//! Complete, architecture-explicit AArch64 register-state ABI carried by the
//! foreign message protocol; used verbatim in both exit and reply records.

#![forbid(unsafe_code)]

use core::mem::size_of;

/// Opaque frozen magic of the embedded state record: little-endian `b"LOLO"`.
///
/// Deliberately distinct from `FOREIGN_MAGIC` so that a frame at offset 0 and
/// its embedded state header (at message offset 48) cannot be confused. This is
/// a frozen wire value; never change it.
pub const STATE_MAGIC: u32 = u32::from_le_bytes(*b"LOLO");
pub const ARCHITECTURE_AARCH64: u16 = 1;
pub const STATE_VERSION_V1: u16 = 1;

pub const STATE_FLAG_EXCEPTION_VALID: u32 = 1 << 0;
pub const STATE_FLAGS_V1: u32 = STATE_FLAG_EXCEPTION_VALID;

pub const EXCEPTION_FLAG_FAR_VALID: u32 = 1 << 0;
pub const EXCEPTION_FLAG_LOWER_EL: u32 = 1 << 1;
pub const EXCEPTION_FLAGS_V1: u32 = EXCEPTION_FLAG_FAR_VALID | EXCEPTION_FLAG_LOWER_EL;

pub const EXCEPTION_NONE: u32 = 0;
pub const EXCEPTION_SUPERVISOR_CALL: u32 = 1;
pub const EXCEPTION_DATA_ABORT: u32 = 2;
pub const EXCEPTION_INSTRUCTION_ABORT: u32 = 3;
pub const EXCEPTION_ILLEGAL_INSTRUCTION: u32 = 4;
pub const EXCEPTION_BREAKPOINT: u32 = 5;
pub const EXCEPTION_OTHER: u32 = u32::MAX;

/// Data/instruction-abort ISS bit indicating that FAR is not valid.
pub const ESR_ABORT_FAR_NOT_VALID: u64 = 1 << 10;

pub const PSTATE_NZCV_MASK: u64 = 0xf000_0000;
pub const PSTATE_MODE_MASK: u64 = 0x1f;
pub const PSTATE_DAIF_MASK: u64 = 0x3c0;
pub const PSTATE_IL: u64 = 1 << 20;
pub const PSTATE_SINGLE_STEP: u64 = 1 << 21;
pub const PSTATE_PAN: u64 = 1 << 22;
pub const PSTATE_UAO: u64 = 1 << 23;
pub const PSTATE_FORBIDDEN_RESUME_MASK: u64 =
    PSTATE_MODE_MASK | PSTATE_DAIF_MASK | PSTATE_IL | PSTATE_SINGLE_STEP;

/// This state ABI does not expose single-step control. A later protocol version may add it
/// only if the foreign execution mechanism needs it.
pub const SINGLE_STEP_SUPPORTED: bool = false;

/// Portable FPCR bits supported by the v1 state ABI.
pub const FPCR_WRITABLE_MASK: u32 = 0x07c0_9f00;
/// Portable FPSR cumulative-status and saturation bits supported by v1.
pub const FPSR_WRITABLE_MASK: u32 = 0x0800_009f;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct StateHeader {
    pub magic: u32,
    pub version: u16,
    pub architecture: u16,
    pub size: u32,
    pub flags: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(C, align(16))]
pub struct Aarch64Vector {
    pub low: u64,
    pub high: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub struct Aarch64ExceptionState {
    pub kind: u32,
    pub flags: u32,
    pub esr: u64,
    pub far: u64,
    /// Architectural exception-return PC captured in ELR_EL1.
    pub pc: u64,
}

impl Aarch64ExceptionState {
    #[must_use]
    pub const fn from_lower_el(esr: u64, far: u64, pc: u64) -> Self {
        let kind = classify_exception(esr);
        let far_valid = matches!(kind, EXCEPTION_DATA_ABORT | EXCEPTION_INSTRUCTION_ABORT)
            && esr & ESR_ABORT_FAR_NOT_VALID == 0;

        Self {
            kind,
            flags: EXCEPTION_FLAG_LOWER_EL
                | if far_valid {
                    EXCEPTION_FLAG_FAR_VALID
                } else {
                    0
                },
            esr,
            far: if far_valid { far } else { 0 },
            pc,
        }
    }
}

/// Complete, architecture-explicit AArch64 state ABI.
///
/// All fields use fixed-width integers. Exception fields describe why the task
/// stopped and are immutable when the structure is submitted for resumption.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C, align(16))]
pub struct Aarch64StateV1 {
    pub header: StateHeader,
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub pstate: u64,
    pub tpidr_el0: u64,
    pub tpidrro_el0: u64,
    pub vectors: [Aarch64Vector; 32],
    pub fpcr: u32,
    pub fpsr: u32,
    pub exception: Aarch64ExceptionState,
    /// Explicitly occupies the tail so the fixed-size ABI has no implicit
    /// padding bytes that could carry uninitialized kernel memory.
    pub reserved: [u64; 5],
}

pub const STATE_V1_SIZE: u32 = size_of::<Aarch64StateV1>() as u32;

impl Default for Aarch64StateV1 {
    fn default() -> Self {
        Self {
            header: StateHeader {
                magic: STATE_MAGIC,
                version: STATE_VERSION_V1,
                architecture: ARCHITECTURE_AARCH64,
                size: STATE_V1_SIZE,
                flags: 0,
            },
            x: [0; 31],
            sp: 0,
            pc: 0,
            pstate: 0,
            tpidr_el0: 0,
            tpidrro_el0: 0,
            vectors: [Aarch64Vector::default(); 32],
            fpcr: 0,
            fpsr: 0,
            exception: Aarch64ExceptionState::default(),
            reserved: [0; 5],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidationError {
    BadMagic,
    UnsupportedVersion,
    WrongArchitecture,
    WrongSize,
    UnknownStateFlags,
    UnknownExceptionFlags,
    InvalidExceptionRecord,
    NonzeroReserved,
    MisalignedPc,
    PcOutsideUserspace,
    MisalignedSp,
    SpOutsideUserspace,
    ForbiddenPstate,
    NonWritablePstateChanged,
    InvalidFpcr,
    InvalidFpsr,
    ExceptionStateChanged,
}

impl Aarch64StateV1 {
    pub fn validate_snapshot(&self) -> Result<(), ValidationError> {
        if self.header.magic != STATE_MAGIC {
            return Err(ValidationError::BadMagic);
        }
        if self.header.version != STATE_VERSION_V1 {
            return Err(ValidationError::UnsupportedVersion);
        }
        if self.header.architecture != ARCHITECTURE_AARCH64 {
            return Err(ValidationError::WrongArchitecture);
        }
        if self.header.size != STATE_V1_SIZE {
            return Err(ValidationError::WrongSize);
        }
        if self.header.flags & !STATE_FLAGS_V1 != 0 {
            return Err(ValidationError::UnknownStateFlags);
        }
        if self.reserved != [0; 5] {
            return Err(ValidationError::NonzeroReserved);
        }
        if self.header.flags & STATE_FLAG_EXCEPTION_VALID == 0 {
            if self.exception != Aarch64ExceptionState::default() {
                return Err(ValidationError::InvalidExceptionRecord);
            }
        } else {
            if self.exception.kind == EXCEPTION_NONE {
                return Err(ValidationError::InvalidExceptionRecord);
            }
            if self.exception.flags & !EXCEPTION_FLAGS_V1 != 0 {
                return Err(ValidationError::UnknownExceptionFlags);
            }
            if self.exception.flags & EXCEPTION_FLAG_LOWER_EL == 0 {
                return Err(ValidationError::InvalidExceptionRecord);
            }
            if self.exception.kind != classify_exception(self.exception.esr) {
                return Err(ValidationError::InvalidExceptionRecord);
            }
            let far_valid = self.exception.flags & EXCEPTION_FLAG_FAR_VALID != 0;
            let far_is_valid = matches!(
                self.exception.kind,
                EXCEPTION_DATA_ABORT | EXCEPTION_INSTRUCTION_ABORT
            ) && self.exception.esr & ESR_ABORT_FAR_NOT_VALID == 0;
            if far_valid != far_is_valid || (!far_valid && self.exception.far != 0) {
                return Err(ValidationError::InvalidExceptionRecord);
            }
        }

        Ok(())
    }

    /// Validate and canonicalize state supplied for a return to EL0.
    ///
    /// Only integer registers, TLS, FP/SIMD state, PC, SP, and NZCV may change.
    /// Exception metadata and all other PSTATE bits must match the stopped
    /// snapshot exactly.
    pub fn sanitized_for_resume(
        &self,
        stopped: &Self,
        user_end: u64,
    ) -> Result<Self, ValidationError> {
        self.validate_snapshot()?;
        stopped.validate_snapshot()?;

        if self.pc & 0b11 != 0 {
            return Err(ValidationError::MisalignedPc);
        }
        if self.pc >= user_end {
            return Err(ValidationError::PcOutsideUserspace);
        }
        if self.sp & 0b1111 != 0 {
            return Err(ValidationError::MisalignedSp);
        }
        if self.sp > user_end {
            return Err(ValidationError::SpOutsideUserspace);
        }
        if self.pstate & PSTATE_FORBIDDEN_RESUME_MASK != 0 {
            return Err(ValidationError::ForbiddenPstate);
        }
        if (self.pstate ^ stopped.pstate) & !PSTATE_NZCV_MASK != 0 {
            return Err(ValidationError::NonWritablePstateChanged);
        }
        if (self.fpcr ^ stopped.fpcr) & !FPCR_WRITABLE_MASK != 0 {
            return Err(ValidationError::InvalidFpcr);
        }
        if (self.fpsr ^ stopped.fpsr) & !FPSR_WRITABLE_MASK != 0 {
            return Err(ValidationError::InvalidFpsr);
        }
        if self.header.flags != stopped.header.flags || self.exception != stopped.exception {
            return Err(ValidationError::ExceptionStateChanged);
        }

        let mut sanitized = *self;
        sanitized.pstate = (stopped.pstate & !PSTATE_NZCV_MASK) | (self.pstate & PSTATE_NZCV_MASK);
        sanitized.fpcr = (stopped.fpcr & !FPCR_WRITABLE_MASK) | (self.fpcr & FPCR_WRITABLE_MASK);
        sanitized.fpsr = (stopped.fpsr & !FPSR_WRITABLE_MASK) | (self.fpsr & FPSR_WRITABLE_MASK);
        sanitized.exception = stopped.exception;
        sanitized.header.flags = stopped.header.flags;
        sanitized.reserved = [0; 5];
        Ok(sanitized)
    }
}

#[must_use]
pub const fn exception_class(esr: u64) -> u8 {
    ((esr >> 26) & 0x3f) as u8
}

#[must_use]
pub const fn classify_exception(esr: u64) -> u32 {
    match exception_class(esr) {
        0x15 => EXCEPTION_SUPERVISOR_CALL,
        0x20 | 0x21 => EXCEPTION_INSTRUCTION_ABORT,
        0x24 | 0x25 => EXCEPTION_DATA_ABORT,
        0x00 => EXCEPTION_ILLEGAL_INSTRUCTION,
        0x30 | 0x31 | 0x3c => EXCEPTION_BREAKPOINT,
        _ => EXCEPTION_OTHER,
    }
}

#[cfg(test)]
mod tests {
    use core::mem::{align_of, offset_of, size_of};

    use super::*;

    const USER_END: u64 = 1 << 48;

    fn stopped_state() -> Aarch64StateV1 {
        let mut state = Aarch64StateV1::default();
        state.pc = 0x4000;
        state.sp = 0x8000;
        state.pstate = 0xa000_0000;
        state
    }

    #[test]
    fn v1_layout_is_stable() {
        assert_eq!(size_of::<StateHeader>(), 16);
        assert_eq!(size_of::<Aarch64Vector>(), 16);
        assert_eq!(align_of::<Aarch64Vector>(), 16);
        assert_eq!(size_of::<Aarch64ExceptionState>(), 32);
        assert_eq!(size_of::<Aarch64StateV1>(), 896);
        assert_eq!(align_of::<Aarch64StateV1>(), 16);
        assert_eq!(offset_of!(Aarch64StateV1, header), 0);
        assert_eq!(offset_of!(Aarch64StateV1, x), 16);
        assert_eq!(offset_of!(Aarch64StateV1, sp), 264);
        assert_eq!(offset_of!(Aarch64StateV1, pc), 272);
        assert_eq!(offset_of!(Aarch64StateV1, pstate), 280);
        assert_eq!(offset_of!(Aarch64StateV1, tpidr_el0), 288);
        assert_eq!(offset_of!(Aarch64StateV1, tpidrro_el0), 296);
        assert_eq!(offset_of!(Aarch64StateV1, vectors), 304);
        assert_eq!(offset_of!(Aarch64StateV1, fpcr), 816);
        assert_eq!(offset_of!(Aarch64StateV1, fpsr), 820);
        assert_eq!(offset_of!(Aarch64StateV1, exception), 824);
        assert_eq!(offset_of!(Aarch64StateV1, reserved), 856);
        assert_eq!(
            offset_of!(Aarch64StateV1, reserved) + size_of::<[u64; 5]>(),
            size_of::<Aarch64StateV1>()
        );
        assert_eq!(STATE_V1_SIZE, 896);
    }

    #[test]
    fn classifies_lower_el_exceptions() {
        assert_eq!(classify_exception(0x15 << 26), EXCEPTION_SUPERVISOR_CALL);
        assert_eq!(classify_exception(0x24 << 26), EXCEPTION_DATA_ABORT);
        assert_eq!(classify_exception(0x20 << 26), EXCEPTION_INSTRUCTION_ABORT);
        assert_eq!(classify_exception(0), EXCEPTION_ILLEGAL_INSTRUCTION);
        assert_eq!(classify_exception(0x3c << 26), EXCEPTION_BREAKPOINT);
        assert_eq!(classify_exception(0x2f << 26), EXCEPTION_OTHER);
    }

    #[test]
    fn abort_records_preserve_far_and_other_records_clear_it() {
        let abort = Aarch64ExceptionState::from_lower_el(0x24 << 26, 0xdead_0000, 0x4000);
        assert_eq!(abort.kind, EXCEPTION_DATA_ABORT);
        assert_eq!(abort.far, 0xdead_0000);
        assert_ne!(abort.flags & EXCEPTION_FLAG_FAR_VALID, 0);

        let svc = Aarch64ExceptionState::from_lower_el(0x15 << 26, 0xdead_0000, 0x4004);
        assert_eq!(svc.kind, EXCEPTION_SUPERVISOR_CALL);
        assert_eq!(svc.far, 0);
        assert_eq!(svc.flags & EXCEPTION_FLAG_FAR_VALID, 0);

        let abort_without_far = Aarch64ExceptionState::from_lower_el(
            (0x24 << 26) | ESR_ABORT_FAR_NOT_VALID,
            0xdead_0000,
            0x4008,
        );
        assert_eq!(abort_without_far.kind, EXCEPTION_DATA_ABORT);
        assert_eq!(abort_without_far.far, 0);
        assert_eq!(abort_without_far.flags & EXCEPTION_FLAG_FAR_VALID, 0);
    }

    #[test]
    fn distinctive_state_round_trips_through_resume_validation() {
        let stopped = stopped_state();
        let mut requested = stopped;
        for (index, value) in requested.x.iter_mut().enumerate() {
            *value = 0x1000_0000_0000_0000 | index as u64;
        }
        requested.pc = 0x9000;
        requested.sp = 0xa000;
        requested.pstate = 0x5000_0000;
        requested.tpidr_el0 = 0x1111_2222_3333_4444;
        requested.tpidrro_el0 = 0x5555_6666_7777_8888;
        for (index, vector) in requested.vectors.iter_mut().enumerate() {
            vector.low = 0xaaaa_0000_0000_0000 | index as u64;
            vector.high = 0xbbbb_0000_0000_0000 | index as u64;
        }
        requested.fpcr = FPCR_WRITABLE_MASK;
        requested.fpsr = FPSR_WRITABLE_MASK;

        assert_eq!(
            requested.sanitized_for_resume(&stopped, USER_END),
            Ok(requested)
        );
    }

    #[test]
    fn rejects_incompatible_headers_and_reserved_fields() {
        let stopped = stopped_state();
        for (mut invalid, expected) in [
            {
                let mut state = stopped;
                state.header.magic = 0;
                (state, ValidationError::BadMagic)
            },
            {
                let mut state = stopped;
                state.header.version += 1;
                (state, ValidationError::UnsupportedVersion)
            },
            {
                let mut state = stopped;
                state.header.architecture += 1;
                (state, ValidationError::WrongArchitecture)
            },
            {
                let mut state = stopped;
                state.header.size -= 1;
                (state, ValidationError::WrongSize)
            },
            {
                let mut state = stopped;
                state.header.flags = 1 << 31;
                (state, ValidationError::UnknownStateFlags)
            },
            {
                let mut state = stopped;
                state.reserved[0] = 1;
                (state, ValidationError::NonzeroReserved)
            },
        ] {
            assert_eq!(
                invalid.sanitized_for_resume(&stopped, USER_END),
                Err(expected)
            );
            invalid = stopped;
            assert_eq!(invalid.validate_snapshot(), Ok(()));
        }
    }

    #[test]
    fn rejects_unsafe_resume_control() {
        let stopped = stopped_state();
        for (invalid, expected) in [
            {
                let mut state = stopped;
                state.pc += 2;
                (state, ValidationError::MisalignedPc)
            },
            {
                let mut state = stopped;
                state.pc = USER_END;
                (state, ValidationError::PcOutsideUserspace)
            },
            {
                let mut state = stopped;
                state.sp += 8;
                (state, ValidationError::MisalignedSp)
            },
            {
                let mut state = stopped;
                state.sp = USER_END + 16;
                (state, ValidationError::SpOutsideUserspace)
            },
            {
                let mut state = stopped;
                state.pstate |= 1;
                (state, ValidationError::ForbiddenPstate)
            },
            {
                let mut state = stopped;
                state.pstate |= PSTATE_DAIF_MASK;
                (state, ValidationError::ForbiddenPstate)
            },
            {
                let mut state = stopped;
                state.pstate |= PSTATE_SINGLE_STEP;
                (state, ValidationError::ForbiddenPstate)
            },
            {
                let mut state = stopped;
                state.pstate |= 1 << 12;
                (state, ValidationError::NonWritablePstateChanged)
            },
            {
                let mut state = stopped;
                state.fpcr = 1;
                (state, ValidationError::InvalidFpcr)
            },
            {
                let mut state = stopped;
                state.fpsr = 1 << 12;
                (state, ValidationError::InvalidFpsr)
            },
        ] {
            assert_eq!(
                invalid.sanitized_for_resume(&stopped, USER_END),
                Err(expected)
            );
        }
    }

    #[test]
    fn exception_metadata_is_validated_and_immutable() {
        let mut stopped = stopped_state();
        stopped.header.flags = STATE_FLAG_EXCEPTION_VALID;
        stopped.exception = Aarch64ExceptionState::from_lower_el(0x24 << 26, 0x1234, stopped.pc);
        assert_eq!(stopped.validate_snapshot(), Ok(()));

        let mut changed = stopped;
        changed.exception.far += 1;
        assert_eq!(
            changed.sanitized_for_resume(&stopped, USER_END),
            Err(ValidationError::ExceptionStateChanged)
        );

        let mut malformed = stopped;
        malformed.exception.flags &= !EXCEPTION_FLAG_FAR_VALID;
        assert_eq!(
            malformed.validate_snapshot(),
            Err(ValidationError::InvalidExceptionRecord)
        );

        let mut no_far = stopped_state();
        no_far.header.flags = STATE_FLAG_EXCEPTION_VALID;
        no_far.exception = Aarch64ExceptionState::from_lower_el(
            (0x24 << 26) | ESR_ABORT_FAR_NOT_VALID,
            0x5678,
            no_far.pc,
        );
        assert_eq!(no_far.validate_snapshot(), Ok(()));

        let mut malformed_no_far = no_far;
        malformed_no_far.exception.flags |= EXCEPTION_FLAG_FAR_VALID;
        assert_eq!(
            malformed_no_far.validate_snapshot(),
            Err(ValidationError::InvalidExceptionRecord)
        );
    }

    #[test]
    fn preserves_unknown_snapshot_bits_but_rejects_changes_to_them() {
        let mut stopped = stopped_state();
        stopped.pstate |= PSTATE_PAN;
        stopped.fpcr |= 1;
        stopped.fpsr |= 1 << 12;

        assert_eq!(
            stopped.sanitized_for_resume(&stopped, USER_END),
            Ok(stopped)
        );

        let mut changed = stopped;
        changed.pstate ^= PSTATE_PAN;
        assert_eq!(
            changed.sanitized_for_resume(&stopped, USER_END),
            Err(ValidationError::NonWritablePstateChanged)
        );

        changed = stopped;
        changed.fpcr ^= 1;
        assert_eq!(
            changed.sanitized_for_resume(&stopped, USER_END),
            Err(ValidationError::InvalidFpcr)
        );

        changed = stopped;
        changed.fpsr ^= 1 << 12;
        assert_eq!(
            changed.sanitized_for_resume(&stopped, USER_END),
            Err(ValidationError::InvalidFpsr)
        );
    }
}
