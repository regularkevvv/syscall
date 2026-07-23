//! Versioned foreign-domain trap/reply protocol shared by the kernel and an
//! unprivileged foreign-task broker.
//!
//! The kernel diverts every `svc` and unhandled architectural exception from an
//! attached foreign task into a broker-owned queue as a [`ForeignMessageV1`]
//! exit record, and the broker replies with the same record shape to resume or
//! terminate the task. The register payload is the [`Aarch64StateV1`] so that
//! every broker-provided register update flows through the same validation
//! boundary; there is exactly one register layout.
//!
//! The broker is any unprivileged supervisor of foreign tasks -- a debugger, a
//! language runtime, or a sandbox. No guest syscall number, errno, or signal
//! number appears here: the broker interprets AArch64 state; the kernel only
//! reports the trap.
//!
//! # Wire layout
//!
//! Every field is little-endian with no implicit padding, so a non-Rust broker
//! can implement the protocol from these tables alone. The fixed exit/reply
//! frame is [`FOREIGN_MESSAGE_V1_WIRE_SIZE`] (976) bytes:
//!
//! | Offset | Size | Field | Notes |
//! |-------:|-----:|-------|-------|
//! | 0 | 4 | `magic` | u32, little-endian `b"LFOR"` ([`FOREIGN_MAGIC`]) |
//! | 4 | 2 | `version` | u16 protocol version |
//! | 6 | 2 | `architecture` | u16, `1` = AArch64 ([`FOREIGN_ARCH_AARCH64`]) |
//! | 8 | 4 | `size` | u32, always 976 |
//! | 12 | 4 | `code` | u32 exit reason or reply kind |
//! | 16 | 8 | `domain_id` | u64 opaque domain token |
//! | 24 | 8 | `task_id` | u64 opaque task token |
//! | 32 | 8 | `sequence` | u64 monotonic exit-sequence token |
//! | 40 | 8 | `flags` | u64, reserved, must be zero |
//! | 48 | 896 | `state` | [`Aarch64StateV1`] register record (see below) |
//! | 944 | 32 | `reserved` | four u64 tail words |
//!
//! The embedded [`Aarch64StateV1`] begins at frame offset 48 and carries its
//! own [`StateHeader`](aarch64::StateHeader), whose magic is
//! [`STATE_MAGIC`](aarch64::STATE_MAGIC) (`b"LOLO"`), deliberately distinct from
//! [`FOREIGN_MAGIC`] so the embedded state header cannot be confused with the
//! surrounding frame. Its fields, at offsets relative to that 48-byte base:
//!
//! | Offset | Size | Field |
//! |-------:|-----:|-------|
//! | 0 | 16 | `header` (magic u32, version u16, architecture u16, size u32, flags u32) |
//! | 16 | 248 | `x[0..31]` general registers, u64 each |
//! | 264 | 8 | `sp` |
//! | 272 | 8 | `pc` |
//! | 280 | 8 | `pstate` |
//! | 288 | 8 | `tpidr_el0` |
//! | 296 | 8 | `tpidrro_el0` |
//! | 304 | 512 | `vectors[0..32]`, each `{ low: u64, high: u64 }` |
//! | 816 | 4 | `fpcr` |
//! | 820 | 4 | `fpsr` |
//! | 824 | 32 | `exception` (kind u32, flags u32, esr u64, far u64, pc u64) |
//! | 856 | 40 | `reserved[0..5]`, u64 each, always zero |
//!
//! The four-word tail at frame offset 944 is all zero for V1 frames and for
//! every reply. On a V2 [`ExitReason::Kick`] its first word (`reserved[0]`, at
//! frame offset 944) carries the [`KickOrigin`] wire value; on a V3-or-later
//! [`ExitReason::WaitComplete`] that same word carries the [`WaitOutcome`] wire
//! value. All other tail words stay zero.
//!
//! # Sub-protocols
//!
//! A broker endpoint mints task-bound capabilities by `dup`-ing a decimal task
//! token appended to a fixed handle prefix:
//!
//! - task-bound memory: [`FOREIGN_MEMORY_HANDLE_PREFIX`] (`"memory/"`), driven
//!   through the positioned read/write scheme interface with no pointer-bearing
//!   record.
//! - atomic-u32: [`FOREIGN_ATOMIC_U32_HANDLE_PREFIX`] (`"atomic-u32/"`).
//! - atomic wait/wake: [`FOREIGN_WAIT_U32_HANDLE_PREFIX`] (`"wait-u32/"`),
//!   carrying the fixed [`ForeignWaitU32RequestV1`] request record.

#![forbid(unsafe_code)]

use core::mem::size_of;

use self::aarch64::{Aarch64ExceptionState, Aarch64StateV1, Aarch64Vector, StateHeader};

/// AArch64 register-state ABI shared by the foreign message protocol.
pub mod aarch64;

/// Magic identifying a foreign exit/reply record: little-endian `b"LFOR"`.
///
/// Distinct from the register-state magic so a stale register state cannot be
/// replayed as a foreign message, or vice versa.
pub const FOREIGN_MAGIC: u32 = u32::from_le_bytes(*b"LFOR");
/// Original protocol version carried by [`ForeignMessageV1`].
///
/// V1 has only supervisor-call, architectural-exception, and task-death exit
/// reasons. Existing domains remain on this version unless their trusted
/// controller explicitly selects V2 before opening a broker endpoint.
pub const FOREIGN_VERSION_V1: u16 = 1;
/// Lifecycle protocol version.
///
/// V2 preserves the fixed 976-byte wire layout, but adds the checked `Kick`
/// exit reason. For a V2 `Kick` exit only, the first word of the existing tail
/// records whether it superseded a simultaneous supervisor call or exception.
/// Every other tail word remains zero. Keeping the extension versioned and
/// layout-stable lets a broker reject an unsupported domain protocol before it
/// starts supervising tasks.
pub const FOREIGN_VERSION_V2: u16 = 2;
/// Atomic wait/wake protocol version.
///
/// V3 preserves the existing fixed exit/reply frame and adds a checked
/// `WaitComplete` exit.  The separate wait-operation request below is also
/// versioned, so neither a broker nor the kernel can reinterpret an older
/// lifecycle record as a wait completion.
pub const FOREIGN_VERSION_V3: u16 = 3;
/// Resource-governance protocol version.
///
/// V4 preserves the fixed V3 exit/reply frame and generic atomic-wait request.
/// It adds a pre-activation domain-control operation for selecting explicit
/// resource limits. The limits are kernel policy for one trust domain, not a
/// guest OS resource policy; V1--V3 callers keep their bounded
/// default limits without having to understand the V4 control operation.
pub const FOREIGN_VERSION_V4: u16 = 4;
/// Whether this crate can safely decode the selected protocol version.
#[must_use]
pub const fn protocol_version_is_supported(version: u16) -> bool {
    matches!(
        version,
        FOREIGN_VERSION_V1 | FOREIGN_VERSION_V2 | FOREIGN_VERSION_V3 | FOREIGN_VERSION_V4
    )
}
/// Architecture identifier: AArch64.
pub const FOREIGN_ARCH_AARCH64: u16 = 1;

/// Exit reason: the foreign task executed `svc`.
pub const EXIT_SUPERVISOR_CALL: u32 = 1;
/// Exit reason: an unhandled lower-EL architectural exception was routed.
pub const EXIT_EXCEPTION: u32 = 2;
/// Exit reason: the task died; terminal, cannot be resumed.
pub const EXIT_TASK_DEATH: u32 = 3;
/// Exit reason: a trusted controller interrupted the foreign task.
///
/// Available only in protocol V2. The record carries the exact stopped state;
/// see [`KickOrigin`] for the versioned tail's meaning.
pub const EXIT_KICK: u32 = 4;
/// Exit reason: a generic atomic wait completed.
///
/// This is deliberately not a guest wait result. The broker receives the
/// generic completion fact and chooses any guest ABI-visible return value in
/// its normal, validated resume reply.
pub const EXIT_WAIT_COMPLETE: u32 = 5;

const KICK_ORIGIN_ASYNCHRONOUS: u64 = 0;
const KICK_ORIGIN_SUPERVISOR_CALL: u64 = EXIT_SUPERVISOR_CALL as u64;
const KICK_ORIGIN_EXCEPTION: u64 = EXIT_EXCEPTION as u64;

/// Reply kind: resume the task with the validated register state.
///
/// Chosen not to overlap the exit reasons, so that a message used in the wrong
/// direction fails validation instead of being silently misinterpreted.
pub const REPLY_RESUME: u32 = 0x101;
/// Reply kind: terminate the task.
pub const REPLY_TERMINATE: u32 = 0x102;

/// Version of the task-bound foreign-memory capability.
///
/// The memory operations themselves use the normal positioned `read`/`write`
/// scheme interface, so there is no pointer-bearing request record. The version
/// and bounds are still part of the UAPI contract: a future incompatible
/// capability shape must use a new version rather than silently changing the
/// meaning of offsets or result counts.
pub const FOREIGN_MEMORY_VERSION_V1: u16 = 1;
/// Maximum bytes transferred by one foreign-memory operation.
///
/// This bounds the kernel-owned staging buffer. A protocol client may compose
/// several exact operations and report progress explicitly; the kernel never
/// performs an implicit partial transfer within this bound.
pub const FOREIGN_MEMORY_MAX_TRANSFER_V1: usize = 64 * 1024;
/// `dup` path prefix used on a broker endpoint to mint a task-bound memory
/// capability. The decimal task token follows this prefix.
pub const FOREIGN_MEMORY_HANDLE_PREFIX: &str = "memory/";
/// `dup` path prefix used on a broker endpoint to mint a task-bound atomic
/// 32-bit capability. The decimal task token follows this prefix.
///
/// This intentionally names a separate endpoint rather than changing the
/// byte-copy endpoint's behavior when a transfer happens to be four bytes.
/// A caller can therefore distinguish exact copies from sequentially
/// consistent atomic load and compare-exchange operations at the ABI level.
pub const FOREIGN_ATOMIC_U32_HANDLE_PREFIX: &str = "atomic-u32/";
/// `dup` path prefix used on a broker endpoint to mint an atomic-wait
/// capability. The decimal task token follows this prefix.
pub const FOREIGN_WAIT_U32_HANDLE_PREFIX: &str = "wait-u32/";

/// Control opcodes written to a foreign-domain control handle. The payload is a
/// sequence of native-width words: `[opcode, arg0, ...]`. These name control
/// actions on the domain, never guest operations.
pub mod domain_op {
    /// `[SET_CAPACITY, capacity]` — set the exit-queue bound before any attach.
    pub const SET_CAPACITY: usize = 0;
    /// `[ATTACH, task_fd]` — quiesce and attach a task capability.
    pub const ATTACH: usize = 1;
    /// `[START, task_fd]` — run an attached, stopped task.
    pub const START: usize = 2;
    /// `[DESTROY]` — tear the domain down (also happens on handle close).
    pub const DESTROY: usize = 3;
    /// `[SET_PROTOCOL_VERSION, version]` — select a supported protocol before
    /// a broker endpoint is opened or a task is attached. Domains default to
    /// V1 so earlier userspace keeps its exact contract.
    pub const SET_PROTOCOL_VERSION: usize = 4;
    /// `[KICK, task_fd]` — request one checked lifecycle interruption for the
    /// attached task referenced by the supplied process capability.
    pub const KICK: usize = 5;
    /// `[CANCEL, task_fd, sequence]` — cancel one exact outstanding broker
    /// request, leaving the task safely stopped. `sequence` prevents a stale
    /// controller action from cancelling a newer exit.
    pub const CANCEL: usize = 6;
    /// `[SET_LIMITS_V1, normal_exits, lifecycle_exits, tasks,
    /// outstanding_replies, transient_copy_pages, wait_registrations]` — set
    /// the complete per-domain resource budget before the broker endpoint
    /// is opened or any task is attached. This operation requires protocol V4
    /// so an older controller cannot silently configure only part of a limit
    /// set it does not understand.
    ///
    /// All six values are counts, not byte sizes, and are carried as
    /// native-width words in the fixed order shown above.
    pub const SET_LIMITS_V1: usize = 7;
}

/// Number of resource values following [`domain_op::SET_LIMITS_V1`].
///
/// The control transport itself uses native-width words because it is a Redox
/// proc-scheme operation. The values are counts rather than pointers, and the
/// selected V4 protocol version makes this fixed ordering part of the contract.
pub const FOREIGN_DOMAIN_LIMITS_V1_VALUE_COUNT: usize = 6;
/// Total native-width words in one V4 `SET_LIMITS_V1` request, including its
/// opcode.
pub const FOREIGN_DOMAIN_LIMITS_V1_CONTROL_WORD_COUNT: usize =
    1 + FOREIGN_DOMAIN_LIMITS_V1_VALUE_COUNT;

/// Fixed message header. Identical for exit records and reply records; `code`
/// carries the exit reason or the reply kind depending on direction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct ForeignHeader {
    pub magic: u32,
    pub version: u16,
    pub architecture: u16,
    pub size: u32,
    pub code: u32,
}

/// A foreign exit record (kernel → broker) or reply record (broker → kernel).
///
/// One fixed-width, architecture-explicit layout is used in both directions.
/// The register payload reuses the [`Aarch64StateV1`] verbatim; there is no
/// second register ABI to keep in sync.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C, align(16))]
pub struct ForeignMessageV1 {
    pub header: ForeignHeader,
    /// Opaque domain token.
    pub domain_id: u64,
    /// Opaque task token, unique within the domain.
    pub task_id: u64,
    /// Monotonic exit-sequence token; a reply must echo the exact outstanding
    /// value.
    pub sequence: u64,
    /// Reserved for future negotiated behavior; must be zero.
    pub flags: u64,
    /// Complete AArch64 machine state: the stopped snapshot on an exit, the
    /// requested state on a resume reply. Ignored for terminate replies and
    /// death exits.
    pub state: Aarch64StateV1,
    /// Explicit tail so the fixed record carries no implicit padding.
    ///
    /// V1 requires all four words to be zero. V2 does too, except that a
    /// `Kick` exit uses `reserved[0]` for its checked [`KickOrigin`]. A reply
    /// always requires all four words to be zero.
    pub reserved: [u64; 4],
}

/// Size of the fixed foreign message record in bytes.
pub const FOREIGN_MESSAGE_V1_SIZE: u32 = size_of::<ForeignMessageV1>() as u32;
/// The fixed wire length in bytes, equal to [`FOREIGN_MESSAGE_V1_SIZE`]; this is
/// the exact buffer length [`ForeignMessageV1::from_wire_bytes`] accepts.
pub const FOREIGN_MESSAGE_V1_WIRE_SIZE: usize = FOREIGN_MESSAGE_V1_SIZE as usize;

impl Default for ForeignHeader {
    fn default() -> Self {
        Self {
            magic: FOREIGN_MAGIC,
            version: FOREIGN_VERSION_V1,
            architecture: FOREIGN_ARCH_AARCH64,
            size: FOREIGN_MESSAGE_V1_SIZE,
            code: 0,
        }
    }
}

impl Default for ForeignMessageV1 {
    fn default() -> Self {
        Self {
            header: ForeignHeader::default(),
            domain_id: 0,
            task_id: 0,
            sequence: 0,
            flags: 0,
            state: Aarch64StateV1::default(),
            reserved: [0; 4],
        }
    }
}

/// Reasons a [`ForeignMessageV1`] fails structural validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForeignError {
    BadMagic,
    UnsupportedVersion,
    WrongArchitecture,
    WrongSize,
    UnknownCode,
    NonzeroFlags,
    NonzeroReserved,
    WrongDomain,
    WrongWireSize,
    WrongProtocolVersion,
    InvalidKickOrigin,
    InvalidWaitOutcome,
    WrongWaitRequestMagic,
    WrongWaitRequestSize,
    InvalidWaitRequest,
}

/// The kind of exit an exit record represents.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExitReason {
    SupervisorCall,
    Exception,
    TaskDeath,
    Kick,
    WaitComplete,
}

/// The event that a V2 [`ExitReason::Kick`] interrupted.
///
/// `Asynchronous` means the task was interrupted while it was merely running
/// or blocked. The other two values preserve the fact that a racing
/// supervisor-call or architectural exception was observed before the kick
/// took ownership of the task's parked state; the event is never silently
/// reinterpreted as a native syscall.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KickOrigin {
    Asynchronous,
    SupervisorCall,
    Exception,
}

/// A kernel-mechanism outcome for an atomic wait.
///
/// These values intentionally do not encode a guest errno, signal, or wait
/// operation. A broker maps the fact to its own guest ABI in a later reply.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaitOutcome {
    Woken,
    TimedOut,
    Interrupted,
    MappingInvalidated,
}

const WAIT_OUTCOME_WOKEN: u64 = 0;
const WAIT_OUTCOME_TIMED_OUT: u64 = 1;
const WAIT_OUTCOME_INTERRUPTED: u64 = 2;
const WAIT_OUTCOME_MAPPING_INVALIDATED: u64 = 3;

/// Magic identifying an atomic-wait operation request: little-endian
/// `b"LWAT"`.
pub const FOREIGN_WAIT_U32_MAGIC: u32 = u32::from_le_bytes(*b"LWAT");
/// First fixed request version for atomic wait/wake operations.
pub const FOREIGN_WAIT_U32_VERSION_V1: u16 = 1;
/// No timeout. Any other timeout is a relative monotonic duration in
/// nanoseconds, not a guest clock or flag combination.
pub const FOREIGN_WAIT_U32_NO_TIMEOUT: u64 = u64::MAX;
/// Fixed byte length of [`ForeignWaitU32RequestV1`] on the wire.
pub const FOREIGN_WAIT_U32_REQUEST_V1_WIRE_SIZE: usize = 64;

/// Generic operations accepted by a task-bound wait capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaitOperation {
    Park,
    Wake,
    Requeue,
    Interrupt,
}

const WAIT_OPERATION_PARK: u16 = 1;
const WAIT_OPERATION_WAKE: u16 = 2;
const WAIT_OPERATION_REQUEUE: u16 = 3;
const WAIT_OPERATION_INTERRUPT: u16 = 4;

/// A fixed, pointer-free atomic wait/wake request.
///
/// `sequence` binds `Park` and `Interrupt` to the exact outstanding foreign
/// exit. `address` and `address2` are guest integers that the kernel resolves
/// through the capability's address space; they are never Rust references.
/// `count` is a wake bound and `requeue_count` is the number to move after the
/// requested wake count. All unused fields must be zero, making a future ABI
/// extension fail closed.
///
/// The record is [`FOREIGN_WAIT_U32_REQUEST_V1_WIRE_SIZE`] (64) little-endian
/// bytes with no implicit padding:
///
/// | Offset | Size | Field | Notes |
/// |-------:|-----:|-------|-------|
/// | 0 | 4 | `magic` | u32, little-endian `b"LWAT"` ([`FOREIGN_WAIT_U32_MAGIC`]) |
/// | 4 | 2 | `version` | u16, `1` ([`FOREIGN_WAIT_U32_VERSION_V1`]) |
/// | 6 | 2 | `operation` | u16 discriminant |
/// | 8 | 4 | `size` | u32, always 64 |
/// | 12 | 4 | reserved | must be zero |
/// | 16 | 8 | `sequence` | u64 |
/// | 24 | 8 | `address` | u64 |
/// | 32 | 4 | `expected` | u32 |
/// | 36 | 4 | `count` | u32 wake bound |
/// | 40 | 8 | `timeout_ns` | u64 |
/// | 48 | 8 | `address2` | u64 requeue target |
/// | 56 | 4 | `requeue_count` | u32 |
/// | 60 | 4 | reserved | must be zero |
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ForeignWaitU32RequestV1 {
    pub operation: WaitOperation,
    pub sequence: u64,
    pub address: u64,
    pub expected: u32,
    pub count: u32,
    pub timeout_ns: u64,
    pub address2: u64,
    pub requeue_count: u32,
}

impl ForeignWaitU32RequestV1 {
    #[must_use]
    pub const fn park(sequence: u64, address: u64, expected: u32, timeout_ns: Option<u64>) -> Self {
        let timeout_ns = match timeout_ns {
            Some(timeout_ns) => timeout_ns,
            None => FOREIGN_WAIT_U32_NO_TIMEOUT,
        };
        Self {
            operation: WaitOperation::Park,
            sequence,
            address,
            expected,
            count: 0,
            timeout_ns,
            address2: 0,
            requeue_count: 0,
        }
    }

    #[must_use]
    pub const fn wake(address: u64, count: u32) -> Self {
        Self {
            operation: WaitOperation::Wake,
            sequence: 0,
            address,
            expected: 0,
            count,
            timeout_ns: 0,
            address2: 0,
            requeue_count: 0,
        }
    }

    #[must_use]
    pub const fn requeue(address: u64, wake_count: u32, address2: u64, requeue_count: u32) -> Self {
        Self {
            operation: WaitOperation::Requeue,
            sequence: 0,
            address,
            expected: 0,
            count: wake_count,
            timeout_ns: 0,
            address2,
            requeue_count,
        }
    }

    #[must_use]
    pub const fn interrupt(sequence: u64) -> Self {
        Self {
            operation: WaitOperation::Interrupt,
            sequence,
            address: 0,
            expected: 0,
            count: 0,
            timeout_ns: 0,
            address2: 0,
            requeue_count: 0,
        }
    }

    /// Validate semantic invariants that can be checked without kernel state.
    pub const fn validate(&self) -> Result<(), ForeignError> {
        let aligned = self.address & 3 == 0;
        match self.operation {
            WaitOperation::Park => {
                if self.sequence == 0
                    || !aligned
                    || self.count != 0
                    || self.address2 != 0
                    || self.requeue_count != 0
                {
                    Err(ForeignError::InvalidWaitRequest)
                } else {
                    Ok(())
                }
            }
            WaitOperation::Wake => {
                if !aligned
                    || self.sequence != 0
                    || self.expected != 0
                    || self.timeout_ns != 0
                    || self.address2 != 0
                    || self.requeue_count != 0
                {
                    Err(ForeignError::InvalidWaitRequest)
                } else {
                    Ok(())
                }
            }
            WaitOperation::Requeue => {
                if !aligned
                    || self.address2 & 3 != 0
                    || self.sequence != 0
                    || self.expected != 0
                    || self.timeout_ns != 0
                {
                    Err(ForeignError::InvalidWaitRequest)
                } else {
                    Ok(())
                }
            }
            WaitOperation::Interrupt => {
                if self.sequence == 0
                    || self.address != 0
                    || self.expected != 0
                    || self.count != 0
                    || self.timeout_ns != 0
                    || self.address2 != 0
                    || self.requeue_count != 0
                {
                    Err(ForeignError::InvalidWaitRequest)
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Encode this request in its explicit, padding-free little-endian form.
    #[must_use]
    pub fn to_wire_bytes(&self) -> [u8; FOREIGN_WAIT_U32_REQUEST_V1_WIRE_SIZE] {
        let mut bytes = [0; FOREIGN_WAIT_U32_REQUEST_V1_WIRE_SIZE];
        bytes[0..4].copy_from_slice(&FOREIGN_WAIT_U32_MAGIC.to_le_bytes());
        bytes[4..6].copy_from_slice(&FOREIGN_WAIT_U32_VERSION_V1.to_le_bytes());
        let operation = match self.operation {
            WaitOperation::Park => WAIT_OPERATION_PARK,
            WaitOperation::Wake => WAIT_OPERATION_WAKE,
            WaitOperation::Requeue => WAIT_OPERATION_REQUEUE,
            WaitOperation::Interrupt => WAIT_OPERATION_INTERRUPT,
        };
        bytes[6..8].copy_from_slice(&operation.to_le_bytes());
        bytes[8..12].copy_from_slice(&(FOREIGN_WAIT_U32_REQUEST_V1_WIRE_SIZE as u32).to_le_bytes());
        // bytes 12..16 are explicit reserved zeroes.
        bytes[16..24].copy_from_slice(&self.sequence.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.address.to_le_bytes());
        bytes[32..36].copy_from_slice(&self.expected.to_le_bytes());
        bytes[36..40].copy_from_slice(&self.count.to_le_bytes());
        bytes[40..48].copy_from_slice(&self.timeout_ns.to_le_bytes());
        bytes[48..56].copy_from_slice(&self.address2.to_le_bytes());
        bytes[56..60].copy_from_slice(&self.requeue_count.to_le_bytes());
        // bytes 60..64 are explicit reserved zeroes.
        bytes
    }

    /// Decode and validate one exact request record.
    pub fn from_wire_bytes(bytes: &[u8]) -> Result<Self, ForeignError> {
        if bytes.len() != FOREIGN_WAIT_U32_REQUEST_V1_WIRE_SIZE {
            return Err(ForeignError::WrongWaitRequestSize);
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != FOREIGN_WAIT_U32_MAGIC {
            return Err(ForeignError::WrongWaitRequestMagic);
        }
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        let size = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != FOREIGN_WAIT_U32_VERSION_V1
            || size != FOREIGN_WAIT_U32_REQUEST_V1_WIRE_SIZE as u32
            || bytes[12..16] != [0; 4]
            || bytes[60..64] != [0; 4]
        {
            return Err(ForeignError::WrongWaitRequestSize);
        }
        let operation = match u16::from_le_bytes(bytes[6..8].try_into().unwrap()) {
            WAIT_OPERATION_PARK => WaitOperation::Park,
            WAIT_OPERATION_WAKE => WaitOperation::Wake,
            WAIT_OPERATION_REQUEUE => WaitOperation::Requeue,
            WAIT_OPERATION_INTERRUPT => WaitOperation::Interrupt,
            _ => return Err(ForeignError::InvalidWaitRequest),
        };
        let request = Self {
            operation,
            sequence: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            address: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            expected: u32::from_le_bytes(bytes[32..36].try_into().unwrap()),
            count: u32::from_le_bytes(bytes[36..40].try_into().unwrap()),
            timeout_ns: u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
            address2: u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
            requeue_count: u32::from_le_bytes(bytes[56..60].try_into().unwrap()),
        };
        request.validate()?;
        Ok(request)
    }
}

/// The action a reply record requests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplyKind {
    Resume,
    Terminate,
}

impl ForeignMessageV1 {
    /// Build an exit record. Used by the kernel; the register `state` must
    /// already be a valid register snapshot for supervisor-call/exception exits, and
    /// is left at its default for death exits.
    #[must_use]
    pub fn new_exit(
        reason: ExitReason,
        domain_id: u64,
        task_id: u64,
        sequence: u64,
        state: Aarch64StateV1,
    ) -> Result<Self, ForeignError> {
        Self::new_exit_for_version(
            FOREIGN_VERSION_V1,
            reason,
            domain_id,
            task_id,
            sequence,
            state,
        )
    }

    /// Build an exit record for an explicitly selected domain protocol.
    ///
    /// The protocol choice belongs to the trusted domain controller and is
    /// fixed before a broker or task exists. Returning an error instead of
    /// silently downgrading means a kernel bug cannot turn a lifecycle `Kick` into a
    /// different V1 event.
    pub fn new_exit_for_version(
        version: u16,
        reason: ExitReason,
        domain_id: u64,
        task_id: u64,
        sequence: u64,
        state: Aarch64StateV1,
    ) -> Result<Self, ForeignError> {
        let code = match reason {
            ExitReason::SupervisorCall => EXIT_SUPERVISOR_CALL,
            ExitReason::Exception => EXIT_EXCEPTION,
            ExitReason::TaskDeath => EXIT_TASK_DEATH,
            ExitReason::Kick => return Err(ForeignError::InvalidKickOrigin),
            ExitReason::WaitComplete => return Err(ForeignError::InvalidWaitRequest),
        };
        if !protocol_version_is_supported(version) {
            return Err(ForeignError::UnsupportedVersion);
        }
        Ok(Self {
            header: ForeignHeader {
                version,
                code,
                ..ForeignHeader::default()
            },
            domain_id,
            task_id,
            sequence,
            flags: 0,
            state,
            reserved: [0; 4],
        })
    }

    /// Build the V2 lifecycle interruption record for a parked foreign task.
    ///
    /// There is no V1 spelling: callers must choose the V2 domain protocol
    /// before attaching a task. The exact stopped state is carried in `state`;
    /// the tail states whether a racing guest-originated event was superseded.
    #[must_use]
    pub fn new_kick_exit(
        domain_id: u64,
        task_id: u64,
        sequence: u64,
        state: Aarch64StateV1,
        origin: KickOrigin,
    ) -> Self {
        Self::new_kick_exit_for_version(
            FOREIGN_VERSION_V2,
            domain_id,
            task_id,
            sequence,
            state,
            origin,
        )
        .expect("V2 is a supported lifecycle protocol")
    }

    /// Build a V2-or-later lifecycle interruption record.
    ///
    /// V3 keeps the exact V2 `Kick` spelling so controlled lifecycle actions
    /// remain available while V3 wait operations are enabled.
    pub fn new_kick_exit_for_version(
        version: u16,
        domain_id: u64,
        task_id: u64,
        sequence: u64,
        state: Aarch64StateV1,
        origin: KickOrigin,
    ) -> Result<Self, ForeignError> {
        if !matches!(
            version,
            FOREIGN_VERSION_V2 | FOREIGN_VERSION_V3 | FOREIGN_VERSION_V4
        ) {
            return Err(ForeignError::InvalidKickOrigin);
        }
        let origin = match origin {
            KickOrigin::Asynchronous => KICK_ORIGIN_ASYNCHRONOUS,
            KickOrigin::SupervisorCall => KICK_ORIGIN_SUPERVISOR_CALL,
            KickOrigin::Exception => KICK_ORIGIN_EXCEPTION,
        };
        Ok(Self {
            header: ForeignHeader {
                version,
                code: EXIT_KICK,
                ..ForeignHeader::default()
            },
            domain_id,
            task_id,
            sequence,
            flags: 0,
            state,
            reserved: [origin, 0, 0, 0],
        })
    }

    /// Build a V3 generic atomic-wait completion record.
    #[must_use]
    pub fn new_wait_complete_exit(
        domain_id: u64,
        task_id: u64,
        sequence: u64,
        state: Aarch64StateV1,
        outcome: WaitOutcome,
    ) -> Self {
        Self::new_wait_complete_exit_for_version(
            FOREIGN_VERSION_V3,
            domain_id,
            task_id,
            sequence,
            state,
            outcome,
        )
        .expect("V3 is a supported atomic-wait protocol")
    }

    /// Build a V3-or-later generic atomic-wait completion record.
    ///
    /// V4 keeps the V3 completion spelling byte-for-byte. The explicit version
    /// argument prevents a V4 domain from accidentally publishing a reply that
    /// its broker must reject as an older protocol frame.
    pub fn new_wait_complete_exit_for_version(
        version: u16,
        domain_id: u64,
        task_id: u64,
        sequence: u64,
        state: Aarch64StateV1,
        outcome: WaitOutcome,
    ) -> Result<Self, ForeignError> {
        if !matches!(version, FOREIGN_VERSION_V3 | FOREIGN_VERSION_V4) {
            return Err(ForeignError::InvalidWaitOutcome);
        }
        let outcome = match outcome {
            WaitOutcome::Woken => WAIT_OUTCOME_WOKEN,
            WaitOutcome::TimedOut => WAIT_OUTCOME_TIMED_OUT,
            WaitOutcome::Interrupted => WAIT_OUTCOME_INTERRUPTED,
            WaitOutcome::MappingInvalidated => WAIT_OUTCOME_MAPPING_INVALIDATED,
        };
        Ok(Self {
            header: ForeignHeader {
                version,
                code: EXIT_WAIT_COMPLETE,
                ..ForeignHeader::default()
            },
            domain_id,
            task_id,
            sequence,
            flags: 0,
            state,
            reserved: [outcome, 0, 0, 0],
        })
    }

    /// Validate the fixed header and flags common to all messages. Exit and
    /// reply validation handle the versioned tail separately because V2 `Kick`
    /// legitimately uses its first word.
    fn validate_frame(&self) -> Result<(), ForeignError> {
        if self.header.magic != FOREIGN_MAGIC {
            return Err(ForeignError::BadMagic);
        }
        if !protocol_version_is_supported(self.header.version) {
            return Err(ForeignError::UnsupportedVersion);
        }
        if self.header.architecture != FOREIGN_ARCH_AARCH64 {
            return Err(ForeignError::WrongArchitecture);
        }
        if self.header.size != FOREIGN_MESSAGE_V1_SIZE {
            return Err(ForeignError::WrongSize);
        }
        if self.flags != 0 {
            return Err(ForeignError::NonzeroFlags);
        }
        Ok(())
    }

    fn validate_exit_tail(&self, reason: ExitReason) -> Result<(), ForeignError> {
        match (self.header.version, reason) {
            (FOREIGN_VERSION_V1, _)
            | (
                FOREIGN_VERSION_V2 | FOREIGN_VERSION_V3 | FOREIGN_VERSION_V4,
                ExitReason::SupervisorCall | ExitReason::Exception | ExitReason::TaskDeath,
            ) if self.reserved == [0; 4] => Ok(()),
            (FOREIGN_VERSION_V1, _)
            | (
                FOREIGN_VERSION_V2 | FOREIGN_VERSION_V3 | FOREIGN_VERSION_V4,
                ExitReason::SupervisorCall | ExitReason::Exception | ExitReason::TaskDeath,
            ) => Err(ForeignError::NonzeroReserved),
            (FOREIGN_VERSION_V2 | FOREIGN_VERSION_V3 | FOREIGN_VERSION_V4, ExitReason::Kick) => {
                if self.reserved[1..] != [0; 3] {
                    return Err(ForeignError::NonzeroReserved);
                }
                match self.reserved[0] {
                    KICK_ORIGIN_ASYNCHRONOUS
                    | KICK_ORIGIN_SUPERVISOR_CALL
                    | KICK_ORIGIN_EXCEPTION => Ok(()),
                    _ => Err(ForeignError::InvalidKickOrigin),
                }
            }
            (FOREIGN_VERSION_V3 | FOREIGN_VERSION_V4, ExitReason::WaitComplete) => {
                if self.reserved[1..] != [0; 3] {
                    return Err(ForeignError::NonzeroReserved);
                }
                match self.reserved[0] {
                    WAIT_OUTCOME_WOKEN
                    | WAIT_OUTCOME_TIMED_OUT
                    | WAIT_OUTCOME_INTERRUPTED
                    | WAIT_OUTCOME_MAPPING_INVALIDATED => Ok(()),
                    _ => Err(ForeignError::InvalidWaitOutcome),
                }
            }
            (_, ExitReason::Kick) => Err(ForeignError::InvalidKickOrigin),
            (_, ExitReason::WaitComplete) => Err(ForeignError::InvalidWaitOutcome),
            _ => Err(ForeignError::UnsupportedVersion),
        }
    }

    /// Interpret this record as an exit and return its reason. Used by the
    /// broker probe to classify what it read.
    pub fn exit_reason(&self) -> Result<ExitReason, ForeignError> {
        self.validate_frame()?;
        let reason = match self.header.code {
            EXIT_SUPERVISOR_CALL => Ok(ExitReason::SupervisorCall),
            EXIT_EXCEPTION => Ok(ExitReason::Exception),
            EXIT_TASK_DEATH => Ok(ExitReason::TaskDeath),
            EXIT_KICK
                if matches!(
                    self.header.version,
                    FOREIGN_VERSION_V2 | FOREIGN_VERSION_V3 | FOREIGN_VERSION_V4
                ) =>
            {
                Ok(ExitReason::Kick)
            }
            EXIT_WAIT_COMPLETE
                if matches!(self.header.version, FOREIGN_VERSION_V3 | FOREIGN_VERSION_V4) =>
            {
                Ok(ExitReason::WaitComplete)
            }
            _ => Err(ForeignError::UnknownCode),
        }?;
        self.validate_exit_tail(reason)?;
        Ok(reason)
    }

    /// Decode the V2 `Kick` origin after [`Self::exit_reason`] has classified
    /// the record. Calling this for a non-kick record is a structural error,
    /// rather than an ambiguous `None` result.
    pub fn kick_origin(&self) -> Result<KickOrigin, ForeignError> {
        if self.exit_reason()? != ExitReason::Kick {
            return Err(ForeignError::InvalidKickOrigin);
        }
        match self.reserved[0] {
            KICK_ORIGIN_ASYNCHRONOUS => Ok(KickOrigin::Asynchronous),
            KICK_ORIGIN_SUPERVISOR_CALL => Ok(KickOrigin::SupervisorCall),
            KICK_ORIGIN_EXCEPTION => Ok(KickOrigin::Exception),
            _ => Err(ForeignError::InvalidKickOrigin),
        }
    }

    /// Decode a V3 generic atomic-wait completion outcome.
    pub fn wait_outcome(&self) -> Result<WaitOutcome, ForeignError> {
        if self.exit_reason()? != ExitReason::WaitComplete {
            return Err(ForeignError::InvalidWaitOutcome);
        }
        match self.reserved[0] {
            WAIT_OUTCOME_WOKEN => Ok(WaitOutcome::Woken),
            WAIT_OUTCOME_TIMED_OUT => Ok(WaitOutcome::TimedOut),
            WAIT_OUTCOME_INTERRUPTED => Ok(WaitOutcome::Interrupted),
            WAIT_OUTCOME_MAPPING_INVALIDATED => Ok(WaitOutcome::MappingInvalidated),
            _ => Err(ForeignError::InvalidWaitOutcome),
        }
    }

    /// Validate this record as a reply bound to `expected_domain_id` and return
    /// the requested action. Stateful checks (task liveness, sequence match) and
    /// register validation happen in the kernel after this passes.
    pub fn validate_reply(&self, expected_domain_id: u64) -> Result<ReplyKind, ForeignError> {
        self.validate_reply_for_version(expected_domain_id, FOREIGN_VERSION_V1)
    }

    /// Validate a reply against the domain's explicitly selected protocol.
    ///
    /// The reply tail is always all zero, including under V2. A lifecycle exit
    /// cannot be replayed as a reply because codes and tail rules are disjoint.
    pub fn validate_reply_for_version(
        &self,
        expected_domain_id: u64,
        expected_version: u16,
    ) -> Result<ReplyKind, ForeignError> {
        self.validate_frame()?;
        if self.header.version != expected_version {
            return Err(ForeignError::WrongProtocolVersion);
        }
        if self.domain_id != expected_domain_id {
            return Err(ForeignError::WrongDomain);
        }
        if self.reserved != [0; 4] {
            return Err(ForeignError::NonzeroReserved);
        }
        match self.header.code {
            REPLY_RESUME => Ok(ReplyKind::Resume),
            REPLY_TERMINATE => Ok(ReplyKind::Terminate),
            _ => Err(ForeignError::UnknownCode),
        }
    }

    /// Turn an exit record the broker read into a resume reply for the same
    /// task and sequence, applying `edit` to the register state. Convenience for
    /// a broker; the kernel re-validates everything.
    #[must_use]
    pub fn to_resume_reply(&self, edit: impl FnOnce(&mut Aarch64StateV1)) -> Self {
        let mut reply = *self;
        reply.header.code = REPLY_RESUME;
        reply.flags = 0;
        reply.reserved = [0; 4];
        edit(&mut reply.state);
        reply
    }

    /// Turn an exit record the broker read into a terminate reply for the same
    /// task and sequence.
    #[must_use]
    pub fn to_terminate_reply(&self) -> Self {
        let mut reply = *self;
        reply.header.code = REPLY_TERMINATE;
        reply.flags = 0;
        reply.reserved = [0; 4];
        reply
    }

    /// Encode this record as its explicit little-endian wire representation.
    ///
    /// The kernel uses this instead of exposing an in-memory Rust layout to a
    /// broker. Keeping this conversion here makes the byte-level contract
    /// reviewable and lets a `#![forbid(unsafe_code)]` broker use the protocol.
    #[must_use]
    pub fn to_wire_bytes(&self) -> [u8; FOREIGN_MESSAGE_V1_WIRE_SIZE] {
        let mut writer = WireWriter::new();
        writer.header(&self.header);
        writer.u64(self.domain_id);
        writer.u64(self.task_id);
        writer.u64(self.sequence);
        writer.u64(self.flags);
        writer.state(&self.state);
        for value in self.reserved {
            writer.u64(value);
        }
        writer.finish()
    }

    /// Decode one complete little-endian wire record.
    ///
    /// This intentionally performs no semantic validation. Call
    /// [`Self::exit_reason`] or [`Self::validate_reply`] after decoding, just
    /// as a kernel would after receiving a C-layout record.
    pub fn from_wire_bytes(bytes: &[u8]) -> Result<Self, ForeignError> {
        if bytes.len() != FOREIGN_MESSAGE_V1_WIRE_SIZE {
            return Err(ForeignError::WrongWireSize);
        }
        let mut reader = WireReader::new(bytes);
        let header = reader.header();
        let domain_id = reader.u64();
        let task_id = reader.u64();
        let sequence = reader.u64();
        let flags = reader.u64();
        let state = reader.state();
        let mut reserved = [0; 4];
        for value in &mut reserved {
            *value = reader.u64();
        }
        debug_assert!(reader.finished());
        Ok(Self {
            header,
            domain_id,
            task_id,
            sequence,
            flags,
            state,
            reserved,
        })
    }
}

/// A deliberately small, bounds-free writer: the fixed record size and the
/// matching unit test make every write statically accounted for.
struct WireWriter {
    bytes: [u8; FOREIGN_MESSAGE_V1_WIRE_SIZE],
    position: usize,
}

impl WireWriter {
    const fn new() -> Self {
        Self {
            bytes: [0; FOREIGN_MESSAGE_V1_WIRE_SIZE],
            position: 0,
        }
    }

    fn u16(&mut self, value: u16) {
        self.bytes[self.position..self.position + 2].copy_from_slice(&value.to_le_bytes());
        self.position += 2;
    }

    fn u32(&mut self, value: u32) {
        self.bytes[self.position..self.position + 4].copy_from_slice(&value.to_le_bytes());
        self.position += 4;
    }

    fn u64(&mut self, value: u64) {
        self.bytes[self.position..self.position + 8].copy_from_slice(&value.to_le_bytes());
        self.position += 8;
    }

    fn header(&mut self, value: &ForeignHeader) {
        self.u32(value.magic);
        self.u16(value.version);
        self.u16(value.architecture);
        self.u32(value.size);
        self.u32(value.code);
    }

    fn state(&mut self, value: &Aarch64StateV1) {
        self.u32(value.header.magic);
        self.u16(value.header.version);
        self.u16(value.header.architecture);
        self.u32(value.header.size);
        self.u32(value.header.flags);
        for register in value.x {
            self.u64(register);
        }
        self.u64(value.sp);
        self.u64(value.pc);
        self.u64(value.pstate);
        self.u64(value.tpidr_el0);
        self.u64(value.tpidrro_el0);
        for vector in value.vectors {
            self.u64(vector.low);
            self.u64(vector.high);
        }
        self.u32(value.fpcr);
        self.u32(value.fpsr);
        self.u32(value.exception.kind);
        self.u32(value.exception.flags);
        self.u64(value.exception.esr);
        self.u64(value.exception.far);
        self.u64(value.exception.pc);
        for reserved in value.reserved {
            self.u64(reserved);
        }
    }

    fn finish(self) -> [u8; FOREIGN_MESSAGE_V1_WIRE_SIZE] {
        debug_assert_eq!(self.position, FOREIGN_MESSAGE_V1_WIRE_SIZE);
        self.bytes
    }
}

struct WireReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> WireReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn u16(&mut self) -> u16 {
        let value = u16::from_le_bytes(
            self.bytes[self.position..self.position + 2]
                .try_into()
                .unwrap(),
        );
        self.position += 2;
        value
    }

    fn u32(&mut self) -> u32 {
        let value = u32::from_le_bytes(
            self.bytes[self.position..self.position + 4]
                .try_into()
                .unwrap(),
        );
        self.position += 4;
        value
    }

    fn u64(&mut self) -> u64 {
        let value = u64::from_le_bytes(
            self.bytes[self.position..self.position + 8]
                .try_into()
                .unwrap(),
        );
        self.position += 8;
        value
    }

    fn header(&mut self) -> ForeignHeader {
        ForeignHeader {
            magic: self.u32(),
            version: self.u16(),
            architecture: self.u16(),
            size: self.u32(),
            code: self.u32(),
        }
    }

    fn state(&mut self) -> Aarch64StateV1 {
        let header = StateHeader {
            magic: self.u32(),
            version: self.u16(),
            architecture: self.u16(),
            size: self.u32(),
            flags: self.u32(),
        };
        let mut x = [0; 31];
        for register in &mut x {
            *register = self.u64();
        }
        let sp = self.u64();
        let pc = self.u64();
        let pstate = self.u64();
        let tpidr_el0 = self.u64();
        let tpidrro_el0 = self.u64();
        let mut vectors = [Aarch64Vector::default(); 32];
        for vector in &mut vectors {
            vector.low = self.u64();
            vector.high = self.u64();
        }
        let fpcr = self.u32();
        let fpsr = self.u32();
        let exception = Aarch64ExceptionState {
            kind: self.u32(),
            flags: self.u32(),
            esr: self.u64(),
            far: self.u64(),
            pc: self.u64(),
        };
        let mut reserved = [0; 5];
        for value in &mut reserved {
            *value = self.u64();
        }
        Aarch64StateV1 {
            header,
            x,
            sp,
            pc,
            pstate,
            tpidr_el0,
            tpidrro_el0,
            vectors,
            fpcr,
            fpsr,
            exception,
            reserved,
        }
    }

    fn finished(&self) -> bool {
        self.position == self.bytes.len()
    }
}

#[cfg(test)]
mod golden;

#[cfg(test)]
mod tests {
    use core::mem::{align_of, offset_of, size_of};

    use super::*;
    use super::aarch64::{STATE_FLAG_EXCEPTION_VALID, STATE_MAGIC};

    #[test]
    fn v1_layout_is_stable() {
        assert_eq!(size_of::<ForeignHeader>(), 16);
        assert_eq!(size_of::<ForeignMessageV1>(), 976);
        assert_eq!(align_of::<ForeignMessageV1>(), 16);
        assert_eq!(offset_of!(ForeignMessageV1, header), 0);
        assert_eq!(offset_of!(ForeignMessageV1, domain_id), 16);
        assert_eq!(offset_of!(ForeignMessageV1, task_id), 24);
        assert_eq!(offset_of!(ForeignMessageV1, sequence), 32);
        assert_eq!(offset_of!(ForeignMessageV1, flags), 40);
        assert_eq!(offset_of!(ForeignMessageV1, state), 48);
        assert_eq!(offset_of!(ForeignMessageV1, reserved), 944);
        assert_eq!(
            offset_of!(ForeignMessageV1, reserved) + size_of::<[u64; 4]>(),
            size_of::<ForeignMessageV1>()
        );
        assert_eq!(FOREIGN_MESSAGE_V1_SIZE, 976);
    }

    #[test]
    fn magic_is_distinct_from_state_magic() {
        assert_ne!(FOREIGN_MAGIC, STATE_MAGIC);
    }

    #[test]
    fn wire_round_trip_preserves_every_field() {
        let mut message = valid_exit();
        message.state.x[0] = 0x0102_0304_0506_0708;
        message.state.x[30] = u64::MAX;
        message.state.vectors[0] = Aarch64Vector {
            low: 0x1111_2222_3333_4444,
            high: 0x5555_6666_7777_8888,
        };
        message.state.vectors[31] = Aarch64Vector {
            low: 0x9999_aaaa_bbbb_cccc,
            high: 0xdddd_eeee_ffff_0001,
        };
        message.state.fpcr = 0x1234_5678;
        message.state.fpsr = 0x90ab_cdef;
        message.state.exception.kind = 4;
        message.state.exception.flags = 2;
        message.state.exception.esr = 0xfeed_face_cafe_beef;
        message.state.exception.pc = 0x4000;
        message.reserved = [1, 2, 3, 4];

        let bytes = message.to_wire_bytes();
        assert_eq!(&bytes[..4], b"LFOR");
        assert_eq!(ForeignMessageV1::from_wire_bytes(&bytes), Ok(message));
    }

    #[test]
    fn wire_decode_rejects_non_exact_lengths() {
        let bytes = valid_exit().to_wire_bytes();
        assert_eq!(
            ForeignMessageV1::from_wire_bytes(&bytes[..bytes.len() - 1]),
            Err(ForeignError::WrongWireSize)
        );
        assert_eq!(
            ForeignMessageV1::from_wire_bytes(&[0; FOREIGN_MESSAGE_V1_WIRE_SIZE + 1]),
            Err(ForeignError::WrongWireSize)
        );
    }

    #[test]
    fn memory_contract_is_bounded_and_versioned() {
        assert_eq!(FOREIGN_MEMORY_VERSION_V1, 1);
        assert_eq!(FOREIGN_MEMORY_MAX_TRANSFER_V1, 64 * 1024);
        assert_eq!(FOREIGN_MEMORY_HANDLE_PREFIX, "memory/");
        assert_eq!(FOREIGN_ATOMIC_U32_HANDLE_PREFIX, "atomic-u32/");
        assert_eq!(FOREIGN_WAIT_U32_HANDLE_PREFIX, "wait-u32/");
    }

    fn valid_exit() -> ForeignMessageV1 {
        let mut state = Aarch64StateV1::default();
        state.pc = 0x4000;
        state.sp = 0x8000;
        ForeignMessageV1::new_exit(ExitReason::SupervisorCall, 7, 11, 3, state)
            .expect("supervisor calls are valid V1 exits")
    }

    #[test]
    fn exit_records_classify() {
        assert_eq!(valid_exit().exit_reason(), Ok(ExitReason::SupervisorCall));

        let mut death = valid_exit();
        death.header.code = EXIT_TASK_DEATH;
        assert_eq!(death.exit_reason(), Ok(ExitReason::TaskDeath));
    }

    #[test]
    fn v2_kick_preserves_a_racing_guest_exit_kind() {
        let state = valid_exit().state;
        let kick = ForeignMessageV1::new_kick_exit(7, 11, 4, state, KickOrigin::SupervisorCall);

        assert_eq!(kick.header.version, FOREIGN_VERSION_V2);
        assert_eq!(kick.exit_reason(), Ok(ExitReason::Kick));
        assert_eq!(kick.kick_origin(), Ok(KickOrigin::SupervisorCall));
        assert_eq!(
            ForeignMessageV1::from_wire_bytes(&kick.to_wire_bytes()),
            Ok(kick)
        );

        let resume = kick.to_resume_reply(|state| state.x[0] = 0x55);
        assert_eq!(
            resume.validate_reply_for_version(7, FOREIGN_VERSION_V2),
            Ok(ReplyKind::Resume)
        );
        assert_eq!(
            resume.validate_reply(7),
            Err(ForeignError::WrongProtocolVersion)
        );
    }

    #[test]
    fn v3_wait_completion_is_versioned_and_replyable() {
        let exit = ForeignMessageV1::new_wait_complete_exit(
            7,
            11,
            4,
            valid_exit().state,
            WaitOutcome::TimedOut,
        );

        assert_eq!(exit.header.version, FOREIGN_VERSION_V3);
        assert_eq!(exit.exit_reason(), Ok(ExitReason::WaitComplete));
        assert_eq!(exit.wait_outcome(), Ok(WaitOutcome::TimedOut));
        assert_eq!(
            ForeignMessageV1::from_wire_bytes(&exit.to_wire_bytes()),
            Ok(exit)
        );
        assert_eq!(
            exit.to_resume_reply(|_| {})
                .validate_reply_for_version(7, FOREIGN_VERSION_V3),
            Ok(ReplyKind::Resume)
        );
    }

    #[test]
    fn v3_preserves_the_v2_kick_contract() {
        let kick = ForeignMessageV1::new_kick_exit_for_version(
            FOREIGN_VERSION_V3,
            7,
            11,
            4,
            valid_exit().state,
            KickOrigin::Asynchronous,
        )
        .expect("V3 retains lifecycle kicks");
        assert_eq!(kick.exit_reason(), Ok(ExitReason::Kick));
        assert_eq!(kick.kick_origin(), Ok(KickOrigin::Asynchronous));
    }

    #[test]
    fn v4_preserves_v3_frames_and_names_the_complete_limit_request() {
        assert!(protocol_version_is_supported(FOREIGN_VERSION_V4));
        assert_eq!(
            FOREIGN_DOMAIN_LIMITS_V1_CONTROL_WORD_COUNT,
            1 + FOREIGN_DOMAIN_LIMITS_V1_VALUE_COUNT
        );

        let state = valid_exit().state;
        let wait = ForeignMessageV1::new_wait_complete_exit_for_version(
            FOREIGN_VERSION_V4,
            7,
            11,
            4,
            state,
            WaitOutcome::Woken,
        )
        .expect("V4 retains atomic-wait completions");
        assert_eq!(wait.header.version, FOREIGN_VERSION_V4);
        assert_eq!(wait.exit_reason(), Ok(ExitReason::WaitComplete));
        assert_eq!(wait.wait_outcome(), Ok(WaitOutcome::Woken));
        assert_eq!(
            wait.to_resume_reply(|_| {})
                .validate_reply_for_version(7, FOREIGN_VERSION_V4),
            Ok(ReplyKind::Resume)
        );

        let kick = ForeignMessageV1::new_kick_exit_for_version(
            FOREIGN_VERSION_V4,
            7,
            11,
            5,
            state,
            KickOrigin::Exception,
        )
        .expect("V4 retains lifecycle kicks");
        assert_eq!(kick.kick_origin(), Ok(KickOrigin::Exception));
    }

    #[test]
    fn deterministic_wire_fuzz_never_reinterprets_or_panics() {
        // This intentionally uses no random crate or test-time entropy: every
        // CI run exercises the same 16k hostile exit/reply/state inputs and
        // can reproduce a failure from the initial seed. `from_wire_bytes`
        // is a lossless decode, while the following semantic checks are the
        // untrusted-broker boundaries used by the kernel.
        const CASES: usize = 16_384;
        let mut seed = 0x4c4f_4c4f_4d38_4655_u64;
        let stopped = Aarch64StateV1::default();
        for _ in 0..CASES {
            let mut bytes = ForeignMessageV1::new_exit_for_version(
                FOREIGN_VERSION_V4,
                ExitReason::SupervisorCall,
                7,
                11,
                13,
                stopped,
            )
            .expect("V4 supervisor-call frame is valid")
            .to_wire_bytes();
            let mutations = usize::try_from(next_fuzz_word(&mut seed) % 8 + 1)
                .expect("small mutation count fits usize");
            for _ in 0..mutations {
                let index = usize::try_from(
                    next_fuzz_word(&mut seed) % FOREIGN_MESSAGE_V1_WIRE_SIZE as u64,
                )
                .expect("wire index fits usize");
                bytes[index] ^= next_fuzz_word(&mut seed) as u8;
            }

            let decoded = ForeignMessageV1::from_wire_bytes(&bytes)
                .expect("an exact-width record must always decode losslessly");
            assert_eq!(decoded.to_wire_bytes(), bytes);
            let _ = decoded.exit_reason();
            let _ = decoded.validate_reply_for_version(decoded.domain_id, FOREIGN_VERSION_V4);
            let _ = decoded.state.validate_snapshot();
            let _ = decoded.state.sanitized_for_resume(&stopped, 1_u64 << 48);
        }

        // The independent wait/capability request decoder must also tolerate
        // arbitrary lengths and bytes without accepting a malformed record as
        // a differently shaped operation.
        for _ in 0..CASES {
            let mut bytes = [0_u8; FOREIGN_WAIT_U32_REQUEST_V1_WIRE_SIZE];
            for byte in &mut bytes {
                *byte = next_fuzz_word(&mut seed) as u8;
            }
            let length = usize::try_from(
                next_fuzz_word(&mut seed) % (FOREIGN_WAIT_U32_REQUEST_V1_WIRE_SIZE as u64 + 2),
            )
            .expect("fuzz length fits usize");
            let input = if length <= bytes.len() {
                &bytes[..length]
            } else {
                &bytes[..]
            };
            let _ = ForeignWaitU32RequestV1::from_wire_bytes(input);
        }
    }

    fn next_fuzz_word(seed: &mut u64) -> u64 {
        // SplitMix64: small, deterministic, and sufficient to distribute
        // bit/field mutations without becoming a protocol dependency.
        *seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = *seed;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    #[test]
    fn wait_request_wire_codec_rejects_ambiguous_inputs() {
        let request = ForeignWaitU32RequestV1::park(9, 0x4000, 0xfeed_beef, Some(42));
        let bytes = request.to_wire_bytes();
        assert_eq!(
            ForeignWaitU32RequestV1::from_wire_bytes(&bytes),
            Ok(request)
        );

        let mut wrong_tail = bytes;
        wrong_tail[60] = 1;
        assert_eq!(
            ForeignWaitU32RequestV1::from_wire_bytes(&wrong_tail),
            Err(ForeignError::WrongWaitRequestSize)
        );

        let unaligned = ForeignWaitU32RequestV1::wake(3, 1).to_wire_bytes();
        assert_eq!(
            ForeignWaitU32RequestV1::from_wire_bytes(&unaligned),
            Err(ForeignError::InvalidWaitRequest)
        );
    }

    #[test]
    fn wait_request_operations_have_disjoint_shapes() {
        assert!(ForeignWaitU32RequestV1::park(1, 0, 0, None)
            .validate()
            .is_ok());
        assert!(ForeignWaitU32RequestV1::wake(0, 0).validate().is_ok());
        assert!(ForeignWaitU32RequestV1::requeue(0, 1, 4, 2)
            .validate()
            .is_ok());
        assert!(ForeignWaitU32RequestV1::interrupt(1).validate().is_ok());
        assert!(ForeignWaitU32RequestV1::park(0, 0, 0, None)
            .validate()
            .is_err());
        assert!(ForeignWaitU32RequestV1::requeue(0, 0, 2, 0)
            .validate()
            .is_err());
    }

    #[test]
    fn kick_is_unrepresentable_in_v1_and_tails_are_version_checked() {
        let state = valid_exit().state;
        assert_eq!(
            ForeignMessageV1::new_exit_for_version(
                FOREIGN_VERSION_V1,
                ExitReason::Kick,
                7,
                11,
                4,
                state,
            ),
            Err(ForeignError::InvalidKickOrigin)
        );

        let mut v1 = valid_exit();
        v1.header.code = EXIT_KICK;
        assert_eq!(v1.exit_reason(), Err(ForeignError::UnknownCode));

        let mut v2 = ForeignMessageV1::new_kick_exit(7, 11, 4, state, KickOrigin::Asynchronous);
        v2.reserved[1] = 1;
        assert_eq!(v2.exit_reason(), Err(ForeignError::NonzeroReserved));
    }

    #[test]
    fn reply_validation_binds_domain_and_kind() {
        let resume = valid_exit().to_resume_reply(|s| s.x[0] = 0x1234);
        assert_eq!(resume.validate_reply(7), Ok(ReplyKind::Resume));
        // Cross-domain reply is rejected structurally.
        assert_eq!(resume.validate_reply(8), Err(ForeignError::WrongDomain));

        let terminate = valid_exit().to_terminate_reply();
        assert_eq!(terminate.validate_reply(7), Ok(ReplyKind::Terminate));
    }

    #[test]
    fn hostile_frames_are_rejected() {
        let mut m = valid_exit().to_resume_reply(|_| {});
        m.header.magic = 0;
        assert_eq!(m.validate_reply(7), Err(ForeignError::BadMagic));

        m = valid_exit().to_resume_reply(|_| {});
        m.header.version = FOREIGN_VERSION_V2;
        assert_eq!(m.validate_reply(7), Err(ForeignError::WrongProtocolVersion));

        m = valid_exit().to_resume_reply(|_| {});
        m.header.architecture = 2;
        assert_eq!(m.validate_reply(7), Err(ForeignError::WrongArchitecture));

        m = valid_exit().to_resume_reply(|_| {});
        m.header.size -= 1;
        assert_eq!(m.validate_reply(7), Err(ForeignError::WrongSize));

        m = valid_exit().to_resume_reply(|_| {});
        m.header.code = 0xdead;
        assert_eq!(m.validate_reply(7), Err(ForeignError::UnknownCode));

        m = valid_exit().to_resume_reply(|_| {});
        m.flags = 1;
        assert_eq!(m.validate_reply(7), Err(ForeignError::NonzeroFlags));

        m = valid_exit().to_resume_reply(|_| {});
        m.reserved[3] = 1;
        assert_eq!(m.validate_reply(7), Err(ForeignError::NonzeroReserved));

        m = valid_exit().to_resume_reply(|_| {});
        m.header.version = 99;
        assert_eq!(
            m.validate_reply_for_version(7, 99),
            Err(ForeignError::UnsupportedVersion)
        );
    }

    #[test]
    fn exit_code_and_reply_code_do_not_overlap() {
        for exit in [EXIT_SUPERVISOR_CALL, EXIT_EXCEPTION, EXIT_TASK_DEATH] {
            assert_ne!(exit, REPLY_RESUME);
            assert_ne!(exit, REPLY_TERMINATE);
        }
        // An exit record cannot be misread as a reply and vice versa.
        let exit = valid_exit();
        assert_eq!(exit.validate_reply(7), Err(ForeignError::UnknownCode));
    }

    #[test]
    fn resume_reply_carries_exception_state_unchanged() {
        let mut state = Aarch64StateV1::default();
        state.pc = 0x4000;
        state.sp = 0x8000;
        state.header.flags = STATE_FLAG_EXCEPTION_VALID;
        state.exception =
            super::aarch64::Aarch64ExceptionState::from_lower_el(0x24 << 26, 0xdead_0000, 0x4000);
        let exit = ForeignMessageV1::new_exit(ExitReason::Exception, 1, 2, 9, state)
            .expect("exceptions are valid V1 exits");
        let reply = exit.to_resume_reply(|s| s.pc += 4);
        // The exception metadata is echoed back verbatim for the kernel's
        // immutability check.
        assert_eq!(reply.state.exception, state.exception);
        assert_eq!(reply.state.pc, 0x4004);
    }
}
