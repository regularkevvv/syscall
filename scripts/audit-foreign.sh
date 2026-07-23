#!/bin/sh
#
# audit-foreign.sh -- prove the extracted foreign ABI stayed generic and
# unprivileged after the move into the syscall crate.
#
# Run from the repository root. Exits nonzero on the first violation:
#   (a) any unsafe block/fn or inline-assembly construct in the foreign sources;
#   (b) any Linux-policy or Lolo naming leaking into the foreign sources;
#   (c) any changed path outside the allowed extraction surface;
#   (d) a missing inner #![forbid(unsafe_code)] in foreign.rs / aarch64.rs.
# It also asserts positively that `pub mod foreign;` is not feature-gated.
#
# The wire values b"LFOR" / b"LWAT" and the b"LOLO" state magic are frozen
# opaque bytes; the case-sensitive, word-boundary patterns below are written so
# they -- and hex byte literals -- never match.

set -eu

BASE="upstream-base"

# The foreign sources scanned by (a) and (b): foreign.rs plus every file under
# src/foreign/ (aarch64.rs and golden.rs today).
FOREIGN_FILES="src/foreign.rs src/foreign/aarch64.rs src/foreign/golden.rs"

fail() {
    echo "audit-foreign: FAIL: $1" >&2
    exit 1
}

command -v rg >/dev/null 2>&1 || fail "ripgrep (rg) is required"
command -v git >/dev/null 2>&1 || fail "git is required"

for f in $FOREIGN_FILES; do
    [ -f "$f" ] || fail "expected source file is missing: $f"
done

# ----------------------------------------------------------------------
# (a) No unsafe blocks/fns or inline assembly.
# \bunsafe\b matches the `unsafe` keyword but not `unsafe_code` (in the
# forbid attribute) or `unsafe_resume` (a test name), which have a word
# character after the token.
# ----------------------------------------------------------------------
if rg -ns '\bunsafe\b|asm!|global_asm!|naked_asm' $FOREIGN_FILES; then
    fail "unsafe or inline-assembly construct found in foreign sources"
fi

# ----------------------------------------------------------------------
# (b) No Linux-policy, Lolo, or milestone identifiers.
# The `SYS_`/`LINUX_`/`Linux` patterns stay case-sensitive and word-bounded.
# `lolo`/`Lolo` are matched as substrings so predecessor naming cannot slip
# through inside a larger token (e.g. a crate name); this is deliberately
# lowercase-`l` only, so the sanctioned uppercase b"LOLO" state magic and 0x..
# byte literals never match. `\bM[0-8]\b` catches leftover milestone tokens
# (M0..M8) in the extracted sources.
# ----------------------------------------------------------------------
if rg -ns 'SYS_[A-Z]|LINUX_|linuxd|\bLinux\b|\bLolo\b|\blolo\b|lolo|Lolo|\bM[0-8]\b' $FOREIGN_FILES; then
    fail "Linux-policy, Lolo, or milestone identifier found in foreign sources"
fi

# ----------------------------------------------------------------------
# (c) The move must not touch anything outside the allowed surface.
# grep -vxE prints changed paths that are NOT wholly one of the allowed
# files; any output is a violation. `|| true` keeps set -e happy when grep
# finds nothing (all paths allowed).
# ----------------------------------------------------------------------
disallowed=$(
    git diff --name-only "$BASE"..HEAD | grep -vxE \
        '\.gitignore|Cargo\.lock|src/lib\.rs|src/foreign\.rs|src/foreign/aarch64\.rs|src/foreign/golden\.rs|scripts/audit-foreign\.sh' \
        || true
)
if [ -n "$disallowed" ]; then
    fail "changed path outside allowed extraction set:
$disallowed"
fi

# ----------------------------------------------------------------------
# (d) The inner #![forbid(unsafe_code)] must be re-imposed at module scope.
# ----------------------------------------------------------------------
for f in src/foreign.rs src/foreign/aarch64.rs; do
    grep -qF '#![forbid(unsafe_code)]' "$f" \
        || fail "missing inner #![forbid(unsafe_code)] in $f"
done

# ----------------------------------------------------------------------
# Positive assertion: `pub mod foreign;` exists in the crate root and is NOT
# feature-gated -- the kernel consumes it with default-features = false.
# ----------------------------------------------------------------------
grep -qE '^pub mod foreign;' src/lib.rs \
    || fail "pub mod foreign; not declared at the crate root"

if rg -Uq '#\[cfg[^]]*\][[:space:]]*\npub mod foreign;' src/lib.rs; then
    fail "pub mod foreign; is feature-gated by a preceding cfg attribute"
fi
if rg -q '#\[cfg[^]]*\][^\n]*pub mod foreign;' src/lib.rs; then
    fail "pub mod foreign; is feature-gated by an inline cfg attribute"
fi

echo "audit-foreign: PASS"
