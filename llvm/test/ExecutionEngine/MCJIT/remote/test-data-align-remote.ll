; RUN:  %lli_mcjit -remote-mcjit -O0 -mcjit-remote-process=lli-child-target %s

; This fails because __main is not resolved in remote mcjit on cygming.
; XFAIL: cygwin,mingw32,mips

; Check that a variable is always aligned as specified.

@var = global i32 0, align 32
define i32 @main() {
  %addr = ptrtoint i32* @var to i64
  %mask = and i64 %addr, 31
  %tst = icmp eq i64 %mask, 0
  br i1 %tst, label %good, label %bad
good:
  ret i32 0
bad:
  ret i32 1
}
