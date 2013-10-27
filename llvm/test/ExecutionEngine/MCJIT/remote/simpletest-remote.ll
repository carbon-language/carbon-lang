; RUN: %lli_mcjit -remote-mcjit -mcjit-remote-process=lli-child-target %s > /dev/null

; This fails because __main is not resolved in remote mcjit on cygming.
; XFAIL: cygwin,mingw32,mips

define i32 @bar() {
	ret i32 0
}

define i32 @main() {
	%r = call i32 @bar( )		; <i32> [#uses=1]
	ret i32 %r
}
