; RUN: %lli_mcjit -remote-mcjit %s > /dev/null
; XFAIL:  mips

define i32 @bar() {
	ret i32 0
}

define i32 @main() {
	%r = call i32 @bar( )		; <i32> [#uses=1]
	ret i32 %r
}
