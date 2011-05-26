; RUN: lli %s > /dev/null
; XFAIL: arm
; FIXME: ExecutionEngine is broken for ARM, please remove the following XFAIL when it will be fixed.

define i32 @bar() {
	ret i32 0
}

define i32 @main() {
	%r = call i32 @bar( )		; <i32> [#uses=1]
	ret i32 %r
}

