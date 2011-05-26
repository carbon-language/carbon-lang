; RUN: lli %s > /dev/null
; XFAIL: arm
; FIXME: ExecutionEngine is broken for ARM, please remove the following XFAIL when it will be fixed.

define i32 @main() {
	%X = malloc i32		; <i32*> [#uses=1]
	%Y = malloc i32, i32 100		; <i32*> [#uses=1]
	%u = add i32 1, 2		; <i32> [#uses=1]
	%Z = malloc i32, i32 %u		; <i32*> [#uses=1]
	free i32* %X
	free i32* %Y
	free i32* %Z
	ret i32 0
}

