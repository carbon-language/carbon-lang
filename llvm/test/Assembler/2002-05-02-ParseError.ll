; RUN: llvm-as %s -o /dev/null -f

; This should parse correctly without an 'implementation', but our current YACC
; based parser doesn't have the required 2 token lookahead...

%T = type i32 *

define %T %test() {
	ret %T null
}
