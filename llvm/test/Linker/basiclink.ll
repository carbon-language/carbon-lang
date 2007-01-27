; Test linking two functions with different prototypes and two globals 
; in different modules.
; RUN: llvm-as %s -o %t.bar.bc -f
; RUN: echo "define i32* @foo(i32 %x) { ret i32* @baz } @baz = external global i32" | \
; RUN:   llvm-as -o %t.foo.bc -f
; RUN: llvm-link %t.bar.bc %t.foo.bc -o %t.bc
; RUN: llvm-link %t.foo.bc %t.bar.bc -o %t.bc
declare i32* @foo(...)
define i32* @bar() {
	%ret = call i32* (...)* @foo( i32 123 )
	ret i32* %ret
}
@baz = global i32 0
