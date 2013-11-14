; RUN: llc < %s -mtriple=arm-linux-gnueabi | FileCheck %s

; CHECK: .globl	test

; CHECK: .globl	foo1
; CHECK: foo1 = bar

; CHECK: .globl	foo2
; CHECK: foo2 = bar

; CHECK: .weak	bar_f
; CHECK: bar_f = foo_f

; CHECK: bar_i = bar

; CHECK: .globl	A
; CHECK: A = bar

@bar = global i32 42
@foo1 = alias i32* @bar
@foo2 = alias i32* @bar

%FunTy = type i32()

define i32 @foo_f() {
  ret i32 0
}
@bar_f = alias weak %FunTy* @foo_f

@bar_i = alias internal i32* @bar

@A = alias bitcast (i32* @bar to i64*)

define i32 @test() {
entry:
   %tmp = load i32* @foo1
   %tmp1 = load i32* @foo2
   %tmp0 = load i32* @bar_i
   %tmp2 = call i32 @foo_f()
   %tmp3 = add i32 %tmp, %tmp2
   %tmp4 = call %FunTy* @bar_f()
   %tmp5 = add i32 %tmp3, %tmp4
   %tmp6 = add i32 %tmp1, %tmp5
   %tmp7 = add i32 %tmp6, %tmp0
   ret i32 %tmp7
}
