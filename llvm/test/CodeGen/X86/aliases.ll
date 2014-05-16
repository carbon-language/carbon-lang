; RUN: llc < %s -mtriple=i686-pc-linux-gnu -asm-verbose=false | FileCheck %s

@bar = global i32 42

; CHECK-DAG: .globl	foo1
@foo1 = alias i32* @bar

; CHECK-DAG: .globl	foo2
@foo2 = alias i32* @bar

%FunTy = type i32()

define i32 @foo_f() {
  ret i32 0
}
; CHECK-DAG: .weak	bar_f
@bar_f = alias weak %FunTy* @foo_f

@bar_l = alias linkonce_odr i32* @bar
; CHECK-DAG: .weak	bar_l

@bar_i = alias internal i32* @bar

; CHECK-DAG: .globl	A
@A = alias i64, i32* @bar

; CHECK-DAG: .globl	bar_h
; CHECK-DAG: .hidden	bar_h
@bar_h = hidden alias i32* @bar

; CHECK-DAG: .globl	bar_p
; CHECK-DAG: .protected	bar_p
@bar_p = protected alias i32* @bar

; CHECK-DAG: .globl	test
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
