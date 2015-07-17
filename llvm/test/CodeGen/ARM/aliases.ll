; RUN: llc < %s -mtriple=arm-linux-gnueabi | FileCheck %s

; CHECK: .globl	test

; CHECK: .globl structvar
; CHECK: .size structvar, 8

; CHECK: .globl	foo1
; CHECK: foo1 = bar
; CHECK: .size foo1, 4

; CHECK: .globl	foo2
; CHECK: foo2 = bar
; CHECK: .size foo2, 4

; CHECK: .weak	bar_f
; CHECK: bar_f = foo_f

; CHECK: bar_i = bar
; CHECK: .size bar_i, 4

; CHECK: .globl	A
; CHECK: A = bar
; CHECK: .size A, 8

; CHECK: .globl elem0
; CHECK: elem0 = structvar
; CHECK: .size elem0, 4

; CHECK: .globl elem1
; CHECK: elem1 = structvar+4
; CHECK: .size elem1, 4

@bar = global i32 42
@foo1 = alias i32* @bar
@foo2 = alias i32* @bar

%FunTy = type i32()

define i32 @foo_f() {
  ret i32 0
}
@bar_f = weak alias %FunTy* @foo_f

@bar_i = internal alias i32* @bar

@A = alias bitcast (i32* @bar to i64*)

@structvar = global {i32, i32} {i32 1, i32 2}
@elem0 = alias getelementptr({i32, i32}, {i32, i32}*  @structvar, i32 0, i32 0)
@elem1 = alias getelementptr({i32, i32}, {i32, i32}*  @structvar, i32 0, i32 1)

define i32 @test() {
entry:
   %tmp = load i32, i32* @foo1
   %tmp1 = load i32, i32* @foo2
   %tmp0 = load i32, i32* @bar_i
   %tmp2 = call i32 @foo_f()
   %tmp3 = add i32 %tmp, %tmp2
   %tmp4 = call i32 @bar_f()
   %tmp5 = add i32 %tmp3, %tmp4
   %tmp6 = add i32 %tmp1, %tmp5
   %tmp7 = add i32 %tmp6, %tmp0
   ret i32 %tmp7
}
