; RUN: opt -indvars -instcombine -S < %s | FileCheck %s

;; Test that loop's exit value is rewritten to its initial
;; value from loop preheader
define i32 @test1(i32* %var) {
; CHECK-LABEL: @test1
entry:
 %cond = icmp eq i32* %var, null 
 br label %header

header:
 %phi_indvar = phi i32 [0, %entry], [%indvar, %loop]
 br i1 %cond, label %loop, label %exit

loop:
 %indvar = add i32 %phi_indvar, 1
 br label %header

exit:
; CHECK: ret i32 0
 ret i32 %phi_indvar
}


;; Test that inner loop's exit value is first rewritten to outer
;; loop's induction variable, and then further rewritten to a 
;; constant when process outer loop.
define i32 @test2(i32* %var1, i32* %var2) {
; CHECK-LABEL: @test2
entry:
 %cond1 = icmp eq i32* %var1, null 
 %cond2 = icmp eq i32* %var2, null 
 br label %outer_header

outer_header:
 %phi_outer = phi i32 [0, %entry], [%indvar_outer, %inner_exit]
 br label %inner_header
 
inner_header:
 %phi_inner = phi i32 [%phi_outer, %outer_header], [%indvar_inner, %loop]
 br i1 %cond1, label %loop, label %exit

loop:
 %indvar_inner = add i32 %phi_inner, 1
 br i1 %cond2, label %inner_header, label %inner_exit
 
inner_exit:
 %indvar_outer = add i32 %phi_outer, 1
 br label %outer_header

exit:
;; %phi_inner is first rewritten to %phi_outer
;; and then %phi_outer is rewritten to 0
 %ret_val = add i32 %phi_inner, %phi_outer
; CHECK: ret i32 0
 ret i32 %ret_val
}

;; Test that we can not rewrite loop exit value if it's not
;; a phi node (%indvar is an add instruction in this test).
define i32 @test3(i32* %var) {
; CHECK-LABEL: @test3
entry:
 %cond = icmp eq i32* %var, null 
 br label %header

header:
 %phi_indvar = phi i32 [0, %entry], [%indvar, %header]
 %indvar = add i32 %phi_indvar, 1
 br i1 %cond, label %header, label %exit

exit:
; CHECK: ret i32 %indvar
 ret i32 %indvar
}