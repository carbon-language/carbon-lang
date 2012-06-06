; RUN: llc < %s -march=x86-64 -mattr=+avx -mtriple=i686-apple-darwin10 | FileCheck %s
; rdar://11496434

; no VLAs or dynamic alignment
define i32 @t1() nounwind uwtable ssp {
entry:
  %a = alloca i32, align 4
  call void @t1_helper(i32* %a) nounwind
  %0 = load i32* %a, align 4
  %add = add nsw i32 %0, 13
  ret i32 %add

; CHECK: _t1
; CHECK-NOT: andq $-{{[0-9]+}}, %rsp
; CHECK: leaq [[OFFSET:[0-9]*]](%rsp), %rdi
; CHECK: callq _t1_helper
; CHECK: movl [[OFFSET]](%rsp), %eax
; CHECK: addl $13, %eax
}

declare void @t1_helper(i32*)

; dynamic realignment
define i32 @t2() nounwind uwtable ssp {
entry:
  %a = alloca i32, align 4
  %v = alloca <8 x float>, align 32
  call void @t2_helper(i32* %a, <8 x float>* %v) nounwind
  %0 = load i32* %a, align 4
  %add = add nsw i32 %0, 13
  ret i32 %add

; CHECK: _t2
; CHECK: pushq %rbp
; CHECK: movq %rsp, %rbp
; CHECK: andq $-32, %rsp
; CHECK: subq ${{[0-9]+}}, %rsp
;
; CHECK: leaq {{[0-9]*}}(%rsp), %rdi
; CHECK: leaq {{[0-9]*}}(%rsp), %rsi
; CHECK: callq _t2_helper
;
; CHECK: movq %rbp, %rsp
; CHECK: popq %rbp
}

declare void @t2_helper(i32*, <8 x float>*)

; VLAs
define i32 @t3(i64 %sz) nounwind uwtable ssp {
entry:
  %a = alloca i32, align 4
  %vla = alloca i32, i64 %sz, align 16
  call void @t3_helper(i32* %a, i32* %vla) nounwind
  %0 = load i32* %a, align 4
  %add = add nsw i32 %0, 13
  ret i32 %add

; CHECK: _t3
; CHECK: pushq %rbp
; CHECK: movq %rsp, %rbp
; CHECK: pushq %rbx
; CHECK-NOT: andq $-{{[0-9]+}}, %rsp
; CHECK: subq ${{[0-9]+}}, %rsp
;
; CHECK: leaq -{{[0-9]+}}(%rbp), %rsp
; CHECK: popq %rbx
; CHECK: popq %rbp
}

declare void @t3_helper(i32*, i32*)

; VLAs + Dynamic realignment
define i32 @t4(i64 %sz) nounwind uwtable ssp {
entry:
  %a = alloca i32, align 4
  %v = alloca <8 x float>, align 32
  %vla = alloca i32, i64 %sz, align 16
  call void @t4_helper(i32* %a, i32* %vla, <8 x float>* %v) nounwind
  %0 = load i32* %a, align 4
  %add = add nsw i32 %0, 13
  ret i32 %add

; CHECK: _t4
; CHECK: pushq %rbp
; CHECK: movq %rsp, %rbp
; CHECK: andq $-32, %rsp
; CHECK: pushq %r14
; CHECK: pushq %rbx
; CHECK: subq $[[STACKADJ:[0-9]+]], %rsp
; CHECK: movq %rsp, %rbx
;
; CHECK: leaq {{[0-9]*}}(%rbx), %rdi
; CHECK: leaq {{[0-9]*}}(%rbx), %rdx
; CHECK: callq   _t4_helper
;
; CHECK: addq $[[STACKADJ]], %rsp
; CHECK: popq %rbx
; CHECK: popq %r14
; CHECK: movq %rbp, %rsp
; CHECK: popq %rbp
}

declare void @t4_helper(i32*, i32*, <8 x float>*)

; Dynamic realignment + Spill
define i32 @t5(float* nocapture %f) nounwind uwtable ssp {
entry:
  %a = alloca i32, align 4
  %0 = bitcast float* %f to <8 x float>*
  %1 = load <8 x float>* %0, align 32
  call void @t5_helper1(i32* %a) nounwind
  call void @t5_helper2(<8 x float> %1) nounwind
  %2 = load i32* %a, align 4
  %add = add nsw i32 %2, 13
  ret i32 %add

; CHECK: _t5
; CHECK: pushq %rbp
; CHECK: movq %rsp, %rbp
; CHECK: andq $-32, %rsp
; CHECK: subq ${{[0-9]+}}, %rsp
;
; CHECK: vmovaps (%rdi), [[AVXREG:%ymm[0-9]+]]
; CHECK: vmovaps [[AVXREG]], (%rsp)
; CHECK: leaq {{[0-9]+}}(%rsp), %rdi
; CHECK: callq   _t5_helper1
; CHECK: vmovaps (%rsp), %ymm0
; CHECK: callq   _t5_helper2
; CHECK: movl {{[0-9]+}}(%rsp), %eax
;
; CHECK: movq %rbp, %rsp
; CHECK: popq %rbp
}

declare void @t5_helper1(i32*)

declare void @t5_helper2(<8 x float>)

; VLAs + Dynamic realignment + Spill
; FIXME: RA has already reserved RBX, so we can't do dynamic realignment.
define i32 @t6(i64 %sz, float* nocapture %f) nounwind uwtable ssp {
entry:
; CHECK: _t6
  %a = alloca i32, align 4
  %0 = bitcast float* %f to <8 x float>*
  %1 = load <8 x float>* %0, align 32
  %vla = alloca i32, i64 %sz, align 16
  call void @t6_helper1(i32* %a, i32* %vla) nounwind
  call void @t6_helper2(<8 x float> %1) nounwind
  %2 = load i32* %a, align 4
  %add = add nsw i32 %2, 13
  ret i32 %add
}

declare void @t6_helper1(i32*, i32*)

declare void @t6_helper2(<8 x float>)
