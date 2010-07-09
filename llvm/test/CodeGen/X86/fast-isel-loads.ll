; RUN: llc -march=x86-64 -O0 -asm-verbose=false < %s | FileCheck %s

; Fast-isel shouldn't reload the argument values from the stack.

; CHECK: foo:
; CHECK-NEXT: movq  %rdi, -8(%rsp)
; CHECK-NEXT: movq  %rsi, -16(%rsp)
; CHECK: movsd 128(%rsi,%rdi,8), %xmm0
; CHECK-NEXT: ret

define double @foo(i64 %x, double* %p) nounwind {
entry:
  %x.addr = alloca i64, align 8                   ; <i64*> [#uses=2]
  %p.addr = alloca double*, align 8               ; <double**> [#uses=2]
  store i64 %x, i64* %x.addr
  store double* %p, double** %p.addr
  %tmp = load i64* %x.addr                        ; <i64> [#uses=1]
  %tmp1 = load double** %p.addr                   ; <double*> [#uses=1]
  %add = add nsw i64 %tmp, 16                     ; <i64> [#uses=1]
  %arrayidx = getelementptr inbounds double* %tmp1, i64 %add ; <double*> [#uses=1]
  %tmp2 = load double* %arrayidx                  ; <double> [#uses=1]
  ret double %tmp2
}
