; RUN: llc < %s -march=x86-64 -mattr=+mmx,+sse2 | FileCheck %s

define i64 @t0(<1 x i64>* %a, i32* %b) {
; CHECK-LABEL: t0:
; CHECK:       # BB#0:{{.*}} %entry
; CHECK:         movq (%[[REG1:[a-z]+]]), %mm0
; CHECK-NEXT:    psllq (%[[REG2:[a-z]+]]), %mm0
; CHECK-NEXT:    movd %mm0, %rax
; CHECK-NEXT:    retq
entry:
  %0 = bitcast <1 x i64>* %a to x86_mmx*
  %1 = load x86_mmx* %0, align 8
  %2 = load i32* %b, align 4
  %3 = tail call x86_mmx @llvm.x86.mmx.pslli.q(x86_mmx %1, i32 %2)
  %4 = bitcast x86_mmx %3 to i64
  ret i64 %4
}
declare x86_mmx @llvm.x86.mmx.pslli.q(x86_mmx, i32)

define i64 @t1(<1 x i64>* %a, i32* %b) {
; CHECK-LABEL: t1:
; CHECK:       # BB#0:{{.*}} %entry
; CHECK:         movq (%[[REG1]]), %mm0
; CHECK-NEXT:    psrlq (%[[REG2]]), %mm0
; CHECK-NEXT:    movd %mm0, %rax
; CHECK-NEXT:    retq
entry:
  %0 = bitcast <1 x i64>* %a to x86_mmx*
  %1 = load x86_mmx* %0, align 8
  %2 = load i32* %b, align 4
  %3 = tail call x86_mmx @llvm.x86.mmx.psrli.q(x86_mmx %1, i32 %2)
  %4 = bitcast x86_mmx %3 to i64
  ret i64 %4
}
declare x86_mmx @llvm.x86.mmx.psrli.q(x86_mmx, i32)

define i64 @t2(<1 x i64>* %a, i32* %b) {
; CHECK-LABEL: t2:
; CHECK:       # BB#0:{{.*}} %entry
; CHECK:         movq (%[[REG1]]), %mm0
; CHECK-NEXT:    psllw (%[[REG2]]), %mm0
; CHECK-NEXT:    movd %mm0, %rax
; CHECK-NEXT:    retq
entry:
  %0 = bitcast <1 x i64>* %a to x86_mmx*
  %1 = load x86_mmx* %0, align 8
  %2 = load i32* %b, align 4
  %3 = tail call x86_mmx @llvm.x86.mmx.pslli.w(x86_mmx %1, i32 %2)
  %4 = bitcast x86_mmx %3 to i64
  ret i64 %4
}
declare x86_mmx @llvm.x86.mmx.pslli.w(x86_mmx, i32)

define i64 @t3(<1 x i64>* %a, i32* %b) {
; CHECK-LABEL: t3:
; CHECK:       # BB#0:{{.*}} %entry
; CHECK:         movq (%[[REG1]]), %mm0
; CHECK-NEXT:    psrlw (%[[REG2]]), %mm0
; CHECK-NEXT:    movd %mm0, %rax
; CHECK-NEXT:    retq
entry:
  %0 = bitcast <1 x i64>* %a to x86_mmx*
  %1 = load x86_mmx* %0, align 8
  %2 = load i32* %b, align 4
  %3 = tail call x86_mmx @llvm.x86.mmx.psrli.w(x86_mmx %1, i32 %2)
  %4 = bitcast x86_mmx %3 to i64
  ret i64 %4
}
declare x86_mmx @llvm.x86.mmx.psrli.w(x86_mmx, i32)

define i64 @t4(<1 x i64>* %a, i32* %b) {
; CHECK-LABEL: t4:
; CHECK:       # BB#0:{{.*}} %entry
; CHECK:         movq (%[[REG1]]), %mm0
; CHECK-NEXT:    pslld (%[[REG2]]), %mm0
; CHECK-NEXT:    movd %mm0, %rax
; CHECK-NEXT:    retq
entry:
  %0 = bitcast <1 x i64>* %a to x86_mmx*
  %1 = load x86_mmx* %0, align 8
  %2 = load i32* %b, align 4
  %3 = tail call x86_mmx @llvm.x86.mmx.pslli.d(x86_mmx %1, i32 %2)
  %4 = bitcast x86_mmx %3 to i64
  ret i64 %4
}
declare x86_mmx @llvm.x86.mmx.pslli.d(x86_mmx, i32)

define i64 @t5(<1 x i64>* %a, i32* %b) {
; CHECK-LABEL: t5:
; CHECK:       # BB#0:{{.*}} %entry
; CHECK:         movq (%[[REG1]]), %mm0
; CHECK-NEXT:    psrld (%[[REG2]]), %mm0
; CHECK-NEXT:    movd %mm0, %rax
; CHECK-NEXT:    retq
entry:
  %0 = bitcast <1 x i64>* %a to x86_mmx*
  %1 = load x86_mmx* %0, align 8
  %2 = load i32* %b, align 4
  %3 = tail call x86_mmx @llvm.x86.mmx.psrli.d(x86_mmx %1, i32 %2)
  %4 = bitcast x86_mmx %3 to i64
  ret i64 %4
}
declare x86_mmx @llvm.x86.mmx.psrli.d(x86_mmx, i32)

define i64 @t6(<1 x i64>* %a, i32* %b) {
; CHECK-LABEL: t6:
; CHECK:       # BB#0:{{.*}} %entry
; CHECK:         movq (%[[REG1]]), %mm0
; CHECK-NEXT:    psraw (%[[REG2]]), %mm0
; CHECK-NEXT:    movd %mm0, %rax
; CHECK-NEXT:    retq
entry:
  %0 = bitcast <1 x i64>* %a to x86_mmx*
  %1 = load x86_mmx* %0, align 8
  %2 = load i32* %b, align 4
  %3 = tail call x86_mmx @llvm.x86.mmx.psrai.w(x86_mmx %1, i32 %2)
  %4 = bitcast x86_mmx %3 to i64
  ret i64 %4
}
declare x86_mmx @llvm.x86.mmx.psrai.w(x86_mmx, i32)

define i64 @t7(<1 x i64>* %a, i32* %b) {
; CHECK-LABEL: t7:
; CHECK:       # BB#0:{{.*}} %entry
; CHECK:         movq (%[[REG1]]), %mm0
; CHECK-NEXT:    psrad (%[[REG2]]), %mm0
; CHECK-NEXT:    movd %mm0, %rax
; CHECK-NEXT:    retq
entry:
  %0 = bitcast <1 x i64>* %a to x86_mmx*
  %1 = load x86_mmx* %0, align 8
  %2 = load i32* %b, align 4
  %3 = tail call x86_mmx @llvm.x86.mmx.psrai.d(x86_mmx %1, i32 %2)
  %4 = bitcast x86_mmx %3 to i64
  ret i64 %4
}
declare x86_mmx @llvm.x86.mmx.psrai.d(x86_mmx, i32)
