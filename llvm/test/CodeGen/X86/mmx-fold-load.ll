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

define i64 @tt0(x86_mmx %t, x86_mmx* %q) {
; CHECK-LABEL: tt0:
; CHECK:       # BB#0:{{.*}} %entry
; CHECK:    paddb (%[[REG3:[a-z]+]]), %mm0
; CHECK-NEXT:    movd %mm0, %rax
; CHECK-NEXT:    emms
; CHECK-NEXT:    retq
entry:
  %v = load x86_mmx* %q
  %u = tail call x86_mmx @llvm.x86.mmx.padd.b(x86_mmx %t, x86_mmx %v)
  %s = bitcast x86_mmx %u to i64
  call void @llvm.x86.mmx.emms()
  ret i64 %s
}
declare x86_mmx @llvm.x86.mmx.padd.b(x86_mmx, x86_mmx)
declare void @llvm.x86.mmx.emms()

define i64 @tt1(x86_mmx %t, x86_mmx* %q) {
; CHECK-LABEL: tt1:
; CHECK:       # BB#0:{{.*}} %entry
; CHECK:    paddw (%[[REG3]]), %mm0
; CHECK-NEXT:    movd %mm0, %rax
; CHECK-NEXT:    emms
; CHECK-NEXT:    retq
entry:
  %v = load x86_mmx* %q
  %u = tail call x86_mmx @llvm.x86.mmx.padd.w(x86_mmx %t, x86_mmx %v)
  %s = bitcast x86_mmx %u to i64
  call void @llvm.x86.mmx.emms()
  ret i64 %s
}
declare x86_mmx @llvm.x86.mmx.padd.w(x86_mmx, x86_mmx)

define i64 @tt2(x86_mmx %t, x86_mmx* %q) {
; CHECK-LABEL: tt2:
; CHECK:       # BB#0:{{.*}} %entry
; CHECK:    paddd (%[[REG3]]), %mm0
; CHECK-NEXT:    movd %mm0, %rax
; CHECK-NEXT:    emms
; CHECK-NEXT:    retq
entry:
  %v = load x86_mmx* %q
  %u = tail call x86_mmx @llvm.x86.mmx.padd.d(x86_mmx %t, x86_mmx %v)
  %s = bitcast x86_mmx %u to i64
  call void @llvm.x86.mmx.emms()
  ret i64 %s
}
declare x86_mmx @llvm.x86.mmx.padd.d(x86_mmx, x86_mmx)

define i64 @tt3(x86_mmx %t, x86_mmx* %q) {
; CHECK-LABEL: tt3:
; CHECK:       # BB#0:{{.*}} %entry
; CHECK:    paddq (%[[REG3]]), %mm0
; CHECK-NEXT:    movd %mm0, %rax
; CHECK-NEXT:    emms
; CHECK-NEXT:    retq
entry:
  %v = load x86_mmx* %q
  %u = tail call x86_mmx @llvm.x86.mmx.padd.q(x86_mmx %t, x86_mmx %v)
  %s = bitcast x86_mmx %u to i64
  call void @llvm.x86.mmx.emms()
  ret i64 %s
}
declare x86_mmx @llvm.x86.mmx.padd.q(x86_mmx, x86_mmx)

define i64 @tt4(x86_mmx %t, x86_mmx* %q) {
; CHECK-LABEL: tt4:
; CHECK:       # BB#0:{{.*}} %entry
; CHECK:    paddusb (%[[REG3]]), %mm0
; CHECK-NEXT:    movd %mm0, %rax
; CHECK-NEXT:    emms
; CHECK-NEXT:    retq
entry:
  %v = load x86_mmx* %q
  %u = tail call x86_mmx @llvm.x86.mmx.paddus.b(x86_mmx %t, x86_mmx %v)
  %s = bitcast x86_mmx %u to i64
  call void @llvm.x86.mmx.emms()
  ret i64 %s
}
declare x86_mmx @llvm.x86.mmx.paddus.b(x86_mmx, x86_mmx)

define i64 @tt5(x86_mmx %t, x86_mmx* %q) {
; CHECK-LABEL: tt5:
; CHECK:       # BB#0:{{.*}} %entry
; CHECK:    paddusw (%[[REG3]]), %mm0
; CHECK-NEXT:    movd %mm0, %rax
; CHECK-NEXT:    emms
; CHECK-NEXT:    retq
entry:
  %v = load x86_mmx* %q
  %u = tail call x86_mmx @llvm.x86.mmx.paddus.w(x86_mmx %t, x86_mmx %v)
  %s = bitcast x86_mmx %u to i64
  call void @llvm.x86.mmx.emms()
  ret i64 %s
}
declare x86_mmx @llvm.x86.mmx.paddus.w(x86_mmx, x86_mmx)

define i64 @tt6(x86_mmx %t, x86_mmx* %q) {
; CHECK-LABEL: tt6:
; CHECK:       # BB#0:{{.*}} %entry
; CHECK:    psrlw (%[[REG3]]), %mm0
; CHECK-NEXT:    movd %mm0, %rax
; CHECK-NEXT:    emms
; CHECK-NEXT:    retq
entry:
  %v = load x86_mmx* %q
  %u = tail call x86_mmx @llvm.x86.mmx.psrl.w(x86_mmx %t, x86_mmx %v)
  %s = bitcast x86_mmx %u to i64
  call void @llvm.x86.mmx.emms()
  ret i64 %s
}
declare x86_mmx @llvm.x86.mmx.psrl.w(x86_mmx, x86_mmx)

define i64 @tt7(x86_mmx %t, x86_mmx* %q) {
; CHECK-LABEL: tt7:
; CHECK:       # BB#0:{{.*}} %entry
; CHECK:    psrld (%[[REG3]]), %mm0
; CHECK-NEXT:    movd %mm0, %rax
; CHECK-NEXT:    emms
; CHECK-NEXT:    retq
entry:
  %v = load x86_mmx* %q
  %u = tail call x86_mmx @llvm.x86.mmx.psrl.d(x86_mmx %t, x86_mmx %v)
  %s = bitcast x86_mmx %u to i64
  call void @llvm.x86.mmx.emms()
  ret i64 %s
}
declare x86_mmx @llvm.x86.mmx.psrl.d(x86_mmx, x86_mmx)

define i64 @tt8(x86_mmx %t, x86_mmx* %q) {
; CHECK-LABEL: tt8:
; CHECK:       # BB#0:{{.*}} %entry
; CHECK:    psrlq (%[[REG3]]), %mm0
; CHECK-NEXT:    movd %mm0, %rax
; CHECK-NEXT:    emms
; CHECK-NEXT:    retq
entry:
  %v = load x86_mmx* %q
  %u = tail call x86_mmx @llvm.x86.mmx.psrl.q(x86_mmx %t, x86_mmx %v)
  %s = bitcast x86_mmx %u to i64
  call void @llvm.x86.mmx.emms()
  ret i64 %s
}
declare x86_mmx @llvm.x86.mmx.psrl.q(x86_mmx, x86_mmx)
