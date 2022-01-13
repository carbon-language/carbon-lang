; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=avx512f -stop-after=finalize-isel | FileCheck %s

; CHECK: %[[REG1:.*]]:vr512_0_15 = COPY %1
; CHECK: %[[REG2:.*]]:vr512_0_15 = COPY %2
; CHECK: INLINEASM &"vpaddq\09$3, $2, $0 {$1}", 0 /* attdialect */, {{.*}}, def %{{.*}}, {{.*}}, %{{.*}}, {{.*}}, %[[REG1]], {{.*}}, %[[REG2]], 12 /* clobber */, implicit-def early-clobber $df, 12 /* clobber */, implicit-def early-clobber $fpsw, 12 /* clobber */, implicit-def early-clobber $eflags

define <8 x i64> @mask_Yk_i8(i8 signext %msk, <8 x i64> %x, <8 x i64> %y) {
entry:
  %0 = tail call <8 x i64> asm "vpaddq\09$3, $2, $0 {$1}", "=x,^Yk,x,x,~{dirflag},~{fpsr},~{flags}"(i8 %msk, <8 x i64> %x, <8 x i64> %y)
  ret <8 x i64> %0
}
