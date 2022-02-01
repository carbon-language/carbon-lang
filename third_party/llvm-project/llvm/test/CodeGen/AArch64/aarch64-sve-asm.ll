; RUN: llc < %s -mtriple aarch64-none-linux-gnu -mattr=+sve -stop-after=finalize-isel | FileCheck %s --check-prefix=CHECK

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-none-linux-gnu"

; Function Attrs: nounwind readnone
; CHECK: [[ARG1:%[0-9]+]]:zpr = COPY $z1
; CHECK: [[ARG2:%[0-9]+]]:zpr = COPY $z0
; CHECK: [[ARG3:%[0-9]+]]:zpr = COPY [[ARG2]]
; CHECK: [[ARG4:%[0-9]+]]:zpr_3b = COPY [[ARG1]]
; CHECK: INLINEASM {{.*}} [[ARG4]]
define <vscale x 16 x i8> @test_svadd_i8(<vscale x 16 x i8> %Zn, <vscale x 16 x i8> %Zm) {
  %1 = tail call <vscale x 16 x i8> asm "add $0.b, $1.b, $2.b", "=w,w,y"(<vscale x 16 x i8> %Zn, <vscale x 16 x i8> %Zm)
  ret <vscale x 16 x i8> %1
}

; Function Attrs: nounwind readnone
; CHECK: [[ARG1:%[0-9]+]]:zpr = COPY $z1
; CHECK: [[ARG2:%[0-9]+]]:zpr = COPY $z0
; CHECK: [[ARG3:%[0-9]+]]:zpr = COPY [[ARG2]]
; CHECK: [[ARG4:%[0-9]+]]:zpr_4b = COPY [[ARG1]]
; CHECK: INLINEASM {{.*}} [[ARG4]]
define <vscale x 2 x i64> @test_svsub_i64(<vscale x 2 x i64> %Zn, <vscale x 2 x i64> %Zm) {
  %1 = tail call <vscale x 2 x i64> asm "sub $0.d, $1.d, $2.d", "=w,w,x"(<vscale x 2 x i64> %Zn, <vscale x 2 x i64> %Zm)
  ret <vscale x 2 x i64> %1
}

; Function Attrs: nounwind readnone
; CHECK: [[ARG1:%[0-9]+]]:zpr = COPY $z1
; CHECK: [[ARG2:%[0-9]+]]:zpr = COPY $z0
; CHECK: [[ARG3:%[0-9]+]]:zpr = COPY [[ARG2]]
; CHECK: [[ARG4:%[0-9]+]]:zpr_3b = COPY [[ARG1]]
; CHECK: INLINEASM {{.*}} [[ARG4]]
define <vscale x 8 x half> @test_svfmul_f16(<vscale x 8 x half> %Zn, <vscale x 8 x half> %Zm) {
  %1 = tail call <vscale x 8 x half> asm "fmul $0.h, $1.h, $2.h", "=w,w,y"(<vscale x 8 x half> %Zn, <vscale x 8 x half> %Zm)
  ret <vscale x 8 x half> %1
}

; Function Attrs: nounwind readnone
; CHECK: [[ARG1:%[0-9]+]]:zpr = COPY $z1
; CHECK: [[ARG2:%[0-9]+]]:zpr = COPY $z0
; CHECK: [[ARG3:%[0-9]+]]:zpr = COPY [[ARG2]]
; CHECK: [[ARG4:%[0-9]+]]:zpr_4b = COPY [[ARG1]]
; CHECK: INLINEASM {{.*}} [[ARG4]]
define <vscale x 4 x float> @test_svfmul_f(<vscale x 4 x float> %Zn, <vscale x 4 x float> %Zm) {
  %1 = tail call <vscale x 4 x float> asm "fmul $0.s, $1.s, $2.s", "=w,w,x"(<vscale x 4 x float> %Zn, <vscale x 4 x float> %Zm)
  ret <vscale x 4 x float> %1
}

; Function Attrs: nounwind readnone
; CHECK: [[ARG1:%[0-9]+]]:zpr = COPY $z1
; CHECK: [[ARG2:%[0-9]+]]:zpr = COPY $z0
; CHECK: [[ARG3:%[0-9]+]]:ppr = COPY $p0
; CHECK: [[ARG4:%[0-9]+]]:ppr_3b = COPY [[ARG3]]
; CHECK: INLINEASM {{.*}} [[ARG4]]
define <vscale x 8 x half> @test_svfadd_f16(<vscale x 16 x i1> %Pg, <vscale x 8 x half> %Zn, <vscale x 8 x half> %Zm) {
  %1 = tail call <vscale x 8 x half> asm "fadd $0.h, $1/m, $2.h, $3.h", "=w,@3Upl,w,w"(<vscale x 16 x i1> %Pg, <vscale x 8 x half> %Zn, <vscale x 8 x half> %Zm)
  ret <vscale x 8 x half> %1
}

; Function Attrs: nounwind readnone
; CHECK: [[ARG1:%[0-9]+]]:zpr = COPY $z0
; CHECK: [[ARG2:%[0-9]+]]:ppr = COPY $p0
; CHECK: [[ARG3:%[0-9]+]]:ppr = COPY [[ARG2]]
; CHECK: [[ARG4:%[0-9]+]]:zpr = COPY [[ARG1]]
; CHECK: INLINEASM {{.*}} [[ARG3]]
define <vscale x 4 x i32> @test_incp(<vscale x 16 x i1> %Pg, <vscale x 4 x i32> %Zn) {
  %1 = tail call <vscale x 4 x i32> asm "incp $0.s, $1", "=w,@3Upa,0"(<vscale x 16 x i1> %Pg, <vscale x 4 x i32> %Zn)
  ret <vscale x 4 x i32> %1
}
