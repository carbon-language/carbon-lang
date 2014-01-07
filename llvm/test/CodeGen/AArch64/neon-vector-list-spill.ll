; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon -fp-contract=fast

; FIXME: We should not generate ld/st for such register spill/fill, because the
; test case seems very simple and the register pressure is not high. If the
; spill/fill algorithm is optimized, this test case may not be triggered. And
; then we can delete it.
define i32 @spill.DPairReg(i8* %arg1, i32 %arg2) {
; CHECK-LABEL: spill.DPairReg:
; CHECK: ld2 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
; CHECK: st1 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
; CHECK: ld1 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
entry:
  %vld = tail call { <2 x i32>, <2 x i32> } @llvm.arm.neon.vld2.v2i32(i8* %arg1, i32 4)
  %cmp = icmp eq i32 %arg2, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @foo()
  br label %if.end

if.end:
  %vld.extract = extractvalue { <2 x i32>, <2 x i32> } %vld, 0
  %res = extractelement <2 x i32> %vld.extract, i32 1
  ret i32 %res
}

define i16 @spill.DTripleReg(i8* %arg1, i32 %arg2) {
; CHECK-LABEL: spill.DTripleReg:
; CHECK: ld3 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
; CHECK: st1 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
; CHECK: ld1 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
entry:
  %vld = tail call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld3.v4i16(i8* %arg1, i32 4)
  %cmp = icmp eq i32 %arg2, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @foo()
  br label %if.end

if.end:
  %vld.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %vld, 0
  %res = extractelement <4 x i16> %vld.extract, i32 1
  ret i16 %res
}

define i16 @spill.DQuadReg(i8* %arg1, i32 %arg2) {
; CHECK-LABEL: spill.DQuadReg:
; CHECK: ld4 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
; CHECK: st1 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
; CHECK: ld1 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
entry:
  %vld = tail call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld4.v4i16(i8* %arg1, i32 4)
  %cmp = icmp eq i32 %arg2, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @foo()
  br label %if.end

if.end:
  %vld.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %vld, 0
  %res = extractelement <4 x i16> %vld.extract, i32 0
  ret i16 %res
}

define i32 @spill.QPairReg(i8* %arg1, i32 %arg2) {
; CHECK-LABEL: spill.QPairReg:
; CHECK: ld3 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
; CHECK: st1 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
; CHECK: ld1 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
entry:
  %vld = tail call { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2.v4i32(i8* %arg1, i32 4)
  %cmp = icmp eq i32 %arg2, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @foo()
  br label %if.end

if.end:
  %vld.extract = extractvalue { <4 x i32>, <4 x i32> } %vld, 0
  %res = extractelement <4 x i32> %vld.extract, i32 1
  ret i32 %res
}

define float @spill.QTripleReg(i8* %arg1, i32 %arg2) {
; CHECK-LABEL: spill.QTripleReg:
; CHECK: ld3 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
; CHECK: st1 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
; CHECK: ld1 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
entry:
  %vld3 = tail call { <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld3.v4f32(i8* %arg1, i32 4)
  %cmp = icmp eq i32 %arg2, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @foo()
  br label %if.end

if.end:
  %vld3.extract = extractvalue { <4 x float>, <4 x float>, <4 x float> } %vld3, 0
  %res = extractelement <4 x float> %vld3.extract, i32 1
  ret float %res
}

define i8 @spill.QQuadReg(i8* %arg1, i32 %arg2) {
; CHECK-LABEL: spill.QQuadReg:
; CHECK: ld4 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
; CHECK: st1 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
; CHECK: ld1 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
entry:
  %vld = tail call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld4.v16i8(i8* %arg1, i32 4)
  %cmp = icmp eq i32 %arg2, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @foo()
  br label %if.end

if.end:
  %vld.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld, 0
  %res = extractelement <16 x i8> %vld.extract, i32 1
  ret i8 %res
}

declare { <2 x i32>, <2 x i32> } @llvm.arm.neon.vld2.v2i32(i8*, i32)
declare { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld3.v4i16(i8*, i32)
declare { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld4.v4i16(i8*, i32)
declare { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2.v4i32(i8*, i32)
declare { <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld3.v4f32(i8*, i32)
declare { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld4.v16i8(i8*, i32)

declare void @foo()

; FIXME: We should not generate ld/st for such register spill/fill, because the
; test case seems very simple and the register pressure is not high. If the
; spill/fill algorithm is optimized, this test case may not be triggered. And
; then we can delete it.
; check the spill for Register Class QPair_with_qsub_0_in_FPR128Lo
define <8 x i16> @test_2xFPR128Lo(i64 %got, i8* %ptr, <1 x i64> %a) {
  tail call void @llvm.arm.neon.vst2lane.v1i64(i8* %ptr, <1 x i64> zeroinitializer, <1 x i64> zeroinitializer, i32 0, i32 8)
  tail call void @foo()
  %sv = shufflevector <1 x i64> zeroinitializer, <1 x i64> %a, <2 x i32> <i32 0, i32 1>
  %1 = bitcast <2 x i64> %sv to <8 x i16>
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %3 = mul <8 x i16> %2, %2
  ret <8 x i16> %3
}

; check the spill for Register Class QTriple_with_qsub_0_in_FPR128Lo
define <8 x i16> @test_3xFPR128Lo(i64 %got, i8* %ptr, <1 x i64> %a) {
  tail call void @llvm.arm.neon.vst3lane.v1i64(i8* %ptr, <1 x i64> zeroinitializer, <1 x i64> zeroinitializer, <1 x i64> zeroinitializer, i32 0, i32 8)
  tail call void @foo()
  %sv = shufflevector <1 x i64> zeroinitializer, <1 x i64> %a, <2 x i32> <i32 0, i32 1>
  %1 = bitcast <2 x i64> %sv to <8 x i16>
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %3 = mul <8 x i16> %2, %2
  ret <8 x i16> %3
}

; check the spill for Register Class QQuad_with_qsub_0_in_FPR128Lo
define <8 x i16> @test_4xFPR128Lo(i64 %got, i8* %ptr, <1 x i64> %a) {
  tail call void @llvm.arm.neon.vst4lane.v1i64(i8* %ptr, <1 x i64> zeroinitializer, <1 x i64> zeroinitializer, <1 x i64> zeroinitializer, <1 x i64> zeroinitializer, i32 0, i32 8)
  tail call void @foo()
  %sv = shufflevector <1 x i64> zeroinitializer, <1 x i64> %a, <2 x i32> <i32 0, i32 1>
  %1 = bitcast <2 x i64> %sv to <8 x i16>
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %3 = mul <8 x i16> %2, %2
  ret <8 x i16> %3
}

declare void @llvm.arm.neon.vst2lane.v1i64(i8*, <1 x i64>, <1 x i64>, i32, i32)
declare void @llvm.arm.neon.vst3lane.v1i64(i8*, <1 x i64>, <1 x i64>, <1 x i64>, i32, i32)
declare void @llvm.arm.neon.vst4lane.v1i64(i8*, <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64>, i32, i32)