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
