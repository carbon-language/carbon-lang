; RUN: llc < %s -mattr=+neon | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32"
target triple = "thumbv7-elf"

define arm_aapcs_vfpcc void @test_vget_laneu16() nounwind {
entry:
; CHECK: vmov.u16 r0, d0[1]
  %arg0_uint16x4_t = alloca <4 x i16>             ; <<4 x i16>*> [#uses=1]
  %out_uint16_t = alloca i16                      ; <i16*> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %0 = load <4 x i16>* %arg0_uint16x4_t, align 8  ; <<4 x i16>> [#uses=1]
  %1 = extractelement <4 x i16> %0, i32 1         ; <i16> [#uses=1]
  store i16 %1, i16* %out_uint16_t, align 2
  br label %return

return:                                           ; preds = %entry
  ret void
}

define arm_aapcs_vfpcc void @test_vget_laneu8() nounwind {
entry:
; CHECK: vmov.u8 r0, d0[1]
  %arg0_uint8x8_t = alloca <8 x i8>               ; <<8 x i8>*> [#uses=1]
  %out_uint8_t = alloca i8                        ; <i8*> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %0 = load <8 x i8>* %arg0_uint8x8_t, align 8    ; <<8 x i8>> [#uses=1]
  %1 = extractelement <8 x i8> %0, i32 1          ; <i8> [#uses=1]
  store i8 %1, i8* %out_uint8_t, align 1
  br label %return

return:                                           ; preds = %entry
  ret void
}

define arm_aapcs_vfpcc void @test_vgetQ_laneu16() nounwind {
entry:
; CHECK: vmov.u16 r0, d0[1]
  %arg0_uint16x8_t = alloca <8 x i16>             ; <<8 x i16>*> [#uses=1]
  %out_uint16_t = alloca i16                      ; <i16*> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %0 = load <8 x i16>* %arg0_uint16x8_t, align 16 ; <<8 x i16>> [#uses=1]
  %1 = extractelement <8 x i16> %0, i32 1         ; <i16> [#uses=1]
  store i16 %1, i16* %out_uint16_t, align 2
  br label %return

return:                                           ; preds = %entry
  ret void
}

define arm_aapcs_vfpcc void @test_vgetQ_laneu8() nounwind {
entry:
; CHECK: vmov.u8 r0, d0[1]
  %arg0_uint8x16_t = alloca <16 x i8>             ; <<16 x i8>*> [#uses=1]
  %out_uint8_t = alloca i8                        ; <i8*> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %0 = load <16 x i8>* %arg0_uint8x16_t, align 16 ; <<16 x i8>> [#uses=1]
  %1 = extractelement <16 x i8> %0, i32 1         ; <i8> [#uses=1]
  store i8 %1, i8* %out_uint8_t, align 1
  br label %return

return:                                           ; preds = %entry
  ret void
}
