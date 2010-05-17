; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s
; Implementing vld / vst as REG_SEQUENCE eliminates the extra vmov's.

%struct.int16x8_t = type { <8 x i16> }
%struct.int32x4_t = type { <4 x i32> }
%struct.__neon_int8x8x2_t = type { <8 x i8>, <8 x i8> }
%struct.__neon_int8x8x3_t = type { <8 x i8>,  <8 x i8>,  <8 x i8> }
%struct.__neon_int16x8x2_t = type { <8 x i16>, <8 x i16> }
%struct.__neon_int32x4x2_t = type { <4 x i32>, <4 x i32> }

define arm_apcscc void @t1(i16* %i_ptr, i16* %o_ptr, %struct.int32x4_t* nocapture %vT0ptr, %struct.int32x4_t* nocapture %vT1ptr) nounwind {
entry:
; CHECK:        t1:
; CHECK:        vld1.16
; CHECK-NOT:    vmov d
; CHECK:        vmovl.s16
; CHECK:        vshrn.i32
; CHECK:        vshrn.i32
; CHECK-NOT:    vmov d
; CHECK-NEXT:   vst1.16
  %0 = getelementptr inbounds %struct.int32x4_t* %vT0ptr, i32 0, i32 0 ; <<4 x i32>*> [#uses=1]
  %1 = load <4 x i32>* %0, align 16               ; <<4 x i32>> [#uses=1]
  %2 = getelementptr inbounds %struct.int32x4_t* %vT1ptr, i32 0, i32 0 ; <<4 x i32>*> [#uses=1]
  %3 = load <4 x i32>* %2, align 16               ; <<4 x i32>> [#uses=1]
  %4 = bitcast i16* %i_ptr to i8*                 ; <i8*> [#uses=1]
  %5 = tail call <8 x i16> @llvm.arm.neon.vld1.v8i16(i8* %4) ; <<8 x i16>> [#uses=1]
  %6 = bitcast <8 x i16> %5 to <2 x double>       ; <<2 x double>> [#uses=2]
  %7 = extractelement <2 x double> %6, i32 0      ; <double> [#uses=1]
  %8 = bitcast double %7 to <4 x i16>             ; <<4 x i16>> [#uses=1]
  %9 = tail call <4 x i32> @llvm.arm.neon.vmovls.v4i32(<4 x i16> %8) ; <<4 x i32>> [#uses=1]
  %10 = extractelement <2 x double> %6, i32 1     ; <double> [#uses=1]
  %11 = bitcast double %10 to <4 x i16>           ; <<4 x i16>> [#uses=1]
  %12 = tail call <4 x i32> @llvm.arm.neon.vmovls.v4i32(<4 x i16> %11) ; <<4 x i32>> [#uses=1]
  %13 = mul <4 x i32> %1, %9                      ; <<4 x i32>> [#uses=1]
  %14 = mul <4 x i32> %3, %12                     ; <<4 x i32>> [#uses=1]
  %15 = tail call <4 x i16> @llvm.arm.neon.vshiftn.v4i16(<4 x i32> %13, <4 x i32> <i32 -12, i32 -12, i32 -12, i32 -12>) ; <<4 x i16>> [#uses=1]
  %16 = tail call <4 x i16> @llvm.arm.neon.vshiftn.v4i16(<4 x i32> %14, <4 x i32> <i32 -12, i32 -12, i32 -12, i32 -12>) ; <<4 x i16>> [#uses=1]
  %17 = shufflevector <4 x i16> %15, <4 x i16> %16, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7> ; <<8 x i16>> [#uses=1]
  %18 = bitcast i16* %o_ptr to i8*                ; <i8*> [#uses=1]
  tail call void @llvm.arm.neon.vst1.v8i16(i8* %18, <8 x i16> %17)
  ret void
}

define arm_apcscc void @t2(i16* %i_ptr, i16* %o_ptr, %struct.int16x8_t* nocapture %vT0ptr, %struct.int16x8_t* nocapture %vT1ptr) nounwind {
entry:
; CHECK:        t2:
; CHECK:        vld1.16
; CHECK-NOT:    vmov
; CHECK:        vmul.i16
; CHECK:        vld1.16
; CHECK:        vst1.16
; CHECK-NOT:    vmov
; CHECK:        vmul.i16
; CHECK:        vst1.16
  %0 = getelementptr inbounds %struct.int16x8_t* %vT0ptr, i32 0, i32 0 ; <<8 x i16>*> [#uses=1]
  %1 = load <8 x i16>* %0, align 16               ; <<8 x i16>> [#uses=1]
  %2 = getelementptr inbounds %struct.int16x8_t* %vT1ptr, i32 0, i32 0 ; <<8 x i16>*> [#uses=1]
  %3 = load <8 x i16>* %2, align 16               ; <<8 x i16>> [#uses=1]
  %4 = bitcast i16* %i_ptr to i8*                 ; <i8*> [#uses=1]
  %5 = tail call <8 x i16> @llvm.arm.neon.vld1.v8i16(i8* %4) ; <<8 x i16>> [#uses=1]
  %6 = getelementptr inbounds i16* %i_ptr, i32 8  ; <i16*> [#uses=1]
  %7 = bitcast i16* %6 to i8*                     ; <i8*> [#uses=1]
  %8 = tail call <8 x i16> @llvm.arm.neon.vld1.v8i16(i8* %7) ; <<8 x i16>> [#uses=1]
  %9 = mul <8 x i16> %1, %5                       ; <<8 x i16>> [#uses=1]
  %10 = mul <8 x i16> %3, %8                      ; <<8 x i16>> [#uses=1]
  %11 = bitcast i16* %o_ptr to i8*                ; <i8*> [#uses=1]
  tail call void @llvm.arm.neon.vst1.v8i16(i8* %11, <8 x i16> %9)
  %12 = getelementptr inbounds i16* %o_ptr, i32 8 ; <i16*> [#uses=1]
  %13 = bitcast i16* %12 to i8*                   ; <i8*> [#uses=1]
  tail call void @llvm.arm.neon.vst1.v8i16(i8* %13, <8 x i16> %10)
  ret void
}

define <8 x i8> @t3(i8* %A, i8* %B) nounwind {
; CHECK:        t3:
; CHECK:        vld3.8
; CHECK:        vmul.i8
; CHECK-NOT:    vmov
; CHECK:        vst3.8
  %tmp1 = call %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8(i8* %A) ; <%struct.__neon_int8x8x3_t> [#uses=2]
  %tmp2 = extractvalue %struct.__neon_int8x8x3_t %tmp1, 0 ; <<8 x i8>> [#uses=1]
  %tmp3 = extractvalue %struct.__neon_int8x8x3_t %tmp1, 2 ; <<8 x i8>> [#uses=1]
  %tmp4 = extractvalue %struct.__neon_int8x8x3_t %tmp1, 1 ; <<8 x i8>> [#uses=1]
  %tmp5 = sub <8 x i8> %tmp3, %tmp4
  %tmp6 = add <8 x i8> %tmp2, %tmp3               ; <<8 x i8>> [#uses=1]
  %tmp7 = mul <8 x i8> %tmp4, %tmp2
  tail call void @llvm.arm.neon.vst3.v8i8(i8* %B, <8 x i8> %tmp5, <8 x i8> %tmp6, <8 x i8> %tmp7)
  ret <8 x i8> %tmp4
}

define arm_apcscc void @t4(i32* %in, i32* %out) nounwind {
entry:
; CHECK:        t4:
; CHECK:        vld2.32
; CHECK-NOT:    vmov
; CHECK:        vld2.32
; CHECK-NOT:    vmov
; CHECK:        bne
  %tmp1 = bitcast i32* %in to i8*                 ; <i8*> [#uses=1]
  %tmp2 = tail call %struct.__neon_int32x4x2_t @llvm.arm.neon.vld2.v4i32(i8* %tmp1) ; <%struct.__neon_int32x4x2_t> [#uses=2]
  %tmp3 = getelementptr inbounds i32* %in, i32 8  ; <i32*> [#uses=1]
  %tmp4 = bitcast i32* %tmp3 to i8*               ; <i8*> [#uses=1]
  %tmp5 = tail call %struct.__neon_int32x4x2_t @llvm.arm.neon.vld2.v4i32(i8* %tmp4) ; <%struct.__neon_int32x4x2_t> [#uses=2]
  %tmp8 = bitcast i32* %out to i8*                ; <i8*> [#uses=1]
  br i1 undef, label %return1, label %return2

return1:
; CHECK:        %return1
; CHECK-NOT:    vmov
; CHECK-NEXT:   vadd.i32
; CHECK-NEXT:   vadd.i32
; CHECK-NEXT:   vst2.32
  %tmp52 = extractvalue %struct.__neon_int32x4x2_t %tmp2, 0 ; <<4 x i32>> [#uses=1]
  %tmp57 = extractvalue %struct.__neon_int32x4x2_t %tmp2, 1 ; <<4 x i32>> [#uses=1]
  %tmp = extractvalue %struct.__neon_int32x4x2_t %tmp5, 0 ; <<4 x i32>> [#uses=1]
  %tmp39 = extractvalue %struct.__neon_int32x4x2_t %tmp5, 1 ; <<4 x i32>> [#uses=1]
  %tmp6 = add <4 x i32> %tmp52, %tmp              ; <<4 x i32>> [#uses=1]
  %tmp7 = add <4 x i32> %tmp57, %tmp39            ; <<4 x i32>> [#uses=1]
  tail call void @llvm.arm.neon.vst2.v4i32(i8* %tmp8, <4 x i32> %tmp6, <4 x i32> %tmp7)
  ret void

return2:
; CHECK:        %return2
; CHECK:        vadd.i32
; CHECK:        vmov q1, q3
; CHECK-NOT:    vmov
; CHECK:        vst2.32 {d0, d1, d2, d3}
  %tmp100 = extractvalue %struct.__neon_int32x4x2_t %tmp2, 0 ; <<4 x i32>> [#uses=1]
  %tmp101 = extractvalue %struct.__neon_int32x4x2_t %tmp5, 1 ; <<4 x i32>> [#uses=1]
  %tmp102 = add <4 x i32> %tmp100, %tmp101              ; <<4 x i32>> [#uses=1]
  tail call void @llvm.arm.neon.vst2.v4i32(i8* %tmp8, <4 x i32> %tmp102, <4 x i32> %tmp101)
  call void @llvm.trap()
  unreachable
}

define <8 x i16> @t5(i16* %A, <8 x i16>* %B) nounwind {
; CHECK:        t5:
; CHECK:        vldmia
; CHECK:        vmov q1, q0
; CHECK-NOT:    vmov
; CHECK:        vld2.16 {d0[1], d2[1]}, [r0]
; CHECK-NOT:    vmov
; CHECK:        vadd.i16
  %tmp0 = bitcast i16* %A to i8*                  ; <i8*> [#uses=1]
  %tmp1 = load <8 x i16>* %B                      ; <<8 x i16>> [#uses=2]
  %tmp2 = call %struct.__neon_int16x8x2_t @llvm.arm.neon.vld2lane.v8i16(i8* %tmp0, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 1) ; <%struct.__neon_int16x8x2_t> [#uses=2]
  %tmp3 = extractvalue %struct.__neon_int16x8x2_t %tmp2, 0 ; <<8 x i16>> [#uses=1]
  %tmp4 = extractvalue %struct.__neon_int16x8x2_t %tmp2, 1 ; <<8 x i16>> [#uses=1]
  %tmp5 = add <8 x i16> %tmp3, %tmp4              ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %tmp5
}

define <8 x i8> @t6(i8* %A, <8 x i8>* %B) nounwind {
; CHECK:        t6:
; CHECK:        vldr.64
; CHECK:        vmov d1, d0
; CHECK-NEXT:   vld2.8 {d0[1], d1[1]}
  %tmp1 = load <8 x i8>* %B                       ; <<8 x i8>> [#uses=2]
  %tmp2 = call %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2lane.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1) ; <%struct.__neon_int8x8x2_t> [#uses=2]
  %tmp3 = extractvalue %struct.__neon_int8x8x2_t %tmp2, 0 ; <<8 x i8>> [#uses=1]
  %tmp4 = extractvalue %struct.__neon_int8x8x2_t %tmp2, 1 ; <<8 x i8>> [#uses=1]
  %tmp5 = add <8 x i8> %tmp3, %tmp4               ; <<8 x i8>> [#uses=1]
  ret <8 x i8> %tmp5
}

define arm_apcscc void @t7(i32* %iptr, i32* %optr) nounwind {
entry:
; CHECK:        t7:
; CHECK:        vld2.32
; CHECK:        vst2.32
; CHECK:        vld1.32 {d0, d1},
; CHECK:        vmov q1, q0
; CHECK-NOT:    vmov
; CHECK:        vuzp.32 q0, q1
; CHECK:        vst1.32
  %0 = bitcast i32* %iptr to i8*                  ; <i8*> [#uses=2]
  %1 = tail call %struct.__neon_int32x4x2_t @llvm.arm.neon.vld2.v4i32(i8* %0) ; <%struct.__neon_int32x4x2_t> [#uses=2]
  %tmp57 = extractvalue %struct.__neon_int32x4x2_t %1, 0 ; <<4 x i32>> [#uses=1]
  %tmp60 = extractvalue %struct.__neon_int32x4x2_t %1, 1 ; <<4 x i32>> [#uses=1]
  %2 = bitcast i32* %optr to i8*                  ; <i8*> [#uses=2]
  tail call void @llvm.arm.neon.vst2.v4i32(i8* %2, <4 x i32> %tmp57, <4 x i32> %tmp60)
  %3 = tail call <4 x i32> @llvm.arm.neon.vld1.v4i32(i8* %0) ; <<4 x i32>> [#uses=1]
  %4 = shufflevector <4 x i32> %3, <4 x i32> undef, <4 x i32> <i32 0, i32 2, i32 0, i32 2> ; <<4 x i32>> [#uses=1]
  tail call void @llvm.arm.neon.vst1.v4i32(i8* %2, <4 x i32> %4)
  ret void
}

declare <4 x i32> @llvm.arm.neon.vld1.v4i32(i8*) nounwind readonly

declare <8 x i16> @llvm.arm.neon.vld1.v8i16(i8*) nounwind readonly

declare <4 x i32> @llvm.arm.neon.vmovls.v4i32(<4 x i16>) nounwind readnone

declare <4 x i16> @llvm.arm.neon.vshiftn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone

declare void @llvm.arm.neon.vst1.v4i32(i8*, <4 x i32>) nounwind

declare void @llvm.arm.neon.vst1.v8i16(i8*, <8 x i16>) nounwind

declare void @llvm.arm.neon.vst3.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>) nounwind

declare %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8(i8*) nounwind readonly

declare %struct.__neon_int32x4x2_t @llvm.arm.neon.vld2.v4i32(i8*) nounwind readonly

declare %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2lane.v8i8(i8*, <8 x i8>, <8 x i8>, i32) nounwind readonly

declare %struct.__neon_int16x8x2_t @llvm.arm.neon.vld2lane.v8i16(i8*, <8 x i16>, <8 x i16>, i32) nounwind readonly

declare void @llvm.arm.neon.vst2.v4i32(i8*, <4 x i32>, <4 x i32>) nounwind

declare void @llvm.trap() nounwind
