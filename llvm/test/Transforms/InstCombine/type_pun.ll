; RUN: opt < %s -instcombine -S | FileCheck %s

; Ensure that type punning using a union of vector and same-sized array
; generates an extract instead of a shuffle with an uncommon vector size:
;
;   typedef uint32_t v4i32 __attribute__((vector_size(16)));
;   union { v4i32 v; uint32_t a[4]; };
;
; This cleans up behind SROA, which inserts the uncommon vector size when
; cleaning up the alloca/store/GEP/load.


; Extracting the zeroth element in an i32 array.
define i32 @type_pun_zeroth(<16 x i8> %in) {
; CHECK-LABEL: @type_pun_zeroth(
; CHECK-NEXT: %[[BC:.*]] = bitcast <16 x i8> %in to <4 x i32>
; CHECK-NEXT: %[[EXT:.*]] = extractelement <4 x i32> %[[BC]], i32 0
; CHECK-NEXT: ret i32 %[[EXT]]
  %sroa = shufflevector <16 x i8> %in, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1 = bitcast <4 x i8> %sroa to i32
  ret i32 %1
}

; Extracting the first element in an i32 array.
define i32 @type_pun_first(<16 x i8> %in) {
; CHECK-LABEL: @type_pun_first(
; CHECK-NEXT: %[[BC:.*]] = bitcast <16 x i8> %in to <4 x i32>
; CHECK-NEXT: %[[EXT:.*]] = extractelement <4 x i32> %[[BC]], i32 1
; CHECK-NEXT: ret i32 %[[EXT]]
  %sroa = shufflevector <16 x i8> %in, <16 x i8> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1 = bitcast <4 x i8> %sroa to i32
  ret i32 %1
}

; Extracting an i32 that isn't aligned to any natural boundary.
define i32 @type_pun_misaligned(<16 x i8> %in) {
; CHECK-LABEL: @type_pun_misaligned(
; CHECK-NEXT: %[[SHUF:.*]] = shufflevector <16 x i8> %in, <16 x i8> undef, <16 x i32> <i32 6, i32 7, i32 8, i32 9, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT: %[[BC:.*]] = bitcast <16 x i8> %[[SHUF]] to <4 x i32>
; CHECK-NEXT: %[[EXT:.*]] = extractelement <4 x i32> %[[BC]], i32 0
; CHECK-NEXT: ret i32 %[[EXT]]
  %sroa = shufflevector <16 x i8> %in, <16 x i8> undef, <4 x i32> <i32 6, i32 7, i32 8, i32 9>
  %1 = bitcast <4 x i8> %sroa to i32
  ret i32 %1
}

; Type punning to an array of pointers.
define i32* @type_pun_pointer(<16 x i8> %in) {
; CHECK-LABEL: @type_pun_pointer(
; CHECK-NEXT: %[[BC:.*]] = bitcast <16 x i8> %in to <4 x i32>
; CHECK-NEXT: %[[EXT:.*]] = extractelement <4 x i32> %[[BC]], i32 0
; CHECK-NEXT: %[[I2P:.*]] = inttoptr i32 %[[EXT]] to i32*
; CHECK-NEXT: ret i32* %[[I2P]]
  %sroa = shufflevector <16 x i8> %in, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1 = bitcast <4 x i8> %sroa to i32
  %2 = inttoptr i32 %1 to i32*
  ret i32* %2
}

; Type punning to an array of 32-bit floating-point values.
define float @type_pun_float(<16 x i8> %in) {
; CHECK-LABEL: @type_pun_float(
; CHECK-NEXT: %[[BC:.*]] = bitcast <16 x i8> %in to <4 x float>
; CHECK-NEXT: %[[EXT:.*]] = extractelement <4 x float> %[[BC]], i32 0
; CHECK-NEXT: ret float %[[EXT]]
  %sroa = shufflevector <16 x i8> %in, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1 = bitcast <4 x i8> %sroa to float
  ret float %1
}

; Type punning to an array of 64-bit floating-point values.
define double @type_pun_double(<16 x i8> %in) {
; CHECK-LABEL: @type_pun_double(
; CHECK-NEXT: %[[BC:.*]] = bitcast <16 x i8> %in to <2 x double>
; CHECK-NEXT: %[[EXT:.*]] = extractelement <2 x double> %[[BC]], i32 0
; CHECK-NEXT: ret double %[[EXT]]
  %sroa = shufflevector <16 x i8> %in, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1 = bitcast <8 x i8> %sroa to double
  ret double %1
}

; Type punning to same-size floating-point and integer values.
; Verify that multiple uses with different bitcast types are properly handled.
define { float, i32 } @type_pun_float_i32(<16 x i8> %in) {
; CHECK-LABEL: @type_pun_float_i32(
; CHECK-NEXT: %[[BCI:.*]] = bitcast <16 x i8> %in to <4 x i32>
; CHECK-NEXT: %[[EXTI:.*]] = extractelement <4 x i32> %[[BCI]], i32 0
; CHECK-NEXT: %[[BCF:.*]] = bitcast <16 x i8> %in to <4 x float>
; CHECK-NEXT: %[[EXTF:.*]] = extractelement <4 x float> %[[BCF]], i32 0
; CHECK-NEXT: %1 = insertvalue { float, i32 } undef, float %[[EXTF]], 0
; CHECK-NEXT: %2 = insertvalue { float, i32 } %1, i32 %[[EXTI]], 1
; CHECK-NEXT: ret { float, i32 } %2
  %sroa = shufflevector <16 x i8> %in, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %f = bitcast <4 x i8> %sroa to float
  %i = bitcast <4 x i8> %sroa to i32
  %1 = insertvalue { float, i32 } undef, float %f, 0
  %2 = insertvalue { float, i32 } %1, i32 %i, 1
  ret { float, i32 } %2
}

; Type punning two i32 values, with control flow.
; Verify that the bitcast is shared and dominates usage.
define i32 @type_pun_i32_ctrl(<16 x i8> %in) {
; CHECK-LABEL: @type_pun_i32_ctrl(
entry: ; CHECK-NEXT: entry:
; CHECK-NEXT: %[[BC:.*]] = bitcast <16 x i8> %in to <4 x i32>
; CHECK-NEXT: br
  %sroa = shufflevector <16 x i8> %in, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  br i1 undef, label %left, label %right
left: ; CHECK: left:
; CHECK-NEXT: %[[EXTL:.*]] = extractelement <4 x i32> %[[BC]], i32 0
; CHECK-NEXT: br
  %lhs = bitcast <4 x i8> %sroa to i32
  br label %tail
right: ; CHECK: right:
; CHECK-NEXT: %[[EXTR:.*]] = extractelement <4 x i32> %[[BC]], i32 0
; CHECK-NEXT: br
  %rhs = bitcast <4 x i8> %sroa to i32
  br label %tail
tail: ; CHECK: tail:
; CHECK-NEXT: %i = phi i32 [ %[[EXTL]], %left ], [ %[[EXTR]], %right ]
; CHECK-NEXT: ret i32 %i
  %i = phi i32 [ %lhs, %left ], [ %rhs, %right ]
  ret i32 %i
}

; Extracting a type that won't fit in a vector isn't handled. The function
; should stay the same.
define i40 @type_pun_unhandled(<16 x i8> %in) {
; CHECK-LABEL: @type_pun_unhandled(
; CHECK-NEXT: %sroa = shufflevector <16 x i8> %in, <16 x i8> undef, <5 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8>
; CHECK-NEXT: %1 = bitcast <5 x i8> %sroa to i40
; CHECK-NEXT: ret i40 %1
  %sroa = shufflevector <16 x i8> %in, <16 x i8> undef, <5 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8>
  %1 = bitcast <5 x i8> %sroa to i40
  ret i40 %1
}
