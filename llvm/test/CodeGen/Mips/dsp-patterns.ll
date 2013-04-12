; RUN: llc -march=mips -mattr=dsp < %s | FileCheck %s -check-prefix=R1
; RUN: llc -march=mips -mattr=dspr2 < %s | FileCheck %s -check-prefix=R2

; R1: test_lbux:
; R1: lbux ${{[0-9]+}}

define zeroext i8 @test_lbux(i8* nocapture %b, i32 %i) {
entry:
  %add.ptr = getelementptr inbounds i8* %b, i32 %i
  %0 = load i8* %add.ptr, align 1
  ret i8 %0
}

; R1: test_lhx:
; R1: lhx ${{[0-9]+}}

define signext i16 @test_lhx(i16* nocapture %b, i32 %i) {
entry:
  %add.ptr = getelementptr inbounds i16* %b, i32 %i
  %0 = load i16* %add.ptr, align 2
  ret i16 %0
}

; R1: test_lwx:
; R1: lwx ${{[0-9]+}}

define i32 @test_lwx(i32* nocapture %b, i32 %i) {
entry:
  %add.ptr = getelementptr inbounds i32* %b, i32 %i
  %0 = load i32* %add.ptr, align 4
  ret i32 %0
}

; R1: test_add_v2q15_:
; R1: addq.ph ${{[0-9]+}}

define { i32 } @test_add_v2q15_(i32 %a.coerce, i32 %b.coerce) {
entry:
  %0 = bitcast i32 %a.coerce to <2 x i16>
  %1 = bitcast i32 %b.coerce to <2 x i16>
  %add = add <2 x i16> %0, %1
  %2 = bitcast <2 x i16> %add to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; R1: test_sub_v2q15_:
; R1: subq.ph ${{[0-9]+}}

define { i32 } @test_sub_v2q15_(i32 %a.coerce, i32 %b.coerce) {
entry:
  %0 = bitcast i32 %a.coerce to <2 x i16>
  %1 = bitcast i32 %b.coerce to <2 x i16>
  %sub = sub <2 x i16> %0, %1
  %2 = bitcast <2 x i16> %sub to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; R2: test_mul_v2q15_:
; R2: mul.ph ${{[0-9]+}}

; mul.ph is an R2 instruction. Check that multiply node gets expanded.
; R1: test_mul_v2q15_:
; R1: mul ${{[0-9]+}}
; R1: mul ${{[0-9]+}}

define { i32 } @test_mul_v2q15_(i32 %a.coerce, i32 %b.coerce) {
entry:
  %0 = bitcast i32 %a.coerce to <2 x i16>
  %1 = bitcast i32 %b.coerce to <2 x i16>
  %mul = mul <2 x i16> %0, %1
  %2 = bitcast <2 x i16> %mul to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; R1: test_add_v4i8_:
; R1: addu.qb ${{[0-9]+}}

define { i32 } @test_add_v4i8_(i32 %a.coerce, i32 %b.coerce) {
entry:
  %0 = bitcast i32 %a.coerce to <4 x i8>
  %1 = bitcast i32 %b.coerce to <4 x i8>
  %add = add <4 x i8> %0, %1
  %2 = bitcast <4 x i8> %add to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; R1: test_sub_v4i8_:
; R1: subu.qb ${{[0-9]+}}

define { i32 } @test_sub_v4i8_(i32 %a.coerce, i32 %b.coerce) {
entry:
  %0 = bitcast i32 %a.coerce to <4 x i8>
  %1 = bitcast i32 %b.coerce to <4 x i8>
  %sub = sub <4 x i8> %0, %1
  %2 = bitcast <4 x i8> %sub to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; DSP-ASE doesn't have a v4i8 multiply instruction. Check that multiply node gets expanded.
; R2: test_mul_v4i8_:
; R2: mul ${{[0-9]+}}
; R2: mul ${{[0-9]+}}
; R2: mul ${{[0-9]+}}
; R2: mul ${{[0-9]+}}

define { i32 } @test_mul_v4i8_(i32 %a.coerce, i32 %b.coerce) {
entry:
  %0 = bitcast i32 %a.coerce to <4 x i8>
  %1 = bitcast i32 %b.coerce to <4 x i8>
  %mul = mul <4 x i8> %0, %1
  %2 = bitcast <4 x i8> %mul to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}
