; RUN: llc < %s -march=x86-64 -mcpu=corei7 -mtriple=x86_64-unknown-linux-gnu | FileCheck %s -check-prefix=CHECK -check-prefix=SSE41
; RUN: llc < %s -march=x86-64 -mcpu=corei7-avx -mtriple=x86_64-unknown-linux-gnu | FileCheck %s -check-prefix=CHECK -check-prefix=AVX


define double @test1_add(double %A, double %B) {
  %1 = bitcast double %A to <2 x i32>
  %2 = bitcast double %B to <2 x i32>
  %add = add <2 x i32> %1, %2
  %3 = bitcast <2 x i32> %add to double
  ret double %3
}
; CHECK-LABEL: test1_add
; SSE41: paddd
; AVX: vpaddd
; CHECK-NEXT: ret


define double @test2_add(double %A, double %B) {
  %1 = bitcast double %A to <4 x i16>
  %2 = bitcast double %B to <4 x i16>
  %add = add <4 x i16> %1, %2
  %3 = bitcast <4 x i16> %add to double
  ret double %3
}
; CHECK-LABEL: test2_add
; SSE41: paddw
; AVX: vpaddw
; CHECK-NEXT: ret

define double @test3_add(double %A, double %B) {
  %1 = bitcast double %A to <8 x i8>
  %2 = bitcast double %B to <8 x i8>
  %add = add <8 x i8> %1, %2
  %3 = bitcast <8 x i8> %add to double
  ret double %3
}
; CHECK-LABEL: test3_add
; SSE41: paddb
; AVX: vpaddb
; CHECK-NEXT: ret


define double @test1_sub(double %A, double %B) {
  %1 = bitcast double %A to <2 x i32>
  %2 = bitcast double %B to <2 x i32>
  %sub = sub <2 x i32> %1, %2
  %3 = bitcast <2 x i32> %sub to double
  ret double %3
}
; CHECK-LABEL: test1_sub
; SSE41: psubd
; AVX: vpsubd
; CHECK-NEXT: ret


define double @test2_sub(double %A, double %B) {
  %1 = bitcast double %A to <4 x i16>
  %2 = bitcast double %B to <4 x i16>
  %sub = sub <4 x i16> %1, %2
  %3 = bitcast <4 x i16> %sub to double
  ret double %3
}
; CHECK-LABEL: test2_sub
; SSE41: psubw
; AVX: vpsubw
; CHECK-NEXT: ret


define double @test3_sub(double %A, double %B) {
  %1 = bitcast double %A to <8 x i8>
  %2 = bitcast double %B to <8 x i8>
  %sub = sub <8 x i8> %1, %2
  %3 = bitcast <8 x i8> %sub to double
  ret double %3
}
; CHECK-LABEL: test3_sub
; SSE41: psubb
; AVX: vpsubb
; CHECK-NEXT: ret


define double @test1_mul(double %A, double %B) {
  %1 = bitcast double %A to <2 x i32>
  %2 = bitcast double %B to <2 x i32>
  %mul = mul <2 x i32> %1, %2
  %3 = bitcast <2 x i32> %mul to double
  ret double %3
}
; CHECK-LABEL: test1_mul
; SSE41: pmulld
; AVX: vpmulld
; CHECK-NEXT: ret


define double @test2_mul(double %A, double %B) {
  %1 = bitcast double %A to <4 x i16>
  %2 = bitcast double %B to <4 x i16>
  %mul = mul <4 x i16> %1, %2
  %3 = bitcast <4 x i16> %mul to double
  ret double %3
}
; CHECK-LABEL: test2_mul
; SSE41: pmullw
; AVX: vpmullw
; CHECK-NEXT: ret

; There is no legal ISD::MUL with type MVT::v8i16.
define double @test3_mul(double %A, double %B) {
  %1 = bitcast double %A to <8 x i8>
  %2 = bitcast double %B to <8 x i8>
  %mul = mul <8 x i8> %1, %2
  %3 = bitcast <8 x i8> %mul to double
  ret double %3
}
; CHECK-LABEL: test3_mul
; CHECK: pmullw
; CHECK-NEXT: pshufb
; CHECK-NEXT: ret


define double @test1_and(double %A, double %B) {
  %1 = bitcast double %A to <2 x i32>
  %2 = bitcast double %B to <2 x i32>
  %and = and <2 x i32> %1, %2
  %3 = bitcast <2 x i32> %and to double
  ret double %3
}
; CHECK-LABEL: test1_and
; SSE41: andps
; AVX: vandps
; CHECK-NEXT: ret


define double @test2_and(double %A, double %B) {
  %1 = bitcast double %A to <4 x i16>
  %2 = bitcast double %B to <4 x i16>
  %and = and <4 x i16> %1, %2
  %3 = bitcast <4 x i16> %and to double
  ret double %3
}
; CHECK-LABEL: test2_and
; SSE41: andps
; AVX: vandps
; CHECK-NEXT: ret


define double @test3_and(double %A, double %B) {
  %1 = bitcast double %A to <8 x i8>
  %2 = bitcast double %B to <8 x i8>
  %and = and <8 x i8> %1, %2
  %3 = bitcast <8 x i8> %and to double
  ret double %3
}
; CHECK-LABEL: test3_and
; SSE41: andps
; AVX: vandps
; CHECK-NEXT: ret


define double @test1_or(double %A, double %B) {
  %1 = bitcast double %A to <2 x i32>
  %2 = bitcast double %B to <2 x i32>
  %or = or <2 x i32> %1, %2
  %3 = bitcast <2 x i32> %or to double
  ret double %3
}
; CHECK-LABEL: test1_or
; SSE41: orps
; AVX: vorps
; CHECK-NEXT: ret


define double @test2_or(double %A, double %B) {
  %1 = bitcast double %A to <4 x i16>
  %2 = bitcast double %B to <4 x i16>
  %or = or <4 x i16> %1, %2
  %3 = bitcast <4 x i16> %or to double
  ret double %3
}
; CHECK-LABEL: test2_or
; SSE41: orps
; AVX: vorps
; CHECK-NEXT: ret


define double @test3_or(double %A, double %B) {
  %1 = bitcast double %A to <8 x i8>
  %2 = bitcast double %B to <8 x i8>
  %or = or <8 x i8> %1, %2
  %3 = bitcast <8 x i8> %or to double
  ret double %3
}
; CHECK-LABEL: test3_or
; SSE41: orps
; AVX: vorps
; CHECK-NEXT: ret


define double @test1_xor(double %A, double %B) {
  %1 = bitcast double %A to <2 x i32>
  %2 = bitcast double %B to <2 x i32>
  %xor = xor <2 x i32> %1, %2
  %3 = bitcast <2 x i32> %xor to double
  ret double %3
}
; CHECK-LABEL: test1_xor
; SSE41: xorps
; AVX: vxorps
; CHECK-NEXT: ret


define double @test2_xor(double %A, double %B) {
  %1 = bitcast double %A to <4 x i16>
  %2 = bitcast double %B to <4 x i16>
  %xor = xor <4 x i16> %1, %2
  %3 = bitcast <4 x i16> %xor to double
  ret double %3
}
; CHECK-LABEL: test2_xor
; SSE41: xorps
; AVX: vxorps
; CHECK-NEXT: ret


define double @test3_xor(double %A, double %B) {
  %1 = bitcast double %A to <8 x i8>
  %2 = bitcast double %B to <8 x i8>
  %xor = xor <8 x i8> %1, %2
  %3 = bitcast <8 x i8> %xor to double
  ret double %3
}
; CHECK-LABEL: test3_xor
; SSE41: xorps
; AVX: vxorps
; CHECK-NEXT: ret


define double @test_fadd(double %A, double %B) {
  %1 = bitcast double %A to <2 x float>
  %2 = bitcast double %B to <2 x float>
  %add = fadd <2 x float> %1, %2
  %3 = bitcast <2 x float> %add to double
  ret double %3
}
; CHECK-LABEL: test_fadd
; SSE41: addps
; AVX: vaddps
; CHECK-NEXT: ret

define double @test_fsub(double %A, double %B) {
  %1 = bitcast double %A to <2 x float>
  %2 = bitcast double %B to <2 x float>
  %sub = fsub <2 x float> %1, %2
  %3 = bitcast <2 x float> %sub to double
  ret double %3
}
; CHECK-LABEL: test_fsub
; SSE41: subps
; AVX: vsubps
; CHECK-NEXT: ret

define double @test_fmul(double %A, double %B) {
  %1 = bitcast double %A to <2 x float>
  %2 = bitcast double %B to <2 x float>
  %mul = fmul <2 x float> %1, %2
  %3 = bitcast <2 x float> %mul to double
  ret double %3
}
; CHECK-LABEL: test_fmul
; SSE41: mulps
; AVX: vmulps
; CHECK-NEXT: ret

