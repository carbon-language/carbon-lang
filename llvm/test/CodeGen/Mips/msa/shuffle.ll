; RUN: llc -march=mips -mattr=+msa,+fp64 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 < %s | FileCheck %s

define void @vshf_v16i8_0(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: vshf_v16i8_0:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <16 x i8> %1, <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; CHECK-DAG: vshf.b [[R3]], [[R1]], [[R1]]
  store <16 x i8> %2, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v16i8_0
}

define void @vshf_v16i8_1(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: vshf_v16i8_1:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <16 x i8> %1, <16 x i8> undef, <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: splati.b [[R3:\$w[0-9]+]], [[R1]][1]
  store <16 x i8> %2, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v16i8_1
}

define void @vshf_v16i8_2(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: vshf_v16i8_2:

  %1 = load <16 x i8>* %a
  %2 = load <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 16>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; CHECK-DAG: vshf.b [[R3]], [[R2]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v16i8_2
}

define void @vshf_v16i8_3(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: vshf_v16i8_3:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2, <16 x i32> <i32 17, i32 24, i32 25, i32 18, i32 19, i32 20, i32 28, i32 19, i32 1, i32 8, i32 9, i32 2, i32 3, i32 4, i32 12, i32 3>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; The concatenation step of vshf is bitwise not vectorwise so we must reverse
  ; the operands to get the right answer.
  ; CHECK-DAG: vshf.b [[R3]], [[R2]], [[R1]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v16i8_3
}

define void @vshf_v16i8_4(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: vshf_v16i8_4:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <16 x i8> %1, <16 x i8> %1, <16 x i32> <i32 1, i32 17, i32 1, i32 17, i32 1, i32 17, i32 1, i32 17, i32 1, i32 17, i32 1, i32 17, i32 1, i32 17, i32 1, i32 17>
  ; CHECK-DAG: splati.b [[R3:\$w[0-9]+]], [[R1]][1]
  store <16 x i8> %2, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v16i8_4
}

define void @vshf_v8i16_0(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: vshf_v8i16_0:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.h [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; CHECK-DAG: vshf.h [[R3]], [[R1]], [[R1]]
  store <8 x i16> %2, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v8i16_0
}

define void @vshf_v8i16_1(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: vshf_v8i16_1:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: splati.h [[R3:\$w[0-9]+]], [[R1]][1]
  store <8 x i16> %2, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v8i16_1
}

define void @vshf_v8i16_2(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: vshf_v8i16_2:

  %1 = load <8 x i16>* %a
  %2 = load <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 8>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.h [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; CHECK-DAG: vshf.h [[R3]], [[R2]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v8i16_2
}

define void @vshf_v8i16_3(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: vshf_v8i16_3:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 1, i32 8, i32 9, i32 2, i32 3, i32 4, i32 12, i32 3>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.h [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; The concatenation step of vshf is bitwise not vectorwise so we must reverse
  ; the operands to get the right answer.
  ; CHECK-DAG: vshf.h [[R3]], [[R2]], [[R1]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v8i16_3
}

define void @vshf_v8i16_4(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: vshf_v8i16_4:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <8 x i16> %1, <8 x i16> %1, <8 x i32> <i32 1, i32 9, i32 1, i32 9, i32 1, i32 9, i32 1, i32 9>
  ; CHECK-DAG: splati.h [[R3:\$w[0-9]+]], [[R1]][1]
  store <8 x i16> %2, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v8i16_4
}

; Note: v4i32 only has one 4-element set so it's impossible to get a vshf.w
; instruction when using a single vector.

define void @vshf_v4i32_0(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: vshf_v4i32_0:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ; CHECK-DAG: shf.w [[R3:\$w[0-9]+]], [[R1]], 27
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v4i32_0
}

define void @vshf_v4i32_1(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: vshf_v4i32_1:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: shf.w [[R3:\$w[0-9]+]], [[R1]], 85
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v4i32_1
}

define void @vshf_v4i32_2(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: vshf_v4i32_2:

  %1 = load <4 x i32>* %a
  %2 = load <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 4, i32 5, i32 6, i32 4>
  ; CHECK-DAG: shf.w [[R3:\$w[0-9]+]], [[R2]], 36
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v4i32_2
}

define void @vshf_v4i32_3(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: vshf_v4i32_3:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 1, i32 5, i32 6, i32 4>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.w [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; The concatenation step of vshf is bitwise not vectorwise so we must reverse
  ; the operands to get the right answer.
  ; CHECK-DAG: vshf.w [[R3]], [[R2]], [[R1]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v4i32_3
}

define void @vshf_v4i32_4(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: vshf_v4i32_4:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <4 x i32> %1, <4 x i32> %1, <4 x i32> <i32 1, i32 5, i32 5, i32 1>
  ; CHECK-DAG: shf.w [[R3:\$w[0-9]+]], [[R1]], 85
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v4i32_4
}

define void @vshf_v2i64_0(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: vshf_v2i64_0:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <2 x i64> %1, <2 x i64> undef, <2 x i32> <i32 1, i32 0>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.d [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; CHECK-DAG: vshf.d [[R3]], [[R1]], [[R1]]
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v2i64_0
}

define void @vshf_v2i64_1(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: vshf_v2i64_1:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <2 x i64> %1, <2 x i64> undef, <2 x i32> <i32 1, i32 1>
  ; CHECK-DAG: splati.d [[R3:\$w[0-9]+]], [[R1]][1]
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v2i64_1
}

define void @vshf_v2i64_2(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: vshf_v2i64_2:

  %1 = load <2 x i64>* %a
  %2 = load <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 3, i32 2>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.d [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; CHECK-DAG: vshf.d [[R3]], [[R2]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v2i64_2
}

define void @vshf_v2i64_3(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: vshf_v2i64_3:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 1, i32 2>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.d [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; The concatenation step of vshf is bitwise not vectorwise so we must reverse
  ; the operands to get the right answer.
  ; CHECK-DAG: vshf.d [[R3]], [[R2]], [[R1]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v2i64_3
}

define void @vshf_v2i64_4(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: vshf_v2i64_4:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <2 x i64> %1, <2 x i64> %1, <2 x i32> <i32 1, i32 3>
  ; CHECK-DAG: splati.d [[R3:\$w[0-9]+]], [[R1]][1]
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v2i64_4
}

define void @shf_v16i8_0(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: shf_v16i8_0:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <16 x i8> %1, <16 x i8> undef, <16 x i32> <i32 1, i32 3, i32 2, i32 0, i32 5, i32 7, i32 6, i32 4, i32 9, i32 11, i32 10, i32 8, i32 13, i32 15, i32 14, i32 12>
  ; CHECK-DAG: shf.b [[R3:\$w[0-9]+]], [[R1]], 45
  store <16 x i8> %2, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size shf_v16i8_0
}

define void @shf_v8i16_0(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: shf_v8i16_0:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ; CHECK-DAG: shf.h [[R3:\$w[0-9]+]], [[R1]], 27
  store <8 x i16> %2, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size shf_v8i16_0
}

define void @shf_v4i32_0(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: shf_v4i32_0:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ; CHECK-DAG: shf.w [[R3:\$w[0-9]+]], [[R1]], 27
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size shf_v4i32_0
}

; shf.d does not exist

define void @ilvev_v16i8_0(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: ilvev_v16i8_0:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
  ; CHECK-DAG: ilvev.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size ilvev_v16i8_0
}

define void @ilvev_v8i16_0(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: ilvev_v8i16_0:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  ; CHECK-DAG: ilvev.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size ilvev_v8i16_0
}

define void @ilvev_v4i32_0(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: ilvev_v4i32_0:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ; CHECK-DAG: ilvev.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ilvev_v4i32_0
}

define void @ilvev_v2i64_0(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: ilvev_v2i64_0:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 0, i32 2>
  ; CHECK-DAG: ilvev.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ilvev_v2i64_0
}

define void @ilvod_v16i8_0(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: ilvod_v16i8_0:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
  ; CHECK-DAG: ilvod.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size ilvod_v16i8_0
}

define void @ilvod_v8i16_0(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: ilvod_v8i16_0:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ; CHECK-DAG: ilvod.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size ilvod_v8i16_0
}

define void @ilvod_v4i32_0(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: ilvod_v4i32_0:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ; CHECK-DAG: ilvod.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ilvod_v4i32_0
}

define void @ilvod_v2i64_0(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: ilvod_v2i64_0:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 1, i32 3>
  ; CHECK-DAG: ilvod.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ilvod_v2i64_0
}

define void @ilvl_v16i8_0(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: ilvl_v16i8_0:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  ; CHECK-DAG: ilvl.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size ilvl_v16i8_0
}

define void @ilvl_v8i16_0(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: ilvl_v8i16_0:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ; CHECK-DAG: ilvl.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size ilvl_v8i16_0
}

define void @ilvl_v4i32_0(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: ilvl_v4i32_0:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ; CHECK-DAG: ilvl.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ilvl_v4i32_0
}

define void @ilvl_v2i64_0(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: ilvl_v2i64_0:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 0, i32 2>
  ; ilvl.d and ilvev.d are equivalent for v2i64
  ; CHECK-DAG: ilvev.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ilvl_v2i64_0
}

define void @ilvr_v16i8_0(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: ilvr_v16i8_0:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  ; CHECK-DAG: ilvr.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size ilvr_v16i8_0
}

define void @ilvr_v8i16_0(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: ilvr_v8i16_0:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  ; CHECK-DAG: ilvr.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size ilvr_v8i16_0
}

define void @ilvr_v4i32_0(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: ilvr_v4i32_0:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  ; CHECK-DAG: ilvr.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ilvr_v4i32_0
}

define void @ilvr_v2i64_0(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: ilvr_v2i64_0:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 1, i32 3>
  ; ilvr.d and ilvod.d are equivalent for v2i64
  ; CHECK-DAG: ilvod.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ilvr_v2i64_0
}

define void @pckev_v16i8_0(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: pckev_v16i8_0:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  ; CHECK-DAG: pckev.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size pckev_v16i8_0
}

define void @pckev_v8i16_0(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: pckev_v8i16_0:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  ; CHECK-DAG: pckev.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size pckev_v8i16_0
}

define void @pckev_v4i32_0(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: pckev_v4i32_0:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  ; CHECK-DAG: pckev.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size pckev_v4i32_0
}

define void @pckev_v2i64_0(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: pckev_v2i64_0:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 0, i32 2>
  ; pckev.d and ilvev.d are equivalent for v2i64
  ; CHECK-DAG: ilvev.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size pckev_v2i64_0
}

define void @pckod_v16i8_0(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: pckod_v16i8_0:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  ; CHECK-DAG: pckod.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size pckod_v16i8_0
}

define void @pckod_v8i16_0(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: pckod_v8i16_0:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  ; CHECK-DAG: pckod.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size pckod_v8i16_0
}

define void @pckod_v4i32_0(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: pckod_v4i32_0:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  ; CHECK-DAG: pckod.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size pckod_v4i32_0
}

define void @pckod_v2i64_0(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: pckod_v2i64_0:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 1, i32 3>
  ; pckod.d and ilvod.d are equivalent for v2i64
  ; CHECK-DAG: ilvod.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size pckod_v2i64_0
}

define void @splati_v16i8_0(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: splati_v16i8_0:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <16 x i8> %1, <16 x i8> undef,
                     <16 x i32> <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  ; CHECK-DAG: splati.b [[R3:\$w[0-9]+]], [[R1]][4]
  store <16 x i8> %2, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size splati_v16i8_0
}

define void @splati_v8i16_0(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: splati_v8i16_0:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  ; CHECK-DAG: splati.h [[R3:\$w[0-9]+]], [[R1]][4]
  store <8 x i16> %2, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size splati_v8i16_0
}

define void @splati_v4i32_0(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: splati_v4i32_0:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  ; shf.w and splati.w are equivalent
  ; CHECK-DAG: shf.w [[R3:\$w[0-9]+]], [[R1]], 255
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size splati_v4i32_0
}

define void @splati_v2i64_0(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: splati_v2i64_0:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <2 x i64> %1, <2 x i64> undef, <2 x i32> <i32 1, i32 1>
  ; CHECK-DAG: splati.d [[R3:\$w[0-9]+]], [[R1]][1]
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size splati_v2i64_0
}
