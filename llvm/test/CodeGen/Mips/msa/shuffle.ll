; RUN: llc -march=mips -mattr=+msa < %s | FileCheck %s

define void @vshf_v16i8_0(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: vshf_v16i8_0:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <16 x i8> %1, <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], %lo
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
  ; CHECK-DAG: ldi.b [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: vshf.b [[R3]], [[R1]], [[R1]]
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
  ; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], %lo
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
  ; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], %lo
  ; CHECK-DAG: vshf.b [[R3]], [[R1]], [[R2]]
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
  ; CHECK-DAG: ldi.b [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: vshf.b [[R3]], [[R1]], [[R1]]
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
  ; CHECK-DAG: ld.h [[R3:\$w[0-9]+]], %lo
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
  ; CHECK-DAG: ldi.h [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: vshf.h [[R3]], [[R1]], [[R1]]
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
  ; CHECK-DAG: ld.h [[R3:\$w[0-9]+]], %lo
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
  ; CHECK-DAG: ld.h [[R3:\$w[0-9]+]], %lo
  ; CHECK-DAG: vshf.h [[R3]], [[R1]], [[R2]]
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
  ; CHECK-DAG: ldi.h [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: vshf.h [[R3]], [[R1]], [[R1]]
  store <8 x i16> %2, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v8i16_4
}

define void @vshf_v4i32_0(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: vshf_v4i32_0:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ; CHECK-DAG: ld.w [[R3:\$w[0-9]+]], %lo
  ; CHECK-DAG: vshf.w [[R3:\$w[0-9]+]], [[R1]], [[R1]]
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
  ; CHECK-DAG: ldi.w [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: vshf.w [[R3:\$w[0-9]+]], [[R1]], [[R1]]
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
  ; CHECK-DAG: ld.w [[R3:\$w[0-9]+]], %lo
  ; CHECK-DAG: vshf.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
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
  ; CHECK-DAG: ld.w [[R3:\$w[0-9]+]], %lo
  ; CHECK-DAG: vshf.w [[R3]], [[R1]], [[R2]]
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
  ; CHECK-DAG: ldi.w [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: vshf.w [[R3:\$w[0-9]+]], [[R1]], [[R1]]
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
  ; CHECK-DAG: ld.d [[R3:\$w[0-9]+]], %lo
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
  ; CHECK-DAG: ldi.d [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: vshf.d [[R3]], [[R1]], [[R1]]
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
  ; CHECK-DAG: ld.d [[R3:\$w[0-9]+]], %lo
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
  ; CHECK-DAG: ld.d [[R3:\$w[0-9]+]], %lo
  ; CHECK-DAG: vshf.d [[R3]], [[R1]], [[R2]]
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
  ; CHECK-DAG: ldi.d [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: vshf.d [[R3]], [[R1]], [[R1]]
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size vshf_v2i64_4
}
