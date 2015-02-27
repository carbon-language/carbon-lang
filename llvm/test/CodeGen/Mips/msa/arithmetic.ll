; RUN: llc -march=mips -mattr=+msa,+fp64 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 < %s | FileCheck %s

define void @add_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: add_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = add <16 x i8> %1, %2
  ; CHECK-DAG: addv.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size add_v16i8
}

define void @add_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: add_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = add <8 x i16> %1, %2
  ; CHECK-DAG: addv.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size add_v8i16
}

define void @add_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: add_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = add <4 x i32> %1, %2
  ; CHECK-DAG: addv.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size add_v4i32
}

define void @add_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: add_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = add <2 x i64> %1, %2
  ; CHECK-DAG: addv.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size add_v2i64
}

define void @add_v16i8_i(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: add_v16i8_i:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = add <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                          i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ; CHECK-DAG: addvi.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %2, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size add_v16i8_i
}

define void @add_v8i16_i(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: add_v8i16_i:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = add <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1,
                          i16 1, i16 1, i16 1, i16 1>
  ; CHECK-DAG: addvi.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %2, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size add_v8i16_i
}

define void @add_v4i32_i(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: add_v4i32_i:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = add <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: addvi.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size add_v4i32_i
}

define void @add_v2i64_i(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: add_v2i64_i:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = add <2 x i64> %1, <i64 1, i64 1>
  ; CHECK-DAG: addvi.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size add_v2i64_i
}

define void @sub_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: sub_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = sub <16 x i8> %1, %2
  ; CHECK-DAG: subv.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size sub_v16i8
}

define void @sub_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: sub_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = sub <8 x i16> %1, %2
  ; CHECK-DAG: subv.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size sub_v8i16
}

define void @sub_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: sub_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = sub <4 x i32> %1, %2
  ; CHECK-DAG: subv.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size sub_v4i32
}

define void @sub_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: sub_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = sub <2 x i64> %1, %2
  ; CHECK-DAG: subv.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size sub_v2i64
}

define void @sub_v16i8_i(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: sub_v16i8_i:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = sub <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                          i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ; CHECK-DAG: subvi.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %2, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size sub_v16i8_i
}

define void @sub_v8i16_i(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: sub_v8i16_i:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = sub <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1,
                          i16 1, i16 1, i16 1, i16 1>
  ; CHECK-DAG: subvi.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %2, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size sub_v8i16_i
}

define void @sub_v4i32_i(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: sub_v4i32_i:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = sub <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: subvi.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size sub_v4i32_i
}

define void @sub_v2i64_i(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: sub_v2i64_i:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = sub <2 x i64> %1, <i64 1, i64 1>
  ; CHECK-DAG: subvi.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size sub_v2i64_i
}

define void @mul_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: mul_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = mul <16 x i8> %1, %2
  ; CHECK-DAG: mulv.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size mul_v16i8
}

define void @mul_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: mul_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = mul <8 x i16> %1, %2
  ; CHECK-DAG: mulv.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size mul_v8i16
}

define void @mul_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: mul_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = mul <4 x i32> %1, %2
  ; CHECK-DAG: mulv.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size mul_v4i32
}

define void @mul_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: mul_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = mul <2 x i64> %1, %2
  ; CHECK-DAG: mulv.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size mul_v2i64
}

define void @maddv_v16i8(<16 x i8>* %d, <16 x i8>* %a, <16 x i8>* %b,
                         <16 x i8>* %c) nounwind {
  ; CHECK: maddv_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = load <16 x i8>, <16 x i8>* %c
  ; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], 0($7)
  %4 = mul <16 x i8> %2, %3
  %5 = add <16 x i8> %4, %1
  ; CHECK-DAG: maddv.b [[R1]], [[R2]], [[R3]]
  store <16 x i8> %5, <16 x i8>* %d
  ; CHECK-DAG: st.b [[R1]], 0($4)

  ret void
  ; CHECK: .size maddv_v16i8
}

define void @maddv_v8i16(<8 x i16>* %d, <8 x i16>* %a, <8 x i16>* %b,
                         <8 x i16>* %c) nounwind {
  ; CHECK: maddv_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = load <8 x i16>, <8 x i16>* %c
  ; CHECK-DAG: ld.h [[R3:\$w[0-9]+]], 0($7)
  %4 = mul <8 x i16> %2, %3
  %5 = add <8 x i16> %4, %1
  ; CHECK-DAG: maddv.h [[R1]], [[R2]], [[R3]]
  store <8 x i16> %5, <8 x i16>* %d
  ; CHECK-DAG: st.h [[R1]], 0($4)

  ret void
  ; CHECK: .size maddv_v8i16
}

define void @maddv_v4i32(<4 x i32>* %d, <4 x i32>* %a, <4 x i32>* %b,
                         <4 x i32>* %c) nounwind {
  ; CHECK: maddv_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = load <4 x i32>, <4 x i32>* %c
  ; CHECK-DAG: ld.w [[R3:\$w[0-9]+]], 0($7)
  %4 = mul <4 x i32> %2, %3
  %5 = add <4 x i32> %4, %1
  ; CHECK-DAG: maddv.w [[R1]], [[R2]], [[R3]]
  store <4 x i32> %5, <4 x i32>* %d
  ; CHECK-DAG: st.w [[R1]], 0($4)

  ret void
  ; CHECK: .size maddv_v4i32
}

define void @maddv_v2i64(<2 x i64>* %d, <2 x i64>* %a, <2 x i64>* %b,
                         <2 x i64>* %c) nounwind {
  ; CHECK: maddv_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = load <2 x i64>, <2 x i64>* %c
  ; CHECK-DAG: ld.d [[R3:\$w[0-9]+]], 0($7)
  %4 = mul <2 x i64> %2, %3
  %5 = add <2 x i64> %4, %1
  ; CHECK-DAG: maddv.d [[R1]], [[R2]], [[R3]]
  store <2 x i64> %5, <2 x i64>* %d
  ; CHECK-DAG: st.d [[R1]], 0($4)

  ret void
  ; CHECK: .size maddv_v2i64
}

define void @msubv_v16i8(<16 x i8>* %d, <16 x i8>* %a, <16 x i8>* %b,
                         <16 x i8>* %c) nounwind {
  ; CHECK: msubv_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = load <16 x i8>, <16 x i8>* %c
  ; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], 0($7)
  %4 = mul <16 x i8> %2, %3
  %5 = sub <16 x i8> %1, %4
  ; CHECK-DAG: msubv.b [[R1]], [[R2]], [[R3]]
  store <16 x i8> %5, <16 x i8>* %d
  ; CHECK-DAG: st.b [[R1]], 0($4)

  ret void
  ; CHECK: .size msubv_v16i8
}

define void @msubv_v8i16(<8 x i16>* %d, <8 x i16>* %a, <8 x i16>* %b,
                         <8 x i16>* %c) nounwind {
  ; CHECK: msubv_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = load <8 x i16>, <8 x i16>* %c
  ; CHECK-DAG: ld.h [[R3:\$w[0-9]+]], 0($7)
  %4 = mul <8 x i16> %2, %3
  %5 = sub <8 x i16> %1, %4
  ; CHECK-DAG: msubv.h [[R1]], [[R2]], [[R3]]
  store <8 x i16> %5, <8 x i16>* %d
  ; CHECK-DAG: st.h [[R1]], 0($4)

  ret void
  ; CHECK: .size msubv_v8i16
}

define void @msubv_v4i32(<4 x i32>* %d, <4 x i32>* %a, <4 x i32>* %b,
                         <4 x i32>* %c) nounwind {
  ; CHECK: msubv_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = load <4 x i32>, <4 x i32>* %c
  ; CHECK-DAG: ld.w [[R3:\$w[0-9]+]], 0($7)
  %4 = mul <4 x i32> %2, %3
  %5 = sub <4 x i32> %1, %4
  ; CHECK-DAG: msubv.w [[R1]], [[R2]], [[R3]]
  store <4 x i32> %5, <4 x i32>* %d
  ; CHECK-DAG: st.w [[R1]], 0($4)

  ret void
  ; CHECK: .size msubv_v4i32
}

define void @msubv_v2i64(<2 x i64>* %d, <2 x i64>* %a, <2 x i64>* %b,
                         <2 x i64>* %c) nounwind {
  ; CHECK: msubv_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = load <2 x i64>, <2 x i64>* %c
  ; CHECK-DAG: ld.d [[R3:\$w[0-9]+]], 0($7)
  %4 = mul <2 x i64> %2, %3
  %5 = sub <2 x i64> %1, %4
  ; CHECK-DAG: msubv.d [[R1]], [[R2]], [[R3]]
  store <2 x i64> %5, <2 x i64>* %d
  ; CHECK-DAG: st.d [[R1]], 0($4)

  ret void
  ; CHECK: .size msubv_v2i64
}

define void @div_s_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: div_s_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = sdiv <16 x i8> %1, %2
  ; CHECK-DAG: div_s.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size div_s_v16i8
}

define void @div_s_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: div_s_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = sdiv <8 x i16> %1, %2
  ; CHECK-DAG: div_s.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size div_s_v8i16
}

define void @div_s_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: div_s_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = sdiv <4 x i32> %1, %2
  ; CHECK-DAG: div_s.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size div_s_v4i32
}

define void @div_s_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: div_s_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = sdiv <2 x i64> %1, %2
  ; CHECK-DAG: div_s.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size div_s_v2i64
}

define void @div_u_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: div_u_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = udiv <16 x i8> %1, %2
  ; CHECK-DAG: div_u.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size div_u_v16i8
}

define void @div_u_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: div_u_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = udiv <8 x i16> %1, %2
  ; CHECK-DAG: div_u.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size div_u_v8i16
}

define void @div_u_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: div_u_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = udiv <4 x i32> %1, %2
  ; CHECK-DAG: div_u.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size div_u_v4i32
}

define void @div_u_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: div_u_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = udiv <2 x i64> %1, %2
  ; CHECK-DAG: div_u.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size div_u_v2i64
}

define void @mod_s_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: mod_s_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = srem <16 x i8> %1, %2
  ; CHECK-DAG: mod_s.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size mod_s_v16i8
}

define void @mod_s_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: mod_s_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = srem <8 x i16> %1, %2
  ; CHECK-DAG: mod_s.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size mod_s_v8i16
}

define void @mod_s_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: mod_s_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = srem <4 x i32> %1, %2
  ; CHECK-DAG: mod_s.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size mod_s_v4i32
}

define void @mod_s_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: mod_s_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = srem <2 x i64> %1, %2
  ; CHECK-DAG: mod_s.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size mod_s_v2i64
}

define void @mod_u_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: mod_u_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = urem <16 x i8> %1, %2
  ; CHECK-DAG: mod_u.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size mod_u_v16i8
}

define void @mod_u_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: mod_u_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = urem <8 x i16> %1, %2
  ; CHECK-DAG: mod_u.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size mod_u_v8i16
}

define void @mod_u_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: mod_u_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = urem <4 x i32> %1, %2
  ; CHECK-DAG: mod_u.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size mod_u_v4i32
}

define void @mod_u_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: mod_u_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = urem <2 x i64> %1, %2
  ; CHECK-DAG: mod_u.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size mod_u_v2i64
}
