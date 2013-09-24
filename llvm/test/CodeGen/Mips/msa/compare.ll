; RUN: llc -march=mips -mattr=+msa < %s | FileCheck %s

define void @ceq_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: ceq_v16i8:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp eq <16 x i8> %1, %2
  %4 = sext <16 x i1> %3 to <16 x i8>
  ; CHECK-DAG: ceq.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size ceq_v16i8
}

define void @ceq_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: ceq_v8i16:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp eq <8 x i16> %1, %2
  %4 = sext <8 x i1> %3 to <8 x i16>
  ; CHECK-DAG: ceq.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size ceq_v8i16
}

define void @ceq_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: ceq_v4i32:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp eq <4 x i32> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: ceq.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ceq_v4i32
}

define void @ceq_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: ceq_v2i64:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp eq <2 x i64> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: ceq.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ceq_v2i64
}

define void @cle_s_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: cle_s_v16i8:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sle <16 x i8> %1, %2
  %4 = sext <16 x i1> %3 to <16 x i8>
  ; CHECK-DAG: cle_s.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size cle_s_v16i8
}

define void @cle_s_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: cle_s_v8i16:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sle <8 x i16> %1, %2
  %4 = sext <8 x i1> %3 to <8 x i16>
  ; CHECK-DAG: cle_s.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size cle_s_v8i16
}

define void @cle_s_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: cle_s_v4i32:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sle <4 x i32> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: cle_s.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size cle_s_v4i32
}

define void @cle_s_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: cle_s_v2i64:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sle <2 x i64> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: cle_s.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size cle_s_v2i64
}

define void @cle_u_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: cle_u_v16i8:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ule <16 x i8> %1, %2
  %4 = sext <16 x i1> %3 to <16 x i8>
  ; CHECK-DAG: cle_u.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size cle_u_v16i8
}

define void @cle_u_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: cle_u_v8i16:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ule <8 x i16> %1, %2
  %4 = sext <8 x i1> %3 to <8 x i16>
  ; CHECK-DAG: cle_u.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size cle_u_v8i16
}

define void @cle_u_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: cle_u_v4i32:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ule <4 x i32> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: cle_u.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size cle_u_v4i32
}

define void @cle_u_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: cle_u_v2i64:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ule <2 x i64> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: cle_u.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size cle_u_v2i64
}

define void @clt_s_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: clt_s_v16i8:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp slt <16 x i8> %1, %2
  %4 = sext <16 x i1> %3 to <16 x i8>
  ; CHECK-DAG: clt_s.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size clt_s_v16i8
}

define void @clt_s_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: clt_s_v8i16:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp slt <8 x i16> %1, %2
  %4 = sext <8 x i1> %3 to <8 x i16>
  ; CHECK-DAG: clt_s.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size clt_s_v8i16
}

define void @clt_s_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: clt_s_v4i32:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp slt <4 x i32> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: clt_s.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size clt_s_v4i32
}

define void @clt_s_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: clt_s_v2i64:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp slt <2 x i64> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: clt_s.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size clt_s_v2i64
}

define void @clt_u_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: clt_u_v16i8:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ult <16 x i8> %1, %2
  %4 = sext <16 x i1> %3 to <16 x i8>
  ; CHECK-DAG: clt_u.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size clt_u_v16i8
}

define void @clt_u_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: clt_u_v8i16:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ult <8 x i16> %1, %2
  %4 = sext <8 x i1> %3 to <8 x i16>
  ; CHECK-DAG: clt_u.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size clt_u_v8i16
}

define void @clt_u_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: clt_u_v4i32:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ult <4 x i32> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: clt_u.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size clt_u_v4i32
}

define void @clt_u_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: clt_u_v2i64:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ult <2 x i64> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: clt_u.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size clt_u_v2i64
}

define void @ceqi_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: ceqi_v16i8:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp eq <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %3 = sext <16 x i1> %2 to <16 x i8>
  ; CHECK-DAG: ceqi.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size ceqi_v16i8
}

define void @ceqi_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: ceqi_v8i16:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp eq <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %3 = sext <8 x i1> %2 to <8 x i16>
  ; CHECK-DAG: ceqi.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size ceqi_v8i16
}

define void @ceqi_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: ceqi_v4i32:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp eq <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  %3 = sext <4 x i1> %2 to <4 x i32>
  ; CHECK-DAG: ceqi.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ceqi_v4i32
}

define void @ceqi_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: ceqi_v2i64:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp eq <2 x i64> %1, <i64 1, i64 1>
  %3 = sext <2 x i1> %2 to <2 x i64>
  ; CHECK-DAG: ceqi.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ceqi_v2i64
}

define void @clei_s_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: clei_s_v16i8:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sle <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %3 = sext <16 x i1> %2 to <16 x i8>
  ; CHECK-DAG: clei_s.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size clei_s_v16i8
}

define void @clei_s_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: clei_s_v8i16:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sle <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %3 = sext <8 x i1> %2 to <8 x i16>
  ; CHECK-DAG: clei_s.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size clei_s_v8i16
}

define void @clei_s_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: clei_s_v4i32:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sle <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  %3 = sext <4 x i1> %2 to <4 x i32>
  ; CHECK-DAG: clei_s.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size clei_s_v4i32
}

define void @clei_s_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: clei_s_v2i64:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sle <2 x i64> %1, <i64 1, i64 1>
  %3 = sext <2 x i1> %2 to <2 x i64>
  ; CHECK-DAG: clei_s.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size clei_s_v2i64
}

define void @clei_u_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: clei_u_v16i8:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ule <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %3 = sext <16 x i1> %2 to <16 x i8>
  ; CHECK-DAG: clei_u.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size clei_u_v16i8
}

define void @clei_u_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: clei_u_v8i16:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ule <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %3 = sext <8 x i1> %2 to <8 x i16>
  ; CHECK-DAG: clei_u.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size clei_u_v8i16
}

define void @clei_u_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: clei_u_v4i32:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ule <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  %3 = sext <4 x i1> %2 to <4 x i32>
  ; CHECK-DAG: clei_u.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size clei_u_v4i32
}

define void @clei_u_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: clei_u_v2i64:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ule <2 x i64> %1, <i64 1, i64 1>
  %3 = sext <2 x i1> %2 to <2 x i64>
  ; CHECK-DAG: clei_u.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size clei_u_v2i64
}

define void @clti_s_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: clti_s_v16i8:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp slt <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %3 = sext <16 x i1> %2 to <16 x i8>
  ; CHECK-DAG: clti_s.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size clti_s_v16i8
}

define void @clti_s_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: clti_s_v8i16:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp slt <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %3 = sext <8 x i1> %2 to <8 x i16>
  ; CHECK-DAG: clti_s.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size clti_s_v8i16
}

define void @clti_s_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: clti_s_v4i32:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp slt <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  %3 = sext <4 x i1> %2 to <4 x i32>
  ; CHECK-DAG: clti_s.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size clti_s_v4i32
}

define void @clti_s_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: clti_s_v2i64:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp slt <2 x i64> %1, <i64 1, i64 1>
  %3 = sext <2 x i1> %2 to <2 x i64>
  ; CHECK-DAG: clti_s.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size clti_s_v2i64
}

define void @clti_u_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: clti_u_v16i8:

  %1 = load <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ult <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %3 = sext <16 x i1> %2 to <16 x i8>
  ; CHECK-DAG: clti_u.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size clti_u_v16i8
}

define void @clti_u_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: clti_u_v8i16:

  %1 = load <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ult <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %3 = sext <8 x i1> %2 to <8 x i16>
  ; CHECK-DAG: clti_u.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size clti_u_v8i16
}

define void @clti_u_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: clti_u_v4i32:

  %1 = load <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ult <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  %3 = sext <4 x i1> %2 to <4 x i32>
  ; CHECK-DAG: clti_u.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size clti_u_v4i32
}

define void @clti_u_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: clti_u_v2i64:

  %1 = load <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ult <2 x i64> %1, <i64 1, i64 1>
  %3 = sext <2 x i1> %2 to <2 x i64>
  ; CHECK-DAG: clti_u.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size clti_u_v2i64
}
