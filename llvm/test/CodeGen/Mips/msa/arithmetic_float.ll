; RUN: llc -march=mips -mattr=+msa < %s | FileCheck %s

define void @add_v4f32(<4 x float>* %c, <4 x float>* %a, <4 x float>* %b) nounwind {
  ; CHECK: add_v4f32:

  %1 = load <4 x float>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fadd <4 x float> %1, %2
  ; CHECK-DAG: fadd.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x float> %3, <4 x float>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size add_v4f32
}

define void @add_v2f64(<2 x double>* %c, <2 x double>* %a, <2 x double>* %b) nounwind {
  ; CHECK: add_v2f64:

  %1 = load <2 x double>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fadd <2 x double> %1, %2
  ; CHECK-DAG: fadd.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x double> %3, <2 x double>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size add_v2f64
}

define void @sub_v4f32(<4 x float>* %c, <4 x float>* %a, <4 x float>* %b) nounwind {
  ; CHECK: sub_v4f32:

  %1 = load <4 x float>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fsub <4 x float> %1, %2
  ; CHECK-DAG: fsub.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x float> %3, <4 x float>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size sub_v4f32
}

define void @sub_v2f64(<2 x double>* %c, <2 x double>* %a, <2 x double>* %b) nounwind {
  ; CHECK: sub_v2f64:

  %1 = load <2 x double>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fsub <2 x double> %1, %2
  ; CHECK-DAG: fsub.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x double> %3, <2 x double>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size sub_v2f64
}

define void @mul_v4f32(<4 x float>* %c, <4 x float>* %a, <4 x float>* %b) nounwind {
  ; CHECK: mul_v4f32:

  %1 = load <4 x float>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fmul <4 x float> %1, %2
  ; CHECK-DAG: fmul.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x float> %3, <4 x float>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size mul_v4f32
}

define void @mul_v2f64(<2 x double>* %c, <2 x double>* %a, <2 x double>* %b) nounwind {
  ; CHECK: mul_v2f64:

  %1 = load <2 x double>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fmul <2 x double> %1, %2
  ; CHECK-DAG: fmul.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x double> %3, <2 x double>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size mul_v2f64
}

define void @fdiv_v4f32(<4 x float>* %c, <4 x float>* %a, <4 x float>* %b) nounwind {
  ; CHECK: fdiv_v4f32:

  %1 = load <4 x float>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fdiv <4 x float> %1, %2
  ; CHECK-DAG: fdiv.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x float> %3, <4 x float>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size fdiv_v4f32
}

define void @fdiv_v2f64(<2 x double>* %c, <2 x double>* %a, <2 x double>* %b) nounwind {
  ; CHECK: fdiv_v2f64:

  %1 = load <2 x double>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fdiv <2 x double> %1, %2
  ; CHECK-DAG: fdiv.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x double> %3, <2 x double>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size fdiv_v2f64
}

define void @fsqrt_v4f32(<4 x float>* %c, <4 x float>* %a) nounwind {
  ; CHECK: fsqrt_v4f32:

  %1 = load <4 x float>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = tail call <4 x float> @llvm.sqrt.v4f32 (<4 x float> %1)
  ; CHECK-DAG: fsqrt.w [[R3:\$w[0-9]+]], [[R1]]
  store <4 x float> %2, <4 x float>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size fsqrt_v4f32
}

define void @fsqrt_v2f64(<2 x double>* %c, <2 x double>* %a) nounwind {
  ; CHECK: fsqrt_v2f64:

  %1 = load <2 x double>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = tail call <2 x double> @llvm.sqrt.v2f64 (<2 x double> %1)
  ; CHECK-DAG: fsqrt.d [[R3:\$w[0-9]+]], [[R1]]
  store <2 x double> %2, <2 x double>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size fsqrt_v2f64
}

declare <4 x float>  @llvm.sqrt.v4f32(<4 x float>  %Val)
declare <2 x double> @llvm.sqrt.v2f64(<2 x double> %Val)
