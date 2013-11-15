; Test the MSA intrinsics that are encoded with the 2R instruction format and
; convert scalars to vectors.

; RUN: llc -march=mips -mattr=+msa,+fp64 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 < %s | FileCheck %s

@llvm_mips_fill_b_ARG1 = global i32 23, align 16
@llvm_mips_fill_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_fill_b_test() nounwind {
entry:
  %0 = load i32* @llvm_mips_fill_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.fill.b(i32 %0)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_fill_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.fill.b(i32) nounwind

; CHECK: llvm_mips_fill_b_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]],
; CHECK-DAG: fill.b [[R2:\$w[0-9]+]], [[R1]]
; CHECK-DAG: st.b [[R2]],
; CHECK: .size llvm_mips_fill_b_test
;
@llvm_mips_fill_h_ARG1 = global i32 23, align 16
@llvm_mips_fill_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_fill_h_test() nounwind {
entry:
  %0 = load i32* @llvm_mips_fill_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.fill.h(i32 %0)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_fill_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.fill.h(i32) nounwind

; CHECK: llvm_mips_fill_h_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]],
; CHECK-DAG: fill.h [[R2:\$w[0-9]+]], [[R1]]
; CHECK-DAG: st.h [[R2]],
; CHECK: .size llvm_mips_fill_h_test
;
@llvm_mips_fill_w_ARG1 = global i32 23, align 16
@llvm_mips_fill_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fill_w_test() nounwind {
entry:
  %0 = load i32* @llvm_mips_fill_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.fill.w(i32 %0)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_fill_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fill.w(i32) nounwind

; CHECK: llvm_mips_fill_w_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]],
; CHECK-DAG: fill.w [[R2:\$w[0-9]+]], [[R1]]
; CHECK-DAG: st.w [[R2]],
; CHECK: .size llvm_mips_fill_w_test
;
@llvm_mips_fill_d_ARG1 = global i64 23, align 16
@llvm_mips_fill_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fill_d_test() nounwind {
entry:
  %0 = load i64* @llvm_mips_fill_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.fill.d(i64 %0)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_fill_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fill.d(i64) nounwind

; CHECK: llvm_mips_fill_d_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], 0(
; CHECK-DAG: lw [[R2:\$[0-9]+]], 4(
; CHECK-DAG: ldi.b [[R3:\$w[0-9]+]], 0
; CHECK-DAG: insert.w [[R3]][0], [[R1]]
; CHECK-DAG: insert.w [[R3]][1], [[R2]]
; CHECK-DAG: insert.w [[R3]][2], [[R1]]
; CHECK-DAG: insert.w [[R3]][3], [[R2]]
; CHECK-DAG: st.w [[R3]],
; CHECK: .size llvm_mips_fill_d_test
;
