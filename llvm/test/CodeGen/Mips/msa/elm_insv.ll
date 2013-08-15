; RUN: llc -march=mips -mattr=+msa < %s | FileCheck %s

@llvm_mips_insert_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_insert_b_ARG3 = global i32 27, align 16
@llvm_mips_insert_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_insert_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_insert_b_ARG1
  %1 = load i32* @llvm_mips_insert_b_ARG3
  %2 = tail call <16 x i8> @llvm.mips.insert.b(<16 x i8> %0, i32 1, i32 %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_insert_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.insert.b(<16 x i8>, i32, i32) nounwind

; CHECK: llvm_mips_insert_b_test:
; CHECK: lw
; CHECK: ld.b
; CHECK: insert.b
; CHECK: st.b
; CHECK: .size llvm_mips_insert_b_test
;
@llvm_mips_insert_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_insert_h_ARG3 = global i32 27, align 16
@llvm_mips_insert_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_insert_h_test() nounwind {
entry:
  %0 = load <8 x i16>* @llvm_mips_insert_h_ARG1
  %1 = load i32* @llvm_mips_insert_h_ARG3
  %2 = tail call <8 x i16> @llvm.mips.insert.h(<8 x i16> %0, i32 1, i32 %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_insert_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.insert.h(<8 x i16>, i32, i32) nounwind

; CHECK: llvm_mips_insert_h_test:
; CHECK: lw
; CHECK: ld.h
; CHECK: insert.h
; CHECK: st.h
; CHECK: .size llvm_mips_insert_h_test
;
@llvm_mips_insert_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_insert_w_ARG3 = global i32 27, align 16
@llvm_mips_insert_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_insert_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_insert_w_ARG1
  %1 = load i32* @llvm_mips_insert_w_ARG3
  %2 = tail call <4 x i32> @llvm.mips.insert.w(<4 x i32> %0, i32 1, i32 %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_insert_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.insert.w(<4 x i32>, i32, i32) nounwind

; CHECK: llvm_mips_insert_w_test:
; CHECK: lw
; CHECK: ld.w
; CHECK: insert.w
; CHECK: st.w
; CHECK: .size llvm_mips_insert_w_test
;
