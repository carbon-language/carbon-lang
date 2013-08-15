; RUN: llc -march=mips -mattr=+msa < %s | FileCheck %s

@llvm_mips_andi_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_andi_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_andi_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_andi_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.andi.b(<16 x i8> %0, i32 25)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_andi_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.andi.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_andi_b_test:
; CHECK: ld.b
; CHECK: andi.b
; CHECK: st.b
; CHECK: .size llvm_mips_andi_b_test
;
@llvm_mips_bmnzi_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bmnzi_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bmnzi_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_bmnzi_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.bmnzi.b(<16 x i8> %0, i32 25)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_bmnzi_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.bmnzi.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_bmnzi_b_test:
; CHECK: ld.b
; CHECK: bmnzi.b
; CHECK: st.b
; CHECK: .size llvm_mips_bmnzi_b_test
;
@llvm_mips_bmzi_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bmzi_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bmzi_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_bmzi_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.bmzi.b(<16 x i8> %0, i32 25)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_bmzi_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.bmzi.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_bmzi_b_test:
; CHECK: ld.b
; CHECK: bmzi.b
; CHECK: st.b
; CHECK: .size llvm_mips_bmzi_b_test
;
@llvm_mips_bseli_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bseli_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bseli_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_bseli_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.bseli.b(<16 x i8> %0, i32 25)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_bseli_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.bseli.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_bseli_b_test:
; CHECK: ld.b
; CHECK: bseli.b
; CHECK: st.b
; CHECK: .size llvm_mips_bseli_b_test
;
