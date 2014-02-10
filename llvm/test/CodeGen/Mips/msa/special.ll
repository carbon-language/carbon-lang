; Test the MSA intrinsics that are encoded with the SPECIAL instruction format.

; RUN: llc -march=mips -mattr=+msa,+fp64 < %s | \
; RUN:   FileCheck %s --check-prefix=MIPS32

define i32 @llvm_mips_lsa_test(i32 %a, i32 %b) nounwind {
entry:
  %0 = tail call i32 @llvm.mips.lsa(i32 %a, i32 %b, i32 2)
  ret i32 %0
}

declare i32 @llvm.mips.lsa(i32, i32, i32) nounwind

; MIPS32: llvm_mips_lsa_test:
; MIPS32: lsa {{\$[0-9]+}}, $5, $4, 2
; MIPS32: .size llvm_mips_lsa_test

define i32 @lsa_test(i32 %a, i32 %b) nounwind {
entry:
  %0 = shl i32 %b, 2
  %1 = add i32 %a, %0
  ret i32 %1
}

; MIPS32: lsa_test:
; MIPS32: lsa {{\$[0-9]+}}, $5, $4, 2
; MIPS32: .size lsa_test
