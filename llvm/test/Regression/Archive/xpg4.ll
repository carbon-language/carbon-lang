; This isn't really an assembly file, its just here to
; run the following test which tests llvm-ar for compatibility
; reading xpg4 style archives.

; RUN: ar t xpg4.a > Output/xpg1
; RUN: llvm-ar t xpg4.a > Output/xpg2
; RUN: diff Output/xpg1 Output/xpg2
; RUN: cp xpg4.a Output/xpg4_mod.a
; RUN: llvm-ranlib Output/xpg4_mod.a
; RUN: llvm-ar t Output/xpg4_mod.a
