; This isn't really an assembly file, its just here to
; run the following test which tests llvm-ar for compatibility
; reading xpg4 style archives.

; RUN: ar t xpg4.a > Output/xpg1
; RUN: llvm-ar t SVR4.a > Output/xpg2
; RUN: diff Output/xpg1 Output/xpg2
