; This isn't really an assembly file, its just here to
; run the following test which tests llvm-ar for compatibility
; reading SVR4 style archives.

; RUN: ar t SVR4.a > Output/svr1
; RUN: llvm-ar t SVR4.a > Output/svr2
; RUN: diff Output/svr1  Output/svr2
; RUN: cp SVR4.a Output/SVR4_mod.a
; RUN: llvm-ranlib Output/SVR4_mod.a
; RUN: llvm-ar t Output/SVR4_mod.a
