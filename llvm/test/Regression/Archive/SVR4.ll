; This isn't really an assembly file, its just here to
; run the following test which tests llvm-ar for compatibility
; reading SVR4 style archives.

; RUN: ar t SVR4.a > %t1
; RUN: llvm-ar t SVR4.a > %t2
; RUN: diff %t1 %t2
