; This isn't really an assembly file, its just here to
; run the following test which tests llvm-ar for compatibility
; reading MacOSX (BSD4.4) style archives.

; RUN: ar t MacOSX.a > Output/osx1
; RUN: llvm-ar t MacOSX.a > Output/osx2
; RUN: diff Output/osx1  Output/osx2
; RUN: cp MacOSX.a Output/MacOSX_mod.a
; RUN: llvm-ranlib Output/MacOSX_mod.a
; RUN: llvm-ar t Output/MacOSX_mod.a
