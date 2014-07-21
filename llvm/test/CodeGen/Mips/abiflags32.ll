; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips32 %s -o - | FileCheck %s
; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips32 -mattr=fp64 %s -o - | FileCheck  -check-prefix=CHECK-64 %s
; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips64 -mattr=-n64,n32 %s -o - | FileCheck  -check-prefix=CHECK-64n %s

; CHECK: .nan    legacy
; We don't emit '.module fp=32' for compatibility with binutils 2.24 which
; doesn't accept .module.
; CHECK-NOT: .module fp=32

; CHECK-64: .nan    legacy
; We do emit '.module fp=64' though since it contradicts the default value.
; CHECK-64: .module fp=64

; CHECK-64n: .nan    legacy
; We don't emit '.module fp=64' for compatibility with binutils 2.24 which
; doesn't accept .module.
; CHECK-64n-NOT: .module fp=64
