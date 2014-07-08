; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips32 %s -o - | FileCheck %s
; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips32 -mattr=fp64 %s -o - | FileCheck  -check-prefix=CHECK-64 %s
; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips64 -mattr=-n64,n32 %s -o - | FileCheck  -check-prefix=CHECK-64n %s

; CHECK: .nan    legacy
; CHECK: .module fp=32

; CHECK-64: .nan    legacy
; CHECK-64: .module fp=64

; CHECK-64n: .nan    legacy
; CHECK-64n: .module fp=64
