; RUN: llvm-as < %s | llc -mtriple=i386-unknown-linux-gnu | FileCheck %s -check-prefix=LINUX

@G1 = common global i32 0

; LINUX: .type   G1,@object
; LINUX: .section .gnu.linkonce.b.G1,"aw",@nobits
; LINUX: .comm  G1,4,4

