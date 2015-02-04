; RUN: llc < %s -mtriple=i386-unknown-linux | FileCheck %s -check-prefix=LINUX
; RUN: llc < %s -mtriple=i386-unknown-linux -data-sections | FileCheck %s -check-prefix=LINUX-SECTIONS

$G16 = comdat any
@G16 = unnamed_addr constant i32 42, comdat

; LINUX: .section	.rodata.G16,"aG",@progbits,G16,comdat
; LINUX-SECTIONS: .section	.rodata.G16,"aG",@progbits,G16,comdat
