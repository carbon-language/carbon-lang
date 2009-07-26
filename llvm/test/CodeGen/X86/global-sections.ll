; RUN: llvm-as < %s | llc -mtriple=i386-unknown-linux-gnu | FileCheck %s -check-prefix=LINUX
; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin9.7 | FileCheck %s -check-prefix=DARWIN


; int G1;
@G1 = common global i32 0

; LINUX: .type   G1,@object
; LINUX: .section .gnu.linkonce.b.G1,"aw",@nobits
; LINUX: .comm  G1,4,4

; DARWIN: .comm	_G1,4,2




; const int G2 __attribute__((weak)) = 42;
@G2 = weak_odr constant i32 42	


; TODO: linux drops this into .rodata, we drop it into ".gnu.linkonce.r.G2"

; DARWIN: .section __TEXT,__const_coal,coalesced
; DARWIN: _G2:
; DARWIN:    .long 42
