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


; int * const G3 = &G1;
@G3 = constant i32* @G1

; DARWIN: .section        __DATA,__const
; DARWIN: .globl _G3
; DARWIN: _G3:
; DARWIN:     .long _G1


; _Complex long long const G4 = 34;
@G4 = constant {i64,i64} { i64 34, i64 0 }

; DARWIN: .section        __TEXT,__const
; DARWIN: _G4:
; DARWIN:     .long 34


; int G5 = 47;
@G5 = global i32 47

; LINUX: .data
; LINUX: .globl G5
; LINUX: G5:
; LINUX:    .long 47

; DARWIN: .section        __DATA,__data
; DARWIN: .globl _G5
; DARWIN: _G5:
; DARWIN:    .long 47


; PR4584
@"foo bar" = linkonce global i32 42

; LINUX: .type	foo_20_bar,@object
; LINUX:.section	.gnu.linkonce.d.foo_20_bar,"aw",@progbits
; LINUX: .weak	foo_20_bar
; LINUX: foo_20_bar:

; DARWIN: .section		__DATA,__datacoal_nt,coalesced
; DARWIN: .globl	"_foo bar"
; DARWIN:	.weak_definition "_foo bar"
; DARWIN: "_foo bar":

; PR4650
@G6 = weak_odr constant [1 x i8] c"\01"

; LINUX:   .type	G6,@object
; LINUX:   .section	.gnu.linkonce.r.G6,"a",@progbits
; LINUX:   .weak	G6
; LINUX: G6:
; LINUX:   .ascii	"\001"
; LINUX:   .size	G6, 1

; DARWIN:  .section __TEXT,__const_coal,coalesced
; DARWIN:  .globl _G6
; DARWIN:  .weak_definition _G6
; DARWIN:_G6:
; DARWIN:  .ascii "\001"


@G7 = constant [10 x i8] c"abcdefghi\00"

; DARWIN:	__TEXT,__cstring,cstring_literals
; DARWIN:	.globl _G7
; DARWIN: _G7:
; DARWIN:	.asciz	"abcdefghi"

; LINUX:	.section		.rodata.str1.1,"aMS",@progbits,1
; LINUX:	.globl G7
; LINUX: G7:
; LINUX:	.asciz	"abcdefghi"


@G8 = constant [4 x i16] [ i16 1, i16 2, i16 3, i16 0 ]

; DARWIN:	.section	__TEXT,__ustring
; DARWIN:	.globl _G8
; DARWIN: _G8:

; LINUX:	.section		.rodata.str2.2,"aMS",@progbits,2
; LINUX:	.globl G8
; LINUX:G8:

@G9 = constant [4 x i32] [ i32 1, i32 2, i32 3, i32 0 ]

; DARWIN:	.section        __TEXT,__const
; DARWIN:	.globl _G9
; DARWIN: _G9:

; LINUX:	.section		.rodata.str4.4,"aMS",@progbits,4
; LINUX:	.globl G9
; LINUX:G9



