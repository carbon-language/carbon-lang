; RUN: llc < %s -mtriple=arm-apple-darwin  | FileCheck %s -check-prefix=DARWIN
; RUN: llc < %s -mtriple=arm-linux-gnu     | FileCheck %s -check-prefix=ELF
; RUN: llc < %s -mtriple=arm-linux-gnueabi | FileCheck %s -check-prefix=GNUEABI

; DARWIN:      .section	__DATA,__mod_init_func,mod_init_funcs
; DARWIN:      .long _f151
; DARWIN-NEXT: .long _f152

; ELF:      .section .ctors,"aw",%progbits
; ELF:      .long    f152
; ELF-NEXT: .long    f151

; GNUEABI:      .section .init_array,"aw",%init_array
; GNUEABI:      .long    f151
; GNUEABI-NEXT: .long    f152


@llvm.global_ctors = appending global [2 x { i32, void ()* }] [ { i32, void ()* } { i32 151, void ()* @f151 }, { i32, void ()* } { i32 152, void ()* @f152 } ]

define void @f151() {
entry:
        ret void
}

define void @f152() {
entry:
        ret void
}
