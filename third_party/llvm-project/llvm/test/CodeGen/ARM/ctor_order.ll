; RUN: llc < %s -mtriple=arm-apple-darwin  | FileCheck %s -check-prefix=DARWIN
; RUN: llc < %s -mtriple=arm-apple-darwin -relocation-model=dynamic-no-pic  | FileCheck %s --check-prefix=DARWIN
; RUN: llc < %s -mtriple=arm-apple-darwin -relocation-model=static  | FileCheck %s -check-prefix=DARWIN-STATIC
; RUN: llc < %s -mtriple=arm-linux-gnu -target-abi=apcs  | FileCheck %s -check-prefix=ELF
; RUN: llc < %s -mtriple=arm-linux-gnueabi | FileCheck %s -check-prefix=GNUEABI

; DARWIN:      .section	__DATA,__mod_init_func,mod_init_funcs
; DARWIN:      .long _f151
; DARWIN-NEXT: .long _f152

; DARWIN-STATIC: .section __TEXT,__constructor

; ELF:      .section .ctors.65384,"aw",%progbits
; ELF:      .long    f151
; ELF:      .section .ctors.65383,"aw",%progbits
; ELF:      .long    f152

; GNUEABI:      .section .init_array.151,"aw",%init_array
; GNUEABI:      .long    f151
; GNUEABI:      .section .init_array.152,"aw",%init_array
; GNUEABI:      .long    f152


@llvm.global_ctors = appending global [2 x { i32, void ()*, i8* }] [ { i32, void ()*, i8* } { i32 151, void ()* @f151, i8* null }, { i32, void ()*, i8* } { i32 152, void ()* @f152, i8* null } ]

define void @f151() {
entry:
        ret void
}

define void @f152() {
entry:
        ret void
}
