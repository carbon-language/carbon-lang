; RUN: llc < %s -relocation-model=static | FileCheck %s -check-prefix=STATIC
; RUN: llc < %s -relocation-model=pic -mtriple=powerpc-apple-darwin8 | FileCheck %s -check-prefix=PIC
; RUN: llc < %s -relocation-model=pic -mtriple=powerpc-unknown-linux | FileCheck %s -check-prefix=PICELF
; RUN: llc < %s -relocation-model=pic -mtriple=powerpc64-apple-darwin8 | FileCheck %s -check-prefix=PIC64
; RUN: llc < %s -relocation-model=dynamic-no-pic -mtriple=powerpc-apple-darwin8 | FileCheck %s -check-prefix=DYNAMIC
; RUN: llc < %s -relocation-model=dynamic-no-pic -mtriple=powerpc64-apple-darwin8 | FileCheck %s -check-prefix=DYNAMIC64
; PR4482
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "powerpc-apple-darwin8"

define i32 @foo(i64 %x) nounwind {
entry:
; STATIC: _foo:
; STATIC: bl _exact_log2
; STATIC: blr
; STATIC: .subsections_via_symbols

; PIC: _foo:
; PIC: bl L_exact_log2$stub
; PIC: blr

; PICELF: foo:
; PICELF: bl exact_log2@PLT
; PICELF: blr

; PIC64: _foo:
; PIC64: bl L_exact_log2$stub
; PIC64: blr

; DYNAMIC: _foo:
; DYNAMIC: bl L_exact_log2$stub
; DYNAMIC: blr

; DYNAMIC64: _foo:
; DYNAMIC64: bl L_exact_log2$stub
; DYNAMIC64: blr

        %A = call i32 @exact_log2(i64 %x) nounwind
	ret i32 %A
}

define available_externally i32 @exact_log2(i64 %x) nounwind {
entry:
	ret i32 42
}


; PIC: .section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
; PIC: L_exact_log2$stub:
; PIC: .indirect_symbol _exact_log2
; PIC: mflr r0
; PIC: bcl 20, 31, L_exact_log2$stub$tmp

; PIC: L_exact_log2$stub$tmp:
; PIC: mflr r11
; PIC: addis r11, r11, ha16(L_exact_log2$lazy_ptr-L_exact_log2$stub$tmp)
; PIC: mtlr r0
; PIC: lwzu r12, lo16(L_exact_log2$lazy_ptr-L_exact_log2$stub$tmp)(r11)
; PIC: mtctr r12
; PIC: bctr

; PIC: .section __DATA,__la_symbol_ptr,lazy_symbol_pointers
; PIC: L_exact_log2$lazy_ptr:
; PIC: .indirect_symbol _exact_log2
; PIC: .long dyld_stub_binding_helper

; PIC: .subsections_via_symbols

; PIC64: .section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
; PIC64: L_exact_log2$stub:
; PIC64: .indirect_symbol _exact_log2
; PIC64: mflr r0
; PIC64: bcl 20, 31, L_exact_log2$stub$tmp

; PIC64: L_exact_log2$stub$tmp:
; PIC64: mflr r11
; PIC64: addis r11, r11, ha16(L_exact_log2$lazy_ptr-L_exact_log2$stub$tmp)
; PIC64: mtlr r0
; PIC64: ldu r12, lo16(L_exact_log2$lazy_ptr-L_exact_log2$stub$tmp)(r11)
; PIC64: mtctr r12
; PIC64: bctr

; PIC64: .section __DATA,__la_symbol_ptr,lazy_symbol_pointers
; PIC64: L_exact_log2$lazy_ptr:
; PIC64: .indirect_symbol _exact_log2
; PIC64: .quad dyld_stub_binding_helper

; PIC64: .subsections_via_symbols

; DYNAMIC: .section __TEXT,__symbol_stub1,symbol_stubs,pure_instructions,16
; DYNAMIC: L_exact_log2$stub:
; DYNAMIC: .indirect_symbol _exact_log2
; DYNAMIC: lis r11, ha16(L_exact_log2$lazy_ptr)
; DYNAMIC: lwzu r12, lo16(L_exact_log2$lazy_ptr)(r11)
; DYNAMIC: mtctr r12
; DYNAMIC: bctr

; DYNAMIC: .section __DATA,__la_symbol_ptr,lazy_symbol_pointers
; DYNAMIC: L_exact_log2$lazy_ptr:
; DYNAMIC: .indirect_symbol _exact_log2
; DYNAMIC: .long dyld_stub_binding_helper

; DYNAMIC64: .section __TEXT,__symbol_stub1,symbol_stubs,pure_instructions,16
; DYNAMIC64: L_exact_log2$stub:
; DYNAMIC64: .indirect_symbol _exact_log2
; DYNAMIC64: lis r11, ha16(L_exact_log2$lazy_ptr)
; DYNAMIC64: ldu r12, lo16(L_exact_log2$lazy_ptr)(r11)
; DYNAMIC64: mtctr r12
; DYNAMIC64: bctr

; DYNAMIC64: .section __DATA,__la_symbol_ptr,lazy_symbol_pointers
; DYNAMIC64: L_exact_log2$lazy_ptr:
; DYNAMIC64: .indirect_symbol _exact_log2
; DYNAMIC64: .quad dyld_stub_binding_helper
