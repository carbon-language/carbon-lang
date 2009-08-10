; RUN: llvm-as < %s | llc -relocation-model=static | FileCheck %s -check-prefix=STATIC
; RUN: llvm-as < %s | llc -relocation-model=pic | FileCheck %s -check-prefix=PIC
; RUN: llvm-as < %s | llc -relocation-model=dynamic-no-pic | FileCheck %s -check-prefix=DYNAMIC
; PR4482
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "armv6-apple-darwin2"

define i32 @foo(i64 %x) nounwind {
entry:
; STATIC: _foo:
; STATIC: bl _exact_log2
; STATIC: ldmfd sp!, {r7, pc}
; STATIC: .subsections_via_symbols

; PIC: _foo:
; PIC: bl L_exact_log2$stub
; PIC: ldmfd sp!, {r7, pc}

; DYNAMIC: _foo:
; DYNAMIC: bl L_exact_log2$stub
; DYNAMIC: ldmfd sp!, {r7, pc}

 	%A = call i32 @exact_log2(i64 %x)
	ret i32 %A
}

define available_externally i32 @exact_log2(i64 %x) nounwind {
  ret i32 4
}


; PIC: .section __TEXT,__picsymbolstub4,symbol_stubs,none,16
; PIC: L_exact_log2$stub:
; PIC: .indirect_symbol _exact_log2
; PIC: ldr ip, L_exact_log2$slp
; PIC: L_exact_log2$scv:
; PIC: add ip, pc, ip
; PIC: ldr pc, [ip, #0]
; PIC: L_exact_log2$slp:
; PIC: .long	L_exact_log2$lazy_ptr-(L_exact_log2$scv+8)

; PIC: .section __DATA,__la_symbol_ptr,lazy_symbol_pointers
; PIC: L_exact_log2$lazy_ptr:
; PIC: .indirect_symbol _exact_log2
; PIC: .long	dyld_stub_binding_helper

; PIC: .subsections_via_symbols


; DYNAMIC: .section __TEXT,__symbol_stub4,symbol_stubs,none,12
; DYNAMIC: L_exact_log2$stub:
; DYNAMIC: .indirect_symbol _exact_log2
; DYNAMIC: ldr ip, L_exact_log2$slp
; DYNAMIC: ldr pc, [ip, #0]
; DYNAMIC: L_exact_log2$slp:
; DYNAMIC: .long	L_exact_log2$lazy_ptr

; DYNAMIC: .section __DATA,__la_symbol_ptr,lazy_symbol_pointers
; DYNAMIC: L_exact_log2$lazy_ptr:
; DYNAMIC: .indirect_symbol _exact_log2
; DYNAMIC: .long	dyld_stub_binding_helper
; DYNAMIC: .subsections_via_symbols





