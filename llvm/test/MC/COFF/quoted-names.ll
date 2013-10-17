; Check that certain symbol and section names are quoted in the asm output.
; RUN: llc -mtriple=i686-pc-win32 %s -o - | FileCheck %s

; Check that the symbol and section names can round-trip through the assembler.
; RUN: llc -mtriple=i686-pc-win32 %s -o - | llvm-mc -triple i686-pc-win32 -filetype=obj | llvm-readobj -s -section-symbols | FileCheck %s --check-prefix=READOBJ

@"\01??__E_Generic_object@?$_Error_objects@H@std@@YAXXZ" = global i32 0

define weak i32 @"\01??_B?$num_put@_WV?$back_insert_iterator@V?$basic_string@_WU?$char_traits@_W@std@@V?$allocator@_W@2@@std@@@std@@@std@@51"() section ".text" {
  %res = load i32* @"\01??__E_Generic_object@?$_Error_objects@H@std@@YAXXZ"
  ret i32 %res
}

; CHECK: .section ".text$??_B?$num_put@_WV?$back_insert_iterator@V?$basic_string@_WU?$char_traits@_W@std@@V?$allocator@_W@2@@std@@@std@@@std@@51","xr"
; CHECK: .globl "??_B?$num_put@_WV?$back_insert_iterator@V?$basic_string@_WU?$char_traits@_W@std@@V?$allocator@_W@2@@std@@@std@@@std@@51"
; CHECK: "??_B?$num_put@_WV?$back_insert_iterator@V?$basic_string@_WU?$char_traits@_W@std@@V?$allocator@_W@2@@std@@@std@@@std@@51"

; READOBJ: Symbol
; READOBJ: Name: ??_B?$num_put@_WV?$back_insert_iterator@V?$basic_string@_WU?$char_traits@_W@std@@V?$allocator@_W@2@@std@@@std@@@std@@51
; READOBJ: Section: .text$??_B?$num_put@_WV?$back_insert_iterator@V?$basic_string@_WU?$char_traits@_W@std@@V?$allocator@_W@2@@std@@@std@@@std@@51
