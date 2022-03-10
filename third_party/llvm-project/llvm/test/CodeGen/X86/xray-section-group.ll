; RUN: llc -mtriple=x86_64-unknown-linux-gnu -function-sections < %s | FileCheck %s
; RUN: llc -filetype=obj -o %t -mtriple=x86_64-unknown-linux-gnu -function-sections < %s
; RUN: llvm-objdump --disassemble-all %t | FileCheck %s --check-prefix=CHECK-OBJ

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK: .section .text.foo,"ax",@progbits
  ret i32 0
; CHECK: .section xray_instr_map,"ao",@progbits,foo{{$}}
}

$bar = comdat any
define i32 @bar() nounwind noinline uwtable "function-instrument"="xray-always" comdat($bar) {
; CHECK: .section .text.bar,"axG",@progbits,bar,comdat
  ret i32 1
; CHECK: .section xray_instr_map,"aGo",@progbits,bar,comdat,bar{{$}}
}

; CHECK-OBJ:      section xray_instr_map:
