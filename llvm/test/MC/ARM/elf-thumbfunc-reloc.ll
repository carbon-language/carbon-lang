; RUN: llc %s -mtriple=thumbv7-linux-gnueabi -relocation-model=pic \
; RUN: -filetype=obj -o - | llvm-readobj -s -sd -r -t | \
; RUN: FileCheck %s

; FIXME: This file needs to be in .s form!
; We want to test relocatable thumb function call,
; but ARMAsmParser cannot handle "bl foo(PLT)" yet

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:32-n32"
target triple = "thumbv7-none--gnueabi"

define void @foo() nounwind {
entry:
  ret void
}

define void @bar() nounwind {
entry:
  call void @foo()
  ret void
}


; make sure that bl 0 <foo> (fff7feff) is correctly encoded
; CHECK: Sections [
; CHECK:   SectionData (
; CHECK:     0000: 704700BF 2DE90048 FFF7FEFF BDE80088
; CHECK:   )
; CHECK: ]

; CHECK:      Relocations [
; CHECK-NEXT:   Section (2) .rel.text {
; CHECK-NEXT:     0x8 R_ARM_THM_CALL foo 0x0
; CHECK-NEXT:   }
; CHECK-NEXT:   Section (7) .rel.ARM.exidx {
; CHECK-NEXT:     0x0 R_ARM_PREL31 .text 0x0
; CHECK-NEXT:     0x8 R_ARM_PREL31 .text 0x0
; CHECK-NEXT:   }
; CHECK-NEXT: ]

; make sure foo is thumb function: bit 0 = 1
; CHECK:      Symbols [
; CHECK:        Symbol {
; CHECK:          Name: foo
; CHECK-NEXT:     Value: 0x1
