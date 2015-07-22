; RUN: llc -x86-asm-syntax=intel < %s | FileCheck %s -check-prefix=INTEL
; RUN: llc -x86-asm-syntax=att < %s | FileCheck %s -check-prefix=ATT

; INTEL: .intel_syntax noprefix
; ATT-NOT: .intel_syntax noprefix
define i32 @test() {
entry:
  ret i32 0
}
