; RUN: llc -mcpu=pwr7 -O0 -code-model=medium -fast-isel=false <%s | FileCheck -check-prefix=MEDIUM %s
; RUN: llc -mcpu=pwr7 -O0 -code-model=large -fast-isel=false <%s | FileCheck -check-prefix=LARGE %s

; Test correct code generation for medium and large code model
; for loading a value from the constant pool (TOC-relative).

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define double @test_double_const() nounwind {
entry:
  ret double 0x3F4FD4920B498CF0
}

; MEDIUM: [[VAR:[a-z0-9A-Z_.]+]]:
; MEDIUM: .quad 4562098671269285104
; MEDIUM: test_double_const:
; MEDIUM: addis [[REG1:[0-9]+]], 2, [[VAR]]@toc@ha
; MEDIUM: addi [[REG2:[0-9]+]], [[REG1]], [[VAR]]@toc@l
; MEDIUM: lfd {{[0-9]+}}, 0([[REG2]])

; LARGE: [[VAR:[a-z0-9A-Z_.]+]]:
; LARGE: .quad 4562098671269285104
; LARGE: test_double_const:
; LARGE: addis [[REG1:[0-9]+]], 2, [[VAR]]@toc@ha
; LARGE: ld [[REG2:[0-9]+]], [[VAR]]@toc@l([[REG1]])
; LARGE: lfd {{[0-9]+}}, 0([[REG2]])
