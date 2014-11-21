; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mattr=+idivl-to-divb < %s | FileCheck -check-prefix=DIV32 %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mattr=+idivq-to-divw < %s | FileCheck -check-prefix=DIV64 %s

define i32 @div32(i32 %a, i32 %b) {
entry:
; DIV32-LABEL: div32:
; DIV32: orl %{{.*}}, [[REG:%[a-z]+]]
; DIV32: testl $-256, [[REG]]
; DIV32: divb
; DIV64-LABEL: div32:
; DIV64-NOT: divb
  %div = sdiv i32 %a, %b
  ret i32 %div
}

define i64 @div64(i64 %a, i64 %b) {
entry:
; DIV32-LABEL: div64:
; DIV32-NOT: divw
; DIV64-LABEL: div64:
; DIV64: orq %{{.*}}, [[REG:%[a-z]+]]
; DIV64: testq   $-65536, [[REG]]
; DIV64: divw
  %div = sdiv i64 %a, %b
  ret i64 %div
}


