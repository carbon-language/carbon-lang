; RUN: llc -verify-machineinstrs -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s

; When generating DAG selection tables, TableGen used to only flag an
; instruction as needing a chain on its own account if it had a built-in pattern
; which used the chain. This meant that the AArch64 load/stores weren't
; recognised and so both loads from %locvar below were coalesced into a single
; LS8_LDR instruction (same operands other than the non-existent chain) and the
; increment was lost at return.

; This was obviously a Bad Thing.

declare void @bar(i8*)

define i64 @test_chains() {
; CHECK-LABEL: test_chains:

  %locvar = alloca i8

  call void @bar(i8* %locvar)
; CHECK: bl bar

  %inc.1 = load i8* %locvar
  %inc.2 = zext i8 %inc.1 to i64
  %inc.3 = add i64 %inc.2, 1
  %inc.4 = trunc i64 %inc.3 to i8
  store i8 %inc.4, i8* %locvar
; CHECK: ldrb {{w[0-9]+}}, [sp, [[LOCADDR:#[0-9]+]]]
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, #1
; CHECK: strb {{w[0-9]+}}, [sp, [[LOCADDR]]]
; CHECK: ldrb {{w[0-9]+}}, [sp, [[LOCADDR]]]

  %ret.1 = load i8* %locvar
  %ret.2 = zext i8 %ret.1 to i64
  ret i64 %ret.2
; CHECK: ret
}
