; RUN: llc  < %s -march=mips64el -mcpu=mips4 -target-abi=n64 | FileCheck %s
; RUN: llc  < %s -march=mips64el -mcpu=mips64 -target-abi=n64 | FileCheck %s

@gl = global i64 1250999896321, align 8

; CHECK: 8byte
define i64 @foo1() nounwind readonly {
entry:
  %0 = load i64, i64* @gl, align 8
  ret i64 %0
}

