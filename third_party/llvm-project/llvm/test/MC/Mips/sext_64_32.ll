; RUN: llc -march=mips64el -filetype=obj -mcpu=mips64r2 %s -o - | \
; RUN:   llvm-objdump -d - | FileCheck %s

; Sign extend from 32 to 64 was creating nonsense opcodes

; CHECK: sll ${{[a-z0-9]+}}, ${{[a-z0-9]+}}, 0

define i64 @foo(i32 %ival) nounwind readnone {
entry:
  %conv = sext i32 %ival to i64
  ret i64 %conv
}

; CHECK-LABEL: <foo_2>:
; CHECK: dext ${{[a-z0-9]+}}, ${{[a-z0-9]+}}, 0, 32

define i64 @foo_2(i32 %ival_2) nounwind readnone {
entry:
  %conv_2 = zext i32 %ival_2 to i64
  ret i64 %conv_2
}

