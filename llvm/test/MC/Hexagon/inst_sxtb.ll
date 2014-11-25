;; RUN: llc -mtriple=hexagon-unknown-elf -filetype=obj %s -o - \
;; RUN: | llvm-objdump -s - | FileCheck %s

define i32 @foo (i8 %a)
{
  %1 = sext i8 %a to i32
  ret i32 %1
}

; CHECK:   0000 0040a070 00c09f52
