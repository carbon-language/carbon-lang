;; RUN: llc -mtriple=hexagon-unknown-elf -filetype=obj %s -o - \
;; RUN: | llvm-objdump -s - | FileCheck %s

define i32 @foo (i1 %a, i32 %b, i32 %c)
{
  %1 = select i1 %a, i32 %b, i32 %c
  ret i32 %1
}

; CHECK:  0000 00400000 004201f4 00c09f52
