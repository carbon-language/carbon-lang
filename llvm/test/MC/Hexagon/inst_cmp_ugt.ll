;; RUN: llc -mtriple=hexagon-unknown-elf -filetype=obj %s -o - \
;; RUN: | llvm-objdump -s - | FileCheck %s

define i1 @foo (i32 %a, i32 %b)
{
  %1 = icmp ugt i32 %a, %b
  ret i1 %1
}

; CHECK:  0000 004160f2 00400000 00c09f52
