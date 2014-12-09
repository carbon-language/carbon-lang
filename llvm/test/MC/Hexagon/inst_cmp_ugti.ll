;; RUN: llc -mtriple=hexagon-unknown-elf -filetype=obj %s -o - \
;; RUN: | llvm-objdump -s - | FileCheck %s

define i1 @foo (i32 %a)
{
  %1 = icmp ugt i32 %a, 42
  ret i1 %1
}

; CHECK:  0000 40458075 00404089 00c09f52
