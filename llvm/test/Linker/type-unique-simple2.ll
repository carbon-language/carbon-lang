; REQUIRES: object-emission

; RUN: llvm-link %S/Inputs/type-unique-simple2-a.ll %S/Inputs/type-unique-simple2-b.ll -S -o %t
; RUN: cat %t | FileCheck %S/Inputs/type-unique-simple2-a.ll -check-prefix=LINK
