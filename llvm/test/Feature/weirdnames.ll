; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

; Test using double quotes to form names that are not legal in the % form

"&^ " = type { int }
"%.*+ foo" = global "&^ " { int 5 }
"0" = global float 0.0                 ; This CANNOT be %0
