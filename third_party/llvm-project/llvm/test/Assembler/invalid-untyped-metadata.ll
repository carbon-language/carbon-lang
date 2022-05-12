; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; Tests bug: https://llvm.org/bugs/show_bug.cgi?id=24645
; CHECK: error: invalid type for inline asm constraint string

     !3=!    {%..d04 *asm" !6!={!H)4" ,""  
