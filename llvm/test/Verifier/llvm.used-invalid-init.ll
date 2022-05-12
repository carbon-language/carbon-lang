; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

@llvm.used = appending global [1 x i8*] zeroinitializer, section "llvm.metadata"

; CHECK: wrong initalizer for intrinsic global variable
; CHECK-NEXT: [1 x i8*] zeroinitializer
