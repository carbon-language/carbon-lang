; REQUIRES: asserts

; RUN: llvm-as %s -o %t.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so  -shared \
; RUN:    -plugin-opt=-stats %t.o -o %t2 2>&1 | FileCheck %s

; CHECK: Statistics Collected
