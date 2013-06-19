;RUN: not llvm-ar r %T/test.a . 2>&1 | FileCheck %s
;CHECK: . Is a directory
