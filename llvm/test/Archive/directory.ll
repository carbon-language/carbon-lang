;RUN: not llvm-ar r %T/test.a . 2>&1 | FileCheck %s
;CHECK: . Is a directory

;RUN: rm -f %T/test.a
;RUN: llvm-ar r %T/test.a %s
;RUN: llvm-ar t %T/test.a | FileCheck -check-prefix=MEMBERS %s
;MEMBERS-NOT: /
;MEMBERS: directory.ll
