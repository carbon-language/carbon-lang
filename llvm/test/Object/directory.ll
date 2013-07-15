;RUN: rm -f %T/test.a
;RUN: not llvm-ar r %T/test.a . 2>&1 | FileCheck %s
;CHECK: .: Is a directory

; opening a directory works on freebsd. On windows we just get a
; "permission denied"
;XFAIL: freebsd, win32, mingw32

;RUN: rm -f %T/test.a
;RUN: touch %T/a-very-long-file-name
;RUN: llvm-ar r %T/test.a %s %T/a-very-long-file-name
;RUN: llvm-ar r %T/test.a %T/a-very-long-file-name
;RUN: llvm-ar t %T/test.a | FileCheck -check-prefix=MEMBERS %s
;MEMBERS-NOT: /
;MEMBERS: directory.ll
;MEMBERS: a-very-long-file-name
;MEMBERS-NOT: a-very-long-file-name
