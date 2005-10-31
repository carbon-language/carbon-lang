;This isn't really an assembly file, its just here to run the test.
;This test just makes sure that llvm-ar can generate a symbol table for
;GNU style archives
;RUN: cp %p/GNU.a %t.GNU.a
;RUN: llvm-ranlib %t.GNU.a
;RUN: llvm-ar t %t.GNU.a > %t1
;RUN: diff %t1 %p/GNU.toc

; XFAIL: alpha
