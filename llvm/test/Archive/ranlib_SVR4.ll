;This isn't really an assembly file, its just here to run the test.
;This test just makes sure that llvm-ar can generate a symbol table for
;SVR4 style archives
;RUN: cp %p/SVR4.a %t.SVR4.a
;RUN: llvm-ranlib %t.SVR4.a
;RUN: llvm-ar t %t.SVR4.a > %t1
;RUN: diff %t1 %p/SVR4.toc

