;This isn't really an assembly file, its just here to run the test.
;This test just makes sure that llvm-ar can generate a symbol table for
;GNU style archives
;RUN: cp %p/GNU.a GNU_copy.a
;RUN: llvm-ranlib GNU_copy.a
;RUN: llvm-ar t GNU_copy.a > %t1
;RUN: sed -e '/^;.*/d' %s >%t2
;RUN: diff %t2 %t1
evenlen
oddlen
very_long_bytecode_file_name.bc
IsNAN.o
