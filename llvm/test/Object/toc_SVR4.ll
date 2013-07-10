;This isn't really an assembly file, its just here to run the test.
;This test just makes sure that llvm-ar can generate a table of contents for
;SVR4 style archives
;This archive was created on Solaris with /usr/ccs/bin/ar
;RUN: llvm-ar t %p/Inputs/SVR4.a | FileCheck %s
;CHECK:      evenlen
;CHECK-NEXT: oddlen
;CHECK-NEXT: very_long_bytecode_file_name.bc
;CHECK-NEXT: IsNAN.o
