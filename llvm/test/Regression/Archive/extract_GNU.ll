;This isn't really an assembly file, its just here to run the test.
;This test just makes sure that llvm-ar can extract bytecode members
;from GNU style archives
;RUN: llvm-ar x %p/GNU.a very_long_bytecode_file_name.bc
;RUN: diff -q %p/very_long_bytecode_file_name.bc very_long_bytecode_file_name.bc
