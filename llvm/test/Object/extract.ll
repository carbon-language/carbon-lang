; This isn't really an assembly file, its just here to run the test.

; This test just makes sure that llvm-ar can extract bytecode members
; from various style archives.

; REQUIRES: shell

; RUN: cd %T

; RUN: rm -f very_long_bytecode_file_name.bc
; RUN: llvm-ar p %p/Inputs/GNU.a very_long_bytecode_file_name.bc | \
; RUN:   cmp -s %p/Inputs/very_long_bytecode_file_name.bc -
; RUN: llvm-ar x %p/Inputs/GNU.a very_long_bytecode_file_name.bc
; RUN: cmp -s %p/Inputs/very_long_bytecode_file_name.bc \
; RUN:        very_long_bytecode_file_name.bc

; RUN: rm -f very_long_bytecode_file_name.bc
; RUN: llvm-ar p %p/Inputs/MacOSX.a very_long_bytecode_file_name.bc | \
; RUN:   cmp -s %p/Inputs/very_long_bytecode_file_name.bc -
; RUN: llvm-ar x %p/Inputs/MacOSX.a very_long_bytecode_file_name.bc
; RUN: cmp -s %p/Inputs/very_long_bytecode_file_name.bc \
; RUN:        very_long_bytecode_file_name.bc

; RUN: rm -f very_long_bytecode_file_name.bc
; RUN: llvm-ar p %p/Inputs/SVR4.a very_long_bytecode_file_name.bc | \
; RUN:   cmp -s %p/Inputs/very_long_bytecode_file_name.bc -
; RUN: llvm-ar x %p/Inputs/SVR4.a very_long_bytecode_file_name.bc
; RUN: cmp -s %p/Inputs/very_long_bytecode_file_name.bc \
; RUN:        very_long_bytecode_file_name.bc

; RUN: rm -f very_long_bytecode_file_name.bc
; RUN: llvm-ar p %p/Inputs/xpg4.a very_long_bytecode_file_name.bc |\
; RUN:   cmp -s %p/Inputs/very_long_bytecode_file_name.bc -
; RUN: llvm-ar x %p/Inputs/xpg4.a very_long_bytecode_file_name.bc
; RUN: cmp -s %p/Inputs/very_long_bytecode_file_name.bc \
; RUN:        very_long_bytecode_file_name.bc


; Test that the 'o' option is working by extracting a file, putting it in o
; new archive and checking that date.
; RUN: rm -f very_long_bytecode_file_name.bc
; RUN: llvm-ar xo %p/Inputs/GNU.a very_long_bytecode_file_name.bc
; RUN: llvm-ar rc %t.a very_long_bytecode_file_name.bc
; RUN: env TZ=GMT llvm-ar tv %t.a | FileCheck %s

CHECK: rwxr-xr-x 1000/1000 1465 Nov 19 03:01 2004 very_long_bytecode_file_name.bc
