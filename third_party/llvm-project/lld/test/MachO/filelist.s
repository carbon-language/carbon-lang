# REQUIRES: x86

## This test verifies that the paths in -filelist get processed in command-line
## order.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/first.s -o %t/first.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/second.s -o %t/second.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o

# FIRST: __TEXT,first _foo
# SECOND: __TEXT,second _foo

# RUN: echo "%t/first.o" > filelist
# RUN: echo "%t/second.o" >> filelist
# RUN: %lld -filelist filelist %t/test.o -o %t/test
# RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=FIRST

# RUN: echo "%t/second.o" > filelist
# RUN: echo "%t/first.o" >> filelist
# RUN: %lld -filelist filelist %t/test.o -o %t/test
# RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=SECOND

# RUN: echo "%t/first.o" > filelist
# RUN: %lld -filelist filelist %t/second.o %t/test.o -o %t/test
# RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=FIRST
# RUN: %lld %t/second.o -filelist filelist %t/test.o -o %t/test
# RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=SECOND

# RUN: echo "%t/first.o" > filelist-1
# RUN: echo "%t/second.o" > filelist-2
# RUN: %lld -filelist filelist-1 -filelist filelist-2 %t/test.o -o %t/test
# RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=FIRST
# RUN: %lld -filelist filelist-2 -filelist filelist-1 %t/test.o -o %t/test
# RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=SECOND

#--- first.s
.globl _foo
.weak_definition _foo
.section __TEXT,first
_foo:

#--- second.s
.globl _foo
.weak_definition _foo
.section __TEXT,second
_foo:

#--- test.s
.globl _main
_main:
  ret
