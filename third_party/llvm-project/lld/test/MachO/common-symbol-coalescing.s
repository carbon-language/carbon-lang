# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/same-size.s -o %t/same-size.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/smaller-size.s -o %t/smaller-size.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/zero-align.s -o %t/zero-align.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/zero-align-round-up.s -o %t/zero-align-round-up.o

## Check that we pick the definition with the larger size, regardless of
## its alignment.
# RUN: %lld %t/test.o %t/smaller-size.o -order_file %t/order -o %t/test
# RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#ALIGN=8
# RUN: %lld %t/smaller-size.o %t/test.o -order_file %t/order -o %t/test
# RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#ALIGN=8

## When the sizes are equal, we pick the symbol whose file occurs later in the
## command-line argument list.
# RUN: %lld %t/test.o %t/same-size.o -order_file %t/order -o %t/test
# RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#ALIGN=16
# RUN: %lld %t/same-size.o %t/test.o -order_file %t/order -o %t/test
# RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#ALIGN=8

# RUN: %lld %t/test.o %t/zero-align.o -order_file %t/order -o %t/test
# RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#ALIGN=16
# RUN: %lld %t/zero-align.o %t/test.o -order_file %t/order -o %t/test
# RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#ALIGN=16

# RUN: %lld %t/test.o %t/zero-align-round-up.o -order_file %t/order -o %t/test
# RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#ALIGN=16
# RUN: %lld %t/zero-align-round-up.o %t/test.o -order_file %t/order -o %t/test
# RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#ALIGN=16

# CHECK-LABEL: Sections:
# CHECK:       __common      {{[0-9a-f]+}} [[#%.16x, COMMON_START:]]  BSS

# CHECK-LABEL: SYMBOL TABLE:
# CHECK-DAG:   [[#%.16x, COMMON_START]]          g     O __DATA,__common _check_size
# CHECK-DAG:   [[#%.16x, COMMON_START + 2]]      g     O __DATA,__common _end_marker
# CHECK-DAG:   [[#%.16x, COMMON_START + ALIGN]]  g     O __DATA,__common _check_alignment

#--- order
## Order is important as we determine the size of a given symbol via the
## address of the next symbol.
_check_size
_end_marker
_check_alignment

#--- smaller-size.s
.comm _check_size, 1, 1
.comm _check_alignment, 1, 4

#--- same-size.s
.comm _check_size, 2, 1
.comm _check_alignment, 2, 4

#--- zero-align.s
.comm _check_size, 2, 1
## If alignment is set to zero, use the size to determine the alignment.
.comm _check_alignment, 16, 0

#--- zero-align-round-up.s
.comm _check_size, 2, 1
## If alignment is set to zero, use the size to determine the alignment. If the
## size is not a power of two, round it up. (In this case, 14 rounds to 16.)
.comm _check_alignment, 14, 0

#--- test.s
.comm _check_size, 2, 1
.comm _end_marker, 1
.comm _check_alignment, 2, 3

.globl _main
_main:
  ret
