# REQUIRES: x86

## i386-got32x-baseless.elf is a file produced using GNU as v.2.27
## using following code and command line:
## (as --32 -o base.o base.s)
##
## .text
## .globl foo
## .type foo, @function
## foo:
##  nop
##
## _start:
##  movl foo@GOT, %eax
##  movl foo@GOT, %ebx
##  movl foo@GOT(%eax), %eax
##  movl foo@GOT(%ebx), %eax
##
## Result file contains four R_386_GOT32X relocations. Generated code
## is also a four mov instructions. And first two has no base register:
## <_start>:
##   1: 8b 05 00 00 00 00 mov 0x0,%eax
##   7: 8b 1d 00 00 00 00 mov 0x0,%ebx
##   d: 8b 80 00 00 00 00 mov 0x0(%eax),%eax
##  13: 8b 83 00 00 00 00 mov 0x0(%ebx),%eax
##
## R_386_GOT32X is computed as G + A - GOT, but if it used without base
## register, it should be calculated as G + A. Using without base register
## is only allowed for non-PIC code.
##
# RUN: ld.lld %S/Inputs/i386-got32x-baseless.elf -o %t1
# RUN: llvm-objdump --section-headers -d %t1 | FileCheck %s

## 73728 == 0x12000 == ADDR(.got)
# CHECK: Sections:
# CHECK:  Name     Size     VMA
# CHECK:  .got.plt 0000000c 00403134
# CHECK:       <_start>:
# CHECK-NEXT:   401115: 8b 05 {{.*}} movl 4202800, %eax
# CHECK-NEXT:   40111b: 8b 1d {{.*}} movl 4202800, %ebx
# CHECK-NEXT:   401121: 8b 80 {{.*}} movl -4100(%eax), %eax
# CHECK-NEXT:   401127: 8b 83 {{.*}} movl -4100(%ebx), %eax

# RUN: not ld.lld %S/Inputs/i386-got32x-baseless.elf -o /dev/null -pie 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR
# ERR-COUNT-2: error: symbol 'foo' cannot be preempted; recompile with -fPIE
