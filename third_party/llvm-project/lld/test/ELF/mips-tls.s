# REQUIRES: mips
# Check MIPS TLS relocations handling.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         %p/Inputs/mips-tls.s -o %t.so.o
# RUN: ld.lld -shared %t.so.o -soname=t.so -o %t.so
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o

# RUN: echo "SECTIONS { \
# RUN:         . = 0x10000; .text : { *(.text) } \
# RUN:         . = 0x30000; .got  : { *(.got)  } \
# RUN:       }" > %t.script

# RUN: ld.lld %t.o %t.so -script %t.script -o %t.exe
# RUN: llvm-objdump -d -s -t --no-show-raw-insn %t.exe \
# RUN:   | FileCheck -check-prefix=DIS %s
# RUN: llvm-readobj -r -A %t.exe | FileCheck %s

# RUN: ld.lld -pie %t.o %t.so -script %t.script -o %t.pie
# RUN: llvm-objdump -d -s -t --no-show-raw-insn %t.pie \
# RUN:   | FileCheck -check-prefix=DIS %s
# RUN: llvm-readobj -r -A %t.pie | FileCheck %s

# RUN: ld.lld -shared %t.o %t.so -script %t.script -o %t-out.so
# RUN: llvm-objdump -d -s -t --no-show-raw-insn %t-out.so \
# RUN:   | FileCheck -check-prefix=DIS-SO %s
# RUN: llvm-readobj -r -A %t-out.so | FileCheck -check-prefix=SO %s

# DIS: 00000000 l      .tdata          00000000 loc
# DIS: 00000000        *UND*           00000000 foo
# DIS: 00000004 g      .tdata          00000000 bar

# DIS:      Contents of section .got:
# DIS-NEXT:  30000 00000000 80000000 00000000 ffff9000
# DIS-NEXT:  30010 ffff9004 00000000 00000000 00000001
# DIS-NEXT:  30020 00000000 00000001 ffff8004

# DIS:      <__start>:
# DIS-NEXT:    addiu   $2, $3, -32732
# DIS-NEXT:    addiu   $2, $3, -32744
# DIS-NEXT:    addiu   $2, $3, -32724
# DIS-NEXT:    addiu   $2, $3, -32740
# DIS-NEXT:    addiu   $2, $3, -32716
# DIS-NEXT:    addiu   $2, $3, -32736

# CHECK:      Relocations [
# CHECK-NEXT:   Section (7) .rel.dyn {
# CHECK-NEXT:     0x30008 R_MIPS_TLS_TPREL32 foo
# CHECK-NEXT:     0x30014 R_MIPS_TLS_DTPMOD32 foo
# CHECK-NEXT:     0x30018 R_MIPS_TLS_DTPREL32 foo
# CHECK-NEXT:   }
# CHECK-NEXT: ]
# CHECK:      Primary GOT {
# CHECK-NEXT:   Canonical gp value: 0x37FF0
# CHECK-NEXT:   Reserved entries [
# CHECK:        ]
# CHECK-NEXT:   Local entries [
# CHECK-NEXT:   ]
# CHECK-NEXT:   Global entries [
# CHECK-NEXT:   ]
# CHECK-NEXT:   Number of TLS and multi-GOT entries: 9
#               ^-- -32744 R_MIPS_TLS_GOTTPREL R_MIPS_TLS_TPREL32  foo
#               ^-- -32740 R_MIPS_TLS_GOTTPREL VA - 0x7000 loc
#               ^-- -32736 R_MIPS_TLS_GOTTPREL VA - 0x7000 bar
#               ^-- -32732 R_MIPS_TLS_GD       R_MIPS_TLS_DTPMOD32 foo
#               ^-- -32728                     R_MIPS_TLS_DTPREL32 foo
#               ^-- -32724 R_MIPS_TLS_LDM      1 loc
#               ^-- -32720                     0 loc
#               ^-- -32716 R_MIPS_TLS_GD       1 bar
#               ^-- -32712                     VA - 0x8000 bar

# DIS-SO:      Contents of section .got:
# DIS-SO-NEXT:  30000 00000000 80000000 00000000 00000000
# DIS-SO-NEXT:  30010 00000004 00000000 00000000 00000000
# DIS-SO-NEXT:  30020 00000000 00000000 00000000

# SO:      Relocations [
# SO-NEXT:   Section (7) .rel.dyn {
# SO-NEXT:     0x3000C R_MIPS_TLS_TPREL32 -
# SO-NEXT:     0x3001C R_MIPS_TLS_DTPMOD32 -
# SO-NEXT:     0x30008 R_MIPS_TLS_TPREL32 foo
# SO-NEXT:     0x30014 R_MIPS_TLS_DTPMOD32 foo
# SO-NEXT:     0x30018 R_MIPS_TLS_DTPREL32 foo
# SO-NEXT:     0x30010 R_MIPS_TLS_TPREL32 bar
# SO-NEXT:     0x30024 R_MIPS_TLS_DTPMOD32 bar
# SO-NEXT:     0x30028 R_MIPS_TLS_DTPREL32 bar
# SO-NEXT:   }
# SO-NEXT: ]
# SO:      Primary GOT {
# SO-NEXT:   Canonical gp value: 0x37FF0
# SO-NEXT:   Reserved entries [
# SO:        ]
# SO-NEXT:   Local entries [
# SO-NEXT:   ]
# SO-NEXT:   Global entries [
# SO-NEXT:   ]
# SO-NEXT:   Number of TLS and multi-GOT entries: 9
#            ^-- -32744 R_MIPS_TLS_GOTTPREL R_MIPS_TLS_TPREL32  foo
#            ^-- -32740 R_MIPS_TLS_GOTTPREL R_MIPS_TLS_TPREL32  loc
#            ^-- -32736 R_MIPS_TLS_GOTTPREL R_MIPS_TLS_TPREL32  bar
#            ^-- -32732 R_MIPS_TLS_GD       R_MIPS_TLS_DTPMOD32 foo
#            ^-- -32728 R_MIPS_TLS_DTPREL32 foo
#            ^-- -32724 R_MIPS_TLS_LDM      R_MIPS_TLS_DTPMOD32 loc
#            ^-- -32720 0 loc
#            ^-- -32716 R_MIPS_TLS_GD       R_MIPS_TLS_DTPMOD32 bar
#            ^-- -32712 R_MIPS_TLS_DTPREL32 bar

  .text
  .global  __start
__start:
  addiu $2, $3, %tlsgd(foo)     # R_MIPS_TLS_GD
  addiu $2, $3, %gottprel(foo)  # R_MIPS_TLS_GOTTPREL
  addiu $2, $3, %tlsldm(loc)    # R_MIPS_TLS_LDM
  addiu $2, $3, %gottprel(loc)  # R_MIPS_TLS_GOTTPREL
  addiu $2, $3, %tlsgd(bar)     # R_MIPS_TLS_GD
  addiu $2, $3, %gottprel(bar)  # R_MIPS_TLS_GOTTPREL

 .section .tdata,"awT",%progbits
 .global bar
loc:
 .word 0
bar:
 .word 0
