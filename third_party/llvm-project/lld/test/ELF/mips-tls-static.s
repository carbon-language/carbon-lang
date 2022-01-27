# REQUIRES: mips
# Check handling TLS related relocations and symbols when linking
# a static executable.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t
# RUN: ld.lld -static %t -o %t.exe
# RUN: llvm-objdump -s -t %t.exe | FileCheck %s

# CHECK: SYMBOL TABLE:
# CHECK:           00000000 g      .tdata          00000000 tls1
# CHECK:  [[TGA:[0-9a-f]+]] g      .text           00000000 __tls_get_addr
#
# CHECK:      Contents of section .data:
# CHECK-NEXT:  {{.*}} [[TGA]] ffff8004 ffff9004
# CHECK:      Contents of section .got:
# CHECK-NEXT:  {{.*}} 00000000 80000000 ffff9000 00000001
# CHECK-NEXT:  {{.*}} ffff8000 00000001 00000000

  .text
  .global __start
__start:
  addiu $2, $3, %tlsgd(tls1)      # R_MIPS_TLS_GD
  addiu $2, $3, %tlsldm(tls2)     # R_MIPS_TLS_LDM
  addiu $2, $3, %gottprel(tls1)   # R_MIPS_TLS_GOTTPREL

  .global __tls_get_addr
__tls_get_addr:
  nop

  .data
loc:
  .word __tls_get_addr
  .dtprelword tls1+4    # R_MIPS_TLS_DTPREL32
  .tprelword tls1+4     # R_MIPS_TLS_TPREL32

 .section .tdata,"awT",%progbits
 .global tls1
tls1:
 .word __tls_get_addr
tls2:
 .word 0
