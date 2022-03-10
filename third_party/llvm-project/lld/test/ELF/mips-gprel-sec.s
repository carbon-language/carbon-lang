# REQUIRES: mips
# Check order of gp-relative sections, i.e. sections with SHF_MIPS_GPREL flag.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -shared -o %t.so
# RUN: llvm-readobj -S %t.so | FileCheck %s

  .text
  nop

  .sdata
  .word 0

# CHECK:      Section {
# CHECK:        Name: .got
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:     SHF_MIPS_GPREL
# CHECK-NEXT:     SHF_WRITE
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address:
# CHECK-NEXT:   Offset:
# CHECK:      }
# CHECK:      Section {
# CHECK-NEXT:   Index:
# CHECK-NEXT:   Name: .sdata
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:     SHF_MIPS_GPREL
# CHECK-NEXT:     SHF_WRITE
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address:
# CHECK-NEXT:   Offset:
# CHECK:      }
