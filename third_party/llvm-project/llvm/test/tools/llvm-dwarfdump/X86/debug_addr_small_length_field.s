# RUN: llvm-mc %s -filetype obj -triple i386-pc-linux -o - | \
# RUN: not llvm-dwarfdump -debug-addr - 2> %t.err | FileCheck %s
# RUN: FileCheck %s -input-file %t.err -check-prefix=ERR

# CHECK: .debug_addr contents:
# CHECK-NOT: {{.}}
# ERR: address table at offset 0x0 has a unit_length value of 0x1, which is too small to contain a complete header
# ERR-NOT: {{.}}

# too small length value
  .section  .debug_addr,"",@progbits
.Ldebug_addr0:
  .long 1  # unit_length
  .short 5 # version
  .byte 4  # address_size
  .byte 0  # segment_selector_size
  .long 0x00000000
  .long 0x00000001
