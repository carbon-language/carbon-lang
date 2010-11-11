// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump   | FileCheck %s

// Test that we produce the two group sections and that they are a the beginning
// of the file.

// CHECK:       # Section 0x00000001
// CHECK-NEXT:  (('sh_name', 0x00000021) # '.group'
// CHECK-NEXT:   ('sh_type', 0x00000011)
// CHECK-NEXT:   ('sh_flags', 0x00000000)
// CHECK-NEXT:   ('sh_addr', 0x00000000)
// CHECK-NEXT:   ('sh_offset', 0x00000040)
// CHECK-NEXT:   ('sh_size', 0x0000000c)
// CHECK-NEXT:   ('sh_link', 0x0000000a)
// CHECK-NEXT:   ('sh_info', 0x00000001)
// CHECK-NEXT:   ('sh_addralign', 0x00000004)
// CHECK-NEXT:   ('sh_entsize', 0x00000004)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Section 0x00000002
// CHECK-NEXT:  (('sh_name', 0x00000021) # '.group'
// CHECK-NEXT:   ('sh_type', 0x00000011)
// CHECK-NEXT:   ('sh_flags', 0x00000000)
// CHECK-NEXT:   ('sh_addr', 0x00000000)
// CHECK-NEXT:   ('sh_offset', 0x0000004c)
// CHECK-NEXT:   ('sh_size', 0x00000008)
// CHECK-NEXT:   ('sh_link', 0x0000000a)
// CHECK-NEXT:   ('sh_info', 0x00000002)
// CHECK-NEXT:   ('sh_addralign', 0x00000004)
// CHECK-NEXT:   ('sh_entsize', 0x00000004)

	.section	.foo,"axG",@progbits,g1,comdat
g1:
        nop

        .section	.bar,"axG",@progbits,g1,comdat
        nop

        .section	.zed,"axG",@progbits,g2,comdat
g2:
        nop
