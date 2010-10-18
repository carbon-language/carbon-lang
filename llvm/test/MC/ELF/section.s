// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that these names are accepted.

.section	.note.GNU-stack,"",@progbits
.section	.note.GNU-,"",@progbits
.section	-.note.GNU,"",@progbits

// CHECK: ('sh_name', 0x12) # '.note.GNU-stack'
// CHECK: ('sh_name', 0x22) # '.note.GNU-'
// CHECK: ('sh_name', 0x2d) # '-.note.GNU'
