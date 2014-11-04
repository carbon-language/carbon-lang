// RUN: llvm-mc -triple x86_64-apple-darwin9 %s -filetype=obj -o - | llvm-readobj -r | FileCheck %s

	.data
L_var1:
L_var2:
	.long L_var2 - L_var1
	.set L_var3, .
	.set L_var4, .
	.long L_var4 - L_var3

// CHECK:      Relocations [
// CHECK-NEXT:   Section __data {
// CHECK-NEXT:     0x4 0 2 0 X86_64_RELOC_SUBTRACTOR 0 0x2
// CHECK-NEXT:     0x4 0 2 0 X86_64_RELOC_UNSIGNED 0 0x2
// CHECK-NEXT:   }
// CHECK-NEXT: ]
