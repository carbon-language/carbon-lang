// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s -t | FileCheck %s

// Test that we produce two foo sections, each in separate groups

// CHECK: Index: 1
// CHECK-NEXT: Name: .group

// CHECK: Index: 2
// CHECK-NEXT: Name: .group

// CHECK: Index: 6
// CHECK-NEXT: Name: .foo

// CHECK: Index: 7
// CHECK-NEXT: Name: .foo

// CHECK: Symbols [

// CHECK: Name: f1
// CHECK-NOT: }
// CHECK: Section: .group (0x1)

// CHECK: Name: f2
// CHECK-NOT: }
// CHECK: Section: .group (0x2)

// CHECK: Name: .foo
// CHECK-NOT: }
// CHECK: Section: .foo (0x6)

// CHECK: Name: .foo
// CHECK-NOT: }
// CHECK: Section: .foo (0x7)


	.section	.foo,"axG",@progbits,f1,comdat
        nop

	.section	.foo,"axG",@progbits,f2,comdat
        nop

