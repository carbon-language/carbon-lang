// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s -t | FileCheck %s

// Test that this produces an undefined reference to .Lfoo

        je	.Lfoo

// CHECK:       Section {
// CHECK:         Name: .strtab

// CHECK:       Symbol {
// CHECK:         Name: .Lfoo
// CHECK-NEXT:    Value:
// CHECK-NEXT:    Size:
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type:
// CHECK-NEXT:    Other:
// CHECK-NEXT:    Section:
// CHECK-NEXT:  }
