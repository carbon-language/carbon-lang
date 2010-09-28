// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that there is an undefined reference to .Lset0, but not .Lset1

        .long	.Lset0
foo:
bar:
.Lset1 = foo - bar
        .long	.Lset1

// CHECK: ('_symbols', [
// CHECK-NOT: Lset1
// CHECK:  (('st_name', 9) # '.Lset0'
// CHECK-NOT: Lset1
// CHECK: (('sh_name', 36) # '.strtab'
