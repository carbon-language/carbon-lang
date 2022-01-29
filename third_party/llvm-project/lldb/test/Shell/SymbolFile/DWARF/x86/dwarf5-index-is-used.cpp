// Test that we use the DWARF v5 name indexes.

// REQUIRES: lld

// RUN: %clang %s -c -o %t.o --target=x86_64-pc-linux -gdwarf-5 -gpubnames
// RUN: ld.lld %t.o -o %t
// RUN: lldb-test symbols %t | FileCheck %s

// CHECK: Name Index
// CHECK: String: 0x{{.*}} "_start"
// CHECK: Tag: DW_TAG_subprogram

extern "C" void _start() {}
