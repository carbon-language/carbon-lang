// REQUIRES: x86-64-registered-target
// RUN: %clang_cc1 -g -triple x86_64-apple-darwin10 -S -masm-verbose -o - %s | FileCheck %s
// Radar 8801045
// Do not emit AT_MIPS_linkage_name for static variable i

// CHECK:  DW_TAG_variable
// CHECK-NEXT:        .byte   105                     ## DW_AT_name
// CHECK-NEXT:        .byte   0
// CHECK-NEXT:        DW_AT_type
// CHECK-NEXT:        DW_AT_decl_file
// CHECK-NEXT:        DW_AT_decl_line
// CHECK-NEXT:        DW_AT_location

@interface A {
}
-(void) foo;
@end

@implementation A 
-(void)foo {
  static int i = 1;
}
@end

