// RUN: %clang_cc1 -o - -emit-llvm -g %s | FileCheck %s

// This test verifies that variadic ObjC methods get the
// DW_TAG_unspecified_parameter marker.

@interface Foo
- (void) Bar: (int) n, ...;
@end

@implementation Foo
- (void) Bar: (int) n, ...
{
  // CHECK: metadata ![[NUM:[^,]*]], null, null, null} ; [ DW_TAG_subroutine_type ]
  // CHECK: ![[NUM]] = {{metadata !{null, metadata ![^,]*, metadata ![^,]*, metadata ![^,]*, null}}}    
}
@end
