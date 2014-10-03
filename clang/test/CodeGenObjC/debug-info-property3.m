// RUN: %clang_cc1 -S -emit-llvm -g %s -o - | FileCheck %s

// CHECK: metadata !{metadata !"0x4200\00p1\005\00\00\002316", {{.*}}} ; [ DW_TAG_APPLE_property ]
@interface I1
@property int p1;
@end

@implementation I1
@synthesize p1;
@end

void foo(I1 *iptr) {}
