// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -g %s -o - | FileCheck %s

__attribute((objc_root_class)) @interface NSObject {
    id isa;
}
@end

@interface BaseClass : NSObject
{
    int i;
    unsigned flag_1 : 9;
    unsigned flag_2 : 9;
    unsigned : 1;
    unsigned flag_3 : 9;
}
@end

@implementation BaseClass
@end

// CHECK: {{.*}} [ DW_TAG_member ] [i] [line 10, size 32, align 32, offset 0] [protected] [from int]
// CHECK: {{.*}} [ DW_TAG_member ] [flag_1] [line 11, size 9, align 32, offset 0] [protected] [from unsigned int]
// CHECK: {{.*}} [ DW_TAG_member ] [flag_2] [line 12, size 9, align 32, offset 1] [protected] [from unsigned int]
// CHECK: {{.*}} [ DW_TAG_member ] [flag_3] [line 14, size 9, align 32, offset 3] [protected] [from unsigned int]
