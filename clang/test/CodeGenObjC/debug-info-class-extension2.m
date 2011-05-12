// RUN: %clang_cc1  -fobjc-nonfragile-abi -masm-verbose -S -g %s -o - | FileCheck %s
// CHECK: AT_APPLE_objc_class_extension

@interface Foo {} @end

@interface Foo () {
    int *bar;
}
@end

@implementation Foo
@end

void bar(Foo *fptr) {}
