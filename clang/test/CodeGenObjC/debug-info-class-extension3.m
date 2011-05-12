// RUN: %clang_cc1 -fobjc-nonfragile-abi -masm-verbose -S -g %s -o - | FileCheck %s

// CHECK-NOT: AT_APPLE_objc_class_extension

@interface Foo {} @end

@interface Foo () {
    int *bar;
}
@end

void bar(Foo *fptr) {}
