// RUN: clang-cc -fobjc-nonfragile-abi -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-X86-64 %s

__attribute__((weak_import)) @interface WeakClass 
@end

@interface MySubclass : WeakClass @end

@implementation MySubclass @end

@implementation WeakClass(MyCategory) @end


__attribute__((weak_import))
@interface WeakClass1 @end

@implementation WeakClass1(MyCategory) @end

@implementation WeakClass1(YourCategory) @end

 __attribute__((weak_import))
@interface WeakClass3 
+ message;
@end

int main() {
     [WeakClass3 message];
}

// CHECK-X86-64: OBJC_METACLASS_$_WeakClass" = extern_weak global
// CHECK-X86-64: OBJC_CLASS_$_WeakClass" = extern_weak global
// CHECK-X86-64: OBJC_CLASS_$_WeakClass1" = extern_weak global
// CHECK-X86-64: OBJC_CLASS_$_WeakClass3" = extern_weak global


