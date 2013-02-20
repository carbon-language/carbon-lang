// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -o - %s -fsanitize=address | FileCheck -check-prefix ASAN %s

@interface MyClass
+ (int) addressSafety:(int*)a;
@end

@implementation MyClass

// CHECK:  +[MyClass load]{{.*}}#0
// ASAN: +[MyClass load]{{.*}}#0
+(void) load { }

// CHECK:  +[MyClass addressSafety:]{{.*}}#0
// ASAN:  +[MyClass addressSafety:]{{.*}}#0
+ (int) addressSafety:(int*)a { return *a; }

@end

// CHECK: attributes #0 = { nounwind "target-features"={{.*}} }
// ASAN: attributes #0 = { address_safety nounwind "target-features"={{.*}} }
