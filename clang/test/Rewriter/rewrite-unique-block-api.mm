// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: FileCheck -check-prefix LP --input-file=%t-rw.cpp %s
// radar 7630551

void f(void (^b)(char c));

@interface a
- (void)processStuff;
@end

@implementation a
- (void)processStuff {
    f(^(char x) { });
}
@end

@interface b
- (void)processStuff;
@end

@implementation b
- (void)processStuff {
    f(^(char x) { });
}
@end

// CHECK-LP: struct __a__processStuff_block_impl_0
// CHECK-LP: static void __a__processStuff_block_func_0

// CHECK-LP: struct __b__processStuff_block_impl_0
// CHECK-LP: static void __b__processStuff_block_func_0
