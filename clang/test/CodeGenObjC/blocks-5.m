// RUN: %clang_cc1 -fblocks -triple i386-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
// radar 7581175
// XFAIL: *

extern void dispatch_async (void (^)(void));

@interface Foo 
@end

@implementation Foo
- (void)dealloc {
        dispatch_async(^{});
}
@end

// CHECK: self.addr
// CHECK: self.addr
