// RUN: %clang_cc1 -fblocks -fobjc-gc -triple x86_64-apple-darwin -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s
// RUN: %clang_cc1 -fblocks -fobjc-gc -triple i386-apple-darwin -S %s -o %t-32.s
// RUN: FileCheck -check-prefix LP32 --input-file=%t-32.s %s

@interface MyView
- (void)MyView_sharedInit;
@end

void foo(MyView *(^obj)(void)) ;

@implementation MyView
- (void)MyView_sharedInit {
    
    __block __weak MyView *weakSelf = self;
    foo(
    ^{
	return weakSelf;
    });

}
@end

// CHECK-LP64: callq    _objc_read_weak
// CHECK-LP64: callq    _objc_read_weak

// CHECK-LP32: calll     L_objc_read_weak
// CHECK-LP32: calll     L_objc_read_weak

