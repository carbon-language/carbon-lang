// RUN: %clang_cc1 -triple x86_64-apple-darwin9  -fobjc-nonfragile-abi -emit-llvm -o - %s | FileCheck -check-prefix LP64 %s
// rdar: // 7866951

@interface NSObject 
- (id)init;
@end

@interface ClangTest : NSObject @end

@implementation ClangTest
- (id)init
{
 [super init];
 return self;
}
@end

// CHECK-LP64: objc_msgSendSuper2_fixup_init
// CHECK-LP64: objc_msgSendSuper2_fixup
