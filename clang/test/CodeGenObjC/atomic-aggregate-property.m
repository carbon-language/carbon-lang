// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -fobjc-nonfragile-abi -emit-llvm -o - %s | FileCheck -check-prefix LP64 %s
// rdar: // 7849824

struct s {
  double a, b, c, d;  
};

@interface A 
@property (readwrite) double x;
@property (readwrite) struct s y;
@end

@implementation A
@synthesize x;
@synthesize y;
@end

// CHECK-LP64: call void @objc_copyStruct
// CHECK-LP64: call void @objc_copyStruct
