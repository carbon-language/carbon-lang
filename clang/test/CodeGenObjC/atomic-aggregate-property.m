// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -fobjc-nonfragile-abi -fobjc-gc -emit-llvm -o - %s | FileCheck -check-prefix LP64 %s
// rdar: // 7849824

struct s {
  double a, b, c, d;  
};

struct s1 {
    int i;
    id j;
    id k;
};

@interface A 
@property (readwrite) double x;
@property (readwrite) struct s y;
@property (nonatomic, readwrite) struct s1 z;
@end

@implementation A
@synthesize x;
@synthesize y;
@synthesize z;
@end

// CHECK-LP64: call void @objc_copyStruct
// CHECK-LP64: call void @objc_copyStruct
// CHECK-LP64: call void @objc_copyStruct
