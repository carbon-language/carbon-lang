// RUN: clang-cc -triple i386-apple-darwin9 -emit-llvm %s -o - | FileCheck %s

@interface I0 
@end

@implementation I0 // expected-warning {{'@end' is missing in implementation context}}
- meth { return 0; }

@interface I1 : I0 
@end

@implementation I1 // expected-warning {{'@end' is missing in implementation context}}
-(void) im0 { self = [super init]; }

@interface I2 : I0
- I2meth;
@end

@implementation I2 // expected-warning {{'@end' is missing in implementation context}}
- I2meth { return 0; }

@implementation  I2(CAT) // expected-warning {{'@end' is missing in implementation context}}

// CHECK: @"\01L_OBJC_CLASS_I1" = internal global
