// RUN: clang-cc -emit-llvm -triple=i686-apple-darwin8 -o %t %s &&
// RUN: grep -e "SiSetOtherThings:" %t

@interface A 
@property(setter=iSetOtherThings:) int otherThings;
@end

@implementation A
@dynamic otherThings;
@end
