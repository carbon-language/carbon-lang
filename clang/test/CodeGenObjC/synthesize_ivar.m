// RUN: clang-cc -arch x86_64 -emit-llvm -o %t %s

@interface I
{
}
@property int IP;
@end

@implementation I
@synthesize IP;
- (int) Meth {
   return IP;
}
@end
