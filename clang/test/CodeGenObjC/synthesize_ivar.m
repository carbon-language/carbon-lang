// RUN: clang-cc -triple x86_64-apple-darwin10 -emit-llvm -o %t %s

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
