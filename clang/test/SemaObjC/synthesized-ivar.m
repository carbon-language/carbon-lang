// RUN: clang-cc -fsyntax-only -fobjc-nonfragile-abi -verify %s
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
