// RUN: clang-cc -fsyntax-only -triple x86_64-apple-darwin9 -verify %s
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
