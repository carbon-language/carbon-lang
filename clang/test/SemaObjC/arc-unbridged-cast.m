// // RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fobjc-nonfragile-abi -fsyntax-only -fobjc-arc -verify %s
// rdar://9744349

typedef const struct __CFString * CFStringRef;

@interface I 
@property CFStringRef P;
@end

@implementation I
@synthesize P;
- (id) Meth {
    I* p1 = (id)[p1 P];
    id p2 = (__bridge_transfer id)[p1 P];
    id p3 = (__bridge I*)[p1 P];
    return (id) p1.P;
}
@end
