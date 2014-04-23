// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://16655340
@protocol X, Y, Z;
@class Foo;

@protocol Proto
@end

@interface Intf <Proto>
{
id <X> IVAR_x;
id <X, Y> IVAR_xy;
id <X, Y, Z> IVAR_xyz;
Foo <X, Y, Z> *IVAR_Fooxyz;
Class <X> IVAR_Classx;
}
@end

@implementation Intf 
@end

int main()
{
    int i;
    typeof(@encode(typeof(i))) e = @encode(typeof(Intf)); // expected-warning {{initializer-string for char array is too long}}
}
