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

// rdar://9255564
typedef short short8 __attribute__((ext_vector_type(8)));

struct foo {
 char a;
 int b;
 long c;
 short8 d;
 int array[4];
 short int bitfield1:5;
 unsigned short bitfield2:11;
 char *string;
};

const char *RetEncode () {
 return @encode(struct foo); // expected-warning {{encoding of 'struct foo' type is incomplete because 'short8' (vector of 8 'short' values) component has unknown encoding}}
}

