// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,c -xc %s

#ifdef __OBJC__
#if !__has_feature(objc_fixed_enum)
#  error Enumerations with a fixed underlying type are not supported
#endif
#endif

#if !__has_extension(cxx_fixed_enum)
#  error Enumerations with a fixed underlying type are not supported
#endif

typedef long Integer;

typedef enum : Integer { Enumerator1, Enumerator2 } Enumeration;

int array[sizeof(Enumeration) == sizeof(long)? 1 : -1];


enum Color { Red, Green, Blue };

struct X { 
  enum Color : 4;
  enum Color field1: 4;
  enum Other : Integer field2; // c-error {{only permitted as a standalone}}
  enum Other : Integer field3 : 4; // c-error {{only permitted as a standalone}}
  enum  : Integer { Blah, Blarg } field4 : 4;
};

void test(void) {
  long value = 2;
  Enumeration e = value;
}

// <rdar://10381507>
typedef enum : long { Foo } IntegerEnum;
int arr[(sizeof(__typeof__(Foo)) == sizeof(__typeof__(IntegerEnum)))? 1 : -1];
int arr1[(sizeof(__typeof__(Foo)) == sizeof(__typeof__(long)))? 1 : -1];
int arr2[(sizeof(__typeof__(IntegerEnum)) == sizeof(__typeof__(long)))? 1 : -1];

// <rdar://problem/10760113>
typedef enum : long long { Bar = -1 } LongLongEnum;
int arr3[(long long)Bar == (long long)-1 ? 1 : -1];

typedef enum : Integer { BaseElem } BaseEnum;
typedef enum : BaseEnum { DerivedElem } DerivedEnum; // expected-error {{non-integral type 'BaseEnum' is an invalid underlying type}}

// <rdar://problem/24999533>
enum MyEnum : _Bool {
  MyThing = 0
};
