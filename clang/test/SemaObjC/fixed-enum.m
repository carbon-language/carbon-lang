// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar://10381507

typedef enum : long { Foo } IntegerEnum;
int arr[(sizeof(typeof(Foo)) == sizeof(typeof(IntegerEnum))) - 1];
int arr1[(sizeof(typeof(Foo)) == sizeof(typeof(long))) - 1];
int arr2[(sizeof(typeof(IntegerEnum)) == sizeof(typeof(long))) - 1];
