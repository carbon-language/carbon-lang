// RUN: %clang_cc1 -fsyntax-only -verify %s

#if !__has_feature(objc_fixed_enum)
#  error Enumerations with a fixed underlying type are not supported
#endif

typedef long Integer;

typedef enum : Integer { Enumerator1, Enumerator2 } Enumeration;

int array[sizeof(Enumeration) == sizeof(long)? 1 : -1];


enum Color { Red, Green, Blue };

struct X { 
  enum Color : 4;
  enum Color field1: 4;
  enum Other : Integer field2;
  enum Other : Integer field3 : 4;
  enum  : Integer { Blah, Blarg } field4 : 4;
};
