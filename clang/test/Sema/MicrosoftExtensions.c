// RUN: %clang_cc1 %s -fsyntax-only -Wno-unused-value -Wmicrosoft -verify -fms-extensions


struct A
{
   int a[];  /* expected-warning {{flexible array member 'a' in otherwise empty struct is a Microsoft extension}} */
};

struct C {
   int l;
   union {
       int c1[];   /* expected-warning {{flexible array member 'c1' in a union is a Microsoft extension}}  */
       char c2[];  /* expected-warning {{flexible array member 'c2' in a union is a Microsoft extension}} */
   };
};


struct D {
   int l;
   int D[];
};


enum ENUM1; // expected-warning {{forward references to 'enum' types are a Microsoft extension}}    
enum ENUM1 var1 = 3;
enum ENUM1* var2 = 0;


enum ENUM2 {
  ENUM2_a = (enum ENUM2) 4,
  ENUM2_b = 0x9FFFFFFF, // expected-warning {{enumerator value is not representable in the underlying type 'int'}}
  ENUM2_c = 0x100000000 // expected-warning {{enumerator value is not representable in the underlying type 'int'}}
};




typedef struct notnested {
  long bad1;
  long bad2;
} NOTNESTED;


typedef struct nested1 {
  long a;
  struct notnested var1;
  NOTNESTED var2;
} NESTED1;

struct nested2 {
  long b;
  NESTED1;  // expected-warning {{anonymous structs are a Microsoft extension}}
};

struct test {
  int c;
  struct nested2;   // expected-warning {{anonymous structs are a Microsoft extension}}
};

void foo()
{
  struct test var;
  var.a;
  var.b;
  var.c;
  var.bad1;   // expected-error {{no member named 'bad1' in 'struct test'}}
  var.bad2;   // expected-error {{no member named 'bad2' in 'struct test'}}
}

// Enumeration types with a fixed underlying type.
const int seventeen = 17;
typedef int Int;

struct X0 {
  enum E1 : Int { SomeOtherValue } field;  // expected-warning{{enumeration types with a fixed underlying type are a Microsoft extension}}
  enum E1 : seventeen;
};

enum : long long {  // expected-warning{{enumeration types with a fixed underlying type are a Microsoft extension}}
  SomeValue = 0x100000000
};


void pointer_to_integral_type_conv(char* ptr) {
   char ch = (char)ptr;
   short sh = (short)ptr;
   ch = (char)ptr;
   sh = (short)ptr;
} 
