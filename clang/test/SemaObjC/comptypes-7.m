// RUN: clang-cc -fsyntax-only -verify -pedantic %s

#define nil (void *)0;
#define Nil (void *)0;

extern void foo();

@protocol MyProtocol
- (void) method;
@end

@interface MyClass
@end

int main()
{
  id obj = nil;
  id <MyProtocol> obj_p = nil;
  MyClass *obj_c = nil;
  Class obj_C = Nil;
  
  int i = 0;
  int *j = nil;

  /* These should all generate warnings.  */
  
  obj = i; // expected-warning {{incompatible integer to pointer conversion assigning 'int', expected 'id'}}
  obj = j; // expected-warning {{incompatible pointer types assigning 'int *', expected 'id'}}

  obj_p = i; // expected-warning {{incompatible integer to pointer conversion assigning 'int', expected 'id<MyProtocol>'}}
  obj_p = j; // expected-warning {{incompatible pointer types assigning 'int *', expected 'id<MyProtocol>'}}
  
  obj_c = i; // expected-warning {{incompatible integer to pointer conversion assigning 'int', expected 'MyClass *'}}
  obj_c = j; // expected-warning {{incompatible pointer types assigning 'int *', expected 'MyClass *'}}

  obj_C = i; // expected-warning {{incompatible integer to pointer conversion assigning 'int', expected 'Class'}}
  obj_C = j; // expected-warning {{incompatible pointer types assigning 'int *', expected 'Class'}}
  
  i = obj;   // expected-warning {{incompatible pointer to integer conversion assigning 'id', expected 'int'}}
  i = obj_p; // expected-warning {{incompatible pointer to integer conversion assigning 'id<MyProtocol>', expected 'int'}}
  i = obj_c; // expected-warning {{incompatible pointer to integer conversion assigning 'MyClass *', expected 'int'}}
  i = obj_C; // expected-warning {{incompatible pointer to integer conversion assigning 'Class', expected 'int'}}
  
  j = obj;   // expected-warning {{incompatible pointer types assigning 'id', expected 'int *'}}
  j = obj_p; // expected-warning {{incompatible pointer types assigning 'id<MyProtocol>', expected 'int *'}}
  j = obj_c; // expected-warning {{incompatible pointer types assigning 'MyClass *', expected 'int *'}}
  j = obj_C; // expected-warning {{incompatible pointer types assigning 'Class', expected 'int *'}}
  
  if (obj == i) foo() ; // expected-warning {{comparison between pointer and integer ('id' and 'int')}}
  if (i == obj) foo() ; // expected-warning {{comparison between pointer and integer ('int' and 'id')}}
  if (obj == j) foo() ; // expected-warning {{comparison of distinct pointer types ('id' and 'int *')}}
  if (j == obj) foo() ; // expected-warning {{comparison of distinct pointer types ('int *' and 'id')}}

  if (obj_c == i) foo() ; // expected-warning {{comparison between pointer and integer ('MyClass *' and 'int')}}
  if (i == obj_c) foo() ; // expected-warning {{comparison between pointer and integer ('int' and 'MyClass *')}}
  if (obj_c == j) foo() ; // expected-warning {{comparison of distinct pointer types ('MyClass *' and 'int *')}}
  if (j == obj_c) foo() ; // expected-warning {{comparison of distinct pointer types ('int *' and 'MyClass *')}}

  if (obj_p == i) foo() ; // expected-warning {{comparison between pointer and integer ('id<MyProtocol>' and 'int')}}
  if (i == obj_p) foo() ; // expected-warning {{comparison between pointer and integer ('int' and 'id<MyProtocol>')}}
  if (obj_p == j) foo() ; // expected-warning {{comparison of distinct pointer types ('id<MyProtocol>' and 'int *')}}
  if (j == obj_p) foo() ; // expected-warning {{comparison of distinct pointer types ('int *' and 'id<MyProtocol>')}}

  if (obj_C == i) foo() ; // expected-warning {{comparison between pointer and integer ('Class' and 'int')}}
  if (i == obj_C) foo() ; // expected-warning {{comparison between pointer and integer ('int' and 'Class')}}
  if (obj_C == j) foo() ; // expected-warning {{comparison of distinct pointer types ('Class' and 'int *')}}
  if (j == obj_C) foo() ; // expected-warning {{comparison of distinct pointer types ('int *' and 'Class')}}

  return 0;
}
