// RUN: clang -fsyntax-only -verify %s

extern void foo();
#include <objc/objc.h>

@protocol MyProtocol
- (void) method;
@end

@interface MyClass
@end

@interface MyClass (Addition) <MyProtocol>
- (void) method;
@end

@interface MyOtherClass : MyClass
@end

int main()
{
  id <MyProtocol> obj_id_p = nil;
  MyClass *obj_c_cat_p = nil;
  MyOtherClass *obj_c_super_p = nil;

  obj_c_cat_p = obj_id_p;   // expected-error {{incompatible types assigning 'id<MyProtocol>' to 'MyClass *'}}
  obj_c_super_p = obj_id_p;  // expected-error {{incompatible types assigning 'id<MyProtocol>' to 'MyOtherClass *'}}
  obj_id_p = obj_c_cat_p;  /* Ok */
  obj_id_p = obj_c_super_p; /* Ok */

  if (obj_c_cat_p == obj_id_p) foo(); /* Ok */
  if (obj_c_super_p == obj_id_p) foo() ; /* Ok */
  if (obj_id_p == obj_c_cat_p)  foo(); /* Ok */
  if (obj_id_p == obj_c_super_p)  foo(); /* Ok */

  return 0;
}
