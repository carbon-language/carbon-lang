// RUN: clang -fsyntax-only -verify %s

#include <objc/Object.h>

@interface Derived: Object
@end

extern Object* foo(void);

static Derived *test(void)
{
   Derived *m = foo();   // expected-warning {{incompatible pointer types assigning 'Object *' to 'Derived *'}}

   return m;
}

