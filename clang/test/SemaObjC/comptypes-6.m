// RUN: clang-cc -fsyntax-only -verify -pedantic %s

@interface Derived
@end

@interface Object @end

extern Object* foo(void);

static Derived *test(void)
{
   Derived *m = foo();   // expected-warning {{incompatible pointer types initializing 'Object *', expected 'Derived *'}}

   return m;
}

