// RUN: clang -fsyntax-only -verify -std=c90 %s
void
foo (void)
{
 struct b;
 struct b* x = 0;
 struct b* y = &*x;
}

void foo2 (void)
{
 typedef int (*arrayptr)[];
 arrayptr x = 0;
 arrayptr y = &*x;
}

void foo3 (void)
{
 void* x = 0;
 void* y = &*x; // expected-error {{address expression must be an lvalue or a function designator}}
}

