// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A
{
    ~A();
};
int foo(A);

void bar()
{
    A a;
    asm("" : : "r"(foo(a)) ); // rdar://8540491
}
