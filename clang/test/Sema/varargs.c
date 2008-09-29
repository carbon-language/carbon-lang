// RUN: clang -fsyntax-only -verify %s &&
// RUN: clang -fsyntax-only -verify %s -triple x86_64-apple-darwin9

void f1(int a)
{
    __builtin_va_list ap;
    
    __builtin_va_start(ap, a, a); // expected-error {{too many arguments to function}}
    __builtin_va_start(ap, a); // expected-error {{'va_start' used in function with fixed args}}
}

void f2(int a, int b, ...)
{
    __builtin_va_list ap;
    
    __builtin_va_start(ap, 10); // expected-warning {{second parameter of 'va_start' not last named argument}}
    __builtin_va_start(ap, a); // expected-warning {{second parameter of 'va_start' not last named argument}}
    __builtin_va_start(ap, b);
}

void f3(float a, ...)
{
    __builtin_va_list ap;
    
    __builtin_va_start(ap, a);
    __builtin_va_start(ap, (a));
}
