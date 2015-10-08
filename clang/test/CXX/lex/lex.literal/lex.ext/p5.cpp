// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s -triple=x86_64-linux-gnu

using size_t = decltype(sizeof(int));

int &operator "" _x1 (const char *);
double &operator "" _x1 (const char *, size_t);
double &i1 = "foo"_x1;
double &i2 = u8"foo"_x1;
double &i3 = L"foo"_x1; // expected-error {{no matching literal operator for call to 'operator""_x1' with arguments of types 'const wchar_t *' and 'unsigned long'}}

char &operator "" _x1(const wchar_t *, size_t);
char &i4 = L"foo"_x1; // ok
double &i5 = R"(foo)"_x1; // ok
double &i6 = u\
8\
R\
"(foo)"\
_\
x\
1; // ok
