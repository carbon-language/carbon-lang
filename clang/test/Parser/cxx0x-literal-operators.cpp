// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

void operator "" (const char *); // expected-error {{expected identifier}}
void operator "k" foo(const char *); // expected-error {{string literal after 'operator' must be '""'}} \
// expected-warning{{user-defined literal with suffix 'foo' is preempted by C99 hexfloat extension}}
void operator "" tester (const char *); // expected-warning{{user-defined literal with suffix 'tester' is preempted by C99 hexfloat extension}}
