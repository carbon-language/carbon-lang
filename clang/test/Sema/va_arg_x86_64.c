// RUN: clang -fsyntax-only -verify -triple=x86_64-unknown-freebsd7.0 %s

char* foo(char *fmt, __builtin_va_list ap)
{
  return __builtin_va_arg((ap), char *);
}
