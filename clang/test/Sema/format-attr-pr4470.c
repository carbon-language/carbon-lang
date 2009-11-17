// RUN: clang-cc -fsyntax-only -verify -Wformat=2 %s

#include <stdarg.h>
int vprintf(const char *, va_list);

const char *foo(const char *format) __attribute__((format_arg(1)));

void __attribute__((format(printf, 1, 0)))
foo2(const char *fmt, va_list va)
{
        vprintf(foo(fmt), va);
}
