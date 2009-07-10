// RUN: clang-cc -fsyntax-only -verify -Wformat=2 %s

#include <stdio.h>
#include <stdarg.h>

const char *foo(const char *format) __attribute__((format_arg(1)));

void __attribute__((format(printf, 1, 0)))
foo2(const char *fmt, va_list va)
{
        vprintf(foo(fmt), va);
}
