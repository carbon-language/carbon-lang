// RUN: %clang_cc1 -isystem %S -Wdouble-promotion -fsyntax-only %s  2>&1 | FileCheck -allow-empty %s
// CHECK-NOT: warning:

#include <no-warn-in-system-macro.c.inc>

#define MACRO(x) x

int main(void)
{
	double foo = 1.0;

	if (isnan(foo))
		return 1;

        MACRO(isnan(foo));

        return 0;
}
