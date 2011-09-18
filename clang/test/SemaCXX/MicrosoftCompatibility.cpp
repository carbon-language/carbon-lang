// RUN: %clang_cc1 %s -triple i686-pc-win32 -fsyntax-only -Wmicrosoft -verify -fms-compatibility



namespace ms_conversion_rules {

void f(float a);
void f(int a);

void test()
{
    long a = 0;
    f((long)0);
	f(a);
}

}

