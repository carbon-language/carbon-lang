// RUN: clang-cc -fsyntax-only -verify -pedantic %s
// PR4287

#include <stdarg.h>
char *foo = "test";
int test(char*,...);

int test(fmt)
        char*fmt;
{
        va_list ap;
        char*a;
        int x;

        va_start(ap,fmt);
        a=va_arg(ap,char*);
        x=(a!=foo);
        va_end(ap);
        return x;
}

void exit();

int main(argc,argv)
        int argc;char**argv;
{
        exit(test("",foo));
}

