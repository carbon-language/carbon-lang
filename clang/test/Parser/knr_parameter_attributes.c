// RUN: %clang_cc1 -fsyntax-only -W -Wall -Werror -verify %s

int f(int i __attribute__((__unused__)))
{
    return 0;
}
int g(i)
    int i __attribute__((__unused__));
{
    return 0;
}
