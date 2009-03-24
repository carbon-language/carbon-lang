// RUN: clang-cc -ast-print %s

#include <stddef.h>

char    test1(void) { return '\\'; }
wchar_t test2(void) { return L'\\'; }
char    test3(void) { return '\''; }
wchar_t test4(void) { return L'\''; }
char    test5(void) { return '\a'; }
wchar_t test6(void) { return L'\a'; }
char    test7(void) { return '\b'; }
wchar_t test8(void) { return L'\b'; }
char    test9(void) { return '\e'; }
wchar_t test10(void) { return L'\e'; }
char    test11(void) { return '\f'; }
wchar_t test12(void) { return L'\f'; }
char    test13(void) { return '\n'; }
wchar_t test14(void) { return L'\n'; }
char    test15(void) { return '\r'; }
wchar_t test16(void) { return L'\r'; }
char    test17(void) { return '\t'; }
wchar_t test18(void) { return L'\t'; }
char    test19(void) { return '\v'; }
wchar_t test20(void) { return L'\v'; }

char    test21(void) { return 'c'; }
wchar_t test22(void) { return L'c'; }
char    test23(void) { return '\x3'; }
wchar_t test24(void) { return L'\x3'; }

wchar_t test25(void) { return L'\x333'; }
