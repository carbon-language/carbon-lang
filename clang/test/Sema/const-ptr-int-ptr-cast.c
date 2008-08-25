// RUN: clang -fsyntax-only -verify %s

char *a = (void*)(unsigned long long)(void*)&a;
