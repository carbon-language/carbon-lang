// RUN: %clang_cc1 -fsyntax-only -verify -std=c99 %s
// PR16138
// expected-no-diagnostics

int alloca;
int stpcpy;
int stpncpy;
int strdup;
int strndup;
int index;
int rindex;
int bzero;
int strcasecmp;
int strncasecmp;
int _exit;
int vfork;
int _setjmp;
int __sigsetjmp;
int sigsetjmp;
int setjmp_syscall;
int savectx;
int qsetjmp;
int getcontext;
int _longjmp;
int siglongjmp;
int strlcpy;
int strlcat;
