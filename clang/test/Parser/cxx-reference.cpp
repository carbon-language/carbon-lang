// RUN: clang -fsyntax-only %s
extern char *bork;
char *& bar = bork;

void foo(int &a) {
}
