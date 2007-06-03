// RUN: clang -fsyntax-only %s &&
// RUN: clang -fsyntax-only %s 2>&1 | grep "error: 'const' qualifier may not be applied to a reference" &&
// RUN: clang -fsyntax-only %s 2>&1 | grep "error: 'volatile' qualifier may not be applied to a reference"

extern char *bork;
char *& bar = bork;

void foo(int &a) {
}

typedef int & A;

void g(const A aref) {
}

int & const X;
int & volatile Y;
int & const volatile Z;

