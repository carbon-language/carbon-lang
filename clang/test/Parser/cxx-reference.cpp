// RUN: clang -parse-ast-check %s

extern char *bork;
char *& bar = bork;

void foo(int &a) {
}

typedef int & A;

void g(const A aref) {
}

int & const X; // expected-error {{'const' qualifier may not be applied to a reference}}
int & volatile Y; // expected-error {{'volatile' qualifier may not be applied to a reference}}
int & const volatile Z; /* expected-error {{'const' qualifier may not be applied}} \
                           expected-error {{'volatile' qualifier may not be applied}} */
