// RUN: clang -fsyntax-only -verify %s

struct foo { int a, b; };

extern void fooFunc(struct foo *pfoo);

int main(int argc, char **argv) {
 fooFunc(&(struct foo){ 1, 2 });
}

