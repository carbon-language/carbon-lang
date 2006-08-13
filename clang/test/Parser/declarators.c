// RUN: clang %s -fsyntax-only

extern void a1[];

void f0();
void f1(int [*]);
void f2(int [const *]);
void f3(int [volatile const*]);
int f4(*XX)(void);

char ((((*X))));

void (*signal(int, void (*)(int)))(int);

int a, ***C, * const D, b(int);

int *A;

struct str;

int test2(int *P, int A) {
  struct str;

  // Hard case for array decl, not Array[*].
  int Array[*(int*)P+A];
}


