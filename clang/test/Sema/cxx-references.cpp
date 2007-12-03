// RUN: clang -fsyntax-only %s
int g(int);

void f() {
  int i;
  int &r = i;
  r = 1;
  int *p = &r;
  int &rr = r;
  int (&rg)(int) = g;
  rg(i);
  int a[3];
  int (&ra)[3] = a;
  ra[1] = i;
  int *Q;
  int *& P = Q;
  P[1] = 1;
}

typedef int t[1];
void test2() {
    t a;
    t& b = a;


    int c[3];
    int (&rc)[3] = c;
}
