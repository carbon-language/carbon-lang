void printf(char*, ...);

typedef struct params_ {
  int     i1;
  float   f1;
  double  d1;
  short   s1;
  double  d2;
  char    c1;
  unsigned short  s2;
  float   f2;
  int     i2;
} params;

void print_param(params p) {
  printf("%d,   %f,   %f,   %d,   %f,   %c,   %d,   %f,   %d\n",
        p.i1, p.f1, p.d1, p.s1, p.d2, p.c1, p.s2, p.f2, p.i2);
}

void print_param_addr(params *p) {
  printf("%d,   %f,   %f,   %d,   %f,   %c,   %d,   %f,   %d\n",
        p->i1, p->f1, p->d1, p->s1, p->d2, p->c1, p->s2, p->f2, p->i2);
}

int main() {
  params p;
  p.i1 = 1;
  p.f1 = 2.0;
  p.d1 = 3.0;
  p.s1 = 4;
  p.d2 = 5.0;
  p.c1 = '6';
  p.s2 = 7;
  p.f2 = 8.0;
  p.i2 = 9;
  print_param(p);
  print_param_addr(&p);
  return 0;
}
