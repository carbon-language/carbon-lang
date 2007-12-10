// RUN: clang %s -emit-llvm

struct  {
  int x;
  int y;
} point;

void fn1() {
  point.x = 42;
}

/* Nested member */
struct  {
  struct {
    int a;
    int b;
  } p1;
} point2;

void fn2() {
  point2.p1.a = 42;
}

/* Indirect reference */
typedef struct __sf {
 unsigned char *c;
 short flags;
} F;

typedef struct __sf2 {
  F *ff;
} F2;

int fn3(F2 *c) {
  if (c->ff->c >= 0)
    return 1;
  else
    return 0;
}

/* Nested structs */
typedef struct NA {
  int data;
  struct NA *next;
} NA;
void f1() {  NA a; }

typedef struct NB {
  int d1;
  struct _B2 {
    int d2;
    struct NB *n2;
  } B2;
} NB;

void f2() { NB b; }

extern NB *f3();
void f4() {
  f3()->d1 = 42;
}

void f5() {
  (f3())->d1 = 42;
}

/* Function calls */
typedef struct {
  int location;
  int length;
} range;

extern range f6();
void f7()
{
  range r = f6();
}

/* Member expressions */
typedef struct {
  range range1;
  range range2;
} rangepair;

void f8()
{
  rangepair p;

  range r = p.range1;
}


/* _Bool types */

struct _w
{
  short a,b;
  short c,d;
  short e,f;
  short g;

  unsigned int h,i;

  _Bool j,k;
} ws;
