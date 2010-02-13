// RUN: %clang_cc1 -fsyntax-only -verify %s

#define EVAL_EXPR(testno, expr) int test##testno = sizeof(struct{char qq[expr];});
int x;
EVAL_EXPR(1, (_Bool)&x)
EVAL_EXPR(2, (int)(1.0+(double)4))
EVAL_EXPR(3, (int)(1.0+(float)4.0))
EVAL_EXPR(4, (_Bool)(1 ? (void*)&x : 0))
EVAL_EXPR(5, (_Bool)(int[]){0})
struct y {int x,y;};
EVAL_EXPR(6, (int)(1+(struct y*)0))
EVAL_EXPR(7, (int)&((struct y*)0)->y)
EVAL_EXPR(8, (_Bool)"asdf")
EVAL_EXPR(9, !!&x)
EVAL_EXPR(10, ((void)1, 12))
void g0(void);
EVAL_EXPR(11, (g0(), 12)) // FIXME: This should give an error
EVAL_EXPR(12, 1.0&&2.0)
EVAL_EXPR(13, x || 3.0)

unsigned int l_19 = 1;
EVAL_EXPR(14, (1 ^ l_19) && 1); // expected-error {{fields must have a constant size}}

void f()
{
  int a;
  EVAL_EXPR(15, (_Bool)&a); // expected-error {{fields must have a constant size}}
}

// FIXME: Turn into EVAL_EXPR test once we have more folding.
_Complex float g16 = (1.0f + 1.0fi);

// ?: in constant expressions.
int g17[(3?:1) - 2]; 

EVAL_EXPR(18, ((int)((void*)10 + 10)) == 20 ? 1 : -1);

struct s {
  int a[(int)-1.0f]; // expected-error {{array size is negative}}
};

EVAL_EXPR(19, ((int)&*(char*)10 == 10 ? 1 : -1));

EVAL_EXPR(20, __builtin_constant_p(*((int*) 10), -1, 1));

EVAL_EXPR(21, (__imag__ 2i) == 2 ? 1 : -1);

EVAL_EXPR(22, (__real__ (2i+3)) == 3 ? 1 : -1);

int g23[(int)(1.0 / 1.0)] = { 1 };
int g24[(int)(1.0 / 1.0)] = { 1 , 2 }; // expected-warning {{excess elements in array initializer}}
int g25[(int)(1.0 + 1.0)], g26 = sizeof(g25);

EVAL_EXPR(26, (_Complex double)0 ? -1 : 1)
EVAL_EXPR(27, (_Complex int)0 ? -1 : 1)
EVAL_EXPR(28, (_Complex double)1 ? 1 : -1)
EVAL_EXPR(29, (_Complex int)1 ? 1 : -1)


// PR4027 + rdar://6808859
struct a { int x, y };
static struct a V2 = (struct a)(struct a){ 1, 2};
static const struct a V1 = (struct a){ 1, 2};

EVAL_EXPR(30, (int)(_Complex float)((1<<30)-1) == (1<<30) ? 1 : -1)
EVAL_EXPR(31, (int*)0 == (int*)0 ? 1 : -1)
EVAL_EXPR(32, (int*)0 != (int*)0 ? -1 : 1)
EVAL_EXPR(33, (void*)0 - (void*)0 == 0 ? 1 : -1)
void foo(void) {}
EVAL_EXPR(34, (foo == (void *)0) ? -1 : 1)

// No PR. Mismatched bitwidths lead to a crash on second evaluation.
const _Bool constbool = 0;
EVAL_EXPR(35, constbool)
EVAL_EXPR(36, constbool)

EVAL_EXPR(37, (1,2.0) == 2.0)
EVAL_EXPR(38, __builtin_expect(1,1) == 1)
