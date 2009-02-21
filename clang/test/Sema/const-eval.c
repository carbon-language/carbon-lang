// RUN: clang -fsyntax-only -verify %s

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
