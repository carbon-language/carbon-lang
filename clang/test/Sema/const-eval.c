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
