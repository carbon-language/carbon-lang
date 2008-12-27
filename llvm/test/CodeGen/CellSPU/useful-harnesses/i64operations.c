#include <stdio.h>

typedef unsigned long long int	uint64_t;
typedef long long int   	int64_t;

const char *boolstring(int val) {
  return val ? "true" : "false";
}

int i64_eq(int64_t a, int64_t b) {
  return (a == b);
}

int i64_neq(int64_t a, int64_t b) {
  return (a != b);
}

int64_t i64_eq_select(int64_t a, int64_t b, int64_t c, int64_t d) {
  return ((a == b) ? c : d);
}

int64_t i64_neq_select(int64_t a, int64_t b, int64_t c, int64_t d) {
  return ((a != b) ? c : d);
}

struct pred_s {
  const char   *name;
  int 		(*predfunc)(int64_t, int64_t);
  int64_t       (*selfunc)(int64_t, int64_t, int64_t, int64_t);
};

struct pred_s preds[] = {
  { "eq",  i64_eq,  i64_eq_select },
  { "neq", i64_neq, i64_neq_select }
};

int main(void) {
  int i;
  int64_t a = 1234567890000LL;
  int64_t b = 2345678901234LL;
  int64_t c = 1234567890001LL;
  int64_t d =         10001LL;
  int64_t e =         10000LL;

  printf("a = %16lld (0x%016llx)\n", a, a);
  printf("b = %16lld (0x%016llx)\n", b, b);
  printf("c = %16lld (0x%016llx)\n", c, c);
  printf("d = %16lld (0x%016llx)\n", d, d);
  printf("e = %16lld (0x%016llx)\n", e, e);
  printf("----------------------------------------\n");

  for (i = 0; i < sizeof(preds)/sizeof(preds[0]); ++i) {
    printf("a %s a = %s\n", preds[i].name, boolstring((*preds[i].predfunc)(a, a)));
    printf("a %s b = %s\n", preds[i].name, boolstring((*preds[i].predfunc)(a, b)));
    printf("a %s c = %s\n", preds[i].name, boolstring((*preds[i].predfunc)(a, c)));
    printf("d %s e = %s\n", preds[i].name, boolstring((*preds[i].predfunc)(d, e)));
    printf("e %s e = %s\n", preds[i].name, boolstring((*preds[i].predfunc)(e, e)));

    printf("a %s a ? c : d = %lld\n", preds[i].name, (*preds[i].selfunc)(a, a, c, d));
    printf("a %s a ? c : d == c (%s)\n", preds[i].name, boolstring((*preds[i].selfunc)(a, a, c, d) == c));
    printf("a %s b ? c : d = %lld\n", preds[i].name, (*preds[i].selfunc)(a, b, c, d));
    printf("a %s b ? c : d == d (%s)\n", preds[i].name, boolstring((*preds[i].selfunc)(a, b, c, d) == d));

    printf("----------------------------------------\n");
  }

  return 0;
}
