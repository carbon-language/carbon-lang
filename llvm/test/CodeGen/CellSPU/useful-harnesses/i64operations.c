#include <stdio.h>

#define TRUE_VAL (!0)
#define FALSE_VAL 0
#define ARR_SIZE(arr) (sizeof(arr)/sizeof(arr[0]))

typedef unsigned long long int uint64_t;
typedef long long int int64_t;

/* ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- */

int64_t         a = 1234567890003LL;
int64_t         b = 2345678901235LL;
int64_t         c = 1234567890001LL;
int64_t         d = 10001LL;
int64_t         e = 10000LL;
int64_t         f = -1068103409991LL;

/* ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- */

int 
i64_eq(int64_t a, int64_t b)
{
  return (a == b);
}

int 
i64_neq(int64_t a, int64_t b)
{
  return (a != b);
}

int 
i64_ugt(uint64_t a, uint64_t b)
{
  return (a > b);
}

int 
i64_ule(uint64_t a, uint64_t b)
{
  return (a <= b);
}

int64_t 
i64_eq_select(int64_t a, int64_t b, int64_t c, int64_t d)
{
  return ((a == b) ? c : d);
}

int64_t 
i64_neq_select(int64_t a, int64_t b, int64_t c, int64_t d)
{
  return ((a != b) ? c : d);
}

uint64_t 
i64_ugt_select(uint64_t a, uint64_t b, uint64_t c, uint64_t d)
{
  return ((a > b) ? c : d);
}

uint64_t 
i64_ule_select(uint64_t a, uint64_t b, uint64_t c, uint64_t d)
{
  return ((a <= b) ? c : d);
}

/* ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- */

struct harness_int64_pred {
  const char     *fmt_string;
  int64_t        *lhs;
  int64_t        *rhs;
  int64_t        *select_a;
  int64_t        *select_b;
  int             expected;
  int64_t        *select_expected;
};

struct harness_uint64_pred {
  const char     *fmt_string;
  uint64_t       *lhs;
  uint64_t       *rhs;
  uint64_t       *select_a;
  uint64_t       *select_b;
  int             expected;
  uint64_t       *select_expected;
};

struct int64_pred_s {
  const char     *name;
  int             (*predfunc) (int64_t, int64_t);
  int64_t         (*selfunc) (int64_t, int64_t, int64_t, int64_t);
  struct harness_int64_pred *tests;
  int             n_tests;
};

struct uint64_pred_s {
  const char     *name;
  int             (*predfunc) (uint64_t, uint64_t);
  uint64_t        (*selfunc) (uint64_t, uint64_t, uint64_t, uint64_t);
  struct harness_uint64_pred *tests;
  int             n_tests;
};

/* ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- */

struct harness_int64_pred int64_tests_eq[] = {
  {"a %s a", &a, &a, &c, &d, TRUE_VAL, &c},
  {"a %s b", &a, &b, &c, &d, FALSE_VAL, &d},
  {"a %s c", &a, &c, &c, &d, FALSE_VAL, &d},
  {"d %s e", &d, &e, &c, &d, FALSE_VAL, &d},
  {"e %s e", &e, &e, &c, &d, TRUE_VAL, &c}
};

struct harness_int64_pred int64_tests_neq[] = {
  {"a %s a", &a, &a, &c, &d, FALSE_VAL, &d},
  {"a %s b", &a, &b, &c, &d, TRUE_VAL, &c},
  {"a %s c", &a, &c, &c, &d, TRUE_VAL, &c},
  {"d %s e", &d, &e, &c, &d, TRUE_VAL, &c},
  {"e %s e", &e, &e, &c, &d, FALSE_VAL, &d}
};

struct int64_pred_s int64_preds[] = {
  {"eq", i64_eq, i64_eq_select,
     int64_tests_eq, ARR_SIZE(int64_tests_eq)},
  {"neq", i64_neq, i64_neq_select,
     int64_tests_neq, ARR_SIZE(int64_tests_neq)}
};

/* ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- */

struct harness_uint64_pred uint64_tests_ugt[] = {
  {"a %s a", (uint64_t *) &a, (uint64_t *) &a, (uint64_t *) &c,
     (uint64_t *) &d, FALSE_VAL, (uint64_t *) &d},
  {"a %s b", (uint64_t *) &a, (uint64_t *) &b, (uint64_t *) &c,
     (uint64_t *) &d, FALSE_VAL, (uint64_t *) &d },
  {"a %s c", (uint64_t *) &a, (uint64_t *) &c, (uint64_t *) &c,
     (uint64_t *) &d, TRUE_VAL, (uint64_t *) &c },
  {"d %s e", (uint64_t *) &d, (uint64_t *) &e, (uint64_t *) &c,
     (uint64_t *) &d, TRUE_VAL, (uint64_t *) &c },
  {"e %s e", (uint64_t *) &e, (uint64_t *) &e, (uint64_t *) &c,
     (uint64_t *) &d, FALSE_VAL, (uint64_t *) &d }
};

struct harness_uint64_pred uint64_tests_ule[] = {
  {"a %s a", (uint64_t *) &a, (uint64_t *) &a, (uint64_t *) &c,
     (uint64_t *) &d, TRUE_VAL, (uint64_t *) &c},
  {"a %s b", (uint64_t *) &a, (uint64_t *) &b, (uint64_t *) &c,
     (uint64_t *) &d, TRUE_VAL, (uint64_t *) &c},
  {"a %s c", (uint64_t *) &a, (uint64_t *) &c, (uint64_t *) &c,
     (uint64_t *) &d, FALSE_VAL, (uint64_t *) &d},
  {"d %s e", (uint64_t *) &d, (uint64_t *) &e, (uint64_t *) &c,
     (uint64_t *) &d, FALSE_VAL, (uint64_t *) &d},
  {"e %s e", (uint64_t *) &e, (uint64_t *) &e, (uint64_t *) &c,
     (uint64_t *) &d, TRUE_VAL, (uint64_t *) &c}
};

struct uint64_pred_s uint64_preds[] = {
  {"ugt", i64_ugt, i64_ugt_select,
     uint64_tests_ugt, ARR_SIZE(uint64_tests_ugt)},
  {"ule", i64_ule, i64_ule_select,
     uint64_tests_ule, ARR_SIZE(uint64_tests_ule)}
};

int 
compare_expect_int64(const struct int64_pred_s * pred)
{
  int             j, failed = 0;

  for (j = 0; j < pred->n_tests; ++j) {
    int             pred_result =
    (*pred->predfunc) (*pred->tests[j].lhs, *pred->tests[j].rhs);

    if (pred_result != pred->tests[j].expected) {
      char            str[64];

      sprintf(str, pred->tests[j].fmt_string, pred->name);
      printf("%s: returned value is %d, expecting %d\n", str,
	     pred_result, pred->tests[j].expected);
      printf("  lhs = %19lld (0x%016llx)\n", *pred->tests[j].lhs, *pred->tests[j].lhs);
      printf("  rhs = %19lld (0x%016llx)\n", *pred->tests[j].rhs, *pred->tests[j].rhs);
      ++failed;
    } else {
      int64_t         selresult = (pred->selfunc) (*pred->tests[j].lhs, *pred->tests[j].rhs,
			  *pred->tests[j].select_a, *pred->tests[j].select_b);
      if (selresult != *pred->tests[j].select_expected) {
	char            str[64];

	sprintf(str, pred->tests[j].fmt_string, pred->name);
	printf("%s select: returned value is %d, expecting %d\n", str,
	       pred_result, pred->tests[j].expected);
	printf("  lhs   = %19lld (0x%016llx)\n", *pred->tests[j].lhs, *pred->tests[j].lhs);
	printf("  rhs   = %19lld (0x%016llx)\n", *pred->tests[j].rhs, *pred->tests[j].rhs);
	printf("  true  = %19lld (0x%016llx)\n", *pred->tests[j].select_a, *pred->tests[j].select_a);
	printf("  false = %19lld (0x%016llx)\n", *pred->tests[j].select_b, *pred->tests[j].select_b);
	++failed;
      }
    }
  }

  return failed;
}

int 
compare_expect_uint64(const struct uint64_pred_s * pred)
{
  int             j, failed = 0;

  for (j = 0; j < pred->n_tests; ++j) {
    int             pred_result = (*pred->predfunc) (*pred->tests[j].lhs, *pred->tests[j].rhs);

    if (pred_result != pred->tests[j].expected) {
      char            str[64];

      sprintf(str, pred->tests[j].fmt_string, pred->name);
      printf("%s: returned value is %d, expecting %d\n", str,
	     pred_result, pred->tests[j].expected);
      printf("  lhs = %19llu (0x%016llx)\n", *pred->tests[j].lhs, *pred->tests[j].lhs);
      printf("  rhs = %19llu (0x%016llx)\n", *pred->tests[j].rhs, *pred->tests[j].rhs);
      ++failed;
    } else {
      uint64_t        selresult = (pred->selfunc) (*pred->tests[j].lhs, *pred->tests[j].rhs,
			  *pred->tests[j].select_a, *pred->tests[j].select_b);
      if (selresult != *pred->tests[j].select_expected) {
	char            str[64];

	sprintf(str, pred->tests[j].fmt_string, pred->name);
	printf("%s select: returned value is %d, expecting %d\n", str,
	       pred_result, pred->tests[j].expected);
	printf("  lhs   = %19llu (0x%016llx)\n", *pred->tests[j].lhs, *pred->tests[j].lhs);
	printf("  rhs   = %19llu (0x%016llx)\n", *pred->tests[j].rhs, *pred->tests[j].rhs);
	printf("  true  = %19llu (0x%016llx)\n", *pred->tests[j].select_a, *pred->tests[j].select_a);
	printf("  false = %19llu (0x%016llx)\n", *pred->tests[j].select_b, *pred->tests[j].select_b);
	++failed;
      }
    }

  }

  return failed;
}

/* ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- */

uint64_t 
i64_shl_const(uint64_t a)
{
  return a << 10;
}

uint64_t 
i64_shl(uint64_t a, int amt)
{
  return a << amt;
}

uint64_t 
i64_srl_const(uint64_t a)
{
  return a >> 10;
}

uint64_t 
i64_srl(uint64_t a, int amt)
{
  return a >> amt;
}

int64_t 
i64_sra_const(int64_t a)
{
  return a >> 10;
}

int64_t 
i64_sra(int64_t a, int amt)
{
  return a >> amt;
}

/* ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- */

int 
main(void)
{
  int             i, j, failed = 0;
  const char     *something_failed = "  %d tests failed.\n";
  const char     *all_tests_passed = "  All tests passed.\n";

  printf("a = %16lld (0x%016llx)\n", a, a);
  printf("b = %16lld (0x%016llx)\n", b, b);
  printf("c = %16lld (0x%016llx)\n", c, c);
  printf("d = %16lld (0x%016llx)\n", d, d);
  printf("e = %16lld (0x%016llx)\n", e, e);
  printf("f = %16lld (0x%016llx)\n", f, f);
  printf("----------------------------------------\n");

  for (i = 0; i < ARR_SIZE(int64_preds); ++i) {
    printf("%s series:\n", int64_preds[i].name);
    if ((failed = compare_expect_int64(int64_preds + i)) > 0) {
      printf(something_failed, failed);
    } else {
      printf(all_tests_passed);
    }

    printf("----------------------------------------\n");
  }

  for (i = 0; i < ARR_SIZE(uint64_preds); ++i) {
    printf("%s series:\n", uint64_preds[i].name);
    if ((failed = compare_expect_uint64(uint64_preds + i)) > 0) {
      printf(something_failed, failed);
    } else {
      printf(all_tests_passed);
    }

    printf("----------------------------------------\n");
  }

  printf("a                = 0x%016llx\n", a);
  printf("i64_shl_const(a) = 0x%016llx\n", i64_shl_const(a));
  printf("i64_shl(a)       = 0x%016llx\n", i64_shl(a, 10));
  printf("i64_srl_const(a) = 0x%016llx\n", i64_srl_const(a));
  printf("i64_srl(a)       = 0x%016llx\n", i64_srl(a, 10));
  printf("i64_sra_const(a) = 0x%016llx\n", i64_sra_const(a));
  printf("i64_sra(a)       = 0x%016llx\n", i64_sra(a, 10));
  printf("----------------------------------------\n");

  printf("f                = 0x%016llx\n", f);
  printf("i64_shl_const(f) = 0x%016llx\n", i64_shl_const(f));
  printf("i64_shl(f)       = 0x%016llx\n", i64_shl(f, 10));
  printf("i64_srl_const(f) = 0x%016llx\n", i64_srl_const(f));
  printf("i64_srl(f)       = 0x%016llx\n", i64_srl(f, 10));
  printf("i64_sra_const(f) = 0x%016llx\n", i64_sra_const(f));
  printf("i64_sra(f)       = 0x%016llx\n", i64_sra(f, 10));
  printf("----------------------------------------\n");

  return 0;
}
