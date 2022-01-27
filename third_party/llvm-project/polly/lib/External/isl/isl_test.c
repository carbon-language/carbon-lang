/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010      INRIA Saclay
 * Copyright 2012-2013 Ecole Normale Superieure
 * Copyright 2014      INRIA Rocquencourt
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 * and Inria Paris - Rocquencourt, Domaine de Voluceau - Rocquencourt,
 * B.P. 105 - 78153 Le Chesnay, France
 */

#include <assert.h>
#include <stdio.h>
#include <limits.h>
#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include <isl_aff_private.h>
#include <isl_space_private.h>
#include <isl/id.h>
#include <isl/set.h>
#include <isl/flow.h>
#include <isl_constraint_private.h>
#include <isl/polynomial.h>
#include <isl/union_set.h>
#include <isl/union_map.h>
#include <isl_factorization.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl_options_private.h>
#include <isl_vertices_private.h>
#include <isl/ast_build.h>
#include <isl/val.h>
#include <isl/ilp.h>
#include <isl_ast_build_expr.h>
#include <isl/options.h>

#include "isl_srcdir.c"

#define ARRAY_SIZE(array) (sizeof(array)/sizeof(*array))

static char *get_filename(isl_ctx *ctx, const char *name, const char *suffix) {
	char *filename;
	int length;
	char *pattern = "%s/test_inputs/%s.%s";

	length = strlen(pattern) - 6 + strlen(srcdir) + strlen(name)
		+ strlen(suffix) + 1;
	filename = isl_alloc_array(ctx, char, length);

	if (!filename)
		return NULL;

	sprintf(filename, pattern, srcdir, name, suffix);

	return filename;
}

void test_parse_map(isl_ctx *ctx, const char *str)
{
	isl_map *map;

	map = isl_map_read_from_str(ctx, str);
	assert(map);
	isl_map_free(map);
}

int test_parse_map_equal(isl_ctx *ctx, const char *str, const char *str2)
{
	isl_map *map, *map2;
	int equal;

	map = isl_map_read_from_str(ctx, str);
	map2 = isl_map_read_from_str(ctx, str2);
	equal = isl_map_is_equal(map, map2);
	isl_map_free(map);
	isl_map_free(map2);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown, "maps not equal",
			return -1);

	return 0;
}

void test_parse_pwqp(isl_ctx *ctx, const char *str)
{
	isl_pw_qpolynomial *pwqp;

	pwqp = isl_pw_qpolynomial_read_from_str(ctx, str);
	assert(pwqp);
	isl_pw_qpolynomial_free(pwqp);
}

static void test_parse_pwaff(isl_ctx *ctx, const char *str)
{
	isl_pw_aff *pwaff;

	pwaff = isl_pw_aff_read_from_str(ctx, str);
	assert(pwaff);
	isl_pw_aff_free(pwaff);
}

/* Check that we can read an isl_multi_val from "str" without errors.
 */
static int test_parse_multi_val(isl_ctx *ctx, const char *str)
{
	isl_multi_val *mv;

	mv = isl_multi_val_read_from_str(ctx, str);
	isl_multi_val_free(mv);

	return mv ? 0 : -1;
}

/* String descriptions of multi piecewise affine expressions
 * that are used for testing printing and parsing.
 */
static const char *reparse_multi_pw_aff_tests[] = {
	"{ A[x, y] -> [] : x + y >= 0 }",
	"{ A[x, y] -> B[] : x + y >= 0 }",
	"{ A[x, y] -> [x] : x + y >= 0 }",
	"[N] -> { A[x, y] -> [x] : x + y <= N }",
	"{ A[x, y] -> [x, y] : x + y >= 0 }",
	"{ A[x, y] -> [(x : x >= 0), (y : y >= 0)] : x + y >= 0 }",
	"[N] -> { [] : N >= 0 }",
	"[N] -> { [] : N >= 0 }",
	"[N] -> { [N] : N >= 0 }",
	"[N] -> { [N, N + 1] : N >= 0 }",
	"[N, M] -> { [(N : N >= 0), (M : M >= 0)] : N + M >= 0 }",
	"{ [a] -> [b = a] }",
	"{ [a] -> [b = a] : a >= 0 }",
};

#undef BASE
#define BASE multi_pw_aff

#include "check_reparse_templ.c"
#include "check_reparse_test_templ.c"

/* String descriptions that cannot be parsed
 * as multi piecewise affine expressions.
 */
static const char *parse_multi_pw_aff_fail_tests[] = {
	"{ [a] -> [b] : b = a }",
	"{ [a] -> [b = a] : b >= 0 }",
};

#include "check_parse_fail_test_templ.c"

/* String descriptions of piecewise multi affine expressions
 * that are used for testing printing and parsing.
 */
static const char *reparse_pw_multi_aff_tests[] = {
	"{ [x] -> [x] }",
	"{ [x] -> [x % 4] }",
	"{ [x] -> [x % 4] : x mod 3 = 1 }",
	"{ [x, x] -> [x % 4] }",
	"{ [x, x + 1] -> [x % 4] : x mod 3 = 1 }",
	"{ [x, x mod 2] -> [x % 4] }",
};

#undef BASE
#define BASE pw_multi_aff

#include "check_reparse_templ.c"
#include "check_reparse_test_templ.c"

/* Test parsing of piecewise multi affine expressions by printing
 * the expressions and checking that parsing the output results
 * in the same expression.
 * Do this for an expression converted from a map with an output
 * dimension name that is equal to an automatically generated name, and
 * a set of expressions parsed from strings.
 */
static isl_stat test_parse_pma(isl_ctx *ctx)
{
	isl_map *map;
	isl_pw_multi_aff *pma;

	map = isl_map_read_from_str(ctx, "{ [a, a] -> [i1 = a + 1] }");
	pma = isl_pw_multi_aff_from_map(map);
	if (check_reparse_pw_multi_aff(ctx, pma) < 0)
		return isl_stat_error;

	if (check_reparse_pw_multi_aff_tests(ctx) < 0)
		return isl_stat_error;

	return isl_stat_ok;
}

/* String descriptions that cannot be parsed
 * as union piecewise multi affine expressions.
 */
static const char *parse_union_pw_multi_aff_fail_tests[] = {
	"{ [a] -> [b] : b = a }",
	"{ [a] -> [b = a] : b >= 0 }",
};

#undef BASE
#define BASE union_pw_multi_aff

#include "check_parse_fail_test_templ.c"

/* Test parsing of union piecewise multi affine expressions.
 *
 * In particular, check some cases where parsing is supposed to fail.
 */
static isl_stat test_parse_upma(isl_ctx *ctx)
{
	if (check_parse_union_pw_multi_aff_fail_tests(ctx) < 0)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Test parsing of multi piecewise affine expressions by printing
 * the expressions and checking that parsing the output results
 * in the same expression.
 * Do this for a couple of manually constructed expressions,
 * an expression converted from a map with an output dimension name
 * that is equal to an automatically generated name, and
 * a set of expressions parsed from strings.
 *
 * Additionally, check some cases where parsing is supposed to fail.
 */
static int test_parse_mpa(isl_ctx *ctx)
{
	isl_space *space;
	isl_set *dom;
	isl_map *map;
	isl_pw_multi_aff *pma;
	isl_multi_pw_aff *mpa;
	isl_stat r;

	space = isl_space_set_alloc(ctx, 0, 0);
	space = isl_space_set_tuple_name(space, isl_dim_set, "A");
	mpa = isl_multi_pw_aff_zero(space);
	r = check_reparse_multi_pw_aff(ctx, mpa);
	if (r < 0)
		return -1;

	space = isl_space_set_alloc(ctx, 1, 0);
	space = isl_space_set_dim_name(space, isl_dim_param, 0, "N");
	space = isl_space_set_tuple_name(space, isl_dim_set, "A");
	dom = isl_set_universe(isl_space_params(isl_space_copy(space)));
	dom = isl_set_lower_bound_si(dom, isl_dim_param, 0, 5);
	mpa = isl_multi_pw_aff_zero(space);
	mpa = isl_multi_pw_aff_intersect_domain(mpa, dom);
	r = check_reparse_multi_pw_aff(ctx, mpa);
	if (r < 0)
		return -1;

	map = isl_map_read_from_str(ctx, "{ [a, a] -> [i1 = a + 1] }");
	pma = isl_pw_multi_aff_from_map(map);
	mpa = isl_multi_pw_aff_from_pw_multi_aff(pma);
	if (check_reparse_multi_pw_aff(ctx, mpa) < 0)
		return -1;

	if (check_reparse_multi_pw_aff_tests(ctx) < 0)
		return -1;
	if (check_parse_multi_pw_aff_fail_tests(ctx) < 0)
		return -1;

	return 0;
}

/* String descriptions of multi union piecewise affine expressions
 * that are used for testing printing and parsing.
 */
static const char *reparse_multi_union_pw_aff_tests[] = {
	"[]",
	"A[]",
	"A[B[] -> C[]]",
	"(A[] : { S[x] : x > 0; T[y] : y >= 0 })",
	"(A[] : { })",
	"[N] -> (A[] : { })",
	"[N] -> (A[] : { : N >= 0 })",
	"[N] -> (A[] : { S[x] : x > N; T[y] : y >= 0 })",
	"(A[] : [N] -> { S[x] : x > N; T[y] : y >= 0 })",
	"A[{ S[x] -> [x + 1]; T[x] -> [x] }]",
	"(A[{ S[x] -> [x + 1]; T[x] -> [x] }] : "
		"{ S[x] : x > 0; T[y] : y >= 0 })",
};

#undef BASE
#define BASE multi_union_pw_aff

#include "check_reparse_templ.c"
#include "check_reparse_test_templ.c"

/* Test parsing of multi union piecewise affine expressions by printing
 * the expressions and checking that parsing the output results
 * in the same expression.
 * Do this for a couple of manually constructed expressions and
 * a set of expressions parsed from strings.
 */
static int test_parse_mupa(isl_ctx *ctx)
{
	isl_space *space;
	isl_multi_union_pw_aff *mupa;
	isl_set *dom;
	isl_union_set *uset;
	isl_stat r;

	space = isl_space_set_alloc(ctx, 0, 0);
	space = isl_space_set_tuple_name(space, isl_dim_set, "A");
	mupa = isl_multi_union_pw_aff_zero(space);
	r = check_reparse_multi_union_pw_aff(ctx, mupa);
	if (r < 0)
		return -1;

	space = isl_space_set_alloc(ctx, 1, 0);
	space = isl_space_set_dim_name(space, isl_dim_param, 0, "N");
	space = isl_space_set_tuple_name(space, isl_dim_set, "A");
	dom = isl_set_universe(space);
	dom = isl_set_lower_bound_si(dom, isl_dim_param, 0, 5);
	uset = isl_union_set_from_set(dom);
	space = isl_space_set_alloc(ctx, 1, 0);
	space = isl_space_set_dim_name(space, isl_dim_param, 0, "N");
	space = isl_space_set_tuple_name(space, isl_dim_set, "B");
	mupa = isl_multi_union_pw_aff_zero(space);
	mupa = isl_multi_union_pw_aff_intersect_domain(mupa, uset);
	r = check_reparse_multi_union_pw_aff(ctx, mupa);
	if (r < 0)
		return -1;

	if (check_reparse_multi_union_pw_aff_tests(ctx) < 0)
		return -1;

	return 0;
}

/* Test parsing of multi expressions.
 */
static int test_parse_multi(isl_ctx *ctx)
{
	if (test_parse_mpa(ctx) < 0)
		return -1;
	if (test_parse_mupa(ctx) < 0)
		return -1;

	return 0;
}

/* Pairs of binary relation representations that should represent
 * the same binary relations.
 */
struct {
	const char *map1;
	const char *map2;
} parse_map_equal_tests[] = {
	{ "{ [x,y]  : [([x/2]+y)/3] >= 1 }",
	  "{ [x, y] : 2y >= 6 - x }" },
	{ "{ [x,y] : x <= min(y, 2*y+3) }",
	  "{ [x,y] : x <= y, 2*y + 3 }" },
	{ "{ [x,y] : x >= min(y, 2*y+3) }",
	  "{ [x, y] : (y <= x and y >= -3) or (2y <= -3 + x and y <= -4) }" },
	{ "[n] -> { [c1] : c1>=0 and c1<=floord(n-4,3) }",
	  "[n] -> { [c1] : c1 >= 0 and 3c1 <= -4 + n }" },
	{ "{ [i,j] -> [i] : i < j; [i,j] -> [j] : j <= i }",
	  "{ [i,j] -> [min(i,j)] }" },
	{ "{ [i,j] : i != j }",
	  "{ [i,j] : i < j or i > j }" },
	{ "{ [i,j] : (i+1)*2 >= j }",
	  "{ [i, j] : j <= 2 + 2i }" },
	{ "{ [i] -> [i > 0 ? 4 : 5] }",
	  "{ [i] -> [5] : i <= 0; [i] -> [4] : i >= 1 }" },
	{ "[N=2,M] -> { [i=[(M+N)/4]] }",
	  "[N, M] -> { [i] : N = 2 and 4i <= 2 + M and 4i >= -1 + M }" },
	{ "{ [x] : x >= 0 }",
	  "{ [x] : x-0 >= 0 }" },
	{ "{ [i] : ((i > 10)) }",
	  "{ [i] : i >= 11 }" },
	{ "{ [i] -> [0] }",
	  "{ [i] -> [0 * i] }" },
	{ "{ [a] -> [b] : (not false) }",
	  "{ [a] -> [b] : true }" },
	{ "{ [i] : i/2 <= 5 }",
	  "{ [i] : i <= 10 }" },
	{ "{Sym=[n] [i] : i <= n }",
	  "[n] -> { [i] : i <= n }" },
	{ "{ [*] }",
	  "{ [a] }" },
	{ "{ [i] : 2*floor(i/2) = i }",
	  "{ [i] : exists a : i = 2 a }" },
	{ "{ [a] -> [b] : a = 5 implies b = 5 }",
	  "{ [a] -> [b] : a != 5 or b = 5 }" },
	{ "{ [a] -> [a - 1 : a > 0] }",
	  "{ [a] -> [a - 1] : a > 0 }" },
	{ "{ [a] -> [a - 1 : a > 0; a : a <= 0] }",
	  "{ [a] -> [a - 1] : a > 0; [a] -> [a] : a <= 0 }" },
	{ "{ [a] -> [(a) * 2 : a >= 0; 0 : a < 0] }",
	  "{ [a] -> [2a] : a >= 0; [a] -> [0] : a < 0 }" },
	{ "{ [a] -> [(a * 2) : a >= 0; 0 : a < 0] }",
	  "{ [a] -> [2a] : a >= 0; [a] -> [0] : a < 0 }" },
	{ "{ [a] -> [(a * 2 : a >= 0); 0 : a < 0] }",
	  "{ [a] -> [2a] : a >= 0; [a] -> [0] : a < 0 }" },
	{ "{ [a] -> [(a * 2 : a >= 0; 0 : a < 0)] }",
	  "{ [a] -> [2a] : a >= 0; [a] -> [0] : a < 0 }" },
	{ "{ [a,b] -> [i,j] : a,b << i,j }",
	  "{ [a,b] -> [i,j] : a < i or (a = i and b < j) }" },
	{ "{ [a,b] -> [i,j] : a,b <<= i,j }",
	  "{ [a,b] -> [i,j] : a < i or (a = i and b <= j) }" },
	{ "{ [a,b] -> [i,j] : a,b >> i,j }",
	  "{ [a,b] -> [i,j] : a > i or (a = i and b > j) }" },
	{ "{ [a,b] -> [i,j] : a,b >>= i,j }",
	  "{ [a,b] -> [i,j] : a > i or (a = i and b >= j) }" },
	{ "{ [n] -> [i] : exists (a, b, c: 8b <= i - 32a and "
			    "8b >= -7 + i - 32 a and b >= 0 and b <= 3 and "
			    "8c < n - 32a and i < n and c >= 0 and "
			    "c <= 3 and c >= -4a) }",
	  "{ [n] -> [i] : 0 <= i < n }" },
	{ "{ [x] -> [] : exists (a, b: 0 <= a <= 1 and 0 <= b <= 3 and "
			    "2b <= x - 8a and 2b >= -1 + x - 8a) }",
	  "{ [x] -> [] : 0 <= x <= 15 }" },
	{ "{ [x] -> [x] : }",
	  "{ [x] -> [x] }" },
	{ "{ [x=4:5] -> [x + 1] }",
	  "{ [x] -> [x + 1] : 4 <= x <= 5 }" },
	{ "{ [x=4:5] -> [x + 1 : x + 1] }",
	  "{ [x=4:5] -> [x + 1] }" },
	{ "{ [x] -> [x - 1 : x + 1] }",
	  "{ [x] -> [y] : x - 1 <= y <= x + 1 }" },
	{ "{ [x=4:] -> [x + 1] }",
	  "{ [x] -> [x + 1] : 4 <= x }" },
	{ "{ [x=:5] -> [x + 1] }",
	  "{ [x] -> [x + 1] : x <= 5 }" },
	{ "{ [x=:] -> [x + 1] }",
	  "{ [x] -> [x + 1] }" },
	{ "{ [:] -> [:] }",
	  "{ [x] -> [y] }" },
	{ "{ [x, x//4] }",
	  "{ [x, floor(x/4)] }" },
	{ "{ [10//4] }",
	  "{ [2] }" },
};

int test_parse(struct isl_ctx *ctx)
{
	int i;
	isl_map *map, *map2;
	const char *str, *str2;

	if (test_parse_multi_val(ctx, "{ A[B[2] -> C[5, 7]] }") < 0)
		return -1;
	if (test_parse_multi_val(ctx, "[n] -> { [2] }") < 0)
		return -1;
	if (test_parse_multi_val(ctx, "{ A[4, infty, NaN, -1/2, 2/3] }") < 0)
		return -1;
	if (test_parse_multi(ctx) < 0)
		return -1;
	if (test_parse_pma(ctx) < 0)
		return -1;
	if (test_parse_upma(ctx) < 0)
		return -1;

	str = "{ [i] -> [-i] }";
	map = isl_map_read_from_str(ctx, str);
	assert(map);
	isl_map_free(map);

	str = "{ A[i] -> L[([i/3])] }";
	map = isl_map_read_from_str(ctx, str);
	assert(map);
	isl_map_free(map);

	test_parse_map(ctx, "{[[s] -> A[i]] -> [[s+1] -> A[i]]}");
	test_parse_map(ctx, "{ [p1, y1, y2] -> [2, y1, y2] : "
				"p1 = 1 && (y1 <= y2 || y2 = 0) }");

	for (i = 0; i < ARRAY_SIZE(parse_map_equal_tests); ++i) {
		str = parse_map_equal_tests[i].map1;
		str2 = parse_map_equal_tests[i].map2;
		if (test_parse_map_equal(ctx, str, str2) < 0)
			return -1;
	}

	str = "{[new,old] -> [new+1-2*[(new+1)/2],old+1-2*[(old+1)/2]]}";
	map = isl_map_read_from_str(ctx, str);
	str = "{ [new, old] -> [o0, o1] : "
	       "exists (e0 = [(-1 - new + o0)/2], e1 = [(-1 - old + o1)/2]: "
	       "2e0 = -1 - new + o0 and 2e1 = -1 - old + o1 and o0 >= 0 and "
	       "o0 <= 1 and o1 >= 0 and o1 <= 1) }";
	map2 = isl_map_read_from_str(ctx, str);
	assert(isl_map_is_equal(map, map2));
	isl_map_free(map);
	isl_map_free(map2);

	str = "{[new,old] -> [new+1-2*[(new+1)/2],old+1-2*[(old+1)/2]]}";
	map = isl_map_read_from_str(ctx, str);
	str = "{[new,old] -> [(new+1)%2,(old+1)%2]}";
	map2 = isl_map_read_from_str(ctx, str);
	assert(isl_map_is_equal(map, map2));
	isl_map_free(map);
	isl_map_free(map2);

	test_parse_pwqp(ctx, "{ [i] -> i + [ (i + [i/3])/2 ] }");
	test_parse_map(ctx, "{ S1[i] -> [([i/10]),i%10] : 0 <= i <= 45 }");
	test_parse_pwaff(ctx, "{ [i] -> [i + 1] : i > 0; [a] -> [a] : a < 0 }");
	test_parse_pwqp(ctx, "{ [x] -> ([(x)/2] * [(x)/3]) }");
	test_parse_pwaff(ctx, "{ [] -> [(100)] }");

	return 0;
}

static int test_read(isl_ctx *ctx)
{
	char *filename;
	FILE *input;
	isl_basic_set *bset1, *bset2;
	const char *str = "{[y]: Exists ( alpha : 2alpha = y)}";
	int equal;

	filename = get_filename(ctx, "set", "omega");
	assert(filename);
	input = fopen(filename, "r");
	assert(input);

	bset1 = isl_basic_set_read_from_file(ctx, input);
	bset2 = isl_basic_set_read_from_str(ctx, str);

	equal = isl_basic_set_is_equal(bset1, bset2);

	isl_basic_set_free(bset1);
	isl_basic_set_free(bset2);
	free(filename);

	fclose(input);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown,
			"read sets not equal", return -1);

	return 0;
}

static int test_bounded(isl_ctx *ctx)
{
	isl_set *set;
	isl_bool bounded;

	set = isl_set_read_from_str(ctx, "[n] -> {[i] : 0 <= i <= n }");
	bounded = isl_set_is_bounded(set);
	isl_set_free(set);

	if (bounded < 0)
		return -1;
	if (!bounded)
		isl_die(ctx, isl_error_unknown,
			"set not considered bounded", return -1);

	set = isl_set_read_from_str(ctx, "{[n, i] : 0 <= i <= n }");
	bounded = isl_set_is_bounded(set);
	assert(!bounded);
	isl_set_free(set);

	if (bounded < 0)
		return -1;
	if (bounded)
		isl_die(ctx, isl_error_unknown,
			"set considered bounded", return -1);

	set = isl_set_read_from_str(ctx, "[n] -> {[i] : i <= n }");
	bounded = isl_set_is_bounded(set);
	isl_set_free(set);

	if (bounded < 0)
		return -1;
	if (bounded)
		isl_die(ctx, isl_error_unknown,
			"set considered bounded", return -1);

	return 0;
}

/* Construct the basic set { [i] : 5 <= i <= N } */
static int test_construction_1(isl_ctx *ctx)
{
	isl_space *space;
	isl_local_space *ls;
	isl_basic_set *bset;
	isl_constraint *c;

	space = isl_space_set_alloc(ctx, 1, 1);
	bset = isl_basic_set_universe(isl_space_copy(space));
	ls = isl_local_space_from_space(space);

	c = isl_constraint_alloc_inequality(isl_local_space_copy(ls));
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, -1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_param, 0, 1);
	bset = isl_basic_set_add_constraint(bset, c);

	c = isl_constraint_alloc_inequality(isl_local_space_copy(ls));
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, 1);
	c = isl_constraint_set_constant_si(c, -5);
	bset = isl_basic_set_add_constraint(bset, c);

	isl_local_space_free(ls);
	isl_basic_set_free(bset);

	return 0;
}

/* Construct the basic set { [x] : -100 <= x <= 100 }
 * using isl_basic_set_{lower,upper}_bound_val and
 * check that it is equal the same basic set parsed from a string.
 */
static int test_construction_2(isl_ctx *ctx)
{
	isl_bool equal;
	isl_val *v;
	isl_space *space;
	isl_basic_set *bset1, *bset2;

	v = isl_val_int_from_si(ctx, 100);
	space = isl_space_set_alloc(ctx, 0, 1);
	bset1 = isl_basic_set_universe(space);
	bset1 = isl_basic_set_upper_bound_val(bset1, isl_dim_set, 0,
						isl_val_copy(v));
	bset1 = isl_basic_set_lower_bound_val(bset1, isl_dim_set, 0,
						isl_val_neg(v));
	bset2 = isl_basic_set_read_from_str(ctx, "{ [x] : -100 <= x <= 100 }");
	equal = isl_basic_set_is_equal(bset1, bset2);
	isl_basic_set_free(bset1);
	isl_basic_set_free(bset2);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown,
			"failed construction", return -1);

	return 0;
}

/* Basic tests for constructing basic sets.
 */
static int test_construction(isl_ctx *ctx)
{
	if (test_construction_1(ctx) < 0)
		return -1;
	if (test_construction_2(ctx) < 0)
		return -1;
	return 0;
}

static int test_dim(isl_ctx *ctx)
{
	const char *str;
	isl_map *map1, *map2;
	int equal;

	map1 = isl_map_read_from_str(ctx,
	    "[n] -> { [i] -> [j] : exists (a = [i/10] : i - 10a <= n ) }");
	map1 = isl_map_add_dims(map1, isl_dim_in, 1);
	map2 = isl_map_read_from_str(ctx,
	    "[n] -> { [i,k] -> [j] : exists (a = [i/10] : i - 10a <= n ) }");
	equal = isl_map_is_equal(map1, map2);
	isl_map_free(map2);

	map1 = isl_map_project_out(map1, isl_dim_in, 0, 1);
	map2 = isl_map_read_from_str(ctx, "[n] -> { [i] -> [j] : n >= 0 }");
	if (equal >= 0 && equal)
		equal = isl_map_is_equal(map1, map2);

	isl_map_free(map1);
	isl_map_free(map2);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown,
			"unexpected result", return -1);

	str = "[n] -> { [i] -> [] : exists a : 0 <= i <= n and i = 2 a }";
	map1 = isl_map_read_from_str(ctx, str);
	str = "{ [i] -> [j] : exists a : 0 <= i <= j and i = 2 a }";
	map2 = isl_map_read_from_str(ctx, str);
	map1 = isl_map_move_dims(map1, isl_dim_out, 0, isl_dim_param, 0, 1);
	equal = isl_map_is_equal(map1, map2);
	isl_map_free(map1);
	isl_map_free(map2);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown,
			"unexpected result", return -1);

	return 0;
}

#undef BASE
#define BASE	multi_val
#include "isl_test_plain_equal_templ.c"

#undef BASE
#define BASE	multi_aff
#include "isl_test_plain_equal_templ.c"

/* Check that "val" is equal to the value described by "str".
 * If "str" is "NaN", then check for a NaN value explicitly.
 */
static isl_stat val_check_equal(__isl_keep isl_val *val, const char *str)
{
	isl_bool ok, is_nan;
	isl_ctx *ctx;
	isl_val *res;

	if (!val)
		return isl_stat_error;

	ctx = isl_val_get_ctx(val);
	res = isl_val_read_from_str(ctx, str);
	is_nan = isl_val_is_nan(res);
	if (is_nan < 0)
		ok = isl_bool_error;
	else if (is_nan)
		ok = isl_val_is_nan(val);
	else
		ok = isl_val_eq(val, res);
	isl_val_free(res);
	if (ok < 0)
		return isl_stat_error;
	if (!ok)
		isl_die(ctx, isl_error_unknown,
			"unexpected result", return isl_stat_error);
	return isl_stat_ok;
}

struct {
	__isl_give isl_val *(*op)(__isl_take isl_val *v);
	const char *arg;
	const char *res;
} val_un_tests[] = {
	{ &isl_val_neg, "0", "0" },
	{ &isl_val_abs, "0", "0" },
	{ &isl_val_pow2, "0", "1" },
	{ &isl_val_floor, "0", "0" },
	{ &isl_val_ceil, "0", "0" },
	{ &isl_val_neg, "1", "-1" },
	{ &isl_val_neg, "-1", "1" },
	{ &isl_val_neg, "1/2", "-1/2" },
	{ &isl_val_neg, "-1/2", "1/2" },
	{ &isl_val_neg, "infty", "-infty" },
	{ &isl_val_neg, "-infty", "infty" },
	{ &isl_val_neg, "NaN", "NaN" },
	{ &isl_val_abs, "1", "1" },
	{ &isl_val_abs, "-1", "1" },
	{ &isl_val_abs, "1/2", "1/2" },
	{ &isl_val_abs, "-1/2", "1/2" },
	{ &isl_val_abs, "infty", "infty" },
	{ &isl_val_abs, "-infty", "infty" },
	{ &isl_val_abs, "NaN", "NaN" },
	{ &isl_val_floor, "1", "1" },
	{ &isl_val_floor, "-1", "-1" },
	{ &isl_val_floor, "1/2", "0" },
	{ &isl_val_floor, "-1/2", "-1" },
	{ &isl_val_floor, "infty", "infty" },
	{ &isl_val_floor, "-infty", "-infty" },
	{ &isl_val_floor, "NaN", "NaN" },
	{ &isl_val_ceil, "1", "1" },
	{ &isl_val_ceil, "-1", "-1" },
	{ &isl_val_ceil, "1/2", "1" },
	{ &isl_val_ceil, "-1/2", "0" },
	{ &isl_val_ceil, "infty", "infty" },
	{ &isl_val_ceil, "-infty", "-infty" },
	{ &isl_val_ceil, "NaN", "NaN" },
	{ &isl_val_pow2, "-3", "1/8" },
	{ &isl_val_pow2, "-1", "1/2" },
	{ &isl_val_pow2, "1", "2" },
	{ &isl_val_pow2, "2", "4" },
	{ &isl_val_pow2, "3", "8" },
	{ &isl_val_inv, "1", "1" },
	{ &isl_val_inv, "2", "1/2" },
	{ &isl_val_inv, "1/2", "2" },
	{ &isl_val_inv, "-2", "-1/2" },
	{ &isl_val_inv, "-1/2", "-2" },
	{ &isl_val_inv, "0", "NaN" },
	{ &isl_val_inv, "NaN", "NaN" },
	{ &isl_val_inv, "infty", "0" },
	{ &isl_val_inv, "-infty", "0" },
};

/* Perform some basic tests of unary operations on isl_val objects.
 */
static int test_un_val(isl_ctx *ctx)
{
	int i;
	isl_val *v;
	__isl_give isl_val *(*fn)(__isl_take isl_val *v);

	for (i = 0; i < ARRAY_SIZE(val_un_tests); ++i) {
		isl_stat r;

		v = isl_val_read_from_str(ctx, val_un_tests[i].arg);
		fn = val_un_tests[i].op;
		v = fn(v);
		r = val_check_equal(v, val_un_tests[i].res);
		isl_val_free(v);
		if (r < 0)
			return -1;
	}

	return 0;
}

struct {
	__isl_give isl_val *(*fn)(__isl_take isl_val *v1,
				__isl_take isl_val *v2);
} val_bin_op[] = {
	['+'] = { &isl_val_add },
	['-'] = { &isl_val_sub },
	['*'] = { &isl_val_mul },
	['/'] = { &isl_val_div },
	['g'] = { &isl_val_gcd },
	['m'] = { &isl_val_min },
	['M'] = { &isl_val_max },
};

struct {
	const char *arg1;
	unsigned char op;
	const char *arg2;
	const char *res;
} val_bin_tests[] = {
	{ "0", '+', "0", "0" },
	{ "1", '+', "0", "1" },
	{ "1", '+', "1", "2" },
	{ "1", '-', "1", "0" },
	{ "1", '*', "1", "1" },
	{ "1", '/', "1", "1" },
	{ "2", '*', "3", "6" },
	{ "2", '*', "1/2", "1" },
	{ "2", '*', "1/3", "2/3" },
	{ "2/3", '*', "3/5", "2/5" },
	{ "2/3", '*', "7/5", "14/15" },
	{ "2", '/', "1/2", "4" },
	{ "-2", '/', "-1/2", "4" },
	{ "-2", '/', "1/2", "-4" },
	{ "2", '/', "-1/2", "-4" },
	{ "2", '/', "2", "1" },
	{ "2", '/', "3", "2/3" },
	{ "2/3", '/', "5/3", "2/5" },
	{ "2/3", '/', "5/7", "14/15" },
	{ "0", '/', "0", "NaN" },
	{ "42", '/', "0", "NaN" },
	{ "-42", '/', "0", "NaN" },
	{ "infty", '/', "0", "NaN" },
	{ "-infty", '/', "0", "NaN" },
	{ "NaN", '/', "0", "NaN" },
	{ "0", '/', "NaN", "NaN" },
	{ "42", '/', "NaN", "NaN" },
	{ "-42", '/', "NaN", "NaN" },
	{ "infty", '/', "NaN", "NaN" },
	{ "-infty", '/', "NaN", "NaN" },
	{ "NaN", '/', "NaN", "NaN" },
	{ "0", '/', "infty", "0" },
	{ "42", '/', "infty", "0" },
	{ "-42", '/', "infty", "0" },
	{ "infty", '/', "infty", "NaN" },
	{ "-infty", '/', "infty", "NaN" },
	{ "NaN", '/', "infty", "NaN" },
	{ "0", '/', "-infty", "0" },
	{ "42", '/', "-infty", "0" },
	{ "-42", '/', "-infty", "0" },
	{ "infty", '/', "-infty", "NaN" },
	{ "-infty", '/', "-infty", "NaN" },
	{ "NaN", '/', "-infty", "NaN" },
	{ "1", '-', "1/3", "2/3" },
	{ "1/3", '+', "1/2", "5/6" },
	{ "1/2", '+', "1/2", "1" },
	{ "3/4", '-', "1/4", "1/2" },
	{ "1/2", '-', "1/3", "1/6" },
	{ "infty", '+', "42", "infty" },
	{ "infty", '+', "infty", "infty" },
	{ "42", '+', "infty", "infty" },
	{ "infty", '-', "infty", "NaN" },
	{ "infty", '*', "infty", "infty" },
	{ "infty", '*', "-infty", "-infty" },
	{ "-infty", '*', "infty", "-infty" },
	{ "-infty", '*', "-infty", "infty" },
	{ "0", '*', "infty", "NaN" },
	{ "1", '*', "infty", "infty" },
	{ "infty", '*', "0", "NaN" },
	{ "infty", '*', "42", "infty" },
	{ "42", '-', "infty", "-infty" },
	{ "infty", '+', "-infty", "NaN" },
	{ "4", 'g', "6", "2" },
	{ "5", 'g', "6", "1" },
	{ "42", 'm', "3", "3" },
	{ "42", 'M', "3", "42" },
	{ "3", 'm', "42", "3" },
	{ "3", 'M', "42", "42" },
	{ "42", 'm', "infty", "42" },
	{ "42", 'M', "infty", "infty" },
	{ "42", 'm', "-infty", "-infty" },
	{ "42", 'M', "-infty", "42" },
	{ "42", 'm', "NaN", "NaN" },
	{ "42", 'M', "NaN", "NaN" },
	{ "infty", 'm', "-infty", "-infty" },
	{ "infty", 'M', "-infty", "infty" },
};

/* Perform some basic tests of binary operations on isl_val objects.
 */
static int test_bin_val(isl_ctx *ctx)
{
	int i;
	isl_val *v1, *v2, *res;
	__isl_give isl_val *(*fn)(__isl_take isl_val *v1,
				__isl_take isl_val *v2);
	int ok;

	for (i = 0; i < ARRAY_SIZE(val_bin_tests); ++i) {
		v1 = isl_val_read_from_str(ctx, val_bin_tests[i].arg1);
		v2 = isl_val_read_from_str(ctx, val_bin_tests[i].arg2);
		res = isl_val_read_from_str(ctx, val_bin_tests[i].res);
		fn = val_bin_op[val_bin_tests[i].op].fn;
		v1 = fn(v1, v2);
		if (isl_val_is_nan(res))
			ok = isl_val_is_nan(v1);
		else
			ok = isl_val_eq(v1, res);
		isl_val_free(v1);
		isl_val_free(res);
		if (ok < 0)
			return -1;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return -1);
	}

	return 0;
}

/* Perform some basic tests on isl_val objects.
 */
static int test_val(isl_ctx *ctx)
{
	if (test_un_val(ctx) < 0)
		return -1;
	if (test_bin_val(ctx) < 0)
		return -1;
	return 0;
}

/* Sets described using existentially quantified variables that
 * can also be described without.
 */
static const char *elimination_tests[] = {
	"{ [i,j] : 2 * [i/2] + 3 * [j/4] <= 10 and 2 i = j }",
	"{ [m, w] : exists a : w - 2m - 5 <= 3a <= m - 2w }",
	"{ [m, w] : exists a : w >= 0 and a < m and -1 + w <= a <= 2m - w }",
};

/* Check that redundant existentially quantified variables are
 * getting removed.
 */
static int test_elimination(isl_ctx *ctx)
{
	int i;
	isl_size n;
	isl_basic_set *bset;

	for (i = 0; i < ARRAY_SIZE(elimination_tests); ++i) {
		bset = isl_basic_set_read_from_str(ctx, elimination_tests[i]);
		n = isl_basic_set_dim(bset, isl_dim_div);
		isl_basic_set_free(bset);
		if (n < 0)
			return -1;
		if (n != 0)
			isl_die(ctx, isl_error_unknown,
				"expecting no existentials", return -1);
	}

	return 0;
}

static int test_div(isl_ctx *ctx)
{
	const char *str;
	int empty;
	isl_space *space;
	isl_set *set;
	isl_local_space *ls;
	struct isl_basic_set *bset;
	struct isl_constraint *c;

	/* test 1 */
	space = isl_space_set_alloc(ctx, 0, 3);
	bset = isl_basic_set_universe(isl_space_copy(space));
	ls = isl_local_space_from_space(space);

	c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
	c = isl_constraint_set_constant_si(c, -1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, 1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 1, 3);
	bset = isl_basic_set_add_constraint(bset, c);

	c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
	c = isl_constraint_set_constant_si(c, 1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, -1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 2, 3);
	bset = isl_basic_set_add_constraint(bset, c);

	bset = isl_basic_set_project_out(bset, isl_dim_set, 1, 2);

	assert(bset && bset->n_div == 1);
	isl_local_space_free(ls);
	isl_basic_set_free(bset);

	/* test 2 */
	space = isl_space_set_alloc(ctx, 0, 3);
	bset = isl_basic_set_universe(isl_space_copy(space));
	ls = isl_local_space_from_space(space);

	c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
	c = isl_constraint_set_constant_si(c, 1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, -1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 1, 3);
	bset = isl_basic_set_add_constraint(bset, c);

	c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
	c = isl_constraint_set_constant_si(c, -1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, 1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 2, 3);
	bset = isl_basic_set_add_constraint(bset, c);

	bset = isl_basic_set_project_out(bset, isl_dim_set, 1, 2);

	assert(bset && bset->n_div == 1);
	isl_local_space_free(ls);
	isl_basic_set_free(bset);

	/* test 3 */
	space = isl_space_set_alloc(ctx, 0, 3);
	bset = isl_basic_set_universe(isl_space_copy(space));
	ls = isl_local_space_from_space(space);

	c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
	c = isl_constraint_set_constant_si(c, 1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, -1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 1, 3);
	bset = isl_basic_set_add_constraint(bset, c);

	c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
	c = isl_constraint_set_constant_si(c, -3);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, 1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 2, 4);
	bset = isl_basic_set_add_constraint(bset, c);

	bset = isl_basic_set_project_out(bset, isl_dim_set, 1, 2);

	assert(bset && bset->n_div == 1);
	isl_local_space_free(ls);
	isl_basic_set_free(bset);

	/* test 4 */
	space = isl_space_set_alloc(ctx, 0, 3);
	bset = isl_basic_set_universe(isl_space_copy(space));
	ls = isl_local_space_from_space(space);

	c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
	c = isl_constraint_set_constant_si(c, 2);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, -1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 1, 3);
	bset = isl_basic_set_add_constraint(bset, c);

	c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
	c = isl_constraint_set_constant_si(c, -1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, 1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 2, 6);
	bset = isl_basic_set_add_constraint(bset, c);

	bset = isl_basic_set_project_out(bset, isl_dim_set, 1, 2);

	assert(isl_basic_set_is_empty(bset));
	isl_local_space_free(ls);
	isl_basic_set_free(bset);

	/* test 5 */
	space = isl_space_set_alloc(ctx, 0, 3);
	bset = isl_basic_set_universe(isl_space_copy(space));
	ls = isl_local_space_from_space(space);

	c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, -1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 2, 3);
	bset = isl_basic_set_add_constraint(bset, c);

	c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, 1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 1, -3);
	bset = isl_basic_set_add_constraint(bset, c);

	bset = isl_basic_set_project_out(bset, isl_dim_set, 2, 1);

	assert(bset && bset->n_div == 0);
	isl_basic_set_free(bset);
	isl_local_space_free(ls);

	/* test 6 */
	space = isl_space_set_alloc(ctx, 0, 3);
	bset = isl_basic_set_universe(isl_space_copy(space));
	ls = isl_local_space_from_space(space);

	c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, -1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 2, 6);
	bset = isl_basic_set_add_constraint(bset, c);

	c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, 1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 1, -3);
	bset = isl_basic_set_add_constraint(bset, c);

	bset = isl_basic_set_project_out(bset, isl_dim_set, 2, 1);

	assert(bset && bset->n_div == 1);
	isl_basic_set_free(bset);
	isl_local_space_free(ls);

	/* test 7 */
	/* This test is a bit tricky.  We set up an equality
	 *		a + 3b + 3c = 6 e0
	 * Normalization of divs creates _two_ divs
	 *		a = 3 e0
	 *		c - b - e0 = 2 e1
	 * Afterwards e0 is removed again because it has coefficient -1
	 * and we end up with the original equality and div again.
	 * Perhaps we can avoid the introduction of this temporary div.
	 */
	space = isl_space_set_alloc(ctx, 0, 4);
	bset = isl_basic_set_universe(isl_space_copy(space));
	ls = isl_local_space_from_space(space);

	c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, -1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 1, -3);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 2, -3);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 3, 6);
	bset = isl_basic_set_add_constraint(bset, c);

	bset = isl_basic_set_project_out(bset, isl_dim_set, 3, 1);

	/* Test disabled for now */
	/*
	assert(bset && bset->n_div == 1);
	*/
	isl_local_space_free(ls);
	isl_basic_set_free(bset);

	/* test 8 */
	space = isl_space_set_alloc(ctx, 0, 5);
	bset = isl_basic_set_universe(isl_space_copy(space));
	ls = isl_local_space_from_space(space);

	c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, -1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 1, -3);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 3, -3);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 4, 6);
	bset = isl_basic_set_add_constraint(bset, c);

	c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, -1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 2, 1);
	c = isl_constraint_set_constant_si(c, 1);
	bset = isl_basic_set_add_constraint(bset, c);

	bset = isl_basic_set_project_out(bset, isl_dim_set, 4, 1);

	/* Test disabled for now */
	/*
	assert(bset && bset->n_div == 1);
	*/
	isl_local_space_free(ls);
	isl_basic_set_free(bset);

	/* test 9 */
	space = isl_space_set_alloc(ctx, 0, 4);
	bset = isl_basic_set_universe(isl_space_copy(space));
	ls = isl_local_space_from_space(space);

	c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, 1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 1, -1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 2, -2);
	bset = isl_basic_set_add_constraint(bset, c);

	c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, -1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 3, 3);
	c = isl_constraint_set_constant_si(c, 2);
	bset = isl_basic_set_add_constraint(bset, c);

	bset = isl_basic_set_project_out(bset, isl_dim_set, 2, 2);

	bset = isl_basic_set_fix_si(bset, isl_dim_set, 0, 2);

	assert(!isl_basic_set_is_empty(bset));

	isl_local_space_free(ls);
	isl_basic_set_free(bset);

	/* test 10 */
	space = isl_space_set_alloc(ctx, 0, 3);
	bset = isl_basic_set_universe(isl_space_copy(space));
	ls = isl_local_space_from_space(space);

	c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, 1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_set, 2, -2);
	bset = isl_basic_set_add_constraint(bset, c);

	bset = isl_basic_set_project_out(bset, isl_dim_set, 2, 1);

	bset = isl_basic_set_fix_si(bset, isl_dim_set, 0, 2);

	isl_local_space_free(ls);
	isl_basic_set_free(bset);

	str = "{ [i] : exists (e0, e1: 3e1 >= 1 + 2e0 and "
	    "8e1 <= -1 + 5i - 5e0 and 2e1 >= 1 + 2i - 5e0) }";
	set = isl_set_read_from_str(ctx, str);
	set = isl_set_compute_divs(set);
	isl_set_free(set);
	if (!set)
		return -1;

	if (test_elimination(ctx) < 0)
		return -1;

	str = "{ [i,j,k] : 3 + i + 2j >= 0 and 2 * [(i+2j)/4] <= k }";
	set = isl_set_read_from_str(ctx, str);
	set = isl_set_remove_divs_involving_dims(set, isl_dim_set, 0, 2);
	set = isl_set_fix_si(set, isl_dim_set, 2, -3);
	empty = isl_set_is_empty(set);
	isl_set_free(set);
	if (empty < 0)
		return -1;
	if (!empty)
		isl_die(ctx, isl_error_unknown,
			"result not as accurate as expected", return -1);

	return 0;
}

void test_application_case(struct isl_ctx *ctx, const char *name)
{
	char *filename;
	FILE *input;
	struct isl_basic_set *bset1, *bset2;
	struct isl_basic_map *bmap;

	filename = get_filename(ctx, name, "omega");
	assert(filename);
	input = fopen(filename, "r");
	assert(input);

	bset1 = isl_basic_set_read_from_file(ctx, input);
	bmap = isl_basic_map_read_from_file(ctx, input);

	bset1 = isl_basic_set_apply(bset1, bmap);

	bset2 = isl_basic_set_read_from_file(ctx, input);

	assert(isl_basic_set_is_equal(bset1, bset2) == 1);

	isl_basic_set_free(bset1);
	isl_basic_set_free(bset2);
	free(filename);

	fclose(input);
}

static int test_application(isl_ctx *ctx)
{
	test_application_case(ctx, "application");
	test_application_case(ctx, "application2");

	return 0;
}

void test_affine_hull_case(struct isl_ctx *ctx, const char *name)
{
	char *filename;
	FILE *input;
	struct isl_basic_set *bset1, *bset2;

	filename = get_filename(ctx, name, "polylib");
	assert(filename);
	input = fopen(filename, "r");
	assert(input);

	bset1 = isl_basic_set_read_from_file(ctx, input);
	bset2 = isl_basic_set_read_from_file(ctx, input);

	bset1 = isl_basic_set_affine_hull(bset1);

	assert(isl_basic_set_is_equal(bset1, bset2) == 1);

	isl_basic_set_free(bset1);
	isl_basic_set_free(bset2);
	free(filename);

	fclose(input);
}

/* Pairs of sets and the corresponding expected results of
 * isl_basic_set_recession_cone.
 */
struct {
	const char *set;
	const char *cone;
} recession_cone_tests[] = {
	{ "{ [i] : 0 <= i <= 10 }", "{ [0] }" },
	{ "{ [i] : 0 <= i }", "{ [i] : 0 <= i }" },
	{ "{ [i] : i <= 10 }", "{ [i] : i <= 0 }" },
	{ "{ [i] : false }", "{ [i] : false }" },
};

/* Perform some basic isl_basic_set_recession_cone tests.
 */
static int test_recession_cone(struct isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(recession_cone_tests); ++i) {
		const char *str;
		isl_basic_set *bset;
		isl_basic_set *cone, *expected;
		isl_bool equal;

		str = recession_cone_tests[i].set;
		bset = isl_basic_set_read_from_str(ctx, str);
		str = recession_cone_tests[i].cone;
		expected = isl_basic_set_read_from_str(ctx, str);
		cone = isl_basic_set_recession_cone(bset);
		equal = isl_basic_set_is_equal(cone, expected);
		isl_basic_set_free(cone);
		isl_basic_set_free(expected);
		if (equal < 0)
			return -1;
		if (!equal)
			isl_die(ctx, isl_error_unknown, "unexpected cone",
				return -1);
	}

	return 0;
}

int test_affine_hull(struct isl_ctx *ctx)
{
	const char *str;
	isl_set *set;
	isl_basic_set *bset, *bset2;
	isl_size n;
	isl_bool subset;

	test_affine_hull_case(ctx, "affine2");
	test_affine_hull_case(ctx, "affine");
	test_affine_hull_case(ctx, "affine3");

	str = "[m] -> { [i0] : exists (e0, e1: e1 <= 1 + i0 and "
			"m >= 3 and 4i0 <= 2 + m and e1 >= i0 and "
			"e1 >= 0 and e1 <= 2 and e1 >= 1 + 2e0 and "
			"2e1 <= 1 + m + 4e0 and 2e1 >= 2 - m + 4i0 - 4e0) }";
	set = isl_set_read_from_str(ctx, str);
	bset = isl_set_affine_hull(set);
	n = isl_basic_set_dim(bset, isl_dim_div);
	isl_basic_set_free(bset);
	if (n < 0)
		return -1;
	if (n != 0)
		isl_die(ctx, isl_error_unknown, "not expecting any divs",
			return -1);

	/* Check that isl_map_affine_hull is not confused by
	 * the reordering of divs in isl_map_align_divs.
	 */
	str = "{ [a, b, c, 0] : exists (e0 = [(b)/32], e1 = [(c)/32]: "
				"32e0 = b and 32e1 = c); "
		"[a, 0, c, 0] : exists (e0 = [(c)/32]: 32e0 = c) }";
	set = isl_set_read_from_str(ctx, str);
	bset = isl_set_affine_hull(set);
	isl_basic_set_free(bset);
	if (!bset)
		return -1;

	str = "{ [a] : exists e0, e1, e2: 32e1 = 31 + 31a + 31e0 and "
			"32e2 = 31 + 31e0 }";
	set = isl_set_read_from_str(ctx, str);
	bset = isl_set_affine_hull(set);
	str = "{ [a] : exists e : a = 32 e }";
	bset2 = isl_basic_set_read_from_str(ctx, str);
	subset = isl_basic_set_is_subset(bset, bset2);
	isl_basic_set_free(bset);
	isl_basic_set_free(bset2);
	if (subset < 0)
		return -1;
	if (!subset)
		isl_die(ctx, isl_error_unknown, "not as accurate as expected",
			return -1);

	return 0;
}

/* Test a special case of isl_set_plain_unshifted_simple_hull
 * where older versions of isl would include a redundant constraint
 * in the result.
 * Check that the result does not have any constraints.
 */
static isl_stat test_plain_unshifted_simple_hull_special(isl_ctx *ctx)
{
	const char *str;
	isl_bool is_universe;
	isl_set *set;
	isl_basic_set *bset;

	str = "{[x, y] : x = 0 or 2*((x+y)//2) <= y + 2 }";
	set = isl_set_read_from_str(ctx, str);
	bset = isl_set_plain_unshifted_simple_hull(set);
	is_universe = isl_basic_set_plain_is_universe(bset);
	isl_basic_set_free(bset);

	if (is_universe < 0)
		return isl_stat_error;
	if (!is_universe)
		isl_die(ctx, isl_error_unknown,
			"hull should not have any constraints",
			return isl_stat_error);

	return isl_stat_ok;
}

/* Inputs for simple hull tests, consisting of
 * the specific simple hull function, the input set and the expected result.
 */
struct {
	__isl_give isl_basic_set *(*fn)(__isl_take isl_set *set);
	const char *set;
	const char *hull;
} simple_hull_tests[] = {
	{ &isl_set_plain_unshifted_simple_hull,
	  "{ [i,j] : i >= 1 and j >= 1 or i >= 2 and j <= 10 }",
	  "{ [i,j] : i >= 1 }" },
	{ &isl_set_plain_unshifted_simple_hull,
	  "{ [n,i,j,k] : (i mod 3 = 2 and j mod 4 = 2) or "
		"(j mod 4 = 2 and k mod 6 = n) }",
	  "{ [n,i,j,k] : j mod 4 = 2 }" },
	{ &isl_set_unshifted_simple_hull,
	  "{ [0,x,y] : x <= -1; [1,x,y] : x <= y <= -x; [2,x,y] : x <= 1 }",
	  "{ [t,x,y] : 0 <= t <= 2 and x <= 1 }" },
	{ &isl_set_simple_hull,
	  "{ [a, b] : b <= 0 and "
			"2*floor((-2*floor((b)/2))/5) >= a - floor((b)/2); "
	    "[a, b] : a mod 2 = 0 }",
	  "{ [a, b] }" },
};

/* Basic tests for various simple hull functions.
 */
static int test_various_simple_hull(isl_ctx *ctx)
{
	int i;
	isl_set *set;
	isl_basic_set *hull, *expected;
	isl_bool equal;

	for (i = 0; i < ARRAY_SIZE(simple_hull_tests); ++i) {
		const char *str;
		str = simple_hull_tests[i].set;
		set = isl_set_read_from_str(ctx, str);
		str = simple_hull_tests[i].hull;
		expected = isl_basic_set_read_from_str(ctx, str);
		hull = simple_hull_tests[i].fn(set);
		equal = isl_basic_set_is_equal(hull, expected);
		isl_basic_set_free(hull);
		isl_basic_set_free(expected);
		if (equal < 0)
			return -1;
		if (!equal)
			isl_die(ctx, isl_error_unknown, "unexpected hull",
				return -1);
	}

	return 0;
}

static int test_simple_hull(struct isl_ctx *ctx)
{
	const char *str;
	isl_set *set;
	isl_basic_set *bset;
	isl_bool is_empty;

	str = "{ [x, y] : 3y <= 2x and y >= -2 + 2x and 2y >= 2 - x;"
		"[y, x] : 3y <= 2x and y >= -2 + 2x and 2y >= 2 - x }";
	set = isl_set_read_from_str(ctx, str);
	bset = isl_set_simple_hull(set);
	is_empty = isl_basic_set_is_empty(bset);
	isl_basic_set_free(bset);

	if (is_empty == isl_bool_error)
		return -1;

	if (is_empty == isl_bool_false)
		isl_die(ctx, isl_error_unknown, "Empty set should be detected",
			return -1);

	if (test_plain_unshifted_simple_hull_special(ctx) < 0)
		return -1;
	if (test_various_simple_hull(ctx) < 0)
		return -1;

	return 0;
}

/* Inputs for isl_set_get_simple_fixed_box_hull tests.
 * "set" is the input set.
 * "offset" is the expected box offset.
 * "size" is the expected box size.
 */
static struct {
	const char *set;
	const char *offset;
	const char *size;
} box_hull_tests[] = {
	{ "{ S[x, y] : 0 <= x, y < 10 }", "{ S[0, 0] }", "{ S[10, 10] }" },
	{ "[N] -> { S[x, y] : N <= x, y < N + 10 }",
	  "[N] -> { S[N, N] }", "{ S[10, 10] }" },
	{ "{ S[x, y] : 0 <= x + y, x - y < 10 }",
	  "{ S[0, -4] }", "{ S[10, 9] }" },
	{ "{ [i=0:10] : exists (e0, e1: 3e1 >= 1 + 2e0 and "
	    "8e1 <= -1 + 5i - 5e0 and 2e1 >= 1 + 2i - 5e0) }",
	  "{ [3] }", "{ [8] }" },
	{ "[N] -> { [w = 0:17] : exists (e0: w < 2N and "
	    "-1 + w <= e0 <= w and 2e0 >= N + w and w <= 2e0 <= 15 + w) }",
	  "[N] -> { [N] }", "{ [9] }" },
};

/* Perform basic isl_set_get_simple_fixed_box_hull tests.
 */
static int test_box_hull(struct isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(box_hull_tests); ++i) {
		const char *str;
		isl_stat r;
		isl_set *set;
		isl_multi_aff *offset;
		isl_multi_val *size;
		isl_fixed_box *box;

		set = isl_set_read_from_str(ctx, box_hull_tests[i].set);
		box = isl_set_get_simple_fixed_box_hull(set);
		offset = isl_fixed_box_get_offset(box);
		size = isl_fixed_box_get_size(box);
		str = box_hull_tests[i].offset;
		r = multi_aff_check_plain_equal(offset, str);
		str = box_hull_tests[i].size;
		if (r >= 0)
			r = multi_val_check_plain_equal(size, str);
		isl_multi_aff_free(offset);
		isl_multi_val_free(size);
		isl_fixed_box_free(box);
		isl_set_free(set);
		if (r < 0)
			return -1;
	}

	return 0;
}

void test_convex_hull_case(struct isl_ctx *ctx, const char *name)
{
	char *filename;
	FILE *input;
	struct isl_basic_set *bset1, *bset2;
	struct isl_set *set;

	filename = get_filename(ctx, name, "polylib");
	assert(filename);
	input = fopen(filename, "r");
	assert(input);

	bset1 = isl_basic_set_read_from_file(ctx, input);
	bset2 = isl_basic_set_read_from_file(ctx, input);

	set = isl_basic_set_union(bset1, bset2);
	bset1 = isl_set_convex_hull(set);

	bset2 = isl_basic_set_read_from_file(ctx, input);

	assert(isl_basic_set_is_equal(bset1, bset2) == 1);

	isl_basic_set_free(bset1);
	isl_basic_set_free(bset2);
	free(filename);

	fclose(input);
}

struct {
	const char *set;
	const char *hull;
} convex_hull_tests[] = {
	{ "{ [i0, i1, i2] : (i2 = 1 and i0 = 0 and i1 >= 0) or "
	       "(i0 = 1 and i1 = 0 and i2 = 1) or "
	       "(i0 = 0 and i1 = 0 and i2 = 0) }",
	  "{ [i0, i1, i2] : i0 >= 0 and i2 >= i0 and i2 <= 1 and i1 >= 0 }" },
	{ "[n] -> { [i0, i1, i0] : i0 <= -4 + n; "
	    "[i0, i0, i2] : n = 6 and i0 >= 0 and i2 <= 7 - i0 and "
	    "i2 <= 5 and i2 >= 4; "
	    "[3, i1, 3] : n = 5 and i1 <= 2 and i1 >= 0 }",
	  "[n] -> { [i0, i1, i2] : i2 <= -1 + n and 2i2 <= -6 + 3n - i0 and "
	    "i2 <= 5 + i0 and i2 >= i0 }" },
	{ "{ [x, y] : 3y <= 2x and y >= -2 + 2x and 2y >= 2 - x }",
	    "{ [x, y] : 1 = 0 }" },
	{ "{ [x, y, z] : 0 <= x, y, z <= 10; [x, y, 0] : x >= 0 and y > 0; "
	    "[x, y, 0] : x >= 0 and y < 0 }",
	    "{ [x, y, z] : x >= 0 and 0 <= z <= 10 }" },
	{ "{ [a, b, c] : a <= 1 and -a < b <= 1 and 0 <= c <= 2 - a - b and "
			    "c <= a; "
	    "[0, 2, 0]; [3, 1, 0] }",
	    "{ [a, b, c] : b > -a and 2b >= -1 + a and 0 <= c <= a and "
			    "5c <= 6 - a - 3b }" },
};

static int test_convex_hull_algo(isl_ctx *ctx, int convex)
{
	int i;
	int orig_convex = ctx->opt->convex;
	ctx->opt->convex = convex;

	test_convex_hull_case(ctx, "convex0");
	test_convex_hull_case(ctx, "convex1");
	test_convex_hull_case(ctx, "convex2");
	test_convex_hull_case(ctx, "convex3");
	test_convex_hull_case(ctx, "convex4");
	test_convex_hull_case(ctx, "convex5");
	test_convex_hull_case(ctx, "convex6");
	test_convex_hull_case(ctx, "convex7");
	test_convex_hull_case(ctx, "convex8");
	test_convex_hull_case(ctx, "convex9");
	test_convex_hull_case(ctx, "convex10");
	test_convex_hull_case(ctx, "convex11");
	test_convex_hull_case(ctx, "convex12");
	test_convex_hull_case(ctx, "convex13");
	test_convex_hull_case(ctx, "convex14");
	test_convex_hull_case(ctx, "convex15");

	for (i = 0; i < ARRAY_SIZE(convex_hull_tests); ++i) {
		isl_set *set1, *set2;
		int equal;

		set1 = isl_set_read_from_str(ctx, convex_hull_tests[i].set);
		set2 = isl_set_read_from_str(ctx, convex_hull_tests[i].hull);
		set1 = isl_set_from_basic_set(isl_set_convex_hull(set1));
		equal = isl_set_is_equal(set1, set2);
		isl_set_free(set1);
		isl_set_free(set2);

		if (equal < 0)
			return -1;
		if (!equal)
			isl_die(ctx, isl_error_unknown,
				"unexpected convex hull", return -1);
	}

	ctx->opt->convex = orig_convex;

	return 0;
}

static int test_convex_hull(isl_ctx *ctx)
{
	if (test_convex_hull_algo(ctx, ISL_CONVEX_HULL_FM) < 0)
		return -1;
	if (test_convex_hull_algo(ctx, ISL_CONVEX_HULL_WRAP) < 0)
		return -1;
	return 0;
}

/* Check that computing the gist of "map" with respect to "context"
 * does not make any copy of "map" get marked empty.
 * Earlier versions of isl would end up doing that.
 */
static isl_stat test_gist_empty_pair(isl_ctx *ctx, const char *map,
	const char *context)
{
	isl_map *m1, *m2, *m3;
	isl_bool empty_before, empty_after;

	m1 = isl_map_read_from_str(ctx, map);
	m2 = isl_map_read_from_str(ctx, context);
	m3 = isl_map_copy(m1);
	empty_before = isl_map_is_empty(m3);
	m1 = isl_map_gist(m1, m2);
	empty_after = isl_map_is_empty(m3);
	isl_map_free(m1);
	isl_map_free(m3);

	if (empty_before < 0 || empty_after < 0)
		return isl_stat_error;
	if (empty_before)
		isl_die(ctx, isl_error_unknown, "map should not be empty",
			return isl_stat_error);
	if (empty_after)
		isl_die(ctx, isl_error_unknown, "map should still not be empty",
			return isl_stat_error);

	return isl_stat_ok;
}

/* Check that computing a gist does not make any copy of the input
 * get marked empty.
 * Earlier versions of isl would end up doing that on some pairs of inputs.
 */
static isl_stat test_gist_empty(isl_ctx *ctx)
{
	const char *map, *context;

	map = "{ [] -> [a, b, c] : 2b = 1 + a }";
	context = "{ [] -> [a, b, c] : 2c = 2 + a }";
	if (test_gist_empty_pair(ctx, map, context) < 0)
		return isl_stat_error;
	map = "{ [] -> [0, 0] }";
	context = "{ [] -> [a, b] : a > b }";
	if (test_gist_empty_pair(ctx, map, context) < 0)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Inputs to isl_map_plain_gist_basic_map, along with the expected output.
 */
struct {
	const char *map;
	const char *context;
	const char *gist;
} plain_gist_tests[] = {
	{ "{ [i] -> [j] : i >= 1 and j >= 1 or i >= 2 and j <= 10 }",
	  "{ [i] -> [j] : i >= 1 }",
	  "{ [i] -> [j] : j >= 1 or i >= 2 and j <= 10 }" },
	{ "{ [n] -> [i,j,k] : (i mod 3 = 2 and j mod 4 = 2) or "
		"(j mod 4 = 2 and k mod 6 = n) }",
	  "{ [n] -> [i,j,k] : j mod 4 = 2 }",
	  "{ [n] -> [i,j,k] : (i mod 3 = 2) or (k mod 6 = n) }" },
	{ "{ [i] -> [j] : i > j and (exists a,b : i <= 2a + 5b <= 2) }",
	  "{ [i] -> [j] : i > j }",
	  "{ [i] -> [j] : exists a,b : i <= 2a + 5b <= 2 }" },
};

/* Basic tests for isl_map_plain_gist_basic_map.
 */
static int test_plain_gist(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(plain_gist_tests); ++i) {
		const char *str;
		int equal;
		isl_map *map, *gist;
		isl_basic_map *context;

		map = isl_map_read_from_str(ctx, plain_gist_tests[i].map);
		str = plain_gist_tests[i].context;
		context = isl_basic_map_read_from_str(ctx, str);
		map = isl_map_plain_gist_basic_map(map, context);
		gist = isl_map_read_from_str(ctx, plain_gist_tests[i].gist);
		equal = isl_map_is_equal(map, gist);
		isl_map_free(map);
		isl_map_free(gist);
		if (equal < 0)
			return -1;
		if (!equal)
			isl_die(ctx, isl_error_unknown,
				"incorrect gist result", return -1);
	}

	return 0;
}

/* Inputs for isl_basic_set_gist tests that are expected to fail.
 */
struct {
	const char *set;
	const char *context;
} gist_fail_tests[] = {
	{ "{ [i] : exists (e0, e1: 3e1 >= 1 + 2e0 and "
	    "8e1 <= -1 + 5i - 5e0 and 2e1 >= 1 + 2i - 5e0) }",
	  "{ [i] : i >= 0 }" },
};

/* Check that isl_basic_set_gist fails (gracefully) when expected.
 * In particular, the user should be able to recover from the failure.
 */
static isl_stat test_gist_fail(struct isl_ctx *ctx)
{
	int i, n;
	int on_error;

	on_error = isl_options_get_on_error(ctx);
	isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);
	n = ARRAY_SIZE(gist_fail_tests);
	for (i = 0; i < n; ++i) {
		const char *str;
		isl_basic_set *bset, *context;

		bset = isl_basic_set_read_from_str(ctx, gist_fail_tests[i].set);
		str = gist_fail_tests[i].context;
		context = isl_basic_set_read_from_str(ctx, str);
		bset = isl_basic_set_gist(bset, context);
		isl_basic_set_free(bset);
		if (bset)
			break;
	}
	isl_options_set_on_error(ctx, on_error);
	if (i < n)
		isl_die(ctx, isl_error_unknown,
			"operation not expected to succeed",
			return isl_stat_error);

	return isl_stat_ok;
}

struct {
	const char *set;
	const char *context;
	const char *gist;
} gist_tests[] = {
	{ "{ [1, -1, 3] }",
	  "{ [1, b, 2 - b] : -1 <= b <= 2 }",
	  "{ [a, -1, c] }" },
	{ "{ [a, b, c] : a <= 15 and a >= 1 }",
	  "{ [a, b, c] : exists (e0 = floor((-1 + a)/16): a >= 1 and "
			"c <= 30 and 32e0 >= -62 + 2a + 2b - c and b >= 0) }",
	  "{ [a, b, c] : a <= 15 }" },
	{ "{ : }", "{ : 1 = 0 }", "{ : }" },
	{ "{ : 1 = 0 }", "{ : 1 = 0 }", "{ : }" },
	{ "[M] -> { [x] : exists (e0 = floor((-2 + x)/3): 3e0 = -2 + x) }",
	  "[M] -> { [3M] }" , "[M] -> { [x] : 1 = 0 }" },
	{ "{ [m, n, a, b] : a <= 2147 + n }",
	  "{ [m, n, a, b] : (m >= 1 and n >= 1 and a <= 2148 - m and "
			"b <= 2148 - n and b >= 0 and b >= 2149 - n - a) or "
			"(n >= 1 and a >= 0 and b <= 2148 - n - a and "
			"b >= 0) }",
	  "{ [m, n, ku, kl] }" },
	{ "{ [a, a, b] : a >= 10 }",
	  "{ [a, b, c] : c >= a and c <= b and c >= 2 }",
	  "{ [a, a, b] : a >= 10 }" },
	{ "{ [i, j] : i >= 0 and i + j >= 0 }", "{ [i, j] : i <= 0 }",
	  "{ [0, j] : j >= 0 }" },
	/* Check that no constraints on i6 are introduced in the gist */
	{ "[t1] -> { [i4, i6] : exists (e0 = floor((1530 - 4t1 - 5i4)/20): "
		"20e0 <= 1530 - 4t1 - 5i4 and 20e0 >= 1511 - 4t1 - 5i4 and "
		"5e0 <= 381 - t1 and i4 <= 1) }",
	  "[t1] -> { [i4, i6] : exists (e0 = floor((-t1 + i6)/5): "
		"5e0 = -t1 + i6 and i6 <= 6 and i6 >= 3) }",
	  "[t1] -> { [i4, i6] : exists (e0 = floor((1530 - 4t1 - 5i4)/20): "
		"i4 <= 1 and 5e0 <= 381 - t1 and 20e0 <= 1530 - 4t1 - 5i4 and "
		"20e0 >= 1511 - 4t1 - 5i4) }" },
	/* Check that no constraints on i6 are introduced in the gist */
	{ "[t1, t2] -> { [i4, i5, i6] : exists (e0 = floor((1 + i4)/2), "
		"e1 = floor((1530 - 4t1 - 5i4)/20), "
		"e2 = floor((-4t1 - 5i4 + 10*floor((1 + i4)/2))/20), "
		"e3 = floor((-1 + i4)/2): t2 = 0 and 2e3 = -1 + i4 and "
			"20e2 >= -19 - 4t1 - 5i4 + 10e0 and 5e2 <= 1 - t1 and "
			"2e0 <= 1 + i4 and 2e0 >= i4 and "
			"20e1 <= 1530 - 4t1 - 5i4 and "
			"20e1 >= 1511 - 4t1 - 5i4 and i4 <= 1 and "
			"5e1 <= 381 - t1 and 20e2 <= -4t1 - 5i4 + 10e0) }",
	  "[t1, t2] -> { [i4, i5, i6] : exists (e0 = floor((-17 + i4)/2), "
		"e1 = floor((-t1 + i6)/5): 5e1 = -t1 + i6 and "
			"2e0 <= -17 + i4 and 2e0 >= -18 + i4 and "
			"10e0 <= -91 + 5i4 + 4i6 and "
			"10e0 >= -105 + 5i4 + 4i6) }",
	  "[t1, t2] -> { [i4, i5, i6] : exists (e0 = floor((381 - t1)/5), "
		"e1 = floor((-1 + i4)/2): t2 = 0 and 2e1 = -1 + i4 and "
		"i4 <= 1 and 5e0 <= 381 - t1 and 20e0 >= 1511 - 4t1 - 5i4) }" },
	{ "{ [0, 0, q, p] : -5 <= q <= 5 and p >= 0 }",
	  "{ [a, b, q, p] : b >= 1 + a }",
	  "{ [a, b, q, p] : false }" },
	{ "[n] -> { [x] : x = n && x mod 32 = 0 }",
	  "[n] -> { [x] : x mod 32 = 0 }",
	  "[n] -> { [x = n] }" },
	{ "{ [x] : x mod 6 = 0 }", "{ [x] : x mod 3 = 0 }",
	  "{ [x] : x mod 2 = 0 }" },
	{ "{ [x] : x mod 3200 = 0 }", "{ [x] : x mod 10000 = 0 }",
	  "{ [x] : x mod 128 = 0 }" },
	{ "{ [x] : x mod 3200 = 0 }", "{ [x] : x mod 10 = 0 }",
	  "{ [x] : x mod 3200 = 0 }" },
	{ "{ [a, b, c] : a mod 2 = 0 and a = c }",
	  "{ [a, b, c] : b mod 2 = 0 and b = c }",
	  "{ [a, b, c = a] }" },
	{ "{ [a, b, c] : a mod 6 = 0 and a = c }",
	  "{ [a, b, c] : b mod 2 = 0 and b = c }",
	  "{ [a, b, c = a] : a mod 3 = 0 }" },
	{ "{ [x] : 0 <= x <= 4 or 6 <= x <= 9 }",
	  "{ [x] : 1 <= x <= 3 or 7 <= x <= 8 }",
	  "{ [x] }" },
	{ "{ [x,y] : x < 0 and 0 <= y <= 4 or x >= -2 and -x <= y <= 10 + x }",
	  "{ [x,y] : 1 <= y <= 3 }",
	  "{ [x,y] }" },
};

/* Check that isl_set_gist behaves as expected.
 *
 * For the test cases in gist_tests, besides checking that the result
 * is as expected, also check that applying the gist operation does
 * not modify the input set (an earlier version of isl would do that) and
 * that the test case is consistent, i.e., that the gist has the same
 * intersection with the context as the input set.
 */
static int test_gist(struct isl_ctx *ctx)
{
	int i;
	const char *str;
	isl_basic_set *bset1, *bset2;
	isl_map *map1, *map2;
	isl_bool equal;
	isl_size n_div;

	for (i = 0; i < ARRAY_SIZE(gist_tests); ++i) {
		isl_bool equal_input, equal_intersection;
		isl_set *set1, *set2, *copy, *context;

		set1 = isl_set_read_from_str(ctx, gist_tests[i].set);
		context = isl_set_read_from_str(ctx, gist_tests[i].context);
		copy = isl_set_copy(set1);
		set1 = isl_set_gist(set1, isl_set_copy(context));
		set2 = isl_set_read_from_str(ctx, gist_tests[i].gist);
		equal = isl_set_is_equal(set1, set2);
		isl_set_free(set1);
		set1 = isl_set_read_from_str(ctx, gist_tests[i].set);
		equal_input = isl_set_is_equal(set1, copy);
		isl_set_free(copy);
		set1 = isl_set_intersect(set1, isl_set_copy(context));
		set2 = isl_set_intersect(set2, context);
		equal_intersection = isl_set_is_equal(set1, set2);
		isl_set_free(set2);
		isl_set_free(set1);
		if (equal < 0 || equal_input < 0 || equal_intersection < 0)
			return -1;
		if (!equal)
			isl_die(ctx, isl_error_unknown,
				"incorrect gist result", return -1);
		if (!equal_input)
			isl_die(ctx, isl_error_unknown,
				"gist modified input", return -1);
		if (!equal_input)
			isl_die(ctx, isl_error_unknown,
				"inconsistent gist test case", return -1);
	}

	if (test_gist_fail(ctx) < 0)
		return -1;

	str = "[p0, p2, p3, p5, p6, p10] -> { [] : "
	    "exists (e0 = [(15 + p0 + 15p6 + 15p10)/16], e1 = [(p5)/8], "
	    "e2 = [(p6)/128], e3 = [(8p2 - p5)/128], "
	    "e4 = [(128p3 - p6)/4096]: 8e1 = p5 and 128e2 = p6 and "
	    "128e3 = 8p2 - p5 and 4096e4 = 128p3 - p6 and p2 >= 0 and "
	    "16e0 >= 16 + 16p6 + 15p10 and  p2 <= 15 and p3 >= 0 and "
	    "p3 <= 31 and  p6 >= 128p3 and p5 >= 8p2 and p10 >= 0 and "
	    "16e0 <= 15 + p0 + 15p6 + 15p10 and 16e0 >= p0 + 15p6 + 15p10 and "
	    "p10 <= 15 and p10 <= -1 + p0 - p6) }";
	bset1 = isl_basic_set_read_from_str(ctx, str);
	str = "[p0, p2, p3, p5, p6, p10] -> { [] : exists (e0 = [(p5)/8], "
	    "e1 = [(p6)/128], e2 = [(8p2 - p5)/128], "
	    "e3 = [(128p3 - p6)/4096]: 8e0 = p5 and 128e1 = p6 and "
	    "128e2 = 8p2 - p5 and 4096e3 = 128p3 - p6 and p5 >= -7 and "
	    "p2 >= 0 and 8p2 <= -1 + p0 and p2 <= 15 and p3 >= 0 and "
	    "p3 <= 31 and 128p3 <= -1 + p0 and p6 >= -127 and "
	    "p5 <= -1 + p0 and p6 <= -1 + p0 and p6 >= 128p3 and "
	    "p0 >= 1 and p5 >= 8p2 and p10 >= 0 and p10 <= 15 ) }";
	bset2 = isl_basic_set_read_from_str(ctx, str);
	bset1 = isl_basic_set_gist(bset1, bset2);
	assert(bset1 && bset1->n_div == 0);
	isl_basic_set_free(bset1);

	/* Check that the integer divisions of the second disjunct
	 * do not spread to the first disjunct.
	 */
	str = "[t1] -> { S_0[] -> A[o0] : (exists (e0 = [(-t1 + o0)/16]: "
		"16e0 = -t1 + o0 and o0 >= 0 and o0 <= 15 and t1 >= 0)) or "
		"(exists (e0 = [(-1 + t1)/16], "
			"e1 = [(-16 + t1 - 16e0)/4294967296]: "
			"4294967296e1 = -16 + t1 - o0 - 16e0 and "
			"16e0 <= -1 + t1 and 16e0 >= -16 + t1 and o0 >= 0 and "
			"o0 <= 4294967295 and t1 <= -1)) }";
	map1 = isl_map_read_from_str(ctx, str);
	str = "[t1] -> { S_0[] -> A[o0] : t1 >= 0 and t1 <= 4294967295 }";
	map2 = isl_map_read_from_str(ctx, str);
	map1 = isl_map_gist(map1, map2);
	if (!map1)
		return -1;
	if (map1->n != 1)
		isl_die(ctx, isl_error_unknown, "expecting single disjunct",
			isl_map_free(map1); return -1);
	n_div = isl_basic_map_dim(map1->p[0], isl_dim_div);
	isl_map_free(map1);
	if (n_div < 0)
		return -1;
	if (n_div != 1)
		isl_die(ctx, isl_error_unknown, "expecting single div",
			return -1);

	if (test_gist_empty(ctx) < 0)
		return -1;
	if (test_plain_gist(ctx) < 0)
		return -1;

	return 0;
}

int test_coalesce_set(isl_ctx *ctx, const char *str, int check_one)
{
	isl_set *set, *set2;
	int equal;
	int one;

	set = isl_set_read_from_str(ctx, str);
	set = isl_set_coalesce(set);
	set2 = isl_set_read_from_str(ctx, str);
	equal = isl_set_is_equal(set, set2);
	one = set && set->n == 1;
	isl_set_free(set);
	isl_set_free(set2);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown,
			"coalesced set not equal to input", return -1);
	if (check_one && !one)
		isl_die(ctx, isl_error_unknown,
			"coalesced set should not be a union", return -1);

	return 0;
}

/* Inputs for coalescing tests with unbounded wrapping.
 * "str" is a string representation of the input set.
 * "single_disjunct" is set if we expect the result to consist of
 *	a single disjunct.
 */
struct {
	int single_disjunct;
	const char *str;
} coalesce_unbounded_tests[] = {
	{ 1, "{ [x,y,z] : y + 2 >= 0 and x - y + 1 >= 0 and "
			"-x - y + 1 >= 0 and -3 <= z <= 3;"
		"[x,y,z] : -x+z + 20 >= 0 and -x-z + 20 >= 0 and "
			"x-z + 20 >= 0 and x+z + 20 >= 0 and "
			"-10 <= y <= 0}" },
	{ 1, "{ [x,y] : 0 <= x,y <= 10; [5,y]: 4 <= y <= 11 }" },
	{ 1, "{ [x,0,0] : -5 <= x <= 5; [0,y,1] : -5 <= y <= 5 }" },
	{ 1, "{ [x,y] : 0 <= x <= 10 and 0 >= y >= -1 and x+y >= 0; [0,1] }" },
	{ 1, "{ [x,y] : (0 <= x,y <= 4) or (2 <= x,y <= 5 and x + y <= 9) }" },
	{ 0, "{ [x, y, z] : 0 <= x,y,z <= 100 and 0 < z <= 2 + 2x + 2y; "
		"[x, y, 0] : x,y <= 100 and y <= 9 + 11x and x <= 9 + 11y }" },
	{ 1, "{ [0:1, 0:1]; [0, 2:3] }" },
	{ 1, "{ [0:1, 0:1]; [0, 2:3]; [1, -2:-1] }" },
	{ 1, "{ [0:3, 0:1]; [1:2, 2:5] }" },
	{ 1, "{ [0:3, 0:1]; [0:2, 2:5] }" },
	{ 1, "{ [0:3, 0:1]; [1:3, 2:5] }" },
	{ 0, "{ [0:3, 0:1]; [1:4, 2:5] }" },
	{ 0, "{ [0:3, 0:1]; [1:5, 2:5] }" },
};

/* Test the functionality of isl_set_coalesce with the bounded wrapping
 * option turned off.
 */
int test_coalesce_unbounded_wrapping(isl_ctx *ctx)
{
	int i;
	int r = 0;
	int bounded;

	bounded = isl_options_get_coalesce_bounded_wrapping(ctx);
	isl_options_set_coalesce_bounded_wrapping(ctx, 0);

	for (i = 0; i < ARRAY_SIZE(coalesce_unbounded_tests); ++i) {
		const char *str = coalesce_unbounded_tests[i].str;
		int check_one = coalesce_unbounded_tests[i].single_disjunct;
		if (test_coalesce_set(ctx, str, check_one) >= 0)
			continue;
		r = -1;
		break;
	}

	isl_options_set_coalesce_bounded_wrapping(ctx, bounded);

	return r;
}

/* Inputs for coalescing tests.
 * "str" is a string representation of the input set.
 * "single_disjunct" is set if we expect the result to consist of
 *	a single disjunct.
 */
struct {
	int single_disjunct;
	const char *str;
} coalesce_tests[] = {
	{ 1, "{[x,y]: x >= 0 & x <= 10 & y >= 0 & y <= 10 or "
		       "y >= x & x >= 2 & 5 >= y }" },
	{ 1, "{[x,y]: y >= 0 & 2x + y <= 30 & y <= 10 & x >= 0 or "
		       "x + y >= 10 & y <= x & x + y <= 20 & y >= 0}" },
	{ 0, "{[x,y]: y >= 0 & 2x + y <= 30 & y <= 10 & x >= 0 or "
		       "x + y >= 10 & y <= x & x + y <= 19 & y >= 0}" },
	{ 1, "{[x,y]: y >= 0 & x <= 5 & y <= x or "
		       "y >= 0 & x >= 6 & x <= 10 & y <= x}" },
	{ 0, "{[x,y]: y >= 0 & x <= 5 & y <= x or "
		       "y >= 0 & x >= 7 & x <= 10 & y <= x}" },
	{ 0, "{[x,y]: y >= 0 & x <= 5 & y <= x or "
		       "y >= 0 & x >= 6 & x <= 10 & y + 1 <= x}" },
	{ 1, "{[x,y]: y >= 0 & x <= 5 & y <= x or y >= 0 & x = 6 & y <= 6}" },
	{ 0, "{[x,y]: y >= 0 & x <= 5 & y <= x or y >= 0 & x = 7 & y <= 6}" },
	{ 1, "{[x,y]: y >= 0 & x <= 5 & y <= x or y >= 0 & x = 6 & y <= 5}" },
	{ 0, "{[x,y]: y >= 0 & x <= 5 & y <= x or y >= 0 & x = 6 & y <= 7}" },
	{ 1, "[n] -> { [i] : i = 1 and n >= 2 or 2 <= i and i <= n }" },
	{ 0, "{[x,y] : x >= 0 and y >= 0 or 0 <= y and y <= 5 and x = -1}" },
	{ 1, "[n] -> { [i] : 1 <= i and i <= n - 1 or 2 <= i and i <= n }" },
	{ 0, "[n] -> { [[i0] -> [o0]] : exists (e0 = [(i0)/4], e1 = [(o0)/4], "
		"e2 = [(n)/2], e3 = [(-2 + i0)/4], e4 = [(-2 + o0)/4], "
		"e5 = [(-2n + i0)/4]: 2e2 = n and 4e3 = -2 + i0 and "
		"4e4 = -2 + o0 and i0 >= 8 + 2n and o0 >= 2 + i0 and "
		"o0 <= 56 + 2n and o0 <= -12 + 4n and i0 <= 57 + 2n and "
		"i0 <= -11 + 4n and o0 >= 6 + 2n and 4e0 <= i0 and "
		"4e0 >= -3 + i0 and 4e1 <= o0 and 4e1 >= -3 + o0 and "
		"4e5 <= -2n + i0 and 4e5 >= -3 - 2n + i0);"
		"[[i0] -> [o0]] : exists (e0 = [(i0)/4], e1 = [(o0)/4], "
		"e2 = [(n)/2], e3 = [(-2 + i0)/4], e4 = [(-2 + o0)/4], "
		"e5 = [(-2n + i0)/4]: 2e2 = n and 4e3 = -2 + i0 and "
		"4e4 = -2 + o0 and 2e0 >= 3 + n and e0 <= -4 + n and "
		"2e0 <= 27 + n and e1 <= -4 + n and 2e1 <= 27 + n and "
		"2e1 >= 2 + n and e1 >= 1 + e0 and i0 >= 7 + 2n and "
		"i0 <= -11 + 4n and i0 <= 57 + 2n and 4e0 <= -2 + i0 and "
		"4e0 >= -3 + i0 and o0 >= 6 + 2n and o0 <= -11 + 4n and "
		"o0 <= 57 + 2n and 4e1 <= -2 + o0 and 4e1 >= -3 + o0 and "
		"4e5 <= -2n + i0 and 4e5 >= -3 - 2n + i0 ) }" },
	{ 0, "[n, m] -> { [o0, o2, o3] : (o3 = 1 and o0 >= 1 + m and "
	      "o0 <= n + m and o2 <= m and o0 >= 2 + n and o2 >= 3) or "
	      "(o0 >= 2 + n and o0 >= 1 + m and o0 <= n + m and n >= 1 and "
	      "o3 <= -1 + o2 and o3 >= 1 - m + o2 and o3 >= 2 and o3 <= n) }" },
	{ 0, "[M, N] -> { [[i0, i1, i2, i3, i4, i5, i6] -> "
	  "[o0, o1, o2, o3, o4, o5, o6]] : "
	  "(o6 <= -4 + 2M - 2N + i0 + i1 - i2 + i6 - o0 - o1 + o2 and "
	  "o3 <= -2 + i3 and o6 >= 2 + i0 + i3 + i6 - o0 - o3 and "
	  "o6 >= 2 - M + N + i3 + i4 + i6 - o3 - o4 and o0 <= -1 + i0 and "
	  "o4 >= 4 - 3M + 3N - i0 - i1 + i2 + 2i3 + i4 + o0 + o1 - o2 - 2o3 "
	  "and o6 <= -3 + 2M - 2N + i3 + i4 - i5 + i6 - o3 - o4 + o5 and "
	  "2o6 <= -5 + 5M - 5N + 2i0 + i1 - i2 - i5 + 2i6 - 2o0 - o1 + o2 + o5 "
	  "and o6 >= 2i0 + i1 + i6 - 2o0 - o1 and "
	  "3o6 <= -5 + 4M - 4N + 2i0 + i1 - i2 + 2i3 + i4 - i5 + 3i6 "
	  "- 2o0 - o1 + o2 - 2o3 - o4 + o5) or "
	  "(N >= 2 and o3 <= -1 + i3 and o0 <= -1 + i0 and "
	  "o6 >= i3 + i6 - o3 and M >= 0 and "
	  "2o6 >= 1 + i0 + i3 + 2i6 - o0 - o3 and "
	  "o6 >= 1 - M + i0 + i6 - o0 and N >= 2M and o6 >= i0 + i6 - o0) }" },
	{ 0, "[M, N] -> { [o0] : (o0 = 0 and M >= 1 and N >= 2) or "
		"(o0 = 0 and M >= 1 and N >= 2M and N >= 2 + M) or "
		"(o0 = 0 and M >= 2 and N >= 3) or "
		"(M = 0 and o0 = 0 and N >= 3) }" },
	{ 0, "{ [i0, i1, i2, i3] : (i1 = 10i0 and i0 >= 1 and 10i0 <= 100 and "
	    "i3 <= 9 + 10 i2 and i3 >= 1 + 10i2 and i3 >= 0) or "
	    "(i1 <= 9 + 10i0 and i1 >= 1 + 10i0 and i2 >= 0 and "
	    "i0 >= 0 and i1 <= 100 and i3 <= 9 + 10i2 and i3 >= 1 + 10i2) }" },
	{ 0, "[M] -> { [i1] : (i1 >= 2 and i1 <= M) or (i1 = M and M >= 1) }" },
	{ 0, "{[x,y] : x,y >= 0; [x,y] : 10 <= x <= 20 and y >= -1 }" },
	{ 1, "{ [x, y] : (x >= 1 and y >= 1 and x <= 2 and y <= 2) or "
		"(y = 3 and x = 1) }" },
	{ 1, "[M] -> { [i0, i1, i2, i3, i4] : (i1 >= 3 and i4 >= 2 + i2 and "
		"i2 >= 2 and i0 >= 2 and i3 >= 1 + i2 and i0 <= M and "
		"i1 <= M and i3 <= M and i4 <= M) or "
		"(i1 >= 2 and i4 >= 1 + i2 and i2 >= 2 and i0 >= 2 and "
		"i3 >= 1 + i2 and i0 <= M and i1 <= -1 + M and i3 <= M and "
		"i4 <= -1 + M) }" },
	{ 1, "{ [x, y] : (x >= 0 and y >= 0 and x <= 10 and y <= 10) or "
		"(x >= 1 and y >= 1 and x <= 11 and y <= 11) }" },
	{ 0, "{[x,0] : x >= 0; [x,1] : x <= 20}" },
	{ 1, "{ [x, 1 - x] : 0 <= x <= 1; [0,0] }" },
	{ 1, "{ [0,0]; [i,i] : 1 <= i <= 10 }" },
	{ 0, "{ [0,0]; [i,j] : 1 <= i,j <= 10 }" },
	{ 1, "{ [0,0]; [i,2i] : 1 <= i <= 10 }" },
	{ 0, "{ [0,0]; [i,2i] : 2 <= i <= 10 }" },
	{ 0, "{ [1,0]; [i,2i] : 1 <= i <= 10 }" },
	{ 0, "{ [0,1]; [i,2i] : 1 <= i <= 10 }" },
	{ 0, "{ [a, b] : exists e : 2e = a and "
		    "a >= 0 and (a <= 3 or (b <= 0 and b >= -4 + a)) }" },
	{ 0, "{ [i, j, i', j'] : i <= 2 and j <= 2 and "
			"j' >= -1 + 2i + j - 2i' and i' <= -1 + i and "
			"j >= 1 and j' <= i + j - i' and i >= 1; "
		"[1, 1, 1, 1] }" },
	{ 1, "{ [i,j] : exists a,b : i = 2a and j = 3b; "
		 "[i,j] : exists a : j = 3a }" },
	{ 1, "{ [a, b, c] : (c <= 7 - b and b <= 1 and b >= 0 and "
			"c >= 3 + b and b <= 3 + 8a and b >= -26 + 8a and "
			"a >= 3) or "
		    "(b <= 1 and c <= 7 and b >= 0 and c >= 4 + b and "
			"b <= 3 + 8a and b >= -26 + 8a and a >= 3) }" },
	{ 1, "{ [a, 0, c] : c >= 1 and c <= 29 and c >= -1 + 8a and "
				"c <= 6 + 8a and a >= 3; "
		"[a, -1, c] : c >= 1 and c <= 30 and c >= 8a and "
				"c <= 7 + 8a and a >= 3 and a <= 4 }" },
	{ 1, "{ [x,y] : 0 <= x <= 2 and y >= 0 and x + 2y <= 4; "
		"[x,0] : 3 <= x <= 4 }" },
	{ 1, "{ [x,y] : 0 <= x <= 3 and y >= 0 and x + 3y <= 6; "
		"[x,0] : 4 <= x <= 5 }" },
	{ 0, "{ [x,y] : 0 <= x <= 2 and y >= 0 and x + 2y <= 4; "
		"[x,0] : 3 <= x <= 5 }" },
	{ 0, "{ [x,y] : 0 <= x <= 2 and y >= 0 and x + y <= 4; "
		"[x,0] : 3 <= x <= 4 }" },
	{ 1, "{ [i0, i1] : i0 <= 122 and i0 >= 1 and 128i1 >= -249 + i0 and "
			"i1 <= 0; "
		"[i0, 0] : i0 >= 123 and i0 <= 124 }" },
	{ 1, "{ [0,0]; [1,1] }" },
	{ 1, "[n] -> { [k] : 16k <= -1 + n and k >= 1; [0] : n >= 2 }" },
	{ 1, "{ [k, ii, k - ii] : ii >= -6 + k and ii <= 6 and ii >= 1 and "
				"ii <= k;"
		"[k, 0, k] : k <= 6 and k >= 1 }" },
	{ 1, "{ [i,j] : i = 4 j and 0 <= i <= 100;"
		"[i,j] : 1 <= i <= 100 and i >= 4j + 1 and i <= 4j + 2 }" },
	{ 1, "{ [x,y] : x % 2 = 0 and y % 2 = 0; [x,x] : x % 2 = 0 }" },
	{ 1, "[n] -> { [1] : n >= 0;"
		    "[x] : exists (e0 = floor((x)/2): x >= 2 and "
			"2e0 >= -1 + x and 2e0 <= x and 2e0 <= n) }" },
	{ 1, "[n] -> { [x, y] : exists (e0 = floor((x)/2), e1 = floor((y)/3): "
			"3e1 = y and x >= 2 and 2e0 >= -1 + x and "
			"2e0 <= x and 2e0 <= n);"
		    "[1, y] : exists (e0 = floor((y)/3): 3e0 = y and "
			"n >= 0) }" },
	{ 1, "[t1] -> { [i0] : (exists (e0 = floor((63t1)/64): "
				"128e0 >= -134 + 127t1 and t1 >= 2 and "
				"64e0 <= 63t1 and 64e0 >= -63 + 63t1)) or "
				"t1 = 1 }" },
	{ 1, "{ [i, i] : exists (e0 = floor((1 + 2i)/3): 3e0 <= 2i and "
				"3e0 >= -1 + 2i and i <= 9 and i >= 1);"
		"[0, 0] }" },
	{ 1, "{ [t1] : exists (e0 = floor((-11 + t1)/2): 2e0 = -11 + t1 and "
				"t1 >= 13 and t1 <= 16);"
		"[t1] : t1 <= 15 and t1 >= 12 }" },
	{ 1, "{ [x,y] : x = 3y and 0 <= y <= 2; [-3,-1] }" },
	{ 1, "{ [x,y] : 2x = 3y and 0 <= y <= 4; [-3,-2] }" },
	{ 0, "{ [x,y] : 2x = 3y and 0 <= y <= 4; [-2,-2] }" },
	{ 0, "{ [x,y] : 2x = 3y and 0 <= y <= 4; [-3,-1] }" },
	{ 1, "{ [i] : exists j : i = 4 j and 0 <= i <= 100;"
		"[i] : exists j : 1 <= i <= 100 and i >= 4j + 1 and "
				"i <= 4j + 2 }" },
	{ 1, "{ [c0] : (exists (e0 : c0 - 1 <= 3e0 <= c0)) or "
		"(exists (e0 : 3e0 = -2 + c0)) }" },
	{ 0, "[n, b0, t0] -> "
		"{ [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12] : "
		"(exists (e0 = floor((-32b0 + i4)/1048576), "
		"e1 = floor((i8)/32): 1048576e0 = -32b0 + i4 and 32e1 = i8 and "
		"n <= 2147483647 and b0 <= 32767 and b0 >= 0 and "
		"32b0 <= -2 + n and t0 <= 31 and t0 >= 0 and i0 >= 8 + n and "
		"3i4 <= -96 + 3t0 + i0 and 3i4 >= -95 - n + 3t0 + i0 and "
		"i8 >= -157 + i0 - 4i4 and i8 >= 0 and "
		"i8 <= -33 + i0 - 4i4 and 3i8 <= -91 + 4n - i0)) or "
		"(exists (e0 = floor((-32b0 + i4)/1048576), "
		"e1 = floor((i8)/32): 1048576e0 = -32b0 + i4 and 32e1 = i8 and "
		"n <= 2147483647 and b0 <= 32767 and b0 >= 0 and "
		"32b0 <= -2 + n and t0 <= 31 and t0 >= 0 and i0 <= 7 + n and "
		"4i4 <= -3 + i0 and 3i4 <= -96 + 3t0 + i0 and "
		"3i4 >= -95 - n + 3t0 + i0 and i8 >= -157 + i0 - 4i4 and "
		"i8 >= 0 and i8 <= -4 + i0 - 3i4 and i8 <= -41 + i0));"
		"[i0, i1, i2, i3, 0, i5, i6, i7, i8, i9, i10, i11, i12] : "
		"(exists (e0 = floor((i8)/32): b0 = 0 and 32e0 = i8 and "
		"n <= 2147483647 and t0 <= 31 and t0 >= 0 and i0 >= 11 and "
		"i0 >= 96 - 3t0 and i0 <= 95 + n - 3t0 and i0 <= 7 + n and "
		"i8 >= -40 + i0 and i8 <= -10 + i0)) }" },
	{ 0, "{ [i0, i1, i2] : "
		"(exists (e0, e1 = floor((i0)/32), e2 = floor((i1)/32): "
		"32e1 = i0 and 32e2 = i1 and i1 >= -31 + i0 and "
		"i1 <= 31 + i0 and i2 >= -30 + i0 and i2 >= -30 + i1 and "
		"32e0 >= -30 + i0 and 32e0 >= -30 + i1 and "
		"32e0 >= -31 + i2 and 32e0 <= 30 + i2 and 32e0 <= 31 + i1 and "
		"32e0 <= 31 + i0)) or "
		"i0 >= 0 }" },
	{ 1, "{ [a, b, c] : 2b = 1 + a and 2c = 2 + a; [0, 0, 0] }" },
	{ 1, "{ [a, a, b, c] : 32*floor((a)/32) = a and 2*floor((b)/2) = b and "
				"2*floor((c)/2) = c and 0 <= a <= 192;"
		"[224, 224, b, c] : 2*floor((b)/2) = b and 2*floor((c)/2) = c }"
	},
	{ 1, "[n] -> { [a,b] : (exists e : 1 <= a <= 7e and 9e <= b <= n) or "
				"(0 <= a <= b <= n) }" },
	{ 1, "{ [a, b] : 0 <= a <= 2 and b >= 0 and "
		"((0 < b <= 13) or (2*floor((a + b)/2) >= -5 + a + 2b)) }" },
	{ 1, "{ [a] : (2 <= a <= 5) or (a mod 2 = 1 and 1 <= a <= 5) }" },
	{ 1, "{ [a, b, c] : (b = -1 + a and 0 < a <= 3 and "
				"9*floor((-4a + 2c)/9) <= -3 - 4a + 2c) or "
			"(exists (e0 = floor((-16 + 2c)/9): a = 4 and "
				"b = 3 and 9e0 <= -19 + 2c)) }" },
	{ 1, "{ [a, b, c] : (b = -1 + a and 0 < a <= 3 and "
				"9*floor((-4a + 2c)/9) <= -3 - 4a + 2c) or "
			"(a = 4 and b = 3 and "
				"9*floor((-16 + 2c)/9) <= -19 + 2c) }" },
	{ 0, "{ [a, b, c] : (b <= 2 and b <= -2 + a) or "
			"(b = -1 + a and 0 < a <= 3 and "
				"9*floor((-4a + 2c)/9) <= -3 - 4a + 2c) or "
			"(exists (e0 = floor((-16 + 2c)/9): a = 4 and "
				"b = 3 and 9e0 <= -19 + 2c)) }" },
	{ 1, "{ [y, x] : (x - y) mod 3 = 2 and 2 <= y <= 200 and 0 <= x <= 2;"
		"[1, 0] }" },
	{ 1, "{ [x, y] : (x - y) mod 3 = 2 and 2 <= y <= 200 and 0 <= x <= 2;"
		"[0, 1] }" },
	{ 1, "{ [1, y] : -1 <= y <= 1; [x, -x] : 0 <= x <= 1 }" },
	{ 1, "{ [1, y] : 0 <= y <= 1; [x, -x] : 0 <= x <= 1 }" },
	{ 1, "{ [x, y] : 0 <= x <= 10 and x - 4*floor(x/4) <= 1 and y <= 0; "
	       "[x, y] : 0 <= x <= 10 and x - 4*floor(x/4) > 1 and y <= 0; "
	       "[x, y] : 0 <= x <= 10 and x - 5*floor(x/5) <= 1 and 0 < y; "
	       "[x, y] : 0 <= x <= 10 and x - 5*floor(x/5) > 1 and 0 < y }" },
	{ 1, "{ [x, 0] : 0 <= x <= 10 and x mod 2 = 0; "
	       "[x, 0] : 0 <= x <= 10 and x mod 2 = 1; "
	       "[x, y] : 0 <= x <= 10 and 1 <= y <= 10 }" },
	{ 1, "{ [a] : a <= 8 and "
			"(a mod 10 = 7 or a mod 10 = 8 or a mod 10 = 9) }" },
	{ 1, "{ [x, y] : 2y = -x and x <= 0 or "
			"x <= -1 and 2y <= -x - 1 and 2y >= x - 1 }" },
	{ 0, "{ [x, y] : 2y = -x and x <= 0 or "
			"x <= -2 and 2y <= -x - 1 and 2y >= x - 1 }" },
	{ 1, "{ [a] : (a <= 0 and 3*floor((a)/3) = a) or "
			"(a < 0 and 3*floor((a)/3) < a) }" },
	{ 1, "{ [a] : (a <= 0 and 3*floor((a)/3) = a) or "
			"(a < -1 and 3*floor((a)/3) < a) }" },
	{ 1, "{ [a, b] : a <= 1024 and b >= 0 and "
		"((-31 - a + b <= 32*floor((-1 - a)/32) <= -33 + b and "
		  "32*floor((-1 - a)/32) <= -16 + b + 16*floor((-1 - a)/16))"
		"or (2 <= a <= 15 and b < a)) }" },
	{ 1, "{ [a] : a > 0 and ((16*floor((a)/16) < a and "
			"32*floor((a)/32) < a) or a <= 15) }" },
	{ 1, "{ [a, b, c, d] : (-a + d) mod 64 = 0 and a <= 8 and b <= 1 and "
			"10 - a <= c <= 3 and d >= 5 and 9 - 64b <= d <= 70;"
	    "[a, b = 1, c, d] : (-a + d) mod 64 = 0 and a <= 8 and c >= 4 and "
			"10 - a <= c <= 5 and 5 <= d <= 73 - c }" },
	{ 1, "[n, m] -> { S_0[i] : (-n + i) mod 3 = 0 and m >= 3 + n and "
			    "i >= n and 3*floor((2 + n + 2m)/3) <= n + 3m - i; "
			 "S_0[n] : n <= m <= 2 + n }" },
	{ 1, "{ [a, b] : exists (e0: 0 <= a <= 1 and b >= 0 and "
			"2e0 >= -5 + a + 2b and 2e0 >= -1 + a + b and "
			"2e0 <= a + b); "
		"[a, b] : exists (e0: 0 <= a <= 1 and 2e0 >= -5 + a + 2b and "
			"2e0 >= -1 - a + b and 2e0 <= -a + b and "
			"2e0 < -a + 2b) }" },
	{ 1, "{ [i, j, i - 8j] : 8 <= i <= 63 and -7 + i <= 8j <= i; "
		"[i, 0, i] : 0 <= i <= 7 }" },
	{ 1, "{ [a, b] : a >= 0 and 0 <= b <= 1 - a; [1, 1] }" },
	{ 0, "{ [a, b] : a >= 0 and 0 <= b <= 1 - a; [0, 2] }" },
	{ 0, "{ [a, b] : a >= 0 and 0 <= b <= 1 - a; [-1, 3] }" },
	{ 1, "{ [a, b] : a, b >= 0 and a + 2b <= 2; [1, 1] }" },
	{ 0, "{ [a, b] : a, b >= 0 and a + 2b <= 2; [2, 1] }" },
	{ 0, "{ [a, c] : (2 + a) mod 4 = 0 or "
		"(c = 4 + a and 4 * floor((a)/4) = a and a >= 0 and a <= 4) or "
		"(c = 3 + a and 4 * floor((-1 + a)/4) = -1 + a and "
		    "a > 0 and a <= 5) }" },
	{ 1, "{ [1, 0, 0]; [a, b, c] : -1 <= -a < b <= 0 and 2c > b }" },
	{ 0, "{ [j, a, l] : a mod 2 = 0 and j <= 29 and a >= 2 and "
			"2a <= -5 + j and 32j + 2a + 2 <= 4l < 33j; "
		"[j, 0, l] : 4 <= j <= 29 and -3 + 33j <= 4l <= 33j }" },
	{ 0, "{ [0:1, 0:1]; [0, 2:3] }" },
	{ 1, "{ [a] : (a = 0 or ((1 + a) mod 2 = 0 and 0 < a <= 15) or "
		"((a) mod 2 = 0 and 0 < a <= 15)) }" },
};

/* A specialized coalescing test case that would result
 * in a segmentation fault or a failed assertion in earlier versions of isl.
 */
static int test_coalesce_special(struct isl_ctx *ctx)
{
	const char *str;
	isl_map *map1, *map2;

	str = "[y] -> { [S_L220_OUT[] -> T7[]] -> "
	    "[[S_L309_IN[] -> T11[]] -> ce_imag2[1, o1]] : "
	    "(y = 201 and o1 <= 239 and o1 >= 212) or "
	    "(exists (e0 = [(y)/3]: 3e0 = y and y <= 198 and y >= 3 and "
		"o1 <= 239 and o1 >= 212)) or "
	    "(exists (e0 = [(y)/3]: 3e0 = y and y <= 201 and y >= 3 and "
		"o1 <= 241 and o1 >= 240));"
	    "[S_L220_OUT[] -> T7[]] -> "
	    "[[S_L309_IN[] -> T11[]] -> ce_imag2[0, o1]] : "
	    "(y = 2 and o1 <= 241 and o1 >= 212) or "
	    "(exists (e0 = [(-2 + y)/3]: 3e0 = -2 + y and y <= 200 and "
		"y >= 5 and o1 <= 241 and o1 >= 212)) }";
	map1 = isl_map_read_from_str(ctx, str);
	map1 = isl_map_align_divs_internal(map1);
	map1 = isl_map_coalesce(map1);
	str = "[y] -> { [S_L220_OUT[] -> T7[]] -> "
	    "[[S_L309_IN[] -> T11[]] -> ce_imag2[o0, o1]] : "
	    "exists (e0 = [(-1 - y + o0)/3]: 3e0 = -1 - y + o0 and "
		"y <= 201 and o0 <= 2 and o1 >= 212 and o1 <= 241 and "
		"o0 >= 3 - y and o0 <= -2 + y and o0 >= 0) }";
	map2 = isl_map_read_from_str(ctx, str);
	map2 = isl_map_union(map2, map1);
	map2 = isl_map_align_divs_internal(map2);
	map2 = isl_map_coalesce(map2);
	isl_map_free(map2);
	if (!map2)
		return -1;

	return 0;
}

/* Check that the union of the basic sets described by "str1" and "str2"
 * can be coalesced.
 * The explicit call to isl_basic_set_union prevents the implicit
 * equality constraints in the basic maps from being detected prior
 * to the call to isl_set_coalesce, at least at the point
 * where this function was introduced.
 */
static isl_stat test_coalesce_union(isl_ctx *ctx, const char *str1,
	const char *str2)
{
	isl_basic_set *bset1, *bset2;
	isl_set *set;

	bset1 = isl_basic_set_read_from_str(ctx, str1);
	bset2 = isl_basic_set_read_from_str(ctx, str2);
	set = isl_basic_set_union(bset1, bset2);
	set = isl_set_coalesce(set);
	isl_set_free(set);

	return isl_stat_non_null(set);
}

/* A specialized coalescing test case that would result in an assertion
 * in an earlier version of isl.  Use test_coalesce_union with
 * an explicit call to isl_basic_set_union to prevent the implicit
 * equality constraints in the first basic map from being detected prior
 * to the call to isl_set_coalesce, at least at the point
 * where this test case was introduced.
 */
static isl_stat test_coalesce_special2(struct isl_ctx *ctx)
{
	const char *str1;
	const char *str2;

	str1 = "{ [x, y] : x, y >= 0 and x + 2y <= 1 and 2x + y <= 1 }";
	str2 = "{ [x,0] : -1 <= x <= 1 and x mod 2 = 1 }";
	return test_coalesce_union(ctx, str1, str2);
}

/* Check that calling isl_set_coalesce does not leave other sets
 * that may share some information with the input to isl_set_coalesce
 * in an inconsistent state.
 * In particular, older versions of isl would modify all copies
 * of the basic sets in the isl_set_coalesce input in a way
 * that could leave them in an inconsistent state.
 * The result of printing any other set containing one of these
 * basic sets would then result in an invalid set description.
 */
static int test_coalesce_special3(isl_ctx *ctx)
{
	const char *str;
	char *s;
	isl_set *set1, *set2;
	isl_printer *p;

	set1 = isl_set_read_from_str(ctx, "{ [0, 0, 0] }");
	str = "{ [a, b, a + b] : a >= 0 and b >= 0 and 0 < a + b }";
	set2 = isl_set_read_from_str(ctx, str);
	set1 = isl_set_union(set1, isl_set_copy(set2));
	set1 = isl_set_coalesce(set1);
	isl_set_free(set1);

	p = isl_printer_to_str(ctx);
	p = isl_printer_print_set(p, set2);
	isl_set_free(set2);
	s = isl_printer_get_str(p);
	isl_printer_free(p);
	set1 = isl_set_read_from_str(ctx, s);
	free(s);
	isl_set_free(set1);

	if (!set1)
		return -1;

	return 0;
}

/* Check that calling isl_set_coalesce on the intersection of
 * the sets described by "s1" and "s2" does not leave other sets
 * that may share some information with the input to isl_set_coalesce
 * in an inconsistent state.
 * In particular, when isl_set_coalesce detects equality constraints,
 * it does not immediately perform Gaussian elimination on them,
 * but then it needs to ensure that it is performed at some point.
 * The input set has implicit equality constraints in the first disjunct.
 * It is constructed as an intersection, because otherwise
 * those equality constraints would already be detected during parsing.
 */
static isl_stat test_coalesce_intersection(isl_ctx *ctx,
	const char *s1, const char *s2)
{
	isl_set *set1, *set2;

	set1 = isl_set_read_from_str(ctx, s1);
	set2 = isl_set_read_from_str(ctx, s2);
	set1 = isl_set_intersect(set1, set2);
	isl_set_free(isl_set_coalesce(isl_set_copy(set1)));
	set1 = isl_set_coalesce(set1);
	isl_set_free(set1);

	if (!set1)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Check that calling isl_set_coalesce does not leave other sets
 * that may share some information with the input to isl_set_coalesce
 * in an inconsistent state, for the case where one disjunct
 * is a subset of the other.
 */
static isl_stat test_coalesce_special4(isl_ctx *ctx)
{
	const char *s1, *s2;

	s1 = "{ [a, b] : b <= 0 or a <= 1 }";
	s2 = "{ [a, b] : -1 <= -a < b }";
	return test_coalesce_intersection(ctx, s1, s2);
}

/* Check that calling isl_set_coalesce does not leave other sets
 * that may share some information with the input to isl_set_coalesce
 * in an inconsistent state, for the case where two disjuncts
 * can be fused.
 */
static isl_stat test_coalesce_special5(isl_ctx *ctx)
{
	const char *s1, *s2;

	s1 = "{ [a, b, c] : b <= 0 }";
	s2 = "{ [a, b, c] : -1 <= -a < b and (c >= 0 or c < 0) }";
	return test_coalesce_intersection(ctx, s1, s2);
}

/* Check that calling isl_set_coalesce does not leave other sets
 * that may share some information with the input to isl_set_coalesce
 * in an inconsistent state, for the case where two disjuncts
 * can be fused and where both disjuncts have implicit equality constraints.
 */
static isl_stat test_coalesce_special6(isl_ctx *ctx)
{
	const char *s1, *s2;

	s1 = "{ [a, b, c] : c <= 0 }";
	s2 = "{ [a, b, c] : 0 <= a <= b <= c or (0 <= b <= c and a > 0) }";
	return test_coalesce_intersection(ctx, s1, s2);
}

/* A specialized coalescing test case that would result in an assertion failure
 * in an earlier version of isl.  Use test_coalesce_union with
 * an explicit call to isl_basic_set_union to prevent the implicit
 * equality constraints in the basic maps from being detected prior
 * to the call to isl_set_coalesce, at least at the point
 * where this test case was introduced.
 */
static isl_stat test_coalesce_special7(isl_ctx *ctx)
{
	const char *str1;
	const char *str2;

	str1 = "{ [a, b, c=0:17] : a <= 7 and 2b <= 11 - a and "
			"c <= -7 + 2a and 2c >= - 3 + 3a - 2b }";
	str2 = "{ [a, b, c] : c > -15a and c >= -7 + 2a and c < 0 and "
			"3c <= -5 + 5a - 3b and 2b >= 11 - a }";
	return test_coalesce_union(ctx, str1, str2);
}

/* Test the functionality of isl_set_coalesce.
 * That is, check that the output is always equal to the input
 * and in some cases that the result consists of a single disjunct.
 */
static int test_coalesce(struct isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(coalesce_tests); ++i) {
		const char *str = coalesce_tests[i].str;
		int check_one = coalesce_tests[i].single_disjunct;
		if (test_coalesce_set(ctx, str, check_one) < 0)
			return -1;
	}

	if (test_coalesce_unbounded_wrapping(ctx) < 0)
		return -1;
	if (test_coalesce_special(ctx) < 0)
		return -1;
	if (test_coalesce_special2(ctx) < 0)
		return -1;
	if (test_coalesce_special3(ctx) < 0)
		return -1;
	if (test_coalesce_special4(ctx) < 0)
		return -1;
	if (test_coalesce_special5(ctx) < 0)
		return -1;
	if (test_coalesce_special6(ctx) < 0)
		return -1;
	if (test_coalesce_special7(ctx) < 0)
		return -1;


	return 0;
}

/* Construct a representation of the graph on the right of Figure 1
 * in "Computing the Transitive Closure of a Union of
 * Affine Integer Tuple Relations".
 */
static __isl_give isl_map *cocoa_fig_1_right_graph(isl_ctx *ctx)
{
	isl_set *dom;
	isl_map *up, *right;

	dom = isl_set_read_from_str(ctx,
		"{ [x,y] : x >= 0 and -2 x + 3 y >= 0 and x <= 3 and "
			"2 x - 3 y + 3 >= 0 }");
	right = isl_map_read_from_str(ctx,
		"{ [x,y] -> [x2,y2] : x2 = x + 1 and y2 = y }");
	up = isl_map_read_from_str(ctx,
		"{ [x,y] -> [x2,y2] : x2 = x and y2 = y + 1 }");
	right = isl_map_intersect_domain(right, isl_set_copy(dom));
	right = isl_map_intersect_range(right, isl_set_copy(dom));
	up = isl_map_intersect_domain(up, isl_set_copy(dom));
	up = isl_map_intersect_range(up, dom);
	return isl_map_union(up, right);
}

/* Construct a representation of the power of the graph
 * on the right of Figure 1 in "Computing the Transitive Closure of
 * a Union of Affine Integer Tuple Relations".
 */
static __isl_give isl_map *cocoa_fig_1_right_power(isl_ctx *ctx)
{
	return isl_map_read_from_str(ctx,
		"{ [1] -> [[0,0] -> [0,1]]; [2] -> [[0,0] -> [1,1]]; "
		"  [1] -> [[0,1] -> [1,1]]; [1] -> [[2,2] -> [3,2]]; "
		"  [2] -> [[2,2] -> [3,3]]; [1] -> [[3,2] -> [3,3]] }");
}

/* Construct a representation of the transitive closure of the graph
 * on the right of Figure 1 in "Computing the Transitive Closure of
 * a Union of Affine Integer Tuple Relations".
 */
static __isl_give isl_map *cocoa_fig_1_right_tc(isl_ctx *ctx)
{
	return isl_set_unwrap(isl_map_range(cocoa_fig_1_right_power(ctx)));
}

static int test_closure(isl_ctx *ctx)
{
	const char *str;
	isl_map *map, *map2;
	isl_bool exact, equal;

	/* COCOA example 1 */
	map = isl_map_read_from_str(ctx,
		"[n] -> { [i,j] -> [i2,j2] : i2 = i + 1 and j2 = j + 1 and "
			"1 <= i and i < n and 1 <= j and j < n or "
			"i2 = i + 1 and j2 = j - 1 and "
			"1 <= i and i < n and 2 <= j and j <= n }");
	map = isl_map_power(map, &exact);
	assert(exact);
	isl_map_free(map);

	/* COCOA example 1 */
	map = isl_map_read_from_str(ctx,
		"[n] -> { [i,j] -> [i2,j2] : i2 = i + 1 and j2 = j + 1 and "
			"1 <= i and i < n and 1 <= j and j < n or "
			"i2 = i + 1 and j2 = j - 1 and "
			"1 <= i and i < n and 2 <= j and j <= n }");
	map = isl_map_transitive_closure(map, &exact);
	assert(exact);
	map2 = isl_map_read_from_str(ctx,
		"[n] -> { [i,j] -> [i2,j2] : exists (k1,k2,k : "
			"1 <= i and i < n and 1 <= j and j <= n and "
			"2 <= i2 and i2 <= n and 1 <= j2 and j2 <= n and "
			"i2 = i + k1 + k2 and j2 = j + k1 - k2 and "
			"k1 >= 0 and k2 >= 0 and k1 + k2 = k and k >= 1 )}");
	assert(isl_map_is_equal(map, map2));
	isl_map_free(map2);
	isl_map_free(map);

	map = isl_map_read_from_str(ctx,
		"[n] -> { [x] -> [y] : y = x + 1 and 0 <= x and x <= n and "
				     " 0 <= y and y <= n }");
	map = isl_map_transitive_closure(map, &exact);
	map2 = isl_map_read_from_str(ctx,
		"[n] -> { [x] -> [y] : y > x and 0 <= x and x <= n and "
				     " 0 <= y and y <= n }");
	assert(isl_map_is_equal(map, map2));
	isl_map_free(map2);
	isl_map_free(map);

	/* COCOA example 2 */
	map = isl_map_read_from_str(ctx,
		"[n] -> { [i,j] -> [i2,j2] : i2 = i + 2 and j2 = j + 2 and "
			"1 <= i and i < n - 1 and 1 <= j and j < n - 1 or "
			"i2 = i + 2 and j2 = j - 2 and "
			"1 <= i and i < n - 1 and 3 <= j and j <= n }");
	map = isl_map_transitive_closure(map, &exact);
	assert(exact);
	map2 = isl_map_read_from_str(ctx,
		"[n] -> { [i,j] -> [i2,j2] : exists (k1,k2,k : "
			"1 <= i and i < n - 1 and 1 <= j and j <= n and "
			"3 <= i2 and i2 <= n and 1 <= j2 and j2 <= n and "
			"i2 = i + 2 k1 + 2 k2 and j2 = j + 2 k1 - 2 k2 and "
			"k1 >= 0 and k2 >= 0 and k1 + k2 = k and k >= 1) }");
	assert(isl_map_is_equal(map, map2));
	isl_map_free(map);
	isl_map_free(map2);

	/* COCOA Fig.2 left */
	map = isl_map_read_from_str(ctx,
		"[n] -> { [i,j] -> [i2,j2] : i2 = i + 2 and j2 = j and "
			"i <= 2 j - 3 and i <= n - 2 and j <= 2 i - 1 and "
			"j <= n or "
			"i2 = i and j2 = j + 2 and i <= 2 j - 1 and i <= n and "
			"j <= 2 i - 3 and j <= n - 2 or "
			"i2 = i + 1 and j2 = j + 1 and i <= 2 j - 1 and "
			"i <= n - 1 and j <= 2 i - 1 and j <= n - 1 }");
	map = isl_map_transitive_closure(map, &exact);
	assert(exact);
	isl_map_free(map);

	/* COCOA Fig.2 right */
	map = isl_map_read_from_str(ctx,
		"[n] -> { [i,j] -> [i2,j2] : i2 = i + 3 and j2 = j and "
			"i <= 2 j - 4 and i <= n - 3 and j <= 2 i - 1 and "
			"j <= n or "
			"i2 = i and j2 = j + 3 and i <= 2 j - 1 and i <= n and "
			"j <= 2 i - 4 and j <= n - 3 or "
			"i2 = i + 1 and j2 = j + 1 and i <= 2 j - 1 and "
			"i <= n - 1 and j <= 2 i - 1 and j <= n - 1 }");
	map = isl_map_power(map, &exact);
	assert(exact);
	isl_map_free(map);

	/* COCOA Fig.2 right */
	map = isl_map_read_from_str(ctx,
		"[n] -> { [i,j] -> [i2,j2] : i2 = i + 3 and j2 = j and "
			"i <= 2 j - 4 and i <= n - 3 and j <= 2 i - 1 and "
			"j <= n or "
			"i2 = i and j2 = j + 3 and i <= 2 j - 1 and i <= n and "
			"j <= 2 i - 4 and j <= n - 3 or "
			"i2 = i + 1 and j2 = j + 1 and i <= 2 j - 1 and "
			"i <= n - 1 and j <= 2 i - 1 and j <= n - 1 }");
	map = isl_map_transitive_closure(map, &exact);
	assert(exact);
	map2 = isl_map_read_from_str(ctx,
		"[n] -> { [i,j] -> [i2,j2] : exists (k1,k2,k3,k : "
			"i <= 2 j - 1 and i <= n and j <= 2 i - 1 and "
			"j <= n and 3 + i + 2 j <= 3 n and "
			"3 + 2 i + j <= 3n and i2 <= 2 j2 -1 and i2 <= n and "
			"i2 <= 3 j2 - 4 and j2 <= 2 i2 -1 and j2 <= n and "
			"13 + 4 j2 <= 11 i2 and i2 = i + 3 k1 + k3 and "
			"j2 = j + 3 k2 + k3 and k1 >= 0 and k2 >= 0 and "
			"k3 >= 0 and k1 + k2 + k3 = k and k > 0) }");
	assert(isl_map_is_equal(map, map2));
	isl_map_free(map2);
	isl_map_free(map);

	map = cocoa_fig_1_right_graph(ctx);
	map = isl_map_transitive_closure(map, &exact);
	assert(exact);
	map2 = cocoa_fig_1_right_tc(ctx);
	assert(isl_map_is_equal(map, map2));
	isl_map_free(map2);
	isl_map_free(map);

	map = cocoa_fig_1_right_graph(ctx);
	map = isl_map_power(map, &exact);
	map2 = cocoa_fig_1_right_power(ctx);
	equal = isl_map_is_equal(map, map2);
	isl_map_free(map2);
	isl_map_free(map);
	if (equal < 0)
		return -1;
	if (!exact)
		isl_die(ctx, isl_error_unknown, "power not exact", return -1);
	if (!equal)
		isl_die(ctx, isl_error_unknown, "unexpected power", return -1);

	/* COCOA Theorem 1 counter example */
	map = isl_map_read_from_str(ctx,
		"{ [i,j] -> [i2,j2] : i = 0 and 0 <= j and j <= 1 and "
			"i2 = 1 and j2 = j or "
			"i = 0 and j = 0 and i2 = 0 and j2 = 1 }");
	map = isl_map_transitive_closure(map, &exact);
	assert(exact);
	isl_map_free(map);

	map = isl_map_read_from_str(ctx,
		"[m,n] -> { [i,j] -> [i2,j2] : i2 = i and j2 = j + 2 and "
			"1 <= i,i2 <= n and 1 <= j,j2 <= m or "
			"i2 = i + 1 and 3 <= j2 - j <= 4 and "
			"1 <= i,i2 <= n and 1 <= j,j2 <= m }");
	map = isl_map_transitive_closure(map, &exact);
	assert(exact);
	isl_map_free(map);

	/* Kelly et al 1996, fig 12 */
	map = isl_map_read_from_str(ctx,
		"[n] -> { [i,j] -> [i2,j2] : i2 = i and j2 = j + 1 and "
			"1 <= i,j,j+1 <= n or "
			"j = n and j2 = 1 and i2 = i + 1 and "
			"1 <= i,i+1 <= n }");
	map = isl_map_transitive_closure(map, &exact);
	assert(exact);
	map2 = isl_map_read_from_str(ctx,
		"[n] -> { [i,j] -> [i2,j2] : 1 <= j < j2 <= n and "
			"1 <= i <= n and i = i2 or "
			"1 <= i < i2 <= n and 1 <= j <= n and "
			"1 <= j2 <= n }");
	assert(isl_map_is_equal(map, map2));
	isl_map_free(map2);
	isl_map_free(map);

	/* Omega's closure4 */
	map = isl_map_read_from_str(ctx,
		"[m,n] -> { [x,y] -> [x2,y2] : x2 = x and y2 = y + 1 and "
			"1 <= x,y <= 10 or "
			"x2 = x + 1 and y2 = y and "
			"1 <= x <= 20 && 5 <= y <= 15 }");
	map = isl_map_transitive_closure(map, &exact);
	assert(exact);
	isl_map_free(map);

	map = isl_map_read_from_str(ctx,
		"[n] -> { [x] -> [y]: 1 <= n <= y - x <= 10 }");
	map = isl_map_transitive_closure(map, &exact);
	assert(!exact);
	map2 = isl_map_read_from_str(ctx,
		"[n] -> { [x] -> [y] : 1 <= n <= 10 and y >= n + x }");
	assert(isl_map_is_equal(map, map2));
	isl_map_free(map);
	isl_map_free(map2);

	str = "[n, m] -> { [i0, i1, i2, i3] -> [o0, o1, o2, o3] : "
	    "i3 = 1 and o0 = i0 and o1 = -1 + i1 and o2 = -1 + i2 and "
	    "o3 = -2 + i2 and i1 <= -1 + i0 and i1 >= 1 - m + i0 and "
	    "i1 >= 2 and i1 <= n and i2 >= 3 and i2 <= 1 + n and i2 <= m }";
	map = isl_map_read_from_str(ctx, str);
	map = isl_map_transitive_closure(map, &exact);
	assert(exact);
	map2 = isl_map_read_from_str(ctx, str);
	assert(isl_map_is_equal(map, map2));
	isl_map_free(map);
	isl_map_free(map2);

	str = "{[0] -> [1]; [2] -> [3]}";
	map = isl_map_read_from_str(ctx, str);
	map = isl_map_transitive_closure(map, &exact);
	assert(exact);
	map2 = isl_map_read_from_str(ctx, str);
	assert(isl_map_is_equal(map, map2));
	isl_map_free(map);
	isl_map_free(map2);

	str = "[n] -> { [[i0, i1, 1, 0, i0] -> [i5, 1]] -> "
	    "[[i0, -1 + i1, 2, 0, i0] -> [-1 + i5, 2]] : "
	    "exists (e0 = [(3 - n)/3]: i5 >= 2 and i1 >= 2 and "
	    "3i0 <= -1 + n and i1 <= -1 + n and i5 <= -1 + n and "
	    "3e0 >= 1 - n and 3e0 <= 2 - n and 3i0 >= -2 + n); "
	    "[[i0, i1, 2, 0, i0] -> [i5, 1]] -> "
	    "[[i0, i1, 1, 0, i0] -> [-1 + i5, 2]] : "
	    "exists (e0 = [(3 - n)/3]: i5 >= 2 and i1 >= 1 and "
	    "3i0 <= -1 + n and i1 <= -1 + n and i5 <= -1 + n and "
	    "3e0 >= 1 - n and 3e0 <= 2 - n and 3i0 >= -2 + n); "
	    "[[i0, i1, 1, 0, i0] -> [i5, 2]] -> "
	    "[[i0, -1 + i1, 2, 0, i0] -> [i5, 1]] : "
	    "exists (e0 = [(3 - n)/3]: i1 >= 2 and i5 >= 1 and "
	    "3i0 <= -1 + n and i1 <= -1 + n and i5 <= -1 + n and "
	    "3e0 >= 1 - n and 3e0 <= 2 - n and 3i0 >= -2 + n); "
	    "[[i0, i1, 2, 0, i0] -> [i5, 2]] -> "
	    "[[i0, i1, 1, 0, i0] -> [i5, 1]] : "
	    "exists (e0 = [(3 - n)/3]: i5 >= 1 and i1 >= 1 and "
	    "3i0 <= -1 + n and i1 <= -1 + n and i5 <= -1 + n and "
	    "3e0 >= 1 - n and 3e0 <= 2 - n and 3i0 >= -2 + n) }";
	map = isl_map_read_from_str(ctx, str);
	map = isl_map_transitive_closure(map, NULL);
	assert(map);
	isl_map_free(map);

	return 0;
}

/* Check that the actual result of a boolean operation is equal
 * to the expected result.
 */
static isl_stat check_bool(isl_ctx *ctx, isl_bool actual, isl_bool expected)
{
	if (actual != expected)
		isl_die(ctx, isl_error_unknown,
			"incorrect boolean operation", return isl_stat_error);
	return isl_stat_ok;
}

/* Test operations on isl_bool values.
 *
 * This tests:
 *
 * 	isl_bool_not
 * 	isl_bool_ok
 */
static int test_isl_bool(isl_ctx *ctx)
{
	if (check_bool(ctx, isl_bool_not(isl_bool_true), isl_bool_false) < 0)
		return -1;
	if (check_bool(ctx, isl_bool_not(isl_bool_false), isl_bool_true) < 0)
		return -1;
	if (check_bool(ctx, isl_bool_not(isl_bool_error), isl_bool_error) < 0)
		return -1;
	if (check_bool(ctx, isl_bool_ok(0), isl_bool_false) < 0)
		return -1;
	if (check_bool(ctx, isl_bool_ok(1), isl_bool_true) < 0)
		return -1;
	if (check_bool(ctx, isl_bool_ok(-1), isl_bool_true) < 0)
		return -1;
	if (check_bool(ctx, isl_bool_ok(2), isl_bool_true) < 0)
		return -1;
	if (check_bool(ctx, isl_bool_ok(-2), isl_bool_true) < 0)
		return -1;

	return 0;
}

static int test_lex(struct isl_ctx *ctx)
{
	isl_space *space;
	isl_map *map;
	int empty;

	space = isl_space_set_alloc(ctx, 0, 0);
	map = isl_map_lex_le(space);
	empty = isl_map_is_empty(map);
	isl_map_free(map);

	if (empty < 0)
		return -1;
	if (empty)
		isl_die(ctx, isl_error_unknown,
			"expecting non-empty result", return -1);

	return 0;
}

/* Inputs for isl_map_lexmin tests.
 * "map" is the input and "lexmin" is the expected result.
 */
struct {
	const char *map;
	const char *lexmin;
} lexmin_tests [] = {
	{ "{ [x] -> [y] : x <= y <= 10; [x] -> [5] : -8 <= x <= 8 }",
	  "{ [x] -> [5] : 6 <= x <= 8; "
	    "[x] -> [x] : x <= 5 or (9 <= x <= 10) }" },
	{ "{ [x] -> [y] : 4y = x or 4y = -1 + x or 4y = -2 + x }",
	  "{ [x] -> [y] : 4y = x or 4y = -1 + x or 4y = -2 + x }" },
	{ "{ [x] -> [y] : x = 4y; [x] -> [y] : x = 2y }",
	  "{ [x] -> [y] : (4y = x and x >= 0) or "
		"(exists (e0 = [(x)/4], e1 = [(-2 + x)/4]: 2y = x and "
		"4e1 = -2 + x and 4e0 <= -1 + x and 4e0 >= -3 + x)) or "
		"(exists (e0 = [(x)/4]: 2y = x and 4e0 = x and x <= -4)) }" },
	{ "{ T[a] -> S[b, c] : a = 4b-2c and c >= b }",
	  "{ T[a] -> S[b, c] : 2b = a and 2c = a }" },
	/* Check that empty pieces are properly combined. */
	{ "[K, N] -> { [x, y] -> [a, b] : K+2<=N<=K+4 and x>=4 and "
		"2N-6<=x<K+N and N-1<=a<=K+N-1 and N+b-6<=a<=2N-4 and "
		"b<=2N-3K+a and 3b<=4N-K+1 and b>=N and a>=x+1 }",
	  "[K, N] -> { [x, y] -> [1 + x, N] : x >= -6 + 2N and "
		"x <= -5 + 2N and x >= -1 + 3K - N and x <= -2 + K + N and "
		"x >= 4 }" },
	{ "{ [i, k, j] -> [a, b, c, d] : 8*floor((b)/8) = b and k <= 255 and "
		"a <= 255 and c <= 255 and d <= 255 - j and "
		"255 - j <= 7d <= 7 - i and 240d <= 239 + a and "
		"247d <= 247 + k - j and 247d <= 247 + k - b and "
		"247d <= 247 + i and 248 - b <= 248d <= c and "
		"254d >= i - a + b and 254d >= -a + b and "
		"255d >= -i + a - b and 1792d >= -63736 + 257b }",
	  "{ [i, k, j] -> "
	    "[-127762 + i + 502j, -62992 + 248j, 63240 - 248j, 255 - j] : "
		"k <= 255 and 7j >= 1778 + i and 246j >= 62738 - k and "
		"247j >= 62738 - i and 509j <= 129795 + i and "
		"742j >= 188724 - i; "
	    "[0, k, j] -> [1, 0, 248, 1] : k <= 255 and 248 <= j <= 254, k }" },
	{ "{ [a] -> [b] : 0 <= b <= 255 and -509 + a <= 512b < a and "
			"16*floor((8 + b)/16) <= 7 + b; "
	    "[a] -> [1] }",
	  "{ [a] -> [b = 1] : a >= 510 or a <= 0; "
	    "[a] -> [b = 0] : 0 < a <= 509 }" },
	{ "{ rat: [i] : 1 <= 2i <= 9 }", "{ rat: [i] : 2i = 1 }" },
	{ "{ rat: [i] : 1 <= 2i <= 9 or i >= 10 }", "{ rat: [i] : 2i = 1 }" },
	{ "{ rat: [i] : 21 <= 2i <= 29 or i = 5 }", "{ rat: [5] }" },
};

static int test_lexmin(struct isl_ctx *ctx)
{
	int i;
	int equal;
	const char *str;
	isl_basic_map *bmap;
	isl_map *map, *map2;
	isl_set *set;
	isl_set *set2;
	isl_pw_multi_aff *pma;

	str = "[p0, p1] -> { [] -> [] : "
	    "exists (e0 = [(2p1)/3], e1, e2, e3 = [(3 - p1 + 3e0)/3], "
	    "e4 = [(p1)/3], e5 = [(p1 + 3e4)/3]: "
	    "3e0 >= -2 + 2p1 and 3e0 >= p1 and 3e3 >= 1 - p1 + 3e0 and "
	    "3e0 <= 2p1 and 3e3 >= -2 + p1 and 3e3 <= -1 + p1 and p1 >= 3 and "
	    "3e5 >= -2 + 2p1 and 3e5 >= p1 and 3e5 <= -1 + p1 + 3e4 and "
	    "3e4 <= p1 and 3e4 >= -2 + p1 and e3 <= -1 + e0 and "
	    "3e4 >= 6 - p1 + 3e1 and 3e1 >= p1 and 3e5 >= -2 + p1 + 3e4 and "
	    "2e4 >= 3 - p1 + 2e1 and e4 <= e1 and 3e3 <= 2 - p1 + 3e0 and "
	    "e5 >= 1 + e1 and 3e4 >= 6 - 2p1 + 3e1 and "
	    "p0 >= 2 and p1 >= p0 and 3e2 >= p1 and 3e4 >= 6 - p1 + 3e2 and "
	    "e2 <= e1 and e3 >= 1 and e4 <= e2) }";
	map = isl_map_read_from_str(ctx, str);
	map = isl_map_lexmin(map);
	isl_map_free(map);
	if (!map)
		return -1;

	str = "[C] -> { [obj,a,b,c] : obj <= 38 a + 7 b + 10 c and "
	    "a + b <= 1 and c <= 10 b and c <= C and a,b,c,C >= 0 }";
	set = isl_set_read_from_str(ctx, str);
	set = isl_set_lexmax(set);
	str = "[C] -> { [obj,a,b,c] : C = 8 }";
	set2 = isl_set_read_from_str(ctx, str);
	set = isl_set_intersect(set, set2);
	assert(!isl_set_is_empty(set));
	isl_set_free(set);

	for (i = 0; i < ARRAY_SIZE(lexmin_tests); ++i) {
		map = isl_map_read_from_str(ctx, lexmin_tests[i].map);
		map = isl_map_lexmin(map);
		map2 = isl_map_read_from_str(ctx, lexmin_tests[i].lexmin);
		equal = isl_map_is_equal(map, map2);
		isl_map_free(map);
		isl_map_free(map2);

		if (equal < 0)
			return -1;
		if (!equal)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return -1);
	}

	str = "{ [i] -> [i', j] : j = i - 8i' and i' >= 0 and i' <= 7 and "
				" 8i' <= i and 8i' >= -7 + i }";
	bmap = isl_basic_map_read_from_str(ctx, str);
	pma = isl_basic_map_lexmin_pw_multi_aff(isl_basic_map_copy(bmap));
	map2 = isl_map_from_pw_multi_aff(pma);
	map = isl_map_from_basic_map(bmap);
	assert(isl_map_is_equal(map, map2));
	isl_map_free(map);
	isl_map_free(map2);

	str = "[i] -> { [i', j] : j = i - 8i' and i' >= 0 and i' <= 7 and "
				" 8i' <= i and 8i' >= -7 + i }";
	set = isl_set_read_from_str(ctx, str);
	pma = isl_set_lexmin_pw_multi_aff(isl_set_copy(set));
	set2 = isl_set_from_pw_multi_aff(pma);
	equal = isl_set_is_equal(set, set2);
	isl_set_free(set);
	isl_set_free(set2);
	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown,
			"unexpected difference between set and "
			"piecewise affine expression", return -1);

	return 0;
}

/* Inputs for isl_pw_multi_aff_max_multi_val tests.
 * "pma" is the input.
 * "res" is the expected result.
 */
static struct {
	const char *pma;
	const char *res;
} opt_pw_tests[] = {
	{ "{ [-1] -> [-1]; [1] -> [1] }", "{ [1] }" },
	{ "{ [a, b] -> [floor((b - 2*floor((-a)/4))/5)] : "
	    "0 <= a, b <= 100 and b mod 2 = 0}", "{ [30] }" },
	{ "[N] -> { [i,j] -> A[i, -i, i + j] : 0 <= i,j <= N <= 10 }",
	  "{ A[10, 0, 20] }" },
	{ "[N] -> {A[N, -N, 2N] : 0 <= N }", "{ A[infty, 0, infty] }" },
};

/* Perform basic isl_pw_multi_aff_max_multi_val tests.
 */
static isl_stat test_pw_max(struct isl_ctx *ctx)
{
	int i;
	isl_pw_multi_aff *pma;
	isl_multi_val *mv;
	isl_stat r;

	for (i = 0; i < ARRAY_SIZE(opt_pw_tests); ++i) {
		pma = isl_pw_multi_aff_read_from_str(ctx, opt_pw_tests[i].pma);
		mv = isl_pw_multi_aff_max_multi_val(pma);
		r = multi_val_check_plain_equal(mv, opt_pw_tests[i].res);
		isl_multi_val_free(mv);

		if (r < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* A specialized isl_set_min_val test case that would return the wrong result
 * in earlier versions of isl.
 * The explicit call to isl_basic_set_union prevents the second basic set
 * from being determined to be empty prior to the call to isl_set_min_val,
 * at least at the point where this test case was introduced.
 */
static int test_min_special(isl_ctx *ctx)
{
	const char *str;
	isl_basic_set *bset1, *bset2;
	isl_set *set;
	isl_aff *obj;
	isl_val *res;
	int ok;

	str = "{ [a, b] : a >= 2 and b >= 0 and 14 - a <= b <= 9 }";
	bset1 = isl_basic_set_read_from_str(ctx, str);
	str = "{ [a, b] : 1 <= a, b and a + b <= 1 }";
	bset2 = isl_basic_set_read_from_str(ctx, str);
	set = isl_basic_set_union(bset1, bset2);
	obj = isl_aff_read_from_str(ctx, "{ [a, b] -> [a] }");

	res = isl_set_min_val(set, obj);
	ok = isl_val_cmp_si(res, 5) == 0;

	isl_aff_free(obj);
	isl_set_free(set);
	isl_val_free(res);

	if (!res)
		return -1;
	if (!ok)
		isl_die(ctx, isl_error_unknown, "unexpected minimum",
			return -1);

	return 0;
}

/* A specialized isl_set_min_val test case that would return an error
 * in earlier versions of isl.
 */
static int test_min_special2(isl_ctx *ctx)
{
	const char *str;
	isl_basic_set *bset;
	isl_aff *obj;
	isl_val *res;

	str = "{ [i, j, k] : 2j = i and 2k = i + 1 and i >= 2 }";
	bset = isl_basic_set_read_from_str(ctx, str);

	obj = isl_aff_read_from_str(ctx, "{ [i, j, k] -> [i] }");

	res = isl_basic_set_max_val(bset, obj);

	isl_basic_set_free(bset);
	isl_aff_free(obj);
	isl_val_free(res);

	if (!res)
		return -1;

	return 0;
}

/* Check that the result of isl_set_min_multi_pw_aff
 * on the union of the sets with string descriptions "s1" and "s2"
 * consists of a single expression (on a single cell).
 */
static isl_stat check_single_expr_min(isl_ctx *ctx, const char *s1,
	const char *s2)
{
	isl_size n;
	isl_set *set1, *set2;
	isl_multi_pw_aff *mpa;
	isl_pw_multi_aff *pma;

	set1 = isl_set_read_from_str(ctx, s1);
	set2 = isl_set_read_from_str(ctx, s2);
	set1 = isl_set_union(set1, set2);
	mpa = isl_set_min_multi_pw_aff(set1);
	pma = isl_pw_multi_aff_from_multi_pw_aff(mpa);
	n = isl_pw_multi_aff_n_piece(pma);
	isl_pw_multi_aff_free(pma);

	if (n < 0)
		return isl_stat_error;
	if (n != 1)
		isl_die(ctx, isl_error_unknown, "expecting single expression",
			return isl_stat_error);
	return isl_stat_ok;
}

/* A specialized isl_set_min_multi_pw_aff test that checks
 * that the minimum of 2N and 3N for N >= 0 is represented
 * by a single expression, without splitting off the special case N = 0.
 * Do this for both orderings.
 */
static int test_min_mpa(isl_ctx *ctx)
{
	const char *s1, *s2;

	s1 = "[N=0:] -> { [1, 3N:] }";
	s2 = "[N=0:] -> { [10, 2N:] }";
	if (check_single_expr_min(ctx, s1, s2) < 0)
		return -1;
	if (check_single_expr_min(ctx, s2, s1) < 0)
		return -1;

	return 0;
}

struct {
	const char *set;
	const char *obj;
	__isl_give isl_val *(*fn)(__isl_keep isl_set *set,
		__isl_keep isl_aff *obj);
	const char *res;
} opt_tests[] = {
	{ "{ [-1]; [1] }", "{ [x] -> [x] }", &isl_set_min_val, "-1" },
	{ "{ [-1]; [1] }", "{ [x] -> [x] }", &isl_set_max_val, "1" },
	{ "{ [a, b] : 0 <= a, b <= 100 and b mod 2 = 0}",
	  "{ [a, b] -> [floor((b - 2*floor((-a)/4))/5)] }",
	  &isl_set_max_val, "30" },

};

/* Perform basic isl_set_min_val and isl_set_max_val tests.
 * In particular, check the results on non-convex inputs.
 */
static int test_min(struct isl_ctx *ctx)
{
	int i;
	isl_set *set;
	isl_aff *obj;
	isl_val *val, *res;
	isl_bool ok;

	for (i = 0; i < ARRAY_SIZE(opt_tests); ++i) {
		set = isl_set_read_from_str(ctx, opt_tests[i].set);
		obj = isl_aff_read_from_str(ctx, opt_tests[i].obj);
		res = isl_val_read_from_str(ctx, opt_tests[i].res);
		val = opt_tests[i].fn(set, obj);
		ok = isl_val_eq(res, val);
		isl_val_free(res);
		isl_val_free(val);
		isl_aff_free(obj);
		isl_set_free(set);

		if (ok < 0)
			return -1;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected optimum", return -1);
	}

	if (test_pw_max(ctx) < 0)
		return -1;
	if (test_min_special(ctx) < 0)
		return -1;
	if (test_min_special2(ctx) < 0)
		return -1;

	return 0;
}

struct must_may {
	isl_map *must;
	isl_map *may;
};

static isl_stat collect_must_may(__isl_take isl_map *dep, int must,
	void *dep_user, void *user)
{
	struct must_may *mm = (struct must_may *)user;

	if (must)
		mm->must = isl_map_union(mm->must, dep);
	else
		mm->may = isl_map_union(mm->may, dep);

	return isl_stat_ok;
}

static int common_space(void *first, void *second)
{
	int depth = *(int *)first;
	return 2 * depth;
}

static int map_is_equal(__isl_keep isl_map *map, const char *str)
{
	isl_map *map2;
	int equal;

	if (!map)
		return -1;

	map2 = isl_map_read_from_str(map->ctx, str);
	equal = isl_map_is_equal(map, map2);
	isl_map_free(map2);

	return equal;
}

static int map_check_equal(__isl_keep isl_map *map, const char *str)
{
	int equal;

	equal = map_is_equal(map, str);
	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(isl_map_get_ctx(map), isl_error_unknown,
			"result not as expected", return -1);
	return 0;
}

/* Is "set" equal to the set described by "str"?
 */
static isl_bool set_is_equal(__isl_keep isl_set *set, const char *str)
{
	isl_set *set2;
	isl_bool equal;

	if (!set)
		return isl_bool_error;

	set2 = isl_set_read_from_str(isl_set_get_ctx(set), str);
	equal = isl_set_is_equal(set, set2);
	isl_set_free(set2);

	return equal;
}

/* Check that "set" is equal to the set described by "str".
 */
static isl_stat set_check_equal(__isl_keep isl_set *set, const char *str)
{
	isl_bool equal;

	equal = set_is_equal(set, str);
	if (equal < 0)
		return isl_stat_error;
	if (!equal)
		isl_die(isl_set_get_ctx(set), isl_error_unknown,
			"result not as expected", return isl_stat_error);
	return isl_stat_ok;
}

/* Is "uset" equal to the union set described by "str"?
 */
static isl_bool uset_is_equal(__isl_keep isl_union_set *uset, const char *str)
{
	isl_union_set *uset2;
	isl_bool equal;

	if (!uset)
		return isl_bool_error;

	uset2 = isl_union_set_read_from_str(isl_union_set_get_ctx(uset), str);
	equal = isl_union_set_is_equal(uset, uset2);
	isl_union_set_free(uset2);

	return equal;
}

/* Check that "uset" is equal to the union set described by "str".
 */
static isl_stat uset_check_equal(__isl_keep isl_union_set *uset,
	const char *str)
{
	isl_bool equal;

	equal = uset_is_equal(uset, str);
	if (equal < 0)
		return isl_stat_error;
	if (!equal)
		isl_die(isl_union_set_get_ctx(uset), isl_error_unknown,
			"result not as expected", return isl_stat_error);
	return isl_stat_ok;
}

static int test_dep(struct isl_ctx *ctx)
{
	const char *str;
	isl_space *space;
	isl_map *map;
	isl_access_info *ai;
	isl_flow *flow;
	int depth;
	struct must_may mm;

	depth = 3;

	str = "{ [2,i,0] -> [i] : 0 <= i <= 10 }";
	map = isl_map_read_from_str(ctx, str);
	ai = isl_access_info_alloc(map, &depth, &common_space, 2);

	str = "{ [0,i,0] -> [i] : 0 <= i <= 10 }";
	map = isl_map_read_from_str(ctx, str);
	ai = isl_access_info_add_source(ai, map, 1, &depth);

	str = "{ [1,i,0] -> [5] : 0 <= i <= 10 }";
	map = isl_map_read_from_str(ctx, str);
	ai = isl_access_info_add_source(ai, map, 1, &depth);

	flow = isl_access_info_compute_flow(ai);
	space = isl_space_alloc(ctx, 0, 3, 3);
	mm.must = isl_map_empty(isl_space_copy(space));
	mm.may = isl_map_empty(space);

	isl_flow_foreach(flow, collect_must_may, &mm);

	str = "{ [0,i,0] -> [2,i,0] : (0 <= i <= 4) or (6 <= i <= 10); "
	      "  [1,10,0] -> [2,5,0] }";
	assert(map_is_equal(mm.must, str));
	str = "{ [i,j,k] -> [l,m,n] : 1 = 0 }";
	assert(map_is_equal(mm.may, str));

	isl_map_free(mm.must);
	isl_map_free(mm.may);
	isl_flow_free(flow);


	str = "{ [2,i,0] -> [i] : 0 <= i <= 10 }";
	map = isl_map_read_from_str(ctx, str);
	ai = isl_access_info_alloc(map, &depth, &common_space, 2);

	str = "{ [0,i,0] -> [i] : 0 <= i <= 10 }";
	map = isl_map_read_from_str(ctx, str);
	ai = isl_access_info_add_source(ai, map, 1, &depth);

	str = "{ [1,i,0] -> [5] : 0 <= i <= 10 }";
	map = isl_map_read_from_str(ctx, str);
	ai = isl_access_info_add_source(ai, map, 0, &depth);

	flow = isl_access_info_compute_flow(ai);
	space = isl_space_alloc(ctx, 0, 3, 3);
	mm.must = isl_map_empty(isl_space_copy(space));
	mm.may = isl_map_empty(space);

	isl_flow_foreach(flow, collect_must_may, &mm);

	str = "{ [0,i,0] -> [2,i,0] : (0 <= i <= 4) or (6 <= i <= 10) }";
	assert(map_is_equal(mm.must, str));
	str = "{ [0,5,0] -> [2,5,0]; [1,i,0] -> [2,5,0] : 0 <= i <= 10 }";
	assert(map_is_equal(mm.may, str));

	isl_map_free(mm.must);
	isl_map_free(mm.may);
	isl_flow_free(flow);


	str = "{ [2,i,0] -> [i] : 0 <= i <= 10 }";
	map = isl_map_read_from_str(ctx, str);
	ai = isl_access_info_alloc(map, &depth, &common_space, 2);

	str = "{ [0,i,0] -> [i] : 0 <= i <= 10 }";
	map = isl_map_read_from_str(ctx, str);
	ai = isl_access_info_add_source(ai, map, 0, &depth);

	str = "{ [1,i,0] -> [5] : 0 <= i <= 10 }";
	map = isl_map_read_from_str(ctx, str);
	ai = isl_access_info_add_source(ai, map, 0, &depth);

	flow = isl_access_info_compute_flow(ai);
	space = isl_space_alloc(ctx, 0, 3, 3);
	mm.must = isl_map_empty(isl_space_copy(space));
	mm.may = isl_map_empty(space);

	isl_flow_foreach(flow, collect_must_may, &mm);

	str = "{ [0,i,0] -> [2,i,0] : 0 <= i <= 10; "
	      "  [1,i,0] -> [2,5,0] : 0 <= i <= 10 }";
	assert(map_is_equal(mm.may, str));
	str = "{ [i,j,k] -> [l,m,n] : 1 = 0 }";
	assert(map_is_equal(mm.must, str));

	isl_map_free(mm.must);
	isl_map_free(mm.may);
	isl_flow_free(flow);


	str = "{ [0,i,2] -> [i] : 0 <= i <= 10 }";
	map = isl_map_read_from_str(ctx, str);
	ai = isl_access_info_alloc(map, &depth, &common_space, 2);

	str = "{ [0,i,0] -> [i] : 0 <= i <= 10 }";
	map = isl_map_read_from_str(ctx, str);
	ai = isl_access_info_add_source(ai, map, 0, &depth);

	str = "{ [0,i,1] -> [5] : 0 <= i <= 10 }";
	map = isl_map_read_from_str(ctx, str);
	ai = isl_access_info_add_source(ai, map, 0, &depth);

	flow = isl_access_info_compute_flow(ai);
	space = isl_space_alloc(ctx, 0, 3, 3);
	mm.must = isl_map_empty(isl_space_copy(space));
	mm.may = isl_map_empty(space);

	isl_flow_foreach(flow, collect_must_may, &mm);

	str = "{ [0,i,0] -> [0,i,2] : 0 <= i <= 10; "
	      "  [0,i,1] -> [0,5,2] : 0 <= i <= 5 }";
	assert(map_is_equal(mm.may, str));
	str = "{ [i,j,k] -> [l,m,n] : 1 = 0 }";
	assert(map_is_equal(mm.must, str));

	isl_map_free(mm.must);
	isl_map_free(mm.may);
	isl_flow_free(flow);


	str = "{ [0,i,1] -> [i] : 0 <= i <= 10 }";
	map = isl_map_read_from_str(ctx, str);
	ai = isl_access_info_alloc(map, &depth, &common_space, 2);

	str = "{ [0,i,0] -> [i] : 0 <= i <= 10 }";
	map = isl_map_read_from_str(ctx, str);
	ai = isl_access_info_add_source(ai, map, 0, &depth);

	str = "{ [0,i,2] -> [5] : 0 <= i <= 10 }";
	map = isl_map_read_from_str(ctx, str);
	ai = isl_access_info_add_source(ai, map, 0, &depth);

	flow = isl_access_info_compute_flow(ai);
	space = isl_space_alloc(ctx, 0, 3, 3);
	mm.must = isl_map_empty(isl_space_copy(space));
	mm.may = isl_map_empty(space);

	isl_flow_foreach(flow, collect_must_may, &mm);

	str = "{ [0,i,0] -> [0,i,1] : 0 <= i <= 10; "
	      "  [0,i,2] -> [0,5,1] : 0 <= i <= 4 }";
	assert(map_is_equal(mm.may, str));
	str = "{ [i,j,k] -> [l,m,n] : 1 = 0 }";
	assert(map_is_equal(mm.must, str));

	isl_map_free(mm.must);
	isl_map_free(mm.may);
	isl_flow_free(flow);


	depth = 5;

	str = "{ [1,i,0,0,0] -> [i,j] : 0 <= i <= 10 and 0 <= j <= 10 }";
	map = isl_map_read_from_str(ctx, str);
	ai = isl_access_info_alloc(map, &depth, &common_space, 1);

	str = "{ [0,i,0,j,0] -> [i,j] : 0 <= i <= 10 and 0 <= j <= 10 }";
	map = isl_map_read_from_str(ctx, str);
	ai = isl_access_info_add_source(ai, map, 1, &depth);

	flow = isl_access_info_compute_flow(ai);
	space = isl_space_alloc(ctx, 0, 5, 5);
	mm.must = isl_map_empty(isl_space_copy(space));
	mm.may = isl_map_empty(space);

	isl_flow_foreach(flow, collect_must_may, &mm);

	str = "{ [0,i,0,j,0] -> [1,i,0,0,0] : 0 <= i,j <= 10 }";
	assert(map_is_equal(mm.must, str));
	str = "{ [0,0,0,0,0] -> [0,0,0,0,0] : 1 = 0 }";
	assert(map_is_equal(mm.may, str));

	isl_map_free(mm.must);
	isl_map_free(mm.may);
	isl_flow_free(flow);

	return 0;
}

/* Check that the dependence analysis proceeds without errors.
 * Earlier versions of isl would break down during the analysis
 * due to the use of the wrong spaces.
 */
static int test_flow(isl_ctx *ctx)
{
	const char *str;
	isl_union_map *access, *schedule;
	isl_union_map *must_dep, *may_dep;
	int r;

	str = "{ S0[j] -> i[]; S1[j,i] -> i[]; S2[] -> i[]; S3[] -> i[] }";
	access = isl_union_map_read_from_str(ctx, str);
	str = "{ S0[j] -> [0,j,0,0] : 0 <= j < 10; "
		"S1[j,i] -> [0,j,1,i] : 0 <= j < i < 10; "
		"S2[] -> [1,0,0,0]; "
		"S3[] -> [-1,0,0,0] }";
	schedule = isl_union_map_read_from_str(ctx, str);
	r = isl_union_map_compute_flow(access, isl_union_map_copy(access),
					isl_union_map_copy(access), schedule,
					&must_dep, &may_dep, NULL, NULL);
	isl_union_map_free(may_dep);
	isl_union_map_free(must_dep);

	return r;
}

struct {
	const char *map;
	int sv;
} sv_tests[] = {
	{ "[N] -> { [i] -> [f] : 0 <= i <= N and 0 <= i - 10 f <= 9 }", 1 },
	{ "[N] -> { [i] -> [f] : 0 <= i <= N and 0 <= i - 10 f <= 10 }", 0 },
	{ "{ [i] -> [3*floor(i/2) + 5*floor(i/3)] }", 1 },
	{ "{ S1[i] -> [i] : 0 <= i <= 9; S2[i] -> [i] : 0 <= i <= 9 }", 1 },
	{ "{ [i] -> S1[i] : 0 <= i <= 9; [i] -> S2[i] : 0 <= i <= 9 }", 0 },
	{ "{ A[i] -> [i]; B[i] -> [i]; B[i] -> [i + 1] }", 0 },
	{ "{ A[i] -> [i]; B[i] -> [i] : i < 0; B[i] -> [i + 1] : i > 0 }", 1 },
	{ "{ A[i] -> [i]; B[i] -> A[i] : i < 0; B[i] -> [i + 1] : i > 0 }", 1 },
	{ "{ A[i] -> [i]; B[i] -> [j] : i - 1 <= j <= i }", 0 },
};

int test_sv(isl_ctx *ctx)
{
	isl_union_map *umap;
	int i;
	int sv;

	for (i = 0; i < ARRAY_SIZE(sv_tests); ++i) {
		umap = isl_union_map_read_from_str(ctx, sv_tests[i].map);
		sv = isl_union_map_is_single_valued(umap);
		isl_union_map_free(umap);
		if (sv < 0)
			return -1;
		if (sv_tests[i].sv && !sv)
			isl_die(ctx, isl_error_internal,
				"map not detected as single valued", return -1);
		if (!sv_tests[i].sv && sv)
			isl_die(ctx, isl_error_internal,
				"map detected as single valued", return -1);
	}

	return 0;
}

struct {
	const char *str;
	int bijective;
} bijective_tests[] = {
	{ "[N,M]->{[i,j] -> [i]}", 0 },
	{ "[N,M]->{[i,j] -> [i] : j=i}", 1 },
	{ "[N,M]->{[i,j] -> [i] : j=0}", 1 },
	{ "[N,M]->{[i,j] -> [i] : j=N}", 1 },
	{ "[N,M]->{[i,j] -> [j,i]}", 1 },
	{ "[N,M]->{[i,j] -> [i+j]}", 0 },
	{ "[N,M]->{[i,j] -> []}", 0 },
	{ "[N,M]->{[i,j] -> [i,j,N]}", 1 },
	{ "[N,M]->{[i,j] -> [2i]}", 0 },
	{ "[N,M]->{[i,j] -> [i,i]}", 0 },
	{ "[N,M]->{[i,j] -> [2i,i]}", 0 },
	{ "[N,M]->{[i,j] -> [2i,j]}", 1 },
	{ "[N,M]->{[i,j] -> [x,y] : 2x=i & y =j}", 1 },
};

static int test_bijective(struct isl_ctx *ctx)
{
	isl_map *map;
	int i;
	int bijective;

	for (i = 0; i < ARRAY_SIZE(bijective_tests); ++i) {
		map = isl_map_read_from_str(ctx, bijective_tests[i].str);
		bijective = isl_map_is_bijective(map);
		isl_map_free(map);
		if (bijective < 0)
			return -1;
		if (bijective_tests[i].bijective && !bijective)
			isl_die(ctx, isl_error_internal,
				"map not detected as bijective", return -1);
		if (!bijective_tests[i].bijective && bijective)
			isl_die(ctx, isl_error_internal,
				"map detected as bijective", return -1);
	}

	return 0;
}

/* Inputs for isl_pw_qpolynomial_gist tests.
 * "pwqp" is the input, "set" is the context and "gist" is the expected result.
 */
struct {
	const char *pwqp;
	const char *set;
	const char *gist;
} pwqp_gist_tests[] = {
	{ "{ [i] -> i }", "{ [k] : exists a : k = 2a }", "{ [i] -> i }" },
	{ "{ [i] -> i + [ (i + [i/3])/2 ] }", "{ [10] }", "{ [i] -> 16 }" },
	{ "{ [i] -> ([(i)/2]) }", "{ [k] : exists a : k = 2a+1 }",
	  "{ [i] -> -1/2 + 1/2 * i }" },
	{ "{ [i] -> i^2 : i != 0 }", "{ [i] : i != 0 }", "{ [i] -> i^2 }" },
};

/* Perform some basic isl_pw_qpolynomial_gist tests.
 */
static isl_stat test_pwqp_gist(isl_ctx *ctx)
{
	int i;
	const char *str;
	isl_set *set;
	isl_pw_qpolynomial *pwqp1, *pwqp2;
	isl_bool equal;

	for (i = 0; i < ARRAY_SIZE(pwqp_gist_tests); ++i) {
		str = pwqp_gist_tests[i].pwqp;
		pwqp1 = isl_pw_qpolynomial_read_from_str(ctx, str);
		str = pwqp_gist_tests[i].set;
		set = isl_set_read_from_str(ctx, str);
		pwqp1 = isl_pw_qpolynomial_gist(pwqp1, set);
		str = pwqp_gist_tests[i].gist;
		pwqp2 = isl_pw_qpolynomial_read_from_str(ctx, str);
		pwqp1 = isl_pw_qpolynomial_sub(pwqp1, pwqp2);
		equal = isl_pw_qpolynomial_is_zero(pwqp1);
		isl_pw_qpolynomial_free(pwqp1);

		if (equal < 0)
			return isl_stat_error;
		if (!equal)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return isl_stat_error);
	}

	return isl_stat_ok;
}

/* Perform a basic isl_pw_qpolynomial_max test.
 */
static isl_stat test_pwqp_max(isl_ctx *ctx)
{
	const char *str;
	isl_pw_qpolynomial *pwqp;
	isl_val *v;
	int ok;

	str = "{ [x=2:9, y] -> floor((x + 1)/4)^3 - floor((2x)/3)^2 }";
	pwqp = isl_pw_qpolynomial_read_from_str(ctx, str);
	v = isl_pw_qpolynomial_max(pwqp);
	ok = isl_val_cmp_si(v, -1) == 0;
	isl_val_free(v);

	if (!v)
		return isl_stat_error;
	if (!ok)
		isl_die(ctx, isl_error_unknown, "unexpected maximum",
			return isl_stat_error);

	return isl_stat_ok;
}

static int test_pwqp(struct isl_ctx *ctx)
{
	const char *str;
	isl_set *set;
	isl_pw_qpolynomial *pwqp1, *pwqp2;
	int equal;

	str = "{ [i,j,k] -> 1 + 9 * [i/5] + 7 * [j/11] + 4 * [k/13] }";
	pwqp1 = isl_pw_qpolynomial_read_from_str(ctx, str);

	pwqp1 = isl_pw_qpolynomial_move_dims(pwqp1, isl_dim_param, 0,
						isl_dim_in, 1, 1);

	str = "[j] -> { [i,k] -> 1 + 9 * [i/5] + 7 * [j/11] + 4 * [k/13] }";
	pwqp2 = isl_pw_qpolynomial_read_from_str(ctx, str);

	pwqp1 = isl_pw_qpolynomial_sub(pwqp1, pwqp2);

	assert(isl_pw_qpolynomial_is_zero(pwqp1));

	isl_pw_qpolynomial_free(pwqp1);

	if (test_pwqp_gist(ctx) < 0)
		return -1;

	str = "{ [i] -> ([([i/2] + [i/2])/5]) }";
	pwqp1 = isl_pw_qpolynomial_read_from_str(ctx, str);
	str = "{ [i] -> ([(2 * [i/2])/5]) }";
	pwqp2 = isl_pw_qpolynomial_read_from_str(ctx, str);

	pwqp1 = isl_pw_qpolynomial_sub(pwqp1, pwqp2);

	assert(isl_pw_qpolynomial_is_zero(pwqp1));

	isl_pw_qpolynomial_free(pwqp1);

	str = "{ [x] -> ([x/2] + [(x+1)/2]) }";
	pwqp1 = isl_pw_qpolynomial_read_from_str(ctx, str);
	str = "{ [x] -> x }";
	pwqp2 = isl_pw_qpolynomial_read_from_str(ctx, str);

	pwqp1 = isl_pw_qpolynomial_sub(pwqp1, pwqp2);

	assert(isl_pw_qpolynomial_is_zero(pwqp1));

	isl_pw_qpolynomial_free(pwqp1);

	str = "{ [i] -> ([i/2]) : i >= 0; [i] -> ([i/3]) : i < 0 }";
	pwqp1 = isl_pw_qpolynomial_read_from_str(ctx, str);
	pwqp2 = isl_pw_qpolynomial_read_from_str(ctx, str);
	pwqp1 = isl_pw_qpolynomial_coalesce(pwqp1);
	pwqp1 = isl_pw_qpolynomial_sub(pwqp1, pwqp2);
	assert(isl_pw_qpolynomial_is_zero(pwqp1));
	isl_pw_qpolynomial_free(pwqp1);

	str = "{ [a,b,a] -> (([(2*[a/3]+b)/5]) * ([(2*[a/3]+b)/5])) }";
	pwqp2 = isl_pw_qpolynomial_read_from_str(ctx, str);
	str = "{ [a,b,c] -> (([(2*[a/3]+b)/5]) * ([(2*[c/3]+b)/5])) }";
	pwqp1 = isl_pw_qpolynomial_read_from_str(ctx, str);
	set = isl_set_read_from_str(ctx, "{ [a,b,a] }");
	pwqp1 = isl_pw_qpolynomial_intersect_domain(pwqp1, set);
	equal = isl_pw_qpolynomial_plain_is_equal(pwqp1, pwqp2);
	isl_pw_qpolynomial_free(pwqp1);
	isl_pw_qpolynomial_free(pwqp2);
	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown, "unexpected result", return -1);

	str = "{ [a,b,c] -> (([(2*[a/3]+1)/5]) * ([(2*[c/3]+1)/5])) : b = 1 }";
	pwqp2 = isl_pw_qpolynomial_read_from_str(ctx, str);
	str = "{ [a,b,c] -> (([(2*[a/3]+b)/5]) * ([(2*[c/3]+b)/5])) }";
	pwqp1 = isl_pw_qpolynomial_read_from_str(ctx, str);
	pwqp1 = isl_pw_qpolynomial_fix_val(pwqp1, isl_dim_set, 1,
						isl_val_one(ctx));
	equal = isl_pw_qpolynomial_plain_is_equal(pwqp1, pwqp2);
	isl_pw_qpolynomial_free(pwqp1);
	isl_pw_qpolynomial_free(pwqp2);
	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown, "unexpected result", return -1);

	if (test_pwqp_max(ctx) < 0)
		return -1;

	return 0;
}

static int test_split_periods(isl_ctx *ctx)
{
	const char *str;
	isl_pw_qpolynomial *pwqp;

	str = "{ [U,V] -> 1/3 * U + 2/3 * V - [(U + 2V)/3] + [U/2] : "
		"U + 2V + 3 >= 0 and - U -2V  >= 0 and - U + 10 >= 0 and "
		"U  >= 0; [U,V] -> U^2 : U >= 100 }";
	pwqp = isl_pw_qpolynomial_read_from_str(ctx, str);

	pwqp = isl_pw_qpolynomial_split_periods(pwqp, 2);

	isl_pw_qpolynomial_free(pwqp);

	if (!pwqp)
		return -1;

	return 0;
}

static int test_union(isl_ctx *ctx)
{
	const char *str;
	isl_union_set *uset1, *uset2;
	isl_union_map *umap1, *umap2;
	int equal;

	str = "{ [i] : 0 <= i <= 1 }";
	uset1 = isl_union_set_read_from_str(ctx, str);
	str = "{ [1] -> [0] }";
	umap1 = isl_union_map_read_from_str(ctx, str);

	umap2 = isl_union_set_lex_gt_union_set(isl_union_set_copy(uset1), uset1);
	equal = isl_union_map_is_equal(umap1, umap2);

	isl_union_map_free(umap1);
	isl_union_map_free(umap2);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown, "union maps not equal",
			return -1);

	str = "{ A[i] -> B[i]; B[i] -> C[i]; A[0] -> C[1] }";
	umap1 = isl_union_map_read_from_str(ctx, str);
	str = "{ A[i]; B[i] }";
	uset1 = isl_union_set_read_from_str(ctx, str);

	uset2 = isl_union_map_domain(umap1);

	equal = isl_union_set_is_equal(uset1, uset2);

	isl_union_set_free(uset1);
	isl_union_set_free(uset2);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown, "union sets not equal",
			return -1);

	return 0;
}

/* Inputs for basic isl_pw_qpolynomial_bound tests.
 * "type" is the type of bound that should be computed.
 * "poly" is a string representation of the input.
 * "bound" is a string representation of the expected result.
 * "tight" is set if the result is expected to be tight.
 */
static struct {
	int tight;
	enum isl_fold type;
	const char *poly;
	const char *bound;
} bound_tests[] = {
	/* Check that computing a bound of a non-zero polynomial
	 * over an unbounded domain does not produce a rational value.
	 * In particular, check that the upper bound is infinity.
	 */
	{ 0, isl_fold_max, "{ [m, n] -> -m * n }", "{ max(infty) }" },
	{ 1, isl_fold_max, "{ [[a, b, c, d] -> [e]] -> 0 }",
	  "{ [a, b, c, d] -> max(0) }" },
	{ 1, isl_fold_max, "{ [[x] -> [x]] -> 1 : exists a : x = 2 a }",
	  "{ [x] -> max(1) : x mod 2 = 0 }" },
	{ 1, isl_fold_min, "{ [x=5:10] -> (x + 2)^2 }", "{ min(49) }" },
	{ 1, isl_fold_max, "{ [0:10] -> 1 }", "{ max(1) }" },
	{ 1, isl_fold_max, "{ [[m] -> [0:m]] -> m^2 }",
	  "{ [m] -> max(m^2) : m >= 0 }" },
};

/* Check that the bound computation can handle differences
 * in domain dimension names of the input polynomial and its domain.
 */
static isl_stat test_bound_space(isl_ctx *ctx)
{
	const char *str;
	isl_set *set;
	isl_pw_qpolynomial *pwqp;
	isl_pw_qpolynomial_fold *pwf;

	str = "{ [[c] -> [c]] }";
	set = isl_set_read_from_str(ctx, str);
	str = "{ [[a] -> [b]] -> 1 }";
	pwqp = isl_pw_qpolynomial_read_from_str(ctx, str);
	pwqp = isl_pw_qpolynomial_intersect_domain(pwqp, set);
	pwf = isl_pw_qpolynomial_bound(pwqp, isl_fold_max, NULL);
	isl_pw_qpolynomial_fold_free(pwf);

	return isl_stat_non_null(pwf);
}

/* Perform basic isl_pw_qpolynomial_bound tests.
 */
static int test_bound(isl_ctx *ctx)
{
	int i;

	if (test_bound_space(ctx) < 0)
		return -1;

	for (i = 0; i < ARRAY_SIZE(bound_tests); ++i) {
		const char *str;
		enum isl_fold type;
		isl_bool equal, tight;
		isl_pw_qpolynomial *pwqp;
		isl_pw_qpolynomial_fold *pwf1, *pwf2;

		str = bound_tests[i].poly;
		pwqp = isl_pw_qpolynomial_read_from_str(ctx, str);
		type = bound_tests[i].type;
		pwf1 = isl_pw_qpolynomial_bound(pwqp, type, &tight);
		str = bound_tests[i].bound;
		pwf2 = isl_pw_qpolynomial_fold_read_from_str(ctx, str);
		equal = isl_pw_qpolynomial_fold_plain_is_equal(pwf1, pwf2);
		isl_pw_qpolynomial_fold_free(pwf2);
		isl_pw_qpolynomial_fold_free(pwf1);
		if (equal < 0)
			return -1;
		if (!equal)
			isl_die(ctx, isl_error_unknown,
				"incorrect bound result", return -1);
		if (bound_tests[i].tight && !tight)
			isl_die(ctx, isl_error_unknown,
				"bound unexpectedly not tight", return -1);
	}

	return 0;
}

/* isl_set is defined to isl_map internally, so the corresponding elements
 * are isl_basic_map objects.
 */
#undef EL_BASE
#undef SET_BASE
#define EL_BASE		basic_map
#define SET_BASE	set
#include "isl_test_list_templ.c"

#undef EL_BASE
#undef SET_BASE
#define EL_BASE		basic_set
#define SET_BASE	union_set
#include "isl_test_list_templ.c"

#undef EL_BASE
#undef SET_BASE
#define EL_BASE		set
#define SET_BASE	union_set
#include "isl_test_list_templ.c"

#undef EL_BASE
#undef SET_BASE
#define EL_BASE		basic_map
#define SET_BASE	map
#include "isl_test_list_templ.c"

#undef EL_BASE
#undef SET_BASE
#define EL_BASE		map
#define SET_BASE	union_map
#include "isl_test_list_templ.c"

/* Check that the conversion from isl objects to lists works as expected.
 */
static int test_get_list(isl_ctx *ctx)
{
	if (test_get_list_basic_map_from_set(ctx, "{ [0]; [2]; [3] }"))
		return -1;
	if (test_get_list_basic_set_from_union_set(ctx, "{ A[0]; B[2]; B[3] }"))
		return -1;
	if (test_get_list_set_from_union_set(ctx, "{ A[0]; A[2]; B[3] }"))
		return -1;
	if (test_get_list_basic_map_from_map(ctx,
				"{ [0] -> [0]; [2] -> [0]; [3] -> [0] }"))
		return -1;
	if (test_get_list_map_from_union_map(ctx,
				"{ A[0] -> [0]; A[2] -> [0]; B[3] -> [0] }"))
		return -1;

	return 0;
}

static int test_lift(isl_ctx *ctx)
{
	const char *str;
	isl_basic_map *bmap;
	isl_basic_set *bset;

	str = "{ [i0] : exists e0 : i0 = 4e0 }";
	bset = isl_basic_set_read_from_str(ctx, str);
	bset = isl_basic_set_lift(bset);
	bmap = isl_basic_map_from_range(bset);
	bset = isl_basic_map_domain(bmap);
	isl_basic_set_free(bset);

	return 0;
}

/* Check that isl_set_is_subset is not confused by identical
 * integer divisions.
 * The call to isl_set_normalize ensures that the equality constraints
 * a = b = 0 are discovered, turning e0 and e1 into identical
 * integer divisions.  Any further simplification would remove
 * the duplicate integer divisions.
 */
static isl_stat test_subset_duplicate_integer_divisions(isl_ctx *ctx)
{
	const char *str;
	isl_bool is_subset;
	isl_set *set1, *set2;

	str = "{ [a, b, c, d] : "
	    "exists (e0 = floor((a + d)/4), e1 = floor((d)/4), "
		    "e2 = floor((-a - d + 4 *floor((a + d)/4))/10), "
		    "e3 = floor((-d + 4*floor((d)/4))/10): "
		"10e2 = -a - 2c - d + 4e0 and 10e3 = -2c - d + 4e1 and "
		"b >= 0 and a <= 0 and b <= a) }";
	set1 = isl_set_read_from_str(ctx, str);
	set2 = isl_set_read_from_str(ctx, str);
	set2 = isl_set_normalize(set2);

	is_subset = isl_set_is_subset(set1, set2);

	isl_set_free(set1);
	isl_set_free(set2);

	if (is_subset < 0)
		return isl_stat_error;
	if (!is_subset)
		isl_die(ctx, isl_error_unknown,
			"set is not considered to be a subset of itself",
			return isl_stat_error);

	return isl_stat_ok;
}

struct {
	const char *set1;
	const char *set2;
	int subset;
} subset_tests[] = {
	{ "{ [112, 0] }",
	  "{ [i0, i1] : exists (e0 = [(i0 - i1)/16], e1: "
		"16e0 <= i0 - i1 and 16e0 >= -15 + i0 - i1 and "
		"16e1 <= i1 and 16e0 >= -i1 and 16e1 >= -i0 + i1) }", 1 },
	{ "{ [65] }",
	  "{ [i] : exists (e0 = [(255i)/256], e1 = [(127i + 65e0)/191], "
		"e2 = [(3i + 61e1)/65], e3 = [(52i + 12e2)/61], "
		"e4 = [(2i + e3)/3], e5 = [(4i + e3)/4], e6 = [(8i + e3)/12]: "
		    "3e4 = 2i + e3 and 4e5 = 4i + e3 and 12e6 = 8i + e3 and "
		    "i <= 255 and 64e3 >= -45 + 67i and i >= 0 and "
		    "256e0 <= 255i and 256e0 >= -255 + 255i and "
		    "191e1 <= 127i + 65e0 and 191e1 >= -190 + 127i + 65e0 and "
		    "65e2 <= 3i + 61e1 and 65e2 >= -64 + 3i + 61e1 and "
		    "61e3 <= 52i + 12e2 and 61e3 >= -60 + 52i + 12e2) }", 1 },
	{ "{ [i] : 0 <= i <= 10 }", "{ rat: [i] : 0 <= i <= 10 }", 1 },
	{ "{ rat: [i] : 0 <= i <= 10 }", "{ [i] : 0 <= i <= 10 }", 0 },
	{ "{ rat: [0] }", "{ [i] : 0 <= i <= 10 }", 1 },
	{ "{ rat: [(1)/2] }", "{ [i] : 0 <= i <= 10 }", 0 },
	{ "{ [t, i] : (exists (e0 = [(2 + t)/4]: 4e0 <= 2 + t and "
			"4e0 >= -1 + t and i >= 57 and i <= 62 and "
			"4e0 <= 62 + t - i and 4e0 >= -61 + t + i and "
			"t >= 0 and t <= 511 and 4e0 <= -57 + t + i and "
			"4e0 >= 58 + t - i and i >= 58 + t and i >= 62 - t)) }",
	  "{ [i0, i1] : (exists (e0 = [(4 + i0)/4]: 4e0 <= 62 + i0 - i1 and "
			"4e0 >= 1 + i0 and i0 >= 0 and i0 <= 511 and "
			"4e0 <= -57 + i0 + i1)) or "
		"(exists (e0 = [(2 + i0)/4]: 4e0 <= i0 and "
			"4e0 >= 58 + i0 - i1 and i0 >= 2 and i0 <= 511 and "
			"4e0 >= -61 + i0 + i1)) or "
		"(i1 <= 66 - i0 and i0 >= 2 and i1 >= 59 + i0) }", 1 },
	{ "[a, b] -> { : a = 0 and b = -1 }", "[b, a] -> { : b >= -10 }", 1 },
};

static int test_subset(isl_ctx *ctx)
{
	int i;
	isl_set *set1, *set2;
	int subset;

	if (test_subset_duplicate_integer_divisions(ctx) < 0)
		return -1;

	for (i = 0; i < ARRAY_SIZE(subset_tests); ++i) {
		set1 = isl_set_read_from_str(ctx, subset_tests[i].set1);
		set2 = isl_set_read_from_str(ctx, subset_tests[i].set2);
		subset = isl_set_is_subset(set1, set2);
		isl_set_free(set1);
		isl_set_free(set2);
		if (subset < 0)
			return -1;
		if (subset != subset_tests[i].subset)
			isl_die(ctx, isl_error_unknown,
				"incorrect subset result", return -1);
	}

	return 0;
}

/* Perform a set subtraction with a set that has a non-obviously empty disjunct.
 * Older versions of isl would fail on such cases.
 */
static isl_stat test_subtract_empty(isl_ctx *ctx)
{
	const char *str;
	isl_set *s1, *s2;

	s1 = isl_set_read_from_str(ctx, "{ [0] }");
	str = "{ [a] : (exists (e0, e1, e2: 1056e1 <= 32 + a - 33e0 and "
			"1089e1 >= a - 33e0 and 1089e1 <= 1 + a - 33e0 and "
			"33e2 >= -a + 33e0 + 1056e1 and "
			"33e2 < -2a + 66e0 + 2112e1)) or a = 0 }";
	s2 = isl_set_read_from_str(ctx, str);
	s1 = isl_set_subtract(s1, s2);
	isl_set_free(s1);

	return isl_stat_non_null(s1);
}

struct {
	const char *minuend;
	const char *subtrahend;
	const char *difference;
} subtract_domain_tests[] = {
	{ "{ A[i] -> B[i] }", "{ A[i] }", "{ }" },
	{ "{ A[i] -> B[i] }", "{ B[i] }", "{ A[i] -> B[i] }" },
	{ "{ A[i] -> B[i] }", "{ A[i] : i > 0 }", "{ A[i] -> B[i] : i <= 0 }" },
};

static int test_subtract(isl_ctx *ctx)
{
	int i;
	isl_union_map *umap1, *umap2;
	isl_union_pw_multi_aff *upma1, *upma2;
	isl_union_set *uset;
	int equal;

	if (test_subtract_empty(ctx) < 0)
		return -1;

	for (i = 0; i < ARRAY_SIZE(subtract_domain_tests); ++i) {
		umap1 = isl_union_map_read_from_str(ctx,
				subtract_domain_tests[i].minuend);
		uset = isl_union_set_read_from_str(ctx,
				subtract_domain_tests[i].subtrahend);
		umap2 = isl_union_map_read_from_str(ctx,
				subtract_domain_tests[i].difference);
		umap1 = isl_union_map_subtract_domain(umap1, uset);
		equal = isl_union_map_is_equal(umap1, umap2);
		isl_union_map_free(umap1);
		isl_union_map_free(umap2);
		if (equal < 0)
			return -1;
		if (!equal)
			isl_die(ctx, isl_error_unknown,
				"incorrect subtract domain result", return -1);
	}

	for (i = 0; i < ARRAY_SIZE(subtract_domain_tests); ++i) {
		upma1 = isl_union_pw_multi_aff_read_from_str(ctx,
				subtract_domain_tests[i].minuend);
		uset = isl_union_set_read_from_str(ctx,
				subtract_domain_tests[i].subtrahend);
		upma2 = isl_union_pw_multi_aff_read_from_str(ctx,
				subtract_domain_tests[i].difference);
		upma1 = isl_union_pw_multi_aff_subtract_domain(upma1, uset);
		equal = isl_union_pw_multi_aff_plain_is_equal(upma1, upma2);
		isl_union_pw_multi_aff_free(upma1);
		isl_union_pw_multi_aff_free(upma2);
		if (equal < 0)
			return -1;
		if (!equal)
			isl_die(ctx, isl_error_unknown,
				"incorrect subtract domain result", return -1);
	}

	return 0;
}

/* Check that intersecting the empty basic set with another basic set
 * does not increase the number of constraints.  In particular,
 * the empty basic set should maintain its canonical representation.
 */
static int test_intersect_1(isl_ctx *ctx)
{
	isl_size n1, n2;
	isl_basic_set *bset1, *bset2;

	bset1 = isl_basic_set_read_from_str(ctx, "{ [a,b,c] : 1 = 0 }");
	bset2 = isl_basic_set_read_from_str(ctx, "{ [1,2,3] }");
	n1 = isl_basic_set_n_constraint(bset1);
	bset1 = isl_basic_set_intersect(bset1, bset2);
	n2 = isl_basic_set_n_constraint(bset1);
	isl_basic_set_free(bset1);
	if (n1 < 0 || n2 < 0)
		return -1;
	if (n1 != n2)
		isl_die(ctx, isl_error_unknown,
			"number of constraints of empty set changed",
			return -1);

	return 0;
}

/* Check that intersecting a set with itself does not cause
 * an explosion in the number of disjuncts.
 */
static isl_stat test_intersect_2(isl_ctx *ctx)
{
	int i;
	isl_set *set;

	set = isl_set_read_from_str(ctx, "{ [x,y] : x >= 0 or y >= 0 }");
	for (i = 0; i < 100; ++i)
		set = isl_set_intersect(set, isl_set_copy(set));
	isl_set_free(set);
	if (!set)
		return isl_stat_error;
	return isl_stat_ok;
}

/* Perform some intersection tests.
 */
static int test_intersect(isl_ctx *ctx)
{
	if (test_intersect_1(ctx) < 0)
		return -1;
	if (test_intersect_2(ctx) < 0)
		return -1;

	return 0;
}

int test_factorize(isl_ctx *ctx)
{
	const char *str;
	isl_basic_set *bset;
	isl_factorizer *f;

	str = "{ [i0, i1, i2, i3, i4, i5, i6, i7] : 3i5 <= 2 - 2i0 and "
	    "i0 >= -2 and i6 >= 1 + i3 and i7 >= 0 and 3i5 >= -2i0 and "
	    "2i4 <= i2 and i6 >= 1 + 2i0 + 3i1 and i4 <= -1 and "
	    "i6 >= 1 + 2i0 + 3i5 and i6 <= 2 + 2i0 + 3i5 and "
	    "3i5 <= 2 - 2i0 - i2 + 3i4 and i6 <= 2 + 2i0 + 3i1 and "
	    "i0 <= -1 and i7 <= i2 + i3 - 3i4 - i6 and "
	    "3i5 >= -2i0 - i2 + 3i4 }";
	bset = isl_basic_set_read_from_str(ctx, str);
	f = isl_basic_set_factorizer(bset);
	isl_basic_set_free(bset);
	isl_factorizer_free(f);
	if (!f)
		isl_die(ctx, isl_error_unknown,
			"failed to construct factorizer", return -1);

	str = "{ [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12] : "
	    "i12 <= 2 + i0 - i11 and 2i8 >= -i4 and i11 >= i1 and "
	    "3i5 <= -i2 and 2i11 >= -i4 - 2i7 and i11 <= 3 + i0 + 3i9 and "
	    "i11 <= -i4 - 2i7 and i12 >= -i10 and i2 >= -2 and "
	    "i11 >= i1 + 3i10 and i11 >= 1 + i0 + 3i9 and "
	    "i11 <= 1 - i4 - 2i8 and 6i6 <= 6 - i2 and 3i6 >= 1 - i2 and "
	    "i11 <= 2 + i1 and i12 <= i4 + i11 and i12 >= i0 - i11 and "
	    "3i5 >= -2 - i2 and i12 >= -1 + i4 + i11 and 3i3 <= 3 - i2 and "
	    "9i6 <= 11 - i2 + 6i5 and 3i3 >= 1 - i2 and "
	    "9i6 <= 5 - i2 + 6i3 and i12 <= -1 and i2 <= 0 }";
	bset = isl_basic_set_read_from_str(ctx, str);
	f = isl_basic_set_factorizer(bset);
	isl_basic_set_free(bset);
	isl_factorizer_free(f);
	if (!f)
		isl_die(ctx, isl_error_unknown,
			"failed to construct factorizer", return -1);

	return 0;
}

static isl_stat check_injective(__isl_take isl_map *map, void *user)
{
	int *injective = user;

	*injective = isl_map_is_injective(map);
	isl_map_free(map);

	if (*injective < 0 || !*injective)
		return isl_stat_error;

	return isl_stat_ok;
}

int test_one_schedule(isl_ctx *ctx, const char *d, const char *w,
	const char *r, const char *s, int tilable, int parallel)
{
	int i;
	isl_union_set *D;
	isl_union_map *W, *R, *S;
	isl_union_map *empty;
	isl_union_map *dep_raw, *dep_war, *dep_waw, *dep;
	isl_union_map *validity, *proximity, *coincidence;
	isl_union_map *schedule;
	isl_union_map *test;
	isl_union_set *delta;
	isl_union_set *domain;
	isl_set *delta_set;
	isl_set *slice;
	isl_set *origin;
	isl_schedule_constraints *sc;
	isl_schedule *sched;
	int is_nonneg, is_parallel, is_tilable, is_injection, is_complete;
	isl_size n;

	D = isl_union_set_read_from_str(ctx, d);
	W = isl_union_map_read_from_str(ctx, w);
	R = isl_union_map_read_from_str(ctx, r);
	S = isl_union_map_read_from_str(ctx, s);

	W = isl_union_map_intersect_domain(W, isl_union_set_copy(D));
	R = isl_union_map_intersect_domain(R, isl_union_set_copy(D));

	empty = isl_union_map_empty(isl_union_map_get_space(S));
        isl_union_map_compute_flow(isl_union_map_copy(R),
				   isl_union_map_copy(W), empty,
				   isl_union_map_copy(S),
				   &dep_raw, NULL, NULL, NULL);
        isl_union_map_compute_flow(isl_union_map_copy(W),
				   isl_union_map_copy(W),
				   isl_union_map_copy(R),
				   isl_union_map_copy(S),
				   &dep_waw, &dep_war, NULL, NULL);

	dep = isl_union_map_union(dep_waw, dep_war);
	dep = isl_union_map_union(dep, dep_raw);
	validity = isl_union_map_copy(dep);
	coincidence = isl_union_map_copy(dep);
	proximity = isl_union_map_copy(dep);

	sc = isl_schedule_constraints_on_domain(isl_union_set_copy(D));
	sc = isl_schedule_constraints_set_validity(sc, validity);
	sc = isl_schedule_constraints_set_coincidence(sc, coincidence);
	sc = isl_schedule_constraints_set_proximity(sc, proximity);
	sched = isl_schedule_constraints_compute_schedule(sc);
	schedule = isl_schedule_get_map(sched);
	isl_schedule_free(sched);
	isl_union_map_free(W);
	isl_union_map_free(R);
	isl_union_map_free(S);

	is_injection = 1;
	isl_union_map_foreach_map(schedule, &check_injective, &is_injection);

	domain = isl_union_map_domain(isl_union_map_copy(schedule));
	is_complete = isl_union_set_is_subset(D, domain);
	isl_union_set_free(D);
	isl_union_set_free(domain);

	test = isl_union_map_reverse(isl_union_map_copy(schedule));
	test = isl_union_map_apply_range(test, dep);
	test = isl_union_map_apply_range(test, schedule);

	delta = isl_union_map_deltas(test);
	n = isl_union_set_n_set(delta);
	if (n < 0) {
		isl_union_set_free(delta);
		return -1;
	}
	if (n == 0) {
		is_tilable = 1;
		is_parallel = 1;
		is_nonneg = 1;
		isl_union_set_free(delta);
	} else {
		isl_size dim;

		delta_set = isl_set_from_union_set(delta);

		slice = isl_set_universe(isl_set_get_space(delta_set));
		for (i = 0; i < tilable; ++i)
			slice = isl_set_lower_bound_si(slice, isl_dim_set, i, 0);
		is_tilable = isl_set_is_subset(delta_set, slice);
		isl_set_free(slice);

		slice = isl_set_universe(isl_set_get_space(delta_set));
		for (i = 0; i < parallel; ++i)
			slice = isl_set_fix_si(slice, isl_dim_set, i, 0);
		is_parallel = isl_set_is_subset(delta_set, slice);
		isl_set_free(slice);

		origin = isl_set_universe(isl_set_get_space(delta_set));
		dim = isl_set_dim(origin, isl_dim_set);
		if (dim < 0)
			origin = isl_set_free(origin);
		for (i = 0; i < dim; ++i)
			origin = isl_set_fix_si(origin, isl_dim_set, i, 0);

		delta_set = isl_set_union(delta_set, isl_set_copy(origin));
		delta_set = isl_set_lexmin(delta_set);

		is_nonneg = isl_set_is_equal(delta_set, origin);

		isl_set_free(origin);
		isl_set_free(delta_set);
	}

	if (is_nonneg < 0 || is_parallel < 0 || is_tilable < 0 ||
	    is_injection < 0 || is_complete < 0)
		return -1;
	if (!is_complete)
		isl_die(ctx, isl_error_unknown,
			"generated schedule incomplete", return -1);
	if (!is_injection)
		isl_die(ctx, isl_error_unknown,
			"generated schedule not injective on each statement",
			return -1);
	if (!is_nonneg)
		isl_die(ctx, isl_error_unknown,
			"negative dependences in generated schedule",
			return -1);
	if (!is_tilable)
		isl_die(ctx, isl_error_unknown,
			"generated schedule not as tilable as expected",
			return -1);
	if (!is_parallel)
		isl_die(ctx, isl_error_unknown,
			"generated schedule not as parallel as expected",
			return -1);

	return 0;
}

/* Compute a schedule for the given instance set, validity constraints,
 * proximity constraints and context and return a corresponding union map
 * representation.
 */
static __isl_give isl_union_map *compute_schedule_with_context(isl_ctx *ctx,
	const char *domain, const char *validity, const char *proximity,
	const char *context)
{
	isl_set *con;
	isl_union_set *dom;
	isl_union_map *dep;
	isl_union_map *prox;
	isl_schedule_constraints *sc;
	isl_schedule *schedule;
	isl_union_map *sched;

	con = isl_set_read_from_str(ctx, context);
	dom = isl_union_set_read_from_str(ctx, domain);
	dep = isl_union_map_read_from_str(ctx, validity);
	prox = isl_union_map_read_from_str(ctx, proximity);
	sc = isl_schedule_constraints_on_domain(dom);
	sc = isl_schedule_constraints_set_context(sc, con);
	sc = isl_schedule_constraints_set_validity(sc, dep);
	sc = isl_schedule_constraints_set_proximity(sc, prox);
	schedule = isl_schedule_constraints_compute_schedule(sc);
	sched = isl_schedule_get_map(schedule);
	isl_schedule_free(schedule);

	return sched;
}

/* Compute a schedule for the given instance set, validity constraints and
 * proximity constraints and return a corresponding union map representation.
 */
static __isl_give isl_union_map *compute_schedule(isl_ctx *ctx,
	const char *domain, const char *validity, const char *proximity)
{
	return compute_schedule_with_context(ctx, domain, validity, proximity,
						"{ : }");
}

/* Check that a schedule can be constructed on the given domain
 * with the given validity and proximity constraints.
 */
static int test_has_schedule(isl_ctx *ctx, const char *domain,
	const char *validity, const char *proximity)
{
	isl_union_map *sched;

	sched = compute_schedule(ctx, domain, validity, proximity);
	if (!sched)
		return -1;

	isl_union_map_free(sched);
	return 0;
}

int test_special_schedule(isl_ctx *ctx, const char *domain,
	const char *validity, const char *proximity, const char *expected_sched)
{
	isl_union_map *sched1, *sched2;
	int equal;

	sched1 = compute_schedule(ctx, domain, validity, proximity);
	sched2 = isl_union_map_read_from_str(ctx, expected_sched);

	equal = isl_union_map_is_equal(sched1, sched2);
	isl_union_map_free(sched1);
	isl_union_map_free(sched2);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown, "unexpected schedule",
			return -1);

	return 0;
}

/* Check that the schedule map is properly padded, i.e., that the range
 * lives in a single space.
 */
static int test_padded_schedule(isl_ctx *ctx)
{
	const char *str;
	isl_union_set *D;
	isl_union_map *validity, *proximity;
	isl_schedule_constraints *sc;
	isl_schedule *sched;
	isl_union_map *umap;
	isl_union_set *range;
	isl_set *set;

	str = "[N] -> { S0[i] : 0 <= i <= N; S1[i, j] : 0 <= i, j <= N }";
	D = isl_union_set_read_from_str(ctx, str);
	validity = isl_union_map_empty(isl_union_set_get_space(D));
	proximity = isl_union_map_copy(validity);
	sc = isl_schedule_constraints_on_domain(D);
	sc = isl_schedule_constraints_set_validity(sc, validity);
	sc = isl_schedule_constraints_set_proximity(sc, proximity);
	sched = isl_schedule_constraints_compute_schedule(sc);
	umap = isl_schedule_get_map(sched);
	isl_schedule_free(sched);
	range = isl_union_map_range(umap);
	set = isl_set_from_union_set(range);
	isl_set_free(set);

	if (!set)
		return -1;

	return 0;
}

/* Check that conditional validity constraints are also taken into
 * account across bands.
 * In particular, try to make sure that live ranges D[1,0]->C[2,1] and
 * D[2,0]->C[3,0] are not local in the outer band of the generated schedule
 * and then check that the adjacent order constraint C[2,1]->D[2,0]
 * is enforced by the rest of the schedule.
 */
static int test_special_conditional_schedule_constraints(isl_ctx *ctx)
{
	const char *str;
	isl_union_set *domain;
	isl_union_map *validity, *proximity, *condition;
	isl_union_map *sink, *source, *dep;
	isl_schedule_constraints *sc;
	isl_schedule *schedule;
	isl_union_access_info *access;
	isl_union_flow *flow;
	int empty;

	str = "[n] -> { C[k, i] : k <= -1 + n and i >= 0 and i <= -1 + k; "
	    "A[k] : k >= 1 and k <= -1 + n; "
	    "B[k, i] : k <= -1 + n and i >= 0 and i <= -1 + k; "
	    "D[k, i] : k <= -1 + n and i >= 0 and i <= -1 + k }";
	domain = isl_union_set_read_from_str(ctx, str);
	sc = isl_schedule_constraints_on_domain(domain);
	str = "[n] -> { D[k, i] -> C[1 + k, k - i] : "
		"k <= -2 + n and i >= 1 and i <= -1 + k; "
		"D[k, i] -> C[1 + k, i] : "
		"k <= -2 + n and i >= 1 and i <= -1 + k; "
		"D[k, 0] -> C[1 + k, k] : k >= 1 and k <= -2 + n; "
		"D[k, 0] -> C[1 + k, 0] : k >= 1 and k <= -2 + n }";
	validity = isl_union_map_read_from_str(ctx, str);
	sc = isl_schedule_constraints_set_validity(sc, validity);
	str = "[n] -> { C[k, i] -> D[k, i] : "
		"0 <= i <= -1 + k and k <= -1 + n }";
	proximity = isl_union_map_read_from_str(ctx, str);
	sc = isl_schedule_constraints_set_proximity(sc, proximity);
	str = "[n] -> { [D[k, i] -> a[]] -> [C[1 + k, k - i] -> b[]] : "
		"i <= -1 + k and i >= 1 and k <= -2 + n; "
		"[B[k, i] -> c[]] -> [B[k, 1 + i] -> c[]] : "
		"k <= -1 + n and i >= 0 and i <= -2 + k }";
	condition = isl_union_map_read_from_str(ctx, str);
	str = "[n] -> { [B[k, i] -> e[]] -> [D[k, i] -> a[]] : "
		"i >= 0 and i <= -1 + k and k <= -1 + n; "
		"[C[k, i] -> b[]] -> [D[k', -1 + k - i] -> a[]] : "
		"i >= 0 and i <= -1 + k and k <= -1 + n and "
		"k' <= -1 + n and k' >= k - i and k' >= 1 + k; "
		"[C[k, i] -> b[]] -> [D[k, -1 + k - i] -> a[]] : "
		"i >= 0 and i <= -1 + k and k <= -1 + n; "
		"[B[k, i] -> c[]] -> [A[k'] -> d[]] : "
		"k <= -1 + n and i >= 0 and i <= -1 + k and "
		"k' >= 1 and k' <= -1 + n and k' >= 1 + k }";
	validity = isl_union_map_read_from_str(ctx, str);
	sc = isl_schedule_constraints_set_conditional_validity(sc, condition,
								validity);
	schedule = isl_schedule_constraints_compute_schedule(sc);
	str = "{ D[2,0] -> [] }";
	sink = isl_union_map_read_from_str(ctx, str);
	access = isl_union_access_info_from_sink(sink);
	str = "{ C[2,1] -> [] }";
	source = isl_union_map_read_from_str(ctx, str);
	access = isl_union_access_info_set_must_source(access, source);
	access = isl_union_access_info_set_schedule(access, schedule);
	flow = isl_union_access_info_compute_flow(access);
	dep = isl_union_flow_get_must_dependence(flow);
	isl_union_flow_free(flow);
	empty = isl_union_map_is_empty(dep);
	isl_union_map_free(dep);

	if (empty < 0)
		return -1;
	if (empty)
		isl_die(ctx, isl_error_unknown,
			"conditional validity not respected", return -1);

	return 0;
}

/* Check that the test for violated conditional validity constraints
 * is not confused by domain compression.
 * In particular, earlier versions of isl would apply
 * a schedule on the compressed domains to the original domains,
 * resulting in a failure to detect that the default schedule
 * violates the conditional validity constraints.
 */
static int test_special_conditional_schedule_constraints_2(isl_ctx *ctx)
{
	const char *str;
	isl_bool empty;
	isl_union_set *domain;
	isl_union_map *validity, *condition;
	isl_schedule_constraints *sc;
	isl_schedule *schedule;
	isl_union_map *umap;
	isl_map *map, *ge;

	str = "{ A[0, i] : 0 <= i <= 10; B[1, i] : 0 <= i <= 10 }";
	domain = isl_union_set_read_from_str(ctx, str);
	sc = isl_schedule_constraints_on_domain(domain);
	str = "{ B[1, i] -> A[0, i + 1] }";
	condition = isl_union_map_read_from_str(ctx, str);
	str = "{ A[0, i] -> B[1, i - 1] }";
	validity = isl_union_map_read_from_str(ctx, str);
	sc = isl_schedule_constraints_set_conditional_validity(sc, condition,
						isl_union_map_copy(validity));
	schedule = isl_schedule_constraints_compute_schedule(sc);
	umap = isl_schedule_get_map(schedule);
	isl_schedule_free(schedule);
	validity = isl_union_map_apply_domain(validity,
						isl_union_map_copy(umap));
	validity = isl_union_map_apply_range(validity, umap);
	map = isl_map_from_union_map(validity);
	ge = isl_map_lex_ge(isl_space_domain(isl_map_get_space(map)));
	map = isl_map_intersect(map, ge);
	empty = isl_map_is_empty(map);
	isl_map_free(map);

	if (empty < 0)
		return -1;
	if (!empty)
		isl_die(ctx, isl_error_unknown,
			"conditional validity constraints not satisfied",
			return -1);

	return 0;
}

/* Input for testing of schedule construction based on
 * conditional constraints.
 *
 * domain is the iteration domain
 * flow are the flow dependences, which determine the validity and
 * 	proximity constraints
 * condition are the conditions on the conditional validity constraints
 * conditional_validity are the conditional validity constraints
 * outer_band_n is the expected number of members in the outer band
 */
struct {
	const char *domain;
	const char *flow;
	const char *condition;
	const char *conditional_validity;
	int outer_band_n;
} live_range_tests[] = {
	/* Contrived example that illustrates that we need to keep
	 * track of tagged condition dependences and
	 * tagged conditional validity dependences
	 * in isl_sched_edge separately.
	 * In particular, the conditional validity constraints on A
	 * cannot be satisfied,
	 * but they can be ignored because there are no corresponding
	 * condition constraints.  However, we do have an additional
	 * conditional validity constraint that maps to the same
	 * dependence relation
	 * as the condition constraint on B.  If we did not make a distinction
	 * between tagged condition and tagged conditional validity
	 * dependences, then we
	 * could end up treating this shared dependence as an condition
	 * constraint on A, forcing a localization of the conditions,
	 * which is impossible.
	 */
	{ "{ S[i] : 0 <= 1 < 100; T[i] : 0 <= 1 < 100 }",
	  "{ S[i] -> S[i+1] : 0 <= i < 99 }",
	  "{ [S[i] -> B[]] -> [S[i+1] -> B[]] : 0 <= i < 99 }",
	  "{ [S[i] -> A[]] -> [T[i'] -> A[]] : 0 <= i', i < 100 and i != i';"
	    "[T[i] -> A[]] -> [S[i'] -> A[]] : 0 <= i', i < 100 and i != i';"
	    "[S[i] -> A[]] -> [S[i+1] -> A[]] : 0 <= i < 99 }",
	  1
	},
	/* TACO 2013 Fig. 7 */
	{ "[n] -> { S1[i,j] : 0 <= i,j < n; S2[i,j] : 0 <= i,j < n }",
	  "[n] -> { S1[i,j] -> S2[i,j] : 0 <= i,j < n;"
		   "S2[i,j] -> S2[i,j+1] : 0 <= i < n and 0 <= j < n - 1 }",
	  "[n] -> { [S1[i,j] -> t[]] -> [S2[i,j] -> t[]] : 0 <= i,j < n;"
		   "[S2[i,j] -> x1[]] -> [S2[i,j+1] -> x1[]] : "
				"0 <= i < n and 0 <= j < n - 1 }",
	  "[n] -> { [S2[i,j] -> t[]] -> [S1[i,j'] -> t[]] : "
				"0 <= i < n and 0 <= j < j' < n;"
		   "[S2[i,j] -> t[]] -> [S1[i',j'] -> t[]] : "
				"0 <= i < i' < n and 0 <= j,j' < n;"
		   "[S2[i,j] -> x1[]] -> [S2[i,j'] -> x1[]] : "
				"0 <= i,j,j' < n and j < j' }",
	    2
	},
	/* TACO 2013 Fig. 7, without tags */
	{ "[n] -> { S1[i,j] : 0 <= i,j < n; S2[i,j] : 0 <= i,j < n }",
	  "[n] -> { S1[i,j] -> S2[i,j] : 0 <= i,j < n;"
		   "S2[i,j] -> S2[i,j+1] : 0 <= i < n and 0 <= j < n - 1 }",
	  "[n] -> { S1[i,j] -> S2[i,j] : 0 <= i,j < n;"
		   "S2[i,j] -> S2[i,j+1] : 0 <= i < n and 0 <= j < n - 1 }",
	  "[n] -> { S2[i,j] -> S1[i,j'] : 0 <= i < n and 0 <= j < j' < n;"
		   "S2[i,j] -> S1[i',j'] : 0 <= i < i' < n and 0 <= j,j' < n;"
		   "S2[i,j] -> S2[i,j'] : 0 <= i,j,j' < n and j < j' }",
	   1
	},
	/* TACO 2013 Fig. 12 */
	{ "{ S1[i,0] : 0 <= i <= 1; S2[i,j] : 0 <= i <= 1 and 1 <= j <= 2;"
	    "S3[i,3] : 0 <= i <= 1 }",
	  "{ S1[i,0] -> S2[i,1] : 0 <= i <= 1;"
	    "S2[i,1] -> S2[i,2] : 0 <= i <= 1;"
	    "S2[i,2] -> S3[i,3] : 0 <= i <= 1 }",
	  "{ [S1[i,0]->t[]] -> [S2[i,1]->t[]] : 0 <= i <= 1;"
	    "[S2[i,1]->t[]] -> [S2[i,2]->t[]] : 0 <= i <= 1;"
	    "[S2[i,2]->t[]] -> [S3[i,3]->t[]] : 0 <= i <= 1 }",
	  "{ [S2[i,1]->t[]] -> [S2[i,2]->t[]] : 0 <= i <= 1;"
	    "[S2[0,j]->t[]] -> [S2[1,j']->t[]] : 1 <= j,j' <= 2;"
	    "[S2[0,j]->t[]] -> [S1[1,0]->t[]] : 1 <= j <= 2;"
	    "[S3[0,3]->t[]] -> [S2[1,j]->t[]] : 1 <= j <= 2;"
	    "[S3[0,3]->t[]] -> [S1[1,0]->t[]] }",
	   1
	}
};

/* Test schedule construction based on conditional constraints.
 * In particular, check the number of members in the outer band node
 * as an indication of whether tiling is possible or not.
 */
static int test_conditional_schedule_constraints(isl_ctx *ctx)
{
	int i;
	isl_union_set *domain;
	isl_union_map *condition;
	isl_union_map *flow;
	isl_union_map *validity;
	isl_schedule_constraints *sc;
	isl_schedule *schedule;
	isl_schedule_node *node;
	isl_size n_member;

	if (test_special_conditional_schedule_constraints(ctx) < 0)
		return -1;
	if (test_special_conditional_schedule_constraints_2(ctx) < 0)
		return -1;

	for (i = 0; i < ARRAY_SIZE(live_range_tests); ++i) {
		domain = isl_union_set_read_from_str(ctx,
				live_range_tests[i].domain);
		flow = isl_union_map_read_from_str(ctx,
				live_range_tests[i].flow);
		condition = isl_union_map_read_from_str(ctx,
				live_range_tests[i].condition);
		validity = isl_union_map_read_from_str(ctx,
				live_range_tests[i].conditional_validity);
		sc = isl_schedule_constraints_on_domain(domain);
		sc = isl_schedule_constraints_set_validity(sc,
				isl_union_map_copy(flow));
		sc = isl_schedule_constraints_set_proximity(sc, flow);
		sc = isl_schedule_constraints_set_conditional_validity(sc,
				condition, validity);
		schedule = isl_schedule_constraints_compute_schedule(sc);
		node = isl_schedule_get_root(schedule);
		while (node &&
		    isl_schedule_node_get_type(node) != isl_schedule_node_band)
			node = isl_schedule_node_first_child(node);
		n_member = isl_schedule_node_band_n_member(node);
		isl_schedule_node_free(node);
		isl_schedule_free(schedule);

		if (!schedule || n_member < 0)
			return -1;
		if (n_member != live_range_tests[i].outer_band_n)
			isl_die(ctx, isl_error_unknown,
				"unexpected number of members in outer band",
				return -1);
	}
	return 0;
}

/* Check that the schedule computed for the given instance set and
 * dependence relation strongly satisfies the dependences.
 * In particular, check that no instance is scheduled before
 * or together with an instance on which it depends.
 * Earlier versions of isl would produce a schedule that
 * only weakly satisfies the dependences.
 */
static int test_strongly_satisfying_schedule(isl_ctx *ctx)
{
	const char *domain, *dep;
	isl_union_map *D, *schedule;
	isl_map *map, *ge;
	int empty;

	domain = "{ B[i0, i1] : 0 <= i0 <= 1 and 0 <= i1 <= 11; "
		    "A[i0] : 0 <= i0 <= 1 }";
	dep = "{ B[i0, i1] -> B[i0, 1 + i1] : 0 <= i0 <= 1 and 0 <= i1 <= 10; "
		"B[0, 11] -> A[1]; A[i0] -> B[i0, 0] : 0 <= i0 <= 1 }";
	schedule = compute_schedule(ctx, domain, dep, dep);
	D = isl_union_map_read_from_str(ctx, dep);
	D = isl_union_map_apply_domain(D, isl_union_map_copy(schedule));
	D = isl_union_map_apply_range(D, schedule);
	map = isl_map_from_union_map(D);
	ge = isl_map_lex_ge(isl_space_domain(isl_map_get_space(map)));
	map = isl_map_intersect(map, ge);
	empty = isl_map_is_empty(map);
	isl_map_free(map);

	if (empty < 0)
		return -1;
	if (!empty)
		isl_die(ctx, isl_error_unknown,
			"dependences not strongly satisfied", return -1);

	return 0;
}

/* Compute a schedule for input where the instance set constraints
 * conflict with the context constraints.
 * Earlier versions of isl did not properly handle this situation.
 */
static int test_conflicting_context_schedule(isl_ctx *ctx)
{
	isl_union_map *schedule;
	const char *domain, *context;

	domain = "[n] -> { A[] : n >= 0 }";
	context = "[n] -> { : n < 0 }";
	schedule = compute_schedule_with_context(ctx,
						domain, "{}", "{}", context);
	isl_union_map_free(schedule);

	if (!schedule)
		return -1;

	return 0;
}

/* Check that a set of schedule constraints that only allow for
 * a coalescing schedule still produces a schedule even if the user
 * request a non-coalescing schedule.  Earlier versions of isl
 * would not handle this case correctly.
 */
static int test_coalescing_schedule(isl_ctx *ctx)
{
	const char *domain, *dep;
	isl_union_set *I;
	isl_union_map *D;
	isl_schedule_constraints *sc;
	isl_schedule *schedule;
	int treat_coalescing;

	domain = "{ S[a, b] : 0 <= a <= 1 and 0 <= b <= 1 }";
	dep = "{ S[a, b] -> S[a + b, 1 - b] }";
	I = isl_union_set_read_from_str(ctx, domain);
	D = isl_union_map_read_from_str(ctx, dep);
	sc = isl_schedule_constraints_on_domain(I);
	sc = isl_schedule_constraints_set_validity(sc, D);
	treat_coalescing = isl_options_get_schedule_treat_coalescing(ctx);
	isl_options_set_schedule_treat_coalescing(ctx, 1);
	schedule = isl_schedule_constraints_compute_schedule(sc);
	isl_options_set_schedule_treat_coalescing(ctx, treat_coalescing);
	isl_schedule_free(schedule);
	if (!schedule)
		return -1;
	return 0;
}

/* Check that the scheduler does not perform any needless
 * compound skewing.  Earlier versions of isl would compute
 * schedules in terms of transformed schedule coefficients and
 * would not accurately keep track of the sum of the original
 * schedule coefficients.  It could then produce the schedule
 * S[t,i,j,k] -> [t, 2t + i, 2t + i + j, 2t + i + j + k]
 * for the input below instead of the schedule below.
 */
static int test_skewing_schedule(isl_ctx *ctx)
{
	const char *D, *V, *P, *S;

	D = "[n] -> { S[t,i,j,k] : 0 <= t,i,j,k < n }";
	V = "[n] -> { S[t,i,j,k] -> S[t+1,a,b,c] : 0 <= t,i,j,k,a,b,c < n and "
		"-2 <= a-i <= 2 and -1 <= a-i + b-j <= 1 and "
		"-1 <= a-i + b-j + c-k <= 1 }";
	P = "{ }";
	S = "{ S[t,i,j,k] -> [t, 2t + i, t + i + j, 2t + k] }";

	return test_special_schedule(ctx, D, V, P, S);
}

int test_schedule(isl_ctx *ctx)
{
	const char *D, *W, *R, *V, *P, *S;
	int max_coincidence;
	int treat_coalescing;

	/* Handle resulting schedule with zero bands. */
	if (test_one_schedule(ctx, "{[]}", "{}", "{}", "{[] -> []}", 0, 0) < 0)
		return -1;

	/* Jacobi */
	D = "[T,N] -> { S1[t,i] : 1 <= t <= T and 2 <= i <= N - 1 }";
	W = "{ S1[t,i] -> a[t,i] }";
	R = "{ S1[t,i] -> a[t-1,i]; S1[t,i] -> a[t-1,i-1]; "
	    	"S1[t,i] -> a[t-1,i+1] }";
	S = "{ S1[t,i] -> [t,i] }";
	if (test_one_schedule(ctx, D, W, R, S, 2, 0) < 0)
		return -1;

	/* Fig. 5 of CC2008 */
	D = "[N] -> { S_0[i, j] : i >= 0 and i <= -1 + N and j >= 2 and "
				"j <= -1 + N }";
	W = "[N] -> { S_0[i, j] -> a[i, j] : i >= 0 and i <= -1 + N and "
				"j >= 2 and j <= -1 + N }";
	R = "[N] -> { S_0[i, j] -> a[j, i] : i >= 0 and i <= -1 + N and "
				"j >= 2 and j <= -1 + N; "
		    "S_0[i, j] -> a[i, -1 + j] : i >= 0 and i <= -1 + N and "
				"j >= 2 and j <= -1 + N }";
	S = "[N] -> { S_0[i, j] -> [0, i, 0, j, 0] }";
	if (test_one_schedule(ctx, D, W, R, S, 2, 0) < 0)
		return -1;

	D = "{ S1[i] : 0 <= i <= 10; S2[i] : 0 <= i <= 9 }";
	W = "{ S1[i] -> a[i] }";
	R = "{ S2[i] -> a[i+1] }";
	S = "{ S1[i] -> [0,i]; S2[i] -> [1,i] }";
	if (test_one_schedule(ctx, D, W, R, S, 1, 1) < 0)
		return -1;

	D = "{ S1[i] : 0 <= i < 10; S2[i] : 0 <= i < 10 }";
	W = "{ S1[i] -> a[i] }";
	R = "{ S2[i] -> a[9-i] }";
	S = "{ S1[i] -> [0,i]; S2[i] -> [1,i] }";
	if (test_one_schedule(ctx, D, W, R, S, 1, 1) < 0)
		return -1;

	D = "[N] -> { S1[i] : 0 <= i < N; S2[i] : 0 <= i < N }";
	W = "{ S1[i] -> a[i] }";
	R = "[N] -> { S2[i] -> a[N-1-i] }";
	S = "{ S1[i] -> [0,i]; S2[i] -> [1,i] }";
	if (test_one_schedule(ctx, D, W, R, S, 1, 1) < 0)
		return -1;
	
	D = "{ S1[i] : 0 < i < 10; S2[i] : 0 <= i < 10 }";
	W = "{ S1[i] -> a[i]; S2[i] -> b[i] }";
	R = "{ S2[i] -> a[i]; S1[i] -> b[i-1] }";
	S = "{ S1[i] -> [i,0]; S2[i] -> [i,1] }";
	if (test_one_schedule(ctx, D, W, R, S, 0, 0) < 0)
		return -1;

	D = "[N] -> { S1[i] : 1 <= i <= N; S2[i,j] : 1 <= i,j <= N }";
	W = "{ S1[i] -> a[0,i]; S2[i,j] -> a[i,j] }";
	R = "{ S2[i,j] -> a[i-1,j] }";
	S = "{ S1[i] -> [0,i,0]; S2[i,j] -> [1,i,j] }";
	if (test_one_schedule(ctx, D, W, R, S, 2, 1) < 0)
		return -1;

	D = "[N] -> { S1[i] : 1 <= i <= N; S2[i,j] : 1 <= i,j <= N }";
	W = "{ S1[i] -> a[i,0]; S2[i,j] -> a[i,j] }";
	R = "{ S2[i,j] -> a[i,j-1] }";
	S = "{ S1[i] -> [0,i,0]; S2[i,j] -> [1,i,j] }";
	if (test_one_schedule(ctx, D, W, R, S, 2, 1) < 0)
		return -1;

	D = "[N] -> { S_0[]; S_1[i] : i >= 0 and i <= -1 + N; S_2[] }";
	W = "[N] -> { S_0[] -> a[0]; S_2[] -> b[0]; "
		    "S_1[i] -> a[1 + i] : i >= 0 and i <= -1 + N }";
	R = "[N] -> { S_2[] -> a[N]; S_1[i] -> a[i] : i >= 0 and i <= -1 + N }";
	S = "[N] -> { S_1[i] -> [1, i, 0]; S_2[] -> [2, 0, 1]; "
		    "S_0[] -> [0, 0, 0] }";
	if (test_one_schedule(ctx, D, W, R, S, 1, 0) < 0)
		return -1;
	ctx->opt->schedule_parametric = 0;
	if (test_one_schedule(ctx, D, W, R, S, 0, 0) < 0)
		return -1;
	ctx->opt->schedule_parametric = 1;

	D = "[N] -> { S1[i] : 1 <= i <= N; S2[i] : 1 <= i <= N; "
		    "S3[i,j] : 1 <= i,j <= N; S4[i] : 1 <= i <= N }";
	W = "{ S1[i] -> a[i,0]; S2[i] -> a[0,i]; S3[i,j] -> a[i,j] }";
	R = "[N] -> { S3[i,j] -> a[i-1,j]; S3[i,j] -> a[i,j-1]; "
		    "S4[i] -> a[i,N] }";
	S = "{ S1[i] -> [0,i,0]; S2[i] -> [1,i,0]; S3[i,j] -> [2,i,j]; "
		"S4[i] -> [4,i,0] }";
	max_coincidence = isl_options_get_schedule_maximize_coincidence(ctx);
	isl_options_set_schedule_maximize_coincidence(ctx, 0);
	if (test_one_schedule(ctx, D, W, R, S, 2, 0) < 0)
		return -1;
	isl_options_set_schedule_maximize_coincidence(ctx, max_coincidence);

	D = "[N] -> { S_0[i, j] : i >= 1 and i <= N and j >= 1 and j <= N }";
	W = "[N] -> { S_0[i, j] -> s[0] : i >= 1 and i <= N and j >= 1 and "
					"j <= N }";
	R = "[N] -> { S_0[i, j] -> s[0] : i >= 1 and i <= N and j >= 1 and "
					"j <= N; "
		    "S_0[i, j] -> a[i, j] : i >= 1 and i <= N and j >= 1 and "
					"j <= N }";
	S = "[N] -> { S_0[i, j] -> [0, i, 0, j, 0] }";
	if (test_one_schedule(ctx, D, W, R, S, 0, 0) < 0)
		return -1;

	D = "[N] -> { S_0[t] : t >= 0 and t <= -1 + N; "
		    " S_2[t] : t >= 0 and t <= -1 + N; "
		    " S_1[t, i] : t >= 0 and t <= -1 + N and i >= 0 and "
				"i <= -1 + N }";
	W = "[N] -> { S_0[t] -> a[t, 0] : t >= 0 and t <= -1 + N; "
		    " S_2[t] -> b[t] : t >= 0 and t <= -1 + N; "
		    " S_1[t, i] -> a[t, 1 + i] : t >= 0 and t <= -1 + N and "
						"i >= 0 and i <= -1 + N }";
	R = "[N] -> { S_1[t, i] -> a[t, i] : t >= 0 and t <= -1 + N and "
					    "i >= 0 and i <= -1 + N; "
		    " S_2[t] -> a[t, N] : t >= 0 and t <= -1 + N }";
	S = "[N] -> { S_2[t] -> [0, t, 2]; S_1[t, i] -> [0, t, 1, i, 0]; "
		    " S_0[t] -> [0, t, 0] }";

	if (test_one_schedule(ctx, D, W, R, S, 2, 1) < 0)
		return -1;
	ctx->opt->schedule_parametric = 0;
	if (test_one_schedule(ctx, D, W, R, S, 0, 0) < 0)
		return -1;
	ctx->opt->schedule_parametric = 1;

	D = "[N] -> { S1[i,j] : 0 <= i,j < N; S2[i,j] : 0 <= i,j < N }";
	S = "{ S1[i,j] -> [0,i,j]; S2[i,j] -> [1,i,j] }";
	if (test_one_schedule(ctx, D, "{}", "{}", S, 2, 2) < 0)
		return -1;

	D = "[M, N] -> { S_1[i] : i >= 0 and i <= -1 + M; "
	    "S_0[i, j] : i >= 0 and i <= -1 + M and j >= 0 and j <= -1 + N }";
	W = "[M, N] -> { S_0[i, j] -> a[j] : i >= 0 and i <= -1 + M and "
					    "j >= 0 and j <= -1 + N; "
			"S_1[i] -> b[0] : i >= 0 and i <= -1 + M }";
	R = "[M, N] -> { S_0[i, j] -> a[0] : i >= 0 and i <= -1 + M and "
					    "j >= 0 and j <= -1 + N; "
			"S_1[i] -> b[0] : i >= 0 and i <= -1 + M }";
	S = "[M, N] -> { S_1[i] -> [1, i, 0]; S_0[i, j] -> [0, i, 0, j, 0] }";
	if (test_one_schedule(ctx, D, W, R, S, 0, 0) < 0)
		return -1;

	D = "{ S_0[i] : i >= 0 }";
	W = "{ S_0[i] -> a[i] : i >= 0 }";
	R = "{ S_0[i] -> a[0] : i >= 0 }";
	S = "{ S_0[i] -> [0, i, 0] }";
	if (test_one_schedule(ctx, D, W, R, S, 0, 0) < 0)
		return -1;

	D = "{ S_0[i] : i >= 0; S_1[i] : i >= 0 }";
	W = "{ S_0[i] -> a[i] : i >= 0; S_1[i] -> b[i] : i >= 0 }";
	R = "{ S_0[i] -> b[0] : i >= 0; S_1[i] -> a[i] : i >= 0 }";
	S = "{ S_1[i] -> [0, i, 1]; S_0[i] -> [0, i, 0] }";
	if (test_one_schedule(ctx, D, W, R, S, 0, 0) < 0)
		return -1;

	D = "[n] -> { S_0[j, k] : j <= -1 + n and j >= 0 and "
				"k <= -1 + n and k >= 0 }";
	W = "[n] -> { S_0[j, k] -> B[j] : j <= -1 + n and j >= 0 and "							"k <= -1 + n and k >= 0 }";
	R = "[n] -> { S_0[j, k] -> B[j] : j <= -1 + n and j >= 0 and "
					"k <= -1 + n and k >= 0; "
		    "S_0[j, k] -> B[k] : j <= -1 + n and j >= 0 and "
					"k <= -1 + n and k >= 0; "
		    "S_0[j, k] -> A[k] : j <= -1 + n and j >= 0 and "
					"k <= -1 + n and k >= 0 }";
	S = "[n] -> { S_0[j, k] -> [2, j, k] }";
	ctx->opt->schedule_outer_coincidence = 1;
	if (test_one_schedule(ctx, D, W, R, S, 0, 0) < 0)
		return -1;
	ctx->opt->schedule_outer_coincidence = 0;

	D = "{Stmt_for_body24[i0, i1, i2, i3]:"
		"i0 >= 0 and i0 <= 1 and i1 >= 0 and i1 <= 6 and i2 >= 2 and "
		"i2 <= 6 - i1 and i3 >= 0 and i3 <= -1 + i2;"
	     "Stmt_for_body24[i0, i1, 1, 0]:"
		"i0 >= 0 and i0 <= 1 and i1 >= 0 and i1 <= 5;"
	     "Stmt_for_body7[i0, i1, i2]:"
		"i0 >= 0 and i0 <= 1 and i1 >= 0 and i1 <= 7 and i2 >= 0 and "
		"i2 <= 7 }";

	V = "{Stmt_for_body24[0, i1, i2, i3] -> "
		"Stmt_for_body24[1, i1, i2, i3]:"
		"i3 >= 0 and i3 <= -1 + i2 and i1 >= 0 and i2 <= 6 - i1 and "
		"i2 >= 1;"
	     "Stmt_for_body24[0, i1, i2, i3] -> "
		"Stmt_for_body7[1, 1 + i1 + i3, 1 + i1 + i2]:"
		"i3 <= -1 + i2 and i2 <= 6 - i1 and i2 >= 1 and i1 >= 0 and "
		"i3 >= 0;"
	      "Stmt_for_body24[0, i1, i2, i3] ->"
		"Stmt_for_body7[1, i1, 1 + i1 + i3]:"
		"i3 >= 0 and i2 <= 6 - i1 and i1 >= 0 and i3 <= -1 + i2;"
	      "Stmt_for_body7[0, i1, i2] -> Stmt_for_body7[1, i1, i2]:"
		"(i2 >= 1 + i1 and i2 <= 6 and i1 >= 0 and i1 <= 4) or "
		"(i2 >= 3 and i2 <= 7 and i1 >= 1 and i2 >= 1 + i1) or "
		"(i2 >= 0 and i2 <= i1 and i2 >= -7 + i1 and i1 <= 7);"
	      "Stmt_for_body7[0, i1, 1 + i1] -> Stmt_for_body7[1, i1, 1 + i1]:"
		"i1 <= 6 and i1 >= 0;"
	      "Stmt_for_body7[0, 0, 7] -> Stmt_for_body7[1, 0, 7];"
	      "Stmt_for_body7[i0, i1, i2] -> "
		"Stmt_for_body24[i0, o1, -1 + i2 - o1, -1 + i1 - o1]:"
		"i0 >= 0 and i0 <= 1 and o1 >= 0 and i2 >= 1 + i1 and "
		"o1 <= -2 + i2 and i2 <= 7 and o1 <= -1 + i1;"
	      "Stmt_for_body7[i0, i1, i2] -> "
		"Stmt_for_body24[i0, i1, o2, -1 - i1 + i2]:"
		"i0 >= 0 and i0 <= 1 and i1 >= 0 and o2 >= -i1 + i2 and "
		"o2 >= 1 and o2 <= 6 - i1 and i2 >= 1 + i1 }";
	P = V;

	treat_coalescing = isl_options_get_schedule_treat_coalescing(ctx);
	isl_options_set_schedule_treat_coalescing(ctx, 0);
	if (test_has_schedule(ctx, D, V, P) < 0)
		return -1;
	isl_options_set_schedule_treat_coalescing(ctx, treat_coalescing);

	D = "{ S_0[i, j] : i >= 1 and i <= 10 and j >= 1 and j <= 8 }";
	V = "{ S_0[i, j] -> S_0[i, 1 + j] : i >= 1 and i <= 10 and "
					   "j >= 1 and j <= 7;"
		"S_0[i, j] -> S_0[1 + i, j] : i >= 1 and i <= 9 and "
					     "j >= 1 and j <= 8 }";
	P = "{ }";
	S = "{ S_0[i, j] -> [i + j, i] }";
	ctx->opt->schedule_algorithm = ISL_SCHEDULE_ALGORITHM_FEAUTRIER;
	if (test_special_schedule(ctx, D, V, P, S) < 0)
		return -1;
	ctx->opt->schedule_algorithm = ISL_SCHEDULE_ALGORITHM_ISL;

	/* Fig. 1 from Feautrier's "Some Efficient Solutions..." pt. 2, 1992 */
	D = "[N] -> { S_0[i, j] : i >= 0 and i <= -1 + N and "
				 "j >= 0 and j <= -1 + i }";
	V = "[N] -> { S_0[i, j] -> S_0[i, 1 + j] : j <= -2 + i and "
					"i <= -1 + N and j >= 0;"
		     "S_0[i, -1 + i] -> S_0[1 + i, 0] : i >= 1 and "
					"i <= -2 + N }";
	P = "{ }";
	S = "{ S_0[i, j] -> [i, j] }";
	ctx->opt->schedule_algorithm = ISL_SCHEDULE_ALGORITHM_FEAUTRIER;
	if (test_special_schedule(ctx, D, V, P, S) < 0)
		return -1;
	ctx->opt->schedule_algorithm = ISL_SCHEDULE_ALGORITHM_ISL;

	/* Test both algorithms on a case with only proximity dependences. */
	D = "{ S[i,j] : 0 <= i <= 10 }";
	V = "{ }";
	P = "{ S[i,j] -> S[i+1,j] : 0 <= i,j <= 10 }";
	S = "{ S[i, j] -> [j, i] }";
	ctx->opt->schedule_algorithm = ISL_SCHEDULE_ALGORITHM_FEAUTRIER;
	if (test_special_schedule(ctx, D, V, P, S) < 0)
		return -1;
	ctx->opt->schedule_algorithm = ISL_SCHEDULE_ALGORITHM_ISL;
	if (test_special_schedule(ctx, D, V, P, S) < 0)
		return -1;
	
	D = "{ A[a]; B[] }";
	V = "{}";
	P = "{ A[a] -> B[] }";
	if (test_has_schedule(ctx, D, V, P) < 0)
		return -1;

	if (test_padded_schedule(ctx) < 0)
		return -1;

	/* Check that check for progress is not confused by rational
	 * solution.
	 */
	D = "[N] -> { S0[i, j] : i >= 0 and i <= N and j >= 0 and j <= N }";
	V = "[N] -> { S0[i0, -1 + N] -> S0[2 + i0, 0] : i0 >= 0 and "
							"i0 <= -2 + N; "
			"S0[i0, i1] -> S0[i0, 1 + i1] : i0 >= 0 and "
				"i0 <= N and i1 >= 0 and i1 <= -1 + N }";
	P = "{}";
	ctx->opt->schedule_algorithm = ISL_SCHEDULE_ALGORITHM_FEAUTRIER;
	if (test_has_schedule(ctx, D, V, P) < 0)
		return -1;
	ctx->opt->schedule_algorithm = ISL_SCHEDULE_ALGORITHM_ISL;

	/* Check that we allow schedule rows that are only non-trivial
	 * on some full-dimensional domains.
	 */
	D = "{ S1[j] : 0 <= j <= 1; S0[]; S2[k] : 0 <= k <= 1 }";
	V = "{ S0[] -> S1[j] : 0 <= j <= 1; S2[0] -> S0[];"
		"S1[j] -> S2[1] : 0 <= j <= 1 }";
	P = "{}";
	ctx->opt->schedule_algorithm = ISL_SCHEDULE_ALGORITHM_FEAUTRIER;
	if (test_has_schedule(ctx, D, V, P) < 0)
		return -1;
	ctx->opt->schedule_algorithm = ISL_SCHEDULE_ALGORITHM_ISL;

	if (test_conditional_schedule_constraints(ctx) < 0)
		return -1;

	if (test_strongly_satisfying_schedule(ctx) < 0)
		return -1;

	if (test_conflicting_context_schedule(ctx) < 0)
		return -1;

	if (test_coalescing_schedule(ctx) < 0)
		return -1;
	if (test_skewing_schedule(ctx) < 0)
		return -1;

	return 0;
}

/* Perform scheduling tests using the whole component scheduler.
 */
static int test_schedule_whole(isl_ctx *ctx)
{
	int whole;
	int r;

	whole = isl_options_get_schedule_whole_component(ctx);
	isl_options_set_schedule_whole_component(ctx, 1);
	r = test_schedule(ctx);
	isl_options_set_schedule_whole_component(ctx, whole);

	return r;
}

/* Perform scheduling tests using the incremental scheduler.
 */
static int test_schedule_incremental(isl_ctx *ctx)
{
	int whole;
	int r;

	whole = isl_options_get_schedule_whole_component(ctx);
	isl_options_set_schedule_whole_component(ctx, 0);
	r = test_schedule(ctx);
	isl_options_set_schedule_whole_component(ctx, whole);

	return r;
}

int test_plain_injective(isl_ctx *ctx, const char *str, int injective)
{
	isl_union_map *umap;
	int test;

	umap = isl_union_map_read_from_str(ctx, str);
	test = isl_union_map_plain_is_injective(umap);
	isl_union_map_free(umap);
	if (test < 0)
		return -1;
	if (test == injective)
		return 0;
	if (injective)
		isl_die(ctx, isl_error_unknown,
			"map not detected as injective", return -1);
	else
		isl_die(ctx, isl_error_unknown,
			"map detected as injective", return -1);
}

int test_injective(isl_ctx *ctx)
{
	const char *str;

	if (test_plain_injective(ctx, "{S[i,j] -> A[0]; T[i,j] -> B[1]}", 0))
		return -1;
	if (test_plain_injective(ctx, "{S[] -> A[0]; T[] -> B[0]}", 1))
		return -1;
	if (test_plain_injective(ctx, "{S[] -> A[0]; T[] -> A[1]}", 1))
		return -1;
	if (test_plain_injective(ctx, "{S[] -> A[0]; T[] -> A[0]}", 0))
		return -1;
	if (test_plain_injective(ctx, "{S[i] -> A[i,0]; T[i] -> A[i,1]}", 1))
		return -1;
	if (test_plain_injective(ctx, "{S[i] -> A[i]; T[i] -> A[i]}", 0))
		return -1;
	if (test_plain_injective(ctx, "{S[] -> A[0,0]; T[] -> A[0,1]}", 1))
		return -1;
	if (test_plain_injective(ctx, "{S[] -> A[0,0]; T[] -> A[1,0]}", 1))
		return -1;

	str = "{S[] -> A[0,0]; T[] -> A[0,1]; U[] -> A[1,0]}";
	if (test_plain_injective(ctx, str, 1))
		return -1;
	str = "{S[] -> A[0,0]; T[] -> A[0,1]; U[] -> A[0,0]}";
	if (test_plain_injective(ctx, str, 0))
		return -1;

	return 0;
}

#undef BASE
#define BASE	aff
#include "isl_test_plain_equal_templ.c"

#undef BASE
#define BASE	pw_multi_aff
#include "isl_test_plain_equal_templ.c"

#undef BASE
#define BASE	union_pw_aff
#include "isl_test_plain_equal_templ.c"

/* Basic tests on isl_union_pw_aff.
 *
 * In particular, check that isl_union_pw_aff_aff_on_domain
 * aligns the parameters of the input objects and
 * that isl_union_pw_aff_param_on_domain_id properly
 * introduces the parameter.
 */
static int test_upa(isl_ctx *ctx)
{
	const char *str;
	isl_id *id;
	isl_aff *aff;
	isl_union_set *domain;
	isl_union_pw_aff *upa;
	isl_stat ok;

	aff = isl_aff_read_from_str(ctx, "[N] -> { [N] }");
	str = "[M] -> { A[i] : 0 <= i < M; B[] }";
	domain = isl_union_set_read_from_str(ctx, str);
	upa = isl_union_pw_aff_aff_on_domain(domain, aff);
	str = "[N, M] -> { A[i] -> [N] : 0 <= i < M; B[] -> [N] }";
	ok = union_pw_aff_check_plain_equal(upa, str);
	isl_union_pw_aff_free(upa);
	if (ok < 0)
		return -1;

	id = isl_id_alloc(ctx, "N", NULL);
	str = "[M] -> { A[i] : 0 <= i < M; B[] }";
	domain = isl_union_set_read_from_str(ctx, str);
	upa = isl_union_pw_aff_param_on_domain_id(domain, id);
	str = "[N, M] -> { A[i] -> [N] : 0 <= i < M; B[] -> [N] }";
	ok = union_pw_aff_check_plain_equal(upa, str);
	isl_union_pw_aff_free(upa);
	if (ok < 0)
		return -1;

	return 0;
}

struct {
	__isl_give isl_aff *(*fn)(__isl_take isl_aff *aff1,
				__isl_take isl_aff *aff2);
} aff_bin_op[] = {
	['+'] = { &isl_aff_add },
	['-'] = { &isl_aff_sub },
	['*'] = { &isl_aff_mul },
	['/'] = { &isl_aff_div },
};

struct {
	const char *arg1;
	unsigned char op;
	const char *arg2;
	const char *res;
} aff_bin_tests[] = {
	{ "{ [i] -> [i] }", '+', "{ [i] -> [i] }",
	  "{ [i] -> [2i] }" },
	{ "{ [i] -> [i] }", '-', "{ [i] -> [i] }",
	  "{ [i] -> [0] }" },
	{ "{ [i] -> [i] }", '*', "{ [i] -> [2] }",
	  "{ [i] -> [2i] }" },
	{ "{ [i] -> [2] }", '*', "{ [i] -> [i] }",
	  "{ [i] -> [2i] }" },
	{ "{ [i] -> [i] }", '/', "{ [i] -> [2] }",
	  "{ [i] -> [i/2] }" },
	{ "{ [i] -> [2i] }", '/', "{ [i] -> [2] }",
	  "{ [i] -> [i] }" },
	{ "{ [i] -> [i] }", '+', "{ [i] -> [NaN] }",
	  "{ [i] -> [NaN] }" },
	{ "{ [i] -> [i] }", '-', "{ [i] -> [NaN] }",
	  "{ [i] -> [NaN] }" },
	{ "{ [i] -> [i] }", '*', "{ [i] -> [NaN] }",
	  "{ [i] -> [NaN] }" },
	{ "{ [i] -> [2] }", '*', "{ [i] -> [NaN] }",
	  "{ [i] -> [NaN] }" },
	{ "{ [i] -> [i] }", '/', "{ [i] -> [NaN] }",
	  "{ [i] -> [NaN] }" },
	{ "{ [i] -> [2] }", '/', "{ [i] -> [NaN] }",
	  "{ [i] -> [NaN] }" },
	{ "{ [i] -> [NaN] }", '+', "{ [i] -> [i] }",
	  "{ [i] -> [NaN] }" },
	{ "{ [i] -> [NaN] }", '-', "{ [i] -> [i] }",
	  "{ [i] -> [NaN] }" },
	{ "{ [i] -> [NaN] }", '*', "{ [i] -> [2] }",
	  "{ [i] -> [NaN] }" },
	{ "{ [i] -> [NaN] }", '*', "{ [i] -> [i] }",
	  "{ [i] -> [NaN] }" },
	{ "{ [i] -> [NaN] }", '/', "{ [i] -> [2] }",
	  "{ [i] -> [NaN] }" },
	{ "{ [i] -> [NaN] }", '/', "{ [i] -> [i] }",
	  "{ [i] -> [NaN] }" },
	{ "{ [i] -> [i] }", '/', "{ [i] -> [0] }",
	  "{ [i] -> [NaN] }" },
};

/* Perform some basic tests of binary operations on isl_aff objects.
 */
static int test_bin_aff(isl_ctx *ctx)
{
	int i;
	isl_aff *aff1, *aff2, *res;
	__isl_give isl_aff *(*fn)(__isl_take isl_aff *aff1,
				__isl_take isl_aff *aff2);
	int ok;

	for (i = 0; i < ARRAY_SIZE(aff_bin_tests); ++i) {
		aff1 = isl_aff_read_from_str(ctx, aff_bin_tests[i].arg1);
		aff2 = isl_aff_read_from_str(ctx, aff_bin_tests[i].arg2);
		res = isl_aff_read_from_str(ctx, aff_bin_tests[i].res);
		fn = aff_bin_op[aff_bin_tests[i].op].fn;
		aff1 = fn(aff1, aff2);
		if (isl_aff_is_nan(res))
			ok = isl_aff_is_nan(aff1);
		else
			ok = isl_aff_plain_is_equal(aff1, res);
		isl_aff_free(aff1);
		isl_aff_free(res);
		if (ok < 0)
			return -1;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return -1);
	}

	return 0;
}

struct {
	__isl_give isl_pw_aff *(*fn)(__isl_take isl_pw_aff *pa1,
				     __isl_take isl_pw_aff *pa2);
} pw_aff_bin_op[] = {
	['m'] = { &isl_pw_aff_min },
	['M'] = { &isl_pw_aff_max },
};

/* Inputs for binary isl_pw_aff operation tests.
 * "arg1" and "arg2" are the two arguments, "op" identifies the operation
 * defined by pw_aff_bin_op, and "res" is the expected result.
 */
struct {
	const char *arg1;
	unsigned char op;
	const char *arg2;
	const char *res;
} pw_aff_bin_tests[] = {
	{ "{ [i] -> [i] }", 'm', "{ [i] -> [i] }",
	  "{ [i] -> [i] }" },
	{ "{ [i] -> [i] }", 'M', "{ [i] -> [i] }",
	  "{ [i] -> [i] }" },
	{ "{ [i] -> [i] }", 'm', "{ [i] -> [0] }",
	  "{ [i] -> [i] : i <= 0; [i] -> [0] : i > 0 }" },
	{ "{ [i] -> [i] }", 'M', "{ [i] -> [0] }",
	  "{ [i] -> [i] : i >= 0; [i] -> [0] : i < 0 }" },
	{ "{ [i] -> [i] }", 'm', "{ [i] -> [NaN] }",
	  "{ [i] -> [NaN] }" },
	{ "{ [i] -> [NaN] }", 'm', "{ [i] -> [i] }",
	  "{ [i] -> [NaN] }" },
};

/* Perform some basic tests of binary operations on isl_pw_aff objects.
 */
static int test_bin_pw_aff(isl_ctx *ctx)
{
	int i;
	isl_bool ok;
	isl_pw_aff *pa1, *pa2, *res;

	for (i = 0; i < ARRAY_SIZE(pw_aff_bin_tests); ++i) {
		pa1 = isl_pw_aff_read_from_str(ctx, pw_aff_bin_tests[i].arg1);
		pa2 = isl_pw_aff_read_from_str(ctx, pw_aff_bin_tests[i].arg2);
		res = isl_pw_aff_read_from_str(ctx, pw_aff_bin_tests[i].res);
		pa1 = pw_aff_bin_op[pw_aff_bin_tests[i].op].fn(pa1, pa2);
		if (isl_pw_aff_involves_nan(res))
			ok = isl_pw_aff_involves_nan(pa1);
		else
			ok = isl_pw_aff_plain_is_equal(pa1, res);
		isl_pw_aff_free(pa1);
		isl_pw_aff_free(res);
		if (ok < 0)
			return -1;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return -1);
	}

	return 0;
}

/* Inputs for basic tests of test operations on
 * isl_union_pw_multi_aff objects.
 * "fn" is the function that is being tested.
 * "arg" is a string description of the input.
 * "res" is the expected result.
 */
static struct {
	isl_bool (*fn)(__isl_keep isl_union_pw_multi_aff *upma1);
	const char *arg;
	isl_bool res;
} upma_test_tests[] = {
	{ &isl_union_pw_multi_aff_involves_nan, "{ A[] -> [0]; B[0] -> [1] }",
	  isl_bool_false },
	{ &isl_union_pw_multi_aff_involves_nan, "{ A[] -> [NaN]; B[0] -> [1] }",
	  isl_bool_true },
	{ &isl_union_pw_multi_aff_involves_nan, "{ A[] -> [0]; B[0] -> [NaN] }",
	  isl_bool_true },
	{ &isl_union_pw_multi_aff_involves_nan,
	  "{ A[] -> [0]; B[0] -> [1, NaN, 5] }",
	  isl_bool_true },
	{ &isl_union_pw_multi_aff_involves_locals,
	  "{ A[] -> [0]; B[0] -> [1] }",
	  isl_bool_false },
	{ &isl_union_pw_multi_aff_involves_locals,
	  "{ A[] -> [0]; B[x] -> [1] : x mod 2 = 0 }",
	  isl_bool_true },
	{ &isl_union_pw_multi_aff_involves_locals,
	  "{ A[] -> [0]; B[x] -> [x // 2] }",
	  isl_bool_true },
	{ &isl_union_pw_multi_aff_involves_locals,
	  "{ A[i] -> [i // 2]; B[0] -> [1] }",
	  isl_bool_true },
};

/* Perform some basic tests of test operations on
 * isl_union_pw_multi_aff objects.
 */
static isl_stat test_upma_test(isl_ctx *ctx)
{
	int i;
	isl_union_pw_multi_aff *upma;
	isl_bool res;

	for (i = 0; i < ARRAY_SIZE(upma_test_tests); ++i) {
		const char *str;

		str = upma_test_tests[i].arg;
		upma = isl_union_pw_multi_aff_read_from_str(ctx, str);
		res = upma_test_tests[i].fn(upma);
		isl_union_pw_multi_aff_free(upma);
		if (res < 0)
			return isl_stat_error;
		if (res != upma_test_tests[i].res)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return isl_stat_error);
	}

	return isl_stat_ok;
}

struct {
	__isl_give isl_union_pw_multi_aff *(*fn)(
		__isl_take isl_union_pw_multi_aff *upma1,
		__isl_take isl_union_pw_multi_aff *upma2);
	const char *arg1;
	const char *arg2;
	const char *res;
} upma_bin_tests[] = {
	{ &isl_union_pw_multi_aff_add, "{ A[] -> [0]; B[0] -> [1] }",
	  "{ B[x] -> [2] : x >= 0 }", "{ B[0] -> [3] }" },
	{ &isl_union_pw_multi_aff_union_add, "{ A[] -> [0]; B[0] -> [1] }",
	  "{ B[x] -> [2] : x >= 0 }",
	  "{ A[] -> [0]; B[0] -> [3]; B[x] -> [2] : x >= 1 }" },
	{ &isl_union_pw_multi_aff_pullback_union_pw_multi_aff,
	  "{ A[] -> B[0]; C[x] -> B[1] : x < 10; C[y] -> B[2] : y >= 10 }",
	  "{ D[i] -> A[] : i < 0; D[i] -> C[i + 5] : i >= 0 }",
	  "{ D[i] -> B[0] : i < 0; D[i] -> B[1] : 0 <= i < 5; "
	    "D[i] -> B[2] : i >= 5 }" },
	{ &isl_union_pw_multi_aff_union_add, "{ B[x] -> A[1] : x <= 0 }",
	  "{ B[x] -> C[2] : x > 0 }",
	  "{ B[x] -> A[1] : x <= 0; B[x] -> C[2] : x > 0 }" },
	{ &isl_union_pw_multi_aff_union_add, "{ B[x] -> A[1] : x <= 0 }",
	  "{ B[x] -> A[2] : x >= 0 }",
	  "{ B[x] -> A[1] : x < 0; B[x] -> A[2] : x > 0; B[0] -> A[3] }" },
	{
  &isl_union_pw_multi_aff_preimage_domain_wrapped_domain_union_pw_multi_aff,
	  "{ B[x] -> C[x + 2] }",
	  "{ D[y] -> B[2y] }",
	  "{ }" },
	{
  &isl_union_pw_multi_aff_preimage_domain_wrapped_domain_union_pw_multi_aff,
	  "{ [A[x] -> B[x + 1]] -> C[x + 2] }",
	  "{ D[y] -> B[2y] }",
	  "{ }" },
	{
  &isl_union_pw_multi_aff_preimage_domain_wrapped_domain_union_pw_multi_aff,
	  "{ [A[x] -> B[x + 1]] -> C[x + 2]; B[x] -> C[x + 2] }",
	  "{ D[y] -> A[2y] }",
	  "{ [D[y] -> B[2y + 1]] -> C[2y + 2] }" },
	{
  &isl_union_pw_multi_aff_preimage_domain_wrapped_domain_union_pw_multi_aff,
	  "{ T[A[x] -> B[x + 1]] -> C[x + 2]; B[x] -> C[x + 2] }",
	  "{ D[y] -> A[2y] }",
	  "{ T[D[y] -> B[2y + 1]] -> C[2y + 2] }" },
};

/* Perform some basic tests of binary operations on
 * isl_union_pw_multi_aff objects.
 */
static int test_bin_upma(isl_ctx *ctx)
{
	int i;
	isl_union_pw_multi_aff *upma1, *upma2, *res;
	int ok;

	for (i = 0; i < ARRAY_SIZE(upma_bin_tests); ++i) {
		upma1 = isl_union_pw_multi_aff_read_from_str(ctx,
							upma_bin_tests[i].arg1);
		upma2 = isl_union_pw_multi_aff_read_from_str(ctx,
							upma_bin_tests[i].arg2);
		res = isl_union_pw_multi_aff_read_from_str(ctx,
							upma_bin_tests[i].res);
		upma1 = upma_bin_tests[i].fn(upma1, upma2);
		ok = isl_union_pw_multi_aff_plain_is_equal(upma1, res);
		isl_union_pw_multi_aff_free(upma1);
		isl_union_pw_multi_aff_free(res);
		if (ok < 0)
			return -1;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return -1);
	}

	return 0;
}

struct {
	__isl_give isl_union_pw_multi_aff *(*fn)(
		__isl_take isl_union_pw_multi_aff *upma1,
		__isl_take isl_union_pw_multi_aff *upma2);
	const char *arg1;
	const char *arg2;
} upma_bin_fail_tests[] = {
	{ &isl_union_pw_multi_aff_union_add, "{ B[x] -> A[1] : x <= 0 }",
	  "{ B[x] -> C[2] : x >= 0 }" },
};

/* Perform some basic tests of binary operations on
 * isl_union_pw_multi_aff objects that are expected to fail.
 */
static int test_bin_upma_fail(isl_ctx *ctx)
{
	int i, n;
	isl_union_pw_multi_aff *upma1, *upma2;
	int on_error;

	on_error = isl_options_get_on_error(ctx);
	isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);
	n = ARRAY_SIZE(upma_bin_fail_tests);
	for (i = 0; i < n; ++i) {
		upma1 = isl_union_pw_multi_aff_read_from_str(ctx,
						upma_bin_fail_tests[i].arg1);
		upma2 = isl_union_pw_multi_aff_read_from_str(ctx,
						upma_bin_fail_tests[i].arg2);
		upma1 = upma_bin_fail_tests[i].fn(upma1, upma2);
		isl_union_pw_multi_aff_free(upma1);
		if (upma1)
			break;
	}
	isl_options_set_on_error(ctx, on_error);
	if (i < n)
		isl_die(ctx, isl_error_unknown,
			"operation not expected to succeed", return -1);

	return 0;
}

/* Inputs for basic tests of binary operations on
 * pairs of isl_union_pw_multi_aff and isl_union_set objects.
 * "fn" is the function that is being tested.
 * "arg1" and "arg2" are string descriptions of the inputs.
 * "res" is a string description of the expected result.
 */
struct {
	__isl_give isl_union_pw_multi_aff *(*fn)(
		__isl_take isl_union_pw_multi_aff *upma,
		__isl_take isl_union_set *uset);
	const char *arg1;
	const char *arg2;
	const char *res;
} upma_uset_tests[] = {
	{ &isl_union_pw_multi_aff_intersect_domain_wrapped_range,
	  "{ A[i] -> B[i] }", "{ B[0] }",
	  "{ }" },
	{ &isl_union_pw_multi_aff_intersect_domain_wrapped_domain,
	  "{ [A[i] -> B[i]] -> C[i + 1] }", "{ A[1]; B[0] }",
	  "{ [A[1] -> B[1]] -> C[2] }" },
	{ &isl_union_pw_multi_aff_intersect_domain_wrapped_range,
	  "{ [A[i] -> B[i]] -> C[i + 1] }", "{ A[1]; B[0] }",
	  "{ [A[0] -> B[0]] -> C[1] }" },
	{ &isl_union_pw_multi_aff_intersect_domain_wrapped_range,
	  "{ [A[i] -> B[i]] -> C[i + 1] }", "[N] -> { B[N] }",
	  "[N] -> { [A[N] -> B[N]] -> C[N + 1] }" },
	{ &isl_union_pw_multi_aff_intersect_domain_wrapped_range,
	  "[M] -> { [A[M] -> B[M]] -> C[M + 1] }", "[N] -> { B[N] }",
	  "[N, M] -> { [A[N] -> B[N]] -> C[N + 1] : N = M }" },
	{ &isl_union_pw_multi_aff_intersect_domain_wrapped_range,
	  "{ [A[] -> B[]] -> C[]; N[A[] -> B[]] -> D[]; [B[] -> A[]] -> E[] }",
	  "{ B[] }",
	  "{ [A[] -> B[]] -> C[]; N[A[] -> B[]] -> D[] }" },
};

/* Perform some basic tests of binary operations on
 * pairs of isl_union_pw_multi_aff and isl_union_set objects.
 */
static isl_stat test_upma_uset(isl_ctx *ctx)
{
	int i;
	isl_bool ok;
	isl_union_pw_multi_aff *upma, *res;
	isl_union_set *uset;

	for (i = 0; i < ARRAY_SIZE(upma_uset_tests); ++i) {
		upma = isl_union_pw_multi_aff_read_from_str(ctx,
						    upma_uset_tests[i].arg1);
		uset = isl_union_set_read_from_str(ctx,
						    upma_uset_tests[i].arg2);
		res = isl_union_pw_multi_aff_read_from_str(ctx,
						    upma_uset_tests[i].res);
		upma = upma_uset_tests[i].fn(upma, uset);
		ok = isl_union_pw_multi_aff_plain_is_equal(upma, res);
		isl_union_pw_multi_aff_free(upma);
		isl_union_pw_multi_aff_free(res);
		if (ok < 0)
			return isl_stat_error;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return isl_stat_error);
	}

	return isl_stat_ok;
}

/* Inputs for basic tests of unary operations on isl_multi_pw_aff objects.
 * "fn" is the function that is tested.
 * "arg" is a string description of the input.
 * "res" is a string description of the expected result.
 */
struct {
	__isl_give isl_multi_pw_aff *(*fn)(__isl_take isl_multi_pw_aff *mpa);
	const char *arg;
	const char *res;
} mpa_un_tests[] = {
	{ &isl_multi_pw_aff_range_factor_domain,
	  "{ A[x] -> [B[(1 : x >= 5)] -> C[(2 : x <= 10)]] }",
	  "{ A[x] -> B[(1 : x >= 5)] }" },
	{ &isl_multi_pw_aff_range_factor_range,
	  "{ A[x] -> [B[(1 : x >= 5)] -> C[(2 : x <= 10)]] }",
	  "{ A[y] -> C[(2 : y <= 10)] }" },
	{ &isl_multi_pw_aff_range_factor_domain,
	  "{ A[x] -> [B[(1 : x >= 5)] -> C[]] }",
	  "{ A[x] -> B[(1 : x >= 5)] }" },
	{ &isl_multi_pw_aff_range_factor_range,
	  "{ A[x] -> [B[(1 : x >= 5)] -> C[]] }",
	  "{ A[y] -> C[] }" },
	{ &isl_multi_pw_aff_range_factor_domain,
	  "{ A[x] -> [B[] -> C[(2 : x <= 10)]] }",
	  "{ A[x] -> B[] }" },
	{ &isl_multi_pw_aff_range_factor_range,
	  "{ A[x] -> [B[] -> C[(2 : x <= 10)]] }",
	  "{ A[y] -> C[(2 : y <= 10)] }" },
	{ &isl_multi_pw_aff_range_factor_domain,
	  "{ A[x] -> [B[] -> C[]] }",
	  "{ A[x] -> B[] }" },
	{ &isl_multi_pw_aff_range_factor_range,
	  "{ A[x] -> [B[] -> C[]] }",
	  "{ A[y] -> C[] }" },
	{ &isl_multi_pw_aff_factor_range,
	  "{ [B[] -> C[]] }",
	  "{ C[] }" },
	{ &isl_multi_pw_aff_range_factor_domain,
	  "{ A[x] -> [B[] -> C[]] : x >= 0 }",
	  "{ A[x] -> B[] : x >= 0 }" },
	{ &isl_multi_pw_aff_range_factor_range,
	  "{ A[x] -> [B[] -> C[]] : x >= 0 }",
	  "{ A[y] -> C[] : y >= 0 }" },
	{ &isl_multi_pw_aff_factor_range,
	  "[N] -> { [B[] -> C[]] : N >= 0 }",
	  "[N] -> { C[] : N >= 0 }" },
};

/* Perform some basic tests of unary operations on isl_multi_pw_aff objects.
 */
static int test_un_mpa(isl_ctx *ctx)
{
	int i;
	isl_bool ok;
	isl_multi_pw_aff *mpa, *res;

	for (i = 0; i < ARRAY_SIZE(mpa_un_tests); ++i) {
		mpa = isl_multi_pw_aff_read_from_str(ctx, mpa_un_tests[i].arg);
		res = isl_multi_pw_aff_read_from_str(ctx, mpa_un_tests[i].res);
		mpa = mpa_un_tests[i].fn(mpa);
		ok = isl_multi_pw_aff_plain_is_equal(mpa, res);
		isl_multi_pw_aff_free(mpa);
		isl_multi_pw_aff_free(res);
		if (ok < 0)
			return -1;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return -1);
	}

	return 0;
}

/* Inputs for basic tests of binary operations on isl_multi_pw_aff objects.
 * "fn" is the function that is tested.
 * "arg1" and "arg2" are string descriptions of the inputs.
 * "res" is a string description of the expected result.
 */
struct {
	__isl_give isl_multi_pw_aff *(*fn)(
		__isl_take isl_multi_pw_aff *mpa1,
		__isl_take isl_multi_pw_aff *mpa2);
	const char *arg1;
	const char *arg2;
	const char *res;
} mpa_bin_tests[] = {
	{ &isl_multi_pw_aff_add, "{ A[] -> [1] }", "{ A[] -> [2] }",
	  "{ A[] -> [3] }" },
	{ &isl_multi_pw_aff_add, "{ A[x] -> [(1 : x >= 5)] }",
	  "{ A[x] -> [(x : x <= 10)] }",
	  "{ A[x] -> [(1 + x : 5 <= x <= 10)] }" },
	{ &isl_multi_pw_aff_add, "{ A[x] -> [] : x >= 5 }",
	  "{ A[x] -> [] : x <= 10 }",
	  "{ A[x] -> [] : 5 <= x <= 10 }" },
	{ &isl_multi_pw_aff_add, "{ A[x] -> [] : x >= 5 }",
	  "[N] -> { A[x] -> [] : x <= N }",
	  "[N] -> { A[x] -> [] : 5 <= x <= N }" },
	{ &isl_multi_pw_aff_add,
	  "[N] -> { A[x] -> [] : x <= N }",
	  "{ A[x] -> [] : x >= 5 }",
	  "[N] -> { A[x] -> [] : 5 <= x <= N }" },
	{ &isl_multi_pw_aff_range_product, "{ A[x] -> B[(1 : x >= 5)] }",
	  "{ A[y] -> C[(2 : y <= 10)] }",
	  "{ A[x] -> [B[(1 : x >= 5)] -> C[(2 : x <= 10)]] }" },
	{ &isl_multi_pw_aff_range_product, "{ A[x] -> B[1] : x >= 5 }",
	  "{ A[y] -> C[2] : y <= 10 }",
	  "{ A[x] -> [B[(1 : x >= 5)] -> C[(2 : x <= 10)]] }" },
	{ &isl_multi_pw_aff_range_product, "{ A[x] -> B[1] : x >= 5 }",
	  "[N] -> { A[y] -> C[2] : y <= N }",
	  "[N] -> { A[x] -> [B[(1 : x >= 5)] -> C[(2 : x <= N)]] }" },
	{ &isl_multi_pw_aff_range_product, "[N] -> { A[x] -> B[1] : x >= N }",
	  "{ A[y] -> C[2] : y <= 10 }",
	  "[N] -> { A[x] -> [B[(1 : x >= N)] -> C[(2 : x <= 10)]] }" },
	{ &isl_multi_pw_aff_range_product, "{ A[] -> B[1] }", "{ A[] -> C[2] }",
	  "{ A[] -> [B[1] -> C[2]] }" },
	{ &isl_multi_pw_aff_range_product, "{ A[] -> B[] }", "{ A[] -> C[] }",
	  "{ A[] -> [B[] -> C[]] }" },
	{ &isl_multi_pw_aff_range_product, "{ A[x] -> B[(1 : x >= 5)] }",
	  "{ A[y] -> C[] : y <= 10 }",
	  "{ A[x] -> [B[(1 : x >= 5)] -> C[]] : x <= 10 }" },
	{ &isl_multi_pw_aff_range_product, "{ A[y] -> C[] : y <= 10 }",
	  "{ A[x] -> B[(1 : x >= 5)] }",
	  "{ A[x] -> [C[] -> B[(1 : x >= 5)]] : x <= 10 }" },
	{ &isl_multi_pw_aff_product, "{ A[x] -> B[(1 : x >= 5)] }",
	  "{ A[y] -> C[(2 : y <= 10)] }",
	  "{ [A[x] -> A[y]] -> [B[(1 : x >= 5)] -> C[(2 : y <= 10)]] }" },
	{ &isl_multi_pw_aff_product, "{ A[x] -> B[(1 : x >= 5)] }",
	  "{ A[y] -> C[] : y <= 10 }",
	  "{ [A[x] -> A[y]] -> [B[(1 : x >= 5)] -> C[]] : y <= 10 }" },
	{ &isl_multi_pw_aff_product, "{ A[y] -> C[] : y <= 10 }",
	  "{ A[x] -> B[(1 : x >= 5)] }",
	  "{ [A[y] -> A[x]] -> [C[] -> B[(1 : x >= 5)]] : y <= 10 }" },
	{ &isl_multi_pw_aff_product, "{ A[x] -> B[(1 : x >= 5)] }",
	  "[N] -> { A[y] -> C[] : y <= N }",
	  "[N] -> { [A[x] -> A[y]] -> [B[(1 : x >= 5)] -> C[]] : y <= N }" },
	{ &isl_multi_pw_aff_product, "[N] -> { A[y] -> C[] : y <= N }",
	  "{ A[x] -> B[(1 : x >= 5)] }",
	  "[N] -> { [A[y] -> A[x]] -> [C[] -> B[(1 : x >= 5)]] : y <= N }" },
	{ &isl_multi_pw_aff_product, "{ A[x] -> B[] : x >= 5 }",
	  "{ A[y] -> C[] : y <= 10 }",
	  "{ [A[x] -> A[y]] -> [B[] -> C[]] : x >= 5 and y <= 10 }" },
	{ &isl_multi_pw_aff_product, "{ A[] -> B[1] }", "{ A[] -> C[2] }",
	  "{ [A[] -> A[]] -> [B[1] -> C[2]] }" },
	{ &isl_multi_pw_aff_product, "{ A[] -> B[] }", "{ A[] -> C[] }",
	  "{ [A[] -> A[]] -> [B[] -> C[]] }" },
	{ &isl_multi_pw_aff_pullback_multi_pw_aff,
	  "{ B[i,j] -> C[i + 2j] }", "{ A[a,b] -> B[b,a] }",
	  "{ A[a,b] -> C[b + 2a] }" },
	{ &isl_multi_pw_aff_pullback_multi_pw_aff,
	  "{ B[i,j] -> C[i + 2j] }",
	  "{ A[a,b] -> B[(b : b > a),(a : b > a)] }",
	  "{ A[a,b] -> C[(b + 2a : b > a)] }" },
	{ &isl_multi_pw_aff_pullback_multi_pw_aff,
	  "{ B[i,j] -> C[(i + 2j : j > 4)] }",
	  "{ A[a,b] -> B[(b : b > a),(a : b > a)] }",
	  "{ A[a,b] -> C[(b + 2a : b > a > 4)] }" },
	{ &isl_multi_pw_aff_pullback_multi_pw_aff,
	  "{ B[i,j] -> C[] }",
	  "{ A[a,b] -> B[(b : b > a),(a : b > a)] }",
	  "{ A[a,b] -> C[] }" },
	{ &isl_multi_pw_aff_pullback_multi_pw_aff,
	  "{ B[i,j] -> C[] : i > j }",
	  "{ A[a,b] -> B[b,a] }",
	  "{ A[a,b] -> C[] : b > a }" },
	{ &isl_multi_pw_aff_pullback_multi_pw_aff,
	  "{ B[i,j] -> C[] : j > 5 }",
	  "{ A[a,b] -> B[(b : b > a),(a : b > a)] }",
	  "{ A[a,b] -> C[] : b > a > 5 }" },
	{ &isl_multi_pw_aff_pullback_multi_pw_aff,
	  "[N] -> { B[i,j] -> C[] : j > N }",
	  "{ A[a,b] -> B[(b : b > a),(a : b > a)] }",
	  "[N] -> { A[a,b] -> C[] : b > a > N }" },
	{ &isl_multi_pw_aff_pullback_multi_pw_aff,
	  "[M,N] -> { B[] -> C[] : N > 5 }",
	  "[M,N] -> { A[] -> B[] : M > N }",
	  "[M,N] -> { A[] -> C[] : M > N > 5 }" },
};

/* Perform some basic tests of binary operations on isl_multi_pw_aff objects.
 */
static int test_bin_mpa(isl_ctx *ctx)
{
	int i;
	isl_bool ok;
	isl_multi_pw_aff *mpa1, *mpa2, *res;

	for (i = 0; i < ARRAY_SIZE(mpa_bin_tests); ++i) {
		mpa1 = isl_multi_pw_aff_read_from_str(ctx,
							mpa_bin_tests[i].arg1);
		mpa2 = isl_multi_pw_aff_read_from_str(ctx,
							mpa_bin_tests[i].arg2);
		res = isl_multi_pw_aff_read_from_str(ctx,
							mpa_bin_tests[i].res);
		mpa1 = mpa_bin_tests[i].fn(mpa1, mpa2);
		ok = isl_multi_pw_aff_plain_is_equal(mpa1, res);
		isl_multi_pw_aff_free(mpa1);
		isl_multi_pw_aff_free(res);
		if (ok < 0)
			return -1;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return -1);
	}

	return 0;
}

/* Inputs for basic tests of unary operations on
 * isl_multi_union_pw_aff objects.
 * "fn" is the function that is tested.
 * "arg" is a string description of the input.
 * "res" is a string description of the expected result.
 */
struct {
	__isl_give isl_multi_union_pw_aff *(*fn)(
		__isl_take isl_multi_union_pw_aff *mupa);
	const char *arg;
	const char *res;
} mupa_un_tests[] = {
	{ &isl_multi_union_pw_aff_factor_range,
	  "[B[{ A[] -> [1] }] -> C[{ A[] -> [2] }]]",
	  "C[{ A[] -> [2] }]" },
	{ &isl_multi_union_pw_aff_factor_range,
	  "[B[] -> C[{ A[] -> [2] }]]",
	  "C[{ A[] -> [2] }]" },
	{ &isl_multi_union_pw_aff_factor_range,
	  "[B[{ A[] -> [1] }] -> C[]]",
	  "C[]" },
	{ &isl_multi_union_pw_aff_factor_range,
	  "[B[] -> C[]]",
	  "C[]" },
	{ &isl_multi_union_pw_aff_factor_range,
	  "([B[] -> C[]] : { A[x] : x >= 0 })",
	  "(C[] : { A[x] : x >= 0 })" },
	{ &isl_multi_union_pw_aff_factor_range,
	  "[N] -> ([B[] -> C[]] : { A[x] : x <= N })",
	  "[N] -> (C[] : { A[x] : x <= N })" },
	{ &isl_multi_union_pw_aff_factor_range,
	  "[N] -> ([B[] -> C[]] : { : N >= 0 })",
	  "[N] -> (C[] : { : N >= 0 })" },
};

/* Perform some basic tests of unary operations on
 * isl_multi_union_pw_aff objects.
 */
static int test_un_mupa(isl_ctx *ctx)
{
	int i;
	isl_bool ok;
	isl_multi_union_pw_aff *mupa, *res;

	for (i = 0; i < ARRAY_SIZE(mupa_un_tests); ++i) {
		mupa = isl_multi_union_pw_aff_read_from_str(ctx,
							mupa_un_tests[i].arg);
		res = isl_multi_union_pw_aff_read_from_str(ctx,
							mupa_un_tests[i].res);
		mupa = mupa_un_tests[i].fn(mupa);
		ok = isl_multi_union_pw_aff_plain_is_equal(mupa, res);
		isl_multi_union_pw_aff_free(mupa);
		isl_multi_union_pw_aff_free(res);
		if (ok < 0)
			return -1;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return -1);
	}

	return 0;
}

/* Inputs for basic tests of binary operations on
 * isl_multi_union_pw_aff objects.
 * "fn" is the function that is tested.
 * "arg1" and "arg2" are string descriptions of the inputs.
 * "res" is a string description of the expected result.
 */
struct {
	__isl_give isl_multi_union_pw_aff *(*fn)(
		__isl_take isl_multi_union_pw_aff *mupa1,
		__isl_take isl_multi_union_pw_aff *mupa2);
	const char *arg1;
	const char *arg2;
	const char *res;
} mupa_bin_tests[] = {
	{ &isl_multi_union_pw_aff_add, "[{ A[] -> [1] }]", "[{ A[] -> [2] }]",
	  "[{ A[] -> [3] }]" },
	{ &isl_multi_union_pw_aff_sub, "[{ A[] -> [1] }]", "[{ A[] -> [2] }]",
	  "[{ A[] -> [-1] }]" },
	{ &isl_multi_union_pw_aff_add,
	  "[{ A[] -> [1]; B[] -> [4] }]",
	  "[{ A[] -> [2]; C[] -> [5] }]",
	  "[{ A[] -> [3] }]" },
	{ &isl_multi_union_pw_aff_union_add,
	  "[{ A[] -> [1]; B[] -> [4] }]",
	  "[{ A[] -> [2]; C[] -> [5] }]",
	  "[{ A[] -> [3]; B[] -> [4]; C[] -> [5] }]" },
	{ &isl_multi_union_pw_aff_add, "[{ A[x] -> [(1)] : x >= 5 }]",
	  "[{ A[x] -> [(x)] : x <= 10 }]",
	  "[{ A[x] -> [(1 + x)] : 5 <= x <= 10 }]" },
	{ &isl_multi_union_pw_aff_add, "([] : { A[x] : x >= 5 })",
	  "([] : { A[x] : x <= 10 })",
	  "([] : { A[x] : 5 <= x <= 10 })" },
	{ &isl_multi_union_pw_aff_add, "([] : { A[x] : x >= 5 })",
	  "[N] -> ([] : { A[x] : x <= N })",
	  "[N] -> ([] : { A[x] : 5 <= x <= N })" },
	{ &isl_multi_union_pw_aff_add, "[N] -> ([] : { A[x] : x >= N })",
	  "([] : { A[x] : x <= 10 })",
	  "[N] -> ([] : { A[x] : N <= x <= 10 })" },
	{ &isl_multi_union_pw_aff_union_add, "[{ A[x] -> [(1)] : x >= 5 }]",
	  "[{ A[x] -> [(x)] : x <= 10 }]",
	  "[{ A[x] -> [(1 + x)] : 5 <= x <= 10; "
	     "A[x] -> [(1)] : x > 10; A[x] -> [(x)] : x < 5 }]" },
	{ &isl_multi_union_pw_aff_union_add, "([] : { A[x] : x >= 5 })",
	  "([] : { A[x] : x <= 10 })",
	  "([] : { A[x] })" },
	{ &isl_multi_union_pw_aff_union_add, "([] : { A[x] : x >= 0 })",
	  "[N] -> ([] : { A[x] : x >= N })",
	  "[N] -> ([] : { A[x] : x >= 0 or x >= N })" },
	{ &isl_multi_union_pw_aff_union_add,
	  "[N] -> ([] : { A[] : N >= 0})",
	  "[N] -> ([] : { A[] : N <= 0})",
	  "[N] -> ([] : { A[] })" },
	{ &isl_multi_union_pw_aff_union_add,
	  "[N] -> ([] : { A[] })",
	  "[N] -> ([] : { : })",
	  "[N] -> ([] : { : })" },
	{ &isl_multi_union_pw_aff_union_add,
	  "[N] -> ([] : { : })",
	  "[N] -> ([] : { A[] })",
	  "[N] -> ([] : { : })" },
	{ &isl_multi_union_pw_aff_union_add,
	  "[N] -> ([] : { : N >= 0})",
	  "[N] -> ([] : { : N <= 0})",
	  "[N] -> ([] : { : })" },
	{ &isl_multi_union_pw_aff_range_product,
	  "B[{ A[] -> [1] }]",
	  "C[{ A[] -> [2] }]",
	  "[B[{ A[] -> [1] }] -> C[{ A[] -> [2] }]]" },
	{ &isl_multi_union_pw_aff_range_product,
	  "(B[] : { A[x] : x >= 5 })",
	  "(C[] : { A[x] : x <= 10 })",
	  "([B[] -> C[]] : { A[x] : 5 <= x <= 10 })" },
	{ &isl_multi_union_pw_aff_range_product,
	  "B[{ A[x] -> [x + 1] : x >= 5 }]",
	  "(C[] : { A[x] : x <= 10 })",
	  "[B[{ A[x] -> [x + 1] : 5 <= x <= 10 }] -> C[]]" },
	{ &isl_multi_union_pw_aff_range_product,
	  "(C[] : { A[x] : x <= 10 })",
	  "B[{ A[x] -> [x + 1] : x >= 5 }]",
	  "[C[] -> B[{ A[x] -> [x + 1] : 5 <= x <= 10 }]]" },
	{ &isl_multi_union_pw_aff_range_product,
	  "B[{ A[x] -> [x + 1] : x >= 5 }]",
	  "[N] -> (C[] : { A[x] : x <= N })",
	  "[N] -> [B[{ A[x] -> [x + 1] : 5 <= x <= N }] -> C[]]" },
	{ &isl_multi_union_pw_aff_range_product,
	  "[N] -> (C[] : { A[x] : x <= N })",
	  "B[{ A[x] -> [x + 1] : x >= 5 }]",
	  "[N] -> [C[] -> B[{ A[x] -> [x + 1] : 5 <= x <= N }]]" },
	{ &isl_multi_union_pw_aff_range_product,
	  "B[{ A[] -> [1]; D[] -> [3] }]",
	  "C[{ A[] -> [2] }]",
	  "[B[{ A[] -> [1]; D[] -> [3] }] -> C[{ A[] -> [2] }]]" },
	{ &isl_multi_union_pw_aff_range_product,
	  "B[] }]",
	  "(C[] : { A[x] })",
	  "([B[] -> C[]] : { A[x] })" },
	{ &isl_multi_union_pw_aff_range_product,
	  "(B[] : { A[x] })",
	  "C[] }]",
	  "([B[] -> C[]] : { A[x] })" },
};

/* Perform some basic tests of binary operations on
 * isl_multi_union_pw_aff objects.
 */
static int test_bin_mupa(isl_ctx *ctx)
{
	int i;
	isl_bool ok;
	isl_multi_union_pw_aff *mupa1, *mupa2, *res;

	for (i = 0; i < ARRAY_SIZE(mupa_bin_tests); ++i) {
		mupa1 = isl_multi_union_pw_aff_read_from_str(ctx,
							mupa_bin_tests[i].arg1);
		mupa2 = isl_multi_union_pw_aff_read_from_str(ctx,
							mupa_bin_tests[i].arg2);
		res = isl_multi_union_pw_aff_read_from_str(ctx,
							mupa_bin_tests[i].res);
		mupa1 = mupa_bin_tests[i].fn(mupa1, mupa2);
		ok = isl_multi_union_pw_aff_plain_is_equal(mupa1, res);
		isl_multi_union_pw_aff_free(mupa1);
		isl_multi_union_pw_aff_free(res);
		if (ok < 0)
			return -1;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return -1);
	}

	return 0;
}

/* Inputs for basic tests of binary operations on
 * pairs of isl_multi_union_pw_aff and isl_set objects.
 * "fn" is the function that is tested.
 * "arg1" and "arg2" are string descriptions of the inputs.
 * "res" is a string description of the expected result.
 */
struct {
	__isl_give isl_multi_union_pw_aff *(*fn)(
		__isl_take isl_multi_union_pw_aff *mupa,
		__isl_take isl_set *set);
	const char *arg1;
	const char *arg2;
	const char *res;
} mupa_set_tests[] = {
	{ &isl_multi_union_pw_aff_intersect_range,
	  "C[{ B[i,j] -> [i + 2j] }]", "{ C[1] }",
	  "C[{ B[i,j] -> [i + 2j] : i + 2j = 1 }]" },
	{ &isl_multi_union_pw_aff_intersect_range,
	  "C[{ B[i,j] -> [i + 2j] }]", "[N] -> { C[N] }",
	  "[N] -> C[{ B[i,j] -> [i + 2j] : i + 2j = N }]" },
	{ &isl_multi_union_pw_aff_intersect_range,
	  "[N] -> C[{ B[i,j] -> [i + 2j + N] }]", "{ C[1] }",
	  "[N] -> C[{ B[i,j] -> [i + 2j + N] : i + 2j + N = 1 }]" },
	{ &isl_multi_union_pw_aff_intersect_range,
	  "C[{ B[i,j] -> [i + 2j] }]", "[N] -> { C[x] : N >= 0 }",
	  "[N] -> C[{ B[i,j] -> [i + 2j] : N >= 0 }]" },
	{ &isl_multi_union_pw_aff_intersect_range,
	  "C[]", "{ C[] }", "C[]" },
	{ &isl_multi_union_pw_aff_intersect_range,
	  "[N] -> (C[] : { : N >= 0 })",
	  "{ C[] }",
	  "[N] -> (C[] : { : N >= 0 })" },
	{ &isl_multi_union_pw_aff_intersect_range,
	  "(C[] : { A[a,b] })",
	  "{ C[] }",
	  "(C[] : { A[a,b] })" },
	{ &isl_multi_union_pw_aff_intersect_range,
	  "[N] -> (C[] : { A[a,b] : a,b <= N })",
	  "{ C[] }",
	  "[N] -> (C[] : { A[a,b] : a,b <= N })" },
	{ &isl_multi_union_pw_aff_intersect_range,
	  "C[]",
	  "[N] -> { C[] : N >= 0 }",
	  "[N] -> (C[] : { : N >= 0 })" },
	{ &isl_multi_union_pw_aff_intersect_range,
	  "(C[] : { A[a,b] })",
	  "[N] -> { C[] : N >= 0 }",
	  "[N] -> (C[] : { A[a,b] : N >= 0 })" },
	{ &isl_multi_union_pw_aff_intersect_range,
	  "[N] -> (C[] : { : N >= 0 })",
	  "[N] -> { C[] : N < 1024 }",
	  "[N] -> (C[] : { : 0 <= N < 1024 })" },
	{ &isl_multi_union_pw_aff_intersect_params,
	  "C[{ B[i,j] -> [i + 2j] }]", "[N] -> { : N >= 0 }",
	  "[N] -> C[{ B[i,j] -> [i + 2j] : N >= 0}]" },
	{ &isl_multi_union_pw_aff_intersect_params,
	  "[N] -> C[{ B[i,j] -> [i + 2j] : N <= 256 }]", "[N] -> { : N >= 0 }",
	  "[N] -> C[{ B[i,j] -> [i + 2j] : 0 <= N <= 256 }]" },
	{ &isl_multi_union_pw_aff_intersect_params,
	  "[N] -> C[{ B[i,j] -> [i + 2j] : N <= 256 }]", "{ : }",
	  "[N] -> C[{ B[i,j] -> [i + 2j] : N <= 256 }]" },
	{ &isl_multi_union_pw_aff_intersect_params,
	  "C[]", "[N] -> { : N >= 0 }",
	  "[N] -> (C[] : { : N >= 0 })" },
	{ &isl_multi_union_pw_aff_intersect_params,
	  "(C[] : { A[a,b] })", "[N] -> { : N >= 0 }",
	  "[N] -> (C[] : { A[a,b] : N >= 0 })" },
	{ &isl_multi_union_pw_aff_intersect_params,
	  "[N] -> (C[] : { A[a,N] })", "{ : }",
	  "[N] -> (C[] : { A[a,N] })" },
	{ &isl_multi_union_pw_aff_intersect_params,
	  "[N] -> (C[] : { A[a,b] : N <= 256 })", "[N] -> { : N >= 0 }",
	  "[N] -> (C[] : { A[a,b] : 0 <= N <= 256 })" },
};

/* Perform some basic tests of binary operations on
 * pairs of isl_multi_union_pw_aff and isl_set objects.
 */
static int test_mupa_set(isl_ctx *ctx)
{
	int i;
	isl_bool ok;
	isl_multi_union_pw_aff *mupa, *res;
	isl_set *set;

	for (i = 0; i < ARRAY_SIZE(mupa_set_tests); ++i) {
		mupa = isl_multi_union_pw_aff_read_from_str(ctx,
						    mupa_set_tests[i].arg1);
		set = isl_set_read_from_str(ctx, mupa_set_tests[i].arg2);
		res = isl_multi_union_pw_aff_read_from_str(ctx,
						    mupa_set_tests[i].res);
		mupa = mupa_set_tests[i].fn(mupa, set);
		ok = isl_multi_union_pw_aff_plain_is_equal(mupa, res);
		isl_multi_union_pw_aff_free(mupa);
		isl_multi_union_pw_aff_free(res);
		if (ok < 0)
			return -1;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return -1);
	}

	return 0;
}

/* Inputs for basic tests of binary operations on
 * pairs of isl_multi_union_pw_aff and isl_union_set objects.
 * "fn" is the function that is tested.
 * "arg1" and "arg2" are string descriptions of the inputs.
 * "res" is a string description of the expected result.
 */
struct {
	__isl_give isl_multi_union_pw_aff *(*fn)(
		__isl_take isl_multi_union_pw_aff *mupa,
		__isl_take isl_union_set *uset);
	const char *arg1;
	const char *arg2;
	const char *res;
} mupa_uset_tests[] = {
	{ &isl_multi_union_pw_aff_intersect_domain,
	  "C[{ B[i,j] -> [i + 2j] }]", "{ B[i,i] }",
	  "C[{ B[i,i] -> [3i] }]" },
	{ &isl_multi_union_pw_aff_intersect_domain,
	  "(C[] : { B[i,j] })", "{ B[i,i] }",
	  "(C[] : { B[i,i] })" },
	{ &isl_multi_union_pw_aff_intersect_domain,
	  "(C[] : { B[i,j] })", "[N] -> { B[N,N] }",
	  "[N] -> (C[] : { B[N,N] })" },
	{ &isl_multi_union_pw_aff_intersect_domain,
	  "C[]", "{ B[i,i] }",
	  "(C[] : { B[i,i] })" },
	{ &isl_multi_union_pw_aff_intersect_domain,
	  "[N] -> (C[] : { : N >= 0 })", "{ B[i,i] }",
	  "[N] -> (C[] : { B[i,i] : N >= 0 })" },
};

/* Perform some basic tests of binary operations on
 * pairs of isl_multi_union_pw_aff and isl_union_set objects.
 */
static int test_mupa_uset(isl_ctx *ctx)
{
	int i;
	isl_bool ok;
	isl_multi_union_pw_aff *mupa, *res;
	isl_union_set *uset;

	for (i = 0; i < ARRAY_SIZE(mupa_uset_tests); ++i) {
		mupa = isl_multi_union_pw_aff_read_from_str(ctx,
						    mupa_uset_tests[i].arg1);
		uset = isl_union_set_read_from_str(ctx,
						    mupa_uset_tests[i].arg2);
		res = isl_multi_union_pw_aff_read_from_str(ctx,
						    mupa_uset_tests[i].res);
		mupa = mupa_uset_tests[i].fn(mupa, uset);
		ok = isl_multi_union_pw_aff_plain_is_equal(mupa, res);
		isl_multi_union_pw_aff_free(mupa);
		isl_multi_union_pw_aff_free(res);
		if (ok < 0)
			return -1;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return -1);
	}

	return 0;
}

/* Inputs for basic tests of binary operations on
 * pairs of isl_multi_union_pw_aff and isl_multi_aff objects.
 * "fn" is the function that is tested.
 * "arg1" and "arg2" are string descriptions of the inputs.
 * "res" is a string description of the expected result.
 */
struct {
	__isl_give isl_multi_union_pw_aff *(*fn)(
		__isl_take isl_multi_union_pw_aff *mupa,
		__isl_take isl_multi_aff *ma);
	const char *arg1;
	const char *arg2;
	const char *res;
} mupa_ma_tests[] = {
	{ &isl_multi_union_pw_aff_apply_multi_aff,
	  "C[{ A[i,j] -> [i]; B[i,j] -> [j] }, "
	    "{ A[i,j] -> [j]; B[i,j] -> [i] }]",
	  "{ C[a,b] -> D[b,a] }",
	  "D[{ A[i,j] -> [j]; B[i,j] -> [i] }, "
	    "{ A[i,j] -> [i]; B[i,j] -> [j] }]" },
	{ &isl_multi_union_pw_aff_apply_multi_aff,
	  "C[{ A[i,j] -> [i] : i >= 0; B[i,j] -> [j] }, "
	    "{ A[i,j] -> [j]; B[i,j] -> [i] }]",
	  "{ C[a,b] -> D[b,a] }",
	  "D[{ A[i,j] -> [j] : i >= 0; B[i,j] -> [i] }, "
	    "{ A[i,j] -> [i] : i >= 0; B[i,j] -> [j] }]" },
	{ &isl_multi_union_pw_aff_apply_multi_aff,
	  "C[{ A[i,j] -> [i]; B[i,j] -> [j] }]",
	  "[N] -> { C[a] -> D[a + N] }",
	  "[N] -> D[{ A[i,j] -> [i + N]; B[i,j] -> [j + N] }] " },
	{ &isl_multi_union_pw_aff_apply_multi_aff,
	  "C[]",
	  "{ C[] -> D[] }",
	  "D[]" },
	{ &isl_multi_union_pw_aff_apply_multi_aff,
	  "[N] -> (C[] : { : N >= 0 })",
	  "{ C[] -> D[] }",
	  "[N] -> (D[] : { : N >= 0 })" },
	{ &isl_multi_union_pw_aff_apply_multi_aff,
	  "C[]",
	  "[N] -> { C[] -> D[N] }",
	  "[N] -> D[{ [N] }]" },
	{ &isl_multi_union_pw_aff_apply_multi_aff,
	  "(C[] : { A[i,j] : i >= j })",
	  "{ C[] -> D[] }",
	  "(D[] : { A[i,j] : i >= j })" },
	{ &isl_multi_union_pw_aff_apply_multi_aff,
	  "[N] -> (C[] : { A[i,j] : N >= 0 })",
	  "{ C[] -> D[] }",
	  "[N] -> (D[] : { A[i,j] : N >= 0 })" },
	{ &isl_multi_union_pw_aff_apply_multi_aff,
	  "(C[] : { A[i,j] : i >= j })",
	  "[N] -> { C[] -> D[N] }",
	  "[N] -> (D[{ A[i,j] -> [N] : i >= j }])" },
};

/* Perform some basic tests of binary operations on
 * pairs of isl_multi_union_pw_aff and isl_multi_aff objects.
 */
static int test_mupa_ma(isl_ctx *ctx)
{
	int i;
	isl_bool ok;
	isl_multi_union_pw_aff *mupa, *res;
	isl_multi_aff *ma;

	for (i = 0; i < ARRAY_SIZE(mupa_ma_tests); ++i) {
		mupa = isl_multi_union_pw_aff_read_from_str(ctx,
						    mupa_ma_tests[i].arg1);
		ma = isl_multi_aff_read_from_str(ctx, mupa_ma_tests[i].arg2);
		res = isl_multi_union_pw_aff_read_from_str(ctx,
						    mupa_ma_tests[i].res);
		mupa = mupa_ma_tests[i].fn(mupa, ma);
		ok = isl_multi_union_pw_aff_plain_is_equal(mupa, res);
		isl_multi_union_pw_aff_free(mupa);
		isl_multi_union_pw_aff_free(res);
		if (ok < 0)
			return -1;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return -1);
	}

	return 0;
}

/* Inputs for basic tests of binary operations on
 * pairs of isl_multi_union_pw_aff and isl_pw_aff objects.
 * "fn" is the function that is tested.
 * "arg1" and "arg2" are string descriptions of the inputs.
 * "res" is a string description of the expected result.
 */
struct {
	__isl_give isl_union_pw_aff *(*fn)(
		__isl_take isl_multi_union_pw_aff *mupa,
		__isl_take isl_pw_aff *pa);
	const char *arg1;
	const char *arg2;
	const char *res;
} mupa_pa_tests[] = {
	{ &isl_multi_union_pw_aff_apply_pw_aff,
	  "C[{ A[i,j] -> [i]; B[i,j] -> [j] }]",
	  "[N] -> { C[a] -> [a + N] }",
	  "[N] -> { A[i,j] -> [i + N]; B[i,j] -> [j + N] }" },
	{ &isl_multi_union_pw_aff_apply_pw_aff,
	  "C[{ A[i,j] -> [i]; B[i,j] -> [j] }]",
	  "{ C[a] -> [a] : a >= 0; C[a] -> [-a] : a < 0 }",
	  "{ A[i,j] -> [i] : i >= 0; A[i,j] -> [-i] : i < 0; "
	    "B[i,j] -> [j] : j >= 0; B[i,j] -> [-j] : j < 0 }" },
	{ &isl_multi_union_pw_aff_apply_pw_aff,
	  "C[]",
	  "[N] -> { C[] -> [N] }",
	  "[N] -> { [N] }" },
	{ &isl_multi_union_pw_aff_apply_pw_aff,
	  "C[]",
	  "[N] -> { C[] -> [N] : N >= 0; C[] -> [-N] : N < 0 }",
	  "[N] -> { [N] : N >= 0; [-N] : N < 0 }" },
	{ &isl_multi_union_pw_aff_apply_pw_aff,
	  "[N] -> (C[] : { : N >= 0 })",
	  "[N] -> { C[] -> [N] }",
	  "[N] -> { [N] : N >= 0 }" },
	{ &isl_multi_union_pw_aff_apply_pw_aff,
	  "[N] -> (C[] : { : N >= 0 })",
	  "[N] -> { C[] -> [N] : N >= 0; C[] -> [-N] : N < 0 }",
	  "[N] -> { [N] : N >= 0 }" },
	{ &isl_multi_union_pw_aff_apply_pw_aff,
	  "[N] -> (C[] : { : N >= 0 })",
	  "{ C[] -> [0] }",
	  "[N] -> { [0] : N >= 0 }" },
	{ &isl_multi_union_pw_aff_apply_pw_aff,
	  "(C[] : { A[i,j] : i >= j })",
	  "[N] -> { C[] -> [N] }",
	  "[N] -> { A[i,j] -> [N] : i >= j }" },
	{ &isl_multi_union_pw_aff_apply_pw_aff,
	  "(C[] : { A[i,j] : i >= j })",
	  "[N] -> { C[] -> [N] : N >= 0 }",
	  "[N] -> { A[i,j] -> [N] : i >= j and N >= 0 }" },
};

/* Perform some basic tests of binary operations on
 * pairs of isl_multi_union_pw_aff and isl_pw_aff objects.
 */
static int test_mupa_pa(isl_ctx *ctx)
{
	int i;
	isl_bool ok;
	isl_multi_union_pw_aff *mupa;
	isl_union_pw_aff *upa, *res;
	isl_pw_aff *pa;

	for (i = 0; i < ARRAY_SIZE(mupa_pa_tests); ++i) {
		mupa = isl_multi_union_pw_aff_read_from_str(ctx,
						    mupa_pa_tests[i].arg1);
		pa = isl_pw_aff_read_from_str(ctx, mupa_pa_tests[i].arg2);
		res = isl_union_pw_aff_read_from_str(ctx,
						    mupa_pa_tests[i].res);
		upa = mupa_pa_tests[i].fn(mupa, pa);
		ok = isl_union_pw_aff_plain_is_equal(upa, res);
		isl_union_pw_aff_free(upa);
		isl_union_pw_aff_free(res);
		if (ok < 0)
			return -1;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return -1);
	}

	return 0;
}

/* Inputs for basic tests of binary operations on
 * pairs of isl_multi_union_pw_aff and isl_pw_multi_aff objects.
 * "fn" is the function that is tested.
 * "arg1" and "arg2" are string descriptions of the inputs.
 * "res" is a string description of the expected result.
 */
struct {
	__isl_give isl_multi_union_pw_aff *(*fn)(
		__isl_take isl_multi_union_pw_aff *mupa,
		__isl_take isl_pw_multi_aff *pma);
	const char *arg1;
	const char *arg2;
	const char *res;
} mupa_pma_tests[] = {
	{ &isl_multi_union_pw_aff_apply_pw_multi_aff,
	  "C[{ A[i,j] -> [i]; B[i,j] -> [j] }, "
	    "{ A[i,j] -> [j]; B[i,j] -> [i] }]",
	  "{ C[a,b] -> D[b,a] }",
	  "D[{ A[i,j] -> [j]; B[i,j] -> [i] }, "
	    "{ A[i,j] -> [i]; B[i,j] -> [j] }]" },
	{ &isl_multi_union_pw_aff_apply_pw_multi_aff,
	  "C[{ A[i,j] -> [i] : i >= 0; B[i,j] -> [j] }, "
	    "{ A[i,j] -> [j]; B[i,j] -> [i] }]",
	  "{ C[a,b] -> D[b,a] }",
	  "D[{ A[i,j] -> [j] : i >= 0; B[i,j] -> [i] }, "
	    "{ A[i,j] -> [i] : i >= 0; B[i,j] -> [j] }]" },
	{ &isl_multi_union_pw_aff_apply_pw_multi_aff,
	  "C[{ A[i,j] -> [i]; B[i,j] -> [j] }]",
	  "[N] -> { C[a] -> D[a + N] }",
	  "[N] -> D[{ A[i,j] -> [i + N]; B[i,j] -> [j + N] }]" },
	{ &isl_multi_union_pw_aff_apply_pw_multi_aff,
	  "C[{ A[i,j] -> [i]; B[i,j] -> [j] }]",
	  "{ C[a] -> D[a] : a >= 0; C[a] -> D[-a] : a < 0 }",
	  "D[{ A[i,j] -> [i] : i >= 0; A[i,j] -> [-i] : i < 0; "
	      "B[i,j] -> [j] : j >= 0; B[i,j] -> [-j] : j < 0 }]" },
	{ &isl_multi_union_pw_aff_apply_pw_multi_aff,
	  "C[{ A[i,j] -> [i]; B[i,j] -> [j] }, "
	    "{ A[i,j] -> [j]; B[i,j] -> [i] }]",
	  "{ C[a,b] -> D[a,b] : a >= b; C[a,b] -> D[b,a] : a < b }",
	  "D[{ A[i,j] -> [i] : i >= j; A[i,j] -> [j] : i < j; "
	      "B[i,j] -> [j] : i <= j; B[i,j] -> [i] : i > j }, "
	    "{ A[i,j] -> [j] : i >= j; A[i,j] -> [i] : i < j; "
	      "B[i,j] -> [i] : i <= j; B[i,j] -> [j] : i > j }]" },
	{ &isl_multi_union_pw_aff_apply_pw_multi_aff,
	  "C[]",
	  "{ C[] -> D[] }",
	  "D[]" },
	{ &isl_multi_union_pw_aff_apply_pw_multi_aff,
	  "[N] -> (C[] : { : N >= 0 })",
	  "{ C[] -> D[] }",
	  "[N] -> (D[] : { : N >= 0 })" },
	{ &isl_multi_union_pw_aff_apply_pw_multi_aff,
	  "C[]",
	  "[N] -> { C[] -> D[N] }",
	  "[N] -> D[{ [N] }]" },
	{ &isl_multi_union_pw_aff_apply_pw_multi_aff,
	  "(C[] : { A[i,j] : i >= j })",
	  "{ C[] -> D[] }",
	  "(D[] : { A[i,j] : i >= j })" },
	{ &isl_multi_union_pw_aff_apply_pw_multi_aff,
	  "[N] -> (C[] : { A[i,j] : N >= 0 })",
	  "{ C[] -> D[] }",
	  "[N] -> (D[] : { A[i,j] : N >= 0 })" },
	{ &isl_multi_union_pw_aff_apply_pw_multi_aff,
	  "(C[] : { A[i,j] : i >= j })",
	  "[N] -> { C[] -> D[N] }",
	  "[N] -> (D[{ A[i,j] -> [N] : i >= j }])" },
	{ &isl_multi_union_pw_aff_apply_pw_multi_aff,
	  "C[]",
	  "[N] -> { C[] -> D[N] : N >= 0; C[] -> D[-N] : N < 0 }",
	  "[N] -> D[{ [N] : N >= 0; [-N] : N < 0 }]" },
	{ &isl_multi_union_pw_aff_apply_pw_multi_aff,
	  "[N] -> (C[] : { : N >= 0 })",
	  "[N] -> { C[] -> D[N] }",
	  "[N] -> D[{ [N] : N >= 0 }]" },
	{ &isl_multi_union_pw_aff_apply_pw_multi_aff,
	  "[N] -> (C[] : { : N >= 0 })",
	  "[N] -> { C[] -> D[N] : N >= 0; C[] -> D[-N] : N < 0 }",
	  "[N] -> D[{ [N] : N >= 0 }]" },
	{ &isl_multi_union_pw_aff_apply_pw_multi_aff,
	  "[N] -> (C[] : { : N >= 0 })",
	  "{ C[] -> D[0] }",
	  "[N] -> D[{ [0] : N >= 0 }]" },
	{ &isl_multi_union_pw_aff_apply_pw_multi_aff,
	  "(C[] : { A[i,j] : i >= j })",
	  "[N] -> { C[] -> D[N] : N >= 0 }",
	  "[N] -> D[{ A[i,j] -> [N] : i >= j and N >= 0 }]" },
};

/* Perform some basic tests of binary operations on
 * pairs of isl_multi_union_pw_aff and isl_pw_multi_aff objects.
 */
static int test_mupa_pma(isl_ctx *ctx)
{
	int i;
	isl_bool ok;
	isl_multi_union_pw_aff *mupa, *res;
	isl_pw_multi_aff *pma;

	for (i = 0; i < ARRAY_SIZE(mupa_pma_tests); ++i) {
		mupa = isl_multi_union_pw_aff_read_from_str(ctx,
						    mupa_pma_tests[i].arg1);
		pma = isl_pw_multi_aff_read_from_str(ctx,
						    mupa_pma_tests[i].arg2);
		res = isl_multi_union_pw_aff_read_from_str(ctx,
						    mupa_pma_tests[i].res);
		mupa = mupa_pma_tests[i].fn(mupa, pma);
		ok = isl_multi_union_pw_aff_plain_is_equal(mupa, res);
		isl_multi_union_pw_aff_free(mupa);
		isl_multi_union_pw_aff_free(res);
		if (ok < 0)
			return -1;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return -1);
	}

	return 0;
}

/* Inputs for basic tests of binary operations on
 * pairs of isl_multi_union_pw_aff and isl_union_pw_multi_aff objects.
 * "fn" is the function that is tested.
 * "arg1" and "arg2" are string descriptions of the inputs.
 * "res" is a string description of the expected result.
 */
struct {
	__isl_give isl_multi_union_pw_aff *(*fn)(
		__isl_take isl_multi_union_pw_aff *mupa,
		__isl_take isl_union_pw_multi_aff *upma);
	const char *arg1;
	const char *arg2;
	const char *res;
} mupa_upma_tests[] = {
	{ &isl_multi_union_pw_aff_pullback_union_pw_multi_aff,
	  "C[{ B[i,j] -> [i + 2j] }]", "{ A[a,b] -> B[b,a] }",
	  "C[{ A[a,b] -> [b + 2a] }]" },
	{ &isl_multi_union_pw_aff_pullback_union_pw_multi_aff,
	  "C[{ B[i,j] -> [i + 2j] }]",
	  "{ A[a,b] -> B[b,a] : b > a }",
	  "C[{ A[a,b] -> [b + 2a] : b > a }]" },
	{ &isl_multi_union_pw_aff_pullback_union_pw_multi_aff,
	  "C[{ B[i,j] -> [i + 2j] : j > 4 }]",
	  "{ A[a,b] -> B[b,a] : b > a }",
	  "C[{ A[a,b] -> [b + 2a] : b > a > 4 }]" },
	{ &isl_multi_union_pw_aff_pullback_union_pw_multi_aff,
	  "C[{ B[i,j] -> [i + 2j] }]",
	  "{ A[a,b] -> B[b,a] : a > b; A[a,b] -> B[a,b] : a <= b }",
	  "C[{ A[a,b] -> [b + 2a] : a > b; A[a,b] -> [a + 2b] : a <= b }]" },
	{ &isl_multi_union_pw_aff_pullback_union_pw_multi_aff,
	  "(C[] : { B[a,b] })",
	  "{ A[a,b] -> B[b,a] }",
	  "(C[] : { A[a,b] })" },
	{ &isl_multi_union_pw_aff_pullback_union_pw_multi_aff,
	  "(C[] : { B[a,b] })",
	  "{ B[a,b] -> A[b,a] }",
	  "(C[] : { })" },
	{ &isl_multi_union_pw_aff_pullback_union_pw_multi_aff,
	  "(C[] : { B[a,b] })",
	  "{ A[a,b] -> B[b,a] : a > b }",
	  "(C[] : { A[a,b] : a > b })" },
	{ &isl_multi_union_pw_aff_pullback_union_pw_multi_aff,
	  "(C[] : { B[a,b] : a > b })",
	  "{ A[a,b] -> B[b,a] }",
	  "(C[] : { A[a,b] : b > a })" },
	{ &isl_multi_union_pw_aff_pullback_union_pw_multi_aff,
	  "[N] -> (C[] : { B[a,b] : a > N })",
	  "{ A[a,b] -> B[b,a] : a > b }",
	  "[N] -> (C[] : { A[a,b] : a > b > N })" },
	{ &isl_multi_union_pw_aff_pullback_union_pw_multi_aff,
	  "(C[] : { B[a,b] : a > b })",
	  "[N] -> { A[a,b] -> B[b,a] : a > N }",
	  "[N] -> (C[] : { A[a,b] : b > a > N })" },
	{ &isl_multi_union_pw_aff_pullback_union_pw_multi_aff,
	  "C[]",
	  "{ A[a,b] -> B[b,a] }",
	  "C[]" },
	{ &isl_multi_union_pw_aff_pullback_union_pw_multi_aff,
	  "[N] -> (C[] : { : N >= 0 })",
	  "{ A[a,b] -> B[b,a] }",
	  "[N] -> (C[] : { : N >= 0 })" },
	{ &isl_multi_union_pw_aff_pullback_union_pw_multi_aff,
	  "C[]",
	  "[N] -> { A[a,b] -> B[b,a] : N >= 0 }",
	  "[N] -> (C[] : { : N >= 0 })" },
};

/* Perform some basic tests of binary operations on
 * pairs of isl_multi_union_pw_aff and isl_union_pw_multi_aff objects.
 */
static int test_mupa_upma(isl_ctx *ctx)
{
	int i;
	isl_bool ok;
	isl_multi_union_pw_aff *mupa, *res;
	isl_union_pw_multi_aff *upma;

	for (i = 0; i < ARRAY_SIZE(mupa_upma_tests); ++i) {
		mupa = isl_multi_union_pw_aff_read_from_str(ctx,
						    mupa_upma_tests[i].arg1);
		upma = isl_union_pw_multi_aff_read_from_str(ctx,
						    mupa_upma_tests[i].arg2);
		res = isl_multi_union_pw_aff_read_from_str(ctx,
						    mupa_upma_tests[i].res);
		mupa = mupa_upma_tests[i].fn(mupa, upma);
		ok = isl_multi_union_pw_aff_plain_is_equal(mupa, res);
		isl_multi_union_pw_aff_free(mupa);
		isl_multi_union_pw_aff_free(res);
		if (ok < 0)
			return -1;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return -1);
	}

	return 0;
}

/* Check that the input tuple of an isl_aff can be set properly.
 */
static isl_stat test_aff_set_tuple_id(isl_ctx *ctx)
{
	isl_id *id;
	isl_aff *aff;
	isl_stat equal;

	aff = isl_aff_read_from_str(ctx, "{ [x] -> [x + 1] }");
	id = isl_id_alloc(ctx, "A", NULL);
	aff = isl_aff_set_tuple_id(aff, isl_dim_in, id);
	equal = aff_check_plain_equal(aff, "{ A[x] -> [x + 1] }");
	isl_aff_free(aff);
	if (equal < 0)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Check that affine expressions get normalized on addition/subtraction.
 * In particular, check that (final) unused integer divisions get removed
 * such that an expression derived from expressions with integer divisions
 * is found to be obviously equal to one that is created directly.
 */
static isl_stat test_aff_normalize(isl_ctx *ctx)
{
	isl_aff *aff, *aff2;
	isl_stat ok;

	aff = isl_aff_read_from_str(ctx, "{ [x] -> [x//2] }");
	aff2 = isl_aff_read_from_str(ctx, "{ [x] -> [1 + x//2] }");
	aff = isl_aff_sub(aff2, aff);
	ok = aff_check_plain_equal(aff, "{ [x] -> [1] }");
	isl_aff_free(aff);

	return ok;
}

int test_aff(isl_ctx *ctx)
{
	const char *str;
	isl_set *set;
	isl_space *space;
	isl_local_space *ls;
	isl_aff *aff;
	int zero;
	isl_stat equal;

	if (test_upa(ctx) < 0)
		return -1;
	if (test_bin_aff(ctx) < 0)
		return -1;
	if (test_bin_pw_aff(ctx) < 0)
		return -1;
	if (test_upma_test(ctx) < 0)
		return -1;
	if (test_bin_upma(ctx) < 0)
		return -1;
	if (test_bin_upma_fail(ctx) < 0)
		return -1;
	if (test_upma_uset(ctx) < 0)
		return -1;
	if (test_un_mpa(ctx) < 0)
		return -1;
	if (test_bin_mpa(ctx) < 0)
		return -1;
	if (test_un_mupa(ctx) < 0)
		return -1;
	if (test_bin_mupa(ctx) < 0)
		return -1;
	if (test_mupa_set(ctx) < 0)
		return -1;
	if (test_mupa_uset(ctx) < 0)
		return -1;
	if (test_mupa_ma(ctx) < 0)
		return -1;
	if (test_mupa_pa(ctx) < 0)
		return -1;
	if (test_mupa_pma(ctx) < 0)
		return -1;
	if (test_mupa_upma(ctx) < 0)
		return -1;

	space = isl_space_set_alloc(ctx, 0, 1);
	ls = isl_local_space_from_space(space);
	aff = isl_aff_zero_on_domain(ls);

	aff = isl_aff_add_coefficient_si(aff, isl_dim_in, 0, 1);
	aff = isl_aff_scale_down_ui(aff, 3);
	aff = isl_aff_floor(aff);
	aff = isl_aff_add_coefficient_si(aff, isl_dim_in, 0, 1);
	aff = isl_aff_scale_down_ui(aff, 2);
	aff = isl_aff_floor(aff);
	aff = isl_aff_add_coefficient_si(aff, isl_dim_in, 0, 1);

	str = "{ [10] }";
	set = isl_set_read_from_str(ctx, str);
	aff = isl_aff_gist(aff, set);

	aff = isl_aff_add_constant_si(aff, -16);
	zero = isl_aff_plain_is_zero(aff);
	isl_aff_free(aff);

	if (zero < 0)
		return -1;
	if (!zero)
		isl_die(ctx, isl_error_unknown, "unexpected result", return -1);

	aff = isl_aff_read_from_str(ctx, "{ [-1] }");
	aff = isl_aff_scale_down_ui(aff, 64);
	aff = isl_aff_floor(aff);
	equal = aff_check_plain_equal(aff, "{ [-1] }");
	isl_aff_free(aff);
	if (equal < 0)
		return -1;

	if (test_aff_set_tuple_id(ctx) < 0)
		return -1;
	if (test_aff_normalize(ctx) < 0)
		return -1;

	return 0;
}

/* Inputs for isl_set_bind tests.
 * "set" is the input set.
 * "tuple" is the binding tuple.
 * "res" is the expected result.
 */
static
struct {
	const char *set;
	const char *tuple;
	const char *res;
} bind_set_tests[] = {
	{ "{ A[M, N] : M mod 2 = 0 and N mod 8 = 3 }",
	  "{ A[M, N] }",
	  "[M, N] -> { : M mod 2 = 0 and N mod 8 = 3 }" },
	{ "{ B[N, M] : M mod 2 = 0 and N mod 8 = 3 }",
	  "{ B[N, M] }",
	  "[M, N] -> { : M mod 2 = 0 and N mod 8 = 3 }" },
	{ "[M] -> { C[N] : M mod 2 = 0 and N mod 8 = 3 }",
	  "{ C[N] }",
	  "[M, N] -> { : M mod 2 = 0 and N mod 8 = 3 }" },
	{ "[M] -> { D[x, N] : x mod 2 = 0 and N mod 8 = 3 and M >= 0 }",
	  "{ D[M, N] }",
	  "[M, N] -> { : M mod 2 = 0 and N mod 8 = 3 and M >= 0 }" },
};

/* Perform basic isl_set_bind tests.
 */
static isl_stat test_bind_set(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(bind_set_tests); ++i) {
		const char *str;
		isl_set *set;
		isl_multi_id *tuple;
		isl_stat r;

		set = isl_set_read_from_str(ctx, bind_set_tests[i].set);
		str = bind_set_tests[i].tuple;
		tuple = isl_multi_id_read_from_str(ctx, str);
		set = isl_set_bind(set, tuple);
		r = set_check_equal(set, bind_set_tests[i].res);
		isl_set_free(set);
		if (r < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Inputs for isl_map_bind_domain tests.
 * "map" is the input map.
 * "tuple" is the binding tuple.
 * "res" is the expected result.
 */
struct {
	const char *map;
	const char *tuple;
	const char *res;
} bind_map_domain_tests[] = {
	{ "{ A[M, N] -> [M + floor(N/2)] }",
	  "{ A[M, N] }",
	  "[M, N] -> { [M + floor(N/2)] }" },
	{ "{ B[N, M] -> [M + floor(N/2)] }",
	  "{ B[N, M] }",
	  "[N, M] -> { [M + floor(N/2)] }" },
	{ "[M] -> { C[N] -> [M + floor(N/2)] }",
	  "{ C[N] }",
	  "[M, N] -> { [M + floor(N/2)] }" },
	{ "[M] -> { C[x, N] -> [x + floor(N/2)] }",
	  "{ C[M, N] }",
	  "[M, N] -> { [M + floor(N/2)] }" },
	{ "[M] -> { C[x, N] -> [M + floor(N/2)] }",
	  "{ C[M, N] }",
	  "[M, N] -> { [M + floor(N/2)] }" },
	{ "[A, M] -> { C[N, x] -> [x + floor(N/2)] }",
	  "{ C[N, M] }",
	  "[A, N, M] -> { [M + floor(N/2)] }" },
};

/* Perform basic isl_map_bind_domain tests.
 */
static isl_stat test_bind_map_domain(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(bind_map_domain_tests); ++i) {
		const char *str;
		isl_map *map;
		isl_set *set;
		isl_multi_id *tuple;
		isl_stat r;

		str = bind_map_domain_tests[i].map;
		map = isl_map_read_from_str(ctx, str);
		str = bind_map_domain_tests[i].tuple;
		tuple = isl_multi_id_read_from_str(ctx, str);
		set = isl_map_bind_domain(map, tuple);
		str = bind_map_domain_tests[i].res;
		r = set_check_equal(set, str);
		isl_set_free(set);
		if (r < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Inputs for isl_union_map_bind_range tests.
 * "map" is the input union map.
 * "tuple" is the binding tuple.
 * "res" is the expected result.
 */
struct {
	const char *map;
	const char *tuple;
	const char *res;
} bind_umap_range_tests[] = {
	{ "{ B[N, M] -> A[M, N] : M mod 2 = 0 and N mod 8 = 3 }",
	  "{ A[M, N] }",
	  "[M, N] -> { B[N, M] : M mod 2 = 0 and N mod 8 = 3 }" },
	{ "{ B[N, M] -> A[M, N] : M mod 2 = 0 and N mod 8 = 3 }",
	  "{ B[M, N] }",
	  "{ }" },
	{ "{ A[] -> B[]; C[] -> D[]; E[] -> B[] }",
	  "{ B[] }",
	  "{ A[]; E[] }" },
};

/* Perform basic isl_union_map_bind_range tests.
 */
static isl_stat test_bind_umap_range(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(bind_umap_range_tests); ++i) {
		const char *str;
		isl_union_map *umap;
		isl_union_set *uset;
		isl_multi_id *tuple;
		isl_stat r;

		str = bind_umap_range_tests[i].map;
		umap = isl_union_map_read_from_str(ctx, str);
		str = bind_umap_range_tests[i].tuple;
		tuple = isl_multi_id_read_from_str(ctx, str);
		uset = isl_union_map_bind_range(umap, tuple);
		str = bind_umap_range_tests[i].res;
		r = uset_check_equal(uset, str);
		isl_union_set_free(uset);
		if (r < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Inputs for isl_pw_multi_aff_bind_domain tests.
 * "pma" is the input expression.
 * "tuple" is the binding tuple.
 * "res" is the expected result.
 */
struct {
	const char *pma;
	const char *tuple;
	const char *res;
} bind_pma_domain_tests[] = {
	{ "{ A[M, N] -> [M + floor(N/2)] }",
	  "{ A[M, N] }",
	  "[M, N] -> { [M + floor(N/2)] }" },
	{ "{ B[N, M] -> [M + floor(N/2)] }",
	  "{ B[N, M] }",
	  "[N, M] -> { [M + floor(N/2)] }" },
	{ "[M] -> { C[N] -> [M + floor(N/2)] }",
	  "{ C[N] }",
	  "[M, N] -> { [M + floor(N/2)] }" },
	{ "[M] -> { C[x, N] -> [x + floor(N/2)] }",
	  "{ C[M, N] }",
	  "[M, N] -> { [M + floor(N/2)] }" },
	{ "[M] -> { C[x, N] -> [M + floor(N/2)] }",
	  "{ C[M, N] }",
	  "[M, N] -> { [M + floor(N/2)] }" },
	{ "[A, M] -> { C[N, x] -> [x + floor(N/2)] }",
	  "{ C[N, M] }",
	  "[A, N, M] -> { [M + floor(N/2)] }" },
};

/* Perform basic isl_pw_multi_aff_bind_domain tests.
 */
static isl_stat test_bind_pma_domain(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(bind_pma_domain_tests); ++i) {
		const char *str;
		isl_pw_multi_aff *pma;
		isl_multi_id *tuple;
		isl_stat r;

		str = bind_pma_domain_tests[i].pma;
		pma = isl_pw_multi_aff_read_from_str(ctx, str);
		str = bind_pma_domain_tests[i].tuple;
		tuple = isl_multi_id_read_from_str(ctx, str);
		pma = isl_pw_multi_aff_bind_domain(pma, tuple);
		str = bind_pma_domain_tests[i].res;
		r = pw_multi_aff_check_plain_equal(pma, str);
		isl_pw_multi_aff_free(pma);
		if (r < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Inputs for isl_pw_multi_aff_bind_domain_wrapped_domain tests.
 * "pma" is the input expression.
 * "tuple" is the binding tuple.
 * "res" is the expected result.
 */
struct {
	const char *pma;
	const char *tuple;
	const char *res;
} bind_pma_domain_wrapped_tests[] = {
	{ "{ [A[M, N] -> B[]] -> [M + floor(N/2)] }",
	  "{ A[M, N] }",
	  "[M, N] -> { B[] -> [M + floor(N/2)] }" },
	{ "{ [B[N, M] -> D[]] -> [M + floor(N/2)] }",
	  "{ B[N, M] }",
	  "[N, M] -> { D[] -> [M + floor(N/2)] }" },
	{ "[M] -> { [C[N] -> B[x]] -> [x + M + floor(N/2)] }",
	  "{ C[N] }",
	  "[M, N] -> { B[x] -> [x + M + floor(N/2)] }" },
	{ "[M] -> { [C[x, N] -> B[]] -> [x + floor(N/2)] }",
	  "{ C[M, N] }",
	  "[M, N] -> { B[] -> [M + floor(N/2)] }" },
	{ "[M] -> { [C[x, N] -> B[]] -> [M + floor(N/2)] }",
	  "{ C[M, N] }",
	  "[M, N] -> { B[] -> [M + floor(N/2)] }" },
	{ "[A, M] -> { [C[N, x] -> B[]] -> [x + floor(N/2)] }",
	  "{ C[N, M] }",
	  "[A, N, M] -> { B[] -> [M + floor(N/2)] }" },
};

/* Perform basic isl_pw_multi_aff_bind_domain_wrapped_domain tests.
 */
static isl_stat test_bind_pma_domain_wrapped(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(bind_pma_domain_wrapped_tests); ++i) {
		const char *str;
		isl_pw_multi_aff *pma;
		isl_multi_id *tuple;
		isl_stat r;

		str = bind_pma_domain_wrapped_tests[i].pma;
		pma = isl_pw_multi_aff_read_from_str(ctx, str);
		str = bind_pma_domain_wrapped_tests[i].tuple;
		tuple = isl_multi_id_read_from_str(ctx, str);
		pma = isl_pw_multi_aff_bind_domain_wrapped_domain(pma, tuple);
		str = bind_pma_domain_wrapped_tests[i].res;
		r = pw_multi_aff_check_plain_equal(pma, str);
		isl_pw_multi_aff_free(pma);
		if (r < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Inputs for isl_aff_bind_id tests.
 * "aff" is the input expression.
 * "id" is the binding id.
 * "res" is the expected result.
 */
static
struct {
	const char *aff;
	const char *id;
	const char *res;
} bind_aff_tests[] = {
	{ "{ [4] }", "M", "[M = 4] -> { : }" },
	{ "{ B[x] -> [floor(x/2)] }", "M", "[M] -> { B[x] : M = floor(x/2) }" },
	{ "[M] -> { [4] }", "M", "[M = 4] -> { : }" },
	{ "[M] -> { [floor(M/2)] }", "M", "[M] -> { : floor(M/2) = M }" },
	{ "{ [NaN] }", "M", "{ : false }" },
	{ "{ A[x] -> [NaN] }", "M", "{ A[x] : false }" },
};

/* Perform basic isl_aff_bind_id tests.
 */
static isl_stat test_bind_aff(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(bind_aff_tests); ++i) {
		isl_aff *aff;
		isl_set *res;
		isl_id *id;
		isl_stat r;

		aff = isl_aff_read_from_str(ctx, bind_aff_tests[i].aff);
		id = isl_id_read_from_str(ctx, bind_aff_tests[i].id);
		res = isl_set_from_basic_set(isl_aff_bind_id(aff, id));
		r = set_check_equal(res, bind_aff_tests[i].res);
		isl_set_free(res);
		if (r < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Inputs for isl_pw_aff_bind_id tests.
 * "pa" is the input expression.
 * "id" is the binding id.
 * "res" is the expected result.
 */
static
struct {
	const char *pa;
	const char *id;
	const char *res;
} bind_pa_tests[] = {
	{ "{ [4] }", "M", "[M = 4] -> { : }" },
	{ "{ B[x] -> [floor(x/2)] }", "M", "[M] -> { B[x] : M = floor(x/2) }" },
	{ "[M] -> { [4] }", "M", "[M = 4] -> { : }" },
	{ "[M] -> { [floor(M/2)] }", "M", "[M] -> { : floor(M/2) = M }" },
	{ "[M] -> { [M] : M >= 0; [-M] : M < 0 }", "M", "[M] -> { : M >= 0 }" },
	{ "{ [NaN] }", "M", "{ : false }" },
	{ "{ A[x] -> [NaN] }", "M", "{ A[x] : false }" },
};

/* Perform basic isl_pw_aff_bind_id tests.
 */
static isl_stat test_bind_pa(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(bind_pa_tests); ++i) {
		isl_pw_aff *pa;
		isl_set *res;
		isl_id *id;
		isl_stat r;

		pa = isl_pw_aff_read_from_str(ctx, bind_pa_tests[i].pa);
		id = isl_id_read_from_str(ctx, bind_pa_tests[i].id);
		res = isl_pw_aff_bind_id(pa, id);
		r = set_check_equal(res, bind_pa_tests[i].res);
		isl_set_free(res);
		if (r < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Inputs for isl_multi_union_pw_aff_bind tests.
 * "mupa" is the input expression.
 * "tuple" is the binding tuple.
 * "res" is the expected result.
 */
static
struct {
	const char *mupa;
	const char *tuple;
	const char *res;
} bind_mupa_tests[] = {
	{ "A[{ [4] }, { [5] }]",
	  "{ A[M, N] }",
	  "[M = 4, N = 5] -> { : }" },
	{ "A[{ B[x] -> [floor(x/2)] }, { B[y] -> [y + 5] }]",
	  "{ A[M, N] }",
	  "[M, N] -> { B[x] : M = floor(x/2) and N = x + 5 }" },
	{ "[M] -> A[{ [4] }, { [M + 1] }]",
	  "{ A[M, N] }",
	  "[M = 4, N = 5] -> { : }" },
};

/* Perform basic isl_multi_union_pw_aff_bind tests.
 */
static isl_stat test_bind_mupa(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(bind_mupa_tests); ++i) {
		const char *str;
		isl_multi_union_pw_aff *mupa;
		isl_union_set *res;
		isl_multi_id *tuple;
		isl_stat r;

		str = bind_mupa_tests[i].mupa;
		mupa = isl_multi_union_pw_aff_read_from_str(ctx, str);
		str = bind_mupa_tests[i].tuple;
		tuple = isl_multi_id_read_from_str(ctx, str);
		res = isl_multi_union_pw_aff_bind(mupa, tuple);
		r = uset_check_equal(res, bind_mupa_tests[i].res);
		isl_union_set_free(res);
		if (r < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Perform tests that reinterpret dimensions as parameters.
 */
static int test_bind(isl_ctx *ctx)
{
	if (test_bind_set(ctx) < 0)
		return -1;
	if (test_bind_map_domain(ctx) < 0)
		return -1;
	if (test_bind_umap_range(ctx) < 0)
		return -1;
	if (test_bind_pma_domain(ctx) < 0)
		return -1;
	if (test_bind_pma_domain_wrapped(ctx) < 0)
		return -1;
	if (test_bind_aff(ctx) < 0)
		return -1;
	if (test_bind_pa(ctx) < 0)
		return -1;
	if (test_bind_mupa(ctx) < 0)
		return -1;

	return 0;
}

/* Inputs for isl_set_unbind_params tests.
 * "set" is the input parameter domain.
 * "tuple" is the tuple of the constructed set.
 * "res" is the expected result.
 */
struct {
	const char *set;
	const char *tuple;
	const char *res;
} unbind_set_tests[] = {
	{ "[M, N] -> { : M mod 2 = 0 and N mod 8 = 3 }",
	  "{ A[M, N] }",
	  "{ A[M, N] : M mod 2 = 0 and N mod 8 = 3 }" },
	{ "[M, N] -> { : M mod 2 = 0 and N mod 8 = 3 }",
	  "{ B[N, M] }",
	  "{ B[N, M] : M mod 2 = 0 and N mod 8 = 3 }" },
	{ "[M, N] -> { : M mod 2 = 0 and N mod 8 = 3 }",
	  "{ C[N] }",
	  "[M] -> { C[N] : M mod 2 = 0 and N mod 8 = 3 }" },
	{ "[M, N] -> { : M mod 2 = 0 and N mod 8 = 3 }",
	  "{ D[T, N] }",
	  "[M] -> { D[x, N] : M mod 2 = 0 and N mod 8 = 3 }" },
};

/* Perform basic isl_set_unbind_params tests.
 */
static isl_stat test_unbind_set(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(unbind_set_tests); ++i) {
		const char *str;
		isl_set *set;
		isl_multi_id *tuple;
		isl_stat r;

		set = isl_set_read_from_str(ctx, unbind_set_tests[i].set);
		str = unbind_set_tests[i].tuple;
		tuple = isl_multi_id_read_from_str(ctx, str);
		set = isl_set_unbind_params(set, tuple);
		r = set_check_equal(set, unbind_set_tests[i].res);
		isl_set_free(set);
		if (r < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Inputs for isl_aff_unbind_params_insert_domain tests.
 * "aff" is the input affine expression defined over a parameter domain.
 * "tuple" is the tuple of the domain that gets introduced.
 * "res" is the expected result.
 */
struct {
	const char *aff;
	const char *tuple;
	const char *res;
} unbind_aff_tests[] = {
	{ "[M, N] -> { [M + floor(N/2)] }",
	  "{ A[M, N] }",
	  "{ A[M, N] -> [M + floor(N/2)] }" },
	{ "[M, N] -> { [M + floor(N/2)] }",
	  "{ B[N, M] }",
	  "{ B[N, M] -> [M + floor(N/2)] }" },
	{ "[M, N] -> { [M + floor(N/2)] }",
	  "{ C[N] }",
	  "[M] -> { C[N] -> [M + floor(N/2)] }" },
	{ "[M, N] -> { [M + floor(N/2)] }",
	  "{ D[A, B, C, N, Z] }",
	  "[M] -> { D[A, B, C, N, Z] -> [M + floor(N/2)] }" },
};

/* Perform basic isl_aff_unbind_params_insert_domain tests.
 */
static isl_stat test_unbind_aff(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(unbind_aff_tests); ++i) {
		const char *str;
		isl_aff *aff;
		isl_multi_id *tuple;
		isl_stat r;

		aff = isl_aff_read_from_str(ctx, unbind_aff_tests[i].aff);
		str = unbind_aff_tests[i].tuple;
		tuple = isl_multi_id_read_from_str(ctx, str);
		aff = isl_aff_unbind_params_insert_domain(aff, tuple);
		r = aff_check_plain_equal(aff, unbind_aff_tests[i].res);
		isl_aff_free(aff);
		if (r < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Inputs for isl_multi_aff_unbind_params_insert_domain tests.
 * "ma" is the input multi affine expression defined over a parameter domain.
 * "tuple" is the tuple of the domain that gets introduced.
 * "res" is the expected result.
 */
static struct {
	const char *ma;
	const char *tuple;
	const char *res;
} unbind_multi_aff_tests[] = {
	{ "[M, N] -> { T[M + floor(N/2)] }",
	  "{ A[M, N] }",
	  "{ A[M, N] -> T[M + floor(N/2)] }" },
	{ "[M, N] -> { [M + floor(N/2)] }",
	  "{ B[N, M] }",
	  "{ B[N, M] -> [M + floor(N/2)] }" },
	{ "[M, N] -> { [M + floor(N/2)] }",
	  "{ C[N] }",
	  "[M] -> { C[N] -> [M + floor(N/2)] }" },
	{ "[M, N] -> { [M + floor(N/2)] }",
	  "{ D[A, B, C, N, Z] }",
	  "[M] -> { D[A, B, C, N, Z] -> [M + floor(N/2)] }" },
	{ "[M, N] -> { T[M + floor(N/2), N + floor(M/3)] }",
	  "{ A[M, N] }",
	  "{ A[M, N] -> T[M + floor(N/2), N + floor(M/3)] }" },
};

/* Perform basic isl_multi_aff_unbind_params_insert_domain tests.
 */
static isl_stat test_unbind_multi_aff(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(unbind_multi_aff_tests); ++i) {
		const char *str;
		isl_multi_aff *ma;
		isl_multi_id *tuple;
		isl_stat r;

		str = unbind_multi_aff_tests[i].ma;
		ma = isl_multi_aff_read_from_str(ctx, str);
		str = unbind_multi_aff_tests[i].tuple;
		tuple = isl_multi_id_read_from_str(ctx, str);
		ma = isl_multi_aff_unbind_params_insert_domain(ma, tuple);
		str = unbind_multi_aff_tests[i].res;
		r = multi_aff_check_plain_equal(ma, str);
		isl_multi_aff_free(ma);
		if (r < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Perform tests that reinterpret parameters.
 */
static int test_unbind(isl_ctx *ctx)
{
	if (test_unbind_set(ctx) < 0)
		return -1;
	if (test_unbind_aff(ctx) < 0)
		return -1;
	if (test_unbind_multi_aff(ctx) < 0)
		return -1;

	return 0;
}

/* Check that "pa" consists of a single expression.
 */
static int check_single_piece(isl_ctx *ctx, __isl_take isl_pw_aff *pa)
{
	isl_size n;

	n = isl_pw_aff_n_piece(pa);
	isl_pw_aff_free(pa);

	if (n < 0)
		return -1;
	if (n != 1)
		isl_die(ctx, isl_error_unknown, "expecting single expression",
			return -1);

	return 0;
}

/* Check that the computation below results in a single expression.
 * One or two expressions may result depending on which constraint
 * ends up being considered as redundant with respect to the other
 * constraints after the projection that is performed internally
 * by isl_set_dim_min.
 */
static int test_dim_max_1(isl_ctx *ctx)
{
	const char *str;
	isl_set *set;
	isl_pw_aff *pa;

	str = "[n] -> { [a, b] : n >= 0 and 4a >= -4 + n and b >= 0 and "
				"-4a <= b <= 3 and b < n - 4a }";
	set = isl_set_read_from_str(ctx, str);
	pa = isl_set_dim_min(set, 0);
	return check_single_piece(ctx, pa);
}

/* Check that the computation below results in a single expression.
 * The PIP problem corresponding to these constraints has a row
 * that causes a split of the solution domain.  The solver should
 * first pick rows that split off empty parts such that the actual
 * solution domain does not get split.
 * Note that the description contains some redundant constraints.
 * If these constraints get removed first, then the row mentioned
 * above does not appear in the PIP problem.
 */
static int test_dim_max_2(isl_ctx *ctx)
{
	const char *str;
	isl_set *set;
	isl_pw_aff *pa;

	str = "[P, N] -> { [a] : a < N and a >= 0 and N > P and a <= P and "
				"N > 0 and P >= 0 }";
	set = isl_set_read_from_str(ctx, str);
	pa = isl_set_dim_max(set, 0);
	return check_single_piece(ctx, pa);
}

int test_dim_max(isl_ctx *ctx)
{
	int equal;
	const char *str;
	isl_set *set1, *set2;
	isl_set *set;
	isl_map *map;
	isl_pw_aff *pwaff;

	if (test_dim_max_1(ctx) < 0)
		return -1;
	if (test_dim_max_2(ctx) < 0)
		return -1;

	str = "[N] -> { [i] : 0 <= i <= min(N,10) }";
	set = isl_set_read_from_str(ctx, str);
	pwaff = isl_set_dim_max(set, 0);
	set1 = isl_set_from_pw_aff(pwaff);
	str = "[N] -> { [10] : N >= 10; [N] : N <= 9 and N >= 0 }";
	set2 = isl_set_read_from_str(ctx, str);
	equal = isl_set_is_equal(set1, set2);
	isl_set_free(set1);
	isl_set_free(set2);
	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown, "unexpected result", return -1);

	str = "[N] -> { [i] : 0 <= i <= max(2N,N+6) }";
	set = isl_set_read_from_str(ctx, str);
	pwaff = isl_set_dim_max(set, 0);
	set1 = isl_set_from_pw_aff(pwaff);
	str = "[N] -> { [6 + N] : -6 <= N <= 5; [2N] : N >= 6 }";
	set2 = isl_set_read_from_str(ctx, str);
	equal = isl_set_is_equal(set1, set2);
	isl_set_free(set1);
	isl_set_free(set2);
	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown, "unexpected result", return -1);

	str = "[N] -> { [i] : 0 <= i <= 2N or 0 <= i <= N+6 }";
	set = isl_set_read_from_str(ctx, str);
	pwaff = isl_set_dim_max(set, 0);
	set1 = isl_set_from_pw_aff(pwaff);
	str = "[N] -> { [6 + N] : -6 <= N <= 5; [2N] : N >= 6 }";
	set2 = isl_set_read_from_str(ctx, str);
	equal = isl_set_is_equal(set1, set2);
	isl_set_free(set1);
	isl_set_free(set2);
	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown, "unexpected result", return -1);

	str = "[N,M] -> { [i,j] -> [([i/16]), i%16, ([j/16]), j%16] : "
			"0 <= i < N and 0 <= j < M }";
	map = isl_map_read_from_str(ctx, str);
	set = isl_map_range(map);

	pwaff = isl_set_dim_max(isl_set_copy(set), 0);
	set1 = isl_set_from_pw_aff(pwaff);
	str = "[N,M] -> { [([(N-1)/16])] : M,N > 0 }";
	set2 = isl_set_read_from_str(ctx, str);
	equal = isl_set_is_equal(set1, set2);
	isl_set_free(set1);
	isl_set_free(set2);

	pwaff = isl_set_dim_max(isl_set_copy(set), 3);
	set1 = isl_set_from_pw_aff(pwaff);
	str = "[N,M] -> { [t] : t = min(M-1,15) and M,N > 0 }";
	set2 = isl_set_read_from_str(ctx, str);
	if (equal >= 0 && equal)
		equal = isl_set_is_equal(set1, set2);
	isl_set_free(set1);
	isl_set_free(set2);

	isl_set_free(set);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown, "unexpected result", return -1);

	/* Check that solutions are properly merged. */
	str = "[n] -> { [a, b, c] : c >= -4a - 2b and "
				"c <= -1 + n - 4a - 2b and c >= -2b and "
				"4a >= -4 + n and c >= 0 }";
	set = isl_set_read_from_str(ctx, str);
	pwaff = isl_set_dim_min(set, 2);
	set1 = isl_set_from_pw_aff(pwaff);
	str = "[n] -> { [(0)] : n >= 1 }";
	set2 = isl_set_read_from_str(ctx, str);
	equal = isl_set_is_equal(set1, set2);
	isl_set_free(set1);
	isl_set_free(set2);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown, "unexpected result", return -1);

	/* Check that empty solution lie in the right space. */
	str = "[n] -> { [t,a] : 1 = 0 }";
	set = isl_set_read_from_str(ctx, str);
	pwaff = isl_set_dim_max(set, 0);
	set1 = isl_set_from_pw_aff(pwaff);
	str = "[n] -> { [t] : 1 = 0 }";
	set2 = isl_set_read_from_str(ctx, str);
	equal = isl_set_is_equal(set1, set2);
	isl_set_free(set1);
	isl_set_free(set2);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown, "unexpected result", return -1);

	return 0;
}

/* Basic test for isl_pw_multi_aff_product.
 *
 * Check that multiple pieces are properly handled.
 */
static int test_product_pma(isl_ctx *ctx)
{
	isl_stat equal;
	const char *str;
	isl_pw_multi_aff *pma1, *pma2;

	str = "{ A[i] -> B[1] : i < 0; A[i] -> B[2] : i >= 0 }";
	pma1 = isl_pw_multi_aff_read_from_str(ctx, str);
	str = "{ C[] -> D[] }";
	pma2 = isl_pw_multi_aff_read_from_str(ctx, str);
	pma1 = isl_pw_multi_aff_product(pma1, pma2);
	str = "{ [A[i] -> C[]] -> [B[(1)] -> D[]] : i < 0;"
		"[A[i] -> C[]] -> [B[(2)] -> D[]] : i >= 0 }";
	equal = pw_multi_aff_check_plain_equal(pma1, str);
	isl_pw_multi_aff_free(pma1);
	if (equal < 0)
		return -1;

	return 0;
}

int test_product(isl_ctx *ctx)
{
	const char *str;
	isl_set *set;
	isl_union_set *uset1, *uset2;
	int ok;

	str = "{ A[i] }";
	set = isl_set_read_from_str(ctx, str);
	set = isl_set_product(set, isl_set_copy(set));
	ok = isl_set_is_wrapping(set);
	isl_set_free(set);
	if (ok < 0)
		return -1;
	if (!ok)
		isl_die(ctx, isl_error_unknown, "unexpected result", return -1);

	str = "{ [] }";
	uset1 = isl_union_set_read_from_str(ctx, str);
	uset1 = isl_union_set_product(uset1, isl_union_set_copy(uset1));
	str = "{ [[] -> []] }";
	uset2 = isl_union_set_read_from_str(ctx, str);
	ok = isl_union_set_is_equal(uset1, uset2);
	isl_union_set_free(uset1);
	isl_union_set_free(uset2);
	if (ok < 0)
		return -1;
	if (!ok)
		isl_die(ctx, isl_error_unknown, "unexpected result", return -1);

	if (test_product_pma(ctx) < 0)
		return -1;

	return 0;
}

/* Check that two sets are not considered disjoint just because
 * they have a different set of (named) parameters.
 */
static int test_disjoint(isl_ctx *ctx)
{
	const char *str;
	isl_set *set, *set2;
	int disjoint;

	str = "[n] -> { [[]->[]] }";
	set = isl_set_read_from_str(ctx, str);
	str = "{ [[]->[]] }";
	set2 = isl_set_read_from_str(ctx, str);
	disjoint = isl_set_is_disjoint(set, set2);
	isl_set_free(set);
	isl_set_free(set2);
	if (disjoint < 0)
		return -1;
	if (disjoint)
		isl_die(ctx, isl_error_unknown, "unexpected result", return -1);

	return 0;
}

/* Inputs for isl_pw_multi_aff_is_equal tests.
 * "f1" and "f2" are the two function that need to be compared.
 * "equal" is the expected result.
 */
struct {
	int equal;
	const char *f1;
	const char *f2;
} pma_equal_tests[] = {
	{ 1, "[N] -> { [floor(N/2)] : 0 <= N <= 1 }",
	     "[N] -> { [0] : 0 <= N <= 1 }" },
	{ 1, "[N] -> { [floor(N/2)] : 0 <= N <= 2 }",
	     "[N] -> { [0] : 0 <= N <= 1; [1] : N = 2 }" },
	{ 0, "[N] -> { [floor(N/2)] : 0 <= N <= 2 }",
	     "[N] -> { [0] : 0 <= N <= 1 }" },
	{ 0, "{ [NaN] }", "{ [NaN] }" },
};

int test_equal(isl_ctx *ctx)
{
	int i;
	const char *str;
	isl_set *set, *set2;
	int equal;

	str = "{ S_6[i] }";
	set = isl_set_read_from_str(ctx, str);
	str = "{ S_7[i] }";
	set2 = isl_set_read_from_str(ctx, str);
	equal = isl_set_is_equal(set, set2);
	isl_set_free(set);
	isl_set_free(set2);
	if (equal < 0)
		return -1;
	if (equal)
		isl_die(ctx, isl_error_unknown, "unexpected result", return -1);

	for (i = 0; i < ARRAY_SIZE(pma_equal_tests); ++i) {
		int expected = pma_equal_tests[i].equal;
		isl_pw_multi_aff *f1, *f2;

		f1 = isl_pw_multi_aff_read_from_str(ctx, pma_equal_tests[i].f1);
		f2 = isl_pw_multi_aff_read_from_str(ctx, pma_equal_tests[i].f2);
		equal = isl_pw_multi_aff_is_equal(f1, f2);
		isl_pw_multi_aff_free(f1);
		isl_pw_multi_aff_free(f2);
		if (equal < 0)
			return -1;
		if (equal != expected)
			isl_die(ctx, isl_error_unknown,
				"unexpected equality result", return -1);
	}

	return 0;
}

static int test_plain_fixed(isl_ctx *ctx, __isl_take isl_map *map,
	enum isl_dim_type type, unsigned pos, int fixed)
{
	isl_bool test;

	test = isl_map_plain_is_fixed(map, type, pos, NULL);
	isl_map_free(map);
	if (test < 0)
		return -1;
	if (test == fixed)
		return 0;
	if (fixed)
		isl_die(ctx, isl_error_unknown,
			"map not detected as fixed", return -1);
	else
		isl_die(ctx, isl_error_unknown,
			"map detected as fixed", return -1);
}

int test_fixed(isl_ctx *ctx)
{
	const char *str;
	isl_map *map;

	str = "{ [i] -> [i] }";
	map = isl_map_read_from_str(ctx, str);
	if (test_plain_fixed(ctx, map, isl_dim_out, 0, 0))
		return -1;
	str = "{ [i] -> [1] }";
	map = isl_map_read_from_str(ctx, str);
	if (test_plain_fixed(ctx, map, isl_dim_out, 0, 1))
		return -1;
	str = "{ S_1[p1] -> [o0] : o0 = -2 and p1 >= 1 and p1 <= 7 }";
	map = isl_map_read_from_str(ctx, str);
	if (test_plain_fixed(ctx, map, isl_dim_out, 0, 1))
		return -1;
	map = isl_map_read_from_str(ctx, str);
	map = isl_map_neg(map);
	if (test_plain_fixed(ctx, map, isl_dim_out, 0, 1))
		return -1;

	return 0;
}

struct isl_vertices_test_data {
	const char *set;
	int n;
	const char *vertex[6];
} vertices_tests[] = {
	{ "{ A[t, i] : t = 12 and i >= 4 and i <= 12 }",
	  2, { "{ A[12, 4] }", "{ A[12, 12] }" } },
	{ "{ A[t, i] : t = 14 and i = 1 }",
	  1, { "{ A[14, 1] }" } },
	{ "[n, m] -> { [a, b, c] : b <= a and a <= n and b > 0 and c >= b and "
				"c <= m and m <= n and m > 0 }",
	  6, {
		"[n, m] -> { [n, m, m] : 0 < m <= n }",
		"[n, m] -> { [n, 1, m] : 0 < m <= n }",
		"[n, m] -> { [n, 1, 1] : 0 < m <= n }",
		"[n, m] -> { [m, m, m] : 0 < m <= n }",
		"[n, m] -> { [1, 1, m] : 0 < m <= n }",
		"[n, m] -> { [1, 1, 1] : 0 < m <= n }"
	    } },
	/* An input with implicit equality constraints among the parameters. */
	{ "[N, M] -> { [a, b] : M >= 3 and 9 + 3M <= a <= 29 + 2N + 11M and "
			    "2b >= M + a and 5 - 2N - M + a <= 2b <= 3 + a and "
			    "3b >= 15 + a }",
	  2, {
		"[N, M] -> { [(21), (12)] : M = 3 and N >= 0 }",
		"[N, M] -> { [(61 + 2N), (32 + N)] : M = 3 and N >= 0 }",
	     }
	},
};

/* Check that "vertex" corresponds to one of the vertices in data->vertex.
 */
static isl_stat find_vertex(__isl_take isl_vertex *vertex, void *user)
{
	struct isl_vertices_test_data *data = user;
	isl_ctx *ctx;
	isl_multi_aff *ma;
	isl_basic_set *bset;
	isl_pw_multi_aff *pma;
	int i;
	isl_bool equal;

	ctx = isl_vertex_get_ctx(vertex);
	bset = isl_vertex_get_domain(vertex);
	ma = isl_vertex_get_expr(vertex);
	pma = isl_pw_multi_aff_alloc(isl_set_from_basic_set(bset), ma);

	for (i = 0; i < data->n; ++i) {
		isl_pw_multi_aff *pma_i;

		pma_i = isl_pw_multi_aff_read_from_str(ctx, data->vertex[i]);
		equal = isl_pw_multi_aff_plain_is_equal(pma, pma_i);
		isl_pw_multi_aff_free(pma_i);

		if (equal < 0 || equal)
			break;
	}

	isl_pw_multi_aff_free(pma);
	isl_vertex_free(vertex);

	if (equal < 0)
		return isl_stat_error;

	return equal ? isl_stat_ok : isl_stat_error;
}

int test_vertices(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(vertices_tests); ++i) {
		isl_basic_set *bset;
		isl_vertices *vertices;
		int ok = 1;
		isl_size n;

		bset = isl_basic_set_read_from_str(ctx, vertices_tests[i].set);
		vertices = isl_basic_set_compute_vertices(bset);
		n = isl_vertices_get_n_vertices(vertices);
		if (vertices_tests[i].n != n)
			ok = 0;
		if (isl_vertices_foreach_vertex(vertices, &find_vertex,
						&vertices_tests[i]) < 0)
			ok = 0;
		isl_vertices_free(vertices);
		isl_basic_set_free(bset);

		if (n < 0)
			return -1;
		if (!ok)
			isl_die(ctx, isl_error_unknown, "unexpected vertices",
				return -1);
	}

	return 0;
}

/* Inputs for basic tests of unary operations on isl_union_map.
 * "fn" is the function that is being tested.
 * "arg" is a string description of the input.
 * "res" is a string description of the expected result.
 */
static struct {
	__isl_give isl_union_map *(*fn)(__isl_take isl_union_map *umap);
	const char *arg;
	const char *res;
} umap_un_tests[] = {
	{ &isl_union_map_range_reverse,
	  "{ A[] -> [B[] -> C[]]; A[] -> B[]; A[0] -> N[B[1] -> B[2]] }",
	  "{ A[] -> [C[] -> B[]]; A[0] -> N[B[2] -> B[1]] }" },
	{ &isl_union_map_range_reverse,
	  "{ A[] -> N[B[] -> C[]] }",
	  "{ A[] -> [C[] -> B[]] }" },
	{ &isl_union_map_range_reverse,
	  "{ A[] -> N[B[x] -> B[y]] }",
	  "{ A[] -> N[B[*] -> B[*]] }" },
};

/* Perform basic tests of unary operations on isl_union_map.
 */
static isl_stat test_un_union_map(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(umap_un_tests); ++i) {
		const char *str;
		isl_union_map *umap, *res;
		isl_bool equal;

		str = umap_un_tests[i].arg;
		umap = isl_union_map_read_from_str(ctx, str);
		str = umap_un_tests[i].res;
		res = isl_union_map_read_from_str(ctx, str);
		umap = umap_un_tests[i].fn(umap);
		equal = isl_union_map_is_equal(umap, res);
		isl_union_map_free(umap);
		isl_union_map_free(res);
		if (equal < 0)
			return isl_stat_error;
		if (!equal)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return isl_stat_error);
	}

	return isl_stat_ok;
}

/* Inputs for basic tests of binary operations on isl_union_map.
 * "fn" is the function that is being tested.
 * "arg1" and "arg2" are string descriptions of the inputs.
 * "res" is a string description of the expected result.
 */
static struct {
	__isl_give isl_union_map *(*fn)(__isl_take isl_union_map *umap1,
				__isl_take isl_union_map *umap2);
	const char *arg1;
	const char *arg2;
	const char *res;
} umap_bin_tests[] = {
	{ &isl_union_map_intersect,
	  "[n] -> { A[i] -> [] : 0 <= i <= n; B[] -> [] }",
	  "[m] -> { A[i] -> [] : 0 <= i <= m; C[] -> [] }",
	  "[m, n] -> { A[i] -> [] : 0 <= i <= n and i <= m }" },
	{ &isl_union_map_intersect_domain_factor_domain,
	  "{ [A[i] -> B[i + 1]] -> C[i + 2] }",
	  "[N] -> { B[i] -> C[N] }",
	  "{ }" },
	{ &isl_union_map_intersect_domain_factor_domain,
	  "{ [A[i] -> B[i + 1]] -> C[i + 2] }",
	  "[N] -> { A[i] -> C[N] }",
	  "[N] -> { [A[N - 2] -> B[N - 1]] -> C[N] }" },
	{ &isl_union_map_intersect_domain_factor_domain,
	  "{ T[A[i] -> B[i + 1]] -> C[i + 2] }",
	  "[N] -> { A[i] -> C[N] }",
	  "[N] -> { T[A[N - 2] -> B[N - 1]] -> C[N] }" },
	{ &isl_union_map_intersect_domain_factor_range,
	  "{ [A[i] -> B[i + 1]] -> C[i + 2] }",
	  "[N] -> { B[i] -> C[N] }",
	  "[N] -> { [A[N - 2] -> B[N - 1]] -> C[N] }" },
	{ &isl_union_map_intersect_domain_factor_range,
	  "{ T[A[i] -> B[i + 1]] -> C[i + 2] }",
	  "[N] -> { B[i] -> C[N] }",
	  "[N] -> { T[A[N - 2] -> B[N - 1]] -> C[N] }" },
	{ &isl_union_map_intersect_domain_factor_range,
	  "{ [A[i] -> B[i + 1]] -> C[i + 2] }",
	  "[N] -> { A[i] -> C[N] }",
	  "{ }" },
	{ &isl_union_map_intersect_range_factor_domain,
	  "{ A[i] -> [B[i + 1] -> C[i + 2]] }",
	  "[N] -> { A[i] -> B[N] }",
	  "[N] -> { A[N - 1] -> [B[N] -> C[N + 1]] }" },
	{ &isl_union_map_intersect_range_factor_domain,
	  "{ A[i] -> T[B[i + 1] -> C[i + 2]] }",
	  "[N] -> { A[i] -> B[N] }",
	  "[N] -> { A[N - 1] -> T[B[N] -> C[N + 1]] }" },
	{ &isl_union_map_intersect_range_factor_domain,
	  "{ A[i] -> [B[i + 1] -> C[i + 2]] }",
	  "[N] -> { A[i] -> C[N] }",
	  "{ }" },
	{ &isl_union_map_intersect_range_factor_range,
	  "{ A[i] -> [B[i + 1] -> C[i + 2]] }",
	  "[N] -> { A[i] -> C[N] }",
	  "[N] -> { A[N - 2] -> [B[N - 1] -> C[N]] }" },
	{ &isl_union_map_intersect_range_factor_range,
	  "{ A[i] -> T[B[i + 1] -> C[i + 2]] }",
	  "[N] -> { A[i] -> C[N] }",
	  "[N] -> { A[N - 2] -> T[B[N - 1] -> C[N]] }" },
	{ &isl_union_map_intersect_range_factor_range,
	  "{ A[i] -> [B[i + 1] -> C[i + 2]] }",
	  "[N] -> { A[i] -> B[N] }",
	  "{ }" },
};

/* Perform basic tests of binary operations on isl_union_map.
 */
static isl_stat test_bin_union_map(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(umap_bin_tests); ++i) {
		const char *str;
		isl_union_map *umap1, *umap2, *res;
		isl_bool equal;

		str = umap_bin_tests[i].arg1;
		umap1 = isl_union_map_read_from_str(ctx, str);
		str = umap_bin_tests[i].arg2;
		umap2 = isl_union_map_read_from_str(ctx, str);
		str = umap_bin_tests[i].res;
		res = isl_union_map_read_from_str(ctx, str);
		umap1 = umap_bin_tests[i].fn(umap1, umap2);
		equal = isl_union_map_is_equal(umap1, res);
		isl_union_map_free(umap1);
		isl_union_map_free(res);
		if (equal < 0)
			return isl_stat_error;
		if (!equal)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return isl_stat_error);
	}

	return isl_stat_ok;
}

/* Check that isl_union_set_contains finds space independently
 * of the parameters.
 */
static isl_stat test_union_set_contains(isl_ctx *ctx)
{
	const char *str;
	isl_bool ok;
	isl_space *space;
	isl_id *id;
	isl_union_set *uset;

	str = "[N] -> { A[0:N]; B[*,*] }";
	uset = isl_union_set_read_from_str(ctx, str);
	space = isl_space_unit(ctx);
	id = isl_id_alloc(ctx, "A", NULL);
	space = isl_space_add_named_tuple_id_ui(space, id, 1);
	ok = isl_union_set_contains(uset, space);
	isl_space_free(space);
	isl_union_set_free(uset);

	if (ok < 0)
		return isl_stat_error;
	if (!ok)
		isl_die(ctx, isl_error_unknown,
			"unexpected result", return isl_stat_error);

	return isl_stat_ok;
}

/* Perform basic tests of operations on isl_union_map or isl_union_set.
 */
static int test_union_map(isl_ctx *ctx)
{
	if (test_un_union_map(ctx) < 0)
		return -1;
	if (test_bin_union_map(ctx) < 0)
		return -1;
	if (test_union_set_contains(ctx) < 0)
		return -1;
	return 0;
}

int test_union_pw(isl_ctx *ctx)
{
	int equal;
	const char *str;
	isl_union_set *uset;
	isl_union_pw_qpolynomial *upwqp1, *upwqp2;

	str = "{ [x] -> x^2 }";
	upwqp1 = isl_union_pw_qpolynomial_read_from_str(ctx, str);
	upwqp2 = isl_union_pw_qpolynomial_copy(upwqp1);
	uset = isl_union_pw_qpolynomial_domain(upwqp1);
	upwqp1 = isl_union_pw_qpolynomial_copy(upwqp2);
	upwqp1 = isl_union_pw_qpolynomial_intersect_domain(upwqp1, uset);
	equal = isl_union_pw_qpolynomial_plain_is_equal(upwqp1, upwqp2);
	isl_union_pw_qpolynomial_free(upwqp1);
	isl_union_pw_qpolynomial_free(upwqp2);
	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown, "unexpected result", return -1);

	return 0;
}

/* Inputs for basic tests of functions that select
 * subparts of the domain of an isl_multi_union_pw_aff.
 * "fn" is the function that is tested.
 * "arg" is a string description of the input.
 * "res" is a string description of the expected result.
 */
struct {
	__isl_give isl_union_set *(*fn)(
		__isl_take isl_multi_union_pw_aff *mupa);
	const char *arg;
	const char *res;
} un_locus_tests[] = {
	{ &isl_multi_union_pw_aff_zero_union_set,
	  "F[{ A[i,j] -> [i]; B[i,j] -> [i] }]",
	  "{ A[0,j]; B[0,j] }" },
	{ &isl_multi_union_pw_aff_zero_union_set,
	  "F[{ A[i,j] -> [i-j]; B[i,j] -> [i-j] : i >= 0 }]",
	  "{ A[i,i]; B[i,i] : i >= 0 }" },
	{ &isl_multi_union_pw_aff_zero_union_set,
	  "(F[] : { A[i,j]; B[i,i] : i >= 0 })",
	  "{ A[i,j]; B[i,i] : i >= 0 }" },
};

/* Perform some basic tests of functions that select
 * subparts of the domain of an isl_multi_union_pw_aff.
 */
static int test_un_locus(isl_ctx *ctx)
{
	int i;
	isl_bool ok;
	isl_union_set *uset, *res;
	isl_multi_union_pw_aff *mupa;

	for (i = 0; i < ARRAY_SIZE(un_locus_tests); ++i) {
		mupa = isl_multi_union_pw_aff_read_from_str(ctx,
						    un_locus_tests[i].arg);
		res = isl_union_set_read_from_str(ctx, un_locus_tests[i].res);
		uset = un_locus_tests[i].fn(mupa);
		ok = isl_union_set_is_equal(uset, res);
		isl_union_set_free(uset);
		isl_union_set_free(res);
		if (ok < 0)
			return -1;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return -1);
	}

	return 0;
}

/* Inputs for basic tests of functions that select
 * subparts of an isl_union_map based on a relation
 * specified by an isl_multi_union_pw_aff.
 * "fn" is the function that is tested.
 * "arg1" and "arg2" are string descriptions of the inputs.
 * "res" is a string description of the expected result.
 */
struct {
	__isl_give isl_union_map *(*fn)(
		__isl_take isl_union_map *umap,
		__isl_take isl_multi_union_pw_aff *mupa);
	const char *arg1;
	const char *arg2;
	const char *res;
} bin_locus_tests[] = {
	{ &isl_union_map_eq_at_multi_union_pw_aff,
	  "{ A[i,j] -> B[i',j'] }",
	  "F[{ A[i,j] -> [i]; B[i,j] -> [i] }]",
	  "{ A[i,j] -> B[i,j'] }" },
	{ &isl_union_map_eq_at_multi_union_pw_aff,
	  "{ A[i,j] -> B[i',j'] }",
	  "F[{ A[i,j] -> [i]; B[i,j] -> [i] }, "
	    "{ A[i,j] -> [j]; B[i,j] -> [j] }]",
	  "{ A[i,j] -> B[i,j] }" },
	{ &isl_union_map_eq_at_multi_union_pw_aff,
	  "{ A[i,j] -> B[i',j']; A[i,j] -> C[i',j'] }",
	  "F[{ A[i,j] -> [i]; B[i,j] -> [i] }]",
	  "{ A[i,j] -> B[i,j'] }" },
	{ &isl_union_map_eq_at_multi_union_pw_aff,
	  "{ A[i,j] -> B[i',j']; A[i,j] -> C[i',j'] }",
	  "F[{ A[i,j] -> [i]; B[i,j] -> [i]; C[i,j] -> [0] }]",
	  "{ A[i,j] -> B[i,j']; A[0,j] -> C[i',j'] }" },
	{ &isl_union_map_eq_at_multi_union_pw_aff,
	  "{ A[i,j] -> B[i',j'] }",
	  "F[{ A[i,j] -> [i] : i > j; B[i,j] -> [i] }]",
	  "{ A[i,j] -> B[i,j'] : i > j }" },
	{ &isl_union_map_lex_le_at_multi_union_pw_aff,
	  "{ A[i,j] -> B[i',j'] }",
	  "F[{ A[i,j] -> [i]; B[i,j] -> [i] }, "
	    "{ A[i,j] -> [j]; B[i,j] -> [j] }]",
	  "{ A[i,j] -> B[i',j'] : i,j <<= i',j' }" },
	{ &isl_union_map_lex_lt_at_multi_union_pw_aff,
	  "{ A[i,j] -> B[i',j'] }",
	  "F[{ A[i,j] -> [i]; B[i,j] -> [i] }, "
	    "{ A[i,j] -> [j]; B[i,j] -> [j] }]",
	  "{ A[i,j] -> B[i',j'] : i,j << i',j' }" },
	{ &isl_union_map_lex_ge_at_multi_union_pw_aff,
	  "{ A[i,j] -> B[i',j'] }",
	  "F[{ A[i,j] -> [i]; B[i,j] -> [i] }, "
	    "{ A[i,j] -> [j]; B[i,j] -> [j] }]",
	  "{ A[i,j] -> B[i',j'] : i,j >>= i',j' }" },
	{ &isl_union_map_lex_gt_at_multi_union_pw_aff,
	  "{ A[i,j] -> B[i',j'] }",
	  "F[{ A[i,j] -> [i]; B[i,j] -> [i] }, "
	    "{ A[i,j] -> [j]; B[i,j] -> [j] }]",
	  "{ A[i,j] -> B[i',j'] : i,j >> i',j' }" },
	{ &isl_union_map_eq_at_multi_union_pw_aff,
	  "{ A[i,j] -> B[i',j']; A[i,j] -> C[i',j'] }",
	  "(F[] : { A[i,j]; B[i,j] })",
	  "{ A[i,j] -> B[i',j'] }" },
	{ &isl_union_map_eq_at_multi_union_pw_aff,
	  "{ A[i,j] -> B[i',j'] }",
	  "(F[] : { A[i,j] : i > j; B[i,j] : i < j })",
	  "{ A[i,j] -> B[i',j'] : i > j and i' < j' }" },
	{ &isl_union_map_eq_at_multi_union_pw_aff,
	  "[N] -> { A[i,j] -> B[i',j'] : i,i' <= N }",
	  "(F[] : { A[i,j] : i > j; B[i,j] : i < j })",
	  "[N] -> { A[i,j] -> B[i',j'] : i > j and i' < j' and i,i' <= N }" },
	{ &isl_union_map_eq_at_multi_union_pw_aff,
	  "{ A[i,j] -> B[i',j'] }",
	  "[N] -> (F[] : { A[i,j] : i < N; B[i,j] : i < N })",
	  "[N] -> { A[i,j] -> B[i',j'] : i,i' < N }" },
	{ &isl_union_map_eq_at_multi_union_pw_aff,
	  "{ A[i,j] -> B[i',j'] }",
	  "[N] -> (F[] : { : N >= 0 })",
	  "[N] -> { A[i,j] -> B[i',j'] : N >= 0 }" },
};

/* Perform some basic tests of functions that select
 * subparts of an isl_union_map based on a relation
 * specified by an isl_multi_union_pw_aff.
 */
static int test_bin_locus(isl_ctx *ctx)
{
	int i;
	isl_bool ok;
	isl_union_map *umap, *res;
	isl_multi_union_pw_aff *mupa;

	for (i = 0; i < ARRAY_SIZE(bin_locus_tests); ++i) {
		umap = isl_union_map_read_from_str(ctx,
						    bin_locus_tests[i].arg1);
		mupa = isl_multi_union_pw_aff_read_from_str(ctx,
						    bin_locus_tests[i].arg2);
		res = isl_union_map_read_from_str(ctx, bin_locus_tests[i].res);
		umap = bin_locus_tests[i].fn(umap, mupa);
		ok = isl_union_map_is_equal(umap, res);
		isl_union_map_free(umap);
		isl_union_map_free(res);
		if (ok < 0)
			return -1;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return -1);
	}

	return 0;
}

/* Inputs for basic tests of functions that determine
 * the part of the domain where two isl_multi_aff objects
 * related to each other in a specific way.
 * "fn" is the function that is being tested.
 * "arg1" and "arg2" are string descriptions of the inputs.
 * "res" is a string description of the expected result.
 */
static struct {
	__isl_give isl_set *(*fn)(__isl_take isl_multi_aff *ma1,
		__isl_take isl_multi_aff *ma2);
	const char *arg1;
	const char *arg2;
	const char *res;
} bin_locus_ma_tests[] = {
	{ &isl_multi_aff_lex_le_set, "{ [] }", "{ [] }", "{ : }" },
	{ &isl_multi_aff_lex_lt_set, "{ [] }", "{ [] }", "{ : false }" },
	{ &isl_multi_aff_lex_le_set,
	  "{ A[i] -> [i] }", "{ A[i] -> [0] }",
	  "{ A[i] : i <= 0 }" },
	{ &isl_multi_aff_lex_lt_set,
	  "{ A[i] -> [i] }", "{ A[i] -> [0] }",
	  "{ A[i] : i < 0 }" },
	{ &isl_multi_aff_lex_le_set,
	  "{ A[i] -> [i, i] }", "{ A[i] -> [0, 0] }",
	  "{ A[i] : i <= 0 }" },
	{ &isl_multi_aff_lex_le_set,
	  "{ A[i] -> [i, 0] }", "{ A[i] -> [0, 0] }",
	  "{ A[i] : i <= 0 }" },
	{ &isl_multi_aff_lex_le_set,
	  "{ A[i] -> [i, 1] }", "{ A[i] -> [0, 0] }",
	  "{ A[i] : i < 0 }" },
};

/* Perform some basic tests of functions that determine
 * the part of the domain where two isl_multi_aff objects
 * related to each other in a specific way.
 */
static isl_stat test_bin_locus_ma(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(bin_locus_ma_tests); ++i) {
		const char *str;
		isl_bool ok;
		isl_multi_aff *ma1, *ma2;
		isl_set *set, *res;

		str = bin_locus_ma_tests[i].arg1;
		ma1 = isl_multi_aff_read_from_str(ctx, str);
		str = bin_locus_ma_tests[i].arg2;
		ma2 = isl_multi_aff_read_from_str(ctx, str);
		res = isl_set_read_from_str(ctx, bin_locus_ma_tests[i].res);
		set = bin_locus_ma_tests[i].fn(ma1, ma2);
		ok = isl_set_is_equal(set, res);
		isl_set_free(set);
		isl_set_free(res);
		if (ok < 0)
			return isl_stat_error;
		if (!ok)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return isl_stat_error);
	}

	return isl_stat_ok;
}

/* Perform basic locus tests.
 */
static int test_locus(isl_ctx *ctx)
{
	if (test_un_locus(ctx) < 0)
		return -1;
	if (test_bin_locus(ctx) < 0)
		return -1;
	if (test_bin_locus_ma(ctx) < 0)
		return -1;
	return 0;
}

/* Test that isl_union_pw_qpolynomial_eval picks up the function
 * defined over the correct domain space.
 */
static int test_eval_1(isl_ctx *ctx)
{
	const char *str;
	isl_point *pnt;
	isl_set *set;
	isl_union_pw_qpolynomial *upwqp;
	isl_val *v;
	int cmp;

	str = "{ A[x] -> x^2; B[x] -> -x^2 }";
	upwqp = isl_union_pw_qpolynomial_read_from_str(ctx, str);
	str = "{ A[6] }";
	set = isl_set_read_from_str(ctx, str);
	pnt = isl_set_sample_point(set);
	v = isl_union_pw_qpolynomial_eval(upwqp, pnt);
	cmp = isl_val_cmp_si(v, 36);
	isl_val_free(v);

	if (!v)
		return -1;
	if (cmp != 0)
		isl_die(ctx, isl_error_unknown, "unexpected value", return -1);

	return 0;
}

/* Check that isl_qpolynomial_eval handles getting called on a void point.
 */
static int test_eval_2(isl_ctx *ctx)
{
	const char *str;
	isl_point *pnt;
	isl_set *set;
	isl_qpolynomial *qp;
	isl_val *v;
	isl_bool ok;

	str = "{ A[x] -> [x] }";
	qp = isl_qpolynomial_from_aff(isl_aff_read_from_str(ctx, str));
	str = "{ A[x] : false }";
	set = isl_set_read_from_str(ctx, str);
	pnt = isl_set_sample_point(set);
	v = isl_qpolynomial_eval(qp, pnt);
	ok = isl_val_is_nan(v);
	isl_val_free(v);

	if (ok < 0)
		return -1;
	if (!ok)
		isl_die(ctx, isl_error_unknown, "expecting NaN", return -1);

	return 0;
}

/* Check that a polynomial (without local variables) can be evaluated
 * in a rational point.
 */
static isl_stat test_eval_3(isl_ctx *ctx)
{
	isl_pw_qpolynomial *pwqp;
	isl_point *pnt;
	isl_val *v;
	isl_stat r;

	pwqp = isl_pw_qpolynomial_read_from_str(ctx, "{ [x] -> x^2 }");
	pnt = isl_point_zero(isl_pw_qpolynomial_get_domain_space(pwqp));
	v = isl_val_read_from_str(ctx, "1/2");
	pnt = isl_point_set_coordinate_val(pnt, isl_dim_set, 0, v);
	v = isl_pw_qpolynomial_eval(pwqp, pnt);
	r = val_check_equal(v, "1/4");
	isl_val_free(v);

	return r;
}

/* Inputs for isl_pw_aff_eval test.
 * "f" is the affine function.
 * "p" is the point where the function should be evaluated.
 * "res" is the expected result.
 */
struct {
	const char *f;
	const char *p;
	const char *res;
} aff_eval_tests[] = {
	{ "{ [i] -> [2 * i] }", "{ [4] }", "8" },
	{ "{ [i] -> [2 * i] }", "{ [x] : false }", "NaN" },
	{ "{ [i] -> [i + floor(i/2) + floor(i/3)] }", "{ [0] }", "0" },
	{ "{ [i] -> [i + floor(i/2) + floor(i/3)] }", "{ [1] }", "1" },
	{ "{ [i] -> [i + floor(i/2) + floor(i/3)] }", "{ [2] }", "3" },
	{ "{ [i] -> [i + floor(i/2) + floor(i/3)] }", "{ [3] }", "5" },
	{ "{ [i] -> [i + floor(i/2) + floor(i/3)] }", "{ [4] }", "7" },
	{ "{ [i] -> [floor((3 * floor(i/2))/5)] }", "{ [0] }", "0" },
	{ "{ [i] -> [floor((3 * floor(i/2))/5)] }", "{ [1] }", "0" },
	{ "{ [i] -> [floor((3 * floor(i/2))/5)] }", "{ [2] }", "0" },
	{ "{ [i] -> [floor((3 * floor(i/2))/5)] }", "{ [3] }", "0" },
	{ "{ [i] -> [floor((3 * floor(i/2))/5)] }", "{ [4] }", "1" },
	{ "{ [i] -> [floor((3 * floor(i/2))/5)] }", "{ [6] }", "1" },
	{ "{ [i] -> [floor((3 * floor(i/2))/5)] }", "{ [8] }", "2" },
	{ "{ [i] -> [i] : i > 0; [i] -> [-i] : i < 0 }", "{ [4] }", "4" },
	{ "{ [i] -> [i] : i > 0; [i] -> [-i] : i < 0 }", "{ [-2] }", "2" },
	{ "{ [i] -> [i] : i > 0; [i] -> [-i] : i < 0 }", "{ [0] }", "NaN" },
	{ "[N] -> { [2 * N] }", "[N] -> { : N = 4 }", "8" },
	{ "{ [i, j] -> [(i + j)/2] }", "{ [1, 1] }", "1" },
	{ "{ [i, j] -> [(i + j)/2] }", "{ [1, 2] }", "3/2" },
	{ "{ [i] -> [i] : i mod 2 = 0 }", "{ [4] }", "4" },
	{ "{ [i] -> [i] : i mod 2 = 0 }", "{ [3] }", "NaN" },
	{ "{ [i] -> [i] : i mod 2 = 0 }", "{ [x] : false }", "NaN" },
};

/* Perform basic isl_pw_aff_eval tests.
 */
static int test_eval_aff(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(aff_eval_tests); ++i) {
		isl_stat r;
		isl_pw_aff *pa;
		isl_set *set;
		isl_point *pnt;
		isl_val *v;

		pa = isl_pw_aff_read_from_str(ctx, aff_eval_tests[i].f);
		set = isl_set_read_from_str(ctx, aff_eval_tests[i].p);
		pnt = isl_set_sample_point(set);
		v = isl_pw_aff_eval(pa, pnt);
		r = val_check_equal(v, aff_eval_tests[i].res);
		isl_val_free(v);
		if (r < 0)
			return -1;
	}
	return 0;
}

/* Perform basic evaluation tests.
 */
static int test_eval(isl_ctx *ctx)
{
	if (test_eval_1(ctx) < 0)
		return -1;
	if (test_eval_2(ctx) < 0)
		return -1;
	if (test_eval_3(ctx) < 0)
		return -1;
	if (test_eval_aff(ctx) < 0)
		return -1;
	return 0;
}

/* Descriptions of sets that are tested for reparsing after printing.
 */
const char *output_tests[] = {
	"{ [1, y] : 0 <= y <= 1; [x, -x] : 0 <= x <= 1 }",
	"{ [x] : 1 = 0 }",
	"{ [x] : false }",
	"{ [x] : x mod 2 = 0 }",
	"{ [x] : x mod 2 = 1 }",
	"{ [x, y] : x mod 2 = 0 and 3*floor(y/2) < x }",
	"{ [y, x] : x mod 2 = 0 and 3*floor(y/2) < x }",
	"{ [x, y] : x mod 2 = 0 and 3*floor(y/2) = x + y }",
	"{ [y, x] : x mod 2 = 0 and 3*floor(y/2) = x + y }",
	"[n] -> { [y, x] : 2*((x + 2y) mod 3) = n }",
	"{ [x, y] : (2*floor(x/3) + 3*floor(y/4)) mod 5 = x }",
};

/* Check that printing a set and reparsing a set from the printed output
 * results in the same set.
 */
static int test_output_set(isl_ctx *ctx)
{
	int i;
	char *str;
	isl_set *set1, *set2;
	isl_bool equal;

	for (i = 0; i < ARRAY_SIZE(output_tests); ++i) {
		set1 = isl_set_read_from_str(ctx, output_tests[i]);
		str = isl_set_to_str(set1);
		set2 = isl_set_read_from_str(ctx, str);
		free(str);
		equal = isl_set_is_equal(set1, set2);
		isl_set_free(set1);
		isl_set_free(set2);
		if (equal < 0)
			return -1;
		if (!equal)
			isl_die(ctx, isl_error_unknown,
				"parsed output not the same", return -1);
	}

	return 0;
}

/* Check that an isl_multi_aff is printed using a consistent space.
 */
static isl_stat test_output_ma(isl_ctx *ctx)
{
	char *str;
	isl_bool equal;
	isl_aff *aff;
	isl_multi_aff *ma, *ma2;

	ma = isl_multi_aff_read_from_str(ctx, "{ [a, b] -> [a + b] }");
	aff = isl_aff_read_from_str(ctx, "{ [c, d] -> [c + d] }");
	ma = isl_multi_aff_set_aff(ma, 0, aff);
	str = isl_multi_aff_to_str(ma);
	ma2 = isl_multi_aff_read_from_str(ctx, str);
	free(str);
	equal = isl_multi_aff_plain_is_equal(ma, ma2);
	isl_multi_aff_free(ma2);
	isl_multi_aff_free(ma);

	if (equal < 0)
		return isl_stat_error;
	if (!equal)
		isl_die(ctx, isl_error_unknown, "bad conversion",
			return isl_stat_error);

	return isl_stat_ok;
}

/* Check that an isl_multi_pw_aff is printed using a consistent space.
 */
static isl_stat test_output_mpa(isl_ctx *ctx)
{
	char *str;
	isl_bool equal;
	isl_pw_aff *pa;
	isl_multi_pw_aff *mpa, *mpa2;

	mpa = isl_multi_pw_aff_read_from_str(ctx, "{ [a, b] -> [a + b] }");
	pa = isl_pw_aff_read_from_str(ctx, "{ [c, d] -> [c + d] }");
	mpa = isl_multi_pw_aff_set_pw_aff(mpa, 0, pa);
	str = isl_multi_pw_aff_to_str(mpa);
	mpa2 = isl_multi_pw_aff_read_from_str(ctx, str);
	free(str);
	equal = isl_multi_pw_aff_plain_is_equal(mpa, mpa2);
	isl_multi_pw_aff_free(mpa2);
	isl_multi_pw_aff_free(mpa);

	if (equal < 0)
		return isl_stat_error;
	if (!equal)
		isl_die(ctx, isl_error_unknown, "bad conversion",
			return isl_stat_error);

	return isl_stat_ok;
}

int test_output(isl_ctx *ctx)
{
	char *s;
	const char *str;
	isl_pw_aff *pa;
	isl_printer *p;
	int equal;

	if (test_output_set(ctx) < 0)
		return -1;
	if (test_output_ma(ctx) < 0)
		return -1;
	if (test_output_mpa(ctx) < 0)
		return -1;

	str = "[x] -> { [1] : x % 4 <= 2; [2] : x = 3 }";
	pa = isl_pw_aff_read_from_str(ctx, str);

	p = isl_printer_to_str(ctx);
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = isl_printer_print_pw_aff(p, pa);
	s = isl_printer_get_str(p);
	isl_printer_free(p);
	isl_pw_aff_free(pa);
	if (!s)
		equal = -1;
	else
		equal = !strcmp(s, "4 * floord(x, 4) + 2 >= x ? 1 : 2");
	free(s);
	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown, "unexpected result", return -1);

	return 0;
}

int test_sample(isl_ctx *ctx)
{
	const char *str;
	isl_basic_set *bset1, *bset2;
	int empty, subset;

	str = "{ [a, b, c, d, e, f, g, h, i, j, k] : "
	    "3i >= 1073741823b - c - 1073741823e + f and c >= 0 and "
	    "3i >= -1 + 3221225466b + c + d - 3221225466e - f and "
	    "2e >= a - b and 3e <= 2a and 3k <= -a and f <= -1 + a and "
	    "3i <= 4 - a + 4b + 2c - e - 2f and 3k <= -a + c - f and "
	    "3h >= -2 + a and 3g >= -3 - a and 3k >= -2 - a and "
	    "3i >= -2 - a - 2c + 3e + 2f and 3h <= a + c - f and "
	    "3h >= a + 2147483646b + 2c - 2147483646e - 2f and "
	    "3g <= -1 - a and 3i <= 1 + c + d - f and a <= 1073741823 and "
	    "f >= 1 - a + 1073741822b + c + d - 1073741822e and "
	    "3i >= 1 + 2b - 2c + e + 2f + 3g and "
	    "1073741822f <= 1073741822 - a + 1073741821b + 1073741822c +"
		"d - 1073741821e and "
	    "3j <= 3 - a + 3b and 3g <= -2 - 2b + c + d - e - f and "
	    "3j >= 1 - a + b + 2e and "
	    "3f >= -3 + a + 3221225462b + 3c + d - 3221225465e and "
	    "3i <= 4 - a + 4b - e and "
	    "f <= 1073741822 + 1073741822b - 1073741822e and 3h <= a and "
	    "f >= 0 and 2e <= 4 - a + 5b - d and 2e <= a - b + d and "
	    "c <= -1 + a and 3i >= -2 - a + 3e and "
	    "1073741822e <= 1073741823 - a + 1073741822b + c and "
	    "3g >= -4 + 3221225464b + 3c + d - 3221225467e - 3f and "
	    "3i >= -1 + 3221225466b + 3c + d - 3221225466e - 3f and "
	    "1073741823e >= 1 + 1073741823b - d and "
	    "3i >= 1073741823b + c - 1073741823e - f and "
	    "3i >= 1 + 2b + e + 3g }";
	bset1 = isl_basic_set_read_from_str(ctx, str);
	bset2 = isl_basic_set_sample(isl_basic_set_copy(bset1));
	empty = isl_basic_set_is_empty(bset2);
	subset = isl_basic_set_is_subset(bset2, bset1);
	isl_basic_set_free(bset1);
	isl_basic_set_free(bset2);
	if (empty < 0 || subset < 0)
		return -1;
	if (empty)
		isl_die(ctx, isl_error_unknown, "point not found", return -1);
	if (!subset)
		isl_die(ctx, isl_error_unknown, "bad point found", return -1);

	return 0;
}

/* Perform a projection on a basic set that is known to be empty
 * but that has not been assigned a canonical representation.
 * Earlier versions of isl would run into a stack overflow
 * on this example.
 */
static int test_empty_projection(isl_ctx *ctx)
{
	const char *str;
	isl_bool empty;
	isl_basic_set *bset;

	str = "{ [a, b, c, d, e, f, g, h] : 5f = 1 + 4a - b + 5c - d - 2e and "
		"3h = 2 + b + c and 14c >= 9 - 3a + 25b and "
		"4c <= 50 - 3a + 23b and 6b <= -39 + a and "
		"9g >= -6 + 3a + b + c and e < a + b - 2d and "
		"7d >= -5 + 2a + 2b and 5g >= -14 + a - 4b + d + 2e and "
		"9g <= -28 - 5b - 2c + 3d + 6e }";
	bset = isl_basic_set_read_from_str(ctx, str);
	empty = isl_basic_set_is_empty(bset);
	bset = isl_basic_set_params(bset);
	isl_basic_set_free(bset);

	if (empty < 0)
		return -1;

	return 0;
}

int test_fixed_power(isl_ctx *ctx)
{
	const char *str;
	isl_map *map;
	isl_val *exp;
	int equal;

	str = "{ [i] -> [i + 1] }";
	map = isl_map_read_from_str(ctx, str);
	exp = isl_val_int_from_si(ctx, 23);
	map = isl_map_fixed_power_val(map, exp);
	equal = map_check_equal(map, "{ [i] -> [i + 23] }");
	isl_map_free(map);
	if (equal < 0)
		return -1;

	return 0;
}

int test_slice(isl_ctx *ctx)
{
	const char *str;
	isl_map *map;
	int equal;

	str = "{ [i] -> [j] }";
	map = isl_map_read_from_str(ctx, str);
	map = isl_map_equate(map, isl_dim_in, 0, isl_dim_out, 0);
	equal = map_check_equal(map, "{ [i] -> [i] }");
	isl_map_free(map);
	if (equal < 0)
		return -1;

	str = "{ [i] -> [j] }";
	map = isl_map_read_from_str(ctx, str);
	map = isl_map_equate(map, isl_dim_in, 0, isl_dim_in, 0);
	equal = map_check_equal(map, "{ [i] -> [j] }");
	isl_map_free(map);
	if (equal < 0)
		return -1;

	str = "{ [i] -> [j] }";
	map = isl_map_read_from_str(ctx, str);
	map = isl_map_oppose(map, isl_dim_in, 0, isl_dim_out, 0);
	equal = map_check_equal(map, "{ [i] -> [-i] }");
	isl_map_free(map);
	if (equal < 0)
		return -1;

	str = "{ [i] -> [j] }";
	map = isl_map_read_from_str(ctx, str);
	map = isl_map_oppose(map, isl_dim_in, 0, isl_dim_in, 0);
	equal = map_check_equal(map, "{ [0] -> [j] }");
	isl_map_free(map);
	if (equal < 0)
		return -1;

	str = "{ [i] -> [j] }";
	map = isl_map_read_from_str(ctx, str);
	map = isl_map_order_gt(map, isl_dim_in, 0, isl_dim_out, 0);
	equal = map_check_equal(map, "{ [i] -> [j] : i > j }");
	isl_map_free(map);
	if (equal < 0)
		return -1;

	str = "{ [i] -> [j] }";
	map = isl_map_read_from_str(ctx, str);
	map = isl_map_order_gt(map, isl_dim_in, 0, isl_dim_in, 0);
	equal = map_check_equal(map, "{ [i] -> [j] : false }");
	isl_map_free(map);
	if (equal < 0)
		return -1;

	return 0;
}

int test_eliminate(isl_ctx *ctx)
{
	const char *str;
	isl_map *map;
	int equal;

	str = "{ [i] -> [j] : i = 2j }";
	map = isl_map_read_from_str(ctx, str);
	map = isl_map_eliminate(map, isl_dim_out, 0, 1);
	equal = map_check_equal(map, "{ [i] -> [j] : exists a : i = 2a }");
	isl_map_free(map);
	if (equal < 0)
		return -1;

	return 0;
}

/* Check basic functionality of isl_map_deltas_map.
 */
static int test_deltas_map(isl_ctx *ctx)
{
	const char *str;
	isl_map *map;
	int equal;

	str = "{ A[i] -> A[i + 1] }";
	map = isl_map_read_from_str(ctx, str);
	map = isl_map_deltas_map(map);
	equal = map_check_equal(map, "{ [A[i] -> A[i + 1]] -> A[1] }");
	isl_map_free(map);
	if (equal < 0)
		return -1;

	return 0;
}

/* Check that isl_set_dim_residue_class detects that the values of j
 * in the set below are all odd and that it does not detect any spurious
 * strides.
 */
static int test_residue_class(isl_ctx *ctx)
{
	const char *str;
	isl_set *set;
	isl_int m, r;
	isl_stat res;

	str = "{ [i,j] : j = 4 i + 1 and 0 <= i <= 100; "
		"[i,j] : j = 4 i + 3 and 500 <= i <= 600 }";
	set = isl_set_read_from_str(ctx, str);
	isl_int_init(m);
	isl_int_init(r);
	res = isl_set_dim_residue_class(set, 1, &m, &r);
	if (res >= 0 &&
	    (isl_int_cmp_si(m, 2) != 0 || isl_int_cmp_si(r, 1) != 0))
		isl_die(ctx, isl_error_unknown, "incorrect residue class",
			res = isl_stat_error);
	isl_int_clear(r);
	isl_int_clear(m);
	isl_set_free(set);

	return res;
}

static int test_align_parameters_1(isl_ctx *ctx)
{
	const char *str;
	isl_space *space;
	isl_multi_aff *ma1, *ma2;
	int equal;

	str = "{ A[B[] -> C[]] -> D[E[] -> F[]] }";
	ma1 = isl_multi_aff_read_from_str(ctx, str);

	space = isl_space_params_alloc(ctx, 1);
	space = isl_space_set_dim_name(space, isl_dim_param, 0, "N");
	ma1 = isl_multi_aff_align_params(ma1, space);

	str = "[N] -> { A[B[] -> C[]] -> D[E[] -> F[]] }";
	ma2 = isl_multi_aff_read_from_str(ctx, str);

	equal = isl_multi_aff_plain_is_equal(ma1, ma2);

	isl_multi_aff_free(ma1);
	isl_multi_aff_free(ma2);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown,
			"result not as expected", return -1);

	return 0;
}

/* Check the isl_multi_*_from_*_list operation in case inputs
 * have unaligned parameters.
 * In particular, older versions of isl would simply fail
 * (without printing any error message).
 */
static isl_stat test_align_parameters_2(isl_ctx *ctx)
{
	isl_space *space;
	isl_map *map;
	isl_aff *aff;
	isl_multi_aff *ma;

	map = isl_map_read_from_str(ctx, "{ A[] -> M[x] }");
	space = isl_map_get_space(map);
	isl_map_free(map);

	aff = isl_aff_read_from_str(ctx, "[N] -> { A[] -> [N] }");
	ma = isl_multi_aff_from_aff_list(space, isl_aff_list_from_aff(aff));
	isl_multi_aff_free(ma);

	if (!ma)
		return isl_stat_error;
	return isl_stat_ok;
}

/* Perform basic parameter alignment tests.
 */
static int test_align_parameters(isl_ctx *ctx)
{
	if (test_align_parameters_1(ctx) < 0)
		return -1;
	if (test_align_parameters_2(ctx) < 0)
		return -1;

	return 0;
}

/* Check that isl_*_drop_unused_params actually drops the unused parameters
 * by comparing the result using isl_*_plain_is_equal.
 * Note that this assumes that isl_*_plain_is_equal does not consider
 * objects that only differ by unused parameters to be equal.
 */
int test_drop_unused_parameters(isl_ctx *ctx)
{
	const char *str_with, *str_without;
	isl_basic_set *bset1, *bset2;
	isl_set *set1, *set2;
	isl_pw_aff *pwa1, *pwa2;
	int equal;

	str_with = "[n, m, o] -> { [m] }";
	str_without = "[m] -> { [m] }";

	bset1 = isl_basic_set_read_from_str(ctx, str_with);
	bset2 = isl_basic_set_read_from_str(ctx, str_without);
	bset1 = isl_basic_set_drop_unused_params(bset1);
	equal = isl_basic_set_plain_is_equal(bset1, bset2);
	isl_basic_set_free(bset1);
	isl_basic_set_free(bset2);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown,
			"result not as expected", return -1);

	set1 = isl_set_read_from_str(ctx, str_with);
	set2 = isl_set_read_from_str(ctx, str_without);
	set1 = isl_set_drop_unused_params(set1);
	equal = isl_set_plain_is_equal(set1, set2);
	isl_set_free(set1);
	isl_set_free(set2);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown,
			"result not as expected", return -1);

	pwa1 = isl_pw_aff_read_from_str(ctx, str_with);
	pwa2 = isl_pw_aff_read_from_str(ctx, str_without);
	pwa1 = isl_pw_aff_drop_unused_params(pwa1);
	equal = isl_pw_aff_plain_is_equal(pwa1, pwa2);
	isl_pw_aff_free(pwa1);
	isl_pw_aff_free(pwa2);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown,
			"result not as expected", return -1);

	return 0;
}

static int test_list(isl_ctx *ctx)
{
	isl_id *a, *b, *c, *d, *id;
	isl_id_list *list;
	isl_size n;
	int ok;

	a = isl_id_alloc(ctx, "a", NULL);
	b = isl_id_alloc(ctx, "b", NULL);
	c = isl_id_alloc(ctx, "c", NULL);
	d = isl_id_alloc(ctx, "d", NULL);

	list = isl_id_list_alloc(ctx, 4);
	list = isl_id_list_add(list, b);
	list = isl_id_list_insert(list, 0, a);
	list = isl_id_list_add(list, c);
	list = isl_id_list_add(list, d);
	list = isl_id_list_drop(list, 1, 1);

	n = isl_id_list_n_id(list);
	if (n < 0)
		return -1;
	if (n != 3) {
		isl_id_list_free(list);
		isl_die(ctx, isl_error_unknown,
			"unexpected number of elements in list", return -1);
	}

	id = isl_id_list_get_id(list, 0);
	ok = id == a;
	isl_id_free(id);
	id = isl_id_list_get_id(list, 1);
	ok = ok && id == c;
	isl_id_free(id);
	id = isl_id_list_get_id(list, 2);
	ok = ok && id == d;
	isl_id_free(id);

	isl_id_list_free(list);

	if (!ok)
		isl_die(ctx, isl_error_unknown,
			"unexpected elements in list", return -1);

	return 0;
}

/* Check the conversion from an isl_multi_aff to an isl_basic_set.
 */
static isl_stat test_ma_conversion(isl_ctx *ctx)
{
	const char *str;
	isl_bool equal;
	isl_multi_aff *ma;
	isl_basic_set *bset1, *bset2;

	str = "[N] -> { A[0, N + 1] }";
	ma = isl_multi_aff_read_from_str(ctx, str);
	bset1 = isl_basic_set_read_from_str(ctx, str);
	bset2 = isl_basic_set_from_multi_aff(ma);
	equal = isl_basic_set_is_equal(bset1, bset2);
	isl_basic_set_free(bset1);
	isl_basic_set_free(bset2);
	if (equal < 0)
		return isl_stat_error;
	if (!equal)
		isl_die(ctx, isl_error_unknown, "bad conversion",
			return isl_stat_error);
	return isl_stat_ok;
}

const char *set_conversion_tests[] = {
	"[N] -> { [i] : N - 1 <= 2 i <= N }",
	"[N] -> { [i] : exists a : i = 4 a and N - 1 <= i <= N }",
	"[N] -> { [i,j] : exists a : i = 4 a and N - 1 <= i, 2j <= N }",
	"[N] -> { [[i]->[j]] : exists a : i = 4 a and N - 1 <= i, 2j <= N }",
	"[N] -> { [3*floor(N/2) + 5*floor(N/3)] }",
	"[a, b] -> { [c, d] : (4*floor((-a + c)/4) = -a + c and "
			"32*floor((-b + d)/32) = -b + d and 5 <= c <= 8 and "
			"-3 + c <= d <= 28 + c) }",
};

/* Check that converting from isl_set to isl_pw_multi_aff and back
 * to isl_set produces the original isl_set.
 */
static int test_set_conversion(isl_ctx *ctx)
{
	int i;
	const char *str;
	isl_set *set1, *set2;
	isl_pw_multi_aff *pma;
	int equal;

	for (i = 0; i < ARRAY_SIZE(set_conversion_tests); ++i) {
		str = set_conversion_tests[i];
		set1 = isl_set_read_from_str(ctx, str);
		pma = isl_pw_multi_aff_from_set(isl_set_copy(set1));
		set2 = isl_set_from_pw_multi_aff(pma);
		equal = isl_set_is_equal(set1, set2);
		isl_set_free(set1);
		isl_set_free(set2);

		if (equal < 0)
			return -1;
		if (!equal)
			isl_die(ctx, isl_error_unknown, "bad conversion",
				return -1);
	}

	return 0;
}

const char *conversion_tests[] = {
	"{ [a, b, c, d] -> s0[a, b, e, f] : "
	    "exists (e0 = [(a - 2c)/3], e1 = [(-4 + b - 5d)/9], "
	    "e2 = [(-d + f)/9]: 3e0 = a - 2c and 9e1 = -4 + b - 5d and "
	    "9e2 = -d + f and f >= 0 and f <= 8 and 9e >= -5 - 2a and "
	    "9e <= -2 - 2a) }",
	"{ [a, b] -> [c] : exists (e0 = floor((-a - b + c)/5): "
	    "5e0 = -a - b + c and c >= -a and c <= 4 - a) }",
	"{ [a, b] -> [c] : exists d : 18 * d = -3 - a + 2c and 1 <= c <= 3 }",
};

/* Check that converting from isl_map to isl_pw_multi_aff and back
 * to isl_map produces the original isl_map.
 */
static int test_map_conversion(isl_ctx *ctx)
{
	int i;
	isl_map *map1, *map2;
	isl_pw_multi_aff *pma;
	int equal;

	for (i = 0; i < ARRAY_SIZE(conversion_tests); ++i) {
		map1 = isl_map_read_from_str(ctx, conversion_tests[i]);
		pma = isl_pw_multi_aff_from_map(isl_map_copy(map1));
		map2 = isl_map_from_pw_multi_aff(pma);
		equal = isl_map_is_equal(map1, map2);
		isl_map_free(map1);
		isl_map_free(map2);

		if (equal < 0)
			return -1;
		if (!equal)
			isl_die(ctx, isl_error_unknown, "bad conversion",
				return -1);
	}

	return 0;
}

/* Descriptions of isl_pw_multi_aff objects for testing conversion
 * to isl_multi_pw_aff and back.
 */
const char *mpa_conversion_tests[] = {
	"{ [x] -> A[x] }",
	"{ [x] -> A[x] : x >= 0 }",
	"{ [x] -> A[x] : x >= 0; [x] -> A[-x] : x < 0 }",
	"{ [x] -> A[x, x + 1] }",
	"{ [x] -> A[] }",
	"{ [x] -> A[] : x >= 0 }",
};

/* Check that conversion from isl_pw_multi_aff to isl_multi_pw_aff and
 * back to isl_pw_multi_aff preserves the original meaning.
 */
static int test_mpa_conversion(isl_ctx *ctx)
{
	int i;
	isl_pw_multi_aff *pma1, *pma2;
	isl_multi_pw_aff *mpa;
	int equal;

	for (i = 0; i < ARRAY_SIZE(mpa_conversion_tests); ++i) {
		const char *str;
		str = mpa_conversion_tests[i];
		pma1 = isl_pw_multi_aff_read_from_str(ctx, str);
		pma2 = isl_pw_multi_aff_copy(pma1);
		mpa = isl_multi_pw_aff_from_pw_multi_aff(pma1);
		pma1 = isl_pw_multi_aff_from_multi_pw_aff(mpa);
		equal = isl_pw_multi_aff_plain_is_equal(pma1, pma2);
		isl_pw_multi_aff_free(pma1);
		isl_pw_multi_aff_free(pma2);

		if (equal < 0)
			return -1;
		if (!equal)
			isl_die(ctx, isl_error_unknown, "bad conversion",
				return -1);
	}

	return 0;
}

/* Descriptions of union maps that should be convertible
 * to an isl_multi_union_pw_aff.
 */
const char *umap_mupa_conversion_tests[] = {
	"{ [a, b, c, d] -> s0[a, b, e, f] : "
	    "exists (e0 = [(a - 2c)/3], e1 = [(-4 + b - 5d)/9], "
	    "e2 = [(-d + f)/9]: 3e0 = a - 2c and 9e1 = -4 + b - 5d and "
	    "9e2 = -d + f and f >= 0 and f <= 8 and 9e >= -5 - 2a and "
	    "9e <= -2 - 2a) }",
	"{ [a, b] -> [c] : exists (e0 = floor((-a - b + c)/5): "
	    "5e0 = -a - b + c and c >= -a and c <= 4 - a) }",
	"{ [a, b] -> [c] : exists d : 18 * d = -3 - a + 2c and 1 <= c <= 3 }",
	"{ A[] -> B[0]; C[] -> B[1] }",
	"{ A[] -> B[]; C[] -> B[] }",
};

/* Check that converting from isl_union_map to isl_multi_union_pw_aff and back
 * to isl_union_map produces the original isl_union_map.
 */
static int test_union_map_mupa_conversion(isl_ctx *ctx)
{
	int i;
	isl_union_map *umap1, *umap2;
	isl_multi_union_pw_aff *mupa;
	int equal;

	for (i = 0; i < ARRAY_SIZE(umap_mupa_conversion_tests); ++i) {
		const char *str;
		str = umap_mupa_conversion_tests[i];
		umap1 = isl_union_map_read_from_str(ctx, str);
		umap2 = isl_union_map_copy(umap1);
		mupa = isl_multi_union_pw_aff_from_union_map(umap2);
		umap2 = isl_union_map_from_multi_union_pw_aff(mupa);
		equal = isl_union_map_is_equal(umap1, umap2);
		isl_union_map_free(umap1);
		isl_union_map_free(umap2);

		if (equal < 0)
			return -1;
		if (!equal)
			isl_die(ctx, isl_error_unknown, "bad conversion",
				return -1);
	}

	return 0;
}

static int test_conversion(isl_ctx *ctx)
{
	if (test_ma_conversion(ctx) < 0)
		return -1;
	if (test_set_conversion(ctx) < 0)
		return -1;
	if (test_map_conversion(ctx) < 0)
		return -1;
	if (test_mpa_conversion(ctx) < 0)
		return -1;
	if (test_union_map_mupa_conversion(ctx) < 0)
		return -1;
	return 0;
}

/* Check that isl_basic_map_curry does not modify input.
 */
static int test_curry(isl_ctx *ctx)
{
	const char *str;
	isl_basic_map *bmap1, *bmap2;
	int equal;

	str = "{ [A[] -> B[]] -> C[] }";
	bmap1 = isl_basic_map_read_from_str(ctx, str);
	bmap2 = isl_basic_map_curry(isl_basic_map_copy(bmap1));
	equal = isl_basic_map_is_equal(bmap1, bmap2);
	isl_basic_map_free(bmap1);
	isl_basic_map_free(bmap2);

	if (equal < 0)
		return -1;
	if (equal)
		isl_die(ctx, isl_error_unknown,
			"curried map should not be equal to original",
			return -1);

	return 0;
}

struct {
	const char *ma1;
	const char *ma;
	const char *res;
} pullback_tests[] = {
	{ "{ B[i,j] -> C[i + 2j] }" , "{ A[a,b] -> B[b,a] }",
	  "{ A[a,b] -> C[b + 2a] }" },
	{ "{ B[i] -> C[2i] }", "{ A[a] -> B[(a)/2] }", "{ A[a] -> C[a] }" },
	{ "{ B[i] -> C[(i)/2] }", "{ A[a] -> B[2a] }", "{ A[a] -> C[a] }" },
	{ "{ B[i] -> C[(i)/2] }", "{ A[a] -> B[(a)/3] }",
	  "{ A[a] -> C[(a)/6] }" },
	{ "{ B[i] -> C[2i] }", "{ A[a] -> B[5a] }", "{ A[a] -> C[10a] }" },
	{ "{ B[i] -> C[2i] }", "{ A[a] -> B[(a)/3] }",
	  "{ A[a] -> C[(2a)/3] }" },
	{ "{ B[i,j] -> C[i + j] }", "{ A[a] -> B[a,a] }", "{ A[a] -> C[2a] }"},
	{ "{ B[a] -> C[a,a] }", "{ A[i,j] -> B[i + j] }",
	  "{ A[i,j] -> C[i + j, i + j] }"},
	{ "{ B[i] -> C[([i/2])] }", "{ B[5] }", "{ C[2] }" },
	{ "[n] -> { B[i,j] -> C[([i/2]) + 2j] }",
	  "[n] -> { B[n,[n/3]] }", "[n] -> { C[([n/2]) + 2*[n/3]] }", },
	{ "{ [i, j] -> [floor((i)/4) + floor((2*i+j)/5)] }",
	  "{ [i, j] -> [floor((i)/3), j] }",
	  "{ [i, j] -> [(floor((i)/12) + floor((j + 2*floor((i)/3))/5))] }" },
};

static int test_pullback(isl_ctx *ctx)
{
	int i;
	isl_multi_aff *ma1, *ma2;
	isl_multi_aff *ma;
	int equal;

	for (i = 0; i < ARRAY_SIZE(pullback_tests); ++i) {
		ma1 = isl_multi_aff_read_from_str(ctx, pullback_tests[i].ma1);
		ma = isl_multi_aff_read_from_str(ctx, pullback_tests[i].ma);
		ma2 = isl_multi_aff_read_from_str(ctx, pullback_tests[i].res);
		ma1 = isl_multi_aff_pullback_multi_aff(ma1, ma);
		equal = isl_multi_aff_plain_is_equal(ma1, ma2);
		isl_multi_aff_free(ma1);
		isl_multi_aff_free(ma2);
		if (equal < 0)
			return -1;
		if (!equal)
			isl_die(ctx, isl_error_unknown, "bad pullback",
				return -1);
	}

	return 0;
}

/* Check that negation is printed correctly and that equal expressions
 * are correctly identified.
 */
static int test_ast(isl_ctx *ctx)
{
	isl_ast_expr *expr, *expr1, *expr2, *expr3;
	char *str;
	int ok, equal;

	expr1 = isl_ast_expr_from_id(isl_id_alloc(ctx, "A", NULL));
	expr2 = isl_ast_expr_from_id(isl_id_alloc(ctx, "B", NULL));
	expr = isl_ast_expr_add(expr1, expr2);
	expr2 = isl_ast_expr_copy(expr);
	expr = isl_ast_expr_neg(expr);
	expr2 = isl_ast_expr_neg(expr2);
	equal = isl_ast_expr_is_equal(expr, expr2);
	str = isl_ast_expr_to_C_str(expr);
	ok = str ? !strcmp(str, "-(A + B)") : -1;
	free(str);
	isl_ast_expr_free(expr);
	isl_ast_expr_free(expr2);

	if (ok < 0 || equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown,
			"equal expressions not considered equal", return -1);
	if (!ok)
		isl_die(ctx, isl_error_unknown,
			"isl_ast_expr printed incorrectly", return -1);

	expr1 = isl_ast_expr_from_id(isl_id_alloc(ctx, "A", NULL));
	expr2 = isl_ast_expr_from_id(isl_id_alloc(ctx, "B", NULL));
	expr = isl_ast_expr_add(expr1, expr2);
	expr3 = isl_ast_expr_from_id(isl_id_alloc(ctx, "C", NULL));
	expr = isl_ast_expr_sub(expr3, expr);
	str = isl_ast_expr_to_C_str(expr);
	ok = str ? !strcmp(str, "C - (A + B)") : -1;
	free(str);
	isl_ast_expr_free(expr);

	if (ok < 0)
		return -1;
	if (!ok)
		isl_die(ctx, isl_error_unknown,
			"isl_ast_expr printed incorrectly", return -1);

	return 0;
}

/* Check that isl_ast_build_expr_from_set returns a valid expression
 * for an empty set.  Note that isl_ast_build_expr_from_set getting
 * called on an empty set probably indicates a bug in the caller.
 */
static int test_ast_build(isl_ctx *ctx)
{
	isl_set *set;
	isl_ast_build *build;
	isl_ast_expr *expr;

	set = isl_set_universe(isl_space_params_alloc(ctx, 0));
	build = isl_ast_build_from_context(set);

	set = isl_set_empty(isl_space_params_alloc(ctx, 0));
	expr = isl_ast_build_expr_from_set(build, set);

	isl_ast_expr_free(expr);
	isl_ast_build_free(build);

	if (!expr)
		return -1;

	return 0;
}

/* Internal data structure for before_for and after_for callbacks.
 *
 * depth is the current depth
 * before is the number of times before_for has been called
 * after is the number of times after_for has been called
 */
struct isl_test_codegen_data {
	int depth;
	int before;
	int after;
};

/* This function is called before each for loop in the AST generated
 * from test_ast_gen1.
 *
 * Increment the number of calls and the depth.
 * Check that the space returned by isl_ast_build_get_schedule_space
 * matches the target space of the schedule returned by
 * isl_ast_build_get_schedule.
 * Return an isl_id that is checked by the corresponding call
 * to after_for.
 */
static __isl_give isl_id *before_for(__isl_keep isl_ast_build *build,
	void *user)
{
	struct isl_test_codegen_data *data = user;
	isl_ctx *ctx;
	isl_space *space;
	isl_union_map *schedule;
	isl_union_set *uset;
	isl_set *set;
	isl_bool empty;
	isl_size n;
	char name[] = "d0";

	ctx = isl_ast_build_get_ctx(build);

	if (data->before >= 3)
		isl_die(ctx, isl_error_unknown,
			"unexpected number of for nodes", return NULL);
	if (data->depth < 0 || data->depth >= 2)
		isl_die(ctx, isl_error_unknown,
			"unexpected depth", return NULL);

	snprintf(name, sizeof(name), "d%d", data->depth);
	data->before++;
	data->depth++;

	schedule = isl_ast_build_get_schedule(build);
	uset = isl_union_map_range(schedule);
	n = isl_union_set_n_set(uset);
	if (n != 1) {
		isl_union_set_free(uset);
		if (n < 0)
			return NULL;
		isl_die(ctx, isl_error_unknown,
			"expecting single range space", return NULL);
	}

	space = isl_ast_build_get_schedule_space(build);
	set = isl_union_set_extract_set(uset, space);
	isl_union_set_free(uset);
	empty = isl_set_is_empty(set);
	isl_set_free(set);

	if (empty < 0)
		return NULL;
	if (empty)
		isl_die(ctx, isl_error_unknown,
			"spaces don't match", return NULL);

	return isl_id_alloc(ctx, name, NULL);
}

/* This function is called after each for loop in the AST generated
 * from test_ast_gen1.
 *
 * Increment the number of calls and decrement the depth.
 * Check that the annotation attached to the node matches
 * the isl_id returned by the corresponding call to before_for.
 */
static __isl_give isl_ast_node *after_for(__isl_take isl_ast_node *node,
	__isl_keep isl_ast_build *build, void *user)
{
	struct isl_test_codegen_data *data = user;
	isl_id *id;
	const char *name;
	int valid;

	data->after++;
	data->depth--;

	if (data->after > data->before)
		isl_die(isl_ast_node_get_ctx(node), isl_error_unknown,
			"mismatch in number of for nodes",
			return isl_ast_node_free(node));

	id = isl_ast_node_get_annotation(node);
	if (!id)
		isl_die(isl_ast_node_get_ctx(node), isl_error_unknown,
			"missing annotation", return isl_ast_node_free(node));

	name = isl_id_get_name(id);
	valid = name && atoi(name + 1) == data->depth;
	isl_id_free(id);

	if (!valid)
		isl_die(isl_ast_node_get_ctx(node), isl_error_unknown,
			"wrong annotation", return isl_ast_node_free(node));

	return node;
}

/* Check that the before_each_for and after_each_for callbacks
 * are called for each for loop in the generated code,
 * that they are called in the right order and that the isl_id
 * returned from the before_each_for callback is attached to
 * the isl_ast_node passed to the corresponding after_each_for call.
 */
static int test_ast_gen1(isl_ctx *ctx)
{
	const char *str;
	isl_set *set;
	isl_union_map *schedule;
	isl_ast_build *build;
	isl_ast_node *tree;
	struct isl_test_codegen_data data;

	str = "[N] -> { : N >= 10 }";
	set = isl_set_read_from_str(ctx, str);
	str = "[N] -> { A[i,j] -> S[8,i,3,j] : 0 <= i,j <= N; "
		    "B[i,j] -> S[8,j,9,i] : 0 <= i,j <= N }";
	schedule = isl_union_map_read_from_str(ctx, str);

	data.before = 0;
	data.after = 0;
	data.depth = 0;
	build = isl_ast_build_from_context(set);
	build = isl_ast_build_set_before_each_for(build,
			&before_for, &data);
	build = isl_ast_build_set_after_each_for(build,
			&after_for, &data);
	tree = isl_ast_build_node_from_schedule_map(build, schedule);
	isl_ast_build_free(build);
	if (!tree)
		return -1;

	isl_ast_node_free(tree);

	if (data.before != 3 || data.after != 3)
		isl_die(ctx, isl_error_unknown,
			"unexpected number of for nodes", return -1);

	return 0;
}

/* Check that the AST generator handles domains that are integrally disjoint
 * but not rationally disjoint.
 */
static int test_ast_gen2(isl_ctx *ctx)
{
	const char *str;
	isl_set *set;
	isl_union_map *schedule;
	isl_union_map *options;
	isl_ast_build *build;
	isl_ast_node *tree;

	str = "{ A[i,j] -> [i,j] : 0 <= i,j <= 1 }";
	schedule = isl_union_map_read_from_str(ctx, str);
	set = isl_set_universe(isl_space_params_alloc(ctx, 0));
	build = isl_ast_build_from_context(set);

	str = "{ [i,j] -> atomic[1] : i + j = 1; [i,j] -> unroll[1] : i = j }";
	options = isl_union_map_read_from_str(ctx, str);
	build = isl_ast_build_set_options(build, options);
	tree = isl_ast_build_node_from_schedule_map(build, schedule);
	isl_ast_build_free(build);
	if (!tree)
		return -1;
	isl_ast_node_free(tree);

	return 0;
}

/* Increment *user on each call.
 */
static __isl_give isl_ast_node *count_domains(__isl_take isl_ast_node *node,
	__isl_keep isl_ast_build *build, void *user)
{
	int *n = user;

	(*n)++;

	return node;
}

/* Test that unrolling tries to minimize the number of instances.
 * In particular, for the schedule given below, make sure it generates
 * 3 nodes (rather than 101).
 */
static int test_ast_gen3(isl_ctx *ctx)
{
	const char *str;
	isl_set *set;
	isl_union_map *schedule;
	isl_union_map *options;
	isl_ast_build *build;
	isl_ast_node *tree;
	int n_domain = 0;

	str = "[n] -> { A[i] -> [i] : 0 <= i <= 100 and n <= i <= n + 2 }";
	schedule = isl_union_map_read_from_str(ctx, str);
	set = isl_set_universe(isl_space_params_alloc(ctx, 0));

	str = "{ [i] -> unroll[0] }";
	options = isl_union_map_read_from_str(ctx, str);

	build = isl_ast_build_from_context(set);
	build = isl_ast_build_set_options(build, options);
	build = isl_ast_build_set_at_each_domain(build,
			&count_domains, &n_domain);
	tree = isl_ast_build_node_from_schedule_map(build, schedule);
	isl_ast_build_free(build);
	if (!tree)
		return -1;

	isl_ast_node_free(tree);

	if (n_domain != 3)
		isl_die(ctx, isl_error_unknown,
			"unexpected number of for nodes", return -1);

	return 0;
}

/* Check that if the ast_build_exploit_nested_bounds options is set,
 * we do not get an outer if node in the generated AST,
 * while we do get such an outer if node if the options is not set.
 */
static int test_ast_gen4(isl_ctx *ctx)
{
	const char *str;
	isl_set *set;
	isl_union_map *schedule;
	isl_ast_build *build;
	isl_ast_node *tree;
	enum isl_ast_node_type type;
	int enb;

	enb = isl_options_get_ast_build_exploit_nested_bounds(ctx);
	str = "[N,M] -> { A[i,j] -> [i,j] : 0 <= i <= N and 0 <= j <= M }";

	isl_options_set_ast_build_exploit_nested_bounds(ctx, 1);

	schedule = isl_union_map_read_from_str(ctx, str);
	set = isl_set_universe(isl_space_params_alloc(ctx, 0));
	build = isl_ast_build_from_context(set);
	tree = isl_ast_build_node_from_schedule_map(build, schedule);
	isl_ast_build_free(build);
	if (!tree)
		return -1;

	type = isl_ast_node_get_type(tree);
	isl_ast_node_free(tree);

	if (type == isl_ast_node_if)
		isl_die(ctx, isl_error_unknown,
			"not expecting if node", return -1);

	isl_options_set_ast_build_exploit_nested_bounds(ctx, 0);

	schedule = isl_union_map_read_from_str(ctx, str);
	set = isl_set_universe(isl_space_params_alloc(ctx, 0));
	build = isl_ast_build_from_context(set);
	tree = isl_ast_build_node_from_schedule_map(build, schedule);
	isl_ast_build_free(build);
	if (!tree)
		return -1;

	type = isl_ast_node_get_type(tree);
	isl_ast_node_free(tree);

	if (type != isl_ast_node_if)
		isl_die(ctx, isl_error_unknown,
			"expecting if node", return -1);

	isl_options_set_ast_build_exploit_nested_bounds(ctx, enb);

	return 0;
}

/* This function is called for each leaf in the AST generated
 * from test_ast_gen5.
 *
 * We finalize the AST generation by extending the outer schedule
 * with a zero-dimensional schedule.  If this results in any for loops,
 * then this means that we did not pass along enough information
 * about the outer schedule to the inner AST generation.
 */
static __isl_give isl_ast_node *create_leaf(__isl_take isl_ast_build *build,
	void *user)
{
	isl_union_map *schedule, *extra;
	isl_ast_node *tree;

	schedule = isl_ast_build_get_schedule(build);
	extra = isl_union_map_copy(schedule);
	extra = isl_union_map_from_domain(isl_union_map_domain(extra));
	schedule = isl_union_map_range_product(schedule, extra);
	tree = isl_ast_build_node_from_schedule_map(build, schedule);
	isl_ast_build_free(build);

	if (!tree)
		return NULL;

	if (isl_ast_node_get_type(tree) == isl_ast_node_for)
		isl_die(isl_ast_node_get_ctx(tree), isl_error_unknown,
			"code should not contain any for loop",
			return isl_ast_node_free(tree));

	return tree;
}

/* Check that we do not lose any information when going back and
 * forth between internal and external schedule.
 *
 * In particular, we create an AST where we unroll the only
 * non-constant dimension in the schedule.  We therefore do
 * not expect any for loops in the AST.  However, older versions
 * of isl would not pass along enough information about the outer
 * schedule when performing an inner code generation from a create_leaf
 * callback, resulting in the inner code generation producing a for loop.
 */
static int test_ast_gen5(isl_ctx *ctx)
{
	const char *str;
	isl_set *set;
	isl_union_map *schedule, *options;
	isl_ast_build *build;
	isl_ast_node *tree;

	str = "{ A[] -> [1, 1, 2]; B[i] -> [1, i, 0] : i >= 1 and i <= 2 }";
	schedule = isl_union_map_read_from_str(ctx, str);

	str = "{ [a, b, c] -> unroll[1] : exists (e0 = [(a)/4]: "
				"4e0 >= -1 + a - b and 4e0 <= -2 + a + b) }";
	options = isl_union_map_read_from_str(ctx, str);

	set = isl_set_universe(isl_space_params_alloc(ctx, 0));
	build = isl_ast_build_from_context(set);
	build = isl_ast_build_set_options(build, options);
        build = isl_ast_build_set_create_leaf(build, &create_leaf, NULL);
	tree = isl_ast_build_node_from_schedule_map(build, schedule);
	isl_ast_build_free(build);
	isl_ast_node_free(tree);
	if (!tree)
		return -1;

	return 0;
}

/* Check that the expression
 *
 *	[n] -> { [n/2] : n <= 0 and n % 2 = 0; [0] : n > 0 }
 *
 * is not combined into
 *
 *	min(n/2, 0)
 *
 * as this would result in n/2 being evaluated in parts of
 * the definition domain where n is not a multiple of 2.
 */
static int test_ast_expr(isl_ctx *ctx)
{
	const char *str;
	isl_pw_aff *pa;
	isl_ast_build *build;
	isl_ast_expr *expr;
	int min_max;
	int is_min;

	min_max = isl_options_get_ast_build_detect_min_max(ctx);
	isl_options_set_ast_build_detect_min_max(ctx, 1);

	str = "[n] -> { [n/2] : n <= 0 and n % 2 = 0; [0] : n > 0 }";
	pa = isl_pw_aff_read_from_str(ctx, str);
	build = isl_ast_build_alloc(ctx);
	expr = isl_ast_build_expr_from_pw_aff(build, pa);
	is_min = isl_ast_expr_get_type(expr) == isl_ast_expr_op &&
		 isl_ast_expr_get_op_type(expr) == isl_ast_expr_op_min;
	isl_ast_build_free(build);
	isl_ast_expr_free(expr);

	isl_options_set_ast_build_detect_min_max(ctx, min_max);

	if (!expr)
		return -1;
	if (is_min)
		isl_die(ctx, isl_error_unknown,
			"expressions should not be combined", return -1);

	return 0;
}

static int test_ast_gen(isl_ctx *ctx)
{
	if (test_ast_gen1(ctx) < 0)
		return -1;
	if (test_ast_gen2(ctx) < 0)
		return -1;
	if (test_ast_gen3(ctx) < 0)
		return -1;
	if (test_ast_gen4(ctx) < 0)
		return -1;
	if (test_ast_gen5(ctx) < 0)
		return -1;
	if (test_ast_expr(ctx) < 0)
		return -1;
	return 0;
}

/* Check if dropping output dimensions from an isl_pw_multi_aff
 * works properly.
 */
static int test_pw_multi_aff(isl_ctx *ctx)
{
	const char *str;
	isl_pw_multi_aff *pma1, *pma2;
	int equal;

	str = "{ [i,j] -> [i+j, 4i-j] }";
	pma1 = isl_pw_multi_aff_read_from_str(ctx, str);
	str = "{ [i,j] -> [4i-j] }";
	pma2 = isl_pw_multi_aff_read_from_str(ctx, str);

	pma1 = isl_pw_multi_aff_drop_dims(pma1, isl_dim_out, 0, 1);

	equal = isl_pw_multi_aff_plain_is_equal(pma1, pma2);

	isl_pw_multi_aff_free(pma1);
	isl_pw_multi_aff_free(pma2);
	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown,
			"expressions not equal", return -1);

	return 0;
}

/* Check that we can properly parse multi piecewise affine expressions
 * where the piecewise affine expressions have different domains.
 */
static int test_multi_pw_aff_1(isl_ctx *ctx)
{
	const char *str;
	isl_set *dom, *dom2;
	isl_multi_pw_aff *mpa1, *mpa2;
	isl_pw_aff *pa;
	int equal;
	int equal_domain;

	mpa1 = isl_multi_pw_aff_read_from_str(ctx, "{ [i] -> [i] }");
	dom = isl_set_read_from_str(ctx, "{ [i] : i > 0 }");
	mpa1 = isl_multi_pw_aff_intersect_domain(mpa1, dom);
	mpa2 = isl_multi_pw_aff_read_from_str(ctx, "{ [i] -> [2i] }");
	mpa2 = isl_multi_pw_aff_flat_range_product(mpa1, mpa2);
	str = "{ [i] -> [(i : i > 0), 2i] }";
	mpa1 = isl_multi_pw_aff_read_from_str(ctx, str);

	equal = isl_multi_pw_aff_plain_is_equal(mpa1, mpa2);

	pa = isl_multi_pw_aff_get_pw_aff(mpa1, 0);
	dom = isl_pw_aff_domain(pa);
	pa = isl_multi_pw_aff_get_pw_aff(mpa1, 1);
	dom2 = isl_pw_aff_domain(pa);
	equal_domain = isl_set_is_equal(dom, dom2);

	isl_set_free(dom);
	isl_set_free(dom2);
	isl_multi_pw_aff_free(mpa1);
	isl_multi_pw_aff_free(mpa2);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown,
			"expressions not equal", return -1);

	if (equal_domain < 0)
		return -1;
	if (equal_domain)
		isl_die(ctx, isl_error_unknown,
			"domains unexpectedly equal", return -1);

	return 0;
}

/* Check that the dimensions in the explicit domain
 * of a multi piecewise affine expression are properly
 * taken into account.
 */
static int test_multi_pw_aff_2(isl_ctx *ctx)
{
	const char *str;
	isl_bool involves1, involves2, involves3, equal;
	isl_multi_pw_aff *mpa, *mpa1, *mpa2;

	str = "{ A[x,y] -> B[] : x >= y }";
	mpa = isl_multi_pw_aff_read_from_str(ctx, str);
	involves1 = isl_multi_pw_aff_involves_dims(mpa, isl_dim_in, 0, 2);
	mpa1 = isl_multi_pw_aff_copy(mpa);

	mpa = isl_multi_pw_aff_insert_dims(mpa, isl_dim_in, 0, 1);
	involves2 = isl_multi_pw_aff_involves_dims(mpa, isl_dim_in, 0, 1);
	involves3 = isl_multi_pw_aff_involves_dims(mpa, isl_dim_in, 1, 2);
	str = "{ [a,x,y] -> B[] : x >= y }";
	mpa2 = isl_multi_pw_aff_read_from_str(ctx, str);
	equal = isl_multi_pw_aff_plain_is_equal(mpa, mpa2);
	isl_multi_pw_aff_free(mpa2);

	mpa = isl_multi_pw_aff_drop_dims(mpa, isl_dim_in, 0, 1);
	mpa = isl_multi_pw_aff_set_tuple_name(mpa, isl_dim_in, "A");
	if (equal >= 0 && equal)
		equal = isl_multi_pw_aff_plain_is_equal(mpa, mpa1);
	isl_multi_pw_aff_free(mpa1);
	isl_multi_pw_aff_free(mpa);

	if (involves1 < 0 || involves2 < 0 || involves3 < 0 || equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown,
			"incorrect result of dimension insertion/removal",
			return isl_stat_error);
	if (!involves1 || involves2 || !involves3)
		isl_die(ctx, isl_error_unknown,
			"incorrect characterization of involved dimensions",
			return isl_stat_error);

	return 0;
}

/* Check that isl_multi_union_pw_aff_multi_val_on_domain
 * sets the explicit domain of a zero-dimensional result,
 * such that it can be converted to an isl_union_map.
 */
static isl_stat test_multi_pw_aff_3(isl_ctx *ctx)
{
	isl_space *space;
	isl_union_set *dom;
	isl_multi_val *mv;
	isl_multi_union_pw_aff *mupa;
	isl_union_map *umap;

	dom = isl_union_set_read_from_str(ctx, "{ A[]; B[] }");
	space = isl_union_set_get_space(dom);
	mv = isl_multi_val_zero(isl_space_set_from_params(space));
	mupa = isl_multi_union_pw_aff_multi_val_on_domain(dom, mv);
	umap = isl_union_map_from_multi_union_pw_aff(mupa);
	isl_union_map_free(umap);
	if (!umap)
		return isl_stat_error;

	return isl_stat_ok;
}

/* String descriptions of boxes that
 * are used for reconstructing box maps from their lower and upper bounds.
 */
static const char *multi_pw_aff_box_tests[] = {
	"{ A[x, y] -> [] : x + y >= 0 }",
	"[N] -> { A[x, y] -> [x] : x + y <= N }",
	"[N] -> { A[x, y] -> [x : y] : x + y <= N }",
};

/* Check that map representations of boxes can be reconstructed
 * from their lower and upper bounds.
 */
static isl_stat test_multi_pw_aff_box(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(multi_pw_aff_box_tests); ++i) {
		const char *str;
		isl_bool equal;
		isl_map *map, *box;
		isl_multi_pw_aff *min, *max;

		str = multi_pw_aff_box_tests[i];
		map = isl_map_read_from_str(ctx, str);
		min = isl_map_min_multi_pw_aff(isl_map_copy(map));
		max = isl_map_max_multi_pw_aff(isl_map_copy(map));
		box = isl_map_universe(isl_map_get_space(map));
		box = isl_map_lower_bound_multi_pw_aff(box, min);
		box = isl_map_upper_bound_multi_pw_aff(box, max);
		equal = isl_map_is_equal(map, box);
		isl_map_free(map);
		isl_map_free(box);
		if (equal < 0)
			return isl_stat_error;
		if (!equal)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return isl_stat_error);
	}

	return isl_stat_ok;
}

/* Perform some tests on multi piecewise affine expressions.
 */
static int test_multi_pw_aff(isl_ctx *ctx)
{
	if (test_multi_pw_aff_1(ctx) < 0)
		return -1;
	if (test_multi_pw_aff_2(ctx) < 0)
		return -1;
	if (test_multi_pw_aff_3(ctx) < 0)
		return -1;
	if (test_multi_pw_aff_box(ctx) < 0)
		return -1;
	return 0;
}

/* This is a regression test for a bug where isl_basic_map_simplify
 * would end up in an infinite loop.  In particular, we construct
 * an empty basic set that is not obviously empty.
 * isl_basic_set_is_empty marks the basic set as empty.
 * After projecting out i3, the variable can be dropped completely,
 * but isl_basic_map_simplify refrains from doing so if the basic set
 * is empty and would end up in an infinite loop if it didn't test
 * explicitly for empty basic maps in the outer loop.
 */
static int test_simplify_1(isl_ctx *ctx)
{
	const char *str;
	isl_basic_set *bset;
	int empty;

	str = "{ [i0, i1, i2, i3] : i0 >= -2 and 6i2 <= 4 + i0 + 5i1 and "
		"i2 <= 22 and 75i2 <= 111 + 13i0 + 60i1 and "
		"25i2 >= 38 + 6i0 + 20i1 and i0 <= -1 and i2 >= 20 and "
		"i3 >= i2 }";
	bset = isl_basic_set_read_from_str(ctx, str);
	empty = isl_basic_set_is_empty(bset);
	bset = isl_basic_set_project_out(bset, isl_dim_set, 3, 1);
	isl_basic_set_free(bset);
	if (!bset)
		return -1;
	if (!empty)
		isl_die(ctx, isl_error_unknown,
			"basic set should be empty", return -1);

	return 0;
}

/* Check that the equality in the set description below
 * is simplified away.
 */
static int test_simplify_2(isl_ctx *ctx)
{
	const char *str;
	isl_basic_set *bset;
	isl_bool universe;

	str = "{ [a] : exists e0, e1: 32e1 = 31 + 31a + 31e0 }";
	bset = isl_basic_set_read_from_str(ctx, str);
	universe = isl_basic_set_plain_is_universe(bset);
	isl_basic_set_free(bset);

	if (universe < 0)
		return -1;
	if (!universe)
		isl_die(ctx, isl_error_unknown,
			"equality not simplified away", return -1);
	return 0;
}

/* Some simplification tests.
 */
static int test_simplify(isl_ctx *ctx)
{
	if (test_simplify_1(ctx) < 0)
		return -1;
	if (test_simplify_2(ctx) < 0)
		return -1;
	return 0;
}

/* This is a regression test for a bug where isl_tab_basic_map_partial_lexopt
 * with gbr context would fail to disable the use of the shifted tableau
 * when transferring equalities for the input to the context, resulting
 * in invalid sample values.
 */
static int test_partial_lexmin(isl_ctx *ctx)
{
	const char *str;
	isl_basic_set *bset;
	isl_basic_map *bmap;
	isl_map *map;

	str = "{ [1, b, c, 1 - c] -> [e] : 2e <= -c and 2e >= -3 + c }";
	bmap = isl_basic_map_read_from_str(ctx, str);
	str = "{ [a, b, c, d] : c <= 1 and 2d >= 6 - 4b - c }";
	bset = isl_basic_set_read_from_str(ctx, str);
	map = isl_basic_map_partial_lexmin(bmap, bset, NULL);
	isl_map_free(map);

	if (!map)
		return -1;

	return 0;
}

/* Check that the variable compression performed on the existentially
 * quantified variables inside isl_basic_set_compute_divs is not confused
 * by the implicit equalities among the parameters.
 */
static int test_compute_divs(isl_ctx *ctx)
{
	const char *str;
	isl_basic_set *bset;
	isl_set *set;

	str = "[a, b, c, d, e] -> { [] : exists (e0: 2d = b and a <= 124 and "
		"b <= 2046 and b >= 0 and b <= 60 + 64a and 2e >= b + 2c and "
		"2e >= b and 2e <= 1 + b and 2e <= 1 + b + 2c and "
		"32768e0 >= -124 + a and 2097152e0 <= 60 + 64a - b) }";
	bset = isl_basic_set_read_from_str(ctx, str);
	set = isl_basic_set_compute_divs(bset);
	isl_set_free(set);
	if (!set)
		return -1;

	return 0;
}

/* Check that isl_schedule_get_map is not confused by a schedule tree
 * with divergent filter node parameters, as can result from a call
 * to isl_schedule_intersect_domain.
 */
static int test_schedule_tree(isl_ctx *ctx)
{
	const char *str;
	isl_union_set *uset;
	isl_schedule *sched1, *sched2;
	isl_union_map *umap;

	uset = isl_union_set_read_from_str(ctx, "{ A[i] }");
	sched1 = isl_schedule_from_domain(uset);
	uset = isl_union_set_read_from_str(ctx, "{ B[] }");
	sched2 = isl_schedule_from_domain(uset);

	sched1 = isl_schedule_sequence(sched1, sched2);
	str = "[n] -> { A[i] : 0 <= i < n; B[] }";
	uset = isl_union_set_read_from_str(ctx, str);
	sched1 = isl_schedule_intersect_domain(sched1, uset);
	umap = isl_schedule_get_map(sched1);
	isl_schedule_free(sched1);
	isl_union_map_free(umap);
	if (!umap)
		return -1;

	return 0;
}

/* Check that a zero-dimensional prefix schedule keeps track
 * of the domain and outer filters.
 */
static int test_schedule_tree_prefix(isl_ctx *ctx)
{
	const char *str;
	isl_bool equal;
	isl_union_set *uset;
	isl_union_set_list *filters;
	isl_multi_union_pw_aff *mupa, *mupa2;
	isl_schedule_node *node;

	str = "{ S1[i,j] : 0 <= i,j < 10; S2[i,j] : 0 <= i,j < 10 }";
	uset = isl_union_set_read_from_str(ctx, str);
	node = isl_schedule_node_from_domain(uset);
	node = isl_schedule_node_child(node, 0);

	str = "{ S1[i,j] : i > j }";
	uset = isl_union_set_read_from_str(ctx, str);
	filters = isl_union_set_list_from_union_set(uset);
	str = "{ S1[i,j] : i <= j; S2[i,j] }";
	uset = isl_union_set_read_from_str(ctx, str);
	filters = isl_union_set_list_add(filters, uset);
	node = isl_schedule_node_insert_sequence(node, filters);

	node = isl_schedule_node_child(node, 0);
	node = isl_schedule_node_child(node, 0);
	mupa = isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(node);
	str = "([] : { S1[i,j] : i > j })";
	mupa2 = isl_multi_union_pw_aff_read_from_str(ctx, str);
	equal = isl_multi_union_pw_aff_plain_is_equal(mupa, mupa2);
	isl_multi_union_pw_aff_free(mupa2);
	isl_multi_union_pw_aff_free(mupa);
	isl_schedule_node_free(node);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown, "unexpected prefix schedule",
			return -1);

	return 0;
}

/* Check that the reaching domain elements and the prefix schedule
 * at a leaf node are the same before and after grouping.
 */
static int test_schedule_tree_group_1(isl_ctx *ctx)
{
	int equal;
	const char *str;
	isl_id *id;
	isl_union_set *uset;
	isl_multi_union_pw_aff *mupa;
	isl_union_pw_multi_aff *upma1, *upma2;
	isl_union_set *domain1, *domain2;
	isl_union_map *umap1, *umap2;
	isl_schedule_node *node;

	str = "{ S1[i,j] : 0 <= i,j < 10; S2[i,j] : 0 <= i,j < 10 }";
	uset = isl_union_set_read_from_str(ctx, str);
	node = isl_schedule_node_from_domain(uset);
	node = isl_schedule_node_child(node, 0);
	str = "[{ S1[i,j] -> [i]; S2[i,j] -> [9 - i] }]";
	mupa = isl_multi_union_pw_aff_read_from_str(ctx, str);
	node = isl_schedule_node_insert_partial_schedule(node, mupa);
	node = isl_schedule_node_child(node, 0);
	str = "[{ S1[i,j] -> [j]; S2[i,j] -> [j] }]";
	mupa = isl_multi_union_pw_aff_read_from_str(ctx, str);
	node = isl_schedule_node_insert_partial_schedule(node, mupa);
	node = isl_schedule_node_child(node, 0);
	umap1 = isl_schedule_node_get_prefix_schedule_union_map(node);
	upma1 = isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(node);
	domain1 = isl_schedule_node_get_domain(node);
	id = isl_id_alloc(ctx, "group", NULL);
	node = isl_schedule_node_parent(node);
	node = isl_schedule_node_group(node, id);
	node = isl_schedule_node_child(node, 0);
	umap2 = isl_schedule_node_get_prefix_schedule_union_map(node);
	upma2 = isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(node);
	domain2 = isl_schedule_node_get_domain(node);
	equal = isl_union_pw_multi_aff_plain_is_equal(upma1, upma2);
	if (equal >= 0 && equal)
		equal = isl_union_set_is_equal(domain1, domain2);
	if (equal >= 0 && equal)
		equal = isl_union_map_is_equal(umap1, umap2);
	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	isl_union_set_free(domain1);
	isl_union_set_free(domain2);
	isl_union_pw_multi_aff_free(upma1);
	isl_union_pw_multi_aff_free(upma2);
	isl_schedule_node_free(node);

	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown,
			"expressions not equal", return -1);

	return 0;
}

/* Check that we can have nested groupings and that the union map
 * schedule representation is the same before and after the grouping.
 * Note that after the grouping, the union map representation contains
 * the domain constraints from the ranges of the expansion nodes,
 * while they are missing from the union map representation of
 * the tree without expansion nodes.
 *
 * Also check that the global expansion is as expected.
 */
static int test_schedule_tree_group_2(isl_ctx *ctx)
{
	int equal, equal_expansion;
	const char *str;
	isl_id *id;
	isl_union_set *uset;
	isl_union_map *umap1, *umap2;
	isl_union_map *expansion1, *expansion2;
	isl_union_set_list *filters;
	isl_multi_union_pw_aff *mupa;
	isl_schedule *schedule;
	isl_schedule_node *node;

	str = "{ S1[i,j] : 0 <= i,j < 10; S2[i,j] : 0 <= i,j < 10; "
		"S3[i,j] : 0 <= i,j < 10 }";
	uset = isl_union_set_read_from_str(ctx, str);
	node = isl_schedule_node_from_domain(uset);
	node = isl_schedule_node_child(node, 0);
	str = "[{ S1[i,j] -> [i]; S2[i,j] -> [i]; S3[i,j] -> [i] }]";
	mupa = isl_multi_union_pw_aff_read_from_str(ctx, str);
	node = isl_schedule_node_insert_partial_schedule(node, mupa);
	node = isl_schedule_node_child(node, 0);
	str = "{ S1[i,j] }";
	uset = isl_union_set_read_from_str(ctx, str);
	filters = isl_union_set_list_from_union_set(uset);
	str = "{ S2[i,j]; S3[i,j] }";
	uset = isl_union_set_read_from_str(ctx, str);
	filters = isl_union_set_list_add(filters, uset);
	node = isl_schedule_node_insert_sequence(node, filters);
	node = isl_schedule_node_child(node, 1);
	node = isl_schedule_node_child(node, 0);
	str = "{ S2[i,j] }";
	uset = isl_union_set_read_from_str(ctx, str);
	filters = isl_union_set_list_from_union_set(uset);
	str = "{ S3[i,j] }";
	uset = isl_union_set_read_from_str(ctx, str);
	filters = isl_union_set_list_add(filters, uset);
	node = isl_schedule_node_insert_sequence(node, filters);

	schedule = isl_schedule_node_get_schedule(node);
	umap1 = isl_schedule_get_map(schedule);
	uset = isl_schedule_get_domain(schedule);
	umap1 = isl_union_map_intersect_domain(umap1, uset);
	isl_schedule_free(schedule);

	node = isl_schedule_node_parent(node);
	node = isl_schedule_node_parent(node);
	id = isl_id_alloc(ctx, "group1", NULL);
	node = isl_schedule_node_group(node, id);
	node = isl_schedule_node_child(node, 1);
	node = isl_schedule_node_child(node, 0);
	id = isl_id_alloc(ctx, "group2", NULL);
	node = isl_schedule_node_group(node, id);

	schedule = isl_schedule_node_get_schedule(node);
	umap2 = isl_schedule_get_map(schedule);
	isl_schedule_free(schedule);

	node = isl_schedule_node_root(node);
	node = isl_schedule_node_child(node, 0);
	expansion1 = isl_schedule_node_get_subtree_expansion(node);
	isl_schedule_node_free(node);

	str = "{ group1[i] -> S1[i,j] : 0 <= i,j < 10; "
		"group1[i] -> S2[i,j] : 0 <= i,j < 10; "
		"group1[i] -> S3[i,j] : 0 <= i,j < 10 }";

	expansion2 = isl_union_map_read_from_str(ctx, str);

	equal = isl_union_map_is_equal(umap1, umap2);
	equal_expansion = isl_union_map_is_equal(expansion1, expansion2);

	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	isl_union_map_free(expansion1);
	isl_union_map_free(expansion2);

	if (equal < 0 || equal_expansion < 0)
		return -1;
	if (!equal)
		isl_die(ctx, isl_error_unknown,
			"expressions not equal", return -1);
	if (!equal_expansion)
		isl_die(ctx, isl_error_unknown,
			"unexpected expansion", return -1);

	return 0;
}

/* Some tests for the isl_schedule_node_group function.
 */
static int test_schedule_tree_group(isl_ctx *ctx)
{
	if (test_schedule_tree_group_1(ctx) < 0)
		return -1;
	if (test_schedule_tree_group_2(ctx) < 0)
		return -1;
	return 0;
}

struct {
	const char *set;
	const char *dual;
} coef_tests[] = {
	{ "{ rat: [i] : 0 <= i <= 10 }",
	  "{ rat: coefficients[[cst] -> [a]] : cst >= 0 and 10a + cst >= 0 }" },
	{ "{ rat: [i] : FALSE }",
	  "{ rat: coefficients[[cst] -> [a]] }" },
	{ "{ rat: [i] : }",
	  "{ rat: coefficients[[cst] -> [0]] : cst >= 0 }" },
	{ "{ [0:,1,2:3] }",
	  "{ rat: coefficients[[c_cst] -> [a, b, c]] : "
		"a >= 0 and 2c >= -c_cst - b and 3c >= -c_cst - b }" },
	{ "[M, N] -> { [x = (1 - N):-1, -4x:(M - 4x)] }",
	  "{ rat: coefficients[[c_cst, c_M = 0:, c_N = 0:] -> [a, b = -c_M:]] :"
		"4b >= -c_N + a and 4b >= -c_cst - 2c_N + a }" },
	{ "{ rat : [x, y] : 1 <= 2x <= 9 and 2 <= 3y <= 16 }",
	  "{ rat: coefficients[[c_cst] -> [c_x, c_y]] : "
		"4c_y >= -6c_cst - 3c_x and 4c_y >= -6c_cst - 27c_x and "
		"32c_y >= -6c_cst - 3c_x and 32c_y >= -6c_cst - 27c_x }" },
	{ "{ [x, y, z] : 3y <= 2x - 2 and y >= -2 + 2x and 2y >= 2 - x }",
	  "{ rat: coefficients[[cst] -> [a, b, c]] }" },
};

struct {
	const char *set;
	const char *dual;
} sol_tests[] = {
	{ "{ rat: coefficients[[cst] -> [a]] : cst >= 0 and 10a + cst >= 0 }",
	  "{ rat: [i] : 0 <= i <= 10 }" },
	{ "{ rat: coefficients[[cst] -> [a]] : FALSE }",
	  "{ rat: [i] }" },
	{ "{ rat: coefficients[[cst] -> [a]] }",
	  "{ rat: [i] : FALSE }" },
};

/* Test the basic functionality of isl_basic_set_coefficients and
 * isl_basic_set_solutions.
 */
static int test_dual(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(coef_tests); ++i) {
		int equal;
		isl_basic_set *bset1, *bset2;

		bset1 = isl_basic_set_read_from_str(ctx, coef_tests[i].set);
		bset2 = isl_basic_set_read_from_str(ctx, coef_tests[i].dual);
		bset1 = isl_basic_set_coefficients(bset1);
		equal = isl_basic_set_is_equal(bset1, bset2);
		isl_basic_set_free(bset1);
		isl_basic_set_free(bset2);
		if (equal < 0)
			return -1;
		if (!equal)
			isl_die(ctx, isl_error_unknown,
				"incorrect dual", return -1);
	}

	for (i = 0; i < ARRAY_SIZE(sol_tests); ++i) {
		int equal;
		isl_basic_set *bset1, *bset2;

		bset1 = isl_basic_set_read_from_str(ctx, sol_tests[i].set);
		bset2 = isl_basic_set_read_from_str(ctx, sol_tests[i].dual);
		bset1 = isl_basic_set_solutions(bset1);
		equal = isl_basic_set_is_equal(bset1, bset2);
		isl_basic_set_free(bset1);
		isl_basic_set_free(bset2);
		if (equal < 0)
			return -1;
		if (!equal)
			isl_die(ctx, isl_error_unknown,
				"incorrect dual", return -1);
	}

	return 0;
}

struct {
	int scale_tile;
	int shift_point;
	const char *domain;
	const char *schedule;
	const char *sizes;
	const char *tile;
	const char *point;
} tile_tests[] = {
	{ 0, 0, "[n] -> { S[i,j] : 0 <= i,j < n }",
	  "[{ S[i,j] -> [i] }, { S[i,j] -> [j] }]",
	  "{ [32,32] }",
	  "[{ S[i,j] -> [floor(i/32)] }, { S[i,j] -> [floor(j/32)] }]",
	  "[{ S[i,j] -> [i] }, { S[i,j] -> [j] }]",
	},
	{ 1, 0, "[n] -> { S[i,j] : 0 <= i,j < n }",
	  "[{ S[i,j] -> [i] }, { S[i,j] -> [j] }]",
	  "{ [32,32] }",
	  "[{ S[i,j] -> [32*floor(i/32)] }, { S[i,j] -> [32*floor(j/32)] }]",
	  "[{ S[i,j] -> [i] }, { S[i,j] -> [j] }]",
	},
	{ 0, 1, "[n] -> { S[i,j] : 0 <= i,j < n }",
	  "[{ S[i,j] -> [i] }, { S[i,j] -> [j] }]",
	  "{ [32,32] }",
	  "[{ S[i,j] -> [floor(i/32)] }, { S[i,j] -> [floor(j/32)] }]",
	  "[{ S[i,j] -> [i%32] }, { S[i,j] -> [j%32] }]",
	},
	{ 1, 1, "[n] -> { S[i,j] : 0 <= i,j < n }",
	  "[{ S[i,j] -> [i] }, { S[i,j] -> [j] }]",
	  "{ [32,32] }",
	  "[{ S[i,j] -> [32*floor(i/32)] }, { S[i,j] -> [32*floor(j/32)] }]",
	  "[{ S[i,j] -> [i%32] }, { S[i,j] -> [j%32] }]",
	},
};

/* Basic tiling tests.  Create a schedule tree with a domain and a band node,
 * tile the band and then check if the tile and point bands have the
 * expected partial schedule.
 */
static int test_tile(isl_ctx *ctx)
{
	int i;
	int scale;
	int shift;

	scale = isl_options_get_tile_scale_tile_loops(ctx);
	shift = isl_options_get_tile_shift_point_loops(ctx);

	for (i = 0; i < ARRAY_SIZE(tile_tests); ++i) {
		int opt;
		int equal;
		const char *str;
		isl_union_set *domain;
		isl_multi_union_pw_aff *mupa, *mupa2;
		isl_schedule_node *node;
		isl_multi_val *sizes;

		opt = tile_tests[i].scale_tile;
		isl_options_set_tile_scale_tile_loops(ctx, opt);
		opt = tile_tests[i].shift_point;
		isl_options_set_tile_shift_point_loops(ctx, opt);

		str = tile_tests[i].domain;
		domain = isl_union_set_read_from_str(ctx, str);
		node = isl_schedule_node_from_domain(domain);
		node = isl_schedule_node_child(node, 0);
		str = tile_tests[i].schedule;
		mupa = isl_multi_union_pw_aff_read_from_str(ctx, str);
		node = isl_schedule_node_insert_partial_schedule(node, mupa);
		str = tile_tests[i].sizes;
		sizes = isl_multi_val_read_from_str(ctx, str);
		node = isl_schedule_node_band_tile(node, sizes);

		str = tile_tests[i].tile;
		mupa = isl_multi_union_pw_aff_read_from_str(ctx, str);
		mupa2 = isl_schedule_node_band_get_partial_schedule(node);
		equal = isl_multi_union_pw_aff_plain_is_equal(mupa, mupa2);
		isl_multi_union_pw_aff_free(mupa);
		isl_multi_union_pw_aff_free(mupa2);

		node = isl_schedule_node_child(node, 0);

		str = tile_tests[i].point;
		mupa = isl_multi_union_pw_aff_read_from_str(ctx, str);
		mupa2 = isl_schedule_node_band_get_partial_schedule(node);
		if (equal >= 0 && equal)
			equal = isl_multi_union_pw_aff_plain_is_equal(mupa,
									mupa2);
		isl_multi_union_pw_aff_free(mupa);
		isl_multi_union_pw_aff_free(mupa2);

		isl_schedule_node_free(node);

		if (equal < 0)
			return -1;
		if (!equal)
			isl_die(ctx, isl_error_unknown,
				"unexpected result", return -1);
	}

	isl_options_set_tile_scale_tile_loops(ctx, scale);
	isl_options_set_tile_shift_point_loops(ctx, shift);

	return 0;
}

/* Check that the domain hash of a space is equal to the hash
 * of the domain of the space, both ignoring parameters.
 */
static int test_domain_hash(isl_ctx *ctx)
{
	isl_map *map;
	isl_space *space;
	uint32_t hash1, hash2;

	map = isl_map_read_from_str(ctx, "[n] -> { A[B[x] -> C[]] -> D[] }");
	space = isl_map_get_space(map);
	isl_map_free(map);
	hash1 = isl_space_get_tuple_domain_hash(space);
	space = isl_space_domain(space);
	hash2 = isl_space_get_tuple_hash(space);
	isl_space_free(space);

	if (!space)
		return -1;
	if (hash1 != hash2)
		isl_die(ctx, isl_error_unknown,
			"domain hash not equal to hash of domain", return -1);

	return 0;
}

/* Check that a universe basic set that is not obviously equal to the universe
 * is still recognized as being equal to the universe.
 */
static int test_universe(isl_ctx *ctx)
{
	const char *s;
	isl_basic_set *bset;
	isl_bool is_univ;

	s = "{ [] : exists x, y : 3y <= 2x and y >= -3 + 2x and 2y >= 2 - x }";
	bset = isl_basic_set_read_from_str(ctx, s);
	is_univ = isl_basic_set_is_universe(bset);
	isl_basic_set_free(bset);

	if (is_univ < 0)
		return -1;
	if (!is_univ)
		isl_die(ctx, isl_error_unknown,
			"not recognized as universe set", return -1);

	return 0;
}

/* Sets for which chambers are computed and checked.
 */
const char *chambers_tests[] = {
	"[A, B, C] -> { [x, y, z] : x >= 0 and y >= 0 and y <= A - x and "
				"z >= 0 and z <= C - y and z <= B - x - y }",
};

/* Add the domain of "cell" to "cells".
 */
static isl_stat add_cell(__isl_take isl_cell *cell, void *user)
{
	isl_basic_set_list **cells = user;
	isl_basic_set *dom;

	dom = isl_cell_get_domain(cell);
	isl_cell_free(cell);
	*cells = isl_basic_set_list_add(*cells, dom);

	return *cells ? isl_stat_ok : isl_stat_error;
}

/* Check that the elements of "list" are pairwise disjoint.
 */
static isl_stat check_pairwise_disjoint(__isl_keep isl_basic_set_list *list)
{
	int i, j;
	isl_size n;

	n = isl_basic_set_list_n_basic_set(list);
	if (n < 0)
		return isl_stat_error;

	for (i = 0; i < n; ++i) {
		isl_basic_set *bset_i;

		bset_i = isl_basic_set_list_get_basic_set(list, i);
		for (j = i + 1; j < n; ++j) {
			isl_basic_set *bset_j;
			isl_bool disjoint;

			bset_j = isl_basic_set_list_get_basic_set(list, j);
			disjoint = isl_basic_set_is_disjoint(bset_i, bset_j);
			isl_basic_set_free(bset_j);
			if (!disjoint)
				isl_die(isl_basic_set_list_get_ctx(list),
					isl_error_unknown, "not disjoint",
					break);
			if (disjoint < 0 || !disjoint)
				break;
		}
		isl_basic_set_free(bset_i);
		if (j < n)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Check that the chambers computed by isl_vertices_foreach_disjoint_cell
 * are pairwise disjoint.
 */
static int test_chambers(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(chambers_tests); ++i) {
		isl_basic_set *bset;
		isl_vertices *vertices;
		isl_basic_set_list *cells;
		isl_stat ok;

		bset = isl_basic_set_read_from_str(ctx, chambers_tests[i]);
		vertices = isl_basic_set_compute_vertices(bset);
		cells = isl_basic_set_list_alloc(ctx, 0);
		if (isl_vertices_foreach_disjoint_cell(vertices, &add_cell,
							&cells) < 0)
			cells = isl_basic_set_list_free(cells);
		ok = check_pairwise_disjoint(cells);
		isl_basic_set_list_free(cells);
		isl_vertices_free(vertices);
		isl_basic_set_free(bset);

		if (ok < 0)
			return -1;
	}

	return 0;
}

struct {
	const char *name;
	int (*fn)(isl_ctx *ctx);
} tests [] = {
	{ "universe", &test_universe },
	{ "domain hash", &test_domain_hash },
	{ "dual", &test_dual },
	{ "dependence analysis", &test_flow },
	{ "val", &test_val },
	{ "compute divs", &test_compute_divs },
	{ "partial lexmin", &test_partial_lexmin },
	{ "simplify", &test_simplify },
	{ "curry", &test_curry },
	{ "piecewise multi affine expressions", &test_pw_multi_aff },
	{ "multi piecewise affine expressions", &test_multi_pw_aff },
	{ "conversion", &test_conversion },
	{ "list", &test_list },
	{ "align parameters", &test_align_parameters },
	{ "drop unused parameters", &test_drop_unused_parameters },
	{ "pullback", &test_pullback },
	{ "AST", &test_ast },
	{ "AST build", &test_ast_build },
	{ "AST generation", &test_ast_gen },
	{ "eliminate", &test_eliminate },
	{ "deltas_map", &test_deltas_map },
	{ "residue class", &test_residue_class },
	{ "div", &test_div },
	{ "slice", &test_slice },
	{ "fixed power", &test_fixed_power },
	{ "sample", &test_sample },
	{ "empty projection", &test_empty_projection },
	{ "output", &test_output },
	{ "vertices", &test_vertices },
	{ "chambers", &test_chambers },
	{ "fixed", &test_fixed },
	{ "equal", &test_equal },
	{ "disjoint", &test_disjoint },
	{ "product", &test_product },
	{ "dim_max", &test_dim_max },
	{ "affine", &test_aff },
	{ "injective", &test_injective },
	{ "schedule (whole component)", &test_schedule_whole },
	{ "schedule (incremental)", &test_schedule_incremental },
	{ "schedule tree", &test_schedule_tree },
	{ "schedule tree prefix", &test_schedule_tree_prefix },
	{ "schedule tree grouping", &test_schedule_tree_group },
	{ "tile", &test_tile },
	{ "union map", &test_union_map },
	{ "union_pw", &test_union_pw },
	{ "locus", &test_locus },
	{ "eval", &test_eval },
	{ "parse", &test_parse },
	{ "single-valued", &test_sv },
	{ "recession cone", &test_recession_cone },
	{ "affine hull", &test_affine_hull },
	{ "simple_hull", &test_simple_hull },
	{ "box hull", &test_box_hull },
	{ "coalesce", &test_coalesce },
	{ "factorize", &test_factorize },
	{ "subset", &test_subset },
	{ "subtract", &test_subtract },
	{ "intersect", &test_intersect },
	{ "lexmin", &test_lexmin },
	{ "min", &test_min },
	{ "set lower bounds", &test_min_mpa },
	{ "gist", &test_gist },
	{ "piecewise quasi-polynomials", &test_pwqp },
	{ "lift", &test_lift },
	{ "bind parameters", &test_bind },
	{ "unbind parameters", &test_unbind },
	{ "bound", &test_bound },
	{ "get lists", &test_get_list },
	{ "union", &test_union },
	{ "split periods", &test_split_periods },
	{ "lexicographic order", &test_lex },
	{ "bijectivity", &test_bijective },
	{ "dataflow analysis", &test_dep },
	{ "reading", &test_read },
	{ "bounded", &test_bounded },
	{ "construction", &test_construction },
	{ "dimension manipulation", &test_dim },
	{ "map application", &test_application },
	{ "convex hull", &test_convex_hull },
	{ "transitive closure", &test_closure },
	{ "isl_bool", &test_isl_bool},
};

int main(int argc, char **argv)
{
	int i;
	struct isl_ctx *ctx;
	struct isl_options *options;

	options = isl_options_new_with_defaults();
	assert(options);
	argc = isl_options_parse(options, argc, argv, ISL_ARG_ALL);

	ctx = isl_ctx_alloc_with_options(&isl_options_args, options);
	for (i = 0; i < ARRAY_SIZE(tests); ++i) {
		printf("%s\n", tests[i].name);
		if (tests[i].fn(ctx) < 0)
			goto error;
	}
	isl_ctx_free(ctx);
	return 0;
error:
	isl_ctx_free(ctx);
	return -1;
}
