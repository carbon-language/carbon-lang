/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010      INRIA Saclay
 * Copyright 2012-2013 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France 
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include "isl_map_private.h"
#include <isl_seq.h>
#include <isl/options.h>
#include "isl_tab.h"
#include <isl_mat_private.h>
#include <isl_local_space_private.h>
#include <isl_vec_private.h>

#define STATUS_ERROR		-1
#define STATUS_REDUNDANT	 1
#define STATUS_VALID	 	 2
#define STATUS_SEPARATE	 	 3
#define STATUS_CUT	 	 4
#define STATUS_ADJ_EQ	 	 5
#define STATUS_ADJ_INEQ	 	 6

static int status_in(isl_int *ineq, struct isl_tab *tab)
{
	enum isl_ineq_type type = isl_tab_ineq_type(tab, ineq);
	switch (type) {
	default:
	case isl_ineq_error:		return STATUS_ERROR;
	case isl_ineq_redundant:	return STATUS_VALID;
	case isl_ineq_separate:		return STATUS_SEPARATE;
	case isl_ineq_cut:		return STATUS_CUT;
	case isl_ineq_adj_eq:		return STATUS_ADJ_EQ;
	case isl_ineq_adj_ineq:		return STATUS_ADJ_INEQ;
	}
}

/* Compute the position of the equalities of basic map "bmap_i"
 * with respect to the basic map represented by "tab_j".
 * The resulting array has twice as many entries as the number
 * of equalities corresponding to the two inequalties to which
 * each equality corresponds.
 */
static int *eq_status_in(__isl_keep isl_basic_map *bmap_i,
	struct isl_tab *tab_j)
{
	int k, l;
	int *eq = isl_calloc_array(bmap_i->ctx, int, 2 * bmap_i->n_eq);
	unsigned dim;

	if (!eq)
		return NULL;

	dim = isl_basic_map_total_dim(bmap_i);
	for (k = 0; k < bmap_i->n_eq; ++k) {
		for (l = 0; l < 2; ++l) {
			isl_seq_neg(bmap_i->eq[k], bmap_i->eq[k], 1+dim);
			eq[2 * k + l] = status_in(bmap_i->eq[k], tab_j);
			if (eq[2 * k + l] == STATUS_ERROR)
				goto error;
		}
		if (eq[2 * k] == STATUS_SEPARATE ||
		    eq[2 * k + 1] == STATUS_SEPARATE)
			break;
	}

	return eq;
error:
	free(eq);
	return NULL;
}

/* Compute the position of the inequalities of basic map "bmap_i"
 * (also represented by "tab_i", if not NULL) with respect to the basic map
 * represented by "tab_j".
 */
static int *ineq_status_in(__isl_keep isl_basic_map *bmap_i,
	struct isl_tab *tab_i, struct isl_tab *tab_j)
{
	int k;
	unsigned n_eq = bmap_i->n_eq;
	int *ineq = isl_calloc_array(bmap_i->ctx, int, bmap_i->n_ineq);

	if (!ineq)
		return NULL;

	for (k = 0; k < bmap_i->n_ineq; ++k) {
		if (tab_i && isl_tab_is_redundant(tab_i, n_eq + k)) {
			ineq[k] = STATUS_REDUNDANT;
			continue;
		}
		ineq[k] = status_in(bmap_i->ineq[k], tab_j);
		if (ineq[k] == STATUS_ERROR)
			goto error;
		if (ineq[k] == STATUS_SEPARATE)
			break;
	}

	return ineq;
error:
	free(ineq);
	return NULL;
}

static int any(int *con, unsigned len, int status)
{
	int i;

	for (i = 0; i < len ; ++i)
		if (con[i] == status)
			return 1;
	return 0;
}

static int count(int *con, unsigned len, int status)
{
	int i;
	int c = 0;

	for (i = 0; i < len ; ++i)
		if (con[i] == status)
			c++;
	return c;
}

static int all(int *con, unsigned len, int status)
{
	int i;

	for (i = 0; i < len ; ++i) {
		if (con[i] == STATUS_REDUNDANT)
			continue;
		if (con[i] != status)
			return 0;
	}
	return 1;
}

static void drop(struct isl_map *map, int i, struct isl_tab **tabs)
{
	isl_basic_map_free(map->p[i]);
	isl_tab_free(tabs[i]);

	if (i != map->n - 1) {
		map->p[i] = map->p[map->n - 1];
		tabs[i] = tabs[map->n - 1];
	}
	tabs[map->n - 1] = NULL;
	map->n--;
}

/* Replace the pair of basic maps i and j by the basic map bounded
 * by the valid constraints in both basic maps and the constraint
 * in extra (if not NULL).
 */
static int fuse(struct isl_map *map, int i, int j,
	struct isl_tab **tabs, int *eq_i, int *ineq_i, int *eq_j, int *ineq_j,
	__isl_keep isl_mat *extra)
{
	int k, l;
	struct isl_basic_map *fused = NULL;
	struct isl_tab *fused_tab = NULL;
	unsigned total = isl_basic_map_total_dim(map->p[i]);
	unsigned extra_rows = extra ? extra->n_row : 0;

	fused = isl_basic_map_alloc_space(isl_space_copy(map->p[i]->dim),
			map->p[i]->n_div,
			map->p[i]->n_eq + map->p[j]->n_eq,
			map->p[i]->n_ineq + map->p[j]->n_ineq + extra_rows);
	if (!fused)
		goto error;

	for (k = 0; k < map->p[i]->n_eq; ++k) {
		if (eq_i && (eq_i[2 * k] != STATUS_VALID ||
			     eq_i[2 * k + 1] != STATUS_VALID))
			continue;
		l = isl_basic_map_alloc_equality(fused);
		if (l < 0)
			goto error;
		isl_seq_cpy(fused->eq[l], map->p[i]->eq[k], 1 + total);
	}

	for (k = 0; k < map->p[j]->n_eq; ++k) {
		if (eq_j && (eq_j[2 * k] != STATUS_VALID ||
			     eq_j[2 * k + 1] != STATUS_VALID))
			continue;
		l = isl_basic_map_alloc_equality(fused);
		if (l < 0)
			goto error;
		isl_seq_cpy(fused->eq[l], map->p[j]->eq[k], 1 + total);
	}

	for (k = 0; k < map->p[i]->n_ineq; ++k) {
		if (ineq_i[k] != STATUS_VALID)
			continue;
		l = isl_basic_map_alloc_inequality(fused);
		if (l < 0)
			goto error;
		isl_seq_cpy(fused->ineq[l], map->p[i]->ineq[k], 1 + total);
	}

	for (k = 0; k < map->p[j]->n_ineq; ++k) {
		if (ineq_j[k] != STATUS_VALID)
			continue;
		l = isl_basic_map_alloc_inequality(fused);
		if (l < 0)
			goto error;
		isl_seq_cpy(fused->ineq[l], map->p[j]->ineq[k], 1 + total);
	}

	for (k = 0; k < map->p[i]->n_div; ++k) {
		int l = isl_basic_map_alloc_div(fused);
		if (l < 0)
			goto error;
		isl_seq_cpy(fused->div[l], map->p[i]->div[k], 1 + 1 + total);
	}

	for (k = 0; k < extra_rows; ++k) {
		l = isl_basic_map_alloc_inequality(fused);
		if (l < 0)
			goto error;
		isl_seq_cpy(fused->ineq[l], extra->row[k], 1 + total);
	}

	fused = isl_basic_map_gauss(fused, NULL);
	ISL_F_SET(fused, ISL_BASIC_MAP_FINAL);
	if (ISL_F_ISSET(map->p[i], ISL_BASIC_MAP_RATIONAL) &&
	    ISL_F_ISSET(map->p[j], ISL_BASIC_MAP_RATIONAL))
		ISL_F_SET(fused, ISL_BASIC_MAP_RATIONAL);

	fused_tab = isl_tab_from_basic_map(fused, 0);
	if (isl_tab_detect_redundant(fused_tab) < 0)
		goto error;

	isl_basic_map_free(map->p[i]);
	map->p[i] = fused;
	isl_tab_free(tabs[i]);
	tabs[i] = fused_tab;
	drop(map, j, tabs);

	return 1;
error:
	isl_tab_free(fused_tab);
	isl_basic_map_free(fused);
	return -1;
}

/* Given a pair of basic maps i and j such that all constraints are either
 * "valid" or "cut", check if the facets corresponding to the "cut"
 * constraints of i lie entirely within basic map j.
 * If so, replace the pair by the basic map consisting of the valid
 * constraints in both basic maps.
 *
 * To see that we are not introducing any extra points, call the
 * two basic maps A and B and the resulting map U and let x
 * be an element of U \setminus ( A \cup B ).
 * Then there is a pair of cut constraints c_1 and c_2 in A and B such that x
 * violates them.  Let X be the intersection of U with the opposites
 * of these constraints.  Then x \in X.
 * The facet corresponding to c_1 contains the corresponding facet of A.
 * This facet is entirely contained in B, so c_2 is valid on the facet.
 * However, since it is also (part of) a facet of X, -c_2 is also valid
 * on the facet.  This means c_2 is saturated on the facet, so c_1 and
 * c_2 must be opposites of each other, but then x could not violate
 * both of them.
 */
static int check_facets(struct isl_map *map, int i, int j,
	struct isl_tab **tabs, int *ineq_i, int *ineq_j)
{
	int k, l;
	struct isl_tab_undo *snap;
	unsigned n_eq = map->p[i]->n_eq;

	snap = isl_tab_snap(tabs[i]);

	for (k = 0; k < map->p[i]->n_ineq; ++k) {
		if (ineq_i[k] != STATUS_CUT)
			continue;
		if (isl_tab_select_facet(tabs[i], n_eq + k) < 0)
			return -1;
		for (l = 0; l < map->p[j]->n_ineq; ++l) {
			int stat;
			if (ineq_j[l] != STATUS_CUT)
				continue;
			stat = status_in(map->p[j]->ineq[l], tabs[i]);
			if (stat != STATUS_VALID)
				break;
		}
		if (isl_tab_rollback(tabs[i], snap) < 0)
			return -1;
		if (l < map->p[j]->n_ineq)
			break;
	}

	if (k < map->p[i]->n_ineq)
		/* BAD CUT PAIR */
		return 0;
	return fuse(map, i, j, tabs, NULL, ineq_i, NULL, ineq_j, NULL);
}

/* Check if basic map "i" contains the basic map represented
 * by the tableau "tab".
 */
static int contains(struct isl_map *map, int i, int *ineq_i,
	struct isl_tab *tab)
{
	int k, l;
	unsigned dim;

	dim = isl_basic_map_total_dim(map->p[i]);
	for (k = 0; k < map->p[i]->n_eq; ++k) {
		for (l = 0; l < 2; ++l) {
			int stat;
			isl_seq_neg(map->p[i]->eq[k], map->p[i]->eq[k], 1+dim);
			stat = status_in(map->p[i]->eq[k], tab);
			if (stat != STATUS_VALID)
				return 0;
		}
	}

	for (k = 0; k < map->p[i]->n_ineq; ++k) {
		int stat;
		if (ineq_i[k] == STATUS_REDUNDANT)
			continue;
		stat = status_in(map->p[i]->ineq[k], tab);
		if (stat != STATUS_VALID)
			return 0;
	}
	return 1;
}

/* Basic map "i" has an inequality (say "k") that is adjacent
 * to some inequality of basic map "j".  All the other inequalities
 * are valid for "j".
 * Check if basic map "j" forms an extension of basic map "i".
 *
 * Note that this function is only called if some of the equalities or
 * inequalities of basic map "j" do cut basic map "i".  The function is
 * correct even if there are no such cut constraints, but in that case
 * the additional checks performed by this function are overkill.
 *
 * In particular, we replace constraint k, say f >= 0, by constraint
 * f <= -1, add the inequalities of "j" that are valid for "i"
 * and check if the result is a subset of basic map "j".
 * If so, then we know that this result is exactly equal to basic map "j"
 * since all its constraints are valid for basic map "j".
 * By combining the valid constraints of "i" (all equalities and all
 * inequalities except "k") and the valid constraints of "j" we therefore
 * obtain a basic map that is equal to their union.
 * In this case, there is no need to perform a rollback of the tableau
 * since it is going to be destroyed in fuse().
 *
 *
 *	|\__			|\__
 *	|   \__			|   \__
 *	|      \_	=>	|      \__
 *	|_______| _		|_________\
 *
 *
 *	|\			|\
 *	| \			| \
 *	|  \			|  \
 *	|  |			|   \
 *	|  ||\		=>      |    \
 *	|  || \			|     \
 *	|  ||  |		|      |
 *	|__||_/			|_____/
 */
static int is_adj_ineq_extension(__isl_keep isl_map *map, int i, int j,
	struct isl_tab **tabs, int *eq_i, int *ineq_i, int *eq_j, int *ineq_j)
{
	int k;
	struct isl_tab_undo *snap;
	unsigned n_eq = map->p[i]->n_eq;
	unsigned total = isl_basic_map_total_dim(map->p[i]);
	int r;

	if (isl_tab_extend_cons(tabs[i], 1 + map->p[j]->n_ineq) < 0)
		return -1;

	for (k = 0; k < map->p[i]->n_ineq; ++k)
		if (ineq_i[k] == STATUS_ADJ_INEQ)
			break;
	if (k >= map->p[i]->n_ineq)
		isl_die(isl_map_get_ctx(map), isl_error_internal,
			"ineq_i should have exactly one STATUS_ADJ_INEQ",
			return -1);

	snap = isl_tab_snap(tabs[i]);

	if (isl_tab_unrestrict(tabs[i], n_eq + k) < 0)
		return -1;

	isl_seq_neg(map->p[i]->ineq[k], map->p[i]->ineq[k], 1 + total);
	isl_int_sub_ui(map->p[i]->ineq[k][0], map->p[i]->ineq[k][0], 1);
	r = isl_tab_add_ineq(tabs[i], map->p[i]->ineq[k]);
	isl_seq_neg(map->p[i]->ineq[k], map->p[i]->ineq[k], 1 + total);
	isl_int_sub_ui(map->p[i]->ineq[k][0], map->p[i]->ineq[k][0], 1);
	if (r < 0)
		return -1;

	for (k = 0; k < map->p[j]->n_ineq; ++k) {
		if (ineq_j[k] != STATUS_VALID)
			continue;
		if (isl_tab_add_ineq(tabs[i], map->p[j]->ineq[k]) < 0)
			return -1;
	}

	if (contains(map, j, ineq_j, tabs[i]))
		return fuse(map, i, j, tabs, eq_i, ineq_i, eq_j, ineq_j, NULL);

	if (isl_tab_rollback(tabs[i], snap) < 0)
		return -1;

	return 0;
}


/* Both basic maps have at least one inequality with and adjacent
 * (but opposite) inequality in the other basic map.
 * Check that there are no cut constraints and that there is only
 * a single pair of adjacent inequalities.
 * If so, we can replace the pair by a single basic map described
 * by all but the pair of adjacent inequalities.
 * Any additional points introduced lie strictly between the two
 * adjacent hyperplanes and can therefore be integral.
 *
 *        ____			  _____
 *       /    ||\		 /     \
 *      /     || \		/       \
 *      \     ||  \	=>	\        \
 *       \    ||  /		 \       /
 *        \___||_/		  \_____/
 *
 * The test for a single pair of adjancent inequalities is important
 * for avoiding the combination of two basic maps like the following
 *
 *       /|
 *      / |
 *     /__|
 *         _____
 *         |   |
 *         |   |
 *         |___|
 *
 * If there are some cut constraints on one side, then we may
 * still be able to fuse the two basic maps, but we need to perform
 * some additional checks in is_adj_ineq_extension.
 */
static int check_adj_ineq(struct isl_map *map, int i, int j,
	struct isl_tab **tabs, int *eq_i, int *ineq_i, int *eq_j, int *ineq_j)
{
	int count_i, count_j;
	int cut_i, cut_j;

	count_i = count(ineq_i, map->p[i]->n_ineq, STATUS_ADJ_INEQ);
	count_j = count(ineq_j, map->p[j]->n_ineq, STATUS_ADJ_INEQ);

	if (count_i != 1 && count_j != 1)
		return 0;

	cut_i = any(eq_i, 2 * map->p[i]->n_eq, STATUS_CUT) ||
		any(ineq_i, map->p[i]->n_ineq, STATUS_CUT);
	cut_j = any(eq_j, 2 * map->p[j]->n_eq, STATUS_CUT) ||
		any(ineq_j, map->p[j]->n_ineq, STATUS_CUT);

	if (!cut_i && !cut_j && count_i == 1 && count_j == 1)
		return fuse(map, i, j, tabs, NULL, ineq_i, NULL, ineq_j, NULL);

	if (count_i == 1 && !cut_i)
		return is_adj_ineq_extension(map, i, j, tabs,
						eq_i, ineq_i, eq_j, ineq_j);

	if (count_j == 1 && !cut_j)
		return is_adj_ineq_extension(map, j, i, tabs,
						eq_j, ineq_j, eq_i, ineq_i);

	return 0;
}

/* Basic map "i" has an inequality "k" that is adjacent to some equality
 * of basic map "j".  All the other inequalities are valid for "j".
 * Check if basic map "j" forms an extension of basic map "i".
 *
 * In particular, we relax constraint "k", compute the corresponding
 * facet and check whether it is included in the other basic map.
 * If so, we know that relaxing the constraint extends the basic
 * map with exactly the other basic map (we already know that this
 * other basic map is included in the extension, because there
 * were no "cut" inequalities in "i") and we can replace the
 * two basic maps by this extension.
 *        ____			  _____
 *       /    || 		 /     |
 *      /     ||  		/      |
 *      \     ||   	=>	\      |
 *       \    ||		 \     |
 *        \___||		  \____|
 */
static int is_adj_eq_extension(struct isl_map *map, int i, int j, int k,
	struct isl_tab **tabs, int *eq_i, int *ineq_i, int *eq_j, int *ineq_j)
{
	int changed = 0;
	int super;
	struct isl_tab_undo *snap, *snap2;
	unsigned n_eq = map->p[i]->n_eq;

	if (isl_tab_is_equality(tabs[i], n_eq + k))
		return 0;

	snap = isl_tab_snap(tabs[i]);
	tabs[i] = isl_tab_relax(tabs[i], n_eq + k);
	snap2 = isl_tab_snap(tabs[i]);
	if (isl_tab_select_facet(tabs[i], n_eq + k) < 0)
		return -1;
	super = contains(map, j, ineq_j, tabs[i]);
	if (super) {
		if (isl_tab_rollback(tabs[i], snap2) < 0)
			return -1;
		map->p[i] = isl_basic_map_cow(map->p[i]);
		if (!map->p[i])
			return -1;
		isl_int_add_ui(map->p[i]->ineq[k][0], map->p[i]->ineq[k][0], 1);
		ISL_F_SET(map->p[i], ISL_BASIC_MAP_FINAL);
		drop(map, j, tabs);
		changed = 1;
	} else
		if (isl_tab_rollback(tabs[i], snap) < 0)
			return -1;

	return changed;
}

/* Data structure that keeps track of the wrapping constraints
 * and of information to bound the coefficients of those constraints.
 *
 * bound is set if we want to apply a bound on the coefficients
 * mat contains the wrapping constraints
 * max is the bound on the coefficients (if bound is set)
 */
struct isl_wraps {
	int bound;
	isl_mat *mat;
	isl_int max;
};

/* Update wraps->max to be greater than or equal to the coefficients
 * in the equalities and inequalities of bmap that can be removed if we end up
 * applying wrapping.
 */
static void wraps_update_max(struct isl_wraps *wraps,
	__isl_keep isl_basic_map *bmap, int *eq, int *ineq)
{
	int k;
	isl_int max_k;
	unsigned total = isl_basic_map_total_dim(bmap);

	isl_int_init(max_k);

	for (k = 0; k < bmap->n_eq; ++k) {
		if (eq[2 * k] == STATUS_VALID &&
		    eq[2 * k + 1] == STATUS_VALID)
			continue;
		isl_seq_abs_max(bmap->eq[k] + 1, total, &max_k);
		if (isl_int_abs_gt(max_k, wraps->max))
			isl_int_set(wraps->max, max_k);
	}

	for (k = 0; k < bmap->n_ineq; ++k) {
		if (ineq[k] == STATUS_VALID || ineq[k] == STATUS_REDUNDANT)
			continue;
		isl_seq_abs_max(bmap->ineq[k] + 1, total, &max_k);
		if (isl_int_abs_gt(max_k, wraps->max))
			isl_int_set(wraps->max, max_k);
	}

	isl_int_clear(max_k);
}

/* Initialize the isl_wraps data structure.
 * If we want to bound the coefficients of the wrapping constraints,
 * we set wraps->max to the largest coefficient
 * in the equalities and inequalities that can be removed if we end up
 * applying wrapping.
 */
static void wraps_init(struct isl_wraps *wraps, __isl_take isl_mat *mat,
	__isl_keep isl_map *map, int i, int j,
	int *eq_i, int *ineq_i, int *eq_j, int *ineq_j)
{
	isl_ctx *ctx;

	wraps->bound = 0;
	wraps->mat = mat;
	if (!mat)
		return;
	ctx = isl_mat_get_ctx(mat);
	wraps->bound = isl_options_get_coalesce_bounded_wrapping(ctx);
	if (!wraps->bound)
		return;
	isl_int_init(wraps->max);
	isl_int_set_si(wraps->max, 0);
	wraps_update_max(wraps, map->p[i], eq_i, ineq_i);
	wraps_update_max(wraps, map->p[j], eq_j, ineq_j);
}

/* Free the contents of the isl_wraps data structure.
 */
static void wraps_free(struct isl_wraps *wraps)
{
	isl_mat_free(wraps->mat);
	if (wraps->bound)
		isl_int_clear(wraps->max);
}

/* Is the wrapping constraint in row "row" allowed?
 *
 * If wraps->bound is set, we check that none of the coefficients
 * is greater than wraps->max.
 */
static int allow_wrap(struct isl_wraps *wraps, int row)
{
	int i;

	if (!wraps->bound)
		return 1;

	for (i = 1; i < wraps->mat->n_col; ++i)
		if (isl_int_abs_gt(wraps->mat->row[row][i], wraps->max))
			return 0;

	return 1;
}

/* For each non-redundant constraint in "bmap" (as determined by "tab"),
 * wrap the constraint around "bound" such that it includes the whole
 * set "set" and append the resulting constraint to "wraps".
 * "wraps" is assumed to have been pre-allocated to the appropriate size.
 * wraps->n_row is the number of actual wrapped constraints that have
 * been added.
 * If any of the wrapping problems results in a constraint that is
 * identical to "bound", then this means that "set" is unbounded in such
 * way that no wrapping is possible.  If this happens then wraps->n_row
 * is reset to zero.
 * Similarly, if we want to bound the coefficients of the wrapping
 * constraints and a newly added wrapping constraint does not
 * satisfy the bound, then wraps->n_row is also reset to zero.
 */
static int add_wraps(struct isl_wraps *wraps, __isl_keep isl_basic_map *bmap,
	struct isl_tab *tab, isl_int *bound, __isl_keep isl_set *set)
{
	int l;
	int w;
	unsigned total = isl_basic_map_total_dim(bmap);

	w = wraps->mat->n_row;

	for (l = 0; l < bmap->n_ineq; ++l) {
		if (isl_seq_is_neg(bound, bmap->ineq[l], 1 + total))
			continue;
		if (isl_seq_eq(bound, bmap->ineq[l], 1 + total))
			continue;
		if (isl_tab_is_redundant(tab, bmap->n_eq + l))
			continue;

		isl_seq_cpy(wraps->mat->row[w], bound, 1 + total);
		if (!isl_set_wrap_facet(set, wraps->mat->row[w], bmap->ineq[l]))
			return -1;
		if (isl_seq_eq(wraps->mat->row[w], bound, 1 + total))
			goto unbounded;
		if (!allow_wrap(wraps, w))
			goto unbounded;
		++w;
	}
	for (l = 0; l < bmap->n_eq; ++l) {
		if (isl_seq_is_neg(bound, bmap->eq[l], 1 + total))
			continue;
		if (isl_seq_eq(bound, bmap->eq[l], 1 + total))
			continue;

		isl_seq_cpy(wraps->mat->row[w], bound, 1 + total);
		isl_seq_neg(wraps->mat->row[w + 1], bmap->eq[l], 1 + total);
		if (!isl_set_wrap_facet(set, wraps->mat->row[w],
					wraps->mat->row[w + 1]))
			return -1;
		if (isl_seq_eq(wraps->mat->row[w], bound, 1 + total))
			goto unbounded;
		if (!allow_wrap(wraps, w))
			goto unbounded;
		++w;

		isl_seq_cpy(wraps->mat->row[w], bound, 1 + total);
		if (!isl_set_wrap_facet(set, wraps->mat->row[w], bmap->eq[l]))
			return -1;
		if (isl_seq_eq(wraps->mat->row[w], bound, 1 + total))
			goto unbounded;
		if (!allow_wrap(wraps, w))
			goto unbounded;
		++w;
	}

	wraps->mat->n_row = w;
	return 0;
unbounded:
	wraps->mat->n_row = 0;
	return 0;
}

/* Check if the constraints in "wraps" from "first" until the last
 * are all valid for the basic set represented by "tab".
 * If not, wraps->n_row is set to zero.
 */
static int check_wraps(__isl_keep isl_mat *wraps, int first,
	struct isl_tab *tab)
{
	int i;

	for (i = first; i < wraps->n_row; ++i) {
		enum isl_ineq_type type;
		type = isl_tab_ineq_type(tab, wraps->row[i]);
		if (type == isl_ineq_error)
			return -1;
		if (type == isl_ineq_redundant)
			continue;
		wraps->n_row = 0;
		return 0;
	}

	return 0;
}

/* Return a set that corresponds to the non-redudant constraints
 * (as recorded in tab) of bmap.
 *
 * It's important to remove the redundant constraints as some
 * of the other constraints may have been modified after the
 * constraints were marked redundant.
 * In particular, a constraint may have been relaxed.
 * Redundant constraints are ignored when a constraint is relaxed
 * and should therefore continue to be ignored ever after.
 * Otherwise, the relaxation might be thwarted by some of
 * these constraints.
 */
static __isl_give isl_set *set_from_updated_bmap(__isl_keep isl_basic_map *bmap,
	struct isl_tab *tab)
{
	bmap = isl_basic_map_copy(bmap);
	bmap = isl_basic_map_cow(bmap);
	bmap = isl_basic_map_update_from_tab(bmap, tab);
	return isl_set_from_basic_set(isl_basic_map_underlying_set(bmap));
}

/* Given a basic set i with a constraint k that is adjacent to either the
 * whole of basic set j or a facet of basic set j, check if we can wrap
 * both the facet corresponding to k and the facet of j (or the whole of j)
 * around their ridges to include the other set.
 * If so, replace the pair of basic sets by their union.
 *
 * All constraints of i (except k) are assumed to be valid for j.
 *
 * However, the constraints of j may not be valid for i and so
 * we have to check that the wrapping constraints for j are valid for i.
 *
 * In the case where j has a facet adjacent to i, tab[j] is assumed
 * to have been restricted to this facet, so that the non-redundant
 * constraints in tab[j] are the ridges of the facet.
 * Note that for the purpose of wrapping, it does not matter whether
 * we wrap the ridges of i around the whole of j or just around
 * the facet since all the other constraints are assumed to be valid for j.
 * In practice, we wrap to include the whole of j.
 *        ____			  _____
 *       /    | 		 /     \
 *      /     ||  		/      |
 *      \     ||   	=>	\      |
 *       \    ||		 \     |
 *        \___||		  \____|
 *
 */
static int can_wrap_in_facet(struct isl_map *map, int i, int j, int k,
	struct isl_tab **tabs, int *eq_i, int *ineq_i, int *eq_j, int *ineq_j)
{
	int changed = 0;
	struct isl_wraps wraps;
	isl_mat *mat;
	struct isl_set *set_i = NULL;
	struct isl_set *set_j = NULL;
	struct isl_vec *bound = NULL;
	unsigned total = isl_basic_map_total_dim(map->p[i]);
	struct isl_tab_undo *snap;
	int n;

	set_i = set_from_updated_bmap(map->p[i], tabs[i]);
	set_j = set_from_updated_bmap(map->p[j], tabs[j]);
	mat = isl_mat_alloc(map->ctx, 2 * (map->p[i]->n_eq + map->p[j]->n_eq) +
					map->p[i]->n_ineq + map->p[j]->n_ineq,
					1 + total);
	wraps_init(&wraps, mat, map, i, j, eq_i, ineq_i, eq_j, ineq_j);
	bound = isl_vec_alloc(map->ctx, 1 + total);
	if (!set_i || !set_j || !wraps.mat || !bound)
		goto error;

	isl_seq_cpy(bound->el, map->p[i]->ineq[k], 1 + total);
	isl_int_add_ui(bound->el[0], bound->el[0], 1);

	isl_seq_cpy(wraps.mat->row[0], bound->el, 1 + total);
	wraps.mat->n_row = 1;

	if (add_wraps(&wraps, map->p[j], tabs[j], bound->el, set_i) < 0)
		goto error;
	if (!wraps.mat->n_row)
		goto unbounded;

	snap = isl_tab_snap(tabs[i]);

	if (isl_tab_select_facet(tabs[i], map->p[i]->n_eq + k) < 0)
		goto error;
	if (isl_tab_detect_redundant(tabs[i]) < 0)
		goto error;

	isl_seq_neg(bound->el, map->p[i]->ineq[k], 1 + total);

	n = wraps.mat->n_row;
	if (add_wraps(&wraps, map->p[i], tabs[i], bound->el, set_j) < 0)
		goto error;

	if (isl_tab_rollback(tabs[i], snap) < 0)
		goto error;
	if (check_wraps(wraps.mat, n, tabs[i]) < 0)
		goto error;
	if (!wraps.mat->n_row)
		goto unbounded;

	changed = fuse(map, i, j, tabs, eq_i, ineq_i, eq_j, ineq_j, wraps.mat);

unbounded:
	wraps_free(&wraps);

	isl_set_free(set_i);
	isl_set_free(set_j);

	isl_vec_free(bound);

	return changed;
error:
	wraps_free(&wraps);
	isl_vec_free(bound);
	isl_set_free(set_i);
	isl_set_free(set_j);
	return -1;
}

/* Set the is_redundant property of the "n" constraints in "cuts",
 * except "k" to "v".
 * This is a fairly tricky operation as it bypasses isl_tab.c.
 * The reason we want to temporarily mark some constraints redundant
 * is that we want to ignore them in add_wraps.
 *
 * Initially all cut constraints are non-redundant, but the
 * selection of a facet right before the call to this function
 * may have made some of them redundant.
 * Likewise, the same constraints are marked non-redundant
 * in the second call to this function, before they are officially
 * made non-redundant again in the subsequent rollback.
 */
static void set_is_redundant(struct isl_tab *tab, unsigned n_eq,
	int *cuts, int n, int k, int v)
{
	int l;

	for (l = 0; l < n; ++l) {
		if (l == k)
			continue;
		tab->con[n_eq + cuts[l]].is_redundant = v;
	}
}

/* Given a pair of basic maps i and j such that j sticks out
 * of i at n cut constraints, each time by at most one,
 * try to compute wrapping constraints and replace the two
 * basic maps by a single basic map.
 * The other constraints of i are assumed to be valid for j.
 *
 * The facets of i corresponding to the cut constraints are
 * wrapped around their ridges, except those ridges determined
 * by any of the other cut constraints.
 * The intersections of cut constraints need to be ignored
 * as the result of wrapping one cut constraint around another
 * would result in a constraint cutting the union.
 * In each case, the facets are wrapped to include the union
 * of the two basic maps.
 *
 * The pieces of j that lie at an offset of exactly one from
 * one of the cut constraints of i are wrapped around their edges.
 * Here, there is no need to ignore intersections because we
 * are wrapping around the union of the two basic maps.
 *
 * If any wrapping fails, i.e., if we cannot wrap to touch
 * the union, then we give up.
 * Otherwise, the pair of basic maps is replaced by their union.
 */
static int wrap_in_facets(struct isl_map *map, int i, int j,
	int *cuts, int n, struct isl_tab **tabs,
	int *eq_i, int *ineq_i, int *eq_j, int *ineq_j)
{
	int changed = 0;
	struct isl_wraps wraps;
	isl_mat *mat;
	isl_set *set = NULL;
	isl_vec *bound = NULL;
	unsigned total = isl_basic_map_total_dim(map->p[i]);
	int max_wrap;
	int k;
	struct isl_tab_undo *snap_i, *snap_j;

	if (isl_tab_extend_cons(tabs[j], 1) < 0)
		goto error;

	max_wrap = 2 * (map->p[i]->n_eq + map->p[j]->n_eq) +
		    map->p[i]->n_ineq + map->p[j]->n_ineq;
	max_wrap *= n;

	set = isl_set_union(set_from_updated_bmap(map->p[i], tabs[i]),
			    set_from_updated_bmap(map->p[j], tabs[j]));
	mat = isl_mat_alloc(map->ctx, max_wrap, 1 + total);
	wraps_init(&wraps, mat, map, i, j, eq_i, ineq_i, eq_j, ineq_j);
	bound = isl_vec_alloc(map->ctx, 1 + total);
	if (!set || !wraps.mat || !bound)
		goto error;

	snap_i = isl_tab_snap(tabs[i]);
	snap_j = isl_tab_snap(tabs[j]);

	wraps.mat->n_row = 0;

	for (k = 0; k < n; ++k) {
		if (isl_tab_select_facet(tabs[i], map->p[i]->n_eq + cuts[k]) < 0)
			goto error;
		if (isl_tab_detect_redundant(tabs[i]) < 0)
			goto error;
		set_is_redundant(tabs[i], map->p[i]->n_eq, cuts, n, k, 1);

		isl_seq_neg(bound->el, map->p[i]->ineq[cuts[k]], 1 + total);
		if (!tabs[i]->empty &&
		    add_wraps(&wraps, map->p[i], tabs[i], bound->el, set) < 0)
			goto error;

		set_is_redundant(tabs[i], map->p[i]->n_eq, cuts, n, k, 0);
		if (isl_tab_rollback(tabs[i], snap_i) < 0)
			goto error;

		if (tabs[i]->empty)
			break;
		if (!wraps.mat->n_row)
			break;

		isl_seq_cpy(bound->el, map->p[i]->ineq[cuts[k]], 1 + total);
		isl_int_add_ui(bound->el[0], bound->el[0], 1);
		if (isl_tab_add_eq(tabs[j], bound->el) < 0)
			goto error;
		if (isl_tab_detect_redundant(tabs[j]) < 0)
			goto error;

		if (!tabs[j]->empty &&
		    add_wraps(&wraps, map->p[j], tabs[j], bound->el, set) < 0)
			goto error;

		if (isl_tab_rollback(tabs[j], snap_j) < 0)
			goto error;

		if (!wraps.mat->n_row)
			break;
	}

	if (k == n)
		changed = fuse(map, i, j, tabs,
				eq_i, ineq_i, eq_j, ineq_j, wraps.mat);

	isl_vec_free(bound);
	wraps_free(&wraps);
	isl_set_free(set);

	return changed;
error:
	isl_vec_free(bound);
	wraps_free(&wraps);
	isl_set_free(set);
	return -1;
}

/* Given two basic sets i and j such that i has no cut equalities,
 * check if relaxing all the cut inequalities of i by one turns
 * them into valid constraint for j and check if we can wrap in
 * the bits that are sticking out.
 * If so, replace the pair by their union.
 *
 * We first check if all relaxed cut inequalities of i are valid for j
 * and then try to wrap in the intersections of the relaxed cut inequalities
 * with j.
 *
 * During this wrapping, we consider the points of j that lie at a distance
 * of exactly 1 from i.  In particular, we ignore the points that lie in
 * between this lower-dimensional space and the basic map i.
 * We can therefore only apply this to integer maps.
 *        ____			  _____
 *       / ___|_		 /     \
 *      / |    |  		/      |
 *      \ |    |   	=>	\      |
 *       \|____|		 \     |
 *        \___| 		  \____/
 *
 *	 _____			 ______
 *	| ____|_		|      \
 *	| |     |		|       |
 *	| |	|	=>	|       |
 *	|_|     |		|       |
 *	  |_____|		 \______|
 *
 *	 _______
 *	|       |
 *	|  |\   |
 *	|  | \  |
 *	|  |  \ |
 *	|  |   \|
 *	|  |    \
 *	|  |_____\
 *	|       |
 *	|_______|
 *
 * Wrapping can fail if the result of wrapping one of the facets
 * around its edges does not produce any new facet constraint.
 * In particular, this happens when we try to wrap in unbounded sets.
 *
 *	 _______________________________________________________________________
 *	|
 *	|  ___
 *	| |   |
 *	|_|   |_________________________________________________________________
 *	  |___|
 *
 * The following is not an acceptable result of coalescing the above two
 * sets as it includes extra integer points.
 *	 _______________________________________________________________________
 *	|
 *	|     
 *	|      
 *	|
 *	 \______________________________________________________________________
 */
static int can_wrap_in_set(struct isl_map *map, int i, int j,
	struct isl_tab **tabs, int *eq_i, int *ineq_i, int *eq_j, int *ineq_j)
{
	int changed = 0;
	int k, m;
	int n;
	int *cuts = NULL;

	if (ISL_F_ISSET(map->p[i], ISL_BASIC_MAP_RATIONAL) ||
	    ISL_F_ISSET(map->p[j], ISL_BASIC_MAP_RATIONAL))
		return 0;

	n = count(ineq_i, map->p[i]->n_ineq, STATUS_CUT);
	if (n == 0)
		return 0;

	cuts = isl_alloc_array(map->ctx, int, n);
	if (!cuts)
		return -1;

	for (k = 0, m = 0; m < n; ++k) {
		enum isl_ineq_type type;

		if (ineq_i[k] != STATUS_CUT)
			continue;

		isl_int_add_ui(map->p[i]->ineq[k][0], map->p[i]->ineq[k][0], 1);
		type = isl_tab_ineq_type(tabs[j], map->p[i]->ineq[k]);
		isl_int_sub_ui(map->p[i]->ineq[k][0], map->p[i]->ineq[k][0], 1);
		if (type == isl_ineq_error)
			goto error;
		if (type != isl_ineq_redundant)
			break;
		cuts[m] = k;
		++m;
	}

	if (m == n)
		changed = wrap_in_facets(map, i, j, cuts, n, tabs,
					 eq_i, ineq_i, eq_j, ineq_j);

	free(cuts);

	return changed;
error:
	free(cuts);
	return -1;
}

/* Check if either i or j has a single cut constraint that can
 * be used to wrap in (a facet of) the other basic set.
 * if so, replace the pair by their union.
 */
static int check_wrap(struct isl_map *map, int i, int j,
	struct isl_tab **tabs, int *eq_i, int *ineq_i, int *eq_j, int *ineq_j)
{
	int changed = 0;

	if (!any(eq_i, 2 * map->p[i]->n_eq, STATUS_CUT))
		changed = can_wrap_in_set(map, i, j, tabs,
					    eq_i, ineq_i, eq_j, ineq_j);
	if (changed)
		return changed;

	if (!any(eq_j, 2 * map->p[j]->n_eq, STATUS_CUT))
		changed = can_wrap_in_set(map, j, i, tabs,
					    eq_j, ineq_j, eq_i, ineq_i);
	return changed;
}

/* At least one of the basic maps has an equality that is adjacent
 * to inequality.  Make sure that only one of the basic maps has
 * such an equality and that the other basic map has exactly one
 * inequality adjacent to an equality.
 * We call the basic map that has the inequality "i" and the basic
 * map that has the equality "j".
 * If "i" has any "cut" (in)equality, then relaxing the inequality
 * by one would not result in a basic map that contains the other
 * basic map.
 */
static int check_adj_eq(struct isl_map *map, int i, int j,
	struct isl_tab **tabs, int *eq_i, int *ineq_i, int *eq_j, int *ineq_j)
{
	int changed = 0;
	int k;

	if (any(eq_i, 2 * map->p[i]->n_eq, STATUS_ADJ_INEQ) &&
	    any(eq_j, 2 * map->p[j]->n_eq, STATUS_ADJ_INEQ))
		/* ADJ EQ TOO MANY */
		return 0;

	if (any(eq_i, 2 * map->p[i]->n_eq, STATUS_ADJ_INEQ))
		return check_adj_eq(map, j, i, tabs,
					eq_j, ineq_j, eq_i, ineq_i);

	/* j has an equality adjacent to an inequality in i */

	if (any(eq_i, 2 * map->p[i]->n_eq, STATUS_CUT))
		return 0;
	if (any(ineq_i, map->p[i]->n_ineq, STATUS_CUT))
		/* ADJ EQ CUT */
		return 0;
	if (count(ineq_i, map->p[i]->n_ineq, STATUS_ADJ_EQ) != 1 ||
	    any(ineq_j, map->p[j]->n_ineq, STATUS_ADJ_EQ) ||
	    any(ineq_i, map->p[i]->n_ineq, STATUS_ADJ_INEQ) ||
	    any(ineq_j, map->p[j]->n_ineq, STATUS_ADJ_INEQ))
		/* ADJ EQ TOO MANY */
		return 0;

	for (k = 0; k < map->p[i]->n_ineq; ++k)
		if (ineq_i[k] == STATUS_ADJ_EQ)
			break;

	changed = is_adj_eq_extension(map, i, j, k, tabs,
					eq_i, ineq_i, eq_j, ineq_j);
	if (changed)
		return changed;

	if (count(eq_j, 2 * map->p[j]->n_eq, STATUS_ADJ_INEQ) != 1)
		return 0;

	changed = can_wrap_in_facet(map, i, j, k, tabs, eq_i, ineq_i, eq_j, ineq_j);

	return changed;
}

/* The two basic maps lie on adjacent hyperplanes.  In particular,
 * basic map "i" has an equality that lies parallel to basic map "j".
 * Check if we can wrap the facets around the parallel hyperplanes
 * to include the other set.
 *
 * We perform basically the same operations as can_wrap_in_facet,
 * except that we don't need to select a facet of one of the sets.
 *				_
 *	\\			\\
 *	 \\		=>	 \\
 *	  \			  \|
 *
 * We only allow one equality of "i" to be adjacent to an equality of "j"
 * to avoid coalescing
 *
 *	[m, n] -> { [x, y] -> [x, 1 + y] : x >= 1 and y >= 1 and
 *					    x <= 10 and y <= 10;
 *		    [x, y] -> [1 + x, y] : x >= 1 and x <= 20 and
 *					    y >= 5 and y <= 15 }
 *
 * to
 *
 *	[m, n] -> { [x, y] -> [x2, y2] : x >= 1 and 10y2 <= 20 - x + 10y and
 *					4y2 >= 5 + 3y and 5y2 <= 15 + 4y and
 *					y2 <= 1 + x + y - x2 and y2 >= y and
 *					y2 >= 1 + x + y - x2 }
 */
static int check_eq_adj_eq(struct isl_map *map, int i, int j,
	struct isl_tab **tabs, int *eq_i, int *ineq_i, int *eq_j, int *ineq_j)
{
	int k;
	int changed = 0;
	struct isl_wraps wraps;
	isl_mat *mat;
	struct isl_set *set_i = NULL;
	struct isl_set *set_j = NULL;
	struct isl_vec *bound = NULL;
	unsigned total = isl_basic_map_total_dim(map->p[i]);

	if (count(eq_i, 2 * map->p[i]->n_eq, STATUS_ADJ_EQ) != 1)
		return 0;

	for (k = 0; k < 2 * map->p[i]->n_eq ; ++k)
		if (eq_i[k] == STATUS_ADJ_EQ)
			break;

	set_i = set_from_updated_bmap(map->p[i], tabs[i]);
	set_j = set_from_updated_bmap(map->p[j], tabs[j]);
	mat = isl_mat_alloc(map->ctx, 2 * (map->p[i]->n_eq + map->p[j]->n_eq) +
					map->p[i]->n_ineq + map->p[j]->n_ineq,
					1 + total);
	wraps_init(&wraps, mat, map, i, j, eq_i, ineq_i, eq_j, ineq_j);
	bound = isl_vec_alloc(map->ctx, 1 + total);
	if (!set_i || !set_j || !wraps.mat || !bound)
		goto error;

	if (k % 2 == 0)
		isl_seq_neg(bound->el, map->p[i]->eq[k / 2], 1 + total);
	else
		isl_seq_cpy(bound->el, map->p[i]->eq[k / 2], 1 + total);
	isl_int_add_ui(bound->el[0], bound->el[0], 1);

	isl_seq_cpy(wraps.mat->row[0], bound->el, 1 + total);
	wraps.mat->n_row = 1;

	if (add_wraps(&wraps, map->p[j], tabs[j], bound->el, set_i) < 0)
		goto error;
	if (!wraps.mat->n_row)
		goto unbounded;

	isl_int_sub_ui(bound->el[0], bound->el[0], 1);
	isl_seq_neg(bound->el, bound->el, 1 + total);

	isl_seq_cpy(wraps.mat->row[wraps.mat->n_row], bound->el, 1 + total);
	wraps.mat->n_row++;

	if (add_wraps(&wraps, map->p[i], tabs[i], bound->el, set_j) < 0)
		goto error;
	if (!wraps.mat->n_row)
		goto unbounded;

	changed = fuse(map, i, j, tabs, eq_i, ineq_i, eq_j, ineq_j, wraps.mat);

	if (0) {
error:		changed = -1;
	}
unbounded:

	wraps_free(&wraps);
	isl_set_free(set_i);
	isl_set_free(set_j);
	isl_vec_free(bound);

	return changed;
}

/* Check if the union of the given pair of basic maps
 * can be represented by a single basic map.
 * If so, replace the pair by the single basic map and return 1.
 * Otherwise, return 0;
 * The two basic maps are assumed to live in the same local space.
 *
 * We first check the effect of each constraint of one basic map
 * on the other basic map.
 * The constraint may be
 *	redundant	the constraint is redundant in its own
 *			basic map and should be ignore and removed
 *			in the end
 *	valid		all (integer) points of the other basic map
 *			satisfy the constraint
 *	separate	no (integer) point of the other basic map
 *			satisfies the constraint
 *	cut		some but not all points of the other basic map
 *			satisfy the constraint
 *	adj_eq		the given constraint is adjacent (on the outside)
 *			to an equality of the other basic map
 *	adj_ineq	the given constraint is adjacent (on the outside)
 *			to an inequality of the other basic map
 *
 * We consider seven cases in which we can replace the pair by a single
 * basic map.  We ignore all "redundant" constraints.
 *
 *	1. all constraints of one basic map are valid
 *		=> the other basic map is a subset and can be removed
 *
 *	2. all constraints of both basic maps are either "valid" or "cut"
 *	   and the facets corresponding to the "cut" constraints
 *	   of one of the basic maps lies entirely inside the other basic map
 *		=> the pair can be replaced by a basic map consisting
 *		   of the valid constraints in both basic maps
 *
 *	3. there is a single pair of adjacent inequalities
 *	   (all other constraints are "valid")
 *		=> the pair can be replaced by a basic map consisting
 *		   of the valid constraints in both basic maps
 *
 *	4. one basic map has a single adjacent inequality, while the other
 *	   constraints are "valid".  The other basic map has some
 *	   "cut" constraints, but replacing the adjacent inequality by
 *	   its opposite and adding the valid constraints of the other
 *	   basic map results in a subset of the other basic map
 *		=> the pair can be replaced by a basic map consisting
 *		   of the valid constraints in both basic maps
 *
 *	5. there is a single adjacent pair of an inequality and an equality,
 *	   the other constraints of the basic map containing the inequality are
 *	   "valid".  Moreover, if the inequality the basic map is relaxed
 *	   and then turned into an equality, then resulting facet lies
 *	   entirely inside the other basic map
 *		=> the pair can be replaced by the basic map containing
 *		   the inequality, with the inequality relaxed.
 *
 *	6. there is a single adjacent pair of an inequality and an equality,
 *	   the other constraints of the basic map containing the inequality are
 *	   "valid".  Moreover, the facets corresponding to both
 *	   the inequality and the equality can be wrapped around their
 *	   ridges to include the other basic map
 *		=> the pair can be replaced by a basic map consisting
 *		   of the valid constraints in both basic maps together
 *		   with all wrapping constraints
 *
 *	7. one of the basic maps extends beyond the other by at most one.
 *	   Moreover, the facets corresponding to the cut constraints and
 *	   the pieces of the other basic map at offset one from these cut
 *	   constraints can be wrapped around their ridges to include
 *	   the union of the two basic maps
 *		=> the pair can be replaced by a basic map consisting
 *		   of the valid constraints in both basic maps together
 *		   with all wrapping constraints
 *
 *	8. the two basic maps live in adjacent hyperplanes.  In principle
 *	   such sets can always be combined through wrapping, but we impose
 *	   that there is only one such pair, to avoid overeager coalescing.
 *
 * Throughout the computation, we maintain a collection of tableaus
 * corresponding to the basic maps.  When the basic maps are dropped
 * or combined, the tableaus are modified accordingly.
 */
static int coalesce_local_pair(__isl_keep isl_map *map, int i, int j,
	struct isl_tab **tabs)
{
	int changed = 0;
	int *eq_i = NULL;
	int *eq_j = NULL;
	int *ineq_i = NULL;
	int *ineq_j = NULL;

	eq_i = eq_status_in(map->p[i], tabs[j]);
	if (map->p[i]->n_eq && !eq_i)
		goto error;
	if (any(eq_i, 2 * map->p[i]->n_eq, STATUS_ERROR))
		goto error;
	if (any(eq_i, 2 * map->p[i]->n_eq, STATUS_SEPARATE))
		goto done;

	eq_j = eq_status_in(map->p[j], tabs[i]);
	if (map->p[j]->n_eq && !eq_j)
		goto error;
	if (any(eq_j, 2 * map->p[j]->n_eq, STATUS_ERROR))
		goto error;
	if (any(eq_j, 2 * map->p[j]->n_eq, STATUS_SEPARATE))
		goto done;

	ineq_i = ineq_status_in(map->p[i], tabs[i], tabs[j]);
	if (map->p[i]->n_ineq && !ineq_i)
		goto error;
	if (any(ineq_i, map->p[i]->n_ineq, STATUS_ERROR))
		goto error;
	if (any(ineq_i, map->p[i]->n_ineq, STATUS_SEPARATE))
		goto done;

	ineq_j = ineq_status_in(map->p[j], tabs[j], tabs[i]);
	if (map->p[j]->n_ineq && !ineq_j)
		goto error;
	if (any(ineq_j, map->p[j]->n_ineq, STATUS_ERROR))
		goto error;
	if (any(ineq_j, map->p[j]->n_ineq, STATUS_SEPARATE))
		goto done;

	if (all(eq_i, 2 * map->p[i]->n_eq, STATUS_VALID) &&
	    all(ineq_i, map->p[i]->n_ineq, STATUS_VALID)) {
		drop(map, j, tabs);
		changed = 1;
	} else if (all(eq_j, 2 * map->p[j]->n_eq, STATUS_VALID) &&
		   all(ineq_j, map->p[j]->n_ineq, STATUS_VALID)) {
		drop(map, i, tabs);
		changed = 1;
	} else if (any(eq_i, 2 * map->p[i]->n_eq, STATUS_ADJ_EQ)) {
		changed = check_eq_adj_eq(map, i, j, tabs,
					eq_i, ineq_i, eq_j, ineq_j);
	} else if (any(eq_j, 2 * map->p[j]->n_eq, STATUS_ADJ_EQ)) {
		changed = check_eq_adj_eq(map, j, i, tabs,
					eq_j, ineq_j, eq_i, ineq_i);
	} else if (any(eq_i, 2 * map->p[i]->n_eq, STATUS_ADJ_INEQ) ||
		   any(eq_j, 2 * map->p[j]->n_eq, STATUS_ADJ_INEQ)) {
		changed = check_adj_eq(map, i, j, tabs,
					eq_i, ineq_i, eq_j, ineq_j);
	} else if (any(ineq_i, map->p[i]->n_ineq, STATUS_ADJ_EQ) ||
		   any(ineq_j, map->p[j]->n_ineq, STATUS_ADJ_EQ)) {
		/* Can't happen */
		/* BAD ADJ INEQ */
	} else if (any(ineq_i, map->p[i]->n_ineq, STATUS_ADJ_INEQ) ||
		   any(ineq_j, map->p[j]->n_ineq, STATUS_ADJ_INEQ)) {
		changed = check_adj_ineq(map, i, j, tabs,
					eq_i, ineq_i, eq_j, ineq_j);
	} else {
		if (!any(eq_i, 2 * map->p[i]->n_eq, STATUS_CUT) &&
		    !any(eq_j, 2 * map->p[j]->n_eq, STATUS_CUT))
			changed = check_facets(map, i, j, tabs, ineq_i, ineq_j);
		if (!changed)
			changed = check_wrap(map, i, j, tabs,
						eq_i, ineq_i, eq_j, ineq_j);
	}

done:
	free(eq_i);
	free(eq_j);
	free(ineq_i);
	free(ineq_j);
	return changed;
error:
	free(eq_i);
	free(eq_j);
	free(ineq_i);
	free(ineq_j);
	return -1;
}

/* Do the two basic maps live in the same local space, i.e.,
 * do they have the same (known) divs?
 * If either basic map has any unknown divs, then we can only assume
 * that they do not live in the same local space.
 */
static int same_divs(__isl_keep isl_basic_map *bmap1,
	__isl_keep isl_basic_map *bmap2)
{
	int i;
	int known;
	int total;

	if (!bmap1 || !bmap2)
		return -1;
	if (bmap1->n_div != bmap2->n_div)
		return 0;

	if (bmap1->n_div == 0)
		return 1;

	known = isl_basic_map_divs_known(bmap1);
	if (known < 0 || !known)
		return known;
	known = isl_basic_map_divs_known(bmap2);
	if (known < 0 || !known)
		return known;

	total = isl_basic_map_total_dim(bmap1);
	for (i = 0; i < bmap1->n_div; ++i)
		if (!isl_seq_eq(bmap1->div[i], bmap2->div[i], 2 + total))
			return 0;

	return 1;
}

/* Given two basic maps "i" and "j", where the divs of "i" form a subset
 * of those of "j", check if basic map "j" is a subset of basic map "i"
 * and, if so, drop basic map "j".
 *
 * We first expand the divs of basic map "i" to match those of basic map "j",
 * using the divs and expansion computed by the caller.
 * Then we check if all constraints of the expanded "i" are valid for "j".
 */
static int coalesce_subset(__isl_keep isl_map *map, int i, int j,
	struct isl_tab **tabs, __isl_keep isl_mat *div, int *exp)
{
	isl_basic_map *bmap;
	int changed = 0;
	int *eq_i = NULL;
	int *ineq_i = NULL;

	bmap = isl_basic_map_copy(map->p[i]);
	bmap = isl_basic_set_expand_divs(bmap, isl_mat_copy(div), exp);

	if (!bmap)
		goto error;

	eq_i = eq_status_in(bmap, tabs[j]);
	if (bmap->n_eq && !eq_i)
		goto error;
	if (any(eq_i, 2 * bmap->n_eq, STATUS_ERROR))
		goto error;
	if (any(eq_i, 2 * bmap->n_eq, STATUS_SEPARATE))
		goto done;

	ineq_i = ineq_status_in(bmap, NULL, tabs[j]);
	if (bmap->n_ineq && !ineq_i)
		goto error;
	if (any(ineq_i, bmap->n_ineq, STATUS_ERROR))
		goto error;
	if (any(ineq_i, bmap->n_ineq, STATUS_SEPARATE))
		goto done;

	if (all(eq_i, 2 * map->p[i]->n_eq, STATUS_VALID) &&
	    all(ineq_i, map->p[i]->n_ineq, STATUS_VALID)) {
		drop(map, j, tabs);
		changed = 1;
	}

done:
	isl_basic_map_free(bmap);
	free(eq_i);
	free(ineq_i);
	return 0;
error:
	isl_basic_map_free(bmap);
	free(eq_i);
	free(ineq_i);
	return -1;
}

/* Check if the basic map "j" is a subset of basic map "i",
 * assuming that "i" has fewer divs that "j".
 * If not, then we change the order.
 *
 * If the two basic maps have the same number of divs, then
 * they must necessarily be different.  Otherwise, we would have
 * called coalesce_local_pair.  We therefore don't try anything
 * in this case.
 *
 * We first check if the divs of "i" are all known and form a subset
 * of those of "j".  If so, we pass control over to coalesce_subset.
 */
static int check_coalesce_subset(__isl_keep isl_map *map, int i, int j,
	struct isl_tab **tabs)
{
	int known;
	isl_mat *div_i, *div_j, *div;
	int *exp1 = NULL;
	int *exp2 = NULL;
	isl_ctx *ctx;
	int subset;

	if (map->p[i]->n_div == map->p[j]->n_div)
		return 0;
	if (map->p[j]->n_div < map->p[i]->n_div)
		return check_coalesce_subset(map, j, i, tabs);

	known = isl_basic_map_divs_known(map->p[i]);
	if (known < 0 || !known)
		return known;

	ctx = isl_map_get_ctx(map);

	div_i = isl_basic_map_get_divs(map->p[i]);
	div_j = isl_basic_map_get_divs(map->p[j]);

	if (!div_i || !div_j)
		goto error;

	exp1 = isl_alloc_array(ctx, int, div_i->n_row);
	exp2 = isl_alloc_array(ctx, int, div_j->n_row);
	if ((div_i->n_row && !exp1) || (div_j->n_row && !exp2))
		goto error;

	div = isl_merge_divs(div_i, div_j, exp1, exp2);
	if (!div)
		goto error;

	if (div->n_row == div_j->n_row)
		subset = coalesce_subset(map, i, j, tabs, div, exp1);
	else
		subset = 0;

	isl_mat_free(div);

	isl_mat_free(div_i);
	isl_mat_free(div_j);

	free(exp2);
	free(exp1);

	return subset;
error:
	isl_mat_free(div_i);
	isl_mat_free(div_j);
	free(exp1);
	free(exp2);
	return -1;
}

/* Check if the union of the given pair of basic maps
 * can be represented by a single basic map.
 * If so, replace the pair by the single basic map and return 1.
 * Otherwise, return 0;
 *
 * We first check if the two basic maps live in the same local space.
 * If so, we do the complete check.  Otherwise, we check if one is
 * an obvious subset of the other.
 */
static int coalesce_pair(__isl_keep isl_map *map, int i, int j,
	struct isl_tab **tabs)
{
	int same;

	same = same_divs(map->p[i], map->p[j]);
	if (same < 0)
		return -1;
	if (same)
		return coalesce_local_pair(map, i, j, tabs);

	return check_coalesce_subset(map, i, j, tabs);
}

static struct isl_map *coalesce(struct isl_map *map, struct isl_tab **tabs)
{
	int i, j;

	for (i = map->n - 2; i >= 0; --i)
restart:
		for (j = i + 1; j < map->n; ++j) {
			int changed;
			changed = coalesce_pair(map, i, j, tabs);
			if (changed < 0)
				goto error;
			if (changed)
				goto restart;
		}
	return map;
error:
	isl_map_free(map);
	return NULL;
}

/* For each pair of basic maps in the map, check if the union of the two
 * can be represented by a single basic map.
 * If so, replace the pair by the single basic map and start over.
 *
 * Since we are constructing the tableaus of the basic maps anyway,
 * we exploit them to detect implicit equalities and redundant constraints.
 * This also helps the coalescing as it can ignore the redundant constraints.
 * In order to avoid confusion, we make all implicit equalities explicit
 * in the basic maps.  We don't call isl_basic_map_gauss, though,
 * as that may affect the number of constraints.
 * This means that we have to call isl_basic_map_gauss at the end
 * of the computation to ensure that the basic maps are not left
 * in an unexpected state.
 */
struct isl_map *isl_map_coalesce(struct isl_map *map)
{
	int i;
	unsigned n;
	struct isl_tab **tabs = NULL;

	map = isl_map_remove_empty_parts(map);
	if (!map)
		return NULL;

	if (map->n <= 1)
		return map;

	map = isl_map_sort_divs(map);
	map = isl_map_cow(map);

	if (!map)
		return NULL;

	tabs = isl_calloc_array(map->ctx, struct isl_tab *, map->n);
	if (!tabs)
		goto error;

	n = map->n;
	for (i = 0; i < map->n; ++i) {
		tabs[i] = isl_tab_from_basic_map(map->p[i], 0);
		if (!tabs[i])
			goto error;
		if (!ISL_F_ISSET(map->p[i], ISL_BASIC_MAP_NO_IMPLICIT))
			if (isl_tab_detect_implicit_equalities(tabs[i]) < 0)
				goto error;
		map->p[i] = isl_tab_make_equalities_explicit(tabs[i],
								map->p[i]);
		if (!map->p[i])
			goto error;
		if (!ISL_F_ISSET(map->p[i], ISL_BASIC_MAP_NO_REDUNDANT))
			if (isl_tab_detect_redundant(tabs[i]) < 0)
				goto error;
	}
	for (i = map->n - 1; i >= 0; --i)
		if (tabs[i]->empty)
			drop(map, i, tabs);

	map = coalesce(map, tabs);

	if (map)
		for (i = 0; i < map->n; ++i) {
			map->p[i] = isl_basic_map_update_from_tab(map->p[i],
								    tabs[i]);
			map->p[i] = isl_basic_map_gauss(map->p[i], NULL);
			map->p[i] = isl_basic_map_finalize(map->p[i]);
			if (!map->p[i])
				goto error;
			ISL_F_SET(map->p[i], ISL_BASIC_MAP_NO_IMPLICIT);
			ISL_F_SET(map->p[i], ISL_BASIC_MAP_NO_REDUNDANT);
		}

	for (i = 0; i < n; ++i)
		isl_tab_free(tabs[i]);

	free(tabs);

	return map;
error:
	if (tabs)
		for (i = 0; i < n; ++i)
			isl_tab_free(tabs[i]);
	free(tabs);
	isl_map_free(map);
	return NULL;
}

/* For each pair of basic sets in the set, check if the union of the two
 * can be represented by a single basic set.
 * If so, replace the pair by the single basic set and start over.
 */
struct isl_set *isl_set_coalesce(struct isl_set *set)
{
	return (struct isl_set *)isl_map_coalesce((struct isl_map *)set);
}
