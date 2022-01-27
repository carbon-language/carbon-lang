/*
 * Copyright 2010-2011 INRIA Saclay
 * Copyright 2014      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France 
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl_map_private.h>
#include <isl_aff_private.h>
#include <isl_morph.h>
#include <isl_seq.h>
#include <isl_mat_private.h>
#include <isl_space_private.h>
#include <isl_equalities.h>
#include <isl_id_private.h>
#include <isl_aff_private.h>
#include <isl_vec_private.h>

isl_ctx *isl_morph_get_ctx(__isl_keep isl_morph *morph)
{
	if (!morph)
		return NULL;
	return isl_basic_set_get_ctx(morph->dom);
}

__isl_give isl_morph *isl_morph_alloc(
	__isl_take isl_basic_set *dom, __isl_take isl_basic_set *ran,
	__isl_take isl_mat *map, __isl_take isl_mat *inv)
{
	isl_morph *morph;

	if (!dom || !ran || !map || !inv)
		goto error;

	morph = isl_alloc_type(dom->ctx, struct isl_morph);
	if (!morph)
		goto error;

	morph->ref = 1;
	morph->dom = dom;
	morph->ran = ran;
	morph->map = map;
	morph->inv = inv;

	return morph;
error:
	isl_basic_set_free(dom);
	isl_basic_set_free(ran);
	isl_mat_free(map);
	isl_mat_free(inv);
	return NULL;
}

__isl_give isl_morph *isl_morph_copy(__isl_keep isl_morph *morph)
{
	if (!morph)
		return NULL;

	morph->ref++;
	return morph;
}

__isl_give isl_morph *isl_morph_dup(__isl_keep isl_morph *morph)
{
	if (!morph)
		return NULL;

	return isl_morph_alloc(isl_basic_set_copy(morph->dom),
		isl_basic_set_copy(morph->ran),
		isl_mat_copy(morph->map), isl_mat_copy(morph->inv));
}

__isl_give isl_morph *isl_morph_cow(__isl_take isl_morph *morph)
{
	if (!morph)
		return NULL;

	if (morph->ref == 1)
		return morph;
	morph->ref--;
	return isl_morph_dup(morph);
}

__isl_null isl_morph *isl_morph_free(__isl_take isl_morph *morph)
{
	if (!morph)
		return NULL;

	if (--morph->ref > 0)
		return NULL;

	isl_basic_set_free(morph->dom);
	isl_basic_set_free(morph->ran);
	isl_mat_free(morph->map);
	isl_mat_free(morph->inv);
	free(morph);

	return NULL;
}

/* Is "morph" an identity on the parameters?
 */
static isl_bool identity_on_parameters(__isl_keep isl_morph *morph)
{
	isl_bool is_identity;
	isl_size nparam, nparam_ran;
	isl_mat *sub;

	nparam = isl_morph_dom_dim(morph, isl_dim_param);
	nparam_ran = isl_morph_ran_dim(morph, isl_dim_param);
	if (nparam < 0 || nparam_ran < 0)
		return isl_bool_error;
	if (nparam != nparam_ran)
		return isl_bool_false;
	if (nparam == 0)
		return isl_bool_true;
	sub = isl_mat_sub_alloc(morph->map, 0, 1 + nparam, 0, 1 + nparam);
	is_identity = isl_mat_is_scaled_identity(sub);
	isl_mat_free(sub);

	return is_identity;
}

/* Return an affine expression of the variables of the range of "morph"
 * in terms of the parameters and the variables of the domain on "morph".
 *
 * In order for the space manipulations to make sense, we require
 * that the parameters are not modified by "morph".
 */
__isl_give isl_multi_aff *isl_morph_get_var_multi_aff(
	__isl_keep isl_morph *morph)
{
	isl_space *dom, *ran, *space;
	isl_local_space *ls;
	isl_multi_aff *ma;
	isl_size nparam, nvar;
	int i;
	isl_bool is_identity;

	if (!morph)
		return NULL;

	is_identity = identity_on_parameters(morph);
	if (is_identity < 0)
		return NULL;
	if (!is_identity)
		isl_die(isl_morph_get_ctx(morph), isl_error_invalid,
			"cannot handle parameter compression", return NULL);

	dom = isl_morph_get_dom_space(morph);
	ls = isl_local_space_from_space(isl_space_copy(dom));
	ran = isl_morph_get_ran_space(morph);
	space = isl_space_map_from_domain_and_range(dom, ran);
	ma = isl_multi_aff_zero(space);

	nparam = isl_multi_aff_dim(ma, isl_dim_param);
	nvar = isl_multi_aff_dim(ma, isl_dim_out);
	if (nparam < 0 || nvar < 0)
		ma = isl_multi_aff_free(ma);
	for (i = 0; i < nvar; ++i) {
		isl_val *val;
		isl_vec *v;
		isl_aff *aff;

		v = isl_mat_get_row(morph->map, 1 + nparam + i);
		v = isl_vec_insert_els(v, 0, 1);
		val = isl_mat_get_element_val(morph->map, 0, 0);
		v = isl_vec_set_element_val(v, 0, val);
		aff = isl_aff_alloc_vec(isl_local_space_copy(ls), v);
		ma = isl_multi_aff_set_aff(ma, i, aff);
	}

	isl_local_space_free(ls);
	return ma;
}

/* Return the domain space of "morph".
 */
static __isl_keep isl_space *isl_morph_peek_dom_space(
	__isl_keep isl_morph *morph)
{
	if (!morph)
		return NULL;

	return isl_basic_set_peek_space(morph->dom);
}

/* Return a copy of the domain space of "morph".
 */
__isl_give isl_space *isl_morph_get_dom_space(__isl_keep isl_morph *morph)
{
	return isl_space_copy(isl_morph_peek_dom_space(morph));
}

/* Check that the match against "space" with result "match" was successful.
 */
static isl_stat check_space_match(__isl_keep isl_space *space, isl_bool match)
{
	if (match < 0)
		return isl_stat_error;
	if (!match)
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"spaces don't match", return isl_stat_error);

	return isl_stat_ok;
}

/* Check that "morph" can be applied to the "space".
 */
isl_stat isl_morph_check_applies(__isl_keep isl_morph *morph,
	__isl_keep isl_space *space)
{
	isl_space *dom_space;
	isl_bool applies;

	dom_space = isl_morph_peek_dom_space(morph);
	applies = isl_space_is_equal(dom_space, space);
	return check_space_match(space, applies);
}

__isl_give isl_space *isl_morph_get_ran_space(__isl_keep isl_morph *morph)
{
	if (!morph)
		return NULL;
	
	return isl_space_copy(morph->ran->dim);
}

isl_size isl_morph_dom_dim(__isl_keep isl_morph *morph, enum isl_dim_type type)
{
	if (!morph)
		return isl_size_error;

	return isl_basic_set_dim(morph->dom, type);
}

isl_size isl_morph_ran_dim(__isl_keep isl_morph *morph, enum isl_dim_type type)
{
	if (!morph)
		return isl_size_error;

	return isl_basic_set_dim(morph->ran, type);
}

__isl_give isl_morph *isl_morph_remove_dom_dims(__isl_take isl_morph *morph,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	unsigned dom_offset;

	if (n == 0)
		return morph;

	morph = isl_morph_cow(morph);
	if (!morph)
		return NULL;

	dom_offset = 1 + isl_space_offset(morph->dom->dim, type);

	morph->dom = isl_basic_set_remove_dims(morph->dom, type, first, n);

	morph->map = isl_mat_drop_cols(morph->map, dom_offset + first, n);

	morph->inv = isl_mat_drop_rows(morph->inv, dom_offset + first, n);

	if (morph->dom && morph->ran && morph->map && morph->inv)
		return morph;

	isl_morph_free(morph);
	return NULL;
}

__isl_give isl_morph *isl_morph_remove_ran_dims(__isl_take isl_morph *morph,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	unsigned ran_offset;

	if (n == 0)
		return morph;

	morph = isl_morph_cow(morph);
	if (!morph)
		return NULL;

	ran_offset = 1 + isl_space_offset(morph->ran->dim, type);

	morph->ran = isl_basic_set_remove_dims(morph->ran, type, first, n);

	morph->map = isl_mat_drop_rows(morph->map, ran_offset + first, n);

	morph->inv = isl_mat_drop_cols(morph->inv, ran_offset + first, n);

	if (morph->dom && morph->ran && morph->map && morph->inv)
		return morph;

	isl_morph_free(morph);
	return NULL;
}

/* Project domain of morph onto its parameter domain.
 */
__isl_give isl_morph *isl_morph_dom_params(__isl_take isl_morph *morph)
{
	isl_size n;

	morph = isl_morph_cow(morph);
	if (!morph)
		return NULL;
	n = isl_basic_set_dim(morph->dom, isl_dim_set);
	if (n < 0)
		return isl_morph_free(morph);
	morph = isl_morph_remove_dom_dims(morph, isl_dim_set, 0, n);
	if (!morph)
		return NULL;
	morph->dom = isl_basic_set_params(morph->dom);
	if (morph->dom)
		return morph;

	isl_morph_free(morph);
	return NULL;
}

/* Project range of morph onto its parameter domain.
 */
__isl_give isl_morph *isl_morph_ran_params(__isl_take isl_morph *morph)
{
	isl_size n;

	morph = isl_morph_cow(morph);
	if (!morph)
		return NULL;
	n = isl_basic_set_dim(morph->ran, isl_dim_set);
	if (n < 0)
		return isl_morph_free(morph);
	morph = isl_morph_remove_ran_dims(morph, isl_dim_set, 0, n);
	if (!morph)
		return NULL;
	morph->ran = isl_basic_set_params(morph->ran);
	if (morph->ran)
		return morph;

	isl_morph_free(morph);
	return NULL;
}

/* Replace the identifier of the tuple of the range of the morph by "id".
 */
static __isl_give isl_morph *isl_morph_set_ran_tuple_id(
	__isl_take isl_morph *morph, __isl_keep isl_id *id)
{
	morph = isl_morph_cow(morph);
	if (!morph)
		return NULL;
	morph->ran = isl_basic_set_set_tuple_id(morph->ran, isl_id_copy(id));
	if (!morph->ran)
		return isl_morph_free(morph);
	return morph;
}

void isl_morph_print_internal(__isl_take isl_morph *morph, FILE *out)
{
	if (!morph)
		return;

	isl_basic_set_dump(morph->dom);
	isl_basic_set_dump(morph->ran);
	isl_mat_print_internal(morph->map, out, 4);
	isl_mat_print_internal(morph->inv, out, 4);
}

void isl_morph_dump(__isl_take isl_morph *morph)
{
	isl_morph_print_internal(morph, stderr);
}

__isl_give isl_morph *isl_morph_identity(__isl_keep isl_basic_set *bset)
{
	isl_mat *id;
	isl_basic_set *universe;
	isl_size total;

	total = isl_basic_set_dim(bset, isl_dim_all);
	if (total < 0)
		return NULL;

	id = isl_mat_identity(bset->ctx, 1 + total);
	universe = isl_basic_set_universe(isl_space_copy(bset->dim));

	return isl_morph_alloc(universe, isl_basic_set_copy(universe),
		id, isl_mat_copy(id));
}

/* Create a(n identity) morphism between empty sets of the same dimension
 * a "bset".
 */
__isl_give isl_morph *isl_morph_empty(__isl_keep isl_basic_set *bset)
{
	isl_mat *id;
	isl_basic_set *empty;
	isl_size total;

	total = isl_basic_set_dim(bset, isl_dim_all);
	if (total < 0)
		return NULL;

	id = isl_mat_identity(bset->ctx, 1 + total);
	empty = isl_basic_set_empty(isl_space_copy(bset->dim));

	return isl_morph_alloc(empty, isl_basic_set_copy(empty),
		id, isl_mat_copy(id));
}

/* Construct a basic set described by the "n" equalities of "bset" starting
 * at "first".
 */
static __isl_give isl_basic_set *copy_equalities(__isl_keep isl_basic_set *bset,
	unsigned first, unsigned n)
{
	int i, k;
	isl_basic_set *eq;
	isl_size total;

	total = isl_basic_set_dim(bset, isl_dim_all);
	if (total < 0 || isl_basic_set_check_no_locals(bset) < 0)
		return NULL;

	eq = isl_basic_set_alloc_space(isl_basic_set_get_space(bset), 0, n, 0);
	if (!eq)
		return NULL;
	for (i = 0; i < n; ++i) {
		k = isl_basic_set_alloc_equality(eq);
		if (k < 0)
			goto error;
		isl_seq_cpy(eq->eq[k], bset->eq[first + i], 1 + total);
	}

	return eq;
error:
	isl_basic_set_free(eq);
	return NULL;
}

/* Given a basic set, exploit the equalities in the basic set to construct
 * a morphism that maps the basic set to a lower-dimensional space.
 * Specifically, the morphism reduces the number of dimensions of type "type".
 *
 * We first select the equalities of interest, that is those that involve
 * variables of type "type" and no later variables.
 * Denote those equalities as
 *
 *		-C(p) + M x = 0
 *
 * where C(p) depends on the parameters if type == isl_dim_set and
 * is a constant if type == isl_dim_param.
 *
 * Use isl_mat_final_variable_compression to construct a compression
 *
 *	x = T x'
 *
 *	x' = Q x
 *
 * If T is a zero-column matrix, then the set of equality constraints
 * do not admit a solution.  In this case, an empty morphism is returned.
 *
 * Both matrices are extended to map the full original space to the full
 * compressed space.
 */
__isl_give isl_morph *isl_basic_set_variable_compression(
	__isl_keep isl_basic_set *bset, enum isl_dim_type type)
{
	unsigned otype;
	isl_size ntype;
	unsigned orest;
	unsigned nrest;
	isl_size total;
	int f_eq, n_eq;
	isl_space *space;
	isl_mat *E, *Q, *C;
	isl_basic_set *dom, *ran;

	if (!bset)
		return NULL;

	if (isl_basic_set_plain_is_empty(bset))
		return isl_morph_empty(bset);

	if (isl_basic_set_check_no_locals(bset) < 0)
		return NULL;

	ntype = isl_basic_set_dim(bset, type);
	total = isl_basic_set_dim(bset, isl_dim_all);
	if (ntype < 0 || total < 0)
		return NULL;
	otype = isl_basic_set_offset(bset, type);
	orest = otype + ntype;
	nrest = total - (orest - 1);

	for (f_eq = 0; f_eq < bset->n_eq; ++f_eq)
		if (isl_seq_first_non_zero(bset->eq[f_eq] + orest, nrest) == -1)
			break;
	for (n_eq = 0; f_eq + n_eq < bset->n_eq; ++n_eq)
		if (isl_seq_first_non_zero(bset->eq[f_eq + n_eq] + otype, ntype) == -1)
			break;
	if (n_eq == 0)
		return isl_morph_identity(bset);

	E = isl_mat_sub_alloc6(bset->ctx, bset->eq, f_eq, n_eq, 0, orest);
	C = isl_mat_final_variable_compression(E, otype - 1, &Q);
	if (!Q)
		C = isl_mat_free(C);
	if (C && C->n_col == 0) {
		isl_mat_free(C);
		isl_mat_free(Q);
		return isl_morph_empty(bset);
	}

	Q = isl_mat_diagonal(Q, isl_mat_identity(bset->ctx, nrest));
	C = isl_mat_diagonal(C, isl_mat_identity(bset->ctx, nrest));

	space = isl_space_copy(bset->dim);
	space = isl_space_drop_dims(space, type, 0, ntype);
	space = isl_space_add_dims(space, type, ntype - n_eq);
	ran = isl_basic_set_universe(space);
	dom = copy_equalities(bset, f_eq, n_eq);

	return isl_morph_alloc(dom, ran, Q, C);
}

/* Given a basic set, exploit the equalities in the basic set to construct
 * a morphism that maps the basic set to a lower-dimensional space
 * with identifier "id".
 * Specifically, the morphism reduces the number of set dimensions.
 */
__isl_give isl_morph *isl_basic_set_variable_compression_with_id(
	__isl_keep isl_basic_set *bset, __isl_keep isl_id *id)
{
	isl_morph *morph;

	morph = isl_basic_set_variable_compression(bset, isl_dim_set);
	morph = isl_morph_set_ran_tuple_id(morph, id);
	return morph;
}

/* Construct a parameter compression for "bset".
 * We basically just call isl_mat_parameter_compression with the right input
 * and then extend the resulting matrix to include the variables.
 *
 * The implementation assumes that "bset" does not have any equalities
 * that only involve the parameters and that isl_basic_set_gauss has
 * been applied to "bset".
 *
 * Let the equalities be given as
 *
 *	B(p) + A x = 0.
 *
 * We use isl_mat_parameter_compression_ext to compute the compression
 *
 *	p = T p'.
 */
__isl_give isl_morph *isl_basic_set_parameter_compression(
	__isl_keep isl_basic_set *bset)
{
	isl_size nparam;
	isl_size nvar;
	isl_size n_div;
	int n_eq;
	isl_mat *H, *B;
	isl_mat *map, *inv;
	isl_basic_set *dom, *ran;

	if (!bset)
		return NULL;

	if (isl_basic_set_plain_is_empty(bset))
		return isl_morph_empty(bset);
	if (bset->n_eq == 0)
		return isl_morph_identity(bset);

	n_eq = bset->n_eq;
	nparam = isl_basic_set_dim(bset, isl_dim_param);
	nvar = isl_basic_set_dim(bset, isl_dim_set);
	n_div = isl_basic_set_dim(bset, isl_dim_div);
	if (nparam < 0 || nvar < 0 || n_div < 0)
		return NULL;

	if (isl_seq_first_non_zero(bset->eq[bset->n_eq - 1] + 1 + nparam,
				    nvar + n_div) == -1)
		isl_die(isl_basic_set_get_ctx(bset), isl_error_invalid,
			"input not allowed to have parameter equalities",
			return NULL);
	if (n_eq > nvar + n_div)
		isl_die(isl_basic_set_get_ctx(bset), isl_error_invalid,
			"input not gaussed", return NULL);

	B = isl_mat_sub_alloc6(bset->ctx, bset->eq, 0, n_eq, 0, 1 + nparam);
	H = isl_mat_sub_alloc6(bset->ctx, bset->eq,
				0, n_eq, 1 + nparam, nvar + n_div);
	inv = isl_mat_parameter_compression_ext(B, H);
	inv = isl_mat_diagonal(inv, isl_mat_identity(bset->ctx, nvar));
	map = isl_mat_right_inverse(isl_mat_copy(inv));

	dom = isl_basic_set_universe(isl_space_copy(bset->dim));
	ran = isl_basic_set_universe(isl_space_copy(bset->dim));

	return isl_morph_alloc(dom, ran, map, inv);
}

/* Construct an isl_multi_aff that corresponds
 * to the affine transformation matrix "mat" and
 * that lives in an anonymous space.
 */
static __isl_give isl_multi_aff *isl_multi_aff_from_aff_mat_anonymous(
	__isl_take isl_mat *mat)
{
	isl_size n_row, n_col;
	isl_ctx *ctx;
	isl_space *space;

	ctx = isl_mat_get_ctx(mat);
	n_row = isl_mat_rows(mat);
	n_col = isl_mat_cols(mat);
	if (n_row < 0 || n_col < 0)
		space = NULL;
	else
		space = isl_space_alloc(ctx, 0, n_col - 1, n_row - 1);

	return isl_multi_aff_from_aff_mat(space, mat);
}

/* Apply the morphism to the basic set.
 * In particular, compute the preimage of "bset" under the inverse mapping
 * in morph and intersect with the range of the morphism.
 * Note that the mapping in morph applies to both parameters and set dimensions,
 * so the parameters need to be treated as set dimensions during the call
 * to isl_basic_set_preimage_multi_aff.
 */
__isl_give isl_basic_set *isl_morph_basic_set(__isl_take isl_morph *morph,
	__isl_take isl_basic_set *bset)
{
	isl_size n_param;
	isl_space *space;
	isl_multi_aff *ma;

	if (!morph || isl_basic_set_check_equal_space(bset, morph->dom) < 0)
		goto error;
	n_param = isl_basic_set_dim(morph->dom, isl_dim_param);
	if (n_param < 0)
		goto error;

	ma = isl_multi_aff_from_aff_mat_anonymous(isl_mat_copy(morph->inv));

	bset = isl_basic_set_move_dims(bset, isl_dim_set, 0,
					isl_dim_param, 0, n_param);
	bset = isl_basic_set_preimage_multi_aff(bset, ma);
	space = isl_basic_set_get_space(morph->ran);
	bset = isl_basic_set_reset_space(bset, space);
	bset = isl_basic_set_intersect(bset, isl_basic_set_copy(morph->ran));

	isl_morph_free(morph);
	return bset;
error:
	isl_morph_free(morph);
	isl_basic_set_free(bset);
	return NULL;
}

/* Apply the morphism to the set.
 * In particular, compute the preimage of "set" under the inverse mapping
 * in morph and intersect with the range of the morphism.
 * Note that the mapping in morph applies to both parameters and set dimensions,
 * so the parameters need to be treated as set dimensions during the call
 * to isl_set_preimage_multi_aff.
 */
__isl_give isl_set *isl_morph_set(__isl_take isl_morph *morph,
	__isl_take isl_set *set)
{
	isl_size n_param;
	isl_space *space;
	isl_multi_aff *ma;
	isl_basic_set *ran;

	if (!morph || isl_set_basic_set_check_equal_space(set, morph->dom) < 0)
		goto error;
	n_param = isl_basic_set_dim(morph->dom, isl_dim_param);
	if (n_param < 0)
		goto error;

	ma = isl_multi_aff_from_aff_mat_anonymous(isl_mat_copy(morph->inv));

	set = isl_set_move_dims(set, isl_dim_set, 0, isl_dim_param, 0, n_param);
	set = isl_set_preimage_multi_aff(set, ma);
	space = isl_basic_set_get_space(morph->ran);
	set = isl_set_reset_space(set, space);
	ran = isl_basic_set_copy(morph->ran);
	set = isl_set_intersect(set, isl_set_from_basic_set(ran));

	isl_morph_free(morph);
	return set;
error:
	isl_set_free(set);
	isl_morph_free(morph);
	return NULL;
}

/* Construct a morphism that first does morph2 and then morph1.
 */
__isl_give isl_morph *isl_morph_compose(__isl_take isl_morph *morph1,
	__isl_take isl_morph *morph2)
{
	isl_mat *map, *inv;
	isl_basic_set *dom, *ran;

	if (!morph1 || !morph2)
		goto error;

	map = isl_mat_product(isl_mat_copy(morph1->map), isl_mat_copy(morph2->map));
	inv = isl_mat_product(isl_mat_copy(morph2->inv), isl_mat_copy(morph1->inv));
	dom = isl_morph_basic_set(isl_morph_inverse(isl_morph_copy(morph2)),
				  isl_basic_set_copy(morph1->dom));
	dom = isl_basic_set_intersect(dom, isl_basic_set_copy(morph2->dom));
	ran = isl_morph_basic_set(isl_morph_copy(morph1),
				  isl_basic_set_copy(morph2->ran));
	ran = isl_basic_set_intersect(ran, isl_basic_set_copy(morph1->ran));

	isl_morph_free(morph1);
	isl_morph_free(morph2);

	return isl_morph_alloc(dom, ran, map, inv);
error:
	isl_morph_free(morph1);
	isl_morph_free(morph2);
	return NULL;
}

__isl_give isl_morph *isl_morph_inverse(__isl_take isl_morph *morph)
{
	isl_basic_set *bset;
	isl_mat *mat;

	morph = isl_morph_cow(morph);
	if (!morph)
		return NULL;

	bset = morph->dom;
	morph->dom = morph->ran;
	morph->ran = bset;

	mat = morph->map;
	morph->map = morph->inv;
	morph->inv = mat;

	return morph;
}

/* We detect all the equalities first to avoid implicit equalities
 * being discovered during the computations.  In particular,
 * the compression on the variables could expose additional stride
 * constraints on the parameters.  This would result in existentially
 * quantified variables after applying the resulting morph, which
 * in turn could break invariants of the calling functions.
 */
__isl_give isl_morph *isl_basic_set_full_compression(
	__isl_keep isl_basic_set *bset)
{
	isl_morph *morph, *morph2;

	bset = isl_basic_set_copy(bset);
	bset = isl_basic_set_detect_equalities(bset);

	morph = isl_basic_set_variable_compression(bset, isl_dim_param);
	bset = isl_morph_basic_set(isl_morph_copy(morph), bset);

	morph2 = isl_basic_set_parameter_compression(bset);
	bset = isl_morph_basic_set(isl_morph_copy(morph2), bset);

	morph = isl_morph_compose(morph2, morph);

	morph2 = isl_basic_set_variable_compression(bset, isl_dim_set);
	isl_basic_set_free(bset);

	morph = isl_morph_compose(morph2, morph);

	return morph;
}

__isl_give isl_vec *isl_morph_vec(__isl_take isl_morph *morph,
	__isl_take isl_vec *vec)
{
	if (!morph)
		goto error;

	vec = isl_mat_vec_product(isl_mat_copy(morph->map), vec);

	isl_morph_free(morph);
	return vec;
error:
	isl_morph_free(morph);
	isl_vec_free(vec);
	return NULL;
}
