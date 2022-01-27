/*
 * Copyright 2011      INRIA Saclay
 * Copyright 2011      Sven Verdoolaege
 * Copyright 2012-2014 Ecole Normale Superieure
 * Copyright 2014      INRIA Rocquencourt
 * Copyright 2016      Sven Verdoolaege
 * Copyright 2018,2020 Cerebras Systems
 * Copyright 2021      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 * and Inria Paris - Rocquencourt, Domaine de Voluceau - Rocquencourt,
 * B.P. 105 - 78153 Le Chesnay, France
 * and Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include <isl_union_map_private.h>
#include <isl_aff_private.h>
#include <isl_space_private.h>
#include <isl_local_space_private.h>
#include <isl_vec_private.h>
#include <isl_mat_private.h>
#include <isl_id_private.h>
#include <isl/constraint.h>
#include <isl_seq.h>
#include <isl/set.h>
#include <isl_val_private.h>
#include <isl_point_private.h>
#include <isl_config.h>

#undef EL_BASE
#define EL_BASE aff

#include <isl_list_templ.c>
#include <isl_list_read_templ.c>

#undef EL_BASE
#define EL_BASE pw_aff

#include <isl_list_templ.c>
#include <isl_list_read_templ.c>

#undef EL_BASE
#define EL_BASE pw_multi_aff

#include <isl_list_templ.c>
#include <isl_list_read_templ.c>

#undef EL_BASE
#define EL_BASE union_pw_aff

#include <isl_list_templ.c>
#include <isl_list_read_templ.c>

#undef EL_BASE
#define EL_BASE union_pw_multi_aff

#include <isl_list_templ.c>

__isl_give isl_aff *isl_aff_alloc_vec(__isl_take isl_local_space *ls,
	__isl_take isl_vec *v)
{
	isl_aff *aff;

	if (!ls || !v)
		goto error;

	aff = isl_calloc_type(v->ctx, struct isl_aff);
	if (!aff)
		goto error;

	aff->ref = 1;
	aff->ls = ls;
	aff->v = v;

	return aff;
error:
	isl_local_space_free(ls);
	isl_vec_free(v);
	return NULL;
}

__isl_give isl_aff *isl_aff_alloc(__isl_take isl_local_space *ls)
{
	isl_ctx *ctx;
	isl_vec *v;
	isl_size total;

	if (!ls)
		return NULL;

	ctx = isl_local_space_get_ctx(ls);
	if (!isl_local_space_divs_known(ls))
		isl_die(ctx, isl_error_invalid, "local space has unknown divs",
			goto error);
	if (!isl_local_space_is_set(ls))
		isl_die(ctx, isl_error_invalid,
			"domain of affine expression should be a set",
			goto error);

	total = isl_local_space_dim(ls, isl_dim_all);
	if (total < 0)
		goto error;
	v = isl_vec_alloc(ctx, 1 + 1 + total);
	return isl_aff_alloc_vec(ls, v);
error:
	isl_local_space_free(ls);
	return NULL;
}

__isl_give isl_aff *isl_aff_copy(__isl_keep isl_aff *aff)
{
	if (!aff)
		return NULL;

	aff->ref++;
	return aff;
}

__isl_give isl_aff *isl_aff_dup(__isl_keep isl_aff *aff)
{
	if (!aff)
		return NULL;

	return isl_aff_alloc_vec(isl_local_space_copy(aff->ls),
				 isl_vec_copy(aff->v));
}

__isl_give isl_aff *isl_aff_cow(__isl_take isl_aff *aff)
{
	if (!aff)
		return NULL;

	if (aff->ref == 1)
		return aff;
	aff->ref--;
	return isl_aff_dup(aff);
}

__isl_give isl_aff *isl_aff_zero_on_domain(__isl_take isl_local_space *ls)
{
	isl_aff *aff;

	aff = isl_aff_alloc(ls);
	if (!aff)
		return NULL;

	isl_int_set_si(aff->v->el[0], 1);
	isl_seq_clr(aff->v->el + 1, aff->v->size - 1);

	return aff;
}

/* Return an affine expression that is equal to zero on domain space "space".
 */
__isl_give isl_aff *isl_aff_zero_on_domain_space(__isl_take isl_space *space)
{
	return isl_aff_zero_on_domain(isl_local_space_from_space(space));
}

/* This function performs the same operation as isl_aff_zero_on_domain_space,
 * but is considered as a function on an isl_space when exported.
 */
__isl_give isl_aff *isl_space_zero_aff_on_domain(__isl_take isl_space *space)
{
	return isl_aff_zero_on_domain_space(space);
}

/* Return a piecewise affine expression defined on the specified domain
 * that is equal to zero.
 */
__isl_give isl_pw_aff *isl_pw_aff_zero_on_domain(__isl_take isl_local_space *ls)
{
	return isl_pw_aff_from_aff(isl_aff_zero_on_domain(ls));
}

/* Change "aff" into a NaN.
 *
 * Note that this function gets called from isl_aff_nan_on_domain,
 * so "aff" may not have been initialized yet.
 */
static __isl_give isl_aff *isl_aff_set_nan(__isl_take isl_aff *aff)
{
	aff = isl_aff_cow(aff);
	if (!aff)
		return NULL;

	aff->v = isl_vec_clr(aff->v);
	if (!aff->v)
		return isl_aff_free(aff);

	return aff;
}

/* Return an affine expression defined on the specified domain
 * that represents NaN.
 */
__isl_give isl_aff *isl_aff_nan_on_domain(__isl_take isl_local_space *ls)
{
	isl_aff *aff;

	aff = isl_aff_alloc(ls);
	return isl_aff_set_nan(aff);
}

/* Return an affine expression defined on the specified domain space
 * that represents NaN.
 */
__isl_give isl_aff *isl_aff_nan_on_domain_space(__isl_take isl_space *space)
{
	return isl_aff_nan_on_domain(isl_local_space_from_space(space));
}

/* Return a piecewise affine expression defined on the specified domain space
 * that represents NaN.
 */
__isl_give isl_pw_aff *isl_pw_aff_nan_on_domain_space(
	__isl_take isl_space *space)
{
	return isl_pw_aff_from_aff(isl_aff_nan_on_domain_space(space));
}

/* Return a piecewise affine expression defined on the specified domain
 * that represents NaN.
 */
__isl_give isl_pw_aff *isl_pw_aff_nan_on_domain(__isl_take isl_local_space *ls)
{
	return isl_pw_aff_from_aff(isl_aff_nan_on_domain(ls));
}

/* Return an affine expression that is equal to "val" on
 * domain local space "ls".
 */
__isl_give isl_aff *isl_aff_val_on_domain(__isl_take isl_local_space *ls,
	__isl_take isl_val *val)
{
	isl_aff *aff;

	if (!ls || !val)
		goto error;
	if (!isl_val_is_rat(val))
		isl_die(isl_val_get_ctx(val), isl_error_invalid,
			"expecting rational value", goto error);

	aff = isl_aff_alloc(isl_local_space_copy(ls));
	if (!aff)
		goto error;

	isl_seq_clr(aff->v->el + 2, aff->v->size - 2);
	isl_int_set(aff->v->el[1], val->n);
	isl_int_set(aff->v->el[0], val->d);

	isl_local_space_free(ls);
	isl_val_free(val);
	return aff;
error:
	isl_local_space_free(ls);
	isl_val_free(val);
	return NULL;
}

/* Return an affine expression that is equal to "val" on domain space "space".
 */
__isl_give isl_aff *isl_aff_val_on_domain_space(__isl_take isl_space *space,
	__isl_take isl_val *val)
{
	return isl_aff_val_on_domain(isl_local_space_from_space(space), val);
}

/* Return an affine expression that is equal to the specified dimension
 * in "ls".
 */
__isl_give isl_aff *isl_aff_var_on_domain(__isl_take isl_local_space *ls,
	enum isl_dim_type type, unsigned pos)
{
	isl_space *space;
	isl_aff *aff;

	if (!ls)
		return NULL;

	space = isl_local_space_get_space(ls);
	if (!space)
		goto error;
	if (isl_space_is_map(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"expecting (parameter) set space", goto error);
	if (isl_local_space_check_range(ls, type, pos, 1) < 0)
		goto error;

	isl_space_free(space);
	aff = isl_aff_alloc(ls);
	if (!aff)
		return NULL;

	pos += isl_local_space_offset(aff->ls, type);

	isl_int_set_si(aff->v->el[0], 1);
	isl_seq_clr(aff->v->el + 1, aff->v->size - 1);
	isl_int_set_si(aff->v->el[1 + pos], 1);

	return aff;
error:
	isl_local_space_free(ls);
	isl_space_free(space);
	return NULL;
}

/* Return a piecewise affine expression that is equal to
 * the specified dimension in "ls".
 */
__isl_give isl_pw_aff *isl_pw_aff_var_on_domain(__isl_take isl_local_space *ls,
	enum isl_dim_type type, unsigned pos)
{
	return isl_pw_aff_from_aff(isl_aff_var_on_domain(ls, type, pos));
}

/* Return an affine expression that is equal to the parameter
 * in the domain space "space" with identifier "id".
 */
__isl_give isl_aff *isl_aff_param_on_domain_space_id(
	__isl_take isl_space *space, __isl_take isl_id *id)
{
	int pos;
	isl_local_space *ls;

	if (!space || !id)
		goto error;
	pos = isl_space_find_dim_by_id(space, isl_dim_param, id);
	if (pos < 0)
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"parameter not found in space", goto error);
	isl_id_free(id);
	ls = isl_local_space_from_space(space);
	return isl_aff_var_on_domain(ls, isl_dim_param, pos);
error:
	isl_space_free(space);
	isl_id_free(id);
	return NULL;
}

/* This function performs the same operation as
 * isl_aff_param_on_domain_space_id,
 * but is considered as a function on an isl_space when exported.
 */
__isl_give isl_aff *isl_space_param_aff_on_domain_id(
	__isl_take isl_space *space, __isl_take isl_id *id)
{
	return isl_aff_param_on_domain_space_id(space, id);
}

__isl_null isl_aff *isl_aff_free(__isl_take isl_aff *aff)
{
	if (!aff)
		return NULL;

	if (--aff->ref > 0)
		return NULL;

	isl_local_space_free(aff->ls);
	isl_vec_free(aff->v);

	free(aff);

	return NULL;
}

isl_ctx *isl_aff_get_ctx(__isl_keep isl_aff *aff)
{
	return aff ? isl_local_space_get_ctx(aff->ls) : NULL;
}

/* Return a hash value that digests "aff".
 */
uint32_t isl_aff_get_hash(__isl_keep isl_aff *aff)
{
	uint32_t hash, ls_hash, v_hash;

	if (!aff)
		return 0;

	hash = isl_hash_init();
	ls_hash = isl_local_space_get_hash(aff->ls);
	isl_hash_hash(hash, ls_hash);
	v_hash = isl_vec_get_hash(aff->v);
	isl_hash_hash(hash, v_hash);

	return hash;
}

/* Return the domain local space of "aff".
 */
static __isl_keep isl_local_space *isl_aff_peek_domain_local_space(
	__isl_keep isl_aff *aff)
{
	return aff ? aff->ls : NULL;
}

/* Return the number of variables of the given type in the domain of "aff".
 */
isl_size isl_aff_domain_dim(__isl_keep isl_aff *aff, enum isl_dim_type type)
{
	isl_local_space *ls;

	ls = isl_aff_peek_domain_local_space(aff);
	return isl_local_space_dim(ls, type);
}

/* Externally, an isl_aff has a map space, but internally, the
 * ls field corresponds to the domain of that space.
 */
isl_size isl_aff_dim(__isl_keep isl_aff *aff, enum isl_dim_type type)
{
	if (!aff)
		return isl_size_error;
	if (type == isl_dim_out)
		return 1;
	if (type == isl_dim_in)
		type = isl_dim_set;
	return isl_aff_domain_dim(aff, type);
}

/* Return the offset of the first coefficient of type "type" in
 * the domain of "aff".
 */
isl_size isl_aff_domain_offset(__isl_keep isl_aff *aff, enum isl_dim_type type)
{
	isl_local_space *ls;

	ls = isl_aff_peek_domain_local_space(aff);
	return isl_local_space_offset(ls, type);
}

/* Return the position of the dimension of the given type and name
 * in "aff".
 * Return -1 if no such dimension can be found.
 */
int isl_aff_find_dim_by_name(__isl_keep isl_aff *aff, enum isl_dim_type type,
	const char *name)
{
	if (!aff)
		return -1;
	if (type == isl_dim_out)
		return -1;
	if (type == isl_dim_in)
		type = isl_dim_set;
	return isl_local_space_find_dim_by_name(aff->ls, type, name);
}

/* Return the domain space of "aff".
 */
static __isl_keep isl_space *isl_aff_peek_domain_space(__isl_keep isl_aff *aff)
{
	return aff ? isl_local_space_peek_space(aff->ls) : NULL;
}

__isl_give isl_space *isl_aff_get_domain_space(__isl_keep isl_aff *aff)
{
	return isl_space_copy(isl_aff_peek_domain_space(aff));
}

__isl_give isl_space *isl_aff_get_space(__isl_keep isl_aff *aff)
{
	isl_space *space;
	if (!aff)
		return NULL;
	space = isl_local_space_get_space(aff->ls);
	space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_out, 1);
	return space;
}

/* Return a copy of the domain space of "aff".
 */
__isl_give isl_local_space *isl_aff_get_domain_local_space(
	__isl_keep isl_aff *aff)
{
	return isl_local_space_copy(isl_aff_peek_domain_local_space(aff));
}

__isl_give isl_local_space *isl_aff_get_local_space(__isl_keep isl_aff *aff)
{
	isl_local_space *ls;
	if (!aff)
		return NULL;
	ls = isl_local_space_copy(aff->ls);
	ls = isl_local_space_from_domain(ls);
	ls = isl_local_space_add_dims(ls, isl_dim_out, 1);
	return ls;
}

/* Return the local space of the domain of "aff".
 * This may be either a copy or the local space itself
 * if there is only one reference to "aff".
 * This allows the local space to be modified inplace
 * if both the expression and its local space have only a single reference.
 * The caller is not allowed to modify "aff" between this call and
 * a subsequent call to isl_aff_restore_domain_local_space.
 * The only exception is that isl_aff_free can be called instead.
 */
__isl_give isl_local_space *isl_aff_take_domain_local_space(
	__isl_keep isl_aff *aff)
{
	isl_local_space *ls;

	if (!aff)
		return NULL;
	if (aff->ref != 1)
		return isl_aff_get_domain_local_space(aff);
	ls = aff->ls;
	aff->ls = NULL;
	return ls;
}

/* Set the local space of the domain of "aff" to "ls",
 * where the local space of "aff" may be missing
 * due to a preceding call to isl_aff_take_domain_local_space.
 * However, in this case, "aff" only has a single reference and
 * then the call to isl_aff_cow has no effect.
 */
__isl_give isl_aff *isl_aff_restore_domain_local_space(
	__isl_keep isl_aff *aff, __isl_take isl_local_space *ls)
{
	if (!aff || !ls)
		goto error;

	if (aff->ls == ls) {
		isl_local_space_free(ls);
		return aff;
	}

	aff = isl_aff_cow(aff);
	if (!aff)
		goto error;
	isl_local_space_free(aff->ls);
	aff->ls = ls;

	return aff;
error:
	isl_aff_free(aff);
	isl_local_space_free(ls);
	return NULL;
}

/* Externally, an isl_aff has a map space, but internally, the
 * ls field corresponds to the domain of that space.
 */
const char *isl_aff_get_dim_name(__isl_keep isl_aff *aff,
	enum isl_dim_type type, unsigned pos)
{
	if (!aff)
		return NULL;
	if (type == isl_dim_out)
		return NULL;
	if (type == isl_dim_in)
		type = isl_dim_set;
	return isl_local_space_get_dim_name(aff->ls, type, pos);
}

__isl_give isl_aff *isl_aff_reset_domain_space(__isl_take isl_aff *aff,
	__isl_take isl_space *space)
{
	aff = isl_aff_cow(aff);
	if (!aff || !space)
		goto error;

	aff->ls = isl_local_space_reset_space(aff->ls, space);
	if (!aff->ls)
		return isl_aff_free(aff);

	return aff;
error:
	isl_aff_free(aff);
	isl_space_free(space);
	return NULL;
}

/* Reset the space of "aff".  This function is called from isl_pw_templ.c
 * and doesn't know if the space of an element object is represented
 * directly or through its domain.  It therefore passes along both.
 */
__isl_give isl_aff *isl_aff_reset_space_and_domain(__isl_take isl_aff *aff,
	__isl_take isl_space *space, __isl_take isl_space *domain)
{
	isl_space_free(space);
	return isl_aff_reset_domain_space(aff, domain);
}

/* Reorder the coefficients of the affine expression based
 * on the given reordering.
 * The reordering r is assumed to have been extended with the local
 * variables.
 */
static __isl_give isl_vec *vec_reorder(__isl_take isl_vec *vec,
	__isl_take isl_reordering *r, int n_div)
{
	isl_space *space;
	isl_vec *res;
	isl_size dim;
	int i;

	if (!vec || !r)
		goto error;

	space = isl_reordering_peek_space(r);
	dim = isl_space_dim(space, isl_dim_all);
	if (dim < 0)
		goto error;
	res = isl_vec_alloc(vec->ctx, 2 + dim + n_div);
	if (!res)
		goto error;
	isl_seq_cpy(res->el, vec->el, 2);
	isl_seq_clr(res->el + 2, res->size - 2);
	for (i = 0; i < r->len; ++i)
		isl_int_set(res->el[2 + r->pos[i]], vec->el[2 + i]);

	isl_reordering_free(r);
	isl_vec_free(vec);
	return res;
error:
	isl_vec_free(vec);
	isl_reordering_free(r);
	return NULL;
}

/* Reorder the dimensions of the domain of "aff" according
 * to the given reordering.
 */
__isl_give isl_aff *isl_aff_realign_domain(__isl_take isl_aff *aff,
	__isl_take isl_reordering *r)
{
	aff = isl_aff_cow(aff);
	if (!aff)
		goto error;

	r = isl_reordering_extend(r, aff->ls->div->n_row);
	aff->v = vec_reorder(aff->v, isl_reordering_copy(r),
				aff->ls->div->n_row);
	aff->ls = isl_local_space_realign(aff->ls, r);

	if (!aff->v || !aff->ls)
		return isl_aff_free(aff);

	return aff;
error:
	isl_aff_free(aff);
	isl_reordering_free(r);
	return NULL;
}

__isl_give isl_aff *isl_aff_align_params(__isl_take isl_aff *aff,
	__isl_take isl_space *model)
{
	isl_bool equal_params;

	if (!aff || !model)
		goto error;

	equal_params = isl_space_has_equal_params(aff->ls->dim, model);
	if (equal_params < 0)
		goto error;
	if (!equal_params) {
		isl_reordering *exp;

		exp = isl_parameter_alignment_reordering(aff->ls->dim, model);
		exp = isl_reordering_extend_space(exp,
					isl_aff_get_domain_space(aff));
		aff = isl_aff_realign_domain(aff, exp);
	}

	isl_space_free(model);
	return aff;
error:
	isl_space_free(model);
	isl_aff_free(aff);
	return NULL;
}

#undef TYPE
#define TYPE isl_aff
#include "isl_unbind_params_templ.c"

/* Is "aff" obviously equal to zero?
 *
 * If the denominator is zero, then "aff" is not equal to zero.
 */
isl_bool isl_aff_plain_is_zero(__isl_keep isl_aff *aff)
{
	int pos;

	if (!aff)
		return isl_bool_error;

	if (isl_int_is_zero(aff->v->el[0]))
		return isl_bool_false;
	pos = isl_seq_first_non_zero(aff->v->el + 1, aff->v->size - 1);
	return isl_bool_ok(pos < 0);
}

/* Does "aff" represent NaN?
 */
isl_bool isl_aff_is_nan(__isl_keep isl_aff *aff)
{
	if (!aff)
		return isl_bool_error;

	return isl_bool_ok(isl_seq_first_non_zero(aff->v->el, 2) < 0);
}

/* Are "aff1" and "aff2" obviously equal?
 *
 * NaN is not equal to anything, not even to another NaN.
 */
isl_bool isl_aff_plain_is_equal(__isl_keep isl_aff *aff1,
	__isl_keep isl_aff *aff2)
{
	isl_bool equal;

	if (!aff1 || !aff2)
		return isl_bool_error;

	if (isl_aff_is_nan(aff1) || isl_aff_is_nan(aff2))
		return isl_bool_false;

	equal = isl_local_space_is_equal(aff1->ls, aff2->ls);
	if (equal < 0 || !equal)
		return equal;

	return isl_vec_is_equal(aff1->v, aff2->v);
}

/* Return the common denominator of "aff" in "v".
 *
 * We cannot return anything meaningful in case of a NaN.
 */
isl_stat isl_aff_get_denominator(__isl_keep isl_aff *aff, isl_int *v)
{
	if (!aff)
		return isl_stat_error;
	if (isl_aff_is_nan(aff))
		isl_die(isl_aff_get_ctx(aff), isl_error_invalid,
			"cannot get denominator of NaN", return isl_stat_error);
	isl_int_set(*v, aff->v->el[0]);
	return isl_stat_ok;
}

/* Return the common denominator of "aff".
 */
__isl_give isl_val *isl_aff_get_denominator_val(__isl_keep isl_aff *aff)
{
	isl_ctx *ctx;

	if (!aff)
		return NULL;

	ctx = isl_aff_get_ctx(aff);
	if (isl_aff_is_nan(aff))
		return isl_val_nan(ctx);
	return isl_val_int_from_isl_int(ctx, aff->v->el[0]);
}

/* Return the constant term of "aff".
 */
__isl_give isl_val *isl_aff_get_constant_val(__isl_keep isl_aff *aff)
{
	isl_ctx *ctx;
	isl_val *v;

	if (!aff)
		return NULL;

	ctx = isl_aff_get_ctx(aff);
	if (isl_aff_is_nan(aff))
		return isl_val_nan(ctx);
	v = isl_val_rat_from_isl_int(ctx, aff->v->el[1], aff->v->el[0]);
	return isl_val_normalize(v);
}

/* Return the coefficient of the variable of type "type" at position "pos"
 * of "aff".
 */
__isl_give isl_val *isl_aff_get_coefficient_val(__isl_keep isl_aff *aff,
	enum isl_dim_type type, int pos)
{
	isl_ctx *ctx;
	isl_val *v;

	if (!aff)
		return NULL;

	ctx = isl_aff_get_ctx(aff);
	if (type == isl_dim_out)
		isl_die(ctx, isl_error_invalid,
			"output/set dimension does not have a coefficient",
			return NULL);
	if (type == isl_dim_in)
		type = isl_dim_set;

	if (isl_local_space_check_range(aff->ls, type, pos, 1) < 0)
		return NULL;

	if (isl_aff_is_nan(aff))
		return isl_val_nan(ctx);
	pos += isl_local_space_offset(aff->ls, type);
	v = isl_val_rat_from_isl_int(ctx, aff->v->el[1 + pos], aff->v->el[0]);
	return isl_val_normalize(v);
}

/* Return the sign of the coefficient of the variable of type "type"
 * at position "pos" of "aff".
 */
int isl_aff_coefficient_sgn(__isl_keep isl_aff *aff, enum isl_dim_type type,
	int pos)
{
	isl_ctx *ctx;

	if (!aff)
		return 0;

	ctx = isl_aff_get_ctx(aff);
	if (type == isl_dim_out)
		isl_die(ctx, isl_error_invalid,
			"output/set dimension does not have a coefficient",
			return 0);
	if (type == isl_dim_in)
		type = isl_dim_set;

	if (isl_local_space_check_range(aff->ls, type, pos, 1) < 0)
		return 0;

	pos += isl_local_space_offset(aff->ls, type);
	return isl_int_sgn(aff->v->el[1 + pos]);
}

/* Replace the numerator of the constant term of "aff" by "v".
 *
 * A NaN is unaffected by this operation.
 */
__isl_give isl_aff *isl_aff_set_constant(__isl_take isl_aff *aff, isl_int v)
{
	if (!aff)
		return NULL;
	if (isl_aff_is_nan(aff))
		return aff;
	aff = isl_aff_cow(aff);
	if (!aff)
		return NULL;

	aff->v = isl_vec_cow(aff->v);
	if (!aff->v)
		return isl_aff_free(aff);

	isl_int_set(aff->v->el[1], v);

	return aff;
}

/* Replace the constant term of "aff" by "v".
 *
 * A NaN is unaffected by this operation.
 */
__isl_give isl_aff *isl_aff_set_constant_val(__isl_take isl_aff *aff,
	__isl_take isl_val *v)
{
	if (!aff || !v)
		goto error;

	if (isl_aff_is_nan(aff)) {
		isl_val_free(v);
		return aff;
	}

	if (!isl_val_is_rat(v))
		isl_die(isl_aff_get_ctx(aff), isl_error_invalid,
			"expecting rational value", goto error);

	if (isl_int_eq(aff->v->el[1], v->n) &&
	    isl_int_eq(aff->v->el[0], v->d)) {
		isl_val_free(v);
		return aff;
	}

	aff = isl_aff_cow(aff);
	if (!aff)
		goto error;
	aff->v = isl_vec_cow(aff->v);
	if (!aff->v)
		goto error;

	if (isl_int_eq(aff->v->el[0], v->d)) {
		isl_int_set(aff->v->el[1], v->n);
	} else if (isl_int_is_one(v->d)) {
		isl_int_mul(aff->v->el[1], aff->v->el[0], v->n);
	} else {
		isl_seq_scale(aff->v->el + 1,
				aff->v->el + 1, v->d, aff->v->size - 1);
		isl_int_mul(aff->v->el[1], aff->v->el[0], v->n);
		isl_int_mul(aff->v->el[0], aff->v->el[0], v->d);
		aff->v = isl_vec_normalize(aff->v);
		if (!aff->v)
			goto error;
	}

	isl_val_free(v);
	return aff;
error:
	isl_aff_free(aff);
	isl_val_free(v);
	return NULL;
}

/* Add "v" to the constant term of "aff".
 *
 * A NaN is unaffected by this operation.
 */
__isl_give isl_aff *isl_aff_add_constant(__isl_take isl_aff *aff, isl_int v)
{
	if (isl_int_is_zero(v))
		return aff;

	if (!aff)
		return NULL;
	if (isl_aff_is_nan(aff))
		return aff;
	aff = isl_aff_cow(aff);
	if (!aff)
		return NULL;

	aff->v = isl_vec_cow(aff->v);
	if (!aff->v)
		return isl_aff_free(aff);

	isl_int_addmul(aff->v->el[1], aff->v->el[0], v);

	return aff;
}

/* Add "v" to the constant term of "aff",
 * in case "aff" is a rational expression.
 */
static __isl_give isl_aff *isl_aff_add_rat_constant_val(__isl_take isl_aff *aff,
	__isl_take isl_val *v)
{
	aff = isl_aff_cow(aff);
	if (!aff)
		goto error;

	aff->v = isl_vec_cow(aff->v);
	if (!aff->v)
		goto error;

	if (isl_int_is_one(v->d)) {
		isl_int_addmul(aff->v->el[1], aff->v->el[0], v->n);
	} else if (isl_int_eq(aff->v->el[0], v->d)) {
		isl_int_add(aff->v->el[1], aff->v->el[1], v->n);
		aff->v = isl_vec_normalize(aff->v);
		if (!aff->v)
			goto error;
	} else {
		isl_seq_scale(aff->v->el + 1,
				aff->v->el + 1, v->d, aff->v->size - 1);
		isl_int_addmul(aff->v->el[1], aff->v->el[0], v->n);
		isl_int_mul(aff->v->el[0], aff->v->el[0], v->d);
		aff->v = isl_vec_normalize(aff->v);
		if (!aff->v)
			goto error;
	}

	isl_val_free(v);
	return aff;
error:
	isl_aff_free(aff);
	isl_val_free(v);
	return NULL;
}

/* Return the first argument and free the second.
 */
static __isl_give isl_aff *pick_free(__isl_take isl_aff *aff,
	__isl_take isl_val *v)
{
	isl_val_free(v);
	return aff;
}

/* Replace the first argument by NaN and free the second argument.
 */
static __isl_give isl_aff *set_nan_free_val(__isl_take isl_aff *aff,
	__isl_take isl_val *v)
{
	isl_val_free(v);
	return isl_aff_set_nan(aff);
}

/* Add "v" to the constant term of "aff".
 *
 * A NaN is unaffected by this operation.
 * Conversely, adding a NaN turns "aff" into a NaN.
 */
__isl_give isl_aff *isl_aff_add_constant_val(__isl_take isl_aff *aff,
	__isl_take isl_val *v)
{
	isl_bool is_nan, is_zero, is_rat;

	is_nan = isl_aff_is_nan(aff);
	is_zero = isl_val_is_zero(v);
	if (is_nan < 0 || is_zero < 0)
		goto error;
	if (is_nan || is_zero)
		return pick_free(aff, v);

	is_nan = isl_val_is_nan(v);
	is_rat = isl_val_is_rat(v);
	if (is_nan < 0 || is_rat < 0)
		goto error;
	if (is_nan)
		return set_nan_free_val(aff, v);
	if (!is_rat)
		isl_die(isl_aff_get_ctx(aff), isl_error_invalid,
			"expecting rational value or NaN", goto error);

	return isl_aff_add_rat_constant_val(aff, v);
error:
	isl_aff_free(aff);
	isl_val_free(v);
	return NULL;
}

__isl_give isl_aff *isl_aff_add_constant_si(__isl_take isl_aff *aff, int v)
{
	isl_int t;

	isl_int_init(t);
	isl_int_set_si(t, v);
	aff = isl_aff_add_constant(aff, t);
	isl_int_clear(t);

	return aff;
}

/* Add "v" to the numerator of the constant term of "aff".
 *
 * A NaN is unaffected by this operation.
 */
__isl_give isl_aff *isl_aff_add_constant_num(__isl_take isl_aff *aff, isl_int v)
{
	if (isl_int_is_zero(v))
		return aff;

	if (!aff)
		return NULL;
	if (isl_aff_is_nan(aff))
		return aff;
	aff = isl_aff_cow(aff);
	if (!aff)
		return NULL;

	aff->v = isl_vec_cow(aff->v);
	if (!aff->v)
		return isl_aff_free(aff);

	isl_int_add(aff->v->el[1], aff->v->el[1], v);

	return aff;
}

/* Add "v" to the numerator of the constant term of "aff".
 *
 * A NaN is unaffected by this operation.
 */
__isl_give isl_aff *isl_aff_add_constant_num_si(__isl_take isl_aff *aff, int v)
{
	isl_int t;

	if (v == 0)
		return aff;

	isl_int_init(t);
	isl_int_set_si(t, v);
	aff = isl_aff_add_constant_num(aff, t);
	isl_int_clear(t);

	return aff;
}

/* Replace the numerator of the constant term of "aff" by "v".
 *
 * A NaN is unaffected by this operation.
 */
__isl_give isl_aff *isl_aff_set_constant_si(__isl_take isl_aff *aff, int v)
{
	if (!aff)
		return NULL;
	if (isl_aff_is_nan(aff))
		return aff;
	aff = isl_aff_cow(aff);
	if (!aff)
		return NULL;

	aff->v = isl_vec_cow(aff->v);
	if (!aff->v)
		return isl_aff_free(aff);

	isl_int_set_si(aff->v->el[1], v);

	return aff;
}

/* Replace the numerator of the coefficient of the variable of type "type"
 * at position "pos" of "aff" by "v".
 *
 * A NaN is unaffected by this operation.
 */
__isl_give isl_aff *isl_aff_set_coefficient(__isl_take isl_aff *aff,
	enum isl_dim_type type, int pos, isl_int v)
{
	if (!aff)
		return NULL;

	if (type == isl_dim_out)
		isl_die(aff->v->ctx, isl_error_invalid,
			"output/set dimension does not have a coefficient",
			return isl_aff_free(aff));
	if (type == isl_dim_in)
		type = isl_dim_set;

	if (isl_local_space_check_range(aff->ls, type, pos, 1) < 0)
		return isl_aff_free(aff);

	if (isl_aff_is_nan(aff))
		return aff;
	aff = isl_aff_cow(aff);
	if (!aff)
		return NULL;

	aff->v = isl_vec_cow(aff->v);
	if (!aff->v)
		return isl_aff_free(aff);

	pos += isl_local_space_offset(aff->ls, type);
	isl_int_set(aff->v->el[1 + pos], v);

	return aff;
}

/* Replace the numerator of the coefficient of the variable of type "type"
 * at position "pos" of "aff" by "v".
 *
 * A NaN is unaffected by this operation.
 */
__isl_give isl_aff *isl_aff_set_coefficient_si(__isl_take isl_aff *aff,
	enum isl_dim_type type, int pos, int v)
{
	if (!aff)
		return NULL;

	if (type == isl_dim_out)
		isl_die(aff->v->ctx, isl_error_invalid,
			"output/set dimension does not have a coefficient",
			return isl_aff_free(aff));
	if (type == isl_dim_in)
		type = isl_dim_set;

	if (isl_local_space_check_range(aff->ls, type, pos, 1) < 0)
		return isl_aff_free(aff);

	if (isl_aff_is_nan(aff))
		return aff;
	pos += isl_local_space_offset(aff->ls, type);
	if (isl_int_cmp_si(aff->v->el[1 + pos], v) == 0)
		return aff;

	aff = isl_aff_cow(aff);
	if (!aff)
		return NULL;

	aff->v = isl_vec_cow(aff->v);
	if (!aff->v)
		return isl_aff_free(aff);

	isl_int_set_si(aff->v->el[1 + pos], v);

	return aff;
}

/* Replace the coefficient of the variable of type "type" at position "pos"
 * of "aff" by "v".
 *
 * A NaN is unaffected by this operation.
 */
__isl_give isl_aff *isl_aff_set_coefficient_val(__isl_take isl_aff *aff,
	enum isl_dim_type type, int pos, __isl_take isl_val *v)
{
	if (!aff || !v)
		goto error;

	if (type == isl_dim_out)
		isl_die(aff->v->ctx, isl_error_invalid,
			"output/set dimension does not have a coefficient",
			goto error);
	if (type == isl_dim_in)
		type = isl_dim_set;

	if (isl_local_space_check_range(aff->ls, type, pos, 1) < 0)
		return isl_aff_free(aff);

	if (isl_aff_is_nan(aff)) {
		isl_val_free(v);
		return aff;
	}
	if (!isl_val_is_rat(v))
		isl_die(isl_aff_get_ctx(aff), isl_error_invalid,
			"expecting rational value", goto error);

	pos += isl_local_space_offset(aff->ls, type);
	if (isl_int_eq(aff->v->el[1 + pos], v->n) &&
	    isl_int_eq(aff->v->el[0], v->d)) {
		isl_val_free(v);
		return aff;
	}

	aff = isl_aff_cow(aff);
	if (!aff)
		goto error;
	aff->v = isl_vec_cow(aff->v);
	if (!aff->v)
		goto error;

	if (isl_int_eq(aff->v->el[0], v->d)) {
		isl_int_set(aff->v->el[1 + pos], v->n);
	} else if (isl_int_is_one(v->d)) {
		isl_int_mul(aff->v->el[1 + pos], aff->v->el[0], v->n);
	} else {
		isl_seq_scale(aff->v->el + 1,
				aff->v->el + 1, v->d, aff->v->size - 1);
		isl_int_mul(aff->v->el[1 + pos], aff->v->el[0], v->n);
		isl_int_mul(aff->v->el[0], aff->v->el[0], v->d);
		aff->v = isl_vec_normalize(aff->v);
		if (!aff->v)
			goto error;
	}

	isl_val_free(v);
	return aff;
error:
	isl_aff_free(aff);
	isl_val_free(v);
	return NULL;
}

/* Add "v" to the coefficient of the variable of type "type"
 * at position "pos" of "aff".
 *
 * A NaN is unaffected by this operation.
 */
__isl_give isl_aff *isl_aff_add_coefficient(__isl_take isl_aff *aff,
	enum isl_dim_type type, int pos, isl_int v)
{
	if (!aff)
		return NULL;

	if (type == isl_dim_out)
		isl_die(aff->v->ctx, isl_error_invalid,
			"output/set dimension does not have a coefficient",
			return isl_aff_free(aff));
	if (type == isl_dim_in)
		type = isl_dim_set;

	if (isl_local_space_check_range(aff->ls, type, pos, 1) < 0)
		return isl_aff_free(aff);

	if (isl_aff_is_nan(aff))
		return aff;
	aff = isl_aff_cow(aff);
	if (!aff)
		return NULL;

	aff->v = isl_vec_cow(aff->v);
	if (!aff->v)
		return isl_aff_free(aff);

	pos += isl_local_space_offset(aff->ls, type);
	isl_int_addmul(aff->v->el[1 + pos], aff->v->el[0], v);

	return aff;
}

/* Add "v" to the coefficient of the variable of type "type"
 * at position "pos" of "aff".
 *
 * A NaN is unaffected by this operation.
 */
__isl_give isl_aff *isl_aff_add_coefficient_val(__isl_take isl_aff *aff,
	enum isl_dim_type type, int pos, __isl_take isl_val *v)
{
	if (!aff || !v)
		goto error;

	if (isl_val_is_zero(v)) {
		isl_val_free(v);
		return aff;
	}

	if (type == isl_dim_out)
		isl_die(aff->v->ctx, isl_error_invalid,
			"output/set dimension does not have a coefficient",
			goto error);
	if (type == isl_dim_in)
		type = isl_dim_set;

	if (isl_local_space_check_range(aff->ls, type, pos, 1) < 0)
		goto error;

	if (isl_aff_is_nan(aff)) {
		isl_val_free(v);
		return aff;
	}
	if (!isl_val_is_rat(v))
		isl_die(isl_aff_get_ctx(aff), isl_error_invalid,
			"expecting rational value", goto error);

	aff = isl_aff_cow(aff);
	if (!aff)
		goto error;

	aff->v = isl_vec_cow(aff->v);
	if (!aff->v)
		goto error;

	pos += isl_local_space_offset(aff->ls, type);
	if (isl_int_is_one(v->d)) {
		isl_int_addmul(aff->v->el[1 + pos], aff->v->el[0], v->n);
	} else if (isl_int_eq(aff->v->el[0], v->d)) {
		isl_int_add(aff->v->el[1 + pos], aff->v->el[1 + pos], v->n);
		aff->v = isl_vec_normalize(aff->v);
		if (!aff->v)
			goto error;
	} else {
		isl_seq_scale(aff->v->el + 1,
				aff->v->el + 1, v->d, aff->v->size - 1);
		isl_int_addmul(aff->v->el[1 + pos], aff->v->el[0], v->n);
		isl_int_mul(aff->v->el[0], aff->v->el[0], v->d);
		aff->v = isl_vec_normalize(aff->v);
		if (!aff->v)
			goto error;
	}

	isl_val_free(v);
	return aff;
error:
	isl_aff_free(aff);
	isl_val_free(v);
	return NULL;
}

__isl_give isl_aff *isl_aff_add_coefficient_si(__isl_take isl_aff *aff,
	enum isl_dim_type type, int pos, int v)
{
	isl_int t;

	isl_int_init(t);
	isl_int_set_si(t, v);
	aff = isl_aff_add_coefficient(aff, type, pos, t);
	isl_int_clear(t);

	return aff;
}

__isl_give isl_aff *isl_aff_get_div(__isl_keep isl_aff *aff, int pos)
{
	if (!aff)
		return NULL;

	return isl_local_space_get_div(aff->ls, pos);
}

/* Return the negation of "aff".
 *
 * As a special case, -NaN = NaN.
 */
__isl_give isl_aff *isl_aff_neg(__isl_take isl_aff *aff)
{
	if (!aff)
		return NULL;
	if (isl_aff_is_nan(aff))
		return aff;
	aff = isl_aff_cow(aff);
	if (!aff)
		return NULL;
	aff->v = isl_vec_cow(aff->v);
	if (!aff->v)
		return isl_aff_free(aff);

	isl_seq_neg(aff->v->el + 1, aff->v->el + 1, aff->v->size - 1);

	return aff;
}

/* Remove divs from the local space that do not appear in the affine
 * expression.
 * We currently only remove divs at the end.
 * Some intermediate divs may also not appear directly in the affine
 * expression, but we would also need to check that no other divs are
 * defined in terms of them.
 */
__isl_give isl_aff *isl_aff_remove_unused_divs(__isl_take isl_aff *aff)
{
	int pos;
	isl_size off;
	isl_size n;

	n = isl_aff_domain_dim(aff, isl_dim_div);
	off = isl_aff_domain_offset(aff, isl_dim_div);
	if (n < 0 || off < 0)
		return isl_aff_free(aff);

	pos = isl_seq_last_non_zero(aff->v->el + 1 + off, n) + 1;
	if (pos == n)
		return aff;

	aff = isl_aff_cow(aff);
	if (!aff)
		return NULL;

	aff->ls = isl_local_space_drop_dims(aff->ls, isl_dim_div, pos, n - pos);
	aff->v = isl_vec_drop_els(aff->v, 1 + off + pos, n - pos);
	if (!aff->ls || !aff->v)
		return isl_aff_free(aff);

	return aff;
}

/* Look for any divs in the aff->ls with a denominator equal to one
 * and plug them into the affine expression and any subsequent divs
 * that may reference the div.
 */
static __isl_give isl_aff *plug_in_integral_divs(__isl_take isl_aff *aff)
{
	int i;
	isl_size n;
	int len;
	isl_int v;
	isl_vec *vec;
	isl_local_space *ls;
	isl_size off;

	n = isl_aff_domain_dim(aff, isl_dim_div);
	off = isl_aff_domain_offset(aff, isl_dim_div);
	if (n < 0 || off < 0)
		return isl_aff_free(aff);
	len = aff->v->size;
	for (i = 0; i < n; ++i) {
		if (!isl_int_is_one(aff->ls->div->row[i][0]))
			continue;
		ls = isl_local_space_copy(aff->ls);
		ls = isl_local_space_substitute_seq(ls, isl_dim_div, i,
				aff->ls->div->row[i], len, i + 1, n - (i + 1));
		vec = isl_vec_copy(aff->v);
		vec = isl_vec_cow(vec);
		if (!ls || !vec)
			goto error;

		isl_int_init(v);

		isl_seq_substitute(vec->el, off + i, aff->ls->div->row[i],
					len, len, v);

		isl_int_clear(v);

		isl_vec_free(aff->v);
		aff->v = vec;
		isl_local_space_free(aff->ls);
		aff->ls = ls;
	}

	return aff;
error:
	isl_vec_free(vec);
	isl_local_space_free(ls);
	return isl_aff_free(aff);
}

/* Look for any divs j that appear with a unit coefficient inside
 * the definitions of other divs i and plug them into the definitions
 * of the divs i.
 *
 * In particular, an expression of the form
 *
 *	floor((f(..) + floor(g(..)/n))/m)
 *
 * is simplified to
 *
 *	floor((n * f(..) + g(..))/(n * m))
 *
 * This simplification is correct because we can move the expression
 * f(..) into the inner floor in the original expression to obtain
 *
 *	floor(floor((n * f(..) + g(..))/n)/m)
 *
 * from which we can derive the simplified expression.
 */
static __isl_give isl_aff *plug_in_unit_divs(__isl_take isl_aff *aff)
{
	int i, j;
	isl_size n;
	isl_size off;

	n = isl_aff_domain_dim(aff, isl_dim_div);
	off = isl_aff_domain_offset(aff, isl_dim_div);
	if (n < 0 || off < 0)
		return isl_aff_free(aff);
	for (i = 1; i < n; ++i) {
		for (j = 0; j < i; ++j) {
			if (!isl_int_is_one(aff->ls->div->row[i][1 + off + j]))
				continue;
			aff->ls = isl_local_space_substitute_seq(aff->ls,
				isl_dim_div, j, aff->ls->div->row[j],
				aff->v->size, i, 1);
			if (!aff->ls)
				return isl_aff_free(aff);
		}
	}

	return aff;
}

/* Swap divs "a" and "b" in "aff", which is assumed to be non-NULL.
 *
 * Even though this function is only called on isl_affs with a single
 * reference, we are careful to only change aff->v and aff->ls together.
 */
static __isl_give isl_aff *swap_div(__isl_take isl_aff *aff, int a, int b)
{
	isl_size off = isl_aff_domain_offset(aff, isl_dim_div);
	isl_local_space *ls;
	isl_vec *v;

	if (off < 0)
		return isl_aff_free(aff);

	ls = isl_local_space_copy(aff->ls);
	ls = isl_local_space_swap_div(ls, a, b);
	v = isl_vec_copy(aff->v);
	v = isl_vec_cow(v);
	if (!ls || !v)
		goto error;

	isl_int_swap(v->el[1 + off + a], v->el[1 + off + b]);
	isl_vec_free(aff->v);
	aff->v = v;
	isl_local_space_free(aff->ls);
	aff->ls = ls;

	return aff;
error:
	isl_vec_free(v);
	isl_local_space_free(ls);
	return isl_aff_free(aff);
}

/* Merge divs "a" and "b" in "aff", which is assumed to be non-NULL.
 *
 * We currently do not actually remove div "b", but simply add its
 * coefficient to that of "a" and then zero it out.
 */
static __isl_give isl_aff *merge_divs(__isl_take isl_aff *aff, int a, int b)
{
	isl_size off = isl_aff_domain_offset(aff, isl_dim_div);

	if (off < 0)
		return isl_aff_free(aff);

	if (isl_int_is_zero(aff->v->el[1 + off + b]))
		return aff;

	aff->v = isl_vec_cow(aff->v);
	if (!aff->v)
		return isl_aff_free(aff);

	isl_int_add(aff->v->el[1 + off + a],
		    aff->v->el[1 + off + a], aff->v->el[1 + off + b]);
	isl_int_set_si(aff->v->el[1 + off + b], 0);

	return aff;
}

/* Sort the divs in the local space of "aff" according to
 * the comparison function "cmp_row" in isl_local_space.c,
 * combining the coefficients of identical divs.
 *
 * Reordering divs does not change the semantics of "aff",
 * so there is no need to call isl_aff_cow.
 * Moreover, this function is currently only called on isl_affs
 * with a single reference.
 */
static __isl_give isl_aff *sort_divs(__isl_take isl_aff *aff)
{
	isl_size n;
	int i, j;

	n = isl_aff_dim(aff, isl_dim_div);
	if (n < 0)
		return isl_aff_free(aff);
	for (i = 1; i < n; ++i) {
		for (j = i - 1; j >= 0; --j) {
			int cmp = isl_mat_cmp_div(aff->ls->div, j, j + 1);
			if (cmp < 0)
				break;
			if (cmp == 0)
				aff = merge_divs(aff, j, j + 1);
			else
				aff = swap_div(aff, j, j + 1);
			if (!aff)
				return NULL;
		}
	}

	return aff;
}

/* Normalize the representation of "aff".
 *
 * This function should only be called on "new" isl_affs, i.e.,
 * with only a single reference.  We therefore do not need to
 * worry about affecting other instances.
 */
__isl_give isl_aff *isl_aff_normalize(__isl_take isl_aff *aff)
{
	if (!aff)
		return NULL;
	aff->v = isl_vec_normalize(aff->v);
	if (!aff->v)
		return isl_aff_free(aff);
	aff = plug_in_integral_divs(aff);
	aff = plug_in_unit_divs(aff);
	aff = sort_divs(aff);
	aff = isl_aff_remove_unused_divs(aff);
	return aff;
}

/* Given f, return floor(f).
 * If f is an integer expression, then just return f.
 * If f is a constant, then return the constant floor(f).
 * Otherwise, if f = g/m, write g = q m + r,
 * create a new div d = [r/m] and return the expression q + d.
 * The coefficients in r are taken to lie between -m/2 and m/2.
 *
 * reduce_div_coefficients performs the same normalization.
 *
 * As a special case, floor(NaN) = NaN.
 */
__isl_give isl_aff *isl_aff_floor(__isl_take isl_aff *aff)
{
	int i;
	int size;
	isl_ctx *ctx;
	isl_vec *div;

	if (!aff)
		return NULL;

	if (isl_aff_is_nan(aff))
		return aff;
	if (isl_int_is_one(aff->v->el[0]))
		return aff;

	aff = isl_aff_cow(aff);
	if (!aff)
		return NULL;

	aff->v = isl_vec_cow(aff->v);
	if (!aff->v)
		return isl_aff_free(aff);

	if (isl_aff_is_cst(aff)) {
		isl_int_fdiv_q(aff->v->el[1], aff->v->el[1], aff->v->el[0]);
		isl_int_set_si(aff->v->el[0], 1);
		return aff;
	}

	div = isl_vec_copy(aff->v);
	div = isl_vec_cow(div);
	if (!div)
		return isl_aff_free(aff);

	ctx = isl_aff_get_ctx(aff);
	isl_int_fdiv_q(aff->v->el[0], aff->v->el[0], ctx->two);
	for (i = 1; i < aff->v->size; ++i) {
		isl_int_fdiv_r(div->el[i], div->el[i], div->el[0]);
		isl_int_fdiv_q(aff->v->el[i], aff->v->el[i], div->el[0]);
		if (isl_int_gt(div->el[i], aff->v->el[0])) {
			isl_int_sub(div->el[i], div->el[i], div->el[0]);
			isl_int_add_ui(aff->v->el[i], aff->v->el[i], 1);
		}
	}

	aff->ls = isl_local_space_add_div(aff->ls, div);
	if (!aff->ls)
		return isl_aff_free(aff);

	size = aff->v->size;
	aff->v = isl_vec_extend(aff->v, size + 1);
	if (!aff->v)
		return isl_aff_free(aff);
	isl_int_set_si(aff->v->el[0], 1);
	isl_int_set_si(aff->v->el[size], 1);

	aff = isl_aff_normalize(aff);

	return aff;
}

/* Compute
 *
 *	aff mod m = aff - m * floor(aff/m)
 *
 * with m an integer value.
 */
__isl_give isl_aff *isl_aff_mod_val(__isl_take isl_aff *aff,
	__isl_take isl_val *m)
{
	isl_aff *res;

	if (!aff || !m)
		goto error;

	if (!isl_val_is_int(m))
		isl_die(isl_val_get_ctx(m), isl_error_invalid,
			"expecting integer modulo", goto error);

	res = isl_aff_copy(aff);
	aff = isl_aff_scale_down_val(aff, isl_val_copy(m));
	aff = isl_aff_floor(aff);
	aff = isl_aff_scale_val(aff, m);
	res = isl_aff_sub(res, aff);

	return res;
error:
	isl_aff_free(aff);
	isl_val_free(m);
	return NULL;
}

/* Compute
 *
 *	pwaff mod m = pwaff - m * floor(pwaff/m)
 */
__isl_give isl_pw_aff *isl_pw_aff_mod(__isl_take isl_pw_aff *pwaff, isl_int m)
{
	isl_pw_aff *res;

	res = isl_pw_aff_copy(pwaff);
	pwaff = isl_pw_aff_scale_down(pwaff, m);
	pwaff = isl_pw_aff_floor(pwaff);
	pwaff = isl_pw_aff_scale(pwaff, m);
	res = isl_pw_aff_sub(res, pwaff);

	return res;
}

/* Compute
 *
 *	pa mod m = pa - m * floor(pa/m)
 *
 * with m an integer value.
 */
__isl_give isl_pw_aff *isl_pw_aff_mod_val(__isl_take isl_pw_aff *pa,
	__isl_take isl_val *m)
{
	if (!pa || !m)
		goto error;
	if (!isl_val_is_int(m))
		isl_die(isl_pw_aff_get_ctx(pa), isl_error_invalid,
			"expecting integer modulo", goto error);
	pa = isl_pw_aff_mod(pa, m->n);
	isl_val_free(m);
	return pa;
error:
	isl_pw_aff_free(pa);
	isl_val_free(m);
	return NULL;
}

/* Given f, return ceil(f).
 * If f is an integer expression, then just return f.
 * Otherwise, let f be the expression
 *
 *	e/m
 *
 * then return
 *
 *	floor((e + m - 1)/m)
 *
 * As a special case, ceil(NaN) = NaN.
 */
__isl_give isl_aff *isl_aff_ceil(__isl_take isl_aff *aff)
{
	if (!aff)
		return NULL;

	if (isl_aff_is_nan(aff))
		return aff;
	if (isl_int_is_one(aff->v->el[0]))
		return aff;

	aff = isl_aff_cow(aff);
	if (!aff)
		return NULL;
	aff->v = isl_vec_cow(aff->v);
	if (!aff->v)
		return isl_aff_free(aff);

	isl_int_add(aff->v->el[1], aff->v->el[1], aff->v->el[0]);
	isl_int_sub_ui(aff->v->el[1], aff->v->el[1], 1);
	aff = isl_aff_floor(aff);

	return aff;
}

/* Apply the expansion computed by isl_merge_divs.
 * The expansion itself is given by "exp" while the resulting
 * list of divs is given by "div".
 */
__isl_give isl_aff *isl_aff_expand_divs(__isl_take isl_aff *aff,
	__isl_take isl_mat *div, int *exp)
{
	isl_size old_n_div;
	isl_size new_n_div;
	isl_size offset;

	aff = isl_aff_cow(aff);

	offset = isl_aff_domain_offset(aff, isl_dim_div);
	old_n_div = isl_aff_domain_dim(aff, isl_dim_div);
	new_n_div = isl_mat_rows(div);
	if (offset < 0 || old_n_div < 0 || new_n_div < 0)
		goto error;

	aff->v = isl_vec_expand(aff->v, 1 + offset, old_n_div, exp, new_n_div);
	aff->ls = isl_local_space_replace_divs(aff->ls, div);
	if (!aff->v || !aff->ls)
		return isl_aff_free(aff);
	return aff;
error:
	isl_aff_free(aff);
	isl_mat_free(div);
	return NULL;
}

/* Add two affine expressions that live in the same local space.
 */
static __isl_give isl_aff *add_expanded(__isl_take isl_aff *aff1,
	__isl_take isl_aff *aff2)
{
	isl_int gcd, f;

	aff1 = isl_aff_cow(aff1);
	if (!aff1 || !aff2)
		goto error;

	aff1->v = isl_vec_cow(aff1->v);
	if (!aff1->v)
		goto error;

	isl_int_init(gcd);
	isl_int_init(f);
	isl_int_gcd(gcd, aff1->v->el[0], aff2->v->el[0]);
	isl_int_divexact(f, aff2->v->el[0], gcd);
	isl_seq_scale(aff1->v->el + 1, aff1->v->el + 1, f, aff1->v->size - 1);
	isl_int_divexact(f, aff1->v->el[0], gcd);
	isl_seq_addmul(aff1->v->el + 1, f, aff2->v->el + 1, aff1->v->size - 1);
	isl_int_divexact(f, aff2->v->el[0], gcd);
	isl_int_mul(aff1->v->el[0], aff1->v->el[0], f);
	isl_int_clear(f);
	isl_int_clear(gcd);

	isl_aff_free(aff2);
	aff1 = isl_aff_normalize(aff1);
	return aff1;
error:
	isl_aff_free(aff1);
	isl_aff_free(aff2);
	return NULL;
}

/* Replace one of the arguments by a NaN and free the other one.
 */
static __isl_give isl_aff *set_nan_free(__isl_take isl_aff *aff1,
	__isl_take isl_aff *aff2)
{
	isl_aff_free(aff2);
	return isl_aff_set_nan(aff1);
}

/* Return the sum of "aff1" and "aff2".
 *
 * If either of the two is NaN, then the result is NaN.
 */
__isl_give isl_aff *isl_aff_add(__isl_take isl_aff *aff1,
	__isl_take isl_aff *aff2)
{
	isl_ctx *ctx;
	int *exp1 = NULL;
	int *exp2 = NULL;
	isl_mat *div;
	isl_size n_div1, n_div2;

	if (!aff1 || !aff2)
		goto error;

	ctx = isl_aff_get_ctx(aff1);
	if (!isl_space_is_equal(aff1->ls->dim, aff2->ls->dim))
		isl_die(ctx, isl_error_invalid,
			"spaces don't match", goto error);

	if (isl_aff_is_nan(aff1)) {
		isl_aff_free(aff2);
		return aff1;
	}
	if (isl_aff_is_nan(aff2)) {
		isl_aff_free(aff1);
		return aff2;
	}

	n_div1 = isl_aff_dim(aff1, isl_dim_div);
	n_div2 = isl_aff_dim(aff2, isl_dim_div);
	if (n_div1 < 0 || n_div2 < 0)
		goto error;
	if (n_div1 == 0 && n_div2 == 0)
		return add_expanded(aff1, aff2);

	exp1 = isl_alloc_array(ctx, int, n_div1);
	exp2 = isl_alloc_array(ctx, int, n_div2);
	if ((n_div1 && !exp1) || (n_div2 && !exp2))
		goto error;

	div = isl_merge_divs(aff1->ls->div, aff2->ls->div, exp1, exp2);
	aff1 = isl_aff_expand_divs(aff1, isl_mat_copy(div), exp1);
	aff2 = isl_aff_expand_divs(aff2, div, exp2);
	free(exp1);
	free(exp2);

	return add_expanded(aff1, aff2);
error:
	free(exp1);
	free(exp2);
	isl_aff_free(aff1);
	isl_aff_free(aff2);
	return NULL;
}

__isl_give isl_aff *isl_aff_sub(__isl_take isl_aff *aff1,
	__isl_take isl_aff *aff2)
{
	return isl_aff_add(aff1, isl_aff_neg(aff2));
}

/* Return the result of scaling "aff" by a factor of "f".
 *
 * As a special case, f * NaN = NaN.
 */
__isl_give isl_aff *isl_aff_scale(__isl_take isl_aff *aff, isl_int f)
{
	isl_int gcd;

	if (!aff)
		return NULL;
	if (isl_aff_is_nan(aff))
		return aff;

	if (isl_int_is_one(f))
		return aff;

	aff = isl_aff_cow(aff);
	if (!aff)
		return NULL;
	aff->v = isl_vec_cow(aff->v);
	if (!aff->v)
		return isl_aff_free(aff);

	if (isl_int_is_pos(f) && isl_int_is_divisible_by(aff->v->el[0], f)) {
		isl_int_divexact(aff->v->el[0], aff->v->el[0], f);
		return aff;
	}

	isl_int_init(gcd);
	isl_int_gcd(gcd, aff->v->el[0], f);
	isl_int_divexact(aff->v->el[0], aff->v->el[0], gcd);
	isl_int_divexact(gcd, f, gcd);
	isl_seq_scale(aff->v->el + 1, aff->v->el + 1, gcd, aff->v->size - 1);
	isl_int_clear(gcd);

	return aff;
}

/* Multiple "aff" by "v".
 */
__isl_give isl_aff *isl_aff_scale_val(__isl_take isl_aff *aff,
	__isl_take isl_val *v)
{
	if (!aff || !v)
		goto error;

	if (isl_val_is_one(v)) {
		isl_val_free(v);
		return aff;
	}

	if (!isl_val_is_rat(v))
		isl_die(isl_aff_get_ctx(aff), isl_error_invalid,
			"expecting rational factor", goto error);

	aff = isl_aff_scale(aff, v->n);
	aff = isl_aff_scale_down(aff, v->d);

	isl_val_free(v);
	return aff;
error:
	isl_aff_free(aff);
	isl_val_free(v);
	return NULL;
}

/* Return the result of scaling "aff" down by a factor of "f".
 *
 * As a special case, NaN/f = NaN.
 */
__isl_give isl_aff *isl_aff_scale_down(__isl_take isl_aff *aff, isl_int f)
{
	isl_int gcd;

	if (!aff)
		return NULL;
	if (isl_aff_is_nan(aff))
		return aff;

	if (isl_int_is_one(f))
		return aff;

	aff = isl_aff_cow(aff);
	if (!aff)
		return NULL;

	if (isl_int_is_zero(f))
		isl_die(isl_aff_get_ctx(aff), isl_error_invalid,
			"cannot scale down by zero", return isl_aff_free(aff));

	aff->v = isl_vec_cow(aff->v);
	if (!aff->v)
		return isl_aff_free(aff);

	isl_int_init(gcd);
	isl_seq_gcd(aff->v->el + 1, aff->v->size - 1, &gcd);
	isl_int_gcd(gcd, gcd, f);
	isl_seq_scale_down(aff->v->el + 1, aff->v->el + 1, gcd, aff->v->size - 1);
	isl_int_divexact(gcd, f, gcd);
	isl_int_mul(aff->v->el[0], aff->v->el[0], gcd);
	isl_int_clear(gcd);

	return aff;
}

/* Divide "aff" by "v".
 */
__isl_give isl_aff *isl_aff_scale_down_val(__isl_take isl_aff *aff,
	__isl_take isl_val *v)
{
	if (!aff || !v)
		goto error;

	if (isl_val_is_one(v)) {
		isl_val_free(v);
		return aff;
	}

	if (!isl_val_is_rat(v))
		isl_die(isl_aff_get_ctx(aff), isl_error_invalid,
			"expecting rational factor", goto error);
	if (!isl_val_is_pos(v))
		isl_die(isl_aff_get_ctx(aff), isl_error_invalid,
			"factor needs to be positive", goto error);

	aff = isl_aff_scale(aff, v->d);
	aff = isl_aff_scale_down(aff, v->n);

	isl_val_free(v);
	return aff;
error:
	isl_aff_free(aff);
	isl_val_free(v);
	return NULL;
}

__isl_give isl_aff *isl_aff_scale_down_ui(__isl_take isl_aff *aff, unsigned f)
{
	isl_int v;

	if (f == 1)
		return aff;

	isl_int_init(v);
	isl_int_set_ui(v, f);
	aff = isl_aff_scale_down(aff, v);
	isl_int_clear(v);

	return aff;
}

__isl_give isl_aff *isl_aff_set_dim_name(__isl_take isl_aff *aff,
	enum isl_dim_type type, unsigned pos, const char *s)
{
	aff = isl_aff_cow(aff);
	if (!aff)
		return NULL;
	if (type == isl_dim_out)
		isl_die(aff->v->ctx, isl_error_invalid,
			"cannot set name of output/set dimension",
			return isl_aff_free(aff));
	if (type == isl_dim_in)
		type = isl_dim_set;
	aff->ls = isl_local_space_set_dim_name(aff->ls, type, pos, s);
	if (!aff->ls)
		return isl_aff_free(aff);

	return aff;
}

__isl_give isl_aff *isl_aff_set_dim_id(__isl_take isl_aff *aff,
	enum isl_dim_type type, unsigned pos, __isl_take isl_id *id)
{
	aff = isl_aff_cow(aff);
	if (!aff)
		goto error;
	if (type == isl_dim_out)
		isl_die(aff->v->ctx, isl_error_invalid,
			"cannot set name of output/set dimension",
			goto error);
	if (type == isl_dim_in)
		type = isl_dim_set;
	aff->ls = isl_local_space_set_dim_id(aff->ls, type, pos, id);
	if (!aff->ls)
		return isl_aff_free(aff);

	return aff;
error:
	isl_id_free(id);
	isl_aff_free(aff);
	return NULL;
}

/* Replace the identifier of the input tuple of "aff" by "id".
 * type is currently required to be equal to isl_dim_in
 */
__isl_give isl_aff *isl_aff_set_tuple_id(__isl_take isl_aff *aff,
	enum isl_dim_type type, __isl_take isl_id *id)
{
	aff = isl_aff_cow(aff);
	if (!aff)
		goto error;
	if (type != isl_dim_in)
		isl_die(aff->v->ctx, isl_error_invalid,
			"cannot only set id of input tuple", goto error);
	aff->ls = isl_local_space_set_tuple_id(aff->ls, isl_dim_set, id);
	if (!aff->ls)
		return isl_aff_free(aff);

	return aff;
error:
	isl_id_free(id);
	isl_aff_free(aff);
	return NULL;
}

/* Exploit the equalities in "eq" to simplify the affine expression
 * and the expressions of the integer divisions in the local space.
 * The integer divisions in this local space are assumed to appear
 * as regular dimensions in "eq".
 */
static __isl_give isl_aff *isl_aff_substitute_equalities_lifted(
	__isl_take isl_aff *aff, __isl_take isl_basic_set *eq)
{
	int i, j;
	unsigned o_div;
	unsigned n_div;

	if (!eq)
		goto error;
	if (eq->n_eq == 0) {
		isl_basic_set_free(eq);
		return aff;
	}

	aff = isl_aff_cow(aff);
	if (!aff)
		goto error;

	aff->ls = isl_local_space_substitute_equalities(aff->ls,
							isl_basic_set_copy(eq));
	aff->v = isl_vec_cow(aff->v);
	if (!aff->ls || !aff->v)
		goto error;

	o_div = isl_basic_set_offset(eq, isl_dim_div);
	n_div = eq->n_div;
	for (i = 0; i < eq->n_eq; ++i) {
		j = isl_seq_last_non_zero(eq->eq[i], o_div + n_div);
		if (j < 0 || j == 0 || j >= o_div)
			continue;

		isl_seq_elim(aff->v->el + 1, eq->eq[i], j, o_div,
				&aff->v->el[0]);
	}

	isl_basic_set_free(eq);
	aff = isl_aff_normalize(aff);
	return aff;
error:
	isl_basic_set_free(eq);
	isl_aff_free(aff);
	return NULL;
}

/* Exploit the equalities in "eq" to simplify the affine expression
 * and the expressions of the integer divisions in the local space.
 */
__isl_give isl_aff *isl_aff_substitute_equalities(__isl_take isl_aff *aff,
	__isl_take isl_basic_set *eq)
{
	isl_size n_div;

	n_div = isl_aff_domain_dim(aff, isl_dim_div);
	if (n_div < 0)
		goto error;
	if (n_div > 0)
		eq = isl_basic_set_add_dims(eq, isl_dim_set, n_div);
	return isl_aff_substitute_equalities_lifted(aff, eq);
error:
	isl_basic_set_free(eq);
	isl_aff_free(aff);
	return NULL;
}

/* Look for equalities among the variables shared by context and aff
 * and the integer divisions of aff, if any.
 * The equalities are then used to eliminate coefficients and/or integer
 * divisions from aff.
 */
__isl_give isl_aff *isl_aff_gist(__isl_take isl_aff *aff,
	__isl_take isl_set *context)
{
	isl_local_space *ls;
	isl_basic_set *hull;

	ls = isl_aff_get_domain_local_space(aff);
	context = isl_local_space_lift_set(ls, context);

	hull = isl_set_affine_hull(context);
	return isl_aff_substitute_equalities_lifted(aff, hull);
}

__isl_give isl_aff *isl_aff_gist_params(__isl_take isl_aff *aff,
	__isl_take isl_set *context)
{
	isl_set *dom_context = isl_set_universe(isl_aff_get_domain_space(aff));
	dom_context = isl_set_intersect_params(dom_context, context);
	return isl_aff_gist(aff, dom_context);
}

/* Return a basic set containing those elements in the space
 * of aff where it is positive.  "rational" should not be set.
 *
 * If "aff" is NaN, then it is not positive.
 */
static __isl_give isl_basic_set *aff_pos_basic_set(__isl_take isl_aff *aff,
	int rational, void *user)
{
	isl_constraint *ineq;
	isl_basic_set *bset;
	isl_val *c;

	if (!aff)
		return NULL;
	if (isl_aff_is_nan(aff)) {
		isl_space *space = isl_aff_get_domain_space(aff);
		isl_aff_free(aff);
		return isl_basic_set_empty(space);
	}
	if (rational)
		isl_die(isl_aff_get_ctx(aff), isl_error_unsupported,
			"rational sets not supported", goto error);

	ineq = isl_inequality_from_aff(aff);
	c = isl_constraint_get_constant_val(ineq);
	c = isl_val_sub_ui(c, 1);
	ineq = isl_constraint_set_constant_val(ineq, c);

	bset = isl_basic_set_from_constraint(ineq);
	bset = isl_basic_set_simplify(bset);
	return bset;
error:
	isl_aff_free(aff);
	return NULL;
}

/* Return a basic set containing those elements in the space
 * of aff where it is non-negative.
 * If "rational" is set, then return a rational basic set.
 *
 * If "aff" is NaN, then it is not non-negative (it's not negative either).
 */
static __isl_give isl_basic_set *aff_nonneg_basic_set(
	__isl_take isl_aff *aff, int rational, void *user)
{
	isl_constraint *ineq;
	isl_basic_set *bset;

	if (!aff)
		return NULL;
	if (isl_aff_is_nan(aff)) {
		isl_space *space = isl_aff_get_domain_space(aff);
		isl_aff_free(aff);
		return isl_basic_set_empty(space);
	}

	ineq = isl_inequality_from_aff(aff);

	bset = isl_basic_set_from_constraint(ineq);
	if (rational)
		bset = isl_basic_set_set_rational(bset);
	bset = isl_basic_set_simplify(bset);
	return bset;
}

/* Return a basic set containing those elements in the space
 * of aff where it is non-negative.
 */
__isl_give isl_basic_set *isl_aff_nonneg_basic_set(__isl_take isl_aff *aff)
{
	return aff_nonneg_basic_set(aff, 0, NULL);
}

/* Return a basic set containing those elements in the domain space
 * of "aff" where it is positive.
 */
__isl_give isl_basic_set *isl_aff_pos_basic_set(__isl_take isl_aff *aff)
{
	aff = isl_aff_add_constant_num_si(aff, -1);
	return isl_aff_nonneg_basic_set(aff);
}

/* Return a basic set containing those elements in the domain space
 * of aff where it is negative.
 */
__isl_give isl_basic_set *isl_aff_neg_basic_set(__isl_take isl_aff *aff)
{
	aff = isl_aff_neg(aff);
	return isl_aff_pos_basic_set(aff);
}

/* Return a basic set containing those elements in the space
 * of aff where it is zero.
 * If "rational" is set, then return a rational basic set.
 *
 * If "aff" is NaN, then it is not zero.
 */
static __isl_give isl_basic_set *aff_zero_basic_set(__isl_take isl_aff *aff,
	int rational, void *user)
{
	isl_constraint *ineq;
	isl_basic_set *bset;

	if (!aff)
		return NULL;
	if (isl_aff_is_nan(aff)) {
		isl_space *space = isl_aff_get_domain_space(aff);
		isl_aff_free(aff);
		return isl_basic_set_empty(space);
	}

	ineq = isl_equality_from_aff(aff);

	bset = isl_basic_set_from_constraint(ineq);
	if (rational)
		bset = isl_basic_set_set_rational(bset);
	bset = isl_basic_set_simplify(bset);
	return bset;
}

/* Return a basic set containing those elements in the space
 * of aff where it is zero.
 */
__isl_give isl_basic_set *isl_aff_zero_basic_set(__isl_take isl_aff *aff)
{
	return aff_zero_basic_set(aff, 0, NULL);
}

/* Return a basic set containing those elements in the shared space
 * of aff1 and aff2 where aff1 is greater than or equal to aff2.
 */
__isl_give isl_basic_set *isl_aff_ge_basic_set(__isl_take isl_aff *aff1,
	__isl_take isl_aff *aff2)
{
	aff1 = isl_aff_sub(aff1, aff2);

	return isl_aff_nonneg_basic_set(aff1);
}

/* Return a basic set containing those elements in the shared domain space
 * of "aff1" and "aff2" where "aff1" is greater than "aff2".
 */
__isl_give isl_basic_set *isl_aff_gt_basic_set(__isl_take isl_aff *aff1,
	__isl_take isl_aff *aff2)
{
	aff1 = isl_aff_sub(aff1, aff2);

	return isl_aff_pos_basic_set(aff1);
}

/* Return a set containing those elements in the shared space
 * of aff1 and aff2 where aff1 is greater than or equal to aff2.
 */
__isl_give isl_set *isl_aff_ge_set(__isl_take isl_aff *aff1,
	__isl_take isl_aff *aff2)
{
	return isl_set_from_basic_set(isl_aff_ge_basic_set(aff1, aff2));
}

/* Return a set containing those elements in the shared domain space
 * of aff1 and aff2 where aff1 is greater than aff2.
 *
 * If either of the two inputs is NaN, then the result is empty,
 * as comparisons with NaN always return false.
 */
__isl_give isl_set *isl_aff_gt_set(__isl_take isl_aff *aff1,
	__isl_take isl_aff *aff2)
{
	return isl_set_from_basic_set(isl_aff_gt_basic_set(aff1, aff2));
}

/* Return a basic set containing those elements in the shared space
 * of aff1 and aff2 where aff1 is smaller than or equal to aff2.
 */
__isl_give isl_basic_set *isl_aff_le_basic_set(__isl_take isl_aff *aff1,
	__isl_take isl_aff *aff2)
{
	return isl_aff_ge_basic_set(aff2, aff1);
}

/* Return a basic set containing those elements in the shared domain space
 * of "aff1" and "aff2" where "aff1" is smaller than "aff2".
 */
__isl_give isl_basic_set *isl_aff_lt_basic_set(__isl_take isl_aff *aff1,
	__isl_take isl_aff *aff2)
{
	return isl_aff_gt_basic_set(aff2, aff1);
}

/* Return a set containing those elements in the shared space
 * of aff1 and aff2 where aff1 is smaller than or equal to aff2.
 */
__isl_give isl_set *isl_aff_le_set(__isl_take isl_aff *aff1,
	__isl_take isl_aff *aff2)
{
	return isl_aff_ge_set(aff2, aff1);
}

/* Return a set containing those elements in the shared domain space
 * of "aff1" and "aff2" where "aff1" is smaller than "aff2".
 */
__isl_give isl_set *isl_aff_lt_set(__isl_take isl_aff *aff1,
	__isl_take isl_aff *aff2)
{
	return isl_set_from_basic_set(isl_aff_lt_basic_set(aff1, aff2));
}

/* Return a basic set containing those elements in the shared space
 * of aff1 and aff2 where aff1 and aff2 are equal.
 */
__isl_give isl_basic_set *isl_aff_eq_basic_set(__isl_take isl_aff *aff1,
	__isl_take isl_aff *aff2)
{
	aff1 = isl_aff_sub(aff1, aff2);

	return isl_aff_zero_basic_set(aff1);
}

/* Return a set containing those elements in the shared space
 * of aff1 and aff2 where aff1 and aff2 are equal.
 */
__isl_give isl_set *isl_aff_eq_set(__isl_take isl_aff *aff1,
	__isl_take isl_aff *aff2)
{
	return isl_set_from_basic_set(isl_aff_eq_basic_set(aff1, aff2));
}

/* Return a set containing those elements in the shared domain space
 * of aff1 and aff2 where aff1 and aff2 are not equal.
 *
 * If either of the two inputs is NaN, then the result is empty,
 * as comparisons with NaN always return false.
 */
__isl_give isl_set *isl_aff_ne_set(__isl_take isl_aff *aff1,
	__isl_take isl_aff *aff2)
{
	isl_set *set_lt, *set_gt;

	set_lt = isl_aff_lt_set(isl_aff_copy(aff1),
				isl_aff_copy(aff2));
	set_gt = isl_aff_gt_set(aff1, aff2);
	return isl_set_union_disjoint(set_lt, set_gt);
}

__isl_give isl_aff *isl_aff_add_on_domain(__isl_keep isl_set *dom,
	__isl_take isl_aff *aff1, __isl_take isl_aff *aff2)
{
	aff1 = isl_aff_add(aff1, aff2);
	aff1 = isl_aff_gist(aff1, isl_set_copy(dom));
	return aff1;
}

isl_bool isl_aff_is_empty(__isl_keep isl_aff *aff)
{
	if (!aff)
		return isl_bool_error;

	return isl_bool_false;
}

#undef TYPE
#define TYPE	isl_aff
static
#include "check_type_range_templ.c"

/* Check whether the given affine expression has non-zero coefficient
 * for any dimension in the given range or if any of these dimensions
 * appear with non-zero coefficients in any of the integer divisions
 * involved in the affine expression.
 */
isl_bool isl_aff_involves_dims(__isl_keep isl_aff *aff,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;
	int *active = NULL;
	isl_bool involves = isl_bool_false;

	if (!aff)
		return isl_bool_error;
	if (n == 0)
		return isl_bool_false;
	if (isl_aff_check_range(aff, type, first, n) < 0)
		return isl_bool_error;

	active = isl_local_space_get_active(aff->ls, aff->v->el + 2);
	if (!active)
		goto error;

	first += isl_local_space_offset(aff->ls, type) - 1;
	for (i = 0; i < n; ++i)
		if (active[first + i]) {
			involves = isl_bool_true;
			break;
		}

	free(active);

	return involves;
error:
	free(active);
	return isl_bool_error;
}

/* Does "aff" involve any local variables, i.e., integer divisions?
 */
isl_bool isl_aff_involves_locals(__isl_keep isl_aff *aff)
{
	isl_size n;

	n = isl_aff_dim(aff, isl_dim_div);
	if (n < 0)
		return isl_bool_error;
	return isl_bool_ok(n > 0);
}

__isl_give isl_aff *isl_aff_drop_dims(__isl_take isl_aff *aff,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	isl_ctx *ctx;

	if (!aff)
		return NULL;
	if (type == isl_dim_out)
		isl_die(aff->v->ctx, isl_error_invalid,
			"cannot drop output/set dimension",
			return isl_aff_free(aff));
	if (type == isl_dim_in)
		type = isl_dim_set;
	if (n == 0 && !isl_local_space_is_named_or_nested(aff->ls, type))
		return aff;

	ctx = isl_aff_get_ctx(aff);
	if (isl_local_space_check_range(aff->ls, type, first, n) < 0)
		return isl_aff_free(aff);

	aff = isl_aff_cow(aff);
	if (!aff)
		return NULL;

	aff->ls = isl_local_space_drop_dims(aff->ls, type, first, n);
	if (!aff->ls)
		return isl_aff_free(aff);

	first += 1 + isl_local_space_offset(aff->ls, type);
	aff->v = isl_vec_drop_els(aff->v, first, n);
	if (!aff->v)
		return isl_aff_free(aff);

	return aff;
}

/* Is the domain of "aff" a product?
 */
static isl_bool isl_aff_domain_is_product(__isl_keep isl_aff *aff)
{
	return isl_space_is_product(isl_aff_peek_domain_space(aff));
}

#undef TYPE
#define TYPE	isl_aff
#include <isl_domain_factor_templ.c>

/* Project the domain of the affine expression onto its parameter space.
 * The affine expression may not involve any of the domain dimensions.
 */
__isl_give isl_aff *isl_aff_project_domain_on_params(__isl_take isl_aff *aff)
{
	isl_space *space;
	isl_size n;

	n = isl_aff_dim(aff, isl_dim_in);
	if (n < 0)
		return isl_aff_free(aff);
	aff = isl_aff_drop_domain(aff, 0, n);
	space = isl_aff_get_domain_space(aff);
	space = isl_space_params(space);
	aff = isl_aff_reset_domain_space(aff, space);
	return aff;
}

/* Convert an affine expression defined over a parameter domain
 * into one that is defined over a zero-dimensional set.
 */
__isl_give isl_aff *isl_aff_from_range(__isl_take isl_aff *aff)
{
	isl_local_space *ls;

	ls = isl_aff_take_domain_local_space(aff);
	ls = isl_local_space_set_from_params(ls);
	aff = isl_aff_restore_domain_local_space(aff, ls);

	return aff;
}

__isl_give isl_aff *isl_aff_insert_dims(__isl_take isl_aff *aff,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	isl_ctx *ctx;

	if (!aff)
		return NULL;
	if (type == isl_dim_out)
		isl_die(aff->v->ctx, isl_error_invalid,
			"cannot insert output/set dimensions",
			return isl_aff_free(aff));
	if (type == isl_dim_in)
		type = isl_dim_set;
	if (n == 0 && !isl_local_space_is_named_or_nested(aff->ls, type))
		return aff;

	ctx = isl_aff_get_ctx(aff);
	if (isl_local_space_check_range(aff->ls, type, first, 0) < 0)
		return isl_aff_free(aff);

	aff = isl_aff_cow(aff);
	if (!aff)
		return NULL;

	aff->ls = isl_local_space_insert_dims(aff->ls, type, first, n);
	if (!aff->ls)
		return isl_aff_free(aff);

	first += 1 + isl_local_space_offset(aff->ls, type);
	aff->v = isl_vec_insert_zero_els(aff->v, first, n);
	if (!aff->v)
		return isl_aff_free(aff);

	return aff;
}

__isl_give isl_aff *isl_aff_add_dims(__isl_take isl_aff *aff,
	enum isl_dim_type type, unsigned n)
{
	isl_size pos;

	pos = isl_aff_dim(aff, type);
	if (pos < 0)
		return isl_aff_free(aff);

	return isl_aff_insert_dims(aff, type, pos, n);
}

/* Move the "n" dimensions of "src_type" starting at "src_pos" of "aff"
 * to dimensions of "dst_type" at "dst_pos".
 *
 * We only support moving input dimensions to parameters and vice versa.
 */
__isl_give isl_aff *isl_aff_move_dims(__isl_take isl_aff *aff,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n)
{
	unsigned g_dst_pos;
	unsigned g_src_pos;
	isl_size src_off, dst_off;

	if (!aff)
		return NULL;
	if (n == 0 &&
	    !isl_local_space_is_named_or_nested(aff->ls, src_type) &&
	    !isl_local_space_is_named_or_nested(aff->ls, dst_type))
		return aff;

	if (dst_type == isl_dim_out || src_type == isl_dim_out)
		isl_die(isl_aff_get_ctx(aff), isl_error_invalid,
			"cannot move output/set dimension",
			return isl_aff_free(aff));
	if (dst_type == isl_dim_div || src_type == isl_dim_div)
		isl_die(isl_aff_get_ctx(aff), isl_error_invalid,
			"cannot move divs", return isl_aff_free(aff));
	if (dst_type == isl_dim_in)
		dst_type = isl_dim_set;
	if (src_type == isl_dim_in)
		src_type = isl_dim_set;

	if (isl_local_space_check_range(aff->ls, src_type, src_pos, n) < 0)
		return isl_aff_free(aff);
	if (dst_type == src_type)
		isl_die(isl_aff_get_ctx(aff), isl_error_unsupported,
			"moving dims within the same type not supported",
			return isl_aff_free(aff));

	aff = isl_aff_cow(aff);
	src_off = isl_aff_domain_offset(aff, src_type);
	dst_off = isl_aff_domain_offset(aff, dst_type);
	if (src_off < 0 || dst_off < 0)
		return isl_aff_free(aff);

	g_src_pos = 1 + src_off + src_pos;
	g_dst_pos = 1 + dst_off + dst_pos;
	if (dst_type > src_type)
		g_dst_pos -= n;

	aff->v = isl_vec_move_els(aff->v, g_dst_pos, g_src_pos, n);
	aff->ls = isl_local_space_move_dims(aff->ls, dst_type, dst_pos,
						src_type, src_pos, n);
	if (!aff->v || !aff->ls)
		return isl_aff_free(aff);

	aff = sort_divs(aff);

	return aff;
}

/* Return a zero isl_aff in the given space.
 *
 * This is a helper function for isl_pw_*_as_* that ensures a uniform
 * interface over all piecewise types.
 */
static __isl_give isl_aff *isl_aff_zero_in_space(__isl_take isl_space *space)
{
	isl_local_space *ls;

	ls = isl_local_space_from_space(isl_space_domain(space));
	return isl_aff_zero_on_domain(ls);
}

#define isl_aff_involves_nan isl_aff_is_nan

#undef PW
#define PW isl_pw_aff
#undef BASE
#define BASE aff
#undef EL_IS_ZERO
#define EL_IS_ZERO is_empty
#undef ZERO
#define ZERO empty
#undef IS_ZERO
#define IS_ZERO is_empty
#undef FIELD
#define FIELD aff
#undef DEFAULT_IS_ZERO
#define DEFAULT_IS_ZERO 0

#include <isl_pw_templ.c>
#include <isl_pw_add_constant_val_templ.c>
#include <isl_pw_bind_domain_templ.c>
#include <isl_pw_eval.c>
#include <isl_pw_hash.c>
#include <isl_pw_insert_dims_templ.c>
#include <isl_pw_insert_domain_templ.c>
#include <isl_pw_move_dims_templ.c>
#include <isl_pw_neg_templ.c>
#include <isl_pw_pullback_templ.c>
#include <isl_pw_sub_templ.c>
#include <isl_pw_union_opt.c>

#undef BASE
#define BASE pw_aff

#include <isl_union_single.c>
#include <isl_union_neg.c>

#undef BASE
#define BASE aff

#include <isl_union_pw_templ.c>

/* Compute a piecewise quasi-affine expression with a domain that
 * is the union of those of pwaff1 and pwaff2 and such that on each
 * cell, the quasi-affine expression is the maximum of those of pwaff1
 * and pwaff2.  If only one of pwaff1 or pwaff2 is defined on a given
 * cell, then the associated expression is the defined one.
 */
__isl_give isl_pw_aff *isl_pw_aff_union_max(__isl_take isl_pw_aff *pwaff1,
	__isl_take isl_pw_aff *pwaff2)
{
	isl_pw_aff_align_params_bin(&pwaff1, &pwaff2);
	return isl_pw_aff_union_opt_cmp(pwaff1, pwaff2, &isl_aff_ge_set);
}

/* Compute a piecewise quasi-affine expression with a domain that
 * is the union of those of pwaff1 and pwaff2 and such that on each
 * cell, the quasi-affine expression is the minimum of those of pwaff1
 * and pwaff2.  If only one of pwaff1 or pwaff2 is defined on a given
 * cell, then the associated expression is the defined one.
 */
__isl_give isl_pw_aff *isl_pw_aff_union_min(__isl_take isl_pw_aff *pwaff1,
	__isl_take isl_pw_aff *pwaff2)
{
	isl_pw_aff_align_params_bin(&pwaff1, &pwaff2);
	return isl_pw_aff_union_opt_cmp(pwaff1, pwaff2, &isl_aff_le_set);
}

__isl_give isl_pw_aff *isl_pw_aff_union_opt(__isl_take isl_pw_aff *pwaff1,
	__isl_take isl_pw_aff *pwaff2, int max)
{
	if (max)
		return isl_pw_aff_union_max(pwaff1, pwaff2);
	else
		return isl_pw_aff_union_min(pwaff1, pwaff2);
}

/* Is the domain of "pa" a product?
 */
static isl_bool isl_pw_aff_domain_is_product(__isl_keep isl_pw_aff *pa)
{
	return isl_space_domain_is_wrapping(isl_pw_aff_peek_space(pa));
}

#undef TYPE
#define TYPE	isl_pw_aff
#include <isl_domain_factor_templ.c>

/* Return a set containing those elements in the domain
 * of "pwaff" where it satisfies "fn" (if complement is 0) or
 * does not satisfy "fn" (if complement is 1).
 *
 * The pieces with a NaN never belong to the result since
 * NaN does not satisfy any property.
 */
static __isl_give isl_set *pw_aff_locus(__isl_take isl_pw_aff *pwaff,
	__isl_give isl_basic_set *(*fn)(__isl_take isl_aff *aff, int rational,
		void *user),
	int complement, void *user)
{
	int i;
	isl_set *set;

	if (!pwaff)
		return NULL;

	set = isl_set_empty(isl_pw_aff_get_domain_space(pwaff));

	for (i = 0; i < pwaff->n; ++i) {
		isl_basic_set *bset;
		isl_set *set_i, *locus;
		isl_bool rational;

		if (isl_aff_is_nan(pwaff->p[i].aff))
			continue;

		rational = isl_set_has_rational(pwaff->p[i].set);
		bset = fn(isl_aff_copy(pwaff->p[i].aff), rational, user);
		locus = isl_set_from_basic_set(bset);
		set_i = isl_set_copy(pwaff->p[i].set);
		if (complement)
			set_i = isl_set_subtract(set_i, locus);
		else
			set_i = isl_set_intersect(set_i, locus);
		set = isl_set_union_disjoint(set, set_i);
	}

	isl_pw_aff_free(pwaff);

	return set;
}

/* Return a set containing those elements in the domain
 * of "pa" where it is positive.
 */
__isl_give isl_set *isl_pw_aff_pos_set(__isl_take isl_pw_aff *pa)
{
	return pw_aff_locus(pa, &aff_pos_basic_set, 0, NULL);
}

/* Return a set containing those elements in the domain
 * of pwaff where it is non-negative.
 */
__isl_give isl_set *isl_pw_aff_nonneg_set(__isl_take isl_pw_aff *pwaff)
{
	return pw_aff_locus(pwaff, &aff_nonneg_basic_set, 0, NULL);
}

/* Return a set containing those elements in the domain
 * of pwaff where it is zero.
 */
__isl_give isl_set *isl_pw_aff_zero_set(__isl_take isl_pw_aff *pwaff)
{
	return pw_aff_locus(pwaff, &aff_zero_basic_set, 0, NULL);
}

/* Return a set containing those elements in the domain
 * of pwaff where it is not zero.
 */
__isl_give isl_set *isl_pw_aff_non_zero_set(__isl_take isl_pw_aff *pwaff)
{
	return pw_aff_locus(pwaff, &aff_zero_basic_set, 1, NULL);
}

/* Bind the affine function "aff" to the parameter "id",
 * returning the elements in the domain where the affine expression
 * is equal to the parameter.
 */
__isl_give isl_basic_set *isl_aff_bind_id(__isl_take isl_aff *aff,
	__isl_take isl_id *id)
{
	isl_space *space;
	isl_aff *aff_id;

	space = isl_aff_get_domain_space(aff);
	space = isl_space_add_param_id(space, isl_id_copy(id));

	aff = isl_aff_align_params(aff, isl_space_copy(space));
	aff_id = isl_aff_param_on_domain_space_id(space, id);

	return isl_aff_eq_basic_set(aff, aff_id);
}

/* Wrapper around isl_aff_bind_id for use as pw_aff_locus callback.
 * "rational" should not be set.
 */
static __isl_give isl_basic_set *aff_bind_id(__isl_take isl_aff *aff,
	int rational, void *user)
{
	isl_id *id = user;

	if (!aff)
		return NULL;
	if (rational)
		isl_die(isl_aff_get_ctx(aff), isl_error_unsupported,
			"rational binding not supported", goto error);
	return isl_aff_bind_id(aff, isl_id_copy(id));
error:
	isl_aff_free(aff);
	return NULL;
}

/* Bind the piecewise affine function "pa" to the parameter "id",
 * returning the elements in the domain where the expression
 * is equal to the parameter.
 */
__isl_give isl_set *isl_pw_aff_bind_id(__isl_take isl_pw_aff *pa,
	__isl_take isl_id *id)
{
	isl_set *bound;

	bound = pw_aff_locus(pa, &aff_bind_id, 0, id);
	isl_id_free(id);

	return bound;
}

/* Return a set containing those elements in the shared domain
 * of pwaff1 and pwaff2 where pwaff1 is greater than (or equal) to pwaff2.
 *
 * We compute the difference on the shared domain and then construct
 * the set of values where this difference is non-negative.
 * If strict is set, we first subtract 1 from the difference.
 * If equal is set, we only return the elements where pwaff1 and pwaff2
 * are equal.
 */
static __isl_give isl_set *pw_aff_gte_set(__isl_take isl_pw_aff *pwaff1,
	__isl_take isl_pw_aff *pwaff2, int strict, int equal)
{
	isl_set *set1, *set2;

	set1 = isl_pw_aff_domain(isl_pw_aff_copy(pwaff1));
	set2 = isl_pw_aff_domain(isl_pw_aff_copy(pwaff2));
	set1 = isl_set_intersect(set1, set2);
	pwaff1 = isl_pw_aff_intersect_domain(pwaff1, isl_set_copy(set1));
	pwaff2 = isl_pw_aff_intersect_domain(pwaff2, isl_set_copy(set1));
	pwaff1 = isl_pw_aff_add(pwaff1, isl_pw_aff_neg(pwaff2));

	if (strict) {
		isl_space *space = isl_set_get_space(set1);
		isl_aff *aff;
		aff = isl_aff_zero_on_domain(isl_local_space_from_space(space));
		aff = isl_aff_add_constant_si(aff, -1);
		pwaff1 = isl_pw_aff_add(pwaff1, isl_pw_aff_alloc(set1, aff));
	} else
		isl_set_free(set1);

	if (equal)
		return isl_pw_aff_zero_set(pwaff1);
	return isl_pw_aff_nonneg_set(pwaff1);
}

/* Return a set containing those elements in the shared domain
 * of pwaff1 and pwaff2 where pwaff1 is equal to pwaff2.
 */
__isl_give isl_set *isl_pw_aff_eq_set(__isl_take isl_pw_aff *pwaff1,
	__isl_take isl_pw_aff *pwaff2)
{
	isl_pw_aff_align_params_bin(&pwaff1, &pwaff2);
	return pw_aff_gte_set(pwaff1, pwaff2, 0, 1);
}

/* Return a set containing those elements in the shared domain
 * of pwaff1 and pwaff2 where pwaff1 is greater than or equal to pwaff2.
 */
__isl_give isl_set *isl_pw_aff_ge_set(__isl_take isl_pw_aff *pwaff1,
	__isl_take isl_pw_aff *pwaff2)
{
	isl_pw_aff_align_params_bin(&pwaff1, &pwaff2);
	return pw_aff_gte_set(pwaff1, pwaff2, 0, 0);
}

/* Return a set containing those elements in the shared domain
 * of pwaff1 and pwaff2 where pwaff1 is strictly greater than pwaff2.
 */
__isl_give isl_set *isl_pw_aff_gt_set(__isl_take isl_pw_aff *pwaff1,
	__isl_take isl_pw_aff *pwaff2)
{
	isl_pw_aff_align_params_bin(&pwaff1, &pwaff2);
	return pw_aff_gte_set(pwaff1, pwaff2, 1, 0);
}

__isl_give isl_set *isl_pw_aff_le_set(__isl_take isl_pw_aff *pwaff1,
	__isl_take isl_pw_aff *pwaff2)
{
	return isl_pw_aff_ge_set(pwaff2, pwaff1);
}

__isl_give isl_set *isl_pw_aff_lt_set(__isl_take isl_pw_aff *pwaff1,
	__isl_take isl_pw_aff *pwaff2)
{
	return isl_pw_aff_gt_set(pwaff2, pwaff1);
}

/* Return a map containing pairs of elements in the domains of "pa1" and "pa2"
 * where the function values are ordered in the same way as "order",
 * which returns a set in the shared domain of its two arguments.
 *
 * Let "pa1" and "pa2" be defined on domains A and B respectively.
 * We first pull back the two functions such that they are defined on
 * the domain [A -> B].  Then we apply "order", resulting in a set
 * in the space [A -> B].  Finally, we unwrap this set to obtain
 * a map in the space A -> B.
 */
static __isl_give isl_map *isl_pw_aff_order_map(
	__isl_take isl_pw_aff *pa1, __isl_take isl_pw_aff *pa2,
	__isl_give isl_set *(*order)(__isl_take isl_pw_aff *pa1,
		__isl_take isl_pw_aff *pa2))
{
	isl_space *space1, *space2;
	isl_multi_aff *ma;
	isl_set *set;

	isl_pw_aff_align_params_bin(&pa1, &pa2);
	space1 = isl_space_domain(isl_pw_aff_get_space(pa1));
	space2 = isl_space_domain(isl_pw_aff_get_space(pa2));
	space1 = isl_space_map_from_domain_and_range(space1, space2);
	ma = isl_multi_aff_domain_map(isl_space_copy(space1));
	pa1 = isl_pw_aff_pullback_multi_aff(pa1, ma);
	ma = isl_multi_aff_range_map(space1);
	pa2 = isl_pw_aff_pullback_multi_aff(pa2, ma);
	set = order(pa1, pa2);

	return isl_set_unwrap(set);
}

/* Return a map containing pairs of elements in the domains of "pa1" and "pa2"
 * where the function values are equal.
 */
__isl_give isl_map *isl_pw_aff_eq_map(__isl_take isl_pw_aff *pa1,
	__isl_take isl_pw_aff *pa2)
{
	return isl_pw_aff_order_map(pa1, pa2, &isl_pw_aff_eq_set);
}

/* Return a map containing pairs of elements in the domains of "pa1" and "pa2"
 * where the function value of "pa1" is less than or equal to
 * the function value of "pa2".
 */
__isl_give isl_map *isl_pw_aff_le_map(__isl_take isl_pw_aff *pa1,
	__isl_take isl_pw_aff *pa2)
{
	return isl_pw_aff_order_map(pa1, pa2, &isl_pw_aff_le_set);
}

/* Return a map containing pairs of elements in the domains of "pa1" and "pa2"
 * where the function value of "pa1" is less than the function value of "pa2".
 */
__isl_give isl_map *isl_pw_aff_lt_map(__isl_take isl_pw_aff *pa1,
	__isl_take isl_pw_aff *pa2)
{
	return isl_pw_aff_order_map(pa1, pa2, &isl_pw_aff_lt_set);
}

/* Return a map containing pairs of elements in the domains of "pa1" and "pa2"
 * where the function value of "pa1" is greater than or equal to
 * the function value of "pa2".
 */
__isl_give isl_map *isl_pw_aff_ge_map(__isl_take isl_pw_aff *pa1,
	__isl_take isl_pw_aff *pa2)
{
	return isl_pw_aff_order_map(pa1, pa2, &isl_pw_aff_ge_set);
}

/* Return a map containing pairs of elements in the domains of "pa1" and "pa2"
 * where the function value of "pa1" is greater than the function value
 * of "pa2".
 */
__isl_give isl_map *isl_pw_aff_gt_map(__isl_take isl_pw_aff *pa1,
	__isl_take isl_pw_aff *pa2)
{
	return isl_pw_aff_order_map(pa1, pa2, &isl_pw_aff_gt_set);
}

/* Return a set containing those elements in the shared domain
 * of the elements of list1 and list2 where each element in list1
 * has the relation specified by "fn" with each element in list2.
 */
static __isl_give isl_set *pw_aff_list_set(__isl_take isl_pw_aff_list *list1,
	__isl_take isl_pw_aff_list *list2,
	__isl_give isl_set *(*fn)(__isl_take isl_pw_aff *pwaff1,
				    __isl_take isl_pw_aff *pwaff2))
{
	int i, j;
	isl_ctx *ctx;
	isl_set *set;

	if (!list1 || !list2)
		goto error;

	ctx = isl_pw_aff_list_get_ctx(list1);
	if (list1->n < 1 || list2->n < 1)
		isl_die(ctx, isl_error_invalid,
			"list should contain at least one element", goto error);

	set = isl_set_universe(isl_pw_aff_get_domain_space(list1->p[0]));
	for (i = 0; i < list1->n; ++i)
		for (j = 0; j < list2->n; ++j) {
			isl_set *set_ij;

			set_ij = fn(isl_pw_aff_copy(list1->p[i]),
				    isl_pw_aff_copy(list2->p[j]));
			set = isl_set_intersect(set, set_ij);
		}

	isl_pw_aff_list_free(list1);
	isl_pw_aff_list_free(list2);
	return set;
error:
	isl_pw_aff_list_free(list1);
	isl_pw_aff_list_free(list2);
	return NULL;
}

/* Return a set containing those elements in the shared domain
 * of the elements of list1 and list2 where each element in list1
 * is equal to each element in list2.
 */
__isl_give isl_set *isl_pw_aff_list_eq_set(__isl_take isl_pw_aff_list *list1,
	__isl_take isl_pw_aff_list *list2)
{
	return pw_aff_list_set(list1, list2, &isl_pw_aff_eq_set);
}

__isl_give isl_set *isl_pw_aff_list_ne_set(__isl_take isl_pw_aff_list *list1,
	__isl_take isl_pw_aff_list *list2)
{
	return pw_aff_list_set(list1, list2, &isl_pw_aff_ne_set);
}

/* Return a set containing those elements in the shared domain
 * of the elements of list1 and list2 where each element in list1
 * is less than or equal to each element in list2.
 */
__isl_give isl_set *isl_pw_aff_list_le_set(__isl_take isl_pw_aff_list *list1,
	__isl_take isl_pw_aff_list *list2)
{
	return pw_aff_list_set(list1, list2, &isl_pw_aff_le_set);
}

__isl_give isl_set *isl_pw_aff_list_lt_set(__isl_take isl_pw_aff_list *list1,
	__isl_take isl_pw_aff_list *list2)
{
	return pw_aff_list_set(list1, list2, &isl_pw_aff_lt_set);
}

__isl_give isl_set *isl_pw_aff_list_ge_set(__isl_take isl_pw_aff_list *list1,
	__isl_take isl_pw_aff_list *list2)
{
	return pw_aff_list_set(list1, list2, &isl_pw_aff_ge_set);
}

__isl_give isl_set *isl_pw_aff_list_gt_set(__isl_take isl_pw_aff_list *list1,
	__isl_take isl_pw_aff_list *list2)
{
	return pw_aff_list_set(list1, list2, &isl_pw_aff_gt_set);
}


/* Return a set containing those elements in the shared domain
 * of pwaff1 and pwaff2 where pwaff1 is not equal to pwaff2.
 */
__isl_give isl_set *isl_pw_aff_ne_set(__isl_take isl_pw_aff *pwaff1,
	__isl_take isl_pw_aff *pwaff2)
{
	isl_set *set_lt, *set_gt;

	isl_pw_aff_align_params_bin(&pwaff1, &pwaff2);
	set_lt = isl_pw_aff_lt_set(isl_pw_aff_copy(pwaff1),
				   isl_pw_aff_copy(pwaff2));
	set_gt = isl_pw_aff_gt_set(pwaff1, pwaff2);
	return isl_set_union_disjoint(set_lt, set_gt);
}

__isl_give isl_pw_aff *isl_pw_aff_scale_down(__isl_take isl_pw_aff *pwaff,
	isl_int v)
{
	int i;

	if (isl_int_is_one(v))
		return pwaff;
	if (!isl_int_is_pos(v))
		isl_die(isl_pw_aff_get_ctx(pwaff), isl_error_invalid,
			"factor needs to be positive",
			return isl_pw_aff_free(pwaff));
	pwaff = isl_pw_aff_cow(pwaff);
	if (!pwaff)
		return NULL;
	if (pwaff->n == 0)
		return pwaff;

	for (i = 0; i < pwaff->n; ++i) {
		pwaff->p[i].aff = isl_aff_scale_down(pwaff->p[i].aff, v);
		if (!pwaff->p[i].aff)
			return isl_pw_aff_free(pwaff);
	}

	return pwaff;
}

__isl_give isl_pw_aff *isl_pw_aff_floor(__isl_take isl_pw_aff *pwaff)
{
	int i;

	pwaff = isl_pw_aff_cow(pwaff);
	if (!pwaff)
		return NULL;
	if (pwaff->n == 0)
		return pwaff;

	for (i = 0; i < pwaff->n; ++i) {
		pwaff->p[i].aff = isl_aff_floor(pwaff->p[i].aff);
		if (!pwaff->p[i].aff)
			return isl_pw_aff_free(pwaff);
	}

	return pwaff;
}

__isl_give isl_pw_aff *isl_pw_aff_ceil(__isl_take isl_pw_aff *pwaff)
{
	int i;

	pwaff = isl_pw_aff_cow(pwaff);
	if (!pwaff)
		return NULL;
	if (pwaff->n == 0)
		return pwaff;

	for (i = 0; i < pwaff->n; ++i) {
		pwaff->p[i].aff = isl_aff_ceil(pwaff->p[i].aff);
		if (!pwaff->p[i].aff)
			return isl_pw_aff_free(pwaff);
	}

	return pwaff;
}

/* Assuming that "cond1" and "cond2" are disjoint,
 * return an affine expression that is equal to pwaff1 on cond1
 * and to pwaff2 on cond2.
 */
static __isl_give isl_pw_aff *isl_pw_aff_select(
	__isl_take isl_set *cond1, __isl_take isl_pw_aff *pwaff1,
	__isl_take isl_set *cond2, __isl_take isl_pw_aff *pwaff2)
{
	pwaff1 = isl_pw_aff_intersect_domain(pwaff1, cond1);
	pwaff2 = isl_pw_aff_intersect_domain(pwaff2, cond2);

	return isl_pw_aff_add_disjoint(pwaff1, pwaff2);
}

/* Return an affine expression that is equal to pwaff_true for elements
 * where "cond" is non-zero and to pwaff_false for elements where "cond"
 * is zero.
 * That is, return cond ? pwaff_true : pwaff_false;
 *
 * If "cond" involves and NaN, then we conservatively return a NaN
 * on its entire domain.  In principle, we could consider the pieces
 * where it is NaN separately from those where it is not.
 *
 * If "pwaff_true" and "pwaff_false" are obviously equal to each other,
 * then only use the domain of "cond" to restrict the domain.
 */
__isl_give isl_pw_aff *isl_pw_aff_cond(__isl_take isl_pw_aff *cond,
	__isl_take isl_pw_aff *pwaff_true, __isl_take isl_pw_aff *pwaff_false)
{
	isl_set *cond_true, *cond_false;
	isl_bool equal;

	if (!cond)
		goto error;
	if (isl_pw_aff_involves_nan(cond)) {
		isl_space *space = isl_pw_aff_get_domain_space(cond);
		isl_local_space *ls = isl_local_space_from_space(space);
		isl_pw_aff_free(cond);
		isl_pw_aff_free(pwaff_true);
		isl_pw_aff_free(pwaff_false);
		return isl_pw_aff_nan_on_domain(ls);
	}

	pwaff_true = isl_pw_aff_align_params(pwaff_true,
					    isl_pw_aff_get_space(pwaff_false));
	pwaff_false = isl_pw_aff_align_params(pwaff_false,
					    isl_pw_aff_get_space(pwaff_true));
	equal = isl_pw_aff_plain_is_equal(pwaff_true, pwaff_false);
	if (equal < 0)
		goto error;
	if (equal) {
		isl_set *dom;

		dom = isl_set_coalesce(isl_pw_aff_domain(cond));
		isl_pw_aff_free(pwaff_false);
		return isl_pw_aff_intersect_domain(pwaff_true, dom);
	}

	cond_true = isl_pw_aff_non_zero_set(isl_pw_aff_copy(cond));
	cond_false = isl_pw_aff_zero_set(cond);
	return isl_pw_aff_select(cond_true, pwaff_true,
				 cond_false, pwaff_false);
error:
	isl_pw_aff_free(cond);
	isl_pw_aff_free(pwaff_true);
	isl_pw_aff_free(pwaff_false);
	return NULL;
}

isl_bool isl_aff_is_cst(__isl_keep isl_aff *aff)
{
	int pos;

	if (!aff)
		return isl_bool_error;

	pos = isl_seq_first_non_zero(aff->v->el + 2, aff->v->size - 2);
	return isl_bool_ok(pos == -1);
}

/* Check whether pwaff is a piecewise constant.
 */
isl_bool isl_pw_aff_is_cst(__isl_keep isl_pw_aff *pwaff)
{
	int i;

	if (!pwaff)
		return isl_bool_error;

	for (i = 0; i < pwaff->n; ++i) {
		isl_bool is_cst = isl_aff_is_cst(pwaff->p[i].aff);
		if (is_cst < 0 || !is_cst)
			return is_cst;
	}

	return isl_bool_true;
}

/* Return the product of "aff1" and "aff2".
 *
 * If either of the two is NaN, then the result is NaN.
 *
 * Otherwise, at least one of "aff1" or "aff2" needs to be a constant.
 */
__isl_give isl_aff *isl_aff_mul(__isl_take isl_aff *aff1,
	__isl_take isl_aff *aff2)
{
	if (!aff1 || !aff2)
		goto error;

	if (isl_aff_is_nan(aff1)) {
		isl_aff_free(aff2);
		return aff1;
	}
	if (isl_aff_is_nan(aff2)) {
		isl_aff_free(aff1);
		return aff2;
	}

	if (!isl_aff_is_cst(aff2) && isl_aff_is_cst(aff1))
		return isl_aff_mul(aff2, aff1);

	if (!isl_aff_is_cst(aff2))
		isl_die(isl_aff_get_ctx(aff1), isl_error_invalid,
			"at least one affine expression should be constant",
			goto error);

	aff1 = isl_aff_cow(aff1);
	if (!aff1 || !aff2)
		goto error;

	aff1 = isl_aff_scale(aff1, aff2->v->el[1]);
	aff1 = isl_aff_scale_down(aff1, aff2->v->el[0]);

	isl_aff_free(aff2);
	return aff1;
error:
	isl_aff_free(aff1);
	isl_aff_free(aff2);
	return NULL;
}

/* Divide "aff1" by "aff2", assuming "aff2" is a constant.
 *
 * If either of the two is NaN, then the result is NaN.
 * A division by zero also results in NaN.
 */
__isl_give isl_aff *isl_aff_div(__isl_take isl_aff *aff1,
	__isl_take isl_aff *aff2)
{
	isl_bool is_cst, is_zero;
	int neg;

	if (!aff1 || !aff2)
		goto error;

	if (isl_aff_is_nan(aff1)) {
		isl_aff_free(aff2);
		return aff1;
	}
	if (isl_aff_is_nan(aff2)) {
		isl_aff_free(aff1);
		return aff2;
	}

	is_cst = isl_aff_is_cst(aff2);
	if (is_cst < 0)
		goto error;
	if (!is_cst)
		isl_die(isl_aff_get_ctx(aff2), isl_error_invalid,
			"second argument should be a constant", goto error);
	is_zero = isl_aff_plain_is_zero(aff2);
	if (is_zero < 0)
		goto error;
	if (is_zero)
		return set_nan_free(aff1, aff2);

	neg = isl_int_is_neg(aff2->v->el[1]);
	if (neg) {
		isl_int_neg(aff2->v->el[0], aff2->v->el[0]);
		isl_int_neg(aff2->v->el[1], aff2->v->el[1]);
	}

	aff1 = isl_aff_scale(aff1, aff2->v->el[0]);
	aff1 = isl_aff_scale_down(aff1, aff2->v->el[1]);

	if (neg) {
		isl_int_neg(aff2->v->el[0], aff2->v->el[0]);
		isl_int_neg(aff2->v->el[1], aff2->v->el[1]);
	}

	isl_aff_free(aff2);
	return aff1;
error:
	isl_aff_free(aff1);
	isl_aff_free(aff2);
	return NULL;
}

__isl_give isl_pw_aff *isl_pw_aff_add(__isl_take isl_pw_aff *pwaff1,
	__isl_take isl_pw_aff *pwaff2)
{
	isl_pw_aff_align_params_bin(&pwaff1, &pwaff2);
	return isl_pw_aff_on_shared_domain(pwaff1, pwaff2, &isl_aff_add);
}

__isl_give isl_pw_aff *isl_pw_aff_union_add(__isl_take isl_pw_aff *pwaff1,
	__isl_take isl_pw_aff *pwaff2)
{
	return isl_pw_aff_union_add_(pwaff1, pwaff2);
}

__isl_give isl_pw_aff *isl_pw_aff_mul(__isl_take isl_pw_aff *pwaff1,
	__isl_take isl_pw_aff *pwaff2)
{
	isl_pw_aff_align_params_bin(&pwaff1, &pwaff2);
	return isl_pw_aff_on_shared_domain(pwaff1, pwaff2, &isl_aff_mul);
}

/* Divide "pa1" by "pa2", assuming "pa2" is a piecewise constant.
 */
__isl_give isl_pw_aff *isl_pw_aff_div(__isl_take isl_pw_aff *pa1,
	__isl_take isl_pw_aff *pa2)
{
	int is_cst;

	is_cst = isl_pw_aff_is_cst(pa2);
	if (is_cst < 0)
		goto error;
	if (!is_cst)
		isl_die(isl_pw_aff_get_ctx(pa2), isl_error_invalid,
			"second argument should be a piecewise constant",
			goto error);
	isl_pw_aff_align_params_bin(&pa1, &pa2);
	return isl_pw_aff_on_shared_domain(pa1, pa2, &isl_aff_div);
error:
	isl_pw_aff_free(pa1);
	isl_pw_aff_free(pa2);
	return NULL;
}

/* Compute the quotient of the integer division of "pa1" by "pa2"
 * with rounding towards zero.
 * "pa2" is assumed to be a piecewise constant.
 *
 * In particular, return
 *
 *	pa1 >= 0 ? floor(pa1/pa2) : ceil(pa1/pa2)
 *
 */
__isl_give isl_pw_aff *isl_pw_aff_tdiv_q(__isl_take isl_pw_aff *pa1,
	__isl_take isl_pw_aff *pa2)
{
	int is_cst;
	isl_set *cond;
	isl_pw_aff *f, *c;

	is_cst = isl_pw_aff_is_cst(pa2);
	if (is_cst < 0)
		goto error;
	if (!is_cst)
		isl_die(isl_pw_aff_get_ctx(pa2), isl_error_invalid,
			"second argument should be a piecewise constant",
			goto error);

	pa1 = isl_pw_aff_div(pa1, pa2);

	cond = isl_pw_aff_nonneg_set(isl_pw_aff_copy(pa1));
	f = isl_pw_aff_floor(isl_pw_aff_copy(pa1));
	c = isl_pw_aff_ceil(pa1);
	return isl_pw_aff_cond(isl_set_indicator_function(cond), f, c);
error:
	isl_pw_aff_free(pa1);
	isl_pw_aff_free(pa2);
	return NULL;
}

/* Compute the remainder of the integer division of "pa1" by "pa2"
 * with rounding towards zero.
 * "pa2" is assumed to be a piecewise constant.
 *
 * In particular, return
 *
 *	pa1 - pa2 * (pa1 >= 0 ? floor(pa1/pa2) : ceil(pa1/pa2))
 *
 */
__isl_give isl_pw_aff *isl_pw_aff_tdiv_r(__isl_take isl_pw_aff *pa1,
	__isl_take isl_pw_aff *pa2)
{
	int is_cst;
	isl_pw_aff *res;

	is_cst = isl_pw_aff_is_cst(pa2);
	if (is_cst < 0)
		goto error;
	if (!is_cst)
		isl_die(isl_pw_aff_get_ctx(pa2), isl_error_invalid,
			"second argument should be a piecewise constant",
			goto error);
	res = isl_pw_aff_tdiv_q(isl_pw_aff_copy(pa1), isl_pw_aff_copy(pa2));
	res = isl_pw_aff_mul(pa2, res);
	res = isl_pw_aff_sub(pa1, res);
	return res;
error:
	isl_pw_aff_free(pa1);
	isl_pw_aff_free(pa2);
	return NULL;
}

/* Does either of "pa1" or "pa2" involve any NaN2?
 */
static isl_bool either_involves_nan(__isl_keep isl_pw_aff *pa1,
	__isl_keep isl_pw_aff *pa2)
{
	isl_bool has_nan;

	has_nan = isl_pw_aff_involves_nan(pa1);
	if (has_nan < 0 || has_nan)
		return has_nan;
	return isl_pw_aff_involves_nan(pa2);
}

/* Replace "pa1" and "pa2" (at least one of which involves a NaN)
 * by a NaN on their shared domain.
 *
 * In principle, the result could be refined to only being NaN
 * on the parts of this domain where at least one of "pa1" or "pa2" is NaN.
 */
static __isl_give isl_pw_aff *replace_by_nan(__isl_take isl_pw_aff *pa1,
	__isl_take isl_pw_aff *pa2)
{
	isl_local_space *ls;
	isl_set *dom;
	isl_pw_aff *pa;

	dom = isl_set_intersect(isl_pw_aff_domain(pa1), isl_pw_aff_domain(pa2));
	ls = isl_local_space_from_space(isl_set_get_space(dom));
	pa = isl_pw_aff_nan_on_domain(ls);
	pa = isl_pw_aff_intersect_domain(pa, dom);

	return pa;
}

static __isl_give isl_pw_aff *pw_aff_min(__isl_take isl_pw_aff *pwaff1,
	__isl_take isl_pw_aff *pwaff2)
{
	isl_set *le;
	isl_set *dom;

	dom = isl_set_intersect(isl_pw_aff_domain(isl_pw_aff_copy(pwaff1)),
				isl_pw_aff_domain(isl_pw_aff_copy(pwaff2)));
	le = isl_pw_aff_le_set(isl_pw_aff_copy(pwaff1),
				isl_pw_aff_copy(pwaff2));
	dom = isl_set_subtract(dom, isl_set_copy(le));
	return isl_pw_aff_select(le, pwaff1, dom, pwaff2);
}

static __isl_give isl_pw_aff *pw_aff_max(__isl_take isl_pw_aff *pwaff1,
	__isl_take isl_pw_aff *pwaff2)
{
	isl_set *ge;
	isl_set *dom;

	dom = isl_set_intersect(isl_pw_aff_domain(isl_pw_aff_copy(pwaff1)),
				isl_pw_aff_domain(isl_pw_aff_copy(pwaff2)));
	ge = isl_pw_aff_ge_set(isl_pw_aff_copy(pwaff1),
				isl_pw_aff_copy(pwaff2));
	dom = isl_set_subtract(dom, isl_set_copy(ge));
	return isl_pw_aff_select(ge, pwaff1, dom, pwaff2);
}

/* Return an expression for the minimum (if "max" is not set) or
 * the maximum (if "max" is set) of "pa1" and "pa2".
 * If either expression involves any NaN, then return a NaN
 * on the shared domain as result.
 */
static __isl_give isl_pw_aff *pw_aff_min_max(__isl_take isl_pw_aff *pa1,
	__isl_take isl_pw_aff *pa2, int max)
{
	isl_bool has_nan;

	has_nan = either_involves_nan(pa1, pa2);
	if (has_nan < 0)
		pa1 = isl_pw_aff_free(pa1);
	else if (has_nan)
		return replace_by_nan(pa1, pa2);

	isl_pw_aff_align_params_bin(&pa1, &pa2);
	if (max)
		return pw_aff_max(pa1, pa2);
	else
		return pw_aff_min(pa1, pa2);
}

/* Return an expression for the minimum of "pwaff1" and "pwaff2".
 */
__isl_give isl_pw_aff *isl_pw_aff_min(__isl_take isl_pw_aff *pwaff1,
	__isl_take isl_pw_aff *pwaff2)
{
	return pw_aff_min_max(pwaff1, pwaff2, 0);
}

/* Return an expression for the maximum of "pwaff1" and "pwaff2".
 */
__isl_give isl_pw_aff *isl_pw_aff_max(__isl_take isl_pw_aff *pwaff1,
	__isl_take isl_pw_aff *pwaff2)
{
	return pw_aff_min_max(pwaff1, pwaff2, 1);
}

static __isl_give isl_pw_aff *pw_aff_list_reduce(
	__isl_take isl_pw_aff_list *list,
	__isl_give isl_pw_aff *(*fn)(__isl_take isl_pw_aff *pwaff1,
					__isl_take isl_pw_aff *pwaff2))
{
	int i;
	isl_ctx *ctx;
	isl_pw_aff *res;

	if (!list)
		return NULL;

	ctx = isl_pw_aff_list_get_ctx(list);
	if (list->n < 1)
		isl_die(ctx, isl_error_invalid,
			"list should contain at least one element", goto error);

	res = isl_pw_aff_copy(list->p[0]);
	for (i = 1; i < list->n; ++i)
		res = fn(res, isl_pw_aff_copy(list->p[i]));

	isl_pw_aff_list_free(list);
	return res;
error:
	isl_pw_aff_list_free(list);
	return NULL;
}

/* Return an isl_pw_aff that maps each element in the intersection of the
 * domains of the elements of list to the minimal corresponding affine
 * expression.
 */
__isl_give isl_pw_aff *isl_pw_aff_list_min(__isl_take isl_pw_aff_list *list)
{
	return pw_aff_list_reduce(list, &isl_pw_aff_min);
}

/* Return an isl_pw_aff that maps each element in the intersection of the
 * domains of the elements of list to the maximal corresponding affine
 * expression.
 */
__isl_give isl_pw_aff *isl_pw_aff_list_max(__isl_take isl_pw_aff_list *list)
{
	return pw_aff_list_reduce(list, &isl_pw_aff_max);
}

/* Mark the domains of "pwaff" as rational.
 */
__isl_give isl_pw_aff *isl_pw_aff_set_rational(__isl_take isl_pw_aff *pwaff)
{
	int i;

	pwaff = isl_pw_aff_cow(pwaff);
	if (!pwaff)
		return NULL;
	if (pwaff->n == 0)
		return pwaff;

	for (i = 0; i < pwaff->n; ++i) {
		pwaff->p[i].set = isl_set_set_rational(pwaff->p[i].set);
		if (!pwaff->p[i].set)
			return isl_pw_aff_free(pwaff);
	}

	return pwaff;
}

/* Mark the domains of the elements of "list" as rational.
 */
__isl_give isl_pw_aff_list *isl_pw_aff_list_set_rational(
	__isl_take isl_pw_aff_list *list)
{
	int i, n;

	if (!list)
		return NULL;
	if (list->n == 0)
		return list;

	n = list->n;
	for (i = 0; i < n; ++i) {
		isl_pw_aff *pa;

		pa = isl_pw_aff_list_get_pw_aff(list, i);
		pa = isl_pw_aff_set_rational(pa);
		list = isl_pw_aff_list_set_pw_aff(list, i, pa);
	}

	return list;
}

/* Do the parameters of "aff" match those of "space"?
 */
isl_bool isl_aff_matching_params(__isl_keep isl_aff *aff,
	__isl_keep isl_space *space)
{
	isl_space *aff_space;
	isl_bool match;

	if (!aff || !space)
		return isl_bool_error;

	aff_space = isl_aff_get_domain_space(aff);

	match = isl_space_has_equal_params(space, aff_space);

	isl_space_free(aff_space);
	return match;
}

/* Check that the domain space of "aff" matches "space".
 */
isl_stat isl_aff_check_match_domain_space(__isl_keep isl_aff *aff,
	__isl_keep isl_space *space)
{
	isl_space *aff_space;
	isl_bool match;

	if (!aff || !space)
		return isl_stat_error;

	aff_space = isl_aff_get_domain_space(aff);

	match = isl_space_has_equal_params(space, aff_space);
	if (match < 0)
		goto error;
	if (!match)
		isl_die(isl_aff_get_ctx(aff), isl_error_invalid,
			"parameters don't match", goto error);
	match = isl_space_tuple_is_equal(space, isl_dim_in,
					aff_space, isl_dim_set);
	if (match < 0)
		goto error;
	if (!match)
		isl_die(isl_aff_get_ctx(aff), isl_error_invalid,
			"domains don't match", goto error);
	isl_space_free(aff_space);
	return isl_stat_ok;
error:
	isl_space_free(aff_space);
	return isl_stat_error;
}

/* Return the shared (universe) domain of the elements of "ma".
 *
 * Since an isl_multi_aff (and an isl_aff) is always total,
 * the domain is always the universe set in its domain space.
 * This is a helper function for use in the generic isl_multi_*_bind.
 */
static __isl_give isl_basic_set *isl_multi_aff_domain(
	__isl_take isl_multi_aff *ma)
{
	isl_space *space;

	space = isl_multi_aff_get_space(ma);
	isl_multi_aff_free(ma);

	return isl_basic_set_universe(isl_space_domain(space));
}

#undef BASE
#define BASE aff

#include <isl_multi_no_explicit_domain.c>
#include <isl_multi_templ.c>
#include <isl_multi_add_constant_templ.c>
#include <isl_multi_apply_set.c>
#include <isl_multi_arith_templ.c>
#include <isl_multi_bind_domain_templ.c>
#include <isl_multi_cmp.c>
#include <isl_multi_dim_id_templ.c>
#include <isl_multi_dims.c>
#include <isl_multi_floor.c>
#include <isl_multi_from_base_templ.c>
#include <isl_multi_identity_templ.c>
#include <isl_multi_insert_domain_templ.c>
#include <isl_multi_locals_templ.c>
#include <isl_multi_move_dims_templ.c>
#include <isl_multi_nan_templ.c>
#include <isl_multi_product_templ.c>
#include <isl_multi_splice_templ.c>
#include <isl_multi_tuple_id_templ.c>
#include <isl_multi_unbind_params_templ.c>
#include <isl_multi_zero_templ.c>

#undef DOMBASE
#define DOMBASE set
#include <isl_multi_gist.c>

#undef DOMBASE
#define DOMBASE basic_set
#include <isl_multi_bind_templ.c>

/* Construct an isl_multi_aff living in "space" that corresponds
 * to the affine transformation matrix "mat".
 */
__isl_give isl_multi_aff *isl_multi_aff_from_aff_mat(
	__isl_take isl_space *space, __isl_take isl_mat *mat)
{
	isl_ctx *ctx;
	isl_local_space *ls = NULL;
	isl_multi_aff *ma = NULL;
	isl_size n_row, n_col, n_out, total;
	int i;

	if (!space || !mat)
		goto error;

	ctx = isl_mat_get_ctx(mat);

	n_row = isl_mat_rows(mat);
	n_col = isl_mat_cols(mat);
	n_out = isl_space_dim(space, isl_dim_out);
	total = isl_space_dim(space, isl_dim_all);
	if (n_row < 0 || n_col < 0 || n_out < 0 || total < 0)
		goto error;
	if (n_row < 1)
		isl_die(ctx, isl_error_invalid,
			"insufficient number of rows", goto error);
	if (n_col < 1)
		isl_die(ctx, isl_error_invalid,
			"insufficient number of columns", goto error);
	if (1 + n_out != n_row || 2 + total != n_row + n_col)
		isl_die(ctx, isl_error_invalid,
			"dimension mismatch", goto error);

	ma = isl_multi_aff_zero(isl_space_copy(space));
	space = isl_space_domain(space);
	ls = isl_local_space_from_space(isl_space_copy(space));

	for (i = 0; i < n_row - 1; ++i) {
		isl_vec *v;
		isl_aff *aff;

		v = isl_vec_alloc(ctx, 1 + n_col);
		if (!v)
			goto error;
		isl_int_set(v->el[0], mat->row[0][0]);
		isl_seq_cpy(v->el + 1, mat->row[1 + i], n_col);
		v = isl_vec_normalize(v);
		aff = isl_aff_alloc_vec(isl_local_space_copy(ls), v);
		ma = isl_multi_aff_set_aff(ma, i, aff);
	}

	isl_space_free(space);
	isl_local_space_free(ls);
	isl_mat_free(mat);
	return ma;
error:
	isl_space_free(space);
	isl_local_space_free(ls);
	isl_mat_free(mat);
	isl_multi_aff_free(ma);
	return NULL;
}

/* Return the constant terms of the affine expressions of "ma".
 */
__isl_give isl_multi_val *isl_multi_aff_get_constant_multi_val(
	__isl_keep isl_multi_aff *ma)
{
	int i;
	isl_size n;
	isl_space *space;
	isl_multi_val *mv;

	n = isl_multi_aff_size(ma);
	if (n < 0)
		return NULL;
	space = isl_space_range(isl_multi_aff_get_space(ma));
	space = isl_space_drop_all_params(space);
	mv = isl_multi_val_zero(space);

	for (i = 0; i < n; ++i) {
		isl_aff *aff;
		isl_val *val;

		aff = isl_multi_aff_get_at(ma, i);
		val = isl_aff_get_constant_val(aff);
		isl_aff_free(aff);
		mv = isl_multi_val_set_at(mv, i, val);
	}

	return mv;
}

/* Remove any internal structure of the domain of "ma".
 * If there is any such internal structure in the input,
 * then the name of the corresponding space is also removed.
 */
__isl_give isl_multi_aff *isl_multi_aff_flatten_domain(
	__isl_take isl_multi_aff *ma)
{
	isl_space *space;

	if (!ma)
		return NULL;

	if (!ma->space->nested[0])
		return ma;

	space = isl_multi_aff_get_space(ma);
	space = isl_space_flatten_domain(space);
	ma = isl_multi_aff_reset_space(ma, space);

	return ma;
}

/* Given a map space, return an isl_multi_aff that maps a wrapped copy
 * of the space to its domain.
 */
__isl_give isl_multi_aff *isl_multi_aff_domain_map(__isl_take isl_space *space)
{
	int i;
	isl_size n_in;
	isl_local_space *ls;
	isl_multi_aff *ma;

	if (!space)
		return NULL;
	if (!isl_space_is_map(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"not a map space", goto error);

	n_in = isl_space_dim(space, isl_dim_in);
	if (n_in < 0)
		goto error;
	space = isl_space_domain_map(space);

	ma = isl_multi_aff_alloc(isl_space_copy(space));
	if (n_in == 0) {
		isl_space_free(space);
		return ma;
	}

	space = isl_space_domain(space);
	ls = isl_local_space_from_space(space);
	for (i = 0; i < n_in; ++i) {
		isl_aff *aff;

		aff = isl_aff_var_on_domain(isl_local_space_copy(ls),
						isl_dim_set, i);
		ma = isl_multi_aff_set_aff(ma, i, aff);
	}
	isl_local_space_free(ls);
	return ma;
error:
	isl_space_free(space);
	return NULL;
}

/* This function performs the same operation as isl_multi_aff_domain_map,
 * but is considered as a function on an isl_space when exported.
 */
__isl_give isl_multi_aff *isl_space_domain_map_multi_aff(
	__isl_take isl_space *space)
{
	return isl_multi_aff_domain_map(space);
}

/* Given a map space, return an isl_multi_aff that maps a wrapped copy
 * of the space to its range.
 */
__isl_give isl_multi_aff *isl_multi_aff_range_map(__isl_take isl_space *space)
{
	int i;
	isl_size n_in, n_out;
	isl_local_space *ls;
	isl_multi_aff *ma;

	if (!space)
		return NULL;
	if (!isl_space_is_map(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"not a map space", goto error);

	n_in = isl_space_dim(space, isl_dim_in);
	n_out = isl_space_dim(space, isl_dim_out);
	if (n_in < 0 || n_out < 0)
		goto error;
	space = isl_space_range_map(space);

	ma = isl_multi_aff_alloc(isl_space_copy(space));
	if (n_out == 0) {
		isl_space_free(space);
		return ma;
	}

	space = isl_space_domain(space);
	ls = isl_local_space_from_space(space);
	for (i = 0; i < n_out; ++i) {
		isl_aff *aff;

		aff = isl_aff_var_on_domain(isl_local_space_copy(ls),
						isl_dim_set, n_in + i);
		ma = isl_multi_aff_set_aff(ma, i, aff);
	}
	isl_local_space_free(ls);
	return ma;
error:
	isl_space_free(space);
	return NULL;
}

/* This function performs the same operation as isl_multi_aff_range_map,
 * but is considered as a function on an isl_space when exported.
 */
__isl_give isl_multi_aff *isl_space_range_map_multi_aff(
	__isl_take isl_space *space)
{
	return isl_multi_aff_range_map(space);
}

/* Given a map space, return an isl_pw_multi_aff that maps a wrapped copy
 * of the space to its domain.
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_domain_map(
	__isl_take isl_space *space)
{
	return isl_pw_multi_aff_from_multi_aff(isl_multi_aff_domain_map(space));
}

/* This function performs the same operation as isl_pw_multi_aff_domain_map,
 * but is considered as a function on an isl_space when exported.
 */
__isl_give isl_pw_multi_aff *isl_space_domain_map_pw_multi_aff(
	__isl_take isl_space *space)
{
	return isl_pw_multi_aff_domain_map(space);
}

/* Given a map space, return an isl_pw_multi_aff that maps a wrapped copy
 * of the space to its range.
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_range_map(
	__isl_take isl_space *space)
{
	return isl_pw_multi_aff_from_multi_aff(isl_multi_aff_range_map(space));
}

/* This function performs the same operation as isl_pw_multi_aff_range_map,
 * but is considered as a function on an isl_space when exported.
 */
__isl_give isl_pw_multi_aff *isl_space_range_map_pw_multi_aff(
	__isl_take isl_space *space)
{
	return isl_pw_multi_aff_range_map(space);
}

/* Given the space of a set and a range of set dimensions,
 * construct an isl_multi_aff that projects out those dimensions.
 */
__isl_give isl_multi_aff *isl_multi_aff_project_out_map(
	__isl_take isl_space *space, enum isl_dim_type type,
	unsigned first, unsigned n)
{
	int i;
	isl_size dim;
	isl_local_space *ls;
	isl_multi_aff *ma;

	if (!space)
		return NULL;
	if (!isl_space_is_set(space))
		isl_die(isl_space_get_ctx(space), isl_error_unsupported,
			"expecting set space", goto error);
	if (type != isl_dim_set)
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"only set dimensions can be projected out", goto error);
	if (isl_space_check_range(space, type, first, n) < 0)
		goto error;

	dim = isl_space_dim(space, isl_dim_set);
	if (dim < 0)
		goto error;

	space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_out, dim - n);

	if (dim == n)
		return isl_multi_aff_alloc(space);

	ma = isl_multi_aff_alloc(isl_space_copy(space));
	space = isl_space_domain(space);
	ls = isl_local_space_from_space(space);

	for (i = 0; i < first; ++i) {
		isl_aff *aff;

		aff = isl_aff_var_on_domain(isl_local_space_copy(ls),
						isl_dim_set, i);
		ma = isl_multi_aff_set_aff(ma, i, aff);
	}

	for (i = 0; i < dim - (first + n); ++i) {
		isl_aff *aff;

		aff = isl_aff_var_on_domain(isl_local_space_copy(ls),
						isl_dim_set, first + n + i);
		ma = isl_multi_aff_set_aff(ma, first + i, aff);
	}

	isl_local_space_free(ls);
	return ma;
error:
	isl_space_free(space);
	return NULL;
}

/* Given the space of a set and a range of set dimensions,
 * construct an isl_pw_multi_aff that projects out those dimensions.
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_project_out_map(
	__isl_take isl_space *space, enum isl_dim_type type,
	unsigned first, unsigned n)
{
	isl_multi_aff *ma;

	ma = isl_multi_aff_project_out_map(space, type, first, n);
	return isl_pw_multi_aff_from_multi_aff(ma);
}

/* This function performs the same operation as isl_pw_multi_aff_from_multi_aff,
 * but is considered as a function on an isl_multi_aff when exported.
 */
__isl_give isl_pw_multi_aff *isl_multi_aff_to_pw_multi_aff(
	__isl_take isl_multi_aff *ma)
{
	return isl_pw_multi_aff_from_multi_aff(ma);
}

/* Create a piecewise multi-affine expression in the given space that maps each
 * input dimension to the corresponding output dimension.
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_identity(
	__isl_take isl_space *space)
{
	return isl_pw_multi_aff_from_multi_aff(isl_multi_aff_identity(space));
}

/* Create a piecewise multi expression that maps elements in the given space
 * to themselves.
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_identity_on_domain_space(
	__isl_take isl_space *space)
{
	isl_multi_aff *ma;

	ma = isl_multi_aff_identity_on_domain_space(space);
	return isl_pw_multi_aff_from_multi_aff(ma);
}

/* This function performs the same operation as
 * isl_pw_multi_aff_identity_on_domain_space,
 * but is considered as a function on an isl_space when exported.
 */
__isl_give isl_pw_multi_aff *isl_space_identity_pw_multi_aff_on_domain(
	__isl_take isl_space *space)
{
	return isl_pw_multi_aff_identity_on_domain_space(space);
}

/* Exploit the equalities in "eq" to simplify the affine expressions.
 */
static __isl_give isl_multi_aff *isl_multi_aff_substitute_equalities(
	__isl_take isl_multi_aff *maff, __isl_take isl_basic_set *eq)
{
	int i;

	maff = isl_multi_aff_cow(maff);
	if (!maff || !eq)
		goto error;

	for (i = 0; i < maff->n; ++i) {
		maff->u.p[i] = isl_aff_substitute_equalities(maff->u.p[i],
						    isl_basic_set_copy(eq));
		if (!maff->u.p[i])
			goto error;
	}

	isl_basic_set_free(eq);
	return maff;
error:
	isl_basic_set_free(eq);
	isl_multi_aff_free(maff);
	return NULL;
}

__isl_give isl_multi_aff *isl_multi_aff_scale(__isl_take isl_multi_aff *maff,
	isl_int f)
{
	int i;

	maff = isl_multi_aff_cow(maff);
	if (!maff)
		return NULL;

	for (i = 0; i < maff->n; ++i) {
		maff->u.p[i] = isl_aff_scale(maff->u.p[i], f);
		if (!maff->u.p[i])
			return isl_multi_aff_free(maff);
	}

	return maff;
}

__isl_give isl_multi_aff *isl_multi_aff_add_on_domain(__isl_keep isl_set *dom,
	__isl_take isl_multi_aff *maff1, __isl_take isl_multi_aff *maff2)
{
	maff1 = isl_multi_aff_add(maff1, maff2);
	maff1 = isl_multi_aff_gist(maff1, isl_set_copy(dom));
	return maff1;
}

isl_bool isl_multi_aff_is_empty(__isl_keep isl_multi_aff *maff)
{
	if (!maff)
		return isl_bool_error;

	return isl_bool_false;
}

/* Return the set of domain elements where "ma1" is lexicographically
 * smaller than or equal to "ma2".
 */
__isl_give isl_set *isl_multi_aff_lex_le_set(__isl_take isl_multi_aff *ma1,
	__isl_take isl_multi_aff *ma2)
{
	return isl_multi_aff_lex_ge_set(ma2, ma1);
}

/* Return the set of domain elements where "ma1" is lexicographically
 * smaller than "ma2".
 */
__isl_give isl_set *isl_multi_aff_lex_lt_set(__isl_take isl_multi_aff *ma1,
	__isl_take isl_multi_aff *ma2)
{
	return isl_multi_aff_lex_gt_set(ma2, ma1);
}

/* Return the set of domain elements where "ma1" is lexicographically
 * greater than to "ma2".  If "equal" is set, then include the domain
 * elements where they are equal.
 * Do this for the case where there are no entries.
 * In this case, "ma1" cannot be greater than "ma2",
 * but it is (greater than or) equal to "ma2".
 */
static __isl_give isl_set *isl_multi_aff_lex_gte_set_0d(
	__isl_take isl_multi_aff *ma1, __isl_take isl_multi_aff *ma2, int equal)
{
	isl_space *space;

	space = isl_multi_aff_get_domain_space(ma1);

	isl_multi_aff_free(ma1);
	isl_multi_aff_free(ma2);

	if (equal)
		return isl_set_universe(space);
	else
		return isl_set_empty(space);
}

/* Return the set where entry "i" of "ma1" and "ma2"
 * satisfy the relation prescribed by "cmp".
 */
static __isl_give isl_set *isl_multi_aff_order_at(__isl_keep isl_multi_aff *ma1,
	__isl_keep isl_multi_aff *ma2, int i,
	__isl_give isl_set *(*cmp)(__isl_take isl_aff *aff1,
		__isl_take isl_aff *aff2))
{
	isl_aff *aff1, *aff2;

	aff1 = isl_multi_aff_get_at(ma1, i);
	aff2 = isl_multi_aff_get_at(ma2, i);
	return cmp(aff1, aff2);
}

/* Return the set of domain elements where "ma1" is lexicographically
 * greater than to "ma2".  If "equal" is set, then include the domain
 * elements where they are equal.
 *
 * In particular, for all but the final entry,
 * include the set of elements where this entry is strictly greater in "ma1"
 * and all previous entries are equal.
 * The final entry is also allowed to be equal in the two functions
 * if "equal" is set.
 *
 * The case where there are no entries is handled separately.
 */
static __isl_give isl_set *isl_multi_aff_lex_gte_set(
	__isl_take isl_multi_aff *ma1, __isl_take isl_multi_aff *ma2, int equal)
{
	int i;
	isl_size n;
	isl_space *space;
	isl_set *res;
	isl_set *equal_set;
	isl_set *gte;

	if (isl_multi_aff_check_equal_space(ma1, ma2) < 0)
		goto error;
	n = isl_multi_aff_size(ma1);
	if (n < 0)
		goto error;
	if (n == 0)
		return isl_multi_aff_lex_gte_set_0d(ma1, ma2, equal);

	space = isl_multi_aff_get_domain_space(ma1);
	res = isl_set_empty(isl_space_copy(space));
	equal_set = isl_set_universe(space);

	for (i = 0; i + 1 < n; ++i) {
		isl_bool empty;
		isl_set *gt, *eq;

		gt = isl_multi_aff_order_at(ma1, ma2, i, &isl_aff_gt_set);
		gt = isl_set_intersect(gt, isl_set_copy(equal_set));
		res = isl_set_union(res, gt);
		eq = isl_multi_aff_order_at(ma1, ma2, i, &isl_aff_eq_set);
		equal_set = isl_set_intersect(equal_set, eq);

		empty = isl_set_is_empty(equal_set);
		if (empty >= 0 && empty)
			break;
	}

	if (equal)
		gte = isl_multi_aff_order_at(ma1, ma2, n - 1, &isl_aff_ge_set);
	else
		gte = isl_multi_aff_order_at(ma1, ma2, n - 1, &isl_aff_gt_set);
	isl_multi_aff_free(ma1);
	isl_multi_aff_free(ma2);

	gte = isl_set_intersect(gte, equal_set);
	return isl_set_union(res, gte);
error:
	isl_multi_aff_free(ma1);
	isl_multi_aff_free(ma2);
	return NULL;
}

/* Return the set of domain elements where "ma1" is lexicographically
 * greater than or equal to "ma2".
 */
__isl_give isl_set *isl_multi_aff_lex_ge_set(__isl_take isl_multi_aff *ma1,
	__isl_take isl_multi_aff *ma2)
{
	return isl_multi_aff_lex_gte_set(ma1, ma2, 1);
}

/* Return the set of domain elements where "ma1" is lexicographically
 * greater than "ma2".
 */
__isl_give isl_set *isl_multi_aff_lex_gt_set(__isl_take isl_multi_aff *ma1,
	__isl_take isl_multi_aff *ma2)
{
	return isl_multi_aff_lex_gte_set(ma1, ma2, 0);
}

#define isl_multi_aff_zero_in_space	isl_multi_aff_zero

#undef PW
#define PW isl_pw_multi_aff
#undef BASE
#define BASE multi_aff
#undef EL_IS_ZERO
#define EL_IS_ZERO is_empty
#undef ZERO
#define ZERO empty
#undef IS_ZERO
#define IS_ZERO is_empty
#undef FIELD
#define FIELD maff
#undef DEFAULT_IS_ZERO
#define DEFAULT_IS_ZERO 0

#include <isl_pw_templ.c>
#include <isl_pw_add_constant_multi_val_templ.c>
#include <isl_pw_add_constant_val_templ.c>
#include <isl_pw_bind_domain_templ.c>
#include <isl_pw_insert_dims_templ.c>
#include <isl_pw_insert_domain_templ.c>
#include <isl_pw_locals_templ.c>
#include <isl_pw_move_dims_templ.c>
#include <isl_pw_neg_templ.c>
#include <isl_pw_pullback_templ.c>
#include <isl_pw_range_tuple_id_templ.c>
#include <isl_pw_union_opt.c>

#undef BASE
#define BASE pw_multi_aff

#include <isl_union_multi.c>
#include "isl_union_locals_templ.c"
#include <isl_union_neg.c>

#undef BASE
#define BASE multi_aff

#include <isl_union_pw_templ.c>

/* Generic function for extracting a factor from a product "pma".
 * "check_space" checks that the space is that of the right kind of product.
 * "space_factor" extracts the factor from the space.
 * "multi_aff_factor" extracts the factor from the constituent functions.
 */
static __isl_give isl_pw_multi_aff *pw_multi_aff_factor(
	__isl_take isl_pw_multi_aff *pma,
	isl_stat (*check_space)(__isl_keep isl_pw_multi_aff *pma),
	__isl_give isl_space *(*space_factor)(__isl_take isl_space *space),
	__isl_give isl_multi_aff *(*multi_aff_factor)(
		__isl_take isl_multi_aff *ma))
{
	int i;
	isl_space *space;

	if (check_space(pma) < 0)
		return isl_pw_multi_aff_free(pma);

	space = isl_pw_multi_aff_take_space(pma);
	space = space_factor(space);

	for (i = 0; pma && i < pma->n; ++i) {
		isl_multi_aff *ma;

		ma = isl_pw_multi_aff_take_base_at(pma, i);
		ma = multi_aff_factor(ma);
		pma = isl_pw_multi_aff_restore_base_at(pma, i, ma);
	}

	pma = isl_pw_multi_aff_restore_space(pma, space);

	return pma;
}

/* Is the range of "pma" a wrapped relation?
 */
static isl_bool isl_pw_multi_aff_range_is_wrapping(
	__isl_keep isl_pw_multi_aff *pma)
{
	return isl_space_range_is_wrapping(isl_pw_multi_aff_peek_space(pma));
}

/* Check that the range of "pma" is a product.
 */
static isl_stat pw_multi_aff_check_range_product(
	__isl_keep isl_pw_multi_aff *pma)
{
	isl_bool wraps;

	wraps = isl_pw_multi_aff_range_is_wrapping(pma);
	if (wraps < 0)
		return isl_stat_error;
	if (!wraps)
		isl_die(isl_pw_multi_aff_get_ctx(pma), isl_error_invalid,
			"range is not a product", return isl_stat_error);
	return isl_stat_ok;
}

/* Given a function A -> [B -> C], extract the function A -> B.
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_range_factor_domain(
	__isl_take isl_pw_multi_aff *pma)
{
	return pw_multi_aff_factor(pma, &pw_multi_aff_check_range_product,
				&isl_space_range_factor_domain,
				&isl_multi_aff_range_factor_domain);
}

/* Given a function A -> [B -> C], extract the function A -> C.
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_range_factor_range(
	__isl_take isl_pw_multi_aff *pma)
{
	return pw_multi_aff_factor(pma, &pw_multi_aff_check_range_product,
				&isl_space_range_factor_range,
				&isl_multi_aff_range_factor_range);
}

/* Given two piecewise multi affine expressions, return a piecewise
 * multi-affine expression defined on the union of the definition domains
 * of the inputs that is equal to the lexicographic maximum of the two
 * inputs on each cell.  If only one of the two inputs is defined on
 * a given cell, then it is considered to be the maximum.
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_union_lexmax(
	__isl_take isl_pw_multi_aff *pma1,
	__isl_take isl_pw_multi_aff *pma2)
{
	isl_pw_multi_aff_align_params_bin(&pma1, &pma2);
	return isl_pw_multi_aff_union_opt_cmp(pma1, pma2,
					    &isl_multi_aff_lex_ge_set);
}

/* Given two piecewise multi affine expressions, return a piecewise
 * multi-affine expression defined on the union of the definition domains
 * of the inputs that is equal to the lexicographic minimum of the two
 * inputs on each cell.  If only one of the two inputs is defined on
 * a given cell, then it is considered to be the minimum.
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_union_lexmin(
	__isl_take isl_pw_multi_aff *pma1,
	__isl_take isl_pw_multi_aff *pma2)
{
	isl_pw_multi_aff_align_params_bin(&pma1, &pma2);
	return isl_pw_multi_aff_union_opt_cmp(pma1, pma2,
					    &isl_multi_aff_lex_le_set);
}

__isl_give isl_pw_multi_aff *isl_pw_multi_aff_add(
	__isl_take isl_pw_multi_aff *pma1, __isl_take isl_pw_multi_aff *pma2)
{
	isl_pw_multi_aff_align_params_bin(&pma1, &pma2);
	return isl_pw_multi_aff_on_shared_domain(pma1, pma2,
						&isl_multi_aff_add);
}

/* Subtract "pma2" from "pma1" and return the result.
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_sub(
	__isl_take isl_pw_multi_aff *pma1, __isl_take isl_pw_multi_aff *pma2)
{
	isl_pw_multi_aff_align_params_bin(&pma1, &pma2);
	return isl_pw_multi_aff_on_shared_domain(pma1, pma2,
						&isl_multi_aff_sub);
}

__isl_give isl_pw_multi_aff *isl_pw_multi_aff_union_add(
	__isl_take isl_pw_multi_aff *pma1, __isl_take isl_pw_multi_aff *pma2)
{
	return isl_pw_multi_aff_union_add_(pma1, pma2);
}

/* Compute the sum of "upa1" and "upa2" on the union of their domains,
 * with the actual sum on the shared domain and
 * the defined expression on the symmetric difference of the domains.
 */
__isl_give isl_union_pw_aff *isl_union_pw_aff_union_add(
	__isl_take isl_union_pw_aff *upa1, __isl_take isl_union_pw_aff *upa2)
{
	return isl_union_pw_aff_union_add_(upa1, upa2);
}

/* Compute the sum of "upma1" and "upma2" on the union of their domains,
 * with the actual sum on the shared domain and
 * the defined expression on the symmetric difference of the domains.
 */
__isl_give isl_union_pw_multi_aff *isl_union_pw_multi_aff_union_add(
	__isl_take isl_union_pw_multi_aff *upma1,
	__isl_take isl_union_pw_multi_aff *upma2)
{
	return isl_union_pw_multi_aff_union_add_(upma1, upma2);
}

/* Given two piecewise multi-affine expressions A -> B and C -> D,
 * construct a piecewise multi-affine expression [A -> C] -> [B -> D].
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_product(
	__isl_take isl_pw_multi_aff *pma1, __isl_take isl_pw_multi_aff *pma2)
{
	int i, j, n;
	isl_space *space;
	isl_pw_multi_aff *res;

	if (isl_pw_multi_aff_align_params_bin(&pma1, &pma2) < 0)
		goto error;

	n = pma1->n * pma2->n;
	space = isl_space_product(isl_space_copy(pma1->dim),
				  isl_space_copy(pma2->dim));
	res = isl_pw_multi_aff_alloc_size(space, n);

	for (i = 0; i < pma1->n; ++i) {
		for (j = 0; j < pma2->n; ++j) {
			isl_set *domain;
			isl_multi_aff *ma;

			domain = isl_set_product(isl_set_copy(pma1->p[i].set),
						 isl_set_copy(pma2->p[j].set));
			ma = isl_multi_aff_product(
					isl_multi_aff_copy(pma1->p[i].maff),
					isl_multi_aff_copy(pma2->p[j].maff));
			res = isl_pw_multi_aff_add_piece(res, domain, ma);
		}
	}

	isl_pw_multi_aff_free(pma1);
	isl_pw_multi_aff_free(pma2);
	return res;
error:
	isl_pw_multi_aff_free(pma1);
	isl_pw_multi_aff_free(pma2);
	return NULL;
}

/* Subtract the initial "n" elements in "ma" with coefficients in "c" and
 * denominator "denom".
 * "denom" is allowed to be negative, in which case the actual denominator
 * is -denom and the expressions are added instead.
 */
static __isl_give isl_aff *subtract_initial(__isl_take isl_aff *aff,
	__isl_keep isl_multi_aff *ma, int n, isl_int *c, isl_int denom)
{
	int i, first;
	int sign;
	isl_int d;

	first = isl_seq_first_non_zero(c, n);
	if (first == -1)
		return aff;

	sign = isl_int_sgn(denom);
	isl_int_init(d);
	isl_int_abs(d, denom);
	for (i = first; i < n; ++i) {
		isl_aff *aff_i;

		if (isl_int_is_zero(c[i]))
			continue;
		aff_i = isl_multi_aff_get_aff(ma, i);
		aff_i = isl_aff_scale(aff_i, c[i]);
		aff_i = isl_aff_scale_down(aff_i, d);
		if (sign >= 0)
			aff = isl_aff_sub(aff, aff_i);
		else
			aff = isl_aff_add(aff, aff_i);
	}
	isl_int_clear(d);

	return aff;
}

/* Extract an affine expression that expresses the output dimension "pos"
 * of "bmap" in terms of the parameters and input dimensions from
 * equality "eq".
 * Note that this expression may involve integer divisions defined
 * in terms of parameters and input dimensions.
 * The equality may also involve references to earlier (but not later)
 * output dimensions.  These are replaced by the corresponding elements
 * in "ma".
 *
 * If the equality is of the form
 *
 *	f(i) + h(j) + a x + g(i) = 0,
 *
 * with f(i) a linear combinations of the parameters and input dimensions,
 * g(i) a linear combination of integer divisions defined in terms of the same
 * and h(j) a linear combinations of earlier output dimensions,
 * then the affine expression is
 *
 *	(-f(i) - g(i))/a - h(j)/a
 *
 * If the equality is of the form
 *
 *	f(i) + h(j) - a x + g(i) = 0,
 *
 * then the affine expression is
 *
 *	(f(i) + g(i))/a - h(j)/(-a)
 *
 *
 * If "div" refers to an integer division (i.e., it is smaller than
 * the number of integer divisions), then the equality constraint
 * does involve an integer division (the one at position "div") that
 * is defined in terms of output dimensions.  However, this integer
 * division can be eliminated by exploiting a pair of constraints
 * x >= l and x <= l + n, with n smaller than the coefficient of "div"
 * in the equality constraint.  "ineq" refers to inequality x >= l, i.e.,
 * -l + x >= 0.
 * In particular, let
 *
 *	x = e(i) + m floor(...)
 *
 * with e(i) the expression derived above and floor(...) the integer
 * division involving output dimensions.
 * From
 *
 *	l <= x <= l + n,
 *
 * we have
 *
 *	0 <= x - l <= n
 *
 * This means
 *
 *	e(i) + m floor(...) - l = (e(i) + m floor(...) - l) mod m
 *	                        = (e(i) - l) mod m
 *
 * Therefore,
 *
 *	x - l = (e(i) - l) mod m
 *
 * or
 *
 *	x = ((e(i) - l) mod m) + l
 *
 * The variable "shift" below contains the expression -l, which may
 * also involve a linear combination of earlier output dimensions.
 */
static __isl_give isl_aff *extract_aff_from_equality(
	__isl_keep isl_basic_map *bmap, int pos, int eq, int div, int ineq,
	__isl_keep isl_multi_aff *ma)
{
	unsigned o_out;
	isl_size n_div, n_out;
	isl_ctx *ctx;
	isl_local_space *ls;
	isl_aff *aff, *shift;
	isl_val *mod;

	ctx = isl_basic_map_get_ctx(bmap);
	ls = isl_basic_map_get_local_space(bmap);
	ls = isl_local_space_domain(ls);
	aff = isl_aff_alloc(isl_local_space_copy(ls));
	if (!aff)
		goto error;
	o_out = isl_basic_map_offset(bmap, isl_dim_out);
	n_out = isl_basic_map_dim(bmap, isl_dim_out);
	n_div = isl_basic_map_dim(bmap, isl_dim_div);
	if (n_out < 0 || n_div < 0)
		goto error;
	if (isl_int_is_neg(bmap->eq[eq][o_out + pos])) {
		isl_seq_cpy(aff->v->el + 1, bmap->eq[eq], o_out);
		isl_seq_cpy(aff->v->el + 1 + o_out,
			    bmap->eq[eq] + o_out + n_out, n_div);
	} else {
		isl_seq_neg(aff->v->el + 1, bmap->eq[eq], o_out);
		isl_seq_neg(aff->v->el + 1 + o_out,
			    bmap->eq[eq] + o_out + n_out, n_div);
	}
	if (div < n_div)
		isl_int_set_si(aff->v->el[1 + o_out + div], 0);
	isl_int_abs(aff->v->el[0], bmap->eq[eq][o_out + pos]);
	aff = subtract_initial(aff, ma, pos, bmap->eq[eq] + o_out,
			    bmap->eq[eq][o_out + pos]);
	if (div < n_div) {
		shift = isl_aff_alloc(isl_local_space_copy(ls));
		if (!shift)
			goto error;
		isl_seq_cpy(shift->v->el + 1, bmap->ineq[ineq], o_out);
		isl_seq_cpy(shift->v->el + 1 + o_out,
			    bmap->ineq[ineq] + o_out + n_out, n_div);
		isl_int_set_si(shift->v->el[0], 1);
		shift = subtract_initial(shift, ma, pos,
					bmap->ineq[ineq] + o_out, ctx->negone);
		aff = isl_aff_add(aff, isl_aff_copy(shift));
		mod = isl_val_int_from_isl_int(ctx,
					    bmap->eq[eq][o_out + n_out + div]);
		mod = isl_val_abs(mod);
		aff = isl_aff_mod_val(aff, mod);
		aff = isl_aff_sub(aff, shift);
	}

	isl_local_space_free(ls);
	return aff;
error:
	isl_local_space_free(ls);
	isl_aff_free(aff);
	return NULL;
}

/* Given a basic map with output dimensions defined
 * in terms of the parameters input dimensions and earlier
 * output dimensions using an equality (and possibly a pair on inequalities),
 * extract an isl_aff that expresses output dimension "pos" in terms
 * of the parameters and input dimensions.
 * Note that this expression may involve integer divisions defined
 * in terms of parameters and input dimensions.
 * "ma" contains the expressions corresponding to earlier output dimensions.
 *
 * This function shares some similarities with
 * isl_basic_map_has_defining_equality and isl_constraint_get_bound.
 */
static __isl_give isl_aff *extract_isl_aff_from_basic_map(
	__isl_keep isl_basic_map *bmap, int pos, __isl_keep isl_multi_aff *ma)
{
	int eq, div, ineq;
	isl_aff *aff;

	if (!bmap)
		return NULL;
	eq = isl_basic_map_output_defining_equality(bmap, pos, &div, &ineq);
	if (eq >= bmap->n_eq)
		isl_die(isl_basic_map_get_ctx(bmap), isl_error_invalid,
			"unable to find suitable equality", return NULL);
	aff = extract_aff_from_equality(bmap, pos, eq, div, ineq, ma);

	aff = isl_aff_remove_unused_divs(aff);
	return aff;
}

/* Given a basic map where each output dimension is defined
 * in terms of the parameters and input dimensions using an equality,
 * extract an isl_multi_aff that expresses the output dimensions in terms
 * of the parameters and input dimensions.
 */
static __isl_give isl_multi_aff *extract_isl_multi_aff_from_basic_map(
	__isl_take isl_basic_map *bmap)
{
	int i;
	isl_size n_out;
	isl_multi_aff *ma;

	if (!bmap)
		return NULL;

	ma = isl_multi_aff_alloc(isl_basic_map_get_space(bmap));
	n_out = isl_basic_map_dim(bmap, isl_dim_out);
	if (n_out < 0)
		ma = isl_multi_aff_free(ma);

	for (i = 0; i < n_out; ++i) {
		isl_aff *aff;

		aff = extract_isl_aff_from_basic_map(bmap, i, ma);
		ma = isl_multi_aff_set_aff(ma, i, aff);
	}

	isl_basic_map_free(bmap);

	return ma;
}

/* Given a basic set where each set dimension is defined
 * in terms of the parameters using an equality,
 * extract an isl_multi_aff that expresses the set dimensions in terms
 * of the parameters.
 */
__isl_give isl_multi_aff *isl_multi_aff_from_basic_set_equalities(
	__isl_take isl_basic_set *bset)
{
	return extract_isl_multi_aff_from_basic_map(bset);
}

/* Create an isl_pw_multi_aff that is equivalent to
 * isl_map_intersect_domain(isl_map_from_basic_map(bmap), domain).
 * The given basic map is such that each output dimension is defined
 * in terms of the parameters and input dimensions using an equality.
 *
 * Since some applications expect the result of isl_pw_multi_aff_from_map
 * to only contain integer affine expressions, we compute the floor
 * of the expression before returning.
 *
 * Remove all constraints involving local variables without
 * an explicit representation (resulting in the removal of those
 * local variables) prior to the actual extraction to ensure
 * that the local spaces in which the resulting affine expressions
 * are created do not contain any unknown local variables.
 * Removing such constraints is safe because constraints involving
 * unknown local variables are not used to determine whether
 * a basic map is obviously single-valued.
 */
static __isl_give isl_pw_multi_aff *plain_pw_multi_aff_from_map(
	__isl_take isl_set *domain, __isl_take isl_basic_map *bmap)
{
	isl_multi_aff *ma;

	bmap = isl_basic_map_drop_constraints_involving_unknown_divs(bmap);
	ma = extract_isl_multi_aff_from_basic_map(bmap);
	ma = isl_multi_aff_floor(ma);
	return isl_pw_multi_aff_alloc(domain, ma);
}

/* Try and create an isl_pw_multi_aff that is equivalent to the given isl_map.
 * This obviously only works if the input "map" is single-valued.
 * If so, we compute the lexicographic minimum of the image in the form
 * of an isl_pw_multi_aff.  Since the image is unique, it is equal
 * to its lexicographic minimum.
 * If the input is not single-valued, we produce an error.
 */
static __isl_give isl_pw_multi_aff *pw_multi_aff_from_map_base(
	__isl_take isl_map *map)
{
	int i;
	int sv;
	isl_pw_multi_aff *pma;

	sv = isl_map_is_single_valued(map);
	if (sv < 0)
		goto error;
	if (!sv)
		isl_die(isl_map_get_ctx(map), isl_error_invalid,
			"map is not single-valued", goto error);
	map = isl_map_make_disjoint(map);
	if (!map)
		return NULL;

	pma = isl_pw_multi_aff_empty(isl_map_get_space(map));

	for (i = 0; i < map->n; ++i) {
		isl_pw_multi_aff *pma_i;
		isl_basic_map *bmap;
		bmap = isl_basic_map_copy(map->p[i]);
		pma_i = isl_basic_map_lexmin_pw_multi_aff(bmap);
		pma = isl_pw_multi_aff_add_disjoint(pma, pma_i);
	}

	isl_map_free(map);
	return pma;
error:
	isl_map_free(map);
	return NULL;
}

/* Try and create an isl_pw_multi_aff that is equivalent to the given isl_map,
 * taking into account that the output dimension at position "d"
 * can be represented as
 *
 *	x = floor((e(...) + c1) / m)
 *
 * given that constraint "i" is of the form
 *
 *	e(...) + c1 - m x >= 0
 *
 *
 * Let "map" be of the form
 *
 *	A -> B
 *
 * We construct a mapping
 *
 *	A -> [A -> x = floor(...)]
 *
 * apply that to the map, obtaining
 *
 *	[A -> x = floor(...)] -> B
 *
 * and equate dimension "d" to x.
 * We then compute a isl_pw_multi_aff representation of the resulting map
 * and plug in the mapping above.
 */
static __isl_give isl_pw_multi_aff *pw_multi_aff_from_map_div(
	__isl_take isl_map *map, __isl_take isl_basic_map *hull, int d, int i)
{
	isl_ctx *ctx;
	isl_space *space = NULL;
	isl_local_space *ls;
	isl_multi_aff *ma;
	isl_aff *aff;
	isl_vec *v;
	isl_map *insert;
	int offset;
	isl_size n;
	isl_size n_in;
	isl_pw_multi_aff *pma;
	isl_bool is_set;

	is_set = isl_map_is_set(map);
	if (is_set < 0)
		goto error;

	offset = isl_basic_map_offset(hull, isl_dim_out);
	ctx = isl_map_get_ctx(map);
	space = isl_space_domain(isl_map_get_space(map));
	n_in = isl_space_dim(space, isl_dim_set);
	n = isl_space_dim(space, isl_dim_all);
	if (n_in < 0 || n < 0)
		goto error;

	v = isl_vec_alloc(ctx, 1 + 1 + n);
	if (v) {
		isl_int_neg(v->el[0], hull->ineq[i][offset + d]);
		isl_seq_cpy(v->el + 1, hull->ineq[i], 1 + n);
	}
	isl_basic_map_free(hull);

	ls = isl_local_space_from_space(isl_space_copy(space));
	aff = isl_aff_alloc_vec(ls, v);
	aff = isl_aff_floor(aff);
	if (is_set) {
		isl_space_free(space);
		ma = isl_multi_aff_from_aff(aff);
	} else {
		ma = isl_multi_aff_identity(isl_space_map_from_set(space));
		ma = isl_multi_aff_range_product(ma,
						isl_multi_aff_from_aff(aff));
	}

	insert = isl_map_from_multi_aff_internal(isl_multi_aff_copy(ma));
	map = isl_map_apply_domain(map, insert);
	map = isl_map_equate(map, isl_dim_in, n_in, isl_dim_out, d);
	pma = isl_pw_multi_aff_from_map(map);
	pma = isl_pw_multi_aff_pullback_multi_aff(pma, ma);

	return pma;
error:
	isl_space_free(space);
	isl_map_free(map);
	isl_basic_map_free(hull);
	return NULL;
}

/* Is constraint "c" of the form
 *
 *	e(...) + c1 - m x >= 0
 *
 * or
 *
 *	-e(...) + c2 + m x >= 0
 *
 * where m > 1 and e only depends on parameters and input dimensions?
 *
 * "offset" is the offset of the output dimensions
 * "pos" is the position of output dimension x.
 */
static int is_potential_div_constraint(isl_int *c, int offset, int d, int total)
{
	if (isl_int_is_zero(c[offset + d]))
		return 0;
	if (isl_int_is_one(c[offset + d]))
		return 0;
	if (isl_int_is_negone(c[offset + d]))
		return 0;
	if (isl_seq_first_non_zero(c + offset, d) != -1)
		return 0;
	if (isl_seq_first_non_zero(c + offset + d + 1,
				    total - (offset + d + 1)) != -1)
		return 0;
	return 1;
}

/* Try and create an isl_pw_multi_aff that is equivalent to the given isl_map.
 *
 * As a special case, we first check if there is any pair of constraints,
 * shared by all the basic maps in "map" that force a given dimension
 * to be equal to the floor of some affine combination of the input dimensions.
 *
 * In particular, if we can find two constraints
 *
 *	e(...) + c1 - m x >= 0		i.e.,		m x <= e(...) + c1
 *
 * and
 *
 *	-e(...) + c2 + m x >= 0		i.e.,		m x >= e(...) - c2
 *
 * where m > 1 and e only depends on parameters and input dimensions,
 * and such that
 *
 *	c1 + c2 < m			i.e.,		-c2 >= c1 - (m - 1)
 *
 * then we know that we can take
 *
 *	x = floor((e(...) + c1) / m)
 *
 * without having to perform any computation.
 *
 * Note that we know that
 *
 *	c1 + c2 >= 1
 *
 * If c1 + c2 were 0, then we would have detected an equality during
 * simplification.  If c1 + c2 were negative, then we would have detected
 * a contradiction.
 */
static __isl_give isl_pw_multi_aff *pw_multi_aff_from_map_check_div(
	__isl_take isl_map *map)
{
	int d;
	isl_size dim;
	int i, j, n;
	int offset;
	isl_size total;
	isl_int sum;
	isl_basic_map *hull;

	hull = isl_map_unshifted_simple_hull(isl_map_copy(map));
	dim = isl_map_dim(map, isl_dim_out);
	total = isl_basic_map_dim(hull, isl_dim_all);
	if (dim < 0 || total < 0)
		goto error;

	isl_int_init(sum);
	offset = isl_basic_map_offset(hull, isl_dim_out);
	n = hull->n_ineq;
	for (d = 0; d < dim; ++d) {
		for (i = 0; i < n; ++i) {
			if (!is_potential_div_constraint(hull->ineq[i],
							offset, d, 1 + total))
				continue;
			for (j = i + 1; j < n; ++j) {
				if (!isl_seq_is_neg(hull->ineq[i] + 1,
						hull->ineq[j] + 1, total))
					continue;
				isl_int_add(sum, hull->ineq[i][0],
						hull->ineq[j][0]);
				if (isl_int_abs_lt(sum,
						    hull->ineq[i][offset + d]))
					break;

			}
			if (j >= n)
				continue;
			isl_int_clear(sum);
			if (isl_int_is_pos(hull->ineq[j][offset + d]))
				j = i;
			return pw_multi_aff_from_map_div(map, hull, d, j);
		}
	}
	isl_int_clear(sum);
	isl_basic_map_free(hull);
	return pw_multi_aff_from_map_base(map);
error:
	isl_map_free(map);
	isl_basic_map_free(hull);
	return NULL;
}

/* Given an affine expression
 *
 *	[A -> B] -> f(A,B)
 *
 * construct an isl_multi_aff
 *
 *	[A -> B] -> B'
 *
 * such that dimension "d" in B' is set to "aff" and the remaining
 * dimensions are set equal to the corresponding dimensions in B.
 * "n_in" is the dimension of the space A.
 * "n_out" is the dimension of the space B.
 *
 * If "is_set" is set, then the affine expression is of the form
 *
 *	[B] -> f(B)
 *
 * and we construct an isl_multi_aff
 *
 *	B -> B'
 */
static __isl_give isl_multi_aff *range_map(__isl_take isl_aff *aff, int d,
	unsigned n_in, unsigned n_out, int is_set)
{
	int i;
	isl_multi_aff *ma;
	isl_space *space, *space2;
	isl_local_space *ls;

	space = isl_aff_get_domain_space(aff);
	ls = isl_local_space_from_space(isl_space_copy(space));
	space2 = isl_space_copy(space);
	if (!is_set)
		space2 = isl_space_range(isl_space_unwrap(space2));
	space = isl_space_map_from_domain_and_range(space, space2);
	ma = isl_multi_aff_alloc(space);
	ma = isl_multi_aff_set_aff(ma, d, aff);

	for (i = 0; i < n_out; ++i) {
		if (i == d)
			continue;
		aff = isl_aff_var_on_domain(isl_local_space_copy(ls),
						isl_dim_set, n_in + i);
		ma = isl_multi_aff_set_aff(ma, i, aff);
	}

	isl_local_space_free(ls);

	return ma;
}

/* Try and create an isl_pw_multi_aff that is equivalent to the given isl_map,
 * taking into account that the dimension at position "d" can be written as
 *
 *	x = m a + f(..)						(1)
 *
 * where m is equal to "gcd".
 * "i" is the index of the equality in "hull" that defines f(..).
 * In particular, the equality is of the form
 *
 *	f(..) - x + m g(existentials) = 0
 *
 * or
 *
 *	-f(..) + x + m g(existentials) = 0
 *
 * We basically plug (1) into "map", resulting in a map with "a"
 * in the range instead of "x".  The corresponding isl_pw_multi_aff
 * defining "a" is then plugged back into (1) to obtain a definition for "x".
 *
 * Specifically, given the input map
 *
 *	A -> B
 *
 * We first wrap it into a set
 *
 *	[A -> B]
 *
 * and define (1) on top of the corresponding space, resulting in "aff".
 * We use this to create an isl_multi_aff that maps the output position "d"
 * from "a" to "x", leaving all other (intput and output) dimensions unchanged.
 * We plug this into the wrapped map, unwrap the result and compute the
 * corresponding isl_pw_multi_aff.
 * The result is an expression
 *
 *	A -> T(A)
 *
 * We adjust that to
 *
 *	A -> [A -> T(A)]
 *
 * so that we can plug that into "aff", after extending the latter to
 * a mapping
 *
 *	[A -> B] -> B'
 *
 *
 * If "map" is actually a set, then there is no "A" space, meaning
 * that we do not need to perform any wrapping, and that the result
 * of the recursive call is of the form
 *
 *	[T]
 *
 * which is plugged into a mapping of the form
 *
 *	B -> B'
 */
static __isl_give isl_pw_multi_aff *pw_multi_aff_from_map_stride(
	__isl_take isl_map *map, __isl_take isl_basic_map *hull, int d, int i,
	isl_int gcd)
{
	isl_set *set;
	isl_space *space;
	isl_local_space *ls;
	isl_aff *aff;
	isl_multi_aff *ma;
	isl_pw_multi_aff *pma, *id;
	isl_size n_in;
	unsigned o_out;
	isl_size n_out;
	isl_bool is_set;

	is_set = isl_map_is_set(map);
	if (is_set < 0)
		goto error;

	n_in = isl_basic_map_dim(hull, isl_dim_in);
	n_out = isl_basic_map_dim(hull, isl_dim_out);
	if (n_in < 0 || n_out < 0)
		goto error;
	o_out = isl_basic_map_offset(hull, isl_dim_out);

	if (is_set)
		set = map;
	else
		set = isl_map_wrap(map);
	space = isl_space_map_from_set(isl_set_get_space(set));
	ma = isl_multi_aff_identity(space);
	ls = isl_local_space_from_space(isl_set_get_space(set));
	aff = isl_aff_alloc(ls);
	if (aff) {
		isl_int_set_si(aff->v->el[0], 1);
		if (isl_int_is_one(hull->eq[i][o_out + d]))
			isl_seq_neg(aff->v->el + 1, hull->eq[i],
				    aff->v->size - 1);
		else
			isl_seq_cpy(aff->v->el + 1, hull->eq[i],
				    aff->v->size - 1);
		isl_int_set(aff->v->el[1 + o_out + d], gcd);
	}
	ma = isl_multi_aff_set_aff(ma, n_in + d, isl_aff_copy(aff));
	set = isl_set_preimage_multi_aff(set, ma);

	ma = range_map(aff, d, n_in, n_out, is_set);

	if (is_set)
		map = set;
	else
		map = isl_set_unwrap(set);
	pma = isl_pw_multi_aff_from_map(map);

	if (!is_set) {
		space = isl_pw_multi_aff_get_domain_space(pma);
		space = isl_space_map_from_set(space);
		id = isl_pw_multi_aff_identity(space);
		pma = isl_pw_multi_aff_range_product(id, pma);
	}
	id = isl_pw_multi_aff_from_multi_aff(ma);
	pma = isl_pw_multi_aff_pullback_pw_multi_aff(id, pma);

	isl_basic_map_free(hull);
	return pma;
error:
	isl_map_free(map);
	isl_basic_map_free(hull);
	return NULL;
}

/* Try and create an isl_pw_multi_aff that is equivalent to the given isl_map.
 * "hull" contains the equalities valid for "map".
 *
 * Check if any of the output dimensions is "strided".
 * That is, we check if it can be written as
 *
 *	x = m a + f(..)
 *
 * with m greater than 1, a some combination of existentially quantified
 * variables and f an expression in the parameters and input dimensions.
 * If so, we remove the stride in pw_multi_aff_from_map_stride.
 *
 * Otherwise, we continue with pw_multi_aff_from_map_check_div for a further
 * special case.
 */
static __isl_give isl_pw_multi_aff *pw_multi_aff_from_map_check_strides(
	__isl_take isl_map *map, __isl_take isl_basic_map *hull)
{
	int i, j;
	isl_size n_out;
	unsigned o_out;
	isl_size n_div;
	unsigned o_div;
	isl_int gcd;

	n_div = isl_basic_map_dim(hull, isl_dim_div);
	n_out = isl_basic_map_dim(hull, isl_dim_out);
	if (n_div < 0 || n_out < 0)
		goto error;

	if (n_div == 0) {
		isl_basic_map_free(hull);
		return pw_multi_aff_from_map_check_div(map);
	}

	isl_int_init(gcd);

	o_div = isl_basic_map_offset(hull, isl_dim_div);
	o_out = isl_basic_map_offset(hull, isl_dim_out);

	for (i = 0; i < n_out; ++i) {
		for (j = 0; j < hull->n_eq; ++j) {
			isl_int *eq = hull->eq[j];
			isl_pw_multi_aff *res;

			if (!isl_int_is_one(eq[o_out + i]) &&
			    !isl_int_is_negone(eq[o_out + i]))
				continue;
			if (isl_seq_first_non_zero(eq + o_out, i) != -1)
				continue;
			if (isl_seq_first_non_zero(eq + o_out + i + 1,
						    n_out - (i + 1)) != -1)
				continue;
			isl_seq_gcd(eq + o_div, n_div, &gcd);
			if (isl_int_is_zero(gcd))
				continue;
			if (isl_int_is_one(gcd))
				continue;

			res = pw_multi_aff_from_map_stride(map, hull,
								i, j, gcd);
			isl_int_clear(gcd);
			return res;
		}
	}

	isl_int_clear(gcd);
	isl_basic_map_free(hull);
	return pw_multi_aff_from_map_check_div(map);
error:
	isl_map_free(map);
	isl_basic_map_free(hull);
	return NULL;
}

/* Try and create an isl_pw_multi_aff that is equivalent to the given isl_map.
 *
 * As a special case, we first check if all output dimensions are uniquely
 * defined in terms of the parameters and input dimensions over the entire
 * domain.  If so, we extract the desired isl_pw_multi_aff directly
 * from the affine hull of "map" and its domain.
 *
 * Otherwise, continue with pw_multi_aff_from_map_check_strides for more
 * special cases.
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_from_map(__isl_take isl_map *map)
{
	isl_bool sv;
	isl_size n;
	isl_basic_map *hull;

	n = isl_map_n_basic_map(map);
	if (n < 0)
		goto error;

	if (n == 1) {
		hull = isl_map_unshifted_simple_hull(isl_map_copy(map));
		hull = isl_basic_map_plain_affine_hull(hull);
		sv = isl_basic_map_plain_is_single_valued(hull);
		if (sv >= 0 && sv)
			return plain_pw_multi_aff_from_map(isl_map_domain(map),
							    hull);
		isl_basic_map_free(hull);
	}
	map = isl_map_detect_equalities(map);
	hull = isl_map_unshifted_simple_hull(isl_map_copy(map));
	sv = isl_basic_map_plain_is_single_valued(hull);
	if (sv >= 0 && sv)
		return plain_pw_multi_aff_from_map(isl_map_domain(map), hull);
	if (sv >= 0)
		return pw_multi_aff_from_map_check_strides(map, hull);
	isl_basic_map_free(hull);
error:
	isl_map_free(map);
	return NULL;
}

/* This function performs the same operation as isl_pw_multi_aff_from_map,
 * but is considered as a function on an isl_map when exported.
 */
__isl_give isl_pw_multi_aff *isl_map_as_pw_multi_aff(__isl_take isl_map *map)
{
	return isl_pw_multi_aff_from_map(map);
}

__isl_give isl_pw_multi_aff *isl_pw_multi_aff_from_set(__isl_take isl_set *set)
{
	return isl_pw_multi_aff_from_map(set);
}

/* This function performs the same operation as isl_pw_multi_aff_from_set,
 * but is considered as a function on an isl_set when exported.
 */
__isl_give isl_pw_multi_aff *isl_set_as_pw_multi_aff(__isl_take isl_set *set)
{
	return isl_pw_multi_aff_from_set(set);
}

/* Convert "map" into an isl_pw_multi_aff (if possible) and
 * add it to *user.
 */
static isl_stat pw_multi_aff_from_map(__isl_take isl_map *map, void *user)
{
	isl_union_pw_multi_aff **upma = user;
	isl_pw_multi_aff *pma;

	pma = isl_pw_multi_aff_from_map(map);
	*upma = isl_union_pw_multi_aff_add_pw_multi_aff(*upma, pma);

	return *upma ? isl_stat_ok : isl_stat_error;
}

/* Create an isl_union_pw_multi_aff with the given isl_aff on a universe
 * domain.
 */
__isl_give isl_union_pw_multi_aff *isl_union_pw_multi_aff_from_aff(
	__isl_take isl_aff *aff)
{
	isl_multi_aff *ma;
	isl_pw_multi_aff *pma;

	ma = isl_multi_aff_from_aff(aff);
	pma = isl_pw_multi_aff_from_multi_aff(ma);
	return isl_union_pw_multi_aff_from_pw_multi_aff(pma);
}

/* Try and create an isl_union_pw_multi_aff that is equivalent
 * to the given isl_union_map.
 * The isl_union_map is required to be single-valued in each space.
 * Otherwise, an error is produced.
 */
__isl_give isl_union_pw_multi_aff *isl_union_pw_multi_aff_from_union_map(
	__isl_take isl_union_map *umap)
{
	isl_space *space;
	isl_union_pw_multi_aff *upma;

	space = isl_union_map_get_space(umap);
	upma = isl_union_pw_multi_aff_empty(space);
	if (isl_union_map_foreach_map(umap, &pw_multi_aff_from_map, &upma) < 0)
		upma = isl_union_pw_multi_aff_free(upma);
	isl_union_map_free(umap);

	return upma;
}

/* This function performs the same operation as
 * isl_union_pw_multi_aff_from_union_map,
 * but is considered as a function on an isl_union_map when exported.
 */
__isl_give isl_union_pw_multi_aff *isl_union_map_as_union_pw_multi_aff(
	__isl_take isl_union_map *umap)
{
	return isl_union_pw_multi_aff_from_union_map(umap);
}

/* Try and create an isl_union_pw_multi_aff that is equivalent
 * to the given isl_union_set.
 * The isl_union_set is required to be a singleton in each space.
 * Otherwise, an error is produced.
 */
__isl_give isl_union_pw_multi_aff *isl_union_pw_multi_aff_from_union_set(
	__isl_take isl_union_set *uset)
{
	return isl_union_pw_multi_aff_from_union_map(uset);
}

/* Return the piecewise affine expression "set ? 1 : 0".
 */
__isl_give isl_pw_aff *isl_set_indicator_function(__isl_take isl_set *set)
{
	isl_pw_aff *pa;
	isl_space *space = isl_set_get_space(set);
	isl_local_space *ls = isl_local_space_from_space(space);
	isl_aff *zero = isl_aff_zero_on_domain(isl_local_space_copy(ls));
	isl_aff *one = isl_aff_zero_on_domain(ls);

	one = isl_aff_add_constant_si(one, 1);
	pa = isl_pw_aff_alloc(isl_set_copy(set), one);
	set = isl_set_complement(set);
	pa = isl_pw_aff_add_disjoint(pa, isl_pw_aff_alloc(set, zero));

	return pa;
}

/* Plug in "subs" for dimension "type", "pos" of "aff".
 *
 * Let i be the dimension to replace and let "subs" be of the form
 *
 *	f/d
 *
 * and "aff" of the form
 *
 *	(a i + g)/m
 *
 * The result is
 *
 *	(a f + d g')/(m d)
 *
 * where g' is the result of plugging in "subs" in each of the integer
 * divisions in g.
 */
__isl_give isl_aff *isl_aff_substitute(__isl_take isl_aff *aff,
	enum isl_dim_type type, unsigned pos, __isl_keep isl_aff *subs)
{
	isl_ctx *ctx;
	isl_int v;
	isl_size n_div;

	aff = isl_aff_cow(aff);
	if (!aff || !subs)
		return isl_aff_free(aff);

	ctx = isl_aff_get_ctx(aff);
	if (!isl_space_is_equal(aff->ls->dim, subs->ls->dim))
		isl_die(ctx, isl_error_invalid,
			"spaces don't match", return isl_aff_free(aff));
	n_div = isl_aff_domain_dim(subs, isl_dim_div);
	if (n_div < 0)
		return isl_aff_free(aff);
	if (n_div != 0)
		isl_die(ctx, isl_error_unsupported,
			"cannot handle divs yet", return isl_aff_free(aff));

	aff->ls = isl_local_space_substitute(aff->ls, type, pos, subs);
	if (!aff->ls)
		return isl_aff_free(aff);

	aff->v = isl_vec_cow(aff->v);
	if (!aff->v)
		return isl_aff_free(aff);

	pos += isl_local_space_offset(aff->ls, type);

	isl_int_init(v);
	isl_seq_substitute(aff->v->el, pos, subs->v->el,
			    aff->v->size, subs->v->size, v);
	isl_int_clear(v);

	return aff;
}

/* Plug in "subs" for dimension "type", "pos" in each of the affine
 * expressions in "maff".
 */
__isl_give isl_multi_aff *isl_multi_aff_substitute(
	__isl_take isl_multi_aff *maff, enum isl_dim_type type, unsigned pos,
	__isl_keep isl_aff *subs)
{
	int i;

	maff = isl_multi_aff_cow(maff);
	if (!maff || !subs)
		return isl_multi_aff_free(maff);

	if (type == isl_dim_in)
		type = isl_dim_set;

	for (i = 0; i < maff->n; ++i) {
		maff->u.p[i] = isl_aff_substitute(maff->u.p[i],
						type, pos, subs);
		if (!maff->u.p[i])
			return isl_multi_aff_free(maff);
	}

	return maff;
}

/* Plug in "subs" for input dimension "pos" of "pma".
 *
 * pma is of the form
 *
 *	A_i(v) -> M_i(v)
 *
 * while subs is of the form
 *
 *	v' = B_j(v) -> S_j
 *
 * Each pair i,j such that C_ij = A_i \cap B_i is non-empty
 * has a contribution in the result, in particular
 *
 *	C_ij(S_j) -> M_i(S_j)
 *
 * Note that plugging in S_j in C_ij may also result in an empty set
 * and this contribution should simply be discarded.
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_substitute(
	__isl_take isl_pw_multi_aff *pma, unsigned pos,
	__isl_keep isl_pw_aff *subs)
{
	int i, j, n;
	isl_pw_multi_aff *res;

	if (!pma || !subs)
		return isl_pw_multi_aff_free(pma);

	n = pma->n * subs->n;
	res = isl_pw_multi_aff_alloc_size(isl_space_copy(pma->dim), n);

	for (i = 0; i < pma->n; ++i) {
		for (j = 0; j < subs->n; ++j) {
			isl_set *common;
			isl_multi_aff *res_ij;
			int empty;

			common = isl_set_intersect(
					isl_set_copy(pma->p[i].set),
					isl_set_copy(subs->p[j].set));
			common = isl_set_substitute(common,
					pos, subs->p[j].aff);
			empty = isl_set_plain_is_empty(common);
			if (empty < 0 || empty) {
				isl_set_free(common);
				if (empty < 0)
					goto error;
				continue;
			}

			res_ij = isl_multi_aff_substitute(
					isl_multi_aff_copy(pma->p[i].maff),
					isl_dim_in, pos, subs->p[j].aff);

			res = isl_pw_multi_aff_add_piece(res, common, res_ij);
		}
	}

	isl_pw_multi_aff_free(pma);
	return res;
error:
	isl_pw_multi_aff_free(pma);
	isl_pw_multi_aff_free(res);
	return NULL;
}

/* Compute the preimage of a range of dimensions in the affine expression "src"
 * under "ma" and put the result in "dst".  The number of dimensions in "src"
 * that precede the range is given by "n_before".  The number of dimensions
 * in the range is given by the number of output dimensions of "ma".
 * The number of dimensions that follow the range is given by "n_after".
 * If "has_denom" is set (to one),
 * then "src" and "dst" have an extra initial denominator.
 * "n_div_ma" is the number of existentials in "ma"
 * "n_div_bset" is the number of existentials in "src"
 * The resulting "dst" (which is assumed to have been allocated by
 * the caller) contains coefficients for both sets of existentials,
 * first those in "ma" and then those in "src".
 * f, c1, c2 and g are temporary objects that have been initialized
 * by the caller.
 *
 * Let src represent the expression
 *
 *	(a(p) + f_u u + b v + f_w w + c(divs))/d
 *
 * and let ma represent the expressions
 *
 *	v_i = (r_i(p) + s_i(y) + t_i(divs'))/m_i
 *
 * We start out with the following expression for dst:
 *
 *	(a(p) + f_u u + 0 y + f_w w + 0 divs' + c(divs) + f \sum_i b_i v_i)/d
 *
 * with the multiplication factor f initially equal to 1
 * and f \sum_i b_i v_i kept separately.
 * For each x_i that we substitute, we multiply the numerator
 * (and denominator) of dst by c_1 = m_i and add the numerator
 * of the x_i expression multiplied by c_2 = f b_i,
 * after removing the common factors of c_1 and c_2.
 * The multiplication factor f also needs to be multiplied by c_1
 * for the next x_j, j > i.
 */
isl_stat isl_seq_preimage(isl_int *dst, isl_int *src,
	__isl_keep isl_multi_aff *ma, int n_before, int n_after,
	int n_div_ma, int n_div_bmap,
	isl_int f, isl_int c1, isl_int c2, isl_int g, int has_denom)
{
	int i;
	isl_size n_param, n_in, n_out;
	int o_dst, o_src;

	n_param = isl_multi_aff_dim(ma, isl_dim_param);
	n_in = isl_multi_aff_dim(ma, isl_dim_in);
	n_out = isl_multi_aff_dim(ma, isl_dim_out);
	if (n_param < 0 || n_in < 0 || n_out < 0)
		return isl_stat_error;

	isl_seq_cpy(dst, src, has_denom + 1 + n_param + n_before);
	o_dst = o_src = has_denom + 1 + n_param + n_before;
	isl_seq_clr(dst + o_dst, n_in);
	o_dst += n_in;
	o_src += n_out;
	isl_seq_cpy(dst + o_dst, src + o_src, n_after);
	o_dst += n_after;
	o_src += n_after;
	isl_seq_clr(dst + o_dst, n_div_ma);
	o_dst += n_div_ma;
	isl_seq_cpy(dst + o_dst, src + o_src, n_div_bmap);

	isl_int_set_si(f, 1);

	for (i = 0; i < n_out; ++i) {
		int offset = has_denom + 1 + n_param + n_before + i;

		if (isl_int_is_zero(src[offset]))
			continue;
		isl_int_set(c1, ma->u.p[i]->v->el[0]);
		isl_int_mul(c2, f, src[offset]);
		isl_int_gcd(g, c1, c2);
		isl_int_divexact(c1, c1, g);
		isl_int_divexact(c2, c2, g);

		isl_int_mul(f, f, c1);
		o_dst = has_denom;
		o_src = 1;
		isl_seq_combine(dst + o_dst, c1, dst + o_dst,
				c2, ma->u.p[i]->v->el + o_src, 1 + n_param);
		o_dst += 1 + n_param;
		o_src += 1 + n_param;
		isl_seq_scale(dst + o_dst, dst + o_dst, c1, n_before);
		o_dst += n_before;
		isl_seq_combine(dst + o_dst, c1, dst + o_dst,
				c2, ma->u.p[i]->v->el + o_src, n_in);
		o_dst += n_in;
		o_src += n_in;
		isl_seq_scale(dst + o_dst, dst + o_dst, c1, n_after);
		o_dst += n_after;
		isl_seq_combine(dst + o_dst, c1, dst + o_dst,
				c2, ma->u.p[i]->v->el + o_src, n_div_ma);
		o_dst += n_div_ma;
		o_src += n_div_ma;
		isl_seq_scale(dst + o_dst, dst + o_dst, c1, n_div_bmap);
		if (has_denom)
			isl_int_mul(dst[0], dst[0], c1);
	}

	return isl_stat_ok;
}

/* Compute the pullback of "aff" by the function represented by "ma".
 * In other words, plug in "ma" in "aff".  The result is an affine expression
 * defined over the domain space of "ma".
 *
 * If "aff" is represented by
 *
 *	(a(p) + b x + c(divs))/d
 *
 * and ma is represented by
 *
 *	x = D(p) + F(y) + G(divs')
 *
 * then the result is
 *
 *	(a(p) + b D(p) + b F(y) + b G(divs') + c(divs))/d
 *
 * The divs in the local space of the input are similarly adjusted
 * through a call to isl_local_space_preimage_multi_aff.
 */
__isl_give isl_aff *isl_aff_pullback_multi_aff(__isl_take isl_aff *aff,
	__isl_take isl_multi_aff *ma)
{
	isl_aff *res = NULL;
	isl_local_space *ls;
	isl_size n_div_aff, n_div_ma;
	isl_int f, c1, c2, g;

	ma = isl_multi_aff_align_divs(ma);
	if (!aff || !ma)
		goto error;

	n_div_aff = isl_aff_dim(aff, isl_dim_div);
	n_div_ma = ma->n ? isl_aff_dim(ma->u.p[0], isl_dim_div) : 0;
	if (n_div_aff < 0 || n_div_ma < 0)
		goto error;

	ls = isl_aff_get_domain_local_space(aff);
	ls = isl_local_space_preimage_multi_aff(ls, isl_multi_aff_copy(ma));
	res = isl_aff_alloc(ls);
	if (!res)
		goto error;

	isl_int_init(f);
	isl_int_init(c1);
	isl_int_init(c2);
	isl_int_init(g);

	if (isl_seq_preimage(res->v->el, aff->v->el, ma, 0, 0,
			    n_div_ma, n_div_aff, f, c1, c2, g, 1) < 0)
		res = isl_aff_free(res);

	isl_int_clear(f);
	isl_int_clear(c1);
	isl_int_clear(c2);
	isl_int_clear(g);

	isl_aff_free(aff);
	isl_multi_aff_free(ma);
	res = isl_aff_normalize(res);
	return res;
error:
	isl_aff_free(aff);
	isl_multi_aff_free(ma);
	isl_aff_free(res);
	return NULL;
}

/* Compute the pullback of "aff1" by the function represented by "aff2".
 * In other words, plug in "aff2" in "aff1".  The result is an affine expression
 * defined over the domain space of "aff1".
 *
 * The domain of "aff1" should match the range of "aff2", which means
 * that it should be single-dimensional.
 */
__isl_give isl_aff *isl_aff_pullback_aff(__isl_take isl_aff *aff1,
	__isl_take isl_aff *aff2)
{
	isl_multi_aff *ma;

	ma = isl_multi_aff_from_aff(aff2);
	return isl_aff_pullback_multi_aff(aff1, ma);
}

/* Compute the pullback of "ma1" by the function represented by "ma2".
 * In other words, plug in "ma2" in "ma1".
 */
__isl_give isl_multi_aff *isl_multi_aff_pullback_multi_aff(
	__isl_take isl_multi_aff *ma1, __isl_take isl_multi_aff *ma2)
{
	int i;
	isl_space *space = NULL;

	isl_multi_aff_align_params_bin(&ma1, &ma2);
	ma2 = isl_multi_aff_align_divs(ma2);
	ma1 = isl_multi_aff_cow(ma1);
	if (!ma1 || !ma2)
		goto error;

	space = isl_space_join(isl_multi_aff_get_space(ma2),
				isl_multi_aff_get_space(ma1));

	for (i = 0; i < ma1->n; ++i) {
		ma1->u.p[i] = isl_aff_pullback_multi_aff(ma1->u.p[i],
						    isl_multi_aff_copy(ma2));
		if (!ma1->u.p[i])
			goto error;
	}

	ma1 = isl_multi_aff_reset_space(ma1, space);
	isl_multi_aff_free(ma2);
	return ma1;
error:
	isl_space_free(space);
	isl_multi_aff_free(ma2);
	isl_multi_aff_free(ma1);
	return NULL;
}

/* Extend the local space of "dst" to include the divs
 * in the local space of "src".
 *
 * If "src" does not have any divs or if the local spaces of "dst" and
 * "src" are the same, then no extension is required.
 */
__isl_give isl_aff *isl_aff_align_divs(__isl_take isl_aff *dst,
	__isl_keep isl_aff *src)
{
	isl_ctx *ctx;
	isl_size src_n_div, dst_n_div;
	int *exp1 = NULL;
	int *exp2 = NULL;
	isl_bool equal;
	isl_mat *div;

	if (!src || !dst)
		return isl_aff_free(dst);

	ctx = isl_aff_get_ctx(src);
	equal = isl_local_space_has_equal_space(src->ls, dst->ls);
	if (equal < 0)
		return isl_aff_free(dst);
	if (!equal)
		isl_die(ctx, isl_error_invalid,
			"spaces don't match", goto error);

	src_n_div = isl_aff_domain_dim(src, isl_dim_div);
	dst_n_div = isl_aff_domain_dim(dst, isl_dim_div);
	if (src_n_div == 0)
		return dst;
	equal = isl_local_space_is_equal(src->ls, dst->ls);
	if (equal < 0 || src_n_div < 0 || dst_n_div < 0)
		return isl_aff_free(dst);
	if (equal)
		return dst;

	exp1 = isl_alloc_array(ctx, int, src_n_div);
	exp2 = isl_alloc_array(ctx, int, dst_n_div);
	if (!exp1 || (dst_n_div && !exp2))
		goto error;

	div = isl_merge_divs(src->ls->div, dst->ls->div, exp1, exp2);
	dst = isl_aff_expand_divs(dst, div, exp2);
	free(exp1);
	free(exp2);

	return dst;
error:
	free(exp1);
	free(exp2);
	return isl_aff_free(dst);
}

/* Adjust the local spaces of the affine expressions in "maff"
 * such that they all have the save divs.
 */
__isl_give isl_multi_aff *isl_multi_aff_align_divs(
	__isl_take isl_multi_aff *maff)
{
	int i;

	if (!maff)
		return NULL;
	if (maff->n == 0)
		return maff;
	maff = isl_multi_aff_cow(maff);
	if (!maff)
		return NULL;

	for (i = 1; i < maff->n; ++i)
		maff->u.p[0] = isl_aff_align_divs(maff->u.p[0], maff->u.p[i]);
	for (i = 1; i < maff->n; ++i) {
		maff->u.p[i] = isl_aff_align_divs(maff->u.p[i], maff->u.p[0]);
		if (!maff->u.p[i])
			return isl_multi_aff_free(maff);
	}

	return maff;
}

__isl_give isl_aff *isl_aff_lift(__isl_take isl_aff *aff)
{
	aff = isl_aff_cow(aff);
	if (!aff)
		return NULL;

	aff->ls = isl_local_space_lift(aff->ls);
	if (!aff->ls)
		return isl_aff_free(aff);

	return aff;
}

/* Lift "maff" to a space with extra dimensions such that the result
 * has no more existentially quantified variables.
 * If "ls" is not NULL, then *ls is assigned the local space that lies
 * at the basis of the lifting applied to "maff".
 */
__isl_give isl_multi_aff *isl_multi_aff_lift(__isl_take isl_multi_aff *maff,
	__isl_give isl_local_space **ls)
{
	int i;
	isl_space *space;
	isl_size n_div;

	if (ls)
		*ls = NULL;

	if (!maff)
		return NULL;

	if (maff->n == 0) {
		if (ls) {
			isl_space *space = isl_multi_aff_get_domain_space(maff);
			*ls = isl_local_space_from_space(space);
			if (!*ls)
				return isl_multi_aff_free(maff);
		}
		return maff;
	}

	maff = isl_multi_aff_cow(maff);
	maff = isl_multi_aff_align_divs(maff);
	if (!maff)
		return NULL;

	n_div = isl_aff_dim(maff->u.p[0], isl_dim_div);
	if (n_div < 0)
		return isl_multi_aff_free(maff);
	space = isl_multi_aff_get_space(maff);
	space = isl_space_lift(isl_space_domain(space), n_div);
	space = isl_space_extend_domain_with_range(space,
						isl_multi_aff_get_space(maff));
	if (!space)
		return isl_multi_aff_free(maff);
	isl_space_free(maff->space);
	maff->space = space;

	if (ls) {
		*ls = isl_aff_get_domain_local_space(maff->u.p[0]);
		if (!*ls)
			return isl_multi_aff_free(maff);
	}

	for (i = 0; i < maff->n; ++i) {
		maff->u.p[i] = isl_aff_lift(maff->u.p[i]);
		if (!maff->u.p[i])
			goto error;
	}

	return maff;
error:
	if (ls)
		isl_local_space_free(*ls);
	return isl_multi_aff_free(maff);
}

#undef TYPE
#define TYPE	isl_pw_multi_aff
static
#include "check_type_range_templ.c"

/* Extract an isl_pw_aff corresponding to output dimension "pos" of "pma".
 */
__isl_give isl_pw_aff *isl_pw_multi_aff_get_at(
	__isl_keep isl_pw_multi_aff *pma, int pos)
{
	int i;
	isl_size n_out;
	isl_space *space;
	isl_pw_aff *pa;

	if (isl_pw_multi_aff_check_range(pma, isl_dim_out, pos, 1) < 0)
		return NULL;

	n_out = isl_pw_multi_aff_dim(pma, isl_dim_out);
	if (n_out < 0)
		return NULL;

	space = isl_pw_multi_aff_get_space(pma);
	space = isl_space_drop_dims(space, isl_dim_out,
				    pos + 1, n_out - pos - 1);
	space = isl_space_drop_dims(space, isl_dim_out, 0, pos);

	pa = isl_pw_aff_alloc_size(space, pma->n);
	for (i = 0; i < pma->n; ++i) {
		isl_aff *aff;
		aff = isl_multi_aff_get_aff(pma->p[i].maff, pos);
		pa = isl_pw_aff_add_piece(pa, isl_set_copy(pma->p[i].set), aff);
	}

	return pa;
}

/* This is an alternative name for the function above.
 */
__isl_give isl_pw_aff *isl_pw_multi_aff_get_pw_aff(
	__isl_keep isl_pw_multi_aff *pma, int pos)
{
	return isl_pw_multi_aff_get_at(pma, pos);
}

/* Return an isl_pw_multi_aff with the given "set" as domain and
 * an unnamed zero-dimensional range.
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_from_domain(
	__isl_take isl_set *set)
{
	isl_multi_aff *ma;
	isl_space *space;

	space = isl_set_get_space(set);
	space = isl_space_from_domain(space);
	ma = isl_multi_aff_zero(space);
	return isl_pw_multi_aff_alloc(set, ma);
}

/* Add an isl_pw_multi_aff with the given "set" as domain and
 * an unnamed zero-dimensional range to *user.
 */
static isl_stat add_pw_multi_aff_from_domain(__isl_take isl_set *set,
	void *user)
{
	isl_union_pw_multi_aff **upma = user;
	isl_pw_multi_aff *pma;

	pma = isl_pw_multi_aff_from_domain(set);
	*upma = isl_union_pw_multi_aff_add_pw_multi_aff(*upma, pma);

	return isl_stat_ok;
}

/* Return an isl_union_pw_multi_aff with the given "uset" as domain and
 * an unnamed zero-dimensional range.
 */
__isl_give isl_union_pw_multi_aff *isl_union_pw_multi_aff_from_domain(
	__isl_take isl_union_set *uset)
{
	isl_space *space;
	isl_union_pw_multi_aff *upma;

	if (!uset)
		return NULL;

	space = isl_union_set_get_space(uset);
	upma = isl_union_pw_multi_aff_empty(space);

	if (isl_union_set_foreach_set(uset,
				    &add_pw_multi_aff_from_domain, &upma) < 0)
		goto error;

	isl_union_set_free(uset);
	return upma;
error:
	isl_union_set_free(uset);
	isl_union_pw_multi_aff_free(upma);
	return NULL;
}

/* Local data for bin_entry and the callback "fn".
 */
struct isl_union_pw_multi_aff_bin_data {
	isl_union_pw_multi_aff *upma2;
	isl_union_pw_multi_aff *res;
	isl_pw_multi_aff *pma;
	isl_stat (*fn)(__isl_take isl_pw_multi_aff *pma, void *user);
};

/* Given an isl_pw_multi_aff from upma1, store it in data->pma
 * and call data->fn for each isl_pw_multi_aff in data->upma2.
 */
static isl_stat bin_entry(__isl_take isl_pw_multi_aff *pma, void *user)
{
	struct isl_union_pw_multi_aff_bin_data *data = user;
	isl_stat r;

	data->pma = pma;
	r = isl_union_pw_multi_aff_foreach_pw_multi_aff(data->upma2,
				   data->fn, data);
	isl_pw_multi_aff_free(pma);

	return r;
}

/* Call "fn" on each pair of isl_pw_multi_affs in "upma1" and "upma2".
 * The isl_pw_multi_aff from upma1 is stored in data->pma (where data is
 * passed as user field) and the isl_pw_multi_aff from upma2 is available
 * as *entry.  The callback should adjust data->res if desired.
 */
static __isl_give isl_union_pw_multi_aff *bin_op(
	__isl_take isl_union_pw_multi_aff *upma1,
	__isl_take isl_union_pw_multi_aff *upma2,
	isl_stat (*fn)(__isl_take isl_pw_multi_aff *pma, void *user))
{
	isl_space *space;
	struct isl_union_pw_multi_aff_bin_data data = { NULL, NULL, NULL, fn };

	space = isl_union_pw_multi_aff_get_space(upma2);
	upma1 = isl_union_pw_multi_aff_align_params(upma1, space);
	space = isl_union_pw_multi_aff_get_space(upma1);
	upma2 = isl_union_pw_multi_aff_align_params(upma2, space);

	if (!upma1 || !upma2)
		goto error;

	data.upma2 = upma2;
	data.res = isl_union_pw_multi_aff_alloc_same_size(upma1);
	if (isl_union_pw_multi_aff_foreach_pw_multi_aff(upma1,
				   &bin_entry, &data) < 0)
		goto error;

	isl_union_pw_multi_aff_free(upma1);
	isl_union_pw_multi_aff_free(upma2);
	return data.res;
error:
	isl_union_pw_multi_aff_free(upma1);
	isl_union_pw_multi_aff_free(upma2);
	isl_union_pw_multi_aff_free(data.res);
	return NULL;
}

/* Given two isl_pw_multi_affs A -> B and C -> D,
 * construct an isl_pw_multi_aff (A * C) -> [B -> D].
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_range_product(
	__isl_take isl_pw_multi_aff *pma1, __isl_take isl_pw_multi_aff *pma2)
{
	isl_space *space;

	isl_pw_multi_aff_align_params_bin(&pma1, &pma2);
	space = isl_space_range_product(isl_pw_multi_aff_get_space(pma1),
					isl_pw_multi_aff_get_space(pma2));
	return isl_pw_multi_aff_on_shared_domain_in(pma1, pma2, space,
					    &isl_multi_aff_range_product);
}

/* Given two isl_pw_multi_affs A -> B and C -> D,
 * construct an isl_pw_multi_aff (A * C) -> (B, D).
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_flat_range_product(
	__isl_take isl_pw_multi_aff *pma1, __isl_take isl_pw_multi_aff *pma2)
{
	isl_space *space;

	isl_pw_multi_aff_align_params_bin(&pma1, &pma2);
	space = isl_space_range_product(isl_pw_multi_aff_get_space(pma1),
					isl_pw_multi_aff_get_space(pma2));
	space = isl_space_flatten_range(space);
	return isl_pw_multi_aff_on_shared_domain_in(pma1, pma2, space,
					    &isl_multi_aff_flat_range_product);
}

/* If data->pma and "pma2" have the same domain space, then use "range_product"
 * to compute some form of range product and add the result to data->res.
 */
static isl_stat gen_range_product_entry(__isl_take isl_pw_multi_aff *pma2,
	__isl_give isl_pw_multi_aff *(*range_product)(
		__isl_take isl_pw_multi_aff *pma1,
		__isl_take isl_pw_multi_aff *pma2),
	void *user)
{
	struct isl_union_pw_multi_aff_bin_data *data = user;
	isl_bool match;
	isl_space *space1, *space2;

	space1 = isl_pw_multi_aff_peek_space(data->pma);
	space2 = isl_pw_multi_aff_peek_space(pma2);
	match = isl_space_tuple_is_equal(space1, isl_dim_in,
					space2, isl_dim_in);
	if (match < 0 || !match) {
		isl_pw_multi_aff_free(pma2);
		return match < 0 ? isl_stat_error : isl_stat_ok;
	}

	pma2 = range_product(isl_pw_multi_aff_copy(data->pma), pma2);

	data->res = isl_union_pw_multi_aff_add_pw_multi_aff(data->res, pma2);

	return isl_stat_ok;
}

/* If data->pma and "pma2" have the same domain space, then compute
 * their flat range product and add the result to data->res.
 */
static isl_stat flat_range_product_entry(__isl_take isl_pw_multi_aff *pma2,
	void *user)
{
	return gen_range_product_entry(pma2,
				&isl_pw_multi_aff_flat_range_product, user);
}

/* Given two isl_union_pw_multi_affs A -> B and C -> D,
 * construct an isl_union_pw_multi_aff (A * C) -> (B, D).
 */
__isl_give isl_union_pw_multi_aff *isl_union_pw_multi_aff_flat_range_product(
	__isl_take isl_union_pw_multi_aff *upma1,
	__isl_take isl_union_pw_multi_aff *upma2)
{
	return bin_op(upma1, upma2, &flat_range_product_entry);
}

/* If data->pma and "pma2" have the same domain space, then compute
 * their range product and add the result to data->res.
 */
static isl_stat range_product_entry(__isl_take isl_pw_multi_aff *pma2,
	void *user)
{
	return gen_range_product_entry(pma2,
				&isl_pw_multi_aff_range_product, user);
}

/* Given two isl_union_pw_multi_affs A -> B and C -> D,
 * construct an isl_union_pw_multi_aff (A * C) -> [B -> D].
 */
__isl_give isl_union_pw_multi_aff *isl_union_pw_multi_aff_range_product(
	__isl_take isl_union_pw_multi_aff *upma1,
	__isl_take isl_union_pw_multi_aff *upma2)
{
	return bin_op(upma1, upma2, &range_product_entry);
}

/* Replace the affine expressions at position "pos" in "pma" by "pa".
 * The parameters are assumed to have been aligned.
 *
 * The implementation essentially performs an isl_pw_*_on_shared_domain,
 * except that it works on two different isl_pw_* types.
 */
static __isl_give isl_pw_multi_aff *pw_multi_aff_set_pw_aff(
	__isl_take isl_pw_multi_aff *pma, unsigned pos,
	__isl_take isl_pw_aff *pa)
{
	int i, j, n;
	isl_pw_multi_aff *res = NULL;

	if (!pma || !pa)
		goto error;

	if (!isl_space_tuple_is_equal(pma->dim, isl_dim_in,
					pa->dim, isl_dim_in))
		isl_die(isl_pw_multi_aff_get_ctx(pma), isl_error_invalid,
			"domains don't match", goto error);
	if (isl_pw_multi_aff_check_range(pma, isl_dim_out, pos, 1) < 0)
		goto error;

	n = pma->n * pa->n;
	res = isl_pw_multi_aff_alloc_size(isl_pw_multi_aff_get_space(pma), n);

	for (i = 0; i < pma->n; ++i) {
		for (j = 0; j < pa->n; ++j) {
			isl_set *common;
			isl_multi_aff *res_ij;
			int empty;

			common = isl_set_intersect(isl_set_copy(pma->p[i].set),
						   isl_set_copy(pa->p[j].set));
			empty = isl_set_plain_is_empty(common);
			if (empty < 0 || empty) {
				isl_set_free(common);
				if (empty < 0)
					goto error;
				continue;
			}

			res_ij = isl_multi_aff_set_aff(
					isl_multi_aff_copy(pma->p[i].maff), pos,
					isl_aff_copy(pa->p[j].aff));
			res_ij = isl_multi_aff_gist(res_ij,
					isl_set_copy(common));

			res = isl_pw_multi_aff_add_piece(res, common, res_ij);
		}
	}

	isl_pw_multi_aff_free(pma);
	isl_pw_aff_free(pa);
	return res;
error:
	isl_pw_multi_aff_free(pma);
	isl_pw_aff_free(pa);
	return isl_pw_multi_aff_free(res);
}

/* Replace the affine expressions at position "pos" in "pma" by "pa".
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_set_pw_aff(
	__isl_take isl_pw_multi_aff *pma, unsigned pos,
	__isl_take isl_pw_aff *pa)
{
	isl_bool equal_params;

	if (!pma || !pa)
		goto error;
	equal_params = isl_space_has_equal_params(pma->dim, pa->dim);
	if (equal_params < 0)
		goto error;
	if (equal_params)
		return pw_multi_aff_set_pw_aff(pma, pos, pa);
	if (isl_pw_multi_aff_check_named_params(pma) < 0 ||
	    isl_pw_aff_check_named_params(pa) < 0)
		goto error;
	pma = isl_pw_multi_aff_align_params(pma, isl_pw_aff_get_space(pa));
	pa = isl_pw_aff_align_params(pa, isl_pw_multi_aff_get_space(pma));
	return pw_multi_aff_set_pw_aff(pma, pos, pa);
error:
	isl_pw_multi_aff_free(pma);
	isl_pw_aff_free(pa);
	return NULL;
}

/* Do the parameters of "pa" match those of "space"?
 */
isl_bool isl_pw_aff_matching_params(__isl_keep isl_pw_aff *pa,
	__isl_keep isl_space *space)
{
	isl_space *pa_space;
	isl_bool match;

	if (!pa || !space)
		return isl_bool_error;

	pa_space = isl_pw_aff_get_space(pa);

	match = isl_space_has_equal_params(space, pa_space);

	isl_space_free(pa_space);
	return match;
}

/* Check that the domain space of "pa" matches "space".
 */
isl_stat isl_pw_aff_check_match_domain_space(__isl_keep isl_pw_aff *pa,
	__isl_keep isl_space *space)
{
	isl_space *pa_space;
	isl_bool match;

	if (!pa || !space)
		return isl_stat_error;

	pa_space = isl_pw_aff_get_space(pa);

	match = isl_space_has_equal_params(space, pa_space);
	if (match < 0)
		goto error;
	if (!match)
		isl_die(isl_pw_aff_get_ctx(pa), isl_error_invalid,
			"parameters don't match", goto error);
	match = isl_space_tuple_is_equal(space, isl_dim_in,
					pa_space, isl_dim_in);
	if (match < 0)
		goto error;
	if (!match)
		isl_die(isl_pw_aff_get_ctx(pa), isl_error_invalid,
			"domains don't match", goto error);
	isl_space_free(pa_space);
	return isl_stat_ok;
error:
	isl_space_free(pa_space);
	return isl_stat_error;
}

#undef BASE
#define BASE pw_aff
#undef DOMBASE
#define DOMBASE set

#include <isl_multi_explicit_domain.c>
#include <isl_multi_pw_aff_explicit_domain.c>
#include <isl_multi_templ.c>
#include <isl_multi_add_constant_templ.c>
#include <isl_multi_apply_set.c>
#include <isl_multi_arith_templ.c>
#include <isl_multi_bind_templ.c>
#include <isl_multi_bind_domain_templ.c>
#include <isl_multi_coalesce.c>
#include <isl_multi_domain_templ.c>
#include <isl_multi_dim_id_templ.c>
#include <isl_multi_dims.c>
#include <isl_multi_from_base_templ.c>
#include <isl_multi_gist.c>
#include <isl_multi_hash.c>
#include <isl_multi_identity_templ.c>
#include <isl_multi_align_set.c>
#include <isl_multi_insert_domain_templ.c>
#include <isl_multi_intersect.c>
#include <isl_multi_min_max_templ.c>
#include <isl_multi_move_dims_templ.c>
#include <isl_multi_nan_templ.c>
#include <isl_multi_param_templ.c>
#include <isl_multi_product_templ.c>
#include <isl_multi_splice_templ.c>
#include <isl_multi_tuple_id_templ.c>
#include <isl_multi_union_add_templ.c>
#include <isl_multi_zero_templ.c>
#include <isl_multi_unbind_params_templ.c>

/* Is every element of "mpa" defined over a single universe domain?
 */
isl_bool isl_multi_pw_aff_isa_multi_aff(__isl_keep isl_multi_pw_aff *mpa)
{
	return isl_multi_pw_aff_every(mpa, &isl_pw_aff_isa_aff);
}

/* Given that every element of "mpa" is defined over a single universe domain,
 * return the corresponding base expressions.
 */
__isl_give isl_multi_aff *isl_multi_pw_aff_as_multi_aff(
	__isl_take isl_multi_pw_aff *mpa)
{
	int i;
	isl_size n;
	isl_multi_aff *ma;

	n = isl_multi_pw_aff_size(mpa);
	if (n < 0)
		mpa = isl_multi_pw_aff_free(mpa);
	ma = isl_multi_aff_alloc(isl_multi_pw_aff_get_space(mpa));
	for (i = 0; i < n; ++i) {
		isl_aff *aff;

		aff = isl_pw_aff_as_aff(isl_multi_pw_aff_get_at(mpa, i));
		ma = isl_multi_aff_set_aff(ma, i, aff);
	}
	isl_multi_pw_aff_free(mpa);
	return ma;
}

/* If "mpa" has an explicit domain, then intersect the domain of "map"
 * with this explicit domain.
 */
__isl_give isl_map *isl_map_intersect_multi_pw_aff_explicit_domain(
	__isl_take isl_map *map, __isl_keep isl_multi_pw_aff *mpa)
{
	isl_set *dom;

	if (!isl_multi_pw_aff_has_explicit_domain(mpa))
		return map;

	dom = isl_multi_pw_aff_domain(isl_multi_pw_aff_copy(mpa));
	map = isl_map_intersect_domain(map, dom);

	return map;
}

/* Are all elements of "mpa" piecewise constants?
 */
isl_bool isl_multi_pw_aff_is_cst(__isl_keep isl_multi_pw_aff *mpa)
{
	return isl_multi_pw_aff_every(mpa, &isl_pw_aff_is_cst);
}

/* Does "mpa" have a non-trivial explicit domain?
 *
 * The explicit domain, if present, is trivial if it represents
 * an (obviously) universe set.
 */
isl_bool isl_multi_pw_aff_has_non_trivial_domain(
	__isl_keep isl_multi_pw_aff *mpa)
{
	if (!mpa)
		return isl_bool_error;
	if (!isl_multi_pw_aff_has_explicit_domain(mpa))
		return isl_bool_false;
	return isl_bool_not(isl_set_plain_is_universe(mpa->u.dom));
}

#undef BASE
#define BASE	set

#include "isl_opt_mpa_templ.c"

/* Compute the minima of the set dimensions as a function of the
 * parameters, but independently of the other set dimensions.
 */
__isl_give isl_multi_pw_aff *isl_set_min_multi_pw_aff(__isl_take isl_set *set)
{
	return set_opt_mpa(set, &isl_set_dim_min);
}

/* Compute the maxima of the set dimensions as a function of the
 * parameters, but independently of the other set dimensions.
 */
__isl_give isl_multi_pw_aff *isl_set_max_multi_pw_aff(__isl_take isl_set *set)
{
	return set_opt_mpa(set, &isl_set_dim_max);
}

#undef BASE
#define BASE	map

#include "isl_opt_mpa_templ.c"

/* Compute the minima of the output dimensions as a function of the
 * parameters and input dimensions, but independently of
 * the other output dimensions.
 */
__isl_give isl_multi_pw_aff *isl_map_min_multi_pw_aff(__isl_take isl_map *map)
{
	return map_opt_mpa(map, &isl_map_dim_min);
}

/* Compute the maxima of the output dimensions as a function of the
 * parameters and input dimensions, but independently of
 * the other output dimensions.
 */
__isl_give isl_multi_pw_aff *isl_map_max_multi_pw_aff(__isl_take isl_map *map)
{
	return map_opt_mpa(map, &isl_map_dim_max);
}

/* Scale the elements of "pma" by the corresponding elements of "mv".
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_scale_multi_val(
	__isl_take isl_pw_multi_aff *pma, __isl_take isl_multi_val *mv)
{
	int i;
	isl_bool equal_params;

	pma = isl_pw_multi_aff_cow(pma);
	if (!pma || !mv)
		goto error;
	if (!isl_space_tuple_is_equal(pma->dim, isl_dim_out,
					mv->space, isl_dim_set))
		isl_die(isl_pw_multi_aff_get_ctx(pma), isl_error_invalid,
			"spaces don't match", goto error);
	equal_params = isl_space_has_equal_params(pma->dim, mv->space);
	if (equal_params < 0)
		goto error;
	if (!equal_params) {
		pma = isl_pw_multi_aff_align_params(pma,
					    isl_multi_val_get_space(mv));
		mv = isl_multi_val_align_params(mv,
					    isl_pw_multi_aff_get_space(pma));
		if (!pma || !mv)
			goto error;
	}

	for (i = 0; i < pma->n; ++i) {
		pma->p[i].maff = isl_multi_aff_scale_multi_val(pma->p[i].maff,
							isl_multi_val_copy(mv));
		if (!pma->p[i].maff)
			goto error;
	}

	isl_multi_val_free(mv);
	return pma;
error:
	isl_multi_val_free(mv);
	isl_pw_multi_aff_free(pma);
	return NULL;
}

/* This function is called for each entry of an isl_union_pw_multi_aff.
 * If the space of the entry matches that of data->mv,
 * then apply isl_pw_multi_aff_scale_multi_val and return the result.
 * Otherwise, return an empty isl_pw_multi_aff.
 */
static __isl_give isl_pw_multi_aff *union_pw_multi_aff_scale_multi_val_entry(
	__isl_take isl_pw_multi_aff *pma, void *user)
{
	isl_multi_val *mv = user;

	if (!pma)
		return NULL;
	if (!isl_space_tuple_is_equal(pma->dim, isl_dim_out,
				    mv->space, isl_dim_set)) {
		isl_space *space = isl_pw_multi_aff_get_space(pma);
		isl_pw_multi_aff_free(pma);
		return isl_pw_multi_aff_empty(space);
	}

	return isl_pw_multi_aff_scale_multi_val(pma, isl_multi_val_copy(mv));
}

/* Scale the elements of "upma" by the corresponding elements of "mv",
 * for those entries that match the space of "mv".
 */
__isl_give isl_union_pw_multi_aff *isl_union_pw_multi_aff_scale_multi_val(
	__isl_take isl_union_pw_multi_aff *upma, __isl_take isl_multi_val *mv)
{
	struct isl_union_pw_multi_aff_transform_control control = {
		.fn = &union_pw_multi_aff_scale_multi_val_entry,
		.fn_user = mv,
	};

	upma = isl_union_pw_multi_aff_align_params(upma,
						isl_multi_val_get_space(mv));
	mv = isl_multi_val_align_params(mv,
					isl_union_pw_multi_aff_get_space(upma));
	if (!upma || !mv)
		goto error;

	return isl_union_pw_multi_aff_transform(upma, &control);

	isl_multi_val_free(mv);
	return upma;
error:
	isl_multi_val_free(mv);
	isl_union_pw_multi_aff_free(upma);
	return NULL;
}

/* Construct and return a piecewise multi affine expression
 * in the given space with value zero in each of the output dimensions and
 * a universe domain.
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_zero(__isl_take isl_space *space)
{
	return isl_pw_multi_aff_from_multi_aff(isl_multi_aff_zero(space));
}

/* Construct and return a piecewise multi affine expression
 * that is equal to the given piecewise affine expression.
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_from_pw_aff(
	__isl_take isl_pw_aff *pa)
{
	int i;
	isl_space *space;
	isl_pw_multi_aff *pma;

	if (!pa)
		return NULL;

	space = isl_pw_aff_get_space(pa);
	pma = isl_pw_multi_aff_alloc_size(space, pa->n);

	for (i = 0; i < pa->n; ++i) {
		isl_set *set;
		isl_multi_aff *ma;

		set = isl_set_copy(pa->p[i].set);
		ma = isl_multi_aff_from_aff(isl_aff_copy(pa->p[i].aff));
		pma = isl_pw_multi_aff_add_piece(pma, set, ma);
	}

	isl_pw_aff_free(pa);
	return pma;
}

/* Construct and return a piecewise multi affine expression
 * that is equal to the given multi piecewise affine expression
 * on the shared domain of the piecewise affine expressions,
 * in the special case of a 0D multi piecewise affine expression.
 *
 * Create a piecewise multi affine expression with the explicit domain of
 * the 0D multi piecewise affine expression as domain.
 */
static __isl_give isl_pw_multi_aff *isl_pw_multi_aff_from_multi_pw_aff_0D(
	__isl_take isl_multi_pw_aff *mpa)
{
	isl_space *space;
	isl_set *dom;
	isl_multi_aff *ma;

	space = isl_multi_pw_aff_get_space(mpa);
	dom = isl_multi_pw_aff_get_explicit_domain(mpa);
	isl_multi_pw_aff_free(mpa);

	ma = isl_multi_aff_zero(space);
	return isl_pw_multi_aff_alloc(dom, ma);
}

/* Construct and return a piecewise multi affine expression
 * that is equal to the given multi piecewise affine expression
 * on the shared domain of the piecewise affine expressions.
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_from_multi_pw_aff(
	__isl_take isl_multi_pw_aff *mpa)
{
	int i;
	isl_space *space;
	isl_pw_aff *pa;
	isl_pw_multi_aff *pma;

	if (!mpa)
		return NULL;

	if (mpa->n == 0)
		return isl_pw_multi_aff_from_multi_pw_aff_0D(mpa);

	space = isl_multi_pw_aff_get_space(mpa);
	pa = isl_multi_pw_aff_get_pw_aff(mpa, 0);
	pma = isl_pw_multi_aff_from_pw_aff(pa);

	for (i = 1; i < mpa->n; ++i) {
		isl_pw_multi_aff *pma_i;

		pa = isl_multi_pw_aff_get_pw_aff(mpa, i);
		pma_i = isl_pw_multi_aff_from_pw_aff(pa);
		pma = isl_pw_multi_aff_range_product(pma, pma_i);
	}

	pma = isl_pw_multi_aff_reset_space(pma, space);

	isl_multi_pw_aff_free(mpa);
	return pma;
}

/* Convenience function that constructs an isl_multi_pw_aff
 * directly from an isl_aff.
 */
__isl_give isl_multi_pw_aff *isl_multi_pw_aff_from_aff(__isl_take isl_aff *aff)
{
	return isl_multi_pw_aff_from_pw_aff(isl_pw_aff_from_aff(aff));
}

/* Construct and return a multi piecewise affine expression
 * that is equal to the given multi affine expression.
 */
__isl_give isl_multi_pw_aff *isl_multi_pw_aff_from_multi_aff(
	__isl_take isl_multi_aff *ma)
{
	int i;
	isl_size n;
	isl_multi_pw_aff *mpa;

	n = isl_multi_aff_dim(ma, isl_dim_out);
	if (n < 0)
		ma = isl_multi_aff_free(ma);
	if (!ma)
		return NULL;

	mpa = isl_multi_pw_aff_alloc(isl_multi_aff_get_space(ma));

	for (i = 0; i < n; ++i) {
		isl_pw_aff *pa;

		pa = isl_pw_aff_from_aff(isl_multi_aff_get_aff(ma, i));
		mpa = isl_multi_pw_aff_set_pw_aff(mpa, i, pa);
	}

	isl_multi_aff_free(ma);
	return mpa;
}

/* This function performs the same operation as isl_multi_pw_aff_from_multi_aff,
 * but is considered as a function on an isl_multi_aff when exported.
 */
__isl_give isl_multi_pw_aff *isl_multi_aff_to_multi_pw_aff(
	__isl_take isl_multi_aff *ma)
{
	return isl_multi_pw_aff_from_multi_aff(ma);
}

/* Construct and return a multi piecewise affine expression
 * that is equal to the given piecewise multi affine expression.
 *
 * If the resulting multi piecewise affine expression has
 * an explicit domain, then assign it the domain of the input.
 * In other cases, the domain is stored in the individual elements.
 */
__isl_give isl_multi_pw_aff *isl_multi_pw_aff_from_pw_multi_aff(
	__isl_take isl_pw_multi_aff *pma)
{
	int i;
	isl_size n;
	isl_space *space;
	isl_multi_pw_aff *mpa;

	n = isl_pw_multi_aff_dim(pma, isl_dim_out);
	if (n < 0)
		pma = isl_pw_multi_aff_free(pma);
	space = isl_pw_multi_aff_get_space(pma);
	mpa = isl_multi_pw_aff_alloc(space);

	for (i = 0; i < n; ++i) {
		isl_pw_aff *pa;

		pa = isl_pw_multi_aff_get_pw_aff(pma, i);
		mpa = isl_multi_pw_aff_set_pw_aff(mpa, i, pa);
	}
	if (isl_multi_pw_aff_has_explicit_domain(mpa)) {
		isl_set *dom;

		dom = isl_pw_multi_aff_domain(isl_pw_multi_aff_copy(pma));
		mpa = isl_multi_pw_aff_intersect_domain(mpa, dom);
	}

	isl_pw_multi_aff_free(pma);
	return mpa;
}

/* This function performs the same operation as
 * isl_multi_pw_aff_from_pw_multi_aff,
 * but is considered as a function on an isl_pw_multi_aff when exported.
 */
__isl_give isl_multi_pw_aff *isl_pw_multi_aff_to_multi_pw_aff(
	__isl_take isl_pw_multi_aff *pma)
{
	return isl_multi_pw_aff_from_pw_multi_aff(pma);
}

/* Do "pa1" and "pa2" represent the same function?
 *
 * We first check if they are obviously equal.
 * If not, we convert them to maps and check if those are equal.
 *
 * If "pa1" or "pa2" contain any NaNs, then they are considered
 * not to be the same.  A NaN is not equal to anything, not even
 * to another NaN.
 */
isl_bool isl_pw_aff_is_equal(__isl_keep isl_pw_aff *pa1,
	__isl_keep isl_pw_aff *pa2)
{
	isl_bool equal;
	isl_bool has_nan;
	isl_map *map1, *map2;

	if (!pa1 || !pa2)
		return isl_bool_error;

	equal = isl_pw_aff_plain_is_equal(pa1, pa2);
	if (equal < 0 || equal)
		return equal;
	has_nan = either_involves_nan(pa1, pa2);
	if (has_nan < 0)
		return isl_bool_error;
	if (has_nan)
		return isl_bool_false;

	map1 = isl_map_from_pw_aff_internal(isl_pw_aff_copy(pa1));
	map2 = isl_map_from_pw_aff_internal(isl_pw_aff_copy(pa2));
	equal = isl_map_is_equal(map1, map2);
	isl_map_free(map1);
	isl_map_free(map2);

	return equal;
}

/* Do "mpa1" and "mpa2" represent the same function?
 *
 * Note that we cannot convert the entire isl_multi_pw_aff
 * to a map because the domains of the piecewise affine expressions
 * may not be the same.
 */
isl_bool isl_multi_pw_aff_is_equal(__isl_keep isl_multi_pw_aff *mpa1,
	__isl_keep isl_multi_pw_aff *mpa2)
{
	int i;
	isl_bool equal, equal_params;

	if (!mpa1 || !mpa2)
		return isl_bool_error;

	equal_params = isl_space_has_equal_params(mpa1->space, mpa2->space);
	if (equal_params < 0)
		return isl_bool_error;
	if (!equal_params) {
		if (!isl_space_has_named_params(mpa1->space))
			return isl_bool_false;
		if (!isl_space_has_named_params(mpa2->space))
			return isl_bool_false;
		mpa1 = isl_multi_pw_aff_copy(mpa1);
		mpa2 = isl_multi_pw_aff_copy(mpa2);
		mpa1 = isl_multi_pw_aff_align_params(mpa1,
					    isl_multi_pw_aff_get_space(mpa2));
		mpa2 = isl_multi_pw_aff_align_params(mpa2,
					    isl_multi_pw_aff_get_space(mpa1));
		equal = isl_multi_pw_aff_is_equal(mpa1, mpa2);
		isl_multi_pw_aff_free(mpa1);
		isl_multi_pw_aff_free(mpa2);
		return equal;
	}

	equal = isl_space_is_equal(mpa1->space, mpa2->space);
	if (equal < 0 || !equal)
		return equal;

	for (i = 0; i < mpa1->n; ++i) {
		equal = isl_pw_aff_is_equal(mpa1->u.p[i], mpa2->u.p[i]);
		if (equal < 0 || !equal)
			return equal;
	}

	return isl_bool_true;
}

/* Do "pma1" and "pma2" represent the same function?
 *
 * First check if they are obviously equal.
 * If not, then convert them to maps and check if those are equal.
 *
 * If "pa1" or "pa2" contain any NaNs, then they are considered
 * not to be the same.  A NaN is not equal to anything, not even
 * to another NaN.
 */
isl_bool isl_pw_multi_aff_is_equal(__isl_keep isl_pw_multi_aff *pma1,
	__isl_keep isl_pw_multi_aff *pma2)
{
	isl_bool equal;
	isl_bool has_nan;
	isl_map *map1, *map2;

	if (!pma1 || !pma2)
		return isl_bool_error;

	equal = isl_pw_multi_aff_plain_is_equal(pma1, pma2);
	if (equal < 0 || equal)
		return equal;
	has_nan = isl_pw_multi_aff_involves_nan(pma1);
	if (has_nan >= 0 && !has_nan)
		has_nan = isl_pw_multi_aff_involves_nan(pma2);
	if (has_nan < 0 || has_nan)
		return isl_bool_not(has_nan);

	map1 = isl_map_from_pw_multi_aff_internal(isl_pw_multi_aff_copy(pma1));
	map2 = isl_map_from_pw_multi_aff_internal(isl_pw_multi_aff_copy(pma2));
	equal = isl_map_is_equal(map1, map2);
	isl_map_free(map1);
	isl_map_free(map2);

	return equal;
}

/* Compute the pullback of "mpa" by the function represented by "ma".
 * In other words, plug in "ma" in "mpa".
 *
 * The parameters of "mpa" and "ma" are assumed to have been aligned.
 *
 * If "mpa" has an explicit domain, then it is this domain
 * that needs to undergo a pullback, i.e., a preimage.
 */
static __isl_give isl_multi_pw_aff *isl_multi_pw_aff_pullback_multi_aff_aligned(
	__isl_take isl_multi_pw_aff *mpa, __isl_take isl_multi_aff *ma)
{
	int i;
	isl_space *space = NULL;

	mpa = isl_multi_pw_aff_cow(mpa);
	if (!mpa || !ma)
		goto error;

	space = isl_space_join(isl_multi_aff_get_space(ma),
				isl_multi_pw_aff_get_space(mpa));
	if (!space)
		goto error;

	for (i = 0; i < mpa->n; ++i) {
		mpa->u.p[i] = isl_pw_aff_pullback_multi_aff(mpa->u.p[i],
						    isl_multi_aff_copy(ma));
		if (!mpa->u.p[i])
			goto error;
	}
	if (isl_multi_pw_aff_has_explicit_domain(mpa)) {
		mpa->u.dom = isl_set_preimage_multi_aff(mpa->u.dom,
							isl_multi_aff_copy(ma));
		if (!mpa->u.dom)
			goto error;
	}

	isl_multi_aff_free(ma);
	isl_space_free(mpa->space);
	mpa->space = space;
	return mpa;
error:
	isl_space_free(space);
	isl_multi_pw_aff_free(mpa);
	isl_multi_aff_free(ma);
	return NULL;
}

/* Compute the pullback of "mpa" by the function represented by "ma".
 * In other words, plug in "ma" in "mpa".
 */
__isl_give isl_multi_pw_aff *isl_multi_pw_aff_pullback_multi_aff(
	__isl_take isl_multi_pw_aff *mpa, __isl_take isl_multi_aff *ma)
{
	isl_bool equal_params;

	if (!mpa || !ma)
		goto error;
	equal_params = isl_space_has_equal_params(mpa->space, ma->space);
	if (equal_params < 0)
		goto error;
	if (equal_params)
		return isl_multi_pw_aff_pullback_multi_aff_aligned(mpa, ma);
	mpa = isl_multi_pw_aff_align_params(mpa, isl_multi_aff_get_space(ma));
	ma = isl_multi_aff_align_params(ma, isl_multi_pw_aff_get_space(mpa));
	return isl_multi_pw_aff_pullback_multi_aff_aligned(mpa, ma);
error:
	isl_multi_pw_aff_free(mpa);
	isl_multi_aff_free(ma);
	return NULL;
}

/* Compute the pullback of "mpa" by the function represented by "pma".
 * In other words, plug in "pma" in "mpa".
 *
 * The parameters of "mpa" and "mpa" are assumed to have been aligned.
 *
 * If "mpa" has an explicit domain, then it is this domain
 * that needs to undergo a pullback, i.e., a preimage.
 */
static __isl_give isl_multi_pw_aff *
isl_multi_pw_aff_pullback_pw_multi_aff_aligned(
	__isl_take isl_multi_pw_aff *mpa, __isl_take isl_pw_multi_aff *pma)
{
	int i;
	isl_space *space = NULL;

	mpa = isl_multi_pw_aff_cow(mpa);
	if (!mpa || !pma)
		goto error;

	space = isl_space_join(isl_pw_multi_aff_get_space(pma),
				isl_multi_pw_aff_get_space(mpa));

	for (i = 0; i < mpa->n; ++i) {
		mpa->u.p[i] = isl_pw_aff_pullback_pw_multi_aff_aligned(
				    mpa->u.p[i], isl_pw_multi_aff_copy(pma));
		if (!mpa->u.p[i])
			goto error;
	}
	if (isl_multi_pw_aff_has_explicit_domain(mpa)) {
		mpa->u.dom = isl_set_preimage_pw_multi_aff(mpa->u.dom,
						    isl_pw_multi_aff_copy(pma));
		if (!mpa->u.dom)
			goto error;
	}

	isl_pw_multi_aff_free(pma);
	isl_space_free(mpa->space);
	mpa->space = space;
	return mpa;
error:
	isl_space_free(space);
	isl_multi_pw_aff_free(mpa);
	isl_pw_multi_aff_free(pma);
	return NULL;
}

/* Compute the pullback of "mpa" by the function represented by "pma".
 * In other words, plug in "pma" in "mpa".
 */
__isl_give isl_multi_pw_aff *isl_multi_pw_aff_pullback_pw_multi_aff(
	__isl_take isl_multi_pw_aff *mpa, __isl_take isl_pw_multi_aff *pma)
{
	isl_bool equal_params;

	if (!mpa || !pma)
		goto error;
	equal_params = isl_space_has_equal_params(mpa->space, pma->dim);
	if (equal_params < 0)
		goto error;
	if (equal_params)
		return isl_multi_pw_aff_pullback_pw_multi_aff_aligned(mpa, pma);
	mpa = isl_multi_pw_aff_align_params(mpa,
					    isl_pw_multi_aff_get_space(pma));
	pma = isl_pw_multi_aff_align_params(pma,
					    isl_multi_pw_aff_get_space(mpa));
	return isl_multi_pw_aff_pullback_pw_multi_aff_aligned(mpa, pma);
error:
	isl_multi_pw_aff_free(mpa);
	isl_pw_multi_aff_free(pma);
	return NULL;
}

/* Apply "aff" to "mpa".  The range of "mpa" needs to be compatible
 * with the domain of "aff".  The domain of the result is the same
 * as that of "mpa".
 * "mpa" and "aff" are assumed to have been aligned.
 *
 * We first extract the parametric constant from "aff", defined
 * over the correct domain.
 * Then we add the appropriate combinations of the members of "mpa".
 * Finally, we add the integer divisions through recursive calls.
 */
static __isl_give isl_pw_aff *isl_multi_pw_aff_apply_aff_aligned(
	__isl_take isl_multi_pw_aff *mpa, __isl_take isl_aff *aff)
{
	int i;
	isl_size n_in, n_div, n_mpa_in;
	isl_space *space;
	isl_val *v;
	isl_pw_aff *pa;
	isl_aff *tmp;

	n_in = isl_aff_dim(aff, isl_dim_in);
	n_div = isl_aff_dim(aff, isl_dim_div);
	n_mpa_in = isl_multi_pw_aff_dim(mpa, isl_dim_in);
	if (n_in < 0 || n_div < 0 || n_mpa_in < 0)
		goto error;

	space = isl_space_domain(isl_multi_pw_aff_get_space(mpa));
	tmp = isl_aff_copy(aff);
	tmp = isl_aff_drop_dims(tmp, isl_dim_div, 0, n_div);
	tmp = isl_aff_drop_dims(tmp, isl_dim_in, 0, n_in);
	tmp = isl_aff_add_dims(tmp, isl_dim_in, n_mpa_in);
	tmp = isl_aff_reset_domain_space(tmp, space);
	pa = isl_pw_aff_from_aff(tmp);

	for (i = 0; i < n_in; ++i) {
		isl_pw_aff *pa_i;

		if (!isl_aff_involves_dims(aff, isl_dim_in, i, 1))
			continue;
		v = isl_aff_get_coefficient_val(aff, isl_dim_in, i);
		pa_i = isl_multi_pw_aff_get_pw_aff(mpa, i);
		pa_i = isl_pw_aff_scale_val(pa_i, v);
		pa = isl_pw_aff_add(pa, pa_i);
	}

	for (i = 0; i < n_div; ++i) {
		isl_aff *div;
		isl_pw_aff *pa_i;

		if (!isl_aff_involves_dims(aff, isl_dim_div, i, 1))
			continue;
		div = isl_aff_get_div(aff, i);
		pa_i = isl_multi_pw_aff_apply_aff_aligned(
					    isl_multi_pw_aff_copy(mpa), div);
		pa_i = isl_pw_aff_floor(pa_i);
		v = isl_aff_get_coefficient_val(aff, isl_dim_div, i);
		pa_i = isl_pw_aff_scale_val(pa_i, v);
		pa = isl_pw_aff_add(pa, pa_i);
	}

	isl_multi_pw_aff_free(mpa);
	isl_aff_free(aff);

	return pa;
error:
	isl_multi_pw_aff_free(mpa);
	isl_aff_free(aff);
	return NULL;
}

/* Apply "aff" to "mpa".  The range of "mpa" needs to be compatible
 * with the domain of "aff".  The domain of the result is the same
 * as that of "mpa".
 */
__isl_give isl_pw_aff *isl_multi_pw_aff_apply_aff(
	__isl_take isl_multi_pw_aff *mpa, __isl_take isl_aff *aff)
{
	isl_bool equal_params;

	if (!aff || !mpa)
		goto error;
	equal_params = isl_space_has_equal_params(aff->ls->dim, mpa->space);
	if (equal_params < 0)
		goto error;
	if (equal_params)
		return isl_multi_pw_aff_apply_aff_aligned(mpa, aff);

	aff = isl_aff_align_params(aff, isl_multi_pw_aff_get_space(mpa));
	mpa = isl_multi_pw_aff_align_params(mpa, isl_aff_get_space(aff));

	return isl_multi_pw_aff_apply_aff_aligned(mpa, aff);
error:
	isl_aff_free(aff);
	isl_multi_pw_aff_free(mpa);
	return NULL;
}

/* Apply "pa" to "mpa".  The range of "mpa" needs to be compatible
 * with the domain of "pa".  The domain of the result is the same
 * as that of "mpa".
 * "mpa" and "pa" are assumed to have been aligned.
 *
 * We consider each piece in turn.  Note that the domains of the
 * pieces are assumed to be disjoint and they remain disjoint
 * after taking the preimage (over the same function).
 */
static __isl_give isl_pw_aff *isl_multi_pw_aff_apply_pw_aff_aligned(
	__isl_take isl_multi_pw_aff *mpa, __isl_take isl_pw_aff *pa)
{
	isl_space *space;
	isl_pw_aff *res;
	int i;

	if (!mpa || !pa)
		goto error;

	space = isl_space_join(isl_multi_pw_aff_get_space(mpa),
				isl_pw_aff_get_space(pa));
	res = isl_pw_aff_empty(space);

	for (i = 0; i < pa->n; ++i) {
		isl_pw_aff *pa_i;
		isl_set *domain;

		pa_i = isl_multi_pw_aff_apply_aff_aligned(
					isl_multi_pw_aff_copy(mpa),
					isl_aff_copy(pa->p[i].aff));
		domain = isl_set_copy(pa->p[i].set);
		domain = isl_set_preimage_multi_pw_aff(domain,
					isl_multi_pw_aff_copy(mpa));
		pa_i = isl_pw_aff_intersect_domain(pa_i, domain);
		res = isl_pw_aff_add_disjoint(res, pa_i);
	}

	isl_pw_aff_free(pa);
	isl_multi_pw_aff_free(mpa);
	return res;
error:
	isl_pw_aff_free(pa);
	isl_multi_pw_aff_free(mpa);
	return NULL;
}

/* Apply "pa" to "mpa".  The range of "mpa" needs to be compatible
 * with the domain of "pa".  The domain of the result is the same
 * as that of "mpa".
 */
__isl_give isl_pw_aff *isl_multi_pw_aff_apply_pw_aff(
	__isl_take isl_multi_pw_aff *mpa, __isl_take isl_pw_aff *pa)
{
	isl_bool equal_params;

	if (!pa || !mpa)
		goto error;
	equal_params = isl_space_has_equal_params(pa->dim, mpa->space);
	if (equal_params < 0)
		goto error;
	if (equal_params)
		return isl_multi_pw_aff_apply_pw_aff_aligned(mpa, pa);

	pa = isl_pw_aff_align_params(pa, isl_multi_pw_aff_get_space(mpa));
	mpa = isl_multi_pw_aff_align_params(mpa, isl_pw_aff_get_space(pa));

	return isl_multi_pw_aff_apply_pw_aff_aligned(mpa, pa);
error:
	isl_pw_aff_free(pa);
	isl_multi_pw_aff_free(mpa);
	return NULL;
}

/* Compute the pullback of "pa" by the function represented by "mpa".
 * In other words, plug in "mpa" in "pa".
 * "pa" and "mpa" are assumed to have been aligned.
 *
 * The pullback is computed by applying "pa" to "mpa".
 */
static __isl_give isl_pw_aff *isl_pw_aff_pullback_multi_pw_aff_aligned(
	__isl_take isl_pw_aff *pa, __isl_take isl_multi_pw_aff *mpa)
{
	return isl_multi_pw_aff_apply_pw_aff_aligned(mpa, pa);
}

/* Compute the pullback of "pa" by the function represented by "mpa".
 * In other words, plug in "mpa" in "pa".
 *
 * The pullback is computed by applying "pa" to "mpa".
 */
__isl_give isl_pw_aff *isl_pw_aff_pullback_multi_pw_aff(
	__isl_take isl_pw_aff *pa, __isl_take isl_multi_pw_aff *mpa)
{
	return isl_multi_pw_aff_apply_pw_aff(mpa, pa);
}

/* Compute the pullback of "mpa1" by the function represented by "mpa2".
 * In other words, plug in "mpa2" in "mpa1".
 *
 * We pullback each member of "mpa1" in turn.
 *
 * If "mpa1" has an explicit domain, then it is this domain
 * that needs to undergo a pullback instead, i.e., a preimage.
 */
__isl_give isl_multi_pw_aff *isl_multi_pw_aff_pullback_multi_pw_aff(
	__isl_take isl_multi_pw_aff *mpa1, __isl_take isl_multi_pw_aff *mpa2)
{
	int i;
	isl_space *space = NULL;

	isl_multi_pw_aff_align_params_bin(&mpa1, &mpa2);
	mpa1 = isl_multi_pw_aff_cow(mpa1);
	if (!mpa1 || !mpa2)
		goto error;

	space = isl_space_join(isl_multi_pw_aff_get_space(mpa2),
				isl_multi_pw_aff_get_space(mpa1));

	for (i = 0; i < mpa1->n; ++i) {
		mpa1->u.p[i] = isl_pw_aff_pullback_multi_pw_aff_aligned(
				mpa1->u.p[i], isl_multi_pw_aff_copy(mpa2));
		if (!mpa1->u.p[i])
			goto error;
	}

	if (isl_multi_pw_aff_has_explicit_domain(mpa1)) {
		mpa1->u.dom = isl_set_preimage_multi_pw_aff(mpa1->u.dom,
						isl_multi_pw_aff_copy(mpa2));
		if (!mpa1->u.dom)
			goto error;
	}
	mpa1 = isl_multi_pw_aff_reset_space(mpa1, space);

	isl_multi_pw_aff_free(mpa2);
	return mpa1;
error:
	isl_space_free(space);
	isl_multi_pw_aff_free(mpa1);
	isl_multi_pw_aff_free(mpa2);
	return NULL;
}

/* Align the parameters of "mpa1" and "mpa2", check that the ranges
 * of "mpa1" and "mpa2" live in the same space, construct map space
 * between the domain spaces of "mpa1" and "mpa2" and call "order"
 * with this map space as extract argument.
 */
static __isl_give isl_map *isl_multi_pw_aff_order_map(
	__isl_take isl_multi_pw_aff *mpa1, __isl_take isl_multi_pw_aff *mpa2,
	__isl_give isl_map *(*order)(__isl_keep isl_multi_pw_aff *mpa1,
		__isl_keep isl_multi_pw_aff *mpa2, __isl_take isl_space *space))
{
	int match;
	isl_space *space1, *space2;
	isl_map *res;

	mpa1 = isl_multi_pw_aff_align_params(mpa1,
					    isl_multi_pw_aff_get_space(mpa2));
	mpa2 = isl_multi_pw_aff_align_params(mpa2,
					    isl_multi_pw_aff_get_space(mpa1));
	if (!mpa1 || !mpa2)
		goto error;
	match = isl_space_tuple_is_equal(mpa1->space, isl_dim_out,
					mpa2->space, isl_dim_out);
	if (match < 0)
		goto error;
	if (!match)
		isl_die(isl_multi_pw_aff_get_ctx(mpa1), isl_error_invalid,
			"range spaces don't match", goto error);
	space1 = isl_space_domain(isl_multi_pw_aff_get_space(mpa1));
	space2 = isl_space_domain(isl_multi_pw_aff_get_space(mpa2));
	space1 = isl_space_map_from_domain_and_range(space1, space2);

	res = order(mpa1, mpa2, space1);
	isl_multi_pw_aff_free(mpa1);
	isl_multi_pw_aff_free(mpa2);
	return res;
error:
	isl_multi_pw_aff_free(mpa1);
	isl_multi_pw_aff_free(mpa2);
	return NULL;
}

/* Return a map containing pairs of elements in the domains of "mpa1" and "mpa2"
 * where the function values are equal.  "space" is the space of the result.
 * The parameters of "mpa1" and "mpa2" are assumed to have been aligned.
 *
 * "mpa1" and "mpa2" are equal when each of the pairs of elements
 * in the sequences are equal.
 */
static __isl_give isl_map *isl_multi_pw_aff_eq_map_on_space(
	__isl_keep isl_multi_pw_aff *mpa1, __isl_keep isl_multi_pw_aff *mpa2,
	__isl_take isl_space *space)
{
	int i;
	isl_size n;
	isl_map *res;

	n = isl_multi_pw_aff_dim(mpa1, isl_dim_out);
	if (n < 0)
		space = isl_space_free(space);
	res = isl_map_universe(space);

	for (i = 0; i < n; ++i) {
		isl_pw_aff *pa1, *pa2;
		isl_map *map;

		pa1 = isl_multi_pw_aff_get_pw_aff(mpa1, i);
		pa2 = isl_multi_pw_aff_get_pw_aff(mpa2, i);
		map = isl_pw_aff_eq_map(pa1, pa2);
		res = isl_map_intersect(res, map);
	}

	return res;
}

/* Return a map containing pairs of elements in the domains of "mpa1" and "mpa2"
 * where the function values are equal.
 */
__isl_give isl_map *isl_multi_pw_aff_eq_map(__isl_take isl_multi_pw_aff *mpa1,
	__isl_take isl_multi_pw_aff *mpa2)
{
	return isl_multi_pw_aff_order_map(mpa1, mpa2,
					    &isl_multi_pw_aff_eq_map_on_space);
}

/* Intersect "map" with the result of applying "order"
 * on two copies of "mpa".
 */
static __isl_give isl_map *isl_map_order_at_multi_pw_aff(
	__isl_take isl_map *map, __isl_take isl_multi_pw_aff *mpa,
	__isl_give isl_map *(*order)(__isl_take isl_multi_pw_aff *mpa1,
		__isl_take isl_multi_pw_aff *mpa2))
{
	return isl_map_intersect(map, order(mpa, isl_multi_pw_aff_copy(mpa)));
}

/* Return the subset of "map" where the domain and the range
 * have equal "mpa" values.
 */
__isl_give isl_map *isl_map_eq_at_multi_pw_aff(__isl_take isl_map *map,
	__isl_take isl_multi_pw_aff *mpa)
{
	return isl_map_order_at_multi_pw_aff(map, mpa,
						&isl_multi_pw_aff_eq_map);
}

/* Return a map containing pairs of elements in the domains of "mpa1" and "mpa2"
 * where the function values of "mpa1" lexicographically satisfies
 * "strict_base"/"base" compared to that of "mpa2".
 * "space" is the space of the result.
 * The parameters of "mpa1" and "mpa2" are assumed to have been aligned.
 *
 * "mpa1" lexicographically satisfies "strict_base"/"base" compared to "mpa2"
 * if, for some i, the i-th element of "mpa1" satisfies "strict_base"/"base"
 * when compared to the i-th element of "mpa2" while all previous elements are
 * pairwise equal.
 * In particular, if i corresponds to the final elements
 * then they need to satisfy "base", while "strict_base" needs to be satisfied
 * for other values of i.
 * If "base" is a strict order, then "base" and "strict_base" are the same.
 */
static __isl_give isl_map *isl_multi_pw_aff_lex_map_on_space(
	__isl_keep isl_multi_pw_aff *mpa1, __isl_keep isl_multi_pw_aff *mpa2,
	__isl_give isl_map *(*strict_base)(__isl_take isl_pw_aff *pa1,
		__isl_take isl_pw_aff *pa2),
	__isl_give isl_map *(*base)(__isl_take isl_pw_aff *pa1,
		__isl_take isl_pw_aff *pa2),
	__isl_take isl_space *space)
{
	int i;
	isl_size n;
	isl_map *res, *rest;

	n = isl_multi_pw_aff_dim(mpa1, isl_dim_out);
	if (n < 0)
		space = isl_space_free(space);
	res = isl_map_empty(isl_space_copy(space));
	rest = isl_map_universe(space);

	for (i = 0; i < n; ++i) {
		int last;
		isl_pw_aff *pa1, *pa2;
		isl_map *map;

		last = i == n - 1;

		pa1 = isl_multi_pw_aff_get_pw_aff(mpa1, i);
		pa2 = isl_multi_pw_aff_get_pw_aff(mpa2, i);
		map = last ? base(pa1, pa2) : strict_base(pa1, pa2);
		map = isl_map_intersect(map, isl_map_copy(rest));
		res = isl_map_union(res, map);

		if (last)
			continue;

		pa1 = isl_multi_pw_aff_get_pw_aff(mpa1, i);
		pa2 = isl_multi_pw_aff_get_pw_aff(mpa2, i);
		map = isl_pw_aff_eq_map(pa1, pa2);
		rest = isl_map_intersect(rest, map);
	}

	isl_map_free(rest);
	return res;
}

#undef ORDER
#define ORDER		le
#undef STRICT_ORDER
#define STRICT_ORDER	lt
#include "isl_aff_lex_templ.c"

#undef ORDER
#define ORDER		lt
#undef STRICT_ORDER
#define STRICT_ORDER	lt
#include "isl_aff_lex_templ.c"

#undef ORDER
#define ORDER		ge
#undef STRICT_ORDER
#define STRICT_ORDER	gt
#include "isl_aff_lex_templ.c"

#undef ORDER
#define ORDER		gt
#undef STRICT_ORDER
#define STRICT_ORDER	gt
#include "isl_aff_lex_templ.c"

/* Compare two isl_affs.
 *
 * Return -1 if "aff1" is "smaller" than "aff2", 1 if "aff1" is "greater"
 * than "aff2" and 0 if they are equal.
 *
 * The order is fairly arbitrary.  We do consider expressions that only involve
 * earlier dimensions as "smaller".
 */
int isl_aff_plain_cmp(__isl_keep isl_aff *aff1, __isl_keep isl_aff *aff2)
{
	int cmp;
	int last1, last2;

	if (aff1 == aff2)
		return 0;

	if (!aff1)
		return -1;
	if (!aff2)
		return 1;

	cmp = isl_local_space_cmp(aff1->ls, aff2->ls);
	if (cmp != 0)
		return cmp;

	last1 = isl_seq_last_non_zero(aff1->v->el + 1, aff1->v->size - 1);
	last2 = isl_seq_last_non_zero(aff2->v->el + 1, aff1->v->size - 1);
	if (last1 != last2)
		return last1 - last2;

	return isl_seq_cmp(aff1->v->el, aff2->v->el, aff1->v->size);
}

/* Compare two isl_pw_affs.
 *
 * Return -1 if "pa1" is "smaller" than "pa2", 1 if "pa1" is "greater"
 * than "pa2" and 0 if they are equal.
 *
 * The order is fairly arbitrary.  We do consider expressions that only involve
 * earlier dimensions as "smaller".
 */
int isl_pw_aff_plain_cmp(__isl_keep isl_pw_aff *pa1,
	__isl_keep isl_pw_aff *pa2)
{
	int i;
	int cmp;

	if (pa1 == pa2)
		return 0;

	if (!pa1)
		return -1;
	if (!pa2)
		return 1;

	cmp = isl_space_cmp(pa1->dim, pa2->dim);
	if (cmp != 0)
		return cmp;

	if (pa1->n != pa2->n)
		return pa1->n - pa2->n;

	for (i = 0; i < pa1->n; ++i) {
		cmp = isl_set_plain_cmp(pa1->p[i].set, pa2->p[i].set);
		if (cmp != 0)
			return cmp;
		cmp = isl_aff_plain_cmp(pa1->p[i].aff, pa2->p[i].aff);
		if (cmp != 0)
			return cmp;
	}

	return 0;
}

/* Return a piecewise affine expression that is equal to "v" on "domain".
 */
__isl_give isl_pw_aff *isl_pw_aff_val_on_domain(__isl_take isl_set *domain,
	__isl_take isl_val *v)
{
	isl_space *space;
	isl_local_space *ls;
	isl_aff *aff;

	space = isl_set_get_space(domain);
	ls = isl_local_space_from_space(space);
	aff = isl_aff_val_on_domain(ls, v);

	return isl_pw_aff_alloc(domain, aff);
}

/* Return a piecewise affine expression that is equal to the parameter
 * with identifier "id" on "domain".
 */
__isl_give isl_pw_aff *isl_pw_aff_param_on_domain_id(
	__isl_take isl_set *domain, __isl_take isl_id *id)
{
	isl_space *space;
	isl_aff *aff;

	space = isl_set_get_space(domain);
	space = isl_space_add_param_id(space, isl_id_copy(id));
	domain = isl_set_align_params(domain, isl_space_copy(space));
	aff = isl_aff_param_on_domain_space_id(space, id);

	return isl_pw_aff_alloc(domain, aff);
}

/* Return a multi affine expression that is equal to "mv" on domain
 * space "space".
 */
__isl_give isl_multi_aff *isl_multi_aff_multi_val_on_domain_space(
	__isl_take isl_space *space, __isl_take isl_multi_val *mv)
{
	int i;
	isl_size n;
	isl_space *space2;
	isl_local_space *ls;
	isl_multi_aff *ma;

	n = isl_multi_val_dim(mv, isl_dim_set);
	if (!space || n < 0)
		goto error;

	space2 = isl_multi_val_get_space(mv);
	space2 = isl_space_align_params(space2, isl_space_copy(space));
	space = isl_space_align_params(space, isl_space_copy(space2));
	space = isl_space_map_from_domain_and_range(space, space2);
	ma = isl_multi_aff_alloc(isl_space_copy(space));
	ls = isl_local_space_from_space(isl_space_domain(space));
	for (i = 0; i < n; ++i) {
		isl_val *v;
		isl_aff *aff;

		v = isl_multi_val_get_val(mv, i);
		aff = isl_aff_val_on_domain(isl_local_space_copy(ls), v);
		ma = isl_multi_aff_set_aff(ma, i, aff);
	}
	isl_local_space_free(ls);

	isl_multi_val_free(mv);
	return ma;
error:
	isl_space_free(space);
	isl_multi_val_free(mv);
	return NULL;
}

/* This is an alternative name for the function above.
 */
__isl_give isl_multi_aff *isl_multi_aff_multi_val_on_space(
	__isl_take isl_space *space, __isl_take isl_multi_val *mv)
{
	return isl_multi_aff_multi_val_on_domain_space(space, mv);
}

/* This function performs the same operation as
 * isl_multi_aff_multi_val_on_domain_space,
 * but is considered as a function on an isl_space when exported.
 */
__isl_give isl_multi_aff *isl_space_multi_aff_on_domain_multi_val(
	__isl_take isl_space *space, __isl_take isl_multi_val *mv)
{
	return isl_multi_aff_multi_val_on_domain_space(space, mv);
}

/* Return a piecewise multi-affine expression
 * that is equal to "mv" on "domain".
 */
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_multi_val_on_domain(
	__isl_take isl_set *domain, __isl_take isl_multi_val *mv)
{
	isl_space *space;
	isl_multi_aff *ma;

	space = isl_set_get_space(domain);
	ma = isl_multi_aff_multi_val_on_space(space, mv);

	return isl_pw_multi_aff_alloc(domain, ma);
}

/* This function performs the same operation as
 * isl_pw_multi_aff_multi_val_on_domain,
 * but is considered as a function on an isl_set when exported.
 */
__isl_give isl_pw_multi_aff *isl_set_pw_multi_aff_on_domain_multi_val(
	__isl_take isl_set *domain, __isl_take isl_multi_val *mv)
{
	return isl_pw_multi_aff_multi_val_on_domain(domain, mv);
}

/* Internal data structure for isl_union_pw_multi_aff_multi_val_on_domain.
 * mv is the value that should be attained on each domain set
 * res collects the results
 */
struct isl_union_pw_multi_aff_multi_val_on_domain_data {
	isl_multi_val *mv;
	isl_union_pw_multi_aff *res;
};

/* Create an isl_pw_multi_aff equal to data->mv on "domain"
 * and add it to data->res.
 */
static isl_stat pw_multi_aff_multi_val_on_domain(__isl_take isl_set *domain,
	void *user)
{
	struct isl_union_pw_multi_aff_multi_val_on_domain_data *data = user;
	isl_pw_multi_aff *pma;
	isl_multi_val *mv;

	mv = isl_multi_val_copy(data->mv);
	pma = isl_pw_multi_aff_multi_val_on_domain(domain, mv);
	data->res = isl_union_pw_multi_aff_add_pw_multi_aff(data->res, pma);

	return data->res ? isl_stat_ok : isl_stat_error;
}

/* Return a union piecewise multi-affine expression
 * that is equal to "mv" on "domain".
 */
__isl_give isl_union_pw_multi_aff *isl_union_pw_multi_aff_multi_val_on_domain(
	__isl_take isl_union_set *domain, __isl_take isl_multi_val *mv)
{
	struct isl_union_pw_multi_aff_multi_val_on_domain_data data;
	isl_space *space;

	space = isl_union_set_get_space(domain);
	data.res = isl_union_pw_multi_aff_empty(space);
	data.mv = mv;
	if (isl_union_set_foreach_set(domain,
			&pw_multi_aff_multi_val_on_domain, &data) < 0)
		data.res = isl_union_pw_multi_aff_free(data.res);
	isl_union_set_free(domain);
	isl_multi_val_free(mv);
	return data.res;
}

/* Compute the pullback of data->pma by the function represented by "pma2",
 * provided the spaces match, and add the results to data->res.
 */
static isl_stat pullback_entry(__isl_take isl_pw_multi_aff *pma2, void *user)
{
	struct isl_union_pw_multi_aff_bin_data *data = user;

	if (!isl_space_tuple_is_equal(data->pma->dim, isl_dim_in,
				 pma2->dim, isl_dim_out)) {
		isl_pw_multi_aff_free(pma2);
		return isl_stat_ok;
	}

	pma2 = isl_pw_multi_aff_pullback_pw_multi_aff(
					isl_pw_multi_aff_copy(data->pma), pma2);

	data->res = isl_union_pw_multi_aff_add_pw_multi_aff(data->res, pma2);
	if (!data->res)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Compute the pullback of "upma1" by the function represented by "upma2".
 */
__isl_give isl_union_pw_multi_aff *
isl_union_pw_multi_aff_pullback_union_pw_multi_aff(
	__isl_take isl_union_pw_multi_aff *upma1,
	__isl_take isl_union_pw_multi_aff *upma2)
{
	return bin_op(upma1, upma2, &pullback_entry);
}

/* Apply "upma2" to "upma1".
 *
 * That is, compute the pullback of "upma2" by "upma1".
 */
__isl_give isl_union_pw_multi_aff *
isl_union_pw_multi_aff_apply_union_pw_multi_aff(
	__isl_take isl_union_pw_multi_aff *upma1,
	__isl_take isl_union_pw_multi_aff *upma2)
{
	return isl_union_pw_multi_aff_pullback_union_pw_multi_aff(upma2, upma1);
}

#undef TYPE
#define TYPE isl_pw_multi_aff
static
#include "isl_copy_tuple_id_templ.c"

/* Given a function "pma1" of the form A[B -> C] -> D and
 * a function "pma2" of the form E -> B,
 * replace the domain of the wrapped relation inside the domain of "pma1"
 * by the preimage with respect to "pma2".
 * In other words, plug in "pma2" in this nested domain.
 * The result is of the form A[E -> C] -> D.
 *
 * In particular, extend E -> B to A[E -> C] -> A[B -> C] and
 * plug that into "pma1".
 */
__isl_give isl_pw_multi_aff *
isl_pw_multi_aff_preimage_domain_wrapped_domain_pw_multi_aff(
	__isl_take isl_pw_multi_aff *pma1, __isl_take isl_pw_multi_aff *pma2)
{
	isl_space *pma1_space, *pma2_space;
	isl_space *space;
	isl_pw_multi_aff *id;

	pma1_space = isl_pw_multi_aff_peek_space(pma1);
	pma2_space = isl_pw_multi_aff_peek_space(pma2);

	if (isl_space_check_domain_is_wrapping(pma1_space) < 0)
		goto error;
	if (isl_space_check_wrapped_tuple_is_equal(pma1_space,
			isl_dim_in, isl_dim_in, pma2_space, isl_dim_out) < 0)
		goto error;

	space = isl_space_domain(isl_space_copy(pma1_space));
	space = isl_space_range(isl_space_unwrap(space));
	id = isl_pw_multi_aff_identity_on_domain_space(space);
	pma2 = isl_pw_multi_aff_product(pma2, id);

	pma2 = isl_pw_multi_aff_copy_tuple_id(pma2, isl_dim_in,
						pma1_space, isl_dim_in);
	pma2 = isl_pw_multi_aff_copy_tuple_id(pma2, isl_dim_out,
						pma1_space, isl_dim_in);

	return isl_pw_multi_aff_pullback_pw_multi_aff(pma1, pma2);
error:
	isl_pw_multi_aff_free(pma1);
	isl_pw_multi_aff_free(pma2);
	return NULL;
}

/* If data->pma and "pma2" are such that
 * data->pma is of the form A[B -> C] -> D and
 * "pma2" is of the form E -> B,
 * then replace the domain of the wrapped relation
 * inside the domain of data->pma by the preimage with respect to "pma2" and
 * add the result to data->res.
 */
static isl_stat preimage_domain_wrapped_domain_entry(
	__isl_take isl_pw_multi_aff *pma2, void *user)
{
	struct isl_union_pw_multi_aff_bin_data *data = user;
	isl_space *pma1_space, *pma2_space;
	isl_bool match;

	pma1_space = isl_pw_multi_aff_peek_space(data->pma);
	pma2_space = isl_pw_multi_aff_peek_space(pma2);

	match = isl_space_domain_is_wrapping(pma1_space);
	if (match >= 0 && match)
		match = isl_space_wrapped_tuple_is_equal(pma1_space, isl_dim_in,
					isl_dim_in, pma2_space, isl_dim_out);
	if (match < 0 || !match) {
		isl_pw_multi_aff_free(pma2);
		return match < 0 ? isl_stat_error : isl_stat_ok;
	}

	pma2 = isl_pw_multi_aff_preimage_domain_wrapped_domain_pw_multi_aff(
		isl_pw_multi_aff_copy(data->pma), pma2);

	data->res = isl_union_pw_multi_aff_add_pw_multi_aff(data->res, pma2);

	return isl_stat_non_null(data->res);
}

/* For each pair of functions A[B -> C] -> D in "upma1" and
 * E -> B in "upma2",
 * replace the domain of the wrapped relation inside the domain of the first
 * by the preimage with respect to the second and collect the results.
 * In other words, plug in the second function in this nested domain.
 * The results are of the form A[E -> C] -> D.
 */
__isl_give isl_union_pw_multi_aff *
isl_union_pw_multi_aff_preimage_domain_wrapped_domain_union_pw_multi_aff(
	__isl_take isl_union_pw_multi_aff *upma1,
	__isl_take isl_union_pw_multi_aff *upma2)
{
	return bin_op(upma1, upma2, &preimage_domain_wrapped_domain_entry);
}

/* Check that the domain space of "upa" matches "space".
 *
 * This function is called from isl_multi_union_pw_aff_set_union_pw_aff and
 * can in principle never fail since the space "space" is that
 * of the isl_multi_union_pw_aff and is a set space such that
 * there is no domain space to match.
 *
 * We check the parameters and double-check that "space" is
 * indeed that of a set.
 */
static isl_stat isl_union_pw_aff_check_match_domain_space(
	__isl_keep isl_union_pw_aff *upa, __isl_keep isl_space *space)
{
	isl_space *upa_space;
	isl_bool match;

	if (!upa || !space)
		return isl_stat_error;

	match = isl_space_is_set(space);
	if (match < 0)
		return isl_stat_error;
	if (!match)
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"expecting set space", return isl_stat_error);

	upa_space = isl_union_pw_aff_get_space(upa);
	match = isl_space_has_equal_params(space, upa_space);
	if (match < 0)
		goto error;
	if (!match)
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"parameters don't match", goto error);

	isl_space_free(upa_space);
	return isl_stat_ok;
error:
	isl_space_free(upa_space);
	return isl_stat_error;
}

/* Do the parameters of "upa" match those of "space"?
 */
static isl_bool isl_union_pw_aff_matching_params(
	__isl_keep isl_union_pw_aff *upa, __isl_keep isl_space *space)
{
	isl_space *upa_space;
	isl_bool match;

	if (!upa || !space)
		return isl_bool_error;

	upa_space = isl_union_pw_aff_get_space(upa);

	match = isl_space_has_equal_params(space, upa_space);

	isl_space_free(upa_space);
	return match;
}

/* Internal data structure for isl_union_pw_aff_reset_domain_space.
 * space represents the new parameters.
 * res collects the results.
 */
struct isl_union_pw_aff_reset_params_data {
	isl_space *space;
	isl_union_pw_aff *res;
};

/* Replace the parameters of "pa" by data->space and
 * add the result to data->res.
 */
static isl_stat reset_params(__isl_take isl_pw_aff *pa, void *user)
{
	struct isl_union_pw_aff_reset_params_data *data = user;
	isl_space *space;

	space = isl_pw_aff_get_space(pa);
	space = isl_space_replace_params(space, data->space);
	pa = isl_pw_aff_reset_space(pa, space);
	data->res = isl_union_pw_aff_add_pw_aff(data->res, pa);

	return data->res ? isl_stat_ok : isl_stat_error;
}

/* Replace the domain space of "upa" by "space".
 * Since a union expression does not have a (single) domain space,
 * "space" is necessarily a parameter space.
 *
 * Since the order and the names of the parameters determine
 * the hash value, we need to create a new hash table.
 */
static __isl_give isl_union_pw_aff *isl_union_pw_aff_reset_domain_space(
	__isl_take isl_union_pw_aff *upa, __isl_take isl_space *space)
{
	struct isl_union_pw_aff_reset_params_data data = { space };
	isl_bool match;

	match = isl_union_pw_aff_matching_params(upa, space);
	if (match < 0)
		upa = isl_union_pw_aff_free(upa);
	else if (match) {
		isl_space_free(space);
		return upa;
	}

	data.res = isl_union_pw_aff_empty(isl_space_copy(space));
	if (isl_union_pw_aff_foreach_pw_aff(upa, &reset_params, &data) < 0)
		data.res = isl_union_pw_aff_free(data.res);

	isl_union_pw_aff_free(upa);
	isl_space_free(space);
	return data.res;
}

/* Return the floor of "pa".
 */
static __isl_give isl_pw_aff *floor_entry(__isl_take isl_pw_aff *pa, void *user)
{
	return isl_pw_aff_floor(pa);
}

/* Given f, return floor(f).
 */
__isl_give isl_union_pw_aff *isl_union_pw_aff_floor(
	__isl_take isl_union_pw_aff *upa)
{
	return isl_union_pw_aff_transform_inplace(upa, &floor_entry, NULL);
}

/* Compute
 *
 *	upa mod m = upa - m * floor(upa/m)
 *
 * with m an integer value.
 */
__isl_give isl_union_pw_aff *isl_union_pw_aff_mod_val(
	__isl_take isl_union_pw_aff *upa, __isl_take isl_val *m)
{
	isl_union_pw_aff *res;

	if (!upa || !m)
		goto error;

	if (!isl_val_is_int(m))
		isl_die(isl_val_get_ctx(m), isl_error_invalid,
			"expecting integer modulo", goto error);
	if (!isl_val_is_pos(m))
		isl_die(isl_val_get_ctx(m), isl_error_invalid,
			"expecting positive modulo", goto error);

	res = isl_union_pw_aff_copy(upa);
	upa = isl_union_pw_aff_scale_down_val(upa, isl_val_copy(m));
	upa = isl_union_pw_aff_floor(upa);
	upa = isl_union_pw_aff_scale_val(upa, m);
	res = isl_union_pw_aff_sub(res, upa);

	return res;
error:
	isl_val_free(m);
	isl_union_pw_aff_free(upa);
	return NULL;
}

/* Internal data structure for isl_union_pw_multi_aff_get_union_pw_aff.
 * pos is the output position that needs to be extracted.
 * res collects the results.
 */
struct isl_union_pw_multi_aff_get_union_pw_aff_data {
	int pos;
	isl_union_pw_aff *res;
};

/* Extract an isl_pw_aff corresponding to output dimension "pos" of "pma"
 * (assuming it has such a dimension) and add it to data->res.
 */
static isl_stat get_union_pw_aff(__isl_take isl_pw_multi_aff *pma, void *user)
{
	struct isl_union_pw_multi_aff_get_union_pw_aff_data *data = user;
	isl_size n_out;
	isl_pw_aff *pa;

	n_out = isl_pw_multi_aff_dim(pma, isl_dim_out);
	if (n_out < 0)
		return isl_stat_error;
	if (data->pos >= n_out) {
		isl_pw_multi_aff_free(pma);
		return isl_stat_ok;
	}

	pa = isl_pw_multi_aff_get_pw_aff(pma, data->pos);
	isl_pw_multi_aff_free(pma);

	data->res = isl_union_pw_aff_add_pw_aff(data->res, pa);

	return data->res ? isl_stat_ok : isl_stat_error;
}

/* Extract an isl_union_pw_aff corresponding to
 * output dimension "pos" of "upma".
 */
__isl_give isl_union_pw_aff *isl_union_pw_multi_aff_get_union_pw_aff(
	__isl_keep isl_union_pw_multi_aff *upma, int pos)
{
	struct isl_union_pw_multi_aff_get_union_pw_aff_data data;
	isl_space *space;

	if (!upma)
		return NULL;

	if (pos < 0)
		isl_die(isl_union_pw_multi_aff_get_ctx(upma), isl_error_invalid,
			"cannot extract at negative position", return NULL);

	space = isl_union_pw_multi_aff_get_space(upma);
	data.res = isl_union_pw_aff_empty(space);
	data.pos = pos;
	if (isl_union_pw_multi_aff_foreach_pw_multi_aff(upma,
						&get_union_pw_aff, &data) < 0)
		data.res = isl_union_pw_aff_free(data.res);

	return data.res;
}

/* Return a union piecewise affine expression
 * that is equal to "aff" on "domain".
 */
__isl_give isl_union_pw_aff *isl_union_pw_aff_aff_on_domain(
	__isl_take isl_union_set *domain, __isl_take isl_aff *aff)
{
	isl_pw_aff *pa;

	pa = isl_pw_aff_from_aff(aff);
	return isl_union_pw_aff_pw_aff_on_domain(domain, pa);
}

/* Return a union piecewise affine expression
 * that is equal to the parameter identified by "id" on "domain".
 *
 * Make sure the parameter appears in the space passed to
 * isl_aff_param_on_domain_space_id.
 */
__isl_give isl_union_pw_aff *isl_union_pw_aff_param_on_domain_id(
	__isl_take isl_union_set *domain, __isl_take isl_id *id)
{
	isl_space *space;
	isl_aff *aff;

	space = isl_union_set_get_space(domain);
	space = isl_space_add_param_id(space, isl_id_copy(id));
	aff = isl_aff_param_on_domain_space_id(space, id);
	return isl_union_pw_aff_aff_on_domain(domain, aff);
}

/* Internal data structure for isl_union_pw_aff_pw_aff_on_domain.
 * "pa" is the piecewise symbolic value that the resulting isl_union_pw_aff
 * needs to attain.
 * "res" collects the results.
 */
struct isl_union_pw_aff_pw_aff_on_domain_data {
	isl_pw_aff *pa;
	isl_union_pw_aff *res;
};

/* Construct a piecewise affine expression that is equal to data->pa
 * on "domain" and add the result to data->res.
 */
static isl_stat pw_aff_on_domain(__isl_take isl_set *domain, void *user)
{
	struct isl_union_pw_aff_pw_aff_on_domain_data *data = user;
	isl_pw_aff *pa;
	isl_size dim;

	pa = isl_pw_aff_copy(data->pa);
	dim = isl_set_dim(domain, isl_dim_set);
	if (dim < 0)
		pa = isl_pw_aff_free(pa);
	pa = isl_pw_aff_from_range(pa);
	pa = isl_pw_aff_add_dims(pa, isl_dim_in, dim);
	pa = isl_pw_aff_reset_domain_space(pa, isl_set_get_space(domain));
	pa = isl_pw_aff_intersect_domain(pa, domain);
	data->res = isl_union_pw_aff_add_pw_aff(data->res, pa);

	return data->res ? isl_stat_ok : isl_stat_error;
}

/* Return a union piecewise affine expression
 * that is equal to "pa" on "domain", assuming "domain" and "pa"
 * have been aligned.
 *
 * Construct an isl_pw_aff on each of the sets in "domain" and
 * collect the results.
 */
static __isl_give isl_union_pw_aff *isl_union_pw_aff_pw_aff_on_domain_aligned(
	__isl_take isl_union_set *domain, __isl_take isl_pw_aff *pa)
{
	struct isl_union_pw_aff_pw_aff_on_domain_data data;
	isl_space *space;

	space = isl_union_set_get_space(domain);
	data.res = isl_union_pw_aff_empty(space);
	data.pa = pa;
	if (isl_union_set_foreach_set(domain, &pw_aff_on_domain, &data) < 0)
		data.res = isl_union_pw_aff_free(data.res);
	isl_union_set_free(domain);
	isl_pw_aff_free(pa);
	return data.res;
}

/* Return a union piecewise affine expression
 * that is equal to "pa" on "domain".
 *
 * Check that "pa" is a parametric expression,
 * align the parameters if needed and call
 * isl_union_pw_aff_pw_aff_on_domain_aligned.
 */
__isl_give isl_union_pw_aff *isl_union_pw_aff_pw_aff_on_domain(
	__isl_take isl_union_set *domain, __isl_take isl_pw_aff *pa)
{
	isl_bool is_set;
	isl_bool equal_params;
	isl_space *domain_space, *pa_space;

	pa_space = isl_pw_aff_peek_space(pa);
	is_set = isl_space_is_set(pa_space);
	if (is_set < 0)
		goto error;
	if (!is_set)
		isl_die(isl_pw_aff_get_ctx(pa), isl_error_invalid,
			"expecting parametric expression", goto error);

	domain_space = isl_union_set_get_space(domain);
	pa_space = isl_pw_aff_get_space(pa);
	equal_params = isl_space_has_equal_params(domain_space, pa_space);
	if (equal_params >= 0 && !equal_params) {
		isl_space *space;

		space = isl_space_align_params(domain_space, pa_space);
		pa = isl_pw_aff_align_params(pa, isl_space_copy(space));
		domain = isl_union_set_align_params(domain, space);
	} else {
		isl_space_free(domain_space);
		isl_space_free(pa_space);
	}

	if (equal_params < 0)
		goto error;
	return isl_union_pw_aff_pw_aff_on_domain_aligned(domain, pa);
error:
	isl_union_set_free(domain);
	isl_pw_aff_free(pa);
	return NULL;
}

/* Internal data structure for isl_union_pw_aff_val_on_domain.
 * "v" is the value that the resulting isl_union_pw_aff needs to attain.
 * "res" collects the results.
 */
struct isl_union_pw_aff_val_on_domain_data {
	isl_val *v;
	isl_union_pw_aff *res;
};

/* Construct a piecewise affine expression that is equal to data->v
 * on "domain" and add the result to data->res.
 */
static isl_stat pw_aff_val_on_domain(__isl_take isl_set *domain, void *user)
{
	struct isl_union_pw_aff_val_on_domain_data *data = user;
	isl_pw_aff *pa;
	isl_val *v;

	v = isl_val_copy(data->v);
	pa = isl_pw_aff_val_on_domain(domain, v);
	data->res = isl_union_pw_aff_add_pw_aff(data->res, pa);

	return data->res ? isl_stat_ok : isl_stat_error;
}

/* Return a union piecewise affine expression
 * that is equal to "v" on "domain".
 *
 * Construct an isl_pw_aff on each of the sets in "domain" and
 * collect the results.
 */
__isl_give isl_union_pw_aff *isl_union_pw_aff_val_on_domain(
	__isl_take isl_union_set *domain, __isl_take isl_val *v)
{
	struct isl_union_pw_aff_val_on_domain_data data;
	isl_space *space;

	space = isl_union_set_get_space(domain);
	data.res = isl_union_pw_aff_empty(space);
	data.v = v;
	if (isl_union_set_foreach_set(domain, &pw_aff_val_on_domain, &data) < 0)
		data.res = isl_union_pw_aff_free(data.res);
	isl_union_set_free(domain);
	isl_val_free(v);
	return data.res;
}

/* Construct a piecewise multi affine expression
 * that is equal to "pa" and add it to upma.
 */
static isl_stat pw_multi_aff_from_pw_aff_entry(__isl_take isl_pw_aff *pa,
	void *user)
{
	isl_union_pw_multi_aff **upma = user;
	isl_pw_multi_aff *pma;

	pma = isl_pw_multi_aff_from_pw_aff(pa);
	*upma = isl_union_pw_multi_aff_add_pw_multi_aff(*upma, pma);

	return *upma ? isl_stat_ok : isl_stat_error;
}

/* Construct and return a union piecewise multi affine expression
 * that is equal to the given union piecewise affine expression.
 */
__isl_give isl_union_pw_multi_aff *isl_union_pw_multi_aff_from_union_pw_aff(
	__isl_take isl_union_pw_aff *upa)
{
	isl_space *space;
	isl_union_pw_multi_aff *upma;

	if (!upa)
		return NULL;

	space = isl_union_pw_aff_get_space(upa);
	upma = isl_union_pw_multi_aff_empty(space);

	if (isl_union_pw_aff_foreach_pw_aff(upa,
				&pw_multi_aff_from_pw_aff_entry, &upma) < 0)
		upma = isl_union_pw_multi_aff_free(upma);

	isl_union_pw_aff_free(upa);
	return upma;
}

/* Compute the set of elements in the domain of "pa" where it is zero and
 * add this set to "uset".
 */
static isl_stat zero_union_set(__isl_take isl_pw_aff *pa, void *user)
{
	isl_union_set **uset = (isl_union_set **)user;

	*uset = isl_union_set_add_set(*uset, isl_pw_aff_zero_set(pa));

	return *uset ? isl_stat_ok : isl_stat_error;
}

/* Return a union set containing those elements in the domain
 * of "upa" where it is zero.
 */
__isl_give isl_union_set *isl_union_pw_aff_zero_union_set(
	__isl_take isl_union_pw_aff *upa)
{
	isl_union_set *zero;

	zero = isl_union_set_empty(isl_union_pw_aff_get_space(upa));
	if (isl_union_pw_aff_foreach_pw_aff(upa, &zero_union_set, &zero) < 0)
		zero = isl_union_set_free(zero);

	isl_union_pw_aff_free(upa);
	return zero;
}

/* Internal data structure for isl_union_pw_aff_bind_id,
 * storing the parameter that needs to be bound and
 * the accumulated results.
 */
struct isl_bind_id_data {
	isl_id *id;
	isl_union_set *bound;
};

/* Bind the piecewise affine function "pa" to the parameter data->id,
 * adding the resulting elements in the domain where the expression
 * is equal to the parameter to data->bound.
 */
static isl_stat bind_id(__isl_take isl_pw_aff *pa, void *user)
{
	struct isl_bind_id_data *data = user;
	isl_set *bound;

	bound = isl_pw_aff_bind_id(pa, isl_id_copy(data->id));
	data->bound = isl_union_set_add_set(data->bound, bound);

	return data->bound ? isl_stat_ok : isl_stat_error;
}

/* Bind the union piecewise affine function "upa" to the parameter "id",
 * returning the elements in the domain where the expression
 * is equal to the parameter.
 */
__isl_give isl_union_set *isl_union_pw_aff_bind_id(
	__isl_take isl_union_pw_aff *upa, __isl_take isl_id *id)
{
	struct isl_bind_id_data data = { id };

	data.bound = isl_union_set_empty(isl_union_pw_aff_get_space(upa));
	if (isl_union_pw_aff_foreach_pw_aff(upa, &bind_id, &data) < 0)
		data.bound = isl_union_set_free(data.bound);

	isl_union_pw_aff_free(upa);
	isl_id_free(id);
	return data.bound;
}

/* Internal data structure for isl_union_pw_aff_pullback_union_pw_multi_aff.
 * upma is the function that is plugged in.
 * pa is the current part of the function in which upma is plugged in.
 * res collects the results.
 */
struct isl_union_pw_aff_pullback_upma_data {
	isl_union_pw_multi_aff *upma;
	isl_pw_aff *pa;
	isl_union_pw_aff *res;
};

/* Check if "pma" can be plugged into data->pa.
 * If so, perform the pullback and add the result to data->res.
 */
static isl_stat pa_pb_pma(__isl_take isl_pw_multi_aff *pma, void *user)
{
	struct isl_union_pw_aff_pullback_upma_data *data = user;
	isl_pw_aff *pa;

	if (!isl_space_tuple_is_equal(data->pa->dim, isl_dim_in,
				 pma->dim, isl_dim_out)) {
		isl_pw_multi_aff_free(pma);
		return isl_stat_ok;
	}

	pa = isl_pw_aff_copy(data->pa);
	pa = isl_pw_aff_pullback_pw_multi_aff(pa, pma);

	data->res = isl_union_pw_aff_add_pw_aff(data->res, pa);

	return data->res ? isl_stat_ok : isl_stat_error;
}

/* Check if any of the elements of data->upma can be plugged into pa,
 * add if so add the result to data->res.
 */
static isl_stat upa_pb_upma(__isl_take isl_pw_aff *pa, void *user)
{
	struct isl_union_pw_aff_pullback_upma_data *data = user;
	isl_stat r;

	data->pa = pa;
	r = isl_union_pw_multi_aff_foreach_pw_multi_aff(data->upma,
				   &pa_pb_pma, data);
	isl_pw_aff_free(pa);

	return r;
}

/* Compute the pullback of "upa" by the function represented by "upma".
 * In other words, plug in "upma" in "upa".  The result contains
 * expressions defined over the domain space of "upma".
 *
 * Run over all pairs of elements in "upa" and "upma", perform
 * the pullback when appropriate and collect the results.
 * If the hash value were based on the domain space rather than
 * the function space, then we could run through all elements
 * of "upma" and directly pick out the corresponding element of "upa".
 */
__isl_give isl_union_pw_aff *isl_union_pw_aff_pullback_union_pw_multi_aff(
	__isl_take isl_union_pw_aff *upa,
	__isl_take isl_union_pw_multi_aff *upma)
{
	struct isl_union_pw_aff_pullback_upma_data data = { NULL, NULL };
	isl_space *space;

	space = isl_union_pw_multi_aff_get_space(upma);
	upa = isl_union_pw_aff_align_params(upa, space);
	space = isl_union_pw_aff_get_space(upa);
	upma = isl_union_pw_multi_aff_align_params(upma, space);

	if (!upa || !upma)
		goto error;

	data.upma = upma;
	data.res = isl_union_pw_aff_alloc_same_size(upa);
	if (isl_union_pw_aff_foreach_pw_aff(upa, &upa_pb_upma, &data) < 0)
		data.res = isl_union_pw_aff_free(data.res);

	isl_union_pw_aff_free(upa);
	isl_union_pw_multi_aff_free(upma);
	return data.res;
error:
	isl_union_pw_aff_free(upa);
	isl_union_pw_multi_aff_free(upma);
	return NULL;
}

#undef BASE
#define BASE union_pw_aff
#undef DOMBASE
#define DOMBASE union_set

#include <isl_multi_explicit_domain.c>
#include <isl_multi_union_pw_aff_explicit_domain.c>
#include <isl_multi_templ.c>
#include <isl_multi_apply_set.c>
#include <isl_multi_apply_union_set.c>
#include <isl_multi_arith_templ.c>
#include <isl_multi_bind_templ.c>
#include <isl_multi_coalesce.c>
#include <isl_multi_dim_id_templ.c>
#include <isl_multi_floor.c>
#include <isl_multi_from_base_templ.c>
#include <isl_multi_gist.c>
#include <isl_multi_align_set.c>
#include <isl_multi_align_union_set.c>
#include <isl_multi_intersect.c>
#include <isl_multi_nan_templ.c>
#include <isl_multi_tuple_id_templ.c>
#include <isl_multi_union_add_templ.c>
#include <isl_multi_zero_space_templ.c>

/* Does "mupa" have a non-trivial explicit domain?
 *
 * The explicit domain, if present, is trivial if it represents
 * an (obviously) universe parameter set.
 */
isl_bool isl_multi_union_pw_aff_has_non_trivial_domain(
	__isl_keep isl_multi_union_pw_aff *mupa)
{
	isl_bool is_params, trivial;
	isl_set *set;

	if (!mupa)
		return isl_bool_error;
	if (!isl_multi_union_pw_aff_has_explicit_domain(mupa))
		return isl_bool_false;
	is_params = isl_union_set_is_params(mupa->u.dom);
	if (is_params < 0 || !is_params)
		return isl_bool_not(is_params);
	set = isl_set_from_union_set(isl_union_set_copy(mupa->u.dom));
	trivial = isl_set_plain_is_universe(set);
	isl_set_free(set);
	return isl_bool_not(trivial);
}

/* Construct a multiple union piecewise affine expression
 * in the given space with value zero in each of the output dimensions.
 *
 * Since there is no canonical zero value for
 * a union piecewise affine expression, we can only construct
 * a zero-dimensional "zero" value.
 */
__isl_give isl_multi_union_pw_aff *isl_multi_union_pw_aff_zero(
	__isl_take isl_space *space)
{
	isl_bool params;
	isl_size dim;

	if (!space)
		return NULL;

	params = isl_space_is_params(space);
	if (params < 0)
		goto error;
	if (params)
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"expecting proper set space", goto error);
	if (!isl_space_is_set(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"expecting set space", goto error);
	dim = isl_space_dim(space, isl_dim_out);
	if (dim < 0)
		goto error;
	if (dim != 0)
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"expecting 0D space", goto error);

	return isl_multi_union_pw_aff_alloc(space);
error:
	isl_space_free(space);
	return NULL;
}

/* Construct and return a multi union piecewise affine expression
 * that is equal to the given multi affine expression.
 */
__isl_give isl_multi_union_pw_aff *isl_multi_union_pw_aff_from_multi_aff(
	__isl_take isl_multi_aff *ma)
{
	isl_multi_pw_aff *mpa;

	mpa = isl_multi_pw_aff_from_multi_aff(ma);
	return isl_multi_union_pw_aff_from_multi_pw_aff(mpa);
}

/* This function performs the same operation as
 * isl_multi_union_pw_aff_from_multi_aff, but is considered as a function on an
 * isl_multi_aff when exported.
 */
__isl_give isl_multi_union_pw_aff *isl_multi_aff_to_multi_union_pw_aff(
        __isl_take isl_multi_aff *ma)
{
        return isl_multi_union_pw_aff_from_multi_aff(ma);
}

/* Construct and return a multi union piecewise affine expression
 * that is equal to the given multi piecewise affine expression.
 */
__isl_give isl_multi_union_pw_aff *isl_multi_union_pw_aff_from_multi_pw_aff(
	__isl_take isl_multi_pw_aff *mpa)
{
	int i;
	isl_size n;
	isl_space *space;
	isl_multi_union_pw_aff *mupa;

	n = isl_multi_pw_aff_dim(mpa, isl_dim_out);
	if (n < 0)
		mpa = isl_multi_pw_aff_free(mpa);
	if (!mpa)
		return NULL;

	space = isl_multi_pw_aff_get_space(mpa);
	space = isl_space_range(space);
	mupa = isl_multi_union_pw_aff_alloc(space);

	for (i = 0; i < n; ++i) {
		isl_pw_aff *pa;
		isl_union_pw_aff *upa;

		pa = isl_multi_pw_aff_get_pw_aff(mpa, i);
		upa = isl_union_pw_aff_from_pw_aff(pa);
		mupa = isl_multi_union_pw_aff_restore_check_space(mupa, i, upa);
	}

	isl_multi_pw_aff_free(mpa);

	return mupa;
}

/* Extract the range space of "pma" and assign it to *space.
 * If *space has already been set (through a previous call to this function),
 * then check that the range space is the same.
 */
static isl_stat extract_space(__isl_take isl_pw_multi_aff *pma, void *user)
{
	isl_space **space = user;
	isl_space *pma_space;
	isl_bool equal;

	pma_space = isl_space_range(isl_pw_multi_aff_get_space(pma));
	isl_pw_multi_aff_free(pma);

	if (!pma_space)
		return isl_stat_error;
	if (!*space) {
		*space = pma_space;
		return isl_stat_ok;
	}

	equal = isl_space_is_equal(pma_space, *space);
	isl_space_free(pma_space);

	if (equal < 0)
		return isl_stat_error;
	if (!equal)
		isl_die(isl_space_get_ctx(*space), isl_error_invalid,
			"range spaces not the same", return isl_stat_error);
	return isl_stat_ok;
}

/* Construct and return a multi union piecewise affine expression
 * that is equal to the given union piecewise multi affine expression.
 *
 * In order to be able to perform the conversion, the input
 * needs to be non-empty and may only involve a single range space.
 *
 * If the resulting multi union piecewise affine expression has
 * an explicit domain, then assign it the domain of the input.
 * In other cases, the domain is stored in the individual elements.
 */
__isl_give isl_multi_union_pw_aff *
isl_multi_union_pw_aff_from_union_pw_multi_aff(
	__isl_take isl_union_pw_multi_aff *upma)
{
	isl_space *space = NULL;
	isl_multi_union_pw_aff *mupa;
	int i;
	isl_size n;

	n = isl_union_pw_multi_aff_n_pw_multi_aff(upma);
	if (n < 0)
		goto error;
	if (n == 0)
		isl_die(isl_union_pw_multi_aff_get_ctx(upma), isl_error_invalid,
			"cannot extract range space from empty input",
			goto error);
	if (isl_union_pw_multi_aff_foreach_pw_multi_aff(upma, &extract_space,
							&space) < 0)
		goto error;

	if (!space)
		goto error;

	n = isl_space_dim(space, isl_dim_set);
	if (n < 0)
		space = isl_space_free(space);
	mupa = isl_multi_union_pw_aff_alloc(space);

	for (i = 0; i < n; ++i) {
		isl_union_pw_aff *upa;

		upa = isl_union_pw_multi_aff_get_union_pw_aff(upma, i);
		mupa = isl_multi_union_pw_aff_set_union_pw_aff(mupa, i, upa);
	}
	if (isl_multi_union_pw_aff_has_explicit_domain(mupa)) {
		isl_union_set *dom;
		isl_union_pw_multi_aff *copy;

		copy = isl_union_pw_multi_aff_copy(upma);
		dom = isl_union_pw_multi_aff_domain(copy);
		mupa = isl_multi_union_pw_aff_intersect_domain(mupa, dom);
	}

	isl_union_pw_multi_aff_free(upma);
	return mupa;
error:
	isl_space_free(space);
	isl_union_pw_multi_aff_free(upma);
	return NULL;
}

/* This function performs the same operation as
 * isl_multi_union_pw_aff_from_union_pw_multi_aff,
 * but is considered as a function on an isl_union_pw_multi_aff when exported.
 */
__isl_give isl_multi_union_pw_aff *
isl_union_pw_multi_aff_as_multi_union_pw_aff(
	__isl_take isl_union_pw_multi_aff *upma)
{
	return isl_multi_union_pw_aff_from_union_pw_multi_aff(upma);
}

/* Try and create an isl_multi_union_pw_aff that is equivalent
 * to the given isl_union_map.
 * The isl_union_map is required to be single-valued in each space.
 * Moreover, it cannot be empty and all range spaces need to be the same.
 * Otherwise, an error is produced.
 */
__isl_give isl_multi_union_pw_aff *isl_multi_union_pw_aff_from_union_map(
	__isl_take isl_union_map *umap)
{
	isl_union_pw_multi_aff *upma;

	upma = isl_union_pw_multi_aff_from_union_map(umap);
	return isl_multi_union_pw_aff_from_union_pw_multi_aff(upma);
}

/* This function performs the same operation as
 * isl_multi_union_pw_aff_from_union_map,
 * but is considered as a function on an isl_union_map when exported.
 */
__isl_give isl_multi_union_pw_aff *isl_union_map_as_multi_union_pw_aff(
	__isl_take isl_union_map *umap)
{
	return isl_multi_union_pw_aff_from_union_map(umap);
}

/* Return a multiple union piecewise affine expression
 * that is equal to "mv" on "domain", assuming "domain" and "mv"
 * have been aligned.
 *
 * If the resulting multi union piecewise affine expression has
 * an explicit domain, then assign it the input domain.
 * In other cases, the domain is stored in the individual elements.
 */
static __isl_give isl_multi_union_pw_aff *
isl_multi_union_pw_aff_multi_val_on_domain_aligned(
	__isl_take isl_union_set *domain, __isl_take isl_multi_val *mv)
{
	int i;
	isl_size n;
	isl_space *space;
	isl_multi_union_pw_aff *mupa;

	n = isl_multi_val_dim(mv, isl_dim_set);
	if (!domain || n < 0)
		goto error;

	space = isl_multi_val_get_space(mv);
	mupa = isl_multi_union_pw_aff_alloc(space);
	for (i = 0; i < n; ++i) {
		isl_val *v;
		isl_union_pw_aff *upa;

		v = isl_multi_val_get_val(mv, i);
		upa = isl_union_pw_aff_val_on_domain(isl_union_set_copy(domain),
							v);
		mupa = isl_multi_union_pw_aff_set_union_pw_aff(mupa, i, upa);
	}
	if (isl_multi_union_pw_aff_has_explicit_domain(mupa))
		mupa = isl_multi_union_pw_aff_intersect_domain(mupa,
						    isl_union_set_copy(domain));

	isl_union_set_free(domain);
	isl_multi_val_free(mv);
	return mupa;
error:
	isl_union_set_free(domain);
	isl_multi_val_free(mv);
	return NULL;
}

/* Return a multiple union piecewise affine expression
 * that is equal to "mv" on "domain".
 */
__isl_give isl_multi_union_pw_aff *isl_multi_union_pw_aff_multi_val_on_domain(
	__isl_take isl_union_set *domain, __isl_take isl_multi_val *mv)
{
	isl_bool equal_params;

	if (!domain || !mv)
		goto error;
	equal_params = isl_space_has_equal_params(domain->dim, mv->space);
	if (equal_params < 0)
		goto error;
	if (equal_params)
		return isl_multi_union_pw_aff_multi_val_on_domain_aligned(
								    domain, mv);
	domain = isl_union_set_align_params(domain,
						isl_multi_val_get_space(mv));
	mv = isl_multi_val_align_params(mv, isl_union_set_get_space(domain));
	return isl_multi_union_pw_aff_multi_val_on_domain_aligned(domain, mv);
error:
	isl_union_set_free(domain);
	isl_multi_val_free(mv);
	return NULL;
}

/* Return a multiple union piecewise affine expression
 * that is equal to "ma" on "domain".
 */
__isl_give isl_multi_union_pw_aff *isl_multi_union_pw_aff_multi_aff_on_domain(
	__isl_take isl_union_set *domain, __isl_take isl_multi_aff *ma)
{
	isl_pw_multi_aff *pma;

	pma = isl_pw_multi_aff_from_multi_aff(ma);
	return isl_multi_union_pw_aff_pw_multi_aff_on_domain(domain, pma);
}

/* Return a multiple union piecewise affine expression
 * that is equal to "pma" on "domain", assuming "domain" and "pma"
 * have been aligned.
 *
 * If the resulting multi union piecewise affine expression has
 * an explicit domain, then assign it the input domain.
 * In other cases, the domain is stored in the individual elements.
 */
static __isl_give isl_multi_union_pw_aff *
isl_multi_union_pw_aff_pw_multi_aff_on_domain_aligned(
	__isl_take isl_union_set *domain, __isl_take isl_pw_multi_aff *pma)
{
	int i;
	isl_size n;
	isl_space *space;
	isl_multi_union_pw_aff *mupa;

	n = isl_pw_multi_aff_dim(pma, isl_dim_set);
	if (!domain || n < 0)
		goto error;
	space = isl_pw_multi_aff_get_space(pma);
	mupa = isl_multi_union_pw_aff_alloc(space);
	for (i = 0; i < n; ++i) {
		isl_pw_aff *pa;
		isl_union_pw_aff *upa;

		pa = isl_pw_multi_aff_get_pw_aff(pma, i);
		upa = isl_union_pw_aff_pw_aff_on_domain(
					    isl_union_set_copy(domain), pa);
		mupa = isl_multi_union_pw_aff_set_union_pw_aff(mupa, i, upa);
	}
	if (isl_multi_union_pw_aff_has_explicit_domain(mupa))
		mupa = isl_multi_union_pw_aff_intersect_domain(mupa,
						    isl_union_set_copy(domain));

	isl_union_set_free(domain);
	isl_pw_multi_aff_free(pma);
	return mupa;
error:
	isl_union_set_free(domain);
	isl_pw_multi_aff_free(pma);
	return NULL;
}

/* Return a multiple union piecewise affine expression
 * that is equal to "pma" on "domain".
 */
__isl_give isl_multi_union_pw_aff *
isl_multi_union_pw_aff_pw_multi_aff_on_domain(__isl_take isl_union_set *domain,
	__isl_take isl_pw_multi_aff *pma)
{
	isl_bool equal_params;
	isl_space *space;

	space = isl_pw_multi_aff_peek_space(pma);
	equal_params = isl_union_set_space_has_equal_params(domain, space);
	if (equal_params < 0)
		goto error;
	if (equal_params)
		return isl_multi_union_pw_aff_pw_multi_aff_on_domain_aligned(
								domain, pma);
	domain = isl_union_set_align_params(domain,
					    isl_pw_multi_aff_get_space(pma));
	pma = isl_pw_multi_aff_align_params(pma,
					    isl_union_set_get_space(domain));
	return isl_multi_union_pw_aff_pw_multi_aff_on_domain_aligned(domain,
									pma);
error:
	isl_union_set_free(domain);
	isl_pw_multi_aff_free(pma);
	return NULL;
}

/* Return a union set containing those elements in the domains
 * of the elements of "mupa" where they are all zero.
 *
 * If there are no elements, then simply return the entire domain.
 */
__isl_give isl_union_set *isl_multi_union_pw_aff_zero_union_set(
	__isl_take isl_multi_union_pw_aff *mupa)
{
	int i;
	isl_size n;
	isl_union_pw_aff *upa;
	isl_union_set *zero;

	n = isl_multi_union_pw_aff_dim(mupa, isl_dim_set);
	if (n < 0)
		mupa = isl_multi_union_pw_aff_free(mupa);
	if (!mupa)
		return NULL;

	if (n == 0)
		return isl_multi_union_pw_aff_domain(mupa);

	upa = isl_multi_union_pw_aff_get_union_pw_aff(mupa, 0);
	zero = isl_union_pw_aff_zero_union_set(upa);

	for (i = 1; i < n; ++i) {
		isl_union_set *zero_i;

		upa = isl_multi_union_pw_aff_get_union_pw_aff(mupa, i);
		zero_i = isl_union_pw_aff_zero_union_set(upa);

		zero = isl_union_set_intersect(zero, zero_i);
	}

	isl_multi_union_pw_aff_free(mupa);
	return zero;
}

/* Construct a union map mapping the shared domain
 * of the union piecewise affine expressions to the range of "mupa"
 * in the special case of a 0D multi union piecewise affine expression.
 *
 * Construct a map between the explicit domain of "mupa" and
 * the range space.
 * Note that this assumes that the domain consists of explicit elements.
 */
static __isl_give isl_union_map *isl_union_map_from_multi_union_pw_aff_0D(
	__isl_take isl_multi_union_pw_aff *mupa)
{
	isl_bool is_params;
	isl_space *space;
	isl_union_set *dom, *ran;

	space = isl_multi_union_pw_aff_get_space(mupa);
	dom = isl_multi_union_pw_aff_domain(mupa);
	ran = isl_union_set_from_set(isl_set_universe(space));

	is_params = isl_union_set_is_params(dom);
	if (is_params < 0)
		dom = isl_union_set_free(dom);
	else if (is_params)
		isl_die(isl_union_set_get_ctx(dom), isl_error_invalid,
			"cannot create union map from expression without "
			"explicit domain elements",
			dom = isl_union_set_free(dom));

	return isl_union_map_from_domain_and_range(dom, ran);
}

/* Construct a union map mapping the shared domain
 * of the union piecewise affine expressions to the range of "mupa"
 * with each dimension in the range equated to the
 * corresponding union piecewise affine expression.
 *
 * If the input is zero-dimensional, then construct a mapping
 * from its explicit domain.
 */
__isl_give isl_union_map *isl_union_map_from_multi_union_pw_aff(
	__isl_take isl_multi_union_pw_aff *mupa)
{
	int i;
	isl_size n;
	isl_space *space;
	isl_union_map *umap;
	isl_union_pw_aff *upa;

	n = isl_multi_union_pw_aff_dim(mupa, isl_dim_set);
	if (n < 0)
		mupa = isl_multi_union_pw_aff_free(mupa);
	if (!mupa)
		return NULL;

	if (n == 0)
		return isl_union_map_from_multi_union_pw_aff_0D(mupa);

	upa = isl_multi_union_pw_aff_get_union_pw_aff(mupa, 0);
	umap = isl_union_map_from_union_pw_aff(upa);

	for (i = 1; i < n; ++i) {
		isl_union_map *umap_i;

		upa = isl_multi_union_pw_aff_get_union_pw_aff(mupa, i);
		umap_i = isl_union_map_from_union_pw_aff(upa);
		umap = isl_union_map_flat_range_product(umap, umap_i);
	}

	space = isl_multi_union_pw_aff_get_space(mupa);
	umap = isl_union_map_reset_range_space(umap, space);

	isl_multi_union_pw_aff_free(mupa);
	return umap;
}

/* Internal data structure for isl_union_pw_multi_aff_reset_range_space.
 * "range" is the space from which to set the range space.
 * "res" collects the results.
 */
struct isl_union_pw_multi_aff_reset_range_space_data {
	isl_space *range;
	isl_union_pw_multi_aff *res;
};

/* Replace the range space of "pma" by the range space of data->range and
 * add the result to data->res.
 */
static isl_stat reset_range_space(__isl_take isl_pw_multi_aff *pma, void *user)
{
	struct isl_union_pw_multi_aff_reset_range_space_data *data = user;
	isl_space *space;

	space = isl_pw_multi_aff_get_space(pma);
	space = isl_space_domain(space);
	space = isl_space_extend_domain_with_range(space,
						isl_space_copy(data->range));
	pma = isl_pw_multi_aff_reset_space(pma, space);
	data->res = isl_union_pw_multi_aff_add_pw_multi_aff(data->res, pma);

	return data->res ? isl_stat_ok : isl_stat_error;
}

/* Replace the range space of all the piecewise affine expressions in "upma" by
 * the range space of "space".
 *
 * This assumes that all these expressions have the same output dimension.
 *
 * Since the spaces of the expressions change, so do their hash values.
 * We therefore need to create a new isl_union_pw_multi_aff.
 * Note that the hash value is currently computed based on the entire
 * space even though there can only be a single expression with a given
 * domain space.
 */
static __isl_give isl_union_pw_multi_aff *
isl_union_pw_multi_aff_reset_range_space(
	__isl_take isl_union_pw_multi_aff *upma, __isl_take isl_space *space)
{
	struct isl_union_pw_multi_aff_reset_range_space_data data = { space };
	isl_space *space_upma;

	space_upma = isl_union_pw_multi_aff_get_space(upma);
	data.res = isl_union_pw_multi_aff_empty(space_upma);
	if (isl_union_pw_multi_aff_foreach_pw_multi_aff(upma,
					&reset_range_space, &data) < 0)
		data.res = isl_union_pw_multi_aff_free(data.res);

	isl_space_free(space);
	isl_union_pw_multi_aff_free(upma);
	return data.res;
}

/* Construct and return a union piecewise multi affine expression
 * that is equal to the given multi union piecewise affine expression,
 * in the special case of a 0D multi union piecewise affine expression.
 *
 * Construct a union piecewise multi affine expression
 * on top of the explicit domain of the input.
 */
__isl_give isl_union_pw_multi_aff *
isl_union_pw_multi_aff_from_multi_union_pw_aff_0D(
	__isl_take isl_multi_union_pw_aff *mupa)
{
	isl_space *space;
	isl_multi_val *mv;
	isl_union_set *domain;

	space = isl_multi_union_pw_aff_get_space(mupa);
	mv = isl_multi_val_zero(space);
	domain = isl_multi_union_pw_aff_domain(mupa);
	return isl_union_pw_multi_aff_multi_val_on_domain(domain, mv);
}

/* Construct and return a union piecewise multi affine expression
 * that is equal to the given multi union piecewise affine expression.
 *
 * If the input is zero-dimensional, then
 * construct a union piecewise multi affine expression
 * on top of the explicit domain of the input.
 */
__isl_give isl_union_pw_multi_aff *
isl_union_pw_multi_aff_from_multi_union_pw_aff(
	__isl_take isl_multi_union_pw_aff *mupa)
{
	int i;
	isl_size n;
	isl_space *space;
	isl_union_pw_multi_aff *upma;
	isl_union_pw_aff *upa;

	n = isl_multi_union_pw_aff_dim(mupa, isl_dim_set);
	if (n < 0)
		mupa = isl_multi_union_pw_aff_free(mupa);
	if (!mupa)
		return NULL;

	if (n == 0)
		return isl_union_pw_multi_aff_from_multi_union_pw_aff_0D(mupa);

	space = isl_multi_union_pw_aff_get_space(mupa);
	upa = isl_multi_union_pw_aff_get_union_pw_aff(mupa, 0);
	upma = isl_union_pw_multi_aff_from_union_pw_aff(upa);

	for (i = 1; i < n; ++i) {
		isl_union_pw_multi_aff *upma_i;

		upa = isl_multi_union_pw_aff_get_union_pw_aff(mupa, i);
		upma_i = isl_union_pw_multi_aff_from_union_pw_aff(upa);
		upma = isl_union_pw_multi_aff_flat_range_product(upma, upma_i);
	}

	upma = isl_union_pw_multi_aff_reset_range_space(upma, space);

	isl_multi_union_pw_aff_free(mupa);
	return upma;
}

/* Intersect the range of "mupa" with "range",
 * in the special case where "mupa" is 0D.
 *
 * Intersect the domain of "mupa" with the constraints on the parameters
 * of "range".
 */
static __isl_give isl_multi_union_pw_aff *mupa_intersect_range_0D(
	__isl_take isl_multi_union_pw_aff *mupa, __isl_take isl_set *range)
{
	range = isl_set_params(range);
	mupa = isl_multi_union_pw_aff_intersect_params(mupa, range);
	return mupa;
}

/* Intersect the range of "mupa" with "range".
 * That is, keep only those domain elements that have a function value
 * in "range".
 */
__isl_give isl_multi_union_pw_aff *isl_multi_union_pw_aff_intersect_range(
	__isl_take isl_multi_union_pw_aff *mupa, __isl_take isl_set *range)
{
	isl_union_pw_multi_aff *upma;
	isl_union_set *domain;
	isl_space *space;
	isl_size n;
	int match;

	n = isl_multi_union_pw_aff_dim(mupa, isl_dim_set);
	if (n < 0 || !range)
		goto error;

	space = isl_set_get_space(range);
	match = isl_space_tuple_is_equal(mupa->space, isl_dim_set,
					space, isl_dim_set);
	isl_space_free(space);
	if (match < 0)
		goto error;
	if (!match)
		isl_die(isl_multi_union_pw_aff_get_ctx(mupa), isl_error_invalid,
			"space don't match", goto error);
	if (n == 0)
		return mupa_intersect_range_0D(mupa, range);

	upma = isl_union_pw_multi_aff_from_multi_union_pw_aff(
					isl_multi_union_pw_aff_copy(mupa));
	domain = isl_union_set_from_set(range);
	domain = isl_union_set_preimage_union_pw_multi_aff(domain, upma);
	mupa = isl_multi_union_pw_aff_intersect_domain(mupa, domain);

	return mupa;
error:
	isl_multi_union_pw_aff_free(mupa);
	isl_set_free(range);
	return NULL;
}

/* Return the shared domain of the elements of "mupa",
 * in the special case where "mupa" is zero-dimensional.
 *
 * Return the explicit domain of "mupa".
 * Note that this domain may be a parameter set, either
 * because "mupa" is meant to live in a set space or
 * because no explicit domain has been set.
 */
__isl_give isl_union_set *isl_multi_union_pw_aff_domain_0D(
	__isl_take isl_multi_union_pw_aff *mupa)
{
	isl_union_set *dom;

	dom = isl_multi_union_pw_aff_get_explicit_domain(mupa);
	isl_multi_union_pw_aff_free(mupa);

	return dom;
}

/* Return the shared domain of the elements of "mupa".
 *
 * If "mupa" is zero-dimensional, then return its explicit domain.
 */
__isl_give isl_union_set *isl_multi_union_pw_aff_domain(
	__isl_take isl_multi_union_pw_aff *mupa)
{
	int i;
	isl_size n;
	isl_union_pw_aff *upa;
	isl_union_set *dom;

	n = isl_multi_union_pw_aff_dim(mupa, isl_dim_set);
	if (n < 0)
		mupa = isl_multi_union_pw_aff_free(mupa);
	if (!mupa)
		return NULL;

	if (n == 0)
		return isl_multi_union_pw_aff_domain_0D(mupa);

	upa = isl_multi_union_pw_aff_get_union_pw_aff(mupa, 0);
	dom = isl_union_pw_aff_domain(upa);
	for (i = 1; i < n; ++i) {
		isl_union_set *dom_i;

		upa = isl_multi_union_pw_aff_get_union_pw_aff(mupa, i);
		dom_i = isl_union_pw_aff_domain(upa);
		dom = isl_union_set_intersect(dom, dom_i);
	}

	isl_multi_union_pw_aff_free(mupa);
	return dom;
}

/* Apply "aff" to "mupa".  The space of "mupa" is equal to the domain of "aff".
 * In particular, the spaces have been aligned.
 * The result is defined over the shared domain of the elements of "mupa"
 *
 * We first extract the parametric constant part of "aff" and
 * define that over the shared domain.
 * Then we iterate over all input dimensions of "aff" and add the corresponding
 * multiples of the elements of "mupa".
 * Finally, we consider the integer divisions, calling the function
 * recursively to obtain an isl_union_pw_aff corresponding to the
 * integer division argument.
 */
static __isl_give isl_union_pw_aff *multi_union_pw_aff_apply_aff(
	__isl_take isl_multi_union_pw_aff *mupa, __isl_take isl_aff *aff)
{
	int i;
	isl_size n_in, n_div;
	isl_union_pw_aff *upa;
	isl_union_set *uset;
	isl_val *v;
	isl_aff *cst;

	n_in = isl_aff_dim(aff, isl_dim_in);
	n_div = isl_aff_dim(aff, isl_dim_div);
	if (n_in < 0 || n_div < 0)
		goto error;

	uset = isl_multi_union_pw_aff_domain(isl_multi_union_pw_aff_copy(mupa));
	cst = isl_aff_copy(aff);
	cst = isl_aff_drop_dims(cst, isl_dim_div, 0, n_div);
	cst = isl_aff_drop_dims(cst, isl_dim_in, 0, n_in);
	cst = isl_aff_project_domain_on_params(cst);
	upa = isl_union_pw_aff_aff_on_domain(uset, cst);

	for (i = 0; i < n_in; ++i) {
		isl_union_pw_aff *upa_i;

		if (!isl_aff_involves_dims(aff, isl_dim_in, i, 1))
			continue;
		v = isl_aff_get_coefficient_val(aff, isl_dim_in, i);
		upa_i = isl_multi_union_pw_aff_get_union_pw_aff(mupa, i);
		upa_i = isl_union_pw_aff_scale_val(upa_i, v);
		upa = isl_union_pw_aff_add(upa, upa_i);
	}

	for (i = 0; i < n_div; ++i) {
		isl_aff *div;
		isl_union_pw_aff *upa_i;

		if (!isl_aff_involves_dims(aff, isl_dim_div, i, 1))
			continue;
		div = isl_aff_get_div(aff, i);
		upa_i = multi_union_pw_aff_apply_aff(
					isl_multi_union_pw_aff_copy(mupa), div);
		upa_i = isl_union_pw_aff_floor(upa_i);
		v = isl_aff_get_coefficient_val(aff, isl_dim_div, i);
		upa_i = isl_union_pw_aff_scale_val(upa_i, v);
		upa = isl_union_pw_aff_add(upa, upa_i);
	}

	isl_multi_union_pw_aff_free(mupa);
	isl_aff_free(aff);

	return upa;
error:
	isl_multi_union_pw_aff_free(mupa);
	isl_aff_free(aff);
	return NULL;
}

/* Apply "aff" to "mupa".  The space of "mupa" needs to be compatible
 * with the domain of "aff".
 * Furthermore, the dimension of this space needs to be greater than zero.
 * The result is defined over the shared domain of the elements of "mupa"
 *
 * We perform these checks and then hand over control to
 * multi_union_pw_aff_apply_aff.
 */
__isl_give isl_union_pw_aff *isl_multi_union_pw_aff_apply_aff(
	__isl_take isl_multi_union_pw_aff *mupa, __isl_take isl_aff *aff)
{
	isl_size dim;
	isl_space *space1, *space2;
	isl_bool equal;

	mupa = isl_multi_union_pw_aff_align_params(mupa,
						isl_aff_get_space(aff));
	aff = isl_aff_align_params(aff, isl_multi_union_pw_aff_get_space(mupa));
	if (!mupa || !aff)
		goto error;

	space1 = isl_multi_union_pw_aff_get_space(mupa);
	space2 = isl_aff_get_domain_space(aff);
	equal = isl_space_is_equal(space1, space2);
	isl_space_free(space1);
	isl_space_free(space2);
	if (equal < 0)
		goto error;
	if (!equal)
		isl_die(isl_aff_get_ctx(aff), isl_error_invalid,
			"spaces don't match", goto error);
	dim = isl_aff_dim(aff, isl_dim_in);
	if (dim < 0)
		goto error;
	if (dim == 0)
		isl_die(isl_aff_get_ctx(aff), isl_error_invalid,
			"cannot determine domains", goto error);

	return multi_union_pw_aff_apply_aff(mupa, aff);
error:
	isl_multi_union_pw_aff_free(mupa);
	isl_aff_free(aff);
	return NULL;
}

/* Apply "ma" to "mupa", in the special case where "mupa" is 0D.
 * The space of "mupa" is known to be compatible with the domain of "ma".
 *
 * Construct an isl_multi_union_pw_aff that is equal to "ma"
 * on the domain of "mupa".
 */
static __isl_give isl_multi_union_pw_aff *mupa_apply_multi_aff_0D(
	__isl_take isl_multi_union_pw_aff *mupa, __isl_take isl_multi_aff *ma)
{
	isl_union_set *dom;

	dom = isl_multi_union_pw_aff_domain(mupa);
	ma = isl_multi_aff_project_domain_on_params(ma);

	return isl_multi_union_pw_aff_multi_aff_on_domain(dom, ma);
}

/* Apply "ma" to "mupa".  The space of "mupa" needs to be compatible
 * with the domain of "ma".
 * The result is defined over the shared domain of the elements of "mupa"
 */
__isl_give isl_multi_union_pw_aff *isl_multi_union_pw_aff_apply_multi_aff(
	__isl_take isl_multi_union_pw_aff *mupa, __isl_take isl_multi_aff *ma)
{
	isl_space *space1, *space2;
	isl_multi_union_pw_aff *res;
	isl_bool equal;
	int i;
	isl_size n_in, n_out;

	mupa = isl_multi_union_pw_aff_align_params(mupa,
						isl_multi_aff_get_space(ma));
	ma = isl_multi_aff_align_params(ma,
					isl_multi_union_pw_aff_get_space(mupa));
	n_in = isl_multi_aff_dim(ma, isl_dim_in);
	n_out = isl_multi_aff_dim(ma, isl_dim_out);
	if (!mupa || n_in < 0 || n_out < 0)
		goto error;

	space1 = isl_multi_union_pw_aff_get_space(mupa);
	space2 = isl_multi_aff_get_domain_space(ma);
	equal = isl_space_is_equal(space1, space2);
	isl_space_free(space1);
	isl_space_free(space2);
	if (equal < 0)
		goto error;
	if (!equal)
		isl_die(isl_multi_aff_get_ctx(ma), isl_error_invalid,
			"spaces don't match", goto error);
	if (n_in == 0)
		return mupa_apply_multi_aff_0D(mupa, ma);

	space1 = isl_space_range(isl_multi_aff_get_space(ma));
	res = isl_multi_union_pw_aff_alloc(space1);

	for (i = 0; i < n_out; ++i) {
		isl_aff *aff;
		isl_union_pw_aff *upa;

		aff = isl_multi_aff_get_aff(ma, i);
		upa = multi_union_pw_aff_apply_aff(
					isl_multi_union_pw_aff_copy(mupa), aff);
		res = isl_multi_union_pw_aff_set_union_pw_aff(res, i, upa);
	}

	isl_multi_aff_free(ma);
	isl_multi_union_pw_aff_free(mupa);
	return res;
error:
	isl_multi_union_pw_aff_free(mupa);
	isl_multi_aff_free(ma);
	return NULL;
}

/* Apply "pa" to "mupa", in the special case where "mupa" is 0D.
 * The space of "mupa" is known to be compatible with the domain of "pa".
 *
 * Construct an isl_multi_union_pw_aff that is equal to "pa"
 * on the domain of "mupa".
 */
static __isl_give isl_union_pw_aff *isl_multi_union_pw_aff_apply_pw_aff_0D(
	__isl_take isl_multi_union_pw_aff *mupa, __isl_take isl_pw_aff *pa)
{
	isl_union_set *dom;

	dom = isl_multi_union_pw_aff_domain(mupa);
	pa = isl_pw_aff_project_domain_on_params(pa);

	return isl_union_pw_aff_pw_aff_on_domain(dom, pa);
}

/* Apply "pa" to "mupa".  The space of "mupa" needs to be compatible
 * with the domain of "pa".
 * Furthermore, the dimension of this space needs to be greater than zero.
 * The result is defined over the shared domain of the elements of "mupa"
 */
__isl_give isl_union_pw_aff *isl_multi_union_pw_aff_apply_pw_aff(
	__isl_take isl_multi_union_pw_aff *mupa, __isl_take isl_pw_aff *pa)
{
	int i;
	isl_bool equal;
	isl_size n_in;
	isl_space *space, *space2;
	isl_union_pw_aff *upa;

	mupa = isl_multi_union_pw_aff_align_params(mupa,
						isl_pw_aff_get_space(pa));
	pa = isl_pw_aff_align_params(pa,
				    isl_multi_union_pw_aff_get_space(mupa));
	if (!mupa || !pa)
		goto error;

	space = isl_multi_union_pw_aff_get_space(mupa);
	space2 = isl_pw_aff_get_domain_space(pa);
	equal = isl_space_is_equal(space, space2);
	isl_space_free(space);
	isl_space_free(space2);
	if (equal < 0)
		goto error;
	if (!equal)
		isl_die(isl_pw_aff_get_ctx(pa), isl_error_invalid,
			"spaces don't match", goto error);
	n_in = isl_pw_aff_dim(pa, isl_dim_in);
	if (n_in < 0)
		goto error;
	if (n_in == 0)
		return isl_multi_union_pw_aff_apply_pw_aff_0D(mupa, pa);

	space = isl_space_params(isl_multi_union_pw_aff_get_space(mupa));
	upa = isl_union_pw_aff_empty(space);

	for (i = 0; i < pa->n; ++i) {
		isl_aff *aff;
		isl_set *domain;
		isl_multi_union_pw_aff *mupa_i;
		isl_union_pw_aff *upa_i;

		mupa_i = isl_multi_union_pw_aff_copy(mupa);
		domain = isl_set_copy(pa->p[i].set);
		mupa_i = isl_multi_union_pw_aff_intersect_range(mupa_i, domain);
		aff = isl_aff_copy(pa->p[i].aff);
		upa_i = multi_union_pw_aff_apply_aff(mupa_i, aff);
		upa = isl_union_pw_aff_union_add(upa, upa_i);
	}

	isl_multi_union_pw_aff_free(mupa);
	isl_pw_aff_free(pa);
	return upa;
error:
	isl_multi_union_pw_aff_free(mupa);
	isl_pw_aff_free(pa);
	return NULL;
}

/* Apply "pma" to "mupa", in the special case where "mupa" is 0D.
 * The space of "mupa" is known to be compatible with the domain of "pma".
 *
 * Construct an isl_multi_union_pw_aff that is equal to "pma"
 * on the domain of "mupa".
 */
static __isl_give isl_multi_union_pw_aff *mupa_apply_pw_multi_aff_0D(
	__isl_take isl_multi_union_pw_aff *mupa,
	__isl_take isl_pw_multi_aff *pma)
{
	isl_union_set *dom;

	dom = isl_multi_union_pw_aff_domain(mupa);
	pma = isl_pw_multi_aff_project_domain_on_params(pma);

	return isl_multi_union_pw_aff_pw_multi_aff_on_domain(dom, pma);
}

/* Apply "pma" to "mupa".  The space of "mupa" needs to be compatible
 * with the domain of "pma".
 * The result is defined over the shared domain of the elements of "mupa"
 */
__isl_give isl_multi_union_pw_aff *isl_multi_union_pw_aff_apply_pw_multi_aff(
	__isl_take isl_multi_union_pw_aff *mupa,
	__isl_take isl_pw_multi_aff *pma)
{
	isl_space *space1, *space2;
	isl_multi_union_pw_aff *res;
	isl_bool equal;
	int i;
	isl_size n_in, n_out;

	mupa = isl_multi_union_pw_aff_align_params(mupa,
					isl_pw_multi_aff_get_space(pma));
	pma = isl_pw_multi_aff_align_params(pma,
					isl_multi_union_pw_aff_get_space(mupa));
	if (!mupa || !pma)
		goto error;

	space1 = isl_multi_union_pw_aff_get_space(mupa);
	space2 = isl_pw_multi_aff_get_domain_space(pma);
	equal = isl_space_is_equal(space1, space2);
	isl_space_free(space1);
	isl_space_free(space2);
	if (equal < 0)
		goto error;
	if (!equal)
		isl_die(isl_pw_multi_aff_get_ctx(pma), isl_error_invalid,
			"spaces don't match", goto error);
	n_in = isl_pw_multi_aff_dim(pma, isl_dim_in);
	n_out = isl_pw_multi_aff_dim(pma, isl_dim_out);
	if (n_in < 0 || n_out < 0)
		goto error;
	if (n_in == 0)
		return mupa_apply_pw_multi_aff_0D(mupa, pma);

	space1 = isl_space_range(isl_pw_multi_aff_get_space(pma));
	res = isl_multi_union_pw_aff_alloc(space1);

	for (i = 0; i < n_out; ++i) {
		isl_pw_aff *pa;
		isl_union_pw_aff *upa;

		pa = isl_pw_multi_aff_get_pw_aff(pma, i);
		upa = isl_multi_union_pw_aff_apply_pw_aff(
					isl_multi_union_pw_aff_copy(mupa), pa);
		res = isl_multi_union_pw_aff_set_union_pw_aff(res, i, upa);
	}

	isl_pw_multi_aff_free(pma);
	isl_multi_union_pw_aff_free(mupa);
	return res;
error:
	isl_multi_union_pw_aff_free(mupa);
	isl_pw_multi_aff_free(pma);
	return NULL;
}

/* Replace the explicit domain of "mupa" by its preimage under "upma".
 * If the explicit domain only keeps track of constraints on the parameters,
 * then only update those constraints.
 */
static __isl_give isl_multi_union_pw_aff *preimage_explicit_domain(
	__isl_take isl_multi_union_pw_aff *mupa,
	__isl_keep isl_union_pw_multi_aff *upma)
{
	isl_bool is_params;

	if (isl_multi_union_pw_aff_check_has_explicit_domain(mupa) < 0)
		return isl_multi_union_pw_aff_free(mupa);

	mupa = isl_multi_union_pw_aff_cow(mupa);
	if (!mupa)
		return NULL;

	is_params = isl_union_set_is_params(mupa->u.dom);
	if (is_params < 0)
		return isl_multi_union_pw_aff_free(mupa);

	upma = isl_union_pw_multi_aff_copy(upma);
	if (is_params)
		mupa->u.dom = isl_union_set_intersect_params(mupa->u.dom,
		    isl_union_set_params(isl_union_pw_multi_aff_domain(upma)));
	else
		mupa->u.dom = isl_union_set_preimage_union_pw_multi_aff(
							    mupa->u.dom, upma);
	if (!mupa->u.dom)
		return isl_multi_union_pw_aff_free(mupa);
	return mupa;
}

/* Compute the pullback of "mupa" by the function represented by "upma".
 * In other words, plug in "upma" in "mupa".  The result contains
 * expressions defined over the domain space of "upma".
 *
 * Run over all elements of "mupa" and plug in "upma" in each of them.
 *
 * If "mupa" has an explicit domain, then it is this domain
 * that needs to undergo a pullback instead, i.e., a preimage.
 */
__isl_give isl_multi_union_pw_aff *
isl_multi_union_pw_aff_pullback_union_pw_multi_aff(
	__isl_take isl_multi_union_pw_aff *mupa,
	__isl_take isl_union_pw_multi_aff *upma)
{
	int i;
	isl_size n;

	mupa = isl_multi_union_pw_aff_align_params(mupa,
				    isl_union_pw_multi_aff_get_space(upma));
	upma = isl_union_pw_multi_aff_align_params(upma,
				    isl_multi_union_pw_aff_get_space(mupa));
	mupa = isl_multi_union_pw_aff_cow(mupa);
	n = isl_multi_union_pw_aff_dim(mupa, isl_dim_set);
	if (n < 0 || !upma)
		goto error;

	for (i = 0; i < n; ++i) {
		isl_union_pw_aff *upa;

		upa = isl_multi_union_pw_aff_get_union_pw_aff(mupa, i);
		upa = isl_union_pw_aff_pullback_union_pw_multi_aff(upa,
					    isl_union_pw_multi_aff_copy(upma));
		mupa = isl_multi_union_pw_aff_set_union_pw_aff(mupa, i, upa);
	}

	if (isl_multi_union_pw_aff_has_explicit_domain(mupa))
		mupa = preimage_explicit_domain(mupa, upma);

	isl_union_pw_multi_aff_free(upma);
	return mupa;
error:
	isl_multi_union_pw_aff_free(mupa);
	isl_union_pw_multi_aff_free(upma);
	return NULL;
}

/* Extract the sequence of elements in "mupa" with domain space "space"
 * (ignoring parameters).
 *
 * For the elements of "mupa" that are not defined on the specified space,
 * the corresponding element in the result is empty.
 */
__isl_give isl_multi_pw_aff *isl_multi_union_pw_aff_extract_multi_pw_aff(
	__isl_keep isl_multi_union_pw_aff *mupa, __isl_take isl_space *space)
{
	int i;
	isl_size n;
	isl_space *space_mpa;
	isl_multi_pw_aff *mpa;

	n = isl_multi_union_pw_aff_dim(mupa, isl_dim_set);
	if (n < 0 || !space)
		goto error;

	space_mpa = isl_multi_union_pw_aff_get_space(mupa);
	space = isl_space_replace_params(space, space_mpa);
	space_mpa = isl_space_map_from_domain_and_range(isl_space_copy(space),
							space_mpa);
	mpa = isl_multi_pw_aff_alloc(space_mpa);

	space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_out, 1);
	for (i = 0; i < n; ++i) {
		isl_union_pw_aff *upa;
		isl_pw_aff *pa;

		upa = isl_multi_union_pw_aff_get_union_pw_aff(mupa, i);
		pa = isl_union_pw_aff_extract_pw_aff(upa,
							isl_space_copy(space));
		mpa = isl_multi_pw_aff_set_pw_aff(mpa, i, pa);
		isl_union_pw_aff_free(upa);
	}

	isl_space_free(space);
	return mpa;
error:
	isl_space_free(space);
	return NULL;
}

/* Data structure that specifies how isl_union_pw_multi_aff_un_op
 * should modify the base expressions in the input.
 *
 * If "filter" is not NULL, then only the base expressions that satisfy "filter"
 * are taken into account.
 * "fn" is applied to each entry in the input.
 */
struct isl_union_pw_multi_aff_un_op_control {
	isl_bool (*filter)(__isl_keep isl_pw_multi_aff *part);
	__isl_give isl_pw_multi_aff *(*fn)(__isl_take isl_pw_multi_aff *pma);
};

/* Wrapper for isl_union_pw_multi_aff_un_op filter functions (which do not take
 * a second argument) for use as an isl_union_pw_multi_aff_transform
 * filter function (which does take a second argument).
 * Simply call control->filter without the second argument.
 */
static isl_bool isl_union_pw_multi_aff_un_op_filter_drop_user(
	__isl_take isl_pw_multi_aff *pma, void *user)
{
	struct isl_union_pw_multi_aff_un_op_control *control = user;

	return control->filter(pma);
}

/* Wrapper for isl_union_pw_multi_aff_un_op base functions (which do not take
 * a second argument) for use as an isl_union_pw_multi_aff_transform
 * base function (which does take a second argument).
 * Simply call control->fn without the second argument.
 */
static __isl_give isl_pw_multi_aff *isl_union_pw_multi_aff_un_op_drop_user(
	__isl_take isl_pw_multi_aff *pma, void *user)
{
	struct isl_union_pw_multi_aff_un_op_control *control = user;

	return control->fn(pma);
}

/* Construct an isl_union_pw_multi_aff that is obtained by
 * modifying "upma" according to "control".
 *
 * isl_union_pw_multi_aff_transform performs essentially
 * the same operation, but takes a filter and a callback function
 * of a different form (with an extra argument).
 * Call isl_union_pw_multi_aff_transform with wrappers
 * that remove this extra argument.
 */
static __isl_give isl_union_pw_multi_aff *isl_union_pw_multi_aff_un_op(
	__isl_take isl_union_pw_multi_aff *upma,
	struct isl_union_pw_multi_aff_un_op_control *control)
{
	struct isl_union_pw_multi_aff_transform_control t_control = {
		.filter = &isl_union_pw_multi_aff_un_op_filter_drop_user,
		.filter_user = control,
		.fn = &isl_union_pw_multi_aff_un_op_drop_user,
		.fn_user = control,
	};

	return isl_union_pw_multi_aff_transform(upma, &t_control);
}

/* For each function in "upma" of the form A -> [B -> C],
 * extract the function A -> B and collect the results.
 */
__isl_give isl_union_pw_multi_aff *isl_union_pw_multi_aff_range_factor_domain(
	__isl_take isl_union_pw_multi_aff *upma)
{
	struct isl_union_pw_multi_aff_un_op_control control = {
		.filter = &isl_pw_multi_aff_range_is_wrapping,
		.fn = &isl_pw_multi_aff_range_factor_domain,
	};
	return isl_union_pw_multi_aff_un_op(upma, &control);
}

/* For each function in "upma" of the form A -> [B -> C],
 * extract the function A -> C and collect the results.
 */
__isl_give isl_union_pw_multi_aff *isl_union_pw_multi_aff_range_factor_range(
	__isl_take isl_union_pw_multi_aff *upma)
{
	struct isl_union_pw_multi_aff_un_op_control control = {
		.filter = &isl_pw_multi_aff_range_is_wrapping,
		.fn = &isl_pw_multi_aff_range_factor_range,
	};
	return isl_union_pw_multi_aff_un_op(upma, &control);
}

/* Evaluate the affine function "aff" in the void point "pnt".
 * In particular, return the value NaN.
 */
static __isl_give isl_val *eval_void(__isl_take isl_aff *aff,
	__isl_take isl_point *pnt)
{
	isl_ctx *ctx;

	ctx = isl_point_get_ctx(pnt);
	isl_aff_free(aff);
	isl_point_free(pnt);
	return isl_val_nan(ctx);
}

/* Evaluate the affine expression "aff"
 * in the coordinates (with denominator) "pnt".
 */
static __isl_give isl_val *eval(__isl_keep isl_vec *aff,
	__isl_keep isl_vec *pnt)
{
	isl_int n, d;
	isl_ctx *ctx;
	isl_val *v;

	if (!aff || !pnt)
		return NULL;

	ctx = isl_vec_get_ctx(aff);
	isl_int_init(n);
	isl_int_init(d);
	isl_seq_inner_product(aff->el + 1, pnt->el, pnt->size, &n);
	isl_int_mul(d, aff->el[0], pnt->el[0]);
	v = isl_val_rat_from_isl_int(ctx, n, d);
	v = isl_val_normalize(v);
	isl_int_clear(n);
	isl_int_clear(d);

	return v;
}

/* Check that the domain space of "aff" is equal to "space".
 */
static isl_stat isl_aff_check_has_domain_space(__isl_keep isl_aff *aff,
	__isl_keep isl_space *space)
{
	isl_bool ok;

	ok = isl_space_is_equal(isl_aff_peek_domain_space(aff), space);
	if (ok < 0)
		return isl_stat_error;
	if (!ok)
		isl_die(isl_aff_get_ctx(aff), isl_error_invalid,
			"incompatible spaces", return isl_stat_error);
	return isl_stat_ok;
}

/* Evaluate the affine function "aff" in "pnt".
 */
__isl_give isl_val *isl_aff_eval(__isl_take isl_aff *aff,
	__isl_take isl_point *pnt)
{
	isl_bool is_void;
	isl_val *v;
	isl_local_space *ls;

	if (isl_aff_check_has_domain_space(aff, isl_point_peek_space(pnt)) < 0)
		goto error;
	is_void = isl_point_is_void(pnt);
	if (is_void < 0)
		goto error;
	if (is_void)
		return eval_void(aff, pnt);

	ls = isl_aff_get_domain_local_space(aff);
	pnt = isl_local_space_lift_point(ls, pnt);

	v = eval(aff->v, isl_point_peek_vec(pnt));

	isl_aff_free(aff);
	isl_point_free(pnt);

	return v;
error:
	isl_aff_free(aff);
	isl_point_free(pnt);
	return NULL;
}
