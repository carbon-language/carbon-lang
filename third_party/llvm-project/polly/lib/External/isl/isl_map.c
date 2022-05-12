/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010      INRIA Saclay
 * Copyright 2012-2014 Ecole Normale Superieure
 * Copyright 2014      INRIA Rocquencourt
 * Copyright 2016      INRIA Paris
 * Copyright 2016      Sven Verdoolaege
 * Copyright 2018-2019 Cerebras Systems
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
 * and Centre de Recherche Inria de Paris, 2 rue Simone Iff - Voie DQ12,
 * CS 42112, 75589 Paris Cedex 12, France
 * and Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

#include <string.h>
#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include <isl_blk.h>
#include <isl_id_private.h>
#include <isl/constraint.h>
#include "isl_space_private.h"
#include "isl_equalities.h"
#include <isl_lp_private.h>
#include <isl_seq.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl_reordering.h>
#include "isl_sample.h"
#include <isl_sort.h>
#include "isl_tab.h"
#include <isl/vec.h>
#include <isl_mat_private.h>
#include <isl_vec_private.h>
#include <isl_dim_map.h>
#include <isl_local_space_private.h>
#include <isl_aff_private.h>
#include <isl_options_private.h>
#include <isl_morph.h>
#include <isl_val_private.h>
#include <isl_printer_private.h>

#include <bset_to_bmap.c>
#include <bset_from_bmap.c>
#include <set_to_map.c>
#include <set_from_map.c>

/* Treat "bset" as a basic map.
 * Internally, isl_basic_set is defined to isl_basic_map, so in practice,
 * this function performs a redundant cast.
 */
static __isl_keep const isl_basic_map *const_bset_to_bmap(
	__isl_keep const isl_basic_set *bset)
{
	return (const isl_basic_map *) bset;
}

#undef TYPE
#define TYPE	isl_basic_map
#include "has_single_reference_templ.c"

static unsigned pos(__isl_keep isl_space *space, enum isl_dim_type type)
{
	switch (type) {
	case isl_dim_param:	return 1;
	case isl_dim_in:	return 1 + space->nparam;
	case isl_dim_out:	return 1 + space->nparam + space->n_in;
	default:		return 0;
	}
}

isl_size isl_basic_map_dim(__isl_keep isl_basic_map *bmap,
				enum isl_dim_type type)
{
	if (!bmap)
		return isl_size_error;
	switch (type) {
	case isl_dim_cst:	return 1;
	case isl_dim_param:
	case isl_dim_in:
	case isl_dim_out:	return isl_space_dim(bmap->dim, type);
	case isl_dim_div:	return bmap->n_div;
	case isl_dim_all:	return isl_basic_map_total_dim(bmap);
	default:		return 0;
	}
}

/* Return the space of "map".
 */
__isl_keep isl_space *isl_map_peek_space(__isl_keep const isl_map *map)
{
	return map ? map->dim : NULL;
}

/* Return the space of "set".
 */
__isl_keep isl_space *isl_set_peek_space(__isl_keep isl_set *set)
{
	return isl_map_peek_space(set_to_map(set));
}

isl_size isl_map_dim(__isl_keep isl_map *map, enum isl_dim_type type)
{
	return isl_space_dim(isl_map_peek_space(map), type);
}

/* Return the dimensionality of the domain (tuple) of the map.
 */
isl_size isl_map_domain_tuple_dim(__isl_keep isl_map *map)
{
	return isl_map_dim(map, isl_dim_in);
}

/* Return the dimensionality of the range (tuple) of the map.
 */
isl_size isl_map_range_tuple_dim(__isl_keep isl_map *map)
{
	return isl_map_dim(map, isl_dim_out);
}

isl_size isl_set_dim(__isl_keep isl_set *set, enum isl_dim_type type)
{
	return isl_map_dim(set_to_map(set), type);
}

/* Return the dimensionality of the (tuple of the) set.
 */
isl_size isl_set_tuple_dim(__isl_keep isl_set *set)
{
	return isl_set_dim(set, isl_dim_set);
}

/* Return the position of the variables of the given type
 * within the sequence of variables of "bmap".
 */
isl_size isl_basic_map_var_offset(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type)
{
	isl_space *space;

	space = isl_basic_map_peek_space(bmap);
	if (!space)
		return isl_size_error;

	switch (type) {
	case isl_dim_param:
	case isl_dim_in:
	case isl_dim_out:	return isl_space_offset(space, type);
	case isl_dim_div:	return isl_space_dim(space, isl_dim_all);
	case isl_dim_cst:
	default:
		isl_die(isl_basic_map_get_ctx(bmap), isl_error_invalid,
			"invalid dimension type", return isl_size_error);
	}
}

/* Return the position of the variables of the given type
 * within the sequence of variables of "bset".
 */
isl_size isl_basic_set_var_offset(__isl_keep isl_basic_set *bset,
	enum isl_dim_type type)
{
	return isl_basic_map_var_offset(bset_to_bmap(bset), type);
}

/* Return the position of the coefficients of the variables of the given type
 * within the sequence of coefficients of "bmap".
 */
unsigned isl_basic_map_offset(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type)
{
	switch (type) {
	case isl_dim_cst:	return 0;
	case isl_dim_param:
	case isl_dim_in:
	case isl_dim_out:
	case isl_dim_div:	return 1 + isl_basic_map_var_offset(bmap, type);
	default:		return 0;
	}
}

unsigned isl_basic_set_offset(__isl_keep isl_basic_set *bset,
					enum isl_dim_type type)
{
	return isl_basic_map_offset(bset, type);
}

static unsigned map_offset(__isl_keep isl_map *map, enum isl_dim_type type)
{
	return pos(map->dim, type);
}

isl_size isl_basic_set_dim(__isl_keep isl_basic_set *bset,
				enum isl_dim_type type)
{
	return isl_basic_map_dim(bset, type);
}

isl_size isl_basic_set_n_dim(__isl_keep isl_basic_set *bset)
{
	return isl_basic_set_dim(bset, isl_dim_set);
}

isl_size isl_basic_set_n_param(__isl_keep isl_basic_set *bset)
{
	return isl_basic_set_dim(bset, isl_dim_param);
}

isl_size isl_basic_set_total_dim(__isl_keep const isl_basic_set *bset)
{
	return isl_basic_map_total_dim(const_bset_to_bmap(bset));
}

isl_size isl_set_n_dim(__isl_keep isl_set *set)
{
	return isl_set_dim(set, isl_dim_set);
}

isl_size isl_set_n_param(__isl_keep isl_set *set)
{
	return isl_set_dim(set, isl_dim_param);
}

isl_size isl_basic_map_total_dim(__isl_keep const isl_basic_map *bmap)
{
	isl_size dim;

	if (!bmap)
		return isl_size_error;
	dim = isl_space_dim(bmap->dim, isl_dim_all);
	if (dim < 0)
		return isl_size_error;
	return dim + bmap->n_div;
}

/* Return the number of equality constraints in the description of "bmap".
 * Return isl_size_error on error.
 */
isl_size isl_basic_map_n_equality(__isl_keep isl_basic_map *bmap)
{
	if (!bmap)
		return isl_size_error;
	return bmap->n_eq;
}

/* Return the number of equality constraints in the description of "bset".
 * Return isl_size_error on error.
 */
isl_size isl_basic_set_n_equality(__isl_keep isl_basic_set *bset)
{
	return isl_basic_map_n_equality(bset_to_bmap(bset));
}

/* Return the number of inequality constraints in the description of "bmap".
 * Return isl_size_error on error.
 */
isl_size isl_basic_map_n_inequality(__isl_keep isl_basic_map *bmap)
{
	if (!bmap)
		return isl_size_error;
	return bmap->n_ineq;
}

/* Return the number of inequality constraints in the description of "bset".
 * Return isl_size_error on error.
 */
isl_size isl_basic_set_n_inequality(__isl_keep isl_basic_set *bset)
{
	return isl_basic_map_n_inequality(bset_to_bmap(bset));
}

/* Do "bmap1" and "bmap2" have the same parameters?
 */
static isl_bool isl_basic_map_has_equal_params(__isl_keep isl_basic_map *bmap1,
	__isl_keep isl_basic_map *bmap2)
{
	isl_space *space1, *space2;

	space1 = isl_basic_map_peek_space(bmap1);
	space2 = isl_basic_map_peek_space(bmap2);
	return isl_space_has_equal_params(space1, space2);
}

/* Do "map1" and "map2" have the same parameters?
 */
isl_bool isl_map_has_equal_params(__isl_keep isl_map *map1,
	__isl_keep isl_map *map2)
{
	isl_space *space1, *space2;

	space1 = isl_map_peek_space(map1);
	space2 = isl_map_peek_space(map2);
	return isl_space_has_equal_params(space1, space2);
}

/* Do "map" and "set" have the same parameters?
 */
static isl_bool isl_map_set_has_equal_params(__isl_keep isl_map *map,
	__isl_keep isl_set *set)
{
	return isl_map_has_equal_params(map, set_to_map(set));
}

/* Is the tuple of type "type" of "bmap" the same as the single tuple of "bset"?
 */
static isl_bool isl_basic_map_set_tuple_is_equal(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type, __isl_keep isl_basic_set *bset)
{
	isl_space *bmap_space, *bset_space;

	bmap_space = isl_basic_map_peek_space(bmap);
	bset_space = isl_basic_set_peek_space(bset);
	return isl_space_tuple_is_equal(bmap_space, type,
					bset_space, isl_dim_set);
}

/* Is the tuple of type "type" of "map" the same as the single tuple of "set"?
 */
static isl_bool isl_map_set_tuple_is_equal(__isl_keep isl_map *map,
	enum isl_dim_type type, __isl_keep isl_set *set)
{
	return isl_map_tuple_is_equal(map, type, set_to_map(set), isl_dim_set);
}

isl_bool isl_map_compatible_domain(__isl_keep isl_map *map,
	__isl_keep isl_set *set)
{
	isl_bool m;
	if (!map || !set)
		return isl_bool_error;
	m = isl_map_has_equal_params(map, set_to_map(set));
	if (m < 0 || !m)
		return m;
	return isl_map_set_tuple_is_equal(map, isl_dim_in, set);
}

isl_bool isl_basic_map_compatible_domain(__isl_keep isl_basic_map *bmap,
	__isl_keep isl_basic_set *bset)
{
	isl_bool m;
	if (!bmap || !bset)
		return isl_bool_error;
	m = isl_basic_map_has_equal_params(bmap, bset_to_bmap(bset));
	if (m < 0 || !m)
		return m;
	return isl_basic_map_set_tuple_is_equal(bmap, isl_dim_in, bset);
}

isl_bool isl_map_compatible_range(__isl_keep isl_map *map,
	__isl_keep isl_set *set)
{
	isl_bool m;
	if (!map || !set)
		return isl_bool_error;
	m = isl_map_has_equal_params(map, set_to_map(set));
	if (m < 0 || !m)
		return m;
	return isl_map_set_tuple_is_equal(map, isl_dim_out, set);
}

isl_bool isl_basic_map_compatible_range(__isl_keep isl_basic_map *bmap,
	__isl_keep isl_basic_set *bset)
{
	isl_bool m;
	if (!bmap || !bset)
		return isl_bool_error;
	m = isl_basic_map_has_equal_params(bmap, bset_to_bmap(bset));
	if (m < 0 || !m)
		return m;
	return isl_basic_map_set_tuple_is_equal(bmap, isl_dim_out, bset);
}

isl_ctx *isl_basic_map_get_ctx(__isl_keep isl_basic_map *bmap)
{
	return bmap ? bmap->ctx : NULL;
}

isl_ctx *isl_basic_set_get_ctx(__isl_keep isl_basic_set *bset)
{
	return bset ? bset->ctx : NULL;
}

isl_ctx *isl_map_get_ctx(__isl_keep isl_map *map)
{
	return map ? map->ctx : NULL;
}

isl_ctx *isl_set_get_ctx(__isl_keep isl_set *set)
{
	return set ? set->ctx : NULL;
}

/* Return the space of "bmap".
 */
__isl_keep isl_space *isl_basic_map_peek_space(
	__isl_keep const isl_basic_map *bmap)
{
	return bmap ? bmap->dim : NULL;
}

/* Return the space of "bset".
 */
__isl_keep isl_space *isl_basic_set_peek_space(__isl_keep isl_basic_set *bset)
{
	return isl_basic_map_peek_space(bset_to_bmap(bset));
}

__isl_give isl_space *isl_basic_map_get_space(__isl_keep isl_basic_map *bmap)
{
	return isl_space_copy(isl_basic_map_peek_space(bmap));
}

__isl_give isl_space *isl_basic_set_get_space(__isl_keep isl_basic_set *bset)
{
	return isl_basic_map_get_space(bset_to_bmap(bset));
}

/* Return the space of "bmap".
 * This may be either a copy or the space itself
 * if there is only one reference to "bmap".
 * This allows the space to be modified inplace
 * if both the basic map and its space have only a single reference.
 * The caller is not allowed to modify "bmap" between this call and
 * a subsequent call to isl_basic_map_restore_space.
 * The only exception is that isl_basic_map_free can be called instead.
 */
static __isl_give isl_space *isl_basic_map_take_space(
	__isl_keep isl_basic_map *bmap)
{
	isl_space *space;

	if (!bmap)
		return NULL;
	if (bmap->ref != 1)
		return isl_basic_map_get_space(bmap);
	space = bmap->dim;
	bmap->dim = NULL;
	return space;
}

/* Set the space of "bmap" to "space", where the space of "bmap" may be missing
 * due to a preceding call to isl_basic_map_take_space.
 * However, in this case, "bmap" only has a single reference and
 * then the call to isl_basic_map_cow has no effect.
 */
static __isl_give isl_basic_map *isl_basic_map_restore_space(
	__isl_take isl_basic_map *bmap, __isl_take isl_space *space)
{
	if (!bmap || !space)
		goto error;

	if (bmap->dim == space) {
		isl_space_free(space);
		return bmap;
	}

	bmap = isl_basic_map_cow(bmap);
	if (!bmap)
		goto error;
	isl_space_free(bmap->dim);
	bmap->dim = space;

	return bmap;
error:
	isl_basic_map_free(bmap);
	isl_space_free(space);
	return NULL;
}

/* Extract the divs in "bmap" as a matrix.
 */
__isl_give isl_mat *isl_basic_map_get_divs(__isl_keep isl_basic_map *bmap)
{
	int i;
	isl_ctx *ctx;
	isl_mat *div;
	isl_size v_div;
	unsigned cols;

	v_div = isl_basic_map_var_offset(bmap, isl_dim_div);
	if (v_div < 0)
		return NULL;

	ctx = isl_basic_map_get_ctx(bmap);
	cols = 1 + 1 + v_div + bmap->n_div;
	div = isl_mat_alloc(ctx, bmap->n_div, cols);
	if (!div)
		return NULL;

	for (i = 0; i < bmap->n_div; ++i)
		isl_seq_cpy(div->row[i], bmap->div[i], cols);

	return div;
}

/* Extract the divs in "bset" as a matrix.
 */
__isl_give isl_mat *isl_basic_set_get_divs(__isl_keep isl_basic_set *bset)
{
	return isl_basic_map_get_divs(bset);
}

__isl_give isl_local_space *isl_basic_map_get_local_space(
	__isl_keep isl_basic_map *bmap)
{
	isl_mat *div;

	if (!bmap)
		return NULL;

	div = isl_basic_map_get_divs(bmap);
	return isl_local_space_alloc_div(isl_space_copy(bmap->dim), div);
}

__isl_give isl_local_space *isl_basic_set_get_local_space(
	__isl_keep isl_basic_set *bset)
{
	return isl_basic_map_get_local_space(bset);
}

/* For each known div d = floor(f/m), add the constraints
 *
 *		f - m d >= 0
 *		-(f-(m-1)) + m d >= 0
 *
 * Do not finalize the result.
 */
static __isl_give isl_basic_map *add_known_div_constraints(
	__isl_take isl_basic_map *bmap)
{
	int i;
	isl_size n_div;

	n_div = isl_basic_map_dim(bmap, isl_dim_div);
	if (n_div < 0)
		return isl_basic_map_free(bmap);
	if (n_div == 0)
		return bmap;
	bmap = isl_basic_map_cow(bmap);
	bmap = isl_basic_map_extend_constraints(bmap, 0, 2 * n_div);
	if (!bmap)
		return NULL;
	for (i = 0; i < n_div; ++i) {
		if (isl_int_is_zero(bmap->div[i][0]))
			continue;
		bmap = isl_basic_map_add_div_constraints(bmap, i);
	}

	return bmap;
}

__isl_give isl_basic_map *isl_basic_map_from_local_space(
	__isl_take isl_local_space *ls)
{
	int i;
	isl_size n_div;
	isl_basic_map *bmap;

	n_div = isl_local_space_dim(ls, isl_dim_div);
	if (n_div < 0)
		ls = isl_local_space_free(ls);
	if (!ls)
		return NULL;

	bmap = isl_basic_map_alloc_space(isl_local_space_get_space(ls),
					n_div, 0, 2 * n_div);

	for (i = 0; i < n_div; ++i)
		if (isl_basic_map_alloc_div(bmap) < 0)
			goto error;

	for (i = 0; i < n_div; ++i)
		isl_seq_cpy(bmap->div[i], ls->div->row[i], ls->div->n_col);
	bmap = add_known_div_constraints(bmap);
					
	isl_local_space_free(ls);
	return bmap;
error:
	isl_local_space_free(ls);
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_from_local_space(
	__isl_take isl_local_space *ls)
{
	return isl_basic_map_from_local_space(ls);
}

__isl_give isl_space *isl_map_get_space(__isl_keep isl_map *map)
{
	return isl_space_copy(isl_map_peek_space(map));
}

__isl_give isl_space *isl_set_get_space(__isl_keep isl_set *set)
{
	if (!set)
		return NULL;
	return isl_space_copy(set->dim);
}

/* Return the space of "map".
 * This may be either a copy or the space itself
 * if there is only one reference to "map".
 * This allows the space to be modified inplace
 * if both the map and its space have only a single reference.
 * The caller is not allowed to modify "map" between this call and
 * a subsequent call to isl_map_restore_space.
 * The only exception is that isl_map_free can be called instead.
 */
static __isl_give isl_space *isl_map_take_space(__isl_keep isl_map *map)
{
	isl_space *space;

	if (!map)
		return NULL;
	if (map->ref != 1)
		return isl_map_get_space(map);
	space = map->dim;
	map->dim = NULL;
	return space;
}

/* Set the space of "map" to "space", where the space of "map" may be missing
 * due to a preceding call to isl_map_take_space.
 * However, in this case, "map" only has a single reference and
 * then the call to isl_map_cow has no effect.
 */
static __isl_give isl_map *isl_map_restore_space(__isl_take isl_map *map,
	__isl_take isl_space *space)
{
	if (!map || !space)
		goto error;

	if (map->dim == space) {
		isl_space_free(space);
		return map;
	}

	map = isl_map_cow(map);
	if (!map)
		goto error;
	isl_space_free(map->dim);
	map->dim = space;

	return map;
error:
	isl_map_free(map);
	isl_space_free(space);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_set_tuple_name(
	__isl_take isl_basic_map *bmap, enum isl_dim_type type, const char *s)
{
	isl_space *space;

	space = isl_basic_map_take_space(bmap);
	space = isl_space_set_tuple_name(space, type, s);
	bmap = isl_basic_map_restore_space(bmap, space);
	bmap = isl_basic_map_finalize(bmap);
	return bmap;
}

__isl_give isl_basic_set *isl_basic_set_set_tuple_name(
	__isl_take isl_basic_set *bset, const char *s)
{
	return isl_basic_map_set_tuple_name(bset, isl_dim_set, s);
}

const char *isl_basic_map_get_tuple_name(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type)
{
	return bmap ? isl_space_get_tuple_name(bmap->dim, type) : NULL;
}

__isl_give isl_map *isl_map_set_tuple_name(__isl_take isl_map *map,
	enum isl_dim_type type, const char *s)
{
	int i;
	isl_space *space;

	map = isl_map_cow(map);
	if (!map)
		return NULL;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_set_tuple_name(map->p[i], type, s);
		if (!map->p[i])
			goto error;
	}

	space = isl_map_take_space(map);
	space = isl_space_set_tuple_name(space, type, s);
	map = isl_map_restore_space(map, space);

	return map;
error:
	isl_map_free(map);
	return NULL;
}

/* Replace the identifier of the tuple of type "type" by "id".
 */
__isl_give isl_basic_map *isl_basic_map_set_tuple_id(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, __isl_take isl_id *id)
{
	isl_space *space;

	space = isl_basic_map_take_space(bmap);
	space = isl_space_set_tuple_id(space, type, id);
	bmap = isl_basic_map_restore_space(bmap, space);
	bmap = isl_basic_map_finalize(bmap);
	return bmap;
}

/* Replace the identifier of the tuple by "id".
 */
__isl_give isl_basic_set *isl_basic_set_set_tuple_id(
	__isl_take isl_basic_set *bset, __isl_take isl_id *id)
{
	return isl_basic_map_set_tuple_id(bset, isl_dim_set, id);
}

/* Does the input or output tuple have a name?
 */
isl_bool isl_map_has_tuple_name(__isl_keep isl_map *map, enum isl_dim_type type)
{
	return map ? isl_space_has_tuple_name(map->dim, type) : isl_bool_error;
}

const char *isl_map_get_tuple_name(__isl_keep isl_map *map,
	enum isl_dim_type type)
{
	return map ? isl_space_get_tuple_name(map->dim, type) : NULL;
}

__isl_give isl_set *isl_set_set_tuple_name(__isl_take isl_set *set,
	const char *s)
{
	return set_from_map(isl_map_set_tuple_name(set_to_map(set),
						isl_dim_set, s));
}

__isl_give isl_map *isl_map_set_tuple_id(__isl_take isl_map *map,
	enum isl_dim_type type, __isl_take isl_id *id)
{
	isl_space *space;

	space = isl_map_take_space(map);
	space = isl_space_set_tuple_id(space, type, id);
	map = isl_map_restore_space(map, space);

	return isl_map_reset_space(map, isl_map_get_space(map));
}

/* Replace the identifier of the domain tuple of "map" by "id".
 */
__isl_give isl_map *isl_map_set_domain_tuple_id(__isl_take isl_map *map,
	__isl_take isl_id *id)
{
	return isl_map_set_tuple_id(map, isl_dim_in, id);
}

/* Replace the identifier of the range tuple of "map" by "id".
 */
__isl_give isl_map *isl_map_set_range_tuple_id(__isl_take isl_map *map,
	__isl_take isl_id *id)
{
	return isl_map_set_tuple_id(map, isl_dim_out, id);
}

__isl_give isl_set *isl_set_set_tuple_id(__isl_take isl_set *set,
	__isl_take isl_id *id)
{
	return isl_map_set_tuple_id(set, isl_dim_set, id);
}

__isl_give isl_map *isl_map_reset_tuple_id(__isl_take isl_map *map,
	enum isl_dim_type type)
{
	isl_space *space;

	space = isl_map_take_space(map);
	space = isl_space_reset_tuple_id(space, type);
	map = isl_map_restore_space(map, space);

	return isl_map_reset_space(map, isl_map_get_space(map));
}

__isl_give isl_set *isl_set_reset_tuple_id(__isl_take isl_set *set)
{
	return isl_map_reset_tuple_id(set, isl_dim_set);
}

isl_bool isl_map_has_tuple_id(__isl_keep isl_map *map, enum isl_dim_type type)
{
	return map ? isl_space_has_tuple_id(map->dim, type) : isl_bool_error;
}

/* Does the domain tuple of "map" have an identifier?
 */
isl_bool isl_map_has_domain_tuple_id(__isl_keep isl_map *map)
{
	return isl_map_has_tuple_id(map, isl_dim_in);
}

/* Does the range tuple of "map" have an identifier?
 */
isl_bool isl_map_has_range_tuple_id(__isl_keep isl_map *map)
{
	return isl_map_has_tuple_id(map, isl_dim_out);
}

__isl_give isl_id *isl_map_get_tuple_id(__isl_keep isl_map *map,
	enum isl_dim_type type)
{
	return map ? isl_space_get_tuple_id(map->dim, type) : NULL;
}

/* Return the identifier of the domain tuple of "map", assuming it has one.
 */
__isl_give isl_id *isl_map_get_domain_tuple_id(__isl_keep isl_map *map)
{
	return isl_map_get_tuple_id(map, isl_dim_in);
}

/* Return the identifier of the range tuple of "map", assuming it has one.
 */
__isl_give isl_id *isl_map_get_range_tuple_id(__isl_keep isl_map *map)
{
	return isl_map_get_tuple_id(map, isl_dim_out);
}

isl_bool isl_set_has_tuple_id(__isl_keep isl_set *set)
{
	return isl_map_has_tuple_id(set, isl_dim_set);
}

__isl_give isl_id *isl_set_get_tuple_id(__isl_keep isl_set *set)
{
	return isl_map_get_tuple_id(set, isl_dim_set);
}

/* Does the set tuple have a name?
 */
isl_bool isl_set_has_tuple_name(__isl_keep isl_set *set)
{
	if (!set)
		return isl_bool_error;
	return isl_space_has_tuple_name(set->dim, isl_dim_set);
}


const char *isl_basic_set_get_tuple_name(__isl_keep isl_basic_set *bset)
{
	return bset ? isl_space_get_tuple_name(bset->dim, isl_dim_set) : NULL;
}

const char *isl_set_get_tuple_name(__isl_keep isl_set *set)
{
	return set ? isl_space_get_tuple_name(set->dim, isl_dim_set) : NULL;
}

const char *isl_basic_map_get_dim_name(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos)
{
	return bmap ? isl_space_get_dim_name(bmap->dim, type, pos) : NULL;
}

const char *isl_basic_set_get_dim_name(__isl_keep isl_basic_set *bset,
	enum isl_dim_type type, unsigned pos)
{
	return bset ? isl_space_get_dim_name(bset->dim, type, pos) : NULL;
}

/* Does the given dimension have a name?
 */
isl_bool isl_map_has_dim_name(__isl_keep isl_map *map,
	enum isl_dim_type type, unsigned pos)
{
	if (!map)
		return isl_bool_error;
	return isl_space_has_dim_name(map->dim, type, pos);
}

const char *isl_map_get_dim_name(__isl_keep isl_map *map,
	enum isl_dim_type type, unsigned pos)
{
	return map ? isl_space_get_dim_name(map->dim, type, pos) : NULL;
}

const char *isl_set_get_dim_name(__isl_keep isl_set *set,
	enum isl_dim_type type, unsigned pos)
{
	return set ? isl_space_get_dim_name(set->dim, type, pos) : NULL;
}

/* Does the given dimension have a name?
 */
isl_bool isl_set_has_dim_name(__isl_keep isl_set *set,
	enum isl_dim_type type, unsigned pos)
{
	if (!set)
		return isl_bool_error;
	return isl_space_has_dim_name(set->dim, type, pos);
}

__isl_give isl_basic_map *isl_basic_map_set_dim_name(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos, const char *s)
{
	isl_space *space;

	space = isl_basic_map_take_space(bmap);
	space = isl_space_set_dim_name(space, type, pos, s);
	bmap = isl_basic_map_restore_space(bmap, space);
	return isl_basic_map_finalize(bmap);
}

__isl_give isl_map *isl_map_set_dim_name(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned pos, const char *s)
{
	int i;
	isl_space *space;

	map = isl_map_cow(map);
	if (!map)
		return NULL;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_set_dim_name(map->p[i], type, pos, s);
		if (!map->p[i])
			goto error;
	}

	space = isl_map_take_space(map);
	space = isl_space_set_dim_name(space, type, pos, s);
	map = isl_map_restore_space(map, space);

	return map;
error:
	isl_map_free(map);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_set_dim_name(
	__isl_take isl_basic_set *bset,
	enum isl_dim_type type, unsigned pos, const char *s)
{
	return bset_from_bmap(isl_basic_map_set_dim_name(bset_to_bmap(bset),
							type, pos, s));
}

__isl_give isl_set *isl_set_set_dim_name(__isl_take isl_set *set,
	enum isl_dim_type type, unsigned pos, const char *s)
{
	return set_from_map(isl_map_set_dim_name(set_to_map(set),
							type, pos, s));
}

isl_bool isl_basic_map_has_dim_id(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos)
{
	if (!bmap)
		return isl_bool_error;
	return isl_space_has_dim_id(bmap->dim, type, pos);
}

__isl_give isl_id *isl_basic_set_get_dim_id(__isl_keep isl_basic_set *bset,
	enum isl_dim_type type, unsigned pos)
{
	return bset ? isl_space_get_dim_id(bset->dim, type, pos) : NULL;
}

isl_bool isl_map_has_dim_id(__isl_keep isl_map *map,
	enum isl_dim_type type, unsigned pos)
{
	return map ? isl_space_has_dim_id(map->dim, type, pos) : isl_bool_error;
}

__isl_give isl_id *isl_map_get_dim_id(__isl_keep isl_map *map,
	enum isl_dim_type type, unsigned pos)
{
	return map ? isl_space_get_dim_id(map->dim, type, pos) : NULL;
}

isl_bool isl_set_has_dim_id(__isl_keep isl_set *set,
	enum isl_dim_type type, unsigned pos)
{
	return isl_map_has_dim_id(set, type, pos);
}

__isl_give isl_id *isl_set_get_dim_id(__isl_keep isl_set *set,
	enum isl_dim_type type, unsigned pos)
{
	return isl_map_get_dim_id(set, type, pos);
}

__isl_give isl_map *isl_map_set_dim_id(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned pos, __isl_take isl_id *id)
{
	isl_space *space;

	space = isl_map_take_space(map);
	space = isl_space_set_dim_id(space, type, pos, id);
	map = isl_map_restore_space(map, space);

	return isl_map_reset_space(map, isl_map_get_space(map));
}

__isl_give isl_set *isl_set_set_dim_id(__isl_take isl_set *set,
	enum isl_dim_type type, unsigned pos, __isl_take isl_id *id)
{
	return isl_map_set_dim_id(set, type, pos, id);
}

int isl_map_find_dim_by_id(__isl_keep isl_map *map, enum isl_dim_type type,
	__isl_keep isl_id *id)
{
	if (!map)
		return -1;
	return isl_space_find_dim_by_id(map->dim, type, id);
}

int isl_set_find_dim_by_id(__isl_keep isl_set *set, enum isl_dim_type type,
	__isl_keep isl_id *id)
{
	return isl_map_find_dim_by_id(set, type, id);
}

/* Return the position of the dimension of the given type and name
 * in "bmap".
 * Return -1 if no such dimension can be found.
 */
int isl_basic_map_find_dim_by_name(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type, const char *name)
{
	if (!bmap)
		return -1;
	return isl_space_find_dim_by_name(bmap->dim, type, name);
}

int isl_map_find_dim_by_name(__isl_keep isl_map *map, enum isl_dim_type type,
	const char *name)
{
	if (!map)
		return -1;
	return isl_space_find_dim_by_name(map->dim, type, name);
}

int isl_set_find_dim_by_name(__isl_keep isl_set *set, enum isl_dim_type type,
	const char *name)
{
	return isl_map_find_dim_by_name(set, type, name);
}

/* Check whether equality i of bset is a pure stride constraint
 * on a single dimension, i.e., of the form
 *
 *	v = k e
 *
 * with k a constant and e an existentially quantified variable.
 */
isl_bool isl_basic_set_eq_is_stride(__isl_keep isl_basic_set *bset, int i)
{
	isl_size nparam;
	isl_size d;
	isl_size n_div;
	int pos1;
	int pos2;

	nparam = isl_basic_set_dim(bset, isl_dim_param);
	d = isl_basic_set_dim(bset, isl_dim_set);
	n_div = isl_basic_set_dim(bset, isl_dim_div);
	if (nparam < 0 || d < 0 || n_div < 0)
		return isl_bool_error;

	if (!isl_int_is_zero(bset->eq[i][0]))
		return isl_bool_false;

	if (isl_seq_first_non_zero(bset->eq[i] + 1, nparam) != -1)
		return isl_bool_false;
	pos1 = isl_seq_first_non_zero(bset->eq[i] + 1 + nparam, d);
	if (pos1 == -1)
		return isl_bool_false;
	if (isl_seq_first_non_zero(bset->eq[i] + 1 + nparam + pos1 + 1,
					d - pos1 - 1) != -1)
		return isl_bool_false;

	pos2 = isl_seq_first_non_zero(bset->eq[i] + 1 + nparam + d, n_div);
	if (pos2 == -1)
		return isl_bool_false;
	if (isl_seq_first_non_zero(bset->eq[i] + 1 + nparam + d  + pos2 + 1,
				   n_div - pos2 - 1) != -1)
		return isl_bool_false;
	if (!isl_int_is_one(bset->eq[i][1 + nparam + pos1]) &&
	    !isl_int_is_negone(bset->eq[i][1 + nparam + pos1]))
		return isl_bool_false;

	return isl_bool_true;
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * of the space of "map".
 */
__isl_give isl_map *isl_map_reset_user(__isl_take isl_map *map)
{
	isl_space *space;

	space = isl_map_get_space(map);
	space = isl_space_reset_user(space);
	map = isl_map_reset_space(map, space);

	return map;
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * of the space of "set".
 */
__isl_give isl_set *isl_set_reset_user(__isl_take isl_set *set)
{
	return isl_map_reset_user(set);
}

isl_bool isl_basic_map_is_rational(__isl_keep isl_basic_map *bmap)
{
	if (!bmap)
		return isl_bool_error;
	return ISL_F_ISSET(bmap, ISL_BASIC_MAP_RATIONAL);
}

/* Has "map" been marked as a rational map?
 * In particular, have all basic maps in "map" been marked this way?
 * An empty map is not considered to be rational.
 * Maps where only some of the basic maps are marked rational
 * are not allowed.
 */
isl_bool isl_map_is_rational(__isl_keep isl_map *map)
{
	int i;
	isl_bool rational;

	if (!map)
		return isl_bool_error;
	if (map->n == 0)
		return isl_bool_false;
	rational = isl_basic_map_is_rational(map->p[0]);
	if (rational < 0)
		return rational;
	for (i = 1; i < map->n; ++i) {
		isl_bool rational_i;

		rational_i = isl_basic_map_is_rational(map->p[i]);
		if (rational_i < 0)
			return rational_i;
		if (rational != rational_i)
			isl_die(isl_map_get_ctx(map), isl_error_unsupported,
				"mixed rational and integer basic maps "
				"not supported", return isl_bool_error);
	}

	return rational;
}

/* Has "set" been marked as a rational set?
 * In particular, have all basic set in "set" been marked this way?
 * An empty set is not considered to be rational.
 * Sets where only some of the basic sets are marked rational
 * are not allowed.
 */
isl_bool isl_set_is_rational(__isl_keep isl_set *set)
{
	return isl_map_is_rational(set);
}

int isl_basic_set_is_rational(__isl_keep isl_basic_set *bset)
{
	return isl_basic_map_is_rational(bset);
}

/* Does "bmap" contain any rational points?
 *
 * If "bmap" has an equality for each dimension, equating the dimension
 * to an integer constant, then it has no rational points, even if it
 * is marked as rational.
 */
isl_bool isl_basic_map_has_rational(__isl_keep isl_basic_map *bmap)
{
	isl_bool has_rational = isl_bool_true;
	isl_size total;

	if (!bmap)
		return isl_bool_error;
	if (isl_basic_map_plain_is_empty(bmap))
		return isl_bool_false;
	if (!isl_basic_map_is_rational(bmap))
		return isl_bool_false;
	bmap = isl_basic_map_copy(bmap);
	bmap = isl_basic_map_implicit_equalities(bmap);
	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return isl_bool_error;
	if (bmap->n_eq == total) {
		int i, j;
		for (i = 0; i < bmap->n_eq; ++i) {
			j = isl_seq_first_non_zero(bmap->eq[i] + 1, total);
			if (j < 0)
				break;
			if (!isl_int_is_one(bmap->eq[i][1 + j]) &&
			    !isl_int_is_negone(bmap->eq[i][1 + j]))
				break;
			j = isl_seq_first_non_zero(bmap->eq[i] + 1 + j + 1,
						    total - j - 1);
			if (j >= 0)
				break;
		}
		if (i == bmap->n_eq)
			has_rational = isl_bool_false;
	}
	isl_basic_map_free(bmap);

	return has_rational;
}

/* Does "map" contain any rational points?
 */
isl_bool isl_map_has_rational(__isl_keep isl_map *map)
{
	int i;
	isl_bool has_rational;

	if (!map)
		return isl_bool_error;
	for (i = 0; i < map->n; ++i) {
		has_rational = isl_basic_map_has_rational(map->p[i]);
		if (has_rational < 0 || has_rational)
			return has_rational;
	}
	return isl_bool_false;
}

/* Does "set" contain any rational points?
 */
isl_bool isl_set_has_rational(__isl_keep isl_set *set)
{
	return isl_map_has_rational(set);
}

/* Is this basic set a parameter domain?
 */
isl_bool isl_basic_set_is_params(__isl_keep isl_basic_set *bset)
{
	if (!bset)
		return isl_bool_error;
	return isl_space_is_params(bset->dim);
}

/* Is this set a parameter domain?
 */
isl_bool isl_set_is_params(__isl_keep isl_set *set)
{
	if (!set)
		return isl_bool_error;
	return isl_space_is_params(set->dim);
}

/* Is this map actually a parameter domain?
 * Users should never call this function.  Outside of isl,
 * a map can never be a parameter domain.
 */
isl_bool isl_map_is_params(__isl_keep isl_map *map)
{
	if (!map)
		return isl_bool_error;
	return isl_space_is_params(map->dim);
}

static __isl_give isl_basic_map *basic_map_init(isl_ctx *ctx,
	__isl_take isl_basic_map *bmap, unsigned extra,
	unsigned n_eq, unsigned n_ineq)
{
	int i;
	isl_space *space = isl_basic_map_peek_space(bmap);
	isl_size n_var = isl_space_dim(space, isl_dim_all);
	size_t row_size = 1 + n_var + extra;

	bmap->ctx = ctx;
	isl_ctx_ref(ctx);

	if (n_var < 0)
		return isl_basic_map_free(bmap);

	bmap->block = isl_blk_alloc(ctx, (n_ineq + n_eq) * row_size);
	if (isl_blk_is_error(bmap->block))
		goto error;

	bmap->ineq = isl_alloc_array(ctx, isl_int *, n_ineq + n_eq);
	if ((n_ineq + n_eq) && !bmap->ineq)
		goto error;

	if (extra == 0) {
		bmap->block2 = isl_blk_empty();
		bmap->div = NULL;
	} else {
		bmap->block2 = isl_blk_alloc(ctx, extra * (1 + row_size));
		if (isl_blk_is_error(bmap->block2))
			goto error;

		bmap->div = isl_alloc_array(ctx, isl_int *, extra);
		if (!bmap->div)
			goto error;
	}

	for (i = 0; i < n_ineq + n_eq; ++i)
		bmap->ineq[i] = bmap->block.data + i * row_size;

	for (i = 0; i < extra; ++i)
		bmap->div[i] = bmap->block2.data + i * (1 + row_size);

	bmap->ref = 1;
	bmap->flags = 0;
	bmap->c_size = n_eq + n_ineq;
	bmap->eq = bmap->ineq + n_ineq;
	bmap->extra = extra;
	bmap->n_eq = 0;
	bmap->n_ineq = 0;
	bmap->n_div = 0;
	bmap->sample = NULL;

	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_alloc(isl_ctx *ctx,
		unsigned nparam, unsigned dim, unsigned extra,
		unsigned n_eq, unsigned n_ineq)
{
	struct isl_basic_map *bmap;
	isl_space *space;

	space = isl_space_set_alloc(ctx, nparam, dim);
	if (!space)
		return NULL;

	bmap = isl_basic_map_alloc_space(space, extra, n_eq, n_ineq);
	return bset_from_bmap(bmap);
}

__isl_give isl_basic_set *isl_basic_set_alloc_space(__isl_take isl_space *space,
		unsigned extra, unsigned n_eq, unsigned n_ineq)
{
	struct isl_basic_map *bmap;
	if (!space)
		return NULL;
	isl_assert(space->ctx, space->n_in == 0, goto error);
	bmap = isl_basic_map_alloc_space(space, extra, n_eq, n_ineq);
	return bset_from_bmap(bmap);
error:
	isl_space_free(space);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_alloc_space(__isl_take isl_space *space,
		unsigned extra, unsigned n_eq, unsigned n_ineq)
{
	struct isl_basic_map *bmap;

	if (!space)
		return NULL;
	bmap = isl_calloc_type(space->ctx, struct isl_basic_map);
	if (!bmap)
		goto error;
	bmap->dim = space;

	return basic_map_init(space->ctx, bmap, extra, n_eq, n_ineq);
error:
	isl_space_free(space);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_alloc(isl_ctx *ctx,
		unsigned nparam, unsigned in, unsigned out, unsigned extra,
		unsigned n_eq, unsigned n_ineq)
{
	struct isl_basic_map *bmap;
	isl_space *space;

	space = isl_space_alloc(ctx, nparam, in, out);
	if (!space)
		return NULL;

	bmap = isl_basic_map_alloc_space(space, extra, n_eq, n_ineq);
	return bmap;
}

static __isl_give isl_basic_map *dup_constraints(__isl_take isl_basic_map *dst,
	__isl_keep isl_basic_map *src)
{
	int i;
	isl_size total = isl_basic_map_dim(src, isl_dim_all);

	if (!dst || total < 0)
		return isl_basic_map_free(dst);

	for (i = 0; i < src->n_eq; ++i) {
		int j = isl_basic_map_alloc_equality(dst);
		if (j < 0)
			return isl_basic_map_free(dst);
		isl_seq_cpy(dst->eq[j], src->eq[i], 1+total);
	}

	for (i = 0; i < src->n_ineq; ++i) {
		int j = isl_basic_map_alloc_inequality(dst);
		if (j < 0)
			return isl_basic_map_free(dst);
		isl_seq_cpy(dst->ineq[j], src->ineq[i], 1+total);
	}

	for (i = 0; i < src->n_div; ++i) {
		int j = isl_basic_map_alloc_div(dst);
		if (j < 0)
			return isl_basic_map_free(dst);
		isl_seq_cpy(dst->div[j], src->div[i], 1+1+total);
	}
	ISL_F_SET(dst, ISL_BASIC_SET_FINAL);
	return dst;
}

__isl_give isl_basic_map *isl_basic_map_dup(__isl_keep isl_basic_map *bmap)
{
	struct isl_basic_map *dup;

	if (!bmap)
		return NULL;
	dup = isl_basic_map_alloc_space(isl_space_copy(bmap->dim),
			bmap->n_div, bmap->n_eq, bmap->n_ineq);
	dup = dup_constraints(dup, bmap);
	if (!dup)
		return NULL;
	dup->flags = bmap->flags;
	dup->sample = isl_vec_copy(bmap->sample);
	return dup;
}

__isl_give isl_basic_set *isl_basic_set_dup(__isl_keep isl_basic_set *bset)
{
	struct isl_basic_map *dup;

	dup = isl_basic_map_dup(bset_to_bmap(bset));
	return bset_from_bmap(dup);
}

__isl_give isl_basic_set *isl_basic_set_copy(__isl_keep isl_basic_set *bset)
{
	if (!bset)
		return NULL;

	if (ISL_F_ISSET(bset, ISL_BASIC_SET_FINAL)) {
		bset->ref++;
		return bset;
	}
	return isl_basic_set_dup(bset);
}

__isl_give isl_set *isl_set_copy(__isl_keep isl_set *set)
{
	if (!set)
		return NULL;

	set->ref++;
	return set;
}

__isl_give isl_basic_map *isl_basic_map_copy(__isl_keep isl_basic_map *bmap)
{
	if (!bmap)
		return NULL;

	if (ISL_F_ISSET(bmap, ISL_BASIC_SET_FINAL)) {
		bmap->ref++;
		return bmap;
	}
	bmap = isl_basic_map_dup(bmap);
	if (bmap)
		ISL_F_SET(bmap, ISL_BASIC_SET_FINAL);
	return bmap;
}

__isl_give isl_map *isl_map_copy(__isl_keep isl_map *map)
{
	if (!map)
		return NULL;

	map->ref++;
	return map;
}

__isl_null isl_basic_map *isl_basic_map_free(__isl_take isl_basic_map *bmap)
{
	if (!bmap)
		return NULL;

	if (--bmap->ref > 0)
		return NULL;

	isl_ctx_deref(bmap->ctx);
	free(bmap->div);
	isl_blk_free(bmap->ctx, bmap->block2);
	free(bmap->ineq);
	isl_blk_free(bmap->ctx, bmap->block);
	isl_vec_free(bmap->sample);
	isl_space_free(bmap->dim);
	free(bmap);

	return NULL;
}

__isl_null isl_basic_set *isl_basic_set_free(__isl_take isl_basic_set *bset)
{
	return isl_basic_map_free(bset_to_bmap(bset));
}

static int room_for_con(__isl_keep isl_basic_map *bmap, unsigned n)
{
	return bmap->n_eq + bmap->n_ineq + n <= bmap->c_size;
}

/* Check that "bset" does not involve any parameters.
 */
isl_stat isl_basic_set_check_no_params(__isl_keep isl_basic_set *bset)
{
	isl_size nparam;

	nparam = isl_basic_set_dim(bset, isl_dim_param);
	if (nparam < 0)
		return isl_stat_error;
	if (nparam != 0)
		isl_die(isl_basic_set_get_ctx(bset), isl_error_invalid,
			"basic set should not have any parameters",
			return isl_stat_error);
	return isl_stat_ok;
}

/* Check that "bset" does not involve any local variables.
 */
isl_stat isl_basic_set_check_no_locals(__isl_keep isl_basic_set *bset)
{
	isl_size n_div;

	n_div = isl_basic_set_dim(bset, isl_dim_div);
	if (n_div < 0)
		return isl_stat_error;
	if (n_div != 0)
		isl_die(isl_basic_set_get_ctx(bset), isl_error_invalid,
			"basic set should not have any local variables",
			return isl_stat_error);
	return isl_stat_ok;
}

#undef TYPE
#define TYPE isl_map

#include "isl_check_named_params_templ.c"

#undef TYPE
#define TYPE isl_basic_map

static
#include "isl_check_named_params_templ.c"

/* Check that "bmap1" and "bmap2" have the same parameters,
 * reporting an error if they do not.
 */
static isl_stat isl_basic_map_check_equal_params(
	__isl_keep isl_basic_map *bmap1, __isl_keep isl_basic_map *bmap2)
{
	isl_bool match;

	match = isl_basic_map_has_equal_params(bmap1, bmap2);
	if (match < 0)
		return isl_stat_error;
	if (!match)
		isl_die(isl_basic_map_get_ctx(bmap1), isl_error_invalid,
			"parameters don't match", return isl_stat_error);
	return isl_stat_ok;
}

#undef TYPE
#define TYPE	isl_map

#include "isl_align_params_bin_templ.c"

#undef SUFFIX
#define SUFFIX	set
#undef ARG1
#define ARG1	isl_map
#undef ARG2
#define ARG2	isl_set

#include "isl_align_params_templ.c"

isl_bool isl_map_align_params_map_map_and_test(__isl_keep isl_map *map1,
	__isl_keep isl_map *map2,
	isl_bool (*fn)(__isl_keep isl_map *map1, __isl_keep isl_map *map2))
{
	isl_bool r;

	if (!map1 || !map2)
		return isl_bool_error;
	if (isl_map_has_equal_params(map1, map2))
		return fn(map1, map2);
	if (isl_map_check_named_params(map1) < 0)
		return isl_bool_error;
	if (isl_map_check_named_params(map2) < 0)
		return isl_bool_error;
	map1 = isl_map_copy(map1);
	map2 = isl_map_copy(map2);
	map1 = isl_map_align_params(map1, isl_map_get_space(map2));
	map2 = isl_map_align_params(map2, isl_map_get_space(map1));
	r = fn(map1, map2);
	isl_map_free(map1);
	isl_map_free(map2);
	return r;
}

int isl_basic_map_alloc_equality(__isl_keep isl_basic_map *bmap)
{
	isl_size total;
	struct isl_ctx *ctx;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return -1;
	ctx = bmap->ctx;
	isl_assert(ctx, room_for_con(bmap, 1), return -1);
	isl_assert(ctx, (bmap->eq - bmap->ineq) + bmap->n_eq <= bmap->c_size,
			return -1);
	ISL_F_CLR(bmap, ISL_BASIC_MAP_NO_REDUNDANT);
	ISL_F_CLR(bmap, ISL_BASIC_MAP_NO_IMPLICIT);
	ISL_F_CLR(bmap, ISL_BASIC_MAP_ALL_EQUALITIES);
	ISL_F_CLR(bmap, ISL_BASIC_MAP_NORMALIZED_DIVS);
	if ((bmap->eq - bmap->ineq) + bmap->n_eq == bmap->c_size) {
		isl_int *t;
		int j = isl_basic_map_alloc_inequality(bmap);
		if (j < 0)
			return -1;
		t = bmap->ineq[j];
		bmap->ineq[j] = bmap->ineq[bmap->n_ineq - 1];
		bmap->ineq[bmap->n_ineq - 1] = bmap->eq[-1];
		bmap->eq[-1] = t;
		bmap->n_eq++;
		bmap->n_ineq--;
		bmap->eq--;
		return 0;
	}
	isl_seq_clr(bmap->eq[bmap->n_eq] + 1 + total,
		      bmap->extra - bmap->n_div);
	return bmap->n_eq++;
}

int isl_basic_set_alloc_equality(__isl_keep isl_basic_set *bset)
{
	return isl_basic_map_alloc_equality(bset_to_bmap(bset));
}

__isl_give isl_basic_map *isl_basic_map_free_equality(
	__isl_take isl_basic_map *bmap, unsigned n)
{
	if (!bmap)
		return NULL;
	if (n > bmap->n_eq)
		isl_die(isl_basic_map_get_ctx(bmap), isl_error_invalid,
			"invalid number of equalities",
			isl_basic_map_free(bmap));
	bmap->n_eq -= n;
	return bmap;
}

__isl_give isl_basic_set *isl_basic_set_free_equality(
	__isl_take isl_basic_set *bset, unsigned n)
{
	return bset_from_bmap(isl_basic_map_free_equality(bset_to_bmap(bset),
							    n));
}

/* Drop the equality constraint at position "pos",
 * preserving the order of the other equality constraints.
 */
int isl_basic_map_drop_equality(__isl_keep isl_basic_map *bmap, unsigned pos)
{
	isl_int *t;
	int r;

	if (!bmap)
		return -1;
	isl_assert(bmap->ctx, pos < bmap->n_eq, return -1);

	t = bmap->eq[pos];
	bmap->n_eq--;
	for (r = pos; r < bmap->n_eq; ++r)
		bmap->eq[r] = bmap->eq[r + 1];
	bmap->eq[bmap->n_eq] = t;

	return 0;
}

/* Turn inequality "pos" of "bmap" into an equality.
 *
 * In particular, we move the inequality in front of the equalities
 * and move the last inequality in the position of the moved inequality.
 * Note that isl_tab_make_equalities_explicit depends on this particular
 * change in the ordering of the constraints.
 */
void isl_basic_map_inequality_to_equality(
		__isl_keep isl_basic_map *bmap, unsigned pos)
{
	isl_int *t;

	t = bmap->ineq[pos];
	bmap->ineq[pos] = bmap->ineq[bmap->n_ineq - 1];
	bmap->ineq[bmap->n_ineq - 1] = bmap->eq[-1];
	bmap->eq[-1] = t;
	bmap->n_eq++;
	bmap->n_ineq--;
	bmap->eq--;
	ISL_F_CLR(bmap, ISL_BASIC_MAP_NO_REDUNDANT);
	ISL_F_CLR(bmap, ISL_BASIC_MAP_SORTED);
	ISL_F_CLR(bmap, ISL_BASIC_MAP_NORMALIZED_DIVS);
	ISL_F_CLR(bmap, ISL_BASIC_MAP_ALL_EQUALITIES);
}

static int room_for_ineq(__isl_keep isl_basic_map *bmap, unsigned n)
{
	return bmap->n_ineq + n <= bmap->eq - bmap->ineq;
}

int isl_basic_map_alloc_inequality(__isl_keep isl_basic_map *bmap)
{
	isl_size total;
	struct isl_ctx *ctx;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return -1;
	ctx = bmap->ctx;
	isl_assert(ctx, room_for_ineq(bmap, 1), return -1);
	ISL_F_CLR(bmap, ISL_BASIC_MAP_NO_IMPLICIT);
	ISL_F_CLR(bmap, ISL_BASIC_MAP_NO_REDUNDANT);
	ISL_F_CLR(bmap, ISL_BASIC_MAP_SORTED);
	ISL_F_CLR(bmap, ISL_BASIC_MAP_ALL_EQUALITIES);
	isl_seq_clr(bmap->ineq[bmap->n_ineq] + 1 + total,
		      bmap->extra - bmap->n_div);
	return bmap->n_ineq++;
}

int isl_basic_set_alloc_inequality(__isl_keep isl_basic_set *bset)
{
	return isl_basic_map_alloc_inequality(bset_to_bmap(bset));
}

__isl_give isl_basic_map *isl_basic_map_free_inequality(
	__isl_take isl_basic_map *bmap, unsigned n)
{
	if (!bmap)
		return NULL;
	if (n > bmap->n_ineq)
		isl_die(isl_basic_map_get_ctx(bmap), isl_error_invalid,
			"invalid number of inequalities",
			return isl_basic_map_free(bmap));
	bmap->n_ineq -= n;
	return bmap;
}

__isl_give isl_basic_set *isl_basic_set_free_inequality(
	__isl_take isl_basic_set *bset, unsigned n)
{
	return bset_from_bmap(isl_basic_map_free_inequality(bset_to_bmap(bset),
							    n));
}

int isl_basic_map_drop_inequality(__isl_keep isl_basic_map *bmap, unsigned pos)
{
	isl_int *t;
	if (!bmap)
		return -1;
	isl_assert(bmap->ctx, pos < bmap->n_ineq, return -1);

	if (pos != bmap->n_ineq - 1) {
		t = bmap->ineq[pos];
		bmap->ineq[pos] = bmap->ineq[bmap->n_ineq - 1];
		bmap->ineq[bmap->n_ineq - 1] = t;
		ISL_F_CLR(bmap, ISL_BASIC_MAP_SORTED);
	}
	bmap->n_ineq--;
	return 0;
}

int isl_basic_set_drop_inequality(__isl_keep isl_basic_set *bset, unsigned pos)
{
	return isl_basic_map_drop_inequality(bset_to_bmap(bset), pos);
}

__isl_give isl_basic_map *isl_basic_map_add_eq(__isl_take isl_basic_map *bmap,
	isl_int *eq)
{
	isl_bool empty;
	isl_size total;
	int k;

	empty = isl_basic_map_plain_is_empty(bmap);
	if (empty < 0)
		return isl_basic_map_free(bmap);
	if (empty)
		return bmap;

	bmap = isl_basic_map_cow(bmap);
	bmap = isl_basic_map_extend_constraints(bmap, 1, 0);
	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return isl_basic_map_free(bmap);
	k = isl_basic_map_alloc_equality(bmap);
	if (k < 0)
		goto error;
	isl_seq_cpy(bmap->eq[k], eq, 1 + total);
	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_add_eq(__isl_take isl_basic_set *bset,
	isl_int *eq)
{
	return bset_from_bmap(isl_basic_map_add_eq(bset_to_bmap(bset), eq));
}

__isl_give isl_basic_map *isl_basic_map_add_ineq(__isl_take isl_basic_map *bmap,
	isl_int *ineq)
{
	isl_size total;
	int k;

	bmap = isl_basic_map_cow(bmap);
	bmap = isl_basic_map_extend_constraints(bmap, 0, 1);
	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return isl_basic_map_free(bmap);
	k = isl_basic_map_alloc_inequality(bmap);
	if (k < 0)
		goto error;
	isl_seq_cpy(bmap->ineq[k], ineq, 1 + total);
	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_add_ineq(__isl_take isl_basic_set *bset,
	isl_int *ineq)
{
	return bset_from_bmap(isl_basic_map_add_ineq(bset_to_bmap(bset), ineq));
}

int isl_basic_map_alloc_div(__isl_keep isl_basic_map *bmap)
{
	isl_size total;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return -1;
	isl_assert(bmap->ctx, bmap->n_div < bmap->extra, return -1);
	isl_seq_clr(bmap->div[bmap->n_div] + 1 + 1 + total,
		      bmap->extra - bmap->n_div);
	ISL_F_CLR(bmap, ISL_BASIC_MAP_NORMALIZED_DIVS);
	return bmap->n_div++;
}

int isl_basic_set_alloc_div(__isl_keep isl_basic_set *bset)
{
	return isl_basic_map_alloc_div(bset_to_bmap(bset));
}

#undef TYPE
#define TYPE	isl_basic_map
#include "check_type_range_templ.c"

/* Check that there are "n" dimensions of type "type" starting at "first"
 * in "bset".
 */
isl_stat isl_basic_set_check_range(__isl_keep isl_basic_set *bset,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return isl_basic_map_check_range(bset_to_bmap(bset),
					type, first, n);
}

/* Insert an extra integer division, prescribed by "div", to "bmap"
 * at (integer division) position "pos".
 *
 * The integer division is first added at the end and then moved
 * into the right position.
 */
__isl_give isl_basic_map *isl_basic_map_insert_div(
	__isl_take isl_basic_map *bmap, int pos, __isl_keep isl_vec *div)
{
	int i, k;
	isl_size total;

	bmap = isl_basic_map_cow(bmap);
	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0 || !div)
		return isl_basic_map_free(bmap);

	if (div->size != 1 + 1 + total)
		isl_die(isl_basic_map_get_ctx(bmap), isl_error_invalid,
			"unexpected size", return isl_basic_map_free(bmap));
	if (isl_basic_map_check_range(bmap, isl_dim_div, pos, 0) < 0)
		return isl_basic_map_free(bmap);

	bmap = isl_basic_map_extend(bmap, 1, 0, 2);
	k = isl_basic_map_alloc_div(bmap);
	if (k < 0)
		return isl_basic_map_free(bmap);
	isl_seq_cpy(bmap->div[k], div->el, div->size);
	isl_int_set_si(bmap->div[k][div->size], 0);

	for (i = k; i > pos; --i)
		bmap = isl_basic_map_swap_div(bmap, i, i - 1);

	return bmap;
}

isl_stat isl_basic_map_free_div(__isl_keep isl_basic_map *bmap, unsigned n)
{
	if (!bmap)
		return isl_stat_error;
	isl_assert(bmap->ctx, n <= bmap->n_div, return isl_stat_error);
	bmap->n_div -= n;
	return isl_stat_ok;
}

static __isl_give isl_basic_map *add_constraints(
	__isl_take isl_basic_map *bmap1, __isl_take isl_basic_map *bmap2,
	unsigned i_pos, unsigned o_pos)
{
	isl_size total, n_param, n_in, n_out, n_div;
	unsigned o_in, o_out;
	isl_ctx *ctx;
	isl_space *space;
	struct isl_dim_map *dim_map;

	space = isl_basic_map_peek_space(bmap2);
	if (!bmap1 || !space)
		goto error;

	total = isl_basic_map_dim(bmap1, isl_dim_all);
	n_param = isl_basic_map_dim(bmap2, isl_dim_param);
	n_in = isl_basic_map_dim(bmap2, isl_dim_in);
	o_in = isl_basic_map_offset(bmap1, isl_dim_in) - 1 + i_pos;
	n_out = isl_basic_map_dim(bmap2, isl_dim_out);
	o_out = isl_basic_map_offset(bmap1, isl_dim_out) - 1 + o_pos;
	n_div = isl_basic_map_dim(bmap2, isl_dim_div);
	if (total < 0 || n_param < 0 || n_in < 0 || n_out < 0 || n_div < 0)
		goto error;
	ctx = isl_basic_map_get_ctx(bmap1);
	dim_map = isl_dim_map_alloc(ctx, total + n_div);
	isl_dim_map_dim_range(dim_map, space, isl_dim_param, 0, n_param, 0);
	isl_dim_map_dim_range(dim_map, space, isl_dim_in, 0, n_in, o_in);
	isl_dim_map_dim_range(dim_map, space, isl_dim_out, 0, n_out, o_out);
	isl_dim_map_div(dim_map, bmap2, total);

	return isl_basic_map_add_constraints_dim_map(bmap1, bmap2, dim_map);
error:
	isl_basic_map_free(bmap1);
	isl_basic_map_free(bmap2);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_extend(__isl_take isl_basic_map *base,
	unsigned extra, unsigned n_eq, unsigned n_ineq)
{
	isl_space *space;
	struct isl_basic_map *ext;
	unsigned flags;
	int dims_ok;

	if (!base)
		goto error;

	dims_ok = base->extra >= base->n_div + extra;

	if (dims_ok && room_for_con(base, n_eq + n_ineq) &&
		       room_for_ineq(base, n_ineq))
		return base;

	extra += base->extra;
	n_eq += base->n_eq;
	n_ineq += base->n_ineq;

	space = isl_basic_map_get_space(base);
	ext = isl_basic_map_alloc_space(space, extra, n_eq, n_ineq);
	if (!ext)
		goto error;

	if (dims_ok)
		ext->sample = isl_vec_copy(base->sample);
	flags = base->flags;
	ext = add_constraints(ext, base, 0, 0);
	if (ext) {
		ext->flags = flags;
		ISL_F_CLR(ext, ISL_BASIC_SET_FINAL);
	}

	return ext;

error:
	isl_basic_map_free(base);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_extend(__isl_take isl_basic_set *base,
	unsigned extra, unsigned n_eq, unsigned n_ineq)
{
	return bset_from_bmap(isl_basic_map_extend(bset_to_bmap(base),
						    extra, n_eq, n_ineq));
}

__isl_give isl_basic_map *isl_basic_map_extend_constraints(
	__isl_take isl_basic_map *base, unsigned n_eq, unsigned n_ineq)
{
	return isl_basic_map_extend(base, 0, n_eq, n_ineq);
}

__isl_give isl_basic_set *isl_basic_set_extend_constraints(
	__isl_take isl_basic_set *base, unsigned n_eq, unsigned n_ineq)
{
	isl_basic_map *bmap = bset_to_bmap(base);
	bmap = isl_basic_map_extend_constraints(bmap, n_eq, n_ineq);
	return bset_from_bmap(bmap);
}

__isl_give isl_basic_set *isl_basic_set_cow(__isl_take isl_basic_set *bset)
{
	return bset_from_bmap(isl_basic_map_cow(bset_to_bmap(bset)));
}

__isl_give isl_basic_map *isl_basic_map_cow(__isl_take isl_basic_map *bmap)
{
	if (!bmap)
		return NULL;

	if (bmap->ref > 1) {
		bmap->ref--;
		bmap = isl_basic_map_dup(bmap);
	}
	if (bmap) {
		ISL_F_CLR(bmap, ISL_BASIC_SET_FINAL);
		ISL_F_CLR(bmap, ISL_BASIC_MAP_REDUCED_COEFFICIENTS);
	}
	return bmap;
}

/* Clear all cached information in "map", either because it is about
 * to be modified or because it is being freed.
 * Always return the same pointer that is passed in.
 * This is needed for the use in isl_map_free.
 */
static __isl_give isl_map *clear_caches(__isl_take isl_map *map)
{
	isl_basic_map_free(map->cached_simple_hull[0]);
	isl_basic_map_free(map->cached_simple_hull[1]);
	map->cached_simple_hull[0] = NULL;
	map->cached_simple_hull[1] = NULL;
	return map;
}

__isl_give isl_set *isl_set_cow(__isl_take isl_set *set)
{
	return isl_map_cow(set);
}

/* Return an isl_map that is equal to "map" and that has only
 * a single reference.
 *
 * If the original input already has only one reference, then
 * simply return it, but clear all cached information, since
 * it may be rendered invalid by the operations that will be
 * performed on the result.
 *
 * Otherwise, create a duplicate (without any cached information).
 */
__isl_give isl_map *isl_map_cow(__isl_take isl_map *map)
{
	if (!map)
		return NULL;

	if (map->ref == 1)
		return clear_caches(map);
	map->ref--;
	return isl_map_dup(map);
}

static void swap_vars(struct isl_blk blk, isl_int *a,
			unsigned a_len, unsigned b_len)
{
	isl_seq_cpy(blk.data, a+a_len, b_len);
	isl_seq_cpy(blk.data+b_len, a, a_len);
	isl_seq_cpy(a, blk.data, b_len+a_len);
}

static __isl_give isl_basic_map *isl_basic_map_swap_vars(
	__isl_take isl_basic_map *bmap, unsigned pos, unsigned n1, unsigned n2)
{
	int i;
	struct isl_blk blk;

	if (isl_basic_map_check_range(bmap, isl_dim_all, pos - 1, n1 + n2) < 0)
		goto error;

	if (n1 == 0 || n2 == 0)
		return bmap;

	bmap = isl_basic_map_cow(bmap);
	if (!bmap)
		return NULL;

	blk = isl_blk_alloc(bmap->ctx, n1 + n2);
	if (isl_blk_is_error(blk))
		goto error;

	for (i = 0; i < bmap->n_eq; ++i)
		swap_vars(blk,
			  bmap->eq[i] + pos, n1, n2);

	for (i = 0; i < bmap->n_ineq; ++i)
		swap_vars(blk,
			  bmap->ineq[i] + pos, n1, n2);

	for (i = 0; i < bmap->n_div; ++i)
		swap_vars(blk,
			  bmap->div[i]+1 + pos, n1, n2);

	isl_blk_free(bmap->ctx, blk);

	ISL_F_CLR(bmap, ISL_BASIC_SET_SORTED);
	bmap = isl_basic_map_gauss(bmap, NULL);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap);
	return NULL;
}

/* The given basic map has turned out to be empty.
 * Explicitly mark it as such and change the representation
 * to a canonical representation of the empty basic map.
 * Since the basic map has conflicting constraints,
 * it must have at least one constraint, except perhaps
 * if it was already explicitly marked as being empty.
 * Do nothing in the latter case, i.e., if it has been marked empty and
 * has no constraints.
 */
__isl_give isl_basic_map *isl_basic_map_set_to_empty(
	__isl_take isl_basic_map *bmap)
{
	int i = 0;
	isl_bool empty;
	isl_size n;
	isl_size total;

	n = isl_basic_map_n_constraint(bmap);
	empty = isl_basic_map_plain_is_empty(bmap);
	if (n < 0 || empty < 0)
		return isl_basic_map_free(bmap);
	if (n == 0 && empty)
		return bmap;
	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return isl_basic_map_free(bmap);
	if (isl_basic_map_free_div(bmap, bmap->n_div) < 0)
		return isl_basic_map_free(bmap);
	bmap = isl_basic_map_free_inequality(bmap, bmap->n_ineq);
	if (!bmap)
		return NULL;
	if (bmap->n_eq > 0) {
		bmap = isl_basic_map_free_equality(bmap, bmap->n_eq - 1);
		if (!bmap)
			return NULL;
	} else {
		i = isl_basic_map_alloc_equality(bmap);
		if (i < 0)
			goto error;
	}
	isl_int_set_si(bmap->eq[i][0], 1);
	isl_seq_clr(bmap->eq[i]+1, total);
	ISL_F_SET(bmap, ISL_BASIC_MAP_EMPTY);
	isl_vec_free(bmap->sample);
	bmap->sample = NULL;
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_set_to_empty(
	__isl_take isl_basic_set *bset)
{
	return bset_from_bmap(isl_basic_map_set_to_empty(bset_to_bmap(bset)));
}

__isl_give isl_basic_map *isl_basic_map_set_rational(
	__isl_take isl_basic_map *bmap)
{
	if (!bmap)
		return NULL;

	if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_RATIONAL))
		return bmap;

	bmap = isl_basic_map_cow(bmap);
	if (!bmap)
		return NULL;

	ISL_F_SET(bmap, ISL_BASIC_MAP_RATIONAL);

	return isl_basic_map_finalize(bmap);
}

__isl_give isl_basic_set *isl_basic_set_set_rational(
	__isl_take isl_basic_set *bset)
{
	return isl_basic_map_set_rational(bset);
}

__isl_give isl_basic_set *isl_basic_set_set_integral(
	__isl_take isl_basic_set *bset)
{
	if (!bset)
		return NULL;

	if (!ISL_F_ISSET(bset, ISL_BASIC_MAP_RATIONAL))
		return bset;

	bset = isl_basic_set_cow(bset);
	if (!bset)
		return NULL;

	ISL_F_CLR(bset, ISL_BASIC_MAP_RATIONAL);

	return isl_basic_set_finalize(bset);
}

__isl_give isl_map *isl_map_set_rational(__isl_take isl_map *map)
{
	int i;

	map = isl_map_cow(map);
	if (!map)
		return NULL;
	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_set_rational(map->p[i]);
		if (!map->p[i])
			goto error;
	}
	return map;
error:
	isl_map_free(map);
	return NULL;
}

__isl_give isl_set *isl_set_set_rational(__isl_take isl_set *set)
{
	return isl_map_set_rational(set);
}

/* Swap divs "a" and "b" in "bmap" (without modifying any of the constraints
 * of "bmap").
 */
static void swap_div(__isl_keep isl_basic_map *bmap, int a, int b)
{
	isl_int *t = bmap->div[a];
	bmap->div[a] = bmap->div[b];
	bmap->div[b] = t;
}

/* Swap divs "a" and "b" in "bmap" and adjust the constraints and
 * div definitions accordingly.
 */
__isl_give isl_basic_map *isl_basic_map_swap_div(__isl_take isl_basic_map *bmap,
	int a, int b)
{
	int i;
	isl_size off;

	off = isl_basic_map_var_offset(bmap, isl_dim_div);
	if (off < 0)
		return isl_basic_map_free(bmap);

	swap_div(bmap, a, b);

	for (i = 0; i < bmap->n_eq; ++i)
		isl_int_swap(bmap->eq[i][1+off+a], bmap->eq[i][1+off+b]);

	for (i = 0; i < bmap->n_ineq; ++i)
		isl_int_swap(bmap->ineq[i][1+off+a], bmap->ineq[i][1+off+b]);

	for (i = 0; i < bmap->n_div; ++i)
		isl_int_swap(bmap->div[i][1+1+off+a], bmap->div[i][1+1+off+b]);
	ISL_F_CLR(bmap, ISL_BASIC_MAP_SORTED);

	return bmap;
}

static void constraint_drop_vars(isl_int *c, unsigned n, unsigned rem)
{
	isl_seq_cpy(c, c + n, rem);
	isl_seq_clr(c + rem, n);
}

/* Drop n dimensions starting at first.
 *
 * In principle, this frees up some extra variables as the number
 * of columns remains constant, but we would have to extend
 * the div array too as the number of rows in this array is assumed
 * to be equal to extra.
 */
__isl_give isl_basic_set *isl_basic_set_drop_dims(
	__isl_take isl_basic_set *bset, unsigned first, unsigned n)
{
	return isl_basic_map_drop(bset_to_bmap(bset), isl_dim_set, first, n);
}

/* Move "n" divs starting at "first" to the end of the list of divs.
 */
static __isl_give isl_basic_map *move_divs_last(__isl_take isl_basic_map *bmap,
	unsigned first, unsigned n)
{
	isl_int **div;
	int i;

	if (first + n == bmap->n_div)
		return bmap;

	div = isl_alloc_array(bmap->ctx, isl_int *, n);
	if (!div)
		goto error;
	for (i = 0; i < n; ++i)
		div[i] = bmap->div[first + i];
	for (i = 0; i < bmap->n_div - first - n; ++i)
		bmap->div[first + i] = bmap->div[first + n + i];
	for (i = 0; i < n; ++i)
		bmap->div[bmap->n_div - n + i] = div[i];
	free(div);
	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

#undef TYPE
#define TYPE	isl_map
static
#include "check_type_range_templ.c"

/* Check that there are "n" dimensions of type "type" starting at "first"
 * in "set".
 */
isl_stat isl_set_check_range(__isl_keep isl_set *set,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return isl_map_check_range(set_to_map(set), type, first, n);
}

/* Drop "n" dimensions of type "type" starting at "first".
 * Perform the core computation, without cowing or
 * simplifying and finalizing the result.
 *
 * In principle, this frees up some extra variables as the number
 * of columns remains constant, but we would have to extend
 * the div array too as the number of rows in this array is assumed
 * to be equal to extra.
 */
__isl_give isl_basic_map *isl_basic_map_drop_core(
	__isl_take isl_basic_map *bmap, enum isl_dim_type type,
	unsigned first, unsigned n)
{
	int i;
	unsigned offset;
	unsigned left;
	isl_size total;

	if (isl_basic_map_check_range(bmap, type, first, n) < 0)
		return isl_basic_map_free(bmap);

	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return isl_basic_map_free(bmap);

	offset = isl_basic_map_offset(bmap, type) + first;
	left = total - (offset - 1) - n;
	for (i = 0; i < bmap->n_eq; ++i)
		constraint_drop_vars(bmap->eq[i]+offset, n, left);

	for (i = 0; i < bmap->n_ineq; ++i)
		constraint_drop_vars(bmap->ineq[i]+offset, n, left);

	for (i = 0; i < bmap->n_div; ++i)
		constraint_drop_vars(bmap->div[i]+1+offset, n, left);

	if (type == isl_dim_div) {
		bmap = move_divs_last(bmap, first, n);
		if (!bmap)
			return NULL;
		if (isl_basic_map_free_div(bmap, n) < 0)
			return isl_basic_map_free(bmap);
	} else
		bmap->dim = isl_space_drop_dims(bmap->dim, type, first, n);
	if (!bmap->dim)
		return isl_basic_map_free(bmap);

	ISL_F_CLR(bmap, ISL_BASIC_MAP_NO_REDUNDANT);
	ISL_F_CLR(bmap, ISL_BASIC_MAP_SORTED);
	return bmap;
}

/* Drop "n" dimensions of type "type" starting at "first".
 *
 * In principle, this frees up some extra variables as the number
 * of columns remains constant, but we would have to extend
 * the div array too as the number of rows in this array is assumed
 * to be equal to extra.
 */
__isl_give isl_basic_map *isl_basic_map_drop(__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	if (!bmap)
		return NULL;
	if (n == 0 && !isl_space_is_named_or_nested(bmap->dim, type))
		return bmap;

	bmap = isl_basic_map_cow(bmap);
	if (!bmap)
		return NULL;

	bmap = isl_basic_map_drop_core(bmap, type, first, n);

	bmap = isl_basic_map_simplify(bmap);
	return isl_basic_map_finalize(bmap);
}

__isl_give isl_basic_set *isl_basic_set_drop(__isl_take isl_basic_set *bset,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return bset_from_bmap(isl_basic_map_drop(bset_to_bmap(bset),
							type, first, n));
}

/* No longer consider "map" to be normalized.
 */
static __isl_give isl_map *isl_map_unmark_normalized(__isl_take isl_map *map)
{
	if (!map)
		return NULL;
	ISL_F_CLR(map, ISL_MAP_NORMALIZED);
	return map;
}

__isl_give isl_map *isl_map_drop(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;
	isl_space *space;

	if (isl_map_check_range(map, type, first, n) < 0)
		return isl_map_free(map);

	if (n == 0 && !isl_space_is_named_or_nested(map->dim, type))
		return map;
	map = isl_map_cow(map);
	if (!map)
		goto error;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_drop(map->p[i], type, first, n);
		if (!map->p[i])
			goto error;
	}
	map = isl_map_unmark_normalized(map);

	space = isl_map_take_space(map);
	space = isl_space_drop_dims(space, type, first, n);
	map = isl_map_restore_space(map, space);

	return map;
error:
	isl_map_free(map);
	return NULL;
}

__isl_give isl_set *isl_set_drop(__isl_take isl_set *set,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return set_from_map(isl_map_drop(set_to_map(set), type, first, n));
}

/* Drop the integer division at position "div", which is assumed
 * not to appear in any of the constraints or
 * in any of the other integer divisions.
 *
 * Since the integer division is redundant, there is no need to cow.
 */
__isl_give isl_basic_map *isl_basic_map_drop_div(
	__isl_take isl_basic_map *bmap, unsigned div)
{
	return isl_basic_map_drop_core(bmap, isl_dim_div, div, 1);
}

/* Eliminate the specified n dimensions starting at first from the
 * constraints, without removing the dimensions from the space.
 * If the set is rational, the dimensions are eliminated using Fourier-Motzkin.
 */
__isl_give isl_map *isl_map_eliminate(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;

	if (n == 0)
		return map;

	if (isl_map_check_range(map, type, first, n) < 0)
		return isl_map_free(map);

	map = isl_map_cow(map);
	if (!map)
		return NULL;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_eliminate(map->p[i], type, first, n);
		if (!map->p[i])
			goto error;
	}
	return map;
error:
	isl_map_free(map);
	return NULL;
}

/* Eliminate the specified n dimensions starting at first from the
 * constraints, without removing the dimensions from the space.
 * If the set is rational, the dimensions are eliminated using Fourier-Motzkin.
 */
__isl_give isl_set *isl_set_eliminate(__isl_take isl_set *set,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return set_from_map(isl_map_eliminate(set_to_map(set), type, first, n));
}

/* Eliminate the specified n dimensions starting at first from the
 * constraints, without removing the dimensions from the space.
 * If the set is rational, the dimensions are eliminated using Fourier-Motzkin.
 */
__isl_give isl_set *isl_set_eliminate_dims(__isl_take isl_set *set,
	unsigned first, unsigned n)
{
	return isl_set_eliminate(set, isl_dim_set, first, n);
}

__isl_give isl_basic_map *isl_basic_map_remove_divs(
	__isl_take isl_basic_map *bmap)
{
	isl_size v_div;

	v_div = isl_basic_map_var_offset(bmap, isl_dim_div);
	if (v_div < 0)
		return isl_basic_map_free(bmap);
	bmap = isl_basic_map_eliminate_vars(bmap, v_div, bmap->n_div);
	if (!bmap)
		return NULL;
	bmap->n_div = 0;
	return isl_basic_map_finalize(bmap);
}

__isl_give isl_basic_set *isl_basic_set_remove_divs(
	__isl_take isl_basic_set *bset)
{
	return bset_from_bmap(isl_basic_map_remove_divs(bset_to_bmap(bset)));
}

__isl_give isl_map *isl_map_remove_divs(__isl_take isl_map *map)
{
	int i;

	if (!map)
		return NULL;
	if (map->n == 0)
		return map;

	map = isl_map_cow(map);
	if (!map)
		return NULL;
	
	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_remove_divs(map->p[i]);
		if (!map->p[i])
			goto error;
	}
	return map;
error:
	isl_map_free(map);
	return NULL;
}

__isl_give isl_set *isl_set_remove_divs(__isl_take isl_set *set)
{
	return isl_map_remove_divs(set);
}

__isl_give isl_basic_map *isl_basic_map_remove_dims(
	__isl_take isl_basic_map *bmap, enum isl_dim_type type,
	unsigned first, unsigned n)
{
	if (isl_basic_map_check_range(bmap, type, first, n) < 0)
		return isl_basic_map_free(bmap);
	if (n == 0 && !isl_space_is_named_or_nested(bmap->dim, type))
		return bmap;
	bmap = isl_basic_map_eliminate_vars(bmap,
			isl_basic_map_offset(bmap, type) - 1 + first, n);
	if (!bmap)
		return bmap;
	if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_EMPTY) && type == isl_dim_div)
		return bmap;
	bmap = isl_basic_map_drop(bmap, type, first, n);
	return bmap;
}

/* Return true if the definition of the given div (recursively) involves
 * any of the given variables.
 */
static isl_bool div_involves_vars(__isl_keep isl_basic_map *bmap, int div,
	unsigned first, unsigned n)
{
	int i;
	unsigned div_offset = isl_basic_map_offset(bmap, isl_dim_div);

	if (isl_int_is_zero(bmap->div[div][0]))
		return isl_bool_false;
	if (isl_seq_first_non_zero(bmap->div[div] + 1 + first, n) >= 0)
		return isl_bool_true;

	for (i = bmap->n_div - 1; i >= 0; --i) {
		isl_bool involves;

		if (isl_int_is_zero(bmap->div[div][1 + div_offset + i]))
			continue;
		involves = div_involves_vars(bmap, i, first, n);
		if (involves < 0 || involves)
			return involves;
	}

	return isl_bool_false;
}

/* Try and add a lower and/or upper bound on "div" to "bmap"
 * based on inequality "i".
 * "total" is the total number of variables (excluding the divs).
 * "v" is a temporary object that can be used during the calculations.
 * If "lb" is set, then a lower bound should be constructed.
 * If "ub" is set, then an upper bound should be constructed.
 *
 * The calling function has already checked that the inequality does not
 * reference "div", but we still need to check that the inequality is
 * of the right form.  We'll consider the case where we want to construct
 * a lower bound.  The construction of upper bounds is similar.
 *
 * Let "div" be of the form
 *
 *	q = floor((a + f(x))/d)
 *
 * We essentially check if constraint "i" is of the form
 *
 *	b + f(x) >= 0
 *
 * so that we can use it to derive a lower bound on "div".
 * However, we allow a slightly more general form
 *
 *	b + g(x) >= 0
 *
 * with the condition that the coefficients of g(x) - f(x) are all
 * divisible by d.
 * Rewriting this constraint as
 *
 *	0 >= -b - g(x)
 *
 * adding a + f(x) to both sides and dividing by d, we obtain
 *
 *	(a + f(x))/d >= (a-b)/d + (f(x)-g(x))/d
 *
 * Taking the floor on both sides, we obtain
 *
 *	q >= floor((a-b)/d) + (f(x)-g(x))/d
 *
 * or
 *
 *	(g(x)-f(x))/d + ceil((b-a)/d) + q >= 0
 *
 * In the case of an upper bound, we construct the constraint
 *
 *	(g(x)+f(x))/d + floor((b+a)/d) - q >= 0
 *
 */
static __isl_give isl_basic_map *insert_bounds_on_div_from_ineq(
	__isl_take isl_basic_map *bmap, int div, int i,
	unsigned total, isl_int v, int lb, int ub)
{
	int j;

	for (j = 0; (lb || ub) && j < total + bmap->n_div; ++j) {
		if (lb) {
			isl_int_sub(v, bmap->ineq[i][1 + j],
					bmap->div[div][1 + 1 + j]);
			lb = isl_int_is_divisible_by(v, bmap->div[div][0]);
		}
		if (ub) {
			isl_int_add(v, bmap->ineq[i][1 + j],
					bmap->div[div][1 + 1 + j]);
			ub = isl_int_is_divisible_by(v, bmap->div[div][0]);
		}
	}
	if (!lb && !ub)
		return bmap;

	bmap = isl_basic_map_cow(bmap);
	bmap = isl_basic_map_extend_constraints(bmap, 0, lb + ub);
	if (lb) {
		int k = isl_basic_map_alloc_inequality(bmap);
		if (k < 0)
			goto error;
		for (j = 0; j < 1 + total + bmap->n_div; ++j) {
			isl_int_sub(bmap->ineq[k][j], bmap->ineq[i][j],
					bmap->div[div][1 + j]);
			isl_int_cdiv_q(bmap->ineq[k][j],
					bmap->ineq[k][j], bmap->div[div][0]);
		}
		isl_int_set_si(bmap->ineq[k][1 + total + div], 1);
	}
	if (ub) {
		int k = isl_basic_map_alloc_inequality(bmap);
		if (k < 0)
			goto error;
		for (j = 0; j < 1 + total + bmap->n_div; ++j) {
			isl_int_add(bmap->ineq[k][j], bmap->ineq[i][j],
					bmap->div[div][1 + j]);
			isl_int_fdiv_q(bmap->ineq[k][j],
					bmap->ineq[k][j], bmap->div[div][0]);
		}
		isl_int_set_si(bmap->ineq[k][1 + total + div], -1);
	}

	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

/* This function is called right before "div" is eliminated from "bmap"
 * using Fourier-Motzkin.
 * Look through the constraints of "bmap" for constraints on the argument
 * of the integer division and use them to construct constraints on the
 * integer division itself.  These constraints can then be combined
 * during the Fourier-Motzkin elimination.
 * Note that it is only useful to introduce lower bounds on "div"
 * if "bmap" already contains upper bounds on "div" as the newly
 * introduce lower bounds can then be combined with the pre-existing
 * upper bounds.  Similarly for upper bounds.
 * We therefore first check if "bmap" contains any lower and/or upper bounds
 * on "div".
 *
 * It is interesting to note that the introduction of these constraints
 * can indeed lead to more accurate results, even when compared to
 * deriving constraints on the argument of "div" from constraints on "div".
 * Consider, for example, the set
 *
 *	{ [i,j,k] : 3 + i + 2j >= 0 and 2 * [(i+2j)/4] <= k }
 *
 * The second constraint can be rewritten as
 *
 *	2 * [(-i-2j+3)/4] + k >= 0
 *
 * from which we can derive
 *
 *	-i - 2j + 3 >= -2k
 *
 * or
 *
 *	i + 2j <= 3 + 2k
 *
 * Combined with the first constraint, we obtain
 *
 *	-3 <= 3 + 2k	or	k >= -3
 *
 * If, on the other hand we derive a constraint on [(i+2j)/4] from
 * the first constraint, we obtain
 *
 *	[(i + 2j)/4] >= [-3/4] = -1
 *
 * Combining this constraint with the second constraint, we obtain
 *
 *	k >= -2
 */
static __isl_give isl_basic_map *insert_bounds_on_div(
	__isl_take isl_basic_map *bmap, int div)
{
	int i;
	int check_lb, check_ub;
	isl_int v;
	isl_size v_div;

	if (!bmap)
		return NULL;

	if (isl_int_is_zero(bmap->div[div][0]))
		return bmap;

	v_div = isl_basic_map_var_offset(bmap, isl_dim_div);
	if (v_div < 0)
		return isl_basic_map_free(bmap);

	check_lb = 0;
	check_ub = 0;
	for (i = 0; (!check_lb || !check_ub) && i < bmap->n_ineq; ++i) {
		int s = isl_int_sgn(bmap->ineq[i][1 + v_div + div]);
		if (s > 0)
			check_ub = 1;
		if (s < 0)
			check_lb = 1;
	}

	if (!check_lb && !check_ub)
		return bmap;

	isl_int_init(v);

	for (i = 0; bmap && i < bmap->n_ineq; ++i) {
		if (!isl_int_is_zero(bmap->ineq[i][1 + v_div + div]))
			continue;

		bmap = insert_bounds_on_div_from_ineq(bmap, div, i, v_div, v,
							check_lb, check_ub);
	}

	isl_int_clear(v);

	return bmap;
}

/* Remove all divs (recursively) involving any of the given dimensions
 * in their definitions.
 */
__isl_give isl_basic_map *isl_basic_map_remove_divs_involving_dims(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;

	if (isl_basic_map_check_range(bmap, type, first, n) < 0)
		return isl_basic_map_free(bmap);
	first += isl_basic_map_offset(bmap, type);

	for (i = bmap->n_div - 1; i >= 0; --i) {
		isl_bool involves;

		involves = div_involves_vars(bmap, i, first, n);
		if (involves < 0)
			return isl_basic_map_free(bmap);
		if (!involves)
			continue;
		bmap = insert_bounds_on_div(bmap, i);
		bmap = isl_basic_map_remove_dims(bmap, isl_dim_div, i, 1);
		if (!bmap)
			return NULL;
		i = bmap->n_div;
	}

	return bmap;
}

__isl_give isl_basic_set *isl_basic_set_remove_divs_involving_dims(
	__isl_take isl_basic_set *bset,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return isl_basic_map_remove_divs_involving_dims(bset, type, first, n);
}

__isl_give isl_map *isl_map_remove_divs_involving_dims(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;

	if (!map)
		return NULL;
	if (map->n == 0)
		return map;

	map = isl_map_cow(map);
	if (!map)
		return NULL;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_remove_divs_involving_dims(map->p[i],
								type, first, n);
		if (!map->p[i])
			goto error;
	}
	return map;
error:
	isl_map_free(map);
	return NULL;
}

__isl_give isl_set *isl_set_remove_divs_involving_dims(__isl_take isl_set *set,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return set_from_map(isl_map_remove_divs_involving_dims(set_to_map(set),
							      type, first, n));
}

/* Does the description of "bmap" depend on the specified dimensions?
 * We also check whether the dimensions appear in any of the div definitions.
 * In principle there is no need for this check.  If the dimensions appear
 * in a div definition, they also appear in the defining constraints of that
 * div.
 */
isl_bool isl_basic_map_involves_dims(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;

	if (isl_basic_map_check_range(bmap, type, first, n) < 0)
		return isl_bool_error;

	first += isl_basic_map_offset(bmap, type);
	for (i = 0; i < bmap->n_eq; ++i)
		if (isl_seq_first_non_zero(bmap->eq[i] + first, n) >= 0)
			return isl_bool_true;
	for (i = 0; i < bmap->n_ineq; ++i)
		if (isl_seq_first_non_zero(bmap->ineq[i] + first, n) >= 0)
			return isl_bool_true;
	for (i = 0; i < bmap->n_div; ++i) {
		if (isl_int_is_zero(bmap->div[i][0]))
			continue;
		if (isl_seq_first_non_zero(bmap->div[i] + 1 + first, n) >= 0)
			return isl_bool_true;
	}

	return isl_bool_false;
}

isl_bool isl_map_involves_dims(__isl_keep isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;

	if (isl_map_check_range(map, type, first, n) < 0)
		return isl_bool_error;

	for (i = 0; i < map->n; ++i) {
		isl_bool involves = isl_basic_map_involves_dims(map->p[i],
							    type, first, n);
		if (involves < 0 || involves)
			return involves;
	}

	return isl_bool_false;
}

isl_bool isl_basic_set_involves_dims(__isl_keep isl_basic_set *bset,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return isl_basic_map_involves_dims(bset, type, first, n);
}

isl_bool isl_set_involves_dims(__isl_keep isl_set *set,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return isl_map_involves_dims(set, type, first, n);
}

/* Does "bset" involve any local variables, i.e., integer divisions?
 */
static isl_bool isl_basic_set_involves_locals(__isl_keep isl_basic_set *bset)
{
	isl_size n;

	n = isl_basic_set_dim(bset, isl_dim_div);
	if (n < 0)
		return isl_bool_error;
	return isl_bool_ok(n > 0);
}

/* isl_set_every_basic_set callback that checks whether "bset"
 * is free of local variables.
 */
static isl_bool basic_set_no_locals(__isl_keep isl_basic_set *bset, void *user)
{
	return isl_bool_not(isl_basic_set_involves_locals(bset));
}

/* Does "set" involve any local variables, i.e., integer divisions?
 */
isl_bool isl_set_involves_locals(__isl_keep isl_set *set)
{
	isl_bool no_locals;

	no_locals = isl_set_every_basic_set(set, &basic_set_no_locals, NULL);
	return isl_bool_not(no_locals);
}

/* Drop all constraints in bmap that involve any of the dimensions
 * first to first+n-1.
 * This function only performs the actual removal of constraints.
 *
 * This function should not call finalize since it is used by
 * remove_redundant_divs, which in turn is called by isl_basic_map_finalize.
 */
__isl_give isl_basic_map *isl_basic_map_drop_constraints_involving(
	__isl_take isl_basic_map *bmap, unsigned first, unsigned n)
{
	int i;

	if (n == 0)
		return bmap;

	bmap = isl_basic_map_cow(bmap);

	if (!bmap)
		return NULL;

	for (i = bmap->n_eq - 1; i >= 0; --i) {
		if (isl_seq_first_non_zero(bmap->eq[i] + 1 + first, n) == -1)
			continue;
		if (isl_basic_map_drop_equality(bmap, i) < 0)
			return isl_basic_map_free(bmap);
	}

	for (i = bmap->n_ineq - 1; i >= 0; --i) {
		if (isl_seq_first_non_zero(bmap->ineq[i] + 1 + first, n) == -1)
			continue;
		if (isl_basic_map_drop_inequality(bmap, i) < 0)
			return isl_basic_map_free(bmap);
	}

	return bmap;
}

/* Drop all constraints in bset that involve any of the dimensions
 * first to first+n-1.
 * This function only performs the actual removal of constraints.
 */
__isl_give isl_basic_set *isl_basic_set_drop_constraints_involving(
	__isl_take isl_basic_set *bset, unsigned first, unsigned n)
{
	return isl_basic_map_drop_constraints_involving(bset, first, n);
}

/* Drop all constraints in bmap that do not involve any of the dimensions
 * first to first + n - 1 of the given type.
 */
__isl_give isl_basic_map *isl_basic_map_drop_constraints_not_involving_dims(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;

	if (n == 0) {
		isl_space *space = isl_basic_map_get_space(bmap);
		isl_basic_map_free(bmap);
		return isl_basic_map_universe(space);
	}
	bmap = isl_basic_map_cow(bmap);
	if (!bmap)
		return NULL;

	if (isl_basic_map_check_range(bmap, type, first, n) < 0)
		return isl_basic_map_free(bmap);

	first += isl_basic_map_offset(bmap, type) - 1;

	for (i = bmap->n_eq - 1; i >= 0; --i) {
		if (isl_seq_first_non_zero(bmap->eq[i] + 1 + first, n) != -1)
			continue;
		if (isl_basic_map_drop_equality(bmap, i) < 0)
			return isl_basic_map_free(bmap);
	}

	for (i = bmap->n_ineq - 1; i >= 0; --i) {
		if (isl_seq_first_non_zero(bmap->ineq[i] + 1 + first, n) != -1)
			continue;
		if (isl_basic_map_drop_inequality(bmap, i) < 0)
			return isl_basic_map_free(bmap);
	}

	bmap = isl_basic_map_add_known_div_constraints(bmap);
	return bmap;
}

/* Drop all constraints in bset that do not involve any of the dimensions
 * first to first + n - 1 of the given type.
 */
__isl_give isl_basic_set *isl_basic_set_drop_constraints_not_involving_dims(
	__isl_take isl_basic_set *bset,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return isl_basic_map_drop_constraints_not_involving_dims(bset,
							    type, first, n);
}

/* Drop all constraints in bmap that involve any of the dimensions
 * first to first + n - 1 of the given type.
 */
__isl_give isl_basic_map *isl_basic_map_drop_constraints_involving_dims(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	if (!bmap)
		return NULL;
	if (n == 0)
		return bmap;

	if (isl_basic_map_check_range(bmap, type, first, n) < 0)
		return isl_basic_map_free(bmap);

	bmap = isl_basic_map_remove_divs_involving_dims(bmap, type, first, n);
	first += isl_basic_map_offset(bmap, type) - 1;
	bmap = isl_basic_map_drop_constraints_involving(bmap, first, n);
	bmap = isl_basic_map_add_known_div_constraints(bmap);
	return bmap;
}

/* Drop all constraints in bset that involve any of the dimensions
 * first to first + n - 1 of the given type.
 */
__isl_give isl_basic_set *isl_basic_set_drop_constraints_involving_dims(
	__isl_take isl_basic_set *bset,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return isl_basic_map_drop_constraints_involving_dims(bset,
							    type, first, n);
}

/* Drop constraints from "map" by applying "drop" to each basic map.
 */
static __isl_give isl_map *drop_constraints(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n,
	__isl_give isl_basic_map *(*drop)(__isl_take isl_basic_map *bmap,
		enum isl_dim_type type, unsigned first, unsigned n))
{
	int i;

	if (isl_map_check_range(map, type, first, n) < 0)
		return isl_map_free(map);

	map = isl_map_cow(map);
	if (!map)
		return NULL;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = drop(map->p[i], type, first, n);
		if (!map->p[i])
			return isl_map_free(map);
	}

	if (map->n > 1)
		ISL_F_CLR(map, ISL_MAP_DISJOINT);

	return map;
}

/* Drop all constraints in map that involve any of the dimensions
 * first to first + n - 1 of the given type.
 */
__isl_give isl_map *isl_map_drop_constraints_involving_dims(
	__isl_take isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	if (n == 0)
		return map;
	return drop_constraints(map, type, first, n,
				&isl_basic_map_drop_constraints_involving_dims);
}

/* Drop all constraints in "map" that do not involve any of the dimensions
 * first to first + n - 1 of the given type.
 */
__isl_give isl_map *isl_map_drop_constraints_not_involving_dims(
	__isl_take isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	if (n == 0) {
		isl_space *space = isl_map_get_space(map);
		isl_map_free(map);
		return isl_map_universe(space);
	}
	return drop_constraints(map, type, first, n,
			    &isl_basic_map_drop_constraints_not_involving_dims);
}

/* Drop all constraints in set that involve any of the dimensions
 * first to first + n - 1 of the given type.
 */
__isl_give isl_set *isl_set_drop_constraints_involving_dims(
	__isl_take isl_set *set,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return isl_map_drop_constraints_involving_dims(set, type, first, n);
}

/* Drop all constraints in "set" that do not involve any of the dimensions
 * first to first + n - 1 of the given type.
 */
__isl_give isl_set *isl_set_drop_constraints_not_involving_dims(
	__isl_take isl_set *set,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return isl_map_drop_constraints_not_involving_dims(set, type, first, n);
}

/* Does local variable "div" of "bmap" have a complete explicit representation?
 * Having a complete explicit representation requires not only
 * an explicit representation, but also that all local variables
 * that appear in this explicit representation in turn have
 * a complete explicit representation.
 */
isl_bool isl_basic_map_div_is_known(__isl_keep isl_basic_map *bmap, int div)
{
	int i;
	unsigned div_offset = isl_basic_map_offset(bmap, isl_dim_div);
	isl_bool marked;

	marked = isl_basic_map_div_is_marked_unknown(bmap, div);
	if (marked < 0 || marked)
		return isl_bool_not(marked);

	for (i = bmap->n_div - 1; i >= 0; --i) {
		isl_bool known;

		if (isl_int_is_zero(bmap->div[div][1 + div_offset + i]))
			continue;
		known = isl_basic_map_div_is_known(bmap, i);
		if (known < 0 || !known)
			return known;
	}

	return isl_bool_true;
}

/* Remove all divs that are unknown or defined in terms of unknown divs.
 */
__isl_give isl_basic_map *isl_basic_map_remove_unknown_divs(
	__isl_take isl_basic_map *bmap)
{
	int i;

	if (!bmap)
		return NULL;

	for (i = bmap->n_div - 1; i >= 0; --i) {
		if (isl_basic_map_div_is_known(bmap, i))
			continue;
		bmap = isl_basic_map_remove_dims(bmap, isl_dim_div, i, 1);
		if (!bmap)
			return NULL;
		i = bmap->n_div;
	}

	return bmap;
}

/* Remove all divs that are unknown or defined in terms of unknown divs.
 */
__isl_give isl_basic_set *isl_basic_set_remove_unknown_divs(
	__isl_take isl_basic_set *bset)
{
	return isl_basic_map_remove_unknown_divs(bset);
}

__isl_give isl_map *isl_map_remove_unknown_divs(__isl_take isl_map *map)
{
	int i;

	if (!map)
		return NULL;
	if (map->n == 0)
		return map;

	map = isl_map_cow(map);
	if (!map)
		return NULL;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_remove_unknown_divs(map->p[i]);
		if (!map->p[i])
			goto error;
	}
	return map;
error:
	isl_map_free(map);
	return NULL;
}

__isl_give isl_set *isl_set_remove_unknown_divs(__isl_take isl_set *set)
{
	return set_from_map(isl_map_remove_unknown_divs(set_to_map(set)));
}

__isl_give isl_basic_set *isl_basic_set_remove_dims(
	__isl_take isl_basic_set *bset,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	isl_basic_map *bmap = bset_to_bmap(bset);
	bmap = isl_basic_map_remove_dims(bmap, type, first, n);
	return bset_from_bmap(bmap);
}

__isl_give isl_map *isl_map_remove_dims(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;

	if (n == 0)
		return map;

	map = isl_map_cow(map);
	if (isl_map_check_range(map, type, first, n) < 0)
		return isl_map_free(map);
	
	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_eliminate_vars(map->p[i],
			isl_basic_map_offset(map->p[i], type) - 1 + first, n);
		if (!map->p[i])
			goto error;
	}
	map = isl_map_drop(map, type, first, n);
	return map;
error:
	isl_map_free(map);
	return NULL;
}

__isl_give isl_set *isl_set_remove_dims(__isl_take isl_set *bset,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return set_from_map(isl_map_remove_dims(set_to_map(bset),
						type, first, n));
}

/* Project out n inputs starting at first using Fourier-Motzkin */
__isl_give isl_map *isl_map_remove_inputs(__isl_take isl_map *map,
	unsigned first, unsigned n)
{
	return isl_map_remove_dims(map, isl_dim_in, first, n);
}

void isl_basic_set_print_internal(__isl_keep isl_basic_set *bset,
	FILE *out, int indent)
{
	isl_printer *p;

	if (!bset) {
		fprintf(out, "null basic set\n");
		return;
	}

	fprintf(out, "%*s", indent, "");
	fprintf(out, "ref: %d, nparam: %d, dim: %d, extra: %d, flags: %x\n",
			bset->ref, bset->dim->nparam, bset->dim->n_out,
			bset->extra, bset->flags);

	p = isl_printer_to_file(isl_basic_set_get_ctx(bset), out);
	p = isl_printer_set_dump(p, 1);
	p = isl_printer_set_indent(p, indent);
	p = isl_printer_start_line(p);
	p = isl_printer_print_basic_set(p, bset);
	p = isl_printer_end_line(p);
	isl_printer_free(p);
}

void isl_basic_map_print_internal(__isl_keep isl_basic_map *bmap,
	FILE *out, int indent)
{
	isl_printer *p;

	if (!bmap) {
		fprintf(out, "null basic map\n");
		return;
	}

	fprintf(out, "%*s", indent, "");
	fprintf(out, "ref: %d, nparam: %d, in: %d, out: %d, extra: %d, "
			"flags: %x, n_name: %d\n",
		bmap->ref,
		bmap->dim->nparam, bmap->dim->n_in, bmap->dim->n_out,
		bmap->extra, bmap->flags, bmap->dim->n_id);

	p = isl_printer_to_file(isl_basic_map_get_ctx(bmap), out);
	p = isl_printer_set_dump(p, 1);
	p = isl_printer_set_indent(p, indent);
	p = isl_printer_start_line(p);
	p = isl_printer_print_basic_map(p, bmap);
	p = isl_printer_end_line(p);
	isl_printer_free(p);
}

__isl_give isl_basic_map *isl_inequality_negate(__isl_take isl_basic_map *bmap,
	unsigned pos)
{
	isl_size total;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return isl_basic_map_free(bmap);
	if (pos >= bmap->n_ineq)
		isl_die(isl_basic_map_get_ctx(bmap), isl_error_invalid,
			"invalid position", return isl_basic_map_free(bmap));
	isl_seq_neg(bmap->ineq[pos], bmap->ineq[pos], 1 + total);
	isl_int_sub_ui(bmap->ineq[pos][0], bmap->ineq[pos][0], 1);
	ISL_F_CLR(bmap, ISL_BASIC_MAP_NO_REDUNDANT);
	ISL_F_CLR(bmap, ISL_BASIC_MAP_SORTED);
	return bmap;
}

__isl_give isl_set *isl_set_alloc_space(__isl_take isl_space *space, int n,
	unsigned flags)
{
	if (isl_space_check_is_set(space) < 0)
		goto error;
	return isl_map_alloc_space(space, n, flags);
error:
	isl_space_free(space);
	return NULL;
}

/* Make sure "map" has room for at least "n" more basic maps.
 */
__isl_give isl_map *isl_map_grow(__isl_take isl_map *map, int n)
{
	int i;
	struct isl_map *grown = NULL;

	if (!map)
		return NULL;
	isl_assert(map->ctx, n >= 0, goto error);
	if (map->n + n <= map->size)
		return map;
	grown = isl_map_alloc_space(isl_map_get_space(map), map->n + n, map->flags);
	if (!grown)
		goto error;
	for (i = 0; i < map->n; ++i) {
		grown->p[i] = isl_basic_map_copy(map->p[i]);
		if (!grown->p[i])
			goto error;
		grown->n++;
	}
	isl_map_free(map);
	return grown;
error:
	isl_map_free(grown);
	isl_map_free(map);
	return NULL;
}

/* Make sure "set" has room for at least "n" more basic sets.
 */
__isl_give isl_set *isl_set_grow(__isl_take isl_set *set, int n)
{
	return set_from_map(isl_map_grow(set_to_map(set), n));
}

__isl_give isl_set *isl_set_from_basic_set(__isl_take isl_basic_set *bset)
{
	return isl_map_from_basic_map(bset);
}

/* This function performs the same operation as isl_set_from_basic_set,
 * but is considered as a function on an isl_basic_set when exported.
 */
__isl_give isl_set *isl_basic_set_to_set(__isl_take isl_basic_set *bset)
{
	return isl_set_from_basic_set(bset);
}

__isl_give isl_map *isl_map_from_basic_map(__isl_take isl_basic_map *bmap)
{
	struct isl_map *map;

	if (!bmap)
		return NULL;

	map = isl_map_alloc_space(isl_space_copy(bmap->dim), 1, ISL_MAP_DISJOINT);
	return isl_map_add_basic_map(map, bmap);
}

__isl_give isl_set *isl_set_add_basic_set(__isl_take isl_set *set,
						__isl_take isl_basic_set *bset)
{
	return set_from_map(isl_map_add_basic_map(set_to_map(set),
						bset_to_bmap(bset)));
}

__isl_null isl_set *isl_set_free(__isl_take isl_set *set)
{
	return isl_map_free(set);
}

void isl_set_print_internal(__isl_keep isl_set *set, FILE *out, int indent)
{
	int i;

	if (!set) {
		fprintf(out, "null set\n");
		return;
	}

	fprintf(out, "%*s", indent, "");
	fprintf(out, "ref: %d, n: %d, nparam: %d, dim: %d, flags: %x\n",
			set->ref, set->n, set->dim->nparam, set->dim->n_out,
			set->flags);
	for (i = 0; i < set->n; ++i) {
		fprintf(out, "%*s", indent, "");
		fprintf(out, "basic set %d:\n", i);
		isl_basic_set_print_internal(set->p[i], out, indent+4);
	}
}

void isl_map_print_internal(__isl_keep isl_map *map, FILE *out, int indent)
{
	int i;

	if (!map) {
		fprintf(out, "null map\n");
		return;
	}

	fprintf(out, "%*s", indent, "");
	fprintf(out, "ref: %d, n: %d, nparam: %d, in: %d, out: %d, "
		     "flags: %x, n_name: %d\n",
			map->ref, map->n, map->dim->nparam, map->dim->n_in,
			map->dim->n_out, map->flags, map->dim->n_id);
	for (i = 0; i < map->n; ++i) {
		fprintf(out, "%*s", indent, "");
		fprintf(out, "basic map %d:\n", i);
		isl_basic_map_print_internal(map->p[i], out, indent+4);
	}
}

/* Check that the space of "bset" is the same as that of the domain of "bmap".
 */
static isl_stat isl_basic_map_check_compatible_domain(
	__isl_keep isl_basic_map *bmap, __isl_keep isl_basic_set *bset)
{
	isl_bool ok;

	ok = isl_basic_map_compatible_domain(bmap, bset);
	if (ok < 0)
		return isl_stat_error;
	if (!ok)
		isl_die(isl_basic_set_get_ctx(bset), isl_error_invalid,
			"incompatible spaces", return isl_stat_error);

	return isl_stat_ok;
}

__isl_give isl_basic_map *isl_basic_map_intersect_domain(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *bset)
{
	struct isl_basic_map *bmap_domain;
	isl_size dim;

	if (isl_basic_map_check_equal_params(bmap, bset_to_bmap(bset)) < 0)
		goto error;

	dim = isl_basic_set_dim(bset, isl_dim_set);
	if (dim < 0)
		goto error;
	if (dim != 0 &&
	    isl_basic_map_check_compatible_domain(bmap, bset) < 0)
		goto error;

	bmap = isl_basic_map_cow(bmap);
	if (!bmap)
		goto error;
	bmap = isl_basic_map_extend(bmap,
			bset->n_div, bset->n_eq, bset->n_ineq);
	bmap_domain = isl_basic_map_from_domain(bset);
	bmap = add_constraints(bmap, bmap_domain, 0, 0);

	bmap = isl_basic_map_simplify(bmap);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap);
	isl_basic_set_free(bset);
	return NULL;
}

/* Check that the space of "bset" is the same as that of the range of "bmap".
 */
static isl_stat isl_basic_map_check_compatible_range(
	__isl_keep isl_basic_map *bmap, __isl_keep isl_basic_set *bset)
{
	isl_bool ok;

	ok = isl_basic_map_compatible_range(bmap, bset);
	if (ok < 0)
		return isl_stat_error;
	if (!ok)
		isl_die(isl_basic_set_get_ctx(bset), isl_error_invalid,
			"incompatible spaces", return isl_stat_error);

	return isl_stat_ok;
}

__isl_give isl_basic_map *isl_basic_map_intersect_range(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *bset)
{
	struct isl_basic_map *bmap_range;
	isl_size dim;

	if (isl_basic_map_check_equal_params(bmap, bset_to_bmap(bset)) < 0)
		goto error;

	dim = isl_basic_set_dim(bset, isl_dim_set);
	if (dim < 0)
		goto error;
	if (dim != 0 && isl_basic_map_check_compatible_range(bmap, bset) < 0)
		goto error;

	if (isl_basic_set_plain_is_universe(bset)) {
		isl_basic_set_free(bset);
		return bmap;
	}

	bmap = isl_basic_map_cow(bmap);
	if (!bmap)
		goto error;
	bmap = isl_basic_map_extend(bmap,
			bset->n_div, bset->n_eq, bset->n_ineq);
	bmap_range = bset_to_bmap(bset);
	bmap = add_constraints(bmap, bmap_range, 0, 0);

	bmap = isl_basic_map_simplify(bmap);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap);
	isl_basic_set_free(bset);
	return NULL;
}

isl_bool isl_basic_map_contains(__isl_keep isl_basic_map *bmap,
	__isl_keep isl_vec *vec)
{
	int i;
	isl_size total;
	isl_int s;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0 || !vec)
		return isl_bool_error;

	if (1 + total != vec->size)
		return isl_bool_false;

	isl_int_init(s);

	for (i = 0; i < bmap->n_eq; ++i) {
		isl_seq_inner_product(vec->el, bmap->eq[i], 1 + total, &s);
		if (!isl_int_is_zero(s)) {
			isl_int_clear(s);
			return isl_bool_false;
		}
	}

	for (i = 0; i < bmap->n_ineq; ++i) {
		isl_seq_inner_product(vec->el, bmap->ineq[i], 1 + total, &s);
		if (isl_int_is_neg(s)) {
			isl_int_clear(s);
			return isl_bool_false;
		}
	}

	isl_int_clear(s);

	return isl_bool_true;
}

isl_bool isl_basic_set_contains(__isl_keep isl_basic_set *bset,
	__isl_keep isl_vec *vec)
{
	return isl_basic_map_contains(bset_to_bmap(bset), vec);
}

__isl_give isl_basic_map *isl_basic_map_intersect(
	__isl_take isl_basic_map *bmap1, __isl_take isl_basic_map *bmap2)
{
	struct isl_vec *sample = NULL;
	isl_space *space1, *space2;
	isl_size dim1, dim2, nparam1, nparam2;

	if (isl_basic_map_check_equal_params(bmap1, bmap2) < 0)
		goto error;
	space1 = isl_basic_map_peek_space(bmap1);
	space2 = isl_basic_map_peek_space(bmap2);
	dim1 = isl_space_dim(space1, isl_dim_all);
	dim2 = isl_space_dim(space2, isl_dim_all);
	nparam1 = isl_space_dim(space1, isl_dim_param);
	nparam2 = isl_space_dim(space2, isl_dim_param);
	if (dim1 < 0 || dim2 < 0 || nparam1 < 0 || nparam2 < 0)
		goto error;
	if (dim1 == nparam1 && dim2 != nparam2)
		return isl_basic_map_intersect(bmap2, bmap1);

	if (dim2 != nparam2 &&
	    isl_basic_map_check_equal_space(bmap1, bmap2) < 0)
		goto error;

	if (isl_basic_map_plain_is_empty(bmap1)) {
		isl_basic_map_free(bmap2);
		return bmap1;
	}
	if (isl_basic_map_plain_is_empty(bmap2)) {
		isl_basic_map_free(bmap1);
		return bmap2;
	}

	if (bmap1->sample &&
	    isl_basic_map_contains(bmap1, bmap1->sample) > 0 &&
	    isl_basic_map_contains(bmap2, bmap1->sample) > 0)
		sample = isl_vec_copy(bmap1->sample);
	else if (bmap2->sample &&
	    isl_basic_map_contains(bmap1, bmap2->sample) > 0 &&
	    isl_basic_map_contains(bmap2, bmap2->sample) > 0)
		sample = isl_vec_copy(bmap2->sample);

	bmap1 = isl_basic_map_cow(bmap1);
	if (!bmap1)
		goto error;
	bmap1 = isl_basic_map_extend(bmap1,
			bmap2->n_div, bmap2->n_eq, bmap2->n_ineq);
	bmap1 = add_constraints(bmap1, bmap2, 0, 0);

	if (!bmap1)
		isl_vec_free(sample);
	else if (sample) {
		isl_vec_free(bmap1->sample);
		bmap1->sample = sample;
	}

	bmap1 = isl_basic_map_simplify(bmap1);
	return isl_basic_map_finalize(bmap1);
error:
	if (sample)
		isl_vec_free(sample);
	isl_basic_map_free(bmap1);
	isl_basic_map_free(bmap2);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_intersect(
	__isl_take isl_basic_set *bset1, __isl_take isl_basic_set *bset2)
{
	return bset_from_bmap(isl_basic_map_intersect(bset_to_bmap(bset1),
							bset_to_bmap(bset2)));
}

__isl_give isl_basic_set *isl_basic_set_intersect_params(
	__isl_take isl_basic_set *bset1, __isl_take isl_basic_set *bset2)
{
	return isl_basic_set_intersect(bset1, bset2);
}

/* Does "map" consist of a single disjunct, without any local variables?
 */
static isl_bool is_convex_no_locals(__isl_keep isl_map *map)
{
	isl_size n_div;

	if (!map)
		return isl_bool_error;
	if (map->n != 1)
		return isl_bool_false;
	n_div = isl_basic_map_dim(map->p[0], isl_dim_div);
	if (n_div < 0)
		return isl_bool_error;
	if (n_div != 0)
		return isl_bool_false;
	return isl_bool_true;
}

/* Check that "map" consists of a single disjunct, without any local variables.
 */
static isl_stat check_convex_no_locals(__isl_keep isl_map *map)
{
	isl_bool ok;

	ok = is_convex_no_locals(map);
	if (ok < 0)
		return isl_stat_error;
	if (ok)
		return isl_stat_ok;

	isl_die(isl_map_get_ctx(map), isl_error_internal,
		"unexpectedly not convex or involving local variables",
		return isl_stat_error);
}

/* Special case of isl_map_intersect, where both map1 and map2
 * are convex, without any divs and such that either map1 or map2
 * contains a single constraint.  This constraint is then simply
 * added to the other map.
 */
static __isl_give isl_map *map_intersect_add_constraint(
	__isl_take isl_map *map1, __isl_take isl_map *map2)
{
	if (check_convex_no_locals(map1) < 0 ||
	    check_convex_no_locals(map2) < 0)
		goto error;

	if (map2->p[0]->n_eq + map2->p[0]->n_ineq != 1)
		return isl_map_intersect(map2, map1);

	map1 = isl_map_cow(map1);
	if (!map1)
		goto error;
	if (isl_map_plain_is_empty(map1)) {
		isl_map_free(map2);
		return map1;
	}
	if (map2->p[0]->n_eq == 1)
		map1->p[0] = isl_basic_map_add_eq(map1->p[0], map2->p[0]->eq[0]);
	else
		map1->p[0] = isl_basic_map_add_ineq(map1->p[0],
							map2->p[0]->ineq[0]);

	map1->p[0] = isl_basic_map_simplify(map1->p[0]);
	map1->p[0] = isl_basic_map_finalize(map1->p[0]);
	if (!map1->p[0])
		goto error;

	if (isl_basic_map_plain_is_empty(map1->p[0])) {
		isl_basic_map_free(map1->p[0]);
		map1->n = 0;
	}

	isl_map_free(map2);

	map1 = isl_map_unmark_normalized(map1);
	return map1;
error:
	isl_map_free(map1);
	isl_map_free(map2);
	return NULL;
}

/* map2 may be either a parameter domain or a map living in the same
 * space as map1.
 */
static __isl_give isl_map *map_intersect_internal(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	unsigned flags = 0;
	isl_bool equal;
	isl_map *result;
	int i, j;
	isl_size dim2, nparam2;

	if (!map1 || !map2)
		goto error;

	if ((isl_map_plain_is_empty(map1) ||
	     isl_map_plain_is_universe(map2)) &&
	    isl_space_is_equal(map1->dim, map2->dim)) {
		isl_map_free(map2);
		return map1;
	}
	if ((isl_map_plain_is_empty(map2) ||
	     isl_map_plain_is_universe(map1)) &&
	    isl_space_is_equal(map1->dim, map2->dim)) {
		isl_map_free(map1);
		return map2;
	}

	if (is_convex_no_locals(map1) == isl_bool_true &&
	    is_convex_no_locals(map2) == isl_bool_true &&
	    isl_space_is_equal(map1->dim, map2->dim) &&
	    (map1->p[0]->n_eq + map1->p[0]->n_ineq == 1 ||
	     map2->p[0]->n_eq + map2->p[0]->n_ineq == 1))
		return map_intersect_add_constraint(map1, map2);

	equal = isl_map_plain_is_equal(map1, map2);
	if (equal < 0)
		goto error;
	if (equal) {
		isl_map_free(map2);
		return map1;
	}

	dim2 = isl_map_dim(map2, isl_dim_all);
	nparam2 = isl_map_dim(map2, isl_dim_param);
	if (dim2 < 0 || nparam2 < 0)
		goto error;
	if (dim2 != nparam2)
		isl_assert(map1->ctx,
			    isl_space_is_equal(map1->dim, map2->dim), goto error);

	if (ISL_F_ISSET(map1, ISL_MAP_DISJOINT) &&
	    ISL_F_ISSET(map2, ISL_MAP_DISJOINT))
		ISL_FL_SET(flags, ISL_MAP_DISJOINT);

	result = isl_map_alloc_space(isl_space_copy(map1->dim),
				map1->n * map2->n, flags);
	if (!result)
		goto error;
	for (i = 0; i < map1->n; ++i)
		for (j = 0; j < map2->n; ++j) {
			struct isl_basic_map *part;
			part = isl_basic_map_intersect(
				    isl_basic_map_copy(map1->p[i]),
				    isl_basic_map_copy(map2->p[j]));
			if (isl_basic_map_is_empty(part) < 0)
				part = isl_basic_map_free(part);
			result = isl_map_add_basic_map(result, part);
			if (!result)
				goto error;
		}
	isl_map_free(map1);
	isl_map_free(map2);
	return result;
error:
	isl_map_free(map1);
	isl_map_free(map2);
	return NULL;
}

static __isl_give isl_map *map_intersect(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	if (isl_map_check_equal_space(map1, map2) < 0)
		goto error;
	return map_intersect_internal(map1, map2);
error:
	isl_map_free(map1);
	isl_map_free(map2);
	return NULL;
}

__isl_give isl_map *isl_map_intersect(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	isl_map_align_params_bin(&map1, &map2);
	return map_intersect(map1, map2);
}

__isl_give isl_set *isl_set_intersect(__isl_take isl_set *set1,
	__isl_take isl_set *set2)
{
	return set_from_map(isl_map_intersect(set_to_map(set1),
					      set_to_map(set2)));
}

/* map_intersect_internal accepts intersections
 * with parameter domains, so we can just call that function.
 */
__isl_give isl_map *isl_map_intersect_params(__isl_take isl_map *map,
	__isl_take isl_set *params)
{
	isl_map_align_params_set(&map, &params);
	return map_intersect_internal(map, params);
}

__isl_give isl_set *isl_set_intersect_params(__isl_take isl_set *set,
		__isl_take isl_set *params)
{
	return isl_map_intersect_params(set, params);
}

__isl_give isl_basic_map *isl_basic_map_reverse(__isl_take isl_basic_map *bmap)
{
	isl_space *space;
	unsigned pos;
	isl_size n1, n2;

	if (!bmap)
		return NULL;
	bmap = isl_basic_map_cow(bmap);
	if (!bmap)
		return NULL;
	space = isl_space_reverse(isl_space_copy(bmap->dim));
	pos = isl_basic_map_offset(bmap, isl_dim_in);
	n1 = isl_basic_map_dim(bmap, isl_dim_in);
	n2 = isl_basic_map_dim(bmap, isl_dim_out);
	if (n1 < 0 || n2 < 0)
		bmap = isl_basic_map_free(bmap);
	bmap = isl_basic_map_swap_vars(bmap, pos, n1, n2);
	return isl_basic_map_reset_space(bmap, space);
}

/* Given a basic map A -> (B -> C), return the corresponding basic map
 * A -> (C -> B).
 */
static __isl_give isl_basic_map *isl_basic_map_range_reverse(
	__isl_take isl_basic_map *bmap)
{
	isl_space *space;
	isl_size offset, n1, n2;

	space = isl_basic_map_peek_space(bmap);
	if (isl_space_check_range_is_wrapping(space) < 0)
		return isl_basic_map_free(bmap);
	offset = isl_basic_map_var_offset(bmap, isl_dim_out);
	n1 = isl_space_wrapped_dim(space, isl_dim_out, isl_dim_in);
	n2 = isl_space_wrapped_dim(space, isl_dim_out, isl_dim_out);
	if (offset < 0 || n1 < 0 || n2 < 0)
		return isl_basic_map_free(bmap);

	bmap = isl_basic_map_swap_vars(bmap, 1 + offset, n1, n2);

	space = isl_basic_map_take_space(bmap);
	space = isl_space_range_reverse(space);
	bmap = isl_basic_map_restore_space(bmap, space);

	return bmap;
}

static __isl_give isl_basic_map *basic_map_space_reset(
	__isl_take isl_basic_map *bmap, enum isl_dim_type type)
{
	isl_space *space;

	if (!bmap)
		return NULL;
	if (!isl_space_is_named_or_nested(bmap->dim, type))
		return bmap;

	space = isl_basic_map_get_space(bmap);
	space = isl_space_reset(space, type);
	bmap = isl_basic_map_reset_space(bmap, space);
	return bmap;
}

__isl_give isl_basic_map *isl_basic_map_insert_dims(
	__isl_take isl_basic_map *bmap, enum isl_dim_type type,
	unsigned pos, unsigned n)
{
	isl_bool rational, is_empty;
	isl_space *res_space;
	struct isl_basic_map *res;
	struct isl_dim_map *dim_map;
	isl_size total;
	unsigned off;
	enum isl_dim_type t;

	if (n == 0)
		return basic_map_space_reset(bmap, type);

	is_empty = isl_basic_map_plain_is_empty(bmap);
	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (is_empty < 0 || total < 0)
		return isl_basic_map_free(bmap);
	res_space = isl_space_insert_dims(isl_basic_map_get_space(bmap),
					type, pos, n);
	if (!res_space)
		return isl_basic_map_free(bmap);
	if (is_empty) {
		isl_basic_map_free(bmap);
		return isl_basic_map_empty(res_space);
	}

	dim_map = isl_dim_map_alloc(bmap->ctx, total + n);
	off = 0;
	for (t = isl_dim_param; t <= isl_dim_out; ++t) {
		isl_size dim;

		if (t != type) {
			isl_dim_map_dim(dim_map, bmap->dim, t, off);
		} else {
			isl_size size = isl_basic_map_dim(bmap, t);
			if (size < 0)
				dim_map = isl_dim_map_free(dim_map);
			isl_dim_map_dim_range(dim_map, bmap->dim, t,
						0, pos, off);
			isl_dim_map_dim_range(dim_map, bmap->dim, t,
						pos, size - pos, off + pos + n);
		}
		dim = isl_space_dim(res_space, t);
		if (dim < 0)
			dim_map = isl_dim_map_free(dim_map);
		off += dim;
	}
	isl_dim_map_div(dim_map, bmap, off);

	res = isl_basic_map_alloc_space(res_space,
			bmap->n_div, bmap->n_eq, bmap->n_ineq);
	rational = isl_basic_map_is_rational(bmap);
	if (rational < 0)
		res = isl_basic_map_free(res);
	if (rational)
		res = isl_basic_map_set_rational(res);
	res = isl_basic_map_add_constraints_dim_map(res, bmap, dim_map);
	return isl_basic_map_finalize(res);
}

__isl_give isl_basic_set *isl_basic_set_insert_dims(
	__isl_take isl_basic_set *bset,
	enum isl_dim_type type, unsigned pos, unsigned n)
{
	return isl_basic_map_insert_dims(bset, type, pos, n);
}

__isl_give isl_basic_map *isl_basic_map_add_dims(__isl_take isl_basic_map *bmap,
		enum isl_dim_type type, unsigned n)
{
	isl_size dim;

	dim = isl_basic_map_dim(bmap, type);
	if (dim < 0)
		return isl_basic_map_free(bmap);
	return isl_basic_map_insert_dims(bmap, type, dim, n);
}

__isl_give isl_basic_set *isl_basic_set_add_dims(__isl_take isl_basic_set *bset,
		enum isl_dim_type type, unsigned n)
{
	if (!bset)
		return NULL;
	isl_assert(bset->ctx, type != isl_dim_in, goto error);
	return isl_basic_map_add_dims(bset, type, n);
error:
	isl_basic_set_free(bset);
	return NULL;
}

static __isl_give isl_map *map_space_reset(__isl_take isl_map *map,
	enum isl_dim_type type)
{
	isl_space *space;

	if (!map || !isl_space_is_named_or_nested(map->dim, type))
		return map;

	space = isl_map_get_space(map);
	space = isl_space_reset(space, type);
	map = isl_map_reset_space(map, space);
	return map;
}

__isl_give isl_map *isl_map_insert_dims(__isl_take isl_map *map,
		enum isl_dim_type type, unsigned pos, unsigned n)
{
	int i;
	isl_space *space;

	if (n == 0)
		return map_space_reset(map, type);

	map = isl_map_cow(map);
	if (!map)
		return NULL;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_insert_dims(map->p[i], type, pos, n);
		if (!map->p[i])
			goto error;
	}

	space = isl_map_take_space(map);
	space = isl_space_insert_dims(space, type, pos, n);
	map = isl_map_restore_space(map, space);

	return map;
error:
	isl_map_free(map);
	return NULL;
}

__isl_give isl_set *isl_set_insert_dims(__isl_take isl_set *set,
		enum isl_dim_type type, unsigned pos, unsigned n)
{
	return isl_map_insert_dims(set, type, pos, n);
}

__isl_give isl_map *isl_map_add_dims(__isl_take isl_map *map,
		enum isl_dim_type type, unsigned n)
{
	isl_size dim;

	dim = isl_map_dim(map, type);
	if (dim < 0)
		return isl_map_free(map);
	return isl_map_insert_dims(map, type, dim, n);
}

__isl_give isl_set *isl_set_add_dims(__isl_take isl_set *set,
		enum isl_dim_type type, unsigned n)
{
	if (!set)
		return NULL;
	isl_assert(set->ctx, type != isl_dim_in, goto error);
	return set_from_map(isl_map_add_dims(set_to_map(set), type, n));
error:
	isl_set_free(set);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_move_dims(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n)
{
	isl_space *space;
	struct isl_dim_map *dim_map;
	struct isl_basic_map *res;
	enum isl_dim_type t;
	isl_size total;
	unsigned off;

	if (!bmap)
		return NULL;
	if (n == 0) {
		bmap = isl_basic_map_reset(bmap, src_type);
		bmap = isl_basic_map_reset(bmap, dst_type);
		return bmap;
	}

	if (isl_basic_map_check_range(bmap, src_type, src_pos, n) < 0)
		return isl_basic_map_free(bmap);

	if (dst_type == src_type && dst_pos == src_pos)
		return bmap;

	isl_assert(bmap->ctx, dst_type != src_type, goto error);

	if (pos(bmap->dim, dst_type) + dst_pos ==
	    pos(bmap->dim, src_type) + src_pos +
					    ((src_type < dst_type) ? n : 0)) {
		space = isl_basic_map_take_space(bmap);
		space = isl_space_move_dims(space, dst_type, dst_pos,
						src_type, src_pos, n);
		bmap = isl_basic_map_restore_space(bmap, space);
		bmap = isl_basic_map_finalize(bmap);

		return bmap;
	}

	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return isl_basic_map_free(bmap);
	dim_map = isl_dim_map_alloc(bmap->ctx, total);

	off = 0;
	space = isl_basic_map_peek_space(bmap);
	for (t = isl_dim_param; t <= isl_dim_out; ++t) {
		isl_size size = isl_space_dim(space, t);
		if (size < 0)
			dim_map = isl_dim_map_free(dim_map);
		if (t == dst_type) {
			isl_dim_map_dim_range(dim_map, space, t,
					    0, dst_pos, off);
			off += dst_pos;
			isl_dim_map_dim_range(dim_map, space, src_type,
					    src_pos, n, off);
			off += n;
			isl_dim_map_dim_range(dim_map, space, t,
					    dst_pos, size - dst_pos, off);
			off += size - dst_pos;
		} else if (t == src_type) {
			isl_dim_map_dim_range(dim_map, space, t,
					    0, src_pos, off);
			off += src_pos;
			isl_dim_map_dim_range(dim_map, space, t,
					src_pos + n, size - src_pos - n, off);
			off += size - src_pos - n;
		} else {
			isl_dim_map_dim(dim_map, space, t, off);
			off += size;
		}
	}
	isl_dim_map_div(dim_map, bmap, off);

	res = isl_basic_map_alloc_space(isl_basic_map_get_space(bmap),
			bmap->n_div, bmap->n_eq, bmap->n_ineq);
	bmap = isl_basic_map_add_constraints_dim_map(res, bmap, dim_map);
	space = isl_basic_map_take_space(bmap);
	space = isl_space_move_dims(space, dst_type, dst_pos,
					src_type, src_pos, n);
	bmap = isl_basic_map_restore_space(bmap, space);
	if (!bmap)
		goto error;

	ISL_F_CLR(bmap, ISL_BASIC_MAP_SORTED);
	bmap = isl_basic_map_gauss(bmap, NULL);
	bmap = isl_basic_map_finalize(bmap);

	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_move_dims(__isl_take isl_basic_set *bset,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n)
{
	isl_basic_map *bmap = bset_to_bmap(bset);
	bmap = isl_basic_map_move_dims(bmap, dst_type, dst_pos,
					src_type, src_pos, n);
	return bset_from_bmap(bmap);
}

__isl_give isl_set *isl_set_move_dims(__isl_take isl_set *set,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n)
{
	if (!set)
		return NULL;
	isl_assert(set->ctx, dst_type != isl_dim_in, goto error);
	return set_from_map(isl_map_move_dims(set_to_map(set),
				    dst_type, dst_pos, src_type, src_pos, n));
error:
	isl_set_free(set);
	return NULL;
}

__isl_give isl_map *isl_map_move_dims(__isl_take isl_map *map,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n)
{
	int i;
	isl_space *space;

	if (n == 0) {
		map = isl_map_reset(map, src_type);
		map = isl_map_reset(map, dst_type);
		return map;
	}

	if (isl_map_check_range(map, src_type, src_pos, n))
		return isl_map_free(map);

	if (dst_type == src_type && dst_pos == src_pos)
		return map;

	isl_assert(map->ctx, dst_type != src_type, goto error);

	map = isl_map_cow(map);
	if (!map)
		return NULL;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_move_dims(map->p[i],
						dst_type, dst_pos,
						src_type, src_pos, n);
		if (!map->p[i])
			goto error;
	}

	space = isl_map_take_space(map);
	space = isl_space_move_dims(space, dst_type, dst_pos,
					    src_type, src_pos, n);
	map = isl_map_restore_space(map, space);

	return map;
error:
	isl_map_free(map);
	return NULL;
}

/* Move the specified dimensions to the last columns right before
 * the divs.  Don't change the dimension specification of bmap.
 * That's the responsibility of the caller.
 */
static __isl_give isl_basic_map *move_last(__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	isl_space *space;
	struct isl_dim_map *dim_map;
	struct isl_basic_map *res;
	enum isl_dim_type t;
	isl_size total;
	unsigned off;

	if (!bmap)
		return NULL;
	if (isl_basic_map_offset(bmap, type) + first + n ==
				isl_basic_map_offset(bmap, isl_dim_div))
		return bmap;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return isl_basic_map_free(bmap);
	dim_map = isl_dim_map_alloc(bmap->ctx, total);

	off = 0;
	space = isl_basic_map_peek_space(bmap);
	for (t = isl_dim_param; t <= isl_dim_out; ++t) {
		isl_size size = isl_space_dim(space, t);
		if (size < 0)
			dim_map = isl_dim_map_free(dim_map);
		if (t == type) {
			isl_dim_map_dim_range(dim_map, space, t,
					    0, first, off);
			off += first;
			isl_dim_map_dim_range(dim_map, space, t,
					    first, n, total - bmap->n_div - n);
			isl_dim_map_dim_range(dim_map, space, t,
					    first + n, size - (first + n), off);
			off += size - (first + n);
		} else {
			isl_dim_map_dim(dim_map, space, t, off);
			off += size;
		}
	}
	isl_dim_map_div(dim_map, bmap, off + n);

	res = isl_basic_map_alloc_space(isl_basic_map_get_space(bmap),
			bmap->n_div, bmap->n_eq, bmap->n_ineq);
	res = isl_basic_map_add_constraints_dim_map(res, bmap, dim_map);
	return res;
}

/* Insert "n" rows in the divs of "bmap".
 *
 * The number of columns is not changed, which means that the last
 * dimensions of "bmap" are being reintepreted as the new divs.
 * The space of "bmap" is not adjusted, however, which means
 * that "bmap" is left in an inconsistent state.  Removing "n" dimensions
 * from the space of "bmap" is the responsibility of the caller.
 */
static __isl_give isl_basic_map *insert_div_rows(__isl_take isl_basic_map *bmap,
	int n)
{
	int i;
	size_t row_size;
	isl_int **new_div;
	isl_int *old;

	bmap = isl_basic_map_cow(bmap);
	if (!bmap)
		return NULL;

	row_size = isl_basic_map_offset(bmap, isl_dim_div) + bmap->extra;
	old = bmap->block2.data;
	bmap->block2 = isl_blk_extend(bmap->ctx, bmap->block2,
					(bmap->extra + n) * (1 + row_size));
	if (!bmap->block2.data)
		return isl_basic_map_free(bmap);
	new_div = isl_alloc_array(bmap->ctx, isl_int *, bmap->extra + n);
	if (!new_div)
		return isl_basic_map_free(bmap);
	for (i = 0; i < n; ++i) {
		new_div[i] = bmap->block2.data +
				(bmap->extra + i) * (1 + row_size);
		isl_seq_clr(new_div[i], 1 + row_size);
	}
	for (i = 0; i < bmap->extra; ++i)
		new_div[n + i] = bmap->block2.data + (bmap->div[i] - old);
	free(bmap->div);
	bmap->div = new_div;
	bmap->n_div += n;
	bmap->extra += n;

	return bmap;
}

/* Drop constraints from "bmap" that only involve the variables
 * of "type" in the range [first, first + n] that are not related
 * to any of the variables outside that interval.
 * These constraints cannot influence the values for the variables
 * outside the interval, except in case they cause "bmap" to be empty.
 * Only drop the constraints if "bmap" is known to be non-empty.
 */
static __isl_give isl_basic_map *drop_irrelevant_constraints(
	__isl_take isl_basic_map *bmap, enum isl_dim_type type,
	unsigned first, unsigned n)
{
	int i;
	int *groups;
	isl_size dim, n_div;
	isl_bool non_empty;

	non_empty = isl_basic_map_plain_is_non_empty(bmap);
	if (non_empty < 0)
		return isl_basic_map_free(bmap);
	if (!non_empty)
		return bmap;

	dim = isl_basic_map_dim(bmap, isl_dim_all);
	n_div = isl_basic_map_dim(bmap, isl_dim_div);
	if (dim < 0 || n_div < 0)
		return isl_basic_map_free(bmap);
	groups = isl_calloc_array(isl_basic_map_get_ctx(bmap), int, dim);
	if (!groups)
		return isl_basic_map_free(bmap);
	first += isl_basic_map_offset(bmap, type) - 1;
	for (i = 0; i < first; ++i)
		groups[i] = -1;
	for (i = first + n; i < dim - n_div; ++i)
		groups[i] = -1;

	bmap = isl_basic_map_drop_unrelated_constraints(bmap, groups);

	return bmap;
}

/* Turn the n dimensions of type type, starting at first
 * into existentially quantified variables.
 *
 * If a subset of the projected out variables are unrelated
 * to any of the variables that remain, then the constraints
 * involving this subset are simply dropped first.
 */
__isl_give isl_basic_map *isl_basic_map_project_out(
		__isl_take isl_basic_map *bmap,
		enum isl_dim_type type, unsigned first, unsigned n)
{
	isl_bool empty;
	isl_space *space;

	if (n == 0)
		return basic_map_space_reset(bmap, type);
	if (type == isl_dim_div)
		isl_die(isl_basic_map_get_ctx(bmap), isl_error_invalid,
			"cannot project out existentially quantified variables",
			return isl_basic_map_free(bmap));

	empty = isl_basic_map_plain_is_empty(bmap);
	if (empty < 0)
		return isl_basic_map_free(bmap);
	if (empty)
		bmap = isl_basic_map_set_to_empty(bmap);

	bmap = drop_irrelevant_constraints(bmap, type, first, n);
	if (!bmap)
		return NULL;

	if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_RATIONAL))
		return isl_basic_map_remove_dims(bmap, type, first, n);

	if (isl_basic_map_check_range(bmap, type, first, n) < 0)
		return isl_basic_map_free(bmap);

	bmap = move_last(bmap, type, first, n);
	bmap = isl_basic_map_cow(bmap);
	bmap = insert_div_rows(bmap, n);

	space = isl_basic_map_take_space(bmap);
	space = isl_space_drop_dims(space, type, first, n);
	bmap = isl_basic_map_restore_space(bmap, space);
	bmap = isl_basic_map_simplify(bmap);
	bmap = isl_basic_map_drop_redundant_divs(bmap);
	return isl_basic_map_finalize(bmap);
}

/* Turn the n dimensions of type type, starting at first
 * into existentially quantified variables.
 */
__isl_give isl_basic_set *isl_basic_set_project_out(
	__isl_take isl_basic_set *bset, enum isl_dim_type type,
	unsigned first, unsigned n)
{
	return bset_from_bmap(isl_basic_map_project_out(bset_to_bmap(bset),
							type, first, n));
}

/* Turn the n dimensions of type type, starting at first
 * into existentially quantified variables.
 */
__isl_give isl_map *isl_map_project_out(__isl_take isl_map *map,
		enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;
	isl_space *space;

	if (n == 0)
		return map_space_reset(map, type);

	if (isl_map_check_range(map, type, first, n) < 0)
		return isl_map_free(map);

	map = isl_map_cow(map);
	if (!map)
		return NULL;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_project_out(map->p[i], type, first, n);
		if (!map->p[i])
			goto error;
	}

	if (map->n > 1)
		ISL_F_CLR(map, ISL_MAP_DISJOINT);
	map = isl_map_unmark_normalized(map);

	space = isl_map_take_space(map);
	space = isl_space_drop_dims(space, type, first, n);
	map = isl_map_restore_space(map, space);

	return map;
error:
	isl_map_free(map);
	return NULL;
}

#undef TYPE
#define TYPE	isl_map
#include "isl_project_out_all_params_templ.c"

/* Turn all the dimensions of type "type", except the "n" starting at "first"
 * into existentially quantified variables.
 */
__isl_give isl_map *isl_map_project_onto(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	isl_size dim;

	dim = isl_map_dim(map, type);
	if (isl_map_check_range(map, type, first, n) < 0 || dim < 0)
		return isl_map_free(map);
	map = isl_map_project_out(map, type, first + n, dim - (first + n));
	map = isl_map_project_out(map, type, 0, first);
	return map;
}

/* Turn the n dimensions of type type, starting at first
 * into existentially quantified variables.
 */
__isl_give isl_set *isl_set_project_out(__isl_take isl_set *set,
		enum isl_dim_type type, unsigned first, unsigned n)
{
	return set_from_map(isl_map_project_out(set_to_map(set),
						type, first, n));
}

/* If "set" involves a parameter with identifier "id",
 * then turn it into an existentially quantified variable.
 */
__isl_give isl_set *isl_set_project_out_param_id(__isl_take isl_set *set,
	__isl_take isl_id *id)
{
	int pos;

	if (!set || !id)
		goto error;
	pos = isl_set_find_dim_by_id(set, isl_dim_param, id);
	isl_id_free(id);
	if (pos < 0)
		return set;
	return isl_set_project_out(set, isl_dim_param, pos, 1);
error:
	isl_set_free(set);
	isl_id_free(id);
	return NULL;
}

/* If "set" involves any of the parameters with identifiers in "list",
 * then turn them into existentially quantified variables.
 */
__isl_give isl_set *isl_set_project_out_param_id_list(__isl_take isl_set *set,
	__isl_take isl_id_list *list)
{
	int i;
	isl_size n;

	n = isl_id_list_size(list);
	if (n < 0)
		goto error;
	for (i = 0; i < n; ++i) {
		isl_id *id;

		id = isl_id_list_get_at(list, i);
		set = isl_set_project_out_param_id(set, id);
	}

	isl_id_list_free(list);
	return set;
error:
	isl_id_list_free(list);
	isl_set_free(set);
	return NULL;
}

/* Project out all parameters from "set" by existentially quantifying
 * over them.
 */
__isl_give isl_set *isl_set_project_out_all_params(__isl_take isl_set *set)
{
	return set_from_map(isl_map_project_out_all_params(set_to_map(set)));
}

/* Return a map that projects the elements in "set" onto their
 * "n" set dimensions starting at "first".
 * "type" should be equal to isl_dim_set.
 */
__isl_give isl_map *isl_set_project_onto_map(__isl_take isl_set *set,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;
	isl_map *map;

	if (type != isl_dim_set)
		isl_die(isl_set_get_ctx(set), isl_error_invalid,
			"only set dimensions can be projected out", goto error);
	if (isl_set_check_range(set, type, first, n) < 0)
		return isl_set_free(set);

	map = isl_map_from_domain(set);
	map = isl_map_add_dims(map, isl_dim_out, n);
	for (i = 0; i < n; ++i)
		map = isl_map_equate(map, isl_dim_in, first + i,
					isl_dim_out, i);
	return map;
error:
	isl_set_free(set);
	return NULL;
}

static __isl_give isl_basic_map *add_divs(__isl_take isl_basic_map *bmap,
	unsigned n)
{
	int i, j;
	isl_size total;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return isl_basic_map_free(bmap);
	for (i = 0; i < n; ++i) {
		j = isl_basic_map_alloc_div(bmap);
		if (j < 0)
			goto error;
		isl_seq_clr(bmap->div[j], 1 + 1 + total);
	}
	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

/* Does "bmap2" apply to the range of "bmap1" (ignoring parameters)?
 */
isl_bool isl_basic_map_applies_range(__isl_keep isl_basic_map *bmap1,
	__isl_keep isl_basic_map *bmap2)
{
	isl_space *space1, *space2;

	space1 = isl_basic_map_peek_space(bmap1);
	space2 = isl_basic_map_peek_space(bmap2);
	return isl_space_tuple_is_equal(space1, isl_dim_out,
					space2, isl_dim_in);
}

/* Check that "bmap2" applies to the range of "bmap1" (ignoring parameters).
 */
static isl_stat isl_basic_map_check_applies_range(
	__isl_keep isl_basic_map *bmap1, __isl_keep isl_basic_map *bmap2)
{
	isl_bool equal;

	equal = isl_basic_map_applies_range(bmap1, bmap2);
	if (equal < 0)
		return isl_stat_error;
	if (!equal)
		isl_die(isl_basic_map_get_ctx(bmap1), isl_error_invalid,
			"spaces don't match", return isl_stat_error);
	return isl_stat_ok;
}

__isl_give isl_basic_map *isl_basic_map_apply_range(
	__isl_take isl_basic_map *bmap1, __isl_take isl_basic_map *bmap2)
{
	isl_space *space_result = NULL;
	struct isl_basic_map *bmap;
	isl_size n_in, n_out, n, nparam;
	unsigned total, pos;
	struct isl_dim_map *dim_map1, *dim_map2;

	if (isl_basic_map_check_equal_params(bmap1, bmap2) < 0)
		goto error;
	if (isl_basic_map_check_applies_range(bmap1, bmap2) < 0)
		goto error;

	n_in = isl_basic_map_dim(bmap1, isl_dim_in);
	n_out = isl_basic_map_dim(bmap2, isl_dim_out);
	n = isl_basic_map_dim(bmap1, isl_dim_out);
	nparam = isl_basic_map_dim(bmap1, isl_dim_param);
	if (n_in < 0 || n_out < 0 || n < 0 || nparam < 0)
		goto error;

	space_result = isl_space_join(isl_basic_map_get_space(bmap1),
				  isl_basic_map_get_space(bmap2));

	total = nparam + n_in + n_out + bmap1->n_div + bmap2->n_div + n;
	dim_map1 = isl_dim_map_alloc(bmap1->ctx, total);
	dim_map2 = isl_dim_map_alloc(bmap1->ctx, total);
	isl_dim_map_dim(dim_map1, bmap1->dim, isl_dim_param, pos = 0);
	isl_dim_map_dim(dim_map2, bmap2->dim, isl_dim_param, pos = 0);
	isl_dim_map_dim(dim_map1, bmap1->dim, isl_dim_in, pos += nparam);
	isl_dim_map_dim(dim_map2, bmap2->dim, isl_dim_out, pos += n_in);
	isl_dim_map_div(dim_map1, bmap1, pos += n_out);
	isl_dim_map_div(dim_map2, bmap2, pos += bmap1->n_div);
	isl_dim_map_dim(dim_map1, bmap1->dim, isl_dim_out, pos += bmap2->n_div);
	isl_dim_map_dim(dim_map2, bmap2->dim, isl_dim_in, pos);

	bmap = isl_basic_map_alloc_space(space_result,
			bmap1->n_div + bmap2->n_div + n,
			bmap1->n_eq + bmap2->n_eq,
			bmap1->n_ineq + bmap2->n_ineq);
	bmap = isl_basic_map_add_constraints_dim_map(bmap, bmap1, dim_map1);
	bmap = isl_basic_map_add_constraints_dim_map(bmap, bmap2, dim_map2);
	bmap = add_divs(bmap, n);
	bmap = isl_basic_map_simplify(bmap);
	bmap = isl_basic_map_drop_redundant_divs(bmap);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap1);
	isl_basic_map_free(bmap2);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_apply(__isl_take isl_basic_set *bset,
	__isl_take isl_basic_map *bmap)
{
	if (isl_basic_map_check_compatible_domain(bmap, bset) < 0)
		goto error;

	return bset_from_bmap(isl_basic_map_apply_range(bset_to_bmap(bset),
							bmap));
error:
	isl_basic_set_free(bset);
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_apply_domain(
	__isl_take isl_basic_map *bmap1, __isl_take isl_basic_map *bmap2)
{
	if (isl_basic_map_check_equal_params(bmap1, bmap2) < 0)
		goto error;
	if (!isl_space_tuple_is_equal(bmap1->dim, isl_dim_in,
					bmap2->dim, isl_dim_in))
		isl_die(isl_basic_map_get_ctx(bmap1), isl_error_invalid,
			"spaces don't match", goto error);

	bmap1 = isl_basic_map_reverse(bmap1);
	bmap1 = isl_basic_map_apply_range(bmap1, bmap2);
	return isl_basic_map_reverse(bmap1);
error:
	isl_basic_map_free(bmap1);
	isl_basic_map_free(bmap2);
	return NULL;
}

/* Given two basic maps A -> f(A) and B -> g(B), construct a basic map
 * A \cap B -> f(A) + f(B)
 */
__isl_give isl_basic_map *isl_basic_map_sum(__isl_take isl_basic_map *bmap1,
	__isl_take isl_basic_map *bmap2)
{
	isl_size n_in, n_out, nparam;
	unsigned total, pos;
	struct isl_basic_map *bmap = NULL;
	struct isl_dim_map *dim_map1, *dim_map2;
	int i;

	if (isl_basic_map_check_equal_space(bmap1, bmap2) < 0)
		goto error;

	nparam = isl_basic_map_dim(bmap1, isl_dim_param);
	n_in = isl_basic_map_dim(bmap1, isl_dim_in);
	n_out = isl_basic_map_dim(bmap1, isl_dim_out);
	if (nparam < 0 || n_in < 0 || n_out < 0)
		goto error;

	total = nparam + n_in + n_out + bmap1->n_div + bmap2->n_div + 2 * n_out;
	dim_map1 = isl_dim_map_alloc(bmap1->ctx, total);
	dim_map2 = isl_dim_map_alloc(bmap2->ctx, total);
	isl_dim_map_dim(dim_map1, bmap1->dim, isl_dim_param, pos = 0);
	isl_dim_map_dim(dim_map2, bmap2->dim, isl_dim_param, pos);
	isl_dim_map_dim(dim_map1, bmap1->dim, isl_dim_in, pos += nparam);
	isl_dim_map_dim(dim_map2, bmap2->dim, isl_dim_in, pos);
	isl_dim_map_div(dim_map1, bmap1, pos += n_in + n_out);
	isl_dim_map_div(dim_map2, bmap2, pos += bmap1->n_div);
	isl_dim_map_dim(dim_map1, bmap1->dim, isl_dim_out, pos += bmap2->n_div);
	isl_dim_map_dim(dim_map2, bmap2->dim, isl_dim_out, pos += n_out);

	bmap = isl_basic_map_alloc_space(isl_space_copy(bmap1->dim),
			bmap1->n_div + bmap2->n_div + 2 * n_out,
			bmap1->n_eq + bmap2->n_eq + n_out,
			bmap1->n_ineq + bmap2->n_ineq);
	for (i = 0; i < n_out; ++i) {
		int j = isl_basic_map_alloc_equality(bmap);
		if (j < 0)
			goto error;
		isl_seq_clr(bmap->eq[j], 1+total);
		isl_int_set_si(bmap->eq[j][1+nparam+n_in+i], -1);
		isl_int_set_si(bmap->eq[j][1+pos+i], 1);
		isl_int_set_si(bmap->eq[j][1+pos-n_out+i], 1);
	}
	bmap = isl_basic_map_add_constraints_dim_map(bmap, bmap1, dim_map1);
	bmap = isl_basic_map_add_constraints_dim_map(bmap, bmap2, dim_map2);
	bmap = add_divs(bmap, 2 * n_out);

	bmap = isl_basic_map_simplify(bmap);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap);
	isl_basic_map_free(bmap1);
	isl_basic_map_free(bmap2);
	return NULL;
}

/* Given two maps A -> f(A) and B -> g(B), construct a map
 * A \cap B -> f(A) + f(B)
 */
__isl_give isl_map *isl_map_sum(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	struct isl_map *result;
	int i, j;

	if (isl_map_check_equal_space(map1, map2) < 0)
		goto error;

	result = isl_map_alloc_space(isl_space_copy(map1->dim),
				map1->n * map2->n, 0);
	if (!result)
		goto error;
	for (i = 0; i < map1->n; ++i)
		for (j = 0; j < map2->n; ++j) {
			struct isl_basic_map *part;
			part = isl_basic_map_sum(
				    isl_basic_map_copy(map1->p[i]),
				    isl_basic_map_copy(map2->p[j]));
			if (isl_basic_map_is_empty(part))
				isl_basic_map_free(part);
			else
				result = isl_map_add_basic_map(result, part);
			if (!result)
				goto error;
		}
	isl_map_free(map1);
	isl_map_free(map2);
	return result;
error:
	isl_map_free(map1);
	isl_map_free(map2);
	return NULL;
}

__isl_give isl_set *isl_set_sum(__isl_take isl_set *set1,
	__isl_take isl_set *set2)
{
	return set_from_map(isl_map_sum(set_to_map(set1), set_to_map(set2)));
}

/* Given a basic map A -> f(A), construct A -> -f(A).
 */
__isl_give isl_basic_map *isl_basic_map_neg(__isl_take isl_basic_map *bmap)
{
	int i, j;
	unsigned off;
	isl_size n;

	bmap = isl_basic_map_cow(bmap);
	n = isl_basic_map_dim(bmap, isl_dim_out);
	if (n < 0)
		return isl_basic_map_free(bmap);

	off = isl_basic_map_offset(bmap, isl_dim_out);
	for (i = 0; i < bmap->n_eq; ++i)
		for (j = 0; j < n; ++j)
			isl_int_neg(bmap->eq[i][off+j], bmap->eq[i][off+j]);
	for (i = 0; i < bmap->n_ineq; ++i)
		for (j = 0; j < n; ++j)
			isl_int_neg(bmap->ineq[i][off+j], bmap->ineq[i][off+j]);
	for (i = 0; i < bmap->n_div; ++i)
		for (j = 0; j < n; ++j)
			isl_int_neg(bmap->div[i][1+off+j], bmap->div[i][1+off+j]);
	bmap = isl_basic_map_gauss(bmap, NULL);
	return isl_basic_map_finalize(bmap);
}

__isl_give isl_basic_set *isl_basic_set_neg(__isl_take isl_basic_set *bset)
{
	return isl_basic_map_neg(bset);
}

/* Given a map A -> f(A), construct A -> -f(A).
 */
__isl_give isl_map *isl_map_neg(__isl_take isl_map *map)
{
	int i;

	map = isl_map_cow(map);
	if (!map)
		return NULL;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_neg(map->p[i]);
		if (!map->p[i])
			goto error;
	}

	return map;
error:
	isl_map_free(map);
	return NULL;
}

__isl_give isl_set *isl_set_neg(__isl_take isl_set *set)
{
	return set_from_map(isl_map_neg(set_to_map(set)));
}

/* Given a basic map A -> f(A) and an integer d, construct a basic map
 * A -> floor(f(A)/d).
 */
__isl_give isl_basic_map *isl_basic_map_floordiv(__isl_take isl_basic_map *bmap,
		isl_int d)
{
	isl_size n_in, n_out, nparam;
	unsigned total, pos;
	struct isl_basic_map *result = NULL;
	struct isl_dim_map *dim_map;
	int i;

	nparam = isl_basic_map_dim(bmap, isl_dim_param);
	n_in = isl_basic_map_dim(bmap, isl_dim_in);
	n_out = isl_basic_map_dim(bmap, isl_dim_out);
	if (nparam < 0 || n_in < 0 || n_out < 0)
		return isl_basic_map_free(bmap);

	total = nparam + n_in + n_out + bmap->n_div + n_out;
	dim_map = isl_dim_map_alloc(bmap->ctx, total);
	isl_dim_map_dim(dim_map, bmap->dim, isl_dim_param, pos = 0);
	isl_dim_map_dim(dim_map, bmap->dim, isl_dim_in, pos += nparam);
	isl_dim_map_div(dim_map, bmap, pos += n_in + n_out);
	isl_dim_map_dim(dim_map, bmap->dim, isl_dim_out, pos += bmap->n_div);

	result = isl_basic_map_alloc_space(isl_space_copy(bmap->dim),
			bmap->n_div + n_out,
			bmap->n_eq, bmap->n_ineq + 2 * n_out);
	result = isl_basic_map_add_constraints_dim_map(result, bmap, dim_map);
	result = add_divs(result, n_out);
	for (i = 0; i < n_out; ++i) {
		int j;
		j = isl_basic_map_alloc_inequality(result);
		if (j < 0)
			goto error;
		isl_seq_clr(result->ineq[j], 1+total);
		isl_int_neg(result->ineq[j][1+nparam+n_in+i], d);
		isl_int_set_si(result->ineq[j][1+pos+i], 1);
		j = isl_basic_map_alloc_inequality(result);
		if (j < 0)
			goto error;
		isl_seq_clr(result->ineq[j], 1+total);
		isl_int_set(result->ineq[j][1+nparam+n_in+i], d);
		isl_int_set_si(result->ineq[j][1+pos+i], -1);
		isl_int_sub_ui(result->ineq[j][0], d, 1);
	}

	result = isl_basic_map_simplify(result);
	return isl_basic_map_finalize(result);
error:
	isl_basic_map_free(result);
	return NULL;
}

/* Given a map A -> f(A) and an integer d, construct a map
 * A -> floor(f(A)/d).
 */
__isl_give isl_map *isl_map_floordiv(__isl_take isl_map *map, isl_int d)
{
	int i;

	map = isl_map_cow(map);
	if (!map)
		return NULL;

	ISL_F_CLR(map, ISL_MAP_DISJOINT);
	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_floordiv(map->p[i], d);
		if (!map->p[i])
			goto error;
	}
	map = isl_map_unmark_normalized(map);

	return map;
error:
	isl_map_free(map);
	return NULL;
}

/* Given a map A -> f(A) and an integer d, construct a map
 * A -> floor(f(A)/d).
 */
__isl_give isl_map *isl_map_floordiv_val(__isl_take isl_map *map,
	__isl_take isl_val *d)
{
	if (!map || !d)
		goto error;
	if (!isl_val_is_int(d))
		isl_die(isl_val_get_ctx(d), isl_error_invalid,
			"expecting integer denominator", goto error);
	map = isl_map_floordiv(map, d->n);
	isl_val_free(d);
	return map;
error:
	isl_map_free(map);
	isl_val_free(d);
	return NULL;
}

static __isl_give isl_basic_map *var_equal(__isl_take isl_basic_map *bmap,
	unsigned pos)
{
	int i;
	isl_size nparam;
	isl_size n_in;
	isl_size total;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	nparam = isl_basic_map_dim(bmap, isl_dim_param);
	n_in = isl_basic_map_dim(bmap, isl_dim_in);
	if (total < 0 || nparam < 0 || n_in < 0)
		return isl_basic_map_free(bmap);
	i = isl_basic_map_alloc_equality(bmap);
	if (i < 0)
		goto error;
	isl_seq_clr(bmap->eq[i], 1 + total);
	isl_int_set_si(bmap->eq[i][1+nparam+pos], -1);
	isl_int_set_si(bmap->eq[i][1+nparam+n_in+pos], 1);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap);
	return NULL;
}

/* Add a constraint to "bmap" expressing i_pos < o_pos
 */
static __isl_give isl_basic_map *var_less(__isl_take isl_basic_map *bmap,
	unsigned pos)
{
	int i;
	isl_size nparam;
	isl_size n_in;
	isl_size total;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	nparam = isl_basic_map_dim(bmap, isl_dim_param);
	n_in = isl_basic_map_dim(bmap, isl_dim_in);
	if (total < 0 || nparam < 0 || n_in < 0)
		return isl_basic_map_free(bmap);
	i = isl_basic_map_alloc_inequality(bmap);
	if (i < 0)
		goto error;
	isl_seq_clr(bmap->ineq[i], 1 + total);
	isl_int_set_si(bmap->ineq[i][0], -1);
	isl_int_set_si(bmap->ineq[i][1+nparam+pos], -1);
	isl_int_set_si(bmap->ineq[i][1+nparam+n_in+pos], 1);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap);
	return NULL;
}

/* Add a constraint to "bmap" expressing i_pos <= o_pos
 */
static __isl_give isl_basic_map *var_less_or_equal(
	__isl_take isl_basic_map *bmap, unsigned pos)
{
	int i;
	isl_size nparam;
	isl_size n_in;
	isl_size total;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	nparam = isl_basic_map_dim(bmap, isl_dim_param);
	n_in = isl_basic_map_dim(bmap, isl_dim_in);
	if (total < 0 || nparam < 0 || n_in < 0)
		return isl_basic_map_free(bmap);
	i = isl_basic_map_alloc_inequality(bmap);
	if (i < 0)
		goto error;
	isl_seq_clr(bmap->ineq[i], 1 + total);
	isl_int_set_si(bmap->ineq[i][1+nparam+pos], -1);
	isl_int_set_si(bmap->ineq[i][1+nparam+n_in+pos], 1);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap);
	return NULL;
}

/* Add a constraint to "bmap" expressing i_pos > o_pos
 */
static __isl_give isl_basic_map *var_more(__isl_take isl_basic_map *bmap,
	unsigned pos)
{
	int i;
	isl_size nparam;
	isl_size n_in;
	isl_size total;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	nparam = isl_basic_map_dim(bmap, isl_dim_param);
	n_in = isl_basic_map_dim(bmap, isl_dim_in);
	if (total < 0 || nparam < 0 || n_in < 0)
		return isl_basic_map_free(bmap);
	i = isl_basic_map_alloc_inequality(bmap);
	if (i < 0)
		goto error;
	isl_seq_clr(bmap->ineq[i], 1 + total);
	isl_int_set_si(bmap->ineq[i][0], -1);
	isl_int_set_si(bmap->ineq[i][1+nparam+pos], 1);
	isl_int_set_si(bmap->ineq[i][1+nparam+n_in+pos], -1);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap);
	return NULL;
}

/* Add a constraint to "bmap" expressing i_pos >= o_pos
 */
static __isl_give isl_basic_map *var_more_or_equal(
	__isl_take isl_basic_map *bmap, unsigned pos)
{
	int i;
	isl_size nparam;
	isl_size n_in;
	isl_size total;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	nparam = isl_basic_map_dim(bmap, isl_dim_param);
	n_in = isl_basic_map_dim(bmap, isl_dim_in);
	if (total < 0 || nparam < 0 || n_in < 0)
		return isl_basic_map_free(bmap);
	i = isl_basic_map_alloc_inequality(bmap);
	if (i < 0)
		goto error;
	isl_seq_clr(bmap->ineq[i], 1 + total);
	isl_int_set_si(bmap->ineq[i][1+nparam+pos], 1);
	isl_int_set_si(bmap->ineq[i][1+nparam+n_in+pos], -1);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_equal(
	__isl_take isl_space *space, unsigned n_equal)
{
	int i;
	struct isl_basic_map *bmap;
	bmap = isl_basic_map_alloc_space(space, 0, n_equal, 0);
	if (!bmap)
		return NULL;
	for (i = 0; i < n_equal && bmap; ++i)
		bmap = var_equal(bmap, i);
	return isl_basic_map_finalize(bmap);
}

/* Return a relation on of dimension "space" expressing i_[0..pos] << o_[0..pos]
 */
__isl_give isl_basic_map *isl_basic_map_less_at(__isl_take isl_space *space,
	unsigned pos)
{
	int i;
	struct isl_basic_map *bmap;
	bmap = isl_basic_map_alloc_space(space, 0, pos, 1);
	if (!bmap)
		return NULL;
	for (i = 0; i < pos && bmap; ++i)
		bmap = var_equal(bmap, i);
	if (bmap)
		bmap = var_less(bmap, pos);
	return isl_basic_map_finalize(bmap);
}

/* Return a relation on "space" expressing i_[0..pos] <<= o_[0..pos]
 */
__isl_give isl_basic_map *isl_basic_map_less_or_equal_at(
	__isl_take isl_space *space, unsigned pos)
{
	int i;
	isl_basic_map *bmap;

	bmap = isl_basic_map_alloc_space(space, 0, pos, 1);
	for (i = 0; i < pos; ++i)
		bmap = var_equal(bmap, i);
	bmap = var_less_or_equal(bmap, pos);
	return isl_basic_map_finalize(bmap);
}

/* Return a relation on "space" expressing i_pos > o_pos
 */
__isl_give isl_basic_map *isl_basic_map_more_at(__isl_take isl_space *space,
	unsigned pos)
{
	int i;
	struct isl_basic_map *bmap;
	bmap = isl_basic_map_alloc_space(space, 0, pos, 1);
	if (!bmap)
		return NULL;
	for (i = 0; i < pos && bmap; ++i)
		bmap = var_equal(bmap, i);
	if (bmap)
		bmap = var_more(bmap, pos);
	return isl_basic_map_finalize(bmap);
}

/* Return a relation on "space" expressing i_[0..pos] >>= o_[0..pos]
 */
__isl_give isl_basic_map *isl_basic_map_more_or_equal_at(
	__isl_take isl_space *space, unsigned pos)
{
	int i;
	isl_basic_map *bmap;

	bmap = isl_basic_map_alloc_space(space, 0, pos, 1);
	for (i = 0; i < pos; ++i)
		bmap = var_equal(bmap, i);
	bmap = var_more_or_equal(bmap, pos);
	return isl_basic_map_finalize(bmap);
}

static __isl_give isl_map *map_lex_lte_first(__isl_take isl_space *space,
	unsigned n, int equal)
{
	struct isl_map *map;
	int i;

	if (n == 0 && equal)
		return isl_map_universe(space);

	map = isl_map_alloc_space(isl_space_copy(space), n, ISL_MAP_DISJOINT);

	for (i = 0; i + 1 < n; ++i)
		map = isl_map_add_basic_map(map,
				  isl_basic_map_less_at(isl_space_copy(space), i));
	if (n > 0) {
		if (equal)
			map = isl_map_add_basic_map(map,
			      isl_basic_map_less_or_equal_at(space, n - 1));
		else
			map = isl_map_add_basic_map(map,
			      isl_basic_map_less_at(space, n - 1));
	} else
		isl_space_free(space);

	return map;
}

static __isl_give isl_map *map_lex_lte(__isl_take isl_space *space, int equal)
{
	if (!space)
		return NULL;
	return map_lex_lte_first(space, space->n_out, equal);
}

__isl_give isl_map *isl_map_lex_lt_first(__isl_take isl_space *space,
	unsigned n)
{
	return map_lex_lte_first(space, n, 0);
}

__isl_give isl_map *isl_map_lex_le_first(__isl_take isl_space *space,
	unsigned n)
{
	return map_lex_lte_first(space, n, 1);
}

__isl_give isl_map *isl_map_lex_lt(__isl_take isl_space *set_space)
{
	return map_lex_lte(isl_space_map_from_set(set_space), 0);
}

__isl_give isl_map *isl_map_lex_le(__isl_take isl_space *set_space)
{
	return map_lex_lte(isl_space_map_from_set(set_space), 1);
}

static __isl_give isl_map *map_lex_gte_first(__isl_take isl_space *space,
	unsigned n, int equal)
{
	struct isl_map *map;
	int i;

	if (n == 0 && equal)
		return isl_map_universe(space);

	map = isl_map_alloc_space(isl_space_copy(space), n, ISL_MAP_DISJOINT);

	for (i = 0; i + 1 < n; ++i)
		map = isl_map_add_basic_map(map,
				  isl_basic_map_more_at(isl_space_copy(space), i));
	if (n > 0) {
		if (equal)
			map = isl_map_add_basic_map(map,
			      isl_basic_map_more_or_equal_at(space, n - 1));
		else
			map = isl_map_add_basic_map(map,
			      isl_basic_map_more_at(space, n - 1));
	} else
		isl_space_free(space);

	return map;
}

static __isl_give isl_map *map_lex_gte(__isl_take isl_space *space, int equal)
{
	if (!space)
		return NULL;
	return map_lex_gte_first(space, space->n_out, equal);
}

__isl_give isl_map *isl_map_lex_gt_first(__isl_take isl_space *space,
	unsigned n)
{
	return map_lex_gte_first(space, n, 0);
}

__isl_give isl_map *isl_map_lex_ge_first(__isl_take isl_space *space,
	unsigned n)
{
	return map_lex_gte_first(space, n, 1);
}

__isl_give isl_map *isl_map_lex_gt(__isl_take isl_space *set_space)
{
	return map_lex_gte(isl_space_map_from_set(set_space), 0);
}

__isl_give isl_map *isl_map_lex_ge(__isl_take isl_space *set_space)
{
	return map_lex_gte(isl_space_map_from_set(set_space), 1);
}

__isl_give isl_map *isl_set_lex_le_set(__isl_take isl_set *set1,
	__isl_take isl_set *set2)
{
	isl_map *map;
	map = isl_map_lex_le(isl_set_get_space(set1));
	map = isl_map_intersect_domain(map, set1);
	map = isl_map_intersect_range(map, set2);
	return map;
}

__isl_give isl_map *isl_set_lex_lt_set(__isl_take isl_set *set1,
	__isl_take isl_set *set2)
{
	isl_map *map;
	map = isl_map_lex_lt(isl_set_get_space(set1));
	map = isl_map_intersect_domain(map, set1);
	map = isl_map_intersect_range(map, set2);
	return map;
}

__isl_give isl_map *isl_set_lex_ge_set(__isl_take isl_set *set1,
	__isl_take isl_set *set2)
{
	isl_map *map;
	map = isl_map_lex_ge(isl_set_get_space(set1));
	map = isl_map_intersect_domain(map, set1);
	map = isl_map_intersect_range(map, set2);
	return map;
}

__isl_give isl_map *isl_set_lex_gt_set(__isl_take isl_set *set1,
	__isl_take isl_set *set2)
{
	isl_map *map;
	map = isl_map_lex_gt(isl_set_get_space(set1));
	map = isl_map_intersect_domain(map, set1);
	map = isl_map_intersect_range(map, set2);
	return map;
}

__isl_give isl_map *isl_map_lex_le_map(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	isl_map *map;
	map = isl_map_lex_le(isl_space_range(isl_map_get_space(map1)));
	map = isl_map_apply_domain(map, isl_map_reverse(map1));
	map = isl_map_apply_range(map, isl_map_reverse(map2));
	return map;
}

__isl_give isl_map *isl_map_lex_lt_map(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	isl_map *map;
	map = isl_map_lex_lt(isl_space_range(isl_map_get_space(map1)));
	map = isl_map_apply_domain(map, isl_map_reverse(map1));
	map = isl_map_apply_range(map, isl_map_reverse(map2));
	return map;
}

__isl_give isl_map *isl_map_lex_ge_map(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	isl_map *map;
	map = isl_map_lex_ge(isl_space_range(isl_map_get_space(map1)));
	map = isl_map_apply_domain(map, isl_map_reverse(map1));
	map = isl_map_apply_range(map, isl_map_reverse(map2));
	return map;
}

__isl_give isl_map *isl_map_lex_gt_map(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	isl_map *map;
	map = isl_map_lex_gt(isl_space_range(isl_map_get_space(map1)));
	map = isl_map_apply_domain(map, isl_map_reverse(map1));
	map = isl_map_apply_range(map, isl_map_reverse(map2));
	return map;
}

/* For the div d = floor(f/m) at position "div", add the constraint
 *
 *		f - m d >= 0
 */
static __isl_give isl_basic_map *add_upper_div_constraint(
	__isl_take isl_basic_map *bmap, unsigned div)
{
	int i;
	isl_size v_div = isl_basic_map_var_offset(bmap, isl_dim_div);
	isl_size n_div;
	unsigned pos;

	n_div = isl_basic_map_dim(bmap, isl_dim_div);
	if (v_div < 0 || n_div < 0)
		return isl_basic_map_free(bmap);
	pos = v_div + div;
	i = isl_basic_map_alloc_inequality(bmap);
	if (i < 0)
		return isl_basic_map_free(bmap);
	isl_seq_cpy(bmap->ineq[i], bmap->div[div] + 1, 1 + v_div + n_div);
	isl_int_neg(bmap->ineq[i][1 + pos], bmap->div[div][0]);

	return bmap;
}

/* For the div d = floor(f/m) at position "div", add the constraint
 *
 *		-(f-(m-1)) + m d >= 0
 */
static __isl_give isl_basic_map *add_lower_div_constraint(
	__isl_take isl_basic_map *bmap, unsigned div)
{
	int i;
	isl_size v_div = isl_basic_map_var_offset(bmap, isl_dim_div);
	isl_size n_div;
	unsigned pos;

	n_div = isl_basic_map_dim(bmap, isl_dim_div);
	if (v_div < 0 || n_div < 0)
		return isl_basic_map_free(bmap);
	pos = v_div + div;
	i = isl_basic_map_alloc_inequality(bmap);
	if (i < 0)
		return isl_basic_map_free(bmap);
	isl_seq_neg(bmap->ineq[i], bmap->div[div] + 1, 1 + v_div + n_div);
	isl_int_set(bmap->ineq[i][1 + pos], bmap->div[div][0]);
	isl_int_add(bmap->ineq[i][0], bmap->ineq[i][0], bmap->ineq[i][1 + pos]);
	isl_int_sub_ui(bmap->ineq[i][0], bmap->ineq[i][0], 1);

	return bmap;
}

/* For the div d = floor(f/m) at position "pos", add the constraints
 *
 *		f - m d >= 0
 *		-(f-(m-1)) + m d >= 0
 *
 * Note that the second constraint is the negation of
 *
 *		f - m d >= m
 */
__isl_give isl_basic_map *isl_basic_map_add_div_constraints(
	__isl_take isl_basic_map *bmap, unsigned pos)
{
	bmap = add_upper_div_constraint(bmap, pos);
	bmap = add_lower_div_constraint(bmap, pos);
	return bmap;
}

/* For each known div d = floor(f/m), add the constraints
 *
 *		f - m d >= 0
 *		-(f-(m-1)) + m d >= 0
 *
 * Remove duplicate constraints in case of some these div constraints
 * already appear in "bmap".
 */
__isl_give isl_basic_map *isl_basic_map_add_known_div_constraints(
	__isl_take isl_basic_map *bmap)
{
	isl_size n_div;

	n_div = isl_basic_map_dim(bmap, isl_dim_div);
	if (n_div < 0)
		return isl_basic_map_free(bmap);
	if (n_div == 0)
		return bmap;

	bmap = add_known_div_constraints(bmap);
	bmap = isl_basic_map_remove_duplicate_constraints(bmap, NULL, 0);
	bmap = isl_basic_map_finalize(bmap);
	return bmap;
}

/* Add the div constraint of sign "sign" for div "div" of "bmap".
 *
 * In particular, if this div is of the form d = floor(f/m),
 * then add the constraint
 *
 *		f - m d >= 0
 *
 * if sign < 0 or the constraint
 *
 *		-(f-(m-1)) + m d >= 0
 *
 * if sign > 0.
 */
__isl_give isl_basic_map *isl_basic_map_add_div_constraint(
	__isl_take isl_basic_map *bmap, unsigned div, int sign)
{
	if (sign < 0)
		return add_upper_div_constraint(bmap, div);
	else
		return add_lower_div_constraint(bmap, div);
}

__isl_give isl_basic_set *isl_basic_map_underlying_set(
	__isl_take isl_basic_map *bmap)
{
	isl_space *space;

	if (!bmap)
		goto error;
	if (bmap->dim->nparam == 0 && bmap->dim->n_in == 0 &&
	    bmap->n_div == 0 &&
	    !isl_space_is_named_or_nested(bmap->dim, isl_dim_in) &&
	    !isl_space_is_named_or_nested(bmap->dim, isl_dim_out))
		return bset_from_bmap(bmap);
	bmap = isl_basic_map_cow(bmap);
	if (!bmap)
		return NULL;
	space = isl_basic_map_take_space(bmap);
	space = isl_space_underlying(space, bmap->n_div);
	bmap = isl_basic_map_restore_space(bmap, space);
	if (!bmap)
		return NULL;
	bmap->extra -= bmap->n_div;
	bmap->n_div = 0;
	bmap = isl_basic_map_finalize(bmap);
	return bset_from_bmap(bmap);
error:
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_underlying_set(
		__isl_take isl_basic_set *bset)
{
	return isl_basic_map_underlying_set(bset_to_bmap(bset));
}

/* Replace each element in "list" by the result of applying
 * isl_basic_map_underlying_set to the element.
 */
__isl_give isl_basic_set_list *isl_basic_map_list_underlying_set(
	__isl_take isl_basic_map_list *list)
{
	int i;
	isl_size n;

	n = isl_basic_map_list_n_basic_map(list);
	if (n < 0)
		goto error;

	for (i = 0; i < n; ++i) {
		isl_basic_map *bmap;
		isl_basic_set *bset;

		bmap = isl_basic_map_list_get_basic_map(list, i);
		bset = isl_basic_set_underlying_set(bmap);
		list = isl_basic_set_list_set_basic_set(list, i, bset);
	}

	return list;
error:
	isl_basic_map_list_free(list);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_overlying_set(
	__isl_take isl_basic_set *bset, __isl_take isl_basic_map *like)
{
	struct isl_basic_map *bmap;
	struct isl_ctx *ctx;
	isl_size dim, bmap_total;
	unsigned total;
	int i;

	if (!bset || !like)
		goto error;
	ctx = bset->ctx;
	if (isl_basic_set_check_no_params(bset) < 0 ||
	    isl_basic_set_check_no_locals(bset) < 0)
		goto error;
	dim = isl_basic_set_dim(bset, isl_dim_set);
	bmap_total = isl_basic_map_dim(like, isl_dim_all);
	if (dim < 0 || bmap_total < 0)
		goto error;
	isl_assert(ctx, dim == bmap_total, goto error);
	if (like->n_div == 0) {
		isl_space *space = isl_basic_map_get_space(like);
		isl_basic_map_free(like);
		return isl_basic_map_reset_space(bset, space);
	}
	bset = isl_basic_set_cow(bset);
	if (!bset)
		goto error;
	total = dim + bset->extra;
	bmap = bset_to_bmap(bset);
	isl_space_free(isl_basic_map_take_space(bmap));
	bmap = isl_basic_map_restore_space(bmap, isl_basic_map_get_space(like));
	if (!bmap)
		goto error;
	bmap->n_div = like->n_div;
	bmap->extra += like->n_div;
	if (bmap->extra) {
		unsigned ltotal;
		isl_int **div;
		ltotal = total - bmap->extra + like->extra;
		if (ltotal > total)
			ltotal = total;
		bmap->block2 = isl_blk_extend(ctx, bmap->block2,
					bmap->extra * (1 + 1 + total));
		if (isl_blk_is_error(bmap->block2))
			goto error;
		div = isl_realloc_array(ctx, bmap->div, isl_int *, bmap->extra);
		if (!div)
			goto error;
		bmap->div = div;
		for (i = 0; i < bmap->extra; ++i)
			bmap->div[i] = bmap->block2.data + i * (1 + 1 + total);
		for (i = 0; i < like->n_div; ++i) {
			isl_seq_cpy(bmap->div[i], like->div[i], 1 + 1 + ltotal);
			isl_seq_clr(bmap->div[i]+1+1+ltotal, total - ltotal);
		}
		bmap = isl_basic_map_add_known_div_constraints(bmap);
	}
	isl_basic_map_free(like);
	bmap = isl_basic_map_simplify(bmap);
	bmap = isl_basic_map_finalize(bmap);
	return bmap;
error:
	isl_basic_map_free(like);
	isl_basic_set_free(bset);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_from_underlying_set(
	__isl_take isl_basic_set *bset, __isl_take isl_basic_set *like)
{
	return bset_from_bmap(isl_basic_map_overlying_set(bset,
							bset_to_bmap(like)));
}

__isl_give isl_set *isl_map_underlying_set(__isl_take isl_map *map)
{
	int i;

	map = isl_map_cow(map);
	if (!map)
		return NULL;
	map->dim = isl_space_cow(map->dim);
	if (!map->dim)
		goto error;

	for (i = 1; i < map->n; ++i)
		isl_assert(map->ctx, map->p[0]->n_div == map->p[i]->n_div,
				goto error);
	for (i = 0; i < map->n; ++i) {
		map->p[i] = bset_to_bmap(
				isl_basic_map_underlying_set(map->p[i]));
		if (!map->p[i])
			goto error;
	}
	if (map->n == 0)
		map->dim = isl_space_underlying(map->dim, 0);
	else {
		isl_space_free(map->dim);
		map->dim = isl_space_copy(map->p[0]->dim);
	}
	if (!map->dim)
		goto error;
	return set_from_map(map);
error:
	isl_map_free(map);
	return NULL;
}

/* Replace the space of "bmap" by "space".
 *
 * If the space of "bmap" is identical to "space" (including the identifiers
 * of the input and output dimensions), then simply return the original input.
 */
__isl_give isl_basic_map *isl_basic_map_reset_space(
	__isl_take isl_basic_map *bmap, __isl_take isl_space *space)
{
	isl_bool equal;
	isl_space *bmap_space;

	bmap_space = isl_basic_map_peek_space(bmap);
	equal = isl_space_is_equal(bmap_space, space);
	if (equal >= 0 && equal)
		equal = isl_space_has_equal_ids(bmap_space, space);
	if (equal < 0)
		goto error;
	if (equal) {
		isl_space_free(space);
		return bmap;
	}
	isl_space_free(isl_basic_map_take_space(bmap));
	bmap = isl_basic_map_restore_space(bmap, space);

	bmap = isl_basic_map_finalize(bmap);

	return bmap;
error:
	isl_basic_map_free(bmap);
	isl_space_free(space);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_reset_space(
	__isl_take isl_basic_set *bset, __isl_take isl_space *space)
{
	return bset_from_bmap(isl_basic_map_reset_space(bset_to_bmap(bset),
							space));
}

/* Check that the total dimensions of "map" and "space" are the same.
 */
static isl_stat check_map_space_equal_total_dim(__isl_keep isl_map *map,
	__isl_keep isl_space *space)
{
	isl_size dim1, dim2;

	dim1 = isl_map_dim(map, isl_dim_all);
	dim2 = isl_space_dim(space, isl_dim_all);
	if (dim1 < 0 || dim2 < 0)
		return isl_stat_error;
	if (dim1 == dim2)
		return isl_stat_ok;
	isl_die(isl_map_get_ctx(map), isl_error_invalid,
		"total dimensions do not match", return isl_stat_error);
}

__isl_give isl_map *isl_map_reset_space(__isl_take isl_map *map,
	__isl_take isl_space *space)
{
	int i;

	map = isl_map_cow(map);
	if (!map || !space)
		goto error;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_reset_space(map->p[i],
						    isl_space_copy(space));
		if (!map->p[i])
			goto error;
	}
	isl_space_free(isl_map_take_space(map));
	map = isl_map_restore_space(map, space);

	return map;
error:
	isl_map_free(map);
	isl_space_free(space);
	return NULL;
}

/* Replace the space of "map" by "space", without modifying
 * the dimension of "map".
 *
 * If the space of "map" is identical to "space" (including the identifiers
 * of the input and output dimensions), then simply return the original input.
 */
__isl_give isl_map *isl_map_reset_equal_dim_space(__isl_take isl_map *map,
	__isl_take isl_space *space)
{
	isl_bool equal;
	isl_space *map_space;

	map_space = isl_map_peek_space(map);
	equal = isl_space_is_equal(map_space, space);
	if (equal >= 0 && equal)
		equal = isl_space_has_equal_ids(map_space, space);
	if (equal < 0)
		goto error;
	if (equal) {
		isl_space_free(space);
		return map;
	}
	if (check_map_space_equal_total_dim(map, space) < 0)
		goto error;
	return isl_map_reset_space(map, space);
error:
	isl_map_free(map);
	isl_space_free(space);
	return NULL;
}

__isl_give isl_set *isl_set_reset_space(__isl_take isl_set *set,
	__isl_take isl_space *space)
{
	return set_from_map(isl_map_reset_space(set_to_map(set), space));
}

/* Compute the parameter domain of the given basic set.
 */
__isl_give isl_basic_set *isl_basic_set_params(__isl_take isl_basic_set *bset)
{
	isl_bool is_params;
	isl_space *space;
	isl_size n;

	is_params = isl_basic_set_is_params(bset);
	if (is_params < 0)
		return isl_basic_set_free(bset);
	if (is_params)
		return bset;

	n = isl_basic_set_dim(bset, isl_dim_set);
	if (n < 0)
		return isl_basic_set_free(bset);
	bset = isl_basic_set_project_out(bset, isl_dim_set, 0, n);
	space = isl_basic_set_get_space(bset);
	space = isl_space_params(space);
	bset = isl_basic_set_reset_space(bset, space);
	return bset;
}

/* Construct a zero-dimensional basic set with the given parameter domain.
 */
__isl_give isl_basic_set *isl_basic_set_from_params(
	__isl_take isl_basic_set *bset)
{
	isl_space *space;
	space = isl_basic_set_get_space(bset);
	space = isl_space_set_from_params(space);
	bset = isl_basic_set_reset_space(bset, space);
	return bset;
}

/* Compute the parameter domain of the given set.
 */
__isl_give isl_set *isl_set_params(__isl_take isl_set *set)
{
	return isl_map_params(set_to_map(set));
}

/* Construct a zero-dimensional set with the given parameter domain.
 */
__isl_give isl_set *isl_set_from_params(__isl_take isl_set *set)
{
	isl_space *space;
	space = isl_set_get_space(set);
	space = isl_space_set_from_params(space);
	set = isl_set_reset_space(set, space);
	return set;
}

/* Compute the parameter domain of the given map.
 */
__isl_give isl_set *isl_map_params(__isl_take isl_map *map)
{
	isl_space *space;
	isl_size n_in, n_out;

	n_in = isl_map_dim(map, isl_dim_in);
	n_out = isl_map_dim(map, isl_dim_out);
	if (n_in < 0 || n_out < 0)
		return isl_map_free(map);
	map = isl_map_project_out(map, isl_dim_in, 0, n_in);
	map = isl_map_project_out(map, isl_dim_out, 0, n_out);
	space = isl_map_get_space(map);
	space = isl_space_params(space);
	map = isl_map_reset_space(map, space);
	return map;
}

__isl_give isl_basic_set *isl_basic_map_domain(__isl_take isl_basic_map *bmap)
{
	isl_space *space;
	isl_size n_out;

	n_out = isl_basic_map_dim(bmap, isl_dim_out);
	if (n_out < 0)
		return isl_basic_map_free(bmap);
	space = isl_space_domain(isl_basic_map_get_space(bmap));

	bmap = isl_basic_map_project_out(bmap, isl_dim_out, 0, n_out);

	return isl_basic_map_reset_space(bmap, space);
}

isl_bool isl_basic_map_may_be_set(__isl_keep isl_basic_map *bmap)
{
	if (!bmap)
		return isl_bool_error;
	return isl_space_may_be_set(bmap->dim);
}

/* Is this basic map actually a set?
 * Users should never call this function.  Outside of isl,
 * the type should indicate whether something is a set or a map.
 */
isl_bool isl_basic_map_is_set(__isl_keep isl_basic_map *bmap)
{
	if (!bmap)
		return isl_bool_error;
	return isl_space_is_set(bmap->dim);
}

__isl_give isl_basic_set *isl_basic_map_range(__isl_take isl_basic_map *bmap)
{
	isl_bool is_set;

	is_set = isl_basic_map_is_set(bmap);
	if (is_set < 0)
		goto error;
	if (is_set)
		return bmap;
	return isl_basic_map_domain(isl_basic_map_reverse(bmap));
error:
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_domain_map(
	__isl_take isl_basic_map *bmap)
{
	int i;
	isl_space *space;
	isl_basic_map *domain;
	isl_size nparam, n_in, n_out;

	nparam = isl_basic_map_dim(bmap, isl_dim_param);
	n_in = isl_basic_map_dim(bmap, isl_dim_in);
	n_out = isl_basic_map_dim(bmap, isl_dim_out);
	if (nparam < 0 || n_in < 0 || n_out < 0)
		return isl_basic_map_free(bmap);

	space = isl_basic_map_get_space(bmap);
	space = isl_space_from_range(isl_space_domain(space));
	domain = isl_basic_map_universe(space);

	bmap = isl_basic_map_from_domain(isl_basic_map_wrap(bmap));
	bmap = isl_basic_map_apply_range(bmap, domain);
	bmap = isl_basic_map_extend_constraints(bmap, n_in, 0);

	for (i = 0; i < n_in; ++i)
		bmap = isl_basic_map_equate(bmap, isl_dim_in, i,
						    isl_dim_out, i);

	bmap = isl_basic_map_gauss(bmap, NULL);
	return isl_basic_map_finalize(bmap);
}

__isl_give isl_basic_map *isl_basic_map_range_map(
	__isl_take isl_basic_map *bmap)
{
	int i;
	isl_space *space;
	isl_basic_map *range;
	isl_size nparam, n_in, n_out;

	nparam = isl_basic_map_dim(bmap, isl_dim_param);
	n_in = isl_basic_map_dim(bmap, isl_dim_in);
	n_out = isl_basic_map_dim(bmap, isl_dim_out);
	if (nparam < 0 || n_in < 0 || n_out < 0)
		return isl_basic_map_free(bmap);

	space = isl_basic_map_get_space(bmap);
	space = isl_space_from_range(isl_space_range(space));
	range = isl_basic_map_universe(space);

	bmap = isl_basic_map_from_domain(isl_basic_map_wrap(bmap));
	bmap = isl_basic_map_apply_range(bmap, range);
	bmap = isl_basic_map_extend_constraints(bmap, n_out, 0);

	for (i = 0; i < n_out; ++i)
		bmap = isl_basic_map_equate(bmap, isl_dim_in, n_in + i,
						    isl_dim_out, i);

	bmap = isl_basic_map_gauss(bmap, NULL);
	return isl_basic_map_finalize(bmap);
}

int isl_map_may_be_set(__isl_keep isl_map *map)
{
	if (!map)
		return -1;
	return isl_space_may_be_set(map->dim);
}

/* Is this map actually a set?
 * Users should never call this function.  Outside of isl,
 * the type should indicate whether something is a set or a map.
 */
isl_bool isl_map_is_set(__isl_keep isl_map *map)
{
	if (!map)
		return isl_bool_error;
	return isl_space_is_set(map->dim);
}

__isl_give isl_set *isl_map_range(__isl_take isl_map *map)
{
	isl_space *space;
	isl_size n_in;

	n_in = isl_map_dim(map, isl_dim_in);
	if (n_in < 0)
		return set_from_map(isl_map_free(map));
	space = isl_space_range(isl_map_get_space(map));

	map = isl_map_project_out(map, isl_dim_in, 0, n_in);

	return set_from_map(isl_map_reset_space(map, space));
}

/* Transform "map" by applying "fn_space" to its space and "fn_bmap"
 * to each of its basic maps.
 */
static __isl_give isl_map *isl_map_transform(__isl_take isl_map *map,
	__isl_give isl_space *(*fn_space)(__isl_take isl_space *space),
	__isl_give isl_basic_map *(*fn_bmap)(__isl_take isl_basic_map *bmap))
{
	int i;
	isl_space *space;

	map = isl_map_cow(map);
	if (!map)
		return NULL;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = fn_bmap(map->p[i]);
		if (!map->p[i])
			return isl_map_free(map);
	}
	map = isl_map_unmark_normalized(map);

	space = isl_map_take_space(map);
	space = fn_space(space);
	map = isl_map_restore_space(map, space);

	return map;
}

__isl_give isl_map *isl_map_domain_map(__isl_take isl_map *map)
{
	return isl_map_transform(map, &isl_space_domain_map,
					&isl_basic_map_domain_map);
}

__isl_give isl_map *isl_map_range_map(__isl_take isl_map *map)
{
	return isl_map_transform(map, &isl_space_range_map,
					&isl_basic_map_range_map);
}

/* Given a wrapped map of the form A[B -> C],
 * return the map A[B -> C] -> B.
 */
__isl_give isl_map *isl_set_wrapped_domain_map(__isl_take isl_set *set)
{
	isl_id *id;
	isl_map *map;

	if (!set)
		return NULL;
	if (!isl_set_has_tuple_id(set))
		return isl_map_domain_map(isl_set_unwrap(set));

	id = isl_set_get_tuple_id(set);
	map = isl_map_domain_map(isl_set_unwrap(set));
	map = isl_map_set_tuple_id(map, isl_dim_in, id);

	return map;
}

__isl_give isl_basic_map *isl_basic_map_from_domain(
	__isl_take isl_basic_set *bset)
{
	return isl_basic_map_reverse(isl_basic_map_from_range(bset));
}

__isl_give isl_basic_map *isl_basic_map_from_range(
	__isl_take isl_basic_set *bset)
{
	isl_space *space;
	space = isl_basic_set_get_space(bset);
	space = isl_space_from_range(space);
	bset = isl_basic_set_reset_space(bset, space);
	return bset_to_bmap(bset);
}

/* Create a relation with the given set as range.
 * The domain of the created relation is a zero-dimensional
 * flat anonymous space.
 */
__isl_give isl_map *isl_map_from_range(__isl_take isl_set *set)
{
	isl_space *space;
	space = isl_set_get_space(set);
	space = isl_space_from_range(space);
	set = isl_set_reset_space(set, space);
	return set_to_map(set);
}

/* Create a relation with the given set as domain.
 * The range of the created relation is a zero-dimensional
 * flat anonymous space.
 */
__isl_give isl_map *isl_map_from_domain(__isl_take isl_set *set)
{
	return isl_map_reverse(isl_map_from_range(set));
}

__isl_give isl_basic_map *isl_basic_map_from_domain_and_range(
	__isl_take isl_basic_set *domain, __isl_take isl_basic_set *range)
{
	return isl_basic_map_apply_range(isl_basic_map_reverse(domain), range);
}

__isl_give isl_map *isl_map_from_domain_and_range(__isl_take isl_set *domain,
	__isl_take isl_set *range)
{
	return isl_map_apply_range(isl_map_reverse(domain), range);
}

/* Return a newly allocated isl_map with given space and flags and
 * room for "n" basic maps.
 * Make sure that all cached information is cleared.
 */
__isl_give isl_map *isl_map_alloc_space(__isl_take isl_space *space, int n,
	unsigned flags)
{
	struct isl_map *map;

	if (!space)
		return NULL;
	if (n < 0)
		isl_die(space->ctx, isl_error_internal,
			"negative number of basic maps", goto error);
	map = isl_calloc(space->ctx, struct isl_map,
			sizeof(struct isl_map) +
			(n - 1) * sizeof(struct isl_basic_map *));
	if (!map)
		goto error;

	map->ctx = space->ctx;
	isl_ctx_ref(map->ctx);
	map->ref = 1;
	map->size = n;
	map->n = 0;
	map->dim = space;
	map->flags = flags;
	return map;
error:
	isl_space_free(space);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_empty(__isl_take isl_space *space)
{
	struct isl_basic_map *bmap;
	bmap = isl_basic_map_alloc_space(space, 0, 1, 0);
	bmap = isl_basic_map_set_to_empty(bmap);
	return bmap;
}

__isl_give isl_basic_set *isl_basic_set_empty(__isl_take isl_space *space)
{
	struct isl_basic_set *bset;
	bset = isl_basic_set_alloc_space(space, 0, 1, 0);
	bset = isl_basic_set_set_to_empty(bset);
	return bset;
}

__isl_give isl_basic_map *isl_basic_map_universe(__isl_take isl_space *space)
{
	struct isl_basic_map *bmap;
	bmap = isl_basic_map_alloc_space(space, 0, 0, 0);
	bmap = isl_basic_map_finalize(bmap);
	return bmap;
}

__isl_give isl_basic_set *isl_basic_set_universe(__isl_take isl_space *space)
{
	struct isl_basic_set *bset;
	bset = isl_basic_set_alloc_space(space, 0, 0, 0);
	bset = isl_basic_set_finalize(bset);
	return bset;
}

__isl_give isl_basic_map *isl_basic_map_nat_universe(
	__isl_take isl_space *space)
{
	int i;
	isl_size total = isl_space_dim(space, isl_dim_all);
	isl_basic_map *bmap;

	if (total < 0)
		space = isl_space_free(space);
	bmap = isl_basic_map_alloc_space(space, 0, 0, total);
	for (i = 0; i < total; ++i) {
		int k = isl_basic_map_alloc_inequality(bmap);
		if (k < 0)
			goto error;
		isl_seq_clr(bmap->ineq[k], 1 + total);
		isl_int_set_si(bmap->ineq[k][1 + i], 1);
	}
	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_nat_universe(
	__isl_take isl_space *space)
{
	return isl_basic_map_nat_universe(space);
}

__isl_give isl_map *isl_map_nat_universe(__isl_take isl_space *space)
{
	return isl_map_from_basic_map(isl_basic_map_nat_universe(space));
}

__isl_give isl_set *isl_set_nat_universe(__isl_take isl_space *space)
{
	return isl_map_nat_universe(space);
}

__isl_give isl_map *isl_map_empty(__isl_take isl_space *space)
{
	return isl_map_alloc_space(space, 0, ISL_MAP_DISJOINT);
}

__isl_give isl_set *isl_set_empty(__isl_take isl_space *space)
{
	return isl_set_alloc_space(space, 0, ISL_MAP_DISJOINT);
}

__isl_give isl_map *isl_map_universe(__isl_take isl_space *space)
{
	struct isl_map *map;
	if (!space)
		return NULL;
	map = isl_map_alloc_space(isl_space_copy(space), 1, ISL_MAP_DISJOINT);
	map = isl_map_add_basic_map(map, isl_basic_map_universe(space));
	return map;
}

/* This function performs the same operation as isl_map_universe,
 * but is considered as a function on an isl_space when exported.
 */
__isl_give isl_map *isl_space_universe_map(__isl_take isl_space *space)
{
	return isl_map_universe(space);
}

__isl_give isl_set *isl_set_universe(__isl_take isl_space *space)
{
	struct isl_set *set;
	if (!space)
		return NULL;
	set = isl_set_alloc_space(isl_space_copy(space), 1, ISL_MAP_DISJOINT);
	set = isl_set_add_basic_set(set, isl_basic_set_universe(space));
	return set;
}

/* This function performs the same operation as isl_set_universe,
 * but is considered as a function on an isl_space when exported.
 */
__isl_give isl_set *isl_space_universe_set(__isl_take isl_space *space)
{
	return isl_set_universe(space);
}

__isl_give isl_map *isl_map_dup(__isl_keep isl_map *map)
{
	int i;
	struct isl_map *dup;

	if (!map)
		return NULL;
	dup = isl_map_alloc_space(isl_space_copy(map->dim), map->n, map->flags);
	for (i = 0; i < map->n; ++i)
		dup = isl_map_add_basic_map(dup, isl_basic_map_copy(map->p[i]));
	return dup;
}

__isl_give isl_map *isl_map_add_basic_map(__isl_take isl_map *map,
						__isl_take isl_basic_map *bmap)
{
	if (!bmap || !map)
		goto error;
	if (isl_basic_map_plain_is_empty(bmap)) {
		isl_basic_map_free(bmap);
		return map;
	}
	if (isl_map_basic_map_check_equal_space(map, bmap) < 0)
		goto error;
	isl_assert(map->ctx, map->n < map->size, goto error);
	map->p[map->n] = bmap;
	map->n++;
	map = isl_map_unmark_normalized(map);
	return map;
error:
	if (map)
		isl_map_free(map);
	if (bmap)
		isl_basic_map_free(bmap);
	return NULL;
}

__isl_null isl_map *isl_map_free(__isl_take isl_map *map)
{
	int i;

	if (!map)
		return NULL;

	if (--map->ref > 0)
		return NULL;

	clear_caches(map);
	isl_ctx_deref(map->ctx);
	for (i = 0; i < map->n; ++i)
		isl_basic_map_free(map->p[i]);
	isl_space_free(map->dim);
	free(map);

	return NULL;
}

static __isl_give isl_basic_map *isl_basic_map_fix_pos_si(
	__isl_take isl_basic_map *bmap, unsigned pos, int value)
{
	int j;
	isl_size total;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return isl_basic_map_free(bmap);

	bmap = isl_basic_map_cow(bmap);
	bmap = isl_basic_map_extend_constraints(bmap, 1, 0);
	j = isl_basic_map_alloc_equality(bmap);
	if (j < 0)
		goto error;
	isl_seq_clr(bmap->eq[j] + 1, total);
	isl_int_set_si(bmap->eq[j][pos], -1);
	isl_int_set_si(bmap->eq[j][0], value);
	bmap = isl_basic_map_simplify(bmap);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap);
	return NULL;
}

static __isl_give isl_basic_map *isl_basic_map_fix_pos(
	__isl_take isl_basic_map *bmap, unsigned pos, isl_int value)
{
	int j;
	isl_size total;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return isl_basic_map_free(bmap);

	bmap = isl_basic_map_cow(bmap);
	bmap = isl_basic_map_extend_constraints(bmap, 1, 0);
	j = isl_basic_map_alloc_equality(bmap);
	if (j < 0)
		goto error;
	isl_seq_clr(bmap->eq[j] + 1, total);
	isl_int_set_si(bmap->eq[j][pos], -1);
	isl_int_set(bmap->eq[j][0], value);
	bmap = isl_basic_map_simplify(bmap);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_fix_si(__isl_take isl_basic_map *bmap,
		enum isl_dim_type type, unsigned pos, int value)
{
	if (isl_basic_map_check_range(bmap, type, pos, 1) < 0)
		return isl_basic_map_free(bmap);
	return isl_basic_map_fix_pos_si(bmap,
		isl_basic_map_offset(bmap, type) + pos, value);
}

__isl_give isl_basic_map *isl_basic_map_fix(__isl_take isl_basic_map *bmap,
		enum isl_dim_type type, unsigned pos, isl_int value)
{
	if (isl_basic_map_check_range(bmap, type, pos, 1) < 0)
		return isl_basic_map_free(bmap);
	return isl_basic_map_fix_pos(bmap,
		isl_basic_map_offset(bmap, type) + pos, value);
}

/* Fix the value of the variable at position "pos" of type "type" of "bmap"
 * to be equal to "v".
 */
__isl_give isl_basic_map *isl_basic_map_fix_val(__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos, __isl_take isl_val *v)
{
	if (!bmap || !v)
		goto error;
	if (!isl_val_is_int(v))
		isl_die(isl_basic_map_get_ctx(bmap), isl_error_invalid,
			"expecting integer value", goto error);
	if (isl_basic_map_check_range(bmap, type, pos, 1) < 0)
		goto error;
	pos += isl_basic_map_offset(bmap, type);
	bmap = isl_basic_map_fix_pos(bmap, pos, v->n);
	isl_val_free(v);
	return bmap;
error:
	isl_basic_map_free(bmap);
	isl_val_free(v);
	return NULL;
}

/* Fix the value of the variable at position "pos" of type "type" of "bset"
 * to be equal to "v".
 */
__isl_give isl_basic_set *isl_basic_set_fix_val(__isl_take isl_basic_set *bset,
	enum isl_dim_type type, unsigned pos, __isl_take isl_val *v)
{
	return isl_basic_map_fix_val(bset, type, pos, v);
}

__isl_give isl_basic_set *isl_basic_set_fix_si(__isl_take isl_basic_set *bset,
		enum isl_dim_type type, unsigned pos, int value)
{
	return bset_from_bmap(isl_basic_map_fix_si(bset_to_bmap(bset),
						    type, pos, value));
}

__isl_give isl_basic_set *isl_basic_set_fix(__isl_take isl_basic_set *bset,
		enum isl_dim_type type, unsigned pos, isl_int value)
{
	return bset_from_bmap(isl_basic_map_fix(bset_to_bmap(bset),
						    type, pos, value));
}

/* Remove the basic map at position "i" from "map" if this basic map
 * is (obviously) empty.
 */
static __isl_give isl_map *remove_if_empty(__isl_take isl_map *map, int i)
{
	isl_bool empty;

	if (!map)
		return NULL;

	empty = isl_basic_map_plain_is_empty(map->p[i]);
	if (empty < 0)
		return isl_map_free(map);
	if (!empty)
		return map;

	isl_basic_map_free(map->p[i]);
	map->n--;
	if (i != map->n) {
		map->p[i] = map->p[map->n];
		map = isl_map_unmark_normalized(map);

	}

	return map;
}

/* Perform "fn" on each basic map of "map", where we may not be holding
 * the only reference to "map".
 * In particular, "fn" should be a semantics preserving operation
 * that we want to apply to all copies of "map".  We therefore need
 * to be careful not to modify "map" in a way that breaks "map"
 * in case anything goes wrong.
 */
__isl_give isl_map *isl_map_inline_foreach_basic_map(__isl_take isl_map *map,
	__isl_give isl_basic_map *(*fn)(__isl_take isl_basic_map *bmap))
{
	struct isl_basic_map *bmap;
	int i;

	if (!map)
		return NULL;

	for (i = map->n - 1; i >= 0; --i) {
		bmap = isl_basic_map_copy(map->p[i]);
		bmap = fn(bmap);
		if (!bmap)
			goto error;
		isl_basic_map_free(map->p[i]);
		map->p[i] = bmap;
		map = remove_if_empty(map, i);
		if (!map)
			return NULL;
	}

	return map;
error:
	isl_map_free(map);
	return NULL;
}

__isl_give isl_map *isl_map_fix_si(__isl_take isl_map *map,
		enum isl_dim_type type, unsigned pos, int value)
{
	int i;

	map = isl_map_cow(map);
	if (isl_map_check_range(map, type, pos, 1) < 0)
		return isl_map_free(map);
	for (i = map->n - 1; i >= 0; --i) {
		map->p[i] = isl_basic_map_fix_si(map->p[i], type, pos, value);
		map = remove_if_empty(map, i);
		if (!map)
			return NULL;
	}
	map = isl_map_unmark_normalized(map);
	return map;
}

__isl_give isl_set *isl_set_fix_si(__isl_take isl_set *set,
		enum isl_dim_type type, unsigned pos, int value)
{
	return set_from_map(isl_map_fix_si(set_to_map(set), type, pos, value));
}

__isl_give isl_map *isl_map_fix(__isl_take isl_map *map,
		enum isl_dim_type type, unsigned pos, isl_int value)
{
	int i;

	map = isl_map_cow(map);
	if (isl_map_check_range(map, type, pos, 1) < 0)
		return isl_map_free(map);
	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_fix(map->p[i], type, pos, value);
		if (!map->p[i])
			goto error;
	}
	map = isl_map_unmark_normalized(map);
	return map;
error:
	isl_map_free(map);
	return NULL;
}

__isl_give isl_set *isl_set_fix(__isl_take isl_set *set,
		enum isl_dim_type type, unsigned pos, isl_int value)
{
	return set_from_map(isl_map_fix(set_to_map(set), type, pos, value));
}

/* Fix the value of the variable at position "pos" of type "type" of "map"
 * to be equal to "v".
 */
__isl_give isl_map *isl_map_fix_val(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned pos, __isl_take isl_val *v)
{
	int i;

	map = isl_map_cow(map);
	if (!map || !v)
		goto error;

	if (!isl_val_is_int(v))
		isl_die(isl_map_get_ctx(map), isl_error_invalid,
			"expecting integer value", goto error);
	if (isl_map_check_range(map, type, pos, 1) < 0)
		goto error;
	for (i = map->n - 1; i >= 0; --i) {
		map->p[i] = isl_basic_map_fix_val(map->p[i], type, pos,
							isl_val_copy(v));
		map = remove_if_empty(map, i);
		if (!map)
			goto error;
	}
	map = isl_map_unmark_normalized(map);
	isl_val_free(v);
	return map;
error:
	isl_map_free(map);
	isl_val_free(v);
	return NULL;
}

/* Fix the value of the variable at position "pos" of type "type" of "set"
 * to be equal to "v".
 */
__isl_give isl_set *isl_set_fix_val(__isl_take isl_set *set,
	enum isl_dim_type type, unsigned pos, __isl_take isl_val *v)
{
	return isl_map_fix_val(set, type, pos, v);
}

__isl_give isl_map *isl_map_fix_input_si(__isl_take isl_map *map,
		unsigned input, int value)
{
	return isl_map_fix_si(map, isl_dim_in, input, value);
}

__isl_give isl_set *isl_set_fix_dim_si(__isl_take isl_set *set, unsigned dim,
	int value)
{
	return set_from_map(isl_map_fix_si(set_to_map(set),
						isl_dim_set, dim, value));
}

static __isl_give isl_basic_map *basic_map_bound_si(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos, int value, int upper)
{
	int j;
	isl_size total;

	if (isl_basic_map_check_range(bmap, type, pos, 1) < 0)
		return isl_basic_map_free(bmap);
	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return isl_basic_map_free(bmap);
	pos += isl_basic_map_offset(bmap, type);
	bmap = isl_basic_map_cow(bmap);
	bmap = isl_basic_map_extend_constraints(bmap, 0, 1);
	j = isl_basic_map_alloc_inequality(bmap);
	if (j < 0)
		goto error;
	isl_seq_clr(bmap->ineq[j], 1 + total);
	if (upper) {
		isl_int_set_si(bmap->ineq[j][pos], -1);
		isl_int_set_si(bmap->ineq[j][0], value);
	} else {
		isl_int_set_si(bmap->ineq[j][pos], 1);
		isl_int_set_si(bmap->ineq[j][0], -value);
	}
	bmap = isl_basic_map_simplify(bmap);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_lower_bound_si(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos, int value)
{
	return basic_map_bound_si(bmap, type, pos, value, 0);
}

/* Constrain the values of the given dimension to be no greater than "value".
 */
__isl_give isl_basic_map *isl_basic_map_upper_bound_si(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos, int value)
{
	return basic_map_bound_si(bmap, type, pos, value, 1);
}

static __isl_give isl_map *map_bound_si(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned pos, int value, int upper)
{
	int i;

	map = isl_map_cow(map);
	if (isl_map_check_range(map, type, pos, 1) < 0)
		return isl_map_free(map);
	for (i = 0; i < map->n; ++i) {
		map->p[i] = basic_map_bound_si(map->p[i],
						 type, pos, value, upper);
		if (!map->p[i])
			goto error;
	}
	map = isl_map_unmark_normalized(map);
	return map;
error:
	isl_map_free(map);
	return NULL;
}

__isl_give isl_map *isl_map_lower_bound_si(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned pos, int value)
{
	return map_bound_si(map, type, pos, value, 0);
}

__isl_give isl_map *isl_map_upper_bound_si(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned pos, int value)
{
	return map_bound_si(map, type, pos, value, 1);
}

__isl_give isl_set *isl_set_lower_bound_si(__isl_take isl_set *set,
		enum isl_dim_type type, unsigned pos, int value)
{
	return set_from_map(isl_map_lower_bound_si(set_to_map(set),
							type, pos, value));
}

__isl_give isl_set *isl_set_upper_bound_si(__isl_take isl_set *set,
	enum isl_dim_type type, unsigned pos, int value)
{
	return isl_map_upper_bound_si(set, type, pos, value);
}

/* Bound the given variable of "bmap" from below (or above is "upper"
 * is set) to "value".
 */
static __isl_give isl_basic_map *basic_map_bound(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos, isl_int value, int upper)
{
	int j;
	isl_size total;

	if (isl_basic_map_check_range(bmap, type, pos, 1) < 0)
		return isl_basic_map_free(bmap);
	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return isl_basic_map_free(bmap);
	pos += isl_basic_map_offset(bmap, type);
	bmap = isl_basic_map_cow(bmap);
	bmap = isl_basic_map_extend_constraints(bmap, 0, 1);
	j = isl_basic_map_alloc_inequality(bmap);
	if (j < 0)
		goto error;
	isl_seq_clr(bmap->ineq[j], 1 + total);
	if (upper) {
		isl_int_set_si(bmap->ineq[j][pos], -1);
		isl_int_set(bmap->ineq[j][0], value);
	} else {
		isl_int_set_si(bmap->ineq[j][pos], 1);
		isl_int_neg(bmap->ineq[j][0], value);
	}
	bmap = isl_basic_map_simplify(bmap);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap);
	return NULL;
}

/* Bound the given variable of "map" from below (or above is "upper"
 * is set) to "value".
 */
static __isl_give isl_map *map_bound(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned pos, isl_int value, int upper)
{
	int i;

	map = isl_map_cow(map);
	if (isl_map_check_range(map, type, pos, 1) < 0)
		return isl_map_free(map);
	for (i = map->n - 1; i >= 0; --i) {
		map->p[i] = basic_map_bound(map->p[i], type, pos, value, upper);
		map = remove_if_empty(map, i);
		if (!map)
			return NULL;
	}
	map = isl_map_unmark_normalized(map);
	return map;
}

__isl_give isl_map *isl_map_lower_bound(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned pos, isl_int value)
{
	return map_bound(map, type, pos, value, 0);
}

__isl_give isl_map *isl_map_upper_bound(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned pos, isl_int value)
{
	return map_bound(map, type, pos, value, 1);
}

__isl_give isl_set *isl_set_lower_bound(__isl_take isl_set *set,
	enum isl_dim_type type, unsigned pos, isl_int value)
{
	return isl_map_lower_bound(set, type, pos, value);
}

__isl_give isl_set *isl_set_upper_bound(__isl_take isl_set *set,
	enum isl_dim_type type, unsigned pos, isl_int value)
{
	return isl_map_upper_bound(set, type, pos, value);
}

/* Force the values of the variable at position "pos" of type "type" of "map"
 * to be no smaller than "value".
 */
__isl_give isl_map *isl_map_lower_bound_val(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned pos, __isl_take isl_val *value)
{
	if (!value)
		goto error;
	if (!isl_val_is_int(value))
		isl_die(isl_map_get_ctx(map), isl_error_invalid,
			"expecting integer value", goto error);
	map = isl_map_lower_bound(map, type, pos, value->n);
	isl_val_free(value);
	return map;
error:
	isl_val_free(value);
	isl_map_free(map);
	return NULL;
}

/* Force the values of the variable at position "pos" of type "type" of "set"
 * to be no smaller than "value".
 */
__isl_give isl_set *isl_set_lower_bound_val(__isl_take isl_set *set,
	enum isl_dim_type type, unsigned pos, __isl_take isl_val *value)
{
	isl_map *map;

	map = set_to_map(set);
	return set_from_map(isl_map_lower_bound_val(map, type, pos, value));
}

/* Force the values of the variable at position "pos" of type "type" of "map"
 * to be no greater than "value".
 */
__isl_give isl_map *isl_map_upper_bound_val(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned pos, __isl_take isl_val *value)
{
	if (!value)
		goto error;
	if (!isl_val_is_int(value))
		isl_die(isl_map_get_ctx(map), isl_error_invalid,
			"expecting integer value", goto error);
	map = isl_map_upper_bound(map, type, pos, value->n);
	isl_val_free(value);
	return map;
error:
	isl_val_free(value);
	isl_map_free(map);
	return NULL;
}

/* Force the values of the variable at position "pos" of type "type" of "set"
 * to be no greater than "value".
 */
__isl_give isl_set *isl_set_upper_bound_val(__isl_take isl_set *set,
	enum isl_dim_type type, unsigned pos, __isl_take isl_val *value)
{
	isl_map *map;

	map = set_to_map(set);
	return set_from_map(isl_map_upper_bound_val(map, type, pos, value));
}

/* If "mv" has an explicit domain, then intersect the domain of "map"
 * with this explicit domain.
 *
 * An isl_multi_val object never has an explicit domain,
 * so simply return "map".
 */
static __isl_give isl_map *isl_map_intersect_multi_val_explicit_domain(
	__isl_take isl_map *map, __isl_keep isl_multi_val *mv)
{
	return map;
}

#undef BASE
#define BASE	val
#include "isl_map_bound_templ.c"

/* Apply "map_bound" to "set" with the corresponding value in "bound"
 * for each set dimension, by treating the set as a map.
 */
static __isl_give isl_set *set_bound_multi_val(__isl_take isl_set *set,
	__isl_take isl_multi_val *bound,
	__isl_give isl_map *map_bound(__isl_take isl_map *map,
		unsigned pos, __isl_take isl_val *value))
{
	isl_map *map;

	map = set_to_map(set);
	return set_from_map(map_bound_multi_val(map, bound, map_bound));
}

#undef BASE
#define BASE	pw_aff
#include "isl_map_bound_templ.c"

/* Apply "map_bound" to "set" with the corresponding value in "bound"
 * for each set dimension, by converting the set and the bound
 * to objects living in a map space.
 */
static __isl_give isl_set *set_bound_multi_pw_aff(__isl_take isl_set *set,
	__isl_take isl_multi_pw_aff *bound,
	__isl_give isl_map *set_bound(__isl_take isl_map *map,
		unsigned pos, __isl_take TYPE *value))
{
	isl_map *map;

	map = isl_map_from_range(set);
	bound = isl_multi_pw_aff_from_range(bound);
	map = map_bound_multi_pw_aff(map, bound, set_bound);
	return isl_map_range(map);
}

/* Wrapper around isl_map_lower_bound_val for use in map_bound_multi_val,
 * setting a bound on the given output dimension.
 */
static __isl_give isl_map *map_lower_bound_val(__isl_take isl_map *map,
	unsigned pos, __isl_take isl_val *v)
{
	return isl_map_lower_bound_val(map, isl_dim_out, pos, v);
}

/* Force the values of the set dimensions of "set"
 * to be no smaller than the corresponding values in "lower".
 */
__isl_give isl_set *isl_set_lower_bound_multi_val(__isl_take isl_set *set,
	__isl_take isl_multi_val *lower)
{
	return set_bound_multi_val(set, lower, &map_lower_bound_val);
}

/* Wrapper around isl_map_upper_bound_val for use in map_bound_multi_val,
 * setting a bound on the given output dimension.
 */
static __isl_give isl_map *map_upper_bound_val(__isl_take isl_map *map,
	unsigned pos, __isl_take isl_val *v)
{
	return isl_map_upper_bound_val(map, isl_dim_out, pos, v);
}

/* Force the values of the set dimensions of "set"
 * to be no greater than the corresponding values in "upper".
 */
__isl_give isl_set *isl_set_upper_bound_multi_val(__isl_take isl_set *set,
	__isl_take isl_multi_val *upper)
{
	return set_bound_multi_val(set, upper, &map_upper_bound_val);
}

/* Force the symbolic constant expression "bound"
 * to satisfy the relation "order" with respect to
 * the output variable at position "pos" of "map".
 *
 * Create an affine expression representing the output variable
 * in terms of the range and
 * compare it using "order" to "bound" (defined on the domain).
 * The result is a relation between elements in domain and range that
 * can be intersected with "map".
 */
static __isl_give isl_map *map_bound_pw_aff(__isl_take isl_map *map,
	unsigned pos, __isl_take isl_pw_aff *bound,
	__isl_give isl_map *(*order)(__isl_take isl_pw_aff *pa1,
		__isl_take isl_pw_aff *pa2))
{
	isl_space *space;
	isl_local_space *ls;
	isl_pw_aff *var;

	space = isl_space_range(isl_map_get_space(map));
	ls = isl_local_space_from_space(space);
	var = isl_pw_aff_var_on_domain(ls, isl_dim_set, pos);
	map = isl_map_intersect(map, order(bound, var));
	return map;
}

/* Force the values of the output variable at position "pos" of "map"
 * to be no smaller than the symbolic constant expression "lower".
 */
static __isl_give isl_map *map_lower_bound_pw_aff(__isl_take isl_map *map,
	unsigned pos, __isl_take isl_pw_aff *lower)
{
	return map_bound_pw_aff(map, pos, lower, &isl_pw_aff_le_map);
}

/* Force the values of the output variable at position "pos" of "map"
 * to be no greater than the symbolic constant expression "upper".
 */
static __isl_give isl_map *map_upper_bound_pw_aff(__isl_take isl_map *map,
	unsigned pos, __isl_take isl_pw_aff *upper)
{
	return map_bound_pw_aff(map, pos, upper, &isl_pw_aff_ge_map);
}

/* Force the values of the set dimensions of "set"
 * to be no smaller than the corresponding constant symbolic expressions
 * in "lower".
 */
__isl_give isl_set *isl_set_lower_bound_multi_pw_aff(__isl_take isl_set *set,
	__isl_take isl_multi_pw_aff *lower)
{
	return set_bound_multi_pw_aff(set, lower, &map_lower_bound_pw_aff);
}

/* Force the values of the set dimensions of "set"
 * to be no greater than the corresponding constant symbolic expressions
 * in "upper".
 */
__isl_give isl_set *isl_set_upper_bound_multi_pw_aff(__isl_take isl_set *set,
	__isl_take isl_multi_pw_aff *upper)
{
	return set_bound_multi_pw_aff(set, upper, &map_upper_bound_pw_aff);
}

/* Force the values of the output dimensions of "map"
 * to be no smaller than the corresponding constant symbolic expressions
 * in "lower".
 */
__isl_give isl_map *isl_map_lower_bound_multi_pw_aff(__isl_take isl_map *map,
	__isl_take isl_multi_pw_aff *lower)
{
	return map_bound_multi_pw_aff(map, lower, &map_lower_bound_pw_aff);
}

/* Force the values of the output dimensions of "map"
 * to be no greater than the corresponding constant symbolic expressions
 * in "upper".
 */
__isl_give isl_map *isl_map_upper_bound_multi_pw_aff(__isl_take isl_map *map,
	__isl_take isl_multi_pw_aff *upper)
{
	return map_bound_multi_pw_aff(map, upper, &map_upper_bound_pw_aff);
}

/* Bound the given variable of "bset" from below (or above is "upper"
 * is set) to "value".
 */
static __isl_give isl_basic_set *isl_basic_set_bound(
	__isl_take isl_basic_set *bset, enum isl_dim_type type, unsigned pos,
	isl_int value, int upper)
{
	return bset_from_bmap(basic_map_bound(bset_to_bmap(bset),
						type, pos, value, upper));
}

/* Bound the given variable of "bset" from below (or above is "upper"
 * is set) to "value".
 */
static __isl_give isl_basic_set *isl_basic_set_bound_val(
	__isl_take isl_basic_set *bset, enum isl_dim_type type, unsigned pos,
	__isl_take isl_val *value, int upper)
{
	if (!value)
		goto error;
	if (!isl_val_is_int(value))
		isl_die(isl_basic_set_get_ctx(bset), isl_error_invalid,
			"expecting integer value", goto error);
	bset = isl_basic_set_bound(bset, type, pos, value->n, upper);
	isl_val_free(value);
	return bset;
error:
	isl_val_free(value);
	isl_basic_set_free(bset);
	return NULL;
}

/* Bound the given variable of "bset" from below to "value".
 */
__isl_give isl_basic_set *isl_basic_set_lower_bound_val(
	__isl_take isl_basic_set *bset, enum isl_dim_type type, unsigned pos,
	__isl_take isl_val *value)
{
	return isl_basic_set_bound_val(bset, type, pos, value, 0);
}

/* Bound the given variable of "bset" from above to "value".
 */
__isl_give isl_basic_set *isl_basic_set_upper_bound_val(
	__isl_take isl_basic_set *bset, enum isl_dim_type type, unsigned pos,
	__isl_take isl_val *value)
{
	return isl_basic_set_bound_val(bset, type, pos, value, 1);
}

__isl_give isl_map *isl_map_reverse(__isl_take isl_map *map)
{
	return isl_map_transform(map, &isl_space_reverse,
					&isl_basic_map_reverse);
}

/* Given a map A -> (B -> C), return the corresponding map A -> (C -> B).
 */
__isl_give isl_map *isl_map_range_reverse(__isl_take isl_map *map)
{
	return isl_map_transform(map, &isl_space_range_reverse,
					&isl_basic_map_range_reverse);
}

#undef TYPE
#define TYPE	isl_pw_multi_aff
#undef SUFFIX
#define SUFFIX	_pw_multi_aff
#undef EMPTY
#define EMPTY	isl_pw_multi_aff_empty
#undef ADD
#define ADD	isl_pw_multi_aff_union_add
#include "isl_map_lexopt_templ.c"

/* Given a map "map", compute the lexicographically minimal
 * (or maximal) image element for each domain element in dom,
 * in the form of an isl_pw_multi_aff.
 * If "empty" is not NULL, then set *empty to those elements in dom that
 * do not have an image element.
 * If "flags" includes ISL_OPT_FULL, then "dom" is NULL and the optimum
 * should be computed over the domain of "map".  "empty" is also NULL
 * in this case.
 *
 * We first compute the lexicographically minimal or maximal element
 * in the first basic map.  This results in a partial solution "res"
 * and a subset "todo" of dom that still need to be handled.
 * We then consider each of the remaining maps in "map" and successively
 * update both "res" and "todo".
 * If "empty" is NULL, then the todo sets are not needed and therefore
 * also not computed.
 */
static __isl_give isl_pw_multi_aff *isl_map_partial_lexopt_aligned_pw_multi_aff(
	__isl_take isl_map *map, __isl_take isl_set *dom,
	__isl_give isl_set **empty, unsigned flags)
{
	int i;
	int full;
	isl_pw_multi_aff *res;
	isl_set *todo;

	full = ISL_FL_ISSET(flags, ISL_OPT_FULL);
	if (!map || (!full && !dom))
		goto error;

	if (isl_map_plain_is_empty(map)) {
		if (empty)
			*empty = dom;
		else
			isl_set_free(dom);
		return isl_pw_multi_aff_from_map(map);
	}

	res = basic_map_partial_lexopt_pw_multi_aff(
					    isl_basic_map_copy(map->p[0]),
					    isl_set_copy(dom), empty, flags);

	if (empty)
		todo = *empty;
	for (i = 1; i < map->n; ++i) {
		isl_pw_multi_aff *res_i;

		res_i = basic_map_partial_lexopt_pw_multi_aff(
					    isl_basic_map_copy(map->p[i]),
					    isl_set_copy(dom), empty, flags);

		if (ISL_FL_ISSET(flags, ISL_OPT_MAX))
			res = isl_pw_multi_aff_union_lexmax(res, res_i);
		else
			res = isl_pw_multi_aff_union_lexmin(res, res_i);

		if (empty)
			todo = isl_set_intersect(todo, *empty);
	}

	isl_set_free(dom);
	isl_map_free(map);

	if (empty)
		*empty = todo;

	return res;
error:
	if (empty)
		*empty = NULL;
	isl_set_free(dom);
	isl_map_free(map);
	return NULL;
}

#undef TYPE
#define TYPE	isl_map
#undef SUFFIX
#define SUFFIX
#undef EMPTY
#define EMPTY	isl_map_empty
#undef ADD
#define ADD	isl_map_union_disjoint
#include "isl_map_lexopt_templ.c"

/* Given a map "map", compute the lexicographically minimal
 * (or maximal) image element for each domain element in "dom",
 * in the form of an isl_map.
 * If "empty" is not NULL, then set *empty to those elements in "dom" that
 * do not have an image element.
 * If "flags" includes ISL_OPT_FULL, then "dom" is NULL and the optimum
 * should be computed over the domain of "map".  "empty" is also NULL
 * in this case.
 *
 * If the input consists of more than one disjunct, then first
 * compute the desired result in the form of an isl_pw_multi_aff and
 * then convert that into an isl_map.
 *
 * This function used to have an explicit implementation in terms
 * of isl_maps, but it would continually intersect the domains of
 * partial results with the complement of the domain of the next
 * partial solution, potentially leading to an explosion in the number
 * of disjuncts if there are several disjuncts in the input.
 * An even earlier implementation of this function would look for
 * better results in the domain of the partial result and for extra
 * results in the complement of this domain, which would lead to
 * even more splintering.
 */
static __isl_give isl_map *isl_map_partial_lexopt_aligned(
	__isl_take isl_map *map, __isl_take isl_set *dom,
	__isl_give isl_set **empty, unsigned flags)
{
	int full;
	struct isl_map *res;
	isl_pw_multi_aff *pma;

	full = ISL_FL_ISSET(flags, ISL_OPT_FULL);
	if (!map || (!full && !dom))
		goto error;

	if (isl_map_plain_is_empty(map)) {
		if (empty)
			*empty = dom;
		else
			isl_set_free(dom);
		return map;
	}

	if (map->n == 1) {
		res = basic_map_partial_lexopt(isl_basic_map_copy(map->p[0]),
						dom, empty, flags);
		isl_map_free(map);
		return res;
	}

	pma = isl_map_partial_lexopt_aligned_pw_multi_aff(map, dom, empty,
							flags);
	return isl_map_from_pw_multi_aff_internal(pma);
error:
	if (empty)
		*empty = NULL;
	isl_set_free(dom);
	isl_map_free(map);
	return NULL;
}

__isl_give isl_map *isl_map_partial_lexmax(
		__isl_take isl_map *map, __isl_take isl_set *dom,
		__isl_give isl_set **empty)
{
	return isl_map_partial_lexopt(map, dom, empty, ISL_OPT_MAX);
}

__isl_give isl_map *isl_map_partial_lexmin(
		__isl_take isl_map *map, __isl_take isl_set *dom,
		__isl_give isl_set **empty)
{
	return isl_map_partial_lexopt(map, dom, empty, 0);
}

__isl_give isl_set *isl_set_partial_lexmin(
		__isl_take isl_set *set, __isl_take isl_set *dom,
		__isl_give isl_set **empty)
{
	return set_from_map(isl_map_partial_lexmin(set_to_map(set),
						    dom, empty));
}

__isl_give isl_set *isl_set_partial_lexmax(
		__isl_take isl_set *set, __isl_take isl_set *dom,
		__isl_give isl_set **empty)
{
	return set_from_map(isl_map_partial_lexmax(set_to_map(set),
						    dom, empty));
}

/* Compute the lexicographic minimum (or maximum if "flags" includes
 * ISL_OPT_MAX) of "bset" over its parametric domain.
 */
__isl_give isl_set *isl_basic_set_lexopt(__isl_take isl_basic_set *bset,
	unsigned flags)
{
	return isl_basic_map_lexopt(bset, flags);
}

__isl_give isl_map *isl_basic_map_lexmax(__isl_take isl_basic_map *bmap)
{
	return isl_basic_map_lexopt(bmap, ISL_OPT_MAX);
}

__isl_give isl_set *isl_basic_set_lexmin(__isl_take isl_basic_set *bset)
{
	return set_from_map(isl_basic_map_lexmin(bset_to_bmap(bset)));
}

__isl_give isl_set *isl_basic_set_lexmax(__isl_take isl_basic_set *bset)
{
	return set_from_map(isl_basic_map_lexmax(bset_to_bmap(bset)));
}

/* Compute the lexicographic minimum of "bset" over its parametric domain
 * for the purpose of quantifier elimination.
 * That is, find an explicit representation for all the existentially
 * quantified variables in "bset" by computing their lexicographic
 * minimum.
 */
static __isl_give isl_set *isl_basic_set_lexmin_compute_divs(
	__isl_take isl_basic_set *bset)
{
	return isl_basic_set_lexopt(bset, ISL_OPT_QE);
}

/* Given a basic map with one output dimension, compute the minimum or
 * maximum of that dimension as an isl_pw_aff.
 *
 * Compute the optimum as a lexicographic optimum over the single
 * output dimension and extract the single isl_pw_aff from the result.
 */
static __isl_give isl_pw_aff *basic_map_dim_opt(__isl_keep isl_basic_map *bmap,
	int max)
{
	isl_pw_multi_aff *pma;
	isl_pw_aff *pwaff;

	bmap = isl_basic_map_copy(bmap);
	pma = isl_basic_map_lexopt_pw_multi_aff(bmap, max ? ISL_OPT_MAX : 0);
	pwaff = isl_pw_multi_aff_get_pw_aff(pma, 0);
	isl_pw_multi_aff_free(pma);

	return pwaff;
}

/* Compute the minimum or maximum of the given output dimension
 * as a function of the parameters and the input dimensions,
 * but independently of the other output dimensions.
 *
 * We first project out the other output dimension and then compute
 * the "lexicographic" maximum in each basic map, combining the results
 * using isl_pw_aff_union_max.
 */
static __isl_give isl_pw_aff *map_dim_opt(__isl_take isl_map *map, int pos,
	int max)
{
	int i;
	isl_pw_aff *pwaff;
	isl_size n_out;

	n_out = isl_map_dim(map, isl_dim_out);
	if (n_out < 0)
		map = isl_map_free(map);
	map = isl_map_project_out(map, isl_dim_out, pos + 1, n_out - (pos + 1));
	map = isl_map_project_out(map, isl_dim_out, 0, pos);
	if (!map)
		return NULL;

	if (map->n == 0) {
		isl_space *space = isl_map_get_space(map);
		isl_map_free(map);
		return isl_pw_aff_empty(space);
	}

	pwaff = basic_map_dim_opt(map->p[0], max);
	for (i = 1; i < map->n; ++i) {
		isl_pw_aff *pwaff_i;

		pwaff_i = basic_map_dim_opt(map->p[i], max);
		pwaff = isl_pw_aff_union_opt(pwaff, pwaff_i, max);
	}

	isl_map_free(map);

	return pwaff;
}

/* Compute the minimum of the given output dimension as a function of the
 * parameters and input dimensions, but independently of
 * the other output dimensions.
 */
__isl_give isl_pw_aff *isl_map_dim_min(__isl_take isl_map *map, int pos)
{
	return map_dim_opt(map, pos, 0);
}

/* Compute the maximum of the given output dimension as a function of the
 * parameters and input dimensions, but independently of
 * the other output dimensions.
 */
__isl_give isl_pw_aff *isl_map_dim_max(__isl_take isl_map *map, int pos)
{
	return map_dim_opt(map, pos, 1);
}

/* Compute the minimum or maximum of the given set dimension
 * as a function of the parameters,
 * but independently of the other set dimensions.
 */
static __isl_give isl_pw_aff *set_dim_opt(__isl_take isl_set *set, int pos,
	int max)
{
	return map_dim_opt(set, pos, max);
}

/* Compute the maximum of the given set dimension as a function of the
 * parameters, but independently of the other set dimensions.
 */
__isl_give isl_pw_aff *isl_set_dim_max(__isl_take isl_set *set, int pos)
{
	return set_dim_opt(set, pos, 1);
}

/* Compute the minimum of the given set dimension as a function of the
 * parameters, but independently of the other set dimensions.
 */
__isl_give isl_pw_aff *isl_set_dim_min(__isl_take isl_set *set, int pos)
{
	return set_dim_opt(set, pos, 0);
}

/* Apply a preimage specified by "mat" on the parameters of "bset".
 * bset is assumed to have only parameters and divs.
 */
static __isl_give isl_basic_set *basic_set_parameter_preimage(
	__isl_take isl_basic_set *bset, __isl_take isl_mat *mat)
{
	isl_size nparam;

	nparam = isl_basic_set_dim(bset, isl_dim_param);
	if (nparam < 0 || !mat)
		goto error;

	bset->dim = isl_space_cow(bset->dim);
	if (!bset->dim)
		goto error;

	isl_assert(bset->ctx, mat->n_row == 1 + nparam, goto error);

	bset->dim->nparam = 0;
	bset->dim->n_out = nparam;
	bset = isl_basic_set_preimage(bset, mat);
	if (bset) {
		bset->dim->nparam = bset->dim->n_out;
		bset->dim->n_out = 0;
	}
	return bset;
error:
	isl_mat_free(mat);
	isl_basic_set_free(bset);
	return NULL;
}

/* Apply a preimage specified by "mat" on the parameters of "set".
 * set is assumed to have only parameters and divs.
 */
static __isl_give isl_set *set_parameter_preimage(__isl_take isl_set *set,
	__isl_take isl_mat *mat)
{
	isl_space *space;
	isl_size nparam;

	nparam = isl_set_dim(set, isl_dim_param);
	if (nparam < 0 || !mat)
		goto error;

	if (mat->n_row != 1 + nparam)
		isl_die(isl_set_get_ctx(set), isl_error_internal,
			"unexpected number of rows", goto error);

	space = isl_set_get_space(set);
	space = isl_space_move_dims(space, isl_dim_set, 0,
				    isl_dim_param, 0, nparam);
	set = isl_set_reset_space(set, space);
	set = isl_set_preimage(set, mat);
	nparam = isl_set_dim(set, isl_dim_out);
	if (nparam < 0)
		set = isl_set_free(set);
	space = isl_set_get_space(set);
	space = isl_space_move_dims(space, isl_dim_param, 0,
				    isl_dim_out, 0, nparam);
	set = isl_set_reset_space(set, space);
	return set;
error:
	isl_mat_free(mat);
	isl_set_free(set);
	return NULL;
}

/* Intersect the basic set "bset" with the affine space specified by the
 * equalities in "eq".
 */
static __isl_give isl_basic_set *basic_set_append_equalities(
	__isl_take isl_basic_set *bset, __isl_take isl_mat *eq)
{
	int i, k;
	unsigned len;

	if (!bset || !eq)
		goto error;

	bset = isl_basic_set_extend(bset, 0, eq->n_row, 0);
	if (!bset)
		goto error;

	len = isl_basic_set_offset(bset, isl_dim_div) + bset->extra;
	for (i = 0; i < eq->n_row; ++i) {
		k = isl_basic_set_alloc_equality(bset);
		if (k < 0)
			goto error;
		isl_seq_cpy(bset->eq[k], eq->row[i], eq->n_col);
		isl_seq_clr(bset->eq[k] + eq->n_col, len - eq->n_col);
	}
	isl_mat_free(eq);

	bset = isl_basic_set_gauss(bset, NULL);
	bset = isl_basic_set_finalize(bset);

	return bset;
error:
	isl_mat_free(eq);
	isl_basic_set_free(bset);
	return NULL;
}

/* Intersect the set "set" with the affine space specified by the
 * equalities in "eq".
 */
static __isl_give isl_set *set_append_equalities(__isl_take isl_set *set,
	__isl_take isl_mat *eq)
{
	int i;

	if (!set || !eq)
		goto error;

	for (i = 0; i < set->n; ++i) {
		set->p[i] = basic_set_append_equalities(set->p[i],
					isl_mat_copy(eq));
		if (!set->p[i])
			goto error;
	}
	isl_mat_free(eq);
	return set;
error:
	isl_mat_free(eq);
	isl_set_free(set);
	return NULL;
}

/* Given a basic set "bset" that only involves parameters and existentially
 * quantified variables, return the index of the first equality
 * that only involves parameters.  If there is no such equality then
 * return bset->n_eq.
 *
 * This function assumes that isl_basic_set_gauss has been called on "bset".
 */
static int first_parameter_equality(__isl_keep isl_basic_set *bset)
{
	int i, j;
	isl_size nparam, n_div;

	nparam = isl_basic_set_dim(bset, isl_dim_param);
	n_div = isl_basic_set_dim(bset, isl_dim_div);
	if (nparam < 0 || n_div < 0)
		return -1;

	for (i = 0, j = n_div - 1; i < bset->n_eq && j >= 0; --j) {
		if (!isl_int_is_zero(bset->eq[i][1 + nparam + j]))
			++i;
	}

	return i;
}

/* Compute an explicit representation for the existentially quantified
 * variables in "bset" by computing the "minimal value" of the set
 * variables.  Since there are no set variables, the computation of
 * the minimal value essentially computes an explicit representation
 * of the non-empty part(s) of "bset".
 *
 * The input only involves parameters and existentially quantified variables.
 * All equalities among parameters have been removed.
 *
 * Since the existentially quantified variables in the result are in general
 * going to be different from those in the input, we first replace
 * them by the minimal number of variables based on their equalities.
 * This should simplify the parametric integer programming.
 */
static __isl_give isl_set *base_compute_divs(__isl_take isl_basic_set *bset)
{
	isl_morph *morph1, *morph2;
	isl_set *set;
	isl_size n;

	if (!bset)
		return NULL;
	if (bset->n_eq == 0)
		return isl_basic_set_lexmin_compute_divs(bset);

	morph1 = isl_basic_set_parameter_compression(bset);
	bset = isl_morph_basic_set(isl_morph_copy(morph1), bset);
	bset = isl_basic_set_lift(bset);
	morph2 = isl_basic_set_variable_compression(bset, isl_dim_set);
	bset = isl_morph_basic_set(morph2, bset);
	n = isl_basic_set_dim(bset, isl_dim_set);
	if (n < 0)
		bset = isl_basic_set_free(bset);
	bset = isl_basic_set_project_out(bset, isl_dim_set, 0, n);

	set = isl_basic_set_lexmin_compute_divs(bset);

	set = isl_morph_set(isl_morph_inverse(morph1), set);

	return set;
}

/* Project the given basic set onto its parameter domain, possibly introducing
 * new, explicit, existential variables in the constraints.
 * The input has parameters and (possibly implicit) existential variables.
 * The output has the same parameters, but only
 * explicit existentially quantified variables.
 *
 * The actual projection is performed by pip, but pip doesn't seem
 * to like equalities very much, so we first remove the equalities
 * among the parameters by performing a variable compression on
 * the parameters.  Afterward, an inverse transformation is performed
 * and the equalities among the parameters are inserted back in.
 *
 * The variable compression on the parameters may uncover additional
 * equalities that were only implicit before.  We therefore check
 * if there are any new parameter equalities in the result and
 * if so recurse.  The removal of parameter equalities is required
 * for the parameter compression performed by base_compute_divs.
 */
static __isl_give isl_set *parameter_compute_divs(
	__isl_take isl_basic_set *bset)
{
	int i;
	struct isl_mat *eq;
	struct isl_mat *T, *T2;
	struct isl_set *set;
	isl_size nparam;

	bset = isl_basic_set_cow(bset);
	if (!bset)
		return NULL;

	if (bset->n_eq == 0)
		return base_compute_divs(bset);

	bset = isl_basic_set_gauss(bset, NULL);
	if (!bset)
		return NULL;
	if (isl_basic_set_plain_is_empty(bset))
		return isl_set_from_basic_set(bset);

	i = first_parameter_equality(bset);
	if (i == bset->n_eq)
		return base_compute_divs(bset);

	nparam = isl_basic_set_dim(bset, isl_dim_param);
	if (nparam < 0)
		return isl_set_from_basic_set(isl_basic_set_free(bset));
	eq = isl_mat_sub_alloc6(bset->ctx, bset->eq, i, bset->n_eq - i,
		0, 1 + nparam);
	eq = isl_mat_cow(eq);
	T = isl_mat_variable_compression(isl_mat_copy(eq), &T2);
	if (T && T->n_col == 0) {
		isl_mat_free(T);
		isl_mat_free(T2);
		isl_mat_free(eq);
		bset = isl_basic_set_set_to_empty(bset);
		return isl_set_from_basic_set(bset);
	}
	bset = basic_set_parameter_preimage(bset, T);

	i = first_parameter_equality(bset);
	if (!bset)
		set = NULL;
	else if (i == bset->n_eq)
		set = base_compute_divs(bset);
	else
		set = parameter_compute_divs(bset);
	set = set_parameter_preimage(set, T2);
	set = set_append_equalities(set, eq);
	return set;
}

/* Insert the divs from "ls" before those of "bmap".
 *
 * The number of columns is not changed, which means that the last
 * dimensions of "bmap" are being reintepreted as the divs from "ls".
 * The caller is responsible for removing the same number of dimensions
 * from the space of "bmap".
 */
static __isl_give isl_basic_map *insert_divs_from_local_space(
	__isl_take isl_basic_map *bmap, __isl_keep isl_local_space *ls)
{
	int i;
	isl_size n_div;
	int old_n_div;

	n_div = isl_local_space_dim(ls, isl_dim_div);
	if (n_div < 0)
		return isl_basic_map_free(bmap);
	if (n_div == 0)
		return bmap;

	old_n_div = bmap->n_div;
	bmap = insert_div_rows(bmap, n_div);
	if (!bmap)
		return NULL;

	for (i = 0; i < n_div; ++i) {
		isl_seq_cpy(bmap->div[i], ls->div->row[i], ls->div->n_col);
		isl_seq_clr(bmap->div[i] + ls->div->n_col, old_n_div);
	}

	return bmap;
}

/* Replace the space of "bmap" by the space and divs of "ls".
 *
 * If "ls" has any divs, then we simplify the result since we may
 * have discovered some additional equalities that could simplify
 * the div expressions.
 */
static __isl_give isl_basic_map *basic_replace_space_by_local_space(
	__isl_take isl_basic_map *bmap, __isl_take isl_local_space *ls)
{
	isl_size n_div;

	bmap = isl_basic_map_cow(bmap);
	n_div = isl_local_space_dim(ls, isl_dim_div);
	if (!bmap || n_div < 0)
		goto error;

	bmap = insert_divs_from_local_space(bmap, ls);
	if (!bmap)
		goto error;

	isl_space_free(bmap->dim);
	bmap->dim = isl_local_space_get_space(ls);
	if (!bmap->dim)
		goto error;

	isl_local_space_free(ls);
	if (n_div > 0)
		bmap = isl_basic_map_simplify(bmap);
	bmap = isl_basic_map_finalize(bmap);
	return bmap;
error:
	isl_basic_map_free(bmap);
	isl_local_space_free(ls);
	return NULL;
}

/* Replace the space of "map" by the space and divs of "ls".
 */
static __isl_give isl_map *replace_space_by_local_space(__isl_take isl_map *map,
	__isl_take isl_local_space *ls)
{
	int i;

	map = isl_map_cow(map);
	if (!map || !ls)
		goto error;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = basic_replace_space_by_local_space(map->p[i],
						    isl_local_space_copy(ls));
		if (!map->p[i])
			goto error;
	}
	isl_space_free(isl_map_take_space(map));
	map = isl_map_restore_space(map, isl_local_space_get_space(ls));

	isl_local_space_free(ls);
	return map;
error:
	isl_local_space_free(ls);
	isl_map_free(map);
	return NULL;
}

/* Compute an explicit representation for the existentially
 * quantified variables for which do not know any explicit representation yet.
 *
 * We first sort the existentially quantified variables so that the
 * existentially quantified variables for which we already have an explicit
 * representation are placed before those for which we do not.
 * The input dimensions, the output dimensions and the existentially
 * quantified variables for which we already have an explicit
 * representation are then turned into parameters.
 * compute_divs returns a map with the same parameters and
 * no input or output dimensions and the dimension specification
 * is reset to that of the input, including the existentially quantified
 * variables for which we already had an explicit representation.
 */
static __isl_give isl_map *compute_divs(__isl_take isl_basic_map *bmap)
{
	struct isl_basic_set *bset;
	struct isl_set *set;
	struct isl_map *map;
	isl_space *space;
	isl_local_space *ls;
	isl_size nparam;
	isl_size n_in;
	isl_size n_out;
	int n_known;
	int i;

	bmap = isl_basic_map_sort_divs(bmap);
	bmap = isl_basic_map_cow(bmap);
	if (!bmap)
		return NULL;

	n_known = isl_basic_map_first_unknown_div(bmap);
	nparam = isl_basic_map_dim(bmap, isl_dim_param);
	n_in = isl_basic_map_dim(bmap, isl_dim_in);
	n_out = isl_basic_map_dim(bmap, isl_dim_out);
	if (n_known < 0 || nparam < 0 || n_in < 0 || n_out < 0)
		return isl_map_from_basic_map(isl_basic_map_free(bmap));

	space = isl_space_set_alloc(bmap->ctx,
				    nparam + n_in + n_out + n_known, 0);
	if (!space)
		goto error;

	ls = isl_basic_map_get_local_space(bmap);
	ls = isl_local_space_drop_dims(ls, isl_dim_div,
					n_known, bmap->n_div - n_known);
	if (n_known > 0) {
		for (i = n_known; i < bmap->n_div; ++i)
			swap_div(bmap, i - n_known, i);
		bmap->n_div -= n_known;
		bmap->extra -= n_known;
	}
	bmap = isl_basic_map_reset_space(bmap, space);
	bset = bset_from_bmap(bmap);

	set = parameter_compute_divs(bset);
	map = set_to_map(set);
	map = replace_space_by_local_space(map, ls);

	return map;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

/* Remove the explicit representation of local variable "div",
 * if there is any.
 */
__isl_give isl_basic_map *isl_basic_map_mark_div_unknown(
	__isl_take isl_basic_map *bmap, int div)
{
	isl_bool unknown;

	unknown = isl_basic_map_div_is_marked_unknown(bmap, div);
	if (unknown < 0)
		return isl_basic_map_free(bmap);
	if (unknown)
		return bmap;

	bmap = isl_basic_map_cow(bmap);
	if (!bmap)
		return NULL;
	isl_int_set_si(bmap->div[div][0], 0);
	return bmap;
}

/* Is local variable "div" of "bmap" marked as not having an explicit
 * representation?
 * Note that even if "div" is not marked in this way and therefore
 * has an explicit representation, this representation may still
 * depend (indirectly) on other local variables that do not
 * have an explicit representation.
 */
isl_bool isl_basic_map_div_is_marked_unknown(__isl_keep isl_basic_map *bmap,
	int div)
{
	if (isl_basic_map_check_range(bmap, isl_dim_div, div, 1) < 0)
		return isl_bool_error;
	return isl_int_is_zero(bmap->div[div][0]);
}

/* Return the position of the first local variable that does not
 * have an explicit representation.
 * Return the total number of local variables if they all have
 * an explicit representation.
 * Return -1 on error.
 */
int isl_basic_map_first_unknown_div(__isl_keep isl_basic_map *bmap)
{
	int i;

	if (!bmap)
		return -1;

	for (i = 0; i < bmap->n_div; ++i) {
		if (!isl_basic_map_div_is_known(bmap, i))
			return i;
	}
	return bmap->n_div;
}

/* Return the position of the first local variable that does not
 * have an explicit representation.
 * Return the total number of local variables if they all have
 * an explicit representation.
 * Return -1 on error.
 */
int isl_basic_set_first_unknown_div(__isl_keep isl_basic_set *bset)
{
	return isl_basic_map_first_unknown_div(bset);
}

/* Does "bmap" have an explicit representation for all local variables?
 */
isl_bool isl_basic_map_divs_known(__isl_keep isl_basic_map *bmap)
{
	int first;
	isl_size n;

	n = isl_basic_map_dim(bmap, isl_dim_div);
	first = isl_basic_map_first_unknown_div(bmap);
	if (n < 0 || first < 0)
		return isl_bool_error;
	return first == n;
}

/* Do all basic maps in "map" have an explicit representation
 * for all local variables?
 */
isl_bool isl_map_divs_known(__isl_keep isl_map *map)
{
	int i;

	if (!map)
		return isl_bool_error;

	for (i = 0; i < map->n; ++i) {
		int known = isl_basic_map_divs_known(map->p[i]);
		if (known <= 0)
			return known;
	}

	return isl_bool_true;
}

/* If bmap contains any unknown divs, then compute explicit
 * expressions for them.  However, this computation may be
 * quite expensive, so first try to remove divs that aren't
 * strictly needed.
 */
__isl_give isl_map *isl_basic_map_compute_divs(__isl_take isl_basic_map *bmap)
{
	int known;
	struct isl_map *map;

	known = isl_basic_map_divs_known(bmap);
	if (known < 0)
		goto error;
	if (known)
		return isl_map_from_basic_map(bmap);

	bmap = isl_basic_map_drop_redundant_divs(bmap);

	known = isl_basic_map_divs_known(bmap);
	if (known < 0)
		goto error;
	if (known)
		return isl_map_from_basic_map(bmap);

	map = compute_divs(bmap);
	return map;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_map *isl_map_compute_divs(__isl_take isl_map *map)
{
	int i;
	int known;
	struct isl_map *res;

	if (!map)
		return NULL;
	if (map->n == 0)
		return map;

	known = isl_map_divs_known(map);
	if (known < 0) {
		isl_map_free(map);
		return NULL;
	}
	if (known)
		return map;

	res = isl_basic_map_compute_divs(isl_basic_map_copy(map->p[0]));
	for (i = 1 ; i < map->n; ++i) {
		struct isl_map *r2;
		r2 = isl_basic_map_compute_divs(isl_basic_map_copy(map->p[i]));
		if (ISL_F_ISSET(map, ISL_MAP_DISJOINT))
			res = isl_map_union_disjoint(res, r2);
		else
			res = isl_map_union(res, r2);
	}
	isl_map_free(map);

	return res;
}

__isl_give isl_set *isl_basic_set_compute_divs(__isl_take isl_basic_set *bset)
{
	return set_from_map(isl_basic_map_compute_divs(bset_to_bmap(bset)));
}

__isl_give isl_set *isl_set_compute_divs(__isl_take isl_set *set)
{
	return set_from_map(isl_map_compute_divs(set_to_map(set)));
}

__isl_give isl_set *isl_map_domain(__isl_take isl_map *map)
{
	isl_space *space;
	isl_size n_out;

	n_out = isl_map_dim(map, isl_dim_out);
	if (n_out < 0)
		return set_from_map(isl_map_free(map));
	space = isl_space_domain(isl_map_get_space(map));

	map = isl_map_project_out(map, isl_dim_out, 0, n_out);

	return set_from_map(isl_map_reset_space(map, space));
}

/* Return the union of "map1" and "map2", where we assume for now that
 * "map1" and "map2" are disjoint.  Note that the basic maps inside
 * "map1" or "map2" may not be disjoint from each other.
 * Also note that this function is also called from isl_map_union,
 * which takes care of handling the situation where "map1" and "map2"
 * may not be disjoint.
 *
 * If one of the inputs is empty, we can simply return the other input.
 * Similarly, if one of the inputs is universal, then it is equal to the union.
 */
static __isl_give isl_map *map_union_disjoint(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	int i;
	unsigned flags = 0;
	struct isl_map *map = NULL;
	int is_universe;

	if (isl_map_check_equal_space(map1, map2) < 0)
		goto error;

	if (map1->n == 0) {
		isl_map_free(map1);
		return map2;
	}
	if (map2->n == 0) {
		isl_map_free(map2);
		return map1;
	}

	is_universe = isl_map_plain_is_universe(map1);
	if (is_universe < 0)
		goto error;
	if (is_universe) {
		isl_map_free(map2);
		return map1;
	}

	is_universe = isl_map_plain_is_universe(map2);
	if (is_universe < 0)
		goto error;
	if (is_universe) {
		isl_map_free(map1);
		return map2;
	}

	if (ISL_F_ISSET(map1, ISL_MAP_DISJOINT) &&
	    ISL_F_ISSET(map2, ISL_MAP_DISJOINT))
		ISL_FL_SET(flags, ISL_MAP_DISJOINT);

	map = isl_map_alloc_space(isl_space_copy(map1->dim),
				map1->n + map2->n, flags);
	if (!map)
		goto error;
	for (i = 0; i < map1->n; ++i) {
		map = isl_map_add_basic_map(map,
				  isl_basic_map_copy(map1->p[i]));
		if (!map)
			goto error;
	}
	for (i = 0; i < map2->n; ++i) {
		map = isl_map_add_basic_map(map,
				  isl_basic_map_copy(map2->p[i]));
		if (!map)
			goto error;
	}
	isl_map_free(map1);
	isl_map_free(map2);
	return map;
error:
	isl_map_free(map);
	isl_map_free(map1);
	isl_map_free(map2);
	return NULL;
}

/* Return the union of "map1" and "map2", where "map1" and "map2" are
 * guaranteed to be disjoint by the caller.
 *
 * Note that this functions is called from within isl_map_make_disjoint,
 * so we have to be careful not to touch the constraints of the inputs
 * in any way.
 */
__isl_give isl_map *isl_map_union_disjoint(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	isl_map_align_params_bin(&map1, &map2);
	return map_union_disjoint(map1, map2);
}

/* Return the union of "map1" and "map2", where "map1" and "map2" may
 * not be disjoint.
 *
 * We currently simply call map_union_disjoint, the internal operation
 * of which does not really depend on the inputs being disjoint.
 * If the result contains more than one basic map, then we clear
 * the disjoint flag since the result may contain basic maps from
 * both inputs and these are not guaranteed to be disjoint.
 *
 * As a special case, if "map1" and "map2" are obviously equal,
 * then we simply return "map1".
 */
__isl_give isl_map *isl_map_union(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	int equal;

	if (isl_map_align_params_bin(&map1, &map2) < 0)
		goto error;

	equal = isl_map_plain_is_equal(map1, map2);
	if (equal < 0)
		goto error;
	if (equal) {
		isl_map_free(map2);
		return map1;
	}

	map1 = map_union_disjoint(map1, map2);
	if (!map1)
		return NULL;
	if (map1->n > 1)
		ISL_F_CLR(map1, ISL_MAP_DISJOINT);
	return map1;
error:
	isl_map_free(map1);
	isl_map_free(map2);
	return NULL;
}

__isl_give isl_set *isl_set_union_disjoint(
	__isl_take isl_set *set1, __isl_take isl_set *set2)
{
	return set_from_map(isl_map_union_disjoint(set_to_map(set1),
						    set_to_map(set2)));
}

__isl_give isl_set *isl_set_union(__isl_take isl_set *set1,
	__isl_take isl_set *set2)
{
	return set_from_map(isl_map_union(set_to_map(set1), set_to_map(set2)));
}

/* Apply "fn" to pairs of elements from "map" and "set" and collect
 * the results in a map living in "space".
 *
 * "map" and "set" are assumed to be compatible and non-NULL.
 */
static __isl_give isl_map *map_intersect_set(__isl_take isl_map *map,
	__isl_take isl_space *space, __isl_take isl_set *set,
	__isl_give isl_basic_map *fn(__isl_take isl_basic_map *bmap,
		__isl_take isl_basic_set *bset))
{
	unsigned flags = 0;
	struct isl_map *result;
	int i, j;

	if (isl_set_plain_is_universe(set)) {
		isl_set_free(set);
		return isl_map_reset_equal_dim_space(map, space);
	}

	if (ISL_F_ISSET(map, ISL_MAP_DISJOINT) &&
	    ISL_F_ISSET(set, ISL_MAP_DISJOINT))
		ISL_FL_SET(flags, ISL_MAP_DISJOINT);

	result = isl_map_alloc_space(space, map->n * set->n, flags);
	for (i = 0; result && i < map->n; ++i)
		for (j = 0; j < set->n; ++j) {
			result = isl_map_add_basic_map(result,
					fn(isl_basic_map_copy(map->p[i]),
					    isl_basic_set_copy(set->p[j])));
			if (!result)
				break;
		}

	isl_map_free(map);
	isl_set_free(set);
	return result;
}

__isl_give isl_map *isl_map_intersect_range(__isl_take isl_map *map,
	__isl_take isl_set *set)
{
	isl_bool ok;
	isl_space *space;

	isl_map_align_params_set(&map, &set);
	ok = isl_map_compatible_range(map, set);
	if (ok < 0)
		goto error;
	if (!ok)
		isl_die(set->ctx, isl_error_invalid,
			"incompatible spaces", goto error);

	space = isl_map_get_space(map);
	return map_intersect_set(map, space, set,
				&isl_basic_map_intersect_range);
error:
	isl_map_free(map);
	isl_set_free(set);
	return NULL;
}

/* Intersect the domain of "map" with "set".
 *
 * If the domain dimensions of "map" do not have any identifiers,
 * then copy them over from "set".
 */
__isl_give isl_map *isl_map_intersect_domain(__isl_take isl_map *map,
	__isl_take isl_set *set)
{
	isl_bool ok;
	isl_space *space;

	isl_map_align_params_set(&map, &set);
	ok = isl_map_compatible_domain(map, set);
	if (ok < 0)
		goto error;
	if (!ok)
		isl_die(set->ctx, isl_error_invalid,
			"incompatible spaces", goto error);

	space = isl_map_get_space(map);
	space = isl_space_copy_ids_if_unset(space, isl_dim_in,
					isl_set_peek_space(set), isl_dim_set);
	return map_intersect_set(map, space, set,
				&isl_basic_map_intersect_domain);
error:
	isl_map_free(map);
	isl_set_free(set);
	return NULL;
}

#undef TYPE
#define TYPE isl_map
static
#include "isl_copy_tuple_id_templ.c"

/* Data structure that specifies how isl_map_intersect_factor
 * should operate.
 *
 * "preserve_type" is the tuple where the factor differs from
 * the input map and of which the identifiers needs
 * to be preserved explicitly.
 * "other_factor" is used to extract the space of the other factor
 * from the space of the product ("map").
 * "product" is used to combine the given factor and a universe map
 * in the space returned by "other_factor" to produce a map
 * that lives in the same space as the input map.
 */
struct isl_intersect_factor_control {
	enum isl_dim_type preserve_type;
	__isl_give isl_space *(*other_factor)(__isl_take isl_space *space);
	__isl_give isl_map *(*product)(__isl_take isl_map *factor,
		__isl_take isl_map *other);
};

/* Given a map "map" in some product space and a map "factor"
 * living in some factor space, return the intersection.
 *
 * After aligning the parameters,
 * the map "factor" is first extended to a map living in the same space
 * as "map" and then a regular intersection is computed.
 *
 * Note that the extension is computed as a product, which is anonymous
 * by default.  If "map" has an identifier on the corresponding tuple,
 * then this identifier needs to be set on the product
 * before the intersection is computed.
 */
static __isl_give isl_map *isl_map_intersect_factor(
	__isl_take isl_map *map, __isl_take isl_map *factor,
	struct isl_intersect_factor_control *control)
{
	isl_bool equal;
	isl_space *space;
	isl_map *other, *product;

	equal = isl_map_has_equal_params(map, factor);
	if (equal < 0)
		goto error;
	if (!equal) {
		map = isl_map_align_params(map, isl_map_get_space(factor));
		factor = isl_map_align_params(factor, isl_map_get_space(map));
	}

	space = isl_map_get_space(map);
	other = isl_map_universe(control->other_factor(space));
	product = control->product(factor, other);

	space = isl_map_peek_space(map);
	product = isl_map_copy_tuple_id(product, control->preserve_type,
					space, control->preserve_type);
	return map_intersect(map, product);
error:
	isl_map_free(map);
	isl_map_free(factor);
	return NULL;
}

/* Return the domain product of "map2" and "map1".
 */
static __isl_give isl_map *isl_map_reverse_domain_product(
	__isl_take isl_map *map1, __isl_take isl_map *map2)
{
	return isl_map_domain_product(map2, map1);
}

/* Return the range product of "map2" and "map1".
 */
static __isl_give isl_map *isl_map_reverse_range_product(
	__isl_take isl_map *map1, __isl_take isl_map *map2)
{
	return isl_map_range_product(map2, map1);
}

/* Given a map "map" in a space [A -> B] -> C and a map "factor"
 * in the space A -> C, return the intersection.
 */
__isl_give isl_map *isl_map_intersect_domain_factor_domain(
	__isl_take isl_map *map, __isl_take isl_map *factor)
{
	struct isl_intersect_factor_control control = {
		.preserve_type = isl_dim_in,
		.other_factor = isl_space_domain_factor_range,
		.product = isl_map_domain_product,
	};

	return isl_map_intersect_factor(map, factor, &control);
}

/* Given a map "map" in a space [A -> B] -> C and a map "factor"
 * in the space B -> C, return the intersection.
 */
__isl_give isl_map *isl_map_intersect_domain_factor_range(
	__isl_take isl_map *map, __isl_take isl_map *factor)
{
	struct isl_intersect_factor_control control = {
		.preserve_type = isl_dim_in,
		.other_factor = isl_space_domain_factor_domain,
		.product = isl_map_reverse_domain_product,
	};

	return isl_map_intersect_factor(map, factor, &control);
}

/* Given a map "map" in a space A -> [B -> C] and a map "factor"
 * in the space A -> B, return the intersection.
 */
__isl_give isl_map *isl_map_intersect_range_factor_domain(
	__isl_take isl_map *map, __isl_take isl_map *factor)
{
	struct isl_intersect_factor_control control = {
		.preserve_type = isl_dim_out,
		.other_factor = isl_space_range_factor_range,
		.product = isl_map_range_product,
	};

	return isl_map_intersect_factor(map, factor, &control);
}

/* Given a map "map" in a space A -> [B -> C] and a map "factor"
 * in the space A -> C, return the intersection.
 */
__isl_give isl_map *isl_map_intersect_range_factor_range(
	__isl_take isl_map *map, __isl_take isl_map *factor)
{
	struct isl_intersect_factor_control control = {
		.preserve_type = isl_dim_out,
		.other_factor = isl_space_range_factor_domain,
		.product = isl_map_reverse_range_product,
	};

	return isl_map_intersect_factor(map, factor, &control);
}

/* Given a set "set" in a space [A -> B] and a set "domain"
 * in the space A, return the intersection.
 *
 * The set "domain" is first extended to a set living in the space
 * [A -> B] and then a regular intersection is computed.
 */
__isl_give isl_set *isl_set_intersect_factor_domain(__isl_take isl_set *set,
	__isl_take isl_set *domain)
{
	struct isl_intersect_factor_control control = {
		.preserve_type = isl_dim_set,
		.other_factor = isl_space_factor_range,
		.product = isl_map_range_product,
	};

	return set_from_map(isl_map_intersect_factor(set_to_map(set),
						set_to_map(domain), &control));
}

/* Given a set "set" in a space [A -> B] and a set "range"
 * in the space B, return the intersection.
 *
 * The set "range" is first extended to a set living in the space
 * [A -> B] and then a regular intersection is computed.
 */
__isl_give isl_set *isl_set_intersect_factor_range(__isl_take isl_set *set,
	__isl_take isl_set *range)
{
	struct isl_intersect_factor_control control = {
		.preserve_type = isl_dim_set,
		.other_factor = isl_space_factor_domain,
		.product = isl_map_reverse_range_product,
	};

	return set_from_map(isl_map_intersect_factor(set_to_map(set),
						set_to_map(range), &control));
}

__isl_give isl_map *isl_map_apply_domain(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	if (isl_map_align_params_bin(&map1, &map2) < 0)
		goto error;
	map1 = isl_map_reverse(map1);
	map1 = isl_map_apply_range(map1, map2);
	return isl_map_reverse(map1);
error:
	isl_map_free(map1);
	isl_map_free(map2);
	return NULL;
}

__isl_give isl_map *isl_map_apply_range(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	isl_space *space;
	struct isl_map *result;
	int i, j;

	if (isl_map_align_params_bin(&map1, &map2) < 0)
		goto error;

	space = isl_space_join(isl_space_copy(map1->dim),
				  isl_space_copy(map2->dim));

	result = isl_map_alloc_space(space, map1->n * map2->n, 0);
	if (!result)
		goto error;
	for (i = 0; i < map1->n; ++i)
		for (j = 0; j < map2->n; ++j) {
			result = isl_map_add_basic_map(result,
			    isl_basic_map_apply_range(
				isl_basic_map_copy(map1->p[i]),
				isl_basic_map_copy(map2->p[j])));
			if (!result)
				goto error;
		}
	isl_map_free(map1);
	isl_map_free(map2);
	if (result && result->n <= 1)
		ISL_F_SET(result, ISL_MAP_DISJOINT);
	return result;
error:
	isl_map_free(map1);
	isl_map_free(map2);
	return NULL;
}

/* Is "bmap" a transformation, i.e.,
 * does it relate elements from the same space.
 */
isl_bool isl_basic_map_is_transformation(__isl_keep isl_basic_map *bmap)
{
	isl_space *space;

	space = isl_basic_map_peek_space(bmap);
	return isl_space_tuple_is_equal(space, isl_dim_in, space, isl_dim_out);
}

/* Check that "bmap" is a transformation, i.e.,
 * that it relates elements from the same space.
 */
static isl_stat isl_basic_map_check_transformation(
	__isl_keep isl_basic_map *bmap)
{
	isl_bool equal;

	equal = isl_basic_map_is_transformation(bmap);
	if (equal < 0)
		return isl_stat_error;
	if (!equal)
		isl_die(isl_basic_map_get_ctx(bmap), isl_error_invalid,
			"domain and range don't match", return isl_stat_error);
	return isl_stat_ok;
}

/*
 * returns range - domain
 */
__isl_give isl_basic_set *isl_basic_map_deltas(__isl_take isl_basic_map *bmap)
{
	isl_space *target_space;
	struct isl_basic_set *bset;
	isl_size dim;
	isl_size nparam;
	isl_size total;
	int i;

	if (isl_basic_map_check_transformation(bmap) < 0)
		return isl_basic_map_free(bmap);
	dim = isl_basic_map_dim(bmap, isl_dim_in);
	nparam = isl_basic_map_dim(bmap, isl_dim_param);
	if (dim < 0 || nparam < 0)
		goto error;
	target_space = isl_space_domain(isl_basic_map_get_space(bmap));
	bmap = isl_basic_map_from_range(isl_basic_map_wrap(bmap));
	bmap = isl_basic_map_add_dims(bmap, isl_dim_in, dim);
	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		bmap = isl_basic_map_free(bmap);
	bmap = isl_basic_map_extend_constraints(bmap, dim, 0);
	for (i = 0; i < dim; ++i) {
		int j = isl_basic_map_alloc_equality(bmap);
		if (j < 0) {
			bmap = isl_basic_map_free(bmap);
			break;
		}
		isl_seq_clr(bmap->eq[j], 1 + total);
		isl_int_set_si(bmap->eq[j][1+nparam+i], 1);
		isl_int_set_si(bmap->eq[j][1+nparam+dim+i], 1);
		isl_int_set_si(bmap->eq[j][1+nparam+2*dim+i], -1);
	}
	bset = isl_basic_map_domain(bmap);
	bset = isl_basic_set_reset_space(bset, target_space);
	return bset;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

/* Is the tuple of type "type1" of "map" the same as
 * the tuple of type "type2" of "space"?
 */
isl_bool isl_map_space_tuple_is_equal(__isl_keep isl_map *map,
	enum isl_dim_type type1, __isl_keep isl_space *space,
	enum isl_dim_type type2)
{
	isl_space *map_space;

	map_space = isl_map_peek_space(map);
	return isl_space_tuple_is_equal(map_space, type1, space, type2);
}

/* Is the tuple of type "type1" of "map1" the same as
 * the tuple of type "type2" of "map2"?
 */
isl_bool isl_map_tuple_is_equal(__isl_keep isl_map *map1,
	enum isl_dim_type type1, __isl_keep isl_map *map2,
	enum isl_dim_type type2)
{
	isl_space *space1, *space2;

	space1 = isl_map_peek_space(map1);
	space2 = isl_map_peek_space(map2);
	return isl_space_tuple_is_equal(space1, type1, space2, type2);
}

/* Is the space of "obj" equal to "space", ignoring parameters?
 */
isl_bool isl_map_has_space_tuples(__isl_keep isl_map *map,
	__isl_keep isl_space *space)
{
	isl_space *map_space;

	map_space = isl_map_peek_space(map);
	return isl_space_has_equal_tuples(map_space, space);
}

/* Check that "map" is a transformation, i.e.,
 * that it relates elements from the same space.
 */
isl_stat isl_map_check_transformation(__isl_keep isl_map *map)
{
	isl_bool equal;

	equal = isl_map_tuple_is_equal(map, isl_dim_in, map, isl_dim_out);
	if (equal < 0)
		return isl_stat_error;
	if (!equal)
		isl_die(isl_map_get_ctx(map), isl_error_invalid,
			"domain and range don't match", return isl_stat_error);
	return isl_stat_ok;
}

/*
 * returns range - domain
 */
__isl_give isl_set *isl_map_deltas(__isl_take isl_map *map)
{
	int i;
	isl_space *space;
	struct isl_set *result;

	if (isl_map_check_transformation(map) < 0)
		goto error;
	space = isl_map_get_space(map);
	space = isl_space_domain(space);
	result = isl_set_alloc_space(space, map->n, 0);
	if (!result)
		goto error;
	for (i = 0; i < map->n; ++i)
		result = isl_set_add_basic_set(result,
			  isl_basic_map_deltas(isl_basic_map_copy(map->p[i])));
	isl_map_free(map);
	return result;
error:
	isl_map_free(map);
	return NULL;
}

/*
 * returns [domain -> range] -> range - domain
 */
__isl_give isl_basic_map *isl_basic_map_deltas_map(
	__isl_take isl_basic_map *bmap)
{
	int i, k;
	isl_space *space;
	isl_basic_map *domain;
	isl_size nparam, n;
	isl_size total;

	if (isl_basic_map_check_transformation(bmap) < 0)
		return isl_basic_map_free(bmap);

	nparam = isl_basic_map_dim(bmap, isl_dim_param);
	n = isl_basic_map_dim(bmap, isl_dim_in);
	if (nparam < 0 || n < 0)
		return isl_basic_map_free(bmap);

	space = isl_basic_map_get_space(bmap);
	space = isl_space_from_range(isl_space_domain(space));
	domain = isl_basic_map_universe(space);

	bmap = isl_basic_map_from_domain(isl_basic_map_wrap(bmap));
	bmap = isl_basic_map_apply_range(bmap, domain);
	bmap = isl_basic_map_extend_constraints(bmap, n, 0);

	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return isl_basic_map_free(bmap);

	for (i = 0; i < n; ++i) {
		k = isl_basic_map_alloc_equality(bmap);
		if (k < 0)
			goto error;
		isl_seq_clr(bmap->eq[k], 1 + total);
		isl_int_set_si(bmap->eq[k][1 + nparam + i], 1);
		isl_int_set_si(bmap->eq[k][1 + nparam + n + i], -1);
		isl_int_set_si(bmap->eq[k][1 + nparam + n + n + i], 1);
	}

	bmap = isl_basic_map_gauss(bmap, NULL);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap);
	return NULL;
}

/*
 * returns [domain -> range] -> range - domain
 */
__isl_give isl_map *isl_map_deltas_map(__isl_take isl_map *map)
{
	if (isl_map_check_transformation(map) < 0)
		return isl_map_free(map);

	return isl_map_transform(map, &isl_space_range_map,
					&isl_basic_map_deltas_map);
}

/* Return pairs of elements { x -> y } such that y - x is in "deltas".
 */
__isl_give isl_map *isl_set_translation(__isl_take isl_set *deltas)
{
	isl_space *space;
	isl_map *map;

	space = isl_space_map_from_set(isl_set_get_space(deltas));
	map = isl_map_deltas_map(isl_map_universe(space));
	map = isl_map_intersect_range(map, deltas);

	return isl_set_unwrap(isl_map_domain(map));
}

__isl_give isl_basic_map *isl_basic_map_identity(__isl_take isl_space *space)
{
	isl_size n_in, n_out;

	n_in = isl_space_dim(space, isl_dim_in);
	n_out = isl_space_dim(space, isl_dim_out);
	if (n_in < 0 || n_out < 0)
		goto error;
	if (n_in != n_out)
		isl_die(space->ctx, isl_error_invalid,
			"number of input and output dimensions needs to be "
			"the same", goto error);
	return isl_basic_map_equal(space, n_in);
error:
	isl_space_free(space);
	return NULL;
}

__isl_give isl_map *isl_map_identity(__isl_take isl_space *space)
{
	return isl_map_from_basic_map(isl_basic_map_identity(space));
}

__isl_give isl_map *isl_set_identity(__isl_take isl_set *set)
{
	isl_space *space = isl_set_get_space(set);
	isl_map *id;
	id = isl_map_identity(isl_space_map_from_set(space));
	return isl_map_intersect_range(id, set);
}

/* Construct a basic set with all set dimensions having only non-negative
 * values.
 */
__isl_give isl_basic_set *isl_basic_set_positive_orthant(
	__isl_take isl_space *space)
{
	int i;
	isl_size nparam;
	isl_size dim;
	isl_size total;
	struct isl_basic_set *bset;

	nparam = isl_space_dim(space, isl_dim_param);
	dim = isl_space_dim(space, isl_dim_set);
	total = isl_space_dim(space, isl_dim_all);
	if (nparam < 0 || dim < 0 || total < 0)
		space = isl_space_free(space);
	bset = isl_basic_set_alloc_space(space, 0, 0, dim);
	if (!bset)
		return NULL;
	for (i = 0; i < dim; ++i) {
		int k = isl_basic_set_alloc_inequality(bset);
		if (k < 0)
			goto error;
		isl_seq_clr(bset->ineq[k], 1 + total);
		isl_int_set_si(bset->ineq[k][1 + nparam + i], 1);
	}
	return bset;
error:
	isl_basic_set_free(bset);
	return NULL;
}

/* Construct the half-space x_pos >= 0.
 */
static __isl_give isl_basic_set *nonneg_halfspace(__isl_take isl_space *space,
	int pos)
{
	int k;
	isl_size total;
	isl_basic_set *nonneg;

	total = isl_space_dim(space, isl_dim_all);
	if (total < 0)
		space = isl_space_free(space);
	nonneg = isl_basic_set_alloc_space(space, 0, 0, 1);
	k = isl_basic_set_alloc_inequality(nonneg);
	if (k < 0)
		goto error;
	isl_seq_clr(nonneg->ineq[k], 1 + total);
	isl_int_set_si(nonneg->ineq[k][pos], 1);

	return isl_basic_set_finalize(nonneg);
error:
	isl_basic_set_free(nonneg);
	return NULL;
}

/* Construct the half-space x_pos <= -1.
 */
static __isl_give isl_basic_set *neg_halfspace(__isl_take isl_space *space,
	int pos)
{
	int k;
	isl_size total;
	isl_basic_set *neg;

	total = isl_space_dim(space, isl_dim_all);
	if (total < 0)
		space = isl_space_free(space);
	neg = isl_basic_set_alloc_space(space, 0, 0, 1);
	k = isl_basic_set_alloc_inequality(neg);
	if (k < 0)
		goto error;
	isl_seq_clr(neg->ineq[k], 1 + total);
	isl_int_set_si(neg->ineq[k][0], -1);
	isl_int_set_si(neg->ineq[k][pos], -1);

	return isl_basic_set_finalize(neg);
error:
	isl_basic_set_free(neg);
	return NULL;
}

__isl_give isl_set *isl_set_split_dims(__isl_take isl_set *set,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;
	unsigned offset;
	isl_basic_set *nonneg;
	isl_basic_set *neg;

	if (n == 0)
		return set;

	if (isl_set_check_range(set, type, first, n) < 0)
		return isl_set_free(set);

	offset = pos(set->dim, type);
	for (i = 0; i < n; ++i) {
		nonneg = nonneg_halfspace(isl_set_get_space(set),
					  offset + first + i);
		neg = neg_halfspace(isl_set_get_space(set), offset + first + i);

		set = isl_set_intersect(set, isl_basic_set_union(nonneg, neg));
	}

	return set;
}

static isl_stat foreach_orthant(__isl_take isl_set *set, int *signs, int first,
	int len,
	isl_stat (*fn)(__isl_take isl_set *orthant, int *signs, void *user),
	void *user)
{
	isl_set *half;

	if (!set)
		return isl_stat_error;
	if (isl_set_plain_is_empty(set)) {
		isl_set_free(set);
		return isl_stat_ok;
	}
	if (first == len)
		return fn(set, signs, user);

	signs[first] = 1;
	half = isl_set_from_basic_set(nonneg_halfspace(isl_set_get_space(set),
							1 + first));
	half = isl_set_intersect(half, isl_set_copy(set));
	if (foreach_orthant(half, signs, first + 1, len, fn, user) < 0)
		goto error;

	signs[first] = -1;
	half = isl_set_from_basic_set(neg_halfspace(isl_set_get_space(set),
							1 + first));
	half = isl_set_intersect(half, set);
	return foreach_orthant(half, signs, first + 1, len, fn, user);
error:
	isl_set_free(set);
	return isl_stat_error;
}

/* Call "fn" on the intersections of "set" with each of the orthants
 * (except for obviously empty intersections).  The orthant is identified
 * by the signs array, with each entry having value 1 or -1 according
 * to the sign of the corresponding variable.
 */
isl_stat isl_set_foreach_orthant(__isl_keep isl_set *set,
	isl_stat (*fn)(__isl_take isl_set *orthant, int *signs, void *user),
	void *user)
{
	isl_size nparam;
	isl_size nvar;
	int *signs;
	isl_stat r;

	if (!set)
		return isl_stat_error;
	if (isl_set_plain_is_empty(set))
		return isl_stat_ok;

	nparam = isl_set_dim(set, isl_dim_param);
	nvar = isl_set_dim(set, isl_dim_set);
	if (nparam < 0 || nvar < 0)
		return isl_stat_error;

	signs = isl_alloc_array(set->ctx, int, nparam + nvar);

	r = foreach_orthant(isl_set_copy(set), signs, 0, nparam + nvar,
			    fn, user);

	free(signs);

	return r;
}

isl_bool isl_set_is_equal(__isl_keep isl_set *set1, __isl_keep isl_set *set2)
{
	return isl_map_is_equal(set_to_map(set1), set_to_map(set2));
}

isl_bool isl_basic_map_is_subset(__isl_keep isl_basic_map *bmap1,
	__isl_keep isl_basic_map *bmap2)
{
	isl_bool is_subset;
	struct isl_map *map1;
	struct isl_map *map2;

	if (!bmap1 || !bmap2)
		return isl_bool_error;

	map1 = isl_map_from_basic_map(isl_basic_map_copy(bmap1));
	map2 = isl_map_from_basic_map(isl_basic_map_copy(bmap2));

	is_subset = isl_map_is_subset(map1, map2);

	isl_map_free(map1);
	isl_map_free(map2);

	return is_subset;
}

isl_bool isl_basic_set_is_subset(__isl_keep isl_basic_set *bset1,
	__isl_keep isl_basic_set *bset2)
{
	return isl_basic_map_is_subset(bset1, bset2);
}

isl_bool isl_basic_map_is_equal(__isl_keep isl_basic_map *bmap1,
	__isl_keep isl_basic_map *bmap2)
{
	isl_bool is_subset;

	if (!bmap1 || !bmap2)
		return isl_bool_error;
	is_subset = isl_basic_map_is_subset(bmap1, bmap2);
	if (is_subset != isl_bool_true)
		return is_subset;
	is_subset = isl_basic_map_is_subset(bmap2, bmap1);
	return is_subset;
}

isl_bool isl_basic_set_is_equal(__isl_keep isl_basic_set *bset1,
	__isl_keep isl_basic_set *bset2)
{
	return isl_basic_map_is_equal(
		bset_to_bmap(bset1), bset_to_bmap(bset2));
}

isl_bool isl_map_is_empty(__isl_keep isl_map *map)
{
	int i;
	int is_empty;

	if (!map)
		return isl_bool_error;
	for (i = 0; i < map->n; ++i) {
		is_empty = isl_basic_map_is_empty(map->p[i]);
		if (is_empty < 0)
			return isl_bool_error;
		if (!is_empty)
			return isl_bool_false;
	}
	return isl_bool_true;
}

isl_bool isl_map_plain_is_empty(__isl_keep isl_map *map)
{
	return map ? map->n == 0 : isl_bool_error;
}

isl_bool isl_set_plain_is_empty(__isl_keep isl_set *set)
{
	return set ? set->n == 0 : isl_bool_error;
}

isl_bool isl_set_is_empty(__isl_keep isl_set *set)
{
	return isl_map_is_empty(set_to_map(set));
}

#undef TYPE
#define TYPE	isl_basic_map

static
#include "isl_type_has_equal_space_bin_templ.c"
#include "isl_type_check_equal_space_templ.c"

/* Check that "bset1" and "bset2" live in the same space,
 * reporting an error if they do not.
 */
isl_stat isl_basic_set_check_equal_space(__isl_keep isl_basic_set *bset1,
	__isl_keep isl_basic_set *bset2)
{
	return isl_basic_map_check_equal_space(bset_to_bmap(bset1),
						bset_to_bmap(bset1));
}

#undef TYPE
#define TYPE	isl_map

#include "isl_type_has_equal_space_bin_templ.c"
#include "isl_type_check_equal_space_templ.c"
#include "isl_type_has_space_templ.c"

isl_bool isl_set_has_equal_space(__isl_keep isl_set *set1,
	__isl_keep isl_set *set2)
{
	return isl_map_has_equal_space(set_to_map(set1), set_to_map(set2));
}

#undef TYPE1
#define TYPE1		isl_map
#undef TYPE2
#define TYPE2		isl_basic_map
#undef TYPE_PAIR
#define TYPE_PAIR	isl_map_basic_map

static
#include "isl_type_has_equal_space_templ.c"
#include "isl_type_check_equal_space_templ.c"

/* Check that "set" and "bset" live in the same space,
 * reporting an error if they do not.
 */
isl_stat isl_set_basic_set_check_equal_space(__isl_keep isl_set *set,
	__isl_keep isl_basic_set *bset)
{
	return isl_map_basic_map_check_equal_space(set_to_map(set),
						    bset_to_bmap(bset));
}

static isl_bool map_is_equal(__isl_keep isl_map *map1, __isl_keep isl_map *map2)
{
	isl_bool is_subset;

	if (!map1 || !map2)
		return isl_bool_error;
	is_subset = isl_map_is_subset(map1, map2);
	if (is_subset != isl_bool_true)
		return is_subset;
	is_subset = isl_map_is_subset(map2, map1);
	return is_subset;
}

/* Is "map1" equal to "map2"?
 *
 * First check if they are obviously equal.
 * If not, then perform a more detailed analysis.
 */
isl_bool isl_map_is_equal(__isl_keep isl_map *map1, __isl_keep isl_map *map2)
{
	isl_bool equal;

	equal = isl_map_plain_is_equal(map1, map2);
	if (equal < 0 || equal)
		return equal;
	return isl_map_align_params_map_map_and_test(map1, map2, &map_is_equal);
}

isl_bool isl_basic_map_is_strict_subset(__isl_keep isl_basic_map *bmap1,
	__isl_keep isl_basic_map *bmap2)
{
	isl_bool is_subset;

	if (!bmap1 || !bmap2)
		return isl_bool_error;
	is_subset = isl_basic_map_is_subset(bmap1, bmap2);
	if (is_subset != isl_bool_true)
		return is_subset;
	is_subset = isl_basic_map_is_subset(bmap2, bmap1);
	return isl_bool_not(is_subset);
}

isl_bool isl_map_is_strict_subset(__isl_keep isl_map *map1,
	__isl_keep isl_map *map2)
{
	isl_bool is_subset;

	if (!map1 || !map2)
		return isl_bool_error;
	is_subset = isl_map_is_subset(map1, map2);
	if (is_subset != isl_bool_true)
		return is_subset;
	is_subset = isl_map_is_subset(map2, map1);
	return isl_bool_not(is_subset);
}

isl_bool isl_set_is_strict_subset(__isl_keep isl_set *set1,
	__isl_keep isl_set *set2)
{
	return isl_map_is_strict_subset(set_to_map(set1), set_to_map(set2));
}

/* Is "bmap" obviously equal to the universe with the same space?
 *
 * That is, does it not have any constraints?
 */
isl_bool isl_basic_map_plain_is_universe(__isl_keep isl_basic_map *bmap)
{
	if (!bmap)
		return isl_bool_error;
	return bmap->n_eq == 0 && bmap->n_ineq == 0;
}

/* Is "bset" obviously equal to the universe with the same space?
 */
isl_bool isl_basic_set_plain_is_universe(__isl_keep isl_basic_set *bset)
{
	return isl_basic_map_plain_is_universe(bset);
}

/* If "c" does not involve any existentially quantified variables,
 * then set *univ to false and abort
 */
static isl_stat involves_divs(__isl_take isl_constraint *c, void *user)
{
	isl_bool *univ = user;
	isl_size n;

	n = isl_constraint_dim(c, isl_dim_div);
	if (n < 0)
		c = isl_constraint_free(c);
	*univ = isl_constraint_involves_dims(c, isl_dim_div, 0, n);
	isl_constraint_free(c);
	if (*univ < 0 || !*univ)
		return isl_stat_error;
	return isl_stat_ok;
}

/* Is "bmap" equal to the universe with the same space?
 *
 * First check if it is obviously equal to the universe.
 * If not and if there are any constraints not involving
 * existentially quantified variables, then it is certainly
 * not equal to the universe.
 * Otherwise, check if the universe is a subset of "bmap".
 */
isl_bool isl_basic_map_is_universe(__isl_keep isl_basic_map *bmap)
{
	isl_size n_div;
	isl_bool univ;
	isl_basic_map *test;

	univ = isl_basic_map_plain_is_universe(bmap);
	if (univ < 0 || univ)
		return univ;
	n_div = isl_basic_map_dim(bmap, isl_dim_div);
	if (n_div < 0)
		return isl_bool_error;
	if (n_div == 0)
		return isl_bool_false;
	univ = isl_bool_true;
	if (isl_basic_map_foreach_constraint(bmap, &involves_divs, &univ) < 0 &&
	    univ)
		return isl_bool_error;
	if (univ < 0 || !univ)
		return univ;
	test = isl_basic_map_universe(isl_basic_map_get_space(bmap));
	univ = isl_basic_map_is_subset(test, bmap);
	isl_basic_map_free(test);
	return univ;
}

/* Is "bset" equal to the universe with the same space?
 */
isl_bool isl_basic_set_is_universe(__isl_keep isl_basic_set *bset)
{
	return isl_basic_map_is_universe(bset);
}

isl_bool isl_map_plain_is_universe(__isl_keep isl_map *map)
{
	int i;

	if (!map)
		return isl_bool_error;

	for (i = 0; i < map->n; ++i) {
		isl_bool r = isl_basic_map_plain_is_universe(map->p[i]);
		if (r < 0 || r)
			return r;
	}

	return isl_bool_false;
}

isl_bool isl_set_plain_is_universe(__isl_keep isl_set *set)
{
	return isl_map_plain_is_universe(set_to_map(set));
}

isl_bool isl_basic_map_is_empty(__isl_keep isl_basic_map *bmap)
{
	struct isl_basic_set *bset = NULL;
	struct isl_vec *sample = NULL;
	isl_bool empty, non_empty;

	if (!bmap)
		return isl_bool_error;

	if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_EMPTY))
		return isl_bool_true;

	if (isl_basic_map_plain_is_universe(bmap))
		return isl_bool_false;

	if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_RATIONAL)) {
		struct isl_basic_map *copy = isl_basic_map_copy(bmap);
		copy = isl_basic_map_remove_redundancies(copy);
		empty = isl_basic_map_plain_is_empty(copy);
		isl_basic_map_free(copy);
		return empty;
	}

	non_empty = isl_basic_map_plain_is_non_empty(bmap);
	if (non_empty < 0)
		return isl_bool_error;
	if (non_empty)
		return isl_bool_false;
	isl_vec_free(bmap->sample);
	bmap->sample = NULL;
	bset = isl_basic_map_underlying_set(isl_basic_map_copy(bmap));
	if (!bset)
		return isl_bool_error;
	sample = isl_basic_set_sample_vec(bset);
	if (!sample)
		return isl_bool_error;
	empty = sample->size == 0;
	isl_vec_free(bmap->sample);
	bmap->sample = sample;
	if (empty)
		ISL_F_SET(bmap, ISL_BASIC_MAP_EMPTY);

	return empty;
}

isl_bool isl_basic_map_plain_is_empty(__isl_keep isl_basic_map *bmap)
{
	if (!bmap)
		return isl_bool_error;
	return ISL_F_ISSET(bmap, ISL_BASIC_MAP_EMPTY);
}

isl_bool isl_basic_set_plain_is_empty(__isl_keep isl_basic_set *bset)
{
	if (!bset)
		return isl_bool_error;
	return ISL_F_ISSET(bset, ISL_BASIC_SET_EMPTY);
}

/* Is "bmap" known to be non-empty?
 *
 * That is, is the cached sample still valid?
 */
isl_bool isl_basic_map_plain_is_non_empty(__isl_keep isl_basic_map *bmap)
{
	isl_size total;

	if (!bmap)
		return isl_bool_error;
	if (!bmap->sample)
		return isl_bool_false;
	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return isl_bool_error;
	if (bmap->sample->size != 1 + total)
		return isl_bool_false;
	return isl_basic_map_contains(bmap, bmap->sample);
}

isl_bool isl_basic_set_is_empty(__isl_keep isl_basic_set *bset)
{
	return isl_basic_map_is_empty(bset_to_bmap(bset));
}

__isl_give isl_map *isl_basic_map_union(__isl_take isl_basic_map *bmap1,
	__isl_take isl_basic_map *bmap2)
{
	struct isl_map *map;

	if (isl_basic_map_check_equal_space(bmap1, bmap2) < 0)
		goto error;

	map = isl_map_alloc_space(isl_space_copy(bmap1->dim), 2, 0);
	if (!map)
		goto error;
	map = isl_map_add_basic_map(map, bmap1);
	map = isl_map_add_basic_map(map, bmap2);
	return map;
error:
	isl_basic_map_free(bmap1);
	isl_basic_map_free(bmap2);
	return NULL;
}

__isl_give isl_set *isl_basic_set_union(__isl_take isl_basic_set *bset1,
	__isl_take isl_basic_set *bset2)
{
	return set_from_map(isl_basic_map_union(bset_to_bmap(bset1),
						bset_to_bmap(bset2)));
}

/* Order divs such that any div only depends on previous divs */
__isl_give isl_basic_map *isl_basic_map_order_divs(
	__isl_take isl_basic_map *bmap)
{
	int i;
	isl_size off;

	off = isl_basic_map_var_offset(bmap, isl_dim_div);
	if (off < 0)
		return isl_basic_map_free(bmap);

	for (i = 0; i < bmap->n_div; ++i) {
		int pos;
		if (isl_int_is_zero(bmap->div[i][0]))
			continue;
		pos = isl_seq_first_non_zero(bmap->div[i]+1+1+off+i,
							    bmap->n_div-i);
		if (pos == -1)
			continue;
		if (pos == 0)
			isl_die(isl_basic_map_get_ctx(bmap), isl_error_internal,
				"integer division depends on itself",
				return isl_basic_map_free(bmap));
		bmap = isl_basic_map_swap_div(bmap, i, i + pos);
		if (!bmap)
			return NULL;
		--i;
	}
	return bmap;
}

__isl_give isl_map *isl_map_order_divs(__isl_take isl_map *map)
{
	int i;

	if (!map)
		return 0;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_order_divs(map->p[i]);
		if (!map->p[i])
			goto error;
	}

	return map;
error:
	isl_map_free(map);
	return NULL;
}

/* Sort the local variables of "bset".
 */
__isl_give isl_basic_set *isl_basic_set_sort_divs(
	__isl_take isl_basic_set *bset)
{
	return bset_from_bmap(isl_basic_map_sort_divs(bset_to_bmap(bset)));
}

/* Apply the expansion computed by isl_merge_divs.
 * The expansion itself is given by "exp" while the resulting
 * list of divs is given by "div".
 *
 * Move the integer divisions of "bmap" into the right position
 * according to "exp" and then introduce the additional integer
 * divisions, adding div constraints.
 * The moving should be done first to avoid moving coefficients
 * in the definitions of the extra integer divisions.
 */
__isl_give isl_basic_map *isl_basic_map_expand_divs(
	__isl_take isl_basic_map *bmap, __isl_take isl_mat *div, int *exp)
{
	int i, j;
	int n_div;

	bmap = isl_basic_map_cow(bmap);
	if (!bmap || !div)
		goto error;

	if (div->n_row < bmap->n_div)
		isl_die(isl_mat_get_ctx(div), isl_error_invalid,
			"not an expansion", goto error);

	n_div = bmap->n_div;
	bmap = isl_basic_map_extend(bmap, div->n_row - n_div, 0,
					    2 * (div->n_row - n_div));

	for (i = n_div; i < div->n_row; ++i)
		if (isl_basic_map_alloc_div(bmap) < 0)
			goto error;

	for (j = n_div - 1; j >= 0; --j) {
		if (exp[j] == j)
			break;
		bmap = isl_basic_map_swap_div(bmap, j, exp[j]);
		if (!bmap)
			goto error;
	}
	j = 0;
	for (i = 0; i < div->n_row; ++i) {
		if (j < n_div && exp[j] == i) {
			j++;
		} else {
			isl_seq_cpy(bmap->div[i], div->row[i], div->n_col);
			if (isl_basic_map_div_is_marked_unknown(bmap, i))
				continue;
			bmap = isl_basic_map_add_div_constraints(bmap, i);
			if (!bmap)
				goto error;
		}
	}

	isl_mat_free(div);
	return bmap;
error:
	isl_basic_map_free(bmap);
	isl_mat_free(div);
	return NULL;
}

/* Apply the expansion computed by isl_merge_divs.
 * The expansion itself is given by "exp" while the resulting
 * list of divs is given by "div".
 */
__isl_give isl_basic_set *isl_basic_set_expand_divs(
	__isl_take isl_basic_set *bset, __isl_take isl_mat *div, int *exp)
{
	return isl_basic_map_expand_divs(bset, div, exp);
}

/* Look for a div in dst that corresponds to the div "div" in src.
 * The divs before "div" in src and dst are assumed to be the same.
 * 
 * Return the position of the corresponding div in dst
 * if there is one.  Otherwise, return a position beyond the integer divisions.
 * Return -1 on error.
 */
static int find_div(__isl_keep isl_basic_map *dst,
	__isl_keep isl_basic_map *src, unsigned div)
{
	int i;
	isl_size n_div;
	isl_size v_div;

	v_div = isl_basic_map_var_offset(src, isl_dim_div);
	n_div = isl_basic_map_dim(dst, isl_dim_div);
	if (n_div < 0 || v_div < 0)
		return -1;
	isl_assert(dst->ctx, div <= n_div, return -1);
	for (i = div; i < n_div; ++i)
		if (isl_seq_eq(dst->div[i], src->div[div], 1+1+v_div+div) &&
		    isl_seq_first_non_zero(dst->div[i] + 1 + 1 + v_div + div,
						n_div - div) == -1)
			return i;
	return n_div;
}

/* Align the divs of "dst" to those of "src", adding divs from "src"
 * if needed.  That is, make sure that the first src->n_div divs
 * of the result are equal to those of src.
 * The integer division of "src" are assumed to be ordered.
 *
 * The integer divisions are swapped into the right position
 * (possibly after adding them first).  This may result
 * in the remaining integer divisions appearing in the wrong order,
 * i.e., with some integer division appearing before
 * some other integer division on which it depends.
 * The integer divisions therefore need to be ordered.
 * This will not affect the integer divisions aligned to those of "src",
 * since "src" is assumed to have ordered integer divisions.
 *
 * The result is not finalized as by design it will have redundant
 * divs if any divs from "src" were copied.
 */
__isl_give isl_basic_map *isl_basic_map_align_divs(
	__isl_take isl_basic_map *dst, __isl_keep isl_basic_map *src)
{
	int i;
	isl_bool known;
	int extended;
	isl_size v_div;
	isl_size dst_n_div;

	if (!dst || !src)
		return isl_basic_map_free(dst);

	if (src->n_div == 0)
		return dst;

	known = isl_basic_map_divs_known(src);
	if (known < 0)
		return isl_basic_map_free(dst);
	if (!known)
		isl_die(isl_basic_map_get_ctx(src), isl_error_invalid,
			"some src divs are unknown",
			return isl_basic_map_free(dst));

	v_div = isl_basic_map_var_offset(src, isl_dim_div);
	if (v_div < 0)
		return isl_basic_map_free(dst);

	extended = 0;
	dst_n_div = isl_basic_map_dim(dst, isl_dim_div);
	if (dst_n_div < 0)
		dst = isl_basic_map_free(dst);
	for (i = 0; i < src->n_div; ++i) {
		int j = find_div(dst, src, i);
		if (j < 0)
			dst = isl_basic_map_free(dst);
		if (j == dst_n_div) {
			if (!extended) {
				int extra = src->n_div - i;
				dst = isl_basic_map_cow(dst);
				if (!dst)
					return isl_basic_map_free(dst);
				dst = isl_basic_map_extend(dst,
						extra, 0, 2 * extra);
				extended = 1;
			}
			j = isl_basic_map_alloc_div(dst);
			if (j < 0)
				return isl_basic_map_free(dst);
			isl_seq_cpy(dst->div[j], src->div[i], 1+1+v_div+i);
			isl_seq_clr(dst->div[j]+1+1+v_div+i, dst->n_div - i);
			dst_n_div++;
			dst = isl_basic_map_add_div_constraints(dst, j);
			if (!dst)
				return isl_basic_map_free(dst);
		}
		if (j != i)
			dst = isl_basic_map_swap_div(dst, i, j);
		if (!dst)
			return isl_basic_map_free(dst);
	}
	return isl_basic_map_order_divs(dst);
}

__isl_give isl_map *isl_map_align_divs_internal(__isl_take isl_map *map)
{
	int i;

	if (!map)
		return NULL;
	if (map->n == 0)
		return map;
	map = isl_map_compute_divs(map);
	map = isl_map_order_divs(map);
	map = isl_map_cow(map);
	if (!map)
		return NULL;

	for (i = 1; i < map->n; ++i)
		map->p[0] = isl_basic_map_align_divs(map->p[0], map->p[i]);
	for (i = 1; i < map->n; ++i) {
		map->p[i] = isl_basic_map_align_divs(map->p[i], map->p[0]);
		if (!map->p[i])
			return isl_map_free(map);
	}

	map = isl_map_unmark_normalized(map);
	return map;
}

__isl_give isl_map *isl_map_align_divs(__isl_take isl_map *map)
{
	return isl_map_align_divs_internal(map);
}

__isl_give isl_set *isl_set_align_divs(__isl_take isl_set *set)
{
	return set_from_map(isl_map_align_divs_internal(set_to_map(set)));
}

/* Align the divs of the basic maps in "map" to those
 * of the basic maps in "list", as well as to the other basic maps in "map".
 * The elements in "list" are assumed to have known divs.
 */
__isl_give isl_map *isl_map_align_divs_to_basic_map_list(
	__isl_take isl_map *map, __isl_keep isl_basic_map_list *list)
{
	int i;
	isl_size n;

	n = isl_basic_map_list_n_basic_map(list);
	map = isl_map_compute_divs(map);
	map = isl_map_cow(map);
	if (!map || n < 0)
		return isl_map_free(map);
	if (map->n == 0)
		return map;

	for (i = 0; i < n; ++i) {
		isl_basic_map *bmap;

		bmap = isl_basic_map_list_get_basic_map(list, i);
		bmap = isl_basic_map_order_divs(bmap);
		map->p[0] = isl_basic_map_align_divs(map->p[0], bmap);
		isl_basic_map_free(bmap);
	}
	if (!map->p[0])
		return isl_map_free(map);

	return isl_map_align_divs_internal(map);
}

/* Align the divs of each element of "list" to those of "bmap".
 * Both "bmap" and the elements of "list" are assumed to have known divs.
 */
__isl_give isl_basic_map_list *isl_basic_map_list_align_divs_to_basic_map(
	__isl_take isl_basic_map_list *list, __isl_keep isl_basic_map *bmap)
{
	int i;
	isl_size n;

	n = isl_basic_map_list_n_basic_map(list);
	if (n < 0 || !bmap)
		return isl_basic_map_list_free(list);

	for (i = 0; i < n; ++i) {
		isl_basic_map *bmap_i;

		bmap_i = isl_basic_map_list_get_basic_map(list, i);
		bmap_i = isl_basic_map_align_divs(bmap_i, bmap);
		list = isl_basic_map_list_set_basic_map(list, i, bmap_i);
	}

	return list;
}

__isl_give isl_set *isl_set_apply( __isl_take isl_set *set,
	__isl_take isl_map *map)
{
	isl_bool ok;

	isl_map_align_params_set(&map, &set);
	ok = isl_map_compatible_domain(map, set);
	if (ok < 0)
		goto error;
	if (!ok)
		isl_die(isl_set_get_ctx(set), isl_error_invalid,
			"incompatible spaces", goto error);
	map = isl_map_intersect_domain(map, set);
	set = isl_map_range(map);
	return set;
error:
	isl_set_free(set);
	isl_map_free(map);
	return NULL;
}

/* There is no need to cow as removing empty parts doesn't change
 * the meaning of the set.
 */
__isl_give isl_map *isl_map_remove_empty_parts(__isl_take isl_map *map)
{
	int i;

	if (!map)
		return NULL;

	for (i = map->n - 1; i >= 0; --i)
		map = remove_if_empty(map, i);

	return map;
}

__isl_give isl_set *isl_set_remove_empty_parts(__isl_take isl_set *set)
{
	return set_from_map(isl_map_remove_empty_parts(set_to_map(set)));
}

/* Create a binary relation that maps the shared initial "pos" dimensions
 * of "bset1" and "bset2" to the remaining dimensions of "bset1" and "bset2".
 */
static __isl_give isl_basic_map *join_initial(__isl_keep isl_basic_set *bset1,
	__isl_keep isl_basic_set *bset2, int pos)
{
	isl_basic_map *bmap1;
	isl_basic_map *bmap2;

	bmap1 = isl_basic_map_from_range(isl_basic_set_copy(bset1));
	bmap2 = isl_basic_map_from_range(isl_basic_set_copy(bset2));
	bmap1 = isl_basic_map_move_dims(bmap1, isl_dim_in, 0,
					isl_dim_out, 0, pos);
	bmap2 = isl_basic_map_move_dims(bmap2, isl_dim_in, 0,
					isl_dim_out, 0, pos);
	return isl_basic_map_range_product(bmap1, bmap2);
}

/* Given two basic sets bset1 and bset2, compute the maximal difference
 * between the values of dimension pos in bset1 and those in bset2
 * for any common value of the parameters and dimensions preceding pos.
 */
static enum isl_lp_result basic_set_maximal_difference_at(
	__isl_keep isl_basic_set *bset1, __isl_keep isl_basic_set *bset2,
	int pos, isl_int *opt)
{
	isl_basic_map *bmap1;
	struct isl_ctx *ctx;
	struct isl_vec *obj;
	isl_size total;
	isl_size nparam;
	isl_size dim1;
	enum isl_lp_result res;

	nparam = isl_basic_set_dim(bset1, isl_dim_param);
	dim1 = isl_basic_set_dim(bset1, isl_dim_set);
	if (nparam < 0 || dim1 < 0 || !bset2)
		return isl_lp_error;

	bmap1 = join_initial(bset1, bset2, pos);
	total = isl_basic_map_dim(bmap1, isl_dim_all);
	if (total < 0)
		return isl_lp_error;

	ctx = bmap1->ctx;
	obj = isl_vec_alloc(ctx, 1 + total);
	if (!obj)
		goto error;
	isl_seq_clr(obj->block.data, 1 + total);
	isl_int_set_si(obj->block.data[1+nparam+pos], 1);
	isl_int_set_si(obj->block.data[1+nparam+pos+(dim1-pos)], -1);
	res = isl_basic_map_solve_lp(bmap1, 1, obj->block.data, ctx->one,
					opt, NULL, NULL);
	isl_basic_map_free(bmap1);
	isl_vec_free(obj);
	return res;
error:
	isl_basic_map_free(bmap1);
	return isl_lp_error;
}

/* Given two _disjoint_ basic sets bset1 and bset2, check whether
 * for any common value of the parameters and dimensions preceding pos
 * in both basic sets, the values of dimension pos in bset1 are
 * smaller or larger than those in bset2.
 *
 * Returns
 *	 1 if bset1 follows bset2
 *	-1 if bset1 precedes bset2
 *	 0 if bset1 and bset2 are incomparable
 *	-2 if some error occurred.
 */
int isl_basic_set_compare_at(__isl_keep isl_basic_set *bset1,
	__isl_keep isl_basic_set *bset2, int pos)
{
	isl_int opt;
	enum isl_lp_result res;
	int cmp;

	isl_int_init(opt);

	res = basic_set_maximal_difference_at(bset1, bset2, pos, &opt);

	if (res == isl_lp_empty)
		cmp = 0;
	else if ((res == isl_lp_ok && isl_int_is_pos(opt)) ||
		  res == isl_lp_unbounded)
		cmp = 1;
	else if (res == isl_lp_ok && isl_int_is_neg(opt))
		cmp = -1;
	else
		cmp = -2;

	isl_int_clear(opt);
	return cmp;
}

/* Given two basic sets bset1 and bset2, check whether
 * for any common value of the parameters and dimensions preceding pos
 * there is a value of dimension pos in bset1 that is larger
 * than a value of the same dimension in bset2.
 *
 * Return
 *	 1 if there exists such a pair
 *	 0 if there is no such pair, but there is a pair of equal values
 *	-1 otherwise
 *	-2 if some error occurred.
 */
int isl_basic_set_follows_at(__isl_keep isl_basic_set *bset1,
	__isl_keep isl_basic_set *bset2, int pos)
{
	isl_bool empty;
	isl_basic_map *bmap;
	isl_size dim1;

	dim1 = isl_basic_set_dim(bset1, isl_dim_set);
	if (dim1 < 0)
		return -2;
	bmap = join_initial(bset1, bset2, pos);
	bmap = isl_basic_map_order_ge(bmap, isl_dim_out, 0,
					    isl_dim_out, dim1 - pos);
	empty = isl_basic_map_is_empty(bmap);
	if (empty < 0)
		goto error;
	if (empty) {
		isl_basic_map_free(bmap);
		return -1;
	}
	bmap = isl_basic_map_order_gt(bmap, isl_dim_out, 0,
					    isl_dim_out, dim1 - pos);
	empty = isl_basic_map_is_empty(bmap);
	if (empty < 0)
		goto error;
	isl_basic_map_free(bmap);
	if (empty)
		return 0;
	return 1;
error:
	isl_basic_map_free(bmap);
	return -2;
}

/* Given two sets set1 and set2, check whether
 * for any common value of the parameters and dimensions preceding pos
 * there is a value of dimension pos in set1 that is larger
 * than a value of the same dimension in set2.
 *
 * Return
 *	 1 if there exists such a pair
 *	 0 if there is no such pair, but there is a pair of equal values
 *	-1 otherwise
 *	-2 if some error occurred.
 */
int isl_set_follows_at(__isl_keep isl_set *set1,
	__isl_keep isl_set *set2, int pos)
{
	int i, j;
	int follows = -1;

	if (!set1 || !set2)
		return -2;

	for (i = 0; i < set1->n; ++i)
		for (j = 0; j < set2->n; ++j) {
			int f;
			f = isl_basic_set_follows_at(set1->p[i], set2->p[j], pos);
			if (f == 1 || f == -2)
				return f;
			if (f > follows)
				follows = f;
		}

	return follows;
}

static isl_bool isl_basic_map_plain_has_fixed_var(
	__isl_keep isl_basic_map *bmap, unsigned pos, isl_int *val)
{
	int i;
	int d;
	isl_size total;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return isl_bool_error;
	for (i = 0, d = total-1; i < bmap->n_eq && d+1 > pos; ++i) {
		for (; d+1 > pos; --d)
			if (!isl_int_is_zero(bmap->eq[i][1+d]))
				break;
		if (d != pos)
			continue;
		if (isl_seq_first_non_zero(bmap->eq[i]+1, d) != -1)
			return isl_bool_false;
		if (isl_seq_first_non_zero(bmap->eq[i]+1+d+1, total-d-1) != -1)
			return isl_bool_false;
		if (!isl_int_is_one(bmap->eq[i][1+d]))
			return isl_bool_false;
		if (val)
			isl_int_neg(*val, bmap->eq[i][0]);
		return isl_bool_true;
	}
	return isl_bool_false;
}

static isl_bool isl_map_plain_has_fixed_var(__isl_keep isl_map *map,
	unsigned pos, isl_int *val)
{
	int i;
	isl_int v;
	isl_int tmp;
	isl_bool fixed;

	if (!map)
		return isl_bool_error;
	if (map->n == 0)
		return isl_bool_false;
	if (map->n == 1)
		return isl_basic_map_plain_has_fixed_var(map->p[0], pos, val); 
	isl_int_init(v);
	isl_int_init(tmp);
	fixed = isl_basic_map_plain_has_fixed_var(map->p[0], pos, &v); 
	for (i = 1; fixed == isl_bool_true && i < map->n; ++i) {
		fixed = isl_basic_map_plain_has_fixed_var(map->p[i], pos, &tmp); 
		if (fixed == isl_bool_true && isl_int_ne(tmp, v))
			fixed = isl_bool_false;
	}
	if (val)
		isl_int_set(*val, v);
	isl_int_clear(tmp);
	isl_int_clear(v);
	return fixed;
}

static isl_bool isl_basic_set_plain_has_fixed_var(
	__isl_keep isl_basic_set *bset, unsigned pos, isl_int *val)
{
	return isl_basic_map_plain_has_fixed_var(bset_to_bmap(bset),
						pos, val);
}

isl_bool isl_basic_map_plain_is_fixed(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos, isl_int *val)
{
	if (isl_basic_map_check_range(bmap, type, pos, 1) < 0)
		return isl_bool_error;
	return isl_basic_map_plain_has_fixed_var(bmap,
		isl_basic_map_offset(bmap, type) - 1 + pos, val);
}

/* If "bmap" obviously lies on a hyperplane where the given dimension
 * has a fixed value, then return that value.
 * Otherwise return NaN.
 */
__isl_give isl_val *isl_basic_map_plain_get_val_if_fixed(
	__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos)
{
	isl_ctx *ctx;
	isl_val *v;
	isl_bool fixed;

	if (!bmap)
		return NULL;
	ctx = isl_basic_map_get_ctx(bmap);
	v = isl_val_alloc(ctx);
	if (!v)
		return NULL;
	fixed = isl_basic_map_plain_is_fixed(bmap, type, pos, &v->n);
	if (fixed < 0)
		return isl_val_free(v);
	if (fixed) {
		isl_int_set_si(v->d, 1);
		return v;
	}
	isl_val_free(v);
	return isl_val_nan(ctx);
}

isl_bool isl_map_plain_is_fixed(__isl_keep isl_map *map,
	enum isl_dim_type type, unsigned pos, isl_int *val)
{
	if (isl_map_check_range(map, type, pos, 1) < 0)
		return isl_bool_error;
	return isl_map_plain_has_fixed_var(map,
		map_offset(map, type) - 1 + pos, val);
}

/* If "map" obviously lies on a hyperplane where the given dimension
 * has a fixed value, then return that value.
 * Otherwise return NaN.
 */
__isl_give isl_val *isl_map_plain_get_val_if_fixed(__isl_keep isl_map *map,
	enum isl_dim_type type, unsigned pos)
{
	isl_ctx *ctx;
	isl_val *v;
	isl_bool fixed;

	if (!map)
		return NULL;
	ctx = isl_map_get_ctx(map);
	v = isl_val_alloc(ctx);
	if (!v)
		return NULL;
	fixed = isl_map_plain_is_fixed(map, type, pos, &v->n);
	if (fixed < 0)
		return isl_val_free(v);
	if (fixed) {
		isl_int_set_si(v->d, 1);
		return v;
	}
	isl_val_free(v);
	return isl_val_nan(ctx);
}

/* If "set" obviously lies on a hyperplane where the given dimension
 * has a fixed value, then return that value.
 * Otherwise return NaN.
 */
__isl_give isl_val *isl_set_plain_get_val_if_fixed(__isl_keep isl_set *set,
	enum isl_dim_type type, unsigned pos)
{
	return isl_map_plain_get_val_if_fixed(set, type, pos);
}

/* Return a sequence of values in the same space as "set"
 * that are equal to the corresponding set dimensions of "set"
 * for those set dimensions that obviously lie on a hyperplane
 * where the dimension has a fixed value.
 * The other elements are set to NaN.
 */
__isl_give isl_multi_val *isl_set_get_plain_multi_val_if_fixed(
	__isl_keep isl_set *set)
{
	int i;
	isl_size n;
	isl_space *space;
	isl_multi_val *mv;

	space = isl_space_drop_all_params(isl_set_get_space(set));
	mv = isl_multi_val_alloc(space);
	n = isl_multi_val_size(mv);
	if (n < 0)
		return isl_multi_val_free(mv);

	for (i = 0; i < n; ++i) {
		isl_val *v;

		v = isl_set_plain_get_val_if_fixed(set, isl_dim_set, i);
		mv = isl_multi_val_set_val(mv, i, v);
	}

	return mv;
}

/* Check if dimension dim has fixed value and if so and if val is not NULL,
 * then return this fixed value in *val.
 */
isl_bool isl_basic_set_plain_dim_is_fixed(__isl_keep isl_basic_set *bset,
	unsigned dim, isl_int *val)
{
	isl_size nparam;

	nparam = isl_basic_set_dim(bset, isl_dim_param);
	if (nparam < 0)
		return isl_bool_error;
	return isl_basic_set_plain_has_fixed_var(bset, nparam + dim, val);
}

/* Return -1 if the constraint "c1" should be sorted before "c2"
 * and 1 if it should be sorted after "c2".
 * Return 0 if the two constraints are the same (up to the constant term).
 *
 * In particular, if a constraint involves later variables than another
 * then it is sorted after this other constraint.
 * uset_gist depends on constraints without existentially quantified
 * variables sorting first.
 *
 * For constraints that have the same latest variable, those
 * with the same coefficient for this latest variable (first in absolute value
 * and then in actual value) are grouped together.
 * This is useful for detecting pairs of constraints that can
 * be chained in their printed representation.
 *
 * Finally, within a group, constraints are sorted according to
 * their coefficients (excluding the constant term).
 */
static int sort_constraint_cmp(const void *p1, const void *p2, void *arg)
{
	isl_int **c1 = (isl_int **) p1;
	isl_int **c2 = (isl_int **) p2;
	int l1, l2;
	unsigned size = *(unsigned *) arg;
	int cmp;

	l1 = isl_seq_last_non_zero(*c1 + 1, size);
	l2 = isl_seq_last_non_zero(*c2 + 1, size);

	if (l1 != l2)
		return l1 - l2;

	cmp = isl_int_abs_cmp((*c1)[1 + l1], (*c2)[1 + l1]);
	if (cmp != 0)
		return cmp;
	cmp = isl_int_cmp((*c1)[1 + l1], (*c2)[1 + l1]);
	if (cmp != 0)
		return -cmp;

	return isl_seq_cmp(*c1 + 1, *c2 + 1, size);
}

/* Return -1 if the constraint "c1" of "bmap" is sorted before "c2"
 * by isl_basic_map_sort_constraints, 1 if it is sorted after "c2"
 * and 0 if the two constraints are the same (up to the constant term).
 */
int isl_basic_map_constraint_cmp(__isl_keep isl_basic_map *bmap,
	isl_int *c1, isl_int *c2)
{
	isl_size total;
	unsigned size;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return -2;
	size = total;
	return sort_constraint_cmp(&c1, &c2, &size);
}

__isl_give isl_basic_map *isl_basic_map_sort_constraints(
	__isl_take isl_basic_map *bmap)
{
	isl_size total;
	unsigned size;

	if (!bmap)
		return NULL;
	if (bmap->n_ineq == 0)
		return bmap;
	if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_SORTED))
		return bmap;
	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return isl_basic_map_free(bmap);
	size = total;
	if (isl_sort(bmap->ineq, bmap->n_ineq, sizeof(isl_int *),
		    &sort_constraint_cmp, &size) < 0)
		return isl_basic_map_free(bmap);
	ISL_F_SET(bmap, ISL_BASIC_MAP_SORTED);
	return bmap;
}

__isl_give isl_basic_set *isl_basic_set_sort_constraints(
	__isl_take isl_basic_set *bset)
{
	isl_basic_map *bmap = bset_to_bmap(bset);
	return bset_from_bmap(isl_basic_map_sort_constraints(bmap));
}

__isl_give isl_basic_map *isl_basic_map_normalize(
	__isl_take isl_basic_map *bmap)
{
	bmap = isl_basic_map_remove_redundancies(bmap);
	bmap = isl_basic_map_sort_constraints(bmap);
	return bmap;
}
int isl_basic_map_plain_cmp(__isl_keep isl_basic_map *bmap1,
	__isl_keep isl_basic_map *bmap2)
{
	int i, cmp;
	isl_size total;
	isl_space *space1, *space2;

	if (!bmap1 || !bmap2)
		return -1;

	if (bmap1 == bmap2)
		return 0;
	space1 = isl_basic_map_peek_space(bmap1);
	space2 = isl_basic_map_peek_space(bmap2);
	cmp = isl_space_cmp(space1, space2);
	if (cmp)
		return cmp;
	if (ISL_F_ISSET(bmap1, ISL_BASIC_MAP_RATIONAL) !=
	    ISL_F_ISSET(bmap2, ISL_BASIC_MAP_RATIONAL))
		return ISL_F_ISSET(bmap1, ISL_BASIC_MAP_RATIONAL) ? -1 : 1;
	if (ISL_F_ISSET(bmap1, ISL_BASIC_MAP_EMPTY) &&
	    ISL_F_ISSET(bmap2, ISL_BASIC_MAP_EMPTY))
		return 0;
	if (ISL_F_ISSET(bmap1, ISL_BASIC_MAP_EMPTY))
		return 1;
	if (ISL_F_ISSET(bmap2, ISL_BASIC_MAP_EMPTY))
		return -1;
	if (bmap1->n_eq != bmap2->n_eq)
		return bmap1->n_eq - bmap2->n_eq;
	if (bmap1->n_ineq != bmap2->n_ineq)
		return bmap1->n_ineq - bmap2->n_ineq;
	if (bmap1->n_div != bmap2->n_div)
		return bmap1->n_div - bmap2->n_div;
	total = isl_basic_map_dim(bmap1, isl_dim_all);
	if (total < 0)
		return -1;
	for (i = 0; i < bmap1->n_eq; ++i) {
		cmp = isl_seq_cmp(bmap1->eq[i], bmap2->eq[i], 1+total);
		if (cmp)
			return cmp;
	}
	for (i = 0; i < bmap1->n_ineq; ++i) {
		cmp = isl_seq_cmp(bmap1->ineq[i], bmap2->ineq[i], 1+total);
		if (cmp)
			return cmp;
	}
	for (i = 0; i < bmap1->n_div; ++i) {
		cmp = isl_seq_cmp(bmap1->div[i], bmap2->div[i], 1+1+total);
		if (cmp)
			return cmp;
	}
	return 0;
}

int isl_basic_set_plain_cmp(__isl_keep isl_basic_set *bset1,
	__isl_keep isl_basic_set *bset2)
{
	return isl_basic_map_plain_cmp(bset1, bset2);
}

int isl_set_plain_cmp(__isl_keep isl_set *set1, __isl_keep isl_set *set2)
{
	int i, cmp;

	if (set1 == set2)
		return 0;
	if (set1->n != set2->n)
		return set1->n - set2->n;

	for (i = 0; i < set1->n; ++i) {
		cmp = isl_basic_set_plain_cmp(set1->p[i], set2->p[i]);
		if (cmp)
			return cmp;
	}

	return 0;
}

isl_bool isl_basic_map_plain_is_equal(__isl_keep isl_basic_map *bmap1,
	__isl_keep isl_basic_map *bmap2)
{
	if (!bmap1 || !bmap2)
		return isl_bool_error;
	return isl_basic_map_plain_cmp(bmap1, bmap2) == 0;
}

isl_bool isl_basic_set_plain_is_equal(__isl_keep isl_basic_set *bset1,
	__isl_keep isl_basic_set *bset2)
{
	return isl_basic_map_plain_is_equal(bset_to_bmap(bset1),
					    bset_to_bmap(bset2));
}

static int qsort_bmap_cmp(const void *p1, const void *p2)
{
	isl_basic_map *bmap1 = *(isl_basic_map **) p1;
	isl_basic_map *bmap2 = *(isl_basic_map **) p2;

	return isl_basic_map_plain_cmp(bmap1, bmap2);
}

/* Sort the basic maps of "map" and remove duplicate basic maps.
 *
 * While removing basic maps, we make sure that the basic maps remain
 * sorted because isl_map_normalize expects the basic maps of the result
 * to be sorted.
 */
static __isl_give isl_map *sort_and_remove_duplicates(__isl_take isl_map *map)
{
	int i, j;

	map = isl_map_remove_empty_parts(map);
	if (!map)
		return NULL;
	qsort(map->p, map->n, sizeof(struct isl_basic_map *), qsort_bmap_cmp);
	for (i = map->n - 1; i >= 1; --i) {
		if (!isl_basic_map_plain_is_equal(map->p[i - 1], map->p[i]))
			continue;
		isl_basic_map_free(map->p[i-1]);
		for (j = i; j < map->n; ++j)
			map->p[j - 1] = map->p[j];
		map->n--;
	}

	return map;
}

/* Remove obvious duplicates among the basic maps of "map".
 *
 * Unlike isl_map_normalize, this function does not remove redundant
 * constraints and only removes duplicates that have exactly the same
 * constraints in the input.  It does sort the constraints and
 * the basic maps to ease the detection of duplicates.
 *
 * If "map" has already been normalized or if the basic maps are
 * disjoint, then there can be no duplicates.
 */
__isl_give isl_map *isl_map_remove_obvious_duplicates(__isl_take isl_map *map)
{
	int i;
	isl_basic_map *bmap;

	if (!map)
		return NULL;
	if (map->n <= 1)
		return map;
	if (ISL_F_ISSET(map, ISL_MAP_NORMALIZED | ISL_MAP_DISJOINT))
		return map;
	for (i = 0; i < map->n; ++i) {
		bmap = isl_basic_map_copy(map->p[i]);
		bmap = isl_basic_map_sort_constraints(bmap);
		if (!bmap)
			return isl_map_free(map);
		isl_basic_map_free(map->p[i]);
		map->p[i] = bmap;
	}

	map = sort_and_remove_duplicates(map);
	return map;
}

/* We normalize in place, but if anything goes wrong we need
 * to return NULL, so we need to make sure we don't change the
 * meaning of any possible other copies of map.
 */
__isl_give isl_map *isl_map_normalize(__isl_take isl_map *map)
{
	int i;
	struct isl_basic_map *bmap;

	if (!map)
		return NULL;
	if (ISL_F_ISSET(map, ISL_MAP_NORMALIZED))
		return map;
	for (i = 0; i < map->n; ++i) {
		bmap = isl_basic_map_normalize(isl_basic_map_copy(map->p[i]));
		if (!bmap)
			goto error;
		isl_basic_map_free(map->p[i]);
		map->p[i] = bmap;
	}

	map = sort_and_remove_duplicates(map);
	if (map)
		ISL_F_SET(map, ISL_MAP_NORMALIZED);
	return map;
error:
	isl_map_free(map);
	return NULL;
}

__isl_give isl_set *isl_set_normalize(__isl_take isl_set *set)
{
	return set_from_map(isl_map_normalize(set_to_map(set)));
}

isl_bool isl_map_plain_is_equal(__isl_keep isl_map *map1,
	__isl_keep isl_map *map2)
{
	int i;
	isl_bool equal;

	if (!map1 || !map2)
		return isl_bool_error;

	if (map1 == map2)
		return isl_bool_true;
	equal = isl_map_has_equal_space(map1, map2);
	if (equal < 0 || !equal)
		return equal;

	map1 = isl_map_copy(map1);
	map2 = isl_map_copy(map2);
	map1 = isl_map_normalize(map1);
	map2 = isl_map_normalize(map2);
	if (!map1 || !map2)
		goto error;
	equal = map1->n == map2->n;
	for (i = 0; equal && i < map1->n; ++i) {
		equal = isl_basic_map_plain_is_equal(map1->p[i], map2->p[i]);
		if (equal < 0)
			goto error;
	}
	isl_map_free(map1);
	isl_map_free(map2);
	return equal;
error:
	isl_map_free(map1);
	isl_map_free(map2);
	return isl_bool_error;
}

isl_bool isl_set_plain_is_equal(__isl_keep isl_set *set1,
	__isl_keep isl_set *set2)
{
	return isl_map_plain_is_equal(set_to_map(set1), set_to_map(set2));
}

/* Return the basic maps in "map" as a list.
 */
__isl_give isl_basic_map_list *isl_map_get_basic_map_list(
	__isl_keep isl_map *map)
{
	int i;
	isl_ctx *ctx;
	isl_basic_map_list *list;

	if (!map)
		return NULL;
	ctx = isl_map_get_ctx(map);
	list = isl_basic_map_list_alloc(ctx, map->n);

	for (i = 0; i < map->n; ++i) {
		isl_basic_map *bmap;

		bmap = isl_basic_map_copy(map->p[i]);
		list = isl_basic_map_list_add(list, bmap);
	}

	return list;
}

/* Return the intersection of the elements in the non-empty list "list".
 * All elements are assumed to live in the same space.
 */
__isl_give isl_basic_map *isl_basic_map_list_intersect(
	__isl_take isl_basic_map_list *list)
{
	int i;
	isl_size n;
	isl_basic_map *bmap;

	n = isl_basic_map_list_n_basic_map(list);
	if (n < 0)
		goto error;
	if (n < 1)
		isl_die(isl_basic_map_list_get_ctx(list), isl_error_invalid,
			"expecting non-empty list", goto error);

	bmap = isl_basic_map_list_get_basic_map(list, 0);
	for (i = 1; i < n; ++i) {
		isl_basic_map *bmap_i;

		bmap_i = isl_basic_map_list_get_basic_map(list, i);
		bmap = isl_basic_map_intersect(bmap, bmap_i);
	}

	isl_basic_map_list_free(list);
	return bmap;
error:
	isl_basic_map_list_free(list);
	return NULL;
}

/* Return the intersection of the elements in the non-empty list "list".
 * All elements are assumed to live in the same space.
 */
__isl_give isl_basic_set *isl_basic_set_list_intersect(
	__isl_take isl_basic_set_list *list)
{
	return isl_basic_map_list_intersect(list);
}

/* Return the union of the elements of "list".
 * The list is required to have at least one element.
 */
__isl_give isl_set *isl_basic_set_list_union(
	__isl_take isl_basic_set_list *list)
{
	int i;
	isl_size n;
	isl_space *space;
	isl_basic_set *bset;
	isl_set *set;

	n = isl_basic_set_list_n_basic_set(list);
	if (n < 0)
		goto error;
	if (n < 1)
		isl_die(isl_basic_set_list_get_ctx(list), isl_error_invalid,
			"expecting non-empty list", goto error);

	bset = isl_basic_set_list_get_basic_set(list, 0);
	space = isl_basic_set_get_space(bset);
	isl_basic_set_free(bset);

	set = isl_set_alloc_space(space, n, 0);
	for (i = 0; i < n; ++i) {
		bset = isl_basic_set_list_get_basic_set(list, i);
		set = isl_set_add_basic_set(set, bset);
	}

	isl_basic_set_list_free(list);
	return set;
error:
	isl_basic_set_list_free(list);
	return NULL;
}

/* Return the union of the elements in the non-empty list "list".
 * All elements are assumed to live in the same space.
 */
__isl_give isl_set *isl_set_list_union(__isl_take isl_set_list *list)
{
	int i;
	isl_size n;
	isl_set *set;

	n = isl_set_list_n_set(list);
	if (n < 0)
		goto error;
	if (n < 1)
		isl_die(isl_set_list_get_ctx(list), isl_error_invalid,
			"expecting non-empty list", goto error);

	set = isl_set_list_get_set(list, 0);
	for (i = 1; i < n; ++i) {
		isl_set *set_i;

		set_i = isl_set_list_get_set(list, i);
		set = isl_set_union(set, set_i);
	}

	isl_set_list_free(list);
	return set;
error:
	isl_set_list_free(list);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_product(
	__isl_take isl_basic_map *bmap1, __isl_take isl_basic_map *bmap2)
{
	isl_space *space_result = NULL;
	struct isl_basic_map *bmap;
	unsigned in1, in2, out1, out2, nparam, total, pos;
	struct isl_dim_map *dim_map1, *dim_map2;

	if (isl_basic_map_check_equal_params(bmap1, bmap2) < 0)
		goto error;
	space_result = isl_space_product(isl_space_copy(bmap1->dim),
						   isl_space_copy(bmap2->dim));

	in1 = isl_basic_map_dim(bmap1, isl_dim_in);
	in2 = isl_basic_map_dim(bmap2, isl_dim_in);
	out1 = isl_basic_map_dim(bmap1, isl_dim_out);
	out2 = isl_basic_map_dim(bmap2, isl_dim_out);
	nparam = isl_basic_map_dim(bmap1, isl_dim_param);

	total = nparam + in1 + in2 + out1 + out2 + bmap1->n_div + bmap2->n_div;
	dim_map1 = isl_dim_map_alloc(bmap1->ctx, total);
	dim_map2 = isl_dim_map_alloc(bmap1->ctx, total);
	isl_dim_map_dim(dim_map1, bmap1->dim, isl_dim_param, pos = 0);
	isl_dim_map_dim(dim_map2, bmap2->dim, isl_dim_param, pos = 0);
	isl_dim_map_dim(dim_map1, bmap1->dim, isl_dim_in, pos += nparam);
	isl_dim_map_dim(dim_map2, bmap2->dim, isl_dim_in, pos += in1);
	isl_dim_map_dim(dim_map1, bmap1->dim, isl_dim_out, pos += in2);
	isl_dim_map_dim(dim_map2, bmap2->dim, isl_dim_out, pos += out1);
	isl_dim_map_div(dim_map1, bmap1, pos += out2);
	isl_dim_map_div(dim_map2, bmap2, pos += bmap1->n_div);

	bmap = isl_basic_map_alloc_space(space_result,
			bmap1->n_div + bmap2->n_div,
			bmap1->n_eq + bmap2->n_eq,
			bmap1->n_ineq + bmap2->n_ineq);
	bmap = isl_basic_map_add_constraints_dim_map(bmap, bmap1, dim_map1);
	bmap = isl_basic_map_add_constraints_dim_map(bmap, bmap2, dim_map2);
	bmap = isl_basic_map_simplify(bmap);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap1);
	isl_basic_map_free(bmap2);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_flat_product(
	__isl_take isl_basic_map *bmap1, __isl_take isl_basic_map *bmap2)
{
	isl_basic_map *prod;

	prod = isl_basic_map_product(bmap1, bmap2);
	prod = isl_basic_map_flatten(prod);
	return prod;
}

__isl_give isl_basic_set *isl_basic_set_flat_product(
	__isl_take isl_basic_set *bset1, __isl_take isl_basic_set *bset2)
{
	return isl_basic_map_flat_range_product(bset1, bset2);
}

__isl_give isl_basic_map *isl_basic_map_domain_product(
	__isl_take isl_basic_map *bmap1, __isl_take isl_basic_map *bmap2)
{
	isl_space *space1, *space2;
	isl_space *space_result = NULL;
	isl_basic_map *bmap;
	isl_size in1, in2, out, nparam;
	unsigned total, pos;
	struct isl_dim_map *dim_map1, *dim_map2;

	in1 = isl_basic_map_dim(bmap1, isl_dim_in);
	in2 = isl_basic_map_dim(bmap2, isl_dim_in);
	out = isl_basic_map_dim(bmap1, isl_dim_out);
	nparam = isl_basic_map_dim(bmap1, isl_dim_param);
	if (in1 < 0 || in2 < 0 || out < 0 || nparam < 0)
		goto error;

	space1 = isl_basic_map_get_space(bmap1);
	space2 = isl_basic_map_get_space(bmap2);
	space_result = isl_space_domain_product(space1, space2);

	total = nparam + in1 + in2 + out + bmap1->n_div + bmap2->n_div;
	dim_map1 = isl_dim_map_alloc(bmap1->ctx, total);
	dim_map2 = isl_dim_map_alloc(bmap1->ctx, total);
	isl_dim_map_dim(dim_map1, bmap1->dim, isl_dim_param, pos = 0);
	isl_dim_map_dim(dim_map2, bmap2->dim, isl_dim_param, pos = 0);
	isl_dim_map_dim(dim_map1, bmap1->dim, isl_dim_in, pos += nparam);
	isl_dim_map_dim(dim_map2, bmap2->dim, isl_dim_in, pos += in1);
	isl_dim_map_dim(dim_map1, bmap1->dim, isl_dim_out, pos += in2);
	isl_dim_map_dim(dim_map2, bmap2->dim, isl_dim_out, pos);
	isl_dim_map_div(dim_map1, bmap1, pos += out);
	isl_dim_map_div(dim_map2, bmap2, pos += bmap1->n_div);

	bmap = isl_basic_map_alloc_space(space_result,
			bmap1->n_div + bmap2->n_div,
			bmap1->n_eq + bmap2->n_eq,
			bmap1->n_ineq + bmap2->n_ineq);
	bmap = isl_basic_map_add_constraints_dim_map(bmap, bmap1, dim_map1);
	bmap = isl_basic_map_add_constraints_dim_map(bmap, bmap2, dim_map2);
	bmap = isl_basic_map_simplify(bmap);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap1);
	isl_basic_map_free(bmap2);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_range_product(
	__isl_take isl_basic_map *bmap1, __isl_take isl_basic_map *bmap2)
{
	isl_bool rational;
	isl_space *space_result = NULL;
	isl_basic_map *bmap;
	isl_size in, out1, out2, nparam;
	unsigned total, pos;
	struct isl_dim_map *dim_map1, *dim_map2;

	rational = isl_basic_map_is_rational(bmap1);
	if (rational >= 0 && rational)
		rational = isl_basic_map_is_rational(bmap2);
	in = isl_basic_map_dim(bmap1, isl_dim_in);
	out1 = isl_basic_map_dim(bmap1, isl_dim_out);
	out2 = isl_basic_map_dim(bmap2, isl_dim_out);
	nparam = isl_basic_map_dim(bmap1, isl_dim_param);
	if (in < 0 || out1 < 0 || out2 < 0 || nparam < 0 || rational < 0)
		goto error;

	if (isl_basic_map_check_equal_params(bmap1, bmap2) < 0)
		goto error;

	space_result = isl_space_range_product(isl_space_copy(bmap1->dim),
					   isl_space_copy(bmap2->dim));

	total = nparam + in + out1 + out2 + bmap1->n_div + bmap2->n_div;
	dim_map1 = isl_dim_map_alloc(bmap1->ctx, total);
	dim_map2 = isl_dim_map_alloc(bmap1->ctx, total);
	isl_dim_map_dim(dim_map1, bmap1->dim, isl_dim_param, pos = 0);
	isl_dim_map_dim(dim_map2, bmap2->dim, isl_dim_param, pos = 0);
	isl_dim_map_dim(dim_map1, bmap1->dim, isl_dim_in, pos += nparam);
	isl_dim_map_dim(dim_map2, bmap2->dim, isl_dim_in, pos);
	isl_dim_map_dim(dim_map1, bmap1->dim, isl_dim_out, pos += in);
	isl_dim_map_dim(dim_map2, bmap2->dim, isl_dim_out, pos += out1);
	isl_dim_map_div(dim_map1, bmap1, pos += out2);
	isl_dim_map_div(dim_map2, bmap2, pos += bmap1->n_div);

	bmap = isl_basic_map_alloc_space(space_result,
			bmap1->n_div + bmap2->n_div,
			bmap1->n_eq + bmap2->n_eq,
			bmap1->n_ineq + bmap2->n_ineq);
	bmap = isl_basic_map_add_constraints_dim_map(bmap, bmap1, dim_map1);
	bmap = isl_basic_map_add_constraints_dim_map(bmap, bmap2, dim_map2);
	if (rational)
		bmap = isl_basic_map_set_rational(bmap);
	bmap = isl_basic_map_simplify(bmap);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap1);
	isl_basic_map_free(bmap2);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_flat_range_product(
	__isl_take isl_basic_map *bmap1, __isl_take isl_basic_map *bmap2)
{
	isl_basic_map *prod;

	prod = isl_basic_map_range_product(bmap1, bmap2);
	prod = isl_basic_map_flatten_range(prod);
	return prod;
}

/* Apply "basic_map_product" to each pair of basic maps in "map1" and "map2"
 * and collect the results.
 * The result live in the space obtained by calling "space_product"
 * on the spaces of "map1" and "map2".
 * If "remove_duplicates" is set then the result may contain duplicates
 * (even if the inputs do not) and so we try and remove the obvious
 * duplicates.
 */
static __isl_give isl_map *map_product(__isl_take isl_map *map1,
	__isl_take isl_map *map2,
	__isl_give isl_space *(*space_product)(__isl_take isl_space *left,
					   __isl_take isl_space *right),
	__isl_give isl_basic_map *(*basic_map_product)(
		__isl_take isl_basic_map *left,
		__isl_take isl_basic_map *right),
	int remove_duplicates)
{
	unsigned flags = 0;
	struct isl_map *result;
	int i, j;
	isl_bool m;

	m = isl_map_has_equal_params(map1, map2);
	if (m < 0)
		goto error;
	if (!m)
		isl_die(isl_map_get_ctx(map1), isl_error_invalid,
			"parameters don't match", goto error);

	if (ISL_F_ISSET(map1, ISL_MAP_DISJOINT) &&
	    ISL_F_ISSET(map2, ISL_MAP_DISJOINT))
		ISL_FL_SET(flags, ISL_MAP_DISJOINT);

	result = isl_map_alloc_space(space_product(isl_space_copy(map1->dim),
					       isl_space_copy(map2->dim)),
				map1->n * map2->n, flags);
	if (!result)
		goto error;
	for (i = 0; i < map1->n; ++i)
		for (j = 0; j < map2->n; ++j) {
			struct isl_basic_map *part;
			part = basic_map_product(isl_basic_map_copy(map1->p[i]),
						 isl_basic_map_copy(map2->p[j]));
			if (isl_basic_map_is_empty(part))
				isl_basic_map_free(part);
			else
				result = isl_map_add_basic_map(result, part);
			if (!result)
				goto error;
		}
	if (remove_duplicates)
		result = isl_map_remove_obvious_duplicates(result);
	isl_map_free(map1);
	isl_map_free(map2);
	return result;
error:
	isl_map_free(map1);
	isl_map_free(map2);
	return NULL;
}

/* Given two maps A -> B and C -> D, construct a map [A -> C] -> [B -> D]
 */
__isl_give isl_map *isl_map_product(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	isl_map_align_params_bin(&map1, &map2);
	return map_product(map1, map2, &isl_space_product,
			&isl_basic_map_product, 0);
}

/* Given two maps A -> B and C -> D, construct a map (A, C) -> (B, D)
 */
__isl_give isl_map *isl_map_flat_product(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	isl_map *prod;

	prod = isl_map_product(map1, map2);
	prod = isl_map_flatten(prod);
	return prod;
}

/* Given two set A and B, construct its Cartesian product A x B.
 */
__isl_give isl_set *isl_set_product(__isl_take isl_set *set1,
	__isl_take isl_set *set2)
{
	return isl_map_range_product(set1, set2);
}

__isl_give isl_set *isl_set_flat_product(__isl_take isl_set *set1,
	__isl_take isl_set *set2)
{
	return isl_map_flat_range_product(set1, set2);
}

/* Given two maps A -> B and C -> D, construct a map [A -> C] -> (B * D)
 */
__isl_give isl_map *isl_map_domain_product(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	isl_map_align_params_bin(&map1, &map2);
	return map_product(map1, map2, &isl_space_domain_product,
				&isl_basic_map_domain_product, 1);
}

/* Given two maps A -> B and C -> D, construct a map (A * C) -> [B -> D]
 */
__isl_give isl_map *isl_map_range_product(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	isl_map_align_params_bin(&map1, &map2);
	return map_product(map1, map2, &isl_space_range_product,
				&isl_basic_map_range_product, 1);
}

/* Given a map of the form [A -> B] -> [C -> D], return the map A -> C.
 */
__isl_give isl_map *isl_map_factor_domain(__isl_take isl_map *map)
{
	isl_space *space;
	isl_size total1, keep1, total2, keep2;

	total1 = isl_map_dim(map, isl_dim_in);
	total2 = isl_map_dim(map, isl_dim_out);
	if (total1 < 0 || total2 < 0)
		return isl_map_free(map);
	if (!isl_space_domain_is_wrapping(map->dim) ||
	    !isl_space_range_is_wrapping(map->dim))
		isl_die(isl_map_get_ctx(map), isl_error_invalid,
			"not a product", return isl_map_free(map));

	space = isl_map_get_space(map);
	space = isl_space_factor_domain(space);
	keep1 = isl_space_dim(space, isl_dim_in);
	keep2 = isl_space_dim(space, isl_dim_out);
	if (keep1 < 0 || keep2 < 0)
		map = isl_map_free(map);
	map = isl_map_project_out(map, isl_dim_in, keep1, total1 - keep1);
	map = isl_map_project_out(map, isl_dim_out, keep2, total2 - keep2);
	map = isl_map_reset_space(map, space);

	return map;
}

/* Given a map of the form [A -> B] -> [C -> D], return the map B -> D.
 */
__isl_give isl_map *isl_map_factor_range(__isl_take isl_map *map)
{
	isl_space *space;
	isl_size total1, keep1, total2, keep2;

	total1 = isl_map_dim(map, isl_dim_in);
	total2 = isl_map_dim(map, isl_dim_out);
	if (total1 < 0 || total2 < 0)
		return isl_map_free(map);
	if (!isl_space_domain_is_wrapping(map->dim) ||
	    !isl_space_range_is_wrapping(map->dim))
		isl_die(isl_map_get_ctx(map), isl_error_invalid,
			"not a product", return isl_map_free(map));

	space = isl_map_get_space(map);
	space = isl_space_factor_range(space);
	keep1 = isl_space_dim(space, isl_dim_in);
	keep2 = isl_space_dim(space, isl_dim_out);
	if (keep1 < 0 || keep2 < 0)
		map = isl_map_free(map);
	map = isl_map_project_out(map, isl_dim_in, 0, total1 - keep1);
	map = isl_map_project_out(map, isl_dim_out, 0, total2 - keep2);
	map = isl_map_reset_space(map, space);

	return map;
}

/* Given a map of the form [A -> B] -> C, return the map A -> C.
 */
__isl_give isl_map *isl_map_domain_factor_domain(__isl_take isl_map *map)
{
	isl_space *space;
	isl_size total, keep;

	total = isl_map_dim(map, isl_dim_in);
	if (total < 0)
		return isl_map_free(map);
	if (!isl_space_domain_is_wrapping(map->dim))
		isl_die(isl_map_get_ctx(map), isl_error_invalid,
			"domain is not a product", return isl_map_free(map));

	space = isl_map_get_space(map);
	space = isl_space_domain_factor_domain(space);
	keep = isl_space_dim(space, isl_dim_in);
	if (keep < 0)
		map = isl_map_free(map);
	map = isl_map_project_out(map, isl_dim_in, keep, total - keep);
	map = isl_map_reset_space(map, space);

	return map;
}

/* Given a map of the form [A -> B] -> C, return the map B -> C.
 */
__isl_give isl_map *isl_map_domain_factor_range(__isl_take isl_map *map)
{
	isl_space *space;
	isl_size total, keep;

	total = isl_map_dim(map, isl_dim_in);
	if (total < 0)
		return isl_map_free(map);
	if (!isl_space_domain_is_wrapping(map->dim))
		isl_die(isl_map_get_ctx(map), isl_error_invalid,
			"domain is not a product", return isl_map_free(map));

	space = isl_map_get_space(map);
	space = isl_space_domain_factor_range(space);
	keep = isl_space_dim(space, isl_dim_in);
	if (keep < 0)
		map = isl_map_free(map);
	map = isl_map_project_out(map, isl_dim_in, 0, total - keep);
	map = isl_map_reset_space(map, space);

	return map;
}

/* Given a map A -> [B -> C], extract the map A -> B.
 */
__isl_give isl_map *isl_map_range_factor_domain(__isl_take isl_map *map)
{
	isl_space *space;
	isl_size total, keep;

	total = isl_map_dim(map, isl_dim_out);
	if (total < 0)
		return isl_map_free(map);
	if (!isl_space_range_is_wrapping(map->dim))
		isl_die(isl_map_get_ctx(map), isl_error_invalid,
			"range is not a product", return isl_map_free(map));

	space = isl_map_get_space(map);
	space = isl_space_range_factor_domain(space);
	keep = isl_space_dim(space, isl_dim_out);
	if (keep < 0)
		map = isl_map_free(map);
	map = isl_map_project_out(map, isl_dim_out, keep, total - keep);
	map = isl_map_reset_space(map, space);

	return map;
}

/* Given a map A -> [B -> C], extract the map A -> C.
 */
__isl_give isl_map *isl_map_range_factor_range(__isl_take isl_map *map)
{
	isl_space *space;
	isl_size total, keep;

	total = isl_map_dim(map, isl_dim_out);
	if (total < 0)
		return isl_map_free(map);
	if (!isl_space_range_is_wrapping(map->dim))
		isl_die(isl_map_get_ctx(map), isl_error_invalid,
			"range is not a product", return isl_map_free(map));

	space = isl_map_get_space(map);
	space = isl_space_range_factor_range(space);
	keep = isl_space_dim(space, isl_dim_out);
	if (keep < 0)
		map = isl_map_free(map);
	map = isl_map_project_out(map, isl_dim_out, 0, total - keep);
	map = isl_map_reset_space(map, space);

	return map;
}

/* Given two maps A -> B and C -> D, construct a map (A, C) -> (B * D)
 */
__isl_give isl_map *isl_map_flat_domain_product(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	isl_map *prod;

	prod = isl_map_domain_product(map1, map2);
	prod = isl_map_flatten_domain(prod);
	return prod;
}

/* Given two maps A -> B and C -> D, construct a map (A * C) -> (B, D)
 */
__isl_give isl_map *isl_map_flat_range_product(__isl_take isl_map *map1,
	__isl_take isl_map *map2)
{
	isl_map *prod;

	prod = isl_map_range_product(map1, map2);
	prod = isl_map_flatten_range(prod);
	return prod;
}

uint32_t isl_basic_map_get_hash(__isl_keep isl_basic_map *bmap)
{
	int i;
	uint32_t hash = isl_hash_init();
	isl_size total;

	if (!bmap)
		return 0;
	bmap = isl_basic_map_copy(bmap);
	bmap = isl_basic_map_normalize(bmap);
	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return 0;
	isl_hash_byte(hash, bmap->n_eq & 0xFF);
	for (i = 0; i < bmap->n_eq; ++i) {
		uint32_t c_hash;
		c_hash = isl_seq_get_hash(bmap->eq[i], 1 + total);
		isl_hash_hash(hash, c_hash);
	}
	isl_hash_byte(hash, bmap->n_ineq & 0xFF);
	for (i = 0; i < bmap->n_ineq; ++i) {
		uint32_t c_hash;
		c_hash = isl_seq_get_hash(bmap->ineq[i], 1 + total);
		isl_hash_hash(hash, c_hash);
	}
	isl_hash_byte(hash, bmap->n_div & 0xFF);
	for (i = 0; i < bmap->n_div; ++i) {
		uint32_t c_hash;
		if (isl_int_is_zero(bmap->div[i][0]))
			continue;
		isl_hash_byte(hash, i & 0xFF);
		c_hash = isl_seq_get_hash(bmap->div[i], 1 + 1 + total);
		isl_hash_hash(hash, c_hash);
	}
	isl_basic_map_free(bmap);
	return hash;
}

uint32_t isl_basic_set_get_hash(__isl_keep isl_basic_set *bset)
{
	return isl_basic_map_get_hash(bset_to_bmap(bset));
}

uint32_t isl_map_get_hash(__isl_keep isl_map *map)
{
	int i;
	uint32_t hash;

	if (!map)
		return 0;
	map = isl_map_copy(map);
	map = isl_map_normalize(map);
	if (!map)
		return 0;

	hash = isl_hash_init();
	for (i = 0; i < map->n; ++i) {
		uint32_t bmap_hash;
		bmap_hash = isl_basic_map_get_hash(map->p[i]);
		isl_hash_hash(hash, bmap_hash);
	}
		
	isl_map_free(map);

	return hash;
}

uint32_t isl_set_get_hash(__isl_keep isl_set *set)
{
	return isl_map_get_hash(set_to_map(set));
}

/* Return the number of basic maps in the (current) representation of "map".
 */
isl_size isl_map_n_basic_map(__isl_keep isl_map *map)
{
	return map ? map->n : isl_size_error;
}

isl_size isl_set_n_basic_set(__isl_keep isl_set *set)
{
	return set ? set->n : isl_size_error;
}

isl_stat isl_map_foreach_basic_map(__isl_keep isl_map *map,
	isl_stat (*fn)(__isl_take isl_basic_map *bmap, void *user), void *user)
{
	int i;

	if (!map)
		return isl_stat_error;

	for (i = 0; i < map->n; ++i)
		if (fn(isl_basic_map_copy(map->p[i]), user) < 0)
			return isl_stat_error;

	return isl_stat_ok;
}

isl_stat isl_set_foreach_basic_set(__isl_keep isl_set *set,
	isl_stat (*fn)(__isl_take isl_basic_set *bset, void *user), void *user)
{
	int i;

	if (!set)
		return isl_stat_error;

	for (i = 0; i < set->n; ++i)
		if (fn(isl_basic_set_copy(set->p[i]), user) < 0)
			return isl_stat_error;

	return isl_stat_ok;
}

/* Does "test" succeed on every basic set in "set"?
 */
isl_bool isl_set_every_basic_set(__isl_keep isl_set *set,
	isl_bool (*test)(__isl_keep isl_basic_set *bset, void *user),
	void *user)
{
	int i;

	if (!set)
		return isl_bool_error;

	for (i = 0; i < set->n; ++i) {
		isl_bool r;

		r = test(set->p[i], user);
		if (r < 0 || !r)
			return r;
	}

	return isl_bool_true;
}

/* Return a list of basic sets, the union of which is equal to "set".
 */
__isl_give isl_basic_set_list *isl_set_get_basic_set_list(
	__isl_keep isl_set *set)
{
	int i;
	isl_basic_set_list *list;

	if (!set)
		return NULL;

	list = isl_basic_set_list_alloc(isl_set_get_ctx(set), set->n);
	for (i = 0; i < set->n; ++i) {
		isl_basic_set *bset;

		bset = isl_basic_set_copy(set->p[i]);
		list = isl_basic_set_list_add(list, bset);
	}

	return list;
}

__isl_give isl_basic_set *isl_basic_set_lift(__isl_take isl_basic_set *bset)
{
	isl_space *space;

	if (!bset)
		return NULL;

	bset = isl_basic_set_cow(bset);
	if (!bset)
		return NULL;

	space = isl_basic_set_get_space(bset);
	space = isl_space_lift(space, bset->n_div);
	if (!space)
		goto error;
	isl_space_free(bset->dim);
	bset->dim = space;
	bset->extra -= bset->n_div;
	bset->n_div = 0;

	bset = isl_basic_set_finalize(bset);

	return bset;
error:
	isl_basic_set_free(bset);
	return NULL;
}

__isl_give isl_set *isl_set_lift(__isl_take isl_set *set)
{
	int i;
	isl_space *space;
	unsigned n_div;

	set = set_from_map(isl_map_align_divs_internal(set_to_map(set)));

	if (!set)
		return NULL;

	set = isl_set_cow(set);
	if (!set)
		return NULL;

	n_div = set->p[0]->n_div;
	space = isl_set_get_space(set);
	space = isl_space_lift(space, n_div);
	if (!space)
		goto error;
	isl_space_free(set->dim);
	set->dim = space;

	for (i = 0; i < set->n; ++i) {
		set->p[i] = isl_basic_set_lift(set->p[i]);
		if (!set->p[i])
			goto error;
	}

	return set;
error:
	isl_set_free(set);
	return NULL;
}

int isl_basic_set_size(__isl_keep isl_basic_set *bset)
{
	isl_size dim;
	int size = 0;

	dim = isl_basic_set_dim(bset, isl_dim_all);
	if (dim < 0)
		return -1;
	size += bset->n_eq * (1 + dim);
	size += bset->n_ineq * (1 + dim);
	size += bset->n_div * (2 + dim);

	return size;
}

int isl_set_size(__isl_keep isl_set *set)
{
	int i;
	int size = 0;

	if (!set)
		return -1;

	for (i = 0; i < set->n; ++i)
		size += isl_basic_set_size(set->p[i]);

	return size;
}

/* Check if there is any lower bound (if lower == 0) and/or upper
 * bound (if upper == 0) on the specified dim.
 */
static isl_bool basic_map_dim_is_bounded(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos, int lower, int upper)
{
	int i;

	if (isl_basic_map_check_range(bmap, type, pos, 1) < 0)
		return isl_bool_error;

	pos += isl_basic_map_offset(bmap, type);

	for (i = 0; i < bmap->n_div; ++i) {
		if (isl_int_is_zero(bmap->div[i][0]))
			continue;
		if (!isl_int_is_zero(bmap->div[i][1 + pos]))
			return isl_bool_true;
	}

	for (i = 0; i < bmap->n_eq; ++i)
		if (!isl_int_is_zero(bmap->eq[i][pos]))
			return isl_bool_true;

	for (i = 0; i < bmap->n_ineq; ++i) {
		int sgn = isl_int_sgn(bmap->ineq[i][pos]);
		if (sgn > 0)
			lower = 1;
		if (sgn < 0)
			upper = 1;
	}

	return lower && upper;
}

isl_bool isl_basic_map_dim_is_bounded(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos)
{
	return basic_map_dim_is_bounded(bmap, type, pos, 0, 0);
}

isl_bool isl_basic_map_dim_has_lower_bound(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos)
{
	return basic_map_dim_is_bounded(bmap, type, pos, 0, 1);
}

isl_bool isl_basic_map_dim_has_upper_bound(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos)
{
	return basic_map_dim_is_bounded(bmap, type, pos, 1, 0);
}

isl_bool isl_map_dim_is_bounded(__isl_keep isl_map *map,
	enum isl_dim_type type, unsigned pos)
{
	int i;

	if (!map)
		return isl_bool_error;

	for (i = 0; i < map->n; ++i) {
		isl_bool bounded;
		bounded = isl_basic_map_dim_is_bounded(map->p[i], type, pos);
		if (bounded < 0 || !bounded)
			return bounded;
	}

	return isl_bool_true;
}

/* Return true if the specified dim is involved in both an upper bound
 * and a lower bound.
 */
isl_bool isl_set_dim_is_bounded(__isl_keep isl_set *set,
	enum isl_dim_type type, unsigned pos)
{
	return isl_map_dim_is_bounded(set_to_map(set), type, pos);
}

/* Does "map" have a bound (according to "fn") for any of its basic maps?
 */
static isl_bool has_any_bound(__isl_keep isl_map *map,
	enum isl_dim_type type, unsigned pos,
	isl_bool (*fn)(__isl_keep isl_basic_map *bmap,
		  enum isl_dim_type type, unsigned pos))
{
	int i;

	if (!map)
		return isl_bool_error;

	for (i = 0; i < map->n; ++i) {
		isl_bool bounded;
		bounded = fn(map->p[i], type, pos);
		if (bounded < 0 || bounded)
			return bounded;
	}

	return isl_bool_false;
}

/* Return 1 if the specified dim is involved in any lower bound.
 */
isl_bool isl_set_dim_has_any_lower_bound(__isl_keep isl_set *set,
	enum isl_dim_type type, unsigned pos)
{
	return has_any_bound(set, type, pos,
				&isl_basic_map_dim_has_lower_bound);
}

/* Return 1 if the specified dim is involved in any upper bound.
 */
isl_bool isl_set_dim_has_any_upper_bound(__isl_keep isl_set *set,
	enum isl_dim_type type, unsigned pos)
{
	return has_any_bound(set, type, pos,
				&isl_basic_map_dim_has_upper_bound);
}

/* Does "map" have a bound (according to "fn") for all of its basic maps?
 */
static isl_bool has_bound(__isl_keep isl_map *map,
	enum isl_dim_type type, unsigned pos,
	isl_bool (*fn)(__isl_keep isl_basic_map *bmap,
		  enum isl_dim_type type, unsigned pos))
{
	int i;

	if (!map)
		return isl_bool_error;

	for (i = 0; i < map->n; ++i) {
		isl_bool bounded;
		bounded = fn(map->p[i], type, pos);
		if (bounded < 0 || !bounded)
			return bounded;
	}

	return isl_bool_true;
}

/* Return 1 if the specified dim has a lower bound (in each of its basic sets).
 */
isl_bool isl_set_dim_has_lower_bound(__isl_keep isl_set *set,
	enum isl_dim_type type, unsigned pos)
{
	return has_bound(set, type, pos, &isl_basic_map_dim_has_lower_bound);
}

/* Return 1 if the specified dim has an upper bound (in each of its basic sets).
 */
isl_bool isl_set_dim_has_upper_bound(__isl_keep isl_set *set,
	enum isl_dim_type type, unsigned pos)
{
	return has_bound(set, type, pos, &isl_basic_map_dim_has_upper_bound);
}

/* For each of the "n" variables starting at "first", determine
 * the sign of the variable and put the results in the first "n"
 * elements of the array "signs".
 * Sign
 *	1 means that the variable is non-negative
 *	-1 means that the variable is non-positive
 *	0 means the variable attains both positive and negative values.
 */
isl_stat isl_basic_set_vars_get_sign(__isl_keep isl_basic_set *bset,
	unsigned first, unsigned n, int *signs)
{
	isl_vec *bound = NULL;
	struct isl_tab *tab = NULL;
	struct isl_tab_undo *snap;
	int i;
	isl_size total;

	total = isl_basic_set_dim(bset, isl_dim_all);
	if (total < 0 || !signs)
		return isl_stat_error;

	bound = isl_vec_alloc(bset->ctx, 1 + total);
	tab = isl_tab_from_basic_set(bset, 0);
	if (!bound || !tab)
		goto error;

	isl_seq_clr(bound->el, bound->size);
	isl_int_set_si(bound->el[0], -1);

	snap = isl_tab_snap(tab);
	for (i = 0; i < n; ++i) {
		int empty;

		isl_int_set_si(bound->el[1 + first + i], -1);
		if (isl_tab_add_ineq(tab, bound->el) < 0)
			goto error;
		empty = tab->empty;
		isl_int_set_si(bound->el[1 + first + i], 0);
		if (isl_tab_rollback(tab, snap) < 0)
			goto error;

		if (empty) {
			signs[i] = 1;
			continue;
		}

		isl_int_set_si(bound->el[1 + first + i], 1);
		if (isl_tab_add_ineq(tab, bound->el) < 0)
			goto error;
		empty = tab->empty;
		isl_int_set_si(bound->el[1 + first + i], 0);
		if (isl_tab_rollback(tab, snap) < 0)
			goto error;

		signs[i] = empty ? -1 : 0;
	}

	isl_tab_free(tab);
	isl_vec_free(bound);
	return isl_stat_ok;
error:
	isl_tab_free(tab);
	isl_vec_free(bound);
	return isl_stat_error;
}

isl_stat isl_basic_set_dims_get_sign(__isl_keep isl_basic_set *bset,
	enum isl_dim_type type, unsigned first, unsigned n, int *signs)
{
	if (!bset || !signs)
		return isl_stat_error;
	if (isl_basic_set_check_range(bset, type, first, n) < 0)
		return isl_stat_error;

	first += pos(bset->dim, type) - 1;
	return isl_basic_set_vars_get_sign(bset, first, n, signs);
}

/* Is it possible for the integer division "div" to depend (possibly
 * indirectly) on any output dimensions?
 *
 * If the div is undefined, then we conservatively assume that it
 * may depend on them.
 * Otherwise, we check if it actually depends on them or on any integer
 * divisions that may depend on them.
 */
static isl_bool div_may_involve_output(__isl_keep isl_basic_map *bmap, int div)
{
	int i;
	isl_size n_out, n_div;
	unsigned o_out, o_div;

	if (isl_int_is_zero(bmap->div[div][0]))
		return isl_bool_true;

	n_out = isl_basic_map_dim(bmap, isl_dim_out);
	if (n_out < 0)
		return isl_bool_error;
	o_out = isl_basic_map_offset(bmap, isl_dim_out);

	if (isl_seq_first_non_zero(bmap->div[div] + 1 + o_out, n_out) != -1)
		return isl_bool_true;

	n_div = isl_basic_map_dim(bmap, isl_dim_div);
	if (n_div < 0)
		return isl_bool_error;
	o_div = isl_basic_map_offset(bmap, isl_dim_div);

	for (i = 0; i < n_div; ++i) {
		isl_bool may_involve;

		if (isl_int_is_zero(bmap->div[div][1 + o_div + i]))
			continue;
		may_involve = div_may_involve_output(bmap, i);
		if (may_involve < 0 || may_involve)
			return may_involve;
	}

	return isl_bool_false;
}

/* Return the first integer division of "bmap" in the range
 * [first, first + n[ that may depend on any output dimensions and
 * that has a non-zero coefficient in "c" (where the first coefficient
 * in "c" corresponds to integer division "first").
 */
static int first_div_may_involve_output(__isl_keep isl_basic_map *bmap,
	isl_int *c, int first, int n)
{
	int k;

	if (!bmap)
		return -1;

	for (k = first; k < first + n; ++k) {
		isl_bool may_involve;

		if (isl_int_is_zero(c[k]))
			continue;
		may_involve = div_may_involve_output(bmap, k);
		if (may_involve < 0)
			return -1;
		if (may_involve)
			return k;
	}

	return first + n;
}

/* Look for a pair of inequality constraints in "bmap" of the form
 *
 *	-l + i >= 0		or		i >= l
 * and
 *	n + l - i >= 0		or		i <= l + n
 *
 * with n < "m" and i the output dimension at position "pos".
 * (Note that n >= 0 as otherwise the two constraints would conflict.)
 * Furthermore, "l" is only allowed to involve parameters, input dimensions
 * and earlier output dimensions, as well as integer divisions that do
 * not involve any of the output dimensions.
 *
 * Return the index of the first inequality constraint or bmap->n_ineq
 * if no such pair can be found.
 */
static int find_modulo_constraint_pair(__isl_keep isl_basic_map *bmap,
	int pos, isl_int m)
{
	int i, j;
	isl_ctx *ctx;
	isl_size total;
	isl_size n_div, n_out;
	unsigned o_div, o_out;
	int less;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	n_out = isl_basic_map_dim(bmap, isl_dim_out);
	n_div = isl_basic_map_dim(bmap, isl_dim_div);
	if (total < 0 || n_out < 0 || n_div < 0)
		return -1;

	ctx = isl_basic_map_get_ctx(bmap);
	o_out = isl_basic_map_offset(bmap, isl_dim_out);
	o_div = isl_basic_map_offset(bmap, isl_dim_div);
	for (i = 0; i < bmap->n_ineq; ++i) {
		if (!isl_int_abs_eq(bmap->ineq[i][o_out + pos], ctx->one))
			continue;
		if (isl_seq_first_non_zero(bmap->ineq[i] + o_out + pos + 1,
					n_out - (pos + 1)) != -1)
			continue;
		if (first_div_may_involve_output(bmap, bmap->ineq[i] + o_div,
						0, n_div) < n_div)
			continue;
		for (j = i + 1; j < bmap->n_ineq; ++j) {
			if (!isl_int_abs_eq(bmap->ineq[j][o_out + pos],
					    ctx->one))
				continue;
			if (!isl_seq_is_neg(bmap->ineq[i] + 1,
					    bmap->ineq[j] + 1, total))
				continue;
			break;
		}
		if (j >= bmap->n_ineq)
			continue;
		isl_int_add(bmap->ineq[i][0],
			    bmap->ineq[i][0], bmap->ineq[j][0]);
		less = isl_int_abs_lt(bmap->ineq[i][0], m);
		isl_int_sub(bmap->ineq[i][0],
			    bmap->ineq[i][0], bmap->ineq[j][0]);
		if (!less)
			continue;
		if (isl_int_is_one(bmap->ineq[i][o_out + pos]))
			return i;
		else
			return j;
	}

	return bmap->n_ineq;
}

/* Return the index of the equality of "bmap" that defines
 * the output dimension "pos" in terms of earlier dimensions.
 * The equality may also involve integer divisions, as long
 * as those integer divisions are defined in terms of
 * parameters or input dimensions.
 * In this case, *div is set to the number of integer divisions and
 * *ineq is set to the number of inequality constraints (provided
 * div and ineq are not NULL).
 *
 * The equality may also involve a single integer division involving
 * the output dimensions (typically only output dimension "pos") as
 * long as the coefficient of output dimension "pos" is 1 or -1 and
 * there is a pair of constraints i >= l and i <= l + n, with i referring
 * to output dimension "pos", l an expression involving only earlier
 * dimensions and n smaller than the coefficient of the integer division
 * in the equality.  In this case, the output dimension can be defined
 * in terms of a modulo expression that does not involve the integer division.
 * *div is then set to this single integer division and
 * *ineq is set to the index of constraint i >= l.
 *
 * Return bmap->n_eq if there is no such equality.
 * Return -1 on error.
 */
int isl_basic_map_output_defining_equality(__isl_keep isl_basic_map *bmap,
	int pos, int *div, int *ineq)
{
	int j, k, l;
	isl_size n_div, n_out;
	unsigned o_div, o_out;

	n_out = isl_basic_map_dim(bmap, isl_dim_out);
	n_div = isl_basic_map_dim(bmap, isl_dim_div);
	if (n_out < 0 || n_div < 0)
		return -1;

	o_out = isl_basic_map_offset(bmap, isl_dim_out);
	o_div = isl_basic_map_offset(bmap, isl_dim_div);

	if (ineq)
		*ineq = bmap->n_ineq;
	if (div)
		*div = n_div;
	for (j = 0; j < bmap->n_eq; ++j) {
		if (isl_int_is_zero(bmap->eq[j][o_out + pos]))
			continue;
		if (isl_seq_first_non_zero(bmap->eq[j] + o_out + pos + 1,
					n_out - (pos + 1)) != -1)
			continue;
		k = first_div_may_involve_output(bmap, bmap->eq[j] + o_div,
						0, n_div);
		if (k >= n_div)
			return j;
		if (!isl_int_is_one(bmap->eq[j][o_out + pos]) &&
		    !isl_int_is_negone(bmap->eq[j][o_out + pos]))
			continue;
		if (first_div_may_involve_output(bmap, bmap->eq[j] + o_div,
						k + 1, n_div - (k+1)) < n_div)
			continue;
		l = find_modulo_constraint_pair(bmap, pos,
						bmap->eq[j][o_div + k]);
		if (l < 0)
			return -1;
		if (l >= bmap->n_ineq)
			continue;
		if (div)
			*div = k;
		if (ineq)
			*ineq = l;
		return j;
	}

	return bmap->n_eq;
}

/* Check if the given basic map is obviously single-valued.
 * In particular, for each output dimension, check that there is
 * an equality that defines the output dimension in terms of
 * earlier dimensions.
 */
isl_bool isl_basic_map_plain_is_single_valued(__isl_keep isl_basic_map *bmap)
{
	int i;
	isl_size n_out;

	n_out = isl_basic_map_dim(bmap, isl_dim_out);
	if (n_out < 0)
		return isl_bool_error;

	for (i = 0; i < n_out; ++i) {
		int eq;

		eq = isl_basic_map_output_defining_equality(bmap, i,
							    NULL, NULL);
		if (eq < 0)
			return isl_bool_error;
		if (eq >= bmap->n_eq)
			return isl_bool_false;
	}

	return isl_bool_true;
}

/* Check if the given basic map is single-valued.
 * We simply compute
 *
 *	M \circ M^-1
 *
 * and check if the result is a subset of the identity mapping.
 */
isl_bool isl_basic_map_is_single_valued(__isl_keep isl_basic_map *bmap)
{
	isl_space *space;
	isl_basic_map *test;
	isl_basic_map *id;
	isl_bool sv;

	sv = isl_basic_map_plain_is_single_valued(bmap);
	if (sv < 0 || sv)
		return sv;

	test = isl_basic_map_reverse(isl_basic_map_copy(bmap));
	test = isl_basic_map_apply_range(test, isl_basic_map_copy(bmap));

	space = isl_basic_map_get_space(bmap);
	space = isl_space_map_from_set(isl_space_range(space));
	id = isl_basic_map_identity(space);

	sv = isl_basic_map_is_subset(test, id);

	isl_basic_map_free(test);
	isl_basic_map_free(id);

	return sv;
}

/* Check if the given map is obviously single-valued.
 */
isl_bool isl_map_plain_is_single_valued(__isl_keep isl_map *map)
{
	if (!map)
		return isl_bool_error;
	if (map->n == 0)
		return isl_bool_true;
	if (map->n >= 2)
		return isl_bool_false;

	return isl_basic_map_plain_is_single_valued(map->p[0]);
}

/* Check if the given map is single-valued.
 * We simply compute
 *
 *	M \circ M^-1
 *
 * and check if the result is a subset of the identity mapping.
 */
isl_bool isl_map_is_single_valued(__isl_keep isl_map *map)
{
	isl_space *space;
	isl_map *test;
	isl_map *id;
	isl_bool sv;

	sv = isl_map_plain_is_single_valued(map);
	if (sv < 0 || sv)
		return sv;

	test = isl_map_reverse(isl_map_copy(map));
	test = isl_map_apply_range(test, isl_map_copy(map));

	space = isl_space_map_from_set(isl_space_range(isl_map_get_space(map)));
	id = isl_map_identity(space);

	sv = isl_map_is_subset(test, id);

	isl_map_free(test);
	isl_map_free(id);

	return sv;
}

isl_bool isl_map_is_injective(__isl_keep isl_map *map)
{
	isl_bool in;

	map = isl_map_copy(map);
	map = isl_map_reverse(map);
	in = isl_map_is_single_valued(map);
	isl_map_free(map);

	return in;
}

/* Check if the given map is obviously injective.
 */
isl_bool isl_map_plain_is_injective(__isl_keep isl_map *map)
{
	isl_bool in;

	map = isl_map_copy(map);
	map = isl_map_reverse(map);
	in = isl_map_plain_is_single_valued(map);
	isl_map_free(map);

	return in;
}

isl_bool isl_map_is_bijective(__isl_keep isl_map *map)
{
	isl_bool sv;

	sv = isl_map_is_single_valued(map);
	if (sv < 0 || !sv)
		return sv;

	return isl_map_is_injective(map);
}

isl_bool isl_set_is_singleton(__isl_keep isl_set *set)
{
	return isl_map_is_single_valued(set_to_map(set));
}

/* Does "map" only map elements to themselves?
 *
 * If the domain and range spaces are different, then "map"
 * is considered not to be an identity relation, even if it is empty.
 * Otherwise, construct the maximal identity relation and
 * check whether "map" is a subset of this relation.
 */
isl_bool isl_map_is_identity(__isl_keep isl_map *map)
{
	isl_map *id;
	isl_bool equal, is_identity;

	equal = isl_map_tuple_is_equal(map, isl_dim_in, map, isl_dim_out);
	if (equal < 0 || !equal)
		return equal;

	id = isl_map_identity(isl_map_get_space(map));
	is_identity = isl_map_is_subset(map, id);
	isl_map_free(id);

	return is_identity;
}

int isl_map_is_translation(__isl_keep isl_map *map)
{
	int ok;
	isl_set *delta;

	delta = isl_map_deltas(isl_map_copy(map));
	ok = isl_set_is_singleton(delta);
	isl_set_free(delta);

	return ok;
}

static int unique(isl_int *p, unsigned pos, unsigned len)
{
	if (isl_seq_first_non_zero(p, pos) != -1)
		return 0;
	if (isl_seq_first_non_zero(p + pos + 1, len - pos - 1) != -1)
		return 0;
	return 1;
}

isl_bool isl_basic_set_is_box(__isl_keep isl_basic_set *bset)
{
	int i, j;
	isl_size nvar, n_div;
	unsigned ovar;

	n_div = isl_basic_set_dim(bset, isl_dim_div);
	if (n_div < 0)
		return isl_bool_error;
	if (n_div != 0)
		return isl_bool_false;

	nvar = isl_basic_set_dim(bset, isl_dim_set);
	if (nvar < 0)
		return isl_bool_error;
	ovar = isl_space_offset(bset->dim, isl_dim_set);
	for (j = 0; j < nvar; ++j) {
		int lower = 0, upper = 0;
		for (i = 0; i < bset->n_eq; ++i) {
			if (isl_int_is_zero(bset->eq[i][1 + ovar + j]))
				continue;
			if (!unique(bset->eq[i] + 1 + ovar, j, nvar))
				return isl_bool_false;
			break;
		}
		if (i < bset->n_eq)
			continue;
		for (i = 0; i < bset->n_ineq; ++i) {
			if (isl_int_is_zero(bset->ineq[i][1 + ovar + j]))
				continue;
			if (!unique(bset->ineq[i] + 1 + ovar, j, nvar))
				return isl_bool_false;
			if (isl_int_is_pos(bset->ineq[i][1 + ovar + j]))
				lower = 1;
			else
				upper = 1;
		}
		if (!lower || !upper)
			return isl_bool_false;
	}

	return isl_bool_true;
}

isl_bool isl_set_is_box(__isl_keep isl_set *set)
{
	if (!set)
		return isl_bool_error;
	if (set->n != 1)
		return isl_bool_false;

	return isl_basic_set_is_box(set->p[0]);
}

isl_bool isl_basic_set_is_wrapping(__isl_keep isl_basic_set *bset)
{
	if (!bset)
		return isl_bool_error;
	
	return isl_space_is_wrapping(bset->dim);
}

isl_bool isl_set_is_wrapping(__isl_keep isl_set *set)
{
	if (!set)
		return isl_bool_error;
	
	return isl_space_is_wrapping(set->dim);
}

/* Modify the space of "map" through a call to "change".
 * If "can_change" is set (not NULL), then first call it to check
 * if the modification is allowed, printing the error message "cannot_change"
 * if it is not.
 */
static __isl_give isl_map *isl_map_change_space(__isl_take isl_map *map,
	isl_bool (*can_change)(__isl_keep isl_map *map),
	const char *cannot_change,
	__isl_give isl_space *(*change)(__isl_take isl_space *space))
{
	isl_bool ok;
	isl_space *space;

	if (!map)
		return NULL;

	ok = can_change ? can_change(map) : isl_bool_true;
	if (ok < 0)
		return isl_map_free(map);
	if (!ok)
		isl_die(isl_map_get_ctx(map), isl_error_invalid, cannot_change,
			return isl_map_free(map));

	space = change(isl_map_get_space(map));
	map = isl_map_reset_space(map, space);

	return map;
}

/* Is the domain of "map" a wrapped relation?
 */
isl_bool isl_map_domain_is_wrapping(__isl_keep isl_map *map)
{
	if (!map)
		return isl_bool_error;

	return isl_space_domain_is_wrapping(map->dim);
}

/* Does "map" have a wrapped relation in both domain and range?
 */
isl_bool isl_map_is_product(__isl_keep isl_map *map)
{
	return isl_space_is_product(isl_map_peek_space(map));
}

/* Is the range of "map" a wrapped relation?
 */
isl_bool isl_map_range_is_wrapping(__isl_keep isl_map *map)
{
	if (!map)
		return isl_bool_error;

	return isl_space_range_is_wrapping(map->dim);
}

__isl_give isl_basic_set *isl_basic_map_wrap(__isl_take isl_basic_map *bmap)
{
	isl_space *space;

	space = isl_basic_map_take_space(bmap);
	space = isl_space_wrap(space);
	bmap = isl_basic_map_restore_space(bmap, space);

	bmap = isl_basic_map_finalize(bmap);

	return bset_from_bmap(bmap);
}

/* Given a map A -> B, return the set (A -> B).
 */
__isl_give isl_set *isl_map_wrap(__isl_take isl_map *map)
{
	return isl_map_change_space(map, NULL, NULL, &isl_space_wrap);
}

__isl_give isl_basic_map *isl_basic_set_unwrap(__isl_take isl_basic_set *bset)
{
	bset = isl_basic_set_cow(bset);
	if (!bset)
		return NULL;

	bset->dim = isl_space_unwrap(bset->dim);
	if (!bset->dim)
		goto error;

	bset = isl_basic_set_finalize(bset);

	return bset_to_bmap(bset);
error:
	isl_basic_set_free(bset);
	return NULL;
}

/* Given a set (A -> B), return the map A -> B.
 * Error out if "set" is not of the form (A -> B).
 */
__isl_give isl_map *isl_set_unwrap(__isl_take isl_set *set)
{
	return isl_map_change_space(set, &isl_set_is_wrapping,
				    "not a wrapping set", &isl_space_unwrap);
}

__isl_give isl_basic_map *isl_basic_map_reset(__isl_take isl_basic_map *bmap,
	enum isl_dim_type type)
{
	isl_space *space;

	space = isl_basic_map_take_space(bmap);
	space = isl_space_reset(space, type);
	bmap = isl_basic_map_restore_space(bmap, space);

	bmap = isl_basic_map_mark_final(bmap);

	return bmap;
}

__isl_give isl_map *isl_map_reset(__isl_take isl_map *map,
	enum isl_dim_type type)
{
	int i;
	isl_space *space;

	if (!map)
		return NULL;

	if (!isl_space_is_named_or_nested(map->dim, type))
		return map;

	map = isl_map_cow(map);
	if (!map)
		return NULL;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_reset(map->p[i], type);
		if (!map->p[i])
			goto error;
	}

	space = isl_map_take_space(map);
	space = isl_space_reset(space, type);
	map = isl_map_restore_space(map, space);

	return map;
error:
	isl_map_free(map);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_flatten(__isl_take isl_basic_map *bmap)
{
	isl_space *space;

	space = isl_basic_map_take_space(bmap);
	space = isl_space_flatten(space);
	bmap = isl_basic_map_restore_space(bmap, space);

	bmap = isl_basic_map_mark_final(bmap);

	return bmap;
}

__isl_give isl_basic_set *isl_basic_set_flatten(__isl_take isl_basic_set *bset)
{
	return bset_from_bmap(isl_basic_map_flatten(bset_to_bmap(bset)));
}

__isl_give isl_basic_map *isl_basic_map_flatten_domain(
	__isl_take isl_basic_map *bmap)
{
	isl_space *space;

	space = isl_basic_map_take_space(bmap);
	space = isl_space_flatten_domain(space);
	bmap = isl_basic_map_restore_space(bmap, space);

	bmap = isl_basic_map_mark_final(bmap);

	return bmap;
}

__isl_give isl_basic_map *isl_basic_map_flatten_range(
	__isl_take isl_basic_map *bmap)
{
	isl_space *space;

	space = isl_basic_map_take_space(bmap);
	space = isl_space_flatten_range(space);
	bmap = isl_basic_map_restore_space(bmap, space);

	bmap = isl_basic_map_mark_final(bmap);

	return bmap;
}

/* Remove any internal structure from the spaces of domain and range of "map".
 */
__isl_give isl_map *isl_map_flatten(__isl_take isl_map *map)
{
	if (!map)
		return NULL;

	if (!map->dim->nested[0] && !map->dim->nested[1])
		return map;

	return isl_map_change_space(map, NULL, NULL, &isl_space_flatten);
}

__isl_give isl_set *isl_set_flatten(__isl_take isl_set *set)
{
	return set_from_map(isl_map_flatten(set_to_map(set)));
}

__isl_give isl_map *isl_set_flatten_map(__isl_take isl_set *set)
{
	isl_space *space, *flat_space;
	isl_map *map;

	space = isl_set_get_space(set);
	flat_space = isl_space_flatten(isl_space_copy(space));
	map = isl_map_identity(isl_space_join(isl_space_reverse(space),
						flat_space));
	map = isl_map_intersect_domain(map, set);

	return map;
}

/* Remove any internal structure from the space of the domain of "map".
 */
__isl_give isl_map *isl_map_flatten_domain(__isl_take isl_map *map)
{
	if (!map)
		return NULL;

	if (!map->dim->nested[0])
		return map;

	return isl_map_change_space(map, NULL, NULL, &isl_space_flatten_domain);
}

/* Remove any internal structure from the space of the range of "map".
 */
__isl_give isl_map *isl_map_flatten_range(__isl_take isl_map *map)
{
	if (!map)
		return NULL;

	if (!map->dim->nested[1])
		return map;

	return isl_map_change_space(map, NULL, NULL, &isl_space_flatten_range);
}

/* Reorder the dimensions of "bmap" according to the given dim_map
 * and set the dimension specification to "space" and
 * perform Gaussian elimination on the result.
 */
__isl_give isl_basic_map *isl_basic_map_realign(__isl_take isl_basic_map *bmap,
	__isl_take isl_space *space, __isl_take struct isl_dim_map *dim_map)
{
	isl_basic_map *res;
	unsigned flags;
	isl_size n_div;

	n_div = isl_basic_map_dim(bmap, isl_dim_div);
	if (n_div < 0 || !space || !dim_map)
		goto error;

	flags = bmap->flags;
	ISL_FL_CLR(flags, ISL_BASIC_MAP_FINAL);
	ISL_FL_CLR(flags, ISL_BASIC_MAP_SORTED);
	ISL_FL_CLR(flags, ISL_BASIC_MAP_NORMALIZED_DIVS);
	res = isl_basic_map_alloc_space(space, n_div, bmap->n_eq, bmap->n_ineq);
	res = isl_basic_map_add_constraints_dim_map(res, bmap, dim_map);
	if (res)
		res->flags = flags;
	res = isl_basic_map_gauss(res, NULL);
	res = isl_basic_map_finalize(res);
	return res;
error:
	isl_dim_map_free(dim_map);
	isl_basic_map_free(bmap);
	isl_space_free(space);
	return NULL;
}

/* Reorder the dimensions of "map" according to given reordering.
 */
__isl_give isl_map *isl_map_realign(__isl_take isl_map *map,
	__isl_take isl_reordering *r)
{
	int i;
	struct isl_dim_map *dim_map;

	map = isl_map_cow(map);
	dim_map = isl_dim_map_from_reordering(r);
	if (!map || !r || !dim_map)
		goto error;

	for (i = 0; i < map->n; ++i) {
		struct isl_dim_map *dim_map_i;
		isl_space *space;

		dim_map_i = isl_dim_map_extend(dim_map, map->p[i]);

		space = isl_reordering_get_space(r);
		map->p[i] = isl_basic_map_realign(map->p[i], space, dim_map_i);

		if (!map->p[i])
			goto error;
	}

	map = isl_map_reset_space(map, isl_reordering_get_space(r));
	map = isl_map_unmark_normalized(map);

	isl_reordering_free(r);
	isl_dim_map_free(dim_map);
	return map;
error:
	isl_dim_map_free(dim_map);
	isl_map_free(map);
	isl_reordering_free(r);
	return NULL;
}

__isl_give isl_set *isl_set_realign(__isl_take isl_set *set,
	__isl_take isl_reordering *r)
{
	return set_from_map(isl_map_realign(set_to_map(set), r));
}

__isl_give isl_map *isl_map_align_params(__isl_take isl_map *map,
	__isl_take isl_space *model)
{
	isl_ctx *ctx;
	isl_bool aligned;

	if (!map || !model)
		goto error;

	ctx = isl_space_get_ctx(model);
	if (!isl_space_has_named_params(model))
		isl_die(ctx, isl_error_invalid,
			"model has unnamed parameters", goto error);
	if (isl_map_check_named_params(map) < 0)
		goto error;
	aligned = isl_map_space_has_equal_params(map, model);
	if (aligned < 0)
		goto error;
	if (!aligned) {
		isl_reordering *exp;

		exp = isl_parameter_alignment_reordering(map->dim, model);
		exp = isl_reordering_extend_space(exp, isl_map_get_space(map));
		map = isl_map_realign(map, exp);
	}

	isl_space_free(model);
	return map;
error:
	isl_space_free(model);
	isl_map_free(map);
	return NULL;
}

__isl_give isl_set *isl_set_align_params(__isl_take isl_set *set,
	__isl_take isl_space *model)
{
	return isl_map_align_params(set, model);
}

/* Align the parameters of "bmap" to those of "model", introducing
 * additional parameters if needed.
 */
__isl_give isl_basic_map *isl_basic_map_align_params(
	__isl_take isl_basic_map *bmap, __isl_take isl_space *model)
{
	isl_ctx *ctx;
	isl_bool equal_params;

	if (!bmap || !model)
		goto error;

	ctx = isl_space_get_ctx(model);
	if (!isl_space_has_named_params(model))
		isl_die(ctx, isl_error_invalid,
			"model has unnamed parameters", goto error);
	if (isl_basic_map_check_named_params(bmap) < 0)
		goto error;
	equal_params = isl_space_has_equal_params(bmap->dim, model);
	if (equal_params < 0)
		goto error;
	if (!equal_params) {
		isl_reordering *exp;
		struct isl_dim_map *dim_map;

		exp = isl_parameter_alignment_reordering(bmap->dim, model);
		exp = isl_reordering_extend_space(exp,
					isl_basic_map_get_space(bmap));
		dim_map = isl_dim_map_from_reordering(exp);
		bmap = isl_basic_map_realign(bmap,
				    isl_reordering_get_space(exp),
				    isl_dim_map_extend(dim_map, bmap));
		isl_reordering_free(exp);
		isl_dim_map_free(dim_map);
	}

	isl_space_free(model);
	return bmap;
error:
	isl_space_free(model);
	isl_basic_map_free(bmap);
	return NULL;
}

/* Do "bset" and "space" have the same parameters?
 */
isl_bool isl_basic_set_space_has_equal_params(__isl_keep isl_basic_set *bset,
	__isl_keep isl_space *space)
{
	isl_space *bset_space;

	bset_space = isl_basic_set_peek_space(bset);
	return isl_space_has_equal_params(bset_space, space);
}

/* Do "map" and "space" have the same parameters?
 */
isl_bool isl_map_space_has_equal_params(__isl_keep isl_map *map,
	__isl_keep isl_space *space)
{
	isl_space *map_space;

	map_space = isl_map_peek_space(map);
	return isl_space_has_equal_params(map_space, space);
}

/* Do "set" and "space" have the same parameters?
 */
isl_bool isl_set_space_has_equal_params(__isl_keep isl_set *set,
	__isl_keep isl_space *space)
{
	return isl_map_space_has_equal_params(set_to_map(set), space);
}

/* Align the parameters of "bset" to those of "model", introducing
 * additional parameters if needed.
 */
__isl_give isl_basic_set *isl_basic_set_align_params(
	__isl_take isl_basic_set *bset, __isl_take isl_space *model)
{
	return isl_basic_map_align_params(bset, model);
}

/* Drop all parameters not referenced by "map".
 */
__isl_give isl_map *isl_map_drop_unused_params(__isl_take isl_map *map)
{
	int i;
	isl_size n;

	n = isl_map_dim(map, isl_dim_param);
	if (isl_map_check_named_params(map) < 0 || n < 0)
		return isl_map_free(map);

	for (i = n - 1; i >= 0; i--) {
		isl_bool involves;

		involves = isl_map_involves_dims(map, isl_dim_param, i, 1);
		if (involves < 0)
			return isl_map_free(map);
		if (!involves)
			map = isl_map_project_out(map, isl_dim_param, i, 1);
	}

	return map;
}

/* Drop all parameters not referenced by "set".
 */
__isl_give isl_set *isl_set_drop_unused_params(
	__isl_take isl_set *set)
{
	return set_from_map(isl_map_drop_unused_params(set_to_map(set)));
}

/* Drop all parameters not referenced by "bmap".
 */
__isl_give isl_basic_map *isl_basic_map_drop_unused_params(
	__isl_take isl_basic_map *bmap)
{
	isl_size nparam;
	int i;

	nparam = isl_basic_map_dim(bmap, isl_dim_param);
	if (nparam < 0 || isl_basic_map_check_named_params(bmap) < 0)
		return isl_basic_map_free(bmap);

	for (i = nparam - 1; i >= 0; i--) {
		isl_bool involves;

		involves = isl_basic_map_involves_dims(bmap,
							isl_dim_param, i, 1);
		if (involves < 0)
			return isl_basic_map_free(bmap);
		if (!involves)
			bmap = isl_basic_map_drop(bmap, isl_dim_param, i, 1);
	}

	return bmap;
}

/* Drop all parameters not referenced by "bset".
 */
__isl_give isl_basic_set *isl_basic_set_drop_unused_params(
	__isl_take isl_basic_set *bset)
{
	return bset_from_bmap(isl_basic_map_drop_unused_params(
							bset_to_bmap(bset)));
}

/* Given a tuple of identifiers "tuple" in a space that corresponds
 * to that of "set", if any of those identifiers appear as parameters
 * in "set", then equate those parameters with the corresponding
 * set dimensions and project out the parameters.
 * The result therefore has no such parameters.
 */
static __isl_give isl_set *equate_params(__isl_take isl_set *set,
	__isl_keep isl_multi_id *tuple)
{
	int i;
	isl_size n;
	isl_space *set_space, *tuple_space;

	set_space = isl_set_peek_space(set);
	tuple_space = isl_multi_id_peek_space(tuple);
	if (isl_space_check_equal_tuples(tuple_space, set_space) < 0)
		return isl_set_free(set);
	n = isl_multi_id_size(tuple);
	if (n < 0)
		return isl_set_free(set);
	for (i = 0; i < n; ++i) {
		isl_id *id;
		int pos;

		id = isl_multi_id_get_at(tuple, i);
		if (!id)
			return isl_set_free(set);
		pos = isl_set_find_dim_by_id(set, isl_dim_param, id);
		isl_id_free(id);
		if (pos < 0)
			continue;
		set = isl_set_equate(set, isl_dim_param, pos, isl_dim_set, i);
		set = isl_set_project_out(set, isl_dim_param, pos, 1);
	}
	return set;
}

/* Bind the set dimensions of "set" to parameters with identifiers
 * specified by "tuple", living in the same space as "set".
 *
 * If no parameters with these identifiers appear in "set" already,
 * then the set dimensions are simply reinterpreted as parameters.
 * Otherwise, the parameters are first equated to the corresponding
 * set dimensions.
 */
__isl_give isl_set *isl_set_bind(__isl_take isl_set *set,
	__isl_take isl_multi_id *tuple)
{
	isl_space *space;

	set = equate_params(set, tuple);
	space = isl_set_get_space(set);
	space = isl_space_bind_set(space, tuple);
	isl_multi_id_free(tuple);
	set = isl_set_reset_space(set, space);

	return set;
}

/* Given a tuple of identifiers "tuple" in a space that corresponds
 * to the domain of "map", if any of those identifiers appear as parameters
 * in "map", then equate those parameters with the corresponding
 * input dimensions and project out the parameters.
 * The result therefore has no such parameters.
 */
static __isl_give isl_map *map_equate_params(__isl_take isl_map *map,
	__isl_keep isl_multi_id *tuple)
{
	int i;
	isl_size n;
	isl_space *map_space, *tuple_space;

	map_space = isl_map_peek_space(map);
	tuple_space = isl_multi_id_peek_space(tuple);
	if (isl_space_check_domain_tuples(tuple_space, map_space) < 0)
		return isl_map_free(map);
	n = isl_multi_id_size(tuple);
	if (n < 0)
		return isl_map_free(map);
	for (i = 0; i < n; ++i) {
		isl_id *id;
		int pos;

		id = isl_multi_id_get_at(tuple, i);
		if (!id)
			return isl_map_free(map);
		pos = isl_map_find_dim_by_id(map, isl_dim_param, id);
		isl_id_free(id);
		if (pos < 0)
			continue;
		map = isl_map_equate(map, isl_dim_param, pos, isl_dim_in, i);
		map = isl_map_project_out(map, isl_dim_param, pos, 1);
	}
	return map;
}

/* Bind the input dimensions of "map" to parameters with identifiers
 * specified by "tuple", living in the domain space of "map".
 *
 * If no parameters with these identifiers appear in "map" already,
 * then the input dimensions are simply reinterpreted as parameters.
 * Otherwise, the parameters are first equated to the corresponding
 * input dimensions.
 */
__isl_give isl_set *isl_map_bind_domain(__isl_take isl_map *map,
	__isl_take isl_multi_id *tuple)
{
	isl_space *space;
	isl_set *set;

	map = map_equate_params(map, tuple);
	space = isl_map_get_space(map);
	space = isl_space_bind_map_domain(space, tuple);
	isl_multi_id_free(tuple);
	set = set_from_map(isl_map_reset_space(map, space));

	return set;
}

/* Bind the output dimensions of "map" to parameters with identifiers
 * specified by "tuple", living in the range space of "map".
 *
 * Since binding is more easily implemented on the domain,
 * bind the input dimensions of the inverse of "map".
 */
__isl_give isl_set *isl_map_bind_range(__isl_take isl_map *map,
	__isl_take isl_multi_id *tuple)
{
	return isl_map_bind_domain(isl_map_reverse(map), tuple);
}

/* Insert a domain corresponding to "tuple"
 * into the nullary or unary relation "set".
 * The result has an extra initial tuple and is therefore
 * either a unary or binary relation.
 * Any parameters with identifiers in "tuple" are reinterpreted
 * as the corresponding domain dimensions.
 */
static __isl_give isl_map *unbind_params_insert_domain(
	__isl_take isl_set *set, __isl_take isl_multi_id *tuple)
{
	isl_space *space;
	isl_reordering *r;

	space = isl_set_peek_space(set);
	r = isl_reordering_unbind_params_insert_domain(space, tuple);
	isl_multi_id_free(tuple);

	return isl_map_realign(set_to_map(set), r);
}

/* Construct a set with "tuple" as domain from the parameter domain "set".
 * Any parameters with identifiers in "tuple" are reinterpreted
 * as the corresponding set dimensions.
 */
__isl_give isl_set *isl_set_unbind_params(__isl_take isl_set *set,
	__isl_take isl_multi_id *tuple)
{
	isl_bool is_params;

	is_params = isl_set_is_params(set);
	if (is_params < 0)
		set = isl_set_free(set);
	else if (!is_params)
		isl_die(isl_set_get_ctx(set), isl_error_invalid,
			"expecting parameter domain", set = isl_set_free(set));
	return set_from_map(unbind_params_insert_domain(set, tuple));
}

/* Check that "set" is a proper set, i.e., that it is not a parameter domain.
 */
static isl_stat isl_set_check_is_set(__isl_keep isl_set *set)
{
	isl_bool is_params;

	is_params = isl_set_is_params(set);
	if (is_params < 0)
		return isl_stat_error;
	else if (is_params)
		isl_die(isl_set_get_ctx(set), isl_error_invalid,
			"expecting proper set", return isl_stat_error);

	return isl_stat_ok;
}

/* Construct a map with "domain" as domain and "set" as range.
 * Any parameters with identifiers in "domain" are reinterpreted
 * as the corresponding domain dimensions.
 */
__isl_give isl_map *isl_set_unbind_params_insert_domain(
	__isl_take isl_set *set, __isl_take isl_multi_id *domain)
{
	if (isl_set_check_is_set(set) < 0)
		set = isl_set_free(set);
	return unbind_params_insert_domain(set, domain);
}

/* Construct a map with "domain" as domain and "set" as range.
 */
__isl_give isl_map *isl_set_insert_domain(__isl_take isl_set *set,
	__isl_take isl_space *domain)
{
	isl_size dim;
	isl_space *space;
	isl_map *map;

	if (isl_set_check_is_set(set) < 0 || isl_space_check_is_set(domain) < 0)
		domain = isl_space_free(domain);
	dim = isl_space_dim(domain, isl_dim_set);
	if (dim < 0)
		domain = isl_space_free(domain);
	space = isl_set_get_space(set);
	domain = isl_space_replace_params(domain, space);
	space = isl_space_map_from_domain_and_range(domain, space);

	map = isl_map_from_range(set);
	map = isl_map_add_dims(map, isl_dim_in, dim);
	map = isl_map_reset_space(map, space);

	return map;
}

__isl_give isl_mat *isl_basic_map_equalities_matrix(
		__isl_keep isl_basic_map *bmap, enum isl_dim_type c1,
		enum isl_dim_type c2, enum isl_dim_type c3,
		enum isl_dim_type c4, enum isl_dim_type c5)
{
	enum isl_dim_type c[5] = { c1, c2, c3, c4, c5 };
	struct isl_mat *mat;
	int i, j, k;
	int pos;
	isl_size total;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return NULL;
	mat = isl_mat_alloc(bmap->ctx, bmap->n_eq, total + 1);
	if (!mat)
		return NULL;
	for (i = 0; i < bmap->n_eq; ++i)
		for (j = 0, pos = 0; j < 5; ++j) {
			int off = isl_basic_map_offset(bmap, c[j]);
			isl_size dim = isl_basic_map_dim(bmap, c[j]);
			if (dim < 0)
				return isl_mat_free(mat);
			for (k = 0; k < dim; ++k) {
				isl_int_set(mat->row[i][pos],
					    bmap->eq[i][off + k]);
				++pos;
			}
		}

	return mat;
}

__isl_give isl_mat *isl_basic_map_inequalities_matrix(
		__isl_keep isl_basic_map *bmap, enum isl_dim_type c1,
		enum isl_dim_type c2, enum isl_dim_type c3,
		enum isl_dim_type c4, enum isl_dim_type c5)
{
	enum isl_dim_type c[5] = { c1, c2, c3, c4, c5 };
	struct isl_mat *mat;
	int i, j, k;
	int pos;
	isl_size total;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return NULL;
	mat = isl_mat_alloc(bmap->ctx, bmap->n_ineq, total + 1);
	if (!mat)
		return NULL;
	for (i = 0; i < bmap->n_ineq; ++i)
		for (j = 0, pos = 0; j < 5; ++j) {
			int off = isl_basic_map_offset(bmap, c[j]);
			isl_size dim = isl_basic_map_dim(bmap, c[j]);
			if (dim < 0)
				return isl_mat_free(mat);
			for (k = 0; k < dim; ++k) {
				isl_int_set(mat->row[i][pos],
					    bmap->ineq[i][off + k]);
				++pos;
			}
		}

	return mat;
}

__isl_give isl_basic_map *isl_basic_map_from_constraint_matrices(
	__isl_take isl_space *space,
	__isl_take isl_mat *eq, __isl_take isl_mat *ineq, enum isl_dim_type c1,
	enum isl_dim_type c2, enum isl_dim_type c3,
	enum isl_dim_type c4, enum isl_dim_type c5)
{
	enum isl_dim_type c[5] = { c1, c2, c3, c4, c5 };
	isl_basic_map *bmap = NULL;
	isl_size dim;
	unsigned total;
	unsigned extra;
	int i, j, k, l;
	int pos;

	dim = isl_space_dim(space, isl_dim_all);
	if (dim < 0 || !eq || !ineq)
		goto error;

	if (eq->n_col != ineq->n_col)
		isl_die(space->ctx, isl_error_invalid,
			"equalities and inequalities matrices should have "
			"same number of columns", goto error);

	total = 1 + dim;

	if (eq->n_col < total)
		isl_die(space->ctx, isl_error_invalid,
			"number of columns too small", goto error);

	extra = eq->n_col - total;

	bmap = isl_basic_map_alloc_space(isl_space_copy(space), extra,
				       eq->n_row, ineq->n_row);
	if (!bmap)
		goto error;
	for (i = 0; i < extra; ++i) {
		k = isl_basic_map_alloc_div(bmap);
		if (k < 0)
			goto error;
		isl_int_set_si(bmap->div[k][0], 0);
	}
	for (i = 0; i < eq->n_row; ++i) {
		l = isl_basic_map_alloc_equality(bmap);
		if (l < 0)
			goto error;
		for (j = 0, pos = 0; j < 5; ++j) {
			int off = isl_basic_map_offset(bmap, c[j]);
			isl_size dim = isl_basic_map_dim(bmap, c[j]);
			if (dim < 0)
				goto error;
			for (k = 0; k < dim; ++k) {
				isl_int_set(bmap->eq[l][off + k], 
					    eq->row[i][pos]);
				++pos;
			}
		}
	}
	for (i = 0; i < ineq->n_row; ++i) {
		l = isl_basic_map_alloc_inequality(bmap);
		if (l < 0)
			goto error;
		for (j = 0, pos = 0; j < 5; ++j) {
			int off = isl_basic_map_offset(bmap, c[j]);
			isl_size dim = isl_basic_map_dim(bmap, c[j]);
			if (dim < 0)
				goto error;
			for (k = 0; k < dim; ++k) {
				isl_int_set(bmap->ineq[l][off + k], 
					    ineq->row[i][pos]);
				++pos;
			}
		}
	}

	isl_space_free(space);
	isl_mat_free(eq);
	isl_mat_free(ineq);

	bmap = isl_basic_map_simplify(bmap);
	return isl_basic_map_finalize(bmap);
error:
	isl_space_free(space);
	isl_mat_free(eq);
	isl_mat_free(ineq);
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_mat *isl_basic_set_equalities_matrix(
	__isl_keep isl_basic_set *bset, enum isl_dim_type c1,
	enum isl_dim_type c2, enum isl_dim_type c3, enum isl_dim_type c4)
{
	return isl_basic_map_equalities_matrix(bset_to_bmap(bset),
						c1, c2, c3, c4, isl_dim_in);
}

__isl_give isl_mat *isl_basic_set_inequalities_matrix(
	__isl_keep isl_basic_set *bset, enum isl_dim_type c1,
	enum isl_dim_type c2, enum isl_dim_type c3, enum isl_dim_type c4)
{
	return isl_basic_map_inequalities_matrix(bset_to_bmap(bset),
						 c1, c2, c3, c4, isl_dim_in);
}

__isl_give isl_basic_set *isl_basic_set_from_constraint_matrices(
	__isl_take isl_space *space,
	__isl_take isl_mat *eq, __isl_take isl_mat *ineq, enum isl_dim_type c1,
	enum isl_dim_type c2, enum isl_dim_type c3, enum isl_dim_type c4)
{
	isl_basic_map *bmap;
	bmap = isl_basic_map_from_constraint_matrices(space, eq, ineq,
						   c1, c2, c3, c4, isl_dim_in);
	return bset_from_bmap(bmap);
}

isl_bool isl_basic_map_can_zip(__isl_keep isl_basic_map *bmap)
{
	if (!bmap)
		return isl_bool_error;
	
	return isl_space_can_zip(bmap->dim);
}

isl_bool isl_map_can_zip(__isl_keep isl_map *map)
{
	if (!map)
		return isl_bool_error;
	
	return isl_space_can_zip(map->dim);
}

/* Given a basic map (A -> B) -> (C -> D), return the corresponding basic map
 * (A -> C) -> (B -> D).
 */
__isl_give isl_basic_map *isl_basic_map_zip(__isl_take isl_basic_map *bmap)
{
	unsigned pos;
	isl_size n_in;
	isl_size n1;
	isl_size n2;

	if (!bmap)
		return NULL;

	if (!isl_basic_map_can_zip(bmap))
		isl_die(bmap->ctx, isl_error_invalid,
			"basic map cannot be zipped", goto error);
	n_in = isl_space_dim(bmap->dim->nested[0], isl_dim_in);
	n1 = isl_space_dim(bmap->dim->nested[0], isl_dim_out);
	n2 = isl_space_dim(bmap->dim->nested[1], isl_dim_in);
	if (n_in < 0 || n1 < 0 || n2 < 0)
		return isl_basic_map_free(bmap);
	pos = isl_basic_map_offset(bmap, isl_dim_in) + n_in;
	bmap = isl_basic_map_cow(bmap);
	bmap = isl_basic_map_swap_vars(bmap, pos, n1, n2);
	if (!bmap)
		return NULL;
	bmap->dim = isl_space_zip(bmap->dim);
	if (!bmap->dim)
		goto error;
	bmap = isl_basic_map_mark_final(bmap);
	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

/* Given a map (A -> B) -> (C -> D), return the corresponding map
 * (A -> C) -> (B -> D).
 */
__isl_give isl_map *isl_map_zip(__isl_take isl_map *map)
{
	if (!map)
		return NULL;

	if (!isl_map_can_zip(map))
		isl_die(map->ctx, isl_error_invalid, "map cannot be zipped",
			goto error);

	return isl_map_transform(map, &isl_space_zip, &isl_basic_map_zip);
error:
	isl_map_free(map);
	return NULL;
}

/* Can we apply isl_basic_map_curry to "bmap"?
 * That is, does it have a nested relation in its domain?
 */
isl_bool isl_basic_map_can_curry(__isl_keep isl_basic_map *bmap)
{
	if (!bmap)
		return isl_bool_error;

	return isl_space_can_curry(bmap->dim);
}

/* Can we apply isl_map_curry to "map"?
 * That is, does it have a nested relation in its domain?
 */
isl_bool isl_map_can_curry(__isl_keep isl_map *map)
{
	if (!map)
		return isl_bool_error;

	return isl_space_can_curry(map->dim);
}

/* Given a basic map (A -> B) -> C, return the corresponding basic map
 * A -> (B -> C).
 */
__isl_give isl_basic_map *isl_basic_map_curry(__isl_take isl_basic_map *bmap)
{

	if (!bmap)
		return NULL;

	if (!isl_basic_map_can_curry(bmap))
		isl_die(bmap->ctx, isl_error_invalid,
			"basic map cannot be curried", goto error);
	bmap = isl_basic_map_cow(bmap);
	if (!bmap)
		return NULL;
	bmap->dim = isl_space_curry(bmap->dim);
	if (!bmap->dim)
		goto error;
	bmap = isl_basic_map_mark_final(bmap);
	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

/* Given a map (A -> B) -> C, return the corresponding map
 * A -> (B -> C).
 */
__isl_give isl_map *isl_map_curry(__isl_take isl_map *map)
{
	return isl_map_change_space(map, &isl_map_can_curry,
				    "map cannot be curried", &isl_space_curry);
}

/* Can isl_map_range_curry be applied to "map"?
 * That is, does it have a nested relation in its range,
 * the domain of which is itself a nested relation?
 */
isl_bool isl_map_can_range_curry(__isl_keep isl_map *map)
{
	if (!map)
		return isl_bool_error;

	return isl_space_can_range_curry(map->dim);
}

/* Given a map A -> ((B -> C) -> D), return the corresponding map
 * A -> (B -> (C -> D)).
 */
__isl_give isl_map *isl_map_range_curry(__isl_take isl_map *map)
{
	return isl_map_change_space(map, &isl_map_can_range_curry,
				    "map range cannot be curried",
				    &isl_space_range_curry);
}

/* Can we apply isl_basic_map_uncurry to "bmap"?
 * That is, does it have a nested relation in its domain?
 */
isl_bool isl_basic_map_can_uncurry(__isl_keep isl_basic_map *bmap)
{
	if (!bmap)
		return isl_bool_error;

	return isl_space_can_uncurry(bmap->dim);
}

/* Can we apply isl_map_uncurry to "map"?
 * That is, does it have a nested relation in its domain?
 */
isl_bool isl_map_can_uncurry(__isl_keep isl_map *map)
{
	if (!map)
		return isl_bool_error;

	return isl_space_can_uncurry(map->dim);
}

/* Given a basic map A -> (B -> C), return the corresponding basic map
 * (A -> B) -> C.
 */
__isl_give isl_basic_map *isl_basic_map_uncurry(__isl_take isl_basic_map *bmap)
{

	if (!bmap)
		return NULL;

	if (!isl_basic_map_can_uncurry(bmap))
		isl_die(bmap->ctx, isl_error_invalid,
			"basic map cannot be uncurried",
			return isl_basic_map_free(bmap));
	bmap = isl_basic_map_cow(bmap);
	if (!bmap)
		return NULL;
	bmap->dim = isl_space_uncurry(bmap->dim);
	if (!bmap->dim)
		return isl_basic_map_free(bmap);
	bmap = isl_basic_map_mark_final(bmap);
	return bmap;
}

/* Given a map A -> (B -> C), return the corresponding map
 * (A -> B) -> C.
 */
__isl_give isl_map *isl_map_uncurry(__isl_take isl_map *map)
{
	return isl_map_change_space(map, &isl_map_can_uncurry,
				"map cannot be uncurried", &isl_space_uncurry);
}

__isl_give isl_set *isl_set_equate(__isl_take isl_set *set,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2)
{
	return isl_map_equate(set, type1, pos1, type2, pos2);
}

/* Construct a basic map where the given dimensions are equal to each other.
 */
static __isl_give isl_basic_map *equator(__isl_take isl_space *space,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2)
{
	isl_basic_map *bmap = NULL;
	int i;
	isl_size total;

	total = isl_space_dim(space, isl_dim_all);
	if (total < 0 ||
	    isl_space_check_range(space, type1, pos1, 1) < 0 ||
	    isl_space_check_range(space, type2, pos2, 1) < 0)
		goto error;

	if (type1 == type2 && pos1 == pos2)
		return isl_basic_map_universe(space);

	bmap = isl_basic_map_alloc_space(isl_space_copy(space), 0, 1, 0);
	i = isl_basic_map_alloc_equality(bmap);
	if (i < 0)
		goto error;
	isl_seq_clr(bmap->eq[i], 1 + total);
	pos1 += isl_basic_map_offset(bmap, type1);
	pos2 += isl_basic_map_offset(bmap, type2);
	isl_int_set_si(bmap->eq[i][pos1], -1);
	isl_int_set_si(bmap->eq[i][pos2], 1);
	bmap = isl_basic_map_finalize(bmap);
	isl_space_free(space);
	return bmap;
error:
	isl_space_free(space);
	isl_basic_map_free(bmap);
	return NULL;
}

/* Add a constraint imposing that the given two dimensions are equal.
 */
__isl_give isl_basic_map *isl_basic_map_equate(__isl_take isl_basic_map *bmap,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2)
{
	isl_basic_map *eq;

	eq = equator(isl_basic_map_get_space(bmap), type1, pos1, type2, pos2);

	bmap = isl_basic_map_intersect(bmap, eq);

	return bmap;
}

/* Add a constraint imposing that the given two dimensions are equal.
 */
__isl_give isl_map *isl_map_equate(__isl_take isl_map *map,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2)
{
	isl_basic_map *bmap;

	bmap = equator(isl_map_get_space(map), type1, pos1, type2, pos2);

	map = isl_map_intersect(map, isl_map_from_basic_map(bmap));

	return map;
}

/* Add a constraint imposing that the given two dimensions have opposite values.
 */
__isl_give isl_map *isl_map_oppose(__isl_take isl_map *map,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2)
{
	isl_basic_map *bmap = NULL;
	int i;
	isl_size total;

	if (isl_map_check_range(map, type1, pos1, 1) < 0)
		return isl_map_free(map);
	if (isl_map_check_range(map, type2, pos2, 1) < 0)
		return isl_map_free(map);

	total = isl_map_dim(map, isl_dim_all);
	if (total < 0)
		return isl_map_free(map);
	bmap = isl_basic_map_alloc_space(isl_map_get_space(map), 0, 1, 0);
	i = isl_basic_map_alloc_equality(bmap);
	if (i < 0)
		goto error;
	isl_seq_clr(bmap->eq[i], 1 + total);
	pos1 += isl_basic_map_offset(bmap, type1);
	pos2 += isl_basic_map_offset(bmap, type2);
	isl_int_set_si(bmap->eq[i][pos1], 1);
	isl_int_set_si(bmap->eq[i][pos2], 1);
	bmap = isl_basic_map_finalize(bmap);

	map = isl_map_intersect(map, isl_map_from_basic_map(bmap));

	return map;
error:
	isl_basic_map_free(bmap);
	isl_map_free(map);
	return NULL;
}

/* Construct a constraint imposing that the value of the first dimension is
 * greater than or equal to that of the second.
 */
static __isl_give isl_constraint *constraint_order_ge(
	__isl_take isl_space *space, enum isl_dim_type type1, int pos1,
	enum isl_dim_type type2, int pos2)
{
	isl_constraint *c;

	if (isl_space_check_range(space, type1, pos1, 1) < 0 ||
	    isl_space_check_range(space, type2, pos2, 1) < 0)
		space = isl_space_free(space);
	if (!space)
		return NULL;

	c = isl_constraint_alloc_inequality(isl_local_space_from_space(space));

	if (type1 == type2 && pos1 == pos2)
		return c;

	c = isl_constraint_set_coefficient_si(c, type1, pos1, 1);
	c = isl_constraint_set_coefficient_si(c, type2, pos2, -1);

	return c;
}

/* Add a constraint imposing that the value of the first dimension is
 * greater than or equal to that of the second.
 */
__isl_give isl_basic_map *isl_basic_map_order_ge(__isl_take isl_basic_map *bmap,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2)
{
	isl_constraint *c;
	isl_space *space;

	if (type1 == type2 && pos1 == pos2)
		return bmap;
	space = isl_basic_map_get_space(bmap);
	c = constraint_order_ge(space, type1, pos1, type2, pos2);
	bmap = isl_basic_map_add_constraint(bmap, c);

	return bmap;
}

/* Add a constraint imposing that the value of the first dimension is
 * greater than or equal to that of the second.
 */
__isl_give isl_map *isl_map_order_ge(__isl_take isl_map *map,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2)
{
	isl_constraint *c;
	isl_space *space;

	if (type1 == type2 && pos1 == pos2)
		return map;
	space = isl_map_get_space(map);
	c = constraint_order_ge(space, type1, pos1, type2, pos2);
	map = isl_map_add_constraint(map, c);

	return map;
}

/* Add a constraint imposing that the value of the first dimension is
 * less than or equal to that of the second.
 */
__isl_give isl_map *isl_map_order_le(__isl_take isl_map *map,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2)
{
	return isl_map_order_ge(map, type2, pos2, type1, pos1);
}

/* Construct a basic map where the value of the first dimension is
 * greater than that of the second.
 */
static __isl_give isl_basic_map *greator(__isl_take isl_space *space,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2)
{
	isl_basic_map *bmap = NULL;
	int i;
	isl_size total;

	if (isl_space_check_range(space, type1, pos1, 1) < 0 ||
	    isl_space_check_range(space, type2, pos2, 1) < 0)
		goto error;

	if (type1 == type2 && pos1 == pos2)
		return isl_basic_map_empty(space);

	bmap = isl_basic_map_alloc_space(space, 0, 0, 1);
	total = isl_basic_map_dim(bmap, isl_dim_all);
	i = isl_basic_map_alloc_inequality(bmap);
	if (total < 0 || i < 0)
		return isl_basic_map_free(bmap);
	isl_seq_clr(bmap->ineq[i], 1 + total);
	pos1 += isl_basic_map_offset(bmap, type1);
	pos2 += isl_basic_map_offset(bmap, type2);
	isl_int_set_si(bmap->ineq[i][pos1], 1);
	isl_int_set_si(bmap->ineq[i][pos2], -1);
	isl_int_set_si(bmap->ineq[i][0], -1);
	bmap = isl_basic_map_finalize(bmap);

	return bmap;
error:
	isl_space_free(space);
	isl_basic_map_free(bmap);
	return NULL;
}

/* Add a constraint imposing that the value of the first dimension is
 * greater than that of the second.
 */
__isl_give isl_basic_map *isl_basic_map_order_gt(__isl_take isl_basic_map *bmap,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2)
{
	isl_basic_map *gt;

	gt = greator(isl_basic_map_get_space(bmap), type1, pos1, type2, pos2);

	bmap = isl_basic_map_intersect(bmap, gt);

	return bmap;
}

/* Add a constraint imposing that the value of the first dimension is
 * greater than that of the second.
 */
__isl_give isl_map *isl_map_order_gt(__isl_take isl_map *map,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2)
{
	isl_basic_map *bmap;

	bmap = greator(isl_map_get_space(map), type1, pos1, type2, pos2);

	map = isl_map_intersect(map, isl_map_from_basic_map(bmap));

	return map;
}

/* Add a constraint imposing that the value of the first dimension is
 * smaller than that of the second.
 */
__isl_give isl_map *isl_map_order_lt(__isl_take isl_map *map,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2)
{
	return isl_map_order_gt(map, type2, pos2, type1, pos1);
}

__isl_give isl_aff *isl_basic_map_get_div(__isl_keep isl_basic_map *bmap,
	int pos)
{
	isl_aff *div;
	isl_local_space *ls;

	if (!bmap)
		return NULL;

	if (!isl_basic_map_divs_known(bmap))
		isl_die(isl_basic_map_get_ctx(bmap), isl_error_invalid,
			"some divs are unknown", return NULL);

	ls = isl_basic_map_get_local_space(bmap);
	div = isl_local_space_get_div(ls, pos);
	isl_local_space_free(ls);

	return div;
}

__isl_give isl_aff *isl_basic_set_get_div(__isl_keep isl_basic_set *bset,
	int pos)
{
	return isl_basic_map_get_div(bset, pos);
}

/* Plug in "subs" for set dimension "pos" of "set".
 */
__isl_give isl_set *isl_set_substitute(__isl_take isl_set *set,
	unsigned pos, __isl_keep isl_aff *subs)
{
	isl_multi_aff *ma;

	if (set && isl_set_plain_is_empty(set))
		return set;

	ma = isl_multi_aff_identity_on_domain_space(isl_set_get_space(set));
	ma = isl_multi_aff_set_aff(ma, pos, isl_aff_copy(subs));
	return isl_set_preimage_multi_aff(set, ma);
}

/* Check if the range of "ma" is compatible with the domain or range
 * (depending on "type") of "bmap".
 */
static isl_stat check_basic_map_compatible_range_multi_aff(
	__isl_keep isl_basic_map *bmap, enum isl_dim_type type,
	__isl_keep isl_multi_aff *ma)
{
	isl_bool m;
	isl_space *ma_space;

	ma_space = isl_multi_aff_get_space(ma);

	m = isl_space_has_equal_params(bmap->dim, ma_space);
	if (m < 0)
		goto error;
	if (!m)
		isl_die(isl_basic_map_get_ctx(bmap), isl_error_invalid,
			"parameters don't match", goto error);
	m = isl_space_tuple_is_equal(bmap->dim, type, ma_space, isl_dim_out);
	if (m < 0)
		goto error;
	if (!m)
		isl_die(isl_basic_map_get_ctx(bmap), isl_error_invalid,
			"spaces don't match", goto error);

	isl_space_free(ma_space);
	return isl_stat_ok;
error:
	isl_space_free(ma_space);
	return isl_stat_error;
}

/* Copy the divs from "ma" to "bmap", adding zeros for the "n_before"
 * coefficients before the transformed range of dimensions,
 * the "n_after" coefficients after the transformed range of dimensions
 * and the coefficients of the other divs in "bmap".
 */
static __isl_give isl_basic_map *set_ma_divs(__isl_take isl_basic_map *bmap,
	__isl_keep isl_multi_aff *ma, int n_before, int n_after, int n_div)
{
	int i;
	isl_size n_param;
	isl_size n_set;
	isl_local_space *ls;

	if (n_div == 0)
		return bmap;

	ls = isl_aff_get_domain_local_space(ma->u.p[0]);
	n_param = isl_local_space_dim(ls, isl_dim_param);
	n_set = isl_local_space_dim(ls, isl_dim_set);
	if (n_param < 0 || n_set < 0)
		return isl_basic_map_free(bmap);

	for (i = 0; i < n_div; ++i) {
		int o_bmap = 0, o_ls = 0;

		isl_seq_cpy(bmap->div[i], ls->div->row[i], 1 + 1 + n_param);
		o_bmap += 1 + 1 + n_param;
		o_ls += 1 + 1 + n_param;
		isl_seq_clr(bmap->div[i] + o_bmap, n_before);
		o_bmap += n_before;
		isl_seq_cpy(bmap->div[i] + o_bmap,
			    ls->div->row[i] + o_ls, n_set);
		o_bmap += n_set;
		o_ls += n_set;
		isl_seq_clr(bmap->div[i] + o_bmap, n_after);
		o_bmap += n_after;
		isl_seq_cpy(bmap->div[i] + o_bmap,
			    ls->div->row[i] + o_ls, n_div);
		o_bmap += n_div;
		o_ls += n_div;
		isl_seq_clr(bmap->div[i] + o_bmap, bmap->n_div - n_div);
		bmap = isl_basic_map_add_div_constraints(bmap, i);
		if (!bmap)
			goto error;
	}

	isl_local_space_free(ls);
	return bmap;
error:
	isl_local_space_free(ls);
	return isl_basic_map_free(bmap);
}

/* How many stride constraints does "ma" enforce?
 * That is, how many of the affine expressions have a denominator
 * different from one?
 */
static int multi_aff_strides(__isl_keep isl_multi_aff *ma)
{
	int i;
	int strides = 0;

	for (i = 0; i < ma->n; ++i)
		if (!isl_int_is_one(ma->u.p[i]->v->el[0]))
			strides++;

	return strides;
}

/* For each affine expression in ma of the form
 *
 *	x_i = (f_i y + h_i)/m_i
 *
 * with m_i different from one, add a constraint to "bmap"
 * of the form
 *
 *	f_i y + h_i = m_i alpha_i
 *
 * with alpha_i an additional existentially quantified variable.
 *
 * The input variables of "ma" correspond to a subset of the variables
 * of "bmap".  There are "n_before" variables in "bmap" before this
 * subset and "n_after" variables after this subset.
 * The integer divisions of the affine expressions in "ma" are assumed
 * to have been aligned.  There are "n_div_ma" of them and
 * they appear first in "bmap", straight after the "n_after" variables.
 */
static __isl_give isl_basic_map *add_ma_strides(
	__isl_take isl_basic_map *bmap, __isl_keep isl_multi_aff *ma,
	int n_before, int n_after, int n_div_ma)
{
	int i, k;
	int div;
	isl_size total;
	isl_size n_param;
	isl_size n_in;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	n_param = isl_multi_aff_dim(ma, isl_dim_param);
	n_in = isl_multi_aff_dim(ma, isl_dim_in);
	if (total < 0 || n_param < 0 || n_in < 0)
		return isl_basic_map_free(bmap);
	for (i = 0; i < ma->n; ++i) {
		int o_bmap = 0, o_ma = 1;

		if (isl_int_is_one(ma->u.p[i]->v->el[0]))
			continue;
		div = isl_basic_map_alloc_div(bmap);
		k = isl_basic_map_alloc_equality(bmap);
		if (div < 0 || k < 0)
			goto error;
		isl_int_set_si(bmap->div[div][0], 0);
		isl_seq_cpy(bmap->eq[k] + o_bmap,
			    ma->u.p[i]->v->el + o_ma, 1 + n_param);
		o_bmap += 1 + n_param;
		o_ma += 1 + n_param;
		isl_seq_clr(bmap->eq[k] + o_bmap, n_before);
		o_bmap += n_before;
		isl_seq_cpy(bmap->eq[k] + o_bmap,
			    ma->u.p[i]->v->el + o_ma, n_in);
		o_bmap += n_in;
		o_ma += n_in;
		isl_seq_clr(bmap->eq[k] + o_bmap, n_after);
		o_bmap += n_after;
		isl_seq_cpy(bmap->eq[k] + o_bmap,
			    ma->u.p[i]->v->el + o_ma, n_div_ma);
		o_bmap += n_div_ma;
		o_ma += n_div_ma;
		isl_seq_clr(bmap->eq[k] + o_bmap, 1 + total - o_bmap);
		isl_int_neg(bmap->eq[k][1 + total], ma->u.p[i]->v->el[0]);
		total++;
	}

	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

/* Replace the domain or range space (depending on "type) of "space" by "set".
 */
static __isl_give isl_space *isl_space_set(__isl_take isl_space *space,
	enum isl_dim_type type, __isl_take isl_space *set)
{
	if (type == isl_dim_in) {
		space = isl_space_range(space);
		space = isl_space_map_from_domain_and_range(set, space);
	} else {
		space = isl_space_domain(space);
		space = isl_space_map_from_domain_and_range(space, set);
	}

	return space;
}

/* Compute the preimage of the domain or range (depending on "type")
 * of "bmap" under the function represented by "ma".
 * In other words, plug in "ma" in the domain or range of "bmap".
 * The result is a basic map that lives in the same space as "bmap"
 * except that the domain or range has been replaced by
 * the domain space of "ma".
 *
 * If bmap is represented by
 *
 *	A(p) + S u + B x + T v + C(divs) >= 0,
 *
 * where u and x are input and output dimensions if type == isl_dim_out
 * while x and v are input and output dimensions if type == isl_dim_in,
 * and ma is represented by
 *
 *	x = D(p) + F(y) + G(divs')
 *
 * then the result is
 *
 *	A(p) + B D(p) + S u + B F(y) + T v + B G(divs') + C(divs) >= 0
 *
 * The divs in the input set are similarly adjusted.
 * In particular
 *
 *	floor((a_i(p) + s u + b_i x + t v + c_i(divs))/n_i)
 *
 * becomes
 *
 *	floor((a_i(p) + b_i D(p) + s u + b_i F(y) + t v +
 *		B_i G(divs') + c_i(divs))/n_i)
 *
 * If bmap is not a rational map and if F(y) involves any denominators
 *
 *	x_i = (f_i y + h_i)/m_i
 *
 * then additional constraints are added to ensure that we only
 * map back integer points.  That is we enforce
 *
 *	f_i y + h_i = m_i alpha_i
 *
 * with alpha_i an additional existentially quantified variable.
 *
 * We first copy over the divs from "ma".
 * Then we add the modified constraints and divs from "bmap".
 * Finally, we add the stride constraints, if needed.
 */
__isl_give isl_basic_map *isl_basic_map_preimage_multi_aff(
	__isl_take isl_basic_map *bmap, enum isl_dim_type type,
	__isl_take isl_multi_aff *ma)
{
	int i, k;
	isl_space *space;
	isl_basic_map *res = NULL;
	isl_size n_before, n_after, n_div_bmap, n_div_ma;
	isl_int f, c1, c2, g;
	isl_bool rational;
	int strides;

	isl_int_init(f);
	isl_int_init(c1);
	isl_int_init(c2);
	isl_int_init(g);

	ma = isl_multi_aff_align_divs(ma);
	if (!bmap || !ma)
		goto error;
	if (check_basic_map_compatible_range_multi_aff(bmap, type, ma) < 0)
		goto error;

	if (type == isl_dim_in) {
		n_before = 0;
		n_after = isl_basic_map_dim(bmap, isl_dim_out);
	} else {
		n_before = isl_basic_map_dim(bmap, isl_dim_in);
		n_after = 0;
	}
	n_div_bmap = isl_basic_map_dim(bmap, isl_dim_div);
	n_div_ma = ma->n ? isl_aff_dim(ma->u.p[0], isl_dim_div) : 0;
	if (n_before < 0 || n_after < 0 || n_div_bmap < 0 || n_div_ma < 0)
		goto error;

	space = isl_multi_aff_get_domain_space(ma);
	space = isl_space_set(isl_basic_map_get_space(bmap), type, space);
	rational = isl_basic_map_is_rational(bmap);
	strides = rational ? 0 : multi_aff_strides(ma);
	res = isl_basic_map_alloc_space(space, n_div_ma + n_div_bmap + strides,
			    bmap->n_eq + strides, bmap->n_ineq + 2 * n_div_ma);
	if (rational)
		res = isl_basic_map_set_rational(res);

	for (i = 0; i < n_div_ma + n_div_bmap; ++i)
		if (isl_basic_map_alloc_div(res) < 0)
			goto error;

	res = set_ma_divs(res, ma, n_before, n_after, n_div_ma);
	if (!res)
		goto error;

	for (i = 0; i < bmap->n_eq; ++i) {
		k = isl_basic_map_alloc_equality(res);
		if (k < 0)
			goto error;
		if (isl_seq_preimage(res->eq[k], bmap->eq[i], ma, n_before,
				    n_after, n_div_ma, n_div_bmap,
				    f, c1, c2, g, 0) < 0)
			goto error;
	}

	for (i = 0; i < bmap->n_ineq; ++i) {
		k = isl_basic_map_alloc_inequality(res);
		if (k < 0)
			goto error;
		if (isl_seq_preimage(res->ineq[k], bmap->ineq[i], ma, n_before,
				    n_after, n_div_ma, n_div_bmap,
				    f, c1, c2, g, 0) < 0)
			goto error;
	}

	for (i = 0; i < bmap->n_div; ++i) {
		if (isl_int_is_zero(bmap->div[i][0])) {
			isl_int_set_si(res->div[n_div_ma + i][0], 0);
			continue;
		}
		if (isl_seq_preimage(res->div[n_div_ma + i], bmap->div[i], ma,
				    n_before, n_after, n_div_ma, n_div_bmap,
				    f, c1, c2, g, 1) < 0)
			goto error;
	}

	if (strides)
		res = add_ma_strides(res, ma, n_before, n_after, n_div_ma);

	isl_int_clear(f);
	isl_int_clear(c1);
	isl_int_clear(c2);
	isl_int_clear(g);
	isl_basic_map_free(bmap);
	isl_multi_aff_free(ma);
	res = isl_basic_map_simplify(res);
	return isl_basic_map_finalize(res);
error:
	isl_int_clear(f);
	isl_int_clear(c1);
	isl_int_clear(c2);
	isl_int_clear(g);
	isl_basic_map_free(bmap);
	isl_multi_aff_free(ma);
	isl_basic_map_free(res);
	return NULL;
}

/* Compute the preimage of "bset" under the function represented by "ma".
 * In other words, plug in "ma" in "bset".  The result is a basic set
 * that lives in the domain space of "ma".
 */
__isl_give isl_basic_set *isl_basic_set_preimage_multi_aff(
	__isl_take isl_basic_set *bset, __isl_take isl_multi_aff *ma)
{
	return isl_basic_map_preimage_multi_aff(bset, isl_dim_set, ma);
}

/* Compute the preimage of the domain of "bmap" under the function
 * represented by "ma".
 * In other words, plug in "ma" in the domain of "bmap".
 * The result is a basic map that lives in the same space as "bmap"
 * except that the domain has been replaced by the domain space of "ma".
 */
__isl_give isl_basic_map *isl_basic_map_preimage_domain_multi_aff(
	__isl_take isl_basic_map *bmap, __isl_take isl_multi_aff *ma)
{
	return isl_basic_map_preimage_multi_aff(bmap, isl_dim_in, ma);
}

/* Compute the preimage of the range of "bmap" under the function
 * represented by "ma".
 * In other words, plug in "ma" in the range of "bmap".
 * The result is a basic map that lives in the same space as "bmap"
 * except that the range has been replaced by the domain space of "ma".
 */
__isl_give isl_basic_map *isl_basic_map_preimage_range_multi_aff(
	__isl_take isl_basic_map *bmap, __isl_take isl_multi_aff *ma)
{
	return isl_basic_map_preimage_multi_aff(bmap, isl_dim_out, ma);
}

/* Check if the range of "ma" is compatible with the domain or range
 * (depending on "type") of "map".
 * Return isl_stat_error if anything is wrong.
 */
static isl_stat check_map_compatible_range_multi_aff(
	__isl_keep isl_map *map, enum isl_dim_type type,
	__isl_keep isl_multi_aff *ma)
{
	isl_bool m;
	isl_space *ma_space;

	ma_space = isl_multi_aff_get_space(ma);
	m = isl_map_space_tuple_is_equal(map, type, ma_space, isl_dim_out);
	isl_space_free(ma_space);
	if (m < 0)
		return isl_stat_error;
	if (!m)
		isl_die(isl_map_get_ctx(map), isl_error_invalid,
			"spaces don't match", return isl_stat_error);
	return isl_stat_ok;
}

/* Compute the preimage of the domain or range (depending on "type")
 * of "map" under the function represented by "ma".
 * In other words, plug in "ma" in the domain or range of "map".
 * The result is a map that lives in the same space as "map"
 * except that the domain or range has been replaced by
 * the domain space of "ma".
 *
 * The parameters are assumed to have been aligned.
 */
static __isl_give isl_map *map_preimage_multi_aff(__isl_take isl_map *map,
	enum isl_dim_type type, __isl_take isl_multi_aff *ma)
{
	int i;
	isl_space *space;

	map = isl_map_cow(map);
	ma = isl_multi_aff_align_divs(ma);
	if (!map || !ma)
		goto error;
	if (check_map_compatible_range_multi_aff(map, type, ma) < 0)
		goto error;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_preimage_multi_aff(map->p[i], type,
							isl_multi_aff_copy(ma));
		if (!map->p[i])
			goto error;
	}

	space = isl_multi_aff_get_domain_space(ma);
	space = isl_space_set(isl_map_get_space(map), type, space);

	isl_space_free(isl_map_take_space(map));
	map = isl_map_restore_space(map, space);
	if (!map)
		goto error;

	isl_multi_aff_free(ma);
	if (map->n > 1)
		ISL_F_CLR(map, ISL_MAP_DISJOINT);
	ISL_F_CLR(map, ISL_SET_NORMALIZED);
	return map;
error:
	isl_multi_aff_free(ma);
	isl_map_free(map);
	return NULL;
}

/* Compute the preimage of the domain or range (depending on "type")
 * of "map" under the function represented by "ma".
 * In other words, plug in "ma" in the domain or range of "map".
 * The result is a map that lives in the same space as "map"
 * except that the domain or range has been replaced by
 * the domain space of "ma".
 */
__isl_give isl_map *isl_map_preimage_multi_aff(__isl_take isl_map *map,
	enum isl_dim_type type, __isl_take isl_multi_aff *ma)
{
	isl_bool aligned;

	if (!map || !ma)
		goto error;

	aligned = isl_map_space_has_equal_params(map, ma->space);
	if (aligned < 0)
		goto error;
	if (aligned)
		return map_preimage_multi_aff(map, type, ma);

	if (isl_map_check_named_params(map) < 0)
		goto error;
	if (!isl_space_has_named_params(ma->space))
		isl_die(map->ctx, isl_error_invalid,
			"unaligned unnamed parameters", goto error);
	map = isl_map_align_params(map, isl_multi_aff_get_space(ma));
	ma = isl_multi_aff_align_params(ma, isl_map_get_space(map));

	return map_preimage_multi_aff(map, type, ma);
error:
	isl_multi_aff_free(ma);
	return isl_map_free(map);
}

/* Compute the preimage of "set" under the function represented by "ma".
 * In other words, plug in "ma" in "set".  The result is a set
 * that lives in the domain space of "ma".
 */
__isl_give isl_set *isl_set_preimage_multi_aff(__isl_take isl_set *set,
	__isl_take isl_multi_aff *ma)
{
	return isl_map_preimage_multi_aff(set, isl_dim_set, ma);
}

/* Compute the preimage of the domain of "map" under the function
 * represented by "ma".
 * In other words, plug in "ma" in the domain of "map".
 * The result is a map that lives in the same space as "map"
 * except that the domain has been replaced by the domain space of "ma".
 */
__isl_give isl_map *isl_map_preimage_domain_multi_aff(__isl_take isl_map *map,
	__isl_take isl_multi_aff *ma)
{
	return isl_map_preimage_multi_aff(map, isl_dim_in, ma);
}

/* Compute the preimage of the range of "map" under the function
 * represented by "ma".
 * In other words, plug in "ma" in the range of "map".
 * The result is a map that lives in the same space as "map"
 * except that the range has been replaced by the domain space of "ma".
 */
__isl_give isl_map *isl_map_preimage_range_multi_aff(__isl_take isl_map *map,
	__isl_take isl_multi_aff *ma)
{
	return isl_map_preimage_multi_aff(map, isl_dim_out, ma);
}

/* Compute the preimage of "map" under the function represented by "pma".
 * In other words, plug in "pma" in the domain or range of "map".
 * The result is a map that lives in the same space as "map",
 * except that the space of type "type" has been replaced by
 * the domain space of "pma".
 *
 * The parameters of "map" and "pma" are assumed to have been aligned.
 */
static __isl_give isl_map *isl_map_preimage_pw_multi_aff_aligned(
	__isl_take isl_map *map, enum isl_dim_type type,
	__isl_take isl_pw_multi_aff *pma)
{
	int i;
	isl_map *res;

	if (!pma)
		goto error;

	if (pma->n == 0) {
		isl_pw_multi_aff_free(pma);
		res = isl_map_empty(isl_map_get_space(map));
		isl_map_free(map);
		return res;
	}

	res = isl_map_preimage_multi_aff(isl_map_copy(map), type,
					isl_multi_aff_copy(pma->p[0].maff));
	if (type == isl_dim_in)
		res = isl_map_intersect_domain(res,
						isl_map_copy(pma->p[0].set));
	else
		res = isl_map_intersect_range(res,
						isl_map_copy(pma->p[0].set));

	for (i = 1; i < pma->n; ++i) {
		isl_map *res_i;

		res_i = isl_map_preimage_multi_aff(isl_map_copy(map), type,
					isl_multi_aff_copy(pma->p[i].maff));
		if (type == isl_dim_in)
			res_i = isl_map_intersect_domain(res_i,
						isl_map_copy(pma->p[i].set));
		else
			res_i = isl_map_intersect_range(res_i,
						isl_map_copy(pma->p[i].set));
		res = isl_map_union(res, res_i);
	}

	isl_pw_multi_aff_free(pma);
	isl_map_free(map);
	return res;
error:
	isl_pw_multi_aff_free(pma);
	isl_map_free(map);
	return NULL;
}

/* Compute the preimage of "map" under the function represented by "pma".
 * In other words, plug in "pma" in the domain or range of "map".
 * The result is a map that lives in the same space as "map",
 * except that the space of type "type" has been replaced by
 * the domain space of "pma".
 */
__isl_give isl_map *isl_map_preimage_pw_multi_aff(__isl_take isl_map *map,
	enum isl_dim_type type, __isl_take isl_pw_multi_aff *pma)
{
	isl_bool aligned;

	if (!map || !pma)
		goto error;

	aligned = isl_map_space_has_equal_params(map, pma->dim);
	if (aligned < 0)
		goto error;
	if (aligned)
		return isl_map_preimage_pw_multi_aff_aligned(map, type, pma);

	if (isl_map_check_named_params(map) < 0)
		goto error;
	if (isl_pw_multi_aff_check_named_params(pma) < 0)
		goto error;
	map = isl_map_align_params(map, isl_pw_multi_aff_get_space(pma));
	pma = isl_pw_multi_aff_align_params(pma, isl_map_get_space(map));

	return isl_map_preimage_pw_multi_aff_aligned(map, type, pma);
error:
	isl_pw_multi_aff_free(pma);
	return isl_map_free(map);
}

/* Compute the preimage of "set" under the function represented by "pma".
 * In other words, plug in "pma" in "set".  The result is a set
 * that lives in the domain space of "pma".
 */
__isl_give isl_set *isl_set_preimage_pw_multi_aff(__isl_take isl_set *set,
	__isl_take isl_pw_multi_aff *pma)
{
	return isl_map_preimage_pw_multi_aff(set, isl_dim_set, pma);
}

/* Compute the preimage of the domain of "map" under the function
 * represented by "pma".
 * In other words, plug in "pma" in the domain of "map".
 * The result is a map that lives in the same space as "map",
 * except that domain space has been replaced by the domain space of "pma".
 */
__isl_give isl_map *isl_map_preimage_domain_pw_multi_aff(
	__isl_take isl_map *map, __isl_take isl_pw_multi_aff *pma)
{
	return isl_map_preimage_pw_multi_aff(map, isl_dim_in, pma);
}

/* Compute the preimage of the range of "map" under the function
 * represented by "pma".
 * In other words, plug in "pma" in the range of "map".
 * The result is a map that lives in the same space as "map",
 * except that range space has been replaced by the domain space of "pma".
 */
__isl_give isl_map *isl_map_preimage_range_pw_multi_aff(
	__isl_take isl_map *map, __isl_take isl_pw_multi_aff *pma)
{
	return isl_map_preimage_pw_multi_aff(map, isl_dim_out, pma);
}

/* Compute the preimage of "map" under the function represented by "mpa".
 * In other words, plug in "mpa" in the domain or range of "map".
 * The result is a map that lives in the same space as "map",
 * except that the space of type "type" has been replaced by
 * the domain space of "mpa".
 *
 * If the map does not involve any constraints that refer to the
 * dimensions of the substituted space, then the only possible
 * effect of "mpa" on the map is to map the space to a different space.
 * We create a separate isl_multi_aff to effectuate this change
 * in order to avoid spurious splitting of the map along the pieces
 * of "mpa".
 * If "mpa" has a non-trivial explicit domain, however,
 * then the full substitution should be performed.
 */
__isl_give isl_map *isl_map_preimage_multi_pw_aff(__isl_take isl_map *map,
	enum isl_dim_type type, __isl_take isl_multi_pw_aff *mpa)
{
	isl_size n;
	isl_bool full;
	isl_pw_multi_aff *pma;

	n = isl_map_dim(map, type);
	if (n < 0 || !mpa)
		goto error;

	full = isl_map_involves_dims(map, type, 0, n);
	if (full >= 0 && !full)
		full = isl_multi_pw_aff_has_non_trivial_domain(mpa);
	if (full < 0)
		goto error;
	if (!full) {
		isl_space *space;
		isl_multi_aff *ma;

		space = isl_multi_pw_aff_get_space(mpa);
		isl_multi_pw_aff_free(mpa);
		ma = isl_multi_aff_zero(space);
		return isl_map_preimage_multi_aff(map, type, ma);
	}

	pma = isl_pw_multi_aff_from_multi_pw_aff(mpa);
	return isl_map_preimage_pw_multi_aff(map, type, pma);
error:
	isl_map_free(map);
	isl_multi_pw_aff_free(mpa);
	return NULL;
}

/* Compute the preimage of "map" under the function represented by "mpa".
 * In other words, plug in "mpa" in the domain "map".
 * The result is a map that lives in the same space as "map",
 * except that domain space has been replaced by the domain space of "mpa".
 */
__isl_give isl_map *isl_map_preimage_domain_multi_pw_aff(
	__isl_take isl_map *map, __isl_take isl_multi_pw_aff *mpa)
{
	return isl_map_preimage_multi_pw_aff(map, isl_dim_in, mpa);
}

/* Compute the preimage of "set" by the function represented by "mpa".
 * In other words, plug in "mpa" in "set".
 */
__isl_give isl_set *isl_set_preimage_multi_pw_aff(__isl_take isl_set *set,
	__isl_take isl_multi_pw_aff *mpa)
{
	return isl_map_preimage_multi_pw_aff(set, isl_dim_set, mpa);
}

/* Return a copy of the equality constraints of "bset" as a matrix.
 */
__isl_give isl_mat *isl_basic_set_extract_equalities(
	__isl_keep isl_basic_set *bset)
{
	isl_ctx *ctx;
	isl_size total;

	total = isl_basic_set_dim(bset, isl_dim_all);
	if (total < 0)
		return NULL;

	ctx = isl_basic_set_get_ctx(bset);
	return isl_mat_sub_alloc6(ctx, bset->eq, 0, bset->n_eq, 0, 1 + total);
}

/* Are the "n" "coefficients" starting at "first" of the integer division
 * expressions at position "pos1" in "bmap1" and "pos2" in "bmap2" equal
 * to each other?
 * The "coefficient" at position 0 is the denominator.
 * The "coefficient" at position 1 is the constant term.
 */
isl_bool isl_basic_map_equal_div_expr_part(__isl_keep isl_basic_map *bmap1,
	int pos1, __isl_keep isl_basic_map *bmap2, int pos2,
	unsigned first, unsigned n)
{
	if (isl_basic_map_check_range(bmap1, isl_dim_div, pos1, 1) < 0)
		return isl_bool_error;
	if (isl_basic_map_check_range(bmap2, isl_dim_div, pos2, 1) < 0)
		return isl_bool_error;
	return isl_seq_eq(bmap1->div[pos1] + first,
			  bmap2->div[pos2] + first, n);
}

/* Are the integer division expressions at position "pos1" in "bmap1" and
 * "pos2" in "bmap2" equal to each other, except that the constant terms
 * are different?
 */
isl_bool isl_basic_map_equal_div_expr_except_constant(
	__isl_keep isl_basic_map *bmap1, int pos1,
	__isl_keep isl_basic_map *bmap2, int pos2)
{
	isl_bool equal;
	isl_size total, total2;

	total = isl_basic_map_dim(bmap1, isl_dim_all);
	total2 = isl_basic_map_dim(bmap2, isl_dim_all);
	if (total < 0 || total2 < 0)
		return isl_bool_error;
	if (total != total2)
		isl_die(isl_basic_map_get_ctx(bmap1), isl_error_invalid,
			"incomparable div expressions", return isl_bool_error);
	equal = isl_basic_map_equal_div_expr_part(bmap1, pos1, bmap2, pos2,
						0, 1);
	if (equal < 0 || !equal)
		return equal;
	equal = isl_basic_map_equal_div_expr_part(bmap1, pos1, bmap2, pos2,
						1, 1);
	if (equal < 0 || equal)
		return isl_bool_not(equal);
	return isl_basic_map_equal_div_expr_part(bmap1, pos1, bmap2, pos2,
						2, total);
}

/* Replace the numerator of the constant term of the integer division
 * expression at position "div" in "bmap" by "value".
 * The caller guarantees that this does not change the meaning
 * of the input.
 */
__isl_give isl_basic_map *isl_basic_map_set_div_expr_constant_num_si_inplace(
	__isl_take isl_basic_map *bmap, int div, int value)
{
	if (isl_basic_map_check_range(bmap, isl_dim_div, div, 1) < 0)
		return isl_basic_map_free(bmap);

	isl_int_set_si(bmap->div[div][1], value);

	return bmap;
}

/* Is the point "inner" internal to inequality constraint "ineq"
 * of "bset"?
 * The point is considered to be internal to the inequality constraint,
 * if it strictly lies on the positive side of the inequality constraint,
 * or if it lies on the constraint and the constraint is lexico-positive.
 */
static isl_bool is_internal(__isl_keep isl_vec *inner,
	__isl_keep isl_basic_set *bset, int ineq)
{
	isl_ctx *ctx;
	int pos;
	isl_size total;

	if (!inner || !bset)
		return isl_bool_error;

	ctx = isl_basic_set_get_ctx(bset);
	isl_seq_inner_product(inner->el, bset->ineq[ineq], inner->size,
				&ctx->normalize_gcd);
	if (!isl_int_is_zero(ctx->normalize_gcd))
		return isl_int_is_nonneg(ctx->normalize_gcd);

	total = isl_basic_set_dim(bset, isl_dim_all);
	if (total < 0)
		return isl_bool_error;
	pos = isl_seq_first_non_zero(bset->ineq[ineq] + 1, total);
	return isl_int_is_pos(bset->ineq[ineq][1 + pos]);
}

/* Tighten the inequality constraints of "bset" that are outward with respect
 * to the point "vec".
 * That is, tighten the constraints that are not satisfied by "vec".
 *
 * "vec" is a point internal to some superset S of "bset" that is used
 * to make the subsets of S disjoint, by tightening one half of the constraints
 * that separate two subsets.  In particular, the constraints of S
 * are all satisfied by "vec" and should not be tightened.
 * Of the internal constraints, those that have "vec" on the outside
 * are tightened.  The shared facet is included in the adjacent subset
 * with the opposite constraint.
 * For constraints that saturate "vec", this criterion cannot be used
 * to determine which of the two sides should be tightened.
 * Instead, the sign of the first non-zero coefficient is used
 * to make this choice.  Note that this second criterion is never used
 * on the constraints of S since "vec" is interior to "S".
 */
__isl_give isl_basic_set *isl_basic_set_tighten_outward(
	__isl_take isl_basic_set *bset, __isl_keep isl_vec *vec)
{
	int j;

	bset = isl_basic_set_cow(bset);
	if (!bset)
		return NULL;
	for (j = 0; j < bset->n_ineq; ++j) {
		isl_bool internal;

		internal = is_internal(vec, bset, j);
		if (internal < 0)
			return isl_basic_set_free(bset);
		if (internal)
			continue;
		isl_int_sub_ui(bset->ineq[j][0], bset->ineq[j][0], 1);
	}

	return bset;
}

/* Replace the variables x of type "type" starting at "first" in "bmap"
 * by x' with x = M x' with M the matrix trans.
 * That is, replace the corresponding coefficients c by c M.
 *
 * The transformation matrix should be a square matrix.
 */
__isl_give isl_basic_map *isl_basic_map_transform_dims(
	__isl_take isl_basic_map *bmap, enum isl_dim_type type, unsigned first,
	__isl_take isl_mat *trans)
{
	unsigned pos;

	bmap = isl_basic_map_cow(bmap);
	if (!bmap || !trans)
		goto error;

	if (trans->n_row != trans->n_col)
		isl_die(trans->ctx, isl_error_invalid,
			"expecting square transformation matrix", goto error);
	if (isl_basic_map_check_range(bmap, type, first, trans->n_row) < 0)
		goto error;

	pos = isl_basic_map_offset(bmap, type) + first;

	if (isl_mat_sub_transform(bmap->eq, bmap->n_eq, pos,
			isl_mat_copy(trans)) < 0)
		goto error;
	if (isl_mat_sub_transform(bmap->ineq, bmap->n_ineq, pos,
		      isl_mat_copy(trans)) < 0)
		goto error;
	if (isl_mat_sub_transform(bmap->div, bmap->n_div, 1 + pos,
		      isl_mat_copy(trans)) < 0)
		goto error;

	ISL_F_CLR(bmap, ISL_BASIC_MAP_SORTED);
	ISL_F_CLR(bmap, ISL_BASIC_MAP_NORMALIZED_DIVS);

	isl_mat_free(trans);
	return bmap;
error:
	isl_mat_free(trans);
	isl_basic_map_free(bmap);
	return NULL;
}

/* Replace the variables x of type "type" starting at "first" in "bset"
 * by x' with x = M x' with M the matrix trans.
 * That is, replace the corresponding coefficients c by c M.
 *
 * The transformation matrix should be a square matrix.
 */
__isl_give isl_basic_set *isl_basic_set_transform_dims(
	__isl_take isl_basic_set *bset, enum isl_dim_type type, unsigned first,
	__isl_take isl_mat *trans)
{
	return isl_basic_map_transform_dims(bset, type, first, trans);
}
