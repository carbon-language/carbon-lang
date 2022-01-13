/*
 * Copyright 2010-2011 INRIA Saclay
 * Copyright 2013-2014 Ecole Normale Superieure
 * Copyright 2014      INRIA Rocquencourt
 * Copyright 2016-2017 Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France 
 * and Inria Paris - Rocquencourt, Domaine de Voluceau - Rocquencourt,
 * B.P. 105 - 78153 Le Chesnay, France
 */

#include <isl_map_private.h>
#include <isl_union_map_private.h>
#include <isl/ctx.h>
#include <isl/hash.h>
#include <isl_aff_private.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl_space_private.h>
#include <isl/union_set.h>
#include <isl_maybe_map.h>
#include <isl_id_private.h>

#include <bset_from_bmap.c>
#include <set_to_map.c>
#include <set_from_map.c>
#include <uset_to_umap.c>
#include <uset_from_umap.c>
#include <set_list_from_map_list_inl.c>

#undef TYPE
#define TYPE	isl_union_map
static
#include "has_single_reference_templ.c"
static
#include "check_single_reference_templ.c"

/* Return the number of parameters of "umap", where "type"
 * is required to be set to isl_dim_param.
 */
isl_size isl_union_map_dim(__isl_keep isl_union_map *umap,
	enum isl_dim_type type)
{
	if (!umap)
		return isl_size_error;

	if (type != isl_dim_param)
		isl_die(isl_union_map_get_ctx(umap), isl_error_invalid,
			"can only reference parameters", return isl_size_error);

	return isl_space_dim(umap->dim, type);
}

/* Return the number of parameters of "uset", where "type"
 * is required to be set to isl_dim_param.
 */
isl_size isl_union_set_dim(__isl_keep isl_union_set *uset,
	enum isl_dim_type type)
{
	return isl_union_map_dim(uset, type);
}

/* Return the id of the specified dimension.
 */
__isl_give isl_id *isl_union_map_get_dim_id(__isl_keep isl_union_map *umap,
	enum isl_dim_type type, unsigned pos)
{
	if (!umap)
		return NULL;

	if (type != isl_dim_param)
		isl_die(isl_union_map_get_ctx(umap), isl_error_invalid,
			"can only reference parameters", return NULL);

	return isl_space_get_dim_id(umap->dim, type, pos);
}

/* Is this union set a parameter domain?
 */
isl_bool isl_union_set_is_params(__isl_keep isl_union_set *uset)
{
	isl_set *set;
	isl_bool params;

	if (!uset)
		return isl_bool_error;
	if (uset->table.n != 1)
		return isl_bool_false;

	set = isl_set_from_union_set(isl_union_set_copy(uset));
	params = isl_set_is_params(set);
	isl_set_free(set);
	return params;
}

/* Is this union map actually a parameter domain?
 * Users should never call this function.  Outside of isl,
 * a union map can never be a parameter domain.
 */
isl_bool isl_union_map_is_params(__isl_keep isl_union_map *umap)
{
	return isl_union_set_is_params(uset_from_umap(umap));
}

static __isl_give isl_union_map *isl_union_map_alloc(
	__isl_take isl_space *space, int size)
{
	isl_union_map *umap;

	space = isl_space_params(space);
	if (!space)
		return NULL;

	umap = isl_calloc_type(space->ctx, isl_union_map);
	if (!umap) {
		isl_space_free(space);
		return NULL;
	}

	umap->ref = 1;
	umap->dim = space;
	if (isl_hash_table_init(space->ctx, &umap->table, size) < 0)
		return isl_union_map_free(umap);

	return umap;
}

/* Create an empty union map without specifying any parameters.
 */
__isl_give isl_union_map *isl_union_map_empty_ctx(isl_ctx *ctx)
{
	return isl_union_map_empty_space(isl_space_unit(ctx));
}

__isl_give isl_union_map *isl_union_map_empty_space(__isl_take isl_space *space)
{
	return isl_union_map_alloc(space, 16);
}

/* This is an alternative name for the function above.
 */
__isl_give isl_union_map *isl_union_map_empty(__isl_take isl_space *space)
{
	return isl_union_map_empty_space(space);
}

/* Create an empty union set without specifying any parameters.
 */
__isl_give isl_union_set *isl_union_set_empty_ctx(isl_ctx *ctx)
{
	return uset_from_umap(isl_union_map_empty_ctx(ctx));
}

__isl_give isl_union_set *isl_union_set_empty_space(__isl_take isl_space *space)
{
	return uset_from_umap(isl_union_map_empty_space(space));
}

/* This is an alternative name for the function above.
 */
__isl_give isl_union_set *isl_union_set_empty(__isl_take isl_space *space)
{
	return isl_union_set_empty_space(space);
}

isl_ctx *isl_union_map_get_ctx(__isl_keep isl_union_map *umap)
{
	return umap ? umap->dim->ctx : NULL;
}

isl_ctx *isl_union_set_get_ctx(__isl_keep isl_union_set *uset)
{
	return uset ? uset->dim->ctx : NULL;
}

/* Return the space of "umap".
 */
__isl_keep isl_space *isl_union_map_peek_space(__isl_keep isl_union_map *umap)
{
	return umap ? umap->dim : NULL;
}

/* Return the space of "uset".
 */
__isl_keep isl_space *isl_union_set_peek_space(__isl_keep isl_union_set *uset)
{
	return isl_union_map_peek_space(uset_to_umap(uset));
}

__isl_give isl_space *isl_union_map_get_space(__isl_keep isl_union_map *umap)
{
	return isl_space_copy(isl_union_map_peek_space(umap));
}

/* Return the position of the parameter with the given name
 * in "umap".
 * Return -1 if no such dimension can be found.
 */
int isl_union_map_find_dim_by_name(__isl_keep isl_union_map *umap,
	enum isl_dim_type type, const char *name)
{
	if (!umap)
		return -1;
	return isl_space_find_dim_by_name(umap->dim, type, name);
}

__isl_give isl_space *isl_union_set_get_space(__isl_keep isl_union_set *uset)
{
	return isl_union_map_get_space(uset);
}

static isl_stat free_umap_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_map_free(map);
	return isl_stat_ok;
}

static isl_stat add_map(__isl_take isl_map *map, void *user)
{
	isl_union_map **umap = (isl_union_map **)user;

	*umap = isl_union_map_add_map(*umap, map);

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_dup(__isl_keep isl_union_map *umap)
{
	isl_union_map *dup;

	if (!umap)
		return NULL;

	dup = isl_union_map_empty(isl_space_copy(umap->dim));
	if (isl_union_map_foreach_map(umap, &add_map, &dup) < 0)
		goto error;
	return dup;
error:
	isl_union_map_free(dup);
	return NULL;
}

__isl_give isl_union_map *isl_union_map_cow(__isl_take isl_union_map *umap)
{
	if (!umap)
		return NULL;

	if (umap->ref == 1)
		return umap;
	umap->ref--;
	return isl_union_map_dup(umap);
}

struct isl_union_align {
	isl_reordering *exp;
	isl_union_map *res;
};

static isl_stat align_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_reordering *exp;
	struct isl_union_align *data = user;

	exp = isl_reordering_extend_space(isl_reordering_copy(data->exp),
				    isl_map_get_space(map));

	data->res = isl_union_map_add_map(data->res,
					isl_map_realign(isl_map_copy(map), exp));

	return isl_stat_ok;
}

/* Align the parameters of umap along those of model.
 * The result has the parameters of model first, in the same order
 * as they appear in model, followed by any remaining parameters of
 * umap that do not appear in model.
 */
__isl_give isl_union_map *isl_union_map_align_params(
	__isl_take isl_union_map *umap, __isl_take isl_space *model)
{
	struct isl_union_align data = { NULL, NULL };
	isl_bool equal_params;

	if (!umap || !model)
		goto error;

	equal_params = isl_space_has_equal_params(umap->dim, model);
	if (equal_params < 0)
		goto error;
	if (equal_params) {
		isl_space_free(model);
		return umap;
	}

	data.exp = isl_parameter_alignment_reordering(umap->dim, model);
	if (!data.exp)
		goto error;

	data.res = isl_union_map_alloc(isl_reordering_get_space(data.exp),
					umap->table.n);
	if (isl_hash_table_foreach(umap->dim->ctx, &umap->table,
					&align_entry, &data) < 0)
		goto error;

	isl_reordering_free(data.exp);
	isl_union_map_free(umap);
	isl_space_free(model);
	return data.res;
error:
	isl_reordering_free(data.exp);
	isl_union_map_free(umap);
	isl_union_map_free(data.res);
	isl_space_free(model);
	return NULL;
}

__isl_give isl_union_set *isl_union_set_align_params(
	__isl_take isl_union_set *uset, __isl_take isl_space *model)
{
	return isl_union_map_align_params(uset, model);
}

__isl_give isl_union_map *isl_union_map_union(__isl_take isl_union_map *umap1,
	__isl_take isl_union_map *umap2)
{
	umap1 = isl_union_map_align_params(umap1, isl_union_map_get_space(umap2));
	umap2 = isl_union_map_align_params(umap2, isl_union_map_get_space(umap1));

	umap1 = isl_union_map_cow(umap1);

	if (!umap1 || !umap2)
		goto error;

	if (isl_union_map_foreach_map(umap2, &add_map, &umap1) < 0)
		goto error;

	isl_union_map_free(umap2);

	return umap1;
error:
	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	return NULL;
}

__isl_give isl_union_set *isl_union_set_union(__isl_take isl_union_set *uset1,
	__isl_take isl_union_set *uset2)
{
	return isl_union_map_union(uset1, uset2);
}

__isl_give isl_union_map *isl_union_map_copy(__isl_keep isl_union_map *umap)
{
	if (!umap)
		return NULL;

	umap->ref++;
	return umap;
}

__isl_give isl_union_set *isl_union_set_copy(__isl_keep isl_union_set *uset)
{
	return isl_union_map_copy(uset);
}

__isl_null isl_union_map *isl_union_map_free(__isl_take isl_union_map *umap)
{
	if (!umap)
		return NULL;

	if (--umap->ref > 0)
		return NULL;

	isl_hash_table_foreach(umap->dim->ctx, &umap->table,
			       &free_umap_entry, NULL);
	isl_hash_table_clear(&umap->table);
	isl_space_free(umap->dim);
	free(umap);
	return NULL;
}

__isl_null isl_union_set *isl_union_set_free(__isl_take isl_union_set *uset)
{
	return isl_union_map_free(uset);
}

/* Do "umap" and "space" have the same parameters?
 */
isl_bool isl_union_map_space_has_equal_params(__isl_keep isl_union_map *umap,
	__isl_keep isl_space *space)
{
	isl_space *umap_space;

	umap_space = isl_union_map_peek_space(umap);
	return isl_space_has_equal_params(umap_space, space);
}

/* Do "uset" and "space" have the same parameters?
 */
isl_bool isl_union_set_space_has_equal_params(__isl_keep isl_union_set *uset,
	__isl_keep isl_space *space)
{
	return isl_union_map_space_has_equal_params(uset_to_umap(uset), space);
}

/* Is the space of the map at "entry" equal to "space", ignoring parameters?
 */
static isl_bool has_space_tuples(const void *entry, const void *val)
{
	isl_map *map = (isl_map *)entry;
	isl_space *space = (isl_space *) val;

	return isl_map_has_space_tuples(map, space);
}

/* Find the entry in "umap" with space "space" (ignoring parameters),
 * returning isl_hash_table_entry_none if no such entry appears in "umap" and
 * NULL on error.
 * If "reserve" is set, then an entry is created if it does
 * not exist already.  Since this modifies the hash table in-place,
 * this means "umap" must have a single reference when "reserve" is set.
 */
static struct isl_hash_table_entry *isl_union_map_find_entry(
	__isl_keep isl_union_map *umap, __isl_keep isl_space *space,
	int reserve)
{
	uint32_t hash;

	if (!umap || !space)
		return NULL;
	if (reserve && isl_union_map_check_single_reference(umap) < 0)
		return NULL;

	hash = isl_space_get_tuple_hash(space);
	return isl_hash_table_find(isl_union_map_get_ctx(umap), &umap->table,
				    hash, &has_space_tuples, space, reserve);
}

/* Find the entry in "uset" with space "space" (ignoring parameters),
 * returning isl_hash_table_entry_none if no such entry appears in "uset" and
 * NULL on error.
 * If "reserve" is set, then an entry is created if it does
 * not exist already.  In this case, a NULL return indicates an error.
 */
struct isl_hash_table_entry *isl_union_set_find_entry(
	__isl_keep isl_union_set *uset, __isl_keep isl_space *space,
	int reserve)
{
	return isl_union_map_find_entry(uset_to_umap(uset), space, reserve);
}

__isl_give isl_union_map *isl_union_map_add_map(__isl_take isl_union_map *umap,
	__isl_take isl_map *map)
{
	struct isl_hash_table_entry *entry;
	isl_bool aligned;
	isl_space *space;

	if (!map || !umap)
		goto error;

	if (isl_map_plain_is_empty(map)) {
		isl_map_free(map);
		return umap;
	}

	aligned = isl_map_space_has_equal_params(map, umap->dim);
	if (aligned < 0)
		goto error;
	if (!aligned) {
		umap = isl_union_map_align_params(umap, isl_map_get_space(map));
		map = isl_map_align_params(map, isl_union_map_get_space(umap));
	}

	umap = isl_union_map_cow(umap);

	space = isl_map_peek_space(map);
	entry = isl_union_map_find_entry(umap, space, 1);
	if (!entry)
		goto error;

	if (!entry->data)
		entry->data = map;
	else {
		entry->data = isl_map_union(entry->data, isl_map_copy(map));
		if (!entry->data)
			goto error;
		isl_map_free(map);
	}

	return umap;
error:
	isl_map_free(map);
	isl_union_map_free(umap);
	return NULL;
}

__isl_give isl_union_set *isl_union_set_add_set(__isl_take isl_union_set *uset,
	__isl_take isl_set *set)
{
	return isl_union_map_add_map(uset, set_to_map(set));
}

__isl_give isl_union_map *isl_union_map_from_map(__isl_take isl_map *map)
{
	isl_space *space;
	isl_union_map *umap;

	if (!map)
		return NULL;

	space = isl_map_get_space(map);
	space = isl_space_params(space);
	umap = isl_union_map_empty(space);
	umap = isl_union_map_add_map(umap, map);

	return umap;
}

/* This function performs the same operation as isl_union_map_from_map,
 * but is considered as a function on an isl_map when exported.
 */
__isl_give isl_union_map *isl_map_to_union_map(__isl_take isl_map *map)
{
	return isl_union_map_from_map(map);
}

__isl_give isl_union_set *isl_union_set_from_set(__isl_take isl_set *set)
{
	return isl_union_map_from_map(set_to_map(set));
}

/* This function performs the same operation as isl_union_set_from_set,
 * but is considered as a function on an isl_set when exported.
 */
__isl_give isl_union_set *isl_set_to_union_set(__isl_take isl_set *set)
{
	return isl_union_set_from_set(set);
}

__isl_give isl_union_map *isl_union_map_from_basic_map(
	__isl_take isl_basic_map *bmap)
{
	return isl_union_map_from_map(isl_map_from_basic_map(bmap));
}

__isl_give isl_union_set *isl_union_set_from_basic_set(
	__isl_take isl_basic_set *bset)
{
	return isl_union_map_from_basic_map(bset);
}

struct isl_union_map_foreach_data
{
	isl_stat (*fn)(__isl_take isl_map *map, void *user);
	void *user;
};

static isl_stat call_on_copy(void **entry, void *user)
{
	isl_map *map = *entry;
	struct isl_union_map_foreach_data *data;
	data = (struct isl_union_map_foreach_data *)user;

	return data->fn(isl_map_copy(map), data->user);
}

isl_size isl_union_map_n_map(__isl_keep isl_union_map *umap)
{
	return umap ? umap->table.n : isl_size_error;
}

isl_size isl_union_set_n_set(__isl_keep isl_union_set *uset)
{
	return uset ? uset->table.n : isl_size_error;
}

isl_stat isl_union_map_foreach_map(__isl_keep isl_union_map *umap,
	isl_stat (*fn)(__isl_take isl_map *map, void *user), void *user)
{
	struct isl_union_map_foreach_data data = { fn, user };

	if (!umap)
		return isl_stat_error;

	return isl_hash_table_foreach(umap->dim->ctx, &umap->table,
				      &call_on_copy, &data);
}

/* Internal data structure for isl_union_map_every_map.
 *
 * "test" is the user-specified callback function.
 * "user" is the user-specified callback function argument.
 *
 * "failed" is initialized to 0 and set to 1 if "test" fails
 * on any map.
 */
struct isl_union_map_every_data {
	isl_bool (*test)(__isl_keep isl_map *map, void *user);
	void *user;
	int failed;
};

/* Call data->test on "map".
 * If this fails, then set data->failed and abort.
 */
static isl_stat call_every(void **entry, void *user)
{
	isl_map *map = *entry;
	struct isl_union_map_every_data *data = user;
	isl_bool r;

	r = data->test(map, data->user);
	if (r < 0)
		return isl_stat_error;
	if (r)
		return isl_stat_ok;
	data->failed = 1;
	return isl_stat_error;
}

/* Does "test" succeed on every map in "umap"?
 */
isl_bool isl_union_map_every_map(__isl_keep isl_union_map *umap,
	isl_bool (*test)(__isl_keep isl_map *map, void *user), void *user)
{
	struct isl_union_map_every_data data = { test, user, 0 };
	isl_stat r;

	if (!umap)
		return isl_bool_error;

	r = isl_hash_table_foreach(isl_union_map_get_ctx(umap), &umap->table,
				      &call_every, &data);
	if (r >= 0)
		return isl_bool_true;
	if (data.failed)
		return isl_bool_false;
	return isl_bool_error;
}

/* Add "map" to "list".
 */
static isl_stat add_list_map(__isl_take isl_map *map, void *user)
{
	isl_map_list **list = user;

	*list = isl_map_list_add(*list, map);

	if (!*list)
		return isl_stat_error;
	return isl_stat_ok;
}

/* Return the maps in "umap" as a list.
 *
 * First construct a list of the appropriate size and then add all the
 * elements.
 */
__isl_give isl_map_list *isl_union_map_get_map_list(
	__isl_keep isl_union_map *umap)
{
	isl_size n_maps;
	isl_ctx *ctx;
	isl_map_list *list;

	n_maps = isl_union_map_n_map(umap);
	if (n_maps < 0)
		return NULL;
	ctx = isl_union_map_get_ctx(umap);
	list = isl_map_list_alloc(ctx, n_maps);

	if (isl_union_map_foreach_map(umap, &add_list_map, &list) < 0)
		list = isl_map_list_free(list);

	return list;
}

/* Return the sets in "uset" as a list.
 */
__isl_give isl_set_list *isl_union_set_get_set_list(
	__isl_keep isl_union_set *uset)
{
	return set_list_from_map_list(
		isl_union_map_get_map_list(uset_to_umap(uset)));
}

/* Can "umap" be converted to an isl_map?
 * That is, does it contain elements in exactly one space?
 */
isl_bool isl_union_map_isa_map(__isl_keep isl_union_map *umap)
{
	isl_size n;

	n = isl_union_map_n_map(umap);
	if (n < 0)
		return isl_bool_error;
	return isl_bool_ok(n == 1);
}

/* Can "uset" be converted to an isl_set?
 * That is, does it contain elements in exactly one space?
 */
isl_bool isl_union_set_isa_set(__isl_keep isl_union_set *uset)
{
	return isl_union_map_isa_map(uset_to_umap(uset));
}

static isl_stat copy_map(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_map **map_p = user;

	*map_p = isl_map_copy(map);

	return isl_stat_error;
}

__isl_give isl_map *isl_map_from_union_map(__isl_take isl_union_map *umap)
{
	isl_bool is_map;
	isl_ctx *ctx;
	isl_map *map = NULL;

	is_map = isl_union_map_isa_map(umap);
	if (is_map < 0)
		goto error;
	ctx = isl_union_map_get_ctx(umap);
	if (!is_map)
		isl_die(ctx, isl_error_invalid,
			"union map needs to contain elements in exactly "
			"one space", goto error);

	isl_hash_table_foreach(ctx, &umap->table, &copy_map, &map);

	isl_union_map_free(umap);

	return map;
error:
	isl_union_map_free(umap);
	return NULL;
}

/* This function performs the same operation as isl_map_from_union_map,
 * but is considered as a function on an isl_union_map when exported.
 */
__isl_give isl_map *isl_union_map_as_map(__isl_take isl_union_map *umap)
{
	return isl_map_from_union_map(umap);
}

__isl_give isl_set *isl_set_from_union_set(__isl_take isl_union_set *uset)
{
	return isl_map_from_union_map(uset);
}

/* This function performs the same operation as isl_set_from_union_set,
 * but is considered as a function on an isl_union_set when exported.
 */
__isl_give isl_set *isl_union_set_as_set(__isl_take isl_union_set *uset)
{
	return isl_set_from_union_set(uset);
}

/* Extract the map in "umap" that lives in the given space (ignoring
 * parameters).
 */
__isl_give isl_map *isl_union_map_extract_map(__isl_keep isl_union_map *umap,
	__isl_take isl_space *space)
{
	struct isl_hash_table_entry *entry;

	entry = isl_union_map_find_entry(umap, space, 0);
	if (!entry)
		goto error;
	if (entry == isl_hash_table_entry_none)
		return isl_map_empty(space);
	isl_space_free(space);
	return isl_map_copy(entry->data);
error:
	isl_space_free(space);
	return NULL;
}

__isl_give isl_set *isl_union_set_extract_set(__isl_keep isl_union_set *uset,
	__isl_take isl_space *space)
{
	return set_from_map(isl_union_map_extract_map(uset, space));
}

/* Check if umap contains a map in the given space (ignoring parameters).
 */
isl_bool isl_union_map_contains(__isl_keep isl_union_map *umap,
	__isl_keep isl_space *space)
{
	struct isl_hash_table_entry *entry;

	space = isl_space_drop_all_params(isl_space_copy(space));
	space = isl_space_align_params(space, isl_union_map_get_space(umap));
	entry = isl_union_map_find_entry(umap, space, 0);
	isl_space_free(space);
	if (!entry)
		return isl_bool_error;
	return isl_bool_ok(entry != isl_hash_table_entry_none);
}

isl_bool isl_union_set_contains(__isl_keep isl_union_set *uset,
	__isl_keep isl_space *space)
{
	return isl_union_map_contains(uset, space);
}

isl_stat isl_union_set_foreach_set(__isl_keep isl_union_set *uset,
	isl_stat (*fn)(__isl_take isl_set *set, void *user), void *user)
{
	return isl_union_map_foreach_map(uset,
		(isl_stat(*)(__isl_take isl_map *, void*))fn, user);
}

/* Internal data structure for isl_union_set_every_set.
 *
 * "test" is the user-specified callback function.
 * "user" is the user-specified callback function argument.
 */
struct isl_test_set_from_map_data {
	isl_bool (*test)(__isl_keep isl_set *set, void *user);
	void *user;
};

/* Call data->test on "map", which is part of an isl_union_set and
 * therefore known to be an isl_set.
 */
static isl_bool test_set_from_map(__isl_keep isl_map *map, void *user)
{
	struct isl_test_set_from_map_data *data = user;

	return data->test(set_from_map(map), data->user);
}

/* Does "test" succeed on every set in "uset"?
 */
isl_bool isl_union_set_every_set(__isl_keep isl_union_set *uset,
	isl_bool (*test)(__isl_keep isl_set *set, void *user), void *user)
{
	struct isl_test_set_from_map_data data = { test, user };

	return isl_union_map_every_map(uset_to_umap(uset),
					&test_set_from_map, &data);
}

struct isl_union_set_foreach_point_data {
	isl_stat (*fn)(__isl_take isl_point *pnt, void *user);
	void *user;
};

static isl_stat foreach_point(__isl_take isl_set *set, void *user)
{
	struct isl_union_set_foreach_point_data *data = user;
	isl_stat r;

	r = isl_set_foreach_point(set, data->fn, data->user);
	isl_set_free(set);

	return r;
}

isl_stat isl_union_set_foreach_point(__isl_keep isl_union_set *uset,
	isl_stat (*fn)(__isl_take isl_point *pnt, void *user), void *user)
{
	struct isl_union_set_foreach_point_data data = { fn, user };
	return isl_union_set_foreach_set(uset, &foreach_point, &data);
}

/* Data structure that specifies how gen_bin_op should
 * construct results from the inputs.
 *
 * If "subtract" is set, then a map in the first input is copied to the result
 * if there is no corresponding map in the second input.
 * Otherwise, a map in the first input with no corresponding map
 * in the second input is ignored.
 * If "filter" is not NULL, then it specifies which maps in the first
 * input may have a matching map in the second input.
 * In particular, it makes sure that "match_space" can be called
 * on the space of the map.
 * "match_space" specifies how to transform the space of a map
 * in the first input to the space of the corresponding map
 * in the second input.
 * "fn_map" specifies how the matching maps, one from each input,
 * should be combined to form a map in the result.
 */
struct isl_bin_op_control {
	int subtract;
	isl_bool (*filter)(__isl_keep isl_map *map);
	__isl_give isl_space *(*match_space)(__isl_take isl_space *space);
	__isl_give isl_map *(*fn_map)(__isl_take isl_map *map1,
		__isl_take isl_map *map2);
};

/* Internal data structure for gen_bin_op.
 * "control" specifies how the maps in the result should be constructed.
 * "umap2" is a pointer to the second argument.
 * "res" collects the results.
 */
struct isl_union_map_gen_bin_data {
	struct isl_bin_op_control *control;
	isl_union_map *umap2;
	isl_union_map *res;
};

/* Add a copy of "map" to "res" and return the result.
 */
static __isl_give isl_union_map *bin_add_map(__isl_take isl_union_map *res,
	__isl_keep isl_map *map)
{
	return isl_union_map_add_map(res, isl_map_copy(map));
}

/* Combine "map1" and "map2", add the result to "res" and return the result.
 * Check whether the result is empty before adding it to "res".
 */
static __isl_give isl_union_map *bin_add_pair(__isl_take isl_union_map *res,
	__isl_keep isl_map *map1, __isl_keep isl_map *map2,
	struct isl_union_map_gen_bin_data *data)
{
	isl_bool empty;
	isl_map *map;

	map = data->control->fn_map(isl_map_copy(map1), isl_map_copy(map2));
	empty = isl_map_is_empty(map);
	if (empty < 0 || empty) {
		isl_map_free(map);
		if (empty < 0)
			return isl_union_map_free(res);
		return res;
	}
	return isl_union_map_add_map(res, map);
}

/* Dummy match_space function that simply returns the input space.
 */
static __isl_give isl_space *identity(__isl_take isl_space *space)
{
	return space;
}

/* Look for the map in data->umap2 that corresponds to "map", if any.
 * Return (isl_bool_true, matching map) if there is one,
 * (isl_bool_false, NULL) if there is no matching map and
 * (isl_bool_error, NULL) on error.
 *
 * If not NULL, then data->control->filter specifies whether "map"
 * can have any matching map.  If so,
 * data->control->match_space specifies which map in data->umap2
 * corresponds to "map".
 */
static __isl_keep isl_maybe_isl_map bin_try_get_match(
	struct isl_union_map_gen_bin_data *data, __isl_keep isl_map *map)
{
	struct isl_hash_table_entry *entry2;
	isl_space *space;
	isl_maybe_isl_map res = { isl_bool_error, NULL };

	if (data->control->filter) {
		res.valid = data->control->filter(map);
		if (res.valid < 0 || !res.valid)
			return res;
		res.valid = isl_bool_error;
	}

	space = isl_map_get_space(map);
	if (data->control->match_space != &identity)
		space = data->control->match_space(space);
	entry2 = isl_union_map_find_entry(data->umap2, space, 0);
	isl_space_free(space);
	if (entry2)
		res.valid = isl_bool_ok(entry2 != isl_hash_table_entry_none);
	if (res.valid >= 0 && res.valid)
		res.value = entry2->data;

	return res;
}

/* isl_hash_table_foreach callback for gen_bin_op.
 * Look for the map in data->umap2 that corresponds
 * to the map that "entry" points to, apply the binary operation and
 * add the result to data->res.
 *
 * If no corresponding map can be found, then the effect depends
 * on data->control->subtract.  If it is set, then the current map
 * is added directly to the result.  Otherwise, it is ignored.
 */
static isl_stat gen_bin_entry(void **entry, void *user)
{
	struct isl_union_map_gen_bin_data *data = user;
	isl_map *map = *entry;
	isl_maybe_isl_map m;

	m = bin_try_get_match(data, map);
	if (m.valid < 0)
		return isl_stat_error;
	if (!m.valid && !data->control->subtract)
		return isl_stat_ok;

	if (!m.valid)
		data->res = bin_add_map(data->res, map);
	else
		data->res = bin_add_pair(data->res, map, m.value, data);
	if (!data->res)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Apply a binary operation to "umap1" and "umap2" based on "control".
 * Run over all maps in "umap1" and look for the corresponding map in "umap2"
 * in gen_bin_entry.
 */
static __isl_give isl_union_map *gen_bin_op(__isl_take isl_union_map *umap1,
	__isl_take isl_union_map *umap2, struct isl_bin_op_control *control)
{
	struct isl_union_map_gen_bin_data data = { control, NULL, NULL };

	umap1 = isl_union_map_align_params(umap1, isl_union_map_get_space(umap2));
	umap2 = isl_union_map_align_params(umap2, isl_union_map_get_space(umap1));

	if (!umap1 || !umap2)
		goto error;

	data.umap2 = umap2;
	data.res = isl_union_map_alloc(isl_space_copy(umap1->dim),
				       umap1->table.n);
	if (isl_hash_table_foreach(umap1->dim->ctx, &umap1->table,
				   &gen_bin_entry, &data) < 0)
		goto error;

	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	return data.res;
error:
	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	isl_union_map_free(data.res);
	return NULL;
}

__isl_give isl_union_map *isl_union_map_subtract(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	struct isl_bin_op_control control = {
		.subtract = 1,
		.match_space = &identity,
		.fn_map = &isl_map_subtract,
	};

	return gen_bin_op(umap1, umap2, &control);
}

__isl_give isl_union_set *isl_union_set_subtract(
	__isl_take isl_union_set *uset1, __isl_take isl_union_set *uset2)
{
	return isl_union_map_subtract(uset1, uset2);
}

struct isl_union_map_gen_bin_set_data {
	isl_set *set;
	isl_union_map *res;
};

static isl_stat intersect_params_entry(void **entry, void *user)
{
	struct isl_union_map_gen_bin_set_data *data = user;
	isl_map *map = *entry;
	int empty;

	map = isl_map_copy(map);
	map = isl_map_intersect_params(map, isl_set_copy(data->set));

	empty = isl_map_is_empty(map);
	if (empty < 0) {
		isl_map_free(map);
		return isl_stat_error;
	}

	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

static __isl_give isl_union_map *gen_bin_set_op(__isl_take isl_union_map *umap,
	__isl_take isl_set *set, isl_stat (*fn)(void **, void *))
{
	struct isl_union_map_gen_bin_set_data data = { NULL, NULL };

	umap = isl_union_map_align_params(umap, isl_set_get_space(set));
	set = isl_set_align_params(set, isl_union_map_get_space(umap));

	if (!umap || !set)
		goto error;

	data.set = set;
	data.res = isl_union_map_alloc(isl_space_copy(umap->dim),
				       umap->table.n);
	if (isl_hash_table_foreach(umap->dim->ctx, &umap->table,
				   fn, &data) < 0)
		goto error;

	isl_union_map_free(umap);
	isl_set_free(set);
	return data.res;
error:
	isl_union_map_free(umap);
	isl_set_free(set);
	isl_union_map_free(data.res);
	return NULL;
}

/* Intersect "umap" with the parameter domain "set".
 *
 * If "set" does not have any constraints, then we can return immediately.
 */
__isl_give isl_union_map *isl_union_map_intersect_params(
	__isl_take isl_union_map *umap, __isl_take isl_set *set)
{
	int is_universe;

	is_universe = isl_set_plain_is_universe(set);
	if (is_universe < 0)
		goto error;
	if (is_universe) {
		isl_set_free(set);
		return umap;
	}

	return gen_bin_set_op(umap, set, &intersect_params_entry);
error:
	isl_union_map_free(umap);
	isl_set_free(set);
	return NULL;
}

__isl_give isl_union_set *isl_union_set_intersect_params(
	__isl_take isl_union_set *uset, __isl_take isl_set *set)
{
	return isl_union_map_intersect_params(uset, set);
}

static __isl_give isl_union_map *union_map_intersect_params(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset)
{
	return isl_union_map_intersect_params(umap,
						isl_set_from_union_set(uset));
}

static __isl_give isl_union_map *union_map_gist_params(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset)
{
	return isl_union_map_gist_params(umap, isl_set_from_union_set(uset));
}

struct isl_union_map_match_bin_data {
	isl_union_map *umap2;
	isl_union_map *res;
	__isl_give isl_map *(*fn)(__isl_take isl_map*, __isl_take isl_map*);
};

static isl_stat match_bin_entry(void **entry, void *user)
{
	struct isl_union_map_match_bin_data *data = user;
	struct isl_hash_table_entry *entry2;
	isl_space *space;
	isl_map *map = *entry;
	int empty;

	space = isl_map_peek_space(map);
	entry2 = isl_union_map_find_entry(data->umap2, space, 0);
	if (!entry2)
		return isl_stat_error;
	if (entry2 == isl_hash_table_entry_none)
		return isl_stat_ok;

	map = isl_map_copy(map);
	map = data->fn(map, isl_map_copy(entry2->data));

	empty = isl_map_is_empty(map);
	if (empty < 0) {
		isl_map_free(map);
		return isl_stat_error;
	}
	if (empty) {
		isl_map_free(map);
		return isl_stat_ok;
	}

	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

static __isl_give isl_union_map *match_bin_op(__isl_take isl_union_map *umap1,
	__isl_take isl_union_map *umap2,
	__isl_give isl_map *(*fn)(__isl_take isl_map*, __isl_take isl_map*))
{
	struct isl_union_map_match_bin_data data = { NULL, NULL, fn };

	umap1 = isl_union_map_align_params(umap1, isl_union_map_get_space(umap2));
	umap2 = isl_union_map_align_params(umap2, isl_union_map_get_space(umap1));

	if (!umap1 || !umap2)
		goto error;

	data.umap2 = umap2;
	data.res = isl_union_map_alloc(isl_space_copy(umap1->dim),
				       umap1->table.n);
	if (isl_hash_table_foreach(umap1->dim->ctx, &umap1->table,
				   &match_bin_entry, &data) < 0)
		goto error;

	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	return data.res;
error:
	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	isl_union_map_free(data.res);
	return NULL;
}

__isl_give isl_union_map *isl_union_map_intersect(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return match_bin_op(umap1, umap2, &isl_map_intersect);
}

/* Compute the intersection of the two union_sets.
 * As a special case, if exactly one of the two union_sets
 * is a parameter domain, then intersect the parameter domain
 * of the other one with this set.
 */
__isl_give isl_union_set *isl_union_set_intersect(
	__isl_take isl_union_set *uset1, __isl_take isl_union_set *uset2)
{
	int p1, p2;

	p1 = isl_union_set_is_params(uset1);
	p2 = isl_union_set_is_params(uset2);
	if (p1 < 0 || p2 < 0)
		goto error;
	if (!p1 && p2)
		return union_map_intersect_params(uset1, uset2);
	if (p1 && !p2)
		return union_map_intersect_params(uset2, uset1);
	return isl_union_map_intersect(uset1, uset2);
error:
	isl_union_set_free(uset1);
	isl_union_set_free(uset2);
	return NULL;
}

static isl_stat gist_params_entry(void **entry, void *user)
{
	struct isl_union_map_gen_bin_set_data *data = user;
	isl_map *map = *entry;
	int empty;

	map = isl_map_copy(map);
	map = isl_map_gist_params(map, isl_set_copy(data->set));

	empty = isl_map_is_empty(map);
	if (empty < 0) {
		isl_map_free(map);
		return isl_stat_error;
	}

	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_gist_params(
	__isl_take isl_union_map *umap, __isl_take isl_set *set)
{
	return gen_bin_set_op(umap, set, &gist_params_entry);
}

__isl_give isl_union_set *isl_union_set_gist_params(
	__isl_take isl_union_set *uset, __isl_take isl_set *set)
{
	return isl_union_map_gist_params(uset, set);
}

__isl_give isl_union_map *isl_union_map_gist(__isl_take isl_union_map *umap,
	__isl_take isl_union_map *context)
{
	return match_bin_op(umap, context, &isl_map_gist);
}

__isl_give isl_union_set *isl_union_set_gist(__isl_take isl_union_set *uset,
	__isl_take isl_union_set *context)
{
	if (isl_union_set_is_params(context))
		return union_map_gist_params(uset, context);
	return isl_union_map_gist(uset, context);
}

/* For each map in "umap", remove the constraints in the corresponding map
 * of "context".
 * Each map in "context" is assumed to consist of a single disjunct and
 * to have explicit representations for all local variables.
 */
__isl_give isl_union_map *isl_union_map_plain_gist(
	__isl_take isl_union_map *umap, __isl_take isl_union_map *context)
{
	return match_bin_op(umap, context, &isl_map_plain_gist);
}

/* For each set in "uset", remove the constraints in the corresponding set
 * of "context".
 * Each set in "context" is assumed to consist of a single disjunct and
 * to have explicit representations for all local variables.
 */
__isl_give isl_union_set *isl_union_set_plain_gist(
	__isl_take isl_union_set *uset, __isl_take isl_union_set *context)
{
	return isl_union_map_plain_gist(uset, context);
}

static __isl_give isl_map *lex_le_set(__isl_take isl_map *set1,
	__isl_take isl_map *set2)
{
	return isl_set_lex_le_set(set_from_map(set1), set_from_map(set2));
}

static __isl_give isl_map *lex_lt_set(__isl_take isl_map *set1,
	__isl_take isl_map *set2)
{
	return isl_set_lex_lt_set(set_from_map(set1), set_from_map(set2));
}

__isl_give isl_union_map *isl_union_set_lex_lt_union_set(
	__isl_take isl_union_set *uset1, __isl_take isl_union_set *uset2)
{
	return match_bin_op(uset1, uset2, &lex_lt_set);
}

__isl_give isl_union_map *isl_union_set_lex_le_union_set(
	__isl_take isl_union_set *uset1, __isl_take isl_union_set *uset2)
{
	return match_bin_op(uset1, uset2, &lex_le_set);
}

__isl_give isl_union_map *isl_union_set_lex_gt_union_set(
	__isl_take isl_union_set *uset1, __isl_take isl_union_set *uset2)
{
	return isl_union_map_reverse(isl_union_set_lex_lt_union_set(uset2, uset1));
}

__isl_give isl_union_map *isl_union_set_lex_ge_union_set(
	__isl_take isl_union_set *uset1, __isl_take isl_union_set *uset2)
{
	return isl_union_map_reverse(isl_union_set_lex_le_union_set(uset2, uset1));
}

__isl_give isl_union_map *isl_union_map_lex_gt_union_map(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return isl_union_map_reverse(isl_union_map_lex_lt_union_map(umap2, umap1));
}

__isl_give isl_union_map *isl_union_map_lex_ge_union_map(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return isl_union_map_reverse(isl_union_map_lex_le_union_map(umap2, umap1));
}

/* Intersect the domain of "umap" with "uset".
 */
static __isl_give isl_union_map *union_map_intersect_domain(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset)
{
	struct isl_bin_op_control control = {
		.match_space = &isl_space_domain,
		.fn_map = &isl_map_intersect_domain,
	};

	return gen_bin_op(umap, uset, &control);
}

/* Intersect the domain of "umap" with "uset".
 * If "uset" is a parameters domain, then intersect the parameter
 * domain of "umap" with this set.
 */
__isl_give isl_union_map *isl_union_map_intersect_domain_union_set(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset)
{
	if (isl_union_set_is_params(uset))
		return union_map_intersect_params(umap, uset);
	else
		return union_map_intersect_domain(umap, uset);
}

/* This is an alternative name for the function above.
 */
__isl_give isl_union_map *isl_union_map_intersect_domain(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset)
{
	return isl_union_map_intersect_domain_union_set(umap, uset);
}

/* Remove the elements of "uset" from the domain of "umap".
 */
__isl_give isl_union_map *isl_union_map_subtract_domain(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *dom)
{
	struct isl_bin_op_control control = {
		.subtract = 1,
		.match_space = &isl_space_domain,
		.fn_map = &isl_map_subtract_domain,
	};

	return gen_bin_op(umap, dom, &control);
}

/* Remove the elements of "uset" from the range of "umap".
 */
__isl_give isl_union_map *isl_union_map_subtract_range(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *dom)
{
	struct isl_bin_op_control control = {
		.subtract = 1,
		.match_space = &isl_space_range,
		.fn_map = &isl_map_subtract_range,
	};

	return gen_bin_op(umap, dom, &control);
}

/* Compute the gist of "umap" with respect to the domain "uset".
 */
static __isl_give isl_union_map *union_map_gist_domain(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset)
{
	struct isl_bin_op_control control = {
		.match_space = &isl_space_domain,
		.fn_map = &isl_map_gist_domain,
	};

	return gen_bin_op(umap, uset, &control);
}

/* Compute the gist of "umap" with respect to the domain "uset".
 * If "uset" is a parameters domain, then compute the gist
 * with respect to this parameter domain.
 */
__isl_give isl_union_map *isl_union_map_gist_domain(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset)
{
	if (isl_union_set_is_params(uset))
		return union_map_gist_params(umap, uset);
	else
		return union_map_gist_domain(umap, uset);
}

/* Compute the gist of "umap" with respect to the range "uset".
 */
__isl_give isl_union_map *isl_union_map_gist_range(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset)
{
	struct isl_bin_op_control control = {
		.match_space = &isl_space_range,
		.fn_map = &isl_map_gist_range,
	};

	return gen_bin_op(umap, uset, &control);
}

__isl_give isl_union_map *isl_union_map_intersect_range_union_set(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset)
{
	struct isl_bin_op_control control = {
		.match_space = &isl_space_range,
		.fn_map = &isl_map_intersect_range,
	};

	return gen_bin_op(umap, uset, &control);
}

/* This is an alternative name for the function above.
 */
__isl_give isl_union_map *isl_union_map_intersect_range(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset)
{
	return isl_union_map_intersect_range_union_set(umap, uset);
}

/* Intersect each map in "umap" in a space [A -> B] -> C
 * with the corresponding map in "factor" in the space A -> C and
 * collect the results.
 */
__isl_give isl_union_map *isl_union_map_intersect_domain_factor_domain(
	__isl_take isl_union_map *umap, __isl_take isl_union_map *factor)
{
	struct isl_bin_op_control control = {
		.filter = &isl_map_domain_is_wrapping,
		.match_space = &isl_space_domain_factor_domain,
		.fn_map = &isl_map_intersect_domain_factor_domain,
	};

	return gen_bin_op(umap, factor, &control);
}

/* Intersect each map in "umap" in a space [A -> B] -> C
 * with the corresponding map in "factor" in the space B -> C and
 * collect the results.
 */
__isl_give isl_union_map *isl_union_map_intersect_domain_factor_range(
	__isl_take isl_union_map *umap, __isl_take isl_union_map *factor)
{
	struct isl_bin_op_control control = {
		.filter = &isl_map_domain_is_wrapping,
		.match_space = &isl_space_domain_factor_range,
		.fn_map = &isl_map_intersect_domain_factor_range,
	};

	return gen_bin_op(umap, factor, &control);
}

/* Intersect each map in "umap" in a space A -> [B -> C]
 * with the corresponding map in "factor" in the space A -> B and
 * collect the results.
 */
__isl_give isl_union_map *isl_union_map_intersect_range_factor_domain(
	__isl_take isl_union_map *umap, __isl_take isl_union_map *factor)
{
	struct isl_bin_op_control control = {
		.filter = &isl_map_range_is_wrapping,
		.match_space = &isl_space_range_factor_domain,
		.fn_map = &isl_map_intersect_range_factor_domain,
	};

	return gen_bin_op(umap, factor, &control);
}

/* Intersect each map in "umap" in a space A -> [B -> C]
 * with the corresponding map in "factor" in the space A -> C and
 * collect the results.
 */
__isl_give isl_union_map *isl_union_map_intersect_range_factor_range(
	__isl_take isl_union_map *umap, __isl_take isl_union_map *factor)
{
	struct isl_bin_op_control control = {
		.filter = &isl_map_range_is_wrapping,
		.match_space = &isl_space_range_factor_range,
		.fn_map = &isl_map_intersect_range_factor_range,
	};

	return gen_bin_op(umap, factor, &control);
}

struct isl_union_map_bin_data {
	isl_union_map *umap2;
	isl_union_map *res;
	isl_map *map;
	isl_stat (*fn)(void **entry, void *user);
};

static isl_stat apply_range_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_map *map2 = *entry;
	isl_bool empty, match;

	match = isl_map_tuple_is_equal(data->map, isl_dim_out,
				map2, isl_dim_in);
	if (match < 0)
		return isl_stat_error;
	if (!match)
		return isl_stat_ok;

	map2 = isl_map_apply_range(isl_map_copy(data->map), isl_map_copy(map2));

	empty = isl_map_is_empty(map2);
	if (empty < 0) {
		isl_map_free(map2);
		return isl_stat_error;
	}
	if (empty) {
		isl_map_free(map2);
		return isl_stat_ok;
	}

	data->res = isl_union_map_add_map(data->res, map2);

	return isl_stat_ok;
}

static isl_stat bin_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_map *map = *entry;

	data->map = map;
	if (isl_hash_table_foreach(data->umap2->dim->ctx, &data->umap2->table,
				   data->fn, data) < 0)
		return isl_stat_error;

	return isl_stat_ok;
}

static __isl_give isl_union_map *bin_op(__isl_take isl_union_map *umap1,
	__isl_take isl_union_map *umap2,
	isl_stat (*fn)(void **entry, void *user))
{
	struct isl_union_map_bin_data data = { NULL, NULL, NULL, fn };

	umap1 = isl_union_map_align_params(umap1, isl_union_map_get_space(umap2));
	umap2 = isl_union_map_align_params(umap2, isl_union_map_get_space(umap1));

	if (!umap1 || !umap2)
		goto error;

	data.umap2 = umap2;
	data.res = isl_union_map_alloc(isl_space_copy(umap1->dim),
				       umap1->table.n);
	if (isl_hash_table_foreach(umap1->dim->ctx, &umap1->table,
				   &bin_entry, &data) < 0)
		goto error;

	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	return data.res;
error:
	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	isl_union_map_free(data.res);
	return NULL;
}

__isl_give isl_union_map *isl_union_map_apply_range(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return bin_op(umap1, umap2, &apply_range_entry);
}

__isl_give isl_union_map *isl_union_map_apply_domain(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	umap1 = isl_union_map_reverse(umap1);
	umap1 = isl_union_map_apply_range(umap1, umap2);
	return isl_union_map_reverse(umap1);
}

__isl_give isl_union_set *isl_union_set_apply(
	__isl_take isl_union_set *uset, __isl_take isl_union_map *umap)
{
	return isl_union_map_apply_range(uset, umap);
}

static isl_stat map_lex_lt_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_map *map2 = *entry;
	isl_bool match;

	match = isl_map_tuple_is_equal(data->map, isl_dim_out,
				 map2, isl_dim_out);
	if (match < 0)
		return isl_stat_error;
	if (!match)
		return isl_stat_ok;

	map2 = isl_map_lex_lt_map(isl_map_copy(data->map), isl_map_copy(map2));

	data->res = isl_union_map_add_map(data->res, map2);

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_lex_lt_union_map(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return bin_op(umap1, umap2, &map_lex_lt_entry);
}

static isl_stat map_lex_le_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_map *map2 = *entry;
	isl_bool match;

	match = isl_map_tuple_is_equal(data->map, isl_dim_out,
				 map2, isl_dim_out);
	if (match < 0)
		return isl_stat_error;
	if (!match)
		return isl_stat_ok;

	map2 = isl_map_lex_le_map(isl_map_copy(data->map), isl_map_copy(map2));

	data->res = isl_union_map_add_map(data->res, map2);

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_lex_le_union_map(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return bin_op(umap1, umap2, &map_lex_le_entry);
}

static isl_stat product_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_map *map2 = *entry;

	map2 = isl_map_product(isl_map_copy(data->map), isl_map_copy(map2));

	data->res = isl_union_map_add_map(data->res, map2);

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_product(__isl_take isl_union_map *umap1,
	__isl_take isl_union_map *umap2)
{
	return bin_op(umap1, umap2, &product_entry);
}

static isl_stat set_product_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_set *set2 = *entry;

	set2 = isl_set_product(isl_set_copy(data->map), isl_set_copy(set2));

	data->res = isl_union_set_add_set(data->res, set2);

	return isl_stat_ok;
}

__isl_give isl_union_set *isl_union_set_product(__isl_take isl_union_set *uset1,
	__isl_take isl_union_set *uset2)
{
	return bin_op(uset1, uset2, &set_product_entry);
}

static isl_stat domain_product_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_map *map2 = *entry;
	isl_bool match;

	match = isl_map_tuple_is_equal(data->map, isl_dim_out,
				 map2, isl_dim_out);
	if (match < 0)
		return isl_stat_error;
	if (!match)
		return isl_stat_ok;

	map2 = isl_map_domain_product(isl_map_copy(data->map),
				     isl_map_copy(map2));

	data->res = isl_union_map_add_map(data->res, map2);

	return isl_stat_ok;
}

/* Given two maps A -> B and C -> D, construct a map [A -> C] -> (B * D)
 */
__isl_give isl_union_map *isl_union_map_domain_product(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return bin_op(umap1, umap2, &domain_product_entry);
}

static isl_stat range_product_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_map *map2 = *entry;
	isl_bool match;

	match = isl_map_tuple_is_equal(data->map, isl_dim_in, map2, isl_dim_in);
	if (match < 0)
		return isl_stat_error;
	if (!match)
		return isl_stat_ok;

	map2 = isl_map_range_product(isl_map_copy(data->map),
				     isl_map_copy(map2));

	data->res = isl_union_map_add_map(data->res, map2);

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_range_product(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return bin_op(umap1, umap2, &range_product_entry);
}

/* If data->map A -> B and "map2" C -> D have the same range space,
 * then add (A, C) -> (B * D) to data->res.
 */
static isl_stat flat_domain_product_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_map *map2 = *entry;
	isl_bool match;

	match = isl_map_tuple_is_equal(data->map, isl_dim_out,
				 map2, isl_dim_out);
	if (match < 0)
		return isl_stat_error;
	if (!match)
		return isl_stat_ok;

	map2 = isl_map_flat_domain_product(isl_map_copy(data->map),
					  isl_map_copy(map2));

	data->res = isl_union_map_add_map(data->res, map2);

	return isl_stat_ok;
}

/* Given two maps A -> B and C -> D, construct a map (A, C) -> (B * D).
 */
__isl_give isl_union_map *isl_union_map_flat_domain_product(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return bin_op(umap1, umap2, &flat_domain_product_entry);
}

static isl_stat flat_range_product_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_map *map2 = *entry;
	isl_bool match;

	match = isl_map_tuple_is_equal(data->map, isl_dim_in, map2, isl_dim_in);
	if (match < 0)
		return isl_stat_error;
	if (!match)
		return isl_stat_ok;

	map2 = isl_map_flat_range_product(isl_map_copy(data->map),
					  isl_map_copy(map2));

	data->res = isl_union_map_add_map(data->res, map2);

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_flat_range_product(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return bin_op(umap1, umap2, &flat_range_product_entry);
}

/* Data structure that specifies how un_op should modify
 * the maps in the union map.
 *
 * If "inplace" is set, then the maps in the input union map
 * are modified in place.  This means that "fn_map" should not
 * change the meaning of the map or that the union map only
 * has a single reference.
 * If "total" is set, then all maps need to be modified and
 * the results need to live in the same space.
 * Otherwise, a new union map is constructed to store the results.
 * If "filter" is not NULL, then only the input maps that satisfy "filter"
 * are taken into account.  "filter_user" is passed as the second argument
 * to "filter".  No filter can be set if "inplace" or
 * "total" is set.
 * At most one of "fn_map" or "fn_map2" can be set, specifying
 * how the maps (selected by "filter") should be transformed.
 * If "fn_map2" is set, then "fn_map2_user" is passed as the second argument.
 */
struct isl_un_op_control {
	int inplace;
	int total;
	isl_bool (*filter)(__isl_keep isl_map *map, void *user);
	void *filter_user;
	__isl_give isl_map *(*fn_map)(__isl_take isl_map *map);
	__isl_give isl_map *(*fn_map2)(__isl_take isl_map *map, void *user);
	void *fn_map2_user;
};

/* Data structure for wrapping the data for un_op_filter_drop_user.
 * "filter" is the function that is being wrapped.
 */
struct isl_un_op_drop_user_data {
	isl_bool (*filter)(__isl_keep isl_map *map);
};

/* Wrapper for isl_un_op_control filters that do not require
 * a second argument.
 * Simply call data->filter without the second argument.
 */
static isl_bool un_op_filter_drop_user(__isl_keep isl_map *map, void *user)
{
	struct isl_un_op_drop_user_data *data = user;
	return data->filter(map);
}

/* Internal data structure for "un_op".
 * "control" specifies how the maps in the union map should be modified.
 * "res" collects the results.
 */
struct isl_union_map_un_data {
	struct isl_un_op_control *control;
	isl_union_map *res;
};

/* isl_hash_table_foreach callback for un_op.
 * Handle the map that "entry" points to.
 *
 * If control->filter is set, then check if this map satisfies the filter.
 * If so (or if control->filter is not set), modify the map
 * by calling control->fn_map or control->fn_map2 (if set) and
 * either add the result to data->res or
 * replace the original entry by the result (if control->inplace is set).
 */
static isl_stat un_entry(void **entry, void *user)
{
	struct isl_union_map_un_data *data = user;
	struct isl_un_op_control *control = data->control;
	isl_map *map = *entry;

	if (control->filter) {
		isl_bool ok;

		ok = control->filter(map, control->filter_user);
		if (ok < 0)
			return isl_stat_error;
		if (!ok)
			return isl_stat_ok;
	}

	map = isl_map_copy(map);
	if (control->fn_map2 != NULL)
		map = control->fn_map2(map, control->fn_map2_user);
	else if (control->fn_map != NULL)
		map = control->fn_map(map);
	if (!map)
		return isl_stat_error;
	if (control->inplace) {
		isl_map_free(*entry);
		*entry = map;
	} else {
		data->res = isl_union_map_add_map(data->res, map);
		if (!data->res)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Modify the maps in "umap" based on "control".
 * If control->inplace is set, then modify the maps in "umap" in-place.
 * Otherwise, create a new union map to hold the results.
 * If control->total is set, then perform an inplace computation
 * if "umap" is only referenced once.  Otherwise, create a new union map
 * to store the results.
 */
static __isl_give isl_union_map *un_op(__isl_take isl_union_map *umap,
	struct isl_un_op_control *control)
{
	struct isl_union_map_un_data data = { control };

	if (!umap)
		return NULL;
	if (!!control->fn_map && !!control->fn_map2)
		isl_die(isl_union_map_get_ctx(umap), isl_error_internal,
			"at most one mapping function can be specified",
			return isl_union_map_free(umap));
	if ((control->inplace || control->total) && control->filter)
		isl_die(isl_union_map_get_ctx(umap), isl_error_invalid,
			"inplace/total modification cannot be filtered",
			return isl_union_map_free(umap));

	if (control->total && umap->ref == 1)
		control->inplace = 1;
	if (control->inplace) {
		data.res = umap;
	} else {
		isl_space *space;

		space = isl_union_map_get_space(umap);
		data.res = isl_union_map_alloc(space, umap->table.n);
	}
	if (isl_hash_table_foreach(isl_union_map_get_ctx(umap),
				    &umap->table, &un_entry, &data) < 0)
		data.res = isl_union_map_free(data.res);

	if (control->inplace)
		return data.res;
	isl_union_map_free(umap);
	return data.res;
}

__isl_give isl_union_map *isl_union_map_from_range(
	__isl_take isl_union_set *uset)
{
	struct isl_un_op_control control = {
		.fn_map = &isl_map_from_range,
	};
	return un_op(uset, &control);
}

__isl_give isl_union_map *isl_union_map_from_domain(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_reverse(isl_union_map_from_range(uset));
}

__isl_give isl_union_map *isl_union_map_from_domain_and_range(
	__isl_take isl_union_set *domain, __isl_take isl_union_set *range)
{
	return isl_union_map_apply_range(isl_union_map_from_domain(domain),
				         isl_union_map_from_range(range));
}

/* Modify the maps in "umap" by applying "fn" on them.
 * "fn" should apply to all maps in "umap" and should not modify the space.
 */
static __isl_give isl_union_map *total(__isl_take isl_union_map *umap,
	__isl_give isl_map *(*fn)(__isl_take isl_map *))
{
	struct isl_un_op_control control = {
		.total = 1,
		.fn_map = fn,
	};

	return un_op(umap, &control);
}

/* Compute the affine hull of "map" and return the result as an isl_map.
 */
static __isl_give isl_map *isl_map_affine_hull_map(__isl_take isl_map *map)
{
	return isl_map_from_basic_map(isl_map_affine_hull(map));
}

__isl_give isl_union_map *isl_union_map_affine_hull(
	__isl_take isl_union_map *umap)
{
	return total(umap, &isl_map_affine_hull_map);
}

__isl_give isl_union_set *isl_union_set_affine_hull(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_affine_hull(uset);
}

/* Wrapper around isl_set_combined_lineality_space
 * that returns the combined lineality space in the form of an isl_set
 * instead of an isl_basic_set.
 */
static __isl_give isl_set *combined_lineality_space(__isl_take isl_set *set)
{
	return isl_set_from_basic_set(isl_set_combined_lineality_space(set));
}

/* For each set in "uset", compute the (linear) hull
 * of the lineality spaces of its basic sets and
 * collect and return the results.
 */
__isl_give isl_union_set *isl_union_set_combined_lineality_space(
	__isl_take isl_union_set *uset)
{
	struct isl_un_op_control control = {
		.fn_map = &combined_lineality_space,
	};
	return un_op(uset, &control);
}

/* Compute the polyhedral hull of "map" and return the result as an isl_map.
 */
static __isl_give isl_map *isl_map_polyhedral_hull_map(__isl_take isl_map *map)
{
	return isl_map_from_basic_map(isl_map_polyhedral_hull(map));
}

__isl_give isl_union_map *isl_union_map_polyhedral_hull(
	__isl_take isl_union_map *umap)
{
	return total(umap, &isl_map_polyhedral_hull_map);
}

__isl_give isl_union_set *isl_union_set_polyhedral_hull(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_polyhedral_hull(uset);
}

/* Compute a superset of the convex hull of "map" that is described
 * by only translates of the constraints in the constituents of "map" and
 * return the result as an isl_map.
 */
static __isl_give isl_map *isl_map_simple_hull_map(__isl_take isl_map *map)
{
	return isl_map_from_basic_map(isl_map_simple_hull(map));
}

__isl_give isl_union_map *isl_union_map_simple_hull(
	__isl_take isl_union_map *umap)
{
	return total(umap, &isl_map_simple_hull_map);
}

__isl_give isl_union_set *isl_union_set_simple_hull(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_simple_hull(uset);
}

static __isl_give isl_union_map *inplace(__isl_take isl_union_map *umap,
	__isl_give isl_map *(*fn)(__isl_take isl_map *))
{
	struct isl_un_op_control control = {
		.inplace = 1,
		.fn_map = fn,
	};

	return un_op(umap, &control);
}

/* Remove redundant constraints in each of the basic maps of "umap".
 * Since removing redundant constraints does not change the meaning
 * or the space, the operation can be performed in-place.
 */
__isl_give isl_union_map *isl_union_map_remove_redundancies(
	__isl_take isl_union_map *umap)
{
	return inplace(umap, &isl_map_remove_redundancies);
}

/* Remove redundant constraints in each of the basic sets of "uset".
 */
__isl_give isl_union_set *isl_union_set_remove_redundancies(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_remove_redundancies(uset);
}

__isl_give isl_union_map *isl_union_map_coalesce(
	__isl_take isl_union_map *umap)
{
	return inplace(umap, &isl_map_coalesce);
}

__isl_give isl_union_set *isl_union_set_coalesce(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_coalesce(uset);
}

__isl_give isl_union_map *isl_union_map_detect_equalities(
	__isl_take isl_union_map *umap)
{
	return inplace(umap, &isl_map_detect_equalities);
}

__isl_give isl_union_set *isl_union_set_detect_equalities(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_detect_equalities(uset);
}

__isl_give isl_union_map *isl_union_map_compute_divs(
	__isl_take isl_union_map *umap)
{
	return inplace(umap, &isl_map_compute_divs);
}

__isl_give isl_union_set *isl_union_set_compute_divs(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_compute_divs(uset);
}

__isl_give isl_union_map *isl_union_map_lexmin(
	__isl_take isl_union_map *umap)
{
	return total(umap, &isl_map_lexmin);
}

__isl_give isl_union_set *isl_union_set_lexmin(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_lexmin(uset);
}

__isl_give isl_union_map *isl_union_map_lexmax(
	__isl_take isl_union_map *umap)
{
	return total(umap, &isl_map_lexmax);
}

__isl_give isl_union_set *isl_union_set_lexmax(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_lexmax(uset);
}

/* Return the universe in the space of "map".
 */
static __isl_give isl_map *universe(__isl_take isl_map *map)
{
	isl_space *space;

	space = isl_map_get_space(map);
	isl_map_free(map);
	return isl_map_universe(space);
}

__isl_give isl_union_map *isl_union_map_universe(__isl_take isl_union_map *umap)
{
	struct isl_un_op_control control = {
		.fn_map = &universe,
	};
	return un_op(umap, &control);
}

__isl_give isl_union_set *isl_union_set_universe(__isl_take isl_union_set *uset)
{
	return isl_union_map_universe(uset);
}

__isl_give isl_union_map *isl_union_map_reverse(__isl_take isl_union_map *umap)
{
	struct isl_un_op_control control = {
		.fn_map = &isl_map_reverse,
	};
	return un_op(umap, &control);
}

/* Given a union map, take the maps of the form A -> (B -> C) and
 * return the union of the corresponding maps A -> (C -> B).
 */
__isl_give isl_union_map *isl_union_map_range_reverse(
	__isl_take isl_union_map *umap)
{
	struct isl_un_op_drop_user_data data = { &isl_map_range_is_wrapping };
	struct isl_un_op_control control = {
		.filter = &un_op_filter_drop_user,
		.filter_user = &data,
		.fn_map = &isl_map_range_reverse,
	};
	return un_op(umap, &control);
}

/* Compute the parameter domain of the given union map.
 */
__isl_give isl_set *isl_union_map_params(__isl_take isl_union_map *umap)
{
	struct isl_un_op_control control = {
		.fn_map = &isl_map_params,
	};
	int empty;

	empty = isl_union_map_is_empty(umap);
	if (empty < 0)
		goto error;
	if (empty) {
		isl_space *space;
		space = isl_union_map_get_space(umap);
		isl_union_map_free(umap);
		return isl_set_empty(space);
	}
	return isl_set_from_union_set(un_op(umap, &control));
error:
	isl_union_map_free(umap);
	return NULL;
}

/* Compute the parameter domain of the given union set.
 */
__isl_give isl_set *isl_union_set_params(__isl_take isl_union_set *uset)
{
	return isl_union_map_params(uset);
}

__isl_give isl_union_set *isl_union_map_domain(__isl_take isl_union_map *umap)
{
	struct isl_un_op_control control = {
		.fn_map = &isl_map_domain,
	};
	return un_op(umap, &control);
}

__isl_give isl_union_set *isl_union_map_range(__isl_take isl_union_map *umap)
{
	struct isl_un_op_control control = {
		.fn_map = &isl_map_range,
	};
	return un_op(umap, &control);
}

__isl_give isl_union_map *isl_union_map_domain_map(
	__isl_take isl_union_map *umap)
{
	struct isl_un_op_control control = {
		.fn_map = &isl_map_domain_map,
	};
	return un_op(umap, &control);
}

/* Construct an isl_pw_multi_aff that maps "map" to its domain and
 * add the result to "res".
 */
static isl_stat domain_map_upma(__isl_take isl_map *map, void *user)
{
	isl_union_pw_multi_aff **res = user;
	isl_multi_aff *ma;
	isl_pw_multi_aff *pma;

	ma = isl_multi_aff_domain_map(isl_map_get_space(map));
	pma = isl_pw_multi_aff_alloc(isl_map_wrap(map), ma);
	*res = isl_union_pw_multi_aff_add_pw_multi_aff(*res, pma);

	return *res ? isl_stat_ok : isl_stat_error;

}

/* Return an isl_union_pw_multi_aff that maps a wrapped copy of "umap"
 * to its domain.
 */
__isl_give isl_union_pw_multi_aff *isl_union_map_domain_map_union_pw_multi_aff(
	__isl_take isl_union_map *umap)
{
	isl_union_pw_multi_aff *res;

	res = isl_union_pw_multi_aff_empty(isl_union_map_get_space(umap));
	if (isl_union_map_foreach_map(umap, &domain_map_upma, &res) < 0)
		res = isl_union_pw_multi_aff_free(res);

	isl_union_map_free(umap);
	return res;
}

__isl_give isl_union_map *isl_union_map_range_map(
	__isl_take isl_union_map *umap)
{
	struct isl_un_op_control control = {
		.fn_map = &isl_map_range_map,
	};
	return un_op(umap, &control);
}

/* Given a collection of wrapped maps of the form A[B -> C],
 * return the collection of maps A[B -> C] -> B.
 */
__isl_give isl_union_map *isl_union_set_wrapped_domain_map(
	__isl_take isl_union_set *uset)
{
	struct isl_un_op_drop_user_data data = { &isl_set_is_wrapping };
	struct isl_un_op_control control = {
		.filter = &un_op_filter_drop_user,
		.filter_user = &data,
		.fn_map = &isl_set_wrapped_domain_map,
	};
	return un_op(uset, &control);
}

/* Does "map" relate elements from the same space?
 */
static isl_bool equal_tuples(__isl_keep isl_map *map, void *user)
{
	return isl_map_tuple_is_equal(map, isl_dim_in, map, isl_dim_out);
}

__isl_give isl_union_set *isl_union_map_deltas(__isl_take isl_union_map *umap)
{
	struct isl_un_op_control control = {
		.filter = &equal_tuples,
		.fn_map = &isl_map_deltas,
	};
	return un_op(umap, &control);
}

__isl_give isl_union_map *isl_union_map_deltas_map(
	__isl_take isl_union_map *umap)
{
	struct isl_un_op_control control = {
		.filter = &equal_tuples,
		.fn_map = &isl_map_deltas_map,
	};
	return un_op(umap, &control);
}

__isl_give isl_union_map *isl_union_set_identity(__isl_take isl_union_set *uset)
{
	struct isl_un_op_control control = {
		.fn_map = &isl_set_identity,
	};
	return un_op(uset, &control);
}

/* Construct an identity isl_pw_multi_aff on "set" and add it to *res.
 */
static isl_stat identity_upma(__isl_take isl_set *set, void *user)
{
	isl_union_pw_multi_aff **res = user;
	isl_space *space;
	isl_pw_multi_aff *pma;

	space = isl_space_map_from_set(isl_set_get_space(set));
	pma = isl_pw_multi_aff_identity(space);
	pma = isl_pw_multi_aff_intersect_domain(pma, set);
	*res = isl_union_pw_multi_aff_add_pw_multi_aff(*res, pma);

	return *res ? isl_stat_ok : isl_stat_error;
}

/* Return an identity function on "uset" in the form
 * of an isl_union_pw_multi_aff.
 */
__isl_give isl_union_pw_multi_aff *isl_union_set_identity_union_pw_multi_aff(
	__isl_take isl_union_set *uset)
{
	isl_union_pw_multi_aff *res;

	res = isl_union_pw_multi_aff_empty(isl_union_set_get_space(uset));
	if (isl_union_set_foreach_set(uset, &identity_upma, &res) < 0)
		res = isl_union_pw_multi_aff_free(res);

	isl_union_set_free(uset);
	return res;
}

/* For each map in "umap" of the form [A -> B] -> C,
 * construct the map A -> C and collect the results.
 */
__isl_give isl_union_map *isl_union_map_domain_factor_domain(
	__isl_take isl_union_map *umap)
{
	struct isl_un_op_drop_user_data data = { &isl_map_domain_is_wrapping };
	struct isl_un_op_control control = {
		.filter = &un_op_filter_drop_user,
		.filter_user = &data,
		.fn_map = &isl_map_domain_factor_domain,
	};
	return un_op(umap, &control);
}

/* For each map in "umap" of the form [A -> B] -> C,
 * construct the map B -> C and collect the results.
 */
__isl_give isl_union_map *isl_union_map_domain_factor_range(
	__isl_take isl_union_map *umap)
{
	struct isl_un_op_drop_user_data data = { &isl_map_domain_is_wrapping };
	struct isl_un_op_control control = {
		.filter = &un_op_filter_drop_user,
		.filter_user = &data,
		.fn_map = &isl_map_domain_factor_range,
	};
	return un_op(umap, &control);
}

/* For each map in "umap" of the form A -> [B -> C],
 * construct the map A -> B and collect the results.
 */
__isl_give isl_union_map *isl_union_map_range_factor_domain(
	__isl_take isl_union_map *umap)
{
	struct isl_un_op_drop_user_data data = { &isl_map_range_is_wrapping };
	struct isl_un_op_control control = {
		.filter = &un_op_filter_drop_user,
		.filter_user = &data,
		.fn_map = &isl_map_range_factor_domain,
	};
	return un_op(umap, &control);
}

/* For each map in "umap" of the form A -> [B -> C],
 * construct the map A -> C and collect the results.
 */
__isl_give isl_union_map *isl_union_map_range_factor_range(
	__isl_take isl_union_map *umap)
{
	struct isl_un_op_drop_user_data data = { &isl_map_range_is_wrapping };
	struct isl_un_op_control control = {
		.filter = &un_op_filter_drop_user,
		.filter_user = &data,
		.fn_map = &isl_map_range_factor_range,
	};
	return un_op(umap, &control);
}

/* For each map in "umap" of the form [A -> B] -> [C -> D],
 * construct the map A -> C and collect the results.
 */
__isl_give isl_union_map *isl_union_map_factor_domain(
	__isl_take isl_union_map *umap)
{
	struct isl_un_op_drop_user_data data = { &isl_map_is_product };
	struct isl_un_op_control control = {
		.filter = &un_op_filter_drop_user,
		.filter_user = &data,
		.fn_map = &isl_map_factor_domain,
	};
	return un_op(umap, &control);
}

/* For each map in "umap" of the form [A -> B] -> [C -> D],
 * construct the map B -> D and collect the results.
 */
__isl_give isl_union_map *isl_union_map_factor_range(
	__isl_take isl_union_map *umap)
{
	struct isl_un_op_drop_user_data data = { &isl_map_is_product };
	struct isl_un_op_control control = {
		.filter = &un_op_filter_drop_user,
		.filter_user = &data,
		.fn_map = &isl_map_factor_range,
	};
	return un_op(umap, &control);
}

__isl_give isl_union_map *isl_union_set_unwrap(__isl_take isl_union_set *uset)
{
	struct isl_un_op_drop_user_data data = { &isl_set_is_wrapping };
	struct isl_un_op_control control = {
		.filter = &un_op_filter_drop_user,
		.filter_user = &data,
		.fn_map = &isl_set_unwrap,
	};
	return un_op(uset, &control);
}

__isl_give isl_union_set *isl_union_map_wrap(__isl_take isl_union_map *umap)
{
	struct isl_un_op_control control = {
		.fn_map = &isl_map_wrap,
	};
	return un_op(umap, &control);
}

struct isl_union_map_is_subset_data {
	isl_union_map *umap2;
	isl_bool is_subset;
};

static isl_stat is_subset_entry(void **entry, void *user)
{
	struct isl_union_map_is_subset_data *data = user;
	struct isl_hash_table_entry *entry2;
	isl_space *space;
	isl_map *map = *entry;

	space = isl_map_peek_space(map);
	entry2 = isl_union_map_find_entry(data->umap2, space, 0);
	if (!entry2)
		return isl_stat_error;
	if (entry2 == isl_hash_table_entry_none) {
		int empty = isl_map_is_empty(map);
		if (empty < 0)
			return isl_stat_error;
		if (empty)
			return isl_stat_ok;
		data->is_subset = isl_bool_false;
		return isl_stat_error;
	}

	data->is_subset = isl_map_is_subset(map, entry2->data);
	if (data->is_subset < 0 || !data->is_subset)
		return isl_stat_error;

	return isl_stat_ok;
}

isl_bool isl_union_map_is_subset(__isl_keep isl_union_map *umap1,
	__isl_keep isl_union_map *umap2)
{
	struct isl_union_map_is_subset_data data = { NULL, isl_bool_true };

	if (!umap1 || !umap2)
		return isl_bool_error;

	data.umap2 = umap2;
	if (isl_hash_table_foreach(umap1->dim->ctx, &umap1->table,
				   &is_subset_entry, &data) < 0 &&
	    data.is_subset)
		return isl_bool_error;

	return data.is_subset;
}

isl_bool isl_union_set_is_subset(__isl_keep isl_union_set *uset1,
	__isl_keep isl_union_set *uset2)
{
	return isl_union_map_is_subset(uset1, uset2);
}

isl_bool isl_union_map_is_equal(__isl_keep isl_union_map *umap1,
	__isl_keep isl_union_map *umap2)
{
	isl_bool is_subset;

	if (!umap1 || !umap2)
		return isl_bool_error;
	is_subset = isl_union_map_is_subset(umap1, umap2);
	if (is_subset != isl_bool_true)
		return is_subset;
	is_subset = isl_union_map_is_subset(umap2, umap1);
	return is_subset;
}

isl_bool isl_union_set_is_equal(__isl_keep isl_union_set *uset1,
	__isl_keep isl_union_set *uset2)
{
	return isl_union_map_is_equal(uset1, uset2);
}

isl_bool isl_union_map_is_strict_subset(__isl_keep isl_union_map *umap1,
	__isl_keep isl_union_map *umap2)
{
	isl_bool is_subset;

	if (!umap1 || !umap2)
		return isl_bool_error;
	is_subset = isl_union_map_is_subset(umap1, umap2);
	if (is_subset != isl_bool_true)
		return is_subset;
	is_subset = isl_union_map_is_subset(umap2, umap1);
	return isl_bool_not(is_subset);
}

isl_bool isl_union_set_is_strict_subset(__isl_keep isl_union_set *uset1,
	__isl_keep isl_union_set *uset2)
{
	return isl_union_map_is_strict_subset(uset1, uset2);
}

/* Internal data structure for isl_union_map_is_disjoint.
 * umap2 is the union map with which we are comparing.
 * is_disjoint is initialized to 1 and is set to 0 as soon
 * as the union maps turn out not to be disjoint.
 */
struct isl_union_map_is_disjoint_data {
	isl_union_map *umap2;
	isl_bool is_disjoint;
};

/* Check if "map" is disjoint from data->umap2 and abort
 * the search if it is not.
 */
static isl_stat is_disjoint_entry(void **entry, void *user)
{
	struct isl_union_map_is_disjoint_data *data = user;
	struct isl_hash_table_entry *entry2;
	isl_space *space;
	isl_map *map = *entry;

	space = isl_map_peek_space(map);
	entry2 = isl_union_map_find_entry(data->umap2, space, 0);
	if (!entry2)
		return isl_stat_error;
	if (entry2 == isl_hash_table_entry_none)
		return isl_stat_ok;

	data->is_disjoint = isl_map_is_disjoint(map, entry2->data);
	if (data->is_disjoint < 0 || !data->is_disjoint)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Are "umap1" and "umap2" disjoint?
 */
isl_bool isl_union_map_is_disjoint(__isl_keep isl_union_map *umap1,
	__isl_keep isl_union_map *umap2)
{
	struct isl_union_map_is_disjoint_data data = { NULL, isl_bool_true };

	umap1 = isl_union_map_copy(umap1);
	umap2 = isl_union_map_copy(umap2);
	umap1 = isl_union_map_align_params(umap1,
						isl_union_map_get_space(umap2));
	umap2 = isl_union_map_align_params(umap2,
						isl_union_map_get_space(umap1));

	if (!umap1 || !umap2)
		goto error;

	data.umap2 = umap2;
	if (isl_hash_table_foreach(umap1->dim->ctx, &umap1->table,
				   &is_disjoint_entry, &data) < 0 &&
	    data.is_disjoint)
		goto error;

	isl_union_map_free(umap1);
	isl_union_map_free(umap2);

	return data.is_disjoint;
error:
	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	return isl_bool_error;
}

/* Are "uset1" and "uset2" disjoint?
 */
isl_bool isl_union_set_is_disjoint(__isl_keep isl_union_set *uset1,
	__isl_keep isl_union_set *uset2)
{
	return isl_union_map_is_disjoint(uset1, uset2);
}

static isl_stat sample_entry(void **entry, void *user)
{
	isl_basic_map **sample = (isl_basic_map **)user;
	isl_map *map = *entry;

	*sample = isl_map_sample(isl_map_copy(map));
	if (!*sample)
		return isl_stat_error;
	if (!isl_basic_map_plain_is_empty(*sample))
		return isl_stat_error;
	return isl_stat_ok;
}

__isl_give isl_basic_map *isl_union_map_sample(__isl_take isl_union_map *umap)
{
	isl_basic_map *sample = NULL;

	if (!umap)
		return NULL;

	if (isl_hash_table_foreach(umap->dim->ctx, &umap->table,
				   &sample_entry, &sample) < 0 &&
	    !sample)
		goto error;

	if (!sample)
		sample = isl_basic_map_empty(isl_union_map_get_space(umap));

	isl_union_map_free(umap);

	return sample;
error:
	isl_union_map_free(umap);
	return NULL;
}

__isl_give isl_basic_set *isl_union_set_sample(__isl_take isl_union_set *uset)
{
	return bset_from_bmap(isl_union_map_sample(uset));
}

/* Return an element in "uset" in the form of an isl_point.
 * Return a void isl_point if "uset" is empty.
 */
__isl_give isl_point *isl_union_set_sample_point(__isl_take isl_union_set *uset)
{
	return isl_basic_set_sample_point(isl_union_set_sample(uset));
}

struct isl_forall_data {
	isl_bool res;
	isl_bool (*fn)(__isl_keep isl_map *map);
};

static isl_stat forall_entry(void **entry, void *user)
{
	struct isl_forall_data *data = user;
	isl_map *map = *entry;

	data->res = data->fn(map);
	if (data->res < 0)
		return isl_stat_error;

	if (!data->res)
		return isl_stat_error;

	return isl_stat_ok;
}

static isl_bool union_map_forall(__isl_keep isl_union_map *umap,
	isl_bool (*fn)(__isl_keep isl_map *map))
{
	struct isl_forall_data data = { isl_bool_true, fn };

	if (!umap)
		return isl_bool_error;

	if (isl_hash_table_foreach(umap->dim->ctx, &umap->table,
				   &forall_entry, &data) < 0 && data.res)
		return isl_bool_error;

	return data.res;
}

struct isl_forall_user_data {
	isl_bool res;
	isl_bool (*fn)(__isl_keep isl_map *map, void *user);
	void *user;
};

static isl_stat forall_user_entry(void **entry, void *user)
{
	struct isl_forall_user_data *data = user;
	isl_map *map = *entry;

	data->res = data->fn(map, data->user);
	if (data->res < 0)
		return isl_stat_error;

	if (!data->res)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Check if fn(map, user) returns true for all maps "map" in umap.
 */
static isl_bool union_map_forall_user(__isl_keep isl_union_map *umap,
	isl_bool (*fn)(__isl_keep isl_map *map, void *user), void *user)
{
	struct isl_forall_user_data data = { isl_bool_true, fn, user };

	if (!umap)
		return isl_bool_error;

	if (isl_hash_table_foreach(umap->dim->ctx, &umap->table,
				   &forall_user_entry, &data) < 0 && data.res)
		return isl_bool_error;

	return data.res;
}

/* Is "umap" obviously empty?
 */
isl_bool isl_union_map_plain_is_empty(__isl_keep isl_union_map *umap)
{
	isl_size n;

	n = isl_union_map_n_map(umap);
	if (n < 0)
		return isl_bool_error;
	return n == 0;
}

isl_bool isl_union_map_is_empty(__isl_keep isl_union_map *umap)
{
	return union_map_forall(umap, &isl_map_is_empty);
}

isl_bool isl_union_set_is_empty(__isl_keep isl_union_set *uset)
{
	return isl_union_map_is_empty(uset);
}

static isl_bool is_subset_of_identity(__isl_keep isl_map *map)
{
	isl_bool is_subset;
	isl_space *space;
	isl_map *id;
	isl_bool match;

	match = isl_map_tuple_is_equal(map, isl_dim_in, map, isl_dim_out);
	if (match < 0)
		return isl_bool_error;
	if (!match)
		return isl_bool_false;

	space = isl_map_get_space(map);
	id = isl_map_identity(space);

	is_subset = isl_map_is_subset(map, id);

	isl_map_free(id);

	return is_subset;
}

/* Given an isl_union_map that consists of a single map, check
 * if it is single-valued.
 */
static isl_bool single_map_is_single_valued(__isl_keep isl_union_map *umap)
{
	isl_map *map;
	isl_bool sv;

	umap = isl_union_map_copy(umap);
	map = isl_map_from_union_map(umap);
	sv = isl_map_is_single_valued(map);
	isl_map_free(map);

	return sv;
}

/* Internal data structure for single_valued_on_domain.
 *
 * "umap" is the union map to be tested.
 * "sv" is set to 1 as long as "umap" may still be single-valued.
 */
struct isl_union_map_is_sv_data {
	isl_union_map *umap;
	isl_bool sv;
};

/* Check if the data->umap is single-valued on "set".
 *
 * If data->umap consists of a single map on "set", then test it
 * as an isl_map.
 *
 * Otherwise, compute
 *
 *	M \circ M^-1
 *
 * check if the result is a subset of the identity mapping and
 * store the result in data->sv.
 *
 * Terminate as soon as data->umap has been determined not to
 * be single-valued.
 */
static isl_stat single_valued_on_domain(__isl_take isl_set *set, void *user)
{
	struct isl_union_map_is_sv_data *data = user;
	isl_union_map *umap, *test;
	isl_size n;

	umap = isl_union_map_copy(data->umap);
	umap = isl_union_map_intersect_domain(umap,
						isl_union_set_from_set(set));

	n = isl_union_map_n_map(umap);
	if (n < 0) {
		data->sv = isl_bool_error;
	} else if (n == 1) {
		data->sv = single_map_is_single_valued(umap);
		isl_union_map_free(umap);
	} else {
		test = isl_union_map_reverse(isl_union_map_copy(umap));
		test = isl_union_map_apply_range(test, umap);

		data->sv = union_map_forall(test, &is_subset_of_identity);

		isl_union_map_free(test);
	}

	if (data->sv < 0 || !data->sv)
		return isl_stat_error;
	return isl_stat_ok;
}

/* Check if the given map is single-valued.
 *
 * If the union map consists of a single map, then test it as an isl_map.
 * Otherwise, check if the union map is single-valued on each of its
 * domain spaces.
 */
isl_bool isl_union_map_is_single_valued(__isl_keep isl_union_map *umap)
{
	isl_union_map *universe;
	isl_union_set *domain;
	struct isl_union_map_is_sv_data data;
	isl_size n;

	n = isl_union_map_n_map(umap);
	if (n < 0)
		return isl_bool_error;
	if (n == 1)
		return single_map_is_single_valued(umap);

	universe = isl_union_map_universe(isl_union_map_copy(umap));
	domain = isl_union_map_domain(universe);

	data.sv = isl_bool_true;
	data.umap = umap;
	if (isl_union_set_foreach_set(domain,
			    &single_valued_on_domain, &data) < 0 && data.sv)
		data.sv = isl_bool_error;
	isl_union_set_free(domain);

	return data.sv;
}

isl_bool isl_union_map_is_injective(__isl_keep isl_union_map *umap)
{
	isl_bool in;

	umap = isl_union_map_copy(umap);
	umap = isl_union_map_reverse(umap);
	in = isl_union_map_is_single_valued(umap);
	isl_union_map_free(umap);

	return in;
}

/* Is "map" obviously not an identity relation because
 * it maps elements from one space to another space?
 * Update *non_identity accordingly.
 *
 * In particular, if the domain and range spaces are the same,
 * then the map is not considered to obviously not be an identity relation.
 * Otherwise, the map is considered to obviously not be an identity relation
 * if it is is non-empty.
 *
 * If "map" is determined to obviously not be an identity relation,
 * then the search is aborted.
 */
static isl_stat map_plain_is_not_identity(__isl_take isl_map *map, void *user)
{
	isl_bool *non_identity = user;
	isl_bool equal;

	equal = isl_map_tuple_is_equal(map, isl_dim_in, map, isl_dim_out);
	if (equal >= 0 && !equal)
		*non_identity = isl_bool_not(isl_map_is_empty(map));
	else
		*non_identity = isl_bool_not(equal);
	isl_map_free(map);

	if (*non_identity < 0 || *non_identity)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Is "umap" obviously not an identity relation because
 * it maps elements from one space to another space?
 *
 * As soon as a map has been found that maps elements to a different space,
 * non_identity is changed and the search is aborted.
 */
static isl_bool isl_union_map_plain_is_not_identity(
	__isl_keep isl_union_map *umap)
{
	isl_bool non_identity;

	non_identity = isl_bool_false;
	if (isl_union_map_foreach_map(umap, &map_plain_is_not_identity,
					&non_identity) < 0 &&
	    non_identity == isl_bool_false)
		return isl_bool_error;

	return non_identity;
}

/* Does "map" only map elements to themselves?
 * Update *identity accordingly.
 *
 * If "map" is determined not to be an identity relation,
 * then the search is aborted.
 */
static isl_stat map_is_identity(__isl_take isl_map *map, void *user)
{
	isl_bool *identity = user;

	*identity = isl_map_is_identity(map);
	isl_map_free(map);

	if (*identity < 0 || !*identity)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Does "umap" only map elements to themselves?
 *
 * First check if there are any maps that map elements to different spaces.
 * If not, then check that all the maps (between identical spaces)
 * are identity relations.
 */
isl_bool isl_union_map_is_identity(__isl_keep isl_union_map *umap)
{
	isl_bool non_identity;
	isl_bool identity;

	non_identity = isl_union_map_plain_is_not_identity(umap);
	if (non_identity < 0 || non_identity)
		return isl_bool_not(non_identity);

	identity = isl_bool_true;
	if (isl_union_map_foreach_map(umap, &map_is_identity, &identity) < 0 &&
	    identity == isl_bool_true)
		return isl_bool_error;

	return identity;
}

/* Represents a map that has a fixed value (v) for one of its
 * range dimensions.
 * The map in this structure is not reference counted, so it
 * is only valid while the isl_union_map from which it was
 * obtained is still alive.
 */
struct isl_fixed_map {
	isl_int v;
	isl_map *map;
};

static struct isl_fixed_map *alloc_isl_fixed_map_array(isl_ctx *ctx,
	int n)
{
	int i;
	struct isl_fixed_map *v;

	v = isl_calloc_array(ctx, struct isl_fixed_map, n);
	if (!v)
		return NULL;
	for (i = 0; i < n; ++i)
		isl_int_init(v[i].v);
	return v;
}

static void free_isl_fixed_map_array(struct isl_fixed_map *v, int n)
{
	int i;

	if (!v)
		return;
	for (i = 0; i < n; ++i)
		isl_int_clear(v[i].v);
	free(v);
}

/* Compare the "v" field of two isl_fixed_map structs.
 */
static int qsort_fixed_map_cmp(const void *p1, const void *p2)
{
	const struct isl_fixed_map *e1 = (const struct isl_fixed_map *) p1;
	const struct isl_fixed_map *e2 = (const struct isl_fixed_map *) p2;

	return isl_int_cmp(e1->v, e2->v);
}

/* Internal data structure used while checking whether all maps
 * in a union_map have a fixed value for a given output dimension.
 * v is the list of maps, with the fixed value for the dimension
 * n is the number of maps considered so far
 * pos is the output dimension under investigation
 */
struct isl_fixed_dim_data {
	struct isl_fixed_map *v;
	int n;
	int pos;
};

static isl_bool fixed_at_pos(__isl_keep isl_map *map, void *user)
{
	struct isl_fixed_dim_data *data = user;

	data->v[data->n].map = map;
	return isl_map_plain_is_fixed(map, isl_dim_out, data->pos,
				      &data->v[data->n++].v);
}

static isl_bool plain_injective_on_range(__isl_take isl_union_map *umap,
	int first, int n_range);

/* Given a list of the maps, with their fixed values at output dimension "pos",
 * check whether the ranges of the maps form an obvious partition.
 *
 * We first sort the maps according to their fixed values.
 * If all maps have a different value, then we know the ranges form
 * a partition.
 * Otherwise, we collect the maps with the same fixed value and
 * check whether each such collection is obviously injective
 * based on later dimensions.
 */
static int separates(struct isl_fixed_map *v, int n,
	__isl_take isl_space *space, int pos, int n_range)
{
	int i;

	if (!v)
		goto error;

	qsort(v, n, sizeof(*v), &qsort_fixed_map_cmp);

	for (i = 0; i + 1 < n; ++i) {
		int j, k;
		isl_union_map *part;
		int injective;

		for (j = i + 1; j < n; ++j)
			if (isl_int_ne(v[i].v, v[j].v))
				break;

		if (j == i + 1)
			continue;

		part = isl_union_map_alloc(isl_space_copy(space), j - i);
		for (k = i; k < j; ++k)
			part = isl_union_map_add_map(part,
						     isl_map_copy(v[k].map));

		injective = plain_injective_on_range(part, pos + 1, n_range);
		if (injective < 0)
			goto error;
		if (!injective)
			break;

		i = j - 1;
	}

	isl_space_free(space);
	free_isl_fixed_map_array(v, n);
	return i + 1 >= n;
error:
	isl_space_free(space);
	free_isl_fixed_map_array(v, n);
	return -1;
}

/* Check whether the maps in umap have obviously distinct ranges.
 * In particular, check for an output dimension in the range
 * [first,n_range) for which all maps have a fixed value
 * and then check if these values, possibly along with fixed values
 * at later dimensions, entail distinct ranges.
 */
static isl_bool plain_injective_on_range(__isl_take isl_union_map *umap,
	int first, int n_range)
{
	isl_ctx *ctx;
	isl_size n;
	struct isl_fixed_dim_data data = { NULL };

	ctx = isl_union_map_get_ctx(umap);

	n = isl_union_map_n_map(umap);
	if (n < 0)
		goto error;

	if (n <= 1) {
		isl_union_map_free(umap);
		return isl_bool_true;
	}

	if (first >= n_range) {
		isl_union_map_free(umap);
		return isl_bool_false;
	}

	data.v = alloc_isl_fixed_map_array(ctx, n);
	if (!data.v)
		goto error;

	for (data.pos = first; data.pos < n_range; ++data.pos) {
		isl_bool fixed;
		int injective;
		isl_space *space;

		data.n = 0;
		fixed = union_map_forall_user(umap, &fixed_at_pos, &data);
		if (fixed < 0)
			goto error;
		if (!fixed)
			continue;
		space = isl_union_map_get_space(umap);
		injective = separates(data.v, n, space, data.pos, n_range);
		isl_union_map_free(umap);
		return injective;
	}

	free_isl_fixed_map_array(data.v, n);
	isl_union_map_free(umap);

	return isl_bool_false;
error:
	free_isl_fixed_map_array(data.v, n);
	isl_union_map_free(umap);
	return isl_bool_error;
}

/* Check whether the maps in umap that map to subsets of "ran"
 * have obviously distinct ranges.
 */
static isl_bool plain_injective_on_range_wrap(__isl_keep isl_set *ran,
	void *user)
{
	isl_size dim;
	isl_union_map *umap = user;

	dim = isl_set_dim(ran, isl_dim_set);
	if (dim < 0)
		return isl_bool_error;

	umap = isl_union_map_copy(umap);
	umap = isl_union_map_intersect_range(umap,
			isl_union_set_from_set(isl_set_copy(ran)));
	return plain_injective_on_range(umap, 0, dim);
}

/* Check if the given union_map is obviously injective.
 *
 * In particular, we first check if all individual maps are obviously
 * injective and then check if all the ranges of these maps are
 * obviously disjoint.
 */
isl_bool isl_union_map_plain_is_injective(__isl_keep isl_union_map *umap)
{
	isl_bool in;
	isl_union_map *univ;
	isl_union_set *ran;

	in = union_map_forall(umap, &isl_map_plain_is_injective);
	if (in < 0)
		return isl_bool_error;
	if (!in)
		return isl_bool_false;

	univ = isl_union_map_universe(isl_union_map_copy(umap));
	ran = isl_union_map_range(univ);

	in = union_map_forall_user(ran, &plain_injective_on_range_wrap, umap);

	isl_union_set_free(ran);

	return in;
}

isl_bool isl_union_map_is_bijective(__isl_keep isl_union_map *umap)
{
	isl_bool sv;

	sv = isl_union_map_is_single_valued(umap);
	if (sv < 0 || !sv)
		return sv;

	return isl_union_map_is_injective(umap);
}

__isl_give isl_union_map *isl_union_map_zip(__isl_take isl_union_map *umap)
{
	struct isl_un_op_drop_user_data data = { &isl_map_can_zip };
	struct isl_un_op_control control = {
		.filter = &un_op_filter_drop_user,
		.filter_user = &data,
		.fn_map = &isl_map_zip,
	};
	return un_op(umap, &control);
}

/* Given a union map, take the maps of the form A -> (B -> C) and
 * return the union of the corresponding maps (A -> B) -> C.
 */
__isl_give isl_union_map *isl_union_map_uncurry(__isl_take isl_union_map *umap)
{
	struct isl_un_op_drop_user_data data = { &isl_map_can_uncurry };
	struct isl_un_op_control control = {
		.filter = &un_op_filter_drop_user,
		.filter_user = &data,
		.fn_map = &isl_map_uncurry,
	};
	return un_op(umap, &control);
}

/* Given a union map, take the maps of the form (A -> B) -> C and
 * return the union of the corresponding maps A -> (B -> C).
 */
__isl_give isl_union_map *isl_union_map_curry(__isl_take isl_union_map *umap)
{
	struct isl_un_op_drop_user_data data = { &isl_map_can_curry };
	struct isl_un_op_control control = {
		.filter = &un_op_filter_drop_user,
		.filter_user = &data,
		.fn_map = &isl_map_curry,
	};
	return un_op(umap, &control);
}

/* Given a union map, take the maps of the form A -> ((B -> C) -> D) and
 * return the union of the corresponding maps A -> (B -> (C -> D)).
 */
__isl_give isl_union_map *isl_union_map_range_curry(
	__isl_take isl_union_map *umap)
{
	struct isl_un_op_drop_user_data data = { &isl_map_can_range_curry };
	struct isl_un_op_control control = {
		.filter = &un_op_filter_drop_user,
		.filter_user = &data,
		.fn_map = &isl_map_range_curry,
	};
	return un_op(umap, &control);
}

__isl_give isl_union_set *isl_union_set_lift(__isl_take isl_union_set *uset)
{
	struct isl_un_op_control control = {
		.fn_map = &isl_set_lift,
	};
	return un_op(uset, &control);
}

static isl_stat coefficients_entry(void **entry, void *user)
{
	isl_set *set = *entry;
	isl_union_set **res = user;

	set = isl_set_copy(set);
	set = isl_set_from_basic_set(isl_set_coefficients(set));
	*res = isl_union_set_add_set(*res, set);

	return isl_stat_ok;
}

__isl_give isl_union_set *isl_union_set_coefficients(
	__isl_take isl_union_set *uset)
{
	isl_ctx *ctx;
	isl_space *space;
	isl_union_set *res;

	if (!uset)
		return NULL;

	ctx = isl_union_set_get_ctx(uset);
	space = isl_space_set_alloc(ctx, 0, 0);
	res = isl_union_map_alloc(space, uset->table.n);
	if (isl_hash_table_foreach(uset->dim->ctx, &uset->table,
				   &coefficients_entry, &res) < 0)
		goto error;

	isl_union_set_free(uset);
	return res;
error:
	isl_union_set_free(uset);
	isl_union_set_free(res);
	return NULL;
}

static isl_stat solutions_entry(void **entry, void *user)
{
	isl_set *set = *entry;
	isl_union_set **res = user;

	set = isl_set_copy(set);
	set = isl_set_from_basic_set(isl_set_solutions(set));
	if (!*res)
		*res = isl_union_set_from_set(set);
	else
		*res = isl_union_set_add_set(*res, set);

	if (!*res)
		return isl_stat_error;

	return isl_stat_ok;
}

__isl_give isl_union_set *isl_union_set_solutions(
	__isl_take isl_union_set *uset)
{
	isl_union_set *res = NULL;

	if (!uset)
		return NULL;

	if (uset->table.n == 0) {
		res = isl_union_set_empty(isl_union_set_get_space(uset));
		isl_union_set_free(uset);
		return res;
	}

	if (isl_hash_table_foreach(uset->dim->ctx, &uset->table,
				   &solutions_entry, &res) < 0)
		goto error;

	isl_union_set_free(uset);
	return res;
error:
	isl_union_set_free(uset);
	isl_union_set_free(res);
	return NULL;
}

/* Is the domain space of "map" equal to "space"?
 */
static int domain_match(__isl_keep isl_map *map, __isl_keep isl_space *space)
{
	return isl_map_space_tuple_is_equal(map, isl_dim_in,
					space, isl_dim_out);
}

/* Is the range space of "map" equal to "space"?
 */
static int range_match(__isl_keep isl_map *map, __isl_keep isl_space *space)
{
	return isl_map_space_tuple_is_equal(map, isl_dim_out,
					space, isl_dim_out);
}

/* Is the set space of "map" equal to "space"?
 */
static int set_match(__isl_keep isl_map *map, __isl_keep isl_space *space)
{
	return isl_map_space_tuple_is_equal(map, isl_dim_set,
					space, isl_dim_out);
}

/* Internal data structure for preimage_pw_multi_aff.
 *
 * "pma" is the function under which the preimage should be taken.
 * "space" is the space of "pma".
 * "res" collects the results.
 * "fn" computes the preimage for a given map.
 * "match" returns true if "fn" can be called.
 */
struct isl_union_map_preimage_data {
	isl_space *space;
	isl_pw_multi_aff *pma;
	isl_union_map *res;
	int (*match)(__isl_keep isl_map *map, __isl_keep isl_space *space);
	__isl_give isl_map *(*fn)(__isl_take isl_map *map,
		__isl_take isl_pw_multi_aff *pma);
};

/* Call data->fn to compute the preimage of the domain or range of *entry
 * under the function represented by data->pma, provided the domain/range
 * space of *entry matches the target space of data->pma
 * (as given by data->match), and add the result to data->res.
 */
static isl_stat preimage_entry(void **entry, void *user)
{
	int m;
	isl_map *map = *entry;
	struct isl_union_map_preimage_data *data = user;
	isl_bool empty;

	m = data->match(map, data->space);
	if (m < 0)
		return isl_stat_error;
	if (!m)
		return isl_stat_ok;

	map = isl_map_copy(map);
	map = data->fn(map, isl_pw_multi_aff_copy(data->pma));

	empty = isl_map_is_empty(map);
	if (empty < 0 || empty) {
		isl_map_free(map);
		return empty < 0 ? isl_stat_error : isl_stat_ok;
	}

	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

/* Compute the preimage of the domain or range of "umap" under the function
 * represented by "pma".
 * In other words, plug in "pma" in the domain or range of "umap".
 * The function "fn" performs the actual preimage computation on a map,
 * while "match" determines to which maps the function should be applied.
 */
static __isl_give isl_union_map *preimage_pw_multi_aff(
	__isl_take isl_union_map *umap, __isl_take isl_pw_multi_aff *pma,
	int (*match)(__isl_keep isl_map *map, __isl_keep isl_space *space),
	__isl_give isl_map *(*fn)(__isl_take isl_map *map,
		__isl_take isl_pw_multi_aff *pma))
{
	isl_ctx *ctx;
	isl_space *space;
	struct isl_union_map_preimage_data data;

	umap = isl_union_map_align_params(umap,
					    isl_pw_multi_aff_get_space(pma));
	pma = isl_pw_multi_aff_align_params(pma, isl_union_map_get_space(umap));

	if (!umap || !pma)
		goto error;

	ctx = isl_union_map_get_ctx(umap);
	space = isl_union_map_get_space(umap);
	data.space = isl_pw_multi_aff_get_space(pma);
	data.pma = pma;
	data.res = isl_union_map_alloc(space, umap->table.n);
	data.match = match;
	data.fn = fn;
	if (isl_hash_table_foreach(ctx, &umap->table, &preimage_entry,
					&data) < 0)
		data.res = isl_union_map_free(data.res);

	isl_space_free(data.space);
	isl_union_map_free(umap);
	isl_pw_multi_aff_free(pma);
	return data.res;
error:
	isl_union_map_free(umap);
	isl_pw_multi_aff_free(pma);
	return NULL;
}

/* Compute the preimage of the domain of "umap" under the function
 * represented by "pma".
 * In other words, plug in "pma" in the domain of "umap".
 * The result contains maps that live in the same spaces as the maps of "umap"
 * with domain space equal to the target space of "pma",
 * except that the domain has been replaced by the domain space of "pma".
 */
__isl_give isl_union_map *isl_union_map_preimage_domain_pw_multi_aff(
	__isl_take isl_union_map *umap, __isl_take isl_pw_multi_aff *pma)
{
	return preimage_pw_multi_aff(umap, pma, &domain_match,
					&isl_map_preimage_domain_pw_multi_aff);
}

/* Compute the preimage of the range of "umap" under the function
 * represented by "pma".
 * In other words, plug in "pma" in the range of "umap".
 * The result contains maps that live in the same spaces as the maps of "umap"
 * with range space equal to the target space of "pma",
 * except that the range has been replaced by the domain space of "pma".
 */
__isl_give isl_union_map *isl_union_map_preimage_range_pw_multi_aff(
	__isl_take isl_union_map *umap, __isl_take isl_pw_multi_aff *pma)
{
	return preimage_pw_multi_aff(umap, pma, &range_match,
					&isl_map_preimage_range_pw_multi_aff);
}

/* Compute the preimage of "uset" under the function represented by "pma".
 * In other words, plug in "pma" in "uset".
 * The result contains sets that live in the same spaces as the sets of "uset"
 * with space equal to the target space of "pma",
 * except that the space has been replaced by the domain space of "pma".
 */
__isl_give isl_union_set *isl_union_set_preimage_pw_multi_aff(
	__isl_take isl_union_set *uset, __isl_take isl_pw_multi_aff *pma)
{
	return preimage_pw_multi_aff(uset, pma, &set_match,
					&isl_set_preimage_pw_multi_aff);
}

/* Compute the preimage of the domain of "umap" under the function
 * represented by "ma".
 * In other words, plug in "ma" in the domain of "umap".
 * The result contains maps that live in the same spaces as the maps of "umap"
 * with domain space equal to the target space of "ma",
 * except that the domain has been replaced by the domain space of "ma".
 */
__isl_give isl_union_map *isl_union_map_preimage_domain_multi_aff(
	__isl_take isl_union_map *umap, __isl_take isl_multi_aff *ma)
{
	return isl_union_map_preimage_domain_pw_multi_aff(umap,
					isl_pw_multi_aff_from_multi_aff(ma));
}

/* Compute the preimage of the range of "umap" under the function
 * represented by "ma".
 * In other words, plug in "ma" in the range of "umap".
 * The result contains maps that live in the same spaces as the maps of "umap"
 * with range space equal to the target space of "ma",
 * except that the range has been replaced by the domain space of "ma".
 */
__isl_give isl_union_map *isl_union_map_preimage_range_multi_aff(
	__isl_take isl_union_map *umap, __isl_take isl_multi_aff *ma)
{
	return isl_union_map_preimage_range_pw_multi_aff(umap,
					isl_pw_multi_aff_from_multi_aff(ma));
}

/* Compute the preimage of "uset" under the function represented by "ma".
 * In other words, plug in "ma" in "uset".
 * The result contains sets that live in the same spaces as the sets of "uset"
 * with space equal to the target space of "ma",
 * except that the space has been replaced by the domain space of "ma".
 */
__isl_give isl_union_map *isl_union_set_preimage_multi_aff(
	__isl_take isl_union_set *uset, __isl_take isl_multi_aff *ma)
{
	return isl_union_set_preimage_pw_multi_aff(uset,
					isl_pw_multi_aff_from_multi_aff(ma));
}

/* Internal data structure for preimage_multi_pw_aff.
 *
 * "mpa" is the function under which the preimage should be taken.
 * "space" is the space of "mpa".
 * "res" collects the results.
 * "fn" computes the preimage for a given map.
 * "match" returns true if "fn" can be called.
 */
struct isl_union_map_preimage_mpa_data {
	isl_space *space;
	isl_multi_pw_aff *mpa;
	isl_union_map *res;
	int (*match)(__isl_keep isl_map *map, __isl_keep isl_space *space);
	__isl_give isl_map *(*fn)(__isl_take isl_map *map,
		__isl_take isl_multi_pw_aff *mpa);
};

/* Call data->fn to compute the preimage of the domain or range of *entry
 * under the function represented by data->mpa, provided the domain/range
 * space of *entry matches the target space of data->mpa
 * (as given by data->match), and add the result to data->res.
 */
static isl_stat preimage_mpa_entry(void **entry, void *user)
{
	int m;
	isl_map *map = *entry;
	struct isl_union_map_preimage_mpa_data *data = user;
	isl_bool empty;

	m = data->match(map, data->space);
	if (m < 0)
		return isl_stat_error;
	if (!m)
		return isl_stat_ok;

	map = isl_map_copy(map);
	map = data->fn(map, isl_multi_pw_aff_copy(data->mpa));

	empty = isl_map_is_empty(map);
	if (empty < 0 || empty) {
		isl_map_free(map);
		return empty < 0 ? isl_stat_error : isl_stat_ok;
	}

	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

/* Compute the preimage of the domain or range of "umap" under the function
 * represented by "mpa".
 * In other words, plug in "mpa" in the domain or range of "umap".
 * The function "fn" performs the actual preimage computation on a map,
 * while "match" determines to which maps the function should be applied.
 */
static __isl_give isl_union_map *preimage_multi_pw_aff(
	__isl_take isl_union_map *umap, __isl_take isl_multi_pw_aff *mpa,
	int (*match)(__isl_keep isl_map *map, __isl_keep isl_space *space),
	__isl_give isl_map *(*fn)(__isl_take isl_map *map,
		__isl_take isl_multi_pw_aff *mpa))
{
	isl_ctx *ctx;
	isl_space *space;
	struct isl_union_map_preimage_mpa_data data;

	umap = isl_union_map_align_params(umap,
					    isl_multi_pw_aff_get_space(mpa));
	mpa = isl_multi_pw_aff_align_params(mpa, isl_union_map_get_space(umap));

	if (!umap || !mpa)
		goto error;

	ctx = isl_union_map_get_ctx(umap);
	space = isl_union_map_get_space(umap);
	data.space = isl_multi_pw_aff_get_space(mpa);
	data.mpa = mpa;
	data.res = isl_union_map_alloc(space, umap->table.n);
	data.match = match;
	data.fn = fn;
	if (isl_hash_table_foreach(ctx, &umap->table, &preimage_mpa_entry,
					&data) < 0)
		data.res = isl_union_map_free(data.res);

	isl_space_free(data.space);
	isl_union_map_free(umap);
	isl_multi_pw_aff_free(mpa);
	return data.res;
error:
	isl_union_map_free(umap);
	isl_multi_pw_aff_free(mpa);
	return NULL;
}

/* Compute the preimage of the domain of "umap" under the function
 * represented by "mpa".
 * In other words, plug in "mpa" in the domain of "umap".
 * The result contains maps that live in the same spaces as the maps of "umap"
 * with domain space equal to the target space of "mpa",
 * except that the domain has been replaced by the domain space of "mpa".
 */
__isl_give isl_union_map *isl_union_map_preimage_domain_multi_pw_aff(
	__isl_take isl_union_map *umap, __isl_take isl_multi_pw_aff *mpa)
{
	return preimage_multi_pw_aff(umap, mpa, &domain_match,
					&isl_map_preimage_domain_multi_pw_aff);
}

/* Internal data structure for preimage_upma.
 *
 * "umap" is the map of which the preimage should be computed.
 * "res" collects the results.
 * "fn" computes the preimage for a given piecewise multi-affine function.
 */
struct isl_union_map_preimage_upma_data {
	isl_union_map *umap;
	isl_union_map *res;
	__isl_give isl_union_map *(*fn)(__isl_take isl_union_map *umap,
		__isl_take isl_pw_multi_aff *pma);
};

/* Call data->fn to compute the preimage of the domain or range of data->umap
 * under the function represented by pma and add the result to data->res.
 */
static isl_stat preimage_upma(__isl_take isl_pw_multi_aff *pma, void *user)
{
	struct isl_union_map_preimage_upma_data *data = user;
	isl_union_map *umap;

	umap = isl_union_map_copy(data->umap);
	umap = data->fn(umap, pma);
	data->res = isl_union_map_union(data->res, umap);

	return data->res ? isl_stat_ok : isl_stat_error;
}

/* Compute the preimage of the domain or range of "umap" under the function
 * represented by "upma".
 * In other words, plug in "upma" in the domain or range of "umap".
 * The function "fn" performs the actual preimage computation
 * on a piecewise multi-affine function.
 */
static __isl_give isl_union_map *preimage_union_pw_multi_aff(
	__isl_take isl_union_map *umap,
	__isl_take isl_union_pw_multi_aff *upma,
	__isl_give isl_union_map *(*fn)(__isl_take isl_union_map *umap,
		__isl_take isl_pw_multi_aff *pma))
{
	struct isl_union_map_preimage_upma_data data;

	data.umap = umap;
	data.res = isl_union_map_empty(isl_union_map_get_space(umap));
	data.fn = fn;
	if (isl_union_pw_multi_aff_foreach_pw_multi_aff(upma,
						    &preimage_upma, &data) < 0)
		data.res = isl_union_map_free(data.res);

	isl_union_map_free(umap);
	isl_union_pw_multi_aff_free(upma);

	return data.res;
}

/* Compute the preimage of the domain of "umap" under the function
 * represented by "upma".
 * In other words, plug in "upma" in the domain of "umap".
 * The result contains maps that live in the same spaces as the maps of "umap"
 * with domain space equal to one of the target spaces of "upma",
 * except that the domain has been replaced by one of the domain spaces that
 * correspond to that target space of "upma".
 */
__isl_give isl_union_map *isl_union_map_preimage_domain_union_pw_multi_aff(
	__isl_take isl_union_map *umap,
	__isl_take isl_union_pw_multi_aff *upma)
{
	return preimage_union_pw_multi_aff(umap, upma,
				&isl_union_map_preimage_domain_pw_multi_aff);
}

/* Compute the preimage of the range of "umap" under the function
 * represented by "upma".
 * In other words, plug in "upma" in the range of "umap".
 * The result contains maps that live in the same spaces as the maps of "umap"
 * with range space equal to one of the target spaces of "upma",
 * except that the range has been replaced by one of the domain spaces that
 * correspond to that target space of "upma".
 */
__isl_give isl_union_map *isl_union_map_preimage_range_union_pw_multi_aff(
	__isl_take isl_union_map *umap,
	__isl_take isl_union_pw_multi_aff *upma)
{
	return preimage_union_pw_multi_aff(umap, upma,
				&isl_union_map_preimage_range_pw_multi_aff);
}

/* Compute the preimage of "uset" under the function represented by "upma".
 * In other words, plug in "upma" in the range of "uset".
 * The result contains sets that live in the same spaces as the sets of "uset"
 * with space equal to one of the target spaces of "upma",
 * except that the space has been replaced by one of the domain spaces that
 * correspond to that target space of "upma".
 */
__isl_give isl_union_set *isl_union_set_preimage_union_pw_multi_aff(
	__isl_take isl_union_set *uset,
	__isl_take isl_union_pw_multi_aff *upma)
{
	return preimage_union_pw_multi_aff(uset, upma,
					&isl_union_set_preimage_pw_multi_aff);
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * of the spaces of "umap".
 */
__isl_give isl_union_map *isl_union_map_reset_user(
	__isl_take isl_union_map *umap)
{
	umap = isl_union_map_cow(umap);
	if (!umap)
		return NULL;
	umap->dim = isl_space_reset_user(umap->dim);
	if (!umap->dim)
		return isl_union_map_free(umap);
	return total(umap, &isl_map_reset_user);
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * of the spaces of "uset".
 */
__isl_give isl_union_set *isl_union_set_reset_user(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_reset_user(uset);
}

/* Remove all existentially quantified variables and integer divisions
 * from "umap" using Fourier-Motzkin elimination.
 */
__isl_give isl_union_map *isl_union_map_remove_divs(
	__isl_take isl_union_map *umap)
{
	return total(umap, &isl_map_remove_divs);
}

/* Remove all existentially quantified variables and integer divisions
 * from "uset" using Fourier-Motzkin elimination.
 */
__isl_give isl_union_set *isl_union_set_remove_divs(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_remove_divs(uset);
}

/* Internal data structure for isl_union_map_project_out.
 * "type", "first" and "n" are the arguments for the isl_map_project_out
 * call.
 * "res" collects the results.
 */
struct isl_union_map_project_out_data {
	enum isl_dim_type type;
	unsigned first;
	unsigned n;

	isl_union_map *res;
};

/* Turn the data->n dimensions of type data->type, starting at data->first
 * into existentially quantified variables and add the result to data->res.
 */
static isl_stat project_out(__isl_take isl_map *map, void *user)
{
	struct isl_union_map_project_out_data *data = user;

	map = isl_map_project_out(map, data->type, data->first, data->n);
	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

/* Turn the "n" dimensions of type "type", starting at "first"
 * into existentially quantified variables.
 * Since the space of an isl_union_map only contains parameters,
 * type is required to be equal to isl_dim_param.
 */
__isl_give isl_union_map *isl_union_map_project_out(
	__isl_take isl_union_map *umap,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	isl_space *space;
	struct isl_union_map_project_out_data data = { type, first, n };

	if (!umap)
		return NULL;

	if (type != isl_dim_param)
		isl_die(isl_union_map_get_ctx(umap), isl_error_invalid,
			"can only project out parameters",
			return isl_union_map_free(umap));

	space = isl_union_map_get_space(umap);
	space = isl_space_drop_dims(space, type, first, n);
	data.res = isl_union_map_empty(space);
	if (isl_union_map_foreach_map(umap, &project_out, &data) < 0)
		data.res = isl_union_map_free(data.res);

	isl_union_map_free(umap);

	return data.res;
}

#undef TYPE
#define TYPE	isl_union_map
#include "isl_project_out_all_params_templ.c"

/* Turn the "n" dimensions of type "type", starting at "first"
 * into existentially quantified variables.
 * Since the space of an isl_union_set only contains parameters,
 * "type" is required to be equal to isl_dim_param.
 */
__isl_give isl_union_set *isl_union_set_project_out(
	__isl_take isl_union_set *uset,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return isl_union_map_project_out(uset, type, first, n);
}

/* Project out all parameters from "uset" by existentially quantifying
 * over them.
 */
__isl_give isl_union_set *isl_union_set_project_out_all_params(
	__isl_take isl_union_set *uset)
{
	return uset_from_umap(
		    isl_union_map_project_out_all_params(uset_to_umap(uset)));
}

/* Internal data structure for isl_union_map_involves_dims.
 * "first" and "n" are the arguments for the isl_map_involves_dims calls.
 */
struct isl_union_map_involves_dims_data {
	unsigned first;
	unsigned n;
};

/* Does "map" _not_ involve the data->n parameters starting at data->first?
 */
static isl_bool map_excludes(__isl_keep isl_map *map, void *user)
{
	struct isl_union_map_involves_dims_data *data = user;
	isl_bool involves;

	involves = isl_map_involves_dims(map,
					isl_dim_param, data->first, data->n);
	return isl_bool_not(involves);
}

/* Does "umap" involve any of the n parameters starting at first?
 * "type" is required to be set to isl_dim_param.
 *
 * "umap" involves any of those parameters if any of its maps
 * involve the parameters.  In other words, "umap" does not
 * involve any of the parameters if all its maps to not
 * involve the parameters.
 */
isl_bool isl_union_map_involves_dims(__isl_keep isl_union_map *umap,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	struct isl_union_map_involves_dims_data data = { first, n };
	isl_bool excludes;

	if (type != isl_dim_param)
		isl_die(isl_union_map_get_ctx(umap), isl_error_invalid,
			"can only reference parameters", return isl_bool_error);

	excludes = union_map_forall_user(umap, &map_excludes, &data);

	return isl_bool_not(excludes);
}

/* Internal data structure for isl_union_map_reset_range_space.
 * "range" is the space from which to set the range space.
 * "res" collects the results.
 */
struct isl_union_map_reset_range_space_data {
	isl_space *range;
	isl_union_map *res;
};

/* Replace the range space of "map" by the range space of data->range and
 * add the result to data->res.
 */
static isl_stat reset_range_space(__isl_take isl_map *map, void *user)
{
	struct isl_union_map_reset_range_space_data *data = user;
	isl_space *space;

	space = isl_map_get_space(map);
	space = isl_space_domain(space);
	space = isl_space_extend_domain_with_range(space,
						isl_space_copy(data->range));
	map = isl_map_reset_space(map, space);
	data->res = isl_union_map_add_map(data->res, map);

	return data->res ? isl_stat_ok : isl_stat_error;
}

/* Replace the range space of all the maps in "umap" by
 * the range space of "space".
 *
 * This assumes that all maps have the same output dimension.
 * This function should therefore not be made publicly available.
 *
 * Since the spaces of the maps change, so do their hash value.
 * We therefore need to create a new isl_union_map.
 */
__isl_give isl_union_map *isl_union_map_reset_range_space(
	__isl_take isl_union_map *umap, __isl_take isl_space *space)
{
	struct isl_union_map_reset_range_space_data data = { space };

	data.res = isl_union_map_empty(isl_union_map_get_space(umap));
	if (isl_union_map_foreach_map(umap, &reset_range_space, &data) < 0)
		data.res = isl_union_map_free(data.res);

	isl_space_free(space);
	isl_union_map_free(umap);
	return data.res;
}

/* Check that "umap" and "space" have the same number of parameters.
 */
static isl_stat check_union_map_space_equal_dim(__isl_keep isl_union_map *umap,
	__isl_keep isl_space *space)
{
	isl_size dim1, dim2;

	dim1 = isl_union_map_dim(umap, isl_dim_param);
	dim2 = isl_space_dim(space, isl_dim_param);
	if (dim1 < 0 || dim2 < 0)
		return isl_stat_error;
	if (dim1 == dim2)
		return isl_stat_ok;
	isl_die(isl_union_map_get_ctx(umap), isl_error_invalid,
		"number of parameters does not match", return isl_stat_error);
}

/* Internal data structure for isl_union_map_reset_equal_dim_space.
 * "space" is the target space.
 * "res" collects the results.
 */
struct isl_union_map_reset_params_data {
	isl_space *space;
	isl_union_map *res;
};

/* Replace the parameters of "map" by those of data->space and
 * add the result to data->res.
 */
static isl_stat reset_params(__isl_take isl_map *map, void *user)
{
	struct isl_union_map_reset_params_data *data = user;
	isl_space *space;

	space = isl_map_get_space(map);
	space = isl_space_replace_params(space, data->space);
	map = isl_map_reset_equal_dim_space(map, space);
	data->res = isl_union_map_add_map(data->res, map);

	return data->res ? isl_stat_ok : isl_stat_error;
}

/* Replace the space of "umap" by "space", without modifying
 * the dimension of "umap", i.e., the number of parameters of "umap".
 *
 * Since the hash values of the maps in the union map depend
 * on the parameters, a new union map needs to be constructed.
 */
__isl_give isl_union_map *isl_union_map_reset_equal_dim_space(
	__isl_take isl_union_map *umap, __isl_take isl_space *space)
{
	struct isl_union_map_reset_params_data data = { space };
	isl_bool equal;
	isl_space *umap_space;

	umap_space = isl_union_map_peek_space(umap);
	equal = isl_space_is_equal(umap_space, space);
	if (equal < 0)
		goto error;
	if (equal) {
		isl_space_free(space);
		return umap;
	}
	if (check_union_map_space_equal_dim(umap, space) < 0)
		goto error;

	data.res = isl_union_map_empty(isl_space_copy(space));
	if (isl_union_map_foreach_map(umap, &reset_params, &data) < 0)
		data.res = isl_union_map_free(data.res);

	isl_space_free(space);
	isl_union_map_free(umap);
	return data.res;
error:
	isl_union_map_free(umap);
	isl_space_free(space);
	return NULL;
}

/* Internal data structure for isl_union_map_order_at_multi_union_pw_aff.
 * "mupa" is the function from which the isl_multi_pw_affs are extracted.
 * "order" is applied to the extracted isl_multi_pw_affs that correspond
 * to the domain and the range of each map.
 * "res" collects the results.
 */
struct isl_union_order_at_data {
	isl_multi_union_pw_aff *mupa;
	__isl_give isl_map *(*order)(__isl_take isl_multi_pw_aff *mpa1,
		__isl_take isl_multi_pw_aff *mpa2);
	isl_union_map *res;
};

/* Intersect "map" with the result of applying data->order to
 * the functions in data->mupa that apply to the domain and the range
 * of "map" and add the result to data->res.
 */
static isl_stat order_at(__isl_take isl_map *map, void *user)
{
	struct isl_union_order_at_data *data = user;
	isl_space *space;
	isl_multi_pw_aff *mpa1, *mpa2;
	isl_map *order;

	space = isl_space_domain(isl_map_get_space(map));
	mpa1 = isl_multi_union_pw_aff_extract_multi_pw_aff(data->mupa, space);
	space = isl_space_range(isl_map_get_space(map));
	mpa2 = isl_multi_union_pw_aff_extract_multi_pw_aff(data->mupa, space);
	order = data->order(mpa1, mpa2);
	map = isl_map_intersect(map, order);
	data->res = isl_union_map_add_map(data->res, map);

	return data->res ? isl_stat_ok : isl_stat_error;
}

/* If "mupa" has a non-trivial explicit domain, then intersect
 * domain and range of "umap" with this explicit domain.
 * If the explicit domain only describes constraints on the parameters,
 * then the intersection only needs to be performed once.
 */
static __isl_give isl_union_map *intersect_explicit_domain(
	__isl_take isl_union_map *umap, __isl_keep isl_multi_union_pw_aff *mupa)
{
	isl_bool non_trivial, is_params;
	isl_union_set *dom;

	non_trivial = isl_multi_union_pw_aff_has_non_trivial_domain(mupa);
	if (non_trivial < 0)
		return isl_union_map_free(umap);
	if (!non_trivial)
		return umap;
	mupa = isl_multi_union_pw_aff_copy(mupa);
	dom = isl_multi_union_pw_aff_domain(mupa);
	is_params = isl_union_set_is_params(dom);
	if (is_params < 0) {
		isl_union_set_free(dom);
		return isl_union_map_free(umap);
	}
	if (is_params) {
		isl_set *set;

		set = isl_union_set_params(dom);
		umap = isl_union_map_intersect_params(umap, set);
		return umap;
	}
	umap = isl_union_map_intersect_domain(umap, isl_union_set_copy(dom));
	umap = isl_union_map_intersect_range(umap, dom);
	return umap;
}

/* Intersect each map in "umap" with the result of calling "order"
 * on the functions is "mupa" that apply to the domain and the range
 * of the map.
 */
static __isl_give isl_union_map *isl_union_map_order_at_multi_union_pw_aff(
	__isl_take isl_union_map *umap, __isl_take isl_multi_union_pw_aff *mupa,
	__isl_give isl_map *(*order)(__isl_take isl_multi_pw_aff *mpa1,
		__isl_take isl_multi_pw_aff *mpa2))
{
	struct isl_union_order_at_data data;

	umap = isl_union_map_align_params(umap,
				isl_multi_union_pw_aff_get_space(mupa));
	mupa = isl_multi_union_pw_aff_align_params(mupa,
				isl_union_map_get_space(umap));
	umap = intersect_explicit_domain(umap, mupa);
	data.mupa = mupa;
	data.order = order;
	data.res = isl_union_map_empty(isl_union_map_get_space(umap));
	if (isl_union_map_foreach_map(umap, &order_at, &data) < 0)
		data.res = isl_union_map_free(data.res);

	isl_multi_union_pw_aff_free(mupa);
	isl_union_map_free(umap);
	return data.res;
}

/* Return the subset of "umap" where the domain and the range
 * have equal "mupa" values.
 */
__isl_give isl_union_map *isl_union_map_eq_at_multi_union_pw_aff(
	__isl_take isl_union_map *umap,
	__isl_take isl_multi_union_pw_aff *mupa)
{
	return isl_union_map_order_at_multi_union_pw_aff(umap, mupa,
						&isl_multi_pw_aff_eq_map);
}

#undef ORDER
#define ORDER		le
#include "isl_union_map_lex_templ.c"

#undef ORDER
#define ORDER		lt
#include "isl_union_map_lex_templ.c"

#undef ORDER
#define ORDER		ge
#include "isl_union_map_lex_templ.c"

#undef ORDER
#define ORDER		gt
#include "isl_union_map_lex_templ.c"

/* Return the union of the elements in the list "list".
 */
__isl_give isl_union_set *isl_union_set_list_union(
	__isl_take isl_union_set_list *list)
{
	int i;
	isl_size n;
	isl_ctx *ctx;
	isl_space *space;
	isl_union_set *res;

	n = isl_union_set_list_n_union_set(list);
	if (n < 0)
		goto error;

	ctx = isl_union_set_list_get_ctx(list);
	space = isl_space_params_alloc(ctx, 0);
	res = isl_union_set_empty(space);

	for (i = 0; i < n; ++i) {
		isl_union_set *uset_i;

		uset_i = isl_union_set_list_get_union_set(list, i);
		res = isl_union_set_union(res, uset_i);
	}

	isl_union_set_list_free(list);
	return res;
error:
	isl_union_set_list_free(list);
	return NULL;
}

/* Update *hash with the hash value of "map".
 */
static isl_stat add_hash(__isl_take isl_map *map, void *user)
{
	uint32_t *hash = user;
	uint32_t map_hash;

	map_hash = isl_map_get_hash(map);
	isl_hash_hash(*hash, map_hash);

	isl_map_free(map);
	return isl_stat_ok;
}

/* Return a hash value that digests "umap".
 */
uint32_t isl_union_map_get_hash(__isl_keep isl_union_map *umap)
{
	uint32_t hash;

	if (!umap)
		return 0;

	hash = isl_hash_init();
	if (isl_union_map_foreach_map(umap, &add_hash, &hash) < 0)
		return 0;

	return hash;
}

/* Return a hash value that digests "uset".
 */
uint32_t isl_union_set_get_hash(__isl_keep isl_union_set *uset)
{
	return isl_union_map_get_hash(uset);
}

/* Add the number of basic sets in "set" to "n".
 */
static isl_stat add_n(__isl_take isl_set *set, void *user)
{
	int *n = user;
	isl_size set_n;

	set_n = isl_set_n_basic_set(set);
	*n += set_n;
	isl_set_free(set);

	return set_n < 0 ? isl_stat_error : isl_stat_ok;
}

/* Return the total number of basic sets in "uset".
 */
int isl_union_set_n_basic_set(__isl_keep isl_union_set *uset)
{
	int n = 0;

	if (isl_union_set_foreach_set(uset, &add_n, &n) < 0)
		return -1;

	return n;
}

/* Add the basic sets in "set" to "list".
 */
static isl_stat add_list(__isl_take isl_set *set, void *user)
{
	isl_basic_set_list **list = user;
	isl_basic_set_list *list_i;

	list_i = isl_set_get_basic_set_list(set);
	*list = isl_basic_set_list_concat(*list, list_i);
	isl_set_free(set);

	if (!*list)
		return isl_stat_error;
	return isl_stat_ok;
}

/* Return a list containing all the basic sets in "uset".
 *
 * First construct a list of the appropriate size and
 * then add all the elements.
 */
__isl_give isl_basic_set_list *isl_union_set_get_basic_set_list(
	__isl_keep isl_union_set *uset)
{
	int n;
	isl_ctx *ctx;
	isl_basic_set_list *list;

	if (!uset)
		return NULL;
	ctx = isl_union_set_get_ctx(uset);
	n = isl_union_set_n_basic_set(uset);
	if (n < 0)
		return NULL;
	list = isl_basic_set_list_alloc(ctx, n);
	if (isl_union_set_foreach_set(uset, &add_list, &list) < 0)
		list = isl_basic_set_list_free(list);

	return list;
}

/* Internal data structure for isl_union_map_remove_map_if.
 * "fn" and "user" are the arguments to isl_union_map_remove_map_if.
 */
struct isl_union_map_remove_map_if_data {
	isl_bool (*fn)(__isl_keep isl_map *map, void *user);
	void *user;
};

/* isl_un_op_control filter that negates the result of data->fn
 * called on "map".
 */
static isl_bool not(__isl_keep isl_map *map, void *user)
{
	struct isl_union_map_remove_map_if_data *data = user;

	return isl_bool_not(data->fn(map, data->user));
}

/* Dummy isl_un_op_control transformation callback that
 * simply returns the input.
 */
static __isl_give isl_map *map_id(__isl_take isl_map *map)
{
	return map;
}

/* Call "fn" on every map in "umap" and remove those maps
 * for which the callback returns true.
 *
 * Use un_op to keep only those maps that are not filtered out,
 * applying an identity transformation on them.
 */
__isl_give isl_union_map *isl_union_map_remove_map_if(
	__isl_take isl_union_map *umap,
	isl_bool (*fn)(__isl_keep isl_map *map, void *user), void *user)
{
	struct isl_union_map_remove_map_if_data data = { fn, user };
	struct isl_un_op_control control = {
		.filter = &not,
		.filter_user = &data,
		.fn_map = &map_id,
	};
	return un_op(umap, &control);
}

/* Does "map" have "space" as domain (ignoring parameters)?
 */
static isl_bool has_domain_space_tuples(__isl_keep isl_map *map, void *user)
{
	isl_space *space = user;

	return isl_space_has_domain_tuples(space, isl_map_peek_space(map));
}

/* Does "map" have "space" as range (ignoring parameters)?
 */
static isl_bool has_range_space_tuples(__isl_keep isl_map *map, void *user)
{
	isl_space *space = user;

	return isl_space_has_range_tuples(space, isl_map_peek_space(map));
}

/* Wrapper around isl_map_bind_range for use as a un_op callback.
 */
static __isl_give isl_map *bind_range(__isl_take isl_map *map, void *user)
{
	isl_multi_id *tuple = user;

	return isl_map_bind_range(map, isl_multi_id_copy(tuple));
}

/* Bind the output dimensions of "umap" to parameters with identifiers
 * specified by "tuple", living in the range space of "umap",
 * for those maps that have a matching range space.
 */
__isl_give isl_union_set *isl_union_map_bind_range(
	__isl_take isl_union_map *umap, __isl_take isl_multi_id *tuple)
{
	struct isl_un_op_control control = {
		.filter = &has_range_space_tuples,
		.filter_user = isl_multi_id_peek_space(tuple),
		.fn_map2 = &bind_range,
		.fn_map2_user = tuple,
	};
	isl_union_set *bound;

	bound = uset_from_umap(un_op(umap, &control));
	isl_multi_id_free(tuple);
	return bound;
}

/* Only keep those elements in "umap" that have a domain in "space".
 */
__isl_give isl_union_map *isl_union_map_intersect_domain_space(
	__isl_take isl_union_map *umap, __isl_take isl_space *space)
{
	struct isl_un_op_control control = {
		.filter = &has_domain_space_tuples,
		.filter_user = space,
	};

	umap = un_op(umap, &control);
	isl_space_free(space);
	return umap;
}

/* Only keep those elements in "umap" that have a range in "space".
 */
__isl_give isl_union_map *isl_union_map_intersect_range_space(
	__isl_take isl_union_map *umap, __isl_take isl_space *space)
{
	struct isl_un_op_control control = {
		.filter = &has_range_space_tuples,
		.filter_user = space,
	};

	umap = un_op(umap, &control);
	isl_space_free(space);
	return umap;
}
