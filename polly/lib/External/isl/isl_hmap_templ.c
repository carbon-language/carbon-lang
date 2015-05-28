/*
 * Copyright 2011      INRIA Saclay
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <isl/ctx.h>
#include <isl/hash.h>

#define xCAT(A,B) A ## B
#define CAT(A,B) xCAT(A,B)
#define KEY CAT(isl_,KEY_BASE)
#define VAL CAT(isl_,VAL_BASE)
#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)
#define xHMAP(KEY,VAL_BASE) KEY ## _to_ ## VAL_BASE
#define yHMAP(KEY,VAL_BASE) xHMAP(KEY,VAL_BASE)
#define HMAP yHMAP(KEY,VAL_BASE)
#define HMAP_BASE yHMAP(KEY_BASE,VAL_BASE)
#define xS(TYPE1,TYPE2,NAME) struct isl_ ## TYPE1 ## _ ## TYPE2 ## _ ## NAME
#define yS(TYPE1,TYPE2,NAME) xS(TYPE1,TYPE2,NAME)
#define S(NAME) yS(KEY_BASE,VAL_BASE,NAME)

struct HMAP {
	int ref;
	isl_ctx *ctx;
	struct isl_hash_table table;
};

S(pair) {
	KEY *key;
	VAL *val;
};

__isl_give HMAP *FN(HMAP,alloc)(isl_ctx *ctx, int min_size)
{
	HMAP *hmap;

	hmap = isl_calloc_type(ctx, HMAP);
	if (!hmap)
		return NULL;

	hmap->ctx = ctx;
	isl_ctx_ref(ctx);
	hmap->ref = 1;

	if (isl_hash_table_init(ctx, &hmap->table, min_size) < 0)
		return FN(HMAP,free)(hmap);

	return hmap;
}

static isl_stat free_pair(void **entry, void *user)
{
	S(pair) *pair = *entry;
	FN(KEY,free)(pair->key);
	FN(VAL,free)(pair->val);
	free(pair);
	*entry = NULL;
	return isl_stat_ok;
}

__isl_null HMAP *FN(HMAP,free)(__isl_take HMAP *hmap)
{
	if (!hmap)
		return NULL;
	if (--hmap->ref > 0)
		return NULL;
	isl_hash_table_foreach(hmap->ctx, &hmap->table, &free_pair, NULL);
	isl_hash_table_clear(&hmap->table);
	isl_ctx_deref(hmap->ctx);
	free(hmap);
	return NULL;
}

isl_ctx *FN(HMAP,get_ctx)(__isl_keep HMAP *hmap)
{
	return hmap ? hmap->ctx : NULL;
}

/* Add a mapping from "key" to "val" to the associative array
 * pointed to by user.
 */
static isl_stat add_key_val(__isl_take KEY *key, __isl_take VAL *val,
	void *user)
{
	HMAP **hmap = (HMAP **) user;

	*hmap = FN(HMAP,set)(*hmap, key, val);

	if (!*hmap)
		return isl_stat_error;

	return isl_stat_ok;
}

__isl_give HMAP *FN(HMAP,dup)(__isl_keep HMAP *hmap)
{
	HMAP *dup;

	if (!hmap)
		return NULL;

	dup = FN(HMAP,alloc)(hmap->ctx, hmap->table.n);
	if (FN(HMAP,foreach)(hmap, &add_key_val, &dup) < 0)
		return FN(HMAP,free)(dup);

	return dup;
}

__isl_give HMAP *FN(HMAP,cow)(__isl_take HMAP *hmap)
{
	if (!hmap)
		return NULL;

	if (hmap->ref == 1)
		return hmap;
	hmap->ref--;
	return FN(HMAP,dup)(hmap);
}

__isl_give HMAP *FN(HMAP,copy)(__isl_keep HMAP *hmap)
{
	if (!hmap)
		return NULL;

	hmap->ref++;
	return hmap;
}

static int has_key(const void *entry, const void *c_key)
{
	const S(pair) *pair = entry;
	KEY *key = (KEY *) c_key;

	return KEY_EQUAL(pair->key, key);
}

isl_bool FN(HMAP,has)(__isl_keep HMAP *hmap, __isl_keep KEY *key)
{
	uint32_t hash;

	if (!hmap)
		return isl_bool_error;

	hash = FN(KEY,get_hash)(key);
	return !!isl_hash_table_find(hmap->ctx, &hmap->table, hash,
					&has_key, key, 0);
}

__isl_give VAL *FN(HMAP,get)(__isl_keep HMAP *hmap, __isl_take KEY *key)
{
	struct isl_hash_table_entry *entry;
	S(pair) *pair;
	uint32_t hash;

	if (!hmap || !key)
		goto error;

	hash = FN(KEY,get_hash)(key);
	entry = isl_hash_table_find(hmap->ctx, &hmap->table, hash,
					&has_key, key, 0);
	FN(KEY,free)(key);

	if (!entry)
		return NULL;

	pair = entry->data;

	return FN(VAL,copy)(pair->val);
error:
	FN(KEY,free)(key);
	return NULL;
}

/* Remove the mapping between "key" and its associated value (if any)
 * from "hmap".
 *
 * If "key" is not mapped to anything, then we leave "hmap" untouched"
 */
__isl_give HMAP *FN(HMAP,drop)(__isl_take HMAP *hmap, __isl_take KEY *key)
{
	struct isl_hash_table_entry *entry;
	S(pair) *pair;
	uint32_t hash;

	if (!hmap || !key)
		goto error;

	hash = FN(KEY,get_hash)(key);
	entry = isl_hash_table_find(hmap->ctx, &hmap->table, hash,
					&has_key, key, 0);
	if (!entry) {
		FN(KEY,free)(key);
		return hmap;
	}

	hmap = FN(HMAP,cow)(hmap);
	if (!hmap)
		goto error;
	entry = isl_hash_table_find(hmap->ctx, &hmap->table, hash,
					&has_key, key, 0);
	FN(KEY,free)(key);

	if (!entry)
		isl_die(hmap->ctx, isl_error_internal,
			"missing entry" , goto error);

	pair = entry->data;
	isl_hash_table_remove(hmap->ctx, &hmap->table, entry);
	FN(KEY,free)(pair->key);
	FN(VAL,free)(pair->val);
	free(pair);

	return hmap;
error:
	FN(KEY,free)(key);
	FN(HMAP,free)(hmap);
	return NULL;
}

/* Add a mapping from "key" to "val" to "hmap".
 * If "key" was already mapped to something else, then that mapping
 * is replaced.
 * If key happened to be mapped to "val" already, then we leave
 * "hmap" untouched.
 */
__isl_give HMAP *FN(HMAP,set)(__isl_take HMAP *hmap,
	__isl_take KEY *key, __isl_take VAL *val)
{
	struct isl_hash_table_entry *entry;
	S(pair) *pair;
	uint32_t hash;

	if (!hmap || !key || !val)
		goto error;

	hash = FN(KEY,get_hash)(key);
	entry = isl_hash_table_find(hmap->ctx, &hmap->table, hash,
					&has_key, key, 0);
	if (entry) {
		int equal;
		pair = entry->data;
		equal = VAL_EQUAL(pair->val, val);
		if (equal < 0)
			goto error;
		if (equal) {
			FN(KEY,free)(key);
			FN(VAL,free)(val);
			return hmap;
		}
	}

	hmap = FN(HMAP,cow)(hmap);
	if (!hmap)
		goto error;

	entry = isl_hash_table_find(hmap->ctx, &hmap->table, hash,
					&has_key, key, 1);

	if (!entry)
		goto error;

	if (entry->data) {
		pair = entry->data;
		FN(VAL,free)(pair->val);
		pair->val = val;
		FN(KEY,free)(key);
		return hmap;
	}

	pair = isl_alloc_type(hmap->ctx, S(pair));
	if (!pair)
		goto error;

	entry->data = pair;
	pair->key = key;
	pair->val = val;
	return hmap;
error:
	FN(KEY,free)(key);
	FN(VAL,free)(val);
	return FN(HMAP,free)(hmap);
}

/* Internal data structure for isl_map_to_basic_set_foreach.
 *
 * fn is the function that should be called on each entry.
 * user is the user-specified final argument to fn.
 */
S(foreach_data) {
	isl_stat (*fn)(__isl_take KEY *key, __isl_take VAL *val, void *user);
	void *user;
};

/* Call data->fn on a copy of the key and value in *entry.
 */
static isl_stat call_on_copy(void **entry, void *user)
{
	S(pair) *pair = *entry;
	S(foreach_data) *data = (S(foreach_data) *) user;

	return data->fn(FN(KEY,copy)(pair->key), FN(VAL,copy)(pair->val),
			data->user);
}

/* Call "fn" on each pair of key and value in "hmap".
 */
isl_stat FN(HMAP,foreach)(__isl_keep HMAP *hmap,
	isl_stat (*fn)(__isl_take KEY *key, __isl_take VAL *val, void *user),
	void *user)
{
	S(foreach_data) data = { fn, user };

	if (!hmap)
		return isl_stat_error;

	return isl_hash_table_foreach(hmap->ctx, &hmap->table,
				      &call_on_copy, &data);
}

/* Internal data structure for print_pair.
 *
 * p is the printer on which the associative array is being printed.
 * first is set if the current key-value pair is the first to be printed.
 */
S(print_data) {
	isl_printer *p;
	int first;
};

/* Print the given key-value pair to data->p.
 */
static isl_stat print_pair(__isl_take KEY *key, __isl_take VAL *val, void *user)
{
	S(print_data) *data = user;

	if (!data->first)
		data->p = isl_printer_print_str(data->p, ", ");
	data->p = FN(isl_printer_print,KEY_BASE)(data->p, key);
	data->p = isl_printer_print_str(data->p, ": ");
	data->p = FN(isl_printer_print,VAL_BASE)(data->p, val);
	data->first = 0;

	FN(KEY,free)(key);
	FN(VAL,free)(val);
	return isl_stat_ok;
}

/* Print the associative array to "p".
 */
__isl_give isl_printer *FN(isl_printer_print,HMAP_BASE)(
	__isl_take isl_printer *p, __isl_keep HMAP *hmap)
{
	S(print_data) data;

	if (!p || !hmap)
		return isl_printer_free(p);

	p = isl_printer_print_str(p, "{");
	data.p = p;
	data.first = 1;
	if (FN(HMAP,foreach)(hmap, &print_pair, &data) < 0)
		data.p = isl_printer_free(data.p);
	p = data.p;
	p = isl_printer_print_str(p, "}");

	return p;
}

void FN(HMAP,dump)(__isl_keep HMAP *hmap)
{
	isl_printer *printer;

	if (!hmap)
		return;

	printer = isl_printer_to_file(FN(HMAP,get_ctx)(hmap), stderr);
	printer = FN(isl_printer_print,HMAP_BASE)(printer, hmap);
	printer = isl_printer_end_line(printer);

	isl_printer_free(printer);
}
