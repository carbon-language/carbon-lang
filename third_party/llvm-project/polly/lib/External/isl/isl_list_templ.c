/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2011      INRIA Saclay
 * Copyright 2012-2013 Ecole Normale Superieure
 * Copyright 2017      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <isl_sort.h>
#include <isl_tarjan.h>
#include <isl/printer.h>

#include <isl_list_macro.h>

#define xS(TYPE,NAME) struct TYPE ## _ ## NAME
#define S(TYPE,NAME) xS(TYPE,NAME)

isl_ctx *FN(LIST(EL),get_ctx)(__isl_keep LIST(EL) *list)
{
	return list ? list->ctx : NULL;
}

__isl_give LIST(EL) *FN(LIST(EL),alloc)(isl_ctx *ctx, int n)
{
	LIST(EL) *list;

	if (n < 0)
		isl_die(ctx, isl_error_invalid,
			"cannot create list of negative length",
			return NULL);
	list = isl_alloc(ctx, LIST(EL),
			 sizeof(LIST(EL)) + (n - 1) * sizeof(struct EL *));
	if (!list)
		return NULL;

	list->ctx = ctx;
	isl_ctx_ref(ctx);
	list->ref = 1;
	list->size = n;
	list->n = 0;
	return list;
}

__isl_give LIST(EL) *FN(LIST(EL),copy)(__isl_keep LIST(EL) *list)
{
	if (!list)
		return NULL;

	list->ref++;
	return list;
}

__isl_give LIST(EL) *FN(LIST(EL),dup)(__isl_keep LIST(EL) *list)
{
	int i;
	LIST(EL) *dup;

	if (!list)
		return NULL;

	dup = FN(LIST(EL),alloc)(FN(LIST(EL),get_ctx)(list), list->n);
	if (!dup)
		return NULL;
	for (i = 0; i < list->n; ++i)
		dup = FN(LIST(EL),add)(dup, FN(EL,copy)(list->p[i]));
	return dup;
}

__isl_give LIST(EL) *FN(LIST(EL),cow)(__isl_take LIST(EL) *list)
{
	if (!list)
		return NULL;

	if (list->ref == 1)
		return list;
	list->ref--;
	return FN(LIST(EL),dup)(list);
}

/* Make sure "list" has room for at least "n" more pieces.
 * Always return a list with a single reference.
 *
 * If there is only one reference to list, we extend it in place.
 * Otherwise, we create a new LIST(EL) and copy the elements.
 */
static __isl_give LIST(EL) *FN(LIST(EL),grow)(__isl_take LIST(EL) *list, int n)
{
	isl_ctx *ctx;
	int i, new_size;
	LIST(EL) *res;

	if (!list)
		return NULL;
	if (list->ref == 1 && list->n + n <= list->size)
		return list;

	ctx = FN(LIST(EL),get_ctx)(list);
	new_size = ((list->n + n + 1) * 3) / 2;
	if (list->ref == 1) {
		res = isl_realloc(ctx, list, LIST(EL),
			    sizeof(LIST(EL)) + (new_size - 1) * sizeof(EL *));
		if (!res)
			return FN(LIST(EL),free)(list);
		res->size = new_size;
		return res;
	}

	if (list->n + n <= list->size && list->size < new_size)
		new_size = list->size;

	res = FN(LIST(EL),alloc)(ctx, new_size);
	if (!res)
		return FN(LIST(EL),free)(list);

	for (i = 0; i < list->n; ++i)
		res = FN(LIST(EL),add)(res, FN(EL,copy)(list->p[i]));

	FN(LIST(EL),free)(list);
	return res;
}

/* Check that "index" is a valid position in "list".
 */
static isl_stat FN(LIST(EL),check_index)(__isl_keep LIST(EL) *list, int index)
{
	if (!list)
		return isl_stat_error;
	if (index < 0 || index >= list->n)
		isl_die(FN(LIST(EL),get_ctx)(list), isl_error_invalid,
			"index out of bounds", return isl_stat_error);
	return isl_stat_ok;
}

__isl_give LIST(EL) *FN(LIST(EL),add)(__isl_take LIST(EL) *list,
	__isl_take struct EL *el)
{
	list = FN(LIST(EL),grow)(list, 1);
	if (!list || !el)
		goto error;
	list->p[list->n] = el;
	list->n++;
	return list;
error:
	FN(EL,free)(el);
	FN(LIST(EL),free)(list);
	return NULL;
}

/* Remove the "n" elements starting at "first" from "list".
 */
__isl_give LIST(EL) *FN(LIST(EL),drop)(__isl_take LIST(EL) *list,
	unsigned first, unsigned n)
{
	int i;

	if (!list)
		return NULL;
	if (first + n > list->n || first + n < first)
		isl_die(list->ctx, isl_error_invalid,
			"index out of bounds", return FN(LIST(EL),free)(list));
	if (n == 0)
		return list;
	list = FN(LIST(EL),cow)(list);
	if (!list)
		return NULL;
	for (i = 0; i < n; ++i)
		FN(EL,free)(list->p[first + i]);
	for (i = first; i + n < list->n; ++i)
		list->p[i] = list->p[i + n];
	list->n -= n;
	return list;
}

/* Remove all elements from "list".
 */
__isl_give LIST(EL) *FN(LIST(EL),clear)(__isl_take LIST(EL) *list)
{
	if (!list)
		return NULL;
	return FN(LIST(EL),drop)(list, 0, list->n);
}

/* Insert "el" at position "pos" in "list".
 *
 * If there is only one reference to "list" and if it already has space
 * for one extra element, we insert it directly into "list".
 * Otherwise, we create a new list consisting of "el" and copied
 * elements from "list".
 */
__isl_give LIST(EL) *FN(LIST(EL),insert)(__isl_take LIST(EL) *list,
	unsigned pos, __isl_take struct EL *el)
{
	int i;
	isl_ctx *ctx;
	LIST(EL) *res;

	if (!list || !el)
		goto error;
	ctx = FN(LIST(EL),get_ctx)(list);
	if (pos > list->n)
		isl_die(ctx, isl_error_invalid,
			"index out of bounds", goto error);

	if (list->ref == 1 && list->size > list->n) {
		for (i = list->n; i > pos; --i)
			list->p[i] = list->p[i - 1];
		list->n++;
		list->p[pos] = el;
		return list;
	}

	res = FN(LIST(EL),alloc)(ctx, list->n + 1);
	for (i = 0; i < pos; ++i)
		res = FN(LIST(EL),add)(res, FN(EL,copy)(list->p[i]));
	res = FN(LIST(EL),add)(res, el);
	for (i = pos; i < list->n; ++i)
		res = FN(LIST(EL),add)(res, FN(EL,copy)(list->p[i]));
	FN(LIST(EL),free)(list);

	return res;
error:
	FN(EL,free)(el);
	FN(LIST(EL),free)(list);
	return NULL;
}

__isl_null LIST(EL) *FN(LIST(EL),free)(__isl_take LIST(EL) *list)
{
	int i;

	if (!list)
		return NULL;

	if (--list->ref > 0)
		return NULL;

	isl_ctx_deref(list->ctx);
	for (i = 0; i < list->n; ++i)
		FN(EL,free)(list->p[i]);
	free(list);

	return NULL;
}

/* Return the number of elements in "list".
 */
isl_size FN(LIST(EL),size)(__isl_keep LIST(EL) *list)
{
	return list ? list->n : isl_size_error;
}

/* This is an alternative name for the function above.
 */
isl_size FN(FN(LIST(EL),n),EL_BASE)(__isl_keep LIST(EL) *list)
{
	return FN(LIST(EL),size)(list);
}

/* Return the element at position "index" in "list".
 */
__isl_keep EL *FN(LIST(EL),peek)(__isl_keep LIST(EL) *list, int index)
{
	if (FN(LIST(EL),check_index)(list, index) < 0)
		return NULL;
	return list->p[index];
}

/* Return a copy of the element at position "index" in "list".
 */
__isl_give EL *FN(LIST(EL),get_at)(__isl_keep LIST(EL) *list, int index)
{
	return FN(EL,copy)(FN(LIST(EL),peek)(list, index));
}

/* This is an alternative name for the function above.
 */
__isl_give EL *FN(FN(LIST(EL),get),EL_BASE)(__isl_keep LIST(EL) *list,
	int index)
{
	return FN(LIST(EL),get_at)(list, index);
}

/* Replace the element at position "index" in "list" by "el".
 */
__isl_give LIST(EL) *FN(FN(LIST(EL),set),EL_BASE)(__isl_take LIST(EL) *list,
	int index, __isl_take EL *el)
{
	if (!list || !el)
		goto error;
	if (FN(LIST(EL),check_index)(list, index) < 0)
		goto error;
	if (list->p[index] == el) {
		FN(EL,free)(el);
		return list;
	}
	list = FN(LIST(EL),cow)(list);
	if (!list)
		goto error;
	FN(EL,free)(list->p[index]);
	list->p[index] = el;
	return list;
error:
	FN(EL,free)(el);
	FN(LIST(EL),free)(list);
	return NULL;
}

/* Return the element at position "index" of "list".
 * This may be either a copy or the element itself
 * if there is only one reference to "list".
 * This allows the element to be modified inplace
 * if both the list and the element have only a single reference.
 * The caller is not allowed to modify "list" between
 * this call to isl_list_*_take_* and a subsequent call
 * to isl_list_*_restore_*.
 * The only exception is that isl_list_*_free can be called instead.
 */
static __isl_give EL *FN(FN(LIST(EL),take),EL_BASE)(__isl_keep LIST(EL) *list,
	int index)
{
	EL *el;

	if (FN(LIST(EL),check_index)(list, index) < 0)
		return NULL;
	if (list->ref != 1)
		return FN(FN(LIST(EL),get),EL_BASE)(list, index);
	el = list->p[index];
	list->p[index] = NULL;
	return el;
}

/* Set the element at position "index" of "list" to "el",
 * where the position may be empty due to a previous call
 * to isl_list_*_take_*.
 */
static __isl_give LIST(EL) *FN(FN(LIST(EL),restore),EL_BASE)(
	__isl_take LIST(EL) *list, int index, __isl_take EL *el)
{
	return FN(FN(LIST(EL),set),EL_BASE)(list, index, el);
}

/* Swap the elements of "list" in positions "pos1" and "pos2".
 */
__isl_give LIST(EL) *FN(LIST(EL),swap)(__isl_take LIST(EL) *list,
	unsigned pos1, unsigned pos2)
{
	EL *el1, *el2;

	if (pos1 == pos2)
		return list;
	el1 = FN(FN(LIST(EL),take),EL_BASE)(list, pos1);
	el2 = FN(FN(LIST(EL),take),EL_BASE)(list, pos2);
	list = FN(FN(LIST(EL),restore),EL_BASE)(list, pos1, el2);
	list = FN(FN(LIST(EL),restore),EL_BASE)(list, pos2, el1);
	return list;
}

/* Reverse the elements of "list".
 */
__isl_give LIST(EL) *FN(LIST(EL),reverse)(__isl_take LIST(EL) *list)
{
	int i, n;

	n = FN(LIST(EL),size)(list);
	for (i = 0; i < n - 1 - i; ++i)
		list = FN(LIST(EL),swap)(list, i, n - 1 - i);
	return list;
}

isl_stat FN(LIST(EL),foreach)(__isl_keep LIST(EL) *list,
	isl_stat (*fn)(__isl_take EL *el, void *user), void *user)
{
	int i;

	if (!list)
		return isl_stat_error;

	for (i = 0; i < list->n; ++i) {
		EL *el = FN(EL,copy)(list->p[i]);
		if (!el)
			return isl_stat_error;
		if (fn(el, user) < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Does "test" succeed on every element of "list"?
 */
isl_bool FN(LIST(EL),every)(__isl_keep LIST(EL) *list,
	isl_bool (*test)(__isl_keep EL *el, void *user), void *user)
{
	int i;

	if (!list)
		return isl_bool_error;

	for (i = 0; i < list->n; ++i) {
		isl_bool r;

		r = test(list->p[i], user);
		if (r < 0 || !r)
			return r;
	}

	return isl_bool_true;
}

/* Replace each element in "list" by the result of calling "fn"
 * on the element.
 */
__isl_give LIST(EL) *FN(LIST(EL),map)(__isl_keep LIST(EL) *list,
	__isl_give EL *(*fn)(__isl_take EL *el, void *user), void *user)
{
	int i, n;

	if (!list)
		return NULL;

	n = list->n;
	for (i = 0; i < n; ++i) {
		EL *el = FN(FN(LIST(EL),take),EL_BASE)(list, i);
		if (!el)
			return FN(LIST(EL),free)(list);
		el = fn(el, user);
		list = FN(FN(LIST(EL),restore),EL_BASE)(list, i, el);
	}

	return list;
}

/* Internal data structure for isl_*_list_sort.
 *
 * "cmp" is the original comparison function.
 * "user" is a user provided pointer that should be passed to "cmp".
 */
S(LIST(EL),sort_data) {
	int (*cmp)(__isl_keep EL *a, __isl_keep EL *b, void *user);
	void *user;
};

/* Compare two entries of an isl_*_list based on the user provided
 * comparison function on pairs of isl_* objects.
 */
static int FN(LIST(EL),cmp)(const void *a, const void *b, void *user)
{
	S(LIST(EL),sort_data) *data = user;
	EL * const *el1 = a;
	EL * const *el2 = b;

	return data->cmp(*el1, *el2, data->user);
}

/* Sort the elements of "list" in ascending order according to
 * comparison function "cmp".
 */
__isl_give LIST(EL) *FN(LIST(EL),sort)(__isl_take LIST(EL) *list,
	int (*cmp)(__isl_keep EL *a, __isl_keep EL *b, void *user), void *user)
{
	S(LIST(EL),sort_data) data = { cmp, user };

	if (!list)
		return NULL;
	if (list->n <= 1)
		return list;
	list = FN(LIST(EL),cow)(list);
	if (!list)
		return NULL;

	if (isl_sort(list->p, list->n, sizeof(list->p[0]),
			&FN(LIST(EL),cmp), &data) < 0)
		return FN(LIST(EL),free)(list);

	return list;
}

/* Internal data structure for isl_*_list_foreach_scc.
 *
 * "list" is the original list.
 * "follows" is the user provided callback that defines the edges of the graph.
 */
S(LIST(EL),foreach_scc_data) {
	LIST(EL) *list;
	isl_bool (*follows)(__isl_keep EL *a, __isl_keep EL *b, void *user);
	void *follows_user;
};

/* Does element i of data->list follow element j?
 *
 * Use the user provided callback to find out.
 */
static isl_bool FN(LIST(EL),follows)(int i, int j, void *user)
{
	S(LIST(EL),foreach_scc_data) *data = user;

	return data->follows(data->list->p[i], data->list->p[j],
				data->follows_user);
}

/* Call "fn" on the sublist of "list" that consists of the elements
 * with indices specified by the "n" elements of "pos".
 */
static isl_stat FN(LIST(EL),call_on_scc)(__isl_keep LIST(EL) *list, int *pos,
	int n, isl_stat (*fn)(__isl_take LIST(EL) *scc, void *user), void *user)
{
	int i;
	isl_ctx *ctx;
	LIST(EL) *slice;

	ctx = FN(LIST(EL),get_ctx)(list);
	slice = FN(LIST(EL),alloc)(ctx, n);
	for (i = 0; i < n; ++i) {
		EL *el;

		el = FN(EL,copy)(list->p[pos[i]]);
		slice = FN(LIST(EL),add)(slice, el);
	}

	return fn(slice, user);
}

/* Call "fn" on each of the strongly connected components (SCCs) of
 * the graph with as vertices the elements of "list" and
 * a directed edge from node b to node a iff follows(a, b)
 * returns 1.  follows should return -1 on error.
 *
 * If SCC a contains a node i that follows a node j in another SCC b
 * (i.e., follows(i, j, user) returns 1), then fn will be called on SCC a
 * after being called on SCC b.
 *
 * We simply call isl_tarjan_graph_init, extract the SCCs from the result and
 * call fn on each of them.
 */
isl_stat FN(LIST(EL),foreach_scc)(__isl_keep LIST(EL) *list,
	isl_bool (*follows)(__isl_keep EL *a, __isl_keep EL *b, void *user),
	void *follows_user,
	isl_stat (*fn)(__isl_take LIST(EL) *scc, void *user), void *fn_user)
{
	S(LIST(EL),foreach_scc_data) data = { list, follows, follows_user };
	int i, n;
	isl_ctx *ctx;
	struct isl_tarjan_graph *g;

	if (!list)
		return isl_stat_error;
	if (list->n == 0)
		return isl_stat_ok;
	if (list->n == 1)
		return fn(FN(LIST(EL),copy)(list), fn_user);

	ctx = FN(LIST(EL),get_ctx)(list);
	n = list->n;
	g = isl_tarjan_graph_init(ctx, n, &FN(LIST(EL),follows), &data);
	if (!g)
		return isl_stat_error;

	i = 0;
	do {
		int first;

		if (g->order[i] == -1)
			isl_die(ctx, isl_error_internal, "cannot happen",
				break);
		first = i;
		while (g->order[i] != -1) {
			++i; --n;
		}
		if (first == 0 && n == 0) {
			isl_tarjan_graph_free(g);
			return fn(FN(LIST(EL),copy)(list), fn_user);
		}
		if (FN(LIST(EL),call_on_scc)(list, g->order + first, i - first,
					    fn, fn_user) < 0)
			break;
		++i;
	} while (n);

	isl_tarjan_graph_free(g);

	return n > 0 ? isl_stat_error : isl_stat_ok;
}

__isl_give LIST(EL) *FN(FN(LIST(EL),from),EL_BASE)(__isl_take EL *el)
{
	isl_ctx *ctx;
	LIST(EL) *list;

	if (!el)
		return NULL;
	ctx = FN(EL,get_ctx)(el);
	list = FN(LIST(EL),alloc)(ctx, 1);
	if (!list)
		goto error;
	list = FN(LIST(EL),add)(list, el);
	return list;
error:
	FN(EL,free)(el);
	return NULL;
}

/* This function performs the same operation as isl_*_list_from_*,
 * but is considered as a function on the element when exported.
 */
__isl_give LIST(EL) *FN(EL,to_list)(__isl_take EL *el)
{
	return FN(FN(LIST(EL),from),EL_BASE)(el);
}

/* Append the elements of "list2" to "list1", where "list1" is known
 * to have only a single reference and enough room to hold
 * the extra elements.
 */
static __isl_give LIST(EL) *FN(LIST(EL),concat_inplace)(
	__isl_take LIST(EL) *list1, __isl_take LIST(EL) *list2)
{
	int i;

	for (i = 0; i < list2->n; ++i)
		list1 = FN(LIST(EL),add)(list1, FN(EL,copy)(list2->p[i]));
	FN(LIST(EL),free)(list2);
	return list1;
}

/* Concatenate "list1" and "list2".
 * If "list1" has only one reference and has enough room
 * for the elements of "list2", the add the elements to "list1" itself.
 * Otherwise, create a new list to store the result.
 */
__isl_give LIST(EL) *FN(LIST(EL),concat)(__isl_take LIST(EL) *list1,
	__isl_take LIST(EL) *list2)
{
	int i;
	isl_ctx *ctx;
	LIST(EL) *res;

	if (!list1 || !list2)
		goto error;

	if (list1->ref == 1 && list1->n + list2->n <= list1->size)
		return FN(LIST(EL),concat_inplace)(list1, list2);

	ctx = FN(LIST(EL),get_ctx)(list1);
	res = FN(LIST(EL),alloc)(ctx, list1->n + list2->n);
	for (i = 0; i < list1->n; ++i)
		res = FN(LIST(EL),add)(res, FN(EL,copy)(list1->p[i]));
	for (i = 0; i < list2->n; ++i)
		res = FN(LIST(EL),add)(res, FN(EL,copy)(list2->p[i]));

	FN(LIST(EL),free)(list1);
	FN(LIST(EL),free)(list2);
	return res;
error:
	FN(LIST(EL),free)(list1);
	FN(LIST(EL),free)(list2);
	return NULL;
}

__isl_give isl_printer *CAT(isl_printer_print_,LIST(EL_BASE))(
	__isl_take isl_printer *p, __isl_keep LIST(EL) *list)
{
	int i;

	if (!p || !list)
		goto error;
	p = isl_printer_print_str(p, "(");
	for (i = 0; i < list->n; ++i) {
		if (i)
			p = isl_printer_print_str(p, ",");
		p = CAT(isl_printer_print_,EL_BASE)(p, list->p[i]);
	}
	p = isl_printer_print_str(p, ")");
	return p;
error:
	isl_printer_free(p);
	return NULL;
}

#undef BASE
#define BASE LIST(EL_BASE)

#define PRINT_DUMP_DEFAULT	0
#include "print_templ.c"
#undef PRINT_DUMP_DEFAULT
