/*
 * Copyright 2011      INRIA Saclay
 * Copyright 2012-2014 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/ctx.h>
#include <isl_aff_private.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl_sort.h>
#include <isl_schedule_private.h>
#include <isl_band_private.h>

__isl_null isl_schedule *isl_schedule_free(__isl_take isl_schedule *sched)
{
	int i;
	if (!sched)
		return NULL;

	if (--sched->ref > 0)
		return NULL;

	for (i = 0; i < sched->n; ++i) {
		isl_multi_aff_free(sched->node[i].sched);
		free(sched->node[i].band_end);
		free(sched->node[i].band_id);
		free(sched->node[i].coincident);
	}
	isl_space_free(sched->dim);
	isl_band_list_free(sched->band_forest);
	free(sched);
	return NULL;
}

isl_ctx *isl_schedule_get_ctx(__isl_keep isl_schedule *schedule)
{
	return schedule ? isl_space_get_ctx(schedule->dim) : NULL;
}

/* Set max_out to the maximal number of output dimensions over
 * all maps.
 */
static int update_max_out(__isl_take isl_map *map, void *user)
{
	int *max_out = user;
	int n_out = isl_map_dim(map, isl_dim_out);

	if (n_out > *max_out)
		*max_out = n_out;

	isl_map_free(map);
	return 0;
}

/* Internal data structure for map_pad_range.
 *
 * "max_out" is the maximal schedule dimension.
 * "res" collects the results.
 */
struct isl_pad_schedule_map_data {
	int max_out;
	isl_union_map *res;
};

/* Pad the range of the given map with zeros to data->max_out and
 * then add the result to data->res.
 */
static int map_pad_range(__isl_take isl_map *map, void *user)
{
	struct isl_pad_schedule_map_data *data = user;
	int i;
	int n_out = isl_map_dim(map, isl_dim_out);

	map = isl_map_add_dims(map, isl_dim_out, data->max_out - n_out);
	for (i = n_out; i < data->max_out; ++i)
		map = isl_map_fix_si(map, isl_dim_out, i, 0);

	data->res = isl_union_map_add_map(data->res, map);
	if (!data->res)
		return -1;

	return 0;
}

/* Pad the ranges of the maps in the union map with zeros such they all have
 * the same dimension.
 */
static __isl_give isl_union_map *pad_schedule_map(
	__isl_take isl_union_map *umap)
{
	struct isl_pad_schedule_map_data data;

	if (!umap)
		return NULL;
	if (isl_union_map_n_map(umap) <= 1)
		return umap;

	data.max_out = 0;
	if (isl_union_map_foreach_map(umap, &update_max_out, &data.max_out) < 0)
		return isl_union_map_free(umap);

	data.res = isl_union_map_empty(isl_union_map_get_space(umap));
	if (isl_union_map_foreach_map(umap, &map_pad_range, &data) < 0)
		data.res = isl_union_map_free(data.res);

	isl_union_map_free(umap);
	return data.res;
}

/* Return an isl_union_map of the schedule.  If we have already constructed
 * a band forest, then this band forest may have been modified so we need
 * to extract the isl_union_map from the forest rather than from
 * the originally computed schedule.  This reconstructed schedule map
 * then needs to be padded with zeros to unify the schedule space
 * since the result of isl_band_list_get_suffix_schedule may not have
 * a unified schedule space.
 */
__isl_give isl_union_map *isl_schedule_get_map(__isl_keep isl_schedule *sched)
{
	int i;
	isl_union_map *umap;

	if (!sched)
		return NULL;

	if (sched->band_forest) {
		umap = isl_band_list_get_suffix_schedule(sched->band_forest);
		return pad_schedule_map(umap);
	}

	umap = isl_union_map_empty(isl_space_copy(sched->dim));
	for (i = 0; i < sched->n; ++i) {
		isl_multi_aff *ma;

		ma = isl_multi_aff_copy(sched->node[i].sched);
		umap = isl_union_map_add_map(umap, isl_map_from_multi_aff(ma));
	}

	return umap;
}

static __isl_give isl_band_list *construct_band_list(
	__isl_keep isl_schedule *schedule, __isl_keep isl_band *parent,
	int band_nr, int *parent_active, int n_active);

/* Construct an isl_band structure for the band in the given schedule
 * with sequence number band_nr for the n_active nodes marked by active.
 * If the nodes don't have a band with the given sequence number,
 * then a band without members is created.
 *
 * Because of the way the schedule is constructed, we know that
 * the position of the band inside the schedule of a node is the same
 * for all active nodes.
 *
 * The partial schedule for the band is created before the children
 * are created to that construct_band_list can refer to the partial
 * schedule of the parent.
 */
static __isl_give isl_band *construct_band(__isl_keep isl_schedule *schedule,
	__isl_keep isl_band *parent,
	int band_nr, int *active, int n_active)
{
	int i, j;
	isl_ctx *ctx = isl_schedule_get_ctx(schedule);
	isl_band *band;
	unsigned start, end;

	band = isl_band_alloc(ctx);
	if (!band)
		return NULL;

	band->schedule = schedule;
	band->parent = parent;

	for (i = 0; i < schedule->n; ++i)
		if (active[i])
			break;

	if (i >= schedule->n)
		isl_die(ctx, isl_error_internal,
			"band without active statements", goto error);

	start = band_nr ? schedule->node[i].band_end[band_nr - 1] : 0;
	end = band_nr < schedule->node[i].n_band ?
		schedule->node[i].band_end[band_nr] : start;
	band->n = end - start;

	band->coincident = isl_alloc_array(ctx, int, band->n);
	if (band->n && !band->coincident)
		goto error;

	for (j = 0; j < band->n; ++j)
		band->coincident[j] = schedule->node[i].coincident[start + j];

	band->pma = isl_union_pw_multi_aff_empty(isl_space_copy(schedule->dim));
	for (i = 0; i < schedule->n; ++i) {
		isl_multi_aff *ma;
		isl_pw_multi_aff *pma;
		unsigned n_out;

		if (!active[i])
			continue;

		ma = isl_multi_aff_copy(schedule->node[i].sched);
		n_out = isl_multi_aff_dim(ma, isl_dim_out);
		ma = isl_multi_aff_drop_dims(ma, isl_dim_out, end, n_out - end);
		ma = isl_multi_aff_drop_dims(ma, isl_dim_out, 0, start);
		pma = isl_pw_multi_aff_from_multi_aff(ma);
		band->pma = isl_union_pw_multi_aff_add_pw_multi_aff(band->pma,
								    pma);
	}
	if (!band->pma)
		goto error;

	for (i = 0; i < schedule->n; ++i)
		if (active[i] && schedule->node[i].n_band > band_nr + 1)
			break;

	if (i < schedule->n) {
		band->children = construct_band_list(schedule, band,
						band_nr + 1, active, n_active);
		if (!band->children)
			goto error;
	}

	return band;
error:
	isl_band_free(band);
	return NULL;
}

/* Internal data structure used inside cmp_band and pw_multi_aff_extract_int.
 *
 * r is set to a negative value if anything goes wrong.
 *
 * c1 stores the result of extract_int.
 * c2 is a temporary value used inside cmp_band_in_ancestor.
 * t is a temporary value used inside extract_int.
 *
 * first and equal are used inside extract_int.
 * first is set if we are looking at the first isl_multi_aff inside
 * the isl_union_pw_multi_aff.
 * equal is set if all the isl_multi_affs have been equal so far.
 */
struct isl_cmp_band_data {
	int r;

	int first;
	int equal;

	isl_int t;
	isl_int c1;
	isl_int c2;
};

/* Check if "ma" assigns a constant value.
 * Note that this function is only called on isl_multi_affs
 * with a single output dimension.
 *
 * If "ma" assigns a constant value then we compare it to data->c1
 * or assign it to data->c1 if this is the first isl_multi_aff we consider.
 * If "ma" does not assign a constant value or if it assigns a value
 * that is different from data->c1, then we set data->equal to zero
 * and terminate the check.
 */
static int multi_aff_extract_int(__isl_take isl_set *set,
	__isl_take isl_multi_aff *ma, void *user)
{
	isl_aff *aff;
	struct isl_cmp_band_data *data = user;

	aff = isl_multi_aff_get_aff(ma, 0);
	data->r = isl_aff_is_cst(aff);
	if (data->r >= 0 && data->r) {
		isl_aff_get_constant(aff, &data->t);
		if (data->first) {
			isl_int_set(data->c1, data->t);
			data->first = 0;
		} else if (!isl_int_eq(data->c1, data->t))
			data->equal = 0;
	} else if (data->r >= 0 && !data->r)
		data->equal = 0;

	isl_aff_free(aff);
	isl_set_free(set);
	isl_multi_aff_free(ma);

	if (data->r < 0)
		return -1;
	if (!data->equal)
		return -1;
	return 0;
}

/* This function is called for each isl_pw_multi_aff in
 * the isl_union_pw_multi_aff checked by extract_int.
 * Check all the isl_multi_affs inside "pma".
 */
static int pw_multi_aff_extract_int(__isl_take isl_pw_multi_aff *pma,
	void *user)
{
	int r;

	r = isl_pw_multi_aff_foreach_piece(pma, &multi_aff_extract_int, user);
	isl_pw_multi_aff_free(pma);

	return r;
}

/* Check if "upma" assigns a single constant value to its domain.
 * If so, return 1 and store the result in data->c1.
 * If not, return 0.
 *
 * A negative return value from isl_union_pw_multi_aff_foreach_pw_multi_aff
 * means that either an error occurred or that we have broken off the check
 * because we already know the result is going to be negative.
 * In the latter case, data->equal is set to zero.
 */
static int extract_int(__isl_keep isl_union_pw_multi_aff *upma,
	struct isl_cmp_band_data *data)
{
	data->first = 1;
	data->equal = 1;

	if (isl_union_pw_multi_aff_foreach_pw_multi_aff(upma,
					&pw_multi_aff_extract_int, data) < 0) {
		if (!data->equal)
			return 0;
		return -1;
	}

	return !data->first && data->equal;
}

/* Compare "b1" and "b2" based on the parent schedule of their ancestor
 * "ancestor".
 *
 * If the parent of "ancestor" also has a single member, then we
 * first try to compare the two band based on the partial schedule
 * of this parent.
 *
 * Otherwise, or if the result is inconclusive, we look at the partial schedule
 * of "ancestor" itself.
 * In particular, we specialize the parent schedule based
 * on the domains of the child schedules, check if both assign
 * a single constant value and, if so, compare the two constant values.
 * If the specialized parent schedules do not assign a constant value,
 * then they cannot be used to order the two bands and so in this case
 * we return 0.
 */
static int cmp_band_in_ancestor(__isl_keep isl_band *b1,
	__isl_keep isl_band *b2, struct isl_cmp_band_data *data,
	__isl_keep isl_band *ancestor)
{
	isl_union_pw_multi_aff *upma;
	isl_union_set *domain;
	int r;

	if (data->r < 0)
		return 0;

	if (ancestor->parent && ancestor->parent->n == 1) {
		r = cmp_band_in_ancestor(b1, b2, data, ancestor->parent);
		if (data->r < 0)
			return 0;
		if (r)
			return r;
	}

	upma = isl_union_pw_multi_aff_copy(b1->pma);
	domain = isl_union_pw_multi_aff_domain(upma);
	upma = isl_union_pw_multi_aff_copy(ancestor->pma);
	upma = isl_union_pw_multi_aff_intersect_domain(upma, domain);
	r = extract_int(upma, data);
	isl_union_pw_multi_aff_free(upma);

	if (r < 0)
		data->r = -1;
	if (r < 0 || !r)
		return 0;

	isl_int_set(data->c2, data->c1);

	upma = isl_union_pw_multi_aff_copy(b2->pma);
	domain = isl_union_pw_multi_aff_domain(upma);
	upma = isl_union_pw_multi_aff_copy(ancestor->pma);
	upma = isl_union_pw_multi_aff_intersect_domain(upma, domain);
	r = extract_int(upma, data);
	isl_union_pw_multi_aff_free(upma);

	if (r < 0)
		data->r = -1;
	if (r < 0 || !r)
		return 0;

	return isl_int_cmp(data->c2, data->c1);
}

/* Compare "a" and "b" based on the parent schedule of their parent.
 */
static int cmp_band(const void *a, const void *b, void *user)
{
	isl_band *b1 = *(isl_band * const *) a;
	isl_band *b2 = *(isl_band * const *) b;
	struct isl_cmp_band_data *data = user;

	return cmp_band_in_ancestor(b1, b2, data, b1->parent);
}

/* Sort the elements in "list" based on the partial schedules of its parent
 * (and ancestors).  In particular if the parent assigns constant values
 * to the domains of the bands in "list", then the elements are sorted
 * according to that order.
 * This order should be a more "natural" order for the user, but otherwise
 * shouldn't have any effect.
 * If we would be constructing an isl_band forest directly in
 * isl_schedule_constraints_compute_schedule then there wouldn't be any need
 * for a reordering, since the children would be added to the list
 * in their natural order automatically.
 *
 * If there is only one element in the list, then there is no need to sort
 * anything.
 * If the partial schedule of the parent has more than one member
 * (or if there is no parent), then it's
 * defnitely not assigning constant values to the different children in
 * the list and so we wouldn't be able to use it to sort the list.
 */
static __isl_give isl_band_list *sort_band_list(__isl_take isl_band_list *list,
	__isl_keep isl_band *parent)
{
	struct isl_cmp_band_data data;

	if (!list)
		return NULL;
	if (list->n <= 1)
		return list;
	if (!parent || parent->n != 1)
		return list;

	data.r = 0;
	isl_int_init(data.c1);
	isl_int_init(data.c2);
	isl_int_init(data.t);
	isl_sort(list->p, list->n, sizeof(list->p[0]), &cmp_band, &data);
	if (data.r < 0)
		list = isl_band_list_free(list);
	isl_int_clear(data.c1);
	isl_int_clear(data.c2);
	isl_int_clear(data.t);

	return list;
}

/* Construct a list of bands that start at the same position (with
 * sequence number band_nr) in the schedules of the nodes that
 * were active in the parent band.
 *
 * A separate isl_band structure is created for each band_id
 * and for each node that does not have a band with sequence
 * number band_nr.  In the latter case, a band without members
 * is created.
 * This ensures that if a band has any children, then each node
 * that was active in the band is active in exactly one of the children.
 */
static __isl_give isl_band_list *construct_band_list(
	__isl_keep isl_schedule *schedule, __isl_keep isl_band *parent,
	int band_nr, int *parent_active, int n_active)
{
	int i, j;
	isl_ctx *ctx = isl_schedule_get_ctx(schedule);
	int *active;
	int n_band;
	isl_band_list *list;

	n_band = 0;
	for (i = 0; i < n_active; ++i) {
		for (j = 0; j < schedule->n; ++j) {
			if (!parent_active[j])
				continue;
			if (schedule->node[j].n_band <= band_nr)
				continue;
			if (schedule->node[j].band_id[band_nr] == i) {
				n_band++;
				break;
			}
		}
	}
	for (j = 0; j < schedule->n; ++j)
		if (schedule->node[j].n_band <= band_nr)
			n_band++;

	if (n_band == 1) {
		isl_band *band;
		list = isl_band_list_alloc(ctx, n_band);
		band = construct_band(schedule, parent, band_nr,
					parent_active, n_active);
		return isl_band_list_add(list, band);
	}

	active = isl_alloc_array(ctx, int, schedule->n);
	if (schedule->n && !active)
		return NULL;

	list = isl_band_list_alloc(ctx, n_band);

	for (i = 0; i < n_active; ++i) {
		int n = 0;
		isl_band *band;

		for (j = 0; j < schedule->n; ++j) {
			active[j] = parent_active[j] &&
					schedule->node[j].n_band > band_nr &&
					schedule->node[j].band_id[band_nr] == i;
			if (active[j])
				n++;
		}
		if (n == 0)
			continue;

		band = construct_band(schedule, parent, band_nr, active, n);

		list = isl_band_list_add(list, band);
	}
	for (i = 0; i < schedule->n; ++i) {
		isl_band *band;
		if (!parent_active[i])
			continue;
		if (schedule->node[i].n_band > band_nr)
			continue;
		for (j = 0; j < schedule->n; ++j)
			active[j] = j == i;
		band = construct_band(schedule, parent, band_nr, active, 1);
		list = isl_band_list_add(list, band);
	}

	free(active);

	list = sort_band_list(list, parent);

	return list;
}

/* Construct a band forest representation of the schedule and
 * return the list of roots.
 */
static __isl_give isl_band_list *construct_forest(
	__isl_keep isl_schedule *schedule)
{
	int i;
	isl_ctx *ctx = isl_schedule_get_ctx(schedule);
	isl_band_list *forest;
	int *active;

	active = isl_alloc_array(ctx, int, schedule->n);
	if (schedule->n && !active)
		return NULL;

	for (i = 0; i < schedule->n; ++i)
		active[i] = 1;

	forest = construct_band_list(schedule, NULL, 0, active, schedule->n);

	free(active);

	return forest;
}

/* Return the roots of a band forest representation of the schedule.
 */
__isl_give isl_band_list *isl_schedule_get_band_forest(
	__isl_keep isl_schedule *schedule)
{
	if (!schedule)
		return NULL;
	if (!schedule->band_forest)
		schedule->band_forest = construct_forest(schedule);
	return isl_band_list_dup(schedule->band_forest);
}

/* Call "fn" on each band in the schedule in depth-first post-order.
 */
int isl_schedule_foreach_band(__isl_keep isl_schedule *sched,
	int (*fn)(__isl_keep isl_band *band, void *user), void *user)
{
	int r;
	isl_band_list *forest;

	if (!sched)
		return -1;

	forest = isl_schedule_get_band_forest(sched);
	r = isl_band_list_foreach_band(forest, fn, user);
	isl_band_list_free(forest);

	return r;
}

static __isl_give isl_printer *print_band_list(__isl_take isl_printer *p,
	__isl_keep isl_band_list *list);

static __isl_give isl_printer *print_band(__isl_take isl_printer *p,
	__isl_keep isl_band *band)
{
	isl_band_list *children;

	p = isl_printer_start_line(p);
	p = isl_printer_print_union_pw_multi_aff(p, band->pma);
	p = isl_printer_end_line(p);

	if (!isl_band_has_children(band))
		return p;

	children = isl_band_get_children(band);

	p = isl_printer_indent(p, 4);
	p = print_band_list(p, children);
	p = isl_printer_indent(p, -4);

	isl_band_list_free(children);

	return p;
}

static __isl_give isl_printer *print_band_list(__isl_take isl_printer *p,
	__isl_keep isl_band_list *list)
{
	int i, n;

	n = isl_band_list_n_band(list);
	for (i = 0; i < n; ++i) {
		isl_band *band;
		band = isl_band_list_get_band(list, i);
		p = print_band(p, band);
		isl_band_free(band);
	}

	return p;
}

__isl_give isl_printer *isl_printer_print_schedule(__isl_take isl_printer *p,
	__isl_keep isl_schedule *schedule)
{
	isl_band_list *forest;

	forest = isl_schedule_get_band_forest(schedule);

	p = print_band_list(p, forest);

	isl_band_list_free(forest);

	return p;
}

void isl_schedule_dump(__isl_keep isl_schedule *schedule)
{
	isl_printer *printer;

	if (!schedule)
		return;

	printer = isl_printer_to_file(isl_schedule_get_ctx(schedule), stderr);
	printer = isl_printer_print_schedule(printer, schedule);

	isl_printer_free(printer);
}
