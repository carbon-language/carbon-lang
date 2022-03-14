/*
 * Copyright 2005-2007 Universiteit Leiden
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010      INRIA Saclay
 * Copyright 2012      Universiteit Leiden
 * Copyright 2014      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, Leiden Institute of Advanced Computer Science,
 * Universiteit Leiden, Niels Bohrweg 1, 2333 CA Leiden, The Netherlands
 * and K.U.Leuven, Departement Computerwetenschappen, Celestijnenlaan 200A,
 * B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France 
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/val.h>
#include <isl/space.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_set.h>
#include <isl/union_map.h>
#include <isl/flow.h>
#include <isl/schedule_node.h>
#include <isl_sort.h>
#include <isl/stream.h>

enum isl_restriction_type {
	isl_restriction_type_empty,
	isl_restriction_type_none,
	isl_restriction_type_input,
	isl_restriction_type_output
};

struct isl_restriction {
	enum isl_restriction_type type;

	isl_set *source;
	isl_set *sink;
};

/* Create a restriction of the given type.
 */
static __isl_give isl_restriction *isl_restriction_alloc(
	__isl_take isl_map *source_map, enum isl_restriction_type type)
{
	isl_ctx *ctx;
	isl_restriction *restr;

	if (!source_map)
		return NULL;

	ctx = isl_map_get_ctx(source_map);
	restr = isl_calloc_type(ctx, struct isl_restriction);
	if (!restr)
		goto error;

	restr->type = type;

	isl_map_free(source_map);
	return restr;
error:
	isl_map_free(source_map);
	return NULL;
}

/* Create a restriction that doesn't restrict anything.
 */
__isl_give isl_restriction *isl_restriction_none(__isl_take isl_map *source_map)
{
	return isl_restriction_alloc(source_map, isl_restriction_type_none);
}

/* Create a restriction that removes everything.
 */
__isl_give isl_restriction *isl_restriction_empty(
	__isl_take isl_map *source_map)
{
	return isl_restriction_alloc(source_map, isl_restriction_type_empty);
}

/* Create a restriction on the input of the maximization problem
 * based on the given source and sink restrictions.
 */
__isl_give isl_restriction *isl_restriction_input(
	__isl_take isl_set *source_restr, __isl_take isl_set *sink_restr)
{
	isl_ctx *ctx;
	isl_restriction *restr;

	if (!source_restr || !sink_restr)
		goto error;

	ctx = isl_set_get_ctx(source_restr);
	restr = isl_calloc_type(ctx, struct isl_restriction);
	if (!restr)
		goto error;

	restr->type = isl_restriction_type_input;
	restr->source = source_restr;
	restr->sink = sink_restr;

	return restr;
error:
	isl_set_free(source_restr);
	isl_set_free(sink_restr);
	return NULL;
}

/* Create a restriction on the output of the maximization problem
 * based on the given source restriction.
 */
__isl_give isl_restriction *isl_restriction_output(
	__isl_take isl_set *source_restr)
{
	isl_ctx *ctx;
	isl_restriction *restr;

	if (!source_restr)
		return NULL;

	ctx = isl_set_get_ctx(source_restr);
	restr = isl_calloc_type(ctx, struct isl_restriction);
	if (!restr)
		goto error;

	restr->type = isl_restriction_type_output;
	restr->source = source_restr;

	return restr;
error:
	isl_set_free(source_restr);
	return NULL;
}

__isl_null isl_restriction *isl_restriction_free(
	__isl_take isl_restriction *restr)
{
	if (!restr)
		return NULL;

	isl_set_free(restr->source);
	isl_set_free(restr->sink);
	free(restr);
	return NULL;
}

isl_ctx *isl_restriction_get_ctx(__isl_keep isl_restriction *restr)
{
	return restr ? isl_set_get_ctx(restr->source) : NULL;
}

/* A private structure to keep track of a mapping together with
 * a user-specified identifier and a boolean indicating whether
 * the map represents a must or may access/dependence.
 */
struct isl_labeled_map {
	struct isl_map	*map;
	void		*data;
	int		must;
};

typedef isl_bool (*isl_access_coscheduled)(void *first, void *second);

/* A structure containing the input for dependence analysis:
 * - a sink
 * - n_must + n_may (<= max_source) sources
 * - a function for determining the relative order of sources and sink
 * - an optional function "coscheduled" for determining whether sources
 *   may be coscheduled.  If "coscheduled" is NULL, then the sources
 *   are assumed not to be coscheduled.
 * The must sources are placed before the may sources.
 *
 * domain_map is an auxiliary map that maps the sink access relation
 * to the domain of this access relation.
 * This field is only needed when restrict_fn is set and
 * the field itself is set by isl_access_info_compute_flow.
 *
 * restrict_fn is a callback that (if not NULL) will be called
 * right before any lexicographical maximization.
 */
struct isl_access_info {
	isl_map				*domain_map;
	struct isl_labeled_map		sink;
	isl_access_level_before		level_before;
	isl_access_coscheduled		coscheduled;

	isl_access_restrict		restrict_fn;
	void				*restrict_user;

	int		    		max_source;
	int		    		n_must;
	int		    		n_may;
	struct isl_labeled_map		source[1];
};

/* A structure containing the output of dependence analysis:
 * - n_source dependences
 * - a wrapped subset of the sink for which definitely no source could be found
 * - a wrapped subset of the sink for which possibly no source could be found
 */
struct isl_flow {
	isl_set			*must_no_source;
	isl_set			*may_no_source;
	int			n_source;
	struct isl_labeled_map	*dep;
};

/* Construct an isl_access_info structure and fill it up with
 * the given data.  The number of sources is set to 0.
 */
__isl_give isl_access_info *isl_access_info_alloc(__isl_take isl_map *sink,
	void *sink_user, isl_access_level_before fn, int max_source)
{
	isl_ctx *ctx;
	struct isl_access_info *acc;

	if (!sink)
		return NULL;

	ctx = isl_map_get_ctx(sink);
	isl_assert(ctx, max_source >= 0, goto error);

	acc = isl_calloc(ctx, struct isl_access_info,
			sizeof(struct isl_access_info) +
			(max_source - 1) * sizeof(struct isl_labeled_map));
	if (!acc)
		goto error;

	acc->sink.map = sink;
	acc->sink.data = sink_user;
	acc->level_before = fn;
	acc->max_source = max_source;
	acc->n_must = 0;
	acc->n_may = 0;

	return acc;
error:
	isl_map_free(sink);
	return NULL;
}

/* Free the given isl_access_info structure.
 */
__isl_null isl_access_info *isl_access_info_free(
	__isl_take isl_access_info *acc)
{
	int i;

	if (!acc)
		return NULL;
	isl_map_free(acc->domain_map);
	isl_map_free(acc->sink.map);
	for (i = 0; i < acc->n_must + acc->n_may; ++i)
		isl_map_free(acc->source[i].map);
	free(acc);
	return NULL;
}

isl_ctx *isl_access_info_get_ctx(__isl_keep isl_access_info *acc)
{
	return acc ? isl_map_get_ctx(acc->sink.map) : NULL;
}

__isl_give isl_access_info *isl_access_info_set_restrict(
	__isl_take isl_access_info *acc, isl_access_restrict fn, void *user)
{
	if (!acc)
		return NULL;
	acc->restrict_fn = fn;
	acc->restrict_user = user;
	return acc;
}

/* Add another source to an isl_access_info structure, making
 * sure the "must" sources are placed before the "may" sources.
 * This function may be called at most max_source times on a
 * given isl_access_info structure, with max_source as specified
 * in the call to isl_access_info_alloc that constructed the structure.
 */
__isl_give isl_access_info *isl_access_info_add_source(
	__isl_take isl_access_info *acc, __isl_take isl_map *source,
	int must, void *source_user)
{
	isl_ctx *ctx;

	if (!acc)
		goto error;
	ctx = isl_map_get_ctx(acc->sink.map);
	isl_assert(ctx, acc->n_must + acc->n_may < acc->max_source, goto error);
	
	if (must) {
		if (acc->n_may)
			acc->source[acc->n_must + acc->n_may] =
				acc->source[acc->n_must];
		acc->source[acc->n_must].map = source;
		acc->source[acc->n_must].data = source_user;
		acc->source[acc->n_must].must = 1;
		acc->n_must++;
	} else {
		acc->source[acc->n_must + acc->n_may].map = source;
		acc->source[acc->n_must + acc->n_may].data = source_user;
		acc->source[acc->n_must + acc->n_may].must = 0;
		acc->n_may++;
	}

	return acc;
error:
	isl_map_free(source);
	isl_access_info_free(acc);
	return NULL;
}

/* A helper struct carrying the isl_access_info and an error condition.
 */
struct access_sort_info {
	isl_access_info *access_info;
	int error;
};

/* Return -n, 0 or n (with n a positive value), depending on whether
 * the source access identified by p1 should be sorted before, together
 * or after that identified by p2.
 *
 * If p1 appears before p2, then it should be sorted first.
 * For more generic initial schedules, it is possible that neither
 * p1 nor p2 appears before the other, or at least not in any obvious way.
 * We therefore also check if p2 appears before p1, in which case p2
 * should be sorted first.
 * If not, we try to order the two statements based on the description
 * of the iteration domains.  This results in an arbitrary, but fairly
 * stable ordering.
 *
 * In case of an error, sort_info.error is set to true and all elements are
 * reported to be equal.
 */
static int access_sort_cmp(const void *p1, const void *p2, void *user)
{
	struct access_sort_info *sort_info = user;
	isl_access_info *acc = sort_info->access_info;

	if (sort_info->error)
		return 0;

	const struct isl_labeled_map *i1, *i2;
	int level1, level2;
	uint32_t h1, h2;
	i1 = (const struct isl_labeled_map *) p1;
	i2 = (const struct isl_labeled_map *) p2;

	level1 = acc->level_before(i1->data, i2->data);
	if (level1 < 0)
		goto error;
	if (level1 % 2)
		return -1;

	level2 = acc->level_before(i2->data, i1->data);
	if (level2 < 0)
		goto error;
	if (level2 % 2)
		return 1;

	h1 = isl_map_get_hash(i1->map);
	h2 = isl_map_get_hash(i2->map);
	return h1 > h2 ? 1 : h1 < h2 ? -1 : 0;
error:
	sort_info->error = 1;
	return 0;
}

/* Sort the must source accesses in their textual order.
 */
static __isl_give isl_access_info *isl_access_info_sort_sources(
	__isl_take isl_access_info *acc)
{
	struct access_sort_info sort_info;

	sort_info.access_info = acc;
	sort_info.error = 0;

	if (!acc)
		return NULL;
	if (acc->n_must <= 1)
		return acc;

	if (isl_sort(acc->source, acc->n_must, sizeof(struct isl_labeled_map),
		    access_sort_cmp, &sort_info) < 0)
		return isl_access_info_free(acc);
	if (sort_info.error)
		return isl_access_info_free(acc);

	return acc;
}

/* Align the parameters of the two spaces if needed and then call
 * isl_space_join.
 */
static __isl_give isl_space *space_align_and_join(__isl_take isl_space *left,
	__isl_take isl_space *right)
{
	isl_bool equal_params;

	equal_params = isl_space_has_equal_params(left, right);
	if (equal_params < 0)
		goto error;
	if (equal_params)
		return isl_space_join(left, right);

	left = isl_space_align_params(left, isl_space_copy(right));
	right = isl_space_align_params(right, isl_space_copy(left));
	return isl_space_join(left, right);
error:
	isl_space_free(left);
	isl_space_free(right);
	return NULL;
}

/* Initialize an empty isl_flow structure corresponding to a given
 * isl_access_info structure.
 * For each must access, two dependences are created (initialized
 * to the empty relation), one for the resulting must dependences
 * and one for the resulting may dependences.  May accesses can
 * only lead to may dependences, so only one dependence is created
 * for each of them.
 * This function is private as isl_flow structures are only supposed
 * to be created by isl_access_info_compute_flow.
 */
static __isl_give isl_flow *isl_flow_alloc(__isl_keep isl_access_info *acc)
{
	int i, n;
	struct isl_ctx *ctx;
	struct isl_flow *dep;

	if (!acc)
		return NULL;

	ctx = isl_map_get_ctx(acc->sink.map);
	dep = isl_calloc_type(ctx, struct isl_flow);
	if (!dep)
		return NULL;

	n = 2 * acc->n_must + acc->n_may;
	dep->dep = isl_calloc_array(ctx, struct isl_labeled_map, n);
	if (n && !dep->dep)
		goto error;

	dep->n_source = n;
	for (i = 0; i < acc->n_must; ++i) {
		isl_space *space;
		space = space_align_and_join(
			isl_map_get_space(acc->source[i].map),
			isl_space_reverse(isl_map_get_space(acc->sink.map)));
		dep->dep[2 * i].map = isl_map_empty(space);
		dep->dep[2 * i + 1].map = isl_map_copy(dep->dep[2 * i].map);
		dep->dep[2 * i].data = acc->source[i].data;
		dep->dep[2 * i + 1].data = acc->source[i].data;
		dep->dep[2 * i].must = 1;
		dep->dep[2 * i + 1].must = 0;
		if (!dep->dep[2 * i].map || !dep->dep[2 * i + 1].map)
			goto error;
	}
	for (i = acc->n_must; i < acc->n_must + acc->n_may; ++i) {
		isl_space *space;
		space = space_align_and_join(
			isl_map_get_space(acc->source[i].map),
			isl_space_reverse(isl_map_get_space(acc->sink.map)));
		dep->dep[acc->n_must + i].map = isl_map_empty(space);
		dep->dep[acc->n_must + i].data = acc->source[i].data;
		dep->dep[acc->n_must + i].must = 0;
		if (!dep->dep[acc->n_must + i].map)
			goto error;
	}

	return dep;
error:
	isl_flow_free(dep);
	return NULL;
}

/* Iterate over all sources and for each resulting flow dependence
 * that is not empty, call the user specfied function.
 * The second argument in this function call identifies the source,
 * while the third argument correspond to the final argument of
 * the isl_flow_foreach call.
 */
isl_stat isl_flow_foreach(__isl_keep isl_flow *deps,
	isl_stat (*fn)(__isl_take isl_map *dep, int must, void *dep_user,
		void *user),
	void *user)
{
	int i;

	if (!deps)
		return isl_stat_error;

	for (i = 0; i < deps->n_source; ++i) {
		if (isl_map_plain_is_empty(deps->dep[i].map))
			continue;
		if (fn(isl_map_copy(deps->dep[i].map), deps->dep[i].must,
				deps->dep[i].data, user) < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Return a copy of the subset of the sink for which no source could be found.
 */
__isl_give isl_map *isl_flow_get_no_source(__isl_keep isl_flow *deps, int must)
{
	if (!deps)
		return NULL;
	
	if (must)
		return isl_set_unwrap(isl_set_copy(deps->must_no_source));
	else
		return isl_set_unwrap(isl_set_copy(deps->may_no_source));
}

__isl_null isl_flow *isl_flow_free(__isl_take isl_flow *deps)
{
	int i;

	if (!deps)
		return NULL;
	isl_set_free(deps->must_no_source);
	isl_set_free(deps->may_no_source);
	if (deps->dep) {
		for (i = 0; i < deps->n_source; ++i)
			isl_map_free(deps->dep[i].map);
		free(deps->dep);
	}
	free(deps);

	return NULL;
}

isl_ctx *isl_flow_get_ctx(__isl_keep isl_flow *deps)
{
	return deps ? isl_set_get_ctx(deps->must_no_source) : NULL;
}

/* Return a map that enforces that the domain iteration occurs after
 * the range iteration at the given level.
 * If level is odd, then the domain iteration should occur after
 * the target iteration in their shared level/2 outermost loops.
 * In this case we simply need to enforce that these outermost
 * loop iterations are the same.
 * If level is even, then the loop iterator of the domain should
 * be greater than the loop iterator of the range at the last
 * of the level/2 shared loops, i.e., loop level/2 - 1.
 */
static __isl_give isl_map *after_at_level(__isl_take isl_space *space,
	int level)
{
	struct isl_basic_map *bmap;

	if (level % 2)
		bmap = isl_basic_map_equal(space, level/2);
	else
		bmap = isl_basic_map_more_at(space, level/2 - 1);

	return isl_map_from_basic_map(bmap);
}

/* Compute the partial lexicographic maximum of "dep" on domain "sink",
 * but first check if the user has set acc->restrict_fn and if so
 * update either the input or the output of the maximization problem
 * with respect to the resulting restriction.
 *
 * Since the user expects a mapping from sink iterations to source iterations,
 * whereas the domain of "dep" is a wrapped map, mapping sink iterations
 * to accessed array elements, we first need to project out the accessed
 * sink array elements by applying acc->domain_map.
 * Similarly, the sink restriction specified by the user needs to be
 * converted back to the wrapped map.
 */
static __isl_give isl_map *restricted_partial_lexmax(
	__isl_keep isl_access_info *acc, __isl_take isl_map *dep,
	int source, __isl_take isl_set *sink, __isl_give isl_set **empty)
{
	isl_map *source_map;
	isl_restriction *restr;
	isl_set *sink_domain;
	isl_set *sink_restr;
	isl_map *res;

	if (!acc->restrict_fn)
		return isl_map_partial_lexmax(dep, sink, empty);

	source_map = isl_map_copy(dep);
	source_map = isl_map_apply_domain(source_map,
					    isl_map_copy(acc->domain_map));
	sink_domain = isl_set_copy(sink);
	sink_domain = isl_set_apply(sink_domain, isl_map_copy(acc->domain_map));
	restr = acc->restrict_fn(source_map, sink_domain,
				acc->source[source].data, acc->restrict_user);
	isl_set_free(sink_domain);
	isl_map_free(source_map);

	if (!restr)
		goto error;
	if (restr->type == isl_restriction_type_input) {
		dep = isl_map_intersect_range(dep, isl_set_copy(restr->source));
		sink_restr = isl_set_copy(restr->sink);
		sink_restr = isl_set_apply(sink_restr,
				isl_map_reverse(isl_map_copy(acc->domain_map)));
		sink = isl_set_intersect(sink, sink_restr);
	} else if (restr->type == isl_restriction_type_empty) {
		isl_space *space = isl_map_get_space(dep);
		isl_map_free(dep);
		dep = isl_map_empty(space);
	}

	res = isl_map_partial_lexmax(dep, sink, empty);

	if (restr->type == isl_restriction_type_output)
		res = isl_map_intersect_range(res, isl_set_copy(restr->source));

	isl_restriction_free(restr);
	return res;
error:
	isl_map_free(dep);
	isl_set_free(sink);
	*empty = NULL;
	return NULL;
}

/* Compute the last iteration of must source j that precedes the sink
 * at the given level for sink iterations in set_C.
 * The subset of set_C for which no such iteration can be found is returned
 * in *empty.
 */
static struct isl_map *last_source(struct isl_access_info *acc, 
				    struct isl_set *set_C,
				    int j, int level, struct isl_set **empty)
{
	struct isl_map *read_map;
	struct isl_map *write_map;
	struct isl_map *dep_map;
	struct isl_map *after;
	struct isl_map *result;

	read_map = isl_map_copy(acc->sink.map);
	write_map = isl_map_copy(acc->source[j].map);
	write_map = isl_map_reverse(write_map);
	dep_map = isl_map_apply_range(read_map, write_map);
	after = after_at_level(isl_map_get_space(dep_map), level);
	dep_map = isl_map_intersect(dep_map, after);
	result = restricted_partial_lexmax(acc, dep_map, j, set_C, empty);
	result = isl_map_reverse(result);

	return result;
}

/* For a given mapping between iterations of must source j and iterations
 * of the sink, compute the last iteration of must source k preceding
 * the sink at level before_level for any of the sink iterations,
 * but following the corresponding iteration of must source j at level
 * after_level.
 */
static struct isl_map *last_later_source(struct isl_access_info *acc,
					 struct isl_map *old_map,
					 int j, int before_level,
					 int k, int after_level,
					 struct isl_set **empty)
{
	isl_space *space;
	struct isl_set *set_C;
	struct isl_map *read_map;
	struct isl_map *write_map;
	struct isl_map *dep_map;
	struct isl_map *after_write;
	struct isl_map *before_read;
	struct isl_map *result;

	set_C = isl_map_range(isl_map_copy(old_map));
	read_map = isl_map_copy(acc->sink.map);
	write_map = isl_map_copy(acc->source[k].map);

	write_map = isl_map_reverse(write_map);
	dep_map = isl_map_apply_range(read_map, write_map);
	space = space_align_and_join(isl_map_get_space(acc->source[k].map),
		    isl_space_reverse(isl_map_get_space(acc->source[j].map)));
	after_write = after_at_level(space, after_level);
	after_write = isl_map_apply_range(after_write, old_map);
	after_write = isl_map_reverse(after_write);
	dep_map = isl_map_intersect(dep_map, after_write);
	before_read = after_at_level(isl_map_get_space(dep_map), before_level);
	dep_map = isl_map_intersect(dep_map, before_read);
	result = restricted_partial_lexmax(acc, dep_map, k, set_C, empty);
	result = isl_map_reverse(result);

	return result;
}

/* Given a shared_level between two accesses, return 1 if the
 * the first can precede the second at the requested target_level.
 * If the target level is odd, i.e., refers to a statement level
 * dimension, then first needs to precede second at the requested
 * level, i.e., shared_level must be equal to target_level.
 * If the target level is odd, then the two loops should share
 * at least the requested number of outer loops.
 */
static int can_precede_at_level(int shared_level, int target_level)
{
	if (shared_level < target_level)
		return 0;
	if ((target_level % 2) && shared_level > target_level)
		return 0;
	return 1;
}

/* Given a possible flow dependence temp_rel[j] between source j and the sink
 * at level sink_level, remove those elements for which
 * there is an iteration of another source k < j that is closer to the sink.
 * The flow dependences temp_rel[k] are updated with the improved sources.
 * Any improved source needs to precede the sink at the same level
 * and needs to follow source j at the same or a deeper level.
 * The lower this level, the later the execution date of source k.
 * We therefore consider lower levels first.
 *
 * If temp_rel[j] is empty, then there can be no improvement and
 * we return immediately.
 *
 * This function returns isl_stat_ok in case it was executed successfully and
 * isl_stat_error in case of errors during the execution of this function.
 */
static isl_stat intermediate_sources(__isl_keep isl_access_info *acc,
	struct isl_map **temp_rel, int j, int sink_level)
{
	int k, level;
	isl_size n_in = isl_map_dim(acc->source[j].map, isl_dim_in);
	int depth = 2 * n_in + 1;

	if (n_in < 0)
		return isl_stat_error;
	if (isl_map_plain_is_empty(temp_rel[j]))
		return isl_stat_ok;

	for (k = j - 1; k >= 0; --k) {
		int plevel, plevel2;
		plevel = acc->level_before(acc->source[k].data, acc->sink.data);
		if (plevel < 0)
			return isl_stat_error;
		if (!can_precede_at_level(plevel, sink_level))
			continue;

		plevel2 = acc->level_before(acc->source[j].data,
						acc->source[k].data);
		if (plevel2 < 0)
			return isl_stat_error;

		for (level = sink_level; level <= depth; ++level) {
			struct isl_map *T;
			struct isl_set *trest;
			struct isl_map *copy;

			if (!can_precede_at_level(plevel2, level))
				continue;

			copy = isl_map_copy(temp_rel[j]);
			T = last_later_source(acc, copy, j, sink_level, k,
					      level, &trest);
			if (isl_map_plain_is_empty(T)) {
				isl_set_free(trest);
				isl_map_free(T);
				continue;
			}
			temp_rel[j] = isl_map_intersect_range(temp_rel[j], trest);
			temp_rel[k] = isl_map_union_disjoint(temp_rel[k], T);
		}
	}

	return isl_stat_ok;
}

/* Compute all iterations of may source j that precedes the sink at the given
 * level for sink iterations in set_C.
 */
static __isl_give isl_map *all_sources(__isl_keep isl_access_info *acc,
				    __isl_take isl_set *set_C, int j, int level)
{
	isl_map *read_map;
	isl_map *write_map;
	isl_map *dep_map;
	isl_map *after;

	read_map = isl_map_copy(acc->sink.map);
	read_map = isl_map_intersect_domain(read_map, set_C);
	write_map = isl_map_copy(acc->source[acc->n_must + j].map);
	write_map = isl_map_reverse(write_map);
	dep_map = isl_map_apply_range(read_map, write_map);
	after = after_at_level(isl_map_get_space(dep_map), level);
	dep_map = isl_map_intersect(dep_map, after);

	return isl_map_reverse(dep_map);
}

/* For a given mapping between iterations of must source k and iterations
 * of the sink, compute all iterations of may source j preceding
 * the sink at level before_level for any of the sink iterations,
 * but following the corresponding iteration of must source k at level
 * after_level.
 */
static __isl_give isl_map *all_later_sources(__isl_keep isl_access_info *acc,
	__isl_take isl_map *old_map,
	int j, int before_level, int k, int after_level)
{
	isl_space *space;
	isl_set *set_C;
	isl_map *read_map;
	isl_map *write_map;
	isl_map *dep_map;
	isl_map *after_write;
	isl_map *before_read;

	set_C = isl_map_range(isl_map_copy(old_map));
	read_map = isl_map_copy(acc->sink.map);
	read_map = isl_map_intersect_domain(read_map, set_C);
	write_map = isl_map_copy(acc->source[acc->n_must + j].map);

	write_map = isl_map_reverse(write_map);
	dep_map = isl_map_apply_range(read_map, write_map);
	space = isl_space_join(isl_map_get_space(
		    acc->source[acc->n_must + j].map),
		    isl_space_reverse(isl_map_get_space(acc->source[k].map)));
	after_write = after_at_level(space, after_level);
	after_write = isl_map_apply_range(after_write, old_map);
	after_write = isl_map_reverse(after_write);
	dep_map = isl_map_intersect(dep_map, after_write);
	before_read = after_at_level(isl_map_get_space(dep_map), before_level);
	dep_map = isl_map_intersect(dep_map, before_read);
	return isl_map_reverse(dep_map);
}

/* Given the must and may dependence relations for the must accesses
 * for level sink_level, check if there are any accesses of may access j
 * that occur in between and return their union.
 * If some of these accesses are intermediate with respect to
 * (previously thought to be) must dependences, then these
 * must dependences are turned into may dependences.
 */
static __isl_give isl_map *all_intermediate_sources(
	__isl_keep isl_access_info *acc, __isl_take isl_map *map,
	struct isl_map **must_rel, struct isl_map **may_rel,
	int j, int sink_level)
{
	int k, level;
	isl_size n_in = isl_map_dim(acc->source[acc->n_must + j].map,
				    isl_dim_in);
	int depth = 2 * n_in + 1;

	if (n_in < 0)
		return isl_map_free(map);
	for (k = 0; k < acc->n_must; ++k) {
		int plevel;

		if (isl_map_plain_is_empty(may_rel[k]) &&
		    isl_map_plain_is_empty(must_rel[k]))
			continue;

		plevel = acc->level_before(acc->source[k].data,
					acc->source[acc->n_must + j].data);
		if (plevel < 0)
			return isl_map_free(map);

		for (level = sink_level; level <= depth; ++level) {
			isl_map *T;
			isl_map *copy;
			isl_set *ran;

			if (!can_precede_at_level(plevel, level))
				continue;

			copy = isl_map_copy(may_rel[k]);
			T = all_later_sources(acc, copy, j, sink_level, k, level);
			map = isl_map_union(map, T);

			copy = isl_map_copy(must_rel[k]);
			T = all_later_sources(acc, copy, j, sink_level, k, level);
			ran = isl_map_range(isl_map_copy(T));
			map = isl_map_union(map, T);
			may_rel[k] = isl_map_union_disjoint(may_rel[k],
			    isl_map_intersect_range(isl_map_copy(must_rel[k]),
						    isl_set_copy(ran)));
			T = isl_map_from_domain_and_range(
			    isl_set_universe(
				isl_space_domain(isl_map_get_space(must_rel[k]))),
			    ran);
			must_rel[k] = isl_map_subtract(must_rel[k], T);
		}
	}

	return map;
}

/* Given a dependence relation "old_map" between a must-source and the sink,
 * return a subset of the dependences, augmented with instances
 * of the source at position "pos" in "acc" that are coscheduled
 * with the must-source and that access the same element.
 * That is, if the input lives in a space T -> K, then the output
 * lives in the space [T -> S] -> K, with S the space of source "pos", and
 * the domain factor of the domain product is a subset of the input.
 * The sources are considered to be coscheduled if they have the same values
 * for the initial "depth" coordinates.
 *
 * First construct a dependence relation S -> K and a mapping
 * between coscheduled sources T -> S.
 * The second is combined with the original dependence relation T -> K
 * to form a relation in T -> [S -> K], which is subsequently
 * uncurried to [T -> S] -> K.
 * This result is then intersected with the dependence relation S -> K
 * to form the output.
 *
 * In case a negative depth is given, NULL is returned to indicate an error.
 */
static __isl_give isl_map *coscheduled_source(__isl_keep isl_access_info *acc,
	__isl_keep isl_map *old_map, int pos, int depth)
{
	isl_space *space;
	isl_set *set_C;
	isl_map *read_map;
	isl_map *write_map;
	isl_map *dep_map;
	isl_map *equal;
	isl_map *map;

	if (depth < 0)
		return NULL;

	set_C = isl_map_range(isl_map_copy(old_map));
	read_map = isl_map_copy(acc->sink.map);
	read_map = isl_map_intersect_domain(read_map, set_C);
	write_map = isl_map_copy(acc->source[pos].map);
	dep_map = isl_map_domain_product(write_map, read_map);
	dep_map = isl_set_unwrap(isl_map_domain(dep_map));
	space = isl_space_join(isl_map_get_space(old_map),
				isl_space_reverse(isl_map_get_space(dep_map)));
	equal = isl_map_from_basic_map(isl_basic_map_equal(space, depth));
	map = isl_map_range_product(equal, isl_map_copy(old_map));
	map = isl_map_uncurry(map);
	map = isl_map_intersect_domain_factor_range(map, dep_map);

	return map;
}

/* After the dependences derived from a must-source have been computed
 * at a certain level, check if any of the sources of the must-dependences
 * may be coscheduled with other sources.
 * If they are any such sources, then there is no way of determining
 * which of the sources actually comes last and the must-dependences
 * need to be turned into may-dependences, while dependences from
 * the other sources need to be added to the may-dependences as well.
 * "acc" describes the sources and a callback for checking whether
 * two sources may be coscheduled.  If acc->coscheduled is NULL then
 * the sources are assumed not to be coscheduled.
 * "must_rel" and "may_rel" describe the must and may-dependence relations
 * computed at the current level for the must-sources.  Some of the dependences
 * may be moved from "must_rel" to "may_rel".
 * "flow" contains all dependences computed so far (apart from those
 * in "must_rel" and "may_rel") and may be updated with additional
 * dependences derived from may-sources.
 *
 * In particular, consider all the must-sources with a non-empty
 * dependence relation in "must_rel".  They are considered in reverse
 * order because that is the order in which they are considered in the caller.
 * If any of the must-sources are coscheduled, then the last one
 * is the one that will have a corresponding dependence relation.
 * For each must-source i, consider both all the previous must-sources
 * and all the may-sources.  If any of those may be coscheduled with
 * must-source i, then compute the coscheduled instances that access
 * the same memory elements.  The result is a relation [T -> S] -> K.
 * The projection onto T -> K is a subset of the must-dependence relation
 * that needs to be turned into may-dependences.
 * The projection onto S -> K needs to be added to the may-dependences
 * of source S.
 * Since a given must-source instance may be coscheduled with several
 * other source instances, the dependences that need to be turned
 * into may-dependences are first collected and only actually removed
 * from the must-dependences after all other sources have been considered.
 */
static __isl_give isl_flow *handle_coscheduled(__isl_keep isl_access_info *acc,
	__isl_keep isl_map **must_rel, __isl_keep isl_map **may_rel,
	__isl_take isl_flow *flow)
{
	int i, j;

	if (!acc->coscheduled)
		return flow;
	for (i = acc->n_must - 1; i >= 0; --i) {
		isl_map *move;

		if (isl_map_plain_is_empty(must_rel[i]))
			continue;
		move = isl_map_empty(isl_map_get_space(must_rel[i]));
		for (j = i - 1; j >= 0; --j) {
			int depth;
			isl_bool coscheduled;
			isl_map *map, *factor;

			coscheduled = acc->coscheduled(acc->source[i].data,
							acc->source[j].data);
			if (coscheduled < 0) {
				isl_map_free(move);
				return isl_flow_free(flow);
			}
			if (!coscheduled)
				continue;
			depth = acc->level_before(acc->source[i].data,
						acc->source[j].data) / 2;
			map = coscheduled_source(acc, must_rel[i], j, depth);
			factor = isl_map_domain_factor_range(isl_map_copy(map));
			may_rel[j] = isl_map_union(may_rel[j], factor);
			map = isl_map_domain_factor_domain(map);
			move = isl_map_union(move, map);
		}
		for (j = 0; j < acc->n_may; ++j) {
			int depth, pos;
			isl_bool coscheduled;
			isl_map *map, *factor;

			pos = acc->n_must + j;
			coscheduled = acc->coscheduled(acc->source[i].data,
							acc->source[pos].data);
			if (coscheduled < 0) {
				isl_map_free(move);
				return isl_flow_free(flow);
			}
			if (!coscheduled)
				continue;
			depth = acc->level_before(acc->source[i].data,
						acc->source[pos].data) / 2;
			map = coscheduled_source(acc, must_rel[i], pos, depth);
			factor = isl_map_domain_factor_range(isl_map_copy(map));
			pos = 2 * acc->n_must + j;
			flow->dep[pos].map = isl_map_union(flow->dep[pos].map,
							    factor);
			map = isl_map_domain_factor_domain(map);
			move = isl_map_union(move, map);
		}
		must_rel[i] = isl_map_subtract(must_rel[i], isl_map_copy(move));
		may_rel[i] = isl_map_union(may_rel[i], move);
	}

	return flow;
}

/* Compute dependences for the case where all accesses are "may"
 * accesses, which boils down to computing memory based dependences.
 * The generic algorithm would also work in this case, but it would
 * be overkill to use it.
 */
static __isl_give isl_flow *compute_mem_based_dependences(
	__isl_keep isl_access_info *acc)
{
	int i;
	isl_set *mustdo;
	isl_set *maydo;
	isl_flow *res;

	res = isl_flow_alloc(acc);
	if (!res)
		return NULL;

	mustdo = isl_map_domain(isl_map_copy(acc->sink.map));
	maydo = isl_set_copy(mustdo);

	for (i = 0; i < acc->n_may; ++i) {
		int plevel;
		int is_before;
		isl_space *space;
		isl_map *before;
		isl_map *dep;

		plevel = acc->level_before(acc->source[i].data, acc->sink.data);
		if (plevel < 0)
			goto error;

		is_before = plevel & 1;
		plevel >>= 1;

		space = isl_map_get_space(res->dep[i].map);
		if (is_before)
			before = isl_map_lex_le_first(space, plevel);
		else
			before = isl_map_lex_lt_first(space, plevel);
		dep = isl_map_apply_range(isl_map_copy(acc->source[i].map),
			isl_map_reverse(isl_map_copy(acc->sink.map)));
		dep = isl_map_intersect(dep, before);
		mustdo = isl_set_subtract(mustdo,
					    isl_map_range(isl_map_copy(dep)));
		res->dep[i].map = isl_map_union(res->dep[i].map, dep);
	}

	res->may_no_source = isl_set_subtract(maydo, isl_set_copy(mustdo));
	res->must_no_source = mustdo;

	return res;
error:
	isl_set_free(mustdo);
	isl_set_free(maydo);
	isl_flow_free(res);
	return NULL;
}

/* Compute dependences for the case where there is at least one
 * "must" access.
 *
 * The core algorithm considers all levels in which a source may precede
 * the sink, where a level may either be a statement level or a loop level.
 * The outermost statement level is 1, the first loop level is 2, etc...
 * The algorithm basically does the following:
 * for all levels l of the read access from innermost to outermost
 *	for all sources w that may precede the sink access at that level
 *	    compute the last iteration of the source that precedes the sink access
 *					    at that level
 *	    add result to possible last accesses at level l of source w
 *	    for all sources w2 that we haven't considered yet at this level that may
 *					    also precede the sink access
 *		for all levels l2 of w from l to innermost
 *		    for all possible last accesses dep of w at l
 *			compute last iteration of w2 between the source and sink
 *								of dep
 *			add result to possible last accesses at level l of write w2
 *			and replace possible last accesses dep by the remainder
 *
 *
 * The above algorithm is applied to the must access.  During the course
 * of the algorithm, we keep track of sink iterations that still
 * need to be considered.  These iterations are split into those that
 * haven't been matched to any source access (mustdo) and those that have only
 * been matched to may accesses (maydo).
 * At the end of each level, must-sources and may-sources that are coscheduled
 * with the sources of the must-dependences at that level are considered.
 * If any coscheduled instances are found, then corresponding may-dependences
 * are added and the original must-dependences are turned into may-dependences.
 * Afterwards, the may accesses that occur after must-dependence sources
 * are considered.
 * In particular, we consider may accesses that precede the remaining
 * sink iterations, moving elements from mustdo to maydo when appropriate,
 * and may accesses that occur between a must source and a sink of any 
 * dependences found at the current level, turning must dependences into
 * may dependences when appropriate.
 * 
 */
static __isl_give isl_flow *compute_val_based_dependences(
	__isl_keep isl_access_info *acc)
{
	isl_ctx *ctx;
	isl_flow *res;
	isl_set *mustdo = NULL;
	isl_set *maydo = NULL;
	int level, j;
	isl_size n_in;
	int depth;
	isl_map **must_rel = NULL;
	isl_map **may_rel = NULL;

	if (!acc)
		return NULL;

	res = isl_flow_alloc(acc);
	if (!res)
		goto error;
	ctx = isl_map_get_ctx(acc->sink.map);

	n_in = isl_map_dim(acc->sink.map, isl_dim_in);
	if (n_in < 0)
		goto error;
	depth = 2 * n_in + 1;
	mustdo = isl_map_domain(isl_map_copy(acc->sink.map));
	maydo = isl_set_empty(isl_set_get_space(mustdo));
	if (!mustdo || !maydo)
		goto error;
	if (isl_set_plain_is_empty(mustdo))
		goto done;

	must_rel = isl_calloc_array(ctx, struct isl_map *, acc->n_must);
	may_rel = isl_calloc_array(ctx, struct isl_map *, acc->n_must);
	if (!must_rel || !may_rel)
		goto error;

	for (level = depth; level >= 1; --level) {
		for (j = acc->n_must-1; j >=0; --j) {
			isl_space *space;
			space = isl_map_get_space(res->dep[2 * j].map);
			must_rel[j] = isl_map_empty(space);
			may_rel[j] = isl_map_copy(must_rel[j]);
		}

		for (j = acc->n_must - 1; j >= 0; --j) {
			struct isl_map *T;
			struct isl_set *rest;
			int plevel;

			plevel = acc->level_before(acc->source[j].data,
						     acc->sink.data);
			if (plevel < 0)
				goto error;
			if (!can_precede_at_level(plevel, level))
				continue;

			T = last_source(acc, mustdo, j, level, &rest);
			must_rel[j] = isl_map_union_disjoint(must_rel[j], T);
			mustdo = rest;

			if (intermediate_sources(acc, must_rel, j, level) < 0)
				goto error;

			T = last_source(acc, maydo, j, level, &rest);
			may_rel[j] = isl_map_union_disjoint(may_rel[j], T);
			maydo = rest;

			if (intermediate_sources(acc, may_rel, j, level) < 0)
				goto error;

			if (isl_set_plain_is_empty(mustdo) &&
			    isl_set_plain_is_empty(maydo))
				break;
		}
		for (j = j - 1; j >= 0; --j) {
			int plevel;

			plevel = acc->level_before(acc->source[j].data,
						     acc->sink.data);
			if (plevel < 0)
				goto error;
			if (!can_precede_at_level(plevel, level))
				continue;

			if (intermediate_sources(acc, must_rel, j, level) < 0)
				goto error;
			if (intermediate_sources(acc, may_rel, j, level) < 0)
				goto error;
		}

		res = handle_coscheduled(acc, must_rel, may_rel, res);
		if (!res)
			goto error;

		for (j = 0; j < acc->n_may; ++j) {
			int plevel;
			isl_map *T;
			isl_set *ran;

			plevel = acc->level_before(acc->source[acc->n_must + j].data,
						     acc->sink.data);
			if (plevel < 0)
				goto error;
			if (!can_precede_at_level(plevel, level))
				continue;

			T = all_sources(acc, isl_set_copy(maydo), j, level);
			res->dep[2 * acc->n_must + j].map =
			    isl_map_union(res->dep[2 * acc->n_must + j].map, T);
			T = all_sources(acc, isl_set_copy(mustdo), j, level);
			ran = isl_map_range(isl_map_copy(T));
			res->dep[2 * acc->n_must + j].map =
			    isl_map_union(res->dep[2 * acc->n_must + j].map, T);
			mustdo = isl_set_subtract(mustdo, isl_set_copy(ran));
			maydo = isl_set_union_disjoint(maydo, ran);

			T = res->dep[2 * acc->n_must + j].map;
			T = all_intermediate_sources(acc, T, must_rel, may_rel,
							j, level);
			res->dep[2 * acc->n_must + j].map = T;
		}

		for (j = acc->n_must - 1; j >= 0; --j) {
			res->dep[2 * j].map =
				isl_map_union_disjoint(res->dep[2 * j].map,
							     must_rel[j]);
			res->dep[2 * j + 1].map =
				isl_map_union_disjoint(res->dep[2 * j + 1].map,
							     may_rel[j]);
		}

		if (isl_set_plain_is_empty(mustdo) &&
		    isl_set_plain_is_empty(maydo))
			break;
	}

	free(must_rel);
	free(may_rel);
done:
	res->must_no_source = mustdo;
	res->may_no_source = maydo;
	return res;
error:
	if (must_rel)
		for (j = 0; j < acc->n_must; ++j)
			isl_map_free(must_rel[j]);
	if (may_rel)
		for (j = 0; j < acc->n_must; ++j)
			isl_map_free(may_rel[j]);
	isl_flow_free(res);
	isl_set_free(mustdo);
	isl_set_free(maydo);
	free(must_rel);
	free(may_rel);
	return NULL;
}

/* Given a "sink" access, a list of n "source" accesses,
 * compute for each iteration of the sink access
 * and for each element accessed by that iteration,
 * the source access in the list that last accessed the
 * element accessed by the sink access before this sink access.
 * Each access is given as a map from the loop iterators
 * to the array indices.
 * The result is a list of n relations between source and sink
 * iterations and a subset of the domain of the sink access,
 * corresponding to those iterations that access an element
 * not previously accessed.
 *
 * To deal with multi-valued sink access relations, the sink iteration
 * domain is first extended with dimensions that correspond to the data
 * space.  However, these extra dimensions are not projected out again.
 * It is up to the caller to decide whether these dimensions should be kept.
 */
static __isl_give isl_flow *access_info_compute_flow_core(
	__isl_take isl_access_info *acc)
{
	struct isl_flow *res = NULL;

	if (!acc)
		return NULL;

	acc->sink.map = isl_map_range_map(acc->sink.map);
	if (!acc->sink.map)
		goto error;

	if (acc->n_must == 0)
		res = compute_mem_based_dependences(acc);
	else {
		acc = isl_access_info_sort_sources(acc);
		res = compute_val_based_dependences(acc);
	}
	acc = isl_access_info_free(acc);
	if (!res)
		return NULL;
	if (!res->must_no_source || !res->may_no_source)
		goto error;
	return res;
error:
	isl_access_info_free(acc);
	isl_flow_free(res);
	return NULL;
}

/* Given a "sink" access, a list of n "source" accesses,
 * compute for each iteration of the sink access
 * and for each element accessed by that iteration,
 * the source access in the list that last accessed the
 * element accessed by the sink access before this sink access.
 * Each access is given as a map from the loop iterators
 * to the array indices.
 * The result is a list of n relations between source and sink
 * iterations and a subset of the domain of the sink access,
 * corresponding to those iterations that access an element
 * not previously accessed.
 *
 * To deal with multi-valued sink access relations,
 * access_info_compute_flow_core extends the sink iteration domain
 * with dimensions that correspond to the data space.  These extra dimensions
 * are projected out from the result of access_info_compute_flow_core.
 */
__isl_give isl_flow *isl_access_info_compute_flow(__isl_take isl_access_info *acc)
{
	int j;
	struct isl_flow *res;

	if (!acc)
		return NULL;

	acc->domain_map = isl_map_domain_map(isl_map_copy(acc->sink.map));
	res = access_info_compute_flow_core(acc);
	if (!res)
		return NULL;

	for (j = 0; j < res->n_source; ++j) {
		res->dep[j].map = isl_map_range_factor_domain(res->dep[j].map);
		if (!res->dep[j].map)
			goto error;
	}

	return res;
error:
	isl_flow_free(res);
	return NULL;
}


/* Keep track of some information about a schedule for a given
 * access.  In particular, keep track of which dimensions
 * have a constant value and of the actual constant values.
 */
struct isl_sched_info {
	int *is_cst;
	isl_vec *cst;
};

static void sched_info_free(__isl_take struct isl_sched_info *info)
{
	if (!info)
		return;
	isl_vec_free(info->cst);
	free(info->is_cst);
	free(info);
}

/* Extract information on the constant dimensions of the schedule
 * for a given access.  The "map" is of the form
 *
 *	[S -> D] -> A
 *
 * with S the schedule domain, D the iteration domain and A the data domain.
 */
static __isl_give struct isl_sched_info *sched_info_alloc(
	__isl_keep isl_map *map)
{
	isl_ctx *ctx;
	isl_space *space;
	struct isl_sched_info *info;
	int i;
	isl_size n;

	if (!map)
		return NULL;

	space = isl_space_unwrap(isl_space_domain(isl_map_get_space(map)));
	if (!space)
		return NULL;
	n = isl_space_dim(space, isl_dim_in);
	isl_space_free(space);
	if (n < 0)
		return NULL;

	ctx = isl_map_get_ctx(map);
	info = isl_alloc_type(ctx, struct isl_sched_info);
	if (!info)
		return NULL;
	info->is_cst = isl_alloc_array(ctx, int, n);
	info->cst = isl_vec_alloc(ctx, n);
	if (n && (!info->is_cst || !info->cst))
		goto error;

	for (i = 0; i < n; ++i) {
		isl_val *v;

		v = isl_map_plain_get_val_if_fixed(map, isl_dim_in, i);
		if (!v)
			goto error;
		info->is_cst[i] = !isl_val_is_nan(v);
		if (info->is_cst[i])
			info->cst = isl_vec_set_element_val(info->cst, i, v);
		else
			isl_val_free(v);
	}

	return info;
error:
	sched_info_free(info);
	return NULL;
}

/* The different types of access relations that isl_union_access_info
 * keeps track of.

 * "isl_access_sink" represents the sink accesses.
 * "isl_access_must_source" represents the definite source accesses.
 * "isl_access_may_source" represents the possible source accesses.
 * "isl_access_kill" represents the kills.
 *
 * isl_access_sink is sometimes treated differently and
 * should therefore appear first.
 */
enum isl_access_type {
	isl_access_sink,
	isl_access_must_source,
	isl_access_may_source,
	isl_access_kill,
	isl_access_end
};

/* This structure represents the input for a dependence analysis computation.
 *
 * "access" contains the access relations.
 *
 * "schedule" or "schedule_map" represents the execution order.
 * Exactly one of these fields should be NULL.  The other field
 * determines the execution order.
 *
 * The domains of these four maps refer to the same iteration spaces(s).
 * The ranges of the first three maps also refer to the same data space(s).
 *
 * After a call to isl_union_access_info_introduce_schedule,
 * the "schedule_map" field no longer contains useful information.
 */
struct isl_union_access_info {
	isl_union_map *access[isl_access_end];

	isl_schedule *schedule;
	isl_union_map *schedule_map;
};

/* Free "access" and return NULL.
 */
__isl_null isl_union_access_info *isl_union_access_info_free(
	__isl_take isl_union_access_info *access)
{
	enum isl_access_type i;

	if (!access)
		return NULL;

	for (i = isl_access_sink; i < isl_access_end; ++i)
		isl_union_map_free(access->access[i]);
	isl_schedule_free(access->schedule);
	isl_union_map_free(access->schedule_map);
	free(access);

	return NULL;
}

/* Return the isl_ctx to which "access" belongs.
 */
isl_ctx *isl_union_access_info_get_ctx(__isl_keep isl_union_access_info *access)
{
	if (!access)
		return NULL;
	return isl_union_map_get_ctx(access->access[isl_access_sink]);
}

/* Construct an empty (invalid) isl_union_access_info object.
 * The caller is responsible for setting the sink access relation and
 * initializing all the other fields, e.g., by calling
 * isl_union_access_info_init.
 */
static __isl_give isl_union_access_info *isl_union_access_info_alloc(
	isl_ctx *ctx)
{
	return isl_calloc_type(ctx, isl_union_access_info);
}

/* Initialize all the fields of "info", except the sink access relation,
 * which is assumed to have been set by the caller.
 *
 * By default, we use the schedule field of the isl_union_access_info,
 * but this may be overridden by a call
 * to isl_union_access_info_set_schedule_map.
 */
static __isl_give isl_union_access_info *isl_union_access_info_init(
	__isl_take isl_union_access_info *info)
{
	isl_space *space;
	isl_union_map *empty;
	enum isl_access_type i;

	if (!info)
		return NULL;
	if (!info->access[isl_access_sink])
		return isl_union_access_info_free(info);

	space = isl_union_map_get_space(info->access[isl_access_sink]);
	empty = isl_union_map_empty(isl_space_copy(space));
	for (i = isl_access_sink + 1; i < isl_access_end; ++i)
		if (!info->access[i])
			info->access[i] = isl_union_map_copy(empty);
	isl_union_map_free(empty);
	if (!info->schedule && !info->schedule_map)
		info->schedule = isl_schedule_empty(isl_space_copy(space));
	isl_space_free(space);

	for (i = isl_access_sink + 1; i < isl_access_end; ++i)
		if (!info->access[i])
			return isl_union_access_info_free(info);
	if (!info->schedule && !info->schedule_map)
		return isl_union_access_info_free(info);

	return info;
}

/* Create a new isl_union_access_info with the given sink accesses and
 * and no other accesses or schedule information.
 */
__isl_give isl_union_access_info *isl_union_access_info_from_sink(
	__isl_take isl_union_map *sink)
{
	isl_ctx *ctx;
	isl_union_access_info *access;

	if (!sink)
		return NULL;
	ctx = isl_union_map_get_ctx(sink);
	access = isl_union_access_info_alloc(ctx);
	if (!access)
		goto error;
	access->access[isl_access_sink] = sink;
	return isl_union_access_info_init(access);
error:
	isl_union_map_free(sink);
	return NULL;
}

/* Replace the access relation of type "type" of "info" by "access".
 */
static __isl_give isl_union_access_info *isl_union_access_info_set(
	__isl_take isl_union_access_info *info,
	enum isl_access_type type, __isl_take isl_union_map *access)
{
	if (!info || !access)
		goto error;

	isl_union_map_free(info->access[type]);
	info->access[type] = access;

	return info;
error:
	isl_union_access_info_free(info);
	isl_union_map_free(access);
	return NULL;
}

/* Replace the definite source accesses of "access" by "must_source".
 */
__isl_give isl_union_access_info *isl_union_access_info_set_must_source(
	__isl_take isl_union_access_info *access,
	__isl_take isl_union_map *must_source)
{
	return isl_union_access_info_set(access, isl_access_must_source,
					must_source);
}

/* Replace the possible source accesses of "access" by "may_source".
 */
__isl_give isl_union_access_info *isl_union_access_info_set_may_source(
	__isl_take isl_union_access_info *access,
	__isl_take isl_union_map *may_source)
{
	return isl_union_access_info_set(access, isl_access_may_source,
					may_source);
}

/* Replace the kills of "info" by "kill".
 */
__isl_give isl_union_access_info *isl_union_access_info_set_kill(
	__isl_take isl_union_access_info *info, __isl_take isl_union_map *kill)
{
	return isl_union_access_info_set(info, isl_access_kill, kill);
}

/* Return the access relation of type "type" of "info".
 */
static __isl_give isl_union_map *isl_union_access_info_get(
	__isl_keep isl_union_access_info *info, enum isl_access_type type)
{
	if (!info)
		return NULL;
	return isl_union_map_copy(info->access[type]);
}

/* Return the definite source accesses of "info".
 */
__isl_give isl_union_map *isl_union_access_info_get_must_source(
	__isl_keep isl_union_access_info *info)
{
	return isl_union_access_info_get(info, isl_access_must_source);
}

/* Return the possible source accesses of "info".
 */
__isl_give isl_union_map *isl_union_access_info_get_may_source(
	__isl_keep isl_union_access_info *info)
{
	return isl_union_access_info_get(info, isl_access_may_source);
}

/* Return the kills of "info".
 */
__isl_give isl_union_map *isl_union_access_info_get_kill(
	__isl_keep isl_union_access_info *info)
{
	return isl_union_access_info_get(info, isl_access_kill);
}

/* Does "info" specify any kills?
 */
static isl_bool isl_union_access_has_kill(
	__isl_keep isl_union_access_info *info)
{
	isl_bool empty;

	if (!info)
		return isl_bool_error;
	empty = isl_union_map_is_empty(info->access[isl_access_kill]);
	return isl_bool_not(empty);
}

/* Replace the schedule of "access" by "schedule".
 * Also free the schedule_map in case it was set last.
 */
__isl_give isl_union_access_info *isl_union_access_info_set_schedule(
	__isl_take isl_union_access_info *access,
	__isl_take isl_schedule *schedule)
{
	if (!access || !schedule)
		goto error;

	access->schedule_map = isl_union_map_free(access->schedule_map);
	isl_schedule_free(access->schedule);
	access->schedule = schedule;

	return access;
error:
	isl_union_access_info_free(access);
	isl_schedule_free(schedule);
	return NULL;
}

/* Replace the schedule map of "access" by "schedule_map".
 * Also free the schedule in case it was set last.
 */
__isl_give isl_union_access_info *isl_union_access_info_set_schedule_map(
	__isl_take isl_union_access_info *access,
	__isl_take isl_union_map *schedule_map)
{
	if (!access || !schedule_map)
		goto error;

	isl_union_map_free(access->schedule_map);
	access->schedule = isl_schedule_free(access->schedule);
	access->schedule_map = schedule_map;

	return access;
error:
	isl_union_access_info_free(access);
	isl_union_map_free(schedule_map);
	return NULL;
}

__isl_give isl_union_access_info *isl_union_access_info_copy(
	__isl_keep isl_union_access_info *access)
{
	isl_union_access_info *copy;
	enum isl_access_type i;

	if (!access)
		return NULL;
	copy = isl_union_access_info_from_sink(
		    isl_union_map_copy(access->access[isl_access_sink]));
	for (i = isl_access_sink + 1; i < isl_access_end; ++i)
		copy = isl_union_access_info_set(copy, i,
					isl_union_map_copy(access->access[i]));
	if (access->schedule)
		copy = isl_union_access_info_set_schedule(copy,
				isl_schedule_copy(access->schedule));
	else
		copy = isl_union_access_info_set_schedule_map(copy,
				isl_union_map_copy(access->schedule_map));

	return copy;
}

#undef BASE
#define BASE union_map
#include "print_yaml_field_templ.c"

/* An enumeration of the various keys that may appear in a YAML mapping
 * of an isl_union_access_info object.
 * The keys for the access relation types are assumed to have the same values
 * as the access relation types in isl_access_type.
 */
enum isl_ai_key {
	isl_ai_key_error = -1,
	isl_ai_key_sink = isl_access_sink,
	isl_ai_key_must_source = isl_access_must_source,
	isl_ai_key_may_source = isl_access_may_source,
	isl_ai_key_kill = isl_access_kill,
	isl_ai_key_schedule_map,
	isl_ai_key_schedule,
	isl_ai_key_end
};

/* Textual representations of the YAML keys for an isl_union_access_info
 * object.
 */
static char *key_str[] = {
	[isl_ai_key_sink] = "sink",
	[isl_ai_key_must_source] = "must_source",
	[isl_ai_key_may_source] = "may_source",
	[isl_ai_key_kill] = "kill",
	[isl_ai_key_schedule_map] = "schedule_map",
	[isl_ai_key_schedule] = "schedule",
};

/* Print a key-value pair corresponding to the access relation of type "type"
 * of a YAML mapping of "info" to "p".
 *
 * The sink access relation is always printed, but any other access relation
 * is only printed if it is non-empty.
 */
static __isl_give isl_printer *print_access_field(__isl_take isl_printer *p,
	__isl_keep isl_union_access_info *info, enum isl_access_type type)
{
	if (type != isl_access_sink) {
		isl_bool empty;

		empty = isl_union_map_is_empty(info->access[type]);
		if (empty < 0)
			return isl_printer_free(p);
		if (empty)
			return p;
	}
	return print_yaml_field_union_map(p, key_str[type], info->access[type]);
}

/* Print the information contained in "access" to "p".
 * The information is printed as a YAML document.
 */
__isl_give isl_printer *isl_printer_print_union_access_info(
	__isl_take isl_printer *p, __isl_keep isl_union_access_info *access)
{
	enum isl_access_type i;

	if (!access)
		return isl_printer_free(p);

	p = isl_printer_yaml_start_mapping(p);
	for (i = isl_access_sink; i < isl_access_end; ++i)
		p = print_access_field(p, access, i);
	if (access->schedule) {
		p = isl_printer_print_str(p, key_str[isl_ai_key_schedule]);
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_schedule(p, access->schedule);
		p = isl_printer_yaml_next(p);
	} else {
		p = print_yaml_field_union_map(p,
			key_str[isl_ai_key_schedule_map], access->schedule_map);
	}
	p = isl_printer_yaml_end_mapping(p);

	return p;
}

/* Return a string representation of the information in "access".
 * The information is printed in flow format.
 */
__isl_give char *isl_union_access_info_to_str(
	__isl_keep isl_union_access_info *access)
{
	isl_printer *p;
	char *s;

	if (!access)
		return NULL;

	p = isl_printer_to_str(isl_union_access_info_get_ctx(access));
	p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_FLOW);
	p = isl_printer_print_union_access_info(p, access);
	s = isl_printer_get_str(p);
	isl_printer_free(p);

	return s;
}

#undef KEY
#define KEY enum isl_ai_key
#undef KEY_ERROR
#define KEY_ERROR isl_ai_key_error
#undef KEY_END
#define KEY_END isl_ai_key_end
#include "extract_key.c"

#undef BASE
#define BASE union_map
#include "read_in_string_templ.c"

/* Read an isl_union_access_info object from "s".
 *
 * Start off with an empty (invalid) isl_union_access_info object and
 * then fill up the fields based on the input.
 * The input needs to contain at least a description of the sink
 * access relation as well as some form of schedule.
 * The other access relations are set to empty relations
 * by isl_union_access_info_init if they are not specified in the input.
 */
__isl_give isl_union_access_info *isl_stream_read_union_access_info(
	isl_stream *s)
{
	isl_ctx *ctx;
	isl_union_access_info *info;
	int more;
	int sink_set = 0;
	int schedule_set = 0;

	if (isl_stream_yaml_read_start_mapping(s))
		return NULL;

	ctx = isl_stream_get_ctx(s);
	info = isl_union_access_info_alloc(ctx);
	while ((more = isl_stream_yaml_next(s)) > 0) {
		enum isl_ai_key key;
		isl_union_map *access, *schedule_map;
		isl_schedule *schedule;

		key = get_key(s);
		if (isl_stream_yaml_next(s) < 0)
			return isl_union_access_info_free(info);
		switch (key) {
		case isl_ai_key_end:
		case isl_ai_key_error:
			return isl_union_access_info_free(info);
		case isl_ai_key_sink:
			sink_set = 1;
		case isl_ai_key_must_source:
		case isl_ai_key_may_source:
		case isl_ai_key_kill:
			access = read_union_map(s);
			info = isl_union_access_info_set(info, key, access);
			if (!info)
				return NULL;
			break;
		case isl_ai_key_schedule_map:
			schedule_set = 1;
			schedule_map = read_union_map(s);
			info = isl_union_access_info_set_schedule_map(info,
								schedule_map);
			if (!info)
				return NULL;
			break;
		case isl_ai_key_schedule:
			schedule_set = 1;
			schedule = isl_stream_read_schedule(s);
			info = isl_union_access_info_set_schedule(info,
								schedule);
			if (!info)
				return NULL;
			break;
		}
	}
	if (more < 0)
		return isl_union_access_info_free(info);

	if (isl_stream_yaml_read_end_mapping(s) < 0) {
		isl_stream_error(s, NULL, "unexpected extra elements");
		return isl_union_access_info_free(info);
	}

	if (!sink_set) {
		isl_stream_error(s, NULL, "no sink specified");
		return isl_union_access_info_free(info);
	}

	if (!schedule_set) {
		isl_stream_error(s, NULL, "no schedule specified");
		return isl_union_access_info_free(info);
	}

	return isl_union_access_info_init(info);
}

/* Read an isl_union_access_info object from the file "input".
 */
__isl_give isl_union_access_info *isl_union_access_info_read_from_file(
	isl_ctx *ctx, FILE *input)
{
	isl_stream *s;
	isl_union_access_info *access;

	s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	access = isl_stream_read_union_access_info(s);
	isl_stream_free(s);

	return access;
}

/* Update the fields of "access" such that they all have the same parameters,
 * keeping in mind that the schedule_map field may be NULL and ignoring
 * the schedule field.
 */
static __isl_give isl_union_access_info *isl_union_access_info_align_params(
	__isl_take isl_union_access_info *access)
{
	isl_space *space;
	enum isl_access_type i;

	if (!access)
		return NULL;

	space = isl_union_map_get_space(access->access[isl_access_sink]);
	for (i = isl_access_sink + 1; i < isl_access_end; ++i)
		space = isl_space_align_params(space,
				isl_union_map_get_space(access->access[i]));
	if (access->schedule_map)
		space = isl_space_align_params(space,
				isl_union_map_get_space(access->schedule_map));
	for (i = isl_access_sink; i < isl_access_end; ++i)
		access->access[i] =
			isl_union_map_align_params(access->access[i],
							isl_space_copy(space));
	if (!access->schedule_map) {
		isl_space_free(space);
	} else {
		access->schedule_map =
		    isl_union_map_align_params(access->schedule_map, space);
		if (!access->schedule_map)
			return isl_union_access_info_free(access);
	}

	for (i = isl_access_sink; i < isl_access_end; ++i)
		if (!access->access[i])
			return isl_union_access_info_free(access);

	return access;
}

/* Prepend the schedule dimensions to the iteration domains.
 *
 * That is, if the schedule is of the form
 *
 *	D -> S
 *
 * while the access relations are of the form
 *
 *	D -> A
 *
 * then the updated access relations are of the form
 *
 *	[S -> D] -> A
 *
 * The schedule map is also replaced by the map
 *
 *	[S -> D] -> D
 *
 * that is used during the internal computation.
 * Neither the original schedule map nor this updated schedule map
 * are used after the call to this function.
 */
static __isl_give isl_union_access_info *
isl_union_access_info_introduce_schedule(
	__isl_take isl_union_access_info *access)
{
	isl_union_map *sm;
	enum isl_access_type i;

	if (!access)
		return NULL;

	sm = isl_union_map_reverse(access->schedule_map);
	sm = isl_union_map_range_map(sm);
	for (i = isl_access_sink; i < isl_access_end; ++i)
		access->access[i] =
			isl_union_map_apply_range(isl_union_map_copy(sm),
						access->access[i]);
	access->schedule_map = sm;

	for (i = isl_access_sink; i < isl_access_end; ++i)
		if (!access->access[i])
			return isl_union_access_info_free(access);
	if (!access->schedule_map)
		return isl_union_access_info_free(access);

	return access;
}

/* This structure represents the result of a dependence analysis computation.
 *
 * "must_dep" represents the full definite dependences
 * "may_dep" represents the full non-definite dependences.
 * Both are of the form
 *
 *	[Source] -> [[Sink -> Data]]
 *
 * (after the schedule dimensions have been projected out).
 * "must_no_source" represents the subset of the sink accesses for which
 * definitely no source was found.
 * "may_no_source" represents the subset of the sink accesses for which
 * possibly, but not definitely, no source was found.
 */
struct isl_union_flow {
	isl_union_map *must_dep;
	isl_union_map *may_dep;
	isl_union_map *must_no_source;
	isl_union_map *may_no_source;
};

/* Return the isl_ctx to which "flow" belongs.
 */
isl_ctx *isl_union_flow_get_ctx(__isl_keep isl_union_flow *flow)
{
	return flow ? isl_union_map_get_ctx(flow->must_dep) : NULL;
}

/* Free "flow" and return NULL.
 */
__isl_null isl_union_flow *isl_union_flow_free(__isl_take isl_union_flow *flow)
{
	if (!flow)
		return NULL;
	isl_union_map_free(flow->must_dep);
	isl_union_map_free(flow->may_dep);
	isl_union_map_free(flow->must_no_source);
	isl_union_map_free(flow->may_no_source);
	free(flow);
	return NULL;
}

void isl_union_flow_dump(__isl_keep isl_union_flow *flow)
{
	if (!flow)
		return;

	fprintf(stderr, "must dependences: ");
	isl_union_map_dump(flow->must_dep);
	fprintf(stderr, "may dependences: ");
	isl_union_map_dump(flow->may_dep);
	fprintf(stderr, "must no source: ");
	isl_union_map_dump(flow->must_no_source);
	fprintf(stderr, "may no source: ");
	isl_union_map_dump(flow->may_no_source);
}

/* Return the full definite dependences in "flow", with accessed elements.
 */
__isl_give isl_union_map *isl_union_flow_get_full_must_dependence(
	__isl_keep isl_union_flow *flow)
{
	if (!flow)
		return NULL;
	return isl_union_map_copy(flow->must_dep);
}

/* Return the full possible dependences in "flow", including the definite
 * dependences, with accessed elements.
 */
__isl_give isl_union_map *isl_union_flow_get_full_may_dependence(
	__isl_keep isl_union_flow *flow)
{
	if (!flow)
		return NULL;
	return isl_union_map_union(isl_union_map_copy(flow->must_dep),
				    isl_union_map_copy(flow->may_dep));
}

/* Return the definite dependences in "flow", without the accessed elements.
 */
__isl_give isl_union_map *isl_union_flow_get_must_dependence(
	__isl_keep isl_union_flow *flow)
{
	isl_union_map *dep;

	if (!flow)
		return NULL;
	dep = isl_union_map_copy(flow->must_dep);
	return isl_union_map_range_factor_domain(dep);
}

/* Return the possible dependences in "flow", including the definite
 * dependences, without the accessed elements.
 */
__isl_give isl_union_map *isl_union_flow_get_may_dependence(
	__isl_keep isl_union_flow *flow)
{
	isl_union_map *dep;

	if (!flow)
		return NULL;
	dep = isl_union_map_union(isl_union_map_copy(flow->must_dep),
				    isl_union_map_copy(flow->may_dep));
	return isl_union_map_range_factor_domain(dep);
}

/* Return the non-definite dependences in "flow".
 */
static __isl_give isl_union_map *isl_union_flow_get_non_must_dependence(
	__isl_keep isl_union_flow *flow)
{
	if (!flow)
		return NULL;
	return isl_union_map_copy(flow->may_dep);
}

/* Return the subset of the sink accesses for which definitely
 * no source was found.
 */
__isl_give isl_union_map *isl_union_flow_get_must_no_source(
	__isl_keep isl_union_flow *flow)
{
	if (!flow)
		return NULL;
	return isl_union_map_copy(flow->must_no_source);
}

/* Return the subset of the sink accesses for which possibly
 * no source was found, including those for which definitely
 * no source was found.
 */
__isl_give isl_union_map *isl_union_flow_get_may_no_source(
	__isl_keep isl_union_flow *flow)
{
	if (!flow)
		return NULL;
	return isl_union_map_union(isl_union_map_copy(flow->must_no_source),
				    isl_union_map_copy(flow->may_no_source));
}

/* Return the subset of the sink accesses for which possibly, but not
 * definitely, no source was found.
 */
static __isl_give isl_union_map *isl_union_flow_get_non_must_no_source(
	__isl_keep isl_union_flow *flow)
{
	if (!flow)
		return NULL;
	return isl_union_map_copy(flow->may_no_source);
}

/* Create a new isl_union_flow object, initialized with empty
 * dependence relations and sink subsets.
 */
static __isl_give isl_union_flow *isl_union_flow_alloc(
	__isl_take isl_space *space)
{
	isl_ctx *ctx;
	isl_union_map *empty;
	isl_union_flow *flow;

	if (!space)
		return NULL;
	ctx = isl_space_get_ctx(space);
	flow = isl_alloc_type(ctx, isl_union_flow);
	if (!flow)
		goto error;

	empty = isl_union_map_empty(space);
	flow->must_dep = isl_union_map_copy(empty);
	flow->may_dep = isl_union_map_copy(empty);
	flow->must_no_source = isl_union_map_copy(empty);
	flow->may_no_source = empty;

	if (!flow->must_dep || !flow->may_dep ||
	    !flow->must_no_source || !flow->may_no_source)
		return isl_union_flow_free(flow);

	return flow;
error:
	isl_space_free(space);
	return NULL;
}

/* Copy this isl_union_flow object.
 */
__isl_give isl_union_flow *isl_union_flow_copy(__isl_keep isl_union_flow *flow)
{
	isl_union_flow *copy;

	if (!flow)
		return NULL;

	copy = isl_union_flow_alloc(isl_union_map_get_space(flow->must_dep));

	if (!copy)
		return NULL;

	copy->must_dep = isl_union_map_union(copy->must_dep,
		isl_union_map_copy(flow->must_dep));
	copy->may_dep = isl_union_map_union(copy->may_dep,
		isl_union_map_copy(flow->may_dep));
	copy->must_no_source = isl_union_map_union(copy->must_no_source,
		isl_union_map_copy(flow->must_no_source));
	copy->may_no_source = isl_union_map_union(copy->may_no_source,
		isl_union_map_copy(flow->may_no_source));

	if (!copy->must_dep || !copy->may_dep ||
	    !copy->must_no_source || !copy->may_no_source)
		return isl_union_flow_free(copy);

	return copy;
}

/* Drop the schedule dimensions from the iteration domains in "flow".
 * In particular, the schedule dimensions have been prepended
 * to the iteration domains prior to the dependence analysis by
 * replacing the iteration domain D, by the wrapped map [S -> D].
 * Replace these wrapped maps by the original D.
 *
 * In particular, the dependences computed by access_info_compute_flow_core
 * are of the form
 *
 *	[S -> D] -> [[S' -> D'] -> A]
 *
 * The schedule dimensions are projected out by first currying the range,
 * resulting in
 *
 *	[S -> D] -> [S' -> [D' -> A]]
 *
 * and then computing the factor range
 *
 *	D -> [D' -> A]
 */
static __isl_give isl_union_flow *isl_union_flow_drop_schedule(
	__isl_take isl_union_flow *flow)
{
	if (!flow)
		return NULL;

	flow->must_dep = isl_union_map_range_curry(flow->must_dep);
	flow->must_dep = isl_union_map_factor_range(flow->must_dep);
	flow->may_dep = isl_union_map_range_curry(flow->may_dep);
	flow->may_dep = isl_union_map_factor_range(flow->may_dep);
	flow->must_no_source =
		isl_union_map_domain_factor_range(flow->must_no_source);
	flow->may_no_source =
		isl_union_map_domain_factor_range(flow->may_no_source);

	if (!flow->must_dep || !flow->may_dep ||
	    !flow->must_no_source || !flow->may_no_source)
		return isl_union_flow_free(flow);

	return flow;
}

struct isl_compute_flow_data {
	isl_union_map *must_source;
	isl_union_map *may_source;
	isl_union_flow *flow;

	int count;
	int must;
	isl_space *dim;
	struct isl_sched_info *sink_info;
	struct isl_sched_info **source_info;
	isl_access_info *accesses;
};

static isl_stat count_matching_array(__isl_take isl_map *map, void *user)
{
	int eq;
	isl_space *space;
	struct isl_compute_flow_data *data;

	data = (struct isl_compute_flow_data *)user;

	space = isl_space_range(isl_map_get_space(map));

	eq = isl_space_is_equal(space, data->dim);

	isl_space_free(space);
	isl_map_free(map);

	if (eq < 0)
		return isl_stat_error;
	if (eq)
		data->count++;

	return isl_stat_ok;
}

static isl_stat collect_matching_array(__isl_take isl_map *map, void *user)
{
	int eq;
	isl_space *space;
	struct isl_sched_info *info;
	struct isl_compute_flow_data *data;

	data = (struct isl_compute_flow_data *)user;

	space = isl_space_range(isl_map_get_space(map));

	eq = isl_space_is_equal(space, data->dim);

	isl_space_free(space);

	if (eq < 0)
		goto error;
	if (!eq) {
		isl_map_free(map);
		return isl_stat_ok;
	}

	info = sched_info_alloc(map);
	data->source_info[data->count] = info;

	data->accesses = isl_access_info_add_source(data->accesses,
						    map, data->must, info);

	data->count++;

	return isl_stat_ok;
error:
	isl_map_free(map);
	return isl_stat_error;
}

/* Determine the shared nesting level and the "textual order" of
 * the given accesses.
 *
 * We first determine the minimal schedule dimension for both accesses.
 *
 * If among those dimensions, we can find one where both have a fixed
 * value and if moreover those values are different, then the previous
 * dimension is the last shared nesting level and the textual order
 * is determined based on the order of the fixed values.
 * If no such fixed values can be found, then we set the shared
 * nesting level to the minimal schedule dimension, with no textual ordering.
 */
static int before(void *first, void *second)
{
	struct isl_sched_info *info1 = first;
	struct isl_sched_info *info2 = second;
	isl_size n1, n2;
	int i;

	n1 = isl_vec_size(info1->cst);
	n2 = isl_vec_size(info2->cst);
	if (n1 < 0 || n2 < 0)
		return -1;

	if (n2 < n1)
		n1 = n2;

	for (i = 0; i < n1; ++i) {
		int r;
		int cmp;

		if (!info1->is_cst[i])
			continue;
		if (!info2->is_cst[i])
			continue;
		cmp = isl_vec_cmp_element(info1->cst, info2->cst, i);
		if (cmp == 0)
			continue;

		r = 2 * i + (cmp < 0);

		return r;
	}

	return 2 * n1;
}

/* Check if the given two accesses may be coscheduled.
 * If so, return isl_bool_true.  Otherwise return isl_bool_false.
 *
 * Two accesses may only be coscheduled if the fixed schedule
 * coordinates have the same values.
 */
static isl_bool coscheduled(void *first, void *second)
{
	struct isl_sched_info *info1 = first;
	struct isl_sched_info *info2 = second;
	isl_size n1, n2;
	int i;

	n1 = isl_vec_size(info1->cst);
	n2 = isl_vec_size(info2->cst);
	if (n1 < 0 || n2 < 0)
		return isl_bool_error;

	if (n2 < n1)
		n1 = n2;

	for (i = 0; i < n1; ++i) {
		int cmp;

		if (!info1->is_cst[i])
			continue;
		if (!info2->is_cst[i])
			continue;
		cmp = isl_vec_cmp_element(info1->cst, info2->cst, i);
		if (cmp != 0)
			return isl_bool_false;
	}

	return isl_bool_true;
}

/* Given a sink access, look for all the source accesses that access
 * the same array and perform dataflow analysis on them using
 * isl_access_info_compute_flow_core.
 */
static isl_stat compute_flow(__isl_take isl_map *map, void *user)
{
	int i;
	isl_ctx *ctx;
	struct isl_compute_flow_data *data;
	isl_flow *flow;
	isl_union_flow *df;

	data = (struct isl_compute_flow_data *)user;
	df = data->flow;

	ctx = isl_map_get_ctx(map);

	data->accesses = NULL;
	data->sink_info = NULL;
	data->source_info = NULL;
	data->count = 0;
	data->dim = isl_space_range(isl_map_get_space(map));

	if (isl_union_map_foreach_map(data->must_source,
					&count_matching_array, data) < 0)
		goto error;
	if (isl_union_map_foreach_map(data->may_source,
					&count_matching_array, data) < 0)
		goto error;

	data->sink_info = sched_info_alloc(map);
	data->source_info = isl_calloc_array(ctx, struct isl_sched_info *,
					     data->count);

	data->accesses = isl_access_info_alloc(isl_map_copy(map),
				data->sink_info, &before, data->count);
	if (!data->sink_info || (data->count && !data->source_info) ||
	    !data->accesses)
		goto error;
	data->accesses->coscheduled = &coscheduled;
	data->count = 0;
	data->must = 1;
	if (isl_union_map_foreach_map(data->must_source,
					&collect_matching_array, data) < 0)
		goto error;
	data->must = 0;
	if (isl_union_map_foreach_map(data->may_source,
					&collect_matching_array, data) < 0)
		goto error;

	flow = access_info_compute_flow_core(data->accesses);
	data->accesses = NULL;

	if (!flow)
		goto error;

	df->must_no_source = isl_union_map_union(df->must_no_source,
		    isl_union_map_from_map(isl_flow_get_no_source(flow, 1)));
	df->may_no_source = isl_union_map_union(df->may_no_source,
		    isl_union_map_from_map(isl_flow_get_no_source(flow, 0)));

	for (i = 0; i < flow->n_source; ++i) {
		isl_union_map *dep;
		dep = isl_union_map_from_map(isl_map_copy(flow->dep[i].map));
		if (flow->dep[i].must)
			df->must_dep = isl_union_map_union(df->must_dep, dep);
		else
			df->may_dep = isl_union_map_union(df->may_dep, dep);
	}

	isl_flow_free(flow);

	sched_info_free(data->sink_info);
	if (data->source_info) {
		for (i = 0; i < data->count; ++i)
			sched_info_free(data->source_info[i]);
		free(data->source_info);
	}
	isl_space_free(data->dim);
	isl_map_free(map);

	return isl_stat_ok;
error:
	isl_access_info_free(data->accesses);
	sched_info_free(data->sink_info);
	if (data->source_info) {
		for (i = 0; i < data->count; ++i)
			sched_info_free(data->source_info[i]);
		free(data->source_info);
	}
	isl_space_free(data->dim);
	isl_map_free(map);

	return isl_stat_error;
}

/* Add the kills of "info" to the must-sources.
 */
static __isl_give isl_union_access_info *
isl_union_access_info_add_kill_to_must_source(
	__isl_take isl_union_access_info *info)
{
	isl_union_map *must, *kill;

	must = isl_union_access_info_get_must_source(info);
	kill = isl_union_access_info_get_kill(info);
	must = isl_union_map_union(must, kill);
	return isl_union_access_info_set_must_source(info, must);
}

/* Drop dependences from "flow" that purely originate from kills.
 * That is, only keep those dependences that originate from
 * the original must-sources "must" and/or the original may-sources "may".
 * In particular, "must" contains the must-sources from before
 * the kills were added and "may" contains the may-source from before
 * the kills were removed.
 *
 * The dependences are of the form
 *
 *	Source -> [Sink -> Data]
 *
 * Only those dependences are kept where the Source -> Data part
 * is a subset of the original may-sources or must-sources.
 * Of those, only the must-dependences that intersect with the must-sources
 * remain must-dependences.
 * If there is some overlap between the may-sources and the must-sources,
 * then the may-dependences and must-dependences may also overlap.
 * This should be fine since the may-dependences are only kept
 * disjoint from the must-dependences for the isl_union_map_compute_flow
 * interface.  This interface does not support kills, so it will
 * not end up calling this function.
 */
static __isl_give isl_union_flow *isl_union_flow_drop_kill_source(
	__isl_take isl_union_flow *flow, __isl_take isl_union_map *must,
	__isl_take isl_union_map *may)
{
	isl_union_map *move;

	if (!flow)
		goto error;
	move = isl_union_map_copy(flow->must_dep);
	move = isl_union_map_intersect_range_factor_range(move,
				isl_union_map_copy(may));
	may = isl_union_map_union(may, isl_union_map_copy(must));
	flow->may_dep = isl_union_map_intersect_range_factor_range(
				flow->may_dep, may);
	flow->must_dep = isl_union_map_intersect_range_factor_range(
				flow->must_dep, must);
	flow->may_dep = isl_union_map_union(flow->may_dep, move);
	if (!flow->must_dep || !flow->may_dep)
		return isl_union_flow_free(flow);

	return flow;
error:
	isl_union_map_free(must);
	isl_union_map_free(may);
	return NULL;
}

/* Remove the must accesses from the may accesses.
 *
 * A must access always trumps a may access, so there is no need
 * for a must access to also be considered as a may access.  Doing so
 * would only cost extra computations only to find out that
 * the duplicated may access does not make any difference.
 */
static __isl_give isl_union_access_info *isl_union_access_info_normalize(
	__isl_take isl_union_access_info *access)
{
	if (!access)
		return NULL;
	access->access[isl_access_may_source] =
		isl_union_map_subtract(access->access[isl_access_may_source],
		    isl_union_map_copy(access->access[isl_access_must_source]));
	if (!access->access[isl_access_may_source])
		return isl_union_access_info_free(access);

	return access;
}

/* Given a description of the "sink" accesses, the "source" accesses and
 * a schedule, compute for each instance of a sink access
 * and for each element accessed by that instance,
 * the possible or definite source accesses that last accessed the
 * element accessed by the sink access before this sink access
 * in the sense that there is no intermediate definite source access.
 *
 * The must_no_source and may_no_source elements of the result
 * are subsets of access->sink.  The elements must_dep and may_dep
 * map domain elements of access->{may,must)_source to
 * domain elements of access->sink.
 *
 * This function is used when only the schedule map representation
 * is available.
 *
 * We first prepend the schedule dimensions to the domain
 * of the accesses so that we can easily compare their relative order.
 * Then we consider each sink access individually in compute_flow.
 */
static __isl_give isl_union_flow *compute_flow_union_map(
	__isl_take isl_union_access_info *access)
{
	struct isl_compute_flow_data data;
	isl_union_map *sink;

	access = isl_union_access_info_align_params(access);
	access = isl_union_access_info_introduce_schedule(access);
	if (!access)
		return NULL;

	data.must_source = access->access[isl_access_must_source];
	data.may_source = access->access[isl_access_may_source];

	sink = access->access[isl_access_sink];
	data.flow = isl_union_flow_alloc(isl_union_map_get_space(sink));

	if (isl_union_map_foreach_map(sink, &compute_flow, &data) < 0)
		goto error;

	data.flow = isl_union_flow_drop_schedule(data.flow);

	isl_union_access_info_free(access);
	return data.flow;
error:
	isl_union_access_info_free(access);
	isl_union_flow_free(data.flow);
	return NULL;
}

/* A schedule access relation.
 *
 * The access relation "access" is of the form [S -> D] -> A,
 * where S corresponds to the prefix schedule at "node".
 * "must" is only relevant for source accesses and indicates
 * whether the access is a must source or a may source.
 */
struct isl_scheduled_access {
	isl_map *access;
	int must;
	isl_schedule_node *node;
};

/* Data structure for keeping track of individual scheduled sink and source
 * accesses when computing dependence analysis based on a schedule tree.
 *
 * "n_sink" is the number of used entries in "sink"
 * "n_source" is the number of used entries in "source"
 *
 * "set_sink", "must" and "node" are only used inside collect_sink_source,
 * to keep track of the current node and
 * of what extract_sink_source needs to do.
 */
struct isl_compute_flow_schedule_data {
	isl_union_access_info *access;

	int n_sink;
	int n_source;

	struct isl_scheduled_access *sink;
	struct isl_scheduled_access *source;

	int set_sink;
	int must;
	isl_schedule_node *node;
};

/* Align the parameters of all sinks with all sources.
 *
 * If there are no sinks or no sources, then no alignment is needed.
 */
static void isl_compute_flow_schedule_data_align_params(
	struct isl_compute_flow_schedule_data *data)
{
	int i;
	isl_space *space;

	if (data->n_sink == 0 || data->n_source == 0)
		return;

	space = isl_map_get_space(data->sink[0].access);

	for (i = 1; i < data->n_sink; ++i)
		space = isl_space_align_params(space,
				isl_map_get_space(data->sink[i].access));
	for (i = 0; i < data->n_source; ++i)
		space = isl_space_align_params(space,
				isl_map_get_space(data->source[i].access));

	for (i = 0; i < data->n_sink; ++i)
		data->sink[i].access =
			isl_map_align_params(data->sink[i].access,
							isl_space_copy(space));
	for (i = 0; i < data->n_source; ++i)
		data->source[i].access =
			isl_map_align_params(data->source[i].access,
							isl_space_copy(space));

	isl_space_free(space);
}

/* Free all the memory referenced from "data".
 * Do not free "data" itself as it may be allocated on the stack.
 */
static void isl_compute_flow_schedule_data_clear(
	struct isl_compute_flow_schedule_data *data)
{
	int i;

	if (!data->sink)
		return;

	for (i = 0; i < data->n_sink; ++i) {
		isl_map_free(data->sink[i].access);
		isl_schedule_node_free(data->sink[i].node);
	}

	for (i = 0; i < data->n_source; ++i) {
		isl_map_free(data->source[i].access);
		isl_schedule_node_free(data->source[i].node);
	}

	free(data->sink);
}

/* isl_schedule_foreach_schedule_node_top_down callback for counting
 * (an upper bound on) the number of sinks and sources.
 *
 * Sinks and sources are only extracted at leaves of the tree,
 * so we skip the node if it is not a leaf.
 * Otherwise we increment data->n_sink and data->n_source with
 * the number of spaces in the sink and source access domains
 * that reach this node.
 */
static isl_bool count_sink_source(__isl_keep isl_schedule_node *node,
	void *user)
{
	struct isl_compute_flow_schedule_data *data = user;
	isl_union_set *domain;
	isl_union_map *umap;
	isl_bool r = isl_bool_false;
	isl_size n;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
		return isl_bool_true;

	domain = isl_schedule_node_get_universe_domain(node);

	umap = isl_union_map_copy(data->access->access[isl_access_sink]);
	umap = isl_union_map_intersect_domain(umap, isl_union_set_copy(domain));
	data->n_sink += n = isl_union_map_n_map(umap);
	isl_union_map_free(umap);
	if (n < 0)
		r = isl_bool_error;

	umap = isl_union_map_copy(data->access->access[isl_access_must_source]);
	umap = isl_union_map_intersect_domain(umap, isl_union_set_copy(domain));
	data->n_source += n = isl_union_map_n_map(umap);
	isl_union_map_free(umap);
	if (n < 0)
		r = isl_bool_error;

	umap = isl_union_map_copy(data->access->access[isl_access_may_source]);
	umap = isl_union_map_intersect_domain(umap, isl_union_set_copy(domain));
	data->n_source += n = isl_union_map_n_map(umap);
	isl_union_map_free(umap);
	if (n < 0)
		r = isl_bool_error;

	isl_union_set_free(domain);

	return r;
}

/* Add a single scheduled sink or source (depending on data->set_sink)
 * with scheduled access relation "map", must property data->must and
 * schedule node data->node to the list of sinks or sources.
 */
static isl_stat extract_sink_source(__isl_take isl_map *map, void *user)
{
	struct isl_compute_flow_schedule_data *data = user;
	struct isl_scheduled_access *access;

	if (data->set_sink)
		access = data->sink + data->n_sink++;
	else
		access = data->source + data->n_source++;

	access->access = map;
	access->must = data->must;
	access->node = isl_schedule_node_copy(data->node);

	return isl_stat_ok;
}

/* isl_schedule_foreach_schedule_node_top_down callback for collecting
 * individual scheduled source and sink accesses (taking into account
 * the domain of the schedule).
 *
 * We only collect accesses at the leaves of the schedule tree.
 * We prepend the schedule dimensions at the leaf to the iteration
 * domains of the source and sink accesses and then extract
 * the individual accesses (per space).
 *
 * In particular, if the prefix schedule at the node is of the form
 *
 *	D -> S
 *
 * while the access relations are of the form
 *
 *	D -> A
 *
 * then the updated access relations are of the form
 *
 *	[S -> D] -> A
 *
 * Note that S consists of a single space such that introducing S
 * in the access relations does not increase the number of spaces.
 */
static isl_bool collect_sink_source(__isl_keep isl_schedule_node *node,
	void *user)
{
	struct isl_compute_flow_schedule_data *data = user;
	isl_union_map *prefix;
	isl_union_map *umap;
	isl_bool r = isl_bool_false;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
		return isl_bool_true;

	data->node = node;

	prefix = isl_schedule_node_get_prefix_schedule_relation(node);
	prefix = isl_union_map_reverse(prefix);
	prefix = isl_union_map_range_map(prefix);

	data->set_sink = 1;
	umap = isl_union_map_copy(data->access->access[isl_access_sink]);
	umap = isl_union_map_apply_range(isl_union_map_copy(prefix), umap);
	if (isl_union_map_foreach_map(umap, &extract_sink_source, data) < 0)
		r = isl_bool_error;
	isl_union_map_free(umap);

	data->set_sink = 0;
	data->must = 1;
	umap = isl_union_map_copy(data->access->access[isl_access_must_source]);
	umap = isl_union_map_apply_range(isl_union_map_copy(prefix), umap);
	if (isl_union_map_foreach_map(umap, &extract_sink_source, data) < 0)
		r = isl_bool_error;
	isl_union_map_free(umap);

	data->set_sink = 0;
	data->must = 0;
	umap = isl_union_map_copy(data->access->access[isl_access_may_source]);
	umap = isl_union_map_apply_range(isl_union_map_copy(prefix), umap);
	if (isl_union_map_foreach_map(umap, &extract_sink_source, data) < 0)
		r = isl_bool_error;
	isl_union_map_free(umap);

	isl_union_map_free(prefix);

	return r;
}

/* isl_access_info_compute_flow callback for determining whether
 * the shared nesting level and the ordering within that level
 * for two scheduled accesses for use in compute_single_flow.
 *
 * The tokens passed to this function refer to the leaves
 * in the schedule tree where the accesses take place.
 *
 * If n is the shared number of loops, then we need to return
 * "2 * n + 1" if "first" precedes "second" inside the innermost
 * shared loop and "2 * n" otherwise.
 *
 * The innermost shared ancestor may be the leaves themselves
 * if the accesses take place in the same leaf.  Otherwise,
 * it is either a set node or a sequence node.  Only in the case
 * of a sequence node do we consider one access to precede the other.
 */
static int before_node(void *first, void *second)
{
	isl_schedule_node *node1 = first;
	isl_schedule_node *node2 = second;
	isl_schedule_node *shared;
	isl_size depth;
	int before = 0;

	shared = isl_schedule_node_get_shared_ancestor(node1, node2);
	depth = isl_schedule_node_get_schedule_depth(shared);
	if (depth < 0) {
		isl_schedule_node_free(shared);
		return -1;
	}

	if (isl_schedule_node_get_type(shared) == isl_schedule_node_sequence) {
		isl_size pos1, pos2;

		pos1 = isl_schedule_node_get_ancestor_child_position(node1,
								    shared);
		pos2 = isl_schedule_node_get_ancestor_child_position(node2,
								    shared);
		if (pos1 < 0 || pos2 < 0) {
			isl_schedule_node_free(shared);
			return -1;
		}
		before = pos1 < pos2;
	}

	isl_schedule_node_free(shared);

	return 2 * depth + before;
}

/* Check if the given two accesses may be coscheduled.
 * If so, return isl_bool_true.  Otherwise return isl_bool_false.
 *
 * Two accesses may only be coscheduled if they appear in the same leaf.
 */
static isl_bool coscheduled_node(void *first, void *second)
{
	isl_schedule_node *node1 = first;
	isl_schedule_node *node2 = second;

	return isl_bool_ok(node1 == node2);
}

/* Add the scheduled sources from "data" that access
 * the same data space as "sink" to "access".
 */
static __isl_give isl_access_info *add_matching_sources(
	__isl_take isl_access_info *access, struct isl_scheduled_access *sink,
	struct isl_compute_flow_schedule_data *data)
{
	int i;
	isl_space *space;

	space = isl_space_range(isl_map_get_space(sink->access));
	for (i = 0; i < data->n_source; ++i) {
		struct isl_scheduled_access *source;
		isl_space *source_space;
		int eq;

		source = &data->source[i];
		source_space = isl_map_get_space(source->access);
		source_space = isl_space_range(source_space);
		eq = isl_space_is_equal(space, source_space);
		isl_space_free(source_space);

		if (!eq)
			continue;
		if (eq < 0)
			goto error;

		access = isl_access_info_add_source(access,
		    isl_map_copy(source->access), source->must, source->node);
	}

	isl_space_free(space);
	return access;
error:
	isl_space_free(space);
	isl_access_info_free(access);
	return NULL;
}

/* Given a scheduled sink access relation "sink", compute the corresponding
 * dependences on the sources in "data" and add the computed dependences
 * to "uf".
 *
 * The dependences computed by access_info_compute_flow_core are of the form
 *
 *	[S -> I] -> [[S' -> I'] -> A]
 *
 * The schedule dimensions are projected out by first currying the range,
 * resulting in
 *
 *	[S -> I] -> [S' -> [I' -> A]]
 *
 * and then computing the factor range
 *
 *	I -> [I' -> A]
 */
static __isl_give isl_union_flow *compute_single_flow(
	__isl_take isl_union_flow *uf, struct isl_scheduled_access *sink,
	struct isl_compute_flow_schedule_data *data)
{
	int i;
	isl_access_info *access;
	isl_flow *flow;
	isl_map *map;

	if (!uf)
		return NULL;

	access = isl_access_info_alloc(isl_map_copy(sink->access), sink->node,
					&before_node, data->n_source);
	if (access)
		access->coscheduled = &coscheduled_node;
	access = add_matching_sources(access, sink, data);

	flow = access_info_compute_flow_core(access);
	if (!flow)
		return isl_union_flow_free(uf);

	map = isl_map_domain_factor_range(isl_flow_get_no_source(flow, 1));
	uf->must_no_source = isl_union_map_union(uf->must_no_source,
						isl_union_map_from_map(map));
	map = isl_map_domain_factor_range(isl_flow_get_no_source(flow, 0));
	uf->may_no_source = isl_union_map_union(uf->may_no_source,
						isl_union_map_from_map(map));

	for (i = 0; i < flow->n_source; ++i) {
		isl_union_map *dep;

		map = isl_map_range_curry(isl_map_copy(flow->dep[i].map));
		map = isl_map_factor_range(map);
		dep = isl_union_map_from_map(map);
		if (flow->dep[i].must)
			uf->must_dep = isl_union_map_union(uf->must_dep, dep);
		else
			uf->may_dep = isl_union_map_union(uf->may_dep, dep);
	}

	isl_flow_free(flow);

	return uf;
}

/* Given a description of the "sink" accesses, the "source" accesses and
 * a schedule, compute for each instance of a sink access
 * and for each element accessed by that instance,
 * the possible or definite source accesses that last accessed the
 * element accessed by the sink access before this sink access
 * in the sense that there is no intermediate definite source access.
 * Only consider dependences between statement instances that belong
 * to the domain of the schedule.
 *
 * The must_no_source and may_no_source elements of the result
 * are subsets of access->sink.  The elements must_dep and may_dep
 * map domain elements of access->{may,must)_source to
 * domain elements of access->sink.
 *
 * This function is used when a schedule tree representation
 * is available.
 *
 * We extract the individual scheduled source and sink access relations
 * (taking into account the domain of the schedule) and
 * then compute dependences for each scheduled sink individually.
 */
static __isl_give isl_union_flow *compute_flow_schedule(
	__isl_take isl_union_access_info *access)
{
	struct isl_compute_flow_schedule_data data = { access };
	int i, n;
	isl_ctx *ctx;
	isl_space *space;
	isl_union_flow *flow;

	ctx = isl_union_access_info_get_ctx(access);

	data.n_sink = 0;
	data.n_source = 0;
	if (isl_schedule_foreach_schedule_node_top_down(access->schedule,
						&count_sink_source, &data) < 0)
		goto error;

	n = data.n_sink + data.n_source;
	data.sink = isl_calloc_array(ctx, struct isl_scheduled_access, n);
	if (n && !data.sink)
		goto error;
	data.source = data.sink + data.n_sink;

	data.n_sink = 0;
	data.n_source = 0;
	if (isl_schedule_foreach_schedule_node_top_down(access->schedule,
					    &collect_sink_source, &data) < 0)
		goto error;

	space = isl_union_map_get_space(access->access[isl_access_sink]);
	flow = isl_union_flow_alloc(space);

	isl_compute_flow_schedule_data_align_params(&data);

	for (i = 0; i < data.n_sink; ++i)
		flow = compute_single_flow(flow, &data.sink[i], &data);

	isl_compute_flow_schedule_data_clear(&data);

	isl_union_access_info_free(access);
	return flow;
error:
	isl_union_access_info_free(access);
	isl_compute_flow_schedule_data_clear(&data);
	return NULL;
}

/* Given a description of the "sink" accesses, the "source" accesses and
 * a schedule, compute for each instance of a sink access
 * and for each element accessed by that instance,
 * the possible or definite source accesses that last accessed the
 * element accessed by the sink access before this sink access
 * in the sense that there is no intermediate definite source access.
 *
 * The must_no_source and may_no_source elements of the result
 * are subsets of access->sink.  The elements must_dep and may_dep
 * map domain elements of access->{may,must)_source to
 * domain elements of access->sink.
 *
 * If any kills have been specified, then they are treated as
 * must-sources internally.  Any dependence that purely derives
 * from an original kill is removed from the output.
 *
 * We check whether the schedule is available as a schedule tree
 * or a schedule map and call the corresponding function to perform
 * the analysis.
 */
__isl_give isl_union_flow *isl_union_access_info_compute_flow(
	__isl_take isl_union_access_info *access)
{
	isl_bool has_kill;
	isl_union_map *must = NULL, *may = NULL;
	isl_union_flow *flow;

	has_kill = isl_union_access_has_kill(access);
	if (has_kill < 0)
		goto error;
	if (has_kill) {
		must = isl_union_access_info_get_must_source(access);
		may = isl_union_access_info_get_may_source(access);
	}
	access = isl_union_access_info_add_kill_to_must_source(access);
	access = isl_union_access_info_normalize(access);
	if (!access)
		goto error;
	if (access->schedule)
		flow = compute_flow_schedule(access);
	else
		flow = compute_flow_union_map(access);
	if (has_kill)
		flow = isl_union_flow_drop_kill_source(flow, must, may);
	return flow;
error:
	isl_union_access_info_free(access);
	isl_union_map_free(must);
	isl_union_map_free(may);
	return NULL;
}

/* Print the information contained in "flow" to "p".
 * The information is printed as a YAML document.
 */
__isl_give isl_printer *isl_printer_print_union_flow(
	__isl_take isl_printer *p, __isl_keep isl_union_flow *flow)
{
	isl_union_map *umap;

	if (!flow)
		return isl_printer_free(p);

	p = isl_printer_yaml_start_mapping(p);
	umap = isl_union_flow_get_full_must_dependence(flow);
	p = print_yaml_field_union_map(p, "must_dependence", umap);
	isl_union_map_free(umap);
	umap = isl_union_flow_get_full_may_dependence(flow);
	p = print_yaml_field_union_map(p, "may_dependence", umap);
	isl_union_map_free(umap);
	p = print_yaml_field_union_map(p, "must_no_source",
					flow->must_no_source);
	umap = isl_union_flow_get_may_no_source(flow);
	p = print_yaml_field_union_map(p, "may_no_source", umap);
	isl_union_map_free(umap);
	p = isl_printer_yaml_end_mapping(p);

	return p;
}

/* Return a string representation of the information in "flow".
 * The information is printed in flow format.
 */
__isl_give char *isl_union_flow_to_str(__isl_keep isl_union_flow *flow)
{
	isl_printer *p;
	char *s;

	if (!flow)
		return NULL;

	p = isl_printer_to_str(isl_union_flow_get_ctx(flow));
	p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_FLOW);
	p = isl_printer_print_union_flow(p, flow);
	s = isl_printer_get_str(p);
	isl_printer_free(p);

	return s;
}

/* Given a collection of "sink" and "source" accesses,
 * compute for each iteration of a sink access
 * and for each element accessed by that iteration,
 * the source access in the list that last accessed the
 * element accessed by the sink access before this sink access.
 * Each access is given as a map from the loop iterators
 * to the array indices.
 * The result is a relations between source and sink
 * iterations and a subset of the domain of the sink accesses,
 * corresponding to those iterations that access an element
 * not previously accessed.
 *
 * We collect the inputs in an isl_union_access_info object,
 * call isl_union_access_info_compute_flow and extract
 * the outputs from the result.
 */
int isl_union_map_compute_flow(__isl_take isl_union_map *sink,
	__isl_take isl_union_map *must_source,
	__isl_take isl_union_map *may_source,
	__isl_take isl_union_map *schedule,
	__isl_give isl_union_map **must_dep, __isl_give isl_union_map **may_dep,
	__isl_give isl_union_map **must_no_source,
	__isl_give isl_union_map **may_no_source)
{
	isl_union_access_info *access;
	isl_union_flow *flow;

	access = isl_union_access_info_from_sink(sink);
	access = isl_union_access_info_set_must_source(access, must_source);
	access = isl_union_access_info_set_may_source(access, may_source);
	access = isl_union_access_info_set_schedule_map(access, schedule);
	flow = isl_union_access_info_compute_flow(access);

	if (must_dep)
		*must_dep = isl_union_flow_get_must_dependence(flow);
	if (may_dep)
		*may_dep = isl_union_flow_get_non_must_dependence(flow);
	if (must_no_source)
		*must_no_source = isl_union_flow_get_must_no_source(flow);
	if (may_no_source)
		*may_no_source = isl_union_flow_get_non_must_no_source(flow);

	isl_union_flow_free(flow);

	if ((must_dep && !*must_dep) || (may_dep && !*may_dep) ||
	    (must_no_source && !*must_no_source) ||
	    (may_no_source && !*may_no_source))
		goto error;

	return 0;
error:
	if (must_dep)
		*must_dep = isl_union_map_free(*must_dep);
	if (may_dep)
		*may_dep = isl_union_map_free(*may_dep);
	if (must_no_source)
		*must_no_source = isl_union_map_free(*must_no_source);
	if (may_no_source)
		*may_no_source = isl_union_map_free(*may_no_source);
	return -1;
}
