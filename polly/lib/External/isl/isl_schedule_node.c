/*
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/set.h>
#include <isl_schedule_band.h>
#include <isl_schedule_private.h>
#include <isl_schedule_node_private.h>

/* Create a new schedule node in the given schedule, point at the given
 * tree with given ancestors and child positions.
 * "child_pos" may be NULL if there are no ancestors.
 */
__isl_give isl_schedule_node *isl_schedule_node_alloc(
	__isl_take isl_schedule *schedule, __isl_take isl_schedule_tree *tree,
	__isl_take isl_schedule_tree_list *ancestors, int *child_pos)
{
	isl_ctx *ctx;
	isl_schedule_node *node;
	int i, n;

	if (!schedule || !tree || !ancestors)
		goto error;
	n = isl_schedule_tree_list_n_schedule_tree(ancestors);
	if (n > 0 && !child_pos)
		goto error;
	ctx = isl_schedule_get_ctx(schedule);
	node = isl_calloc_type(ctx, isl_schedule_node);
	if (!node)
		goto error;
	node->ref = 1;
	node->schedule = schedule;
	node->tree = tree;
	node->ancestors = ancestors;
	node->child_pos = isl_alloc_array(ctx, int, n);
	if (n && !node->child_pos)
		return isl_schedule_node_free(node);
	for (i = 0; i < n; ++i)
		node->child_pos[i] = child_pos[i];

	return node;
error:
	isl_schedule_free(schedule);
	isl_schedule_tree_free(tree);
	isl_schedule_tree_list_free(ancestors);
	return NULL;
}

/* Return a pointer to the root of a schedule tree with as single
 * node a domain node with the given domain.
 */
__isl_give isl_schedule_node *isl_schedule_node_from_domain(
	__isl_take isl_union_set *domain)
{
	isl_schedule *schedule;
	isl_schedule_node *node;

	schedule = isl_schedule_from_domain(domain);
	node = isl_schedule_get_root(schedule);
	isl_schedule_free(schedule);

	return node;
}

/* Return the isl_ctx to which "node" belongs.
 */
isl_ctx *isl_schedule_node_get_ctx(__isl_keep isl_schedule_node *node)
{
	return node ? isl_schedule_get_ctx(node->schedule) : NULL;
}

/* Return a pointer to the leaf of the schedule into which "node" points.
 *
 * Even though these leaves are not reference counted, we still
 * indicate that this function does not return a copy.
 */
__isl_keep isl_schedule_tree *isl_schedule_node_peek_leaf(
	__isl_keep isl_schedule_node *node)
{
	return node ? isl_schedule_peek_leaf(node->schedule) : NULL;
}

/* Return a pointer to the leaf of the schedule into which "node" points.
 *
 * Even though these leaves are not reference counted, we still
 * return a "copy" of the leaf here such that it can still be "freed"
 * by the user.
 */
__isl_give isl_schedule_tree *isl_schedule_node_get_leaf(
	__isl_keep isl_schedule_node *node)
{
	return isl_schedule_tree_copy(isl_schedule_node_peek_leaf(node));
}

/* Return the type of the node or isl_schedule_node_error on error.
 */
enum isl_schedule_node_type isl_schedule_node_get_type(
	__isl_keep isl_schedule_node *node)
{
	return node ? isl_schedule_tree_get_type(node->tree)
		    : isl_schedule_node_error;
}

/* Return the type of the parent of "node" or isl_schedule_node_error on error.
 */
enum isl_schedule_node_type isl_schedule_node_get_parent_type(
	__isl_keep isl_schedule_node *node)
{
	int pos;
	int has_parent;
	isl_schedule_tree *parent;
	enum isl_schedule_node_type type;

	if (!node)
		return isl_schedule_node_error;
	has_parent = isl_schedule_node_has_parent(node);
	if (has_parent < 0)
		return isl_schedule_node_error;
	if (!has_parent)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"node has no parent", return isl_schedule_node_error);

	pos = isl_schedule_tree_list_n_schedule_tree(node->ancestors) - 1;
	parent = isl_schedule_tree_list_get_schedule_tree(node->ancestors, pos);
	type = isl_schedule_tree_get_type(parent);
	isl_schedule_tree_free(parent);

	return type;
}

/* Return a copy of the subtree that this node points to.
 */
__isl_give isl_schedule_tree *isl_schedule_node_get_tree(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_tree_copy(node->tree);
}

/* Return a copy of the schedule into which "node" points.
 */
__isl_give isl_schedule *isl_schedule_node_get_schedule(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;
	return isl_schedule_copy(node->schedule);
}

/* Return a fresh copy of "node".
 */
__isl_take isl_schedule_node *isl_schedule_node_dup(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_node_alloc(isl_schedule_copy(node->schedule),
				isl_schedule_tree_copy(node->tree),
				isl_schedule_tree_list_copy(node->ancestors),
				node->child_pos);
}

/* Return an isl_schedule_node that is equal to "node" and that has only
 * a single reference.
 */
__isl_give isl_schedule_node *isl_schedule_node_cow(
	__isl_take isl_schedule_node *node)
{
	if (!node)
		return NULL;

	if (node->ref == 1)
		return node;
	node->ref--;
	return isl_schedule_node_dup(node);
}

/* Return a new reference to "node".
 */
__isl_give isl_schedule_node *isl_schedule_node_copy(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	node->ref++;
	return node;
}

/* Free "node" and return NULL.
 *
 * Since the node may point to a leaf of its schedule, which
 * point to a field inside the schedule, we need to make sure
 * we free the tree before freeing the schedule.
 */
__isl_null isl_schedule_node *isl_schedule_node_free(
	__isl_take isl_schedule_node *node)
{
	if (!node)
		return NULL;
	if (--node->ref > 0)
		return NULL;

	isl_schedule_tree_list_free(node->ancestors);
	free(node->child_pos);
	isl_schedule_tree_free(node->tree);
	isl_schedule_free(node->schedule);
	free(node);

	return NULL;
}

/* Internal data structure for
 * isl_schedule_node_get_prefix_schedule_union_pw_multi_aff
 *
 * "initialized" is set if the filter field has been initialized.
 * "universe_filter" is set if we are only collecting the universes of filters
 * "collect_prefix" is set if we are collecting prefixes.
 * "filter" collects all outer filters and is NULL until "initialized" is set.
 * "prefix" collects all outer band partial schedules (if "collect_prefix"
 * is set).  If it is used, then it is initialized by the caller
 * of collect_filter_prefix to a zero-dimensional function.
 */
struct isl_schedule_node_get_filter_prefix_data {
	int initialized;
	int universe_filter;
	int collect_prefix;
	isl_union_set *filter;
	isl_multi_union_pw_aff *prefix;
};

/* Update "data" based on the tree node "tree" in case "data" has
 * not been initialized yet.
 *
 * Return 0 on success and -1 on error.
 *
 * If "tree" is a filter, then we set data->filter to this filter
 * (or its universe).
 * If "tree" is a domain, then this means we have reached the root
 * of the schedule tree without being able to extract any information.
 * We therefore initialize data->filter to the universe of the domain.
 * If "tree" is a band with at least one member, then we set data->filter
 * to the universe of the schedule domain and replace the zero-dimensional
 * data->prefix by the band schedule (if data->collect_prefix is set).
 */
static int collect_filter_prefix_init(__isl_keep isl_schedule_tree *tree,
	struct isl_schedule_node_get_filter_prefix_data *data)
{
	enum isl_schedule_node_type type;
	isl_multi_union_pw_aff *mupa;
	isl_union_set *filter;

	type = isl_schedule_tree_get_type(tree);
	switch (type) {
	case isl_schedule_node_error:
		return -1;
	case isl_schedule_node_leaf:
	case isl_schedule_node_sequence:
	case isl_schedule_node_set:
		return 0;
	case isl_schedule_node_domain:
		filter = isl_schedule_tree_domain_get_domain(tree);
		filter = isl_union_set_universe(filter);
		data->filter = filter;
		break;
	case isl_schedule_node_band:
		if (isl_schedule_tree_band_n_member(tree) == 0)
			return 0;
		mupa = isl_schedule_tree_band_get_partial_schedule(tree);
		if (data->collect_prefix) {
			isl_multi_union_pw_aff_free(data->prefix);
			mupa = isl_multi_union_pw_aff_reset_tuple_id(mupa,
								isl_dim_set);
			data->prefix = isl_multi_union_pw_aff_copy(mupa);
		}
		filter = isl_multi_union_pw_aff_domain(mupa);
		filter = isl_union_set_universe(filter);
		data->filter = filter;
		break;
	case isl_schedule_node_filter:
		filter = isl_schedule_tree_filter_get_filter(tree);
		if (data->universe_filter)
			filter = isl_union_set_universe(filter);
		data->filter = filter;
		break;
	}

	if ((data->collect_prefix && !data->prefix) || !data->filter)
		return -1;

	data->initialized = 1;

	return 0;
}

/* Update "data" based on the tree node "tree" in case "data" has
 * already been initialized.
 *
 * Return 0 on success and -1 on error.
 *
 * If "tree" is a filter, then we intersect data->filter with this filter
 * (or its universe).
 * If "tree" is a band with at least one member and data->collect_prefix
 * is set, then we extend data->prefix with the band schedule.
 */
static int collect_filter_prefix_update(__isl_keep isl_schedule_tree *tree,
	struct isl_schedule_node_get_filter_prefix_data *data)
{
	enum isl_schedule_node_type type;
	isl_multi_union_pw_aff *mupa;
	isl_union_set *filter;

	type = isl_schedule_tree_get_type(tree);
	switch (type) {
	case isl_schedule_node_error:
		return -1;
	case isl_schedule_node_domain:
	case isl_schedule_node_leaf:
	case isl_schedule_node_sequence:
	case isl_schedule_node_set:
		break;
	case isl_schedule_node_band:
		if (isl_schedule_tree_band_n_member(tree) == 0)
			break;
		if (!data->collect_prefix)
			break;
		mupa = isl_schedule_tree_band_get_partial_schedule(tree);
		data->prefix = isl_multi_union_pw_aff_flat_range_product(mupa,
								data->prefix);
		if (!data->prefix)
			return -1;
		break;
	case isl_schedule_node_filter:
		filter = isl_schedule_tree_filter_get_filter(tree);
		if (data->universe_filter)
			filter = isl_union_set_universe(filter);
		data->filter = isl_union_set_intersect(data->filter, filter);
		if (!data->filter)
			return -1;
		break;
	}

	return 0;
}

/* Collect filter and/or prefix information from the elements
 * in "list" (which represent the ancestors of a node).
 * Store the results in "data".
 *
 * Return 0 on success and -1 on error.
 *
 * We traverse the list from innermost ancestor (last element)
 * to outermost ancestor (first element), calling collect_filter_prefix_init
 * on each node as long as we have not been able to extract any information
 * yet and collect_filter_prefix_update afterwards.
 * On successful return, data->initialized will be set since the outermost
 * ancestor is a domain node, which always results in an initialization.
 */
static int collect_filter_prefix(__isl_keep isl_schedule_tree_list *list,
	struct isl_schedule_node_get_filter_prefix_data *data)
{
	int i, n;

	data->initialized = 0;
	data->filter = NULL;

	if (!list)
		return -1;

	n = isl_schedule_tree_list_n_schedule_tree(list);
	for (i = n - 1; i >= 0; --i) {
		isl_schedule_tree *tree;
		int r;

		tree = isl_schedule_tree_list_get_schedule_tree(list, i);
		if (!tree)
			return -1;
		if (!data->initialized)
			r = collect_filter_prefix_init(tree, data);
		else
			r = collect_filter_prefix_update(tree, data);
		isl_schedule_tree_free(tree);
		if (r < 0)
			return -1;
	}

	return 0;
}

/* Return the concatenation of the partial schedules of all outer band
 * nodes of "node" interesected with all outer filters
 * as an isl_union_pw_multi_aff.
 *
 * If "node" is pointing at the root of the schedule tree, then
 * there are no domain elements reaching the current node, so
 * we return an empty result.
 *
 * We collect all the filters and partial schedules in collect_filter_prefix.
 * The partial schedules are collected as an isl_multi_union_pw_aff.
 * If this isl_multi_union_pw_aff is zero-dimensional, then it does not
 * contain any domain information, so we construct the isl_union_pw_multi_aff
 * result as a zero-dimensional function on the collected filter.
 * Otherwise, we convert the isl_multi_union_pw_aff to
 * an isl_multi_union_pw_aff and intersect the domain with the filter.
 */
__isl_give isl_union_pw_multi_aff *
isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(
	__isl_keep isl_schedule_node *node)
{
	isl_space *space;
	isl_union_pw_multi_aff *prefix;
	struct isl_schedule_node_get_filter_prefix_data data;

	if (!node)
		return NULL;

	space = isl_schedule_get_space(node->schedule);
	if (node->tree == node->schedule->root)
		return isl_union_pw_multi_aff_empty(space);

	space = isl_space_set_from_params(space);
	data.universe_filter = 0;
	data.collect_prefix = 1;
	data.prefix = isl_multi_union_pw_aff_zero(space);

	if (collect_filter_prefix(node->ancestors, &data) < 0)
		data.prefix = isl_multi_union_pw_aff_free(data.prefix);

	if (data.prefix &&
	    isl_multi_union_pw_aff_dim(data.prefix, isl_dim_set) == 0) {
		isl_multi_union_pw_aff_free(data.prefix);
		prefix = isl_union_pw_multi_aff_from_domain(data.filter);
	} else {
		prefix =
		    isl_union_pw_multi_aff_from_multi_union_pw_aff(data.prefix);
		prefix = isl_union_pw_multi_aff_intersect_domain(prefix,
								data.filter);
	}

	return prefix;
}

/* Return the concatenation of the partial schedules of all outer band
 * nodes of "node" interesected with all outer filters
 * as an isl_union_map.
 */
__isl_give isl_union_map *isl_schedule_node_get_prefix_schedule_union_map(
	__isl_keep isl_schedule_node *node)
{
	isl_union_pw_multi_aff *upma;

	upma = isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(node);
	return isl_union_map_from_union_pw_multi_aff(upma);
}

/* Return the union of universe sets of the domain elements that reach "node".
 *
 * If "node" is pointing at the root of the schedule tree, then
 * there are no domain elements reaching the current node, so
 * we return an empty result.
 *
 * Otherwise, we collect the universes of all filters reaching the node
 * in collect_filter_prefix.
 */
__isl_give isl_union_set *isl_schedule_node_get_universe_domain(
	__isl_keep isl_schedule_node *node)
{
	struct isl_schedule_node_get_filter_prefix_data data;

	if (!node)
		return NULL;

	if (node->tree == node->schedule->root) {
		isl_space *space;

		space = isl_schedule_get_space(node->schedule);
		return isl_union_set_empty(space);
	}

	data.universe_filter = 1;
	data.collect_prefix = 0;
	data.prefix = NULL;

	if (collect_filter_prefix(node->ancestors, &data) < 0)
		data.filter = isl_union_set_free(data.filter);

	return data.filter;
}

/* Return the subtree schedule of "node".
 *
 * Since isl_schedule_tree_get_subtree_schedule_union_map does not handle
 * trees that do not contain any schedule information, we first
 * move down to the first relevant descendant and handle leaves ourselves.
 */
__isl_give isl_union_map *isl_schedule_node_get_subtree_schedule_union_map(
	__isl_keep isl_schedule_node *node)
{
	isl_schedule_tree *tree, *leaf;
	isl_union_map *umap;

	tree = isl_schedule_node_get_tree(node);
	leaf = isl_schedule_node_peek_leaf(node);
	tree = isl_schedule_tree_first_schedule_descendant(tree, leaf);
	if (!tree)
		return NULL;
	if (tree == leaf) {
		isl_union_set *domain;
		domain = isl_schedule_node_get_universe_domain(node);
		isl_schedule_tree_free(tree);
		return isl_union_map_from_domain(domain);
	}

	umap = isl_schedule_tree_get_subtree_schedule_union_map(tree);
	isl_schedule_tree_free(tree);
	return umap;
}

/* Return the number of ancestors of "node" in its schedule tree.
 */
int isl_schedule_node_get_tree_depth(__isl_keep isl_schedule_node *node)
{
	if (!node)
		return -1;
	return isl_schedule_tree_list_n_schedule_tree(node->ancestors);
}

/* Does "node" have a parent?
 *
 * That is, does it point to any node of the schedule other than the root?
 */
int isl_schedule_node_has_parent(__isl_keep isl_schedule_node *node)
{
	if (!node)
		return -1;
	if (!node->ancestors)
		return -1;

	return isl_schedule_tree_list_n_schedule_tree(node->ancestors) != 0;
}

/* Return the position of "node" among the children of its parent.
 */
int isl_schedule_node_get_child_position(__isl_keep isl_schedule_node *node)
{
	int n;
	int has_parent;

	if (!node)
		return -1;
	has_parent = isl_schedule_node_has_parent(node);
	if (has_parent < 0)
		return -1;
	if (!has_parent)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"node has no parent", return -1);

	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	return node->child_pos[n - 1];
}

/* Does the parent (if any) of "node" have any children with a smaller child
 * position than this one?
 */
int isl_schedule_node_has_previous_sibling(__isl_keep isl_schedule_node *node)
{
	int n;
	int has_parent;

	if (!node)
		return -1;
	has_parent = isl_schedule_node_has_parent(node);
	if (has_parent < 0 || !has_parent)
		return has_parent;

	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);

	return node->child_pos[n - 1] > 0;
}

/* Does the parent (if any) of "node" have any children with a greater child
 * position than this one?
 */
int isl_schedule_node_has_next_sibling(__isl_keep isl_schedule_node *node)
{
	int n, n_child;
	int has_parent;
	isl_schedule_tree *tree;

	if (!node)
		return -1;
	has_parent = isl_schedule_node_has_parent(node);
	if (has_parent < 0 || !has_parent)
		return has_parent;

	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	tree = isl_schedule_tree_list_get_schedule_tree(node->ancestors, n - 1);
	if (!tree)
		return -1;
	n_child = isl_schedule_tree_list_n_schedule_tree(tree->children);
	isl_schedule_tree_free(tree);

	return node->child_pos[n - 1] + 1 < n_child;
}

/* Does "node" have any children?
 *
 * Any node other than the leaf nodes is considered to have at least
 * one child, even if the corresponding isl_schedule_tree does not
 * have any children.
 */
int isl_schedule_node_has_children(__isl_keep isl_schedule_node *node)
{
	if (!node)
		return -1;
	return !isl_schedule_tree_is_leaf(node->tree);
}

/* Return the number of children of "node"?
 *
 * Any node other than the leaf nodes is considered to have at least
 * one child, even if the corresponding isl_schedule_tree does not
 * have any children.  That is, the number of children of "node" is
 * only zero if its tree is the explicit empty tree.  Otherwise,
 * if the isl_schedule_tree has any children, then it is equal
 * to the number of children of "node".  If it has zero children,
 * then "node" still has a leaf node as child.
 */
int isl_schedule_node_n_children(__isl_keep isl_schedule_node *node)
{
	int n;

	if (!node)
		return -1;

	if (isl_schedule_tree_is_leaf(node->tree))
		return 0;

	n = isl_schedule_tree_n_children(node->tree);
	if (n == 0)
		return 1;

	return n;
}

/* Move the "node" pointer to the parent of the node it currently points to.
 */
__isl_give isl_schedule_node *isl_schedule_node_parent(
	__isl_take isl_schedule_node *node)
{
	int n;
	isl_schedule_tree *tree;

	node = isl_schedule_node_cow(node);
	if (!node)
		return NULL;
	if (!isl_schedule_node_has_parent(node))
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"node has no parent",
			return isl_schedule_node_free(node));
	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	tree = isl_schedule_tree_list_get_schedule_tree(node->ancestors, n - 1);
	isl_schedule_tree_free(node->tree);
	node->tree = tree;
	node->ancestors = isl_schedule_tree_list_drop(node->ancestors,
								n - 1, 1);
	if (!node->ancestors || !node->tree)
		return isl_schedule_node_free(node);

	return node;
}

/* Move the "node" pointer to the child at position "pos" of the node
 * it currently points to.
 */
__isl_give isl_schedule_node *isl_schedule_node_child(
	__isl_take isl_schedule_node *node, int pos)
{
	int n;
	isl_ctx *ctx;
	isl_schedule_tree *tree;
	int *child_pos;

	node = isl_schedule_node_cow(node);
	if (!node)
		return NULL;
	if (!isl_schedule_node_has_children(node))
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"node has no children",
			return isl_schedule_node_free(node));

	ctx = isl_schedule_node_get_ctx(node);
	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	child_pos = isl_realloc_array(ctx, node->child_pos, int, n + 1);
	if (!child_pos)
		return isl_schedule_node_free(node);
	node->child_pos = child_pos;
	node->child_pos[n] = pos;

	node->ancestors = isl_schedule_tree_list_add(node->ancestors,
				isl_schedule_tree_copy(node->tree));
	tree = node->tree;
	if (isl_schedule_tree_has_children(tree))
		tree = isl_schedule_tree_get_child(tree, pos);
	else
		tree = isl_schedule_node_get_leaf(node);
	isl_schedule_tree_free(node->tree);
	node->tree = tree;

	if (!node->tree || !node->ancestors)
		return isl_schedule_node_free(node);

	return node;
}

/* Move the "node" pointer to the first child of the node
 * it currently points to.
 */
__isl_give isl_schedule_node *isl_schedule_node_first_child(
	__isl_take isl_schedule_node *node)
{
	return isl_schedule_node_child(node, 0);
}

/* Move the "node" pointer to the child of this node's parent in
 * the previous child position.
 */
__isl_give isl_schedule_node *isl_schedule_node_previous_sibling(
	__isl_take isl_schedule_node *node)
{
	int n;
	isl_schedule_tree *parent, *tree;

	node = isl_schedule_node_cow(node);
	if (!node)
		return NULL;
	if (!isl_schedule_node_has_previous_sibling(node))
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"node has no previous sibling",
			return isl_schedule_node_free(node));

	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	parent = isl_schedule_tree_list_get_schedule_tree(node->ancestors,
									n - 1);
	if (!parent)
		return isl_schedule_node_free(node);
	node->child_pos[n - 1]--;
	tree = isl_schedule_tree_list_get_schedule_tree(parent->children,
							node->child_pos[n - 1]);
	isl_schedule_tree_free(parent);
	if (!tree)
		return isl_schedule_node_free(node);
	isl_schedule_tree_free(node->tree);
	node->tree = tree;

	return node;
}

/* Move the "node" pointer to the child of this node's parent in
 * the next child position.
 */
__isl_give isl_schedule_node *isl_schedule_node_next_sibling(
	__isl_take isl_schedule_node *node)
{
	int n;
	isl_schedule_tree *parent, *tree;

	node = isl_schedule_node_cow(node);
	if (!node)
		return NULL;
	if (!isl_schedule_node_has_next_sibling(node))
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"node has no next sibling",
			return isl_schedule_node_free(node));

	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	parent = isl_schedule_tree_list_get_schedule_tree(node->ancestors,
									n - 1);
	if (!parent)
		return isl_schedule_node_free(node);
	node->child_pos[n - 1]++;
	tree = isl_schedule_tree_list_get_schedule_tree(parent->children,
							node->child_pos[n - 1]);
	isl_schedule_tree_free(parent);
	if (!tree)
		return isl_schedule_node_free(node);
	isl_schedule_tree_free(node->tree);
	node->tree = tree;

	return node;
}

/* Return a copy to the child at position "pos" of "node".
 */
__isl_give isl_schedule_node *isl_schedule_node_get_child(
	__isl_keep isl_schedule_node *node, int pos)
{
	return isl_schedule_node_child(isl_schedule_node_copy(node), pos);
}

/* Traverse the descendant of "node" in depth-first order, including
 * "node" itself.  Call "enter" whenever a node is entered and "leave"
 * whenever a node is left.  The callback "enter" is responsible
 * for moving to the deepest initial subtree of its argument that
 * should be traversed.
 */
static __isl_give isl_schedule_node *traverse(
	__isl_take isl_schedule_node *node,
	__isl_give isl_schedule_node *(*enter)(
		__isl_take isl_schedule_node *node, void *user),
	__isl_give isl_schedule_node *(*leave)(
		__isl_take isl_schedule_node *node, void *user),
	void *user)
{
	int depth;

	if (!node)
		return NULL;

	depth = isl_schedule_node_get_tree_depth(node);
	do {
		node = enter(node, user);
		node = leave(node, user);
		while (node && isl_schedule_node_get_tree_depth(node) > depth &&
				!isl_schedule_node_has_next_sibling(node)) {
			node = isl_schedule_node_parent(node);
			node = leave(node, user);
		}
		if (node && isl_schedule_node_get_tree_depth(node) > depth)
			node = isl_schedule_node_next_sibling(node);
	} while (node && isl_schedule_node_get_tree_depth(node) > depth);

	return node;
}

/* Internal data structure for isl_schedule_node_foreach_descendant.
 *
 * "fn" is the user-specified callback function.
 * "user" is the user-specified argument for the callback.
 */
struct isl_schedule_node_preorder_data {
	int (*fn)(__isl_keep isl_schedule_node *node, void *user);
	void *user;
};

/* Callback for "traverse" to enter a node and to move
 * to the deepest initial subtree that should be traversed
 * for use in a preorder visit.
 *
 * If the user callback returns a negative value, then we abort
 * the traversal.  If this callback returns zero, then we skip
 * the subtree rooted at the current node.  Otherwise, we move
 * down to the first child and repeat the process until a leaf
 * is reached.
 */
static __isl_give isl_schedule_node *preorder_enter(
	__isl_take isl_schedule_node *node, void *user)
{
	struct isl_schedule_node_preorder_data *data = user;

	if (!node)
		return NULL;

	do {
		int r;

		r = data->fn(node, data->user);
		if (r < 0)
			return isl_schedule_node_free(node);
		if (r == 0)
			return node;
	} while (isl_schedule_node_has_children(node) &&
		(node = isl_schedule_node_first_child(node)) != NULL);

	return node;
}

/* Callback for "traverse" to leave a node
 * for use in a preorder visit.
 * Since we already visited the node when we entered it,
 * we do not need to do anything here.
 */
static __isl_give isl_schedule_node *preorder_leave(
	__isl_take isl_schedule_node *node, void *user)
{
	return node;
}

/* Traverse the descendants of "node" (including the node itself)
 * in depth first preorder.
 *
 * If "fn" returns -1 on any of the nodes, then the traversal is aborted.
 * If "fn" returns 0 on any of the nodes, then the subtree rooted
 * at that node is skipped.
 *
 * Return 0 on success and -1 on failure.
 */
int isl_schedule_node_foreach_descendant(__isl_keep isl_schedule_node *node,
	int (*fn)(__isl_keep isl_schedule_node *node, void *user), void *user)
{
	struct isl_schedule_node_preorder_data data = { fn, user };

	node = isl_schedule_node_copy(node);
	node = traverse(node, &preorder_enter, &preorder_leave, &data);
	isl_schedule_node_free(node);

	return node ? 0 : -1;
}

/* Internal data structure for isl_schedule_node_map_descendant.
 *
 * "fn" is the user-specified callback function.
 * "user" is the user-specified argument for the callback.
 */
struct isl_schedule_node_postorder_data {
	__isl_give isl_schedule_node *(*fn)(__isl_take isl_schedule_node *node,
		void *user);
	void *user;
};

/* Callback for "traverse" to enter a node and to move
 * to the deepest initial subtree that should be traversed
 * for use in a postorder visit.
 *
 * Since we are performing a postorder visit, we only need
 * to move to the deepest initial leaf here.
 */
static __isl_give isl_schedule_node *postorder_enter(
	__isl_take isl_schedule_node *node, void *user)
{
	while (node && isl_schedule_node_has_children(node))
		node = isl_schedule_node_first_child(node);

	return node;
}

/* Callback for "traverse" to leave a node
 * for use in a postorder visit.
 *
 * Since we are performing a postorder visit, we need
 * to call the user callback here.
 */
static __isl_give isl_schedule_node *postorder_leave(
	__isl_take isl_schedule_node *node, void *user)
{
	struct isl_schedule_node_postorder_data *data = user;

	return data->fn(node, data->user);
}

/* Traverse the descendants of "node" (including the node itself)
 * in depth first postorder, allowing the user to modify the visited node.
 * The traversal continues from the node returned by the callback function.
 * It is the responsibility of the user to ensure that this does not
 * lead to an infinite loop.  It is safest to always return a pointer
 * to the same position (same ancestors and child positions) as the input node.
 */
__isl_give isl_schedule_node *isl_schedule_node_map_descendant(
	__isl_take isl_schedule_node *node,
	__isl_give isl_schedule_node *(*fn)(__isl_take isl_schedule_node *node,
		void *user), void *user)
{
	struct isl_schedule_node_postorder_data data = { fn, user };

	return traverse(node, &postorder_enter, &postorder_leave, &data);
}

/* Return the number of members in the given band node.
 */
unsigned isl_schedule_node_band_n_member(__isl_keep isl_schedule_node *node)
{
	return node ? isl_schedule_tree_band_n_member(node->tree) : 0;
}

/* Is the band member at position "pos" of the band node "node"
 * marked coincident?
 */
int isl_schedule_node_band_member_get_coincident(
	__isl_keep isl_schedule_node *node, int pos)
{
	if (!node)
		return -1;
	return isl_schedule_tree_band_member_get_coincident(node->tree, pos);
}

/* Mark the band member at position "pos" the band node "node"
 * as being coincident or not according to "coincident".
 */
__isl_give isl_schedule_node *isl_schedule_node_band_member_set_coincident(
	__isl_take isl_schedule_node *node, int pos, int coincident)
{
	int c;
	isl_schedule_tree *tree;

	if (!node)
		return NULL;
	c = isl_schedule_node_band_member_get_coincident(node, pos);
	if (c == coincident)
		return node;

	tree = isl_schedule_tree_copy(node->tree);
	tree = isl_schedule_tree_band_member_set_coincident(tree, pos,
							    coincident);
	node = isl_schedule_node_graft_tree(node, tree);

	return node;
}

/* Is the band node "node" marked permutable?
 */
int isl_schedule_node_band_get_permutable(__isl_keep isl_schedule_node *node)
{
	if (!node)
		return -1;

	return isl_schedule_tree_band_get_permutable(node->tree);
}

/* Mark the band node "node" permutable or not according to "permutable"?
 */
__isl_give isl_schedule_node *isl_schedule_node_band_set_permutable(
	__isl_take isl_schedule_node *node, int permutable)
{
	isl_schedule_tree *tree;

	if (!node)
		return NULL;
	if (isl_schedule_node_band_get_permutable(node) == permutable)
		return node;

	tree = isl_schedule_tree_copy(node->tree);
	tree = isl_schedule_tree_band_set_permutable(tree, permutable);
	node = isl_schedule_node_graft_tree(node, tree);

	return node;
}

/* Return the schedule space of the band node.
 */
__isl_give isl_space *isl_schedule_node_band_get_space(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_tree_band_get_space(node->tree);
}

/* Return the schedule of the band node in isolation.
 */
__isl_give isl_multi_union_pw_aff *isl_schedule_node_band_get_partial_schedule(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_tree_band_get_partial_schedule(node->tree);
}

/* Return the schedule of the band node in isolation in the form of
 * an isl_union_map.
 *
 * If the band does not have any members, then we construct a universe map
 * with the universe of the domain elements reaching the node as domain.
 * Otherwise, we extract an isl_multi_union_pw_aff representation and
 * convert that to an isl_union_map.
 */
__isl_give isl_union_map *isl_schedule_node_band_get_partial_schedule_union_map(
	__isl_keep isl_schedule_node *node)
{
	isl_multi_union_pw_aff *mupa;

	if (!node)
		return NULL;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"not a band node", return NULL);
	if (isl_schedule_node_band_n_member(node) == 0) {
		isl_union_set *domain;

		domain = isl_schedule_node_get_universe_domain(node);
		return isl_union_map_from_domain(domain);
	}

	mupa = isl_schedule_node_band_get_partial_schedule(node);
	return isl_union_map_from_multi_union_pw_aff(mupa);
}

/* Make sure that that spaces of "node" and "mv" are the same.
 * Return -1 on error, reporting the error to the user.
 */
static int check_space_multi_val(__isl_keep isl_schedule_node *node,
	__isl_keep isl_multi_val *mv)
{
	isl_space *node_space, *mv_space;
	int equal;

	node_space = isl_schedule_node_band_get_space(node);
	mv_space = isl_multi_val_get_space(mv);
	equal = isl_space_tuple_is_equal(node_space, isl_dim_set,
					mv_space, isl_dim_set);
	isl_space_free(mv_space);
	isl_space_free(node_space);
	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"spaces don't match", return -1);

	return 0;
}

/* Multiply the partial schedule of the band node "node"
 * with the factors in "mv".
 */
__isl_give isl_schedule_node *isl_schedule_node_band_scale(
	__isl_take isl_schedule_node *node, __isl_take isl_multi_val *mv)
{
	isl_schedule_tree *tree;

	if (!node || !mv)
		goto error;
	if (check_space_multi_val(node, mv) < 0)
		goto error;

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_band_scale(tree, mv);
	return isl_schedule_node_graft_tree(node, tree);
error:
	isl_multi_val_free(mv);
	isl_schedule_node_free(node);
	return NULL;
}

/* Divide the partial schedule of the band node "node"
 * by the factors in "mv".
 */
__isl_give isl_schedule_node *isl_schedule_node_band_scale_down(
	__isl_take isl_schedule_node *node, __isl_take isl_multi_val *mv)
{
	isl_schedule_tree *tree;

	if (!node || !mv)
		goto error;
	if (check_space_multi_val(node, mv) < 0)
		goto error;

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_band_scale_down(tree, mv);
	return isl_schedule_node_graft_tree(node, tree);
error:
	isl_multi_val_free(mv);
	isl_schedule_node_free(node);
	return NULL;
}

/* Tile "node" with tile sizes "sizes".
 *
 * The current node is replaced by two nested nodes corresponding
 * to the tile dimensions and the point dimensions.
 *
 * Return a pointer to the outer (tile) node.
 *
 * If the scale tile loops option is set, then the tile loops
 * are scaled by the tile sizes.  If the shift point loops option is set,
 * then the point loops are shifted to start at zero.
 * In particular, these options affect the tile and point loop schedules
 * as follows
 *
 *	scale	shift	original	tile		point
 *
 *	0	0	i		floor(i/s)	i
 *	1	0	i		s * floor(i/s)	i
 *	0	1	i		floor(i/s)	i - s * floor(i/s)
 *	1	1	i		s * floor(i/s)	i - s * floor(i/s)
 */
__isl_give isl_schedule_node *isl_schedule_node_band_tile(
	__isl_take isl_schedule_node *node, __isl_take isl_multi_val *sizes)
{
	isl_schedule_tree *tree;

	if (!node || !sizes)
		goto error;

	if (check_space_multi_val(node, sizes) < 0)
		goto error;

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_band_tile(tree, sizes);
	return isl_schedule_node_graft_tree(node, tree);
error:
	isl_multi_val_free(sizes);
	isl_schedule_node_free(node);
	return NULL;
}

/* Move the band node "node" down to all the leaves in the subtree
 * rooted at "node".
 * Return a pointer to the node in the resulting tree that is in the same
 * position as the node pointed to by "node" in the original tree.
 *
 * If the node only has a leaf child, then nothing needs to be done.
 * Otherwise, the child of the node is removed and the result is
 * appended to all the leaves in the subtree rooted at the original child.
 * The original node is then replaced by the result of this operation.
 */
__isl_give isl_schedule_node *isl_schedule_node_band_sink(
	__isl_take isl_schedule_node *node)
{
	enum isl_schedule_node_type type;
	isl_schedule_tree *tree, *child;

	if (!node)
		return NULL;

	type = isl_schedule_node_get_type(node);
	if (type != isl_schedule_node_band)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"not a band node", isl_schedule_node_free(node));
	if (isl_schedule_tree_n_children(node->tree) == 0)
		return node;

	tree = isl_schedule_node_get_tree(node);
	child = isl_schedule_tree_get_child(tree, 0);
	tree = isl_schedule_tree_reset_children(tree);
	tree = isl_schedule_tree_append_to_leaves(child, tree);

	return isl_schedule_node_graft_tree(node, tree);
}

/* Split "node" into two nested band nodes, one with the first "pos"
 * dimensions and one with the remaining dimensions.
 * The schedules of the two band nodes live in anonymous spaces.
 */
__isl_give isl_schedule_node *isl_schedule_node_band_split(
	__isl_take isl_schedule_node *node, int pos)
{
	isl_schedule_tree *tree;

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_band_split(tree, pos);
	return isl_schedule_node_graft_tree(node, tree);
}

/* Return the domain of the domain node "node".
 */
__isl_give isl_union_set *isl_schedule_node_domain_get_domain(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_tree_domain_get_domain(node->tree);
}

/* Return the filter of the filter node "node".
 */
__isl_give isl_union_set *isl_schedule_node_filter_get_filter(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_tree_filter_get_filter(node->tree);
}

/* Replace the filter of filter node "node" by "filter".
 */
__isl_give isl_schedule_node *isl_schedule_node_filter_set_filter(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *filter)
{
	isl_schedule_tree *tree;

	if (!node || !filter)
		goto error;

	tree = isl_schedule_tree_copy(node->tree);
	tree = isl_schedule_tree_filter_set_filter(tree, filter);
	return isl_schedule_node_graft_tree(node, tree);
error:
	isl_schedule_node_free(node);
	isl_union_set_free(filter);
	return NULL;
}

/* Update the ancestors of "node" to point to the tree that "node"
 * now points to.
 * That is, replace the child in the original parent that corresponds
 * to the current tree position by node->tree and continue updating
 * the ancestors in the same way until the root is reached.
 *
 * If "node" originally points to a leaf of the schedule tree, then make sure
 * that in the end it points to a leaf in the updated schedule tree.
 */
static __isl_give isl_schedule_node *update_ancestors(
	__isl_take isl_schedule_node *node)
{
	int i, n;
	int is_leaf;
	isl_ctx *ctx;
	isl_schedule_tree *tree;

	node = isl_schedule_node_cow(node);
	if (!node)
		return NULL;

	ctx = isl_schedule_node_get_ctx(node);
	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	tree = isl_schedule_tree_copy(node->tree);

	for (i = n - 1; i >= 0; --i) {
		isl_schedule_tree *parent;

		parent = isl_schedule_tree_list_get_schedule_tree(
						    node->ancestors, i);
		parent = isl_schedule_tree_replace_child(parent,
						    node->child_pos[i], tree);
		node->ancestors = isl_schedule_tree_list_set_schedule_tree(
			    node->ancestors, i, isl_schedule_tree_copy(parent));

		tree = parent;
	}

	is_leaf = isl_schedule_tree_is_leaf(node->tree);
	node->schedule = isl_schedule_set_root(node->schedule, tree);
	if (is_leaf) {
		isl_schedule_tree_free(node->tree);
		node->tree = isl_schedule_node_get_leaf(node);
	}

	if (!node->schedule || !node->ancestors)
		return isl_schedule_node_free(node);

	return node;
}

/* Replace the subtree that "pos" points to by "tree", updating
 * the ancestors to maintain a consistent state.
 */
__isl_give isl_schedule_node *isl_schedule_node_graft_tree(
	__isl_take isl_schedule_node *pos, __isl_take isl_schedule_tree *tree)
{
	if (!tree || !pos)
		goto error;
	if (pos->tree == tree) {
		isl_schedule_tree_free(tree);
		return pos;
	}

	pos = isl_schedule_node_cow(pos);
	if (!pos)
		goto error;

	isl_schedule_tree_free(pos->tree);
	pos->tree = tree;

	return update_ancestors(pos);
error:
	isl_schedule_node_free(pos);
	isl_schedule_tree_free(tree);
	return NULL;
}

/* Make sure we can insert a node between "node" and its parent.
 * Return -1 on error, reporting the reason why we cannot insert a node.
 */
static int check_insert(__isl_keep isl_schedule_node *node)
{
	int has_parent;
	enum isl_schedule_node_type type;

	has_parent = isl_schedule_node_has_parent(node);
	if (has_parent < 0)
		return -1;
	if (!has_parent)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"cannot insert node outside of root", return -1);

	type = isl_schedule_node_get_parent_type(node);
	if (type == isl_schedule_node_error)
		return -1;
	if (type == isl_schedule_node_set || type == isl_schedule_node_sequence)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"cannot insert node between set or sequence node "
			"and its filter children", return -1);

	return 0;
}

/* Insert a band node with partial schedule "mupa" between "node" and
 * its parent.
 * Return a pointer to the new band node.
 */
__isl_give isl_schedule_node *isl_schedule_node_insert_partial_schedule(
	__isl_take isl_schedule_node *node,
	__isl_take isl_multi_union_pw_aff *mupa)
{
	isl_schedule_band *band;
	isl_schedule_tree *tree;

	if (check_insert(node) < 0)
		node = isl_schedule_node_free(node);

	tree = isl_schedule_node_get_tree(node);
	band = isl_schedule_band_from_multi_union_pw_aff(mupa);
	tree = isl_schedule_tree_insert_band(tree, band);
	node = isl_schedule_node_graft_tree(node, tree);

	return node;
}

/* Insert a filter node with filter "filter" between "node" and its parent.
 * Return a pointer to the new filter node.
 */
__isl_give isl_schedule_node *isl_schedule_node_insert_filter(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *filter)
{
	isl_schedule_tree *tree;

	if (check_insert(node) < 0)
		node = isl_schedule_node_free(node);

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_insert_filter(tree, filter);
	node = isl_schedule_node_graft_tree(node, tree);

	return node;
}

/* Attach the current subtree of "node" to a sequence of filter tree nodes
 * with filters described by "filters", attach this sequence
 * of filter tree nodes as children to a new tree of type "type" and
 * replace the original subtree of "node" by this new tree.
 */
static __isl_give isl_schedule_node *isl_schedule_node_insert_children(
	__isl_take isl_schedule_node *node,
	enum isl_schedule_node_type type,
	__isl_take isl_union_set_list *filters)
{
	int i, n;
	isl_ctx *ctx;
	isl_schedule_tree *tree;
	isl_schedule_tree_list *list;

	if (check_insert(node) < 0)
		node = isl_schedule_node_free(node);

	if (!node || !filters)
		goto error;

	ctx = isl_schedule_node_get_ctx(node);
	n = isl_union_set_list_n_union_set(filters);
	list = isl_schedule_tree_list_alloc(ctx, n);
	for (i = 0; i < n; ++i) {
		isl_schedule_tree *tree;
		isl_union_set *filter;

		tree = isl_schedule_node_get_tree(node);
		filter = isl_union_set_list_get_union_set(filters, i);
		tree = isl_schedule_tree_insert_filter(tree, filter);
		list = isl_schedule_tree_list_add(list, tree);
	}
	tree = isl_schedule_tree_from_children(type, list);
	node = isl_schedule_node_graft_tree(node, tree);

	isl_union_set_list_free(filters);
	return node;
error:
	isl_union_set_list_free(filters);
	isl_schedule_node_free(node);
	return NULL;
}

/* Insert a sequence node with child filters "filters" between "node" and
 * its parent.  That is, the tree that "node" points to is attached
 * to each of the child nodes of the filter nodes.
 * Return a pointer to the new sequence node.
 */
__isl_give isl_schedule_node *isl_schedule_node_insert_sequence(
	__isl_take isl_schedule_node *node,
	__isl_take isl_union_set_list *filters)
{
	return isl_schedule_node_insert_children(node,
					isl_schedule_node_sequence, filters);
}

/* Insert a set node with child filters "filters" between "node" and
 * its parent.  That is, the tree that "node" points to is attached
 * to each of the child nodes of the filter nodes.
 * Return a pointer to the new set node.
 */
__isl_give isl_schedule_node *isl_schedule_node_insert_set(
	__isl_take isl_schedule_node *node,
	__isl_take isl_union_set_list *filters)
{
	return isl_schedule_node_insert_children(node,
					isl_schedule_node_set, filters);
}

/* Print "node" to "p".
 */
__isl_give isl_printer *isl_printer_print_schedule_node(
	__isl_take isl_printer *p, __isl_keep isl_schedule_node *node)
{
	if (!node)
		return isl_printer_free(p);
	return isl_printer_print_schedule_tree_mark(p, node->schedule->root,
			isl_schedule_tree_list_n_schedule_tree(node->ancestors),
			node->child_pos);
}

void isl_schedule_node_dump(__isl_keep isl_schedule_node *node)
{
	isl_ctx *ctx;
	isl_printer *printer;

	if (!node)
		return;

	ctx = isl_schedule_node_get_ctx(node);
	printer = isl_printer_to_file(ctx, stderr);
	printer = isl_printer_set_yaml_style(printer, ISL_YAML_STYLE_BLOCK);
	printer = isl_printer_print_schedule_node(printer, node);

	isl_printer_free(printer);
}
