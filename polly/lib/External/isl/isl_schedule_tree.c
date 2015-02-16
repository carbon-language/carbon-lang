/*
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/map.h>
#include <isl_schedule_band.h>
#include <isl_schedule_private.h>

#undef EL
#define EL isl_schedule_tree

#include <isl_list_templ.h>

#undef BASE
#define BASE schedule_tree

#include <isl_list_templ.c>

/* Is "tree" the leaf of a schedule tree?
 */
int isl_schedule_tree_is_leaf(__isl_keep isl_schedule_tree *tree)
{
	return isl_schedule_tree_get_type(tree) == isl_schedule_node_leaf;
}

/* Create a new schedule tree of type "type".
 * The caller is responsible for filling in the type specific fields and
 * the children.
 */
static __isl_give isl_schedule_tree *isl_schedule_tree_alloc(isl_ctx *ctx,
	enum isl_schedule_node_type type)
{
	isl_schedule_tree *tree;

	if (type == isl_schedule_node_error)
		return NULL;

	tree = isl_calloc_type(ctx, isl_schedule_tree);
	if (!tree)
		return NULL;

	tree->ref = 1;
	tree->ctx = ctx;
	isl_ctx_ref(ctx);
	tree->type = type;

	return tree;
}

/* Return a fresh copy of "tree".
 */
__isl_take isl_schedule_tree *isl_schedule_tree_dup(
	__isl_keep isl_schedule_tree *tree)
{
	isl_ctx *ctx;
	isl_schedule_tree *dup;

	if (!tree)
		return NULL;

	ctx = isl_schedule_tree_get_ctx(tree);
	dup = isl_schedule_tree_alloc(ctx, tree->type);
	if (!dup)
		return NULL;

	switch (tree->type) {
	case isl_schedule_node_error:
		isl_die(ctx, isl_error_internal,
			"allocation should have failed",
			isl_schedule_tree_free(dup));
	case isl_schedule_node_band:
		dup->band = isl_schedule_band_copy(tree->band);
		if (!dup->band)
			return isl_schedule_tree_free(dup);
		break;
	case isl_schedule_node_domain:
		dup->domain = isl_union_set_copy(tree->domain);
		if (!dup->domain)
			return isl_schedule_tree_free(dup);
		break;
	case isl_schedule_node_filter:
		dup->filter = isl_union_set_copy(tree->filter);
		if (!dup->filter)
			return isl_schedule_tree_free(dup);
		break;
	case isl_schedule_node_leaf:
	case isl_schedule_node_sequence:
	case isl_schedule_node_set:
		break;
	}

	if (tree->children) {
		dup->children = isl_schedule_tree_list_copy(tree->children);
		if (!dup->children)
			return isl_schedule_tree_free(dup);
	}

	return dup;
}

/* Return an isl_schedule_tree that is equal to "tree" and that has only
 * a single reference.
 *
 * This function is called before a tree is modified.
 * A static tree (with negative reference count) should never be modified,
 * so it is not allowed to call this function on a static tree.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_cow(
	__isl_take isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	if (tree->ref < 0)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"static trees cannot be modified",
			return isl_schedule_tree_free(tree));

	if (tree->ref == 1)
		return tree;
	tree->ref--;
	return isl_schedule_tree_dup(tree);
}

/* Return a new reference to "tree".
 *
 * A static tree (with negative reference count) does not keep track
 * of the number of references and should not be modified.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_copy(
	__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	if (tree->ref < 0)
		return tree;

	tree->ref++;
	return tree;
}

/* Free "tree" and return NULL.
 */
__isl_null isl_schedule_tree *isl_schedule_tree_free(
	__isl_take isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;
	if (tree->ref < 0)
		return NULL;
	if (--tree->ref > 0)
		return NULL;

	switch (tree->type) {
	case isl_schedule_node_band:
		isl_schedule_band_free(tree->band);
		break;
	case isl_schedule_node_domain:
		isl_union_set_free(tree->domain);
		break;
	case isl_schedule_node_filter:
		isl_union_set_free(tree->filter);
		break;
	case isl_schedule_node_sequence:
	case isl_schedule_node_set:
	case isl_schedule_node_error:
	case isl_schedule_node_leaf:
		break;
	}
	isl_schedule_tree_list_free(tree->children);
	isl_ctx_deref(tree->ctx);
	free(tree);

	return NULL;
}

/* Create and return a new leaf schedule tree.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_leaf(isl_ctx *ctx)
{
	return isl_schedule_tree_alloc(ctx, isl_schedule_node_leaf);
}

/* Create a new band schedule tree referring to "band"
 * with no children.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_from_band(
	__isl_take isl_schedule_band *band)
{
	isl_ctx *ctx;
	isl_schedule_tree *tree;

	if (!band)
		return NULL;

	ctx = isl_schedule_band_get_ctx(band);
	tree = isl_schedule_tree_alloc(ctx, isl_schedule_node_band);
	if (!tree)
		goto error;

	tree->band = band;

	return tree;
error:
	isl_schedule_band_free(band);
	return NULL;
}

/* Create a new domain schedule tree with the given domain and no children.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_from_domain(
	__isl_take isl_union_set *domain)
{
	isl_ctx *ctx;
	isl_schedule_tree *tree;

	if (!domain)
		return NULL;

	ctx = isl_union_set_get_ctx(domain);
	tree = isl_schedule_tree_alloc(ctx, isl_schedule_node_domain);
	if (!tree)
		goto error;

	tree->domain = domain;

	return tree;
error:
	isl_union_set_free(domain);
	return NULL;
}

/* Create a new filter schedule tree with the given filter and no children.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_from_filter(
	__isl_take isl_union_set *filter)
{
	isl_ctx *ctx;
	isl_schedule_tree *tree;

	if (!filter)
		return NULL;

	ctx = isl_union_set_get_ctx(filter);
	tree = isl_schedule_tree_alloc(ctx, isl_schedule_node_filter);
	if (!tree)
		goto error;

	tree->filter = filter;

	return tree;
error:
	isl_union_set_free(filter);
	return NULL;
}

/* Create a new tree of the given type (isl_schedule_node_sequence or
 * isl_schedule_node_set) with the given children.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_from_children(
	enum isl_schedule_node_type type,
	__isl_take isl_schedule_tree_list *list)
{
	isl_ctx *ctx;
	isl_schedule_tree *tree;

	if (!list)
		return NULL;

	ctx = isl_schedule_tree_list_get_ctx(list);
	tree = isl_schedule_tree_alloc(ctx, type);
	if (!tree)
		goto error;

	tree->children = list;

	return tree;
error:
	isl_schedule_tree_list_free(list);
	return NULL;
}

/* Return the isl_ctx to which "tree" belongs.
 */
isl_ctx *isl_schedule_tree_get_ctx(__isl_keep isl_schedule_tree *tree)
{
	return tree ? tree->ctx : NULL;
}

/* Return the type of the root of the tree or isl_schedule_node_error
 * on error.
 */
enum isl_schedule_node_type isl_schedule_tree_get_type(
	__isl_keep isl_schedule_tree *tree)
{
	return tree ? tree->type : isl_schedule_node_error;
}

/* Does "tree" have any children, other than an implicit leaf.
 */
int isl_schedule_tree_has_children(__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return -1;

	return tree->children != NULL;
}

/* Return the number of children of "tree", excluding implicit leaves.
 */
int isl_schedule_tree_n_children(__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return -1;

	return isl_schedule_tree_list_n_schedule_tree(tree->children);
}

/* Return a copy of the (explicit) child at position "pos" of "tree".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_get_child(
	__isl_keep isl_schedule_tree *tree, int pos)
{
	if (!tree)
		return NULL;
	if (!tree->children)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"schedule tree has no explicit children", return NULL);
	return isl_schedule_tree_list_get_schedule_tree(tree->children, pos);
}

/* Return a copy of the (explicit) child at position "pos" of "tree" and
 * free "tree".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_child(
	__isl_take isl_schedule_tree *tree, int pos)
{
	isl_schedule_tree *child;

	child = isl_schedule_tree_get_child(tree, pos);
	isl_schedule_tree_free(tree);
	return child;
}

/* Remove all (explicit) children from "tree".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_reset_children(
	__isl_take isl_schedule_tree *tree)
{
	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		return NULL;
	tree->children = isl_schedule_tree_list_free(tree->children);
	return tree;
}

/* Replace the child at position "pos" of "tree" by "child".
 *
 * If the new child is a leaf, then it is not explicitly
 * recorded in the list of children.  Instead, the list of children
 * (which is assumed to have only one element) is removed.
 * Note that the children of set and sequence nodes are always
 * filters, so they cannot be replaced by empty trees.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_replace_child(
	__isl_take isl_schedule_tree *tree, int pos,
	__isl_take isl_schedule_tree *child)
{
	tree = isl_schedule_tree_cow(tree);
	if (!tree || !child)
		goto error;

	if (isl_schedule_tree_is_leaf(child)) {
		isl_schedule_tree_free(child);
		if (!tree->children && pos == 0)
			return tree;
		if (isl_schedule_tree_n_children(tree) != 1)
			isl_die(isl_schedule_tree_get_ctx(tree),
				isl_error_internal,
				"can only replace single child by leaf",
				goto error);
		return isl_schedule_tree_reset_children(tree);
	}

	if (!tree->children && pos == 0)
		tree->children =
			isl_schedule_tree_list_from_schedule_tree(child);
	else
		tree->children = isl_schedule_tree_list_set_schedule_tree(
				tree->children, pos, child);

	if (!tree->children)
		return isl_schedule_tree_free(tree);

	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_schedule_tree_free(child);
	return NULL;
}

/* Replace the (explicit) children of "tree" by "children"?
 */
__isl_give isl_schedule_tree *isl_schedule_tree_set_children(
	__isl_take isl_schedule_tree *tree,
	__isl_take isl_schedule_tree_list *children)
{
	tree = isl_schedule_tree_cow(tree);
	if (!tree || !children)
		goto error;
	isl_schedule_tree_list_free(tree->children);
	tree->children = children;
	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_schedule_tree_list_free(children);
	return NULL;
}

/* Create a new band schedule tree referring to "band"
 * with "tree" as single child.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_insert_band(
	__isl_take isl_schedule_tree *tree, __isl_take isl_schedule_band *band)
{
	isl_schedule_tree *res;

	res = isl_schedule_tree_from_band(band);
	return isl_schedule_tree_replace_child(res, 0, tree);
}

/* Create a new domain schedule tree with the given domain and
 * with "tree" as single child.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_insert_domain(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *domain)
{
	isl_schedule_tree *res;

	res = isl_schedule_tree_from_domain(domain);
	return isl_schedule_tree_replace_child(res, 0, tree);
}

/* Create a new filter schedule tree with the given filter and single child.
 *
 * If the root of "tree" is itself a filter node, then the two
 * filter nodes are merged into one node.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_insert_filter(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *filter)
{
	isl_schedule_tree *res;

	if (isl_schedule_tree_get_type(tree) == isl_schedule_node_filter) {
		isl_union_set *tree_filter;

		tree_filter = isl_schedule_tree_filter_get_filter(tree);
		tree_filter = isl_union_set_intersect(tree_filter, filter);
		tree = isl_schedule_tree_filter_set_filter(tree, tree_filter);
		return tree;
	}

	res = isl_schedule_tree_from_filter(filter);
	return isl_schedule_tree_replace_child(res, 0, tree);
}

/* Return the number of members in the band tree root.
 */
unsigned isl_schedule_tree_band_n_member(__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return 0;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return 0);

	return isl_schedule_band_n_member(tree->band);
}

/* Is the band member at position "pos" of the band tree root
 * marked coincident?
 */
int isl_schedule_tree_band_member_get_coincident(
	__isl_keep isl_schedule_tree *tree, int pos)
{
	if (!tree)
		return -1;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return -1);

	return isl_schedule_band_member_get_coincident(tree->band, pos);
}

/* Mark the given band member as being coincident or not
 * according to "coincident".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_member_set_coincident(
	__isl_take isl_schedule_tree *tree, int pos, int coincident)
{
	if (!tree)
		return NULL;
	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return isl_schedule_tree_free(tree));
	if (isl_schedule_tree_band_member_get_coincident(tree, pos) ==
								    coincident)
		return tree;
	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		return NULL;

	tree->band = isl_schedule_band_member_set_coincident(tree->band, pos,
							coincident);
	if (!tree->band)
		return isl_schedule_tree_free(tree);
	return tree;
}

/* Is the band tree root marked permutable?
 */
int isl_schedule_tree_band_get_permutable(__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return -1;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return -1);

	return isl_schedule_band_get_permutable(tree->band);
}

/* Mark the band tree root permutable or not according to "permutable"?
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_set_permutable(
	__isl_take isl_schedule_tree *tree, int permutable)
{
	if (!tree)
		return NULL;
	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return isl_schedule_tree_free(tree));
	if (isl_schedule_tree_band_get_permutable(tree) == permutable)
		return tree;
	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		return NULL;

	tree->band = isl_schedule_band_set_permutable(tree->band, permutable);
	if (!tree->band)
		return isl_schedule_tree_free(tree);
	return tree;
}

/* Return the schedule space of the band tree root.
 */
__isl_give isl_space *isl_schedule_tree_band_get_space(
	__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return NULL);

	return isl_schedule_band_get_space(tree->band);
}

/* Return the schedule of the band tree root in isolation.
 */
__isl_give isl_multi_union_pw_aff *isl_schedule_tree_band_get_partial_schedule(
	__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return NULL);

	return isl_schedule_band_get_partial_schedule(tree->band);
}

/* Return the domain of the domain tree root.
 */
__isl_give isl_union_set *isl_schedule_tree_domain_get_domain(
	__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	if (tree->type != isl_schedule_node_domain)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a domain node", return NULL);

	return isl_union_set_copy(tree->domain);
}

/* Replace the domain of domain tree root "tree" by "domain".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_domain_set_domain(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *domain)
{
	tree = isl_schedule_tree_cow(tree);
	if (!tree || !domain)
		goto error;

	if (tree->type != isl_schedule_node_domain)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a domain node", goto error);

	isl_union_set_free(tree->domain);
	tree->domain = domain;

	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_union_set_free(domain);
	return NULL;
}

/* Return the filter of the filter tree root.
 */
__isl_give isl_union_set *isl_schedule_tree_filter_get_filter(
	__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	if (tree->type != isl_schedule_node_filter)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a filter node", return NULL);

	return isl_union_set_copy(tree->filter);
}

/* Replace the filter of the filter tree root by "filter".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_filter_set_filter(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *filter)
{
	tree = isl_schedule_tree_cow(tree);
	if (!tree || !filter)
		goto error;

	if (tree->type != isl_schedule_node_filter)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a filter node", return NULL);

	isl_union_set_free(tree->filter);
	tree->filter = filter;

	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_union_set_free(filter);
	return NULL;
}

/* Set dim to the range dimension of "map" and abort the search.
 */
static int set_range_dim(__isl_take isl_map *map, void *user)
{
	int *dim = user;

	*dim = isl_map_dim(map, isl_dim_out);
	isl_map_free(map);

	return -1;
}

/* Return the dimension of the range of "umap".
 * "umap" is assumed not to be empty and
 * all maps inside "umap" are assumed to have the same range.
 *
 * We extract the range dimension from the first map in "umap".
 */
static int range_dim(__isl_keep isl_union_map *umap)
{
	int dim = -1;

	if (!umap)
		return -1;
	if (isl_union_map_n_map(umap) == 0)
		isl_die(isl_union_map_get_ctx(umap), isl_error_internal,
			"unexpected empty input", return -1);

	isl_union_map_foreach_map(umap, &set_range_dim, &dim);

	return dim;
}

/* Append an "extra" number of zeros to the range of "umap" and
 * return the result.
 */
static __isl_give isl_union_map *append_range(__isl_take isl_union_map *umap,
	int extra)
{
	isl_union_set *dom;
	isl_space *space;
	isl_multi_val *mv;
	isl_union_pw_multi_aff *suffix;
	isl_union_map *universe;
	isl_union_map *suffix_umap;

	universe = isl_union_map_universe(isl_union_map_copy(umap));
	dom = isl_union_map_domain(universe);
	space = isl_union_set_get_space(dom);
	space = isl_space_set_from_params(space);
	space = isl_space_add_dims(space, isl_dim_set, extra);
	mv = isl_multi_val_zero(space);

	suffix = isl_union_pw_multi_aff_multi_val_on_domain(dom, mv);
	suffix_umap = isl_union_map_from_union_pw_multi_aff(suffix);
	umap = isl_union_map_flat_range_product(umap, suffix_umap);

	return umap;
}

/* Move down to the first descendant of "tree" that contains any schedule
 * information or return "leaf" if there is no such descendant.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_first_schedule_descendant(
	__isl_take isl_schedule_tree *tree, __isl_keep isl_schedule_tree *leaf)
{
	while (isl_schedule_tree_get_type(tree) == isl_schedule_node_band &&
	    isl_schedule_tree_band_n_member(tree) == 0) {
		if (!isl_schedule_tree_has_children(tree)) {
			isl_schedule_tree_free(tree);
			return isl_schedule_tree_copy(leaf);
		}
		tree = isl_schedule_tree_child(tree, 0);
	}

	return tree;
}

static __isl_give isl_union_map *subtree_schedule_extend(
	__isl_keep isl_schedule_tree *tree, __isl_take isl_union_map *outer);

/* Extend the schedule map "outer" with the subtree schedule
 * of the (single) child of "tree", if any.
 *
 * If "tree" does not have any descendants (apart from those that
 * do not carry any schedule information), then we simply return "outer".
 * Otherwise, we extend the schedule map "outer" with the subtree schedule
 * of the single child.
 */
static __isl_give isl_union_map *subtree_schedule_extend_child(
	__isl_keep isl_schedule_tree *tree, __isl_take isl_union_map *outer)
{
	isl_schedule_tree *child;
	isl_union_map *res;

	if (!tree)
		return isl_union_map_free(outer);
	if (!isl_schedule_tree_has_children(tree))
		return outer;
	child = isl_schedule_tree_get_child(tree, 0);
	if (!child)
		return isl_union_map_free(outer);
	res = subtree_schedule_extend(child, outer);
	isl_schedule_tree_free(child);
	return res;
}

/* Extract the parameter space from one of the children of "tree",
 * which are assumed to be filters.
 */
static __isl_give isl_space *extract_space_from_filter_child(
	__isl_keep isl_schedule_tree *tree)
{
	isl_space *space;
	isl_union_set *dom;
	isl_schedule_tree *child;

	child = isl_schedule_tree_list_get_schedule_tree(tree->children, 0);
	dom = isl_schedule_tree_filter_get_filter(child);
	space = isl_union_set_get_space(dom);
	isl_union_set_free(dom);
	isl_schedule_tree_free(child);

	return space;
}

/* Extend the schedule map "outer" with the subtree schedule
 * of a set or sequence node.
 *
 * The schedule for the set or sequence node itself is composed of
 * pieces of the form
 *
 *	filter -> []
 *
 * or
 *
 *	filter -> [index]
 *
 * The first form is used if there is only a single child or
 * if the current node is a set node and the schedule_separate_components
 * option is not set.
 *
 * Each of the pieces above is extended with the subtree schedule of
 * the child of the corresponding filter, if any, padded with zeros
 * to ensure that all pieces have the same range dimension.
 */
static __isl_give isl_union_map *subtree_schedule_extend_from_children(
	__isl_keep isl_schedule_tree *tree, __isl_take isl_union_map *outer)
{
	int i, n;
	int dim;
	int separate;
	isl_ctx *ctx;
	isl_val *v = NULL;
	isl_multi_val *mv;
	isl_space *space;
	isl_union_map *umap;

	if (!tree)
		return NULL;

	ctx = isl_schedule_tree_get_ctx(tree);
	if (!tree->children)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"missing children", return NULL);
	n = isl_schedule_tree_list_n_schedule_tree(tree->children);
	if (n == 0)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"missing children", return NULL);

	separate = n > 1 && (tree->type == isl_schedule_node_sequence ||
			    isl_options_get_schedule_separate_components(ctx));

	space = extract_space_from_filter_child(tree);

	umap = isl_union_map_empty(isl_space_copy(space));
	space = isl_space_set_from_params(space);
	if (separate) {
		space = isl_space_add_dims(space, isl_dim_set, 1);
		v = isl_val_zero(ctx);
	}
	mv = isl_multi_val_zero(space);

	dim = isl_multi_val_dim(mv, isl_dim_set);
	for (i = 0; i < n; ++i) {
		isl_union_pw_multi_aff *upma;
		isl_union_map *umap_i;
		isl_union_set *dom;
		isl_schedule_tree *child;
		int dim_i;
		int empty;

		child = isl_schedule_tree_list_get_schedule_tree(
							tree->children, i);
		dom = isl_schedule_tree_filter_get_filter(child);

		if (separate) {
			mv = isl_multi_val_set_val(mv, 0, isl_val_copy(v));
			v = isl_val_add_ui(v, 1);
		}
		upma = isl_union_pw_multi_aff_multi_val_on_domain(dom,
							isl_multi_val_copy(mv));
		umap_i = isl_union_map_from_union_pw_multi_aff(upma);
		umap_i = isl_union_map_flat_range_product(
					    isl_union_map_copy(outer), umap_i);
		umap_i = subtree_schedule_extend_child(child, umap_i);
		isl_schedule_tree_free(child);

		empty = isl_union_map_is_empty(umap_i);
		if (empty < 0)
			umap_i = isl_union_map_free(umap_i);
		else if (empty) {
			isl_union_map_free(umap_i);
			continue;
		}

		dim_i = range_dim(umap_i);
		if (dim_i < 0) {
			umap = isl_union_map_free(umap);
		} else if (dim < dim_i) {
			umap = append_range(umap, dim_i - dim);
			dim = dim_i;
		} else if (dim_i < dim) {
			umap_i = append_range(umap_i, dim - dim_i);
		}
		umap = isl_union_map_union(umap, umap_i);
	}

	isl_val_free(v);
	isl_multi_val_free(mv);
	isl_union_map_free(outer);

	return umap;
}

/* Extend the schedule map "outer" with the subtree schedule of "tree".
 *
 * If the root of the tree is a set or a sequence, then we extend
 * the schedule map in subtree_schedule_extend_from_children.
 * Otherwise, we extend the schedule map with the partial schedule
 * corresponding to the root of the tree and then continue with
 * the single child of this root.
 */
static __isl_give isl_union_map *subtree_schedule_extend(
	__isl_keep isl_schedule_tree *tree, __isl_take isl_union_map *outer)
{
	isl_multi_union_pw_aff *mupa;
	isl_union_map *umap;
	isl_union_set *domain;

	if (!tree)
		return NULL;

	switch (tree->type) {
	case isl_schedule_node_error:
		return isl_union_map_free(outer);
	case isl_schedule_node_band:
		if (isl_schedule_tree_band_n_member(tree) == 0)
			return subtree_schedule_extend_child(tree, outer);
		mupa = isl_schedule_band_get_partial_schedule(tree->band);
		umap = isl_union_map_from_multi_union_pw_aff(mupa);
		outer = isl_union_map_flat_range_product(outer, umap);
		umap = subtree_schedule_extend_child(tree, outer);
		break;
	case isl_schedule_node_domain:
		domain = isl_schedule_tree_domain_get_domain(tree);
		umap = isl_union_map_from_domain(domain);
		outer = isl_union_map_flat_range_product(outer, umap);
		umap = subtree_schedule_extend_child(tree, outer);
		break;
	case isl_schedule_node_filter:
		domain = isl_schedule_tree_filter_get_filter(tree);
		umap = isl_union_map_from_domain(domain);
		outer = isl_union_map_flat_range_product(outer, umap);
		umap = subtree_schedule_extend_child(tree, outer);
		break;
	case isl_schedule_node_leaf:
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"leaf node should be handled by caller", return NULL);
	case isl_schedule_node_set:
	case isl_schedule_node_sequence:
		umap = subtree_schedule_extend_from_children(tree, outer);
		break;
	}

	return umap;
}

static __isl_give isl_union_set *initial_domain(
	__isl_keep isl_schedule_tree *tree);

/* Extract a universe domain from the children of the tree root "tree",
 * which is a set or sequence, meaning that its children are filters.
 * In particular, return the union of the universes of the filters.
 */
static __isl_give isl_union_set *initial_domain_from_children(
	__isl_keep isl_schedule_tree *tree)
{
	int i, n;
	isl_space *space;
	isl_union_set *domain;

	if (!tree->children)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"missing children", return NULL);
	n = isl_schedule_tree_list_n_schedule_tree(tree->children);
	if (n == 0)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"missing children", return NULL);

	space = extract_space_from_filter_child(tree);
	domain = isl_union_set_empty(space);

	for (i = 0; i < n; ++i) {
		isl_schedule_tree *child;
		isl_union_set *domain_i;

		child = isl_schedule_tree_get_child(tree, i);
		domain_i = initial_domain(child);
		domain = isl_union_set_union(domain, domain_i);
		isl_schedule_tree_free(child);
	}

	return domain;
}

/* Extract a universe domain from the tree root "tree".
 * The caller is responsible for making sure that this node
 * would not be skipped by isl_schedule_tree_first_schedule_descendant
 * and that it is not a leaf node.
 */
static __isl_give isl_union_set *initial_domain(
	__isl_keep isl_schedule_tree *tree)
{
	isl_multi_union_pw_aff *mupa;
	isl_union_set *domain;

	if (!tree)
		return NULL;

	switch (tree->type) {
	case isl_schedule_node_error:
		return NULL;
	case isl_schedule_node_band:
		if (isl_schedule_tree_band_n_member(tree) == 0)
			isl_die(isl_schedule_tree_get_ctx(tree),
				isl_error_internal,
				"0D band should be handled by caller",
				return NULL);
		mupa = isl_schedule_band_get_partial_schedule(tree->band);
		domain = isl_multi_union_pw_aff_domain(mupa);
		domain = isl_union_set_universe(domain);
		break;
	case isl_schedule_node_domain:
		domain = isl_schedule_tree_domain_get_domain(tree);
		domain = isl_union_set_universe(domain);
		break;
	case isl_schedule_node_filter:
		domain = isl_schedule_tree_filter_get_filter(tree);
		domain = isl_union_set_universe(domain);
		break;
	case isl_schedule_node_leaf:
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"leaf node should be handled by caller", return NULL);
	case isl_schedule_node_set:
	case isl_schedule_node_sequence:
		domain = initial_domain_from_children(tree);
		break;
	}

	return domain;
}

/* Return the subtree schedule of a node that contains some schedule
 * information, i.e., a node that would not be skipped by
 * isl_schedule_tree_first_schedule_descendant and that is not a leaf.
 *
 * We start with an initial zero-dimensional subtree schedule based
 * on the domain information in the root node and then extend it
 * based on the schedule information in the root node and its descendants.
 */
__isl_give isl_union_map *isl_schedule_tree_get_subtree_schedule_union_map(
	__isl_keep isl_schedule_tree *tree)
{
	isl_union_set *domain;
	isl_union_map *umap;

	domain = initial_domain(tree);
	umap = isl_union_map_from_domain(domain);
	return subtree_schedule_extend(tree, umap);
}

/* Multiply the partial schedule of the band root node of "tree"
 * with the factors in "mv".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_scale(
	__isl_take isl_schedule_tree *tree, __isl_take isl_multi_val *mv)
{
	if (!tree || !mv)
		goto error;
	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", goto error);

	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		goto error;

	tree->band = isl_schedule_band_scale(tree->band, mv);
	if (!tree->band)
		return isl_schedule_tree_free(tree);

	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_multi_val_free(mv);
	return NULL;
}

/* Divide the partial schedule of the band root node of "tree"
 * by the factors in "mv".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_scale_down(
	__isl_take isl_schedule_tree *tree, __isl_take isl_multi_val *mv)
{
	if (!tree || !mv)
		goto error;
	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", goto error);

	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		goto error;

	tree->band = isl_schedule_band_scale_down(tree->band, mv);
	if (!tree->band)
		return isl_schedule_tree_free(tree);

	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_multi_val_free(mv);
	return NULL;
}

/* Tile the band root node of "tree" with tile sizes "sizes".
 *
 * We duplicate the band node, change the schedule of one of them
 * to the tile schedule and the other to the point schedule and then
 * attach the point band as a child to the tile band.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_tile(
	__isl_take isl_schedule_tree *tree, __isl_take isl_multi_val *sizes)
{
	isl_schedule_tree *child = NULL;

	if (!tree || !sizes)
		goto error;
	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", goto error);

	child = isl_schedule_tree_copy(tree);
	tree = isl_schedule_tree_cow(tree);
	child = isl_schedule_tree_cow(child);
	if (!tree || !child)
		goto error;

	tree->band = isl_schedule_band_tile(tree->band,
					    isl_multi_val_copy(sizes));
	if (!tree->band)
		goto error;
	child->band = isl_schedule_band_point(child->band, tree->band, sizes);
	if (!child->band)
		child = isl_schedule_tree_free(child);

	tree = isl_schedule_tree_replace_child(tree, 0, child);

	return tree;
error:
	isl_schedule_tree_free(child);
	isl_schedule_tree_free(tree);
	isl_multi_val_free(sizes);
	return NULL;
}

/* Split the band root node of "tree" into two nested band nodes,
 * one with the first "pos" dimensions and
 * one with the remaining dimensions.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_split(
	__isl_take isl_schedule_tree *tree, int pos)
{
	int n;
	isl_schedule_tree *child;

	if (!tree)
		return NULL;
	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return isl_schedule_tree_free(tree));

	n = isl_schedule_tree_band_n_member(tree);
	if (pos < 0 || pos > n)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"position out of bounds",
			return isl_schedule_tree_free(tree));

	child = isl_schedule_tree_copy(tree);
	tree = isl_schedule_tree_cow(tree);
	child = isl_schedule_tree_cow(child);
	if (!tree || !child)
		goto error;

	child->band = isl_schedule_band_drop(child->band, 0, pos);
	tree->band = isl_schedule_band_drop(tree->band, pos, n - pos);
	if (!child->band || !tree->band)
		goto error;

	tree = isl_schedule_tree_replace_child(tree, 0, child);

	return tree;
error:
	isl_schedule_tree_free(child);
	isl_schedule_tree_free(tree);
	return NULL;
}

/* Attach "tree2" at each of the leaves of "tree1".
 *
 * If "tree1" does not have any explicit children, then make "tree2"
 * its single child.  Otherwise, attach "tree2" to the leaves of
 * each of the children of "tree1".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_append_to_leaves(
	__isl_take isl_schedule_tree *tree1,
	__isl_take isl_schedule_tree *tree2)
{
	int i, n;

	if (!tree1 || !tree2)
		goto error;
	n = isl_schedule_tree_n_children(tree1);
	if (n == 0) {
		isl_schedule_tree_list *list;
		list = isl_schedule_tree_list_from_schedule_tree(tree2);
		tree1 = isl_schedule_tree_set_children(tree1, list);
		return tree1;
	}
	for (i = 0; i < n; ++i) {
		isl_schedule_tree *child;

		child = isl_schedule_tree_get_child(tree1, i);
		child = isl_schedule_tree_append_to_leaves(child,
					isl_schedule_tree_copy(tree2));
		tree1 = isl_schedule_tree_replace_child(tree1, i, child);
	}

	isl_schedule_tree_free(tree2);
	return tree1;
error:
	isl_schedule_tree_free(tree1);
	isl_schedule_tree_free(tree2);
	return NULL;
}

/* Are any members in "band" marked coincident?
 */
static int any_coincident(__isl_keep isl_schedule_band *band)
{
	int i, n;

	n = isl_schedule_band_n_member(band);
	for (i = 0; i < n; ++i)
		if (isl_schedule_band_member_get_coincident(band, i))
			return 1;

	return 0;
}

/* Print the band node "band" to "p".
 *
 * The permutable and coincident properties are only printed if they
 * are different from the defaults.
 * The coincident property is always printed in YAML flow style.
 */
static __isl_give isl_printer *print_tree_band(__isl_take isl_printer *p,
	__isl_keep isl_schedule_band *band)
{
	p = isl_printer_print_str(p, "schedule");
	p = isl_printer_yaml_next(p);
	p = isl_printer_print_str(p, "\"");
	p = isl_printer_print_multi_union_pw_aff(p, band->mupa);
	p = isl_printer_print_str(p, "\"");
	if (isl_schedule_band_get_permutable(band)) {
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, "permutable");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_int(p, 1);
	}
	if (any_coincident(band)) {
		int i, n;
		int style;

		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, "coincident");
		p = isl_printer_yaml_next(p);
		style = isl_printer_get_yaml_style(p);
		p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_FLOW);
		p = isl_printer_yaml_start_sequence(p);
		n = isl_schedule_band_n_member(band);
		for (i = 0; i < n; ++i) {
			p = isl_printer_print_int(p,
			    isl_schedule_band_member_get_coincident(band, i));
			p = isl_printer_yaml_next(p);
		}
		p = isl_printer_yaml_end_sequence(p);
		p = isl_printer_set_yaml_style(p, style);
	}

	return p;
}

/* Print "tree" to "p".
 *
 * If "n_ancestor" is non-negative, then "child_pos" contains the child
 * positions of a descendant of the current node that should be marked
 * (by the comment "YOU ARE HERE").  In particular, if "n_ancestor"
 * is zero, then the current node should be marked.
 * The marking is only printed in YAML block format.
 *
 * Implicit leaf nodes are not printed, except if they correspond
 * to the node that should be marked.
 */
__isl_give isl_printer *isl_printer_print_schedule_tree_mark(
	__isl_take isl_printer *p, __isl_keep isl_schedule_tree *tree,
	int n_ancestor, int *child_pos)
{
	int i, n;
	int sequence = 0;
	int block;

	block = isl_printer_get_yaml_style(p) == ISL_YAML_STYLE_BLOCK;

	p = isl_printer_yaml_start_mapping(p);
	if (n_ancestor == 0 && block) {
		p = isl_printer_print_str(p, "# YOU ARE HERE");
		p = isl_printer_end_line(p);
		p = isl_printer_start_line(p);
	}
	switch (tree->type) {
	case isl_schedule_node_error:
		p = isl_printer_print_str(p, "ERROR");
		break;
	case isl_schedule_node_leaf:
		p = isl_printer_print_str(p, "leaf");
		break;
	case isl_schedule_node_sequence:
		p = isl_printer_print_str(p, "sequence");
		sequence = 1;
		break;
	case isl_schedule_node_set:
		p = isl_printer_print_str(p, "set");
		sequence = 1;
		break;
	case isl_schedule_node_domain:
		p = isl_printer_print_str(p, "domain");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, "\"");
		p = isl_printer_print_union_set(p, tree->domain);
		p = isl_printer_print_str(p, "\"");
		break;
	case isl_schedule_node_filter:
		p = isl_printer_print_str(p, "filter");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, "\"");
		p = isl_printer_print_union_set(p, tree->filter);
		p = isl_printer_print_str(p, "\"");
		break;
	case isl_schedule_node_band:
		p = print_tree_band(p, tree->band);
		break;
	}
	p = isl_printer_yaml_next(p);

	if (!tree->children) {
		if (n_ancestor > 0 && block) {
			isl_schedule_tree *leaf;

			p = isl_printer_print_str(p, "child");
			p = isl_printer_yaml_next(p);
			leaf = isl_schedule_tree_leaf(isl_printer_get_ctx(p));
			p = isl_printer_print_schedule_tree_mark(p,
					leaf, 0, NULL);
			isl_schedule_tree_free(leaf);
			p = isl_printer_yaml_next(p);
		}
		return isl_printer_yaml_end_mapping(p);
	}

	if (sequence) {
		p = isl_printer_yaml_start_sequence(p);
	} else {
		p = isl_printer_print_str(p, "child");
		p = isl_printer_yaml_next(p);
	}

	n = isl_schedule_tree_list_n_schedule_tree(tree->children);
	for (i = 0; i < n; ++i) {
		isl_schedule_tree *t;

		t = isl_schedule_tree_get_child(tree, i);
		if (n_ancestor > 0 && child_pos[0] == i)
			p = isl_printer_print_schedule_tree_mark(p, t,
						n_ancestor - 1, child_pos + 1);
		else
			p = isl_printer_print_schedule_tree_mark(p, t,
						-1, NULL);
		isl_schedule_tree_free(t);

		p = isl_printer_yaml_next(p);
	}

	if (sequence)
		p = isl_printer_yaml_end_sequence(p);
	p = isl_printer_yaml_end_mapping(p);

	return p;
}

/* Print "tree" to "p".
 */
__isl_give isl_printer *isl_printer_print_schedule_tree(
	__isl_take isl_printer *p, __isl_keep isl_schedule_tree *tree)
{
	return isl_printer_print_schedule_tree_mark(p, tree, -1, NULL);
}

void isl_schedule_tree_dump(__isl_keep isl_schedule_tree *tree)
{
	isl_ctx *ctx;
	isl_printer *printer;

	if (!tree)
		return;

	ctx = isl_schedule_tree_get_ctx(tree);
	printer = isl_printer_to_file(ctx, stderr);
	printer = isl_printer_set_yaml_style(printer, ISL_YAML_STYLE_BLOCK);
	printer = isl_printer_print_schedule_tree(printer, tree);

	isl_printer_free(printer);
}
