#ifndef ISL_SCHEDLUE_TREE_H
#define ISL_SCHEDLUE_TREE_H

#include <isl_schedule_band.h>
#include <isl/schedule.h>
#include <isl/set.h>
#include <isl/union_set.h>

struct isl_schedule_tree;
typedef struct isl_schedule_tree isl_schedule_tree;

ISL_DECLARE_LIST(schedule_tree)

/* A schedule (sub)tree.
 *
 * The leaves of a tree are not explicitly represented inside
 * the isl_schedule_tree.  If a tree consists of only a leaf,
 * then it is equal to the static object isl_schedule_tree_empty.
 *
 * ctx may be NULL if type is isl_schedule_node_leaf.
 * In this case, ref has a negative value.
 *
 * The "band" field is valid when type is isl_schedule_node_band.
 * The "domain" field is valid when type is isl_schedule_node_domain
 * and introduces the statement instances scheduled by the tree.
 * The "filter" field is valid when type is isl_schedule_node_filter
 * and represents the statement instances selected by the node.
 *
 * The "children" field is valid for all types except
 * isl_schedule_node_leaf.  This field is NULL if there are
 * no children (except for the implicit leaves).
 */
struct isl_schedule_tree {
	int ref;
	isl_ctx *ctx;
	enum isl_schedule_node_type type;
	union {
		isl_schedule_band *band;
		isl_union_set *domain;
		isl_union_set *filter;
	};
	isl_schedule_tree_list *children;
};

isl_ctx *isl_schedule_tree_get_ctx(__isl_keep isl_schedule_tree *tree);
enum isl_schedule_node_type isl_schedule_tree_get_type(
	__isl_keep isl_schedule_tree *tree);

__isl_give isl_schedule_tree *isl_schedule_tree_leaf(isl_ctx *ctx);
int isl_schedule_tree_is_leaf(__isl_keep isl_schedule_tree *tree);

int isl_schedule_tree_plain_is_equal(__isl_keep isl_schedule_tree *tree1,
	__isl_keep isl_schedule_tree *tree2);

__isl_give isl_schedule_tree *isl_schedule_tree_copy(
	__isl_keep isl_schedule_tree *tree);
__isl_null isl_schedule_tree *isl_schedule_tree_free(
	__isl_take isl_schedule_tree *tree);

__isl_give isl_schedule_tree *isl_schedule_tree_from_band(
	__isl_take isl_schedule_band *band);
__isl_give isl_schedule_tree *isl_schedule_tree_from_domain(
	__isl_take isl_union_set *domain);
__isl_give isl_schedule_tree *isl_schedule_tree_from_filter(
	__isl_take isl_union_set *filter);
__isl_give isl_schedule_tree *isl_schedule_tree_from_children(
	enum isl_schedule_node_type type,
	__isl_take isl_schedule_tree_list *list);
__isl_give isl_schedule_tree *isl_schedule_tree_from_pair(
	enum isl_schedule_node_type type, __isl_take isl_schedule_tree *tree1,
	__isl_take isl_schedule_tree *tree2);

__isl_give isl_space *isl_schedule_tree_band_get_space(
	__isl_keep isl_schedule_tree *tree);
__isl_give isl_multi_union_pw_aff *isl_schedule_tree_band_get_partial_schedule(
	__isl_keep isl_schedule_tree *tree);
__isl_give isl_union_set *isl_schedule_tree_domain_get_domain(
	__isl_keep isl_schedule_tree *tree);
__isl_give isl_schedule_tree *isl_schedule_tree_domain_set_domain(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *domain);
__isl_give isl_union_set *isl_schedule_tree_filter_get_filter(
	__isl_keep isl_schedule_tree *tree);
__isl_give isl_schedule_tree *isl_schedule_tree_filter_set_filter(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *filter);

__isl_give isl_schedule_tree *isl_schedule_tree_first_schedule_descendant(
	__isl_take isl_schedule_tree *tree, __isl_keep isl_schedule_tree *leaf);
__isl_give isl_union_map *isl_schedule_tree_get_subtree_schedule_union_map(
	__isl_keep isl_schedule_tree *tree);

unsigned isl_schedule_tree_band_n_member(__isl_keep isl_schedule_tree *tree);

int isl_schedule_tree_band_member_get_coincident(
	__isl_keep isl_schedule_tree *tree, int pos);
__isl_give isl_schedule_tree *isl_schedule_tree_band_member_set_coincident(
	__isl_take isl_schedule_tree *tree, int pos, int coincident);
int isl_schedule_tree_band_get_permutable(__isl_keep isl_schedule_tree *tree);
__isl_give isl_schedule_tree *isl_schedule_tree_band_set_permutable(
	__isl_take isl_schedule_tree *tree, int permutable);

int isl_schedule_tree_has_children(__isl_keep isl_schedule_tree *tree);
int isl_schedule_tree_n_children(__isl_keep isl_schedule_tree *tree);
__isl_give isl_schedule_tree *isl_schedule_tree_get_child(
	__isl_keep isl_schedule_tree *tree, int pos);

__isl_give isl_schedule_tree *isl_schedule_tree_insert_band(
	__isl_take isl_schedule_tree *tree, __isl_take isl_schedule_band *band);
__isl_give isl_schedule_tree *isl_schedule_tree_insert_domain(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *domain);
__isl_give isl_schedule_tree *isl_schedule_tree_insert_filter(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *filter);
__isl_give isl_schedule_tree *isl_schedule_tree_children_insert_filter(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *filter);

__isl_give isl_schedule_tree *isl_schedule_tree_append_to_leaves(
	__isl_take isl_schedule_tree *tree1,
	__isl_take isl_schedule_tree *tree2);

__isl_give isl_schedule_tree *isl_schedule_tree_band_scale(
	__isl_take isl_schedule_tree *tree, __isl_take isl_multi_val *mv);
__isl_give isl_schedule_tree *isl_schedule_tree_band_scale_down(
	__isl_take isl_schedule_tree *tree, __isl_take isl_multi_val *mv);
__isl_give isl_schedule_tree *isl_schedule_tree_band_tile(
	__isl_take isl_schedule_tree *tree, __isl_take isl_multi_val *sizes);
__isl_give isl_schedule_tree *isl_schedule_tree_band_split(
	__isl_take isl_schedule_tree *tree, int pos);
__isl_give isl_schedule_tree *isl_schedule_tree_band_gist(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *context);

__isl_give isl_schedule_tree *isl_schedule_tree_child(
	__isl_take isl_schedule_tree *tree, int pos);
__isl_give isl_schedule_tree *isl_schedule_tree_reset_children(
	__isl_take isl_schedule_tree *tree);
__isl_give isl_schedule_tree *isl_schedule_tree_drop_child(
	__isl_take isl_schedule_tree *tree, int pos);
__isl_give isl_schedule_tree *isl_schedule_tree_replace_child(
	__isl_take isl_schedule_tree *tree, int pos,
	__isl_take isl_schedule_tree *new_child);

__isl_give isl_schedule_tree *isl_schedule_tree_reset_user(
	__isl_take isl_schedule_tree *tree);
__isl_give isl_schedule_tree *isl_schedule_tree_align_params(
	__isl_take isl_schedule_tree *tree, __isl_take isl_space *space);
__isl_give isl_schedule_tree *isl_schedule_tree_pullback_union_pw_multi_aff(
	__isl_take isl_schedule_tree *tree,
	__isl_take isl_union_pw_multi_aff *upma);

__isl_give isl_printer *isl_printer_print_schedule_tree(
	__isl_take isl_printer *p, __isl_keep isl_schedule_tree *tree);
__isl_give isl_printer *isl_printer_print_schedule_tree_mark(
	__isl_take isl_printer *p, __isl_keep isl_schedule_tree *tree,
	int n_ancestor, int *child_pos);

#endif
