#ifndef ISL_SCHEDULE_NODE_H
#define ISL_SCHEDULE_NODE_H

#include <isl/schedule_type.h>
#include <isl/union_set_type.h>
#include <isl/aff_type.h>
#include <isl/val.h>
#include <isl/space.h>

#if defined(__cplusplus)
extern "C" {
#endif

__isl_give isl_schedule_node *isl_schedule_node_from_domain(
	__isl_take isl_union_set *domain);
__isl_give isl_schedule_node *isl_schedule_node_copy(
	__isl_keep isl_schedule_node *node);
__isl_null isl_schedule_node *isl_schedule_node_free(
	__isl_take isl_schedule_node *node);

int isl_schedule_node_is_equal(__isl_keep isl_schedule_node *node1,
	__isl_keep isl_schedule_node *node2);

isl_ctx *isl_schedule_node_get_ctx(__isl_keep isl_schedule_node *node);
enum isl_schedule_node_type isl_schedule_node_get_type(
	__isl_keep isl_schedule_node *node);
enum isl_schedule_node_type isl_schedule_node_get_parent_type(
	__isl_keep isl_schedule_node *node);
__isl_give isl_schedule *isl_schedule_node_get_schedule(
	__isl_keep isl_schedule_node *node);

int isl_schedule_node_foreach_descendant(__isl_keep isl_schedule_node *node,
	int (*fn)(__isl_keep isl_schedule_node *node, void *user), void *user);
int isl_schedule_node_foreach_ancestor_top_down(
	__isl_keep isl_schedule_node *node,
	int (*fn)(__isl_keep isl_schedule_node *node, void *user), void *user);
__isl_give isl_schedule_node *isl_schedule_node_map_descendant(
	__isl_take isl_schedule_node *node,
	__isl_give isl_schedule_node *(*fn)(__isl_take isl_schedule_node *node,
		void *user), void *user);

int isl_schedule_node_get_tree_depth(__isl_keep isl_schedule_node *node);
int isl_schedule_node_has_parent(__isl_keep isl_schedule_node *node);
int isl_schedule_node_has_children(__isl_keep isl_schedule_node *node);
int isl_schedule_node_has_previous_sibling(__isl_keep isl_schedule_node *node);
int isl_schedule_node_has_next_sibling(__isl_keep isl_schedule_node *node);
int isl_schedule_node_n_children(__isl_keep isl_schedule_node *node);
int isl_schedule_node_get_child_position(__isl_keep isl_schedule_node *node);
int isl_schedule_node_get_ancestor_child_position(
	__isl_keep isl_schedule_node *node,
	__isl_keep isl_schedule_node *ancestor);
__isl_give isl_schedule_node *isl_schedule_node_get_child(
	__isl_keep isl_schedule_node *node, int pos);
__isl_give isl_schedule_node *isl_schedule_node_get_shared_ancestor(
	__isl_keep isl_schedule_node *node1,
	__isl_keep isl_schedule_node *node2);

__isl_give isl_schedule_node *isl_schedule_node_root(
	__isl_take isl_schedule_node *node);
__isl_give isl_schedule_node *isl_schedule_node_parent(
	__isl_take isl_schedule_node *node);
__isl_give isl_schedule_node *isl_schedule_node_ancestor(
	__isl_take isl_schedule_node *node, int generation);
__isl_give isl_schedule_node *isl_schedule_node_child(
	__isl_take isl_schedule_node *node, int pos);
__isl_give isl_schedule_node *isl_schedule_node_first_child(
	__isl_take isl_schedule_node *node);
__isl_give isl_schedule_node *isl_schedule_node_previous_sibling(
	__isl_take isl_schedule_node *node);
__isl_give isl_schedule_node *isl_schedule_node_next_sibling(
	__isl_take isl_schedule_node *node);

__isl_give isl_space *isl_schedule_node_band_get_space(
	__isl_keep isl_schedule_node *node);
__isl_give isl_multi_union_pw_aff *isl_schedule_node_band_get_partial_schedule(
	__isl_keep isl_schedule_node *node);
__isl_give isl_union_map *isl_schedule_node_band_get_partial_schedule_union_map(
	__isl_keep isl_schedule_node *node);
unsigned isl_schedule_node_band_n_member(__isl_keep isl_schedule_node *node);
int isl_schedule_node_band_member_get_coincident(
	__isl_keep isl_schedule_node *node, int pos);
__isl_give isl_schedule_node *isl_schedule_node_band_member_set_coincident(
	__isl_take isl_schedule_node *node, int pos, int coincident);
int isl_schedule_node_band_get_permutable(__isl_keep isl_schedule_node *node);
__isl_give isl_schedule_node *isl_schedule_node_band_set_permutable(
	__isl_take isl_schedule_node *node, int permutable);

int isl_options_set_tile_scale_tile_loops(isl_ctx *ctx, int val);
int isl_options_get_tile_scale_tile_loops(isl_ctx *ctx);
int isl_options_set_tile_shift_point_loops(isl_ctx *ctx, int val);
int isl_options_get_tile_shift_point_loops(isl_ctx *ctx);

__isl_give isl_schedule_node *isl_schedule_node_band_scale(
	__isl_take isl_schedule_node *node, __isl_take isl_multi_val *mv);
__isl_give isl_schedule_node *isl_schedule_node_band_scale_down(
	__isl_take isl_schedule_node *node, __isl_take isl_multi_val *mv);
__isl_give isl_schedule_node *isl_schedule_node_band_tile(
	__isl_take isl_schedule_node *node, __isl_take isl_multi_val *sizes);
__isl_give isl_schedule_node *isl_schedule_node_band_sink(
	__isl_take isl_schedule_node *node);
__isl_give isl_schedule_node *isl_schedule_node_band_split(
	__isl_take isl_schedule_node *node, int pos);

__isl_give isl_union_set *isl_schedule_node_domain_get_domain(
	__isl_keep isl_schedule_node *node);
__isl_give isl_union_set *isl_schedule_node_filter_get_filter(
	__isl_keep isl_schedule_node *node);

int isl_schedule_node_get_schedule_depth(__isl_keep isl_schedule_node *node);
__isl_give isl_union_set *isl_schedule_node_get_universe_domain(
	__isl_keep isl_schedule_node *node);
__isl_give isl_union_pw_multi_aff *
isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(
	__isl_keep isl_schedule_node *node);
__isl_give isl_union_map *isl_schedule_node_get_prefix_schedule_union_map(
	__isl_keep isl_schedule_node *node);
__isl_give isl_union_map *isl_schedule_node_get_subtree_schedule_union_map(
	__isl_keep isl_schedule_node *node);

__isl_give isl_schedule_node *isl_schedule_node_insert_partial_schedule(
	__isl_take isl_schedule_node *node,
	__isl_take isl_multi_union_pw_aff *schedule);
__isl_give isl_schedule_node *isl_schedule_node_insert_filter(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *filter);
__isl_give isl_schedule_node *isl_schedule_node_insert_sequence(
	__isl_take isl_schedule_node *node,
	__isl_take isl_union_set_list *filters);
__isl_give isl_schedule_node *isl_schedule_node_insert_set(
	__isl_take isl_schedule_node *node,
	__isl_take isl_union_set_list *filters);

__isl_give isl_schedule_node *isl_schedule_node_cut(
	__isl_take isl_schedule_node *node);
__isl_give isl_schedule_node *isl_schedule_node_delete(
	__isl_take isl_schedule_node *node);

__isl_give isl_schedule_node *isl_schedule_node_reset_user(
	__isl_take isl_schedule_node *node);
__isl_give isl_schedule_node *isl_schedule_node_align_params(
	__isl_take isl_schedule_node *node, __isl_take isl_space *space);

__isl_give isl_printer *isl_printer_print_schedule_node(
	__isl_take isl_printer *p, __isl_keep isl_schedule_node *node);
void isl_schedule_node_dump(__isl_keep isl_schedule_node *node);

#if defined(__cplusplus)
}
#endif

#endif
