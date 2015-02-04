#ifndef ISL_DIM_H
#define ISL_DIM_H

#include <isl/space.h>
#include <isl/local_space.h>
#include <isl/aff_type.h>
#include <isl/constraint.h>
#include <isl/map_type.h>
#include <isl/set_type.h>
#include <isl/point.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/polynomial_type.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define isl_dim isl_space

ISL_DEPRECATED
isl_ctx *isl_dim_get_ctx(__isl_keep isl_space *dim);
ISL_DEPRECATED
__isl_give isl_space *isl_dim_alloc(isl_ctx *ctx,
			unsigned nparam, unsigned n_in, unsigned n_out);
ISL_DEPRECATED
__isl_give isl_space *isl_dim_set_alloc(isl_ctx *ctx,
			unsigned nparam, unsigned dim);
ISL_DEPRECATED
__isl_give isl_space *isl_dim_copy(__isl_keep isl_space *dim);
ISL_DEPRECATED
void isl_dim_free(__isl_take isl_space *dim);

ISL_DEPRECATED
unsigned isl_dim_size(__isl_keep isl_space *dim, enum isl_dim_type type);

ISL_DEPRECATED
__isl_give isl_space *isl_dim_set_dim_id(__isl_take isl_space *dim,
	enum isl_dim_type type, unsigned pos, __isl_take isl_id *id);
ISL_DEPRECATED
int isl_dim_has_dim_id(__isl_keep isl_space *dim,
	enum isl_dim_type type, unsigned pos);
ISL_DEPRECATED
__isl_give isl_id *isl_dim_get_dim_id(__isl_keep isl_space *dim,
	enum isl_dim_type type, unsigned pos);

ISL_DEPRECATED
int isl_dim_find_dim_by_id(__isl_keep isl_space *dim,
	enum isl_dim_type type, __isl_keep isl_id *id);

ISL_DEPRECATED
__isl_give isl_space *isl_dim_set_tuple_id(__isl_take isl_space *dim,
	enum isl_dim_type type, __isl_take isl_id *id);
ISL_DEPRECATED
__isl_give isl_space *isl_dim_reset_tuple_id(__isl_take isl_space *dim,
	enum isl_dim_type type);
ISL_DEPRECATED
int isl_dim_has_tuple_id(__isl_keep isl_space *dim, enum isl_dim_type type);
ISL_DEPRECATED
__isl_give isl_id *isl_dim_get_tuple_id(__isl_keep isl_space *dim,
	enum isl_dim_type type);

ISL_DEPRECATED
__isl_give isl_space *isl_dim_set_name(__isl_take isl_space *dim,
	enum isl_dim_type type, unsigned pos, __isl_keep const char *name);
ISL_DEPRECATED
__isl_keep const char *isl_dim_get_name(__isl_keep isl_space *dim,
	enum isl_dim_type type, unsigned pos);

ISL_DEPRECATED
__isl_give isl_space *isl_dim_set_tuple_name(__isl_take isl_space *dim,
	enum isl_dim_type type, const char *s);
ISL_DEPRECATED
const char *isl_dim_get_tuple_name(__isl_keep isl_space *dim,
				 enum isl_dim_type type);

ISL_DEPRECATED
int isl_dim_is_wrapping(__isl_keep isl_space *dim);
ISL_DEPRECATED
__isl_give isl_space *isl_dim_wrap(__isl_take isl_space *dim);
ISL_DEPRECATED
__isl_give isl_space *isl_dim_unwrap(__isl_take isl_space *dim);

ISL_DEPRECATED
__isl_give isl_space *isl_dim_domain(__isl_take isl_space *dim);
ISL_DEPRECATED
__isl_give isl_space *isl_dim_from_domain(__isl_take isl_space *dim);
ISL_DEPRECATED
__isl_give isl_space *isl_dim_range(__isl_take isl_space *dim);
ISL_DEPRECATED
__isl_give isl_space *isl_dim_from_range(__isl_take isl_space *dim);
ISL_DEPRECATED
__isl_give isl_space *isl_dim_reverse(__isl_take isl_space *dim);
ISL_DEPRECATED
__isl_give isl_space *isl_dim_join(__isl_take isl_space *left,
	__isl_take isl_space *right);
ISL_DEPRECATED
__isl_give isl_space *isl_dim_align_params(__isl_take isl_space *dim1,
	__isl_take isl_space *dim2);
ISL_DEPRECATED
__isl_give isl_space *isl_dim_insert(__isl_take isl_space *dim,
	enum isl_dim_type type, unsigned pos, unsigned n);
ISL_DEPRECATED
__isl_give isl_space *isl_dim_add(__isl_take isl_space *dim,
	enum isl_dim_type type, unsigned n);
ISL_DEPRECATED
__isl_give isl_space *isl_dim_drop(__isl_take isl_space *dim,
	enum isl_dim_type type, unsigned first, unsigned n);
ISL_DEPRECATED
__isl_give isl_space *isl_dim_move(__isl_take isl_space *dim,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n);
ISL_DEPRECATED
__isl_give isl_space *isl_dim_map_from_set(
	__isl_take isl_space *dim);
ISL_DEPRECATED
__isl_give isl_space *isl_dim_zip(__isl_take isl_space *dim);

ISL_DEPRECATED
__isl_give isl_local_space *isl_local_space_from_dim(
	__isl_take isl_space *dim);
ISL_DEPRECATED
__isl_give isl_space *isl_local_space_get_dim(
	__isl_keep isl_local_space *ls);

ISL_DEPRECATED
__isl_give isl_space *isl_aff_get_dim(__isl_keep isl_aff *aff);
ISL_DEPRECATED
__isl_give isl_space *isl_pw_aff_get_dim(__isl_keep isl_pw_aff *pwaff);

ISL_DEPRECATED
__isl_give isl_space *isl_constraint_get_dim(
	__isl_keep isl_constraint *constraint);

ISL_DEPRECATED
__isl_give isl_space *isl_basic_map_get_dim(__isl_keep isl_basic_map *bmap);
ISL_DEPRECATED
__isl_give isl_space *isl_map_get_dim(__isl_keep isl_map *map);
ISL_DEPRECATED
__isl_give isl_space *isl_union_map_get_dim(__isl_keep isl_union_map *umap);

ISL_DEPRECATED
__isl_give isl_space *isl_basic_set_get_dim(__isl_keep isl_basic_set *bset);
ISL_DEPRECATED
__isl_give isl_space *isl_set_get_dim(__isl_keep isl_set *set);
ISL_DEPRECATED
__isl_give isl_space *isl_union_set_get_dim(__isl_keep isl_union_set *uset);

ISL_DEPRECATED
__isl_give isl_space *isl_point_get_dim(__isl_keep isl_point *pnt);

ISL_DEPRECATED
__isl_give isl_space *isl_qpolynomial_get_dim(__isl_keep isl_qpolynomial *qp);
ISL_DEPRECATED
__isl_give isl_space *isl_pw_qpolynomial_get_dim(
	__isl_keep isl_pw_qpolynomial *pwqp);
ISL_DEPRECATED
__isl_give isl_space *isl_qpolynomial_fold_get_dim(
	__isl_keep isl_qpolynomial_fold *fold);
ISL_DEPRECATED
__isl_give isl_space *isl_pw_qpolynomial_fold_get_dim(
	__isl_keep isl_pw_qpolynomial_fold *pwf);
ISL_DEPRECATED
__isl_give isl_space *isl_union_pw_qpolynomial_get_dim(
	__isl_keep isl_union_pw_qpolynomial *upwqp);
ISL_DEPRECATED
__isl_give isl_space *isl_union_pw_qpolynomial_fold_get_dim(
	__isl_keep isl_union_pw_qpolynomial_fold *upwf);

#if defined(__cplusplus)
}
#endif

#endif
