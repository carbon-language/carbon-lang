#include <isl/dim.h>
#include <isl/aff.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/polynomial.h>

isl_ctx *isl_dim_get_ctx(__isl_keep isl_space *dim)
{
	return isl_space_get_ctx(dim);
}

__isl_give isl_space *isl_dim_alloc(isl_ctx *ctx,
	unsigned nparam, unsigned n_in, unsigned n_out)
{
	return isl_space_alloc(ctx, nparam, n_in, n_out);
}
__isl_give isl_space *isl_dim_set_alloc(isl_ctx *ctx,
	unsigned nparam, unsigned dim)
{
	return isl_space_set_alloc(ctx, nparam, dim);
}
__isl_give isl_space *isl_dim_copy(__isl_keep isl_space *dim)
{
	return isl_space_copy(dim);
}
void isl_dim_free(__isl_take isl_space *dim)
{
	isl_space_free(dim);
}

unsigned isl_dim_size(__isl_keep isl_space *dim, enum isl_dim_type type)
{
	return isl_space_dim(dim, type);
}

__isl_give isl_space *isl_dim_set_dim_id(__isl_take isl_space *dim,
	enum isl_dim_type type, unsigned pos, __isl_take isl_id *id)
{
	return isl_space_set_dim_id(dim, type, pos, id);
}
int isl_dim_has_dim_id(__isl_keep isl_space *dim,
	enum isl_dim_type type, unsigned pos)
{
	return isl_space_has_dim_id(dim, type, pos);
}
__isl_give isl_id *isl_dim_get_dim_id(__isl_keep isl_space *dim,
	enum isl_dim_type type, unsigned pos)
{
	return isl_space_get_dim_id(dim, type, pos);
}

int isl_dim_find_dim_by_id(__isl_keep isl_space *dim,
	enum isl_dim_type type, __isl_keep isl_id *id)
{
	return isl_space_find_dim_by_id(dim, type, id);
}

__isl_give isl_space *isl_dim_set_tuple_id(__isl_take isl_space *dim,
	enum isl_dim_type type, __isl_take isl_id *id)
{
	return isl_space_set_tuple_id(dim, type, id);
}
__isl_give isl_space *isl_dim_reset_tuple_id(__isl_take isl_space *dim,
	enum isl_dim_type type)
{
	return isl_space_reset_tuple_id(dim, type);
}
int isl_dim_has_tuple_id(__isl_keep isl_space *dim, enum isl_dim_type type)
{
	return isl_space_has_tuple_id(dim, type);
}
__isl_give isl_id *isl_dim_get_tuple_id(__isl_keep isl_space *dim,
	enum isl_dim_type type)
{
	return isl_space_get_tuple_id(dim, type);
}

__isl_give isl_space *isl_dim_set_name(__isl_take isl_space *dim,
	enum isl_dim_type type, unsigned pos, __isl_keep const char *name)
{
	return isl_space_set_dim_name(dim, type, pos, name);
}
__isl_keep const char *isl_dim_get_name(__isl_keep isl_space *dim,
	enum isl_dim_type type, unsigned pos)
{
	return isl_space_get_dim_name(dim, type, pos);
}

__isl_give isl_space *isl_dim_set_tuple_name(__isl_take isl_space *dim,
	enum isl_dim_type type, const char *s)
{
	return isl_space_set_tuple_name(dim, type, s);
}
const char *isl_dim_get_tuple_name(__isl_keep isl_space *dim,
	enum isl_dim_type type)
{
	return isl_space_get_tuple_name(dim, type);
}

int isl_dim_is_wrapping(__isl_keep isl_space *dim)
{
	return isl_space_is_wrapping(dim);
}
__isl_give isl_space *isl_dim_wrap(__isl_take isl_space *dim)
{
	return isl_space_wrap(dim);
}
__isl_give isl_space *isl_dim_unwrap(__isl_take isl_space *dim)
{
	return isl_space_unwrap(dim);
}

__isl_give isl_space *isl_dim_domain(__isl_take isl_space *dim)
{
	return isl_space_domain(dim);
}
__isl_give isl_space *isl_dim_from_domain(__isl_take isl_space *dim)
{
	return isl_space_from_domain(dim);
}
__isl_give isl_space *isl_dim_range(__isl_take isl_space *dim)
{
	return isl_space_range(dim);
}
__isl_give isl_space *isl_dim_from_range(__isl_take isl_space *dim)
{
	return isl_space_from_range(dim);
}
__isl_give isl_space *isl_dim_reverse(__isl_take isl_space *dim)
{
	return isl_space_reverse(dim);
}
__isl_give isl_space *isl_dim_join(__isl_take isl_space *left,
	__isl_take isl_space *right)
{
	return isl_space_join(left, right);
}
__isl_give isl_space *isl_dim_align_params(__isl_take isl_space *dim1,
	__isl_take isl_space *dim2)
{
	return isl_space_align_params(dim1, dim2);
}
__isl_give isl_space *isl_dim_insert(__isl_take isl_space *dim,
	enum isl_dim_type type, unsigned pos, unsigned n)
{
	return isl_space_insert_dims(dim, type, pos, n);
}
__isl_give isl_space *isl_dim_add(__isl_take isl_space *dim,
	enum isl_dim_type type, unsigned n)
{
	return isl_space_add_dims(dim, type, n);
}
__isl_give isl_space *isl_dim_drop(__isl_take isl_space *dim,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return isl_space_drop_dims(dim, type, first, n);
}
__isl_give isl_space *isl_dim_move(__isl_take isl_space *dim,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n)
{
	return isl_space_move_dims(dim, dst_type, dst_pos, src_type, src_pos, n);
}
__isl_give isl_space *isl_dim_map_from_set(__isl_take isl_space *dim)
{
	return isl_space_map_from_set(dim);
}
__isl_give isl_space *isl_dim_zip(__isl_take isl_space *dim)
{
	return isl_space_zip(dim);
}

__isl_give isl_local_space *isl_local_space_from_dim(
	__isl_take isl_space *dim)
{
	return isl_local_space_from_space(dim);
}
__isl_give isl_space *isl_local_space_get_dim(
	__isl_keep isl_local_space *ls)
{
	return isl_local_space_get_space(ls);
}

__isl_give isl_space *isl_aff_get_dim(__isl_keep isl_aff *aff)
{
	return isl_aff_get_space(aff);
}
__isl_give isl_space *isl_pw_aff_get_dim(__isl_keep isl_pw_aff *pwaff)
{
	return isl_pw_aff_get_space(pwaff);
}

__isl_give isl_space *isl_constraint_get_dim(
	__isl_keep isl_constraint *constraint)
{
	return isl_constraint_get_space(constraint);
}

__isl_give isl_space *isl_basic_map_get_dim(__isl_keep isl_basic_map *bmap)
{
	return isl_basic_map_get_space(bmap);
}
__isl_give isl_space *isl_map_get_dim(__isl_keep isl_map *map)
{
	return isl_map_get_space(map);
}
__isl_give isl_space *isl_union_map_get_dim(__isl_keep isl_union_map *umap)
{
	return isl_union_map_get_space(umap);
}

__isl_give isl_space *isl_basic_set_get_dim(__isl_keep isl_basic_set *bset)
{
	return isl_basic_set_get_space(bset);
}
__isl_give isl_space *isl_set_get_dim(__isl_keep isl_set *set)
{
	return isl_set_get_space(set);
}
__isl_give isl_space *isl_union_set_get_dim(__isl_keep isl_union_set *uset)
{
	return isl_union_set_get_space(uset);
}

__isl_give isl_space *isl_point_get_dim(__isl_keep isl_point *pnt)
{
	return isl_point_get_space(pnt);
}

__isl_give isl_space *isl_qpolynomial_get_dim(__isl_keep isl_qpolynomial *qp)
{
	return isl_qpolynomial_get_space(qp);
}
__isl_give isl_space *isl_pw_qpolynomial_get_dim(
	__isl_keep isl_pw_qpolynomial *pwqp)
{
	return isl_pw_qpolynomial_get_space(pwqp);
}
__isl_give isl_space *isl_qpolynomial_fold_get_dim(
	__isl_keep isl_qpolynomial_fold *fold)
{
	return isl_qpolynomial_fold_get_space(fold);
}
__isl_give isl_space *isl_pw_qpolynomial_fold_get_dim(
	__isl_keep isl_pw_qpolynomial_fold *pwf)
{
	return isl_pw_qpolynomial_fold_get_space(pwf);
}
__isl_give isl_space *isl_union_pw_qpolynomial_get_dim(
	__isl_keep isl_union_pw_qpolynomial *upwqp)
{
	return isl_union_pw_qpolynomial_get_space(upwqp);
}
__isl_give isl_space *isl_union_pw_qpolynomial_fold_get_dim(
	__isl_keep isl_union_pw_qpolynomial_fold *upwf)
{
	return isl_union_pw_qpolynomial_fold_get_space(upwf);
}
