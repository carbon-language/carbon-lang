#ifndef ISL_VAL_PRIVATE_H
#define ISL_VAL_PRIVATE_H

#include <isl_int.h>
#include <isl/val.h>
#include <isl/local_space.h>
#include <isl_reordering.h>

/* Represents a "value", which may be an integer value, a rational value,
 * plus or minus infinity or "not a number".
 *
 * Internally, +infinity is represented as 1/0,
 * -infinity as -1/0 and NaN as 0/0.
 *
 * A rational value is always normalized before it is passed to the user.
 */
struct isl_val {
	int ref;
	isl_ctx *ctx;

	isl_int n;
	isl_int d;
};

#undef EL
#define EL isl_val

#include <isl_list_templ.h>

__isl_give isl_val *isl_val_alloc(isl_ctx *ctx);
__isl_give isl_val *isl_val_normalize(__isl_take isl_val *v);
__isl_give isl_val *isl_val_int_from_isl_int(isl_ctx *ctx, isl_int n);
__isl_give isl_val *isl_val_rat_from_isl_int(isl_ctx *ctx,
	isl_int n, isl_int d);
__isl_give isl_val *isl_val_cow(__isl_take isl_val *val);

isl_stat isl_val_get_num_isl_int(__isl_keep isl_val *v, isl_int *n);

isl_bool isl_val_involves_dims(__isl_keep isl_val *v, enum isl_dim_type type,
	unsigned first, unsigned n);
__isl_give isl_val *isl_val_insert_dims(__isl_take isl_val *v,
	enum isl_dim_type type, unsigned first, unsigned n);
__isl_give isl_val *isl_val_set_dim_name(__isl_take isl_val *v,
	enum isl_dim_type type, unsigned pos, const char *s);

__isl_give isl_val *isl_val_scale_val(__isl_take isl_val *v1,
	__isl_take isl_val *v2);
__isl_give isl_val *isl_val_scale_down_val(__isl_take isl_val *v1,
	__isl_take isl_val *v2);
__isl_give isl_val *isl_val_mod_val(__isl_take isl_val *v1,
	__isl_take isl_val *v2);

isl_bool isl_val_plain_is_equal(__isl_keep isl_val *val1,
	__isl_keep isl_val *val2);

#undef BASE
#define BASE val

#include <isl_multi_templ.h>

#endif
