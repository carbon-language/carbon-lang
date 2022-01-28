#ifndef ISL_SPACE_PRIVATE
#define ISL_SPACE_PRIVATE

#include <isl/space.h>
#include <isl/hash.h>
#include <isl/id_type.h>

struct isl_name;
struct isl_space {
	int ref;

	struct isl_ctx *ctx;

	unsigned nparam;
	unsigned n_in;		/* zero for sets */
	unsigned n_out;		/* dim for sets */

	isl_id *tuple_id[2];
	isl_space *nested[2];

	unsigned n_id;
	isl_id **ids;
};

__isl_give isl_space *isl_space_cow(__isl_take isl_space *space);

__isl_give isl_space *isl_space_underlying(__isl_take isl_space *space,
	unsigned n_div);

uint32_t isl_space_get_tuple_hash(__isl_keep isl_space *space);
uint32_t isl_space_get_tuple_domain_hash(__isl_keep isl_space *space);
uint32_t isl_space_get_full_hash(__isl_keep isl_space *space);

isl_bool isl_space_has_domain_tuples(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2);
isl_bool isl_space_has_range_tuples(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2);
isl_stat isl_space_check_domain_tuples(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2);
isl_bool isl_space_is_domain_internal(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2);
isl_bool isl_space_is_range_internal(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2);
isl_stat isl_space_check_domain_wrapped_domain_tuples(
	__isl_keep isl_space *space1, __isl_keep isl_space *space2);
isl_bool isl_space_wrapped_tuple_is_equal(__isl_keep isl_space *space1,
	enum isl_dim_type outer, enum isl_dim_type inner,
	__isl_keep isl_space *space2, enum isl_dim_type type2);
isl_stat isl_space_check_wrapped_tuple_is_equal(__isl_keep isl_space *space1,
	enum isl_dim_type outer, enum isl_dim_type inner,
	__isl_keep isl_space *space2, enum isl_dim_type type2);

isl_size isl_space_wrapped_dim(__isl_keep isl_space *space,
	enum isl_dim_type outer, enum isl_dim_type inner);
unsigned isl_space_offset(__isl_keep isl_space *space, enum isl_dim_type type);

isl_stat isl_space_check_range(__isl_keep isl_space *space,
	enum isl_dim_type type, unsigned first, unsigned n);
isl_stat isl_space_check_is_set(__isl_keep isl_space *space);
isl_bool isl_space_may_be_set(__isl_keep isl_space *space);
isl_bool isl_space_is_named_or_nested(__isl_keep isl_space *space,
	enum isl_dim_type type);
isl_bool isl_space_has_equal_ids(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2);
isl_bool isl_space_has_named_params(__isl_keep isl_space *space);
isl_stat isl_space_check_named_params(__isl_keep isl_space *space);
isl_stat isl_space_check_equal_params(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2);
isl_stat isl_space_check_equal_tuples(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2);
__isl_give isl_space *isl_space_reset(__isl_take isl_space *space,
	enum isl_dim_type type);
__isl_give isl_space *isl_space_flatten(__isl_take isl_space *space);

isl_stat isl_space_check_domain_is_wrapping(__isl_keep isl_space *space);
isl_stat isl_space_check_range_is_wrapping(__isl_keep isl_space *space);

__isl_give isl_space *isl_space_replace_params(__isl_take isl_space *dst,
	__isl_keep isl_space *src);
__isl_give isl_space *isl_space_copy_ids_if_unset(__isl_take isl_space *dst,
	enum isl_dim_type dst_type, __isl_keep isl_space *src,
	enum isl_dim_type src_type);

__isl_give isl_space *isl_space_lift(__isl_take isl_space *space,
	unsigned n_local);

__isl_give isl_space *isl_space_extend_domain_with_range(
	__isl_take isl_space *domain, __isl_take isl_space *model);
__isl_give isl_space *isl_space_bind_set(__isl_take isl_space *space,
	__isl_keep isl_multi_id *tuple);
__isl_give isl_space *isl_space_bind_map_domain(__isl_take isl_space *space,
	__isl_keep isl_multi_id *tuple);
__isl_give isl_space *isl_space_bind_domain_wrapped_domain(
	__isl_take isl_space *space, __isl_keep isl_multi_id *tuple);
__isl_give isl_space *isl_space_unbind_params_insert_domain(
	__isl_take isl_space *space, __isl_keep isl_multi_id *tuple);

int isl_space_cmp(__isl_keep isl_space *space1, __isl_keep isl_space *space2);

#endif
