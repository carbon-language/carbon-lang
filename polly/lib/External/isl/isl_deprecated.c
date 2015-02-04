#include <isl/set.h>

/* This function was never documented and has been replaced by
 * isl_basic_set_add_dims.
 */
__isl_give isl_basic_set *isl_basic_set_add(__isl_take isl_basic_set *bset,
	enum isl_dim_type type, unsigned n)
{
	return isl_basic_set_add_dims(bset, type, n);
}
