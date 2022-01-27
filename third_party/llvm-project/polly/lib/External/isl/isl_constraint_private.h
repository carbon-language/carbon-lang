#ifndef ISL_CONSTRAINT_PRIVATE_H
#define ISL_CONSTRAINT_PRIVATE_H

#include <isl/constraint.h>
#include <isl/local_space.h>
#include <isl/vec.h>

struct isl_constraint {
	int ref;

	int eq;
	isl_local_space	*ls;
	isl_vec		*v;
};

#undef EL
#define EL isl_constraint

#include <isl_list_templ.h>

__isl_give isl_constraint *isl_basic_set_constraint(
	__isl_take isl_basic_set *bset, isl_int **line);

void isl_constraint_get_constant(__isl_keep isl_constraint *constraint,
	isl_int *v);
void isl_constraint_get_coefficient(__isl_keep isl_constraint *constraint,
	enum isl_dim_type type, int pos, isl_int *v);

isl_bool isl_constraint_is_div_equality(__isl_keep isl_constraint *constraint,
	unsigned div);

#endif
