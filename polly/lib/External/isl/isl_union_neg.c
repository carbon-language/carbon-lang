/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 */

#include <isl_union_macro.h>

/* Replace *entry by its opposite.
 *
 * Return isl_stat_ok on success and isl_stat_error on error.
 */
static isl_stat FN(UNION,neg_entry)(void **entry, void *user)
{
	PW **pw = (PW **) entry;

	*pw = FN(PW,neg)(*pw);

	return *pw ? isl_stat_ok : isl_stat_error;
}

/* Return the opposite of "u".
 */
__isl_give UNION *FN(UNION,neg)(__isl_take UNION *u)
{
	u = FN(UNION,cow)(u);
	if (!u)
		return NULL;

	if (isl_hash_table_foreach(u->space->ctx, &u->table,
				   &FN(UNION,neg_entry), NULL) < 0)
		return FN(UNION,free)(u);

	return u;
}
