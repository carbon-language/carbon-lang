/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#include <isl/set.h>

int main(int argc, char **argv)
{
	struct isl_ctx *ctx = isl_ctx_alloc();
	struct isl_basic_set *bset;

	bset = isl_basic_set_read_from_file(ctx, stdin);
	bset = isl_basic_set_detect_equalities(bset);
	isl_basic_set_print(bset, stdout, 0, "", "", ISL_FORMAT_POLYLIB);
	isl_basic_set_free(bset);
	isl_ctx_free(ctx);

	return 0;
}
