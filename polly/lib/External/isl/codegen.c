/*
 * Copyright 2012      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

/* This program prints an AST that scans the domain elements of
 * the domain of a given schedule in the order of their image(s).
 *
 * The input consists of three sets/relations.
 * - a schedule
 * - a context
 * - a relation describing AST generation options
 */

#include <assert.h>
#include <isl/ast.h>
#include <isl/ast_build.h>
#include <isl/options.h>
#include <isl/set.h>

struct options {
	struct isl_options	*isl;
	unsigned		 atomic;
	unsigned		 separate;
};

ISL_ARGS_START(struct options, options_args)
ISL_ARG_CHILD(struct options, isl, "isl", &isl_options_args, "isl options")
ISL_ARG_BOOL(struct options, atomic, 0, "atomic", 0,
	"globally set the atomic option")
ISL_ARG_BOOL(struct options, separate, 0, "separate", 0,
	"globally set the separate option")
ISL_ARGS_END

ISL_ARG_DEF(options, struct options, options_args)

/* Return a universal, 1-dimensional set with the given name.
 */
static __isl_give isl_union_set *universe(isl_ctx *ctx, const char *name)
{
	isl_space *space;

	space = isl_space_set_alloc(ctx, 0, 1);
	space = isl_space_set_tuple_name(space, isl_dim_set, name);
	return isl_union_set_from_set(isl_set_universe(space));
}

/* Set the "name" option for the entire schedule domain.
 */
static __isl_give isl_union_map *set_universe(__isl_take isl_union_map *opt,
	__isl_keep isl_union_map *schedule, const char *name)
{
	isl_ctx *ctx;
	isl_union_set *domain, *target;
	isl_union_map *option;

	ctx = isl_union_map_get_ctx(opt);

	domain = isl_union_map_range(isl_union_map_copy(schedule));
	domain = isl_union_set_universe(domain);
	target = universe(ctx, name);
	option = isl_union_map_from_domain_and_range(domain, target);
	opt = isl_union_map_union(opt, option);

	return opt;
}

/* Update the build options based on the user-specified options.
 *
 * If the --separate or --atomic options were specified, then
 * we clear any separate or atomic options that may already exist in "opt".
 */
static __isl_give isl_ast_build *set_options(__isl_take isl_ast_build *build,
	__isl_take isl_union_map *opt, struct options *options,
	__isl_keep isl_union_map *schedule)
{
	if (options->separate || options->atomic) {
		isl_ctx *ctx;
		isl_union_set *target;

		ctx = isl_union_map_get_ctx(schedule);

		target = universe(ctx, "separate");
		opt = isl_union_map_subtract_range(opt, target);
		target = universe(ctx, "atomic");
		opt = isl_union_map_subtract_range(opt, target);
	}

	if (options->separate)
		opt = set_universe(opt, schedule, "separate");
	if (options->atomic)
		opt = set_universe(opt, schedule, "atomic");

	build = isl_ast_build_set_options(build, opt);

	return build;
}

int main(int argc, char **argv)
{
	isl_ctx *ctx;
	isl_set *context;
	isl_union_map *schedule;
	isl_union_map *options_map;
	isl_ast_build *build;
	isl_ast_node *tree;
	struct options *options;
	isl_printer *p;

	options = options_new_with_defaults();
	assert(options);
	argc = options_parse(options, argc, argv, ISL_ARG_ALL);

	ctx = isl_ctx_alloc_with_options(&options_args, options);

	schedule = isl_union_map_read_from_file(ctx, stdin);
	context = isl_set_read_from_file(ctx, stdin);
	options_map = isl_union_map_read_from_file(ctx, stdin);

	build = isl_ast_build_from_context(context);
	build = set_options(build, options_map, options, schedule);
	tree = isl_ast_build_ast_from_schedule(build, schedule);
	isl_ast_build_free(build);

	p = isl_printer_to_file(ctx, stdout);
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = isl_printer_print_ast_node(p, tree);
	isl_printer_free(p);

	isl_ast_node_free(tree);

	isl_ctx_free(ctx);
	return 0;
}
