#include <assert.h>
#include <isl/map.h>
#include <isl/options.h>

int main(int argc, char **argv)
{
	struct isl_ctx *ctx;
	struct isl_map *map;
	struct isl_options *options;
	int exact;

	options = isl_options_new_with_defaults();
	assert(options);
	argc = isl_options_parse(options, argc, argv, ISL_ARG_ALL);

	ctx = isl_ctx_alloc_with_options(&isl_options_args, options);

	map = isl_map_read_from_file(ctx, stdin);
	map = isl_map_transitive_closure(map, &exact);
	if (!exact)
		printf("# NOT exact\n");
	isl_map_print(map, stdout, 0, ISL_FORMAT_ISL);
	printf("\n");
	map = isl_map_compute_divs(map);
	map = isl_map_coalesce(map);
	printf("# coalesced\n");
	isl_map_print(map, stdout, 0, ISL_FORMAT_ISL);
	printf("\n");
	isl_map_free(map);

	isl_ctx_free(ctx);

	return 0;
}
