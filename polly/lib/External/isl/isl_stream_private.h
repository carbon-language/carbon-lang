#include <isl_int.h>
#include <isl/stream.h>

struct isl_token {
	int type;

	unsigned int on_new_line : 1;
	unsigned is_keyword : 1;
	int line;
	int col;

	union {
		isl_int	v;
		char	*s;
		isl_map *map;
		isl_pw_aff *pwaff;
	} u;
};

struct isl_token *isl_token_new(isl_ctx *ctx,
	int line, int col, unsigned on_new_line);
