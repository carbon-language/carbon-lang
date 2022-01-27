#ifndef ISL_SET_TO_GRAFT_LIST_H
#define ISL_SET_TO_GRAFT_LIST_H

#include <isl/set_type.h>
#include "isl_ast_graft_private.h"
#include "isl_maybe_ast_graft_list.h"

#define ISL_KEY			isl_set
#define ISL_VAL			isl_ast_graft_list
#define ISL_HMAP_SUFFIX		set_to_ast_graft_list
#define ISL_HMAP		isl_set_to_ast_graft_list
#include <isl/hmap.h>
#undef ISL_KEY
#undef ISL_VAL
#undef ISL_HMAP_SUFFIX
#undef ISL_HMAP

#endif
