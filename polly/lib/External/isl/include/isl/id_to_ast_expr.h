#ifndef ISL_ID_TO_AST_EXPR_H
#define ISL_ID_TO_AST_EXPR_H

#include <isl/id.h>
#include <isl/ast_type.h>

#define ISL_KEY_BASE	id
#define ISL_VAL_BASE	ast_expr
#include <isl/hmap.h>
#undef ISL_KEY_BASE
#undef ISL_VAL_BASE

#endif
