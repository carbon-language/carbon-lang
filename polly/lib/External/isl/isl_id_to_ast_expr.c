#include <isl/id_to_ast_expr.h>
#include <isl/ast.h>

#define isl_id_is_equal(id1,id2)	id1 == id2

#define KEY_BASE	id
#define KEY_EQUAL	isl_id_is_equal
#define VAL_BASE	ast_expr
#define VAL_EQUAL	isl_ast_expr_is_equal

#include <isl_hmap_templ.c>
