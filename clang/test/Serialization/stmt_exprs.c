// RUN: clang %s --test-pickling 2>&1 | grep -q 'SUCCESS'
#include "../Sema/stmt_exprs.c"