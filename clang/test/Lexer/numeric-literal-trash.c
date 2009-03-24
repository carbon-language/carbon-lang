/* RUN: clang-cc -fsyntax-only -verify %s
 */
# define XRECORD(x, c_name) e##c (x, __LINE__)






 void x() {

XRECORD (XRECORD (1, 1), 1);
    }
