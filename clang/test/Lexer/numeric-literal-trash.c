/* RUN: %clang_cc1 -fsyntax-only -verify %s
 */
# define XRECORD(x, c_name) e##c (x, __LINE__)



int ec(int, int);


 void x() {

XRECORD (XRECORD (1, 1), 1);
    }
