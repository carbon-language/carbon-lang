// RUN: clang-cc %s -ast-print

typedef void func_typedef();
func_typedef xxx;

typedef void func_t(int x);
func_t a;

