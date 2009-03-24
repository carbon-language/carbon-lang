// RUN: clang-cc < %s -emit-llvm

int* a = &(int){1};
struct s {int a, b, c;} * b = &(struct s) {1, 2, 3};
// Not working; complex constants are broken
// _Complex double * x = &(_Complex double){1.0f};

int xxx() {
int* a = &(int){1};
struct s {int a, b, c;} * b = &(struct s) {1, 2, 3};
_Complex double * x = &(_Complex double){1.0f};
}
