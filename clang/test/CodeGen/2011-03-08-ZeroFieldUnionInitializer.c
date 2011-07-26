// RUN: %clang_cc1 -emit-llvm %s -o 0
typedef struct {
  union {
    struct { } __attribute((packed));
  };
} fenv_t;
const fenv_t _FE_DFL_ENV = {{{ 0, 0, 0, 0 }}};
