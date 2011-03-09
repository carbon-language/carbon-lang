// RUN: %llvmgcc -S %s
typedef struct {
  union {
    struct { } __attribute((packed));
  };
} fenv_t;
const fenv_t _FE_DFL_ENV = {{{ 0, 0, 0, 0 }}};
