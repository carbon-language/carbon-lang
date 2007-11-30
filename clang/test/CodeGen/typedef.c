// RUN: clang -emit-llvm %s

typedef struct { int i; } Value;
typedef Value *PValue;

int get_value(PValue v) {
  return v->i;
}
