// RUN: clang -o %t --emit-llvm -g %s

// PR3023
void convert(void) {
  struct { typeof(0) f0; } v0;
}

// PR2784
struct OPAQUE;
typedef struct OPAQUE *PTR;
PTR p;


// PR2950
struct s0;
struct s0 { struct s0 *p; } g0;

struct s0 *f0(struct s0 *a0) {
  return a0->p;
}
  
  
