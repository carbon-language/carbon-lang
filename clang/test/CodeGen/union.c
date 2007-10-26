// RUN: clang %s -emit-llvm

union {
  int a;
  float b;
} u;

void f() {
  u.b = 11;
}

int f2( float __x ) { 
  union{ 
    float __f; 
    unsigned int __u; 
  }__u;
  return (int)(__u.__u >> 31); 
}
