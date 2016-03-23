// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// Make sure we handle target overloads correctly.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:    -fsyntax-only -fcuda-target-overloads -verify %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda \
// RUN:    -fsyntax-only -fcuda-target-overloads -fcuda-is-device -verify %s

// Check target overloads handling with disabled call target checks.
// RUN: %clang_cc1 -DNOCHECKS -triple x86_64-unknown-linux-gnu -fsyntax-only \
// RUN:    -fcuda-disable-target-call-checks -fcuda-target-overloads -verify %s
// RUN: %clang_cc1 -DNOCHECKS -triple nvptx64-nvidia-cuda -fsyntax-only \
// RUN:    -fcuda-disable-target-call-checks -fcuda-target-overloads \
// RUN:    -fcuda-is-device -verify %s

#include "Inputs/cuda.h"

typedef int (*fp_t)();
typedef void (*gp_t)();

// Host and unattributed functions can't be overloaded.
__host__ void hh() {} // expected-note {{previous definition is here}}
void hh() {} // expected-error {{redefinition of 'hh'}}

// H/D overloading is OK.
__host__ int dh() { return 2; }
__device__ int dh() { return 2; }

// H/HD and D/HD are not allowed.
__host__ __device__ int hdh() { return 5; } // expected-note {{previous definition is here}}
__host__ int hdh() { return 4; } // expected-error {{redefinition of 'hdh'}}

__host__ int hhd() { return 4; } // expected-note {{previous definition is here}}
__host__ __device__ int hhd() { return 5; } // expected-error {{redefinition of 'hhd'}}
// expected-warning@-1 {{attribute declaration must precede definition}}
// expected-note@-3 {{previous definition is here}}

__host__ __device__ int hdd() { return 7; } // expected-note {{previous definition is here}}
__device__ int hdd() { return 6; } // expected-error {{redefinition of 'hdd'}}

__device__ int dhd() { return 6; } // expected-note {{previous definition is here}}
__host__ __device__ int dhd() { return 7; } // expected-error {{redefinition of 'dhd'}}
// expected-warning@-1 {{attribute declaration must precede definition}}
// expected-note@-3 {{previous definition is here}}

// Same tests for extern "C" functions.
extern "C" __host__ int chh() {return 11;} // expected-note {{previous definition is here}}
extern "C" int chh() {return 11;} // expected-error {{redefinition of 'chh'}}

// H/D overloading is OK.
extern "C" __device__ int cdh() {return 10;}
extern "C" __host__ int cdh() {return 11;}

// H/HD and D/HD overloading is not allowed.
extern "C" __host__ __device__ int chhd1() {return 12;} // expected-note {{previous definition is here}}
extern "C" __host__ int chhd1() {return 13;} // expected-error {{redefinition of 'chhd1'}}

extern "C" __host__ int chhd2() {return 13;} // expected-note {{previous definition is here}}
extern "C" __host__ __device__ int chhd2() {return 12;} // expected-error {{redefinition of 'chhd2'}}
// expected-warning@-1 {{attribute declaration must precede definition}}
// expected-note@-3 {{previous definition is here}}

// Helper functions to verify calling restrictions.
__device__ int d() { return 8; }
// expected-note@-1 1+ {{'d' declared here}}
// expected-note@-2 1+ {{candidate function not viable: call to __device__ function from __host__ function}}
// expected-note@-3 0+ {{candidate function not viable: call to __device__ function from __host__ __device__ function}}

__host__ int h() { return 9; }
// expected-note@-1 1+ {{'h' declared here}}
// expected-note@-2 1+ {{candidate function not viable: call to __host__ function from __device__ function}}
// expected-note@-3 0+ {{candidate function not viable: call to __host__ function from __host__ __device__ function}}
// expected-note@-4 1+ {{candidate function not viable: call to __host__ function from __global__ function}}

__global__ void g() {}
// expected-note@-1 1+ {{'g' declared here}}
// expected-note@-2 1+ {{candidate function not viable: call to __global__ function from __device__ function}}
// expected-note@-3 0+ {{candidate function not viable: call to __global__ function from __host__ __device__ function}}
// expected-note@-4 1+ {{candidate function not viable: call to __global__ function from __global__ function}}

extern "C" __device__ int cd() {return 10;}
// expected-note@-1 1+ {{'cd' declared here}}
// expected-note@-2 1+ {{candidate function not viable: call to __device__ function from __host__ function}}
// expected-note@-3 0+ {{candidate function not viable: call to __device__ function from __host__ __device__ function}}

extern "C" __host__ int ch() {return 11;}
// expected-note@-1 1+ {{'ch' declared here}}
// expected-note@-2 1+ {{candidate function not viable: call to __host__ function from __device__ function}}
// expected-note@-3 0+ {{candidate function not viable: call to __host__ function from __host__ __device__ function}}
// expected-note@-4 1+ {{candidate function not viable: call to __host__ function from __global__ function}}

__host__ void hostf() {
  fp_t dp = d;
  // expected-error@-1 {{reference to __device__ function 'd' in __host__ function}}
  fp_t cdp = cd;
  // expected-error@-1 {{reference to __device__ function 'cd' in __host__ function}}
  fp_t hp = h;
  fp_t chp = ch;
  fp_t dhp = dh;
  fp_t cdhp = cdh;
  gp_t gp = g;

  d();
  // expected-error@-1 {{no matching function for call to 'd'}}
  cd();
  // expected-error@-1 {{no matching function for call to 'cd'}}
  h();
  ch();
  dh();
  cdh();
  g(); // expected-error {{call to global function g not configured}}
  g<<<0,0>>>();
}

__device__ void devicef() {
  fp_t dp = d;
  fp_t cdp = cd;
  fp_t hp = h;
  // expected-error@-1 {{reference to __host__ function 'h' in __device__ function}}
  fp_t chp = ch;
  // expected-error@-1 {{reference to __host__ function 'ch' in __device__ function}}
  fp_t dhp = dh;
  fp_t cdhp = cdh;
  gp_t gp = g; // expected-error {{reference to __global__ function 'g' in __device__ function}}

  d();
  cd();
  h(); // expected-error {{no matching function for call to 'h'}}
  ch(); // expected-error {{no matching function for call to 'ch'}}
  dh();
  cdh();
  g(); // expected-error {{no matching function for call to 'g'}}
  g<<<0,0>>>(); // expected-error {{reference to __global__ function 'g' in __device__ function}}
}

__global__ void globalf() {
  fp_t dp = d;
  fp_t cdp = cd;
  fp_t hp = h;
  // expected-error@-1 {{reference to __host__ function 'h' in __global__ function}}
  fp_t chp = ch;
  // expected-error@-1 {{reference to __host__ function 'ch' in __global__ function}}
  fp_t dhp = dh;
  fp_t cdhp = cdh;
  gp_t gp = g;
  // expected-error@-1 {{reference to __global__ function 'g' in __global__ function}}

  d();
  cd();
  h();
  // expected-error@-1 {{no matching function for call to 'h'}}
  ch();
  // expected-error@-1 {{no matching function for call to 'ch'}}
  dh();
  cdh();
  g(); // expected-error {{no matching function for call to 'g'}}
  g<<<0,0>>>(); // expected-error {{reference to __global__ function 'g' in __global__ function}}
}

__host__ __device__ void hostdevicef() {
  fp_t dp = d;
  fp_t cdp = cd;
#if !defined(NOCHECKS) && !defined(__CUDA_ARCH__)
  // expected-error@-3 {{reference to __device__ function 'd' in __host__ __device__ function}}
  // expected-error@-3 {{reference to __device__ function 'cd' in __host__ __device__ function}}
#endif

  fp_t hp = h;
  fp_t chp = ch;
#if !defined(NOCHECKS) && defined(__CUDA_ARCH__)
  // expected-error@-3 {{reference to __host__ function 'h' in __host__ __device__ function}}
  // expected-error@-3 {{reference to __host__ function 'ch' in __host__ __device__ function}}
#endif

  fp_t dhp = dh;
  fp_t cdhp = cdh;
  gp_t gp = g;
#if defined(__CUDA_ARCH__)
  // expected-error@-2 {{reference to __global__ function 'g' in __host__ __device__ function}}
#endif

  d();
  cd();
#if !defined(NOCHECKS) && !defined(__CUDA_ARCH__)
  // expected-error@-3 {{no matching function for call to 'd'}}
  // expected-error@-3 {{no matching function for call to 'cd'}}
#endif

  h();
  ch();
#if !defined(NOCHECKS) && defined(__CUDA_ARCH__)
  // expected-error@-3 {{no matching function for call to 'h'}}
  // expected-error@-3 {{no matching function for call to 'ch'}}
#endif

  dh();
  cdh();
  g();
  g<<<0,0>>>();
#if !defined(__CUDA_ARCH__)
  // expected-error@-3 {{call to global function g not configured}}
#else
  // expected-error@-5 {{no matching function for call to 'g'}}
  // expected-error@-5 {{reference to __global__ function 'g' in __host__ __device__ function}}
#endif  // __CUDA_ARCH__
}

// Test for address of overloaded function resolution in the global context.
fp_t hp = h;
fp_t chp = ch;
fp_t dhp = dh;
fp_t cdhp = cdh;
gp_t gp = g;


// Test overloading of destructors
// Can't mix H and unattributed destructors
struct d_h {
  ~d_h() {} // expected-note {{previous declaration is here}}
  __host__ ~d_h() {} // expected-error {{destructor cannot be redeclared}}
};

// H/D overloading is OK
struct d_dh {
  __device__ ~d_dh() {}
  __host__ ~d_dh() {}
};

// HD is OK
struct d_hd {
  __host__ __device__ ~d_hd() {}
};

// Mixing H/D and HD is not allowed.
struct d_dhhd {
  __device__ ~d_dhhd() {}
  __host__ ~d_dhhd() {} // expected-note {{previous declaration is here}}
  __host__ __device__ ~d_dhhd() {} // expected-error {{destructor cannot be redeclared}}
};

struct d_hhd {
  __host__ ~d_hhd() {} // expected-note {{previous declaration is here}}
  __host__ __device__ ~d_hhd() {} // expected-error {{destructor cannot be redeclared}}
};

struct d_hdh {
  __host__ __device__ ~d_hdh() {} // expected-note {{previous declaration is here}}
  __host__ ~d_hdh() {} // expected-error {{destructor cannot be redeclared}}
};

struct d_dhd {
  __device__ ~d_dhd() {} // expected-note {{previous declaration is here}}
  __host__ __device__ ~d_dhd() {} // expected-error {{destructor cannot be redeclared}}
};

struct d_hdd {
  __host__ __device__ ~d_hdd() {} // expected-note {{previous declaration is here}}
  __device__ ~d_hdd() {} // expected-error {{destructor cannot be redeclared}}
};

// Test overloading of member functions
struct m_h {
  void operator delete(void *ptr); // expected-note {{previous declaration is here}}
  __host__ void operator delete(void *ptr); // expected-error {{class member cannot be redeclared}}
};

// D/H overloading is OK
struct m_dh {
  __device__ void operator delete(void *ptr);
  __host__ void operator delete(void *ptr);
};

// HD by itself is OK
struct m_hd {
  __device__ __host__ void operator delete(void *ptr);
};

struct m_hhd {
  __host__ void operator delete(void *ptr) {} // expected-note {{previous declaration is here}}
  __host__ __device__ void operator delete(void *ptr) {} // expected-error {{class member cannot be redeclared}}
};

struct m_hdh {
  __host__ __device__ void operator delete(void *ptr) {} // expected-note {{previous declaration is here}}
  __host__ void operator delete(void *ptr) {} // expected-error {{class member cannot be redeclared}}
};

struct m_dhd {
  __device__ void operator delete(void *ptr) {} // expected-note {{previous declaration is here}}
  __host__ __device__ void operator delete(void *ptr) {} // expected-error {{class member cannot be redeclared}}
};

struct m_hdd {
  __host__ __device__ void operator delete(void *ptr) {} // expected-note {{previous declaration is here}}
  __device__ void operator delete(void *ptr) {} // expected-error {{class member cannot be redeclared}}
};

// __global__ functions can't be overloaded based on attribute
// difference.
struct G {
  friend void friend_of_g(G &arg);
private:
  int x;
};
__global__ void friend_of_g(G &arg) { int x = arg.x; } // expected-note {{previous definition is here}}
void friend_of_g(G &arg) { int x = arg.x; } // expected-error {{redefinition of 'friend_of_g'}}
