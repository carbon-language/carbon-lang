// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s

// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp-simd -ferror-limit 100 -o - %s

struct St{
 int a;
};

int sss;
#pragma omp allocate(sss) allocate // expected-warning {{extra tokens at the end of '#pragma omp allocate' are ignored}}
#pragma omp allocate(sss) allocator // expected-error {{expected '(' after 'allocator'}}
#pragma omp allocate(sss) allocator(0,  // expected-error {{expected ')'}} expected-error {{omp_allocator_handle_t type not found; include <omp.h>}} expected-note {{to match this '('}}
#pragma omp allocate(sss) allocator(0,sss  // expected-error {{expected ')'}} expected-error {{omp_allocator_handle_t type not found; include <omp.h>}} expected-note {{to match this '('}}
#pragma omp allocate(sss) allocator(0,sss)  // expected-error {{expected ')'}} expected-error {{omp_allocator_handle_t type not found; include <omp.h>}} expected-note {{to match this '('}}
#pragma omp allocate(sss) allocator(sss)  // expected-error {{omp_allocator_handle_t type not found; include <omp.h>}}

typedef void *omp_allocator_handle_t;

struct St1{
 int a;
 static int b;
#pragma omp allocate(b) allocator(sss) // expected-error {{initializing 'omp_allocator_handle_t' (aka 'void *') with an expression of incompatible type 'int'}}
} d; // expected-note 2 {{'d' defined here}}

// expected-error@+1 {{expected one of the predefined allocators for the variables with the static storage: 'omp_default_mem_alloc', 'omp_large_cap_mem_alloc', 'omp_const_mem_alloc', 'omp_high_bw_mem_alloc', 'omp_low_lat_mem_alloc', 'omp_cgroup_mem_alloc', 'omp_pteam_mem_alloc' or 'omp_thread_mem_alloc'}}
#pragma omp allocate(d) allocator(nullptr)
extern void *allocator;
// expected-error@+1 {{expected one of the predefined allocators for the variables with the static storage: 'omp_default_mem_alloc', 'omp_large_cap_mem_alloc', 'omp_const_mem_alloc', 'omp_high_bw_mem_alloc', 'omp_low_lat_mem_alloc', 'omp_cgroup_mem_alloc', 'omp_pteam_mem_alloc' or 'omp_thread_mem_alloc'}}
#pragma omp allocate(d) allocator(allocator)
