// Test host codegen.
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

enum omp_allocator_handle_t {
  omp_null_allocator = 0,
  omp_default_mem_alloc = 1,
  omp_large_cap_mem_alloc = 2,
  omp_const_mem_alloc = 3,
  omp_high_bw_mem_alloc = 4,
  omp_low_lat_mem_alloc = 5,
  omp_cgroup_mem_alloc = 6,
  omp_pteam_mem_alloc = 7,
  omp_thread_mem_alloc = 8,
  KMP_ALLOCATOR_MAX_HANDLE = __UINTPTR_MAX__
};

typedef enum omp_alloctrait_key_t { omp_atk_sync_hint = 1,
                                    omp_atk_alignment = 2,
                                    omp_atk_access = 3,
                                    omp_atk_pool_size = 4,
                                    omp_atk_fallback = 5,
                                    omp_atk_fb_data = 6,
                                    omp_atk_pinned = 7,
                                    omp_atk_partition = 8
} omp_alloctrait_key_t;
typedef enum omp_alloctrait_value_t {
  omp_atv_false = 0,
  omp_atv_true = 1,
  omp_atv_default = 2,
  omp_atv_contended = 3,
  omp_atv_uncontended = 4,
  omp_atv_sequential = 5,
  omp_atv_private = 6,
  omp_atv_all = 7,
  omp_atv_thread = 8,
  omp_atv_pteam = 9,
  omp_atv_cgroup = 10,
  omp_atv_default_mem_fb = 11,
  omp_atv_null_fb = 12,
  omp_atv_abort_fb = 13,
  omp_atv_allocator_fb = 14,
  omp_atv_environment = 15,
  omp_atv_nearest = 16,
  omp_atv_blocked = 17,
  omp_atv_interleaved = 18
} omp_alloctrait_value_t;

typedef struct omp_alloctrait_t {
  omp_alloctrait_key_t key;
  __UINTPTR_TYPE__ value;
} omp_alloctrait_t;

// Just map the traits variable as a firstprivate variable.
// CHECK-DAG: [[SIZES:@.+]] = private unnamed_addr constant [1 x i64] [i64 160]
// CHECK-DAG: [[MAPTYPES:@.+]] = private unnamed_addr constant [1 x i64] [i64 673]

// CHECK: define {{.*}}[[FOO:@.+]]()
void foo() {
  omp_alloctrait_t traits[10];
  omp_allocator_handle_t my_allocator;

// CHECK: [[RES:%.+]] = call i32 @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @.[[TGT_REGION:.+]].region_id, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* getelementptr inbounds ([1 x i64], [1 x i64]* [[SIZES]], i32 0, i32 0), i64* getelementptr inbounds ([1 x i64], [1 x i64]* [[MAPTYPES]], i32 0, i32 0), i8** null, i8** null, i32 1, i32 0)
// CHECK: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[CMP]], label %[[FAILED:.+]], label %[[DONE:.+]]
// CHECK: [[FAILED]]:
// CHECK: call void @[[TGT_REGION]]([10 x %struct.omp_alloctrait_t]* %{{[^,]+}})
#pragma omp target parallel for simd uses_allocators(omp_null_allocator, omp_thread_mem_alloc, my_allocator(traits))
  for (int i = 0; i < 10; ++i)
    ;
}

// CHECK: define internal void @[[TGT_REGION]]([10 x %struct.omp_alloctrait_t]* {{.+}})
// CHECK: [[TRAITS_ADDR_REF:%.+]] = alloca [10 x %struct.omp_alloctrait_t]*,
// CHECK: [[MY_ALLOCATOR_ADDR:%.+]] = alloca i64,
// CHECK: [[TRAITS_ADDR:%.+]] = load [10 x %struct.omp_alloctrait_t]*, [10 x %struct.omp_alloctrait_t]** [[TRAITS_ADDR_REF]],
// CHECK: [[TRAITS_ADDR_VOIDPTR:%.+]] = bitcast [10 x %struct.omp_alloctrait_t]* [[TRAITS_ADDR]] to i8**
// CHECK: [[TRAITS:%.+]] = load i8*, i8** [[TRAITS_ADDR_VOIDPTR]],
// CHECK: [[ALLOCATOR:%.+]] = call i8* @__kmpc_init_allocator(i32 %{{.+}}, i8* null, i32 10, i8* [[TRAITS]])
// CHECK: [[CONV:%.+]] = ptrtoint i8* [[ALLOCATOR]] to i64
// CHECK: store i64 [[CONV]], i64* [[MY_ALLOCATOR_ADDR]],

// Destroy allocator upon exit from the region.
// CHECK: [[ALLOCATOR:%.+]] = load i64, i64* [[MY_ALLOCATOR_ADDR]],
// CHECK: [[CONV:%.+]] = inttoptr i64 [[ALLOCATOR]] to i8*
// CHECK: call void @__kmpc_destroy_allocator(i32 %{{.+}}, i8* [[CONV]])
// CHECK: ret void

#endif
