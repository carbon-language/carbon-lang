//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#include <omp.h>
#include "offload.h"
#include "compiler_if_target.h"

// OpenMP API

void omp_set_default_device(int num)
{
}

int omp_get_default_device(void)
{
    return mic_index;
}

int omp_get_num_devices()
{
    return mic_engines_total;
}

// OpenMP API wrappers

static void omp_send_int_to_host(
    void *ofld_,
    int setting
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[1] = {0};

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_out;
    vars[0].ptr = &setting;

    OFFLOAD_TARGET_ENTER(ofld, 1, vars, NULL);
    OFFLOAD_TARGET_LEAVE(ofld);
}

static int omp_get_int_from_host(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[1] = {0};
    int setting;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_in;
    vars[0].ptr = &setting;

    OFFLOAD_TARGET_ENTER(ofld, 1, vars, NULL);
    OFFLOAD_TARGET_LEAVE(ofld);

    return setting;
}

void omp_set_num_threads_lrb(
    void *ofld
)
{
    int num_threads;

    num_threads = omp_get_int_from_host(ofld);
    omp_set_num_threads(num_threads);
}

void omp_get_max_threads_lrb(
    void *ofld
)
{
    int num_threads;

    num_threads = omp_get_max_threads();
    omp_send_int_to_host(ofld, num_threads);
}

void omp_get_num_procs_lrb(
    void *ofld
)
{
    int num_procs;

    num_procs = omp_get_num_procs();
    omp_send_int_to_host(ofld, num_procs);
}

void omp_set_dynamic_lrb(
    void *ofld
)
{
    int dynamic;

    dynamic = omp_get_int_from_host(ofld);
    omp_set_dynamic(dynamic);
}

void omp_get_dynamic_lrb(
    void *ofld
)
{
    int dynamic;

    dynamic = omp_get_dynamic();
    omp_send_int_to_host(ofld, dynamic);
}

void omp_set_nested_lrb(
    void *ofld
)
{
    int nested;

    nested = omp_get_int_from_host(ofld);
    omp_set_nested(nested);
}

void omp_get_nested_lrb(
    void *ofld
)
{
    int nested;

    nested = omp_get_nested();
    omp_send_int_to_host(ofld, nested);
}

void omp_set_schedule_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[2] = {0};
    omp_sched_t kind;
    int modifier;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_in;
    vars[0].ptr = &kind;

    vars[1].type.src = c_data;
    vars[1].type.dst = c_data;
    vars[1].direction.bits = c_parameter_in;
    vars[1].ptr = &modifier;

    OFFLOAD_TARGET_ENTER(ofld, 2, vars, NULL);
    omp_set_schedule(kind, modifier);
    OFFLOAD_TARGET_LEAVE(ofld);
}

void omp_get_schedule_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[2] = {0};
    omp_sched_t kind;
    int modifier;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_out;
    vars[0].ptr = &kind;

    vars[1].type.src = c_data;
    vars[1].type.dst = c_data;
    vars[1].direction.bits = c_parameter_out;
    vars[1].ptr = &modifier;

    OFFLOAD_TARGET_ENTER(ofld, 2, vars, NULL);
    omp_get_schedule(&kind, &modifier);
    OFFLOAD_TARGET_LEAVE(ofld);
}

// lock API functions

void omp_init_lock_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[1] = {0};
    omp_lock_target_t lock;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_out;
    vars[0].ptr = &lock;

    OFFLOAD_TARGET_ENTER(ofld, 1, vars, NULL);
    omp_init_lock(&lock.lock);
    OFFLOAD_TARGET_LEAVE(ofld);
}

void omp_destroy_lock_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[1] = {0};
    omp_lock_target_t lock;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_in;
    vars[0].ptr = &lock;

    OFFLOAD_TARGET_ENTER(ofld, 1, vars, NULL);
    omp_destroy_lock(&lock.lock);
    OFFLOAD_TARGET_LEAVE(ofld);
}

void omp_set_lock_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[1] = {0};
    omp_lock_target_t lock;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_inout;
    vars[0].ptr = &lock;

    OFFLOAD_TARGET_ENTER(ofld, 1, vars, NULL);
    omp_set_lock(&lock.lock);
    OFFLOAD_TARGET_LEAVE(ofld);
}

void omp_unset_lock_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[1] = {0};
    omp_lock_target_t lock;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_inout;
    vars[0].ptr = &lock;

    OFFLOAD_TARGET_ENTER(ofld, 1, vars, NULL);
    omp_unset_lock(&lock.lock);
    OFFLOAD_TARGET_LEAVE(ofld);
}

void omp_test_lock_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[2] = {0};
    omp_lock_target_t lock;
    int result;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_inout;
    vars[0].ptr = &lock;

    vars[1].type.src = c_data;
    vars[1].type.dst = c_data;
    vars[1].direction.bits = c_parameter_out;
    vars[1].ptr = &result;

    OFFLOAD_TARGET_ENTER(ofld, 2, vars, NULL);
    result = omp_test_lock(&lock.lock);
    OFFLOAD_TARGET_LEAVE(ofld);
}

// nested lock API functions

void omp_init_nest_lock_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[1] = {0};
    omp_nest_lock_target_t lock;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_out;
    vars[0].ptr = &lock;

    OFFLOAD_TARGET_ENTER(ofld, 1, vars, NULL);
    omp_init_nest_lock(&lock.lock);
    OFFLOAD_TARGET_LEAVE(ofld);
}

void omp_destroy_nest_lock_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[1] = {0};
    omp_nest_lock_target_t lock;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_in;
    vars[0].ptr = &lock;

    OFFLOAD_TARGET_ENTER(ofld, 1, vars, NULL);
    omp_destroy_nest_lock(&lock.lock);
    OFFLOAD_TARGET_LEAVE(ofld);
}

void omp_set_nest_lock_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[1] = {0};
    omp_nest_lock_target_t lock;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_inout;
    vars[0].ptr = &lock;

    OFFLOAD_TARGET_ENTER(ofld, 1, vars, NULL);
    omp_set_nest_lock(&lock.lock);
    OFFLOAD_TARGET_LEAVE(ofld);
}

void omp_unset_nest_lock_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[1] = {0};
    omp_nest_lock_target_t lock;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_inout;
    vars[0].ptr = &lock;

    OFFLOAD_TARGET_ENTER(ofld, 1, vars, NULL);
    omp_unset_nest_lock(&lock.lock);
    OFFLOAD_TARGET_LEAVE(ofld);
}

void omp_test_nest_lock_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[2] = {0};
    omp_nest_lock_target_t lock;
    int result;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_inout;
    vars[0].ptr = &lock;

    vars[1].type.src = c_data;
    vars[1].type.dst = c_data;
    vars[1].direction.bits = c_parameter_out;
    vars[1].ptr = &result;

    OFFLOAD_TARGET_ENTER(ofld, 2, vars, NULL);
    result = omp_test_nest_lock(&lock.lock);
    OFFLOAD_TARGET_LEAVE(ofld);
}

// kmp API functions

void kmp_set_stacksize_lrb(
    void *ofld
)
{
    int size;

    size = omp_get_int_from_host(ofld);
    kmp_set_stacksize(size);
}

void kmp_get_stacksize_lrb(
    void *ofld
)
{
    int size;

    size = kmp_get_stacksize();
    omp_send_int_to_host(ofld, size);
}

void kmp_set_stacksize_s_lrb(
    void *ofld
)
{
    int size;

    size = omp_get_int_from_host(ofld);
    kmp_set_stacksize_s(size);
}

void kmp_get_stacksize_s_lrb(
    void *ofld
)
{
    int size;

    size = kmp_get_stacksize_s();
    omp_send_int_to_host(ofld, size);
}

void kmp_set_blocktime_lrb(
    void *ofld
)
{
    int time;

    time = omp_get_int_from_host(ofld);
    kmp_set_blocktime(time);
}

void kmp_get_blocktime_lrb(
    void *ofld
)
{
    int time;

    time = kmp_get_blocktime();
    omp_send_int_to_host(ofld, time);
}

void kmp_set_library_serial_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;

    OFFLOAD_TARGET_ENTER(ofld, 0, 0, 0);
    kmp_set_library_serial();
    OFFLOAD_TARGET_LEAVE(ofld);
}

void kmp_set_library_turnaround_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;

    OFFLOAD_TARGET_ENTER(ofld, 0, 0, 0);
    kmp_set_library_turnaround();
    OFFLOAD_TARGET_LEAVE(ofld);
}

void kmp_set_library_throughput_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;

    OFFLOAD_TARGET_ENTER(ofld, 0, 0, 0);
    kmp_set_library_throughput();
    OFFLOAD_TARGET_LEAVE(ofld);
}

void kmp_set_library_lrb(
    void *ofld
)
{
    int mode;

    mode = omp_get_int_from_host(ofld);
    kmp_set_library(mode);
}

void kmp_get_library_lrb(
    void *ofld
)
{
    int mode;

    mode = kmp_get_library();
    omp_send_int_to_host(ofld, mode);
}

void kmp_set_defaults_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[1] = {0};
    char *defaults = 0;

    vars[0].type.src = c_string_ptr;
    vars[0].type.dst = c_string_ptr;
    vars[0].direction.bits = c_parameter_in;
    vars[0].ptr = &defaults;

    OFFLOAD_TARGET_ENTER(ofld, 1, vars, NULL);
    kmp_set_defaults(defaults);
    OFFLOAD_TARGET_LEAVE(ofld);
}

// affinity API functions

void kmp_create_affinity_mask_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[1] = {0};
    kmp_affinity_mask_target_t mask;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_out;
    vars[0].ptr = &mask;

    OFFLOAD_TARGET_ENTER(ofld, 1, vars, NULL);
    kmp_create_affinity_mask(&mask.mask);
    OFFLOAD_TARGET_LEAVE(ofld);
}

void kmp_destroy_affinity_mask_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[1] = {0};
    kmp_affinity_mask_target_t mask;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_in;
    vars[0].ptr = &mask;

    OFFLOAD_TARGET_ENTER(ofld, 1, vars, NULL);
    kmp_destroy_affinity_mask(&mask.mask);
    OFFLOAD_TARGET_LEAVE(ofld);
}

void kmp_set_affinity_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[2] = {0};
    kmp_affinity_mask_target_t mask;
    int result;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_in;
    vars[0].ptr = &mask;

    vars[1].type.src = c_data;
    vars[1].type.dst = c_data;
    vars[1].direction.bits = c_parameter_out;
    vars[1].ptr = &result;

    OFFLOAD_TARGET_ENTER(ofld, 2, vars, NULL);
    result = kmp_set_affinity(&mask.mask);
    OFFLOAD_TARGET_LEAVE(ofld);
}

void kmp_get_affinity_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[2] = {0};
    kmp_affinity_mask_target_t mask;
    int result;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_inout;
    vars[0].ptr = &mask;

    vars[1].type.src = c_data;
    vars[1].type.dst = c_data;
    vars[1].direction.bits = c_parameter_out;
    vars[1].ptr = &result;

    OFFLOAD_TARGET_ENTER(ofld, 2, vars, NULL);
    result = kmp_get_affinity(&mask.mask);
    OFFLOAD_TARGET_LEAVE(ofld);
}

void kmp_get_affinity_max_proc_lrb(
    void *ofld
)
{
    int max_proc;

    max_proc = kmp_get_affinity_max_proc();
    omp_send_int_to_host(ofld, max_proc);
}

void kmp_set_affinity_mask_proc_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[3] = {0};
    kmp_affinity_mask_target_t mask;
    int proc, result;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_in;
    vars[0].ptr = &proc;

    vars[1].type.src = c_data;
    vars[1].type.dst = c_data;
    vars[1].direction.bits = c_parameter_inout;
    vars[1].ptr = &mask;

    vars[2].type.src = c_data;
    vars[2].type.dst = c_data;
    vars[2].direction.bits = c_parameter_out;
    vars[2].ptr = &result;

    OFFLOAD_TARGET_ENTER(ofld, 3, vars, NULL);
    result = kmp_set_affinity_mask_proc(proc, &mask.mask);
    OFFLOAD_TARGET_LEAVE(ofld);
}

void kmp_unset_affinity_mask_proc_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[3] = {0};
    kmp_affinity_mask_target_t mask;
    int proc, result;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_in;
    vars[0].ptr = &proc;

    vars[1].type.src = c_data;
    vars[1].type.dst = c_data;
    vars[1].direction.bits = c_parameter_inout;
    vars[1].ptr = &mask;

    vars[2].type.src = c_data;
    vars[2].type.dst = c_data;
    vars[2].direction.bits = c_parameter_out;
    vars[2].ptr = &result;

    OFFLOAD_TARGET_ENTER(ofld, 3, vars, NULL);
    result = kmp_unset_affinity_mask_proc(proc, &mask.mask);
    OFFLOAD_TARGET_LEAVE(ofld);
}

void kmp_get_affinity_mask_proc_lrb(
    void *ofld_
)
{
    OFFLOAD ofld = (OFFLOAD) ofld_;
    VarDesc vars[3] = {0};
    kmp_affinity_mask_target_t mask;
    int proc, result;

    vars[0].type.src = c_data;
    vars[0].type.dst = c_data;
    vars[0].direction.bits = c_parameter_in;
    vars[0].ptr = &proc;

    vars[1].type.src = c_data;
    vars[1].type.dst = c_data;
    vars[1].direction.bits = c_parameter_in;
    vars[1].ptr = &mask;

    vars[2].type.src = c_data;
    vars[2].type.dst = c_data;
    vars[2].direction.bits = c_parameter_out;
    vars[2].ptr = &result;

    OFFLOAD_TARGET_ENTER(ofld, 3, vars, NULL);
    result = kmp_get_affinity_mask_proc(proc, &mask.mask);
    OFFLOAD_TARGET_LEAVE(ofld);
}

// Target-side stubs for the host functions (to avoid unresolveds)
// These are needed for the offloadm table

void omp_set_num_threads_target(
    TARGET_TYPE target_type,
    int target_number,
    int num_threads
)
{
}

int omp_get_max_threads_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    return 0;
}

int omp_get_num_procs_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    return 0;
}

void omp_set_dynamic_target(
    TARGET_TYPE target_type,
    int target_number,
    int num_threads
)
{
}

int omp_get_dynamic_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    return 0;
}

void omp_set_nested_target(
    TARGET_TYPE target_type,
    int target_number,
    int num_threads
)
{
}

int omp_get_nested_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    return 0;
}

void omp_set_schedule_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_sched_t kind,
    int modifier
)
{
}

void omp_get_schedule_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_sched_t *kind,
    int *modifier
)
{
}

void omp_init_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_lock_target_t *lock
)
{
}

void omp_destroy_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_lock_target_t *lock
)
{
}

void omp_set_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_lock_target_t *lock
)
{
}

void omp_unset_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_lock_target_t *lock
)
{
}

int omp_test_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_lock_target_t *lock
)
{
    return 0;
}

void omp_init_nest_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_nest_lock_target_t *lock
)
{
}

void omp_destroy_nest_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_nest_lock_target_t *lock
)
{
}

void omp_set_nest_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_nest_lock_target_t *lock
)
{
}

void omp_unset_nest_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_nest_lock_target_t *lock
)
{
}

int omp_test_nest_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_nest_lock_target_t *lock
)
{
    return 0;
}

void kmp_set_stacksize_target(
    TARGET_TYPE target_type,
    int target_number,
    int size
)
{
}

int kmp_get_stacksize_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    return 0;
}

void kmp_set_stacksize_s_target(
    TARGET_TYPE target_type,
    int target_number,
    size_t size
)
{
}

size_t kmp_get_stacksize_s_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    return 0;
}

void kmp_set_blocktime_target(
    TARGET_TYPE target_type,
    int target_number,
    int time
)
{
}

int kmp_get_blocktime_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    return 0;
}

void kmp_set_library_serial_target(
    TARGET_TYPE target_type,
    int target_number
)
{
}

void kmp_set_library_turnaround_target(
    TARGET_TYPE target_type,
    int target_number
)
{
}

void kmp_set_library_throughput_target(
    TARGET_TYPE target_type,
    int target_number
)
{
}

void kmp_set_library_target(
    TARGET_TYPE target_type,
    int target_number,
    int mode
)
{
}

int kmp_get_library_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    return 0;
}

void kmp_set_defaults_target(
    TARGET_TYPE target_type,
    int target_number,
    char const *defaults
)
{
}

void kmp_create_affinity_mask_target(
    TARGET_TYPE target_type,
    int target_number,
    kmp_affinity_mask_target_t *mask
)
{
}

void kmp_destroy_affinity_mask_target(
    TARGET_TYPE target_type,
    int target_number,
    kmp_affinity_mask_target_t *mask
)
{
}

int kmp_set_affinity_target(
    TARGET_TYPE target_type,
    int target_number,
    kmp_affinity_mask_target_t *mask
)
{
    return 0;
}

int kmp_get_affinity_target(
    TARGET_TYPE target_type,
    int target_number,
    kmp_affinity_mask_target_t *mask
)
{
    return 0;
}

int kmp_get_affinity_max_proc_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    return 0;
}

int kmp_set_affinity_mask_proc_target(
    TARGET_TYPE target_type,
    int target_number,
    int proc,
    kmp_affinity_mask_target_t *mask
)
{
    return 0;
}

int kmp_unset_affinity_mask_proc_target(
    TARGET_TYPE target_type,
    int target_number,
    int proc,
    kmp_affinity_mask_target_t *mask
)
{
    return 0;
}

int kmp_get_affinity_mask_proc_target(
    TARGET_TYPE target_type,
    int target_number,
    int proc,
    kmp_affinity_mask_target_t *mask
)
{
    return 0;
}
