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
#include "compiler_if_host.h"

// OpenMP API

void omp_set_default_device(int num)
{
    if (num >= 0) {
        __omp_device_num = num;
    }
}

int omp_get_default_device(void)
{
    return __omp_device_num;
}

int omp_get_num_devices()
{
    __offload_init_library();
    return mic_engines_total;
}

// OpenMP API wrappers

static void omp_set_int_target(
    TARGET_TYPE target_type,
    int target_number,
    int setting,
    const char* f_name
)
{
    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          f_name, 0);
    if (ofld) {
        VarDesc vars[1] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_in;
        vars[0].size = sizeof(int);
        vars[0].count = 1;
        vars[0].ptr = &setting;

        OFFLOAD_OFFLOAD(ofld, f_name, 0, 1, vars, NULL, 0, 0, 0);
    }
}

static int omp_get_int_target(
    TARGET_TYPE target_type,
    int target_number,
    const char * f_name
)
{
    int setting = 0;

    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          f_name, 0);
    if (ofld) {
        VarDesc vars[1] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_out;
        vars[0].size = sizeof(int);
        vars[0].count = 1;
        vars[0].ptr = &setting;

        OFFLOAD_OFFLOAD(ofld, f_name, 0, 1, vars, NULL, 0, 0, 0);
    }
    return setting;
}

void omp_set_num_threads_target(
    TARGET_TYPE target_type,
    int target_number,
    int num_threads
)
{
    omp_set_int_target(target_type, target_number, num_threads,
                       "omp_set_num_threads_target");
}

int omp_get_max_threads_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    return omp_get_int_target(target_type, target_number,
                              "omp_get_max_threads_target");
}

int omp_get_num_procs_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    return omp_get_int_target(target_type, target_number,
                              "omp_get_num_procs_target");
}

void omp_set_dynamic_target(
    TARGET_TYPE target_type,
    int target_number,
    int num_threads
)
{
    omp_set_int_target(target_type, target_number, num_threads,
                       "omp_set_dynamic_target");
}

int omp_get_dynamic_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    return omp_get_int_target(target_type, target_number,
                              "omp_get_dynamic_target");
}

void omp_set_nested_target(
    TARGET_TYPE target_type,
    int target_number,
    int nested
)
{
    omp_set_int_target(target_type, target_number, nested,
                       "omp_set_nested_target");
}

int omp_get_nested_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    return omp_get_int_target(target_type, target_number,
                              "omp_get_nested_target");
}

void omp_set_schedule_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_sched_t kind,
    int modifier
)
{
    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[2] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_in;
        vars[0].size = sizeof(omp_sched_t);
        vars[0].count = 1;
        vars[0].ptr = &kind;

        vars[1].type.src = c_data;
        vars[1].type.dst = c_data;
        vars[1].direction.bits = c_parameter_in;
        vars[1].size = sizeof(int);
        vars[1].count = 1;
        vars[1].ptr = &modifier;

        OFFLOAD_OFFLOAD(ofld, "omp_set_schedule_target",
                        0, 2, vars, NULL, 0, 0, 0);
    }
}

void omp_get_schedule_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_sched_t *kind,
    int *modifier
)
{
    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[2] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_out;
        vars[0].size = sizeof(omp_sched_t);
        vars[0].count = 1;
        vars[0].ptr = kind;

        vars[1].type.src = c_data;
        vars[1].type.dst = c_data;
        vars[1].direction.bits = c_parameter_out;
        vars[1].size = sizeof(int);
        vars[1].count = 1;
        vars[1].ptr = modifier;

        OFFLOAD_OFFLOAD(ofld, "omp_get_schedule_target",
                        0, 2, vars, NULL, 0, 0, 0);
    }
}

// lock API functions

void omp_init_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_lock_target_t *lock
)
{
    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[1] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_out;
        vars[0].size = sizeof(omp_lock_target_t);
        vars[0].count = 1;
        vars[0].ptr = lock;

        OFFLOAD_OFFLOAD(ofld, "omp_init_lock_target",
                        0, 1, vars, NULL, 0, 0, 0);
    }
}

void omp_destroy_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_lock_target_t *lock
)
{
    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[1] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_in;
        vars[0].size = sizeof(omp_lock_target_t);
        vars[0].count = 1;
        vars[0].ptr = lock;

        OFFLOAD_OFFLOAD(ofld, "omp_destroy_lock_target",
                        0, 1, vars, NULL, 0, 0, 0);
    }
}

void omp_set_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_lock_target_t *lock
)
{
    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[1] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_inout;
        vars[0].size = sizeof(omp_lock_target_t);
        vars[0].count = 1;
        vars[0].ptr = lock;

        OFFLOAD_OFFLOAD(ofld, "omp_set_lock_target",
                        0, 1, vars, NULL, 0, 0, 0);
    }
}

void omp_unset_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_lock_target_t *lock
)
{
    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[1] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_inout;
        vars[0].size = sizeof(omp_lock_target_t);
        vars[0].count = 1;
        vars[0].ptr = lock;

        OFFLOAD_OFFLOAD(ofld, "omp_unset_lock_target",
                        0, 1, vars, NULL, 0, 0, 0);
    }
}

int omp_test_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_lock_target_t *lock
)
{
    int result = 0;

    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[2] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_inout;
        vars[0].size = sizeof(omp_lock_target_t);
        vars[0].count = 1;
        vars[0].ptr = lock;

        vars[1].type.src = c_data;
        vars[1].type.dst = c_data;
        vars[1].direction.bits = c_parameter_out;
        vars[1].size = sizeof(int);
        vars[1].count = 1;
        vars[1].ptr = &result;

        OFFLOAD_OFFLOAD(ofld, "omp_test_lock_target",
                        0, 2, vars, NULL, 0, 0, 0);
    }
    return result;
}

// nested lock API functions

void omp_init_nest_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_nest_lock_target_t *lock
)
{
    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[1] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_out;
        vars[0].size = sizeof(omp_nest_lock_target_t);
        vars[0].count = 1;
        vars[0].ptr = lock;

        OFFLOAD_OFFLOAD(ofld, "omp_init_nest_lock_target",
                        0, 1, vars, NULL, 0, 0, 0);
    }
}

void omp_destroy_nest_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_nest_lock_target_t *lock
)
{
    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[1] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_in;
        vars[0].size = sizeof(omp_nest_lock_target_t);
        vars[0].count = 1;
        vars[0].ptr = lock;

        OFFLOAD_OFFLOAD(ofld, "omp_destroy_nest_lock_target",
                        0, 1, vars, NULL, 0, 0, 0);
    }
}

void omp_set_nest_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_nest_lock_target_t *lock
)
{
    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[1] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_inout;
        vars[0].size = sizeof(omp_nest_lock_target_t);
        vars[0].count = 1;
        vars[0].ptr = lock;

        OFFLOAD_OFFLOAD(ofld, "omp_set_nest_lock_target",
                        0, 1, vars, NULL, 0, 0, 0);
    }
}

void omp_unset_nest_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_nest_lock_target_t *lock
)
{
    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[1] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_inout;
        vars[0].size = sizeof(omp_nest_lock_target_t);
        vars[0].count = 1;
        vars[0].ptr = lock;

        OFFLOAD_OFFLOAD(ofld, "omp_unset_nest_lock_target",
                        0, 1, vars, NULL, 0, 0, 0);
    }
}

int omp_test_nest_lock_target(
    TARGET_TYPE target_type,
    int target_number,
    omp_nest_lock_target_t *lock
)
{
    int result = 0;

    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[2] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_inout;
        vars[0].size = sizeof(omp_nest_lock_target_t);
        vars[0].count = 1;
        vars[0].ptr = lock;

        vars[1].type.src = c_data;
        vars[1].type.dst = c_data;
        vars[1].direction.bits = c_parameter_out;
        vars[1].size = sizeof(int);
        vars[1].count = 1;
        vars[1].ptr = &result;

        OFFLOAD_OFFLOAD(ofld, "omp_test_nest_lock_target",
                        0, 2, vars, NULL, 0, 0, 0);
    }
    return result;
}

// kmp API functions

void kmp_set_stacksize_target(
    TARGET_TYPE target_type,
    int target_number,
    int size
)
{
    omp_set_int_target(target_type, target_number, size,
                       "kmp_set_stacksize_target");
}

int kmp_get_stacksize_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    return omp_get_int_target(target_type, target_number,
                              "kmp_get_stacksize_target");
}

void kmp_set_stacksize_s_target(
    TARGET_TYPE target_type,
    int target_number,
    size_t size
)
{
    omp_set_int_target(target_type, target_number, size,
                       "kmp_set_stacksize_s_target");
}

size_t kmp_get_stacksize_s_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    return omp_get_int_target(target_type, target_number,
                              "kmp_get_stacksize_s_target");
}

void kmp_set_blocktime_target(
    TARGET_TYPE target_type,
    int target_number,
    int time
)
{
    omp_set_int_target(target_type, target_number, time,
                       "kmp_set_blocktime_target");
}

int kmp_get_blocktime_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    return omp_get_int_target(target_type, target_number,
                              "kmp_get_blocktime_target");
}

void kmp_set_library_serial_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        OFFLOAD_OFFLOAD(ofld, "kmp_set_library_serial_target",
                        0, 0, 0, 0, 0, 0, 0);
    }
}

void kmp_set_library_turnaround_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        OFFLOAD_OFFLOAD(ofld, "kmp_set_library_turnaround_target",
                        0, 0, 0, 0, 0, 0, 0);
    }
}

void kmp_set_library_throughput_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        OFFLOAD_OFFLOAD(ofld, "kmp_set_library_throughput_target",
                        0, 0, 0, 0, 0, 0, 0);
    }
}

void kmp_set_library_target(
    TARGET_TYPE target_type,
    int target_number,
    int mode
)
{
    omp_set_int_target(target_type, target_number, mode,
                       "kmp_set_library_target");
}

int kmp_get_library_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    return omp_get_int_target(target_type, target_number,
                              "kmp_get_library_target");
}

void kmp_set_defaults_target(
    TARGET_TYPE target_type,
    int target_number,
    char const *defaults
)
{
    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[1] = {0};

        vars[0].type.src = c_string_ptr;
        vars[0].type.dst = c_string_ptr;
        vars[0].direction.bits = c_parameter_in;
        vars[0].alloc_if = 1;
        vars[0].free_if = 1;
        vars[0].ptr = &defaults;

        OFFLOAD_OFFLOAD(ofld, "kmp_set_defaults_target",
                        0, 1, vars, NULL, 0, 0, 0);
    }
}

// affinity API functions

void kmp_create_affinity_mask_target(
    TARGET_TYPE target_type,
    int target_number,
    kmp_affinity_mask_target_t *mask
)
{
    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[1] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_out;
        vars[0].size = sizeof(kmp_affinity_mask_target_t);
        vars[0].count = 1;
        vars[0].ptr = mask;

        OFFLOAD_OFFLOAD(ofld, "kmp_create_affinity_mask_target",
                        0, 1, vars, NULL, 0, 0, 0);
    }
}

void kmp_destroy_affinity_mask_target(
    TARGET_TYPE target_type,
    int target_number,
    kmp_affinity_mask_target_t *mask
)
{
    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[1] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_in;
        vars[0].size = sizeof(kmp_affinity_mask_target_t);
        vars[0].count = 1;
        vars[0].ptr = mask;

        OFFLOAD_OFFLOAD(ofld, "kmp_destroy_affinity_mask_target",
                        0, 1, vars, NULL, 0, 0, 0);
    }
}

int kmp_set_affinity_target(
    TARGET_TYPE target_type,
    int target_number,
    kmp_affinity_mask_target_t *mask
)
{
    int result = 1;

    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[2] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_in;
        vars[0].size = sizeof(kmp_affinity_mask_target_t);
        vars[0].count = 1;
        vars[0].ptr = mask;

        vars[1].type.src = c_data;
        vars[1].type.dst = c_data;
        vars[1].direction.bits = c_parameter_out;
        vars[1].size = sizeof(int);
        vars[1].count = 1;
        vars[1].ptr = &result;

        OFFLOAD_OFFLOAD(ofld, "kmp_set_affinity_target",
                        0, 2, vars, NULL, 0, 0, 0);
    }
    return result;
}

int kmp_get_affinity_target(
    TARGET_TYPE target_type,
    int target_number,
    kmp_affinity_mask_target_t *mask
)
{
    int result = 1;

    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[2] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_inout;
        vars[0].size = sizeof(kmp_affinity_mask_target_t);
        vars[0].count = 1;
        vars[0].ptr = mask;

        vars[1].type.src = c_data;
        vars[1].type.dst = c_data;
        vars[1].direction.bits = c_parameter_out;
        vars[1].size = sizeof(int);
        vars[1].count = 1;
        vars[1].ptr = &result;

        OFFLOAD_OFFLOAD(ofld, "kmp_get_affinity_target",
                        0, 2, vars, NULL, 0, 0, 0);
    }
    return result;
}

int kmp_get_affinity_max_proc_target(
    TARGET_TYPE target_type,
    int target_number
)
{
    return omp_get_int_target(target_type, target_number,
                              "kmp_get_affinity_max_proc_target");
}

int kmp_set_affinity_mask_proc_target(
    TARGET_TYPE target_type,
    int target_number,
    int proc,
    kmp_affinity_mask_target_t *mask
)
{
    int result = 1;

    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[3] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_in;
        vars[0].size = sizeof(int);
        vars[0].count = 1;
        vars[0].ptr = &proc;

        vars[1].type.src = c_data;
        vars[1].type.dst = c_data;
        vars[1].direction.bits = c_parameter_inout;
        vars[1].size = sizeof(kmp_affinity_mask_target_t);
        vars[1].count = 1;
        vars[1].ptr = mask;

        vars[2].type.src = c_data;
        vars[2].type.dst = c_data;
        vars[2].direction.bits = c_parameter_out;
        vars[2].size = sizeof(int);
        vars[2].count = 1;
        vars[2].ptr = &result;

        OFFLOAD_OFFLOAD(ofld, "kmp_set_affinity_mask_proc_target",
                        0, 3, vars, NULL, 0, 0, 0);
    }
    return result;
}

int kmp_unset_affinity_mask_proc_target(
    TARGET_TYPE target_type,
    int target_number,
    int proc,
    kmp_affinity_mask_target_t *mask
)
{
    int result = 1;

    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[3] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_in;
        vars[0].size = sizeof(int);
        vars[0].count = 1;
        vars[0].ptr = &proc;

        vars[1].type.src = c_data;
        vars[1].type.dst = c_data;
        vars[1].direction.bits = c_parameter_inout;
        vars[1].size = sizeof(kmp_affinity_mask_target_t);
        vars[1].count = 1;
        vars[1].ptr = mask;

        vars[2].type.src = c_data;
        vars[2].type.dst = c_data;
        vars[2].direction.bits = c_parameter_out;
        vars[2].size = sizeof(int);
        vars[2].count = 1;
        vars[2].ptr = &result;

        OFFLOAD_OFFLOAD(ofld, "kmp_unset_affinity_mask_proc_target",
                        0, 3, vars, NULL, 0, 0, 0);
    }
    return result;
}

int kmp_get_affinity_mask_proc_target(
    TARGET_TYPE target_type,
    int target_number,
    int proc,
    kmp_affinity_mask_target_t *mask
)
{
    int result = 1;

    OFFLOAD ofld = OFFLOAD_TARGET_ACQUIRE(target_type, target_number, 0, NULL,
                                          __func__, 0);
    if (ofld != 0) {
        VarDesc vars[3] = {0};

        vars[0].type.src = c_data;
        vars[0].type.dst = c_data;
        vars[0].direction.bits = c_parameter_in;
        vars[0].size = sizeof(int);
        vars[0].count = 1;
        vars[0].ptr = &proc;

        vars[1].type.src = c_data;
        vars[1].type.dst = c_data;
        vars[1].direction.bits = c_parameter_in;
        vars[1].size = sizeof(kmp_affinity_mask_target_t);
        vars[1].count = 1;
        vars[1].ptr = mask;

        vars[2].type.src = c_data;
        vars[2].type.dst = c_data;
        vars[2].direction.bits = c_parameter_out;
        vars[2].size = sizeof(int);
        vars[2].count = 1;
        vars[2].ptr = &result;

        OFFLOAD_OFFLOAD(ofld, "kmp_get_affinity_mask_proc_target",
                        0, 3, vars, NULL, 0, 0, 0);
    }
    return result;
}
