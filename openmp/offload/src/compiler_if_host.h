//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


/*! \file
    \brief The interface between compiler-generated host code and runtime library
*/

#ifndef COMPILER_IF_HOST_H_INCLUDED
#define COMPILER_IF_HOST_H_INCLUDED

#include "offload_host.h"

#define OFFLOAD_TARGET_ACQUIRE          OFFLOAD_PREFIX(target_acquire)
#define OFFLOAD_TARGET_ACQUIRE1         OFFLOAD_PREFIX(target_acquire1)
#define OFFLOAD_OFFLOAD                 OFFLOAD_PREFIX(offload)
#define OFFLOAD_OFFLOAD1                OFFLOAD_PREFIX(offload1)
#define OFFLOAD_OFFLOAD2                OFFLOAD_PREFIX(offload2)
#define OFFLOAD_CALL_COUNT              OFFLOAD_PREFIX(offload_call_count)


/*! \fn OFFLOAD_TARGET_ACQUIRE
    \brief Attempt to acquire the target.
    \param target_type   The type of target.
    \param target_number The device number.
    \param is_optional   Whether CPU fall-back is allowed.
    \param status        Address of variable to hold offload status.
    \param file          Filename in which this offload occurred.
    \param line          Line number in the file where this offload occurred.
*/
extern "C" OFFLOAD OFFLOAD_TARGET_ACQUIRE(
    TARGET_TYPE      target_type,
    int              target_number,
    int              is_optional,
    _Offload_status* status,
    const char*      file,
    uint64_t         line
);

/*! \fn OFFLOAD_TARGET_ACQUIRE1
    \brief Acquire the target for offload (OpenMP).
    \param device_number Device number or null if not specified.
    \param file          Filename in which this offload occurred
    \param line          Line number in the file where this offload occurred.
*/
extern "C" OFFLOAD OFFLOAD_TARGET_ACQUIRE1(
    const int*      device_number,
    const char*     file,
    uint64_t        line
);

/*! \fn OFFLOAD_OFFLOAD1
    \brief Run function on target using interface for old data persistence.
    \param o Offload descriptor created by OFFLOAD_TARGET_ACQUIRE.
    \param name Name of offload entry point.
    \param is_empty If no code to execute (e.g. offload_transfer)
    \param num_vars Number of variable descriptors.
    \param vars Pointer to VarDesc array.
    \param vars2 Pointer to VarDesc2 array.
    \param num_waits Number of "wait" values.
    \param waits Pointer to array of wait values.
    \param signal Pointer to signal value or NULL.
*/
extern "C" int OFFLOAD_OFFLOAD1(
    OFFLOAD o,
    const char *name,
    int is_empty,
    int num_vars,
    VarDesc *vars,
    VarDesc2 *vars2,
    int num_waits,
    const void** waits,
    const void** signal
);

/*! \fn OFFLOAD_OFFLOAD2
    \brief Run function on target using interface for new data persistence.
    \param o Offload descriptor created by OFFLOAD_TARGET_ACQUIRE.
    \param name Name of offload entry point.
    \param is_empty If no code to execute (e.g. offload_transfer)
    \param num_vars Number of variable descriptors.
    \param vars Pointer to VarDesc array.
    \param vars2 Pointer to VarDesc2 array.
    \param num_waits Number of "wait" values.
    \param waits Pointer to array of wait values.
    \param signal Pointer to signal value or NULL.
    \param entry_id A signature for the function doing the offload.
    \param stack_addr The stack frame address of the function doing offload.
*/
extern "C" int OFFLOAD_OFFLOAD2(
    OFFLOAD o,
    const char *name,
    int is_empty,
    int num_vars,
    VarDesc *vars,
    VarDesc2 *vars2,
    int num_waits,
    const void** waits,
    const void** signal,
    int entry_id,
    const void *stack_addr
);

// Run function on target (obsolete).
// @param o    OFFLOAD object
// @param name function name
extern "C" int OFFLOAD_OFFLOAD(
    OFFLOAD o,
    const char *name,
    int is_empty,
    int num_vars,
    VarDesc *vars,
    VarDesc2 *vars2,
    int num_waits,
    const void** waits,
    const void* signal,
    int entry_id = 0,
    const void *stack_addr = NULL
);

// Global counter on host.
// This variable is used if P2OPT_offload_do_data_persistence == 2.
// The variable used to identify offload constructs contained in one procedure.
// Call to OFFLOAD_CALL_COUNT() is inserted at HOST on entry of the routine.
extern "C" int  OFFLOAD_CALL_COUNT();

#endif // COMPILER_IF_HOST_H_INCLUDED
