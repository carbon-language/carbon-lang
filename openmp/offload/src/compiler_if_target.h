//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


/*! \file
    \brief The interface between compiler-generated target code and runtime library
*/

#ifndef COMPILER_IF_TARGET_H_INCLUDED
#define COMPILER_IF_TARGET_H_INCLUDED

#include "offload_target.h"

#define OFFLOAD_TARGET_ENTER            OFFLOAD_PREFIX(target_enter)
#define OFFLOAD_TARGET_LEAVE            OFFLOAD_PREFIX(target_leave)
#define OFFLOAD_TARGET_MAIN             OFFLOAD_PREFIX(target_main)

/*! \fn OFFLOAD_TARGET_ENTER
    \brief Fill in variable addresses using VarDesc array.
    \brief Then call back the runtime library to fetch data.
    \param ofld         Offload descriptor created by runtime.
    \param var_desc_num Number of variable descriptors.
    \param var_desc     Pointer to VarDesc array.
    \param var_desc2    Pointer to VarDesc2 array.
*/
extern "C" void OFFLOAD_TARGET_ENTER(
    OFFLOAD ofld,
    int var_desc_num,
    VarDesc *var_desc,
    VarDesc2 *var_desc2
);

/*! \fn OFFLOAD_TARGET_LEAVE
    \brief Call back the runtime library to gather outputs using VarDesc array.
    \param ofld Offload descriptor created by OFFLOAD_TARGET_ACQUIRE.
*/
extern "C" void OFFLOAD_TARGET_LEAVE(
    OFFLOAD ofld
);

// Entry point for the target application.
extern "C" void OFFLOAD_TARGET_MAIN(void);

#endif // COMPILER_IF_TARGET_H_INCLUDED
