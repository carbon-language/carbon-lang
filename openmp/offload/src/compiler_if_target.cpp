//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#include "compiler_if_target.h"

extern "C" void OFFLOAD_TARGET_ENTER(
    OFFLOAD ofld,
    int vars_total,
    VarDesc *vars,
    VarDesc2 *vars2
)
{
    OFFLOAD_DEBUG_TRACE(3, "%s(%p, %d, %p, %p)\n", __func__, ofld,
                        vars_total, vars, vars2);
    ofld->merge_var_descs(vars, vars2, vars_total);
    ofld->scatter_copyin_data();
}

extern "C" void OFFLOAD_TARGET_LEAVE(
    OFFLOAD ofld
)
{
    OFFLOAD_DEBUG_TRACE(3, "%s(%p)\n", __func__, ofld);
    ofld->gather_copyout_data();
}

extern "C" void OFFLOAD_TARGET_MAIN(void)
{
    // initialize target part
    __offload_target_init();

    // pass control to COI
    PipelineStartExecutingRunFunctions();
    ProcessWaitForShutdown();

    OFFLOAD_DEBUG_TRACE(2, "Exiting main...\n");
}
