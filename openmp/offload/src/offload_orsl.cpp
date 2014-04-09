//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#include "offload_orsl.h"
#include <stdlib.h>
#include "offload_host.h"
#include "orsl-lite/include/orsl-lite.h"

namespace ORSL {

static bool            is_enabled = false;
static const ORSLTag   my_tag = "Offload";

void init()
{
    const char *env_var = getenv("OFFLOAD_ENABLE_ORSL");
    if (env_var != 0 && *env_var != '\0') {
        int64_t new_val;
        if (__offload_parse_int_string(env_var, new_val)) {
            is_enabled = new_val;
        }
        else {
            LIBOFFLOAD_ERROR(c_invalid_env_var_int_value,
                             "OFFLOAD_ENABLE_ORSL");
        }
    }

    if (is_enabled) {
        OFFLOAD_DEBUG_TRACE(2, "ORSL is enabled\n");
    }
    else {
        OFFLOAD_DEBUG_TRACE(2, "ORSL is disabled\n");
    }
}

bool reserve(int device)
{
    if (is_enabled) {
        int pnum = mic_engines[device].get_physical_index();
        ORSLBusySet bset;

        bset.type = BUSY_SET_FULL;
        if (ORSLReserve(1, &pnum, &bset, my_tag) != 0) {
            return false;
        }
    }
    return true;
}

bool try_reserve(int device)
{
    if (is_enabled) {
        int pnum = mic_engines[device].get_physical_index();
        ORSLBusySet bset;

        bset.type = BUSY_SET_FULL;
        if (ORSLTryReserve(1, &pnum, &bset, my_tag) != 0) {
            return false;
        }
    }
    return true;
}

void release(int device)
{
    if (is_enabled) {
        int pnum = mic_engines[device].get_physical_index();
        ORSLBusySet bset;

        bset.type = BUSY_SET_FULL;
        if (ORSLRelease(1, &pnum, &bset, my_tag) != 0) {
            // should never get here
        }
    }
}

} // namespace ORSL
