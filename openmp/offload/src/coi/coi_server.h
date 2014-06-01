//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


//The interface between offload library and the COI API on the target.

#ifndef COI_SERVER_H_INCLUDED
#define COI_SERVER_H_INCLUDED

#include <common/COIEngine_common.h>
#include <common/COIPerf_common.h>
#include <sink/COIProcess_sink.h>
#include <sink/COIPipeline_sink.h>
#include <sink/COIBuffer_sink.h>
#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../liboffload_error_codes.h"

// wrappers for COI API
#define PipelineStartExecutingRunFunctions() \
    { \
        COIRESULT res = COIPipelineStartExecutingRunFunctions(); \
        if (res != COI_SUCCESS) { \
            LIBOFFLOAD_ERROR(c_pipeline_start_run_funcs, mic_index, res); \
            exit(1); \
        } \
    }

#define ProcessWaitForShutdown() \
    { \
        COIRESULT res = COIProcessWaitForShutdown(); \
        if (res != COI_SUCCESS) { \
            LIBOFFLOAD_ERROR(c_process_wait_shutdown, mic_index, res); \
            exit(1); \
        } \
    }

#define BufferAddRef(buf) \
    { \
        COIRESULT res = COIBufferAddRef(buf); \
        if (res != COI_SUCCESS) { \
            LIBOFFLOAD_ERROR(c_buf_add_ref, mic_index, res); \
            exit(1); \
        } \
    }

#define BufferReleaseRef(buf) \
    { \
        COIRESULT res = COIBufferReleaseRef(buf); \
        if (res != COI_SUCCESS) { \
            LIBOFFLOAD_ERROR(c_buf_release_ref, mic_index, res); \
            exit(1); \
        } \
    }

#define EngineGetIndex(index) \
    { \
        COI_ISA_TYPE isa_type; \
        COIRESULT res = COIEngineGetIndex(&isa_type, index); \
        if (res != COI_SUCCESS) { \
            LIBOFFLOAD_ERROR(c_get_engine_index, mic_index, res); \
            exit(1); \
        } \
    }

#endif // COI_SERVER_H_INCLUDED
