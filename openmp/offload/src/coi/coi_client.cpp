//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


// The COI host interface

#include "coi_client.h"
#include "../offload_common.h"

namespace COI {

#define COI_VERSION1    "COI_1.0"
#define COI_VERSION2    "COI_2.0"

bool            is_available;
static void*    lib_handle;

// pointers to functions from COI library
COIRESULT (*EngineGetCount)(COI_ISA_TYPE, uint32_t*);
COIRESULT (*EngineGetHandle)(COI_ISA_TYPE, uint32_t, COIENGINE*);

COIRESULT (*ProcessCreateFromMemory)(COIENGINE, const char*, const void*,
                                     uint64_t, int, const char**, uint8_t,
                                     const char**, uint8_t, const char*,
                                     uint64_t, const char*, const char*,
                                     uint64_t, COIPROCESS*);
COIRESULT (*ProcessDestroy)(COIPROCESS, int32_t, uint8_t, int8_t*, uint32_t*);
COIRESULT (*ProcessGetFunctionHandles)(COIPROCESS, uint32_t, const char**,
                                       COIFUNCTION*);
COIRESULT (*ProcessLoadLibraryFromMemory)(COIPROCESS, const void*, uint64_t,
                                          const char*, const char*,
                                          const char*, uint64_t, uint32_t,
                                          COILIBRARY*);
COIRESULT (*ProcessRegisterLibraries)(uint32_t, const void**, const uint64_t*,
                                      const char**, const uint64_t*);

COIRESULT (*PipelineCreate)(COIPROCESS, COI_CPU_MASK, uint32_t, COIPIPELINE*);
COIRESULT (*PipelineDestroy)(COIPIPELINE);
COIRESULT (*PipelineRunFunction)(COIPIPELINE, COIFUNCTION, uint32_t,
                                 const COIBUFFER*, const COI_ACCESS_FLAGS*,
                                 uint32_t, const COIEVENT*, const void*,
                                 uint16_t, void*, uint16_t, COIEVENT*);

COIRESULT (*BufferCreate)(uint64_t, COI_BUFFER_TYPE, uint32_t, const void*,
                          uint32_t, const COIPROCESS*, COIBUFFER*);
COIRESULT (*BufferCreateFromMemory)(uint64_t, COI_BUFFER_TYPE, uint32_t,
                                    void*, uint32_t, const COIPROCESS*,
                                    COIBUFFER*);
COIRESULT (*BufferDestroy)(COIBUFFER);
COIRESULT (*BufferMap)(COIBUFFER, uint64_t, uint64_t, COI_MAP_TYPE, uint32_t,
                       const COIEVENT*, COIEVENT*, COIMAPINSTANCE*, void**);
COIRESULT (*BufferUnmap)(COIMAPINSTANCE, uint32_t, const COIEVENT*, COIEVENT*);
COIRESULT (*BufferWrite)(COIBUFFER, uint64_t, const void*, uint64_t,
                         COI_COPY_TYPE, uint32_t, const COIEVENT*, COIEVENT*);
COIRESULT (*BufferRead)(COIBUFFER, uint64_t, void*, uint64_t, COI_COPY_TYPE,
                        uint32_t, const COIEVENT*, COIEVENT*);
COIRESULT (*BufferCopy)(COIBUFFER, COIBUFFER, uint64_t, uint64_t, uint64_t,
                        COI_COPY_TYPE, uint32_t, const COIEVENT*, COIEVENT*);
COIRESULT (*BufferGetSinkAddress)(COIBUFFER, uint64_t*);
COIRESULT (*BufferSetState)(COIBUFFER, COIPROCESS, COI_BUFFER_STATE,
                            COI_BUFFER_MOVE_FLAG, uint32_t,
                            const   COIEVENT*, COIEVENT*);

COIRESULT (*EventWait)(uint16_t, const COIEVENT*, int32_t, uint8_t, uint32_t*,
                       uint32_t*);

uint64_t  (*PerfGetCycleFrequency)(void);

bool init(void)
{
#ifndef TARGET_WINNT
    const char *lib_name = "libcoi_host.so.0";
#else // TARGET_WINNT
    const char *lib_name = "coi_host.dll";
#endif // TARGET_WINNT

    OFFLOAD_DEBUG_TRACE(2, "Loading COI library %s ...\n", lib_name);
    lib_handle = DL_open(lib_name);
    if (lib_handle == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to load the library\n");
        return false;
    }

    EngineGetCount =
        (COIRESULT (*)(COI_ISA_TYPE, uint32_t*))
            DL_sym(lib_handle, "COIEngineGetCount", COI_VERSION1);
    if (EngineGetCount == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIEngineGetCount");
        fini();
        return false;
    }

    EngineGetHandle =
        (COIRESULT (*)(COI_ISA_TYPE, uint32_t, COIENGINE*))
            DL_sym(lib_handle, "COIEngineGetHandle", COI_VERSION1);
    if (EngineGetHandle == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIEngineGetHandle");
        fini();
        return false;
    }

    ProcessCreateFromMemory =
        (COIRESULT (*)(COIENGINE, const char*, const void*, uint64_t, int,
                       const char**, uint8_t, const char**, uint8_t,
                       const char*, uint64_t, const char*, const char*,
                       uint64_t, COIPROCESS*))
            DL_sym(lib_handle, "COIProcessCreateFromMemory", COI_VERSION1);
    if (ProcessCreateFromMemory == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIProcessCreateFromMemory");
        fini();
        return false;
    }

    ProcessDestroy =
        (COIRESULT (*)(COIPROCESS, int32_t, uint8_t, int8_t*,
                       uint32_t*))
            DL_sym(lib_handle, "COIProcessDestroy", COI_VERSION1);
    if (ProcessDestroy == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIProcessDestroy");
        fini();
        return false;
    }

    ProcessGetFunctionHandles =
        (COIRESULT (*)(COIPROCESS, uint32_t, const char**, COIFUNCTION*))
            DL_sym(lib_handle, "COIProcessGetFunctionHandles", COI_VERSION1);
    if (ProcessGetFunctionHandles == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIProcessGetFunctionHandles");
        fini();
        return false;
    }

    ProcessLoadLibraryFromMemory =
        (COIRESULT (*)(COIPROCESS, const void*, uint64_t, const char*,
                       const char*, const char*, uint64_t, uint32_t,
                       COILIBRARY*))
            DL_sym(lib_handle, "COIProcessLoadLibraryFromMemory", COI_VERSION2);
    if (ProcessLoadLibraryFromMemory == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIProcessLoadLibraryFromMemory");
        fini();
        return false;
    }

    ProcessRegisterLibraries =
        (COIRESULT (*)(uint32_t, const void**, const uint64_t*, const char**,
                       const uint64_t*))
            DL_sym(lib_handle, "COIProcessRegisterLibraries", COI_VERSION1);
    if (ProcessRegisterLibraries == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIProcessRegisterLibraries");
        fini();
        return false;
    }

    PipelineCreate =
        (COIRESULT (*)(COIPROCESS, COI_CPU_MASK, uint32_t, COIPIPELINE*))
            DL_sym(lib_handle, "COIPipelineCreate", COI_VERSION1);
    if (PipelineCreate == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIPipelineCreate");
        fini();
        return false;
    }

    PipelineDestroy =
        (COIRESULT (*)(COIPIPELINE))
            DL_sym(lib_handle, "COIPipelineDestroy", COI_VERSION1);
    if (PipelineDestroy == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIPipelineDestroy");
        fini();
        return false;
    }

    PipelineRunFunction =
        (COIRESULT (*)(COIPIPELINE, COIFUNCTION, uint32_t, const COIBUFFER*,
                       const COI_ACCESS_FLAGS*, uint32_t, const COIEVENT*,
                       const void*, uint16_t, void*, uint16_t, COIEVENT*))
            DL_sym(lib_handle, "COIPipelineRunFunction", COI_VERSION1);
    if (PipelineRunFunction == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIPipelineRunFunction");
        fini();
        return false;
    }

    BufferCreate =
        (COIRESULT (*)(uint64_t, COI_BUFFER_TYPE, uint32_t, const void*,
                       uint32_t, const COIPROCESS*, COIBUFFER*))
            DL_sym(lib_handle, "COIBufferCreate", COI_VERSION1);
    if (BufferCreate == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIBufferCreate");
        fini();
        return false;
    }

    BufferCreateFromMemory =
        (COIRESULT (*)(uint64_t, COI_BUFFER_TYPE, uint32_t, void*,
                       uint32_t, const COIPROCESS*, COIBUFFER*))
            DL_sym(lib_handle, "COIBufferCreateFromMemory", COI_VERSION1);
    if (BufferCreateFromMemory == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIBufferCreateFromMemory");
        fini();
        return false;
    }

    BufferDestroy =
        (COIRESULT (*)(COIBUFFER))
            DL_sym(lib_handle, "COIBufferDestroy", COI_VERSION1);
    if (BufferDestroy == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIBufferDestroy");
        fini();
        return false;
    }

    BufferMap =
        (COIRESULT (*)(COIBUFFER, uint64_t, uint64_t, COI_MAP_TYPE, uint32_t,
                       const COIEVENT*, COIEVENT*, COIMAPINSTANCE*,
                       void**))
            DL_sym(lib_handle, "COIBufferMap", COI_VERSION1);
    if (BufferMap == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIBufferMap");
        fini();
        return false;
    }

    BufferUnmap =
        (COIRESULT (*)(COIMAPINSTANCE, uint32_t, const COIEVENT*,
                       COIEVENT*))
            DL_sym(lib_handle, "COIBufferUnmap", COI_VERSION1);
    if (BufferUnmap == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIBufferUnmap");
        fini();
        return false;
    }

    BufferWrite =
        (COIRESULT (*)(COIBUFFER, uint64_t, const void*, uint64_t,
                       COI_COPY_TYPE, uint32_t, const COIEVENT*,
                       COIEVENT*))
            DL_sym(lib_handle, "COIBufferWrite", COI_VERSION1);
    if (BufferWrite == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIBufferWrite");
        fini();
        return false;
    }

    BufferRead =
        (COIRESULT (*)(COIBUFFER, uint64_t, void*, uint64_t,
                                     COI_COPY_TYPE, uint32_t,
                                     const COIEVENT*, COIEVENT*))
            DL_sym(lib_handle, "COIBufferRead", COI_VERSION1);
    if (BufferRead == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIBufferRead");
        fini();
        return false;
    }

    BufferCopy =
        (COIRESULT (*)(COIBUFFER, COIBUFFER, uint64_t, uint64_t, uint64_t,
                       COI_COPY_TYPE, uint32_t, const COIEVENT*,
                       COIEVENT*))
            DL_sym(lib_handle, "COIBufferCopy", COI_VERSION1);
    if (BufferCopy == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIBufferCopy");
        fini();
        return false;
    }

    BufferGetSinkAddress =
        (COIRESULT (*)(COIBUFFER, uint64_t*))
            DL_sym(lib_handle, "COIBufferGetSinkAddress", COI_VERSION1);
    if (BufferGetSinkAddress == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIBufferGetSinkAddress");
        fini();
        return false;
    }

    BufferSetState =
        (COIRESULT(*)(COIBUFFER, COIPROCESS, COI_BUFFER_STATE,
                      COI_BUFFER_MOVE_FLAG, uint32_t, const COIEVENT*,
                      COIEVENT*))
            DL_sym(lib_handle, "COIBufferSetState", COI_VERSION1);
    if (BufferSetState == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIBufferSetState");
        fini();
        return false;
    }

    EventWait =
        (COIRESULT (*)(uint16_t, const COIEVENT*, int32_t, uint8_t,
                       uint32_t*, uint32_t*))
            DL_sym(lib_handle, "COIEventWait", COI_VERSION1);
    if (EventWait == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIEventWait");
        fini();
        return false;
    }

    PerfGetCycleFrequency =
        (uint64_t (*)(void))
            DL_sym(lib_handle, "COIPerfGetCycleFrequency", COI_VERSION1);
    if (PerfGetCycleFrequency == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in COI library\n",
                            "COIPerfGetCycleFrequency");
        fini();
        return false;
    }

    is_available = true;

    return true;
}

void fini(void)
{
    is_available = false;

    if (lib_handle != 0) {
#ifndef TARGET_WINNT
        DL_close(lib_handle);
#endif // TARGET_WINNT
        lib_handle = 0;
    }
}

} // namespace COI
