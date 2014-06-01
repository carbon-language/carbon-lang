//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


// The interface between offload library and the COI API on the host

#ifndef COI_CLIENT_H_INCLUDED
#define COI_CLIENT_H_INCLUDED

#include <common/COIPerf_common.h>
#include <source/COIEngine_source.h>
#include <source/COIProcess_source.h>
#include <source/COIPipeline_source.h>
#include <source/COIBuffer_source.h>
#include <source/COIEvent_source.h>

#include <string.h>

#include "../liboffload_error_codes.h"
#include "../offload_util.h"

#define MIC_ENGINES_MAX     128

#if MIC_ENGINES_MAX < COI_MAX_ISA_MIC_DEVICES
#error MIC_ENGINES_MAX need to be increased
#endif

// COI library interface
namespace COI {

extern bool init(void);
extern void fini(void);

extern bool is_available;

// pointers to functions from COI library
extern COIRESULT (*EngineGetCount)(COI_ISA_TYPE, uint32_t*);
extern COIRESULT (*EngineGetHandle)(COI_ISA_TYPE, uint32_t, COIENGINE*);

extern COIRESULT (*ProcessCreateFromMemory)(COIENGINE, const char*,
                                           const void*, uint64_t, int,
                                           const char**, uint8_t,
                                           const char**, uint8_t,
                                           const char*, uint64_t,
                                           const char*,
                                           const char*, uint64_t,
                                           COIPROCESS*);
extern COIRESULT (*ProcessDestroy)(COIPROCESS, int32_t, uint8_t,
                                  int8_t*, uint32_t*);
extern COIRESULT (*ProcessGetFunctionHandles)(COIPROCESS, uint32_t,
                                             const char**,
                                             COIFUNCTION*);
extern COIRESULT (*ProcessLoadLibraryFromMemory)(COIPROCESS,
                                                const void*,
                                                uint64_t,
                                                const char*,
                                                const char*,
                                                const char*,
                                                uint64_t,
                                                uint32_t,
                                                COILIBRARY*);
extern COIRESULT (*ProcessRegisterLibraries)(uint32_t,
                                            const void**,
                                            const uint64_t*,
                                            const char**,
                                            const uint64_t*);

extern COIRESULT (*PipelineCreate)(COIPROCESS, COI_CPU_MASK, uint32_t,
                                  COIPIPELINE*);
extern COIRESULT (*PipelineDestroy)(COIPIPELINE);
extern COIRESULT (*PipelineRunFunction)(COIPIPELINE, COIFUNCTION,
                                       uint32_t, const COIBUFFER*,
                                       const COI_ACCESS_FLAGS*,
                                       uint32_t, const COIEVENT*,
                                       const void*, uint16_t, void*,
                                       uint16_t, COIEVENT*);

extern COIRESULT (*BufferCreate)(uint64_t, COI_BUFFER_TYPE, uint32_t,
                                const void*, uint32_t,
                                const COIPROCESS*, COIBUFFER*);
extern COIRESULT (*BufferCreateFromMemory)(uint64_t, COI_BUFFER_TYPE,
                                          uint32_t, void*,
                                          uint32_t, const COIPROCESS*,
                                          COIBUFFER*);
extern COIRESULT (*BufferDestroy)(COIBUFFER);
extern COIRESULT (*BufferMap)(COIBUFFER, uint64_t, uint64_t,
                             COI_MAP_TYPE, uint32_t, const COIEVENT*,
                             COIEVENT*, COIMAPINSTANCE*, void**);
extern COIRESULT (*BufferUnmap)(COIMAPINSTANCE, uint32_t,
                               const COIEVENT*, COIEVENT*);
extern COIRESULT (*BufferWrite)(COIBUFFER, uint64_t, const void*,
                               uint64_t, COI_COPY_TYPE, uint32_t,
                               const COIEVENT*, COIEVENT*);
extern COIRESULT (*BufferRead)(COIBUFFER, uint64_t, void*, uint64_t,
                              COI_COPY_TYPE, uint32_t,
                              const COIEVENT*, COIEVENT*);
extern COIRESULT (*BufferCopy)(COIBUFFER, COIBUFFER, uint64_t, uint64_t,
                              uint64_t, COI_COPY_TYPE, uint32_t,
                              const COIEVENT*, COIEVENT*);
extern COIRESULT (*BufferGetSinkAddress)(COIBUFFER, uint64_t*);
extern COIRESULT (*BufferSetState)(COIBUFFER, COIPROCESS, COI_BUFFER_STATE,
                                   COI_BUFFER_MOVE_FLAG, uint32_t,
                                   const   COIEVENT*, COIEVENT*);

extern COIRESULT (*EventWait)(uint16_t, const COIEVENT*, int32_t,
                           uint8_t, uint32_t*, uint32_t*);

extern uint64_t  (*PerfGetCycleFrequency)(void);

} // namespace COI

#endif // COI_CLIENT_H_INCLUDED
