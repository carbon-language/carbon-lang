//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#ifndef OFFLOAD_MYO_HOST_H_INCLUDED
#define OFFLOAD_MYO_HOST_H_INCLUDED

#include <myotypes.h>
#include <myoimpl.h>
#include <myo.h>
#include "offload.h"

typedef MyoiSharedVarEntry      SharedTableEntry;
//typedef MyoiHostSharedFptrEntry FptrTableEntry;
typedef struct {
    //! Function Name
    const char *funcName;
    //! Function Address
    void *funcAddr;
    //! Local Thunk Address
    void *localThunkAddr;
#ifdef TARGET_WINNT
    // Dummy to pad up to 32 bytes
    void *dummy;
#endif // TARGET_WINNT
} FptrTableEntry;

struct InitTableEntry {
#ifdef TARGET_WINNT
    // Dummy to pad up to 16 bytes
    // Function Name
    const char *funcName;
#endif // TARGET_WINNT
    void (*func)(void);
};

#ifdef TARGET_WINNT
#define OFFLOAD_MYO_SHARED_TABLE_SECTION_START          ".MyoSharedTable$a"
#define OFFLOAD_MYO_SHARED_TABLE_SECTION_END            ".MyoSharedTable$z"

#define OFFLOAD_MYO_SHARED_INIT_TABLE_SECTION_START     ".MyoSharedInitTable$a"
#define OFFLOAD_MYO_SHARED_INIT_TABLE_SECTION_END       ".MyoSharedInitTable$z"

#define OFFLOAD_MYO_FPTR_TABLE_SECTION_START            ".MyoFptrTable$a"
#define OFFLOAD_MYO_FPTR_TABLE_SECTION_END              ".MyoFptrTable$z"
#else  // TARGET_WINNT
#define OFFLOAD_MYO_SHARED_TABLE_SECTION_START          ".MyoSharedTable."
#define OFFLOAD_MYO_SHARED_TABLE_SECTION_END            ".MyoSharedTable."

#define OFFLOAD_MYO_SHARED_INIT_TABLE_SECTION_START     ".MyoSharedInitTable."
#define OFFLOAD_MYO_SHARED_INIT_TABLE_SECTION_END       ".MyoSharedInitTable."

#define OFFLOAD_MYO_FPTR_TABLE_SECTION_START            ".MyoFptrTable."
#define OFFLOAD_MYO_FPTR_TABLE_SECTION_END              ".MyoFptrTable."
#endif // TARGET_WINNT

#pragma section(OFFLOAD_MYO_SHARED_TABLE_SECTION_START, read, write)
#pragma section(OFFLOAD_MYO_SHARED_TABLE_SECTION_END, read, write)

#pragma section(OFFLOAD_MYO_SHARED_INIT_TABLE_SECTION_START, read, write)
#pragma section(OFFLOAD_MYO_SHARED_INIT_TABLE_SECTION_END, read, write)

#pragma section(OFFLOAD_MYO_FPTR_TABLE_SECTION_START, read, write)
#pragma section(OFFLOAD_MYO_FPTR_TABLE_SECTION_END, read, write)

extern "C" void __offload_myoRegisterTables(
    InitTableEntry *init_table,
    SharedTableEntry *shared_table,
    FptrTableEntry *fptr_table
);

extern void __offload_myoFini(void);

#endif // OFFLOAD_MYO_HOST_H_INCLUDED
