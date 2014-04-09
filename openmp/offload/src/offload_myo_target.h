//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#ifndef OFFLOAD_MYO_TARGET_H_INCLUDED
#define OFFLOAD_MYO_TARGET_H_INCLUDED

#include <myotypes.h>
#include <myoimpl.h>
#include <myo.h>
#include "offload.h"

typedef MyoiSharedVarEntry          SharedTableEntry;
typedef MyoiTargetSharedFptrEntry   FptrTableEntry;

#ifdef TARGET_WINNT
#define OFFLOAD_MYO_SHARED_TABLE_SECTION_START          ".MyoSharedTable$a"
#define OFFLOAD_MYO_SHARED_TABLE_SECTION_END            ".MyoSharedTable$z"

#define OFFLOAD_MYO_FPTR_TABLE_SECTION_START            ".MyoFptrTable$a"
#define OFFLOAD_MYO_FPTR_TABLE_SECTION_END              ".MyoFptrTable$z"
#else  // TARGET_WINNT
#define OFFLOAD_MYO_SHARED_TABLE_SECTION_START          ".MyoSharedTable."
#define OFFLOAD_MYO_SHARED_TABLE_SECTION_END            ".MyoSharedTable."

#define OFFLOAD_MYO_FPTR_TABLE_SECTION_START            ".MyoFptrTable."
#define OFFLOAD_MYO_FPTR_TABLE_SECTION_END              ".MyoFptrTable."
#endif // TARGET_WINNT

#pragma section(OFFLOAD_MYO_SHARED_TABLE_SECTION_START, read, write)
#pragma section(OFFLOAD_MYO_SHARED_TABLE_SECTION_END, read, write)

#pragma section(OFFLOAD_MYO_FPTR_TABLE_SECTION_START, read, write)
#pragma section(OFFLOAD_MYO_FPTR_TABLE_SECTION_END, read, write)

extern "C" void __offload_myoRegisterTables(
    SharedTableEntry *shared_table,
    FptrTableEntry *fptr_table
);

extern "C" void __offload_myoAcquire(void);
extern "C" void __offload_myoRelease(void);

// temporary workaround for blocking behavior for myoiLibInit/Fini calls
extern "C" void __offload_myoLibInit();
extern "C" void __offload_myoLibFini();

#endif // OFFLOAD_MYO_TARGET_H_INCLUDED
