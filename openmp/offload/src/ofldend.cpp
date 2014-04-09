//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#if HOST_LIBRARY
#include "offload_host.h"
#include "offload_myo_host.h"
#else
#include "offload_target.h"
#include "offload_myo_target.h"
#endif

#ifdef TARGET_WINNT
#define ALLOCATE(name) __declspec(allocate(name))
#else // TARGET_WINNT
#define ALLOCATE(name) __attribute__((section(name)))
#endif // TARGET_WINNT

// offload entry table
ALLOCATE(OFFLOAD_ENTRY_TABLE_SECTION_END)
#ifdef TARGET_WINNT
__declspec(align(sizeof(FuncTable::Entry)))
#endif // TARGET_WINNT
static FuncTable::Entry __offload_entry_table_end = { (const char*)-1 };

// offload function table
ALLOCATE(OFFLOAD_FUNC_TABLE_SECTION_END)
#ifdef TARGET_WINNT
__declspec(align(sizeof(FuncTable::Entry)))
#endif // TARGET_WINNT
static FuncTable::Entry __offload_func_table_end = { (const char*)-1 };

// data table
ALLOCATE(OFFLOAD_VAR_TABLE_SECTION_END)
#ifdef TARGET_WINNT
__declspec(align(sizeof(VarTable::Entry)))
#endif // TARGET_WINNT
static VarTable::Entry __offload_var_table_end = { (const char*)-1 };

#ifdef MYO_SUPPORT

// offload myo shared var section epilog
ALLOCATE(OFFLOAD_MYO_SHARED_TABLE_SECTION_END)
#ifdef TARGET_WINNT
__declspec(align(sizeof(SharedTableEntry)))
static SharedTableEntry __offload_myo_shared_table_end = { (const char*)-1, 0 };
#else // TARGET_WINNT
static SharedTableEntry __offload_myo_shared_table_end = { 0 };
#endif // TARGET_WINNT

#if HOST_LIBRARY
// offload myo shared var init section epilog
ALLOCATE(OFFLOAD_MYO_SHARED_INIT_TABLE_SECTION_END)
#ifdef TARGET_WINNT
__declspec(align(sizeof(InitTableEntry)))
static InitTableEntry __offload_myo_shared_init_table_end = { (const char*)-1, 0 };
#else // TARGET_WINNT
static InitTableEntry __offload_myo_shared_init_table_end = { 0 };
#endif // TARGET_WINNT
#endif // HOST_LIBRARY

// offload myo fptr section epilog
ALLOCATE(OFFLOAD_MYO_FPTR_TABLE_SECTION_END)
#ifdef TARGET_WINNT
__declspec(align(sizeof(FptrTableEntry)))
static FptrTableEntry __offload_myo_fptr_table_end = { (const char*)-1, 0, 0 };
#else // TARGET_WINNT
static FptrTableEntry __offload_myo_fptr_table_end = { 0 };
#endif // TARGET_WINNT

#endif // MYO_SUPPORT
