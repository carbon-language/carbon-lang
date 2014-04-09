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
#include "compiler_if_target.h"
#include "offload_target.h"
#include "offload_myo_target.h"
#endif

#ifdef TARGET_WINNT
#define ALLOCATE(name) __declspec(allocate(name))
#define DLL_LOCAL
#else // TARGET_WINNT
#define ALLOCATE(name) __attribute__((section(name)))
#define DLL_LOCAL  __attribute__((visibility("hidden")))
#endif // TARGET_WINNT

#if HOST_LIBRARY
// the host program/shared library should always have __offload_target_image
// symbol defined. This symbol specifies the beginning of the target program
// image.
extern "C" DLL_LOCAL const void* __offload_target_image;
#else // HOST_LIBRARY
// Define a weak main which would be used on target side in case usere's
// source file containing main does not have offload code.
#pragma weak main
int main(void)
{
    OFFLOAD_TARGET_MAIN();
    return 0;
}

#pragma weak MAIN__
extern "C" int MAIN__(void)
{
    OFFLOAD_TARGET_MAIN();
    return 0;
}
#endif // HOST_LIBRARY

// offload section prolog
ALLOCATE(OFFLOAD_ENTRY_TABLE_SECTION_START)
#ifdef TARGET_WINNT
__declspec(align(sizeof(FuncTable::Entry)))
#endif // TARGET_WINNT
static FuncTable::Entry __offload_entry_table_start = { 0 };

// list element for the current module
static FuncList::Node __offload_entry_node = {
    { &__offload_entry_table_start + 1, -1 },
    0, 0
};

// offload fp section prolog
ALLOCATE(OFFLOAD_FUNC_TABLE_SECTION_START)
#ifdef TARGET_WINNT
__declspec(align(sizeof(FuncTable::Entry)))
#endif // TARGET_WINNT
static FuncTable::Entry __offload_func_table_start = { 0 };

// list element for the current module
static FuncList::Node __offload_func_node = {
    { &__offload_func_table_start + 1, -1 },
    0, 0
};

// offload fp section prolog
ALLOCATE(OFFLOAD_VAR_TABLE_SECTION_START)
#ifdef TARGET_WINNT
__declspec(align(sizeof(VarTable::Entry)))
#endif // TARGET_WINNT
static VarTable::Entry __offload_var_table_start = { 0 };

// list element for the current module
static VarList::Node __offload_var_node = {
    { &__offload_var_table_start + 1 },
    0, 0
};

#ifdef MYO_SUPPORT

// offload myo shared var section prolog
ALLOCATE(OFFLOAD_MYO_SHARED_TABLE_SECTION_START)
#ifdef TARGET_WINNT
__declspec(align(sizeof(SharedTableEntry)))
#endif // TARGET_WINNT
static SharedTableEntry __offload_myo_shared_table_start = { 0 };

#if HOST_LIBRARY
// offload myo shared var init section prolog
ALLOCATE(OFFLOAD_MYO_SHARED_INIT_TABLE_SECTION_START)
#ifdef TARGET_WINNT
__declspec(align(sizeof(InitTableEntry)))
#endif // TARGET_WINNT
static InitTableEntry __offload_myo_shared_init_table_start = { 0 };
#endif

// offload myo fptr section prolog
ALLOCATE(OFFLOAD_MYO_FPTR_TABLE_SECTION_START)
#ifdef TARGET_WINNT
__declspec(align(sizeof(FptrTableEntry)))
#endif // TARGET_WINNT
static FptrTableEntry __offload_myo_fptr_table_start = { 0 };

#endif // MYO_SUPPORT

// init/fini code which adds/removes local lookup data to/from the global list

static void offload_fini();

#ifndef TARGET_WINNT
static void offload_init() __attribute__((constructor(101)));
#else // TARGET_WINNT
static void offload_init();

// Place offload initialization before user constructors
ALLOCATE(OFFLOAD_CRTINIT_SECTION_START)
static void (*addressof_offload_init)() = offload_init;
#endif // TARGET_WINNT

static void offload_init()
{
    // register offload tables
    __offload_register_tables(&__offload_entry_node,
                              &__offload_func_node,
                              &__offload_var_node);

#if HOST_LIBRARY
    __offload_register_image(&__offload_target_image);
    atexit(offload_fini);
#endif // HOST_LIBRARY

#ifdef MYO_SUPPORT
    __offload_myoRegisterTables(
#if HOST_LIBRARY
        &__offload_myo_shared_init_table_start + 1,
#endif // HOST_LIBRARY
        &__offload_myo_shared_table_start + 1,
        &__offload_myo_fptr_table_start + 1
    );
#endif // MYO_SUPPORT
}

static void offload_fini()
{
#if HOST_LIBRARY
    __offload_unregister_image(&__offload_target_image);
#endif // HOST_LIBRARY

    // unregister offload tables
    __offload_unregister_tables(&__offload_entry_node,
                                &__offload_func_node,
                                &__offload_var_node);
}
