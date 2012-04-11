//===-- head_find.c ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file compiles into a dylib and can be used on darwin to find data that
// is contained in active malloc blocks. To use this make the project, then
// load the shared library in a debug session while you are stopped:
//
// (lldb) process load /path/to/libheap.dylib
//
// Now you can use the "find_pointer_in_heap" and "find_cstring_in_heap" 
// functions in the expression parser.
//
// This will grep everything in all active allocation blocks and print and 
// malloc blocks that contain the pointer 0x112233000000:
//
// (lldb) expression find_pointer_in_heap (0x112233000000)
//
// This will grep everything in all active allocation blocks and print and 
// malloc blocks that contain the C string "hello" (as a substring, no
// NULL termination included):
//
// (lldb) expression find_cstring_in_heap ("hello")
//
// The results will be printed to the STDOUT of the inferior program. The 
// return value of the "find_pointer_in_heap" function is the number of 
// pointer references that were found. A quick example shows
//
// (lldb) expr find_pointer_in_heap(0x0000000104000410)
// (uint32_t) $5 = 0x00000002
// 0x104000740: 0x0000000104000410 found in malloc block 0x104000730 + 16 (malloc_size = 48)
// 0x100820060: 0x0000000104000410 found in malloc block 0x100820000 + 96 (malloc_size = 4096)
//
// From the above output we see that 0x104000410 was found in the malloc block
// at 0x104000730 and 0x100820000. If we want to see what these blocks are, we
// can display the memory for this block using the "address" ("A" for short) 
// format. The address format shows pointers, and if those pointers point to
// objects that have symbols or know data contents, it will display information
// about the pointers:
//
// (lldb) memory read --format address --count 1 0x104000730 
// 0x104000730: 0x0000000100002460 (void *)0x0000000100002488: MyString
// 
// We can see that the first block is a "MyString" object that contains our
// pointer value at offset 16.
//
// Looking at the next pointers, are a bit more tricky:
// (lldb) memory read -fA 0x100820000 -c1
// 0x100820000: 0x4f545541a1a1a1a1
// (lldb) memory read 0x100820000
// 0x100820000: a1 a1 a1 a1 41 55 54 4f 52 45 4c 45 41 53 45 21  ....AUTORELEASE!
// 0x100820010: 78 00 82 00 01 00 00 00 60 f9 e8 75 ff 7f 00 00  x.......`..u....
// 
// This is an objective C auto release pool object that contains our pointer.
// C++ classes will show up if they are virtual as something like:
// (lldb) memory read --format address --count 1 0x104008000
// 0x104008000: 0x109008000 vtable for lldb_private::Process
//
// This is a clue that the 0x104008000 is a "lldb_private::Process *".
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <ctype.h>
#include <mach/mach.h>
#include <malloc/malloc.h>
#include <stack_logging.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

struct range_callback_info_t;

typedef void range_callback_t (task_t task, void *baton, unsigned type, uint64_t ptr_addr, uint64_t ptr_size);
typedef void zone_callback_t (void *info, const malloc_zone_t *zone);

struct range_callback_info_t
{
    zone_callback_t *zone_callback;
    range_callback_t *range_callback;
    void *baton;
};

enum data_type_t
{
    eDataTypeBytes,
    eDataTypeCStr,
    eDataTypeInteger
};

struct range_contains_data_callback_info_t
{
    const uint8_t *data;
    const size_t data_len;
    const uint32_t align;
    const data_type_t data_type;
    uint32_t match_count;
};

struct malloc_match
{
    void *addr;
    intptr_t size;
    intptr_t offset;
};

std::vector<malloc_match> g_matches;

static kern_return_t
task_peek (task_t task, vm_address_t remote_address, vm_size_t size, void **local_memory)
{
    *local_memory = (void*) remote_address;
    return KERN_SUCCESS;
}


static const void
foreach_zone_in_this_process (range_callback_info_t *info)
{
    //printf ("foreach_zone_in_this_process ( info->zone_callback = %p, info->range_callback = %p, info->baton = %p)", info->zone_callback, info->range_callback, info->baton);
    if (info == NULL || info->zone_callback == NULL)
        return;

    vm_address_t *zones = NULL;
    unsigned int num_zones = 0;
        
    kern_return_t err = malloc_get_all_zones (0, task_peek, &zones, &num_zones);
    if (KERN_SUCCESS == err)
    {
        for (unsigned int i=0; i<num_zones; ++i)
        {
            info->zone_callback (info, (const malloc_zone_t *)zones[i]);
        }
    }
}

static void
range_callback (task_t task, void *baton, unsigned type, uint64_t ptr_addr, uint64_t ptr_size)
{
    printf ("task = 0x%4.4x: baton = %p, type = %u, ptr_addr = 0x%llx + 0x%llu\n", task, baton, type, ptr_addr, ptr_size);
}

static void 
ranges_callback (task_t task, void *baton, unsigned type, vm_range_t *ptrs, unsigned count) 
{
    range_callback_info_t *info = (range_callback_info_t *)baton;
    while(count--) {
        info->range_callback (task, info->baton, type, ptrs->address, ptrs->size);
        ptrs++;
    }
}

static void
enumerate_range_in_zone (void *baton, const malloc_zone_t *zone)
{
    range_callback_info_t *info = (range_callback_info_t *)baton;

    if (zone && zone->introspect)
        zone->introspect->enumerator (mach_task_self(), 
                                      info, 
                                      MALLOC_PTR_IN_USE_RANGE_TYPE, 
                                      (vm_address_t)zone, 
                                      task_peek, 
                                      ranges_callback);    
}

const void
foreach_range_in_this_process (range_callback_t *callback, void *baton)
{
    range_callback_info_t info = { enumerate_range_in_zone, callback ? callback : range_callback, baton };
    foreach_zone_in_this_process (&info);
}


static void
range_contains_ptr_callback (task_t task, void *baton, unsigned type, uint64_t ptr_addr, uint64_t ptr_size)
{
    uint8_t *data = NULL;
    range_contains_data_callback_info_t *data_info = (range_contains_data_callback_info_t *)baton;
    if (data_info->data_len <= 0)
    {
        printf ("error: invalid data size: %zu\n", data_info->data_len);
    }
    else if (data_info->data_len > ptr_size)
    {
        // This block is too short to contain the data we are looking for...
        return;
    }
    else if (task_peek (task, ptr_addr, ptr_size, (void **)&data) == KERN_SUCCESS)
    {
        assert (data);
        const uint64_t end_addr = ptr_addr + ptr_size;
        for (uint64_t addr = ptr_addr; 
             addr < end_addr && ((end_addr - addr) >= data_info->data_len);
             addr += data_info->align, data += data_info->align)
        {
            if (memcmp (data_info->data, data, data_info->data_len) == 0)
            {
                ++data_info->match_count;
                malloc_match match = { (void *)ptr_addr, ptr_size, addr - ptr_addr };
                g_matches.push_back(match);
                // printf ("0x%llx: ", addr);
                // uint32_t i;
                // switch (data_info->data_type)
                // {
                // case eDataTypeInteger:
                //     {
                //         // NOTE: little endian specific, but all darwin platforms are little endian now..
                //         for (i=0; i<data_info->data_len; ++i)
                //             printf (i ? "%2.2x" : "0x%2.2x", data[data_info->data_len - (i + 1)]);
                //     }
                //     break;
                // case eDataTypeBytes:
                //     {
                //         for (i=0; i<data_info->data_len; ++i)
                //             printf (" %2.2x", data[i]);
                //     }
                //     break;
                // case eDataTypeCStr:
                //     {
                //         putchar ('"');
                //         for (i=0; i<data_info->data_len; ++i)
                //         {
                //             if (isprint (data[i]))
                //                 putchar (data[i]);
                //             else
                //                 printf ("\\x%2.2x", data[i]);
                //         }
                //         putchar ('"');
                //     }
                //     break;
                //     
                // }
                // printf (" found in malloc block 0x%llx + %llu (malloc_size = %llu)\n", ptr_addr, addr - ptr_addr, ptr_size);
            }
        }
    }
    else
    {
        printf ("0x%llx: error: couldn't read %llu bytes\n", ptr_addr, ptr_size);
    }   
}

malloc_match *
find_pointer_in_heap (intptr_t addr)
{
    g_matches.clear();
    range_contains_data_callback_info_t data_info = { (uint8_t *)&addr, sizeof(addr), sizeof(addr), eDataTypeInteger, 0};
    range_callback_info_t info = { enumerate_range_in_zone, range_contains_ptr_callback, &data_info };
    foreach_zone_in_this_process (&info);
    if (g_matches.empty())
        return NULL;
    malloc_match match = { NULL, 0, 0 };
    g_matches.push_back(match);
    return g_matches.data();
}


malloc_match *
find_cstring_in_heap (const char *s)
{
    if (s && s[0])
    {
        g_matches.clear();
        range_contains_data_callback_info_t data_info = { (uint8_t *)s, strlen(s), 1, eDataTypeCStr, 0};
        range_callback_info_t info = { enumerate_range_in_zone, range_contains_ptr_callback, &data_info };
        foreach_zone_in_this_process (&info);
        if (g_matches.empty())
            return NULL;
        malloc_match match = { NULL, 0, 0 };
        g_matches.push_back(match);
        return g_matches.data();
    }
    else
    {
        printf ("error: invalid argument (empty cstring)\n");
    }
    return 0;
}
