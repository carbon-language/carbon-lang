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
#include <stdio.h>
#include <stdlib.h>
#include <vector>

//----------------------------------------------------------------------
// Redefine private types from "/usr/local/include/stack_logging.h"
//----------------------------------------------------------------------
typedef struct {
	uint32_t		type_flags;
	uint64_t		stack_identifier;
	uint64_t		argument;
	mach_vm_address_t	address;
} mach_stack_logging_record_t;

//----------------------------------------------------------------------
// Redefine private defines from "/usr/local/include/stack_logging.h"
//----------------------------------------------------------------------
#define stack_logging_type_free		0
#define stack_logging_type_generic	1
#define stack_logging_type_alloc	2
#define stack_logging_type_dealloc	4

//----------------------------------------------------------------------
// Redefine private function prototypes from 
// "/usr/local/include/stack_logging.h"
//----------------------------------------------------------------------
extern "C" kern_return_t 
__mach_stack_logging_set_file_path (
    task_t task, 
    char* file_path
);

extern "C" kern_return_t 
__mach_stack_logging_get_frames (
    task_t task, 
    mach_vm_address_t address, 
    mach_vm_address_t *stack_frames_buffer, 
    uint32_t max_stack_frames, 
    uint32_t *count
);

extern "C" kern_return_t
__mach_stack_logging_enumerate_records (
    task_t task,
    mach_vm_address_t address, 
    void enumerator(mach_stack_logging_record_t, void *), 
    void *context
);

extern "C" kern_return_t
__mach_stack_logging_frames_for_uniqued_stack (
    task_t task, 
    uint64_t stack_identifier, 
    mach_vm_address_t *stack_frames_buffer, 
    uint32_t max_stack_frames, 
    uint32_t *count
);

//----------------------------------------------------------------------
// Redefine private gloval variables prototypes from 
// "/usr/local/include/stack_logging.h"
//----------------------------------------------------------------------

extern "C" int stack_logging_enable_logging;
extern "C" int stack_logging_dontcompact;

//----------------------------------------------------------------------
// Local defines
//----------------------------------------------------------------------
#define MAX_FRAMES 1024

//----------------------------------------------------------------------
// Local Typedefs and Types
//----------------------------------------------------------------------
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
    eDataTypeAddress,
    eDataTypeContainsData
};

struct aligned_data_t
{
    const uint8_t *buffer;
    uint32_t size;
    uint32_t align;
};

struct range_contains_data_callback_info_t
{
    data_type_t type;
    const void *lookup_addr;
    union
    {
        uintptr_t addr;
        aligned_data_t data;
    };
    uint32_t match_count;
    bool done;
};

struct malloc_match
{
    void *addr;
    intptr_t size;
    intptr_t offset;
};

struct malloc_stack_entry
{
    const void *address;
    uint64_t argument;
    uint32_t type_flags;
    std::vector<uintptr_t> frames;
};

//----------------------------------------------------------------------
// Local global variables
//----------------------------------------------------------------------
std::vector<malloc_match> g_matches;
const void *g_lookup_addr = 0;
std::vector<malloc_stack_entry> g_malloc_stack_history;
mach_vm_address_t g_stack_frames[MAX_FRAMES];
char g_error_string[PATH_MAX];

//----------------------------------------------------------------------
// task_peek
//
// Reads memory from this tasks address space. This callback is needed
// by the code that iterates through all of the malloc blocks to read
// the memory in this process.
//----------------------------------------------------------------------
static kern_return_t
task_peek (task_t task, vm_address_t remote_address, vm_size_t size, void **local_memory)
{
    *local_memory = (void*) remote_address;
    return KERN_SUCCESS;
}


static const void
foreach_zone_in_this_process (range_callback_info_t *info)
{
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

//----------------------------------------------------------------------
// dump_malloc_block_callback
//
// A simple callback that will dump each malloc block and all available
// info from the enumeration callback perpective.
//----------------------------------------------------------------------
static void
dump_malloc_block_callback (task_t task, void *baton, unsigned type, uint64_t ptr_addr, uint64_t ptr_size)
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

static void
range_info_callback (task_t task, void *baton, unsigned type, uint64_t ptr_addr, uint64_t ptr_size)
{
    const uint64_t end_addr = ptr_addr + ptr_size;
    
    range_contains_data_callback_info_t *info = (range_contains_data_callback_info_t *)baton;
    switch (info->type)
    {
    case eDataTypeAddress:
        if (ptr_addr <= info->addr && info->addr < end_addr)
        {
            ++info->match_count;
            malloc_match match = { (void *)ptr_addr, ptr_size, info->addr - ptr_addr };
            g_matches.push_back(match);            
        }
        break;
    
    case eDataTypeContainsData:
        {
            const uint32_t size = info->data.size;
            if (size < ptr_size) // Make sure this block can contain this data
            {
                uint8_t *ptr_data = NULL;
                if (task_peek (task, ptr_addr, ptr_size, (void **)&ptr_data) == KERN_SUCCESS)
                {
                    const void *buffer = info->data.buffer;
                    assert (ptr_data);
                    const uint32_t align = info->data.align;
                    for (uint64_t addr = ptr_addr; 
                         addr < end_addr && ((end_addr - addr) >= size);
                         addr += align, ptr_data += align)
                    {
                        if (memcmp (buffer, ptr_data, size) == 0)
                        {
                            ++info->match_count;
                            malloc_match match = { (void *)ptr_addr, ptr_size, addr - ptr_addr };
                            g_matches.push_back(match);
                        }
                    }
                }
                else
                {
                    printf ("0x%llx: error: couldn't read %llu bytes\n", ptr_addr, ptr_size);
                }   
            }
        }
        break;
    }
}

static void 
get_stack_for_address_enumerator(mach_stack_logging_record_t stack_record, void *task_ptr)
{
    uint32_t num_frames = 0;
    kern_return_t err = __mach_stack_logging_frames_for_uniqued_stack (*(task_t *)task_ptr, 
                                                                       stack_record.stack_identifier,
                                                                       g_stack_frames,
                                                                       MAX_FRAMES,
                                                                       &num_frames);    
    g_malloc_stack_history.resize(g_malloc_stack_history.size() + 1);
    g_malloc_stack_history.back().address = (void *)stack_record.address;
    g_malloc_stack_history.back().type_flags = stack_record.type_flags;
    g_malloc_stack_history.back().argument = stack_record.argument;
    if (num_frames > 0)
        g_malloc_stack_history.back().frames.assign(g_stack_frames, g_stack_frames + num_frames);
    g_malloc_stack_history.back().frames.push_back(0); // Terminate the frames with zero
}

malloc_stack_entry *
get_stack_history_for_address (const void * addr, int history)
{
    std::vector<malloc_stack_entry> empty;
    g_malloc_stack_history.swap(empty);
    if (!stack_logging_enable_logging || (history && !stack_logging_dontcompact))
    {
        if (history)
            strncpy(g_error_string, "error: stack history logging is not enabled, set MallocStackLoggingNoCompact=1 in the environment when launching to enable stack history logging.", sizeof(g_error_string));
        else
            strncpy(g_error_string, "error: stack logging is not enabled, set MallocStackLogging=1 in the environment when launching to enable stack logging.", sizeof(g_error_string));
        return NULL;
    }
    kern_return_t err;
    task_t task = mach_task_self();
    if (history)
    {
        err = __mach_stack_logging_enumerate_records (task,
                                                      (mach_vm_address_t)addr, 
                                                      get_stack_for_address_enumerator,
                                                      &task);
    }
    else
    {
        uint32_t num_frames = 0;
        err = __mach_stack_logging_get_frames(task, (mach_vm_address_t)addr, g_stack_frames, MAX_FRAMES, &num_frames);
        if (err == 0 && num_frames > 0)
        {
            g_malloc_stack_history.resize(1);
            g_malloc_stack_history.back().address = addr;
            g_malloc_stack_history.back().type_flags = stack_logging_type_alloc;
            g_malloc_stack_history.back().argument = 0;
            if (num_frames > 0)
                g_malloc_stack_history.back().frames.assign(g_stack_frames, g_stack_frames + num_frames);
            g_malloc_stack_history.back().frames.push_back(0); // Terminate the frames with zero
        }
    }
    // Append an empty entry
    if (g_malloc_stack_history.empty())
        return NULL;
    g_malloc_stack_history.resize(g_malloc_stack_history.size() + 1);
    g_malloc_stack_history.back().address = 0;
    g_malloc_stack_history.back().type_flags = 0;
    g_malloc_stack_history.back().argument = 0;
    return g_malloc_stack_history.data();
}

//----------------------------------------------------------------------
// find_pointer_in_heap
//
// Finds a pointer value inside one or more currently valid malloc
// blocks.
//----------------------------------------------------------------------
malloc_match *
find_pointer_in_heap (const void * addr)
{
    g_matches.clear();
    // Setup "info" to look for a malloc block that contains data
    // that is the a pointer 
    range_contains_data_callback_info_t data_info;
    data_info.type = eDataTypeContainsData;      // Check each block for data
    g_lookup_addr = addr;
    data_info.data.buffer = (uint8_t *)&addr;    // What data? The pointer value passed in
    data_info.data.size = sizeof(addr);          // How many bytes? The byte size of a pointer
    data_info.data.align = sizeof(addr);         // Align to a pointer byte size
    data_info.match_count = 0;                   // Initialize the match count to zero
    data_info.done = false;                      // Set done to false so searching doesn't stop
    range_callback_info_t info = { enumerate_range_in_zone, range_info_callback, &data_info };
    foreach_zone_in_this_process (&info);
    if (g_matches.empty())
        return NULL;
    malloc_match match = { NULL, 0, 0 };
    g_matches.push_back(match);
    return g_matches.data();
}


//----------------------------------------------------------------------
// find_cstring_in_heap
//
// Finds a C string inside one or more currently valid malloc blocks.
//----------------------------------------------------------------------
malloc_match *
find_cstring_in_heap (const char *s)
{
    g_matches.clear();
    if (s == NULL || s[0] == '\0')
    {
        printf ("error: invalid argument (empty cstring)\n");
        return NULL;
    }
    // Setup "info" to look for a malloc block that contains data
    // that is the C string passed in aligned on a 1 byte boundary
    range_contains_data_callback_info_t data_info;
    data_info.type = eDataTypeContainsData;  // Check each block for data
    g_lookup_addr = s;               // If an expression was used, then fill in the resolved address we are looking up
    data_info.data.buffer = (uint8_t *)s;    // What data? The C string passed in
    data_info.data.size = strlen(s);         // How many bytes? The length of the C string
    data_info.data.align = 1;                // Data doesn't need to be aligned, so set the alignment to 1
    data_info.match_count = 0;               // Initialize the match count to zero
    data_info.done = false;                  // Set done to false so searching doesn't stop
    range_callback_info_t info = { enumerate_range_in_zone, range_info_callback, &data_info };
    foreach_zone_in_this_process (&info);
    if (g_matches.empty())
        return NULL;
    malloc_match match = { NULL, 0, 0 };
    g_matches.push_back(match);
    return g_matches.data();
}

//----------------------------------------------------------------------
// find_block_for_address
//
// Find the malloc block that whose address range contains "addr".
//----------------------------------------------------------------------
malloc_match *
find_block_for_address (const void *addr)
{
    g_matches.clear();
    // Setup "info" to look for a malloc block that contains data
    // that is the C string passed in aligned on a 1 byte boundary
    range_contains_data_callback_info_t data_info;
    g_lookup_addr = addr;               // If an expression was used, then fill in the resolved address we are looking up
    data_info.type = eDataTypeAddress;  // Check each block to see if the block contains the address passed in
    data_info.addr = (uintptr_t)addr;   // What data? The C string passed in
    data_info.match_count = 0;          // Initialize the match count to zero
    data_info.done = false;             // Set done to false so searching doesn't stop
    range_callback_info_t info = { enumerate_range_in_zone, range_info_callback, &data_info };
    foreach_zone_in_this_process (&info);
    if (g_matches.empty())
        return NULL;
    malloc_match match = { NULL, 0, 0 };
    g_matches.push_back(match);
    return g_matches.data();
}
