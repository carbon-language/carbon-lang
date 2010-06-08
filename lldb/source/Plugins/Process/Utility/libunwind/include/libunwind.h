/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- libunwind.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//            C interface to libuwind 
//
// Source compatible with Level 1 Base ABI documented at:
//    http://www.codesourcery.com/public/cxx-abi/abi-eh.html
// 
//===----------------------------------------------------------------------===//


#ifndef __LIBUNWIND__
#define __LIBUNWIND__

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <mach/mach_types.h>
#include <Availability.h>

namespace lldb_private {

#pragma mark Error codes

enum {
    UNW_ESUCCESS           = 0,            /* no error */
    UNW_EUNSPEC            = -6540,        /* unspecified (general) error */
    UNW_ENOMEM             = -6541,        /* out of memory */
    UNW_EBADREG            = -6542,        /* bad register number */
    UNW_EREADONLYREG       = -6543,        /* attempt to write read-only register */
    UNW_ESTOPUNWIND        = -6544,        /* stop unwinding */
    UNW_EINVALIDIP         = -6545,        /* invalid IP */
    UNW_EBADFRAME          = -6546,        /* bad frame */
    UNW_EINVAL             = -6547,        /* unsupported operation or bad value */
    UNW_EBADVERSION        = -6548,        /* unwind info has unsupported version */
    UNW_ENOINFO            = -6549,        /* no unwind info found */
    UNW_EREGUNAVAILABLE    = -6550         /* contents of requested reg are not available */
};

#pragma mark General data structures

struct unw_context_t { uint64_t data[128]; };
typedef struct unw_context_t     unw_context_t;

struct unw_cursor_t { uint64_t data[140]; };
typedef struct unw_cursor_t      unw_cursor_t;

enum unw_as_type { UNW_LOCAL, UNW_REMOTE };
struct unw_addr_space
{ 
	enum unw_as_type type; 
	uint8_t data[]; 
};
typedef struct unw_addr_space* unw_addr_space_t;

typedef int                      unw_regnum_t;
typedef uint64_t                 unw_word_t;
typedef double                   unw_fpreg_t;

enum unw_vecreg_format {
	UNW_VECREG_SIGNED,
	UNW_VECREG_UNSIGNED,
	UNW_VECREG_FLOAT
};

typedef struct
{
	union {
		double   doubles[8];
		float    floats [16];
		
		uint64_t dwords	[8];
		uint32_t words	[16];
		uint16_t hwords	[32];
		uint8_t  bytes	[64];
	} data;
	uint16_t unit_size; // bits
	uint16_t num_units;
	uint8_t format;
} unw_vecreg_t;

struct unw_proc_info_t
{
    unw_word_t    start_ip;         /* start address of function */
    unw_word_t    end_ip;           /* address after end of function */
    unw_word_t    lsda;             /* address of language specific data area, or zero if not used */
    unw_word_t    handler;          /* personality routine, or zero if not used */
    unw_word_t    gp;               /* not used */
    unw_word_t    flags;            /* not used */
    uint32_t      format;           /* compact unwind encoding, or zero if none */
    uint32_t      unwind_info_size; /* size of dwarf unwind info, or zero if none */
    unw_word_t    unwind_info;      /* address of dwarf unwind info, or zero if none */
    unw_word_t    extra;            /* mach_header of mach-o image containing function */
};
typedef struct unw_proc_info_t unw_proc_info_t;

#pragma mark Local API

#ifdef __cplusplus
extern "C" {
#endif

extern int         unw_getcontext(unw_context_t*)                               __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);
extern int         unw_init_local(unw_cursor_t*, unw_context_t*)                __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);
extern int         unw_step(unw_cursor_t*)                                      __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);
extern int         unw_get_reg(unw_cursor_t*, unw_regnum_t, unw_word_t*)        __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);
extern int         unw_get_fpreg(unw_cursor_t*, unw_regnum_t, unw_fpreg_t*)     __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);
extern int         unw_set_reg(unw_cursor_t*, unw_regnum_t, unw_word_t)         __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);
extern int         unw_set_fpreg(unw_cursor_t*, unw_regnum_t, unw_fpreg_t)      __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);
extern int         unw_resume(unw_cursor_t*)                                    __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);

extern const char* unw_regname(unw_cursor_t*, unw_regnum_t)                     __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);
extern int         unw_get_proc_info(unw_cursor_t*, unw_proc_info_t*)           __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);
extern int         unw_is_fpreg(unw_cursor_t*, unw_regnum_t)                    __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);
extern int         unw_is_signal_frame(unw_cursor_t*)                           __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);
extern int         unw_get_proc_name(unw_cursor_t*, char*, size_t, unw_word_t*) __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);
//extern int       unw_get_save_loc(unw_cursor_t*, int, unw_save_loc_t*);


#pragma mark Remote data structures
	
typedef enum {
	UNW_NOT_A_REG = 0,
	UNW_INTEGER_REG,
	UNW_FLOATING_POINT_REG,
	UNW_VECTOR_REG,
	UNW_OTHER_REG
} unw_regtype_t;

typedef enum {
	UNW_TARGET_UNSPECIFIED = 0,
	UNW_TARGET_I386,
	UNW_TARGET_X86_64,
	UNW_TARGET_PPC,
	UNW_TARGET_ARM
} unw_targettype_t;

typedef enum {
    UNW_LOG_LEVEL_NONE =     0x00000000,
    UNW_LOG_LEVEL_INFO =     0x00000001,
    UNW_LOG_LEVEL_API =      0x00000002,
    UNW_LOG_LEVEL_VERBOSE =  0x00000004,
    UNW_LOG_LEVEL_TIMINGS =  0x00000008,
    UNW_LOG_LEVEL_DEBUG =    0x00000010,
    UNW_LOG_LEVEL_ALL =      0x0FFFFFFF
} unw_log_level_t;

typedef enum {
	UNW_CACHE_NONE = 0,
	UNW_CACHE_GLOBAL,
	UNW_CACHE_PER_THREAD
} unw_caching_policy_t;

typedef struct {
	int (*find_proc_info)(unw_addr_space_t as, unw_word_t ip, unw_proc_info_t *pip, int need_unwind_info, void *arg);
	int (*put_unwind_info)(unw_addr_space_t as, unw_proc_info_t *pip, void *arg);
	int (*get_dyn_info_list_addr)(unw_addr_space_t as, unw_word_t *dilap, void *arg);

	// Reads or writes a memory object the size of a target pointer.  
    // Byte-swaps if necessary.
	int (*access_mem)(unw_addr_space_t as, unw_word_t addr, unw_word_t *valp, int write, void *arg);
	
    // Register contents sent as-is (i.e. not byte-swapped).  
    // Register numbers are the driver program's numbering scheme as 
    // determined by the reg_info callbacks; libunwind will interrogate 
    // the driver program to figure out which numbers it uses to refer to 
    // which registers.
	int (*access_reg)(unw_addr_space_t as, unw_regnum_t regnum, unw_word_t *valp, int write, void *arg);
	int (*access_fpreg)(unw_addr_space_t as, unw_regnum_t regnum, unw_fpreg_t *valp, int write, void *arg);
	int (*resume)(unw_addr_space_t as, unw_cursor_t *cp, void *arg);
	int (*get_proc_name)(unw_addr_space_t as, unw_word_t addr, char *bufp, size_t buf_len, unw_word_t *offp, void *arg);


	// Added to find the start of the image (executable, bundle, dylib, etc) 
    // for a given address.
    //   as                     - The address space to use
    //   ip                     - The address libunwind wants to know about
    //   mh                     - The Mach-O header address for this image
    //   text_start             - The start of __TEXT segment (all its sections)
    //   text_end               - The end address of __TEXT segment (all its sections)
    //   eh_frame               - The start of __TEXT,__eh_frame
    //   eh_frame_len           - The length of __TEXT,__eh_frame
    //   compact_unwind_start   - The start of __TEXT,__unwind_info
    //   compact_unwind_len     - The length of __TEXT,__unwind_info
    //   arg                    - The driver-provided generic argument
    // All addresses are the in-memory, slid, addresses. 
    // If eh_frame or unwind_info are missing, addr and len is returned as 0.
    int (*find_image_info)(unw_addr_space_t as, unw_word_t ip, unw_word_t *mh, 
                           unw_word_t *text_start, unw_word_t *text_end,
                           unw_word_t *eh_frame, unw_word_t *eh_frame_len, 
                           unw_word_t *compact_unwind_start, 
                           unw_word_t *compact_unwind_len, void *arg);

    // Added to get the start and end address of a function without needing
    // all of the information (and potential allocation) that the
    // find_proc_info() call entails.
    //   as         - The address space to use
    //   ip         - The address libunwind wants to know about
    //   low        - The start address of the function at 'ip'
    //   high       - The first address past the function at 'ip'
    //   arg        - The driver-provided generic argument
    // If HIGH is unknown, it should be set to 0.  All addresses
    // are the in-memory, slid, addresses.
	int (*get_proc_bounds)(unw_addr_space_t as, unw_word_t ip, 
                           unw_word_t *low, unw_word_t *high, void *arg);

    // Added to support accessing non-word-size memory objects across 
    // platforms.  No byte swapping should be done.
    //   as     - The address space to use
    //   addr   - The starting address to access
    //   extent - The extent of the region to access, in bytes
    //   valp   - The local region to be written from / read into
    //   write  - non-zero if the data is to be written into the target
    //            rather than read
    //   arg    - The driver-provided generic argument (see unw_init_remote)
    int (*access_raw)(unw_addr_space_t as, unw_word_t addr, unw_word_t extent, 
                      uint8_t *valp, int write, void *arg);

    // Added to support identifying registers.
    // libunwind will interrogate the driver program via this callback to
    // identify what register numbers it is using; the register names are
    // used to correlate that the driver program's register numbers with
    // libunwind's internal register numbers.  The driver program should
    // use its own register numbers when requesting registers with
    // unw_get_reg() and libunwind will provide the driver program's
    // register numbers to the access_reg callback function.
    //   as         - The address space to use
    //   regnum     - The register number
    //   type       - Write the register type to this address
    //                For a non-existent register, return UNW_ESUCCESS but 
    //                write UNW_NOT_A_REG to type
    //   buf        - If non-NULL, the register name is written to this address
    //   buf_len    - The size of the buffer provided for the register name
    //   arg        - The driver-provided generic argument (see unw_init_remote)
    int (*reg_info)(unw_addr_space_t as, unw_regnum_t regnum, 
                    unw_regtype_t* type, char *bufp, size_t buf_len, void *arg);

	// Added to read a vector register's value from the remote machine.
	//   as			- The address space to use
	//   regnum		- The register number
	//   valp		- The local region to be written from / read into
	//   write		- non-zero if the data is to be written into the target 
    //                rather than read
	//   arg		- The driver-specified generic argument
	int (*access_vecreg)(unw_addr_space_t as, unw_regnum_t regnum, 
                         unw_vecreg_t* valp, int write, void *arg);

    // Added to identify if an unwind cursor is pointing to _sigtramp().
    // After a _sigtramp we have an entire register set available and we should
    // return any of the registers requested.
    //  as          - The address space to use
    //  ip          - The address of the function libunwind is examining
    //  arg         - The driver-provided generic argument
    // This function returns non-zero if ip is in _sigtramp.
    int (*proc_is_sigtramp) (unw_addr_space_t as, unw_word_t ip, void *arg);

    // Added to identify if an unwind cursor is pointing to a debugger's
    // inferior function call dummy frame.
    // The driver program will need to provide the full register set (via the
    // standard access_reg callback) for the function that was executing 
    // when the inferior function call was made; it will use these register
    // values and not try to unwind out of the inferior function call dummy
    // frame.
    // After a inf func call we have an entire register set available and 
    // we should return any of the registers requested.
    //  as          - The address space to use
    //  ip          - The address of the function libunwind is examining
    //  sp          - The stack pointer value of the frame
    //  arg         - The driver-provided generic argument (see unw_init_remote)
    // This function returns non-zero if ip/sp is an inferior function call 
    // dummy frame.
    int (*proc_is_inferior_function_call) (unw_addr_space_t as, unw_word_t ip, 
                                           unw_word_t sp, void *arg);

    // Added to retrieve a register value from a above a debugger's inferior
    // function call dummy frame.  Similar to _sigtramp but the debugger will
    // have the register context squirreled away in its own memory (or possibly
    // saved on the stack somewhere).
    // May be NULL if the program being unwound will not have a debugger
    // calling functions mid-execution.
    //   as         - The address space to use
    //   ip         - The pc value for the dummy frame
    //   sp         - The stack pointer for the dummy frame
    //   regnum     - The register number in the driver program's register
    //                numbering scheme.
    //   valp       - Pointer to a word of memory to be read/written
    //   write      - Non-zero if libunwind is writing a new value to the reg,
    //                else it is reading the contents of that register.
    //   arg        - The driver-provided generic argument (see unw_init_remote)
	int (*access_reg_inf_func_call)(unw_addr_space_t as, unw_word_t ip, 
                                    unw_word_t sp, unw_regnum_t regnum, 
                                    unw_word_t *valp, int write, void *arg);

    // Added to iterate over unknown assembly instructions when analyzing a
    // function prologue.  Needed for ISAs with variable length instructions
    // (i386, x86_64) or multiple instruction sizes (arm, thumb).
    // Returns zero if the instruction length was successfully measured.
    //  as         - The address space to use
    //  addr       - The address of the instruction being measured
    //  length     - Set to the length of the instruction
    //  arg        - The driver-provided generic argument (see unw_init_remote)
    int (*instruction_length)(unw_addr_space_t as, unw_word_t addr, 
                              int *length, void *arg);

} unw_accessors_t;

extern int               unw_init_remote(unw_cursor_t*, unw_addr_space_t, void*)			__OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);
extern unw_accessors_t*  unw_get_accessors(unw_addr_space_t)								__OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);
extern unw_addr_space_t	 unw_create_addr_space(unw_accessors_t*, unw_targettype_t)	        __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);
extern void              unw_flush_caches(unw_addr_space_t)			                        __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);
extern int               unw_set_caching_policy(unw_addr_space_t, unw_caching_policy_t)		__OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);
extern void              unw_destroy_addr_space(unw_addr_space_t asp)                       __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);
extern void              unw_set_logging_level(unw_addr_space_t, FILE *, unw_log_level_t)   __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA); 

// Should be called when remote unwinding if a bundle in the remote process 
// is unloaded
extern void              unw_image_was_unloaded(unw_addr_space_t, unw_word_t mh)            __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);

// Try to discern where the function's prologue instructions end
//   start - start address of the function, required
//   end   - first address beyond the function, or zero if unknown
//   endofprologue - set to the address after the last prologue instruction if successful
extern int               unw_end_of_prologue_setup(unw_cursor_t*, unw_word_t start, unw_word_t end, unw_word_t *endofprologue) __OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_NA);

/*
 * Dynamic unwinding API
 *     NOT IMPLEMENTED on Mac OS X 
 * extern void              _U_dyn_register(unw_dyn_info_t*);
 * extern void              _U_dyn_cancel(unw_dyn_info_t*);
 */

#ifdef __cplusplus
}
#endif

#pragma mark Register numbers

// architecture independent register numbers 
enum {
    UNW_REG_IP = -1,        // instruction pointer
    UNW_REG_SP = -2,        // stack pointer
};


// 32-bit x86 registers
enum {
    UNW_X86_EAX = 0,
    UNW_X86_ECX = 1,
    UNW_X86_EDX = 2,
    UNW_X86_EBX = 3,
    UNW_X86_EBP = 4,
    UNW_X86_ESP = 5,
    UNW_X86_ESI = 6,
    UNW_X86_EDI = 7
};


// 64-bit x86_64 registers
enum {
    UNW_X86_64_RAX =  0,
    UNW_X86_64_RDX =  1,
    UNW_X86_64_RCX =  2,
    UNW_X86_64_RBX =  3,
    UNW_X86_64_RSI =  4,
    UNW_X86_64_RDI =  5,
    UNW_X86_64_RBP =  6,
    UNW_X86_64_RSP =  7,
    UNW_X86_64_R8  =  8,
    UNW_X86_64_R9  =  9,
    UNW_X86_64_R10 = 10,
    UNW_X86_64_R11 = 11,
    UNW_X86_64_R12 = 12,
    UNW_X86_64_R13 = 13,
    UNW_X86_64_R14 = 14,
    UNW_X86_64_R15 = 15
};


// 32-bit ppc register numbers
enum {
    UNW_PPC_R0  =  0,
    UNW_PPC_R1  =  1,
    UNW_PPC_R2  =  2,
    UNW_PPC_R3  =  3,
    UNW_PPC_R4  =  4,
    UNW_PPC_R5  =  5,
    UNW_PPC_R6  =  6,
    UNW_PPC_R7  =  7,
    UNW_PPC_R8  =  8,
    UNW_PPC_R9  =  9,
    UNW_PPC_R10 = 10,
    UNW_PPC_R11 = 11,
    UNW_PPC_R12 = 12,
    UNW_PPC_R13 = 13,
    UNW_PPC_R14 = 14,
    UNW_PPC_R15 = 15,
    UNW_PPC_R16 = 16,
    UNW_PPC_R17 = 17,
    UNW_PPC_R18 = 18,
    UNW_PPC_R19 = 19,
    UNW_PPC_R20 = 20,
    UNW_PPC_R21 = 21,
    UNW_PPC_R22 = 22,
    UNW_PPC_R23 = 23,
    UNW_PPC_R24 = 24,
    UNW_PPC_R25 = 25,
    UNW_PPC_R26 = 26,
    UNW_PPC_R27 = 27,
    UNW_PPC_R28 = 28,
    UNW_PPC_R29 = 29,
    UNW_PPC_R30 = 30,
    UNW_PPC_R31 = 31,
    UNW_PPC_F0  = 32,
    UNW_PPC_F1  = 33,
    UNW_PPC_F2  = 34,
    UNW_PPC_F3  = 35,
    UNW_PPC_F4  = 36,
    UNW_PPC_F5  = 37,
    UNW_PPC_F6  = 38,
    UNW_PPC_F7  = 39,
    UNW_PPC_F8  = 40,
    UNW_PPC_F9  = 41,
    UNW_PPC_F10 = 42,
    UNW_PPC_F11 = 43,
    UNW_PPC_F12 = 44,
    UNW_PPC_F13 = 45,
    UNW_PPC_F14 = 46,
    UNW_PPC_F15 = 47,
    UNW_PPC_F16 = 48,
    UNW_PPC_F17 = 49,
    UNW_PPC_F18 = 50,
    UNW_PPC_F19 = 51,
    UNW_PPC_F20 = 52,
    UNW_PPC_F21 = 53,
    UNW_PPC_F22 = 54,
    UNW_PPC_F23 = 55,
    UNW_PPC_F24 = 56,
    UNW_PPC_F25 = 57,
    UNW_PPC_F26 = 58,
    UNW_PPC_F27 = 59,
    UNW_PPC_F28 = 60,
    UNW_PPC_F29 = 61,
    UNW_PPC_F30 = 62,
    UNW_PPC_F31 = 63,
    UNW_PPC_MQ  = 64,
    UNW_PPC_LR  = 65,
    UNW_PPC_CTR = 66,
    UNW_PPC_AP  = 67,
	UNW_PPC_CR0 = 68,
    UNW_PPC_CR1 = 69,
	UNW_PPC_CR2 = 70,
	UNW_PPC_CR3 = 71,
	UNW_PPC_CR4 = 72,
	UNW_PPC_CR5 = 73,
	UNW_PPC_CR6 = 74,
	UNW_PPC_CR7 = 75,
	UNW_PPC_XER = 76,
	UNW_PPC_V0  = 77,
    UNW_PPC_V1  = 78,
    UNW_PPC_V2  = 79,
    UNW_PPC_V3  = 80,
    UNW_PPC_V4  = 81,
    UNW_PPC_V5  = 82,
    UNW_PPC_V6  = 83,
    UNW_PPC_V7  = 84,
    UNW_PPC_V8  = 85,
    UNW_PPC_V9  = 86,
    UNW_PPC_V10 = 87,
    UNW_PPC_V11 = 88,
    UNW_PPC_V12 = 89,
    UNW_PPC_V13 = 90,
    UNW_PPC_V14 = 91,
    UNW_PPC_V15 = 92,
    UNW_PPC_V16 = 93,
    UNW_PPC_V17 = 94,
    UNW_PPC_V18 = 95,
    UNW_PPC_V19 = 96,
    UNW_PPC_V20 = 97,
    UNW_PPC_V21 = 98,
    UNW_PPC_V22 = 99,
    UNW_PPC_V23 = 100,
    UNW_PPC_V24 = 101,
    UNW_PPC_V25 = 102,
    UNW_PPC_V26 = 103,
    UNW_PPC_V27 = 104,
    UNW_PPC_V28 = 105,
    UNW_PPC_V29 = 106,
    UNW_PPC_V30 = 107,
    UNW_PPC_V31 = 108,
    UNW_PPC_VRSAVE  = 109,
    UNW_PPC_VSCR    = 110,
    UNW_PPC_SPE_ACC = 111,
    UNW_PPC_SPEFSCR = 112
	
};


}; // namespace lldb_private


#endif

