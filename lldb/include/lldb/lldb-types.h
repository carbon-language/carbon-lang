//===-- lldb-types.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_types_h_
#define LLDB_types_h_

#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "lldb/Utility/SharingPtr.h"

//----------------------------------------------------------------------
//----------------------------------------------------------------------
// MACOSX START
//----------------------------------------------------------------------
//----------------------------------------------------------------------

#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

//----------------------------------------------------------------------
// All host systems must define:
//  liblldb::condition_t       The native condition type (or a substitute class) for conditions on the host system.
//  liblldb::mutex_t           The native mutex type for mutex objects on the host system.
//  liblldb::thread_t          The native thread type for spawned threads on the system
//  liblldb::thread_arg_t      The type of the one any only thread creation argument for the host system
//  liblldb::thread_result_t   The return type that gets returned when a thread finishes.
//  liblldb::thread_func_t     The function prototype used to spawn a thread on the host system.
//  liblldb::SharedPtr         The template that wraps up the host version of a reference counted pointer (like boost::shared_ptr)
//  #define LLDB_INVALID_PROCESS_ID ...
//  #define LLDB_INVALID_THREAD_ID ...
//  #define LLDB_INVALID_HOST_THREAD ...
//----------------------------------------------------------------------

// TODO: Add a bunch of ifdefs to determine the host system and what
// things should be defined. Currently MacOSX is being assumed by default
// since that is what lldb was first developed for.

namespace lldb {
        //----------------------------------------------------------------------
        // MacOSX Types
        //----------------------------------------------------------------------
        typedef ::pthread_mutex_t   mutex_t;
        typedef pthread_cond_t      condition_t;
        typedef pthread_t           thread_t;                   // Host thread type
        typedef void *              thread_arg_t;               // Host thread argument type
        typedef void *              thread_result_t;            // Host thread result type
        typedef void *              (*thread_func_t)(void *);   // Host thread function type

        // The template below can be used in a few useful ways:
        //
        //      // Make a single shared pointer a class Foo
        //      lldb::SharePtr<Foo>::Type foo_sp;
        //
        //      // Make a typedef to a Foo shared pointer
        //      typedef lldb::SharePtr<Foo>::Type FooSP;
        //
        template<typename _Tp>
        struct SharedPtr
        {
            typedef lldb_private::SharingPtr<_Tp> Type;
        };

} // namespace lldb

#define LLDB_INVALID_HOST_THREAD         ((lldb::thread_t)NULL)
#define LLDB_INVALID_HOST_TIME           { 0, 0 }

//----------------------------------------------------------------------
//----------------------------------------------------------------------
// MACOSX END
//----------------------------------------------------------------------
//----------------------------------------------------------------------

#ifdef SWIG
#define CONST_CHAR_PTR char *
#else
#define CONST_CHAR_PTR const char *
#endif

namespace lldb {
    typedef uint64_t    addr_t;
    typedef uint32_t    user_id_t;
    typedef int32_t     pid_t;
    typedef uint32_t    tid_t;
    typedef int32_t     break_id_t;

    //----------------------------------------------------------------------
    // Every register is described in detail including its name, alternate
    // name (optional), encoding, size in bytes and the default display
    // format.
    //----------------------------------------------------------------------
    typedef struct
    {
        CONST_CHAR_PTR  name;           // Name of this register, can't be NULL
        CONST_CHAR_PTR  alt_name;       // Alternate name of this register, can be NULL
        uint32_t        byte_size;      // Size in bytes of the register
        uint32_t        byte_offset;    // The byte offset in the register context data where this register's value is found
        lldb::Encoding  encoding;       // Encoding of the register bits
        lldb::Format    format;         // Default display format
        uint32_t        kinds[kNumRegisterKinds];   // Holds all of the various register numbers for all register kinds
    } RegisterInfo;

    //----------------------------------------------------------------------
    // Registers are grouped into register sets
    //----------------------------------------------------------------------
    typedef struct
    {
        CONST_CHAR_PTR name;           // Name of this register set
        CONST_CHAR_PTR short_name;     // A short name for this register set
        size_t num_registers;       // The number of registers in REGISTERS array below
        const uint32_t *registers;  // An array of register numbers in this set
    } RegisterSet;

    typedef struct
    {
        int value;
        CONST_CHAR_PTR string_value;
        CONST_CHAR_PTR usage;
    } OptionEnumValueElement;
    
    typedef struct
    {
        uint32_t        usage_mask;    // Used to mark options that can be used together.  If (1 << n & usage_mask) != 0
                                       // then this option belongs to option set n.
        bool            required;       // This option is required (in the current usage level)
        CONST_CHAR_PTR  long_option;    // Full name for this option.
        char            short_option;   // Single character for this option.
        int             option_has_arg; // no_argument, required_argument or optional_argument
        OptionEnumValueElement *enum_values;// If non-NULL an array of enum values.
        uint32_t        completionType; // Cookie the option class can use to do define the argument completion.
        CONST_CHAR_PTR  argument_name;  // Text name to be use in usage text to refer to the option's value.
        CONST_CHAR_PTR  usage_text;     // Full text explaining what this options does and what (if any) argument to
                                        // pass it.
    } OptionDefinition;


    typedef int (*comparison_function)(const void *, const void *);
}

#undef CONST_CHAR_PTR

#endif  // LLDB_types_h_
