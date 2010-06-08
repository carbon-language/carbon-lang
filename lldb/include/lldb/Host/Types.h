//===-- Types.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#if 0
#ifndef liblldb_host_types_h_
#define liblldb_host_types_h_

//----------------------------------------------------------------------
//----------------------------------------------------------------------
// MACOSX START
//----------------------------------------------------------------------
//----------------------------------------------------------------------

#include <assert.h>
#include <mach/mach_types.h>
#include <machine/endian.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/syslimits.h>
#include <unistd.h>

#ifndef NO_RTTI

//----------------------------------------------------------------------
// And source files that may not have RTTI enabled during their
// compilation will want to do a "#define NO_RTTI" before including the
// lldb-include.h file.
//----------------------------------------------------------------------

#include <tr1/memory> // for std::tr1::shared_ptr

#endif

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

namespace lldb_private {
        //----------------------------------------------------------------------
        // MacOSX Types
        //----------------------------------------------------------------------
        typedef ::pthread_mutex_t   mutex_t;
        typedef pthread_cond_t      condition_t;
        typedef pthread_t           thread_t;                   // Host thread type
        typedef void *              thread_arg_t;               // Host thread argument type
        typedef void *              thread_result_t;            // Host thread result type
        typedef void *              (*thread_func_t)(void *);   // Host thread function type

#ifndef NO_RTTI
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
            typedef std::tr1::shared_ptr<_Tp> Type;
        };
#endif

} // namespace lldb_private

#define LLDB_INVALID_HOST_THREAD         ((lldb::thread_t)NULL)
#define LLDB_INVALID_HOST_TIME           { 0, 0 }

//----------------------------------------------------------------------
//----------------------------------------------------------------------
// MACOSX END
//----------------------------------------------------------------------
//----------------------------------------------------------------------

#endif  // liblldb_host_types_h_
#endif
