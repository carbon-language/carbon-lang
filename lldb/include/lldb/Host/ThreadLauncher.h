//===-- ThreadLauncher.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_ThreadLauncher_h_
#define lldb_Host_ThreadLauncher_h_

#include "lldb/Core/Error.h"
#include "lldb/Host/HostThread.h"
#include "lldb/lldb-types.h"

#include "llvm/ADT/StringRef.h"

namespace lldb_private
{

class ThreadLauncher
{
  public:
    static HostThread LaunchThread(llvm::StringRef name, lldb::thread_func_t thread_function, lldb::thread_arg_t thread_arg,
                                   Error *error_ptr);

    struct HostThreadCreateInfo
    {
        std::string thread_name;
        lldb::thread_func_t thread_fptr;
        lldb::thread_arg_t thread_arg;

        HostThreadCreateInfo(const char *name, lldb::thread_func_t fptr, lldb::thread_arg_t arg)
            : thread_name(name ? name : "")
            , thread_fptr(fptr)
            , thread_arg(arg)
        {
        }
    };
};
}

#endif
