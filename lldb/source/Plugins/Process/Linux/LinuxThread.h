//===-- LinuxThread.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_LinuxThread_H_
#define liblldb_LinuxThread_H_

// Other libraries and framework includes
#include "POSIXThread.h"

//------------------------------------------------------------------------------
// @class LinuxThread
// @brief Abstraction of a Linux thread.
class LinuxThread
    : public POSIXThread
{
public:

    //------------------------------------------------------------------
    // Constructors and destructors
    //------------------------------------------------------------------
    LinuxThread(lldb_private::Process &process, lldb::tid_t tid);

    virtual ~LinuxThread();

    //--------------------------------------------------------------------------
    // LinuxThread internal API.

    // POSIXThread override
    virtual void
    RefreshStateAfterStop();
};

#endif // #ifndef liblldb_LinuxThread_H_
