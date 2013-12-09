//===-- FreeBSDThread.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_FreeBSDThread_H_
#define liblldb_FreeBSDThread_H_

// Other libraries and framework includes
#include "POSIXThread.h"

//------------------------------------------------------------------------------
// @class FreeBSDThread
// @brief Abstraction of a FreeBSD thread.
class FreeBSDThread
    : public POSIXThread
{
public:

    //------------------------------------------------------------------
    // Constructors and destructors
    //------------------------------------------------------------------
    FreeBSDThread(lldb_private::Process &process, lldb::tid_t tid);

    virtual ~FreeBSDThread();

    //--------------------------------------------------------------------------
    // FreeBSDThread internal API.

    // POSIXThread override
    virtual void
    WillResume(lldb::StateType resume_state);
};

#endif // #ifndef liblldb_FreeBSDThread_H_
