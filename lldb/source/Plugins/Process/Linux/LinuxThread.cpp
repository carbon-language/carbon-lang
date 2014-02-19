//===-- LinuxThread.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "LinuxThread.h"

using namespace lldb;
using namespace lldb_private;

//------------------------------------------------------------------------------
// Constructors and destructors.

LinuxThread::LinuxThread(Process &process, lldb::tid_t tid)
    : POSIXThread(process, tid)
{
}

LinuxThread::~LinuxThread()
{
}

//------------------------------------------------------------------------------
// ProcessInterface protocol.

void
LinuxThread::RefreshStateAfterStop()
{
    // Invalidate the thread names every time we get a stop event on Linux so we
    // will re-read the procfs comm virtual file when folks ask for the thread name.
    m_thread_name_valid = false;

    POSIXThread::RefreshStateAfterStop();
}
