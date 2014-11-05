//===-- LocalDebugDelegate.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LocalDebugDelegate.h"
#include "ProcessWindows.h"

using namespace lldb;
using namespace lldb_private;

LocalDebugDelegate::LocalDebugDelegate(ProcessSP process)
    : m_process(process)
{
}

void
LocalDebugDelegate::OnProcessLaunched(const ProcessMessageCreateProcess &message)
{
    ((ProcessWindows &)*m_process).OnProcessLaunched(message);
}

void
LocalDebugDelegate::OnExitProcess(const ProcessMessageExitProcess &message)
{
    ((ProcessWindows &)*m_process).OnExitProcess(message);
}

void
LocalDebugDelegate::OnDebuggerConnected(const ProcessMessageDebuggerConnected &message)
{
    ((ProcessWindows &)*m_process).OnDebuggerConnected(message);
}

void
LocalDebugDelegate::OnDebugException(const ProcessMessageException &message)
{
    ((ProcessWindows &)*m_process).OnDebugException(message);
}

void
LocalDebugDelegate::OnCreateThread(const ProcessMessageCreateThread &message)
{
    ((ProcessWindows &)*m_process).OnCreateThread(message);
}

void
LocalDebugDelegate::OnExitThread(const ProcessMessageExitThread &message)
{
    ((ProcessWindows &)*m_process).OnExitThread(message);
}

void
LocalDebugDelegate::OnLoadDll(const ProcessMessageLoadDll &message)
{
    ((ProcessWindows &)*m_process).OnLoadDll(message);
}

void
LocalDebugDelegate::OnUnloadDll(const ProcessMessageUnloadDll &message)
{
    ((ProcessWindows &)*m_process).OnUnloadDll(message);
}

void
LocalDebugDelegate::OnDebugString(const ProcessMessageDebugString &message)
{
    ((ProcessWindows &)*m_process).OnDebugString(message);
}

void
LocalDebugDelegate::OnDebuggerError(const ProcessMessageDebuggerError &message)
{
    ((ProcessWindows &)*m_process).OnDebuggerError(message);
}
