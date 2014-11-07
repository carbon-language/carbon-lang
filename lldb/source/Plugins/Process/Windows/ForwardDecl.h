//===-- ForwardDecl.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_ForwardDecl_H_
#define liblldb_Plugins_Process_Windows_ForwardDecl_H_

class ProcessWindows;

#include <memory>

namespace lldb_private
{
class IDebugDelegate;

class DebuggerThread;

// Process message forward declarations.
class ProcessMessageBase;
class ProcessMessageCreateProcess;
class ProcessMessageExitProcess;
class ProcessMessageDebuggerConnected;
class ProcessMessageException;
class ProcessMessageCreateThread;
class ProcessMessageExitThread;
class ProcessMessageLoadDll;
class ProcessMessageUnloadDll;
class ProcessMessageDebugString;
class ProcessMessageDebuggerError;

typedef std::shared_ptr<IDebugDelegate> DebugDelegateSP;
typedef std::unique_ptr<DebuggerThread> DebuggerThreadUP;
}

#endif