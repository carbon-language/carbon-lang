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
// Driver message forward declarations
class DriverMessage;
class DriverLaunchProcessMessage;

// Driver message result forward declarations
class DriverMessageResult;
class DriverLaunchProcessMessageResult;

class IDebugDelegate;

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
}

#endif