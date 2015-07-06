//===-- MICmdCmd.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Overview:    CMICmdCmdEnablePrettyPrinting   interface.
//              CMICmdCmdSource                 interface.
//
//              To implement new MI commands derive a new command class from the command base
//              class. To enable the new command for interpretation add the new command class
//              to the command factory. The files of relevance are:
//                  MICmdCommands.cpp
//                  MICmdBase.h / .cpp
//                  MICmdCmd.h / .cpp
//              For an introduction to adding a new command see CMICmdCmdSupportInfoMiCmdQuery
//              command class as an example.

/*
MI commands implemented are:
        See MICmdCommands.cpp
*/

#pragma once

// Third party headers:
#include <vector>
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBCommandReturnObject.h"

// In-house headers:
#include "MICmdBase.h"
#include "MICmnMIValueTuple.h"
#include "MICmnMIValueList.h"

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "enable-pretty-printing".
//          Enables Python base pretty printing.
// Ref:     http://sourceware.org/gdb/onlinedocs/gdb/GDB_002fMI-Variable-Objects.html
// Gotchas: None.
// Authors: Illya Rudkin 03/03/2014.
// Changes: None.
//--
class CMICmdCmdEnablePrettyPrinting : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdEnablePrettyPrinting(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdEnablePrettyPrinting(void) override;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "source".
// Gotchas: None.
// Authors: Illya Rudkin 05/03/2014.
// Changes: None.
//--
class CMICmdCmdSource : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdSource(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdSource(void) override;
};
