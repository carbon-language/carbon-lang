//===-- MICmdCmdMiscellanous.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Overview:    CMICmdCmdGdbExit                interface.
//              CMICmdCmdListThreadGroups       interface.
//              CMICmdCmdInterpreterExec        interface.
//              CMICmdCmdInferiorTtySet         interface.
//
//              To implement new MI commands derive a new command class from the command base
//              class. To enable the new command for interpretation add the new command class
//              to the command factory. The files of relevance are:
//                  MICmdCommands.cpp
//                  MICmdBase.h / .cpp
//                  MICmdCmd.h / .cpp
//              For an introduction to adding a new command see CMICmdCmdSupportInfoMiCmdQuery
//              command class as an example.

#pragma once

// Third party headers:
#include "lldb/API/SBCommandReturnObject.h"

// In-house headers:
#include "MICmdBase.h"
#include "MICmnMIValueTuple.h"
#include "MICmnMIValueList.h"

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "gdb-exit".
// Gotchas: None.
// Authors: Illya Rudkin 04/03/2014.
// Changes: None.
//--
class CMICmdCmdGdbExit : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdGdbExit(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdGdbExit(void) override;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "list-thread-groups".
//          This command does not follow the MI documentation exactly.
//          http://sourceware.org/gdb/onlinedocs/gdb/GDB_002fMI-Miscellaneous-Commands.html#GDB_002fMI-Miscellaneous-Commands
// Gotchas: None.
// Authors: Illya Rudkin 06/03/2014.
// Changes: None.
//--
class CMICmdCmdListThreadGroups : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdListThreadGroups(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    bool ParseArgs(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdListThreadGroups(void) override;

    // Typedefs:
  private:
    typedef std::vector<CMICmnMIValueTuple> VecMIValueTuple_t;

    // Attributes:
  private:
    bool m_bIsI1;           // True = Yes command argument equal "i1", false = no match
    bool m_bHaveArgOption;  // True = Yes "--available" present, false = not found
    bool m_bHaveArgRecurse; // True = Yes command argument "--recurse", false = no found
    VecMIValueTuple_t m_vecMIValueTuple;
    const CMIUtilString m_constStrArgNamedAvailable;
    const CMIUtilString m_constStrArgNamedRecurse;
    const CMIUtilString m_constStrArgNamedGroup;
    const CMIUtilString m_constStrArgNamedThreadGroup;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "interpreter-exec".
// Gotchas: None.
// Authors: Illya Rudkin 16/05/2014.
// Changes: None.
//--
class CMICmdCmdInterpreterExec : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdInterpreterExec(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    bool ParseArgs(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdInterpreterExec(void) override;

    // Attributes:
  private:
    const CMIUtilString m_constStrArgNamedInterpreter;
    const CMIUtilString m_constStrArgNamedCommand;
    lldb::SBCommandReturnObject m_lldbResult;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "inferior-tty-set".
// Gotchas: None.
// Authors: Illya Rudkin 22/07/2014.
// Changes: None.
//--
class CMICmdCmdInferiorTtySet : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdInferiorTtySet(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdInferiorTtySet(void) override;
};
