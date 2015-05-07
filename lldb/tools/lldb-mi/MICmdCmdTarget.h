//===-- MICmdCmdTarget.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Overview:    CMICmdCmdTargetSelect           interface.
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

// In-house headers:
#include "MICmdBase.h"
#include "MICmnMIValueTuple.h"
#include "MICmnMIValueList.h"

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "target-select".
//          http://sourceware.org/gdb/onlinedocs/gdb/GDB_002fMI-Target-Manipulation.html#GDB_002fMI-Target-Manipulation
// Gotchas: None.
// Authors: Illya Rudkin 05/03/2014.
// Changes: None.
//--
class CMICmdCmdTargetSelect : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdTargetSelect(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    virtual bool ParseArgs(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdTargetSelect(void);

    // Attributes:
  private:
    const CMIUtilString m_constStrArgNamedType;
    const CMIUtilString m_constStrArgNamedParameters;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "target-attach".
//          http://sourceware.org/gdb/onlinedocs/gdb/GDB_002fMI-Target-Manipulation.html#GDB_002fMI-Target-Manipulation
//--
class CMICmdCmdTargetAttach : public CMICmdBase
{
    // Statics:
public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);
    
    // Methods:
public:
    /* ctor */ CMICmdCmdTargetAttach(void);
    
    // Overridden:
public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    virtual bool ParseArgs(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdTargetAttach(void);
    
    // Attributes:
private:
    const CMIUtilString m_constStrArgPid;
    const CMIUtilString m_constStrArgNamedFile;
    const CMIUtilString m_constStrArgWaitFor;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "target-attach".
//          http://sourceware.org/gdb/onlinedocs/gdb/GDB_002fMI-Target-Manipulation.html#GDB_002fMI-Target-Manipulation
//--
class CMICmdCmdTargetDetach : public CMICmdBase
{
    // Statics:
public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);
    
    // Methods:
public:
    /* ctor */ CMICmdCmdTargetDetach(void);
    
    // Overridden:
public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    virtual bool ParseArgs(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdTargetDetach(void);
};

