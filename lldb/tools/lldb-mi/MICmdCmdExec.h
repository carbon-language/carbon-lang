//===-- MICmdCmdExec.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Overview:    CMICmdCmdExecRun                interface.
//              CMICmdCmdExecContinue           interface.
//              CMICmdCmdExecNext               interface.
//              CMICmdCmdExecStep               interface.
//              CMICmdCmdExecNextInstruction    interface.
//              CMICmdCmdExecStepInstruction    interface.
//              CMICmdCmdExecFinish             interface.
//              CMICmdCmdExecInterrupt          interface.
//              CMICmdCmdExecArguments          interface.
//              CMICmdCmdExecAbort              interface.
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

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "exec-run".
// Gotchas: None.
// Authors: Illya Rudkin 07/03/2014.
// Changes: None.
//--
class CMICmdCmdExecRun : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdExecRun(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdExecRun(void);

    // Attributes:
  private:
    lldb::SBCommandReturnObject m_lldbResult;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "exec-continue".
// Gotchas: None.
// Authors: Illya Rudkin 07/03/2014.
// Changes: None.
//--
class CMICmdCmdExecContinue : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdExecContinue(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdExecContinue(void);

    // Attributes:
  private:
    lldb::SBCommandReturnObject m_lldbResult;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "exec-next".
// Gotchas: None.
// Authors: Illya Rudkin 25/03/2014.
// Changes: None.
//--
class CMICmdCmdExecNext : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdExecNext(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    virtual bool ParseArgs(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdExecNext(void);

    // Attributes:
  private:
    lldb::SBCommandReturnObject m_lldbResult;
    const CMIUtilString m_constStrArgThread; // Not specified in MI spec but Eclipse gives this option
    const CMIUtilString m_constStrArgNumber; // Not specified in MI spec but Eclipse gives this option
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "exec-step".
// Gotchas: None.
// Authors: Illya Rudkin 25/03/2014.
// Changes: None.
//--
class CMICmdCmdExecStep : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdExecStep(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    virtual bool ParseArgs(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdExecStep(void);

    // Attributes:
  private:
    lldb::SBCommandReturnObject m_lldbResult;
    const CMIUtilString m_constStrArgThread; // Not specified in MI spec but Eclipse gives this option
    const CMIUtilString m_constStrArgNumber; // Not specified in MI spec but Eclipse gives this option
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "exec-next-instruction".
// Gotchas: None.
// Authors: Illya Rudkin 25/03/2014.
// Changes: None.
//--
class CMICmdCmdExecNextInstruction : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdExecNextInstruction(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    virtual bool ParseArgs(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdExecNextInstruction(void);

    // Attributes:
  private:
    lldb::SBCommandReturnObject m_lldbResult;
    const CMIUtilString m_constStrArgThread; // Not specified in MI spec but Eclipse gives this option
    const CMIUtilString m_constStrArgNumber; // Not specified in MI spec but Eclipse gives this option
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "exec-step-instruction".
// Gotchas: None.
// Authors: Illya Rudkin 25/03/2014.
// Changes: None.
//--
class CMICmdCmdExecStepInstruction : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdExecStepInstruction(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    virtual bool ParseArgs(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdExecStepInstruction(void);

    // Attributes:
  private:
    lldb::SBCommandReturnObject m_lldbResult;
    const CMIUtilString m_constStrArgThread; // Not specified in MI spec but Eclipse gives this option
    const CMIUtilString m_constStrArgNumber; // Not specified in MI spec but Eclipse gives this option
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "exec-finish".
// Gotchas: None.
// Authors: Illya Rudkin 25/03/2014.
// Changes: None.
//--
class CMICmdCmdExecFinish : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdExecFinish(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    virtual bool ParseArgs(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdExecFinish(void);

    // Attributes:
  private:
    lldb::SBCommandReturnObject m_lldbResult;
    const CMIUtilString m_constStrArgThread; // Not specified in MI spec but Eclipse gives this option
    const CMIUtilString m_constStrArgFrame;  // Not specified in MI spec but Eclipse gives this option
};

// CODETAG_DEBUG_SESSION_RUNNING_PROG_RECEIVED_SIGINT_PAUSE_PROGRAM
//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "exec-interrupt".
// Gotchas: Using Eclipse this command is injected into the command system when a
//          SIGINT signal is received while running an inferior program.
// Authors: Illya Rudkin 03/06/2014.
// Changes: None.
//--
class CMICmdCmdExecInterrupt : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdExecInterrupt(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdExecInterrupt(void);

    // Attributes:
  private:
    lldb::SBCommandReturnObject m_lldbResult;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "exec-arguments".
// Gotchas: None.
// Authors: Ilia Kirianovskii 25/11/2014.
// Changes: None.
//--
class CMICmdCmdExecArguments : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdExecArguments(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    virtual bool ParseArgs(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdExecArguments(void);

    // Attributes:
  private:
    const CMIUtilString m_constStrArgArguments;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "exec-abort".
//--
class CMICmdCmdExecAbort : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdExecAbort(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdExecAbort(void);
};
