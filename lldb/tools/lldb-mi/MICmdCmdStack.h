//===-- MICmdCmdStack.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MICmdCmdStack.h
//
// Overview:    CMICmdCmdStackInfoDepth         interface.
//              CMICmdCmdStackListFrames        interface.
//              CMICmdCmdStackListArguments     interface.
//              CMICmdCmdStackListLocals        interface.
//
//              To implement new MI commands derive a new command class from the command base
//              class. To enable the new command for interpretation add the new command class
//              to the command factory. The files of relevance are:
//                  MICmdCommands.cpp
//                  MICmdBase.h / .cpp
//                  MICmdCmd.h / .cpp
//              For an introduction to adding a new command see CMICmdCmdSupportInfoMiCmdQuery
//              command class as an example.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--

#pragma once

// In-house headers:
#include "MICmdBase.h"
#include "MICmnMIValueList.h"

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "stack-info-depth".
// Gotchas: None.
// Authors: Illya Rudkin 21/03/2014.
// Changes: None.
//--
class CMICmdCmdStackInfoDepth : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdStackInfoDepth(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    virtual bool ParseArgs(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdStackInfoDepth(void);

    // Attributes:
  private:
    MIuint m_nThreadFrames;
    const CMIUtilString m_constStrArgThread;   // Not specified in MI spec but Eclipse gives this option
    const CMIUtilString m_constStrArgMaxDepth; // Not handled by *this command
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "stack-list-frames".
// Gotchas: None.
// Authors: Illya Rudkin 21/03/2014.
// Changes: None.
//--
class CMICmdCmdStackListFrames : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdStackListFrames(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    virtual bool ParseArgs(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdStackListFrames(void);

    // Typedefs:
  private:
    typedef std::vector<CMICmnMIValueResult> VecMIValueResult_t;

    // Attributes:
  private:
    MIuint m_nThreadFrames;
    VecMIValueResult_t m_vecMIValueResult;
    const CMIUtilString m_constStrArgThread; // Not specified in MI spec but Eclipse gives this option
    const CMIUtilString m_constStrArgFrameLow;
    const CMIUtilString m_constStrArgFrameHigh;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "stack-list-arguments".
// Gotchas: None.
// Authors: Illya Rudkin 24/03/2014.
// Changes: None.
//--
class CMICmdCmdStackListArguments : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdStackListArguments(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    virtual bool ParseArgs(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdStackListArguments(void);

    // Attributes:
  private:
    bool m_bThreadInvalid; // True = yes invalid thread, false = thread object valid
    CMICmnMIValueList m_miValueList;
    const CMIUtilString m_constStrArgThread;      // Not specified in MI spec but Eclipse gives this option
    const CMIUtilString m_constStrArgPrintValues; // Not handled by *this command
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "stack-list-locals".
// Gotchas: None.
// Authors: Illya Rudkin 24/03/2014.
// Changes: None.
//--
class CMICmdCmdStackListLocals : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdStackListLocals(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    virtual bool ParseArgs(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdStackListLocals(void);

    // Attributes:
  private:
    bool m_bThreadInvalid; // True = yes invalid thread, false = thread object valid
    CMICmnMIValueList m_miValueList;
    const CMIUtilString m_constStrArgThread;      // Not specified in MI spec but Eclipse gives this option
    const CMIUtilString m_constStrArgFrame;       // Not specified in MI spec but Eclipse gives this option
    const CMIUtilString m_constStrArgPrintValues; // Not handled by *this command
};
