//===-- MICmdCmdEnviro.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MICmdCmdEnviro.h
//
// Overview:    CMICmdCmdEnvironmentCd          interface.
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
#include "MICmnMIValueTuple.h"
#include "MICmnMIValueList.h"

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "environment-cd".
// Gotchas: None.
// Authors: Illya Rudkin 03/03/2014.
// Changes: None.
//--
class CMICmdCmdEnvironmentCd : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdEnvironmentCd(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    virtual bool ParseArgs(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdEnvironmentCd(void);

    // Attributes:
  private:
    const CMIUtilString m_constStrArgNamePathDir;
};
