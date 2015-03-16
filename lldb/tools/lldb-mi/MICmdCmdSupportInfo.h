//===-- MICmdCmdSupportInfo.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Overview:    CMICmdCmdSupportInfoMiCmdQuery          interface.
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

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "info-gdb-mi-command".
//          This command does not follow the MI documentation exactly.
// Gotchas: None.
// Authors: Illya Rudkin 06/05/2014.
// Changes: None.
//--
class CMICmdCmdSupportInfoMiCmdQuery : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdSupportInfoMiCmdQuery(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    virtual bool ParseArgs(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdSupportInfoMiCmdQuery(void);

    // Attributes:
  private:
    bool m_bCmdFound; // True = query for the command in command factory found, false = not found not recognised
    const CMIUtilString m_constStrArgCmdName;
};
