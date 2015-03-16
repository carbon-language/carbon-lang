//===-- MICmdCmdGdbThread.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Overview:    CMICmdCmdGdbThread      interface.
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
//          *this class implements GDB command "thread".
// Gotchas: None.
// Authors: Illya Rudkin 25/02/2014.
// Changes: None.
//--
class CMICmdCmdGdbThread : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdGdbThread(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    virtual bool Execute(void);
    virtual bool Acknowledge(void);
    // From CMICmnBase
    /* dtor */ virtual ~CMICmdCmdGdbThread(void);
};
