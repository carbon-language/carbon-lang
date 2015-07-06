//===-- MICmdInterpreter.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

// In-house headers:
#include "MICmnBase.h"
#include "MICmdData.h"
#include "MIUtilSingletonBase.h"

// Declarations:
class CMICmdFactory;

//++ ============================================================================
// Details: MI command interpreter. It takes text data from the MI driver
//          (which got it from Stdin singleton) and validate the text to see if
//          matches Machine Interface (MI) format and commands defined in the
//          MI application.
//          A singleton class.
// Gotchas: None.
// Authors: Illya Rudkin 18/02/2014.
// Changes: None.
//--
class CMICmdInterpreter : public CMICmnBase, public MI::ISingleton<CMICmdInterpreter>
{
    friend MI::ISingleton<CMICmdInterpreter>;

    // Methods:
  public:
    // Methods:
  public:
    bool Initialize(void) override;
    bool Shutdown(void) override;
    bool ValidateIsMi(const CMIUtilString &vTextLine, bool &vwbYesValid, bool &vwbCmdNotInCmdFactor, SMICmdData &rwCmdData);

    // Methods:
  private:
    /* ctor */ CMICmdInterpreter(void);
    /* ctor */ CMICmdInterpreter(const CMICmdInterpreter &);
    void operator=(const CMICmdInterpreter &);

    bool HasCmdFactoryGotMiCmd(const SMICmdData &vCmdData) const;
    bool MiHasCmdTokenEndingHyphen(const CMIUtilString &vTextLine);
    bool MiHasCmdTokenEndingAlpha(const CMIUtilString &vTextLine);
    bool MiHasCmd(const CMIUtilString &vTextLine);
    bool MiHasCmdTokenPresent(const CMIUtilString &vTextLine);
    const SMICmdData &MiGetCmdData() const;

    // Overridden:
  private:
    // From CMICmnBase
    /* dtor */ ~CMICmdInterpreter(void) override;

    // Attributes:
  private:
    SMICmdData m_miCmdData; // Filled in on each new line being interpreted
    CMICmdFactory &m_rCmdFactory;
};
