//===-- MICmnBase.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

// In-house headers:
#include "MIDataTypes.h"
#include "MIUtilString.h"

// Declarations:
class CMICmnLog;

//++ ============================================================================
// Details: MI common code implementation base class.
// Gotchas: None.
// Authors: Illya Rudkin 28/01/2014.
// Changes: None.
//--
class CMICmnBase
{
    // Methods:
  public:
    /* ctor */ CMICmnBase(void);

    bool HaveErrorDescription(void) const;
    const CMIUtilString &GetErrorDescription(void) const;
    void SetErrorDescription(const CMIUtilString &vrTxt) const;
    void SetErrorDescriptionn(const CMIUtilString vFormat, ...) const;
    void SetErrorDescriptionNoLog(const CMIUtilString &vrTxt) const;
    void ClrErrorDescription(void) const;

    // Overrideable:
  public:
    /* dtor */ virtual ~CMICmnBase(void);

    // Attributes:
  protected:
    mutable CMIUtilString m_strMILastErrorDescription;
    bool m_bInitialized;       // True = yes successfully initialized, false = no yet or failed
    CMICmnLog *m_pLog;         // Allow all derived classes to use the logger
    MIint m_clientUsageRefCnt; // Count of client using *this object so not shutdown() object to early
};
