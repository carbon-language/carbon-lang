//===-- MICmdArgContext.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MICmdArgContext.h
//
// Overview:    CMICmdArgContext interface.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--

#pragma once

// In-house headers:
#include "MIUtilString.h"

//++ ============================================================================
// Details: MI common code class. Command arguments and options string. Holds
//          the context string.
//          Based on the Interpreter pattern.
// Gotchas: None.
// Authors: Illya Rudkin 14/04/2014.
// Changes: None.
//--
class CMICmdArgContext
{
    // Methods:
  public:
    /* ctor */ CMICmdArgContext(void);
    /* ctor */ CMICmdArgContext(const CMIUtilString &vrCmdLineArgsRaw);
    //
    const CMIUtilString &GetArgsLeftToParse(void) const;
    MIuint GetNumberArgsPresent(void) const;
    CMIUtilString::VecString_t GetArgs(void) const;
    bool IsEmpty(void) const;
    bool RemoveArg(const CMIUtilString &vArg);
    bool RemoveArgAtPos(const CMIUtilString &vArg, const MIuint nArgIndex);
    //
    CMICmdArgContext &operator=(const CMICmdArgContext &vOther);

    // Overridden:
  public:
    // From CMIUtilString
    /* dtor */ virtual ~CMICmdArgContext(void);

    // Attributes:
  private:
    CMIUtilString m_strCmdArgsAndOptions;
    const MIchar m_constCharSpace;
    const CMIUtilString m_constStrSpace;
};
