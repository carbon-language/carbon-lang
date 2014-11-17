//===-- MICmnLLDBProxySBValue.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MICmnLLDBProxySBValue.h
//
// Overview:    CMICmnLLDBProxySBValue interface.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--

#pragma once

// Third Party Headers:
#include <lldb/API/SBValue.h>

// In-house headers:
#include "MIDataTypes.h"

// Declerations:
class CMIUtilString;

//++ ============================================================================
// Details: MI proxy wrapper class to lldb::SBValue. The class provides functionality
//          to assist in the use of SBValue's parculiar function usage.
// Gotchas: None.
// Authors: Illya Rudkin 03/04/2014.
// Changes: None.
//--
class CMICmnLLDBProxySBValue
{
    // Statics:
  public:
    static bool GetValueAsSigned(const lldb::SBValue &vrValue, MIint64 &vwValue);
    static bool GetValueAsUnsigned(const lldb::SBValue &vrValue, MIuint64 &vwValue);
    static bool GetCString(const lldb::SBValue &vrValue, CMIUtilString &vwCString);
};
