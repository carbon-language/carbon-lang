//===-- MIUtilSystemWindows.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#if defined(_MSC_VER)

// In-house headers:
#include "MIUtilString.h"

//++ ============================================================================
// Details: MI common code utility class. Used to set or retrieve information
//          about the current system or user.
//          *** If you change, remove or add functionality it must be replicated
//          *** for the all platforms supported; Windows, OSX, LINUX
//--
class CMIUtilSystemWindows
{
    // Methods:
  public:
    /* ctor */ CMIUtilSystemWindows();

    bool GetOSErrorMsg(const MIint vError, CMIUtilString &vrwErrorMsg) const;
    CMIUtilString GetOSLastError() const;
    bool GetExecutablesPath(CMIUtilString &vrwFileNamePath) const;
    bool GetLogFilesPath(CMIUtilString &vrwFileNamePath) const;

    // Overrideable:
  public:
    // From CMICmnBase
    /* dtor */ virtual ~CMIUtilSystemWindows();
};

typedef CMIUtilSystemWindows CMIUtilSystem;

#endif // #if defined( _MSC_VER )
