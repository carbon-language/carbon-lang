//===-- MICmnStreamStdinWindows.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MIUtilStreamStdin.h
//
// Overview:    CMICmnStreamStdinWindows interface.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--

#pragma once

// In-house headers:
#include "MICmnBase.h"
#include "MICmnStreamStdin.h"
#include "MIUtilSingletonBase.h"

//++ ============================================================================
// Details: MI common code class. Specific OS stdin handling implementation.
//          CMICmnStreamStdin instance is set with stdin handler before using the
//          the stream. An instance of this class must be set up and ready to give
//          to the CMICmnStreamStdin before it initialises other CMICmnStreamStdin
//          will give an error.
// Gotchas: None.
// Authors: Illya Rudkin 16/06/2014.
// Changes: None.
//--
class CMICmnStreamStdinWindows : public CMICmnBase,
                                 public CMICmnStreamStdin::IOSStdinHandler,
                                 public MI::ISingleton<CMICmnStreamStdinWindows>
{
    // Give singleton access to private constructors
    friend MI::ISingleton<CMICmnStreamStdinWindows>;

    // Methods:
  public:
    bool Initialize(void);
    bool Shutdown(void);

    // Overridden:
  public:
    // From CMICmnStreamStdin::IOSpecificReadStreamStdin
    virtual bool InputAvailable(bool &vwbAvail);
    virtual const MIchar *ReadLine(CMIUtilString &vwErrMsg);

    // Methods:
  private:
    /* ctor */ CMICmnStreamStdinWindows(void);
    /* ctor */ CMICmnStreamStdinWindows(const CMICmnStreamStdinWindows &);
    void operator=(const CMICmnStreamStdinWindows &);
    //
    bool InputAvailableConsoleWin(bool &vwbAvail);
    bool InputAvailableApplication(bool &vwbAvail);

    // Overridden:
  private:
    // From CMICmnBase
    /* dtor */ virtual ~CMICmnStreamStdinWindows(void);

    // Attributes:
  private:
    const MIuint m_constBufferSize;
    FILE *m_pStdin;
    MIchar *m_pCmdBuffer;
    MIchar *m_pStdinBuffer;  // Custom buffer to store std input
    MIuint m_nBytesToBeRead; // Checks that ::fgets() is holding on to data while ::PeekNamedPipe() returns nothing which causes a problem
    bool m_bRunningInConsoleWin; // True = The application is being run in a Windows command line prompt window, false = by other means
};
