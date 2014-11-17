//===-- MIUtilStreamStdin.h -------------------------------------*- C++ -*-===//
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
// Overview:    CMICmnStreamStdin interface.
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
#include "MIUtilThreadBaseStd.h"
#include "MICmnBase.h"
#include "MIUtilSingletonBase.h"

//++ ============================================================================
// Details: MI common code class. Used to handle stream data from Stdin.
//          Singleton class using the Visitor pattern. A driver using the interface
//          provide can receive callbacks when a new line of data is received.
//          Each line is determined by a carriage return.
//          A singleton class.
// Gotchas: None.
// Authors: Illya Rudkin 10/02/2014.
// Changes: Factored out OS specific handling of reading stdin  - IOR 16/06/2014.
//--
class CMICmnStreamStdin : public CMICmnBase, public CMIUtilThreadActiveObjBase, public MI::ISingleton<CMICmnStreamStdin>
{
    // Give singleton access to private constructors
    friend MI::ISingleton<CMICmnStreamStdin>;

    // Class:
  public:
    //++
    // Description: Visitor pattern. Driver(s) use this interface to get a callback
    //              on each new line of data received from stdin.
    //--
    class IStreamStdin
    {
      public:
        virtual bool ReadLine(const CMIUtilString &vStdInBuffer, bool &vrwbYesExit) = 0;

        /* dtor */ virtual ~IStreamStdin(void){};
    };

    //++
    // Description: Specific OS stdin handling implementations are created and used by *this
    //              class. Seperates out functionality and enables handler to be set
    //              dynamically depended on the OS detected.
    //--
    class IOSStdinHandler
    {
      public:
        virtual bool InputAvailable(bool &vwbAvail) = 0;
        virtual const MIchar *ReadLine(CMIUtilString &vwErrMsg) = 0;

        /* dtor */ virtual ~IOSStdinHandler(void){};
    };

    // Methods:
  public:
    bool Initialize(void);
    bool Shutdown(void);
    //
    const CMIUtilString &GetPrompt(void) const;
    bool SetPrompt(const CMIUtilString &vNewPrompt);
    void SetEnablePrompt(const bool vbYes);
    bool GetEnablePrompt(void) const;
    void SetCtrlCHit(void);
    bool SetVisitor(IStreamStdin &vrVisitor);
    bool SetOSStdinHandler(IOSStdinHandler &vrHandler);

    // Overridden:
  public:
    // From CMIUtilThreadActiveObjBase
    virtual const CMIUtilString &ThreadGetName(void) const;

    // Overridden:
  protected:
    // From CMIUtilThreadActiveObjBase
    virtual bool ThreadRun(bool &vrIsAlive);
    virtual bool
    ThreadFinish(void); // Let this thread clean up after itself

    // Methods:
  private:
    /* ctor */ CMICmnStreamStdin(void);
    /* ctor */ CMICmnStreamStdin(const CMICmnStreamStdin &);
    void operator=(const CMICmnStreamStdin &);

    bool MonitorStdin(bool &vrwbYesExit);
    const MIchar *ReadLine(CMIUtilString &vwErrMsg);
    bool
    InputAvailable(bool &vbAvail); // Bytes are available on stdin

    // Overridden:
  private:
    // From CMICmnBase
    /* dtor */ virtual ~CMICmnStreamStdin(void);

    // Attributes:
  private:
    const CMIUtilString m_constStrThisThreadname;
    IStreamStdin *m_pVisitor;
    CMIUtilString m_strPromptCurrent; // Command line prompt as shown to the user
    volatile bool m_bKeyCtrlCHit;     // True = User hit Ctrl-C, false = has not yet
    bool m_bShowPrompt;               // True = Yes prompt is shown/output to the user (stdout), false = no prompt
    bool m_bRedrawPrompt;             // True = Prompt needs to be redrawn
    IOSStdinHandler *m_pStdinReadHandler;
};
