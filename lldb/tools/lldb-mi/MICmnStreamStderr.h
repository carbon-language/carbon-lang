//===-- MICmnStreamStderr.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

// In-house headers:
#include "MIUtilString.h"
#include "MICmnBase.h"
#include "MIUtilSingletonBase.h"
#include "MIUtilThreadBaseStd.h"

//++ ============================================================================
// Details: MI common code class. The MI driver requires this object.
//          CMICmnStreamStderr sets up and tears downs stderr for the driver.
//
//          Singleton class.
// Gotchas: None.
// Authors: Illya Rudkin 19/03/2014.
// Changes: None.
//--
class CMICmnStreamStderr : public CMICmnBase, public MI::ISingleton<CMICmnStreamStderr>
{
    friend class MI::ISingleton<CMICmnStreamStderr>;

    // Statics:
  public:
    static bool TextToStderr(const CMIUtilString &vrTxt);
    static bool LLDBMsgToConsole(const CMIUtilString &vrTxt);

    // Methods:
  public:
    bool Initialize(void) override;
    bool Shutdown(void) override;
    //
    bool Lock(void);
    bool Unlock(void);
    bool Write(const CMIUtilString &vText, const bool vbSendToLog = true);
    bool WriteLLDBMsg(const CMIUtilString &vText, const bool vbSendToLog = true);

    // Methods:
  private:
    /* ctor */ CMICmnStreamStderr(void);
    /* ctor */ CMICmnStreamStderr(const CMICmnStreamStderr &);
    void operator=(const CMICmnStreamStderr &);
    //
    bool WritePriv(const CMIUtilString &vText, const CMIUtilString &vTxtForLogFile, const bool vbSendToLog = true);

    // Overridden:
  private:
    // From CMICmnBase
    /* dtor */ ~CMICmnStreamStderr(void) override;

    // Attributes:
  private:
    CMIUtilThreadMutex m_mutex; // Mutex object for sync during Write()
};
