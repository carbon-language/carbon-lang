//===-- MICmdInvoker.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

// Third party headers
#include <map>

// In-house headers:
#include "MICmnBase.h"
#include "MICmdData.h"
#include "MICmdMgrSetCmdDeleteCallback.h"
#include "MIUtilSingletonBase.h"

// Declarations:
class CMICmdBase;
class CMICmnStreamStdout;

//++ ============================================================================
// Details: MI Command Invoker. The Invoker works on the command pattern design.
//          There two main jobs; action command Execute() function, followed by
//          the command's Acknowledge() function. When a command has finished its
//          execute function it returns to the invoker. The invoker then calls the
//          command's Acknowledge() function to do more work, form and give
//          back a MI result. In the meantime the Command Monitor is monitoring
//          the each command doing their Execute() function work so they do not
//          exceed a time limit which if it exceeds informs the command(s) to
//          stop work.
//          The work by the Invoker is carried out in the main thread.
//          The Invoker takes ownership of any commands created which means it
//          is the only object to delete them when a command is finished working.
//          A singleton class.
// Gotchas: None.
// Authors: Illya Rudkin 19/02/2014.
// Changes: None.
//--
class CMICmdInvoker : public CMICmnBase, public CMICmdMgrSetCmdDeleteCallback::ICallback, public MI::ISingleton<CMICmdInvoker>
{
    friend class MI::ISingleton<CMICmdInvoker>;

    // Class:
  public:
    //++
    // Description: Invoker's interface for commands to implement.
    //--
    class ICmd
    {
      public:
        virtual bool Acknowledge(void) = 0;
        virtual bool Execute(void) = 0;
        virtual bool ParseArgs(void) = 0;
        virtual bool SetCmdData(const SMICmdData &vCmdData) = 0;
        virtual const SMICmdData &GetCmdData(void) const = 0;
        virtual const CMIUtilString &GetErrorDescription(void) const = 0;
        virtual void CmdFinishedTellInvoker(void) const = 0;
        virtual const CMIUtilString &GetMIResultRecord(void) const = 0;
        virtual const CMIUtilString &GetMIResultRecordExtra(void) const = 0;
        virtual bool HasMIResultRecordExtra(void) const = 0;

        /* dtor */ virtual ~ICmd(void){};
    };

    // Methods:
  public:
    bool Initialize(void) override;
    bool Shutdown(void) override;
    bool CmdExecute(CMICmdBase &vCmd);
    bool CmdExecuteFinished(CMICmdBase &vCmd);

    // Typedefs:
  private:
    typedef std::map<MIuint, CMICmdBase *> MapCmdIdToCmd_t;
    typedef std::pair<MIuint, CMICmdBase *> MapPairCmdIdToCmd_t;

    // Methods:
  private:
    /* ctor */ CMICmdInvoker(void);
    /* ctor */ CMICmdInvoker(const CMICmdInvoker &);
    void operator=(const CMICmdInvoker &);
    void CmdDeleteAll(void);
    bool CmdDelete(const MIuint vCmdId, const bool vbYesDeleteCmd = false);
    bool CmdAdd(const CMICmdBase &vCmd);
    bool CmdStdout(const SMICmdData &vCmdData) const;
    void CmdCauseAppExit(const CMICmdBase &vCmd) const;

    // Overridden:
  private:
    // From CMICmnBase
    /* dtor */ ~CMICmdInvoker(void) override;
    // From CMICmdMgrSetCmdDeleteCallback::ICallback
    void Delete(SMICmdData &vCmd) override;

    // Attributes:
  private:
    MapCmdIdToCmd_t m_mapCmdIdToCmd;
    CMICmnStreamStdout &m_rStreamOut;
};
