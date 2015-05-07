//===-- MIDriver.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

// Third party headers
#include <queue>

// In-house headers:
#include "MICmnConfig.h"
#include "MICmnBase.h"
#include "MIDriverBase.h"
#include "MIDriverMgr.h"
#include "MICmnStreamStdin.h"
#include "MICmdData.h"
#include "MIUtilSingletonBase.h"

// Declarations:
class CMICmnLLDBDebugger;
class CMICmnStreamStdout;

//++ ============================================================================
// Details: MI driver implementation class. A singleton class derived from
//          LLDB SBBroadcaster class. Register the instance of *this class with
//          the CMIDriverMgr. The CMIDriverMgr sets the driver(s) of to start
//          work depending on the one selected to work. A driver can if not able
//          to handle an instruction or 'command' can pass that command onto
//          another driver object registered with the Driver Manager.
// Gotchas: None.
// Authors: Illya Rudkin 29/01/2014.
// Changes: None.
//--
class CMIDriver : public CMICmnBase,
                  public CMIDriverMgr::IDriver,
                  public CMIDriverBase,
                  public MI::ISingleton<CMIDriver>
{
    friend class MI::ISingleton<CMIDriver>;

    // Enumerations:
  public:
    //++ ----------------------------------------------------------------------
    // Details: The MI Driver has a running state which is used to help determin
    //          which specific action(s) it should take or not allow.
    //          The driver when operational and not shutting down alternates
    //          between eDriverState_RunningNotDebugging and
    //          eDriverState_RunningDebugging. eDriverState_RunningNotDebugging
    //          is normally set when a breakpoint is hit or halted.
    //          eDriverState_RunningDebugging is normally set when "exec-continue"
    //          or "exec-run" is issued.
    //--
    enum DriverState_e
    {
        eDriverState_NotRunning = 0,      // The MI Driver is not operating
        eDriverState_Initialising,        // The MI Driver is setting itself up
        eDriverState_RunningNotDebugging, // The MI Driver is operational acting on any MI commands sent to it
        eDriverState_RunningDebugging,    // The MI Driver is currently overseeing an inferior program that is running
        eDriverState_ShuttingDown,        // The MI Driver is tearing down resources and about exit
        eDriverState_count                // Always last
    };

    // Methods:
  public:
    // MI system
    bool Initialize(void);
    bool Shutdown(void);

    // MI state
    bool GetExitApplicationFlag(void) const;
    DriverState_e GetCurrentDriverState(void) const;
    bool SetDriverStateRunningNotDebugging(void);
    bool SetDriverStateRunningDebugging(void);
    void SetDriverDebuggingArgExecutable(void);
    bool IsDriverDebuggingArgExecutable(void) const;

    // MI information about itself
    const CMIUtilString &GetAppNameShort(void) const;
    const CMIUtilString &GetAppNameLong(void) const;
    const CMIUtilString &GetVersionDescription(void) const;

    // MI do work
    bool WriteMessageToLog(const CMIUtilString &vMessage);
    bool SetEnableFallThru(const bool vbYes);
    bool GetEnableFallThru(void) const;
    bool HaveExecutableFileNamePathOnCmdLine(void) const;
    const CMIUtilString &GetExecutableFileNamePathOnCmdLine(void) const;

    // Overridden:
  public:
    // From CMIDriverMgr::IDriver
    virtual bool DoInitialize(void);
    virtual bool DoShutdown(void);
    virtual bool DoMainLoop(void);
    virtual lldb::SBError DoParseArgs(const int argc, const char *argv[], FILE *vpStdOut, bool &vwbExiting);
    virtual CMIUtilString GetError(void) const;
    virtual const CMIUtilString &GetName(void) const;
    virtual lldb::SBDebugger &GetTheDebugger(void);
    virtual bool GetDriverIsGDBMICompatibleDriver(void) const;
    virtual bool SetId(const CMIUtilString &vId);
    virtual const CMIUtilString &GetId(void) const;
    // From CMIDriverBase
    virtual void SetExitApplicationFlag(const bool vbForceExit);
    virtual bool DoFallThruToAnotherDriver(const CMIUtilString &vCmd, CMIUtilString &vwErrMsg);
    virtual bool SetDriverToFallThruTo(const CMIDriverBase &vrOtherDriver);
    virtual FILE *GetStdin(void) const;
    virtual FILE *GetStdout(void) const;
    virtual FILE *GetStderr(void) const;
    virtual const CMIUtilString &GetDriverName(void) const;
    virtual const CMIUtilString &GetDriverId(void) const;
    virtual void DeliverSignal(int signal);

    // Typedefs:
  private:
    typedef std::queue<CMIUtilString> QueueStdinLine_t;

    // Methods:
  private:
    /* ctor */ CMIDriver(void);
    /* ctor */ CMIDriver(const CMIDriver &);
    void operator=(const CMIDriver &);

    lldb::SBError ParseArgs(const int argc, const char *argv[], FILE *vpStdOut, bool &vwbExiting);
    bool DoAppQuit(void);
    bool InterpretCommand(const CMIUtilString &vTextLine);
    bool InterpretCommandThisDriver(const CMIUtilString &vTextLine, bool &vwbCmdYesValid);
    CMIUtilString WrapCLICommandIntoMICommand(const CMIUtilString &vTextLine) const;
    bool InterpretCommandFallThruDriver(const CMIUtilString &vTextLine, bool &vwbCmdYesValid);
    bool ExecuteCommand(const SMICmdData &vCmdData);
    bool StartWorkerThreads(void);
    bool StopWorkerThreads(void);
    bool InitClientIDEToMIDriver(void) const;
    bool InitClientIDEEclipse(void) const;
    bool LocalDebugSessionStartupExecuteCommands(void);
    bool ExecuteCommandFile(const bool vbAsyncMode);

    // Overridden:
  private:
    // From CMICmnBase
    /* dtor */ virtual ~CMIDriver(void);

    // Attributes:
  private:
    static const CMIUtilString ms_constAppNameShort;
    static const CMIUtilString ms_constAppNameLong;
    static const CMIUtilString ms_constMIVersion;
    //
    bool m_bFallThruToOtherDriverEnabled; // True = yes fall through, false = do not pass on command
    CMIUtilThreadMutex m_threadMutex;
    bool m_bDriverIsExiting;           // True = yes, driver told to quit, false = continue working
    void *m_handleMainThread;          // *this driver is run by the main thread
    CMICmnStreamStdin &m_rStdin;
    CMICmnLLDBDebugger &m_rLldbDebugger;
    CMICmnStreamStdout &m_rStdOut;
    DriverState_e m_eCurrentDriverState;
    bool m_bHaveExecutableFileNamePathOnCmdLine; // True = yes, executable given as one of the parameters to the MI Driver, false = not found
    CMIUtilString m_strCmdLineArgExecuteableFileNamePath;
    bool m_bDriverDebuggingArgExecutable; // True = the MI Driver (MI mode) is debugging executable passed as argument,
                                          // false = running via a client (e.g. Eclipse)
    bool m_bHaveCommandFileNamePathOnCmdLine; // True = file with initial commands given as one of the parameters to the MI Driver, false = not found
    CMIUtilString m_strCmdLineArgCommandFileNamePath;
};
