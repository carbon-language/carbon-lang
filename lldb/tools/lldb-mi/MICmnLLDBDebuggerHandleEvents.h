//===-- MICmnLLDBDebuggerHandleEvents.h -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MICmnLLDBDebuggerHandleEvents.h
//
// Overview:    CMICmnLLDBDebuggerHandleEvents interface.
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
#include "MICmnMIValueTuple.h"
#include "MIUtilSingletonBase.h"

// Declarations:
class CMICmnLLDBDebugSessionInfo;
class CMICmnMIResultRecord;
class CMICmnStreamStdout;
class CMICmnMIOutOfBandRecord;

//++ ============================================================================
// Details: MI class to take LLDB SBEvent objects, filter them and form
//          MI Out-of-band records from the information inside the event object.
//          These records are then pushed to stdout.
//          A singleton class.
// Gotchas: None.
// Authors: Illya Rudkin 02/03/2014.
// Changes: None.
//--
class CMICmnLLDBDebuggerHandleEvents : public CMICmnBase, public MI::ISingleton<CMICmnLLDBDebuggerHandleEvents>
{
    friend class MI::ISingleton<CMICmnLLDBDebuggerHandleEvents>;

    // Methods:
  public:
    bool Initialize(void);
    bool Shutdown(void);
    //
    bool HandleEvent(const lldb::SBEvent &vEvent, bool &vrbHandledEvent, bool &vrbExitAppEvent);

    // Methods:
  private:
    /* ctor */ CMICmnLLDBDebuggerHandleEvents(void);
    /* ctor */ CMICmnLLDBDebuggerHandleEvents(const CMICmnLLDBDebuggerHandleEvents &);
    void operator=(const CMICmnLLDBDebuggerHandleEvents &);
    //
    bool ChkForStateChanges(void);
    bool GetProcessStdout(void);
    bool GetProcessStderr(void);
    bool HandleEventSBBreakPoint(const lldb::SBEvent &vEvent);
    bool HandleEventSBBreakpointCmn(const lldb::SBEvent &vEvent);
    bool HandleEventSBBreakpointAdded(const lldb::SBEvent &vEvent);
    bool HandleEventSBBreakpointLocationsAdded(const lldb::SBEvent &vEvent);
    bool HandleEventSBProcess(const lldb::SBEvent &vEvent, bool &vrbExitAppEvent);
    bool HandleEventSBThread(const lldb::SBEvent &vEvent);
    bool HandleEventSBThreadBitStackChanged(const lldb::SBEvent &vEvent);
    bool HandleEventSBThreadSuspended(const lldb::SBEvent &vEvent);
    bool HandleEventSBCommandInterpreter(const lldb::SBEvent &vEvent);
    bool HandleProcessEventBroadcastBitStateChanged(const lldb::SBEvent &vEvent, bool &vrbExitAppEvent);
    bool HandleProcessEventStateRunning(void);
    bool HandleProcessEventStateExited(void);
    bool HandleProcessEventStateStopped(bool &vwrbShouldBrk);
    bool HandleProcessEventStopReasonTrace(void);
    bool HandleProcessEventStopReasonBreakpoint(void);
    bool HandleProcessEventStopSignal(bool &vwrbShouldBrk);
    bool HandleProcessEventStateSuspended(const lldb::SBEvent &vEvent);
    bool MiHelpGetCurrentThreadFrame(CMICmnMIValueTuple &vwrMiValueTuple);
    bool MiResultRecordToStdout(const CMICmnMIResultRecord &vrMiResultRecord);
    bool MiOutOfBandRecordToStdout(const CMICmnMIOutOfBandRecord &vrMiResultRecord);
    bool MiStoppedAtBreakPoint(const MIuint64 vBrkPtId, const lldb::SBBreakpoint &vBrkPt);
    bool TextToStdout(const CMIUtilString &vrTxt);
    bool TextToStderr(const CMIUtilString &vrTxt);
    bool UpdateSelectedThread(void);
    bool ConvertPrintfCtrlCodeToString(const MIchar vCtrl, CMIUtilString &vwrStrEquivalent);

    // Overridden:
  private:
    // From CMICmnBase
    /* dtor */ virtual ~CMICmnLLDBDebuggerHandleEvents(void);
};
