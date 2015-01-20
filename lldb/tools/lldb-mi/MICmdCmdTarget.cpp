//===-- MICmdCmdTarget.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MICmdCmdTarget.cpp
//
// Overview:    CMICmdCmdTargetSelect           implementation.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--

// Third Party Headers:
#include "lldb/API/SBStream.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"

// In-house headers:
#include "MICmdCmdTarget.h"
#include "MICmnMIResultRecord.h"
#include "MICmnMIValueConst.h"
#include "MICmnMIOutOfBandRecord.h"
#include "MICmnLLDBDebugger.h"
#include "MICmnLLDBDebugSessionInfo.h"
#include "MICmdArgValString.h"

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdTargetSelect constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdTargetSelect::CMICmdCmdTargetSelect(void)
    : m_constStrArgNamedType("type")
    , m_constStrArgNamedParameters("parameters")
{
    // Command factory matches this name with that received from the stdin stream
    m_strMiCmd = "target-select";

    // Required by the CMICmdFactory when registering *this command
    m_pSelfCreatorFn = &CMICmdCmdTargetSelect::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdTargetSelect destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdTargetSelect::~CMICmdCmdTargetSelect(void)
{
}

//++ ------------------------------------------------------------------------------------
// Details: The invoker requires this function. The parses the command line options
//          arguments to extract values for each of those arguments.
// Type:    Overridden.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdCmdTargetSelect::ParseArgs(void)
{
    bool bOk = m_setCmdArgs.Add(*(new CMICmdArgValString(m_constStrArgNamedType, true, true)));
    bOk = bOk && m_setCmdArgs.Add(*(new CMICmdArgValString(m_constStrArgNamedParameters, true, true)));
    return (bOk && ParseValidateCmdOptions());
}

//++ ------------------------------------------------------------------------------------
// Details: The invoker requires this function. The command does work in this function.
//          The command is likely to communicate with the LLDB SBDebugger in here.
//          Synopsis: -target-select type parameters ...
//          Ref: http://sourceware.org/gdb/onlinedocs/gdb/GDB_002fMI-Target-Manipulation.html#GDB_002fMI-Target-Manipulation
// Type:    Overridden.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdCmdTargetSelect::Execute(void)
{
    CMICMDBASE_GETOPTION(pArgType, String, m_constStrArgNamedType);
    CMICMDBASE_GETOPTION(pArgParameters, String, m_constStrArgNamedParameters);

    CMICmnLLDBDebugSessionInfo &rSessionInfo(CMICmnLLDBDebugSessionInfo::Instance());

    // Check we have a valid target
    // Note: target created via 'file-exec-and-symbols' command
    if (!rSessionInfo.m_lldbTarget.IsValid())
    {
        SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_INVALID_TARGET_CURRENT), m_cmdData.strMiCmd.c_str()));
        return MIstatus::failure;
    }

    // Verify that we are executing remotely
    const CMIUtilString &rRemoteType(pArgType->GetValue());
    if (rRemoteType != "remote")
    {
        SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_INVALID_TARGET_TYPE), m_cmdData.strMiCmd.c_str(), rRemoteType.c_str()));
        return MIstatus::failure;
    }

    // Create a URL pointing to the remote gdb stub
    const CMIUtilString strUrl = CMIUtilString::Format("connect://%s", pArgParameters->GetValue().c_str());

    // Ask LLDB to collect to the target port
    const MIchar *pPlugin("gdb-remote");
    lldb::SBError error;
    lldb::SBProcess process = rSessionInfo.m_lldbTarget.ConnectRemote(rSessionInfo.m_rLlldbListener, strUrl.c_str(), pPlugin, error);

    // Verify that we have managed to connect successfully
    lldb::SBStream errMsg;
    if (!process.IsValid())
    {
        SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_INVALID_TARGET_PLUGIN), m_cmdData.strMiCmd.c_str(), errMsg.GetData()));
        return MIstatus::failure;
    }
    if (error.Fail())
    {
        SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_CONNECT_TO_TARGET), m_cmdData.strMiCmd.c_str(), errMsg.GetData()));
        return MIstatus::failure;
    }

    // Save the process in the session info
    // Note: Order is important here since this process handle may be used by CMICmnLLDBDebugHandleEvents
    //       which can fire when interpreting via HandleCommand() below.
    rSessionInfo.m_lldbProcess = process;

    // Set the environment path if we were given one
    CMIUtilString strWkDir;
    if (rSessionInfo.SharedDataRetrieve<CMIUtilString>(rSessionInfo.m_constStrSharedDataKeyWkDir, strWkDir))
    {
        lldb::SBDebugger &rDbgr = rSessionInfo.m_rLldbDebugger;
        if (!rDbgr.SetCurrentPlatformSDKRoot(strWkDir.c_str()))
        {
            SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_FNFAILED), m_cmdData.strMiCmd.c_str(), "target-select"));
            return MIstatus::failure;
        }
    }

    // Set the shared object path if we were given one
    CMIUtilString strSolibPath;
    if (rSessionInfo.SharedDataRetrieve<CMIUtilString>(rSessionInfo.m_constStrSharedDataSolibPath, strSolibPath))
    {
        lldb::SBDebugger &rDbgr = rSessionInfo.m_rLldbDebugger;
        lldb::SBCommandInterpreter cmdIterpreter = rDbgr.GetCommandInterpreter();

        CMIUtilString strCmdString = CMIUtilString::Format("target modules search-paths add . %s", strSolibPath.c_str());

        lldb::SBCommandReturnObject retObj;
        cmdIterpreter.HandleCommand(strCmdString.c_str(), retObj, false);

        if (!retObj.Succeeded())
        {
            SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_FNFAILED), m_cmdData.strMiCmd.c_str(), "target-select"));
            return MIstatus::failure;
        }
    }

    return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: The invoker requires this function. The command prepares a MI Record Result
//          for the work carried out in the Execute().
// Type:    Overridden.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdCmdTargetSelect::Acknowledge(void)
{
    const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Connected);
    m_miResultRecord = miRecordResult;

    CMICmnLLDBDebugSessionInfo &rSessionInfo(CMICmnLLDBDebugSessionInfo::Instance());
    lldb::pid_t pid = rSessionInfo.m_lldbProcess.GetProcessID();
    // Prod the client i.e. Eclipse with out-of-band results to help it 'continue' because it is using LLDB debugger
    // Give the client '=thread-group-started,id="i1"'
    m_bHasResultRecordExtra = true;
    const CMICmnMIValueConst miValueConst2("i1");
    const CMICmnMIValueResult miValueResult2("id", miValueConst2);
    const CMIUtilString strPid(CMIUtilString::Format("%lld", pid));
    const CMICmnMIValueConst miValueConst(strPid);
    const CMICmnMIValueResult miValueResult("pid", miValueConst);
    CMICmnMIOutOfBandRecord miOutOfBand(CMICmnMIOutOfBandRecord::eOutOfBand_ThreadGroupStarted, miValueResult2);
    miOutOfBand.Add(miValueResult);
    m_miResultRecordExtra = miOutOfBand.GetString();

    return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: Required by the CMICmdFactory when registering *this command. The factory
//          calls this function to create an instance of *this command.
// Type:    Static method.
// Args:    None.
// Return:  CMICmdBase * - Pointer to a new command.
// Throws:  None.
//--
CMICmdBase *
CMICmdCmdTargetSelect::CreateSelf(void)
{
    return new CMICmdCmdTargetSelect();
}
