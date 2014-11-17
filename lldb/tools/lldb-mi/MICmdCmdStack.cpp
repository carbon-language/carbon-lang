//===-- MICmdCmdStack.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MICmdCmdStack.cpp
//
// Overview:    CMICmdCmdStackInfoDepth         implementation.
//              CMICmdCmdStackListFrames        implementation.
//              CMICmdCmdStackListArguments     implementation.
//              CMICmdCmdStackListLocals        implementation.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--

// Third Party Headers:
#include <lldb/API/SBThread.h>

// In-house headers:
#include "MICmdCmdStack.h"
#include "MICmnMIResultRecord.h"
#include "MICmnMIValueConst.h"
#include "MICmnMIOutOfBandRecord.h"
#include "MICmnLLDBDebugger.h"
#include "MICmnLLDBDebugSessionInfo.h"
#include "MICmdArgValNumber.h"
#include "MICmdArgValString.h"
#include "MICmdArgValThreadGrp.h"
#include "MICmdArgValOptionLong.h"
#include "MICmdArgValOptionShort.h"
#include "MICmdArgValListOfN.h"

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdStackInfoDepth constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdStackInfoDepth::CMICmdCmdStackInfoDepth(void)
    : m_nThreadFrames(0)
    , m_constStrArgThread("thread")
    , m_constStrArgMaxDepth("max-depth")
{
    // Command factory matches this name with that received from the stdin stream
    m_strMiCmd = "stack-info-depth";

    // Required by the CMICmdFactory when registering *this command
    m_pSelfCreatorFn = &CMICmdCmdStackInfoDepth::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdStackInfoDepth destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdStackInfoDepth::~CMICmdCmdStackInfoDepth(void)
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
CMICmdCmdStackInfoDepth::ParseArgs(void)
{
    bool bOk =
        m_setCmdArgs.Add(*(new CMICmdArgValOptionLong(m_constStrArgThread, true, true, CMICmdArgValListBase::eArgValType_Number, 1)));
    bOk = bOk && m_setCmdArgs.Add(*(new CMICmdArgValNumber(m_constStrArgMaxDepth, false, false)));
    return (bOk && ParseValidateCmdOptions());
}

//++ ------------------------------------------------------------------------------------
// Details: The invoker requires this function. The command does work in this function.
//          The command is likely to communicate with the LLDB SBDebugger in here.
// Type:    Overridden.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdCmdStackInfoDepth::Execute(void)
{
    CMICMDBASE_GETOPTION(pArgThread, OptionLong, m_constStrArgThread);
    CMICMDBASE_GETOPTION(pArgMaxDepth, Number, m_constStrArgMaxDepth);

    // Retrieve the --thread option's thread ID (only 1)
    MIuint64 nThreadId = UINT64_MAX;
    if (!pArgThread->GetExpectedOption<CMICmdArgValNumber, MIuint64>(nThreadId))
    {
        SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_OPTION_NOT_FOUND), m_cmdData.strMiCmd.c_str(), m_constStrArgThread.c_str()));
        return MIstatus::failure;
    }

    CMICmnLLDBDebugSessionInfo &rSessionInfo(CMICmnLLDBDebugSessionInfo::Instance());
    lldb::SBProcess &rProcess = rSessionInfo.m_lldbProcess;
    lldb::SBThread thread = (nThreadId != UINT64_MAX) ? rProcess.GetThreadByIndexID(nThreadId) : rProcess.GetSelectedThread();
    m_nThreadFrames = thread.GetNumFrames();

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
CMICmdCmdStackInfoDepth::Acknowledge(void)
{
    const CMIUtilString strDepth(CMIUtilString::Format("%d", m_nThreadFrames));
    const CMICmnMIValueConst miValueConst(strDepth);
    const CMICmnMIValueResult miValueResult("depth", miValueConst);
    const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult);
    m_miResultRecord = miRecordResult;

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
CMICmdCmdStackInfoDepth::CreateSelf(void)
{
    return new CMICmdCmdStackInfoDepth();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdStackListFrames constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdStackListFrames::CMICmdCmdStackListFrames(void)
    : m_nThreadFrames(0)
    , m_constStrArgThread("thread")
    , m_constStrArgFrameLow("low-frame")
    , m_constStrArgFrameHigh("high-frame")
{
    // Command factory matches this name with that received from the stdin stream
    m_strMiCmd = "stack-list-frames";

    // Required by the CMICmdFactory when registering *this command
    m_pSelfCreatorFn = &CMICmdCmdStackListFrames::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdStackListFrames destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdStackListFrames::~CMICmdCmdStackListFrames(void)
{
    m_vecMIValueResult.clear();
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
CMICmdCmdStackListFrames::ParseArgs(void)
{
    bool bOk =
        m_setCmdArgs.Add(*(new CMICmdArgValOptionLong(m_constStrArgThread, true, true, CMICmdArgValListBase::eArgValType_Number, 1)));
    bOk = bOk && m_setCmdArgs.Add(*(new CMICmdArgValNumber(m_constStrArgFrameLow, false, true)));
    bOk = bOk && m_setCmdArgs.Add(*(new CMICmdArgValNumber(m_constStrArgFrameHigh, false, true)));
    return (bOk && ParseValidateCmdOptions());
}

//++ ------------------------------------------------------------------------------------
// Details: The invoker requires this function. The command does work in this function.
//          The command is likely to communicate with the LLDB SBDebugger in here.
// Type:    Overridden.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdCmdStackListFrames::Execute(void)
{
    CMICMDBASE_GETOPTION(pArgThread, OptionLong, m_constStrArgThread);
    CMICMDBASE_GETOPTION(pArgFrameLow, Number, m_constStrArgFrameLow);
    CMICMDBASE_GETOPTION(pArgFrameHigh, Number, m_constStrArgFrameHigh);

    // Retrieve the --thread option's thread ID (only 1)
    MIuint64 nThreadId = UINT64_MAX;
    if (!pArgThread->GetExpectedOption<CMICmdArgValNumber, MIuint64>(nThreadId))
    {
        SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_OPTION_NOT_FOUND), m_cmdData.strMiCmd.c_str(), m_constStrArgThread.c_str()));
        return MIstatus::failure;
    }

    // Frame low and high options are not mandatory
    MIuint nFrameHigh = pArgFrameHigh->GetFound() ? pArgFrameHigh->GetValue() : UINT32_MAX;
    const MIuint nFrameLow = pArgFrameLow->GetFound() ? pArgFrameLow->GetValue() : 0;

    CMICmnLLDBDebugSessionInfo &rSessionInfo(CMICmnLLDBDebugSessionInfo::Instance());
    lldb::SBProcess &rProcess = rSessionInfo.m_lldbProcess;
    lldb::SBThread thread = (nThreadId != UINT64_MAX) ? rProcess.GetThreadByIndexID(nThreadId) : rProcess.GetSelectedThread();
    MIuint nThreadFrames = thread.GetNumFrames();

    // Adjust nThreadFrames for the nFrameHigh argument as we use nFrameHigh+1 in the min calc as the arg
    // is not an index, but a frame id value.
    if (nFrameHigh < UINT32_MAX)
    {
        nFrameHigh++;
        nThreadFrames = (nFrameHigh < nThreadFrames) ? nFrameHigh : nThreadFrames;
    }

    m_nThreadFrames = nThreadFrames;
    if (nThreadFrames == 0)
        return MIstatus::success;

    m_vecMIValueResult.clear();
    for (MIuint nLevel = nFrameLow; nLevel < nThreadFrames; nLevel++)
    {
        CMICmnMIValueTuple miValueTuple;
        if (!rSessionInfo.MIResponseFormFrameInfo(thread, nLevel, miValueTuple))
            return MIstatus::failure;

        const CMICmnMIValueResult miValueResult8("frame", miValueTuple);
        m_vecMIValueResult.push_back(miValueResult8);
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
CMICmdCmdStackListFrames::Acknowledge(void)
{
    if (m_nThreadFrames == 0)
    {
        // MI print "3^done,stack=[{}]"
        const CMICmnMIValueTuple miValueTuple;
        const CMICmnMIValueList miValueList(miValueTuple);
        const CMICmnMIValueResult miValueResult("stack", miValueList);
        const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult);
        m_miResultRecord = miRecordResult;

        return MIstatus::success;
    }

    // Build up a list of thread information from tuples
    VecMIValueResult_t::const_iterator it = m_vecMIValueResult.begin();
    if (it == m_vecMIValueResult.end())
    {
        // MI print "3^done,stack=[{}]"
        const CMICmnMIValueTuple miValueTuple;
        const CMICmnMIValueList miValueList(miValueTuple);
        const CMICmnMIValueResult miValueResult("stack", miValueList);
        const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult);
        m_miResultRecord = miRecordResult;
        return MIstatus::success;
    }
    CMICmnMIValueList miValueList(*it);
    ++it;
    while (it != m_vecMIValueResult.end())
    {
        const CMICmnMIValueResult &rTuple(*it);
        miValueList.Add(rTuple);

        // Next
        ++it;
    }
    const CMICmnMIValueResult miValueResult("stack", miValueList);
    const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult);
    m_miResultRecord = miRecordResult;

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
CMICmdCmdStackListFrames::CreateSelf(void)
{
    return new CMICmdCmdStackListFrames();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdStackListArguments constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdStackListArguments::CMICmdCmdStackListArguments(void)
    : m_bThreadInvalid(false)
    , m_miValueList(true)
    , m_constStrArgThread("thread")
    , m_constStrArgPrintValues("print-values")
{
    // Command factory matches this name with that received from the stdin stream
    m_strMiCmd = "stack-list-arguments";

    // Required by the CMICmdFactory when registering *this command
    m_pSelfCreatorFn = &CMICmdCmdStackListArguments::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdStackListArguments destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdStackListArguments::~CMICmdCmdStackListArguments(void)
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
CMICmdCmdStackListArguments::ParseArgs(void)
{
    bool bOk =
        m_setCmdArgs.Add(*(new CMICmdArgValOptionLong(m_constStrArgThread, false, true, CMICmdArgValListBase::eArgValType_Number, 1)));
    bOk = bOk && m_setCmdArgs.Add(*(new CMICmdArgValNumber(m_constStrArgPrintValues, true, false)));
    return (bOk && ParseValidateCmdOptions());
}

//++ ------------------------------------------------------------------------------------
// Details: The invoker requires this function. The command does work in this function.
//          The command is likely to communicate with the LLDB SBDebugger in here.
// Type:    Overridden.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdCmdStackListArguments::Execute(void)
{
    CMICMDBASE_GETOPTION(pArgThread, OptionLong, m_constStrArgThread);
    CMICMDBASE_GETOPTION(pArgPrintValues, Number, m_constStrArgPrintValues);

    // Retrieve the --thread option's thread ID (only 1)
    MIuint64 nThreadId = UINT64_MAX;
    if (pArgThread->GetFound())
    {
        if (!pArgThread->GetExpectedOption<CMICmdArgValNumber, MIuint64>(nThreadId))
        {
            SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_OPTION_NOT_FOUND), m_cmdData.strMiCmd.c_str(), m_constStrArgThread.c_str()));
            return MIstatus::failure;
        }
    }

    CMICmnLLDBDebugSessionInfo &rSessionInfo(CMICmnLLDBDebugSessionInfo::Instance());
    lldb::SBProcess &rProcess = rSessionInfo.m_lldbProcess;
    lldb::SBThread thread = (nThreadId != UINT64_MAX) ? rProcess.GetThreadByIndexID(nThreadId) : rProcess.GetSelectedThread();
    m_bThreadInvalid = !thread.IsValid();
    if (m_bThreadInvalid)
        return MIstatus::success;

    const lldb::StopReason eStopReason = thread.GetStopReason();
    if ((eStopReason == lldb::eStopReasonNone) || (eStopReason == lldb::eStopReasonInvalid))
    {
        m_bThreadInvalid = true;
        return MIstatus::success;
    }

    const MIuint nFrames = thread.GetNumFrames();
    for (MIuint i = 0; i < nFrames; i++)
    {
        lldb::SBFrame frame = thread.GetFrameAtIndex(i);
        CMICmnMIValueList miValueList(true);
        const MIuint maskVarTypes = 0x1000;
        if (!rSessionInfo.MIResponseFormVariableInfo3(frame, maskVarTypes, miValueList))
            return MIstatus::failure;
        const CMICmnMIValueConst miValueConst(CMIUtilString::Format("%d", i));
        const CMICmnMIValueResult miValueResult("level", miValueConst);
        CMICmnMIValueTuple miValueTuple(miValueResult);
        const CMICmnMIValueResult miValueResult2("args", miValueList);
        miValueTuple.Add(miValueResult2);
        const CMICmnMIValueResult miValueResult3("frame", miValueTuple);
        m_miValueList.Add(miValueResult3);
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
CMICmdCmdStackListArguments::Acknowledge(void)
{
    if (m_bThreadInvalid)
    {
        // MI print "%s^done,stack-args=[]"
        const CMICmnMIValueList miValueList(true);
        const CMICmnMIValueResult miValueResult("stack-args", miValueList);
        const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult);
        m_miResultRecord = miRecordResult;
        return MIstatus::success;
    }

    // MI print "%s^done,stack-args=[frame={level=\"0\",args=[%s]},frame={level=\"1\",args=[%s]}]"
    const CMICmnMIValueResult miValueResult4("stack-args", m_miValueList);
    const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult4);
    m_miResultRecord = miRecordResult;

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
CMICmdCmdStackListArguments::CreateSelf(void)
{
    return new CMICmdCmdStackListArguments();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdStackListLocals constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdStackListLocals::CMICmdCmdStackListLocals(void)
    : m_bThreadInvalid(false)
    , m_miValueList(true)
    , m_constStrArgThread("thread")
    , m_constStrArgFrame("frame")
    , m_constStrArgPrintValues("print-values")
{
    // Command factory matches this name with that received from the stdin stream
    m_strMiCmd = "stack-list-locals";

    // Required by the CMICmdFactory when registering *this command
    m_pSelfCreatorFn = &CMICmdCmdStackListLocals::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdStackListLocals destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdStackListLocals::~CMICmdCmdStackListLocals(void)
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
CMICmdCmdStackListLocals::ParseArgs(void)
{
    bool bOk =
        m_setCmdArgs.Add(*(new CMICmdArgValOptionLong(m_constStrArgThread, false, true, CMICmdArgValListBase::eArgValType_Number, 1)));
    bOk = bOk &&
          m_setCmdArgs.Add(*(new CMICmdArgValOptionLong(m_constStrArgFrame, false, true, CMICmdArgValListBase::eArgValType_Number, 1)));
    bOk = bOk && m_setCmdArgs.Add(*(new CMICmdArgValNumber(m_constStrArgPrintValues, true, false)));
    return (bOk && ParseValidateCmdOptions());
}

//++ ------------------------------------------------------------------------------------
// Details: The invoker requires this function. The command does work in this function.
//          The command is likely to communicate with the LLDB SBDebugger in here.
// Type:    Overridden.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdCmdStackListLocals::Execute(void)
{
    CMICMDBASE_GETOPTION(pArgThread, OptionLong, m_constStrArgThread);
    CMICMDBASE_GETOPTION(pArgFrame, OptionLong, m_constStrArgFrame);

    // Retrieve the --thread option's thread ID (only 1)
    MIuint64 nThreadId = UINT64_MAX;
    if (pArgThread->GetFound())
    {
        if (!pArgThread->GetExpectedOption<CMICmdArgValNumber, MIuint64>(nThreadId))
        {
            SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_OPTION_NOT_FOUND), m_cmdData.strMiCmd.c_str(), m_constStrArgThread.c_str()));
            return MIstatus::failure;
        }
    }
    MIuint64 nFrame = UINT64_MAX;
    if (pArgFrame->GetFound())
    {
        if (!pArgFrame->GetExpectedOption<CMICmdArgValNumber, MIuint64>(nFrame))
        {
            SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_OPTION_NOT_FOUND), m_cmdData.strMiCmd.c_str(), m_constStrArgFrame.c_str()));
            return MIstatus::failure;
        }
    }

    CMICmnLLDBDebugSessionInfo &rSessionInfo(CMICmnLLDBDebugSessionInfo::Instance());
    lldb::SBProcess &rProcess = rSessionInfo.m_lldbProcess;
    lldb::SBThread thread = (nThreadId != UINT64_MAX) ? rProcess.GetThreadByIndexID(nThreadId) : rProcess.GetSelectedThread();
    m_bThreadInvalid = !thread.IsValid();
    if (m_bThreadInvalid)
        return MIstatus::success;

    const lldb::StopReason eStopReason = thread.GetStopReason();
    if ((eStopReason == lldb::eStopReasonNone) || (eStopReason == lldb::eStopReasonInvalid))
    {
        m_bThreadInvalid = true;
        return MIstatus::success;
    }

    const MIuint nFrames = thread.GetNumFrames();
    MIunused(nFrames);
    lldb::SBFrame frame = (nFrame != UINT64_MAX) ? thread.GetFrameAtIndex(nFrame) : thread.GetSelectedFrame();
    CMICmnMIValueList miValueList(true);
    const MIuint maskVarTypes = 0x0110;
    if (!rSessionInfo.MIResponseFormVariableInfo(frame, maskVarTypes, miValueList))
        return MIstatus::failure;

    m_miValueList = miValueList;

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
CMICmdCmdStackListLocals::Acknowledge(void)
{
    if (m_bThreadInvalid)
    {
        // MI print "%s^done,locals=[]"
        const CMICmnMIValueList miValueList(true);
        const CMICmnMIValueResult miValueResult("locals", miValueList);
        const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult);
        m_miResultRecord = miRecordResult;
        return MIstatus::success;
    }

    // MI print "%s^done,locals=[%s]"
    const CMICmnMIValueResult miValueResult("locals", m_miValueList);
    const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult);
    m_miResultRecord = miRecordResult;

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
CMICmdCmdStackListLocals::CreateSelf(void)
{
    return new CMICmdCmdStackListLocals();
}
