//===-- MICmdBase.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// In-house headers:
#include "MICmdBase.h"
#include "MICmnMIValueConst.h"
#include "MICmnLLDBDebugSessionInfo.h"

//++ ------------------------------------------------------------------------------------
// Details: CMICmdBase constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdBase::CMICmdBase(void)
    : m_pSelfCreatorFn(nullptr)
    , m_rLLDBDebugSessionInfo(CMICmnLLDBDebugSessionInfo::Instance())
    , m_bHasResultRecordExtra(false)
{
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdBase destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdBase::~CMICmdBase(void)
{
}

//++ ------------------------------------------------------------------------------------
// Details: The invoker requires this function.
// Type:    Overridden.
// Args:    None.
// Return:  SMICmdData & -  *this command's present status/data/information.
// Throws:  None.
//--
const SMICmdData &
CMICmdBase::GetCmdData(void) const
{
    return m_cmdData;
}

//++ ------------------------------------------------------------------------------------
// Details: The invoker requires this function.
// Type:    Overridden.
// Args:    None.
// Return:  CMIUtilString & -   *this command's current error description.
//                              Empty string indicates command status ok.
// Throws:  None.
//--
const CMIUtilString &
CMICmdBase::GetErrorDescription(void) const
{
    return m_strCurrentErrDescription;
}

//++ ------------------------------------------------------------------------------------
// Details: The CMICmdFactory requires this function. Retrieve the command and argument
//          options description string.
// Type:    Overridden.
// Args:    None.
// Return:  CMIUtilString & -   Command decription.
// Throws:  None.
//--
const CMIUtilString &
CMICmdBase::GetMiCmd(void) const
{
    return m_strMiCmd;
}

//++ ------------------------------------------------------------------------------------
// Details: The invoker requires this function. A command must be given working data and
//          provide data about its status or provide information to other objects.
// Type:    Overridden.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdBase::SetCmdData(const SMICmdData &vCmdData)
{
    m_cmdData = vCmdData;

    return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: The command factory requires this function. The factory calls this function
//          so it can obtain *this command's creation function.
// Type:    Overridden.
// Args:    None.
// Return:  CMICmdFactory::CmdCreatorFnPtr - Function pointer.
// Throws:  None.
//--
CMICmdFactory::CmdCreatorFnPtr
CMICmdBase::GetCmdCreatorFn(void) const
{
    return m_pSelfCreatorFn;
}

//++ ------------------------------------------------------------------------------------
// Details: If a command is an event type (has callbacks registered with SBListener) it
//          needs to inform the Invoker that it has finished its work so that the
//          Invoker can tidy up and call the commands Acknowledge function (yes the
//          command itself could call the Acknowledge itself but not doing that way).
// Type:    Overridden.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
void
CMICmdBase::CmdFinishedTellInvoker(void) const
{
    CMICmdInvoker::Instance().CmdExecuteFinished(const_cast<CMICmdBase &>(*this));
}

//++ ------------------------------------------------------------------------------------
// Details: Returns the final version of the MI result record built up in the command's
//          Acknowledge function. The one line text of MI result.
// Type:    Overridden.
// Args:    None.
// Return:  CMIUtilString & - MI text version of the MI result record.
// Throws:  None.
//--
const CMIUtilString &
CMICmdBase::GetMIResultRecord(void) const
{
    return m_miResultRecord.GetString();
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve from the command additional MI result to its 1 line response.
//          Because of using LLDB addtional 'fake'/hack output is sometimes required to
//          help the driver client operate i.e. Eclipse.
// Type:    Overridden.
// Args:    None.
// Return:  CMIUtilString & - MI text version of the MI result record.
// Throws:  None.
//--
const CMIUtilString &
CMICmdBase::GetMIResultRecordExtra(void) const
{
    return m_miResultRecordExtra;
}

//++ ------------------------------------------------------------------------------------
// Details: Hss *this command got additional MI result to its 1 line response.
//          Because of using LLDB addtional 'fake'/hack output is sometimes required to
//          help the driver client operate i.e. Eclipse.
// Type:    Overridden.
// Args:    None.
// Return:  bool    - True = Yes have additional MI output, false = no nothing extra.
// Throws:  None.
//--
bool
CMICmdBase::HasMIResultRecordExtra(void) const
{
    return m_bHasResultRecordExtra;
}

//++ ------------------------------------------------------------------------------------
// Details: Short cut function to enter error information into the command's metadata
//          object and set the command's error status.
// Type:    Method.
// Args:    rErrMsg - (R) Error description.
// Return:  None.
// Throws:  None.
//--
void
CMICmdBase::SetError(const CMIUtilString &rErrMsg)
{
    m_cmdData.bCmdValid = false;
    m_cmdData.strErrorDescription = rErrMsg;
    m_cmdData.bCmdExecutedSuccessfully = false;

    const CMICmnMIValueResult valueResult("msg", CMICmnMIValueConst(rErrMsg));
    const CMICmnMIResultRecord miResultRecord(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error, valueResult);
    m_miResultRecord = miResultRecord;
    m_cmdData.strMiCmdResultRecord = miResultRecord.GetString();
}

//++ ------------------------------------------------------------------------------------
// Details: Ask a command to provide its unique identifier.
// Type:    Method.
// Args:    A unique identifier for this command class.
// Return:  None.
// Throws:  None.
//--
MIuint
CMICmdBase::GetGUID(void)
{
    MIuint64 vptr = reinterpret_cast<MIuint64>(this);
    MIuint id = (vptr)&0xFFFFFFFF;
    id ^= (vptr >> 32) & 0xFFFFFFFF;

    return id;
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
CMICmdBase::ParseArgs(void)
{
    // Do nothing - override to implement

    return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: Having previously given CMICmdArgSet m_setCmdArgs all the argument or option
//          definitions for the command to handle proceed to parse and validate the
//          command's options text for those arguments and extract the values for each if
//          any.
// Type:    Method.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdBase::ParseValidateCmdOptions(void)
{
    CMICmdArgContext argCntxt(m_cmdData.strMiCmdOption);
    if (m_setCmdArgs.Validate(m_cmdData.strMiCmd, argCntxt))
        return MIstatus::success;

    SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_ARGS), m_cmdData.strMiCmd.c_str(), m_setCmdArgs.GetErrorDescription().c_str()));

    return MIstatus::failure;
}

//++ ------------------------------------------------------------------------------------
// Details: If the MI Driver is not operating via a client i.e. Eclipse but say operating
//          on a executable passed in as a argument to the drive then what should the driver
//          do on a command failing? Either continue operating or exit the application.
//          Override this function where a command failure cannot allow the driver to
//          continue operating.
// Type:    Overrideable.
// Args:    None.
// Return:  bool - True = Fatal if command fails, false = can continue if command fails.
// Throws:  None.
//--
bool
CMICmdBase::GetExitAppOnCommandFailure(void) const
{
    return false;
}
