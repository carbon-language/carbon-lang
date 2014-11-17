//===-- MICmdCmdVar.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MICmdCmdVar.cpp
//
// Overview:    CMICmdCmdVarCreate                  implementation.
//              CMICmdCmdVarUpdate                  implementation.
//              CMICmdCmdVarDelete                  implementation.
//              CMICmdCmdVarAssign                  implementation.
//              CMICmdCmdVarSetFormat               implementation.
//              CMICmdCmdVarListChildren            implementation.
//              CMICmdCmdVarEvaluateExpression      implementation.
//              CMICmdCmdVarInfoPathExpression      implementation.
//              CMICmdCmdVarShowAttributes          implementation.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--

// Third Party Headers:
#include <lldb/API/SBStream.h>
#include <lldb/API/SBThread.h>

// In-house headers:
#include "MICmdCmdVar.h"
#include "MICmnMIResultRecord.h"
#include "MICmnMIValueConst.h"
#include "MICmnLLDBDebugger.h"
#include "MICmnLLDBDebugSessionInfo.h"
#include "MICmdArgValNumber.h"
#include "MICmdArgValString.h"
#include "MICmdArgValThreadGrp.h"
#include "MICmdArgValOptionLong.h"
#include "MICmdArgValOptionShort.h"
#include "MICmdArgValListOfN.h"
#include "MICmnLLDBProxySBValue.h"
#include "MICmnLLDBUtilSBValue.h"

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdVarCreate constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdVarCreate::CMICmdCmdVarCreate(void)
    : m_nChildren(0)
    , m_nThreadId(0)
    , m_strType("??")
    , m_bValid(false)
    , m_strValue("??")
    , m_constStrArgThread("thread")
    , m_constStrArgThreadGroup("thread-group")
    , m_constStrArgFrame("frame")
    , m_constStrArgName("name")
    , m_constStrArgFrameAddr("frame-addr")
    , m_constStrArgExpression("expression")
{
    // Command factory matches this name with that received from the stdin stream
    m_strMiCmd = "var-create";

    // Required by the CMICmdFactory when registering *this command
    m_pSelfCreatorFn = &CMICmdCmdVarCreate::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdVarCreate destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdVarCreate::~CMICmdCmdVarCreate(void)
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
CMICmdCmdVarCreate::ParseArgs(void)
{
    bool bOk =
        m_setCmdArgs.Add(*(new CMICmdArgValOptionLong(m_constStrArgThread, false, true, CMICmdArgValListBase::eArgValType_Number, 1)));
    bOk = bOk &&
          m_setCmdArgs.Add(
              *(new CMICmdArgValOptionLong(m_constStrArgThreadGroup, false, false, CMICmdArgValListBase::eArgValType_ThreadGrp, 1)));
    bOk = bOk &&
          m_setCmdArgs.Add(*(new CMICmdArgValOptionLong(m_constStrArgFrame, false, true, CMICmdArgValListBase::eArgValType_Number, 1)));
    bOk = bOk && m_setCmdArgs.Add(*(new CMICmdArgValString(m_constStrArgName, false, true)));
    bOk = bOk && m_setCmdArgs.Add(*(new CMICmdArgValString(m_constStrArgFrameAddr, false, true)));
    bOk = bOk && m_setCmdArgs.Add(*(new CMICmdArgValString(m_constStrArgExpression, true, true, true, true)));
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
CMICmdCmdVarCreate::Execute(void)
{
    CMICMDBASE_GETOPTION(pArgThread, OptionLong, m_constStrArgThread);
    CMICMDBASE_GETOPTION(pArgFrame, OptionLong, m_constStrArgFrame);
    CMICMDBASE_GETOPTION(pArgName, String, m_constStrArgName);
    CMICMDBASE_GETOPTION(pArgFrameAddr, String, m_constStrArgFrameAddr);
    CMICMDBASE_GETOPTION(pArgExpression, String, m_constStrArgExpression);

    // Retrieve the --thread option's thread ID (only 1)
    MIuint64 nThreadId = UINT64_MAX;
    if (!pArgThread->GetExpectedOption<CMICmdArgValNumber, MIuint64>(nThreadId))
    {
        SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_OPTION_NOT_FOUND), m_cmdData.strMiCmd.c_str(), m_constStrArgThread.c_str()));
        return MIstatus::failure;
    }
    m_nThreadId = nThreadId;

    // Retrieve the --frame option's number
    MIuint64 nFrame = UINT64_MAX;
    if (!pArgFrame->GetExpectedOption<CMICmdArgValNumber, MIuint64>(nFrame))
    {
        SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_OPTION_NOT_FOUND), m_cmdData.strMiCmd.c_str(), m_constStrArgFrame.c_str()));
        return MIstatus::failure;
    }

    const CMICmdArgValOptionLong::VecArgObjPtr_t &rVecFrameId(pArgFrame->GetExpectedOptions());
    CMICmdArgValOptionLong::VecArgObjPtr_t::const_iterator it2 = rVecFrameId.begin();
    if (it2 != rVecFrameId.end())
    {
        const CMICmdArgValNumber *pOption = static_cast<CMICmdArgValNumber *>(*it2);
        nFrame = pOption->GetValue();
    }

    bool bAutoName = false;
    const CMIUtilString strArgName;
    if (pArgName->GetFound())
    {
        const CMIUtilString &rArg = pArgName->GetValue();
        bAutoName = (rArg == "-");
    }

    const CMIUtilString &rStrExpression(pArgExpression->GetValue());
    m_strExpression = rStrExpression;

    CMICmnLLDBDebugSessionInfo &rSessionInfo(CMICmnLLDBDebugSessionInfo::Instance());
    m_strVarName = "<unnamedvariable>";
    if (bAutoName)
    {
        m_strVarName = CMIUtilString::Format("var%u", CMICmnLLDBDebugSessionInfoVarObj::VarObjIdGet());
        CMICmnLLDBDebugSessionInfoVarObj::VarObjIdInc();
    }
    lldb::SBProcess &rProcess = rSessionInfo.m_lldbProcess;
    lldb::SBThread thread = rProcess.GetThreadByIndexID(nThreadId);
    lldb::SBFrame frame = thread.GetFrameAtIndex(nFrame);
    lldb::SBValue value = frame.FindVariable(rStrExpression.c_str());
    if (!value.IsValid())
        value = frame.EvaluateExpression(rStrExpression.c_str());
    if (value.IsValid())
    {
        m_bValid = true;
        m_nChildren = value.GetNumChildren();
        m_strType = CMICmnLLDBUtilSBValue(value).GetTypeNameDisplay();

        // This gets added to CMICmnLLDBDebugSessionInfoVarObj static container of varObjs
        CMICmnLLDBDebugSessionInfoVarObj varObj(rStrExpression, m_strVarName, value);
        m_strValue = varObj.GetValueFormatted();
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
CMICmdCmdVarCreate::Acknowledge(void)
{
    if (m_bValid)
    {
        // MI print "%s^done,name=\"%s\",numchild=\"%d\",value=\"%s\",type=\"%s\",thread-id=\"%llu\",has_more=\"%u\""
        const CMICmnMIValueConst miValueConst(m_strVarName);
        CMICmnMIValueResult miValueResultAll("name", miValueConst);
        const CMIUtilString strNumChild(CMIUtilString::Format("%d", m_nChildren));
        const CMICmnMIValueConst miValueConst2(strNumChild);
        miValueResultAll.Add("numchild", miValueConst2);
        const CMICmnMIValueConst miValueConst3(m_strValue);
        miValueResultAll.Add("value", miValueConst3);
        const CMICmnMIValueConst miValueConst4(m_strType);
        miValueResultAll.Add("type", miValueConst4);
        const CMIUtilString strThreadId(CMIUtilString::Format("%llu", m_nThreadId));
        const CMICmnMIValueConst miValueConst5(strThreadId);
        miValueResultAll.Add("thread-id", miValueConst5);
        const CMICmnMIValueConst miValueConst6("0");
        miValueResultAll.Add("has_more", miValueConst6);

        const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResultAll);
        m_miResultRecord = miRecordResult;

        return MIstatus::success;
    }

    const CMICmnMIValueConst miValueConst(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_VARIABLE_CREATION_FAILED), m_strExpression.c_str()));
    CMICmnMIValueResult miValueResult("msg", miValueConst);
    const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error, miValueResult);
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
CMICmdCmdVarCreate::CreateSelf(void)
{
    return new CMICmdCmdVarCreate();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdVarUpdate constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdVarUpdate::CMICmdCmdVarUpdate(void)
    : m_constStrArgPrintValues("print-values")
    , m_constStrArgName("name")
    , m_bValueChangedArrayType(false)
    , m_bValueChangedCompositeType(false)
    , m_bValueChangedNormalType(false)
    , m_miValueList(true)
{
    // Command factory matches this name with that received from the stdin stream
    m_strMiCmd = "var-update";

    // Required by the CMICmdFactory when registering *this command
    m_pSelfCreatorFn = &CMICmdCmdVarUpdate::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdVarUpdate destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdVarUpdate::~CMICmdCmdVarUpdate(void)
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
CMICmdCmdVarUpdate::ParseArgs(void)
{
    bool bOk = m_setCmdArgs.Add(*(new CMICmdArgValNumber(m_constStrArgPrintValues, false, false)));
    bOk = bOk && m_setCmdArgs.Add(*(new CMICmdArgValString(m_constStrArgName, true, true)));
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
CMICmdCmdVarUpdate::Execute(void)
{
    CMICMDBASE_GETOPTION(pArgName, String, m_constStrArgName);

    const CMIUtilString &rVarObjName(pArgName->GetValue());
    CMICmnLLDBDebugSessionInfoVarObj varObj;
    if (!CMICmnLLDBDebugSessionInfoVarObj::VarObjGet(rVarObjName, varObj))
    {
        SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_VARIABLE_DOESNOTEXIST), m_cmdData.strMiCmd.c_str(), rVarObjName.c_str()));
        return MIstatus::failure;
    }

    const CMIUtilString &rVarRealName(varObj.GetNameReal());
    MIunused(rVarRealName);
    lldb::SBValue &rValue = const_cast<lldb::SBValue &>(varObj.GetValue());
    const bool bValid = rValue.IsValid();
    if (bValid && rValue.GetValueDidChange())
    {
        m_bValueChangedNormalType = true;
        varObj.UpdateValue();
        m_strValueName = rVarObjName;
        return MIstatus::success;
    }

    // Examine an array type variable
    if (!ExamineSBValueForChange(varObj, false, m_bValueChangedArrayType))
        return MIstatus::failure;

    // Handle composite types i.e. struct or arrays
    const MIuint nChildren = rValue.GetNumChildren();
    for (MIuint i = 0; i < nChildren; i++)
    {
        lldb::SBValue member = rValue.GetChildAtIndex(i);
        if (!member.IsValid())
            continue;

        const CMIUtilString varName(CMIUtilString::Format("%s.%s", rVarObjName.c_str(), member.GetName()));
        if (member.GetValueDidChange())
        {
            // Handle composite
            const CMIUtilString strValue(
                CMICmnLLDBDebugSessionInfoVarObj::GetValueStringFormatted(member, CMICmnLLDBDebugSessionInfoVarObj::eVarFormat_Natural));
            const CMIUtilString strInScope(member.IsInScope() ? "true" : "false");
            MIFormResponse(varName, strValue, strInScope);

            m_bValueChangedCompositeType = true;
        }
        else
        {
            // Handle array of composites
            CMICmnLLDBDebugSessionInfoVarObj varObj;
            if (CMICmnLLDBDebugSessionInfoVarObj::VarObjGet(varName, varObj))
            {
                bool bValueChanged = false;
                if (ExamineSBValueForChange(varObj, true, bValueChanged))
                {
                    if (bValueChanged && CMICmnLLDBDebugSessionInfoVarObj::VarObjGet(varName, varObj))
                    {
                        lldb::SBValue &rValue = const_cast<lldb::SBValue &>(varObj.GetValue());
                        const bool bValid = rValue.IsValid();
                        const CMIUtilString strValue(bValid ? varObj.GetValueFormatted() : "<unknown>");
                        const CMIUtilString strInScope((bValid && rValue.IsInScope()) ? "true" : "false");
                        MIFormResponse(varName, strValue, strInScope);

                        m_bValueChangedCompositeType = true;
                    }
                }
                else
                    return MIstatus::failure;
            }
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
CMICmdCmdVarUpdate::Acknowledge(void)
{
    if (m_bValueChangedArrayType || m_bValueChangedNormalType)
    {
        CMICmnLLDBDebugSessionInfoVarObj varObj;
        CMICmnLLDBDebugSessionInfoVarObj::VarObjGet(m_strValueName, varObj);
        lldb::SBValue &rValue = const_cast<lldb::SBValue &>(varObj.GetValue());
        const bool bValid = rValue.IsValid();
        const CMIUtilString strValue(bValid ? varObj.GetValueFormatted() : "<unknown>");
        const CMIUtilString strInScope((bValid && rValue.IsInScope()) ? "true" : "false");

        // MI print "%s^done,changelist=[{name=\"%s\",value=\"%s\",in_scope=\"%s\",type_changed=\"false\",has_more=\"0\"}]"
        const CMICmnMIValueConst miValueConst(m_strValueName);
        CMICmnMIValueResult miValueResult("name", miValueConst);
        CMICmnMIValueTuple miValueTuple(miValueResult);
        const CMICmnMIValueConst miValueConst2(strValue);
        CMICmnMIValueResult miValueResult2("value", miValueConst2);
        miValueTuple.Add(miValueResult2);
        const CMICmnMIValueConst miValueConst3(strInScope);
        CMICmnMIValueResult miValueResult3("in_scope", miValueConst3);
        miValueTuple.Add(miValueResult3);
        const CMICmnMIValueConst miValueConst4("false");
        CMICmnMIValueResult miValueResult4("type_changed", miValueConst4);
        miValueTuple.Add(miValueResult4);
        const CMICmnMIValueConst miValueConst5("0");
        CMICmnMIValueResult miValueResult5("has_more", miValueConst5);
        miValueTuple.Add(miValueResult5);
        const CMICmnMIValueList miValueList(miValueTuple);
        CMICmnMIValueResult miValueResult6("changelist", miValueList);
        const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult6);
        m_miResultRecord = miRecordResult;

        return MIstatus::success;
    }
    else if (m_bValueChangedCompositeType)
    {
        // MI print
        // "%s^done,changelist=[{name=\"%s\",value=\"%s\",in_scope=\"%s\",type_changed=\"false\",has_more=\"0\"},{name=\"%s\",value=\"%s\",in_scope=\"%s\",type_changed=\"false\",has_more=\"0\"}]"
        CMICmnMIValueResult miValueResult6("changelist", m_miValueList);
        const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult6);
        m_miResultRecord = miRecordResult;
    }
    else
    {
        // MI: "%s^done,changelist=[]"
        const CMICmnMIValueList miValueList(true);
        CMICmnMIValueResult miValueResult6("changelist", miValueList);
        const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult6);
        m_miResultRecord = miRecordResult;
        return MIstatus::success;
    }

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
CMICmdCmdVarUpdate::CreateSelf(void)
{
    return new CMICmdCmdVarUpdate();
}

//++ ------------------------------------------------------------------------------------
// Details: Form the MI response for multiple variables.
// Type:    Method.
// Args:    vrStrVarName    - (R)   Session var object's name.
//          vrStrValue      - (R)   Text version of the value held in the variable.
//          vrStrScope      - (R)   In scope "yes" or "no".
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdCmdVarUpdate::MIFormResponse(const CMIUtilString &vrStrVarName, const CMIUtilString &vrStrValue, const CMIUtilString &vrStrScope)
{
    // MI print "[{name=\"%s\",value=\"%s\",in_scope=\"%s\",type_changed=\"false\",has_more=\"0\"}]"
    const CMICmnMIValueConst miValueConst(vrStrVarName);
    CMICmnMIValueResult miValueResult("name", miValueConst);
    CMICmnMIValueTuple miValueTuple(miValueResult);
    const CMICmnMIValueConst miValueConst2(vrStrValue);
    CMICmnMIValueResult miValueResult2("value", miValueConst2);
    bool bOk = miValueTuple.Add(miValueResult2);
    const CMICmnMIValueConst miValueConst3(vrStrScope);
    CMICmnMIValueResult miValueResult3("in_scope", miValueConst3);
    bOk = bOk && miValueTuple.Add(miValueResult3);
    const CMICmnMIValueConst miValueConst4("false");
    CMICmnMIValueResult miValueResult4("type_changed", miValueConst4);
    bOk = bOk && miValueTuple.Add(miValueResult4);
    const CMICmnMIValueConst miValueConst5("0");
    CMICmnMIValueResult miValueResult5("has_more", miValueConst5);
    bOk = bOk && miValueTuple.Add(miValueResult5);
    bOk = bOk && m_miValueList.Add(miValueTuple);

    return bOk;
}

//++ ------------------------------------------------------------------------------------
// Details: Determine if the var object is a array type variable. LLDB does not 'detect'
//          a value change for some types like elements in an array so have to re-evaluate
//          the expression again.
// Type:    Method.
// Args:    vrVarObj    - (R)   Session var object to examine.
//          vrwbChanged - (W)   True = Is an array type and it changed,
//                              False = Not an array type or not changed.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdCmdVarUpdate::ExamineSBValueForChange(const CMICmnLLDBDebugSessionInfoVarObj &vrVarObj, const bool vbIgnoreVarType, bool &vrwbChanged)
{
    vrwbChanged = false;

    CMICmnLLDBDebugSessionInfo &rSessionInfo(CMICmnLLDBDebugSessionInfo::Instance());
    lldb::SBProcess &rProcess = rSessionInfo.m_lldbProcess;
    lldb::SBThread thread = rProcess.GetSelectedThread();
    if (thread.GetNumFrames() == 0)
    {
        return MIstatus::success;
    }

    const CMIUtilString &strVarObjParentName = vrVarObj.GetVarParentName();
    lldb::SBFrame frame = thread.GetSelectedFrame();
    const CMIUtilString &rExpression(vrVarObj.GetNameReal());
    CMIUtilString varExpression;
    if (strVarObjParentName.empty())
    {
        varExpression = rExpression;
    }
    else
    {
        CMICmnLLDBDebugSessionInfoVarObj varObjParent;
        if (CMICmnLLDBDebugSessionInfoVarObj::VarObjGet(strVarObjParentName, varObjParent))
            varExpression = CMIUtilString::Format("%s.%s", varObjParent.GetNameReal().c_str(), rExpression.c_str());
        else
        {
            // The parent is only assigned in the CMICmdCmdVarListChildren command, we have a problem, need to investigate
            SetError(
                CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_VARIABLE_DOESNOTEXIST), m_cmdData.strMiCmd.c_str(), strVarObjParentName.c_str()));
            return MIstatus::failure;
        }
    }

    lldb::SBValue value = frame.EvaluateExpression(varExpression.c_str());
    if (!value.IsValid())
        value = frame.FindVariable(rExpression.c_str());
    if (value.IsValid())
    {
        lldb::SBType valueType = value.GetType();
        const lldb::BasicType eValueType = valueType.GetBasicType();
        if (vbIgnoreVarType || (eValueType != lldb::BasicType::eBasicTypeInvalid))
        {
            MIuint64 nPrevValue = 0;
            MIuint64 nRevaluateValue = 0;
            lldb::SBValue &rValue = const_cast<lldb::SBValue &>(vrVarObj.GetValue());
            if (CMICmnLLDBProxySBValue::GetValueAsUnsigned(rValue, nPrevValue) &&
                CMICmnLLDBProxySBValue::GetValueAsUnsigned(value, nRevaluateValue) && (nPrevValue != nRevaluateValue))
            {
                // Have a value change so update the var object
                vrwbChanged = true;
                const CMICmnLLDBDebugSessionInfoVarObj varObj(rExpression, vrVarObj.GetName(), value, strVarObjParentName);
            }
        }
    }

    return MIstatus::success;
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdVarDelete constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdVarDelete::CMICmdCmdVarDelete(void)
    : m_constStrArgName("name")
{
    // Command factory matches this name with that received from the stdin stream
    m_strMiCmd = "var-delete";

    // Required by the CMICmdFactory when registering *this command
    m_pSelfCreatorFn = &CMICmdCmdVarDelete::CreateSelf;
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
CMICmdCmdVarDelete::ParseArgs(void)
{
    bool bOk = m_setCmdArgs.Add(*(new CMICmdArgValString(m_constStrArgName, true, true)));
    return (bOk && ParseValidateCmdOptions());
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdVarDelete destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdVarDelete::~CMICmdCmdVarDelete(void)
{
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
CMICmdCmdVarDelete::Execute(void)
{
    CMICMDBASE_GETOPTION(pArgName, String, m_constStrArgName);

    const CMIUtilString &rVarObjName(pArgName->GetValue());
    CMICmnLLDBDebugSessionInfoVarObj::VarObjDelete(rVarObjName);

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
CMICmdCmdVarDelete::Acknowledge(void)
{
    const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done);
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
CMICmdCmdVarDelete::CreateSelf(void)
{
    return new CMICmdCmdVarDelete();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdVarAssign constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdVarAssign::CMICmdCmdVarAssign(void)
    : m_bOk(true)
    , m_constStrArgName("name")
    , m_constStrArgExpression("expression")
{
    // Command factory matches this name with that received from the stdin stream
    m_strMiCmd = "var-assign";

    // Required by the CMICmdFactory when registering *this command
    m_pSelfCreatorFn = &CMICmdCmdVarAssign::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdVarAssign destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdVarAssign::~CMICmdCmdVarAssign(void)
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
CMICmdCmdVarAssign::ParseArgs(void)
{
    bool bOk = m_setCmdArgs.Add(*(new CMICmdArgValString(m_constStrArgName, true, true)));
    bOk = bOk && m_setCmdArgs.Add(*(new CMICmdArgValString(m_constStrArgExpression, true, true)));
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
CMICmdCmdVarAssign::Execute(void)
{
    CMICMDBASE_GETOPTION(pArgName, String, m_constStrArgName);
    CMICMDBASE_GETOPTION(pArgExpression, String, m_constStrArgExpression);

    const CMIUtilString &rVarObjName(pArgName->GetValue());
    const CMIUtilString &rExpression(pArgExpression->GetValue());

    CMICmnLLDBDebugSessionInfoVarObj varObj;
    if (!CMICmnLLDBDebugSessionInfoVarObj::VarObjGet(rVarObjName, varObj))
    {
        SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_VARIABLE_DOESNOTEXIST), m_cmdData.strMiCmd.c_str(), rVarObjName.c_str()));
        return MIstatus::failure;
    }
    m_varObjName = rVarObjName;

    CMIUtilString strExpression(rExpression.Trim());
    strExpression = strExpression.Trim('"');
    lldb::SBValue &rValue(const_cast<lldb::SBValue &>(varObj.GetValue()));
    m_bOk = rValue.SetValueFromCString(strExpression.c_str());
    if (m_bOk)
        varObj.UpdateValue();

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
CMICmdCmdVarAssign::Acknowledge(void)
{
    if (m_bOk)
    {
        // MI print "%s^done,value=\"%s\""
        CMICmnLLDBDebugSessionInfoVarObj varObj;
        CMICmnLLDBDebugSessionInfoVarObj::VarObjGet(m_varObjName, varObj);
        const CMICmnMIValueConst miValueConst(varObj.GetValueFormatted());
        const CMICmnMIValueResult miValueResult("value", miValueConst);
        const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult);
        m_miResultRecord = miRecordResult;

        return MIstatus::success;
    }

    const CMICmnMIValueConst miValueConst("expression could not be evaluated");
    const CMICmnMIValueResult miValueResult("msg", miValueConst);
    const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error, miValueResult);
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
CMICmdCmdVarAssign::CreateSelf(void)
{
    return new CMICmdCmdVarAssign();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdVarSetFormat constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdVarSetFormat::CMICmdCmdVarSetFormat(void)
    : m_constStrArgName("name")
    , m_constStrArgFormatSpec("format-spec")
{
    // Command factory matches this name with that received from the stdin stream
    m_strMiCmd = "var-set-format";

    // Required by the CMICmdFactory when registering *this command
    m_pSelfCreatorFn = &CMICmdCmdVarSetFormat::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdVarSetFormat destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdVarSetFormat::~CMICmdCmdVarSetFormat(void)
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
CMICmdCmdVarSetFormat::ParseArgs(void)
{
    bool bOk = m_setCmdArgs.Add(*(new CMICmdArgValString(m_constStrArgName, true, true)));
    bOk = bOk && m_setCmdArgs.Add(*(new CMICmdArgValString(m_constStrArgFormatSpec, true, true)));
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
CMICmdCmdVarSetFormat::Execute(void)
{
    CMICMDBASE_GETOPTION(pArgName, String, m_constStrArgName);
    CMICMDBASE_GETOPTION(pArgFormatSpec, String, m_constStrArgFormatSpec);

    const CMIUtilString &rVarObjName(pArgName->GetValue());
    const CMIUtilString &rExpression(pArgFormatSpec->GetValue());

    CMICmnLLDBDebugSessionInfoVarObj varObj;
    if (!CMICmnLLDBDebugSessionInfoVarObj::VarObjGet(rVarObjName, varObj))
    {
        SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_VARIABLE_DOESNOTEXIST), m_cmdData.strMiCmd.c_str(), rVarObjName.c_str()));
        return MIstatus::failure;
    }
    if (!varObj.SetVarFormat(CMICmnLLDBDebugSessionInfoVarObj::GetVarFormatForString(rExpression)))
    {
        SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_VARIABLE_ENUM_INVALID), m_cmdData.strMiCmd.c_str(), rVarObjName.c_str(),
                                       rExpression.c_str()));
        return MIstatus::failure;
    }
    varObj.UpdateValue();

    m_varObjName = rVarObjName;

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
CMICmdCmdVarSetFormat::Acknowledge(void)
{
    // MI print "%s^done,changelist=[{name=\"%s\",value=\"%s\",in_scope=\"%s\",type_changed=\"false\",has_more=\"0\"}]"
    CMICmnLLDBDebugSessionInfoVarObj varObj;
    CMICmnLLDBDebugSessionInfoVarObj::VarObjGet(m_varObjName, varObj);
    const CMICmnMIValueConst miValueConst(m_varObjName);
    const CMICmnMIValueResult miValueResult("name", miValueConst);
    CMICmnMIValueTuple miValueTuple(miValueResult);
    const CMICmnMIValueConst miValueConst2(varObj.GetValueFormatted());
    const CMICmnMIValueResult miValueResult2("value", miValueConst2);
    miValueTuple.Add(miValueResult2);
    lldb::SBValue &rValue = const_cast<lldb::SBValue &>(varObj.GetValue());
    const CMICmnMIValueConst miValueConst3(rValue.IsInScope() ? "true" : "false");
    const CMICmnMIValueResult miValueResult3("in_scope", miValueConst3);
    miValueTuple.Add(miValueResult3);
    const CMICmnMIValueConst miValueConst4("false");
    const CMICmnMIValueResult miValueResult4("type_changed", miValueConst4);
    miValueTuple.Add(miValueResult4);
    const CMICmnMIValueConst miValueConst5("0");
    const CMICmnMIValueResult miValueResult5("type_changed", miValueConst5);
    miValueTuple.Add(miValueResult5);
    const CMICmnMIValueList miValueList(miValueTuple);
    const CMICmnMIValueResult miValueResult6("changelist", miValueList);

    const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult6);
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
CMICmdCmdVarSetFormat::CreateSelf(void)
{
    return new CMICmdCmdVarSetFormat();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdVarListChildren constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdVarListChildren::CMICmdCmdVarListChildren(void)
    : m_bValueValid(false)
    , m_nChildren(0)
    , m_constStrArgPrintValues("print-values")
    , m_constStrArgName("name")
{
    // Command factory matches this name with that received from the stdin stream
    m_strMiCmd = "var-list-children";

    // Required by the CMICmdFactory when registering *this command
    m_pSelfCreatorFn = &CMICmdCmdVarListChildren::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdVarListChildren destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdVarListChildren::~CMICmdCmdVarListChildren(void)
{
    m_vecMiValueResult.clear();
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
CMICmdCmdVarListChildren::ParseArgs(void)
{
    bool bOk = m_setCmdArgs.Add(*(new CMICmdArgValNumber(m_constStrArgPrintValues, false, false)));
    bOk = bOk && m_setCmdArgs.Add(*(new CMICmdArgValString(m_constStrArgName, true, true)));
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
CMICmdCmdVarListChildren::Execute(void)
{
    CMICMDBASE_GETOPTION(pArgName, String, m_constStrArgName);

    const CMIUtilString &rVarObjName(pArgName->GetValue());
    CMICmnLLDBDebugSessionInfoVarObj varObj;
    if (!CMICmnLLDBDebugSessionInfoVarObj::VarObjGet(rVarObjName, varObj))
    {
        SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_VARIABLE_DOESNOTEXIST), m_cmdData.strMiCmd.c_str(), rVarObjName.c_str()));
        return MIstatus::failure;
    }

    lldb::SBValue &rValue = const_cast<lldb::SBValue &>(varObj.GetValue());
    m_bValueValid = rValue.IsValid();
    if (!m_bValueValid)
        return MIstatus::success;

    m_vecMiValueResult.clear();
    m_nChildren = rValue.GetNumChildren();
    for (MIuint i = 0; i < m_nChildren; i++)
    {
        lldb::SBValue member = rValue.GetChildAtIndex(i);
        if (!member.IsValid())
            continue;
        const CMICmnLLDBUtilSBValue utilValue(member);
        const CMIUtilString strExp = utilValue.GetName();
        const CMIUtilString name(CMIUtilString::Format("%s.%s", rVarObjName.c_str(), strExp.c_str()));
        const MIuint nChildren = member.GetNumChildren();
        const CMIUtilString strThreadId(CMIUtilString::Format("%u", member.GetThread().GetIndexID()));

        // Varobj gets added to CMICmnLLDBDebugSessionInfoVarObj static container of varObjs
        CMICmnLLDBDebugSessionInfoVarObj var(strExp, name, member, rVarObjName);

        // MI print "child={name=\"%s\",exp=\"%s\",numchild=\"%d\",value=\"%s\",type=\"%s\",thread-id=\"%u\",has_more=\"%u\"}"
        const CMICmnMIValueConst miValueConst(name);
        const CMICmnMIValueResult miValueResult("name", miValueConst);
        CMICmnMIValueTuple miValueTuple(miValueResult);
        const CMICmnMIValueConst miValueConst2(strExp);
        const CMICmnMIValueResult miValueResult2("exp", miValueConst2);
        miValueTuple.Add(miValueResult2);
        const CMIUtilString strNumChild(CMIUtilString::Format("%d", nChildren));
        const CMICmnMIValueConst miValueConst3(strNumChild);
        const CMICmnMIValueResult miValueResult3("numchild", miValueConst3);
        miValueTuple.Add(miValueResult3);
        const CMICmnMIValueConst miValueConst5(utilValue.GetTypeNameDisplay());
        const CMICmnMIValueResult miValueResult5("type", miValueConst5);
        miValueTuple.Add(miValueResult5);
        const CMICmnMIValueConst miValueConst6(strThreadId);
        const CMICmnMIValueResult miValueResult6("thread-id", miValueConst6);
        miValueTuple.Add(miValueResult6);
        const CMICmnMIValueConst miValueConst7("0");
        const CMICmnMIValueResult miValueResult7("has_more", miValueConst7);
        miValueTuple.Add(miValueResult7);
        const CMICmnMIValueResult miValueResult8("child", miValueTuple);
        m_vecMiValueResult.push_back(miValueResult8);
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
CMICmdCmdVarListChildren::Acknowledge(void)
{
    if (m_bValueValid)
    {
        // MI print "%s^done,numchild=\"%u\",children=[]""
        const CMIUtilString strNumChild(CMIUtilString::Format("%u", m_nChildren));
        const CMICmnMIValueConst miValueConst(strNumChild);
        CMICmnMIValueResult miValueResult("numchild", miValueConst);

        VecMIValueResult_t::const_iterator it = m_vecMiValueResult.begin();
        if (it == m_vecMiValueResult.end())
        {
            const CMICmnMIValueConst miValueConst("[]");
            miValueResult.Add("children", miValueConst);
        }
        else
        {
            CMICmnMIValueList miValueList(*it);
            ++it;
            while (it != m_vecMiValueResult.end())
            {
                const CMICmnMIValueResult &rResult(*it);
                miValueList.Add(rResult);

                // Next
                ++it;
            }
            miValueResult.Add("children", miValueList);
        }

        const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult);
        m_miResultRecord = miRecordResult;
        return MIstatus::success;
    }

    // MI print "%s^done,numchild=\"0\""
    const CMICmnMIValueConst miValueConst("0");
    const CMICmnMIValueResult miValueResult("numchild", miValueConst);
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
CMICmdCmdVarListChildren::CreateSelf(void)
{
    return new CMICmdCmdVarListChildren();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdVarEvaluateExpression constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdVarEvaluateExpression::CMICmdCmdVarEvaluateExpression(void)
    : m_bValueValid(true)
    , m_constStrArgFormatSpec("-f")
    , m_constStrArgName("name")
{
    // Command factory matches this name with that received from the stdin stream
    m_strMiCmd = "var-evaluate-expression";

    // Required by the CMICmdFactory when registering *this command
    m_pSelfCreatorFn = &CMICmdCmdVarEvaluateExpression::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdVarEvaluateExpression destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdVarEvaluateExpression::~CMICmdCmdVarEvaluateExpression(void)
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
CMICmdCmdVarEvaluateExpression::ParseArgs(void)
{
    bool bOk = m_setCmdArgs.Add(
        *(new CMICmdArgValOptionShort(m_constStrArgFormatSpec, false, false, CMICmdArgValListBase::eArgValType_String, 1)));
    bOk = bOk && m_setCmdArgs.Add(*(new CMICmdArgValString(m_constStrArgName, true, true)));
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
CMICmdCmdVarEvaluateExpression::Execute(void)
{
    CMICMDBASE_GETOPTION(pArgName, String, m_constStrArgName);

    const CMIUtilString &rVarObjName(pArgName->GetValue());
    CMICmnLLDBDebugSessionInfoVarObj varObj;
    if (!CMICmnLLDBDebugSessionInfoVarObj::VarObjGet(rVarObjName, varObj))
    {
        SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_VARIABLE_DOESNOTEXIST), m_cmdData.strMiCmd.c_str(), rVarObjName.c_str()));
        return MIstatus::failure;
    }

    lldb::SBValue &rValue = const_cast<lldb::SBValue &>(varObj.GetValue());
    m_bValueValid = rValue.IsValid();
    if (!m_bValueValid)
        return MIstatus::success;

    m_varObjName = rVarObjName;
    varObj.UpdateValue();

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
CMICmdCmdVarEvaluateExpression::Acknowledge(void)
{
    if (m_bValueValid)
    {
        CMICmnLLDBDebugSessionInfoVarObj varObj;
        CMICmnLLDBDebugSessionInfoVarObj::VarObjGet(m_varObjName, varObj);
        const CMICmnMIValueConst miValueConst(varObj.GetValueFormatted());
        const CMICmnMIValueResult miValueResult("value", miValueConst);
        const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult);
        m_miResultRecord = miRecordResult;
        return MIstatus::success;
    }

    const CMICmnMIValueConst miValueConst("variable invalid");
    const CMICmnMIValueResult miValueResult("msg", miValueConst);
    const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error, miValueResult);
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
CMICmdCmdVarEvaluateExpression::CreateSelf(void)
{
    return new CMICmdCmdVarEvaluateExpression();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdVarInfoPathExpression constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdVarInfoPathExpression::CMICmdCmdVarInfoPathExpression(void)
    : m_bValueValid(true)
    , m_constStrArgName("name")
{
    // Command factory matches this name with that received from the stdin stream
    m_strMiCmd = "var-info-path-expression";

    // Required by the CMICmdFactory when registering *this command
    m_pSelfCreatorFn = &CMICmdCmdVarInfoPathExpression::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdVarInfoPathExpression destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdVarInfoPathExpression::~CMICmdCmdVarInfoPathExpression(void)
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
CMICmdCmdVarInfoPathExpression::ParseArgs(void)
{
    bool bOk = m_setCmdArgs.Add(*(new CMICmdArgValString(m_constStrArgName, true, true)));
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
CMICmdCmdVarInfoPathExpression::Execute(void)
{
    CMICMDBASE_GETOPTION(pArgName, String, m_constStrArgName);

    const CMIUtilString &rVarObjName(pArgName->GetValue());
    CMICmnLLDBDebugSessionInfoVarObj varObj;
    if (!CMICmnLLDBDebugSessionInfoVarObj::VarObjGet(rVarObjName, varObj))
    {
        SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_VARIABLE_DOESNOTEXIST), m_cmdData.strMiCmd.c_str(), rVarObjName.c_str()));
        return MIstatus::failure;
    }

    lldb::SBValue &rValue = const_cast<lldb::SBValue &>(varObj.GetValue());
    m_bValueValid = rValue.IsValid();
    if (!m_bValueValid)
        return MIstatus::success;

    lldb::SBStream stream;
    if (!rValue.GetExpressionPath(stream, true))
    {
        SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_VARIABLE_EXPRESSIONPATH), m_cmdData.strMiCmd.c_str(), rVarObjName.c_str()));
        return MIstatus::failure;
    }

    const MIchar *pPathExpression = stream.GetData();
    if (pPathExpression == nullptr)
    {
        // Build expression from what we do know
        m_strPathExpression = varObj.GetNameReal();
        return MIstatus::success;
    }

    // Has LLDB returned a var signature of it's own
    if (pPathExpression[0] != '$')
    {
        m_strPathExpression = pPathExpression;
        return MIstatus::success;
    }

    // Build expression from what we do know
    const CMIUtilString &rVarParentName(varObj.GetVarParentName());
    if (rVarParentName.empty())
    {
        m_strPathExpression = varObj.GetNameReal();
    }
    else
    {
        CMICmnLLDBDebugSessionInfoVarObj varObjParent;
        if (!CMICmnLLDBDebugSessionInfoVarObj::VarObjGet(rVarParentName, varObjParent))
        {
            SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_VARIABLE_DOESNOTEXIST), m_cmdData.strMiCmd.c_str(), rVarParentName.c_str()));
            return MIstatus::failure;
        }
        m_strPathExpression = CMIUtilString::Format("%s.%s", varObjParent.GetNameReal().c_str(), varObj.GetNameReal().c_str());
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
CMICmdCmdVarInfoPathExpression::Acknowledge(void)
{
    if (m_bValueValid)
    {
        const CMICmnMIValueConst miValueConst(m_strPathExpression);
        const CMICmnMIValueResult miValueResult("path_expr", miValueConst);
        const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult);
        m_miResultRecord = miRecordResult;
        return MIstatus::success;
    }

    const CMICmnMIValueConst miValueConst("variable invalid");
    const CMICmnMIValueResult miValueResult("msg", miValueConst);
    const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error, miValueResult);
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
CMICmdCmdVarInfoPathExpression::CreateSelf(void)
{
    return new CMICmdCmdVarInfoPathExpression();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdVarShowAttributes constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdVarShowAttributes::CMICmdCmdVarShowAttributes(void)
    : m_constStrArgName("name")
{
    // Command factory matches this name with that received from the stdin stream
    m_strMiCmd = "var-show-attributes";

    // Required by the CMICmdFactory when registering *this command
    m_pSelfCreatorFn = &CMICmdCmdVarShowAttributes::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdVarShowAttributes destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdVarShowAttributes::~CMICmdCmdVarShowAttributes(void)
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
CMICmdCmdVarShowAttributes::ParseArgs(void)
{
    bool bOk = m_setCmdArgs.Add(*(new CMICmdArgValString(m_constStrArgName, true, true)));
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
CMICmdCmdVarShowAttributes::Execute(void)
{
    CMICMDBASE_GETOPTION(pArgName, String, m_constStrArgName);

    const CMIUtilString &rVarObjName(pArgName->GetValue());
    CMICmnLLDBDebugSessionInfoVarObj varObj;
    if (CMICmnLLDBDebugSessionInfoVarObj::VarObjGet(rVarObjName, varObj))
    {
        SetError(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_VARIABLE_DOESNOTEXIST), m_cmdData.strMiCmd.c_str(), rVarObjName.c_str()));
        return MIstatus::failure;
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
CMICmdCmdVarShowAttributes::Acknowledge(void)
{
    // MI output: "%s^done,status=\"editable\"]"
    const CMICmnMIValueConst miValueConst("editable");
    const CMICmnMIValueResult miValueResult("status", miValueConst);
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
CMICmdCmdVarShowAttributes::CreateSelf(void)
{
    return new CMICmdCmdVarShowAttributes();
}
