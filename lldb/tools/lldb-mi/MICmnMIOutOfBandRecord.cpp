//===-- MICmnMIOutOfBandRecord.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// In-house headers:
#include "MICmnMIOutOfBandRecord.h"
#include "MICmnResources.h"

// Instantiations:
CMICmnMIOutOfBandRecord::MapOutOfBandToOutOfBandText_t ms_MapOutOfBandToOutOfBandText = {
    {CMICmnMIOutOfBandRecord::eOutOfBand_Running, "running"},
    {CMICmnMIOutOfBandRecord::eOutOfBand_Stopped, "stopped"},
    {CMICmnMIOutOfBandRecord::eOutOfBand_BreakPointCreated, "breakpoint-created"},
    {CMICmnMIOutOfBandRecord::eOutOfBand_BreakPointModified, "breakpoint-modified"},
    {CMICmnMIOutOfBandRecord::eOutOfBand_Thread, ""}, // "" Meant to be empty
    {CMICmnMIOutOfBandRecord::eOutOfBand_ThreadGroupAdded, "thread-group-added"},
    {CMICmnMIOutOfBandRecord::eOutOfBand_ThreadGroupExited, "thread-group-exited"},
    {CMICmnMIOutOfBandRecord::eOutOfBand_ThreadGroupRemoved, "thread-group-removed"},
    {CMICmnMIOutOfBandRecord::eOutOfBand_ThreadGroupStarted, "thread-group-started"},
    {CMICmnMIOutOfBandRecord::eOutOfBand_ThreadCreated, "thread-created"},
    {CMICmnMIOutOfBandRecord::eOutOfBand_ThreadExited, "thread-exited"},
    {CMICmnMIOutOfBandRecord::eOutOfBand_ThreadSelected, "thread-selected"},
    {CMICmnMIOutOfBandRecord::eOutOfBand_TargetModuleLoaded, "library-loaded"},
    {CMICmnMIOutOfBandRecord::eOutOfBand_TargetModuleUnloaded, "library-unloaded"},
    {CMICmnMIOutOfBandRecord::eOutOfBand_TargetStreamOutput, ""}};
CMICmnMIOutOfBandRecord::MapOutOfBandToOutOfBandText_t ms_constMapAsyncRecordTextToToken = {
    {CMICmnMIOutOfBandRecord::eOutOfBand_Running, "*"},
    {CMICmnMIOutOfBandRecord::eOutOfBand_Stopped, "*"},
    {CMICmnMIOutOfBandRecord::eOutOfBand_BreakPointCreated, "="},
    {CMICmnMIOutOfBandRecord::eOutOfBand_BreakPointModified, "="},
    {CMICmnMIOutOfBandRecord::eOutOfBand_Thread, "@"},
    {CMICmnMIOutOfBandRecord::eOutOfBand_ThreadGroupAdded, "="},
    {CMICmnMIOutOfBandRecord::eOutOfBand_ThreadGroupExited, "="},
    {CMICmnMIOutOfBandRecord::eOutOfBand_ThreadGroupRemoved, "="},
    {CMICmnMIOutOfBandRecord::eOutOfBand_ThreadGroupStarted, "="},
    {CMICmnMIOutOfBandRecord::eOutOfBand_ThreadCreated, "="},
    {CMICmnMIOutOfBandRecord::eOutOfBand_ThreadExited, "="},
    {CMICmnMIOutOfBandRecord::eOutOfBand_ThreadSelected, "="},
    {CMICmnMIOutOfBandRecord::eOutOfBand_TargetModuleLoaded, "="},
    {CMICmnMIOutOfBandRecord::eOutOfBand_TargetModuleUnloaded, "="},
    {CMICmnMIOutOfBandRecord::eOutOfBand_TargetStreamOutput, "@"}};

//++ ------------------------------------------------------------------------------------
// Details: CMICmnMIOutOfBandRecord constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmnMIOutOfBandRecord::CMICmnMIOutOfBandRecord(void)
    : m_strAsyncRecord(MIRSRC(IDS_CMD_ERR_EVENT_HANDLED_BUT_NO_ACTION))
{
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmnMIOutOfBandRecord constructor.
// Type:    Method.
// Args:    veType      - (R) A MI Out-of-Bound enumeration.
// Return:  None.
// Throws:  None.
//--
CMICmnMIOutOfBandRecord::CMICmnMIOutOfBandRecord(const OutOfBand_e veType)
    : m_eResultAsyncRecordClass(veType)
    , m_strAsyncRecord(MIRSRC(IDS_CMD_ERR_EVENT_HANDLED_BUT_NO_ACTION))
{
    BuildAsyncRecord();
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmnMIOutOfBandRecord constructor.
// Type:    Method.
// Args:    veType      - (R) A MI Out-of-Bound enumeration.
//          vConst      - (R) A MI const object.
// Return:  None.
// Throws:  None.
//--
CMICmnMIOutOfBandRecord::CMICmnMIOutOfBandRecord(const OutOfBand_e veType, const CMICmnMIValueConst &vConst)
    : m_eResultAsyncRecordClass(veType)
    , m_strAsyncRecord(MIRSRC(IDS_CMD_ERR_EVENT_HANDLED_BUT_NO_ACTION))
{
    BuildAsyncRecord();
    m_strAsyncRecord += vConst.GetString();
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmnMIOutOfBandRecord constructor.
// Type:    Method.
// Args:    veType      - (R) A MI Out-of-Bound enumeration.
//          vResult     - (R) A MI result object.
// Return:  None.
// Throws:  None.
//--
CMICmnMIOutOfBandRecord::CMICmnMIOutOfBandRecord(const OutOfBand_e veType, const CMICmnMIValueResult &vResult)
    : m_eResultAsyncRecordClass(veType)
    , m_strAsyncRecord(MIRSRC(IDS_CMD_ERR_EVENT_HANDLED_BUT_NO_ACTION))
    , m_partResult(vResult)
{
    BuildAsyncRecord();
    Add(m_partResult);
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmnMIOutOfBandRecord destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmnMIOutOfBandRecord::~CMICmnMIOutOfBandRecord(void)
{
}

//++ ------------------------------------------------------------------------------------
// Details: Return the MI Out-of-band record as a string. The string is a direct result of
//          work done on *this Out-of-band record so if not enough data is added then it is
//          possible to return a malformed Out-of-band record. If nothing has been set or
//          added to *this MI Out-of-band record object then text "<Invalid>" will be returned.
// Type:    Method.
// Args:    None.
// Return:  CMIUtilString & - MI output text.
// Throws:  None.
//--
const CMIUtilString &
CMICmnMIOutOfBandRecord::GetString(void) const
{
    return m_strAsyncRecord;
}

//++ ------------------------------------------------------------------------------------
// Details: Build the Out-of-band record's mandatory data part. The part up to the first
//          (additional) result i.e. async-record ==>  "*" type.
// Type:    Method.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmnMIOutOfBandRecord::BuildAsyncRecord(void)
{
    const char *pFormat = "%s%s";
    const CMIUtilString &rStrAsyncRecord(ms_MapOutOfBandToOutOfBandText[m_eResultAsyncRecordClass]);
    const CMIUtilString &rStrToken(ms_constMapAsyncRecordTextToToken[m_eResultAsyncRecordClass]);
    m_strAsyncRecord = CMIUtilString::Format(pFormat, rStrToken.c_str(), rStrAsyncRecord.c_str());

    return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: Add to *this Out-of-band record additional information.
// Type:    Method.
// Args:    vResult           - (R) A MI result object.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmnMIOutOfBandRecord::Add(const CMICmnMIValueResult &vResult)
{
    m_strAsyncRecord += ",";
    m_strAsyncRecord += vResult.GetString();

    return MIstatus::success;
}
