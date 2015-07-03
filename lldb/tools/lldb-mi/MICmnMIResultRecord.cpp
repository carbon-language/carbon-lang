//===-- Platform.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// In-house headers:
#include "MICmnMIResultRecord.h"
#include "MICmnResources.h"

// Instantiations:
CMICmnMIResultRecord::MapResultClassToResultClassText_t ms_MapResultClassToResultClassText = {
    {CMICmnMIResultRecord::eResultClass_Done, "done"},
    {CMICmnMIResultRecord::eResultClass_Running, "running"},
    {CMICmnMIResultRecord::eResultClass_Connected, "connected"},
    {CMICmnMIResultRecord::eResultClass_Error, "error"},
    {CMICmnMIResultRecord::eResultClass_Exit, "exit"}};
const CMIUtilString CMICmnMIResultRecord::ms_constStrResultRecordHat("^");

//++ ------------------------------------------------------------------------------------
// Details: CMICmnMIResultRecord constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmnMIResultRecord::CMICmnMIResultRecord(void)
    : m_strResultRecord(MIRSRC(IDS_CMD_ERR_CMD_RUN_BUT_NO_ACTION))
{
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmnMIResultRecord constructor.
// Type:    Method.
// Args:    vrToken - (R) The command's transaction ID or token.
//          veType  - (R) A MI result class enumeration.
// Return:  None.
// Throws:  None.
//--
CMICmnMIResultRecord::CMICmnMIResultRecord(const CMIUtilString &vrToken, const ResultClass_e veType)
    : m_strResultRecordToken(vrToken)
    , m_eResultRecordResultClass(veType)
    , m_strResultRecord(MIRSRC(IDS_CMD_ERR_CMD_RUN_BUT_NO_ACTION))
{
    BuildResultRecord();
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmnMIResultRecord constructor.
// Type:    Method.
// Args:    vrToken     - (R) The command's transaction ID or token.
//          veType      - (R) A MI result class enumeration.
//          vMIResult   - (R) A MI result object.
// Return:  None.
// Throws:  None.
//--
CMICmnMIResultRecord::CMICmnMIResultRecord(const CMIUtilString &vrToken, const ResultClass_e veType, const CMICmnMIValueResult &vValue)
    : m_strResultRecordToken(vrToken)
    , m_eResultRecordResultClass(veType)
    , m_strResultRecord(MIRSRC(IDS_CMD_ERR_CMD_RUN_BUT_NO_ACTION))
    , m_partResult(vValue)
{
    BuildResultRecord();
    Add(m_partResult);
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmnMIResultRecord destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmnMIResultRecord::~CMICmnMIResultRecord(void)
{
}

//++ ------------------------------------------------------------------------------------
// Details: Return the MI result record as a string. The string is a direct result of
//          work done on *this result record so if not enough data is added then it is
//          possible to return a malformed result record. If nothing has been set or
//          added to *this MI result record object then text "<Invalid>" will be returned.
// Type:    Method.
// Args:    None.
// Return:  CMIUtilString & - MI output text.
// Throws:  None.
//--
const CMIUtilString &
CMICmnMIResultRecord::GetString(void) const
{
    return m_strResultRecord;
}

//++ ------------------------------------------------------------------------------------
// Details: Build the result record's mandatory data part. The part up to the first
//          (additional) result i.e. result-record ==>  [ token ] "^" result-class.
// Type:    Method.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmnMIResultRecord::BuildResultRecord(void)
{
    const char *pFormat = "%s%s%s";
    const CMIUtilString &rStrResultRecord(ms_MapResultClassToResultClassText[m_eResultRecordResultClass]);
    m_strResultRecord =
        CMIUtilString::Format(pFormat, m_strResultRecordToken.c_str(), ms_constStrResultRecordHat.c_str(), rStrResultRecord.c_str());

    return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: Add to *this result record additional information.
// Type:    Method.
// Args:    vMIValue    - (R) A MI value derived object.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmnMIResultRecord::Add(const CMICmnMIValue &vMIValue)
{
    m_strResultRecord += ",";
    m_strResultRecord += vMIValue.GetString();

    return MIstatus::success;
}
