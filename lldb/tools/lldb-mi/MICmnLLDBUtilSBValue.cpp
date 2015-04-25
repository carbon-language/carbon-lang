//===-- MICmnLLDBUtilSBValue.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// In-house headers:
#include "MICmnLLDBUtilSBValue.h"
#include "MICmnLLDBDebugSessionInfo.h"
#include "MICmnMIValueConst.h"
#include "MICmnMIValueTuple.h"
#include "MIUtilString.h"

//++ ------------------------------------------------------------------------------------
// Details: CMICmnLLDBUtilSBValue constructor.
// Type:    Method.
// Args:    vrValue             - (R) The LLDb value object.
//          vbHandleCharType    - (R) True = Yes return text molding to char type,
//                                    False = just return data.
// Return:  None.
// Throws:  None.
//--
CMICmnLLDBUtilSBValue::CMICmnLLDBUtilSBValue(const lldb::SBValue &vrValue, const bool vbHandleCharType /* = false */,
                                             const bool vbHandleArrayType /* = true */)
    : m_rValue(const_cast<lldb::SBValue &>(vrValue))
    , m_pUnkwn("??")
    , m_pComposite("{...}")
    , m_bHandleCharType(vbHandleCharType)
    , m_bHandleArrayType(vbHandleArrayType)
{
    m_bValidSBValue = m_rValue.IsValid();
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmnLLDBUtilSBValue destructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmnLLDBUtilSBValue::~CMICmnLLDBUtilSBValue(void)
{
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve from the LLDB SB Value object the name of the variable. If the name
//          is invalid (or the SBValue object invalid) then "??" is returned.
// Type:    Method.
// Args:    None.
// Return:  CMIUtilString   - Name of the variable or "??" for unknown.
// Throws:  None.
//--
CMIUtilString
CMICmnLLDBUtilSBValue::GetName(void) const
{
    const MIchar *pName = m_bValidSBValue ? m_rValue.GetName() : nullptr;
    const CMIUtilString text((pName != nullptr) ? pName : m_pUnkwn);

    return text;
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve from the LLDB SB Value object the value of the variable described in
//          text. If the value is invalid (or the SBValue object invalid) then "??" is
//          returned.
// Type:    Method.
// Args:    None.
// Return:  CMIUtilString   - Text description of the variable's value or "??".
// Throws:  None.
//--
CMIUtilString
CMICmnLLDBUtilSBValue::GetValue(const bool vbExpandAggregates /* = false */) const
{
    if (!m_bValidSBValue)
        return m_pUnkwn;

    CMICmnLLDBDebugSessionInfo &rSessionInfo(CMICmnLLDBDebugSessionInfo::Instance());
    bool bPrintExpandAggregates = false;
    bPrintExpandAggregates = rSessionInfo.SharedDataRetrieve<bool>(rSessionInfo.m_constStrPrintExpandAggregates,
                                                                   bPrintExpandAggregates) && bPrintExpandAggregates;

    const bool bHandleArrayTypeAsSimple = m_bHandleArrayType && !vbExpandAggregates && !bPrintExpandAggregates;
    CMIUtilString value;
    const bool bIsSimpleValue = GetSimpleValue(bHandleArrayTypeAsSimple, value);
    if (bIsSimpleValue)
        return value;

    if (!vbExpandAggregates && !bPrintExpandAggregates)
        return m_pComposite;

    bool bPrintAggregateFieldNames = false;
    bPrintAggregateFieldNames = !rSessionInfo.SharedDataRetrieve<bool>(rSessionInfo.m_constStrPrintAggregateFieldNames,
                                                                       bPrintAggregateFieldNames) || bPrintAggregateFieldNames;

    CMICmnMIValueTuple miValueTuple;
    const bool bOk = GetCompositeValue(bPrintAggregateFieldNames, miValueTuple);
    if (!bOk)
        return m_pUnkwn;

    value = miValueTuple.GetString();
    return value;
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve from the LLDB SB Value object the value of the variable described in
//          text if it has a simple format (not composite).
// Type:    Method.
// Args:    vwrValue          - (W) The SBValue in a string format.
// Return:  MIstatus::success - Function succeeded.
//          MIstatus::failure - Function failed.
// Throws:  None.
//--
bool
CMICmnLLDBUtilSBValue::GetSimpleValue(const bool vbHandleArrayType, CMIUtilString &vwrValue) const
{
    const MIuint nChildren = m_rValue.GetNumChildren();
    if (nChildren == 0)
    {
        if (m_bHandleCharType && IsCharType())
        {
            const uint8_t value = m_rValue.GetValueAsUnsigned();
            const CMIUtilString prefix(CMIUtilString::Format("%c", value).Escape().AddSlashes());
            vwrValue = CMIUtilString::Format("%hhu '%s'", value, prefix.c_str());
            return MIstatus::success;
        }
        else
        {
            const MIchar *pValue = m_bValidSBValue ? m_rValue.GetValue() : nullptr;
            vwrValue = pValue != nullptr ? pValue : m_pUnkwn;
            return MIstatus::success;
        }
    }
    else if (IsPointerType())
    {
        if (m_bHandleCharType && IsFirstChildCharType())
        {
            const MIchar *pValue = m_bValidSBValue ? m_rValue.GetValue() : nullptr;
            const CMIUtilString value = pValue != nullptr ? pValue : m_pUnkwn;
            const CMIUtilString prefix(GetChildValueCString().Escape().AddSlashes());
            // Note code that has const in will not show the text suffix to the string pointer
            // i.e. const char * pMyStr = "blah"; ==> "0x00007000"" <-- Eclipse shows this
            // but        char * pMyStr = "blah"; ==> "0x00007000" "blah"" <-- Eclipse shows this
            vwrValue = CMIUtilString::Format("%s \"%s\"", value.c_str(), prefix.c_str());
            return MIstatus::success;
        }
        else
        {
            const MIchar *pValue = m_bValidSBValue ? m_rValue.GetValue() : nullptr;
            vwrValue = pValue != nullptr ? pValue : m_pUnkwn;
            return MIstatus::success;
        }
    }
    else if (IsArrayType())
    {
        CMICmnLLDBDebugSessionInfo &rSessionInfo(CMICmnLLDBDebugSessionInfo::Instance());
        bool bPrintCharArrayAsString = false;
        bPrintCharArrayAsString = rSessionInfo.SharedDataRetrieve<bool>(rSessionInfo.m_constStrPrintCharArrayAsString,
                                                                        bPrintCharArrayAsString) && bPrintCharArrayAsString;
        if (bPrintCharArrayAsString && m_bHandleCharType && IsFirstChildCharType())
        {
            // TODO: to match char* it should be the following
            //       vwrValue = CMIUtilString::Format("[%u] \"%s\"", nChildren, prefix.c_str());
            const CMIUtilString prefix(GetValueCString().Escape().AddSlashes());
            vwrValue = CMIUtilString::Format("\"%s\"", prefix.c_str());
            return MIstatus::success;
        }
        else if (vbHandleArrayType)
        {
            vwrValue = CMIUtilString::Format("[%u]", nChildren);
            return MIstatus::success;
        }
    }

    // Composite variable type i.e. struct
    return MIstatus::failure;
}

bool
CMICmnLLDBUtilSBValue::GetCompositeValue(const bool vbPrintFieldNames, CMICmnMIValueTuple &vwrMiValueTuple,
                                         const MIuint vnDepth /* = 1 */) const
{
    const MIuint nMaxDepth = 10;
    const MIuint nChildren = m_rValue.GetNumChildren();
    for (MIuint i = 0; i < nChildren; ++i)
    {
        const lldb::SBValue member = m_rValue.GetChildAtIndex(i);
        const CMICmnLLDBUtilSBValue utilMember(member, m_bHandleCharType, m_bHandleArrayType);
        const bool bHandleArrayTypeAsSimple = false;
        CMIUtilString value;
        const bool bIsSimpleValue = utilMember.GetSimpleValue(bHandleArrayTypeAsSimple, value);
        if (bIsSimpleValue)
        {
            // OK. Value is simple (not composite) and was successfully got
        }
        else if (vnDepth < nMaxDepth)
        {
            // Need to get value from composite type
            CMICmnMIValueTuple miValueTuple;
            const bool bOk = utilMember.GetCompositeValue(vbPrintFieldNames, miValueTuple, vnDepth + 1);
            if (!bOk)
                // Can't obtain composite type
                value = m_pUnkwn;
            else
                // OK. Value is composite and was successfully got
                value = miValueTuple.GetString();
        }
        else
        {
            // Need to get value from composite type, but vnMaxDepth is reached
            value = m_pComposite;
        }
        const bool bNoQuotes = true;
        const CMICmnMIValueConst miValueConst(value, bNoQuotes);
        if (vbPrintFieldNames)
        {
            const bool bUseSpacing = true;
            const CMICmnMIValueResult miValueResult(utilMember.GetName(), miValueConst, bUseSpacing);
            const bool bOk = vwrMiValueTuple.Add(miValueResult, bUseSpacing);
            if (!bOk)
                return MIstatus::failure;
        }
        else
        {
            const bool bUseSpacing = false;
            const bool bOk = vwrMiValueTuple.Add(miValueConst, bUseSpacing);
            if (!bOk)
                return MIstatus::failure;
        }
    }

    return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: If the LLDB SB Value object is a char or char[] type then form the text data
//          string otherwise return nothing. m_bHandleCharType must be true to return
//          text data if any.
// Type:    Method.
// Args:    None.
// Return:  CMIUtilString   - Text description of the variable's value.
// Throws:  None.
//--
CMIUtilString
CMICmnLLDBUtilSBValue::GetValueCString(void) const
{
    CMIUtilString text;

    if (m_bHandleCharType && (IsCharType() || (IsArrayType() && IsFirstChildCharType())))
    {
        text = ReadCStringFromHostMemory(m_rValue);
    }

    return text;
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve the flag stating whether this value object is a char type or some
//          other type. Char type can be signed or unsigned.
// Type:    Method.
// Args:    None.
// Return:  bool    - True = Yes is a char type, false = some other type.
// Throws:  None.
//--
bool
CMICmnLLDBUtilSBValue::IsCharType(void) const
{
    const lldb::BasicType eType = m_rValue.GetType().GetBasicType();
    return ((eType == lldb::eBasicTypeChar) || (eType == lldb::eBasicTypeSignedChar) || (eType == lldb::eBasicTypeUnsignedChar));
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve the flag stating whether first child value object of *this object is
//          a char type or some other type. Returns false if there are not children. Char
//          type can be signed or unsigned.
// Type:    Method.
// Args:    None.
// Return:  bool    - True = Yes is a char type, false = some other type.
// Throws:  None.
//--
bool
CMICmnLLDBUtilSBValue::IsFirstChildCharType(void) const
{
    const MIuint nChildren = m_rValue.GetNumChildren();

    // Is it a basic type
    if (nChildren == 0)
        return false;

    const lldb::SBValue member = m_rValue.GetChildAtIndex(0);
    const CMICmnLLDBUtilSBValue utilValue(member);
    return utilValue.IsCharType();
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve the flag stating whether this value object is a integer type or some
//          other type. Char type can be signed or unsigned and short or long/very long.
// Type:    Method.
// Args:    None.
// Return:  bool    - True = Yes is a integer type, false = some other type.
// Throws:  None.
//--
bool
CMICmnLLDBUtilSBValue::IsIntegerType(void) const
{
    const lldb::BasicType eType = m_rValue.GetType().GetBasicType();
    return ((eType == lldb::eBasicTypeShort) || (eType == lldb::eBasicTypeUnsignedShort) ||
            (eType == lldb::eBasicTypeInt) || (eType == lldb::eBasicTypeUnsignedInt) ||
            (eType == lldb::eBasicTypeLong) || (eType == lldb::eBasicTypeUnsignedLong) ||
            (eType == lldb::eBasicTypeLongLong) || (eType == lldb::eBasicTypeUnsignedLongLong) ||
            (eType == lldb::eBasicTypeInt128) || (eType == lldb::eBasicTypeUnsignedInt128));
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve the flag stating whether this value object is a pointer type or some
//          other type.
// Type:    Method.
// Args:    None.
// Return:  bool    - True = Yes is a pointer type, false = some other type.
// Throws:  None.
//--
bool
CMICmnLLDBUtilSBValue::IsPointerType(void) const
{
    return m_rValue.GetType().IsPointerType();
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve the flag stating whether this value object is an array type or some
//          other type.
// Type:    Method.
// Args:    None.
// Return:  bool    - True = Yes is an array type, false = some other type.
// Throws:  None.
//--
bool
CMICmnLLDBUtilSBValue::IsArrayType(void) const
{
    return m_rValue.GetType().IsArrayType();
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve the C string data for a child of char type (one and only child) for
//          the parent value object. If the child is not a char type or the parent has
//          more than one child then an empty string is returned. Char type can be
//          signed or unsigned.
// Type:    Method.
// Args:    None.
// Return:  CMIUtilString   - Text description of the variable's value.
// Throws:  None.
//--
CMIUtilString
CMICmnLLDBUtilSBValue::GetChildValueCString(void) const
{
    CMIUtilString text;
    const MIuint nChildren = m_rValue.GetNumChildren();

    // Is it a basic type
    if (nChildren == 0)
        return text;

    // Is it a composite type
    if (nChildren > 1)
        return text;

    lldb::SBValue member = m_rValue.GetChildAtIndex(0);
    const CMICmnLLDBUtilSBValue utilValue(member);
    if (m_bHandleCharType && utilValue.IsCharType())
    {
        text = ReadCStringFromHostMemory(member);
    }

    return text;
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve the C string data of value object by read the memory where the
//          variable is held.
// Type:    Method.
// Args:    vrValueObj  - (R) LLDB SBValue variable object.
// Return:  CMIUtilString   - Text description of the variable's value.
// Throws:  None.
//--
CMIUtilString
CMICmnLLDBUtilSBValue::ReadCStringFromHostMemory(const lldb::SBValue &vrValueObj) const
{
    CMIUtilString text;

    lldb::SBValue &rValue = const_cast<lldb::SBValue &>(vrValueObj);
    const lldb::addr_t addr = rValue.GetLoadAddress();
    CMICmnLLDBDebugSessionInfo &rSessionInfo(CMICmnLLDBDebugSessionInfo::Instance());
    const MIuint nBytes(128);
    std::unique_ptr<char[]> apBufferMemory(new char[nBytes]);
    lldb::SBError error;
    const MIuint64 nReadBytes = rSessionInfo.GetProcess().ReadMemory(addr, apBufferMemory.get(), nBytes, error);
    MIunused(nReadBytes);
    return CMIUtilString(apBufferMemory.get());
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve the state of the value object's name.
// Type:    Method.
// Args:    None.
// Return:  bool    - True = yes name is indeterminate, false = name is valid.
// Throws:  None.
//--
bool
CMICmnLLDBUtilSBValue::IsNameUnknown(void) const
{
    const CMIUtilString name(GetName());
    return (name == m_pUnkwn);
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve the state of the value object's value data.
// Type:    Method.
// Args:    None.
// Return:  bool    - True = yes value is indeterminate, false = value valid.
// Throws:  None.
//--
bool
CMICmnLLDBUtilSBValue::IsValueUnknown(void) const
{
    const CMIUtilString value(GetValue());
    return (value == m_pUnkwn);
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve the value object's type name if valid.
// Type:    Method.
// Args:    None.
// Return:  CMIUtilString   - The type name or "??".
// Throws:  None.
//--
CMIUtilString
CMICmnLLDBUtilSBValue::GetTypeName(void) const
{
    const MIchar *pName = m_bValidSBValue ? m_rValue.GetTypeName() : nullptr;
    const CMIUtilString text((pName != nullptr) ? pName : m_pUnkwn);

    return text;
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve the value object's display type name if valid.
// Type:    Method.
// Args:    None.
// Return:  CMIUtilString   - The type name or "??".
// Throws:  None.
//--
CMIUtilString
CMICmnLLDBUtilSBValue::GetTypeNameDisplay(void) const
{
    const MIchar *pName = m_bValidSBValue ? m_rValue.GetDisplayTypeName() : nullptr;
    const CMIUtilString text((pName != nullptr) ? pName : m_pUnkwn);

    return text;
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve whether the value object's is valid or not.
// Type:    Method.
// Args:    None.
// Return:  bool    - True = valid, false = not valid.
// Throws:  None.
//--
bool
CMICmnLLDBUtilSBValue::IsValid(void) const
{
    return m_bValidSBValue;
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve the value object' has a name. A value object can be valid but still
//          have no name which suggest it is not a variable.
// Type:    Method.
// Args:    None.
// Return:  bool    - True = valid, false = not valid.
// Throws:  None.
//--
bool
CMICmnLLDBUtilSBValue::HasName(void) const
{
    bool bHasAName = false;

    const MIchar *pName = m_bValidSBValue ? m_rValue.GetDisplayTypeName() : nullptr;
    if (pName != nullptr)
    {
        bHasAName = (CMIUtilString(pName).length() > 0);
    }

    return bHasAName;
}

//++ ------------------------------------------------------------------------------------
// Details: Determine if the value object' respresents a LLDB variable i.e. "$0".
// Type:    Method.
// Args:    None.
// Return:  bool    - True = Yes LLDB variable, false = no.
// Throws:  None.
//--
bool
CMICmnLLDBUtilSBValue::IsLLDBVariable(void) const
{
    return (GetName().at(0) == '$');
}
