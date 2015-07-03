//===-- MICmnLLDBUtilSBValue.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

// Third Party Headers:
#include "lldb/API/SBValue.h"

// In-house headers:
#include "MIDataTypes.h"
#include "MICmnMIValueTuple.h"

// Declerations:
class CMIUtilString;

//++ ============================================================================
// Details: Utility helper class to lldb::SBValue. Using a lldb::SBValue extract
//          value object information to help form verbose debug information.
// Gotchas: None.
// Authors: Illya Rudkin 08/07/2014.
// Changes: None.
//--
class CMICmnLLDBUtilSBValue
{
    // Methods:
  public:
    /* ctor */ CMICmnLLDBUtilSBValue(const lldb::SBValue &vrValue, const bool vbHandleCharType = false,
                                     const bool vbHandleArrayType = true);
    /* dtor */ ~CMICmnLLDBUtilSBValue(void);
    //
    CMIUtilString GetName(void) const;
    CMIUtilString GetValue(const bool vbExpandAggregates = false) const;
    CMIUtilString GetTypeName(void) const;
    CMIUtilString GetTypeNameDisplay(void) const;
    bool IsCharType(void) const;
    bool IsFirstChildCharType(void) const;
    bool IsIntegerType(void) const;
    bool IsPointerType(void) const;
    bool IsArrayType(void) const;
    bool IsLLDBVariable(void) const;
    bool IsNameUnknown(void) const;
    bool IsValueUnknown(void) const;
    bool IsValid(void) const;
    bool HasName(void) const;

    // Methods:
  private:
    template <typename charT> CMIUtilString ReadCStringFromHostMemory(lldb::SBValue &vrValue, const MIuint vnMaxLen = UINT32_MAX) const;
    bool GetSimpleValue(const bool vbHandleArrayType, CMIUtilString &vrValue) const;
    CMIUtilString GetSimpleValueChar(void) const;
    CMIUtilString GetSimpleValueCStringPointer(void) const;
    CMIUtilString GetSimpleValueCStringArray(void) const;
    bool GetCompositeValue(const bool vbPrintFieldNames, CMICmnMIValueTuple &vwrMiValueTuple, const MIuint vnDepth = 1) const;

    // Attributes:
  private:
    lldb::SBValue &m_rValue;
    const char *m_pUnkwn;
    const char *m_pComposite;
    bool m_bValidSBValue;    // True = SBValue is a valid object, false = not valid.
    bool m_bHandleCharType;  // True = Yes return text molding to char type, false = just return data.
    bool m_bHandleArrayType; // True = Yes return special stub for array type, false = just return data.
};
