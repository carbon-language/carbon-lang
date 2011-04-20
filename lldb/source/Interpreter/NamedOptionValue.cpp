//===-- NamedOptionValue.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/NamedOptionValue.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Stream.h"
#include "lldb/Interpreter/Args.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// NamedOptionValue
//-------------------------------------------------------------------------

void
NamedOptionValue::GetQualifiedName (Stream &strm)
{
    if (m_parent)
    {
        m_parent->GetQualifiedName (strm);
        strm.PutChar('.');
    }
    strm << m_name;
}

OptionValue::Type
NamedOptionValue::GetValueType ()
{
    if (m_value_sp)
        return m_value_sp->GetType();
    return OptionValue::eTypeInvalid;
}

bool
NamedOptionValue::DumpValue (Stream &strm)
{
    if (m_value_sp)
    {
        m_value_sp->DumpValue (strm);
        return true;
    }
    return false;
}

bool
NamedOptionValue::SetValueFromCString (const char *value_cstr)
{
    if (m_value_sp)
        return m_value_sp->SetValueFromCString (value_cstr);
    return false;
}

bool
NamedOptionValue::ResetValueToDefault ()
{
    if (m_value_sp)
        return m_value_sp->ResetValueToDefault ();
    return false;
}


OptionValueBoolean *
NamedOptionValue::GetBooleanValue ()
{
    if (GetValueType() == OptionValue::eTypeBoolean)
        return static_cast<OptionValueBoolean *>(m_value_sp.get());
    return NULL;
}

OptionValueSInt64 *
NamedOptionValue::GetSInt64Value ()
{
    if (GetValueType() == OptionValue::eTypeSInt64)
        return static_cast<OptionValueSInt64 *>(m_value_sp.get());
    return NULL;
}

OptionValueUInt64 *
NamedOptionValue::GetUInt64Value ()
{
    if (GetValueType() == OptionValue::eTypeUInt64)
        return static_cast<OptionValueUInt64 *>(m_value_sp.get());
    return NULL;
}

OptionValueString *
NamedOptionValue::GetStringValue ()
{
    if (GetValueType() == OptionValue::eTypeString)
        return static_cast<OptionValueString *>(m_value_sp.get());
    return NULL;
}

OptionValueFileSpec *
NamedOptionValue::GetFileSpecValue ()
{
    if (GetValueType() == OptionValue::eTypeFileSpec)
        return static_cast<OptionValueFileSpec *>(m_value_sp.get());
    return NULL;
}

OptionValueArray *
NamedOptionValue::GetArrayValue ()
{
    if (GetValueType() == OptionValue::eTypeArray)
        return static_cast<OptionValueArray *>(m_value_sp.get());
    return NULL;
}

OptionValueDictionary *
NamedOptionValue::GetDictionaryValue ()
{
    if (GetValueType() == OptionValue::eTypeDictionary)
        return static_cast<OptionValueDictionary *>(m_value_sp.get());
    return NULL;
}

//-------------------------------------------------------------------------
// OptionValueBoolean
//-------------------------------------------------------------------------
void
OptionValueBoolean::DumpValue (Stream &strm)
{
    strm.PutCString (m_current_value ? "true" : "false");
}

bool
OptionValueBoolean::SetValueFromCString (const char *value_cstr)
{
    bool success = false;
    bool value = Args::StringToBoolean(value_cstr, false, &success);
    if (success)
    {
        m_current_value = value;
        return true;
    }
    return false;
}

//-------------------------------------------------------------------------
// OptionValueSInt64
//-------------------------------------------------------------------------
void
OptionValueSInt64::DumpValue (Stream &strm)
{
    strm.Printf ("%lli", m_current_value);
}

bool
OptionValueSInt64::SetValueFromCString (const char *value_cstr)
{
    bool success = false;
    int64_t value = Args::StringToSInt64 (value_cstr, 0, 0, &success);
    if (success)
    {
        m_current_value = value;
        return true;
    }
    return false;
}

//-------------------------------------------------------------------------
// OptionValueUInt64
//-------------------------------------------------------------------------
void
OptionValueUInt64::DumpValue (Stream &strm)
{
    strm.Printf ("0x%llx", m_current_value);
}

bool
OptionValueUInt64::SetValueFromCString (const char *value_cstr)
{
    bool success = false;
    uint64_t value = Args::StringToUInt64 (value_cstr, 0, 0, &success);
    if (success)
    {
        m_current_value = value;
        return true;
    }
    return false;
}

//-------------------------------------------------------------------------
// OptionValueDictionary
//-------------------------------------------------------------------------
void
OptionValueString::DumpValue (Stream &strm)
{
    strm.Printf ("\"%s\"", m_current_value.c_str());
}

bool
OptionValueString::SetValueFromCString (const char *value_cstr)
{
    SetCurrentValue (value_cstr);
    return true;
}



//-------------------------------------------------------------------------
// OptionValueFileSpec
//-------------------------------------------------------------------------
void
OptionValueFileSpec::DumpValue (Stream &strm)
{
    if (m_current_value)
    {
        if (m_current_value.GetDirectory())
        {
            strm << '"' << m_current_value.GetDirectory();
            if (m_current_value.GetFilename())
                strm << '/' << m_current_value.GetFilename();
            strm << '"';
        }
        else
        {
            strm << '"' << m_current_value.GetFilename() << '"';
        }
    }
}

bool
OptionValueFileSpec::SetValueFromCString (const char *value_cstr)
{
    if (value_cstr && value_cstr[0])
        m_current_value.SetFile(value_cstr, false);
    else
        m_current_value.Clear();
    return true;
}


//-------------------------------------------------------------------------
// OptionValueArray
//-------------------------------------------------------------------------
void
OptionValueArray::DumpValue (Stream &strm)
{
    const uint32_t size = m_values.size();
    for (uint32_t i = 0; i<size; ++i)
    {
        strm.Printf("[%u] ", i);
        m_values[i]->DumpValue (strm);
    }
}

bool
OptionValueArray::SetValueFromCString (const char *value_cstr)
{
    // We must be able to set this using the array specific functions
    return false;
}

//-------------------------------------------------------------------------
// OptionValueDictionary
//-------------------------------------------------------------------------
void
OptionValueDictionary::DumpValue (Stream &strm)
{
    collection::iterator pos, end = m_values.end();

    for (pos = m_values.begin(); pos != end; ++pos)
    {
        strm.Printf("%s=", pos->first.GetCString());
        pos->second->DumpValue (strm);
    }
}

bool
OptionValueDictionary::SetValueFromCString (const char *value_cstr)
{
    // We must be able to set this using the array specific functions
    return false;
}


