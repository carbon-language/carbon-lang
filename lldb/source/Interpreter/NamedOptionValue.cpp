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
// OptionValue
//-------------------------------------------------------------------------
OptionValueBoolean *
OptionValue::GetAsBooleanValue ()
{
    if (GetType () == OptionValue::eTypeBoolean)
        return static_cast<OptionValueBoolean *>(this);
    return NULL;
}

OptionValueSInt64 *
OptionValue::GetAsSInt64Value ()
{
    if (GetType () == OptionValue::eTypeSInt64)
        return static_cast<OptionValueSInt64 *>(this);
    return NULL;
}

OptionValueUInt64 *
OptionValue::GetAsUInt64Value ()
{
    if (GetType () == OptionValue::eTypeUInt64)
        return static_cast<OptionValueUInt64 *>(this);
    return NULL;
}

OptionValueString *
OptionValue::GetAsStringValue ()
{
    if (GetType () == OptionValue::eTypeString)
        return static_cast<OptionValueString *>(this);
    return NULL;
}

OptionValueFileSpec *
OptionValue::GetAsFileSpecValue ()
{
    if (GetType () == OptionValue::eTypeFileSpec)
        return static_cast<OptionValueFileSpec *>(this);
    return NULL;
}

OptionValueArray *
OptionValue::GetAsArrayValue ()
{
    if (GetType () == OptionValue::eTypeArray)
        return static_cast<OptionValueArray *>(this);
    return NULL;
}

OptionValueDictionary *
OptionValue::GetAsDictionaryValue ()
{
    if (GetType () == OptionValue::eTypeDictionary)
        return static_cast<OptionValueDictionary *>(this);
    return NULL;
}


//-------------------------------------------------------------------------
// OptionValueCollection
//-------------------------------------------------------------------------

void
OptionValueCollection::GetQualifiedName (Stream &strm)
{
    if (m_parent)
    {
        m_parent->GetQualifiedName (strm);
        strm.PutChar('.');
    }
    strm << m_name;
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

lldb::OptionValueSP
OptionValueDictionary::GetValueForKey (const ConstString &key) const
{
    lldb::OptionValueSP value_sp;
    collection::const_iterator pos = m_values.find (key);
    if (pos != m_values.end())
        value_sp = pos->second;
    return value_sp;
}

const char *
OptionValueDictionary::GetStringValueForKey (const ConstString &key)
{
    collection::const_iterator pos = m_values.find (key);
    if (pos != m_values.end())
    {
        if (pos->second->GetType() == OptionValue::eTypeString)
            return static_cast<OptionValueString *>(pos->second.get())->GetCurrentValue();
    }
    return NULL;
}


bool
OptionValueDictionary::SetStringValueForKey (const ConstString &key, 
                                             const char *value, 
                                             bool can_replace)
{
    collection::const_iterator pos = m_values.find (key);
    if (pos != m_values.end())
    {
        if (!can_replace)
            return false;
        if (pos->second->GetType() == OptionValue::eTypeString)
        {
            pos->second->SetValueFromCString(value);
            return true;
        }
    }
    m_values[key] = OptionValueSP (new OptionValueString (value));
    return true;

}

bool
OptionValueDictionary::SetValueForKey (const ConstString &key, 
                                       const lldb::OptionValueSP &value_sp, 
                                       bool can_replace)
{
    // Make sure the value_sp object is allowed to contain
    // values of the type passed in...
    if (value_sp && (m_type_mask & value_sp->GetTypeAsMask()))
    {
        if (!can_replace)
        {
            collection::const_iterator pos = m_values.find (key);
            if (pos != m_values.end())
                return false;
        }
        m_values[key] = value_sp;
        return true;
    }
    return false;
}

bool
OptionValueDictionary::DeleteValueForKey (const ConstString &key)
{
    collection::iterator pos = m_values.find (key);
    if (pos != m_values.end())
    {
        m_values.erase(pos);
        return true;
    }
    return false;
}


