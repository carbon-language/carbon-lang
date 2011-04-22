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

// Get this value as a uint64_t value if it is encoded as a boolean,
// uint64_t or int64_t. Other types will cause "fail_value" to be 
// returned
uint64_t
OptionValue::GetUInt64Value (uint64_t fail_value, bool *success_ptr)
{
    if (success_ptr)
        *success_ptr = true;
    switch (GetType())
    {
    case OptionValue::eTypeBoolean: return static_cast<OptionValueBoolean *>(this)->GetCurrentValue();
    case OptionValue::eTypeSInt64:  return static_cast<OptionValueSInt64 *>(this)->GetCurrentValue();
    case OptionValue::eTypeUInt64:  return static_cast<OptionValueUInt64 *>(this)->GetCurrentValue();
    default: 
        break;
    }
    if (success_ptr)
        *success_ptr = false;
    return fail_value;
}


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

const char *
OptionValue::GetStringValue ()
{
    if (GetType () == OptionValue::eTypeString)
        return static_cast<OptionValueString *>(this)->GetCurrentValue();
    return NULL;
}

uint64_t
OptionValue::GetUInt64Value ()
{
    if (GetType () == OptionValue::eTypeUInt64)
        return static_cast<OptionValueUInt64 *>(this)->GetCurrentValue();
    return 0;
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

Error
OptionValueBoolean::SetValueFromCString (const char *value_cstr)
{
    Error error;
    bool success = false;
    bool value = Args::StringToBoolean(value_cstr, false, &success);
    if (success)
    {
        m_value_was_set = true;
        m_current_value = value;
    }
    else
    {
        error.SetErrorStringWithFormat ("invalid boolean string value: '%s'\n", value_cstr);
    }
    return error;
}

//-------------------------------------------------------------------------
// OptionValueSInt64
//-------------------------------------------------------------------------
void
OptionValueSInt64::DumpValue (Stream &strm)
{
    strm.Printf ("%lli", m_current_value);
}

Error
OptionValueSInt64::SetValueFromCString (const char *value_cstr)
{
 
    Error error;
    bool success = false;
    int64_t value = Args::StringToSInt64 (value_cstr, 0, 0, &success);
    if (success)
    {
        m_value_was_set = true;
        m_current_value = value;
    }
    else
    {
        error.SetErrorStringWithFormat ("invalid int64_t string value: '%s'\n", value_cstr);
    }
    return error;
}

//-------------------------------------------------------------------------
// OptionValueUInt64
//-------------------------------------------------------------------------

lldb::OptionValueSP
OptionValueUInt64::Create (const char *value_cstr, Error &error)
{
    lldb::OptionValueSP value_sp (new OptionValueUInt64());
    error = value_sp->SetValueFromCString (value_cstr);
    if (error.Fail())
        value_sp.reset();
    return value_sp;
}


void
OptionValueUInt64::DumpValue (Stream &strm)
{
    strm.Printf ("0x%llx", m_current_value);
}

Error
OptionValueUInt64::SetValueFromCString (const char *value_cstr)
{
    Error error;
    bool success = false;
    uint64_t value = Args::StringToUInt64 (value_cstr, 0, 0, &success);
    if (success)
    {
        m_value_was_set = true;
        m_current_value = value;
    }
    else
    {
        error.SetErrorStringWithFormat ("invalid uint64_t string value: '%s'\n", value_cstr);
    }
    return error;
}

//-------------------------------------------------------------------------
// OptionValueDictionary
//-------------------------------------------------------------------------
void
OptionValueString::DumpValue (Stream &strm)
{
    strm.Printf ("\"%s\"", m_current_value.c_str());
}

Error
OptionValueString::SetValueFromCString (const char *value_cstr)
{
    m_value_was_set = true;
    SetCurrentValue (value_cstr);
    return Error ();
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

Error
OptionValueFileSpec::SetValueFromCString (const char *value_cstr)
{
    if (value_cstr && value_cstr[0])
        m_current_value.SetFile(value_cstr, false);
    else
        m_current_value.Clear();
    m_value_was_set = true;
    return Error();
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

Error
OptionValueArray::SetValueFromCString (const char *value_cstr)
{
    Error error;
    error.SetErrorStringWithFormat ("array option values don't yet support being set by string: '%s'\n", value_cstr);
    return error;
}


uint64_t
OptionValueArray::GetUInt64ValueAtIndex (uint32_t idx, uint64_t fail_value, bool *success_ptr) const
{
    if (idx < m_values.size())
        return m_values[idx]->GetUInt64Value (fail_value, success_ptr);
    return fail_value;
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

Error
OptionValueDictionary::SetValueFromCString (const char *value_cstr)
{
    Error error;
    error.SetErrorStringWithFormat ("dictionary option values don't yet support being set by string: '%s'\n", value_cstr);
    return error;
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


