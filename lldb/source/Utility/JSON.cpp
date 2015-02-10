//===--------------------- JSON.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/JSON.h"

using namespace lldb_private;

std::string
JSONString::json_string_quote_metachars (const std::string &s)
{
    if (s.find('"') == std::string::npos)
        return s;
    
    std::string output;
    const size_t s_size = s.size();
    const char *s_chars = s.c_str();
    for (size_t i = 0; i < s_size; i++)
    {
        unsigned char ch = *(s_chars + i);
        if (ch == '"')
        {
            output.push_back ('\\');
        }
        output.push_back (ch);
    }
    return output;
}

JSONString::JSONString () :
JSONValue(JSONValue::Kind::String),
m_data()
{
}

JSONString::JSONString (const char* s) :
JSONValue(JSONValue::Kind::String),
m_data(s ? s : "")
{
}

JSONString::JSONString (const std::string& s) :
JSONValue(JSONValue::Kind::String),
m_data(s)
{
}

void
JSONString::Write (Stream& s)
{
    s.Printf("\"%s\"", json_string_quote_metachars(m_data).c_str());
}

JSONNumber::JSONNumber () :
JSONValue(JSONValue::Kind::Number),
m_data(0)
{
}

JSONNumber::JSONNumber (int64_t i) :
JSONValue(JSONValue::Kind::Number),
m_data(i)
{
}

void
JSONNumber::Write (Stream& s)
{
    s.Printf("%" PRId64, m_data);
}

JSONTrue::JSONTrue () :
JSONValue(JSONValue::Kind::True)
{
}

void
JSONTrue::Write(Stream& s)
{
    s.Printf("true");
}

JSONFalse::JSONFalse () :
JSONValue(JSONValue::Kind::False)
{
}

void
JSONFalse::Write(Stream& s)
{
    s.Printf("false");
}

JSONNull::JSONNull () :
JSONValue(JSONValue::Kind::Null)
{
}

void
JSONNull::Write(Stream& s)
{
    s.Printf("null");
}

JSONObject::JSONObject () :
JSONValue(JSONValue::Kind::Object)
{
}

void
JSONObject::Write (Stream& s)
{
    bool first = true;
    s.PutChar('{');
    auto iter = m_elements.begin(), end = m_elements.end();
    for (;iter != end; iter++)
    {
        if (first)
            first = false;
        else
            s.PutChar(',');
        JSONString key(iter->first);
        JSONValue::SP value(iter->second);
        key.Write(s);
        s.PutChar(':');
        value->Write(s);
    }
    s.PutChar('}');
}

bool
JSONObject::SetObject (const std::string& key,
                       JSONValue::SP value)
{
    if (key.empty() || nullptr == value.get())
        return false;
    m_elements[key] = value;
    return true;
}

JSONValue::SP
JSONObject::GetObject (const std::string& key)
{
    auto iter = m_elements.find(key), end = m_elements.end();
    if (iter == end)
        return JSONValue::SP();
    return iter->second;
}

JSONArray::JSONArray () :
JSONValue(JSONValue::Kind::Array)
{
}

void
JSONArray::Write (Stream& s)
{
    bool first = true;
    s.PutChar('[');
    auto iter = m_elements.begin(), end = m_elements.end();
    for (;iter != end; iter++)
    {
        if (first)
            first = false;
        else
            s.PutChar(',');
        (*iter)->Write(s);
    }
    s.PutChar(']');
}

bool
JSONArray::SetObject (Index i,
                      JSONValue::SP value)
{
    if (value.get() == nullptr)
        return false;
    if (i < m_elements.size())
    {
        m_elements[i] = value;
        return true;
    }
    if (i == m_elements.size())
    {
        m_elements.push_back(value);
        return true;
    }
    return false;
}

bool
JSONArray::AppendObject (JSONValue::SP value)
{
    if (value.get() == nullptr)
        return false;
    m_elements.push_back(value);
    return true;
}

JSONValue::SP
JSONArray::GetObject (Index i)
{
    if (i < m_elements.size())
        return m_elements[i];
    return JSONValue::SP();
}

JSONArray::Size
JSONArray::GetNumElements ()
{
    return m_elements.size();
}
