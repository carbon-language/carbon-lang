//===---------------------StructuredData.cpp ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/StructuredData.h"

#include <errno.h>
#include <stdlib.h>
#include <inttypes.h>

#include "lldb/Core/StreamString.h"

using namespace lldb_private;


static StructuredData::ObjectSP read_json_object (const char **ch);
static StructuredData::ObjectSP read_json_array (const char **ch);

static StructuredData::ObjectSP
read_json_number (const char **ch)
{
    StructuredData::ObjectSP object_sp;
    while (isspace (**ch))
        (*ch)++;
    const char *start_of_number = *ch;
    bool is_integer = true;
    bool is_float = false;
    while (isdigit(**ch) || **ch == '-' || **ch == '.' || **ch == '+' || **ch == 'e' || **ch == 'E')
    {
        if (isdigit(**ch) == false && **ch != '-')
        {
            is_integer = false;
            is_float = true;
        }
        (*ch)++;
    }
    while (isspace (**ch))
        (*ch)++;
    if (**ch == ',' || **ch == ']' || **ch == '}')
    {
        if (is_integer)
        {
            errno = 0;
            uint64_t val = strtoul (start_of_number, NULL, 10);
            if (errno == 0)
            {
                object_sp.reset(new StructuredData::Integer());
                object_sp->GetAsInteger()->SetValue (val);
            }
        }
        if (is_float)
        {
            char *end_of_number = NULL;
            errno = 0;
            double val = strtod (start_of_number, &end_of_number);
            if (errno == 0 && end_of_number != start_of_number && end_of_number != NULL)
            {
                object_sp.reset(new StructuredData::Float());
                object_sp->GetAsFloat()->SetValue (val);
            }
        }
    }
    return object_sp;
}

static std::string
read_json_string (const char **ch)
{
    std::string string;
    if (**ch == '"')
    {
        (*ch)++;
        while (**ch != '\0')
        {
            if (**ch == '"')
            {
                (*ch)++;
                while (isspace (**ch))
                    (*ch)++;
                break;
            }
            else if (**ch == '\\')
            {
                switch (**ch)
                {
                    case '"':
                        string.push_back('"');
                        *ch += 2;
                        break;
                    case '\\':
                        string.push_back('\\');
                        *ch += 2;
                        break;
                    case '/':
                        string.push_back('/');
                        *ch += 2;
                        break;
                    case 'b':
                        string.push_back('\b');
                        *ch += 2;
                        break;
                    case 'f':
                        string.push_back('\f');
                        *ch += 2;
                        break;
                    case 'n':
                        string.push_back('\n');
                        *ch += 2;
                        break;
                    case 'r':
                        string.push_back('\r');
                        *ch += 2;
                        break;
                    case 't':
                        string.push_back('\t');
                        *ch += 2;
                        break;
                    case 'u':
                        // FIXME handle four-hex-digits 
                        *ch += 10;
                        break;
                    default:
                        *ch += 1;
                }
            }
            else
            {
                string.push_back (**ch);
            }
            (*ch)++;
        }
    }
    return string;
}

static StructuredData::ObjectSP
read_json_value (const char **ch)
{
    StructuredData::ObjectSP object_sp;
    while (isspace (**ch))
        (*ch)++;

    if (**ch == '{')
    {
        object_sp = read_json_object (ch);
    }
    else if (**ch == '[')
    {
        object_sp = read_json_array (ch);
    }
    else if (**ch == '"')
    {
        std::string string = read_json_string (ch);
        object_sp.reset(new StructuredData::String());
        object_sp->GetAsString()->SetValue(string);
    }
    else
    {
        if (strncmp (*ch, "true", 4) == 0)
        {
            object_sp.reset(new StructuredData::Boolean());
            object_sp->GetAsBoolean()->SetValue(true);
            *ch += 4;
        }
        else if (strncmp (*ch, "false", 5) == 0)
        {
            object_sp.reset(new StructuredData::Boolean());
            object_sp->GetAsBoolean()->SetValue(false);
            *ch += 5;
        }
        else if (strncmp (*ch, "null", 4) == 0)
        {
            object_sp.reset(new StructuredData::Null());
            *ch += 4;
        }
        else
        {
            object_sp = read_json_number (ch);
        }
    }
    return object_sp;
}

static StructuredData::ObjectSP
read_json_array (const char **ch)
{
    StructuredData::ObjectSP object_sp;
    if (**ch == '[')
    {
        (*ch)++;
        while (isspace (**ch))
            (*ch)++;

        bool first_value = true;
        while (**ch != '\0' && (first_value || **ch == ','))
        {
            if (**ch == ',')
                (*ch)++;
            first_value = false;
            while (isspace (**ch))
                (*ch)++;
            lldb_private::StructuredData::ObjectSP value_sp = read_json_value (ch);
            if (value_sp)
            {
                if (object_sp.get() == NULL)
                {
                    object_sp.reset(new StructuredData::Array());
                }
                object_sp->GetAsArray()->Push (value_sp);
            }
            while (isspace (**ch))
                (*ch)++;
        }
        if (**ch == ']')
        {
            // FIXME should throw an error if we don't see a } to close out the JSON object
            (*ch)++;
            while (isspace (**ch))
                (*ch)++;
        }
    }
    return object_sp;
}

static StructuredData::ObjectSP
read_json_object (const char **ch)
{
    StructuredData::ObjectSP object_sp;
    if (**ch == '{')
    {
        (*ch)++;
        while (isspace (**ch))
            (*ch)++;
        bool first_pair = true;
        while (**ch != '\0' && (first_pair || **ch == ','))
        {
            first_pair = false;
            if (**ch == ',')
                (*ch)++;
            while (isspace (**ch))
                (*ch)++;
            if (**ch != '"')
                break;
            std::string key_string = read_json_string (ch);
            while (isspace (**ch))
                (*ch)++;
            if (key_string.size() > 0 && **ch == ':')
            {
                (*ch)++;
                while (isspace (**ch))
                    (*ch)++;
                lldb_private::StructuredData::ObjectSP value_sp = read_json_value (ch);
                if (value_sp.get())
                {
                    if (object_sp.get() == NULL)
                    {
                        object_sp.reset(new StructuredData::Dictionary());
                    }
                    object_sp->GetAsDictionary()->AddItem (key_string.c_str(), value_sp);
                }
            }
            while (isspace (**ch))
                (*ch)++;
        }
        if (**ch == '}')
        {
            // FIXME should throw an error if we don't see a } to close out the JSON object
            (*ch)++;
            while (isspace (**ch))
                (*ch)++;
        }
    }
    return object_sp;
}


StructuredData::ObjectSP
StructuredData::ParseJSON (std::string json_text)
{
    StructuredData::ObjectSP object_sp;
    const size_t json_text_size = json_text.size();
    if (json_text_size > 0)
    {
        const char *start_of_json_text = json_text.c_str();
        const char *c = json_text.c_str();
        while (*c != '\0' &&
               static_cast<size_t>(c - start_of_json_text) <= json_text_size)
        {
            while (isspace (*c) &&
                   static_cast<size_t>(c - start_of_json_text) < json_text_size)
                c++;
            if (*c == '{')
            {
                object_sp = read_json_object (&c);
            }
            else if (*c == '[')
            {
                object_sp = read_json_array (&c);
            }
            else
            {
                // We have bad characters here, this is likely an illegal JSON string.
                return object_sp;
            }
        }
    }
    return object_sp;
}

StructuredData::ObjectSP
StructuredData::Object::GetObjectForDotSeparatedPath (llvm::StringRef path)
{
    if (this->GetType() == Type::eTypeDictionary)
    {
        std::pair<llvm::StringRef, llvm::StringRef> match = path.split('.');
        std::string key = match.first.str();
        ObjectSP value = this->GetAsDictionary()->GetValueForKey (key.c_str());
        if (value.get())
        {
            // Do we have additional words to descend?  If not, return the
            // value we're at right now.
            if (match.second.empty())
            {
                return value;
            }
            else
            {
                return value->GetObjectForDotSeparatedPath (match.second);
            }
        }
        return ObjectSP();
    }

    if (this->GetType() == Type::eTypeArray)
    {
        std::pair<llvm::StringRef, llvm::StringRef> match = path.split('[');
        if (match.second.size() == 0)
        {
            return this->shared_from_this();
        }
        errno = 0;
        uint64_t val = strtoul (match.second.str().c_str(), NULL, 10);
        if (errno == 0)
        {
            return this->GetAsArray()->GetItemAtIndex(val);
        }
        return ObjectSP();
    }

    return this->shared_from_this();
}

void
StructuredData::Object::DumpToStdout() const
{
    StreamString stream;
    Dump(stream);
    printf("%s\n", stream.GetString().c_str());
}

void
StructuredData::Array::Dump(Stream &s) const
{
    bool first = true;
    s << "[\n";
    s.IndentMore();
    for (const auto &item_sp : m_items)
    {
        if (first)
            first = false;
        else
            s << ",\n";

        s.Indent();
        item_sp->Dump(s);
    }
    s.IndentLess();
    s.EOL();
    s.Indent();
    s << "]";
}

void
StructuredData::Integer::Dump (Stream &s) const
{
    s.Printf ("%" PRIu64, m_value);
}


void
StructuredData::Float::Dump (Stream &s) const
{
    s.Printf ("%lf", m_value);
}

void
StructuredData::Boolean::Dump (Stream &s) const
{
    if (m_value == true)
        s.PutCString ("true");
    else
        s.PutCString ("false");
}


void
StructuredData::String::Dump (Stream &s) const
{
    std::string quoted;
    const size_t strsize = m_value.size();
    for (size_t i = 0; i < strsize ; ++i)
    {
        char ch = m_value[i];
        if (ch == '"')
            quoted.push_back ('\\');
        quoted.push_back (ch);
    }
    s.Printf ("\"%s\"", quoted.c_str());
}

void
StructuredData::Dictionary::Dump (Stream &s) const
{
    bool first = true;
    s << "{\n";
    s.IndentMore();
    for (const auto &pair : m_dict)
    {
        if (first)
            first = false;
        else
            s << ",\n";
        s.Indent();
        s << "\"" << pair.first.AsCString() << "\" : ";
        pair.second->Dump(s);
    }
    s.IndentLess();
    s.EOL();
    s.Indent();
    s << "}";
}

void
StructuredData::Null::Dump (Stream &s) const
{
    s << "null";
}

void
StructuredData::Generic::Dump(Stream &s) const
{
    s << "0x" << m_object;
}
