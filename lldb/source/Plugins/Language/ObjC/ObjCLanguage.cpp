//===-- ObjCLanguage.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ObjCLanguage.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamString.h"

using namespace lldb;
using namespace lldb_private;

void
ObjCLanguage::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   "Objective-C Language",
                                   CreateInstance);
}

void
ObjCLanguage::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

lldb_private::ConstString
ObjCLanguage::GetPluginNameStatic()
{
    static ConstString g_name("objc");
    return g_name;
}


//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString
ObjCLanguage::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ObjCLanguage::GetPluginVersion()
{
    return 1;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
Language *
ObjCLanguage::CreateInstance (lldb::LanguageType language)
{
    switch (language)
    {
        case lldb::eLanguageTypeObjC:
            return new ObjCLanguage();
        default:
            return nullptr;
    }
}

void
ObjCLanguage::MethodName::Clear()
{
    m_full.Clear();
    m_class.Clear();
    m_category.Clear();
    m_selector.Clear();
    m_type = eTypeUnspecified;
    m_category_is_valid = false;
}

bool
ObjCLanguage::MethodName::SetName (const char *name, bool strict)
{
    Clear();
    if (name && name[0])
    {
        // If "strict" is true. then the method must be specified with a
        // '+' or '-' at the beginning. If "strict" is false, then the '+'
        // or '-' can be omitted
        bool valid_prefix = false;
        
        if (name[0] == '+' || name[0] == '-')
        {
            valid_prefix = name[1] == '[';
            if (name[0] == '+')
                m_type = eTypeClassMethod;
            else
                m_type = eTypeInstanceMethod;
        }
        else if (!strict)
        {
            // "strict" is false, the name just needs to start with '['
            valid_prefix = name[0] == '[';
        }
        
        if (valid_prefix)
        {
            int name_len = strlen (name);
            // Objective C methods must have at least:
            //      "-[" or "+[" prefix
            //      One character for a class name
            //      One character for the space between the class name
            //      One character for the method name
            //      "]" suffix
            if (name_len >= (5 + (strict ? 1 : 0)) && name[name_len - 1] == ']')
            {
                m_full.SetCStringWithLength(name, name_len);
            }
        }
    }
    return IsValid(strict);
}

const ConstString &
ObjCLanguage::MethodName::GetClassName ()
{
    if (!m_class)
    {
        if (IsValid(false))
        {
            const char *full = m_full.GetCString();
            const char *class_start = (full[0] == '[' ? full + 1 : full + 2);
            const char *paren_pos = strchr (class_start, '(');
            if (paren_pos)
            {
                m_class.SetCStringWithLength (class_start, paren_pos - class_start);
            }
            else
            {
                // No '(' was found in the full name, we can definitively say
                // that our category was valid (and empty).
                m_category_is_valid = true;
                const char *space_pos = strchr (full, ' ');
                if (space_pos)
                {
                    m_class.SetCStringWithLength (class_start, space_pos - class_start);
                    if (!m_class_category)
                    {
                        // No category in name, so we can also fill in the m_class_category
                        m_class_category = m_class;
                    }
                }
            }
        }
    }
    return m_class;
}

const ConstString &
ObjCLanguage::MethodName::GetClassNameWithCategory () 
{
    if (!m_class_category)
    {
        if (IsValid(false))
        {
            const char *full = m_full.GetCString();
            const char *class_start = (full[0] == '[' ? full + 1 : full + 2);
            const char *space_pos = strchr (full, ' ');
            if (space_pos)
            {
                m_class_category.SetCStringWithLength (class_start, space_pos - class_start);
                // If m_class hasn't been filled in and the class with category doesn't
                // contain a '(', then we can also fill in the m_class
                if (!m_class && strchr (m_class_category.GetCString(), '(') == NULL)
                {
                    m_class = m_class_category;
                    // No '(' was found in the full name, we can definitively say
                    // that our category was valid (and empty).
                    m_category_is_valid = true;

                }
            }
        }
    }
    return m_class_category;
}

const ConstString &
ObjCLanguage::MethodName::GetSelector ()
{
    if (!m_selector)
    {
        if (IsValid(false))
        {
            const char *full = m_full.GetCString();
            const char *space_pos = strchr (full, ' ');
            if (space_pos)
            {
                ++space_pos; // skip the space
                m_selector.SetCStringWithLength (space_pos, m_full.GetLength() - (space_pos - full) - 1);
            }
        }
    }
    return m_selector;
}

const ConstString &
ObjCLanguage::MethodName::GetCategory ()
{
    if (!m_category_is_valid && !m_category)
    {
        if (IsValid(false))
        {
            m_category_is_valid = true;
            const char *full = m_full.GetCString();
            const char *class_start = (full[0] == '[' ? full + 1 : full + 2);
            const char *open_paren_pos = strchr (class_start, '(');
            if (open_paren_pos)
            {
                ++open_paren_pos; // Skip the open paren
                const char *close_paren_pos = strchr (open_paren_pos, ')');
                if (close_paren_pos)
                    m_category.SetCStringWithLength (open_paren_pos, close_paren_pos - open_paren_pos);
            }
        }
    }
    return m_category;
}

ConstString
ObjCLanguage::MethodName::GetFullNameWithoutCategory (bool empty_if_no_category)
{
    if (IsValid(false))
    {
        if (HasCategory())
        {
            StreamString strm;
            if (m_type == eTypeClassMethod)
                strm.PutChar('+');
            else if (m_type == eTypeInstanceMethod)
                strm.PutChar('-');
            strm.Printf("[%s %s]", GetClassName().GetCString(), GetSelector().GetCString());
            return ConstString(strm.GetString().c_str());
        }
        
        if (!empty_if_no_category)
        {
            // Just return the full name since it doesn't have a category
            return GetFullName();
        }
    }
    return ConstString();
}

size_t
ObjCLanguage::MethodName::GetFullNames (std::vector<ConstString> &names, bool append)
{
    if (!append)
        names.clear();
    if (IsValid(false))
    {
        StreamString strm;
        const bool is_class_method = m_type == eTypeClassMethod;
        const bool is_instance_method = m_type == eTypeInstanceMethod;
        const ConstString &category = GetCategory();
        if (is_class_method || is_instance_method)
        {
            names.push_back (m_full);
            if (category)
            {
                strm.Printf("%c[%s %s]",
                            is_class_method ? '+' : '-',
                            GetClassName().GetCString(),
                            GetSelector().GetCString());
                names.push_back(ConstString(strm.GetString().c_str()));
            }
        }
        else
        {
            const ConstString &class_name = GetClassName();
            const ConstString &selector = GetSelector();
            strm.Printf("+[%s %s]", class_name.GetCString(), selector.GetCString());
            names.push_back(ConstString(strm.GetString().c_str()));
            strm.Clear();
            strm.Printf("-[%s %s]", class_name.GetCString(), selector.GetCString());
            names.push_back(ConstString(strm.GetString().c_str()));
            strm.Clear();
            if (category)
            {
                strm.Printf("+[%s(%s) %s]", class_name.GetCString(), category.GetCString(), selector.GetCString());
                names.push_back(ConstString(strm.GetString().c_str()));
                strm.Clear();
                strm.Printf("-[%s(%s) %s]", class_name.GetCString(), category.GetCString(), selector.GetCString());
                names.push_back(ConstString(strm.GetString().c_str()));
            }
        }
    }
    return names.size();
}
