//===-- FormatClasses.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_FormatClasses_h_
#define lldb_FormatClasses_h_

// C++ Includes
#include <string>
#include <vector>

// Other libraries and framework includes

// Project includes
#include "lldb/lldb-public.h"
#include "lldb/lldb-enumerations.h"

#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/Type.h"

namespace lldb_private {

class FormattersMatchCandidate
{
public:
    
    FormattersMatchCandidate (ConstString name,
                              uint32_t reason,
                              bool strip_ptr,
                              bool strip_ref,
                              bool strip_tydef) :
    m_type_name(name),
    m_reason(reason),
    m_stripped_pointer(strip_ptr),
    m_stripped_reference(strip_ref),
    m_stripped_typedef(strip_tydef)
    {
    }
    
    ~FormattersMatchCandidate ()
    {}
    
    ConstString
    GetTypeName () const
    {
        return m_type_name;
    }
    
    uint32_t
    GetReason () const
    {
        return m_reason;
    }
    
    bool
    DidStripPointer () const
    {
        return m_stripped_pointer;
    }
    
    bool
    DidStripReference () const
    {
        return m_stripped_reference;
    }
    
    bool
    DidStripTypedef () const
    {
        return m_stripped_typedef;
    }
    
    template <class Formatter>
    bool
    IsMatch (const std::shared_ptr<Formatter>& formatter_sp) const
    {
        if (!formatter_sp)
            return false;
        if (formatter_sp->Cascades() == false && DidStripTypedef())
            return false;
        if (formatter_sp->SkipsPointers() && DidStripPointer())
            return false;
        if (formatter_sp->SkipsReferences() && DidStripReference())
            return false;
        return true;
    }
    
private:
    ConstString m_type_name;
    uint32_t m_reason;
    bool m_stripped_pointer;
    bool m_stripped_reference;
    bool m_stripped_typedef;
};

typedef std::vector<FormattersMatchCandidate> FormattersMatchVector;
    
class TypeNameSpecifierImpl
{
public:
    TypeNameSpecifierImpl() :
    m_is_regex(false),
    m_type()
    {
    }
    
    TypeNameSpecifierImpl (const char* name, bool is_regex) :
    m_is_regex(is_regex),
    m_type()
    {
        if (name)
            m_type.m_type_name.assign(name);
    }
    
    // if constructing with a given type, is_regex cannot be true since we are
    // giving an exact type to match
    TypeNameSpecifierImpl (lldb::TypeSP type) :
    m_is_regex(false),
    m_type()
    {
        if (type)
        {
            m_type.m_type_name.assign(type->GetName().GetCString());
            m_type.m_type_pair.SetType(type);
        }
    }

    TypeNameSpecifierImpl (ClangASTType type) :
    m_is_regex(false),
    m_type()
    {
        if (type.IsValid())
        {
            m_type.m_type_name.assign(type.GetConstTypeName().GetCString());
            m_type.m_type_pair.SetType(type);
        }
    }
    
    const char*
    GetName()
    {
        if (m_type.m_type_name.size())
            return m_type.m_type_name.c_str();
        return NULL;
    }
    
    lldb::TypeSP
    GetTypeSP ()
    {
        if (m_type.m_type_pair.IsValid())
            return m_type.m_type_pair.GetTypeSP();
        return lldb::TypeSP();
    }
    
    ClangASTType
    GetClangASTType ()
    {
        if (m_type.m_type_pair.IsValid())
            return m_type.m_type_pair.GetClangASTType();
        return ClangASTType();
    }
    
    bool
    IsRegex()
    {
        return m_is_regex;
    }
    
private:
    bool m_is_regex;
    // this works better than TypeAndOrName because the latter only wraps a TypeSP
    // whereas TypePair can also be backed by a ClangASTType
    struct TypeOrName
    {
        std::string m_type_name;
        TypePair m_type_pair;
    };
    TypeOrName m_type;
    
    
private:
    DISALLOW_COPY_AND_ASSIGN(TypeNameSpecifierImpl);
};
    
} // namespace lldb_private

#endif	// lldb_FormatClasses_h_
