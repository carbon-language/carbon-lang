//===-- FormattersContainer.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_FormattersContainer_h_
#define lldb_FormattersContainer_h_

// C Includes
// C++ Includes

// Other libraries and framework includes
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "clang/AST/DeclObjC.h"

// Project includes
#include "lldb/lldb-public.h"

#include "lldb/Core/Log.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/ValueObject.h"

#include "lldb/DataFormatters/FormatClasses.h"
#include "lldb/DataFormatters/TypeFormat.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"

#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangASTType.h"

#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/TargetList.h"

namespace lldb_private {
    
// this file (and its. cpp) contain the low-level implementation of LLDB Data Visualization
// class DataVisualization is the high-level front-end of this feature
// clients should refer to that class as the entry-point into the data formatters
// unless they have a good reason to bypass it and prefer to use this file's objects directly
class IFormatChangeListener
{
public:
    virtual void
    Changed () = 0;
    
    virtual
    ~IFormatChangeListener () {}
    
    virtual uint32_t
    GetCurrentRevision () = 0;
    
};
    
static inline bool
IsWhitespace (char c)
{
    return ( (c == ' ') || (c == '\t') || (c == '\v') || (c == '\f') );
}

static inline bool
HasPrefix (const char* str1, const char* str2)
{
    return ( ::strstr(str1, str2) == str1 );
}
    
// if the user tries to add formatters for, say, "struct Foo"
// those will not match any type because of the way we strip qualifiers from typenames
// this method looks for the case where the user is adding a "class","struct","enum" or "union" Foo
// and strips the unnecessary qualifier
static inline ConstString
GetValidTypeName_Impl (const ConstString& type)
{
    int strip_len = 0;
    
    if ((bool)type == false)
        return type;
    
    const char* type_cstr = type.AsCString();
    
    if ( HasPrefix(type_cstr, "class ") )
        strip_len = 6;
    else if ( HasPrefix(type_cstr, "enum ") )
        strip_len = 5;
    else if ( HasPrefix(type_cstr, "struct ") )
        strip_len = 7;
    else if ( HasPrefix(type_cstr, "union ") )
        strip_len = 6;
    
    if (strip_len == 0)
        return type;
    
    type_cstr += strip_len;
    while (IsWhitespace(*type_cstr) && ++type_cstr)
        ;
    
    return ConstString(type_cstr);
}
    
template<typename KeyType, typename ValueType>
class FormattersContainer;

template<typename KeyType, typename ValueType>
class FormatMap
{
public:

    typedef typename ValueType::SharedPointer ValueSP;
    typedef std::map<KeyType, ValueSP> MapType;
    typedef typename MapType::iterator MapIterator;
    typedef bool(*CallbackType)(void*, KeyType, const ValueSP&);
    
    FormatMap(IFormatChangeListener* lst) :
    m_map(),
    m_map_mutex(Mutex::eMutexTypeRecursive),
    listener(lst)
    {
    }
    
    void
    Add(KeyType name,
        const ValueSP& entry)
    {
        if (listener)
            entry->GetRevision() = listener->GetCurrentRevision();
        else
            entry->GetRevision() = 0;

        Mutex::Locker locker(m_map_mutex);
        m_map[name] = entry;
        if (listener)
            listener->Changed();
    }
    
    bool
    Delete (KeyType name)
    {
        Mutex::Locker locker(m_map_mutex);
        MapIterator iter = m_map.find(name);
        if (iter == m_map.end())
            return false;
        m_map.erase(name);
        if (listener)
            listener->Changed();
        return true;
    }
    
    void
    Clear ()
    {
        Mutex::Locker locker(m_map_mutex);
        m_map.clear();
        if (listener)
            listener->Changed();
    }
    
    bool
    Get(KeyType name,
        ValueSP& entry)
    {
        Mutex::Locker locker(m_map_mutex);
        MapIterator iter = m_map.find(name);
        if (iter == m_map.end())
            return false;
        entry = iter->second;
        return true;
    }
    
    void
    LoopThrough (CallbackType callback, void* param)
    {
        if (callback)
        {
            Mutex::Locker locker(m_map_mutex);
            MapIterator pos, end = m_map.end();
            for (pos = m_map.begin(); pos != end; pos++)
            {
                KeyType type = pos->first;
                if (!callback(param, type, pos->second))
                    break;
            }
        }
    }
    
    uint32_t
    GetCount ()
    {
        return m_map.size();
    }
    
    ValueSP
    GetValueAtIndex (size_t index)
    {
        Mutex::Locker locker(m_map_mutex);
        MapIterator iter = m_map.begin();
        MapIterator end = m_map.end();
        while (index > 0)
        {
            iter++;
            index--;
            if (end == iter)
                return ValueSP();
        }
        return iter->second;
    }
    
    KeyType
    GetKeyAtIndex (size_t index)
    {
        Mutex::Locker locker(m_map_mutex);
        MapIterator iter = m_map.begin();
        MapIterator end = m_map.end();
        while (index > 0)
        {
            iter++;
            index--;
            if (end == iter)
                return KeyType();
        }
        return iter->first;
    }
    
protected:
    MapType m_map;    
    Mutex m_map_mutex;
    IFormatChangeListener* listener;
    
    MapType&
    map ()
    {
        return m_map;
    }
    
    Mutex&
    mutex ()
    {
        return m_map_mutex;
    }
    
    friend class FormattersContainer<KeyType, ValueType>;
    friend class FormatManager;
    
};
    
template<typename KeyType, typename ValueType>
class FormattersContainer
{
protected:
    typedef FormatMap<KeyType,ValueType> BackEndType;
    
public:
    typedef typename BackEndType::MapType MapType;
    typedef typename MapType::iterator MapIterator;
    typedef typename MapType::key_type MapKeyType;
    typedef typename MapType::mapped_type MapValueType;
    typedef typename BackEndType::CallbackType CallbackType;
    typedef typename std::shared_ptr<FormattersContainer<KeyType, ValueType> > SharedPointer;
    
    friend class TypeCategoryImpl;

    FormattersContainer(std::string name,
                    IFormatChangeListener* lst) :
    m_format_map(lst),
    m_name(name),
    m_id_cs(ConstString("id"))
    {
    }
    
    void
    Add (const MapKeyType &type, const MapValueType& entry)
    {
        Add_Impl(type, entry, (KeyType*)NULL);
    }
    
    bool
    Delete (ConstString type)
    {
        return Delete_Impl(type, (KeyType*)NULL);
    }
        
    bool
    Get(ValueObject& valobj,
        MapValueType& entry,
        lldb::DynamicValueType use_dynamic,
        uint32_t* why = NULL)
    {
        uint32_t value = lldb_private::eFormatterChoiceCriterionDirectChoice;
        ClangASTType ast_type(valobj.GetClangType());
        bool ret = Get(valobj, ast_type, entry, use_dynamic, value);
        if (ret)
            entry = MapValueType(entry);
        else
            entry = MapValueType();        
        if (why)
            *why = value;
        return ret;
    }
    
    bool
    Get (ConstString type, MapValueType& entry)
    {
        return Get_Impl(type, entry, (KeyType*)NULL);
    }
    
    bool
    GetExact (ConstString type, MapValueType& entry)
    {
        return GetExact_Impl(type, entry, (KeyType*)NULL);
    }
    
    MapValueType
    GetAtIndex (size_t index)
    {
        return m_format_map.GetValueAtIndex(index);
    }
    
    lldb::TypeNameSpecifierImplSP
    GetTypeNameSpecifierAtIndex (size_t index)
    {
        return GetTypeNameSpecifierAtIndex_Impl(index, (KeyType*)NULL);
    }
    
    void
    Clear ()
    {
        m_format_map.Clear();
    }
    
    void
    LoopThrough (CallbackType callback, void* param)
    {
        m_format_map.LoopThrough(callback,param);
    }
    
    uint32_t
    GetCount ()
    {
        return m_format_map.GetCount();
    }
    
protected:
        
    BackEndType m_format_map;
    
    std::string m_name;
    
    DISALLOW_COPY_AND_ASSIGN(FormattersContainer);
    
    ConstString m_id_cs;
                           
    void
    Add_Impl (const MapKeyType &type, const MapValueType& entry, lldb::RegularExpressionSP *dummy)
    {
       m_format_map.Add(type,entry);
    }

    void Add_Impl (const ConstString &type, const MapValueType& entry, ConstString *dummy)
    {
       m_format_map.Add(GetValidTypeName_Impl(type), entry);
    }

    bool
    Delete_Impl (ConstString type, ConstString *dummy)
    {
       return m_format_map.Delete(type);
    }

    bool
    Delete_Impl (ConstString type, lldb::RegularExpressionSP *dummy)
    {
       Mutex& x_mutex = m_format_map.mutex();
        lldb_private::Mutex::Locker locker(x_mutex);
       MapIterator pos, end = m_format_map.map().end();
       for (pos = m_format_map.map().begin(); pos != end; pos++)
       {
           lldb::RegularExpressionSP regex = pos->first;
           if ( ::strcmp(type.AsCString(),regex->GetText()) == 0)
           {
               m_format_map.map().erase(pos);
               if (m_format_map.listener)
                   m_format_map.listener->Changed();
               return true;
           }
       }
       return false;
    }    

    bool
    Get_Impl (ConstString type, MapValueType& entry, ConstString *dummy)
    {
       return m_format_map.Get(type, entry);
    }

    bool
    GetExact_Impl (ConstString type, MapValueType& entry, ConstString *dummy)
    {
        return Get_Impl(type,entry, (KeyType*)0);
    }
    
    lldb::TypeNameSpecifierImplSP
    GetTypeNameSpecifierAtIndex_Impl (size_t index, ConstString *dummy)
    {
        ConstString key = m_format_map.GetKeyAtIndex(index);
        if (key)
            return lldb::TypeNameSpecifierImplSP(new TypeNameSpecifierImpl(key.AsCString(),
                                                                           false));
        else
            return lldb::TypeNameSpecifierImplSP();
    }
    
    lldb::TypeNameSpecifierImplSP
    GetTypeNameSpecifierAtIndex_Impl (size_t index, lldb::RegularExpressionSP *dummy)
    {
        lldb::RegularExpressionSP regex = m_format_map.GetKeyAtIndex(index);
        if (regex.get() == NULL)
            return lldb::TypeNameSpecifierImplSP();
        return lldb::TypeNameSpecifierImplSP(new TypeNameSpecifierImpl(regex->GetText(),
                                                                       true));
    }

    bool
    Get_Impl (ConstString key, MapValueType& value, lldb::RegularExpressionSP *dummy)
    {
       const char* key_cstr = key.AsCString();
       if (!key_cstr)
           return false;
       Mutex& x_mutex = m_format_map.mutex();
       lldb_private::Mutex::Locker locker(x_mutex);
       MapIterator pos, end = m_format_map.map().end();
       for (pos = m_format_map.map().begin(); pos != end; pos++)
       {
           lldb::RegularExpressionSP regex = pos->first;
           if (regex->Execute(key_cstr))
           {
               value = pos->second;
               return true;
           }
       }
       return false;
    }
    
    bool
    GetExact_Impl (ConstString key, MapValueType& value, lldb::RegularExpressionSP *dummy)
    {
        Mutex& x_mutex = m_format_map.mutex();
        lldb_private::Mutex::Locker locker(x_mutex);
        MapIterator pos, end = m_format_map.map().end();
        for (pos = m_format_map.map().begin(); pos != end; pos++)
        {
            lldb::RegularExpressionSP regex = pos->first;
            if (strcmp(regex->GetText(),key.AsCString()) == 0)
            {
                value = pos->second;
                return true;
            }
        }
        return false;
    }

    bool
    Get (const FormattersMatchVector& candidates,
         MapValueType& entry,
         uint32_t *reason)
    {
        for (const FormattersMatchCandidate& candidate : candidates)
        {
            // FIXME: could we do the IsMatch() check first?
            if (Get(candidate.GetTypeName(),entry))
            {
                if (candidate.IsMatch(entry) == false)
                {
                    entry.reset();
                    continue;
                }
                else
                {
                    if(reason)
                        *reason = candidate.GetReason();
                    return true;
                }
            }
        }
        return false;
    }
};

} // namespace lldb_private

#endif	// lldb_FormattersContainer_h_
