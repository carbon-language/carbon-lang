//===-- FormatManager.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_FormatManager_h_
#define lldb_FormatManager_h_

// C Includes

#include <stdint.h>
#include <unistd.h>

// C++ Includes

#ifdef __GNUC__
#include <ext/hash_map>

namespace std
{
    using namespace __gnu_cxx;
}

#else
#include <hash_map>
#endif

#include <map>
#include <stack>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"
#include "lldb/lldb-enumerations.h"

#include "lldb/Core/Communication.h"
#include "lldb/Core/InputReaderStack.h"
#include "lldb/Core/Listener.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Core/UserID.h"
#include "lldb/Core/UserSettingsController.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/TargetList.h"

#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"

namespace lldb_private {
    
class IFormatChangeListener
{
public:
    virtual void
    Changed() = 0;
    
    virtual
    ~IFormatChangeListener() {}
    
};

struct SummaryFormat
{
    std::string m_format;
    bool m_dont_show_children;
    bool m_dont_show_value;
    bool m_show_members_oneliner;
    bool m_cascades;
    SummaryFormat(std::string f = "", bool c = false, bool nochildren = true, bool novalue = true, bool oneliner = false) :
    m_format(f),
    m_dont_show_children(nochildren),
    m_dont_show_value(novalue),
    m_show_members_oneliner(oneliner),
    m_cascades(c)
    {
    }
    
    bool
    DoesPrintChildren() const
    {
        return !m_dont_show_children;
    }
    
    bool
    DoesPrintValue() const
    {
        return !m_dont_show_value;
    }
    
    bool
    IsOneliner() const
    {
        return m_show_members_oneliner;
    }
    
    typedef lldb::SharedPtr<SummaryFormat>::Type SharedPointer;
    
};

struct ValueFormat
{
    lldb::Format m_format;
    bool m_cascades;
    ValueFormat (lldb::Format f = lldb::eFormatInvalid, bool c = false) : 
    m_format (f), 
    m_cascades (c)
    {
    }
    
    typedef lldb::SharedPtr<ValueFormat>::Type SharedPointer;
    
    ~ValueFormat()
    {
    }
    
};
    
template<typename MapType, typename CallbackType>
class FormatNavigator
{
public:

    typedef typename MapType::iterator MapIterator;
    typedef typename MapType::key_type MapKeyType;
    typedef typename MapType::mapped_type MapValueType;
    
    FormatNavigator(IFormatChangeListener* lst = NULL) :
    m_map_mutex(Mutex::eMutexTypeRecursive),
    m_map(MapType()),
    listener(lst)
    {
    }
        
    bool
    Get(ValueObject& vobj, MapValueType& entry)
    {
        Mutex::Locker(m_map_mutex);
        clang::QualType type = clang::QualType::getFromOpaquePtr(vobj.GetClangType());
        bool ret = Get(vobj, type, entry);
        if(ret)
            entry = MapValueType(entry);
        else
            entry = MapValueType();
        return ret;
    }
    
    void
    Add(const MapKeyType &type, const MapValueType& entry)
    {
        Mutex::Locker(m_map_mutex);
        m_map[type] = MapValueType(entry);
        if(listener)
            listener->Changed();
    }
    
    bool
    Delete(const MapKeyType& type)
    {
        Mutex::Locker(m_map_mutex);
        MapIterator iter = m_map.find(type);
        if (iter == m_map.end())
            return false;
        m_map.erase(type);
        if(listener)
            listener->Changed();
        return true;
    }
    
    void
    Clear()
    {
        Mutex::Locker(m_map_mutex);
        m_map.clear();
        if(listener)
            listener->Changed();
    }
    
    void
    LoopThrough(CallbackType callback, void* param)
    {
        if (callback)
        {
            Mutex::Locker(m_map_mutex);
            MapIterator pos, end = m_map.end();
            for (pos = m_map.begin(); pos != end; pos++)
            {
                MapKeyType type = pos->first;
                if(!callback(param, type, MapValueType(pos->second)))
                    break;
            }
        }
    }
    
    ~FormatNavigator()
    {
    }
    
private:
    
    Mutex m_map_mutex;
    MapType m_map;
    IFormatChangeListener* listener;
    
    DISALLOW_COPY_AND_ASSIGN(FormatNavigator);
    
    bool
    Get(const MapKeyType &type, MapValueType& entry)
    {
        Mutex::Locker(m_map_mutex);
        MapIterator iter = m_map.find(type);
        if (iter == m_map.end())
            return false;
        entry = iter->second;
        return true;
    }
    
    bool Get(ValueObject& vobj,
             const clang::QualType& q_type,
             MapValueType& entry)
    {
        if (q_type.isNull())
            return false;
        clang::QualType type = q_type.getUnqualifiedType();
        type.removeLocalConst(); type.removeLocalVolatile(); type.removeLocalRestrict();
        ConstString name(type.getAsString().c_str());
        //printf("trying to get format for VO name %s of type %s\n",vobj.GetName().AsCString(),name.AsCString());
        if (Get(name.GetCString(), entry))
            return true;
        // look for a "base type", whatever that means
        const clang::Type* typePtr = type.getTypePtrOrNull();
        if (!typePtr)
            return false;
        if (typePtr->isReferenceType())
            return Get(vobj,type.getNonReferenceType(),entry);
        // for C++ classes, navigate up the hierarchy
        if (typePtr->isRecordType())
        {
            clang::CXXRecordDecl* record = typePtr->getAsCXXRecordDecl();
            if (record)
            {
                if (!record->hasDefinition())
                    // dummy call to do the complete
                    ClangASTContext::GetNumChildren(vobj.GetClangAST(), vobj.GetClangType(), false);
                clang::IdentifierInfo *info = record->getIdentifier();
                if (info) {
                    // this is the class name, plain and simple
                    ConstString id_info(info->getName().str().c_str());
                    if (Get(id_info.GetCString(), entry))
                        return true;
                }
                if (record->hasDefinition())
                {
                    clang::CXXRecordDecl::base_class_iterator pos,end;
                    if( record->getNumBases() > 0)
                    {
                        end = record->bases_end();
                        for (pos = record->bases_begin(); pos != end; pos++)
                        {
                            if((Get(vobj, pos->getType(), entry)) && entry->m_cascades)
                                return true; // if it does not cascade, just move on to other base classes which might
                        }
                    }
                    if (record->getNumVBases() > 0)
                    {
                        end = record->vbases_end();
                        for (pos = record->vbases_begin(); pos != end; pos++)
                        {
                            if((Get(vobj, pos->getType(), entry)) && entry->m_cascades)
                                return true;
                        }
                    }
                }
            }
        }
        // try to strip typedef chains
        const clang::TypedefType* type_tdef = type->getAs<clang::TypedefType>();
        if (type_tdef)
            if ((Get(vobj, type_tdef->getDecl()->getUnderlyingType(), entry)) && entry->m_cascades)
                return true;
        return false;
    }
    
};

class FormatManager : public IFormatChangeListener
{
    
public:
    
    typedef bool(*ValueCallback)(void*, const char*, const ValueFormat::SharedPointer&);
    typedef bool(*SummaryCallback)(void*, const char*, const SummaryFormat::SharedPointer&);
    
private:
    
    typedef std::map<const char*, ValueFormat::SharedPointer> ValueMap;
    typedef std::map<const char*, SummaryFormat::SharedPointer> SummaryMap;
    
    typedef FormatNavigator<ValueMap, ValueCallback> ValueNavigator;
    typedef FormatNavigator<SummaryMap, SummaryCallback> SummaryNavigator;
    
    ValueNavigator m_value_nav;
    SummaryNavigator m_summary_nav;
        
    uint32_t m_last_revision;
    
public:
        
    FormatManager() : 
    m_value_nav(this),
    m_summary_nav(this),
    m_last_revision(0)
    {
    }


    ValueNavigator& Value() { return m_value_nav; }
    SummaryNavigator& Summary() { return m_summary_nav; }
    
    static bool
    GetFormatFromCString (const char *format_cstr,
                          bool partial_match_ok,
                          lldb::Format &format);

    static char
    GetFormatAsFormatChar (lldb::Format format);

    static const char *
    GetFormatAsCString (lldb::Format format);
    
    void
    Changed()
    {
        __sync_add_and_fetch(&m_last_revision, +1);
    }
    
    uint32_t
    GetCurrentRevision() const
    {
        return m_last_revision;
    }
    
    ~FormatManager()
    {
    }

};

} // namespace lldb_private

#endif	// lldb_FormatManager_h_
