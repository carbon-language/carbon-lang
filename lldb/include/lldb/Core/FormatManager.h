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
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "clang/AST/DeclObjC.h"

// Project includes
#include "lldb/lldb-public.h"
#include "lldb/lldb-enumerations.h"

#include "lldb/Core/Communication.h"
#include "lldb/Core/InputReaderStack.h"
#include "lldb/Core/Listener.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Core/UserID.h"
#include "lldb/Core/UserSettingsController.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/TargetList.h"

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
    bool m_skip_references;
    bool m_skip_pointers;
    SummaryFormat(std::string f = "",
                  bool c = false,
                  bool nochildren = true,
                  bool novalue = true,
                  bool oneliner = false,
                  bool skipptr = false,
                  bool skipref = false) :
    m_format(f),
    m_dont_show_children(nochildren),
    m_dont_show_value(novalue),
    m_show_members_oneliner(oneliner),
    m_cascades(c),
    m_skip_references(skipref),
    m_skip_pointers(skipptr)
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
    typedef bool(*SummaryCallback)(void*, const char*, const SummaryFormat::SharedPointer&);
    typedef bool(*RegexSummaryCallback)(void*, lldb::RegularExpressionSP, const SummaryFormat::SharedPointer&);
    
};

struct ValueFormat
{
    lldb::Format m_format;
    bool m_cascades;
    bool m_skip_references;
    bool m_skip_pointers;
    ValueFormat (lldb::Format f = lldb::eFormatInvalid,
                 bool c = false,
                 bool skipptr = false,
                 bool skipref = false) : 
    m_format (f), 
    m_cascades (c),
    m_skip_references(skipref),
    m_skip_pointers(skipptr)
    {
    }
    
    typedef lldb::SharedPtr<ValueFormat>::Type SharedPointer;
    typedef bool(*ValueCallback)(void*, const char*, const ValueFormat::SharedPointer&);
    
    ~ValueFormat()
    {
    }
    
};

template<typename KeyType, typename ValueType>
class FormatNavigator;
    
template<typename KeyType, typename ValueType>
class FormatMap
{
private:
    typedef typename ValueType::SharedPointer ValueSP;
    Mutex m_map_mutex;
    IFormatChangeListener* listener;
    
    friend class FormatNavigator<KeyType, ValueType>;
    
public:
    typedef std::map<KeyType, ValueSP> MapType;

private:    
    MapType m_map;
    
    MapType& map()
    {
        return m_map;
    }
    
    Mutex& mutex()
    {
        return m_map_mutex;
    }

public:

    typedef typename MapType::iterator MapIterator;
    typedef bool(*CallbackType)(void*, KeyType, const ValueSP&);
    
    FormatMap(IFormatChangeListener* lst = NULL) :
    m_map(),
    m_map_mutex(Mutex::eMutexTypeRecursive),
    listener(lst)
    {
    }
    
    void
    Add(KeyType name,
        const ValueSP& entry)
    {
        Mutex::Locker(m_map_mutex);
        m_map[name] = entry;
        if (listener)
            listener->Changed();
    }
    
    bool
    Delete(KeyType name)
    {
        Mutex::Locker(m_map_mutex);
        MapIterator iter = m_map.find(name);
        if (iter == m_map.end())
            return false;
        m_map.erase(name);
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
    
    bool
    Get(KeyType name,
        ValueSP& entry)
    {
        Mutex::Locker(m_map_mutex);
        MapIterator iter = m_map.find(name);
        if (iter == m_map.end())
            return false;
        entry = iter->second;
        return true;
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
                KeyType type = pos->first;
                if(!callback(param, type, pos->second))
                    break;
            }
        }
    }
    
    uint32_t
    GetCount()
    {
        return m_map.size();
    }
    
};

template<typename KeyType, typename ValueType>
class FormatNavigator
{
private:
    typedef FormatMap<KeyType,ValueType> BackEndType;
    
    BackEndType m_format_map;
        
public:
    typedef typename BackEndType::MapType MapType;
    typedef typename MapType::iterator MapIterator;
    typedef typename MapType::key_type MapKeyType;
    typedef typename MapType::mapped_type MapValueType;
    typedef typename BackEndType::CallbackType CallbackType;

    FormatNavigator(IFormatChangeListener* lst = NULL) :
    m_format_map(lst)
    {
    }
    
    void
    Add(const MapKeyType &type, const MapValueType& entry)
    {
        m_format_map.Add(type,entry);
    }
    
    // using const char* instead of MapKeyType is necessary here
    // to make the partial template specializations below work
    bool
    Delete(const char *type)
    {
        return m_format_map.Delete(type);
    }
    
    bool
    Get(ValueObject& vobj, MapValueType& entry)
    {
        clang::QualType type = clang::QualType::getFromOpaquePtr(vobj.GetClangType());
        bool ret = Get(vobj, type, entry);
        if(ret)
            entry = MapValueType(entry);
        else
            entry = MapValueType();
        return ret;
    }
    
    void
    Clear()
    {
        m_format_map.Clear();
    }
    
    void
    LoopThrough(CallbackType callback, void* param)
    {
        m_format_map.LoopThrough(callback,param);
    }
    
    uint32_t
    GetCount()
    {
        return m_format_map.GetCount();
    }

private:
    
    DISALLOW_COPY_AND_ASSIGN(FormatNavigator);
    
    // using const char* instead of MapKeyType is necessary here
    // to make the partial template specializations below work
    bool
    Get(const char* type, MapValueType& entry)
    {
        return m_format_map.Get(type, entry);
    }
    
    bool Get(ValueObject& vobj,
             clang::QualType type,
             MapValueType& entry)
    {
        if (type.isNull())
            return false;
        // clang::QualType type = q_type.getUnqualifiedType();
        type.removeLocalConst(); type.removeLocalVolatile(); type.removeLocalRestrict();
        const clang::Type* typePtr = type.getTypePtrOrNull();
        if (!typePtr)
            return false;
        ConstString name(ClangASTType::GetTypeNameForQualType(type).c_str());
        if(vobj.GetBitfieldBitSize() > 0)
        {
            // for bitfields, append size to the typename so one can custom format them
            StreamString sstring;
            sstring.Printf("%s:%d",name.AsCString(),vobj.GetBitfieldBitSize());
            name = ConstString(sstring.GetData());
        }
        //printf("trying to get format for VO name %s of type %s\n",vobj.GetName().AsCString(),name.AsCString());
        if (Get(name.GetCString(), entry))
            return true;
        // look for a "base type", whatever that means
        if (typePtr->isReferenceType())
        {
            if (Get(vobj,type.getNonReferenceType(),entry) && !entry->m_skip_references)
                return true;
        }
        if (typePtr->isPointerType())
        {
            if (Get(vobj, typePtr->getPointeeType(), entry) && !entry->m_skip_pointers)
                return true;
        }
        if (typePtr->isObjCObjectPointerType())
        {
            /*
             for some reason, C++ can quite easily obtain the type hierarchy for a ValueObject
             even if the VO represent a pointer-to-class, as long as the typePtr is right
             Objective-C on the other hand cannot really complete an @interface when
             the VO refers to a pointer-to-@interface
             */
            Error error;
            ValueObject* target = vobj.Dereference(error).get();
            if(error.Fail() || !target)
                return false;
            if (Get(*target, typePtr->getPointeeType(), entry) && !entry->m_skip_pointers)
                return true;
        }
        const clang::ObjCObjectType *objc_class_type = typePtr->getAs<clang::ObjCObjectType>();
        if (objc_class_type)
        {
            //printf("working with ObjC\n");
            clang::ASTContext *ast = vobj.GetClangAST();
            if (ClangASTContext::GetCompleteType(ast, vobj.GetClangType()) && !objc_class_type->isObjCId())
            {
                clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                if(class_interface_decl)
                {
                    //printf("down here\n");
                    clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                    //printf("one further step and we're there...\n");
                    if(superclass_interface_decl)
                    {
                        //printf("the end is here\n");
                        clang::QualType ivar_qual_type(ast->getObjCInterfaceType(superclass_interface_decl));
                        if (Get(vobj, ivar_qual_type, entry) && entry->m_cascades)
                            return true;
                    }
                }
            }
        }
        // for C++ classes, navigate up the hierarchy
        if (typePtr->isRecordType())
        {
            clang::CXXRecordDecl* record = typePtr->getAsCXXRecordDecl();
            if (record)
            {
                if (!record->hasDefinition())
                    ClangASTContext::GetCompleteType(vobj.GetClangAST(), vobj.GetClangType());
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
        {
            if ((Get(vobj, type_tdef->getDecl()->getUnderlyingType(), entry)) && entry->m_cascades)
                return true;
        }
        return false;
    }
};

template<>
bool
FormatNavigator<lldb::RegularExpressionSP, SummaryFormat>::Get(const char* key, SummaryFormat::SharedPointer& value);

template<>
bool
FormatNavigator<lldb::RegularExpressionSP, SummaryFormat>::Delete(const char* type);

class FormatManager : public IFormatChangeListener
{
private:
    
    typedef FormatNavigator<const char*, ValueFormat> ValueNavigator;
    typedef FormatNavigator<const char*, SummaryFormat> SummaryNavigator;
    typedef FormatNavigator<lldb::RegularExpressionSP, SummaryFormat> RegexSummaryNavigator;
    
    typedef ValueNavigator::MapType ValueMap;
    typedef SummaryNavigator::MapType SummaryMap;
    typedef RegexSummaryNavigator::MapType RegexSummaryMap;
    typedef FormatMap<const char*, SummaryFormat> NamedSummariesMap;
    
    ValueNavigator m_value_nav;
    SummaryNavigator m_summary_nav;
    RegexSummaryNavigator m_regex_summary_nav;
    
    NamedSummariesMap m_named_summaries_map;
    
    uint32_t m_last_revision;
    
public:
        
    FormatManager() : 
    m_value_nav(this),
    m_summary_nav(this),
    m_regex_summary_nav(this),
    m_named_summaries_map(this),
    m_last_revision(0)
    {
    }


    ValueNavigator& Value() { return m_value_nav; }
    SummaryNavigator& Summary() { return m_summary_nav; }
    RegexSummaryNavigator& RegexSummary() { return m_regex_summary_nav; }
    NamedSummariesMap& NamedSummary() { return m_named_summaries_map; }
    
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
