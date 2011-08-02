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

#include <list>
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
#include "lldb/Core/FormatClasses.h"
#include "lldb/Core/InputReaderStack.h"
#include "lldb/Core/Listener.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Core/UserID.h"
#include "lldb/Core/UserSettingsController.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/TargetList.h"

using lldb::LogSP;

namespace lldb_private {
    
class IFormatChangeListener
{
public:
    virtual void
    Changed() = 0;
    
    virtual
    ~IFormatChangeListener() {}
    
};
    
template<typename KeyType, typename ValueType>
class FormatNavigator;

template<typename KeyType, typename ValueType>
class FormatMap
{
    friend class FormatNavigator<KeyType, ValueType>;
    friend class FormatManager;

public:

    typedef typename ValueType::SharedPointer ValueSP;
    typedef std::map<KeyType, ValueSP> MapType;
    typedef typename MapType::iterator MapIterator;
    typedef bool(*CallbackType)(void*, KeyType, const ValueSP&);
    
private:
    MapType m_map;    
    Mutex m_map_mutex;
    IFormatChangeListener* listener;
    
    MapType& map()
    {
        return m_map;
    }
    
    Mutex& mutex()
    {
        return m_map_mutex;
    }

public:
    
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
        if (listener)
            listener->Changed();
        return true;
    }
    
    void
    Clear()
    {
        Mutex::Locker(m_map_mutex);
        m_map.clear();
        if (listener)
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
                if (!callback(param, type, pos->second))
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
    
    std::string m_name;
        
public:
    typedef typename BackEndType::MapType MapType;
    typedef typename MapType::iterator MapIterator;
    typedef typename MapType::key_type MapKeyType;
    typedef typename MapType::mapped_type MapValueType;
    typedef typename BackEndType::CallbackType CallbackType;
    
    typedef typename lldb::SharedPtr<FormatNavigator<KeyType, ValueType> >::Type SharedPointer;
    
    friend class FormatCategory;

    FormatNavigator(std::string name,
                    IFormatChangeListener* lst = NULL) :
    m_format_map(lst),
    m_name(name)
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
    Get(ValueObject& vobj,
        MapValueType& entry,
        lldb::DynamicValueType use_dynamic,
        uint32_t* why = NULL)
    {
        uint32_t value = lldb::eFormatterChoiceCriterionDirectChoice;
        clang::QualType type = clang::QualType::getFromOpaquePtr(vobj.GetClangType());
        bool ret = Get(vobj, type, entry, use_dynamic, value);
        if (ret)
            entry = MapValueType(entry);
        else
            entry = MapValueType();        
        if (why)
            *why = value;
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
    
    bool Get_ObjC(ValueObject& vobj,
             ObjCLanguageRuntime::ObjCISA isa,
             MapValueType& entry,
             uint32_t& reason)
    {
        LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_TYPES));
        if (log)
            log->Printf("going to an Objective-C dynamic scanning");
        Process* process = vobj.GetUpdatePoint().GetProcessSP().get();
        ObjCLanguageRuntime* runtime = process->GetObjCLanguageRuntime();
        if (runtime == NULL)
        {
            if (log)
                log->Printf("no valid ObjC runtime, bailing out");
            return false;
        }
        if (runtime->IsValidISA(isa) == false)
        {
            if (log)
                log->Printf("invalid ISA, bailing out");
            return false;
        }
        ConstString name = runtime->GetActualTypeName(isa);
        if (log)
            log->Printf("looking for formatter for %s", name.GetCString());
        if (Get(name.GetCString(), entry))
        {
            if (log)
                log->Printf("direct match found, returning");
            return true;
        }
        if (log)
            log->Printf("no direct match");
        ObjCLanguageRuntime::ObjCISA parent = runtime->GetParentClass(isa);
        if (runtime->IsValidISA(parent) == false)
        {
            if (log)
                log->Printf("invalid parent ISA, bailing out");
            return false;
        }
        if (parent == isa)
        {
            if (log)
                log->Printf("parent-child loop, bailing out");
            return false;
        }
        if (Get_ObjC(vobj, parent, entry, reason))
        {
            reason |= lldb::eFormatterChoiceCriterionNavigatedBaseClasses;
            return true;
        }
        return false;
    }
    
    bool Get(ValueObject& vobj,
             clang::QualType type,
             MapValueType& entry,
             lldb::DynamicValueType use_dynamic,
             uint32_t& reason)
    {
        LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_TYPES));
        if (type.isNull())
        {
            if (log)
                log->Printf("type is NULL, returning");
            return false;
        }
        // clang::QualType type = q_type.getUnqualifiedType();
        type.removeLocalConst(); type.removeLocalVolatile(); type.removeLocalRestrict();
        const clang::Type* typePtr = type.getTypePtrOrNull();
        if (!typePtr)
        {
            if (log)
                log->Printf("type is NULL, returning");
            return false;
        }
        ConstString name(ClangASTType::GetTypeNameForQualType(type).c_str());
        if (vobj.GetBitfieldBitSize() > 0)
        {
            // for bitfields, append size to the typename so one can custom format them
            StreamString sstring;
            sstring.Printf("%s:%d",name.AsCString(),vobj.GetBitfieldBitSize());
            name = ConstString(sstring.GetData());
            if (log)
                log->Printf("appended bitfield info, final result is %s", name.GetCString());
        }
        if (log)
            log->Printf("trying to get %s for VO name %s of type %s",
                        m_name.c_str(),
                        vobj.GetName().AsCString(),
                        name.AsCString());
        if (Get(name.GetCString(), entry))
        {
            if (log)
                log->Printf("direct match found, returning");
            return true;
        }
        if (log)
            log->Printf("no direct match");
        // look for a "base type", whatever that means
        if (typePtr->isReferenceType())
        {
            if (log)
                log->Printf("stripping reference");
            if (Get(vobj,type.getNonReferenceType(),entry, use_dynamic, reason) && !entry->m_skip_references)
            {
                reason |= lldb::eFormatterChoiceCriterionStrippedPointerReference;
                return true;
            }
        }
        if (use_dynamic != lldb::eNoDynamicValues &&
            typePtr == vobj.GetClangAST()->ObjCBuiltinIdTy.getTypePtr())
        {
            if (log)
                log->Printf("this is an ObjC 'id', let's do dynamic search");
            Process* process = vobj.GetUpdatePoint().GetProcessSP().get();
            ObjCLanguageRuntime* runtime = process->GetObjCLanguageRuntime();
            if (runtime == NULL)
            {
                if (log)
                    log->Printf("no valid ObjC runtime, skipping dynamic");
            }
            else
            {
                if (Get_ObjC(vobj, runtime->GetISA(vobj), entry, reason))
                {
                    reason |= lldb::eFormatterChoiceCriterionDynamicObjCHierarchy;
                    return true;
                }
            }
        }
        else if (use_dynamic != lldb::eNoDynamicValues && log)
        {
            log->Printf("typename: %s, typePtr = %p, id = %p",
                        name.AsCString(), typePtr, vobj.GetClangAST()->ObjCBuiltinIdTy.getTypePtr());
        }
        else if (log)
        {
            log->Printf("no dynamic");
        }
        if (typePtr->isPointerType())
        {
            if (log)
                log->Printf("stripping pointer");
            clang::QualType pointee = typePtr->getPointeeType();
            if (Get(vobj, pointee, entry, use_dynamic, reason) && !entry->m_skip_pointers)
            {
                reason |= lldb::eFormatterChoiceCriterionStrippedPointerReference;
                return true;
            }
        }
        if (typePtr->isObjCObjectPointerType())
        {
            if (use_dynamic != lldb::eNoDynamicValues &&
                name.GetCString() == ConstString("id").GetCString())
            {
                if (log)
                    log->Printf("this is an ObjC 'id', let's do dynamic search");
                Process* process = vobj.GetUpdatePoint().GetProcessSP().get();
                ObjCLanguageRuntime* runtime = process->GetObjCLanguageRuntime();
                if (runtime == NULL)
                {
                    if (log)
                        log->Printf("no valid ObjC runtime, skipping dynamic");
                }
                else
                {
                    if (Get_ObjC(vobj, runtime->GetISA(vobj), entry, reason))
                    {
                        reason |= lldb::eFormatterChoiceCriterionDynamicObjCHierarchy;
                        return true;
                    }
                }
            }
            if (log)
                log->Printf("stripping ObjC pointer");
            /*
             for some reason, C++ can quite easily obtain the type hierarchy for a ValueObject
             even if the VO represent a pointer-to-class, as long as the typePtr is right
             Objective-C on the other hand cannot really complete an @interface when
             the VO refers to a pointer-to-@interface
             */
            Error error;
            ValueObject* target = vobj.Dereference(error).get();
            if (error.Fail() || !target)
                return false;
            if (Get(*target, typePtr->getPointeeType(), entry, use_dynamic, reason) && !entry->m_skip_pointers)
            {
                reason |= lldb::eFormatterChoiceCriterionStrippedPointerReference;
                return true;
            }
        }
        const clang::ObjCObjectType *objc_class_type = typePtr->getAs<clang::ObjCObjectType>();
        if (objc_class_type)
        {
            if (log)
                log->Printf("working with ObjC");
            clang::ASTContext *ast = vobj.GetClangAST();
            if (ClangASTContext::GetCompleteType(ast, vobj.GetClangType()) && !objc_class_type->isObjCId())
            {
                clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                if (class_interface_decl)
                {
                    if (log)
                        log->Printf("got an ObjCInterfaceDecl");
                    clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                    if (superclass_interface_decl)
                    {
                        if (log)
                            log->Printf("got a parent class for this ObjC class");
                        clang::QualType ivar_qual_type(ast->getObjCInterfaceType(superclass_interface_decl));
                        if (Get(vobj, ivar_qual_type, entry, use_dynamic, reason) && entry->m_cascades)
                        {
                            reason |= lldb::eFormatterChoiceCriterionNavigatedBaseClasses;
                            return true;
                        }
                    }
                }
            }
        }
        // for C++ classes, navigate up the hierarchy
        if (typePtr->isRecordType())
        {
            if (log)
                log->Printf("working with C++");
            clang::CXXRecordDecl* record = typePtr->getAsCXXRecordDecl();
            if (record)
            {
                if (!record->hasDefinition())
                    ClangASTContext::GetCompleteType(vobj.GetClangAST(), vobj.GetClangType());
                if (record->hasDefinition())
                {
                    clang::CXXRecordDecl::base_class_iterator pos,end;
                    if (record->getNumBases() > 0)
                    {
                        if (log)
                            log->Printf("look into bases");
                        end = record->bases_end();
                        for (pos = record->bases_begin(); pos != end; pos++)
                        {
                            if ((Get(vobj, pos->getType(), entry, use_dynamic, reason)) && entry->m_cascades)
                            {
                                reason |= lldb::eFormatterChoiceCriterionNavigatedBaseClasses;
                                return true;
                            }
                        }
                    }
                    if (record->getNumVBases() > 0)
                    {
                        if (log)
                            log->Printf("look into VBases");
                        end = record->vbases_end();
                        for (pos = record->vbases_begin(); pos != end; pos++)
                        {
                            if ((Get(vobj, pos->getType(), entry, use_dynamic, reason)) && entry->m_cascades)
                            {
                                reason |= lldb::eFormatterChoiceCriterionNavigatedBaseClasses;
                                return true;
                            }
                        }
                    }
                }
            }
        }
        // try to strip typedef chains
        const clang::TypedefType* type_tdef = type->getAs<clang::TypedefType>();
        if (type_tdef)
        {
            if (log)
                log->Printf("stripping typedef");
            if ((Get(vobj, type_tdef->getDecl()->getUnderlyingType(), entry, use_dynamic, reason)) && entry->m_cascades)
            {
                reason |= lldb::eFormatterChoiceCriterionNavigatedTypedefs;
                return true;
            }
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

class CategoryMap;
    
class FormatCategory
{
private:
    typedef FormatNavigator<const char*, SummaryFormat> SummaryNavigator;
    typedef FormatNavigator<lldb::RegularExpressionSP, SummaryFormat> RegexSummaryNavigator;
    typedef FormatNavigator<const char*, SyntheticFilter> FilterNavigator;
    
    typedef SummaryNavigator::MapType SummaryMap;
    typedef RegexSummaryNavigator::MapType RegexSummaryMap;
    typedef FilterNavigator::MapType FilterMap;
        
    SummaryNavigator::SharedPointer m_summary_nav;
    RegexSummaryNavigator::SharedPointer m_regex_summary_nav;
    FilterNavigator::SharedPointer m_filter_nav;
    
    bool m_enabled;
    
    IFormatChangeListener* m_change_listener;
    
    Mutex m_mutex;
    
    std::string m_name;
    
    void
    Enable(bool value = true)
    {
        Mutex::Locker(m_mutex);
        m_enabled = value;        
        if (m_change_listener)
            m_change_listener->Changed();
    }
    
    void
    Disable()
    {
        Enable(false);
    }
    
    friend class CategoryMap;
    
public:
    
    typedef SummaryNavigator::SharedPointer SummaryNavigatorSP;
    typedef RegexSummaryNavigator::SharedPointer RegexSummaryNavigatorSP;
    typedef FilterNavigator::SharedPointer FilterNavigatorSP;
    
    FormatCategory(IFormatChangeListener* clist,
                   std::string name) :
    m_summary_nav(new SummaryNavigator("summary",clist)),
    m_regex_summary_nav(new RegexSummaryNavigator("regex-summary",clist)),
    m_filter_nav(new FilterNavigator("filter",clist)),
    m_enabled(false),
    m_change_listener(clist),
    m_mutex(Mutex::eMutexTypeRecursive),
    m_name(name)
    {}
    
    SummaryNavigatorSP
    Summary()
    {
        return SummaryNavigatorSP(m_summary_nav);
    }
    
    RegexSummaryNavigatorSP
    RegexSummary()
    {
        return RegexSummaryNavigatorSP(m_regex_summary_nav);
    }
    
    FilterNavigatorSP
    Filter()
    {
        return FilterNavigatorSP(m_filter_nav);
    }
    
    bool
    IsEnabled() const
    {
        return m_enabled;
    }
        
    bool
    Get(ValueObject& vobj,
        lldb::SummaryFormatSP& entry,
        lldb::DynamicValueType use_dynamic,
        uint32_t* reason = NULL)
    {
        if (!IsEnabled())
            return false;
        if (Summary()->Get(vobj, entry, use_dynamic, reason))
            return true;
        bool regex = RegexSummary()->Get(vobj, entry, use_dynamic, reason);
        if (regex && reason)
            *reason |= lldb::eFormatterChoiceCriterionRegularExpressionSummary;
        return regex;
    }
    
    bool
    Get(ValueObject& vobj,
        lldb::SyntheticChildrenSP& entry,
        lldb::DynamicValueType use_dynamic,
        uint32_t* reason = NULL)
    {
        if (!IsEnabled())
            return false;
        return (Filter()->Get(vobj, entry, use_dynamic, reason));
    }
    
    // just a shortcut for Summary()->Clear; RegexSummary()->Clear()
    void
    ClearSummaries()
    {
        m_summary_nav->Clear();
        m_regex_summary_nav->Clear();
    }
    
    // just a shortcut for (Summary()->Delete(name) || RegexSummary()->Delete(name))
    bool
    DeleteSummaries(const char* name)
    {
        bool del_sum = m_summary_nav->Delete(name);
        bool del_rex = m_regex_summary_nav->Delete(name);
        
        return (del_sum || del_rex);
    }
    
    uint32_t
    GetCount()
    {
        return Summary()->GetCount() + RegexSummary()->GetCount();
    }
    
    std::string
    GetName()
    {
        return m_name;
    }
    
    typedef lldb::SharedPtr<FormatCategory>::Type SharedPointer;
};

class CategoryMap
{
private:
    typedef const char* KeyType;
    typedef FormatCategory ValueType;
    typedef ValueType::SharedPointer ValueSP;
    typedef std::list<FormatCategory::SharedPointer> ActiveCategoriesList;
    typedef ActiveCategoriesList::iterator ActiveCategoriesIterator;
    
    Mutex m_map_mutex;
    IFormatChangeListener* listener;
    
    
    friend class FormatNavigator<KeyType, ValueType>;
    friend class FormatManager;
    
public:
    typedef std::map<KeyType, ValueSP> MapType;
    
private:    
    MapType m_map;
    ActiveCategoriesList m_active_categories;
    
    MapType& map()
    {
        return m_map;
    }
    
    ActiveCategoriesList& active_list()
    {
        return m_active_categories;
    }
    
    Mutex& mutex()
    {
        return m_map_mutex;
    }
    
public:
    
    typedef MapType::iterator MapIterator;
    typedef bool(*CallbackType)(void*, KeyType, const ValueSP&);
    
    CategoryMap(IFormatChangeListener* lst = NULL) :
    m_map_mutex(Mutex::eMutexTypeRecursive),
    listener(lst),
    m_map(),
    m_active_categories()
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
        DisableCategory(name);
        if (listener)
            listener->Changed();
        return true;
    }
    
    void
    EnableCategory(KeyType category_name)
    {
        Mutex::Locker(m_map_mutex);
        ValueSP category;
        if (!Get(category_name,category))
            return;
        category->Enable();
        m_active_categories.push_front(category);
    }
    
    class delete_matching_categories
    {
        FormatCategory::SharedPointer ptr;
    public:
        delete_matching_categories(FormatCategory::SharedPointer p) : ptr(p)
        {}
        
        bool operator()(const FormatCategory::SharedPointer& other)
        {
            return ptr.get() == other.get();
        }
    };
    
    void
    DisableCategory(KeyType category_name)
    {
        Mutex::Locker(m_map_mutex);
        ValueSP category;
        if (!Get(category_name,category))
            return;
        category->Disable();
        m_active_categories.remove_if(delete_matching_categories(category));
    }
    
    void
    Clear()
    {
        Mutex::Locker(m_map_mutex);
        m_map.clear();
        m_active_categories.clear();
        if (listener)
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
            
            // loop through enabled categories in respective order
            {
                ActiveCategoriesIterator begin, end = m_active_categories.end();
                for (begin = m_active_categories.begin(); begin != end; begin++)
                {
                    FormatCategory::SharedPointer category = *begin;
                    const char* type = category->GetName().c_str();
                    if (!callback(param, type, category))
                        break;
                }
            }
            
            // loop through disabled categories in just any order
            {
                MapIterator pos, end = m_map.end();
                for (pos = m_map.begin(); pos != end; pos++)
                {
                    if (pos->second->IsEnabled())
                        continue;
                    KeyType type = pos->first;
                    if (!callback(param, type, pos->second))
                        break;
                }
            }
        }
    }
    
    uint32_t
    GetCount()
    {
        return m_map.size();
    }
    
    bool
    Get(ValueObject& vobj,
        lldb::SummaryFormatSP& entry,
        lldb::DynamicValueType use_dynamic)
    {
        Mutex::Locker(m_map_mutex);
        
        uint32_t reason_why;        
        ActiveCategoriesIterator begin, end = m_active_categories.end();
        
        for (begin = m_active_categories.begin(); begin != end; begin++)
        {
            FormatCategory::SharedPointer category = *begin;
            lldb::SummaryFormatSP current_format;
            if (!category->Get(vobj, current_format, use_dynamic, &reason_why))
                continue;
            /*if (reason_why == lldb::eFormatterChoiceCriterionDirectChoice)
             {
             entry = current_format;
             return true;
             }
             else if (first)
             {
             entry = current_format;
             first = false;
             }*/
            entry = current_format;
            return true;
        }
        return false;
    }
    
    bool
    Get(ValueObject& vobj,
        lldb::SyntheticChildrenSP& entry,
        lldb::DynamicValueType use_dynamic)
    {
        Mutex::Locker(m_map_mutex);
        
        uint32_t reason_why;
        
        ActiveCategoriesIterator begin, end = m_active_categories.end();
        
        for (begin = m_active_categories.begin(); begin != end; begin++)
        {
            FormatCategory::SharedPointer category = *begin;
            lldb::SyntheticChildrenSP current_format;
            if (!category->Get(vobj, current_format, use_dynamic, &reason_why))
                continue;
            /*if (reason_why == lldb::eFormatterChoiceCriterionDirectChoice)
            {
                entry = current_format;
                return true;
            }
            else if (first)
            {
                entry = current_format;
                first = false;
            }*/
            entry = current_format;
            return true;
        }
        return false;
    }

};


class FormatManager : public IFormatChangeListener
{
private:
    
    typedef FormatNavigator<const char*, ValueFormat> ValueNavigator;

    typedef ValueNavigator::MapType ValueMap;
    typedef FormatMap<const char*, SummaryFormat> NamedSummariesMap;
        
    ValueNavigator m_value_nav;
    NamedSummariesMap m_named_summaries_map;
    uint32_t m_last_revision;
    CategoryMap m_categories_map;
    
    const char* m_default_category_name;
    const char* m_system_category_name;
        
    typedef CategoryMap::MapType::iterator CategoryMapIterator;
        
public:
    
    typedef bool (*CategoryCallback)(void*, const char*, const FormatCategory::SharedPointer&);
    
    FormatManager() : 
    m_value_nav("format",this),
    m_named_summaries_map(this),
    m_last_revision(0),
    m_categories_map(this)
    {
        
        // build default categories
        
        m_default_category_name = ConstString("default").GetCString();
        m_system_category_name = ConstString("system").GetCString();

        // add some default stuff
        // most formats, summaries, ... actually belong to the users' lldbinit file rather than here
        SummaryFormat::SharedPointer string_format(new StringSummaryFormat(false,
                                                                           true,
                                                                           false,
                                                                           true,
                                                                           false,
                                                                           false,
                                                                           "${var%s}"));
        
        
        SummaryFormat::SharedPointer string_array_format(new StringSummaryFormat(false,
                                                                                 true,
                                                                                 false,
                                                                                 false,
                                                                                 false,
                                                                                 false,
                                                                                 "${var%s}"));
        
        lldb::RegularExpressionSP any_size_char_arr(new RegularExpression("char \\[[0-9]+\\]"));
        
        
        Category(m_system_category_name)->Summary()->Add(ConstString("char *").GetCString(), string_format);
        Category(m_system_category_name)->Summary()->Add(ConstString("const char *").GetCString(), string_format);
        Category(m_system_category_name)->RegexSummary()->Add(any_size_char_arr, string_array_format);
        
        Category(m_default_category_name); // this call is there to force LLDB into creating an empty "default" category
        
        // the order of these two calls IS important, if you invert it "system" summaries will prevail over the user's
        EnableCategory(m_system_category_name);
        EnableCategory(m_default_category_name);
        
    }

    
    CategoryMap& Categories() { return m_categories_map; }
    ValueNavigator& Value() { return m_value_nav; }
    NamedSummariesMap& NamedSummary() { return m_named_summaries_map; }

    void
    EnableCategory(const char* category_name)
    {
        m_categories_map.EnableCategory(category_name);
    }
    
    void
    DisableCategory(const char* category_name)
    {
        m_categories_map.DisableCategory(category_name);
    }
    
    void
    LoopThroughCategories(CategoryCallback callback, void* param)
    {
        m_categories_map.LoopThrough(callback, param);
    }
    
    FormatCategory::SummaryNavigatorSP
    Summary(const char* category_name = NULL)
    {
        return Category(category_name)->Summary();
    }
    
    FormatCategory::RegexSummaryNavigatorSP
    RegexSummary(const char* category_name = NULL)
    {
        return Category(category_name)->RegexSummary();
    }
    
    lldb::FormatCategorySP
    Category(const char* category_name = NULL)
    {
        if (!category_name)
            return Category(m_default_category_name);
        lldb::FormatCategorySP category;
        if (m_categories_map.Get(category_name, category))
            return category;
        Categories().Add(category_name,lldb::FormatCategorySP(new FormatCategory(this, category_name)));
        return Category(category_name);
    }
    
    bool
    Get(ValueObject& vobj,
        lldb::SummaryFormatSP& entry,
        lldb::DynamicValueType use_dynamic)
    {
        return m_categories_map.Get(vobj, entry, use_dynamic);
    }
    bool
    Get(ValueObject& vobj,
        lldb::SyntheticChildrenSP& entry,
        lldb::DynamicValueType use_dynamic)
    {
        return m_categories_map.Get(vobj, entry, use_dynamic);
    }

    static bool
    GetFormatFromCString (const char *format_cstr,
                          bool partial_match_ok,
                          lldb::Format &format);

    static char
    GetFormatAsFormatChar (lldb::Format format);

    static const char *
    GetFormatAsCString (lldb::Format format);
    
    // when DataExtractor dumps a vectorOfT, it uses a predefined format for each item
    // this method returns it, or eFormatInvalid if vector_format is not a vectorOf
    static lldb::Format
    GetSingleItemFormat(lldb::Format vector_format);
    
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
