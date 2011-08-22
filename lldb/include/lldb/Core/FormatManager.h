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
// C++ Includes

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"
#include "lldb/lldb-enumerations.h"

#include "lldb/Core/FormatNavigator.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"

using lldb::LogSP;

namespace lldb_private {
    
// this file (and its. cpp) contain the low-level implementation of LLDB Data Visualization
// class DataVisualization is the high-level front-end of this feature
// clients should refer to that class as the entry-point into the data formatters
// unless they have a good reason to bypass it and prefer to use this file's objects directly
    
class CategoryMap;
    
class FormatCategory
{
private:
    
    typedef FormatNavigator<ConstString, SummaryFormat> SummaryNavigator;
    typedef FormatNavigator<lldb::RegularExpressionSP, SummaryFormat> RegexSummaryNavigator;
    
    typedef FormatNavigator<ConstString, SyntheticFilter> FilterNavigator;
    typedef FormatNavigator<lldb::RegularExpressionSP, SyntheticFilter> RegexFilterNavigator;
    
    typedef FormatNavigator<ConstString, SyntheticScriptProvider> SynthNavigator;
    typedef FormatNavigator<lldb::RegularExpressionSP, SyntheticScriptProvider> RegexSynthNavigator;

    typedef SummaryNavigator::MapType SummaryMap;
    typedef RegexSummaryNavigator::MapType RegexSummaryMap;
    typedef FilterNavigator::MapType FilterMap;
    typedef RegexFilterNavigator::MapType RegexFilterMap;
    typedef SynthNavigator::MapType SynthMap;
    typedef RegexSynthNavigator::MapType RegexSynthMap;

public:
        
    typedef uint16_t FormatCategoryItems;
    static const uint16_t ALL_ITEM_TYPES = 0xFFFF;
    
    typedef SummaryNavigator::SharedPointer SummaryNavigatorSP;
    typedef RegexSummaryNavigator::SharedPointer RegexSummaryNavigatorSP;
    typedef FilterNavigator::SharedPointer FilterNavigatorSP;
    typedef RegexFilterNavigator::SharedPointer RegexFilterNavigatorSP;
    typedef SynthNavigator::SharedPointer SynthNavigatorSP;
    typedef RegexSynthNavigator::SharedPointer RegexSynthNavigatorSP;

    FormatCategory (IFormatChangeListener* clist,
                    std::string name);
    
    SummaryNavigatorSP
    GetSummaryNavigator ()
    {
        return SummaryNavigatorSP(m_summary_nav);
    }
    
    RegexSummaryNavigatorSP
    GetRegexSummaryNavigator ()
    {
        return RegexSummaryNavigatorSP(m_regex_summary_nav);
    }
    
    FilterNavigatorSP
    GetFilterNavigator ()
    {
        return FilterNavigatorSP(m_filter_nav);
    }
    
    RegexFilterNavigatorSP
    GetRegexFilterNavigator ()
    {
        return RegexFilterNavigatorSP(m_regex_filter_nav);
    }
    
    SynthNavigatorSP
    GetSyntheticNavigator ()
    {
        return SynthNavigatorSP(m_synth_nav);
    }
    
    RegexSynthNavigatorSP
    GetRegexSyntheticNavigator ()
    {
        return RegexSynthNavigatorSP(m_regex_synth_nav);
    }
    
    bool
    IsEnabled () const
    {
        return m_enabled;
    }
        
    bool
    Get (ValueObject& valobj,
         lldb::SummaryFormatSP& entry,
         lldb::DynamicValueType use_dynamic,
         uint32_t* reason = NULL)
    {
        if (!IsEnabled())
            return false;
        if (GetSummaryNavigator()->Get(valobj, entry, use_dynamic, reason))
            return true;
        bool regex = GetRegexSummaryNavigator()->Get(valobj, entry, use_dynamic, reason);
        if (regex && reason)
            *reason |= lldb::eFormatterChoiceCriterionRegularExpressionSummary;
        return regex;
    }
    
    bool
    Get (ValueObject& valobj,
         lldb::SyntheticChildrenSP& entry,
         lldb::DynamicValueType use_dynamic,
         uint32_t* reason = NULL);
    
    // just a shortcut for GetSummaryNavigator()->Clear; GetRegexSummaryNavigator()->Clear()
    void
    ClearSummaries ()
    {
        Clear(eFormatCategoryItemSummary | eFormatCategoryItemRegexSummary);
    }
    
    // just a shortcut for (GetSummaryNavigator()->Delete(name) || GetRegexSummaryNavigator()->Delete(name))
    bool
    DeleteSummaries (ConstString name)
    {
        return Delete(name, (eFormatCategoryItemSummary | eFormatCategoryItemRegexSummary));
    }
    
    
    void
    Clear (FormatCategoryItems items = ALL_ITEM_TYPES);
    
    bool
    Delete (ConstString name,
            FormatCategoryItems items = ALL_ITEM_TYPES);
    
    uint32_t
    GetCount (FormatCategoryItems items = ALL_ITEM_TYPES);
    
    std::string
    GetName ()
    {
        return m_name;
    }
    
    bool
    AnyMatches (ConstString type_name,
                FormatCategoryItems items = ALL_ITEM_TYPES,
                bool only_enabled = true,
                const char** matching_category = NULL,
                FormatCategoryItems* matching_type = NULL);
    
    typedef lldb::SharedPtr<FormatCategory>::Type SharedPointer;
    
private:
    SummaryNavigator::SharedPointer m_summary_nav;
    RegexSummaryNavigator::SharedPointer m_regex_summary_nav;
    FilterNavigator::SharedPointer m_filter_nav;
    RegexFilterNavigator::SharedPointer m_regex_filter_nav;
    SynthNavigator::SharedPointer m_synth_nav;
    RegexSynthNavigator::SharedPointer m_regex_synth_nav;
    
    bool m_enabled;
    
    IFormatChangeListener* m_change_listener;
    
    Mutex m_mutex;
    
    std::string m_name;
    
    void
    Enable (bool value = true)
    {
        Mutex::Locker(m_mutex);
        m_enabled = value;        
        if (m_change_listener)
            m_change_listener->Changed();
    }
    
    void
    Disable ()
    {
        Enable(false);
    }
    
    friend class CategoryMap;
    
    friend class FormatNavigator<ConstString, SummaryFormat>;
    friend class FormatNavigator<lldb::RegularExpressionSP, SummaryFormat>;
    
    friend class FormatNavigator<ConstString, SyntheticFilter>;
    friend class FormatNavigator<lldb::RegularExpressionSP, SyntheticFilter>;
    
    friend class FormatNavigator<ConstString, SyntheticScriptProvider>;
    friend class FormatNavigator<lldb::RegularExpressionSP, SyntheticScriptProvider>;
    

};

class CategoryMap
{
private:
    typedef ConstString KeyType;
    typedef FormatCategory ValueType;
    typedef ValueType::SharedPointer ValueSP;
    typedef std::list<lldb::FormatCategorySP> ActiveCategoriesList;
    typedef ActiveCategoriesList::iterator ActiveCategoriesIterator;
        
public:
    typedef std::map<KeyType, ValueSP> MapType;
    typedef MapType::iterator MapIterator;
    typedef bool(*CallbackType)(void*, const ValueSP&);
    
    CategoryMap (IFormatChangeListener* lst) :
        m_map_mutex(Mutex::eMutexTypeRecursive),
        listener(lst),
        m_map(),
        m_active_categories()
    {
    }
    
    void
    Add (KeyType name,
         const ValueSP& entry)
    {
        Mutex::Locker(m_map_mutex);
        m_map[name] = entry;
        if (listener)
            listener->Changed();
    }
    
    bool
    Delete (KeyType name)
    {
        Mutex::Locker(m_map_mutex);
        MapIterator iter = m_map.find(name);
        if (iter == m_map.end())
            return false;
        m_map.erase(name);
        Disable(name);
        if (listener)
            listener->Changed();
        return true;
    }
    
    void
    Enable (KeyType category_name)
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
        lldb::FormatCategorySP ptr;
    public:
        delete_matching_categories(lldb::FormatCategorySP p) : ptr(p)
        {}
        
        bool operator()(const lldb::FormatCategorySP& other)
        {
            return ptr.get() == other.get();
        }
    };
    
    void
    Disable (KeyType category_name)
    {
        Mutex::Locker(m_map_mutex);
        ValueSP category;
        if (!Get(category_name,category))
            return;
        category->Disable();
        m_active_categories.remove_if(delete_matching_categories(category));
    }
    
    void
    Clear ()
    {
        Mutex::Locker(m_map_mutex);
        m_map.clear();
        m_active_categories.clear();
        if (listener)
            listener->Changed();
    }
    
    bool
    Get (KeyType name,
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
    LoopThrough (CallbackType callback, void* param);
    
    bool
    AnyMatches (ConstString type_name,
                FormatCategory::FormatCategoryItems items = FormatCategory::ALL_ITEM_TYPES,
                bool only_enabled = true,
                const char** matching_category = NULL,
                FormatCategory::FormatCategoryItems* matching_type = NULL)
    {
        Mutex::Locker(m_map_mutex);
        
        MapIterator pos, end = m_map.end();
        for (pos = m_map.begin(); pos != end; pos++)
        {
            if (pos->second->AnyMatches(type_name,
                                        items,
                                        only_enabled,
                                        matching_category,
                                        matching_type))
                return true;
        }
        return false;
    }
    
    uint32_t
    GetCount ()
    {
        return m_map.size();
    }
    
    bool
    Get (ValueObject& valobj,
         lldb::SummaryFormatSP& entry,
         lldb::DynamicValueType use_dynamic)
    {
        Mutex::Locker(m_map_mutex);
        
        uint32_t reason_why;        
        ActiveCategoriesIterator begin, end = m_active_categories.end();
        
        for (begin = m_active_categories.begin(); begin != end; begin++)
        {
            lldb::FormatCategorySP category = *begin;
            lldb::SummaryFormatSP current_format;
            if (!category->Get(valobj, current_format, use_dynamic, &reason_why))
                continue;
            entry = current_format;
            return true;
        }
        return false;
    }
    
    bool
    Get (ValueObject& valobj,
         lldb::SyntheticChildrenSP& entry,
         lldb::DynamicValueType use_dynamic)
    {
        Mutex::Locker(m_map_mutex);
        
        uint32_t reason_why;
        
        ActiveCategoriesIterator begin, end = m_active_categories.end();
        
        for (begin = m_active_categories.begin(); begin != end; begin++)
        {
            lldb::FormatCategorySP category = *begin;
            lldb::SyntheticChildrenSP current_format;
            if (!category->Get(valobj, current_format, use_dynamic, &reason_why))
                continue;
            entry = current_format;
            return true;
        }
        return false;
    }
    
private:
    Mutex m_map_mutex;
    IFormatChangeListener* listener;
    
    MapType m_map;
    ActiveCategoriesList m_active_categories;
    
    MapType& map ()
    {
        return m_map;
    }
    
    ActiveCategoriesList& active_list ()
    {
        return m_active_categories;
    }
    
    Mutex& mutex ()
    {
        return m_map_mutex;
    }
    
    friend class FormatNavigator<KeyType, ValueType>;
    friend class FormatManager;
};

class FormatManager : public IFormatChangeListener
{
private:
    
    typedef FormatNavigator<ConstString, ValueFormat> ValueNavigator;

    typedef ValueNavigator::MapType ValueMap;
    typedef FormatMap<ConstString, SummaryFormat> NamedSummariesMap;
    typedef CategoryMap::MapType::iterator CategoryMapIterator;

public:
    
    typedef CategoryMap::CallbackType CategoryCallback;
    
    FormatManager ();
    
    ValueNavigator&
    GetValueNavigator ()
    {
        return m_value_nav;
    }
    
    NamedSummariesMap&
    GetNamedSummaryNavigator ()
    {
        return m_named_summaries_map;
    }
    
    void
    EnableCategory (const ConstString& category_name)
    {
        m_categories_map.Enable(category_name);
    }
    
    void
    DisableCategory (const ConstString& category_name)
    {
        m_categories_map.Disable(category_name);
    }
    
    bool
    DeleteCategory (const ConstString& category_name)
    {
        return m_categories_map.Delete(category_name);
    }
    
    void
    ClearCategories ()
    {
        return m_categories_map.Clear();
    }
    
    uint32_t
    GetCategoriesCount ()
    {
        return m_categories_map.GetCount();
    }
    
    void
    LoopThroughCategories (CategoryCallback callback, void* param)
    {
        m_categories_map.LoopThrough(callback, param);
    }
    
    lldb::FormatCategorySP
    Category (const char* category_name = NULL,
              bool can_create = true)
    {
        if (!category_name)
            return Category(m_default_category_name);
        return Category(ConstString(category_name));
    }
    
    lldb::FormatCategorySP
    Category (const ConstString& category_name,
              bool can_create = true)
    {
        if (!category_name)
            return Category(m_default_category_name);
        lldb::FormatCategorySP category;
        if (m_categories_map.Get(category_name, category))
            return category;
        
        if (!can_create)
            return lldb::FormatCategorySP();
        
        m_categories_map.Add(category_name,lldb::FormatCategorySP(new FormatCategory(this, category_name.AsCString())));
        return Category(category_name);
    }
    
    bool
    Get (ValueObject& valobj,
         lldb::SummaryFormatSP& entry,
         lldb::DynamicValueType use_dynamic)
    {
        return m_categories_map.Get(valobj, entry, use_dynamic);
    }
    bool
    Get (ValueObject& valobj,
         lldb::SyntheticChildrenSP& entry,
         lldb::DynamicValueType use_dynamic)
    {
        return m_categories_map.Get(valobj, entry, use_dynamic);
    }
    
    bool
    AnyMatches (ConstString type_name,
                FormatCategory::FormatCategoryItems items = FormatCategory::ALL_ITEM_TYPES,
                bool only_enabled = true,
                const char** matching_category = NULL,
                FormatCategory::FormatCategoryItems* matching_type = NULL)
    {
        return m_categories_map.AnyMatches(type_name,
                                           items,
                                           only_enabled,
                                           matching_category,
                                           matching_type);
    }

    static bool
    GetFormatFromCString (const char *format_cstr,
                          bool partial_match_ok,
                          lldb::Format &format);

    static char
    GetFormatAsFormatChar (lldb::Format format);

    static const char *
    GetFormatAsCString (lldb::Format format);
    
    // if the user tries to add formatters for, say, "struct Foo"
    // those will not match any type because of the way we strip qualifiers from typenames
    // this method looks for the case where the user is adding a "class","struct","enum" or "union" Foo
    // and strips the unnecessary qualifier
    static ConstString
    GetValidTypeName (const ConstString& type);
    
    // when DataExtractor dumps a vectorOfT, it uses a predefined format for each item
    // this method returns it, or eFormatInvalid if vector_format is not a vectorOf
    static lldb::Format
    GetSingleItemFormat (lldb::Format vector_format);
    
    void
    Changed ()
    {
        __sync_add_and_fetch(&m_last_revision, +1);
    }
    
    uint32_t
    GetCurrentRevision ()
    {
        return m_last_revision;
    }
    
    ~FormatManager ()
    {
    }
    
private:    
    ValueNavigator m_value_nav;
    NamedSummariesMap m_named_summaries_map;
    uint32_t m_last_revision;
    CategoryMap m_categories_map;
    
    ConstString m_default_category_name;
    ConstString m_system_category_name;
    ConstString m_gnu_cpp_category_name;
    
    CategoryMap&
    GetCategories ()
    {
        return m_categories_map;
    }
    
};
    
} // namespace lldb_private

#endif	// lldb_FormatManager_h_
