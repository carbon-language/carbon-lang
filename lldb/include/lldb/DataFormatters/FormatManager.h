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

#include "lldb/DataFormatters/FormatCache.h"
#include "lldb/DataFormatters/FormatClasses.h"
#include "lldb/DataFormatters/FormattersContainer.h"
#include "lldb/DataFormatters/TypeCategory.h"
#include "lldb/DataFormatters/TypeCategoryMap.h"

#include <atomic>
#include <functional>

namespace lldb_private {
    
// this file (and its. cpp) contain the low-level implementation of LLDB Data Visualization
// class DataVisualization is the high-level front-end of this feature
// clients should refer to that class as the entry-point into the data formatters
// unless they have a good reason to bypass it and prefer to use this file's objects directly

class FormatManager : public IFormatChangeListener
{
    typedef FormatMap<ConstString, TypeSummaryImpl> NamedSummariesMap;
    typedef TypeCategoryMap::MapType::iterator CategoryMapIterator;
public:
    
    template <typename FormatterType>
    using HardcodedFormatterFinder = std::function<typename FormatterType::SharedPointer (lldb_private::ValueObject&,
                                                                                          lldb::DynamicValueType,
                                                                                          FormatManager&)>;
    
    template <typename FormatterType>
    using HardcodedFormatterFinders = std::vector<HardcodedFormatterFinder<FormatterType>>;
    
    typedef TypeCategoryMap::CallbackType CategoryCallback;
    
    FormatManager ();
    
    NamedSummariesMap&
    GetNamedSummaryContainer ()
    {
        return m_named_summaries_map;
    }
    
    void
    EnableCategory (const ConstString& category_name,
                    TypeCategoryMap::Position pos = TypeCategoryMap::Default)
    {
        m_categories_map.Enable(category_name,
                                pos);
    }
    
    void
    DisableCategory (const ConstString& category_name)
    {
        m_categories_map.Disable(category_name);
    }
    
    void
    EnableCategory (const lldb::TypeCategoryImplSP& category,
                    TypeCategoryMap::Position pos = TypeCategoryMap::Default)
    {
        m_categories_map.Enable(category,
                                pos);
    }
    
    void
    DisableCategory (const lldb::TypeCategoryImplSP& category)
    {
        m_categories_map.Disable(category);
    }
    
    void
    EnableAllCategories ()
    {
        m_categories_map.EnableAllCategories ();
    }
    
    void
    DisableAllCategories ()
    {
        m_categories_map.DisableAllCategories ();
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
    
    lldb::TypeCategoryImplSP
    GetCategoryAtIndex (size_t index)
    {
        return m_categories_map.GetAtIndex(index);
    }
    
    void
    LoopThroughCategories (CategoryCallback callback, void* param)
    {
        m_categories_map.LoopThrough(callback, param);
    }
    
    lldb::TypeCategoryImplSP
    GetCategory (const char* category_name = NULL,
                 bool can_create = true)
    {
        if (!category_name)
            return GetCategory(m_default_category_name);
        return GetCategory(ConstString(category_name));
    }
    
    lldb::TypeCategoryImplSP
    GetCategory (const ConstString& category_name,
                 bool can_create = true);

    lldb::TypeFormatImplSP
    GetFormatForType (lldb::TypeNameSpecifierImplSP type_sp);
    
    lldb::TypeSummaryImplSP
    GetSummaryForType (lldb::TypeNameSpecifierImplSP type_sp);

    lldb::TypeFilterImplSP
    GetFilterForType (lldb::TypeNameSpecifierImplSP type_sp);

#ifndef LLDB_DISABLE_PYTHON
    lldb::ScriptedSyntheticChildrenSP
    GetSyntheticForType (lldb::TypeNameSpecifierImplSP type_sp);
#endif
    
#ifndef LLDB_DISABLE_PYTHON
    lldb::SyntheticChildrenSP
    GetSyntheticChildrenForType (lldb::TypeNameSpecifierImplSP type_sp);
#endif
    
    lldb::TypeValidatorImplSP
    GetValidatorForType (lldb::TypeNameSpecifierImplSP type_sp);
    
    lldb::TypeFormatImplSP
    GetFormat (ValueObject& valobj,
               lldb::DynamicValueType use_dynamic);
    
    lldb::TypeSummaryImplSP
    GetSummaryFormat (ValueObject& valobj,
                      lldb::DynamicValueType use_dynamic);

#ifndef LLDB_DISABLE_PYTHON
    lldb::SyntheticChildrenSP
    GetSyntheticChildren (ValueObject& valobj,
                          lldb::DynamicValueType use_dynamic);
#endif
    
    lldb::TypeValidatorImplSP
    GetValidator (ValueObject& valobj,
                  lldb::DynamicValueType use_dynamic);
    
    bool
    AnyMatches (ConstString type_name,
                TypeCategoryImpl::FormatCategoryItems items = TypeCategoryImpl::ALL_ITEM_TYPES,
                bool only_enabled = true,
                const char** matching_category = NULL,
                TypeCategoryImpl::FormatCategoryItems* matching_type = NULL)
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
    
    // this returns true if the ValueObjectPrinter is *highly encouraged*
    // to actually represent this ValueObject in one-liner format
    // If this object has a summary formatter, however, we should not
    // try and do one-lining, just let the summary do the right thing
    bool
    ShouldPrintAsOneLiner (ValueObject& valobj);
    
    void
    Changed ()
    {
        ++m_last_revision;
        m_format_cache.Clear ();
    }
    
    uint32_t
    GetCurrentRevision ()
    {
        return m_last_revision;
    }
    
    ~FormatManager ()
    {
    }
    
    static FormattersMatchVector
    GetPossibleMatches (ValueObject& valobj,
                        lldb::DynamicValueType use_dynamic)
    {
        FormattersMatchVector matches;
        GetPossibleMatches (valobj,
                            valobj.GetClangType(),
                            lldb_private::eFormatterChoiceCriterionDirectChoice,
                            use_dynamic,
                            matches,
                            false,
                            false,
                            false,
                            true);
        return matches;
    }

private:
    
    static void
    GetPossibleMatches (ValueObject& valobj,
                        ClangASTType clang_type,
                        uint32_t reason,
                        lldb::DynamicValueType use_dynamic,
                        FormattersMatchVector& entries,
                        bool did_strip_ptr,
                        bool did_strip_ref,
                        bool did_strip_typedef,
                        bool root_level = false);
    
    FormatCache m_format_cache;
    NamedSummariesMap m_named_summaries_map;
    std::atomic<uint32_t> m_last_revision;
    TypeCategoryMap m_categories_map;
    
    ConstString m_default_category_name;
    ConstString m_system_category_name;
    ConstString m_gnu_cpp_category_name;
    ConstString m_libcxx_category_name;
    ConstString m_objc_category_name;
    ConstString m_corefoundation_category_name;
    ConstString m_coregraphics_category_name;
    ConstString m_coreservices_category_name;
    ConstString m_vectortypes_category_name;
    ConstString m_appkit_category_name;
    ConstString m_coremedia_category_name;
    
    HardcodedFormatterFinders<TypeFormatImpl> m_hardcoded_formats;
    HardcodedFormatterFinders<TypeSummaryImpl> m_hardcoded_summaries;
    HardcodedFormatterFinders<SyntheticChildren> m_hardcoded_synthetics;
    HardcodedFormatterFinders<TypeValidatorImpl> m_hardcoded_validators;
    
    lldb::TypeFormatImplSP
    GetHardcodedFormat (ValueObject&,lldb::DynamicValueType);
    
    lldb::TypeSummaryImplSP
    GetHardcodedSummaryFormat (ValueObject&,lldb::DynamicValueType);

    lldb::SyntheticChildrenSP
    GetHardcodedSyntheticChildren (ValueObject&,lldb::DynamicValueType);
    
    lldb::TypeValidatorImplSP
    GetHardcodedValidator (ValueObject&,lldb::DynamicValueType);
    
    TypeCategoryMap&
    GetCategories ()
    {
        return m_categories_map;
    }
    
    // WARNING: these are temporary functions that setup formatters
    // while a few of these actually should be globally available and setup by LLDB itself
    // most would actually belong to the users' lldbinit file or to some other form of configurable
    // storage
    void
    LoadLibStdcppFormatters ();
    
    void
    LoadLibcxxFormatters ();
    
    void
    LoadSystemFormatters ();
    
    void
    LoadObjCFormatters ();

    void
    LoadCoreMediaFormatters ();
    
    void
    LoadHardcodedFormatters ();
};
    
} // namespace lldb_private
    
#endif	// lldb_FormatManager_h_
