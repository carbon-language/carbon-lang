//===-- TypeCategory.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_TypeCategory_h_
#define lldb_TypeCategory_h_

// C Includes
// C++ Includes

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"
#include "lldb/lldb-enumerations.h"

#include "lldb/DataFormatters/FormatClasses.h"
#include "lldb/DataFormatters/FormattersContainer.h"

namespace lldb_private {    
    class TypeCategoryImpl
    {
    private:
        typedef FormattersContainer<ConstString, TypeFormatImpl> FormatContainer;
        typedef FormattersContainer<lldb::RegularExpressionSP, TypeFormatImpl> RegexFormatContainer;
        
        typedef FormattersContainer<ConstString, TypeSummaryImpl> SummaryContainer;
        typedef FormattersContainer<lldb::RegularExpressionSP, TypeSummaryImpl> RegexSummaryContainer;
        
        typedef FormattersContainer<ConstString, TypeFilterImpl> FilterContainer;
        typedef FormattersContainer<lldb::RegularExpressionSP, TypeFilterImpl> RegexFilterContainer;
        
#ifndef LLDB_DISABLE_PYTHON
        typedef FormattersContainer<ConstString, ScriptedSyntheticChildren> SynthContainer;
        typedef FormattersContainer<lldb::RegularExpressionSP, ScriptedSyntheticChildren> RegexSynthContainer;
#endif // #ifndef LLDB_DISABLE_PYTHON

        typedef FormatContainer::MapType FormatMap;
        typedef RegexFormatContainer::MapType RegexFormatMap;

        typedef SummaryContainer::MapType SummaryMap;
        typedef RegexSummaryContainer::MapType RegexSummaryMap;
        
        typedef FilterContainer::MapType FilterMap;
        typedef RegexFilterContainer::MapType RegexFilterMap;

#ifndef LLDB_DISABLE_PYTHON
        typedef SynthContainer::MapType SynthMap;
        typedef RegexSynthContainer::MapType RegexSynthMap;
#endif // #ifndef LLDB_DISABLE_PYTHON
        
    public:
        
        typedef uint16_t FormatCategoryItems;
        static const uint16_t ALL_ITEM_TYPES = UINT16_MAX;

        typedef FormatContainer::SharedPointer FormatContainerSP;
        typedef RegexFormatContainer::SharedPointer RegexFormatContainerSP;
        
        typedef SummaryContainer::SharedPointer SummaryContainerSP;
        typedef RegexSummaryContainer::SharedPointer RegexSummaryContainerSP;

        typedef FilterContainer::SharedPointer FilterContainerSP;
        typedef RegexFilterContainer::SharedPointer RegexFilterContainerSP;
#ifndef LLDB_DISABLE_PYTHON
        typedef SynthContainer::SharedPointer SynthContainerSP;
        typedef RegexSynthContainer::SharedPointer RegexSynthContainerSP;
#endif // #ifndef LLDB_DISABLE_PYTHON
        
        TypeCategoryImpl (IFormatChangeListener* clist,
                          ConstString name);
        
        FormatContainerSP
        GetTypeFormatsContainer ()
        {
            return FormatContainerSP(m_format_cont);
        }
        
        RegexFormatContainerSP
        GetRegexTypeFormatsContainer ()
        {
            return RegexFormatContainerSP(m_regex_format_cont);
        }
        
        SummaryContainerSP
        GetTypeSummariesContainer ()
        {
            return SummaryContainerSP(m_summary_cont);
        }
        
        RegexSummaryContainerSP
        GetRegexTypeSummariesContainer ()
        {
            return RegexSummaryContainerSP(m_regex_summary_cont);
        }
        
        FilterContainerSP
        GetTypeFiltersContainer ()
        {
            return FilterContainerSP(m_filter_cont);
        }
        
        RegexFilterContainerSP
        GetRegexTypeFiltersContainer ()
        {
            return RegexFilterContainerSP(m_regex_filter_cont);
        }

        FormatContainer::MapValueType
        GetFormatForType (lldb::TypeNameSpecifierImplSP type_sp);
        
        SummaryContainer::MapValueType
        GetSummaryForType (lldb::TypeNameSpecifierImplSP type_sp);
        
        FilterContainer::MapValueType
        GetFilterForType (lldb::TypeNameSpecifierImplSP type_sp);
        
#ifndef LLDB_DISABLE_PYTHON
        SynthContainer::MapValueType
        GetSyntheticForType (lldb::TypeNameSpecifierImplSP type_sp);
#endif
        
        lldb::TypeNameSpecifierImplSP
        GetTypeNameSpecifierForFormatAtIndex (size_t index);
        
        lldb::TypeNameSpecifierImplSP
        GetTypeNameSpecifierForSummaryAtIndex (size_t index);

        FormatContainer::MapValueType
        GetFormatAtIndex (size_t index);
        
        SummaryContainer::MapValueType
        GetSummaryAtIndex (size_t index);
        
        FilterContainer::MapValueType
        GetFilterAtIndex (size_t index);
        
        lldb::TypeNameSpecifierImplSP
        GetTypeNameSpecifierForFilterAtIndex (size_t index);
        
#ifndef LLDB_DISABLE_PYTHON
        SynthContainerSP
        GetTypeSyntheticsContainer ()
        {
            return SynthContainerSP(m_synth_cont);
        }
        
        RegexSynthContainerSP
        GetRegexTypeSyntheticsContainer ()
        {
            return RegexSynthContainerSP(m_regex_synth_cont);
        }
        
        SynthContainer::MapValueType
        GetSyntheticAtIndex (size_t index);
        
        lldb::TypeNameSpecifierImplSP
        GetTypeNameSpecifierForSyntheticAtIndex (size_t index);
        
#endif // #ifndef LLDB_DISABLE_PYTHON
        
        bool
        IsEnabled () const
        {
            return m_enabled;
        }
        
        uint32_t
        GetEnabledPosition()
        {
            if (m_enabled == false)
                return UINT32_MAX;
            else
                return m_enabled_position;
        }
        
        bool
        Get (ValueObject& valobj,
             const FormattersMatchVector& candidates,
             lldb::TypeFormatImplSP& entry,
             uint32_t* reason = NULL);
        
        bool
        Get (ValueObject& valobj,
             const FormattersMatchVector& candidates,
             lldb::TypeSummaryImplSP& entry,
             uint32_t* reason = NULL);
        
        bool
        Get (ValueObject& valobj,
             const FormattersMatchVector& candidates,
             lldb::SyntheticChildrenSP& entry,
             uint32_t* reason = NULL);
        
        void
        Clear (FormatCategoryItems items = ALL_ITEM_TYPES);
        
        bool
        Delete (ConstString name,
                FormatCategoryItems items = ALL_ITEM_TYPES);
        
        uint32_t
        GetCount (FormatCategoryItems items = ALL_ITEM_TYPES);
        
        const char*
        GetName ()
        {
            return m_name.GetCString();
        }
        
        bool
        AnyMatches (ConstString type_name,
                    FormatCategoryItems items = ALL_ITEM_TYPES,
                    bool only_enabled = true,
                    const char** matching_category = NULL,
                    FormatCategoryItems* matching_type = NULL);
        
        typedef std::shared_ptr<TypeCategoryImpl> SharedPointer;
        
    private:
        FormatContainer::SharedPointer m_format_cont;
        RegexFormatContainer::SharedPointer m_regex_format_cont;
        
        SummaryContainer::SharedPointer m_summary_cont;
        RegexSummaryContainer::SharedPointer m_regex_summary_cont;

        FilterContainer::SharedPointer m_filter_cont;
        RegexFilterContainer::SharedPointer m_regex_filter_cont;

#ifndef LLDB_DISABLE_PYTHON
        SynthContainer::SharedPointer m_synth_cont;
        RegexSynthContainer::SharedPointer m_regex_synth_cont;
#endif // #ifndef LLDB_DISABLE_PYTHON
        
        bool m_enabled;
        
        IFormatChangeListener* m_change_listener;
        
        Mutex m_mutex;
        
        ConstString m_name;
        
        uint32_t m_enabled_position;
        
        void
        Enable (bool value, uint32_t position);
        
        void
        Disable ()
        {
            Enable(false, UINT32_MAX);
        }
        
        friend class TypeCategoryMap;
        
        friend class FormattersContainer<ConstString, TypeFormatImpl>;
        friend class FormattersContainer<lldb::RegularExpressionSP, TypeFormatImpl>;
        
        friend class FormattersContainer<ConstString, TypeSummaryImpl>;
        friend class FormattersContainer<lldb::RegularExpressionSP, TypeSummaryImpl>;
        
        friend class FormattersContainer<ConstString, TypeFilterImpl>;
        friend class FormattersContainer<lldb::RegularExpressionSP, TypeFilterImpl>;
        
#ifndef LLDB_DISABLE_PYTHON
        friend class FormattersContainer<ConstString, ScriptedSyntheticChildren>;
        friend class FormattersContainer<lldb::RegularExpressionSP, ScriptedSyntheticChildren>;
#endif // #ifndef LLDB_DISABLE_PYTHON
    };
    
} // namespace lldb_private

#endif	// lldb_TypeCategory_h_
