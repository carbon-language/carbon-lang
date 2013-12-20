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
    
    template <typename FormatterImpl>
    class FormatterContainerPair
    {
    public:
        typedef FormattersContainer<ConstString, FormatterImpl> ExactMatchContainer;
        typedef FormattersContainer<lldb::RegularExpressionSP, FormatterImpl> RegexMatchContainer;
        
        typedef typename ExactMatchContainer::MapType ExactMatchMap;
        typedef typename RegexMatchContainer::MapType RegexMatchMap;

        typedef typename ExactMatchContainer::MapValueType MapValueType;
        
        typedef typename ExactMatchContainer::SharedPointer ExactMatchContainerSP;
        typedef typename RegexMatchContainer::SharedPointer RegexMatchContainerSP;
        
        FormatterContainerPair (const char* exact_name,
                                const char* regex_name,
                                IFormatChangeListener* clist) :
            m_exact_sp(new ExactMatchContainer(std::string(exact_name),clist)),
            m_regex_sp(new RegexMatchContainer(std::string(regex_name),clist))
        {
        }
        
        ~FormatterContainerPair () = default;
        
        ExactMatchContainerSP
        GetExactMatch () const
        {
            return m_exact_sp;
        }
        
        RegexMatchContainerSP
        GetRegexMatch () const
        {
            return m_regex_sp;
        }
        
    private:
        ExactMatchContainerSP m_exact_sp;
        RegexMatchContainerSP m_regex_sp;
    };

    class TypeCategoryImpl
    {
    private:
        typedef FormatterContainerPair<TypeFormatImpl> FormatContainer;
        typedef FormatterContainerPair<TypeSummaryImpl> SummaryContainer;
        typedef FormatterContainerPair<TypeFilterImpl> FilterContainer;
        
#ifndef LLDB_DISABLE_PYTHON
        typedef FormatterContainerPair<ScriptedSyntheticChildren> SynthContainer;
#endif // #ifndef LLDB_DISABLE_PYTHON

    public:
        
        typedef uint16_t FormatCategoryItems;
        static const uint16_t ALL_ITEM_TYPES = UINT16_MAX;

        typedef FormatContainer::ExactMatchContainerSP FormatContainerSP;
        typedef FormatContainer::RegexMatchContainerSP RegexFormatContainerSP;
        
        typedef SummaryContainer::ExactMatchContainerSP SummaryContainerSP;
        typedef SummaryContainer::RegexMatchContainerSP RegexSummaryContainerSP;

        typedef FilterContainer::ExactMatchContainerSP FilterContainerSP;
        typedef FilterContainer::RegexMatchContainerSP RegexFilterContainerSP;
#ifndef LLDB_DISABLE_PYTHON
        typedef SynthContainer::ExactMatchContainerSP SynthContainerSP;
        typedef SynthContainer::RegexMatchContainerSP RegexSynthContainerSP;
#endif // #ifndef LLDB_DISABLE_PYTHON
        
        TypeCategoryImpl (IFormatChangeListener* clist,
                          ConstString name);
        
        FormatContainerSP
        GetTypeFormatsContainer ()
        {
            return m_format_cont.GetExactMatch();
        }
        
        RegexFormatContainerSP
        GetRegexTypeFormatsContainer ()
        {
            return m_format_cont.GetRegexMatch();
        }
        
        SummaryContainerSP
        GetTypeSummariesContainer ()
        {
            return m_summary_cont.GetExactMatch();
        }
        
        RegexSummaryContainerSP
        GetRegexTypeSummariesContainer ()
        {
            return m_summary_cont.GetRegexMatch();
        }
        
        FilterContainerSP
        GetTypeFiltersContainer ()
        {
            return m_filter_cont.GetExactMatch();
        }
        
        RegexFilterContainerSP
        GetRegexTypeFiltersContainer ()
        {
            return m_filter_cont.GetRegexMatch();
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
            return m_synth_cont.GetExactMatch();
        }
        
        RegexSynthContainerSP
        GetRegexTypeSyntheticsContainer ()
        {
            return m_synth_cont.GetRegexMatch();
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
        FormatContainer m_format_cont;
        
        SummaryContainer m_summary_cont;

        FilterContainer m_filter_cont;

#ifndef LLDB_DISABLE_PYTHON
        SynthContainer m_synth_cont;
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
