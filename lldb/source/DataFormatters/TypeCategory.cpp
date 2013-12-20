//===-- TypeCategory.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/DataFormatters/TypeCategory.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb;
using namespace lldb_private;

TypeCategoryImpl::TypeCategoryImpl(IFormatChangeListener* clist,
                                   ConstString name) :
m_format_cont(new FormatContainer("format",clist)),
m_regex_format_cont(new RegexFormatContainer("regex-format",clist)),
m_summary_cont(new SummaryContainer("summary",clist)),
m_regex_summary_cont(new RegexSummaryContainer("regex-summary",clist)),
m_filter_cont(new FilterContainer("filter",clist)),
m_regex_filter_cont(new RegexFilterContainer("regex-filter",clist)),
#ifndef LLDB_DISABLE_PYTHON
m_synth_cont(new SynthContainer("synth",clist)),
m_regex_synth_cont(new RegexSynthContainer("regex-synth",clist)),
#endif
m_enabled(false),
m_change_listener(clist),
m_mutex(Mutex::eMutexTypeRecursive),
m_name(name)
{}

bool
TypeCategoryImpl::Get (ValueObject& valobj,
                       const FormattersMatchVector& candidates,
                       lldb::TypeFormatImplSP& entry,
                       uint32_t* reason)
{
    if (!IsEnabled())
        return false;
    if (GetTypeFormatsContainer()->Get(candidates, entry, reason))
        return true;
    bool regex = GetRegexTypeFormatsContainer()->Get(candidates, entry, reason);
    if (regex && reason)
        *reason |= lldb_private::eFormatterChoiceCriterionRegularExpressionSummary;
    return regex;
}

bool
TypeCategoryImpl::Get (ValueObject& valobj,
                       const FormattersMatchVector& candidates,
                       lldb::TypeSummaryImplSP& entry,
                       uint32_t* reason)
{
    if (!IsEnabled())
        return false;
    if (GetTypeSummariesContainer()->Get(candidates, entry, reason))
        return true;
    bool regex = GetRegexTypeSummariesContainer()->Get(candidates, entry, reason);
    if (regex && reason)
        *reason |= lldb_private::eFormatterChoiceCriterionRegularExpressionSummary;
    return regex;
}

bool
TypeCategoryImpl::Get (ValueObject& valobj,
                       const FormattersMatchVector& candidates,
                       lldb::SyntheticChildrenSP& entry,
                       uint32_t* reason)
{
    if (!IsEnabled())
        return false;
    TypeFilterImpl::SharedPointer filter_sp;
    uint32_t reason_filter = 0;
    bool regex_filter = false;
    // first find both Filter and Synth, and then check which is most recent
    
    if (!GetTypeFiltersContainer()->Get(candidates, filter_sp, &reason_filter))
        regex_filter = GetRegexTypeFiltersContainer()->Get (candidates, filter_sp, &reason_filter);
    
#ifndef LLDB_DISABLE_PYTHON
    bool regex_synth = false;
    uint32_t reason_synth = 0;
    bool pick_synth = false;
    ScriptedSyntheticChildren::SharedPointer synth;
    if (!GetTypeSyntheticsContainer()->Get(candidates, synth, &reason_synth))
        regex_synth = GetRegexTypeSyntheticsContainer()->Get (candidates, synth, &reason_synth);
    if (!filter_sp.get() && !synth.get())
        return false;
    else if (!filter_sp.get() && synth.get())
        pick_synth = true;
    
    else if (filter_sp.get() && !synth.get())
        pick_synth = false;
    
    else /*if (filter_sp.get() && synth.get())*/
    {
        if (filter_sp->GetRevision() > synth->GetRevision())
            pick_synth = false;
        else
            pick_synth = true;
    }
    if (pick_synth)
    {
        if (regex_synth && reason)
            *reason |= lldb_private::eFormatterChoiceCriterionRegularExpressionFilter;
        entry = synth;
        return true;
    }
    else
    {
        if (regex_filter && reason)
            *reason |= lldb_private::eFormatterChoiceCriterionRegularExpressionFilter;
        entry = filter_sp;
        return true;
    }
    
#else
    if (filter_sp)
    {
        entry = filter_sp;
        return true;
    }
#endif
    
    return false;
}

void
TypeCategoryImpl::Clear (FormatCategoryItems items)
{
    if ( (items & eFormatCategoryItemValue)  == eFormatCategoryItemValue )
        m_format_cont->Clear();
    if ( (items & eFormatCategoryItemRegexValue) == eFormatCategoryItemRegexValue )
        m_regex_format_cont->Clear();

    if ( (items & eFormatCategoryItemSummary) == eFormatCategoryItemSummary )
        m_summary_cont->Clear();
    if ( (items & eFormatCategoryItemRegexSummary) == eFormatCategoryItemRegexSummary )
        m_regex_summary_cont->Clear();

    if ( (items & eFormatCategoryItemFilter)  == eFormatCategoryItemFilter )
        m_filter_cont->Clear();
    if ( (items & eFormatCategoryItemRegexFilter) == eFormatCategoryItemRegexFilter )
        m_regex_filter_cont->Clear();

#ifndef LLDB_DISABLE_PYTHON
    if ( (items & eFormatCategoryItemSynth)  == eFormatCategoryItemSynth )
        m_synth_cont->Clear();
    if ( (items & eFormatCategoryItemRegexSynth) == eFormatCategoryItemRegexSynth )
        m_regex_synth_cont->Clear();
#endif
}

bool
TypeCategoryImpl::Delete (ConstString name,
                          FormatCategoryItems items)
{
    bool success = false;
    
    if ( (items & eFormatCategoryItemValue)  == eFormatCategoryItemValue )
        success = m_format_cont->Delete(name) || success;
    if ( (items & eFormatCategoryItemRegexValue) == eFormatCategoryItemRegexValue )
        success = m_regex_format_cont->Delete(name) || success;

    if ( (items & eFormatCategoryItemSummary) == eFormatCategoryItemSummary )
        success = m_summary_cont->Delete(name) || success;
    if ( (items & eFormatCategoryItemRegexSummary) == eFormatCategoryItemRegexSummary )
        success = m_regex_summary_cont->Delete(name) || success;

    if ( (items & eFormatCategoryItemFilter)  == eFormatCategoryItemFilter )
        success = m_filter_cont->Delete(name) || success;
    if ( (items & eFormatCategoryItemRegexFilter) == eFormatCategoryItemRegexFilter )
        success = m_regex_filter_cont->Delete(name) || success;

#ifndef LLDB_DISABLE_PYTHON
    if ( (items & eFormatCategoryItemSynth)  == eFormatCategoryItemSynth )
        success = m_synth_cont->Delete(name) || success;
    if ( (items & eFormatCategoryItemRegexSynth) == eFormatCategoryItemRegexSynth )
        success = m_regex_synth_cont->Delete(name) || success;
#endif
    return success;
}

uint32_t
TypeCategoryImpl::GetCount (FormatCategoryItems items)
{
    uint32_t count = 0;

    if ( (items & eFormatCategoryItemValue) == eFormatCategoryItemValue )
        count += m_format_cont->GetCount();
    if ( (items & eFormatCategoryItemRegexValue) == eFormatCategoryItemRegexValue )
        count += m_regex_format_cont->GetCount();
    
    if ( (items & eFormatCategoryItemSummary) == eFormatCategoryItemSummary )
        count += m_summary_cont->GetCount();
    if ( (items & eFormatCategoryItemRegexSummary) == eFormatCategoryItemRegexSummary )
        count += m_regex_summary_cont->GetCount();

    if ( (items & eFormatCategoryItemFilter)  == eFormatCategoryItemFilter )
        count += m_filter_cont->GetCount();
    if ( (items & eFormatCategoryItemRegexFilter) == eFormatCategoryItemRegexFilter )
        count += m_regex_filter_cont->GetCount();

#ifndef LLDB_DISABLE_PYTHON
    if ( (items & eFormatCategoryItemSynth)  == eFormatCategoryItemSynth )
        count += m_synth_cont->GetCount();
    if ( (items & eFormatCategoryItemRegexSynth) == eFormatCategoryItemRegexSynth )
        count += m_regex_synth_cont->GetCount();
#endif
    return count;
}

bool
TypeCategoryImpl::AnyMatches(ConstString type_name,
                             FormatCategoryItems items,
                             bool only_enabled,
                             const char** matching_category,
                             FormatCategoryItems* matching_type)
{
    if (!IsEnabled() && only_enabled)
        return false;
    
    lldb::TypeFormatImplSP format_sp;
    lldb::TypeSummaryImplSP summary_sp;
    TypeFilterImpl::SharedPointer filter_sp;
#ifndef LLDB_DISABLE_PYTHON
    ScriptedSyntheticChildren::SharedPointer synth_sp;
#endif
    
    if ( (items & eFormatCategoryItemValue) == eFormatCategoryItemValue )
    {
        if (m_format_cont->Get(type_name, format_sp))
        {
            if (matching_category)
                *matching_category = m_name.GetCString();
            if (matching_type)
                *matching_type = eFormatCategoryItemValue;
            return true;
        }
    }
    if ( (items & eFormatCategoryItemRegexValue) == eFormatCategoryItemRegexValue )
    {
        if (m_regex_format_cont->Get(type_name, format_sp))
        {
            if (matching_category)
                *matching_category = m_name.GetCString();
            if (matching_type)
                *matching_type = eFormatCategoryItemRegexValue;
            return true;
        }
    }
    
    if ( (items & eFormatCategoryItemSummary) == eFormatCategoryItemSummary )
    {
        if (m_summary_cont->Get(type_name, summary_sp))
        {
            if (matching_category)
                *matching_category = m_name.GetCString();
            if (matching_type)
                *matching_type = eFormatCategoryItemSummary;
            return true;
        }
    }
    if ( (items & eFormatCategoryItemRegexSummary) == eFormatCategoryItemRegexSummary )
    {
        if (m_regex_summary_cont->Get(type_name, summary_sp))
        {
            if (matching_category)
                *matching_category = m_name.GetCString();
            if (matching_type)
                *matching_type = eFormatCategoryItemRegexSummary;
            return true;
        }
    }
    
    if ( (items & eFormatCategoryItemFilter)  == eFormatCategoryItemFilter )
    {
        if (m_filter_cont->Get(type_name, filter_sp))
        {
            if (matching_category)
                *matching_category = m_name.GetCString();
            if (matching_type)
                *matching_type = eFormatCategoryItemFilter;
            return true;
        }
    }
    if ( (items & eFormatCategoryItemRegexFilter) == eFormatCategoryItemRegexFilter )
    {
        if (m_regex_filter_cont->Get(type_name, filter_sp))
        {
            if (matching_category)
                *matching_category = m_name.GetCString();
            if (matching_type)
                *matching_type = eFormatCategoryItemRegexFilter;
            return true;
        }
    }
    
#ifndef LLDB_DISABLE_PYTHON
    if ( (items & eFormatCategoryItemSynth)  == eFormatCategoryItemSynth )
    {
        if (m_synth_cont->Get(type_name, synth_sp))
        {
            if (matching_category)
                *matching_category = m_name.GetCString();
            if (matching_type)
                *matching_type = eFormatCategoryItemSynth;
            return true;
        }
    }
    if ( (items & eFormatCategoryItemRegexSynth) == eFormatCategoryItemRegexSynth )
    {
        if (m_regex_synth_cont->Get(type_name, synth_sp))
        {
            if (matching_category)
                *matching_category = m_name.GetCString();
            if (matching_type)
                *matching_type = eFormatCategoryItemRegexSynth;
            return true;
        }
    }
#endif
    return false;
}

TypeCategoryImpl::FormatContainer::MapValueType
TypeCategoryImpl::GetFormatForType (lldb::TypeNameSpecifierImplSP type_sp)
{
    FormatContainer::MapValueType retval;
    
    if (type_sp)
    {
        if (type_sp->IsRegex())
            m_regex_format_cont->GetExact(ConstString(type_sp->GetName()),retval);
        else
            m_format_cont->GetExact(ConstString(type_sp->GetName()),retval);
    }
    
    return retval;
}

TypeCategoryImpl::SummaryContainer::MapValueType
TypeCategoryImpl::GetSummaryForType (lldb::TypeNameSpecifierImplSP type_sp)
{
    SummaryContainer::MapValueType retval;
    
    if (type_sp)
    {
        if (type_sp->IsRegex())
            m_regex_summary_cont->GetExact(ConstString(type_sp->GetName()),retval);
        else
            m_summary_cont->GetExact(ConstString(type_sp->GetName()),retval);
    }
    
    return retval;
}

TypeCategoryImpl::FilterContainer::MapValueType
TypeCategoryImpl::GetFilterForType (lldb::TypeNameSpecifierImplSP type_sp)
{
    FilterContainer::MapValueType retval;
    
    if (type_sp)
    {
        if (type_sp->IsRegex())
            m_regex_filter_cont->GetExact(ConstString(type_sp->GetName()),retval);
        else
            m_filter_cont->GetExact(ConstString(type_sp->GetName()),retval);
    }
    
    return retval;
}

#ifndef LLDB_DISABLE_PYTHON
TypeCategoryImpl::SynthContainer::MapValueType
TypeCategoryImpl::GetSyntheticForType (lldb::TypeNameSpecifierImplSP type_sp)
{
    SynthContainer::MapValueType retval;
    
    if (type_sp)
    {
        if (type_sp->IsRegex())
            m_regex_synth_cont->GetExact(ConstString(type_sp->GetName()),retval);
        else
            m_synth_cont->GetExact(ConstString(type_sp->GetName()),retval);
    }
    
    return retval;
}
#endif

lldb::TypeNameSpecifierImplSP
TypeCategoryImpl::GetTypeNameSpecifierForSummaryAtIndex (size_t index)
{
    if (index < m_summary_cont->GetCount())
        return m_summary_cont->GetTypeNameSpecifierAtIndex(index);
    else
        return m_regex_summary_cont->GetTypeNameSpecifierAtIndex(index-m_summary_cont->GetCount());
}

TypeCategoryImpl::FormatContainer::MapValueType
TypeCategoryImpl::GetFormatAtIndex (size_t index)
{
    if (index < m_format_cont->GetCount())
        return m_format_cont->GetAtIndex(index);
    else
        return m_regex_format_cont->GetAtIndex(index-m_format_cont->GetCount());
}

TypeCategoryImpl::SummaryContainer::MapValueType
TypeCategoryImpl::GetSummaryAtIndex (size_t index)
{
    if (index < m_summary_cont->GetCount())
        return m_summary_cont->GetAtIndex(index);
    else
        return m_regex_summary_cont->GetAtIndex(index-m_summary_cont->GetCount());
}

TypeCategoryImpl::FilterContainer::MapValueType
TypeCategoryImpl::GetFilterAtIndex (size_t index)
{
    if (index < m_filter_cont->GetCount())
        return m_filter_cont->GetAtIndex(index);
    else
        return m_regex_filter_cont->GetAtIndex(index-m_filter_cont->GetCount());
}

lldb::TypeNameSpecifierImplSP
TypeCategoryImpl::GetTypeNameSpecifierForFormatAtIndex (size_t index)
{
    if (index < m_format_cont->GetCount())
        return m_format_cont->GetTypeNameSpecifierAtIndex(index);
    else
        return m_regex_format_cont->GetTypeNameSpecifierAtIndex(index-m_format_cont->GetCount());
}

lldb::TypeNameSpecifierImplSP
TypeCategoryImpl::GetTypeNameSpecifierForFilterAtIndex (size_t index)
{
    if (index < m_filter_cont->GetCount())
        return m_filter_cont->GetTypeNameSpecifierAtIndex(index);
    else
        return m_regex_filter_cont->GetTypeNameSpecifierAtIndex(index-m_filter_cont->GetCount());
}

#ifndef LLDB_DISABLE_PYTHON
TypeCategoryImpl::SynthContainer::MapValueType
TypeCategoryImpl::GetSyntheticAtIndex (size_t index)
{
    if (index < m_synth_cont->GetCount())
        return m_synth_cont->GetAtIndex(index);
    else
        return m_regex_synth_cont->GetAtIndex(index-m_synth_cont->GetCount());
}

lldb::TypeNameSpecifierImplSP
TypeCategoryImpl::GetTypeNameSpecifierForSyntheticAtIndex (size_t index)
{
    if (index < m_synth_cont->GetCount())
        return m_synth_cont->GetTypeNameSpecifierAtIndex(index);
    else
        return m_regex_synth_cont->GetTypeNameSpecifierAtIndex(index - m_synth_cont->GetCount());
}
#endif

void
TypeCategoryImpl::Enable (bool value, uint32_t position)
{
    Mutex::Locker locker(m_mutex);
    m_enabled = value;
    m_enabled_position = position;
    if (m_change_listener)
        m_change_listener->Changed();
}
