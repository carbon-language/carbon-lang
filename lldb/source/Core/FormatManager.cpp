//===-- FormatManager.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/FormatManager.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/Core/Debugger.h"

using namespace lldb;
using namespace lldb_private;


struct FormatInfo
{
    Format format;
    const char format_char; // One or more format characters that can be used for this format.
    const char *format_name;    // Long format name that can be used to specify the current format
};

static FormatInfo 
g_format_infos[] = 
{
    { eFormatDefault        , '\0'  , "default"             },
    { eFormatBoolean        , 'B'   , "boolean"             },
    { eFormatBinary         , 'b'   , "binary"              },
    { eFormatBytes          , 'y'   , "bytes"               },
    { eFormatBytesWithASCII , 'Y'   , "bytes with ASCII"    },
    { eFormatChar           , 'c'   , "character"           },
    { eFormatCharPrintable  , 'C'   , "printable character" },
    { eFormatComplexFloat   , 'F'   , "complex float"       },
    { eFormatCString        , 's'   , "c-string"            },
    { eFormatDecimal        , 'i'   , "decimal"             },
    { eFormatEnum           , 'E'   , "enumeration"         },
    { eFormatHex            , 'x'   , "hex"                 },
    { eFormatFloat          , 'f'   , "float"               },
    { eFormatOctal          , 'o'   , "octal"               },
    { eFormatOSType         , 'O'   , "OSType"              },
    { eFormatUnicode16      , 'U'   , "unicode16"           },
    { eFormatUnicode32      , '\0'  , "unicode32"           },
    { eFormatUnsigned       , 'u'   , "unsigned decimal"    },
    { eFormatPointer        , 'p'   , "pointer"             },
    { eFormatVectorOfChar   , '\0'  , "char[]"              },
    { eFormatVectorOfSInt8  , '\0'  , "int8_t[]"            },
    { eFormatVectorOfUInt8  , '\0'  , "uint8_t[]"           },
    { eFormatVectorOfSInt16 , '\0'  , "int16_t[]"           },
    { eFormatVectorOfUInt16 , '\0'  , "uint16_t[]"          },
    { eFormatVectorOfSInt32 , '\0'  , "int32_t[]"           },
    { eFormatVectorOfUInt32 , '\0'  , "uint32_t[]"          },
    { eFormatVectorOfSInt64 , '\0'  , "int64_t[]"           },
    { eFormatVectorOfUInt64 , '\0'  , "uint64_t[]"          },
    { eFormatVectorOfFloat32, '\0'  , "float32[]"           },
    { eFormatVectorOfFloat64, '\0'  , "float64[]"           },
    { eFormatVectorOfUInt128, '\0'  , "uint128_t[]"         },
    { eFormatComplexInteger , 'I'   , "complex integer"     },
    { eFormatCharArray      , 'a'   , "character array"     }
};

static uint32_t 
g_num_format_infos = sizeof(g_format_infos)/sizeof(FormatInfo);

static bool
GetFormatFromFormatChar (char format_char, Format &format)
{
    for (uint32_t i=0; i<g_num_format_infos; ++i)
    {
        if (g_format_infos[i].format_char == format_char)
        {
            format = g_format_infos[i].format;
            return true;
        }
    }
    format = eFormatInvalid;
    return false;
}

static bool
GetFormatFromFormatName (const char *format_name, bool partial_match_ok, Format &format)
{
    uint32_t i;
    for (i=0; i<g_num_format_infos; ++i)
    {
        if (strcasecmp (g_format_infos[i].format_name, format_name) == 0)
        {
            format = g_format_infos[i].format;
            return true;
        }
    }
    
    if (partial_match_ok)
    {
        for (i=0; i<g_num_format_infos; ++i)
        {
            if (strcasestr (g_format_infos[i].format_name, format_name) == g_format_infos[i].format_name)
            {
                format = g_format_infos[i].format;
                return true;
            }
        }
    }
    format = eFormatInvalid;
    return false;
}

bool
FormatManager::GetFormatFromCString (const char *format_cstr,
                                     bool partial_match_ok,
                                     lldb::Format &format)
{
    bool success = false;
    if (format_cstr && format_cstr[0])
    {
        if (format_cstr[1] == '\0')
        {
            success = GetFormatFromFormatChar (format_cstr[0], format);
            if (success)
                return true;
        }
        
        success = GetFormatFromFormatName (format_cstr, partial_match_ok, format);
    }
    if (!success)
        format = eFormatInvalid;
    return success;
}

char
FormatManager::GetFormatAsFormatChar (lldb::Format format)
{
    for (uint32_t i=0; i<g_num_format_infos; ++i)
    {
        if (g_format_infos[i].format == format)
            return g_format_infos[i].format_char;
    }
    return '\0';
}
    


const char *
FormatManager::GetFormatAsCString (Format format)
{
    if (format >= eFormatDefault && format < kNumFormats)
        return g_format_infos[format].format_name;
    return NULL;
}

FormatCategory::FormatCategory(IFormatChangeListener* clist,
                               std::string name) :
    m_summary_nav(new SummaryNavigator("summary",clist)),
    m_regex_summary_nav(new RegexSummaryNavigator("regex-summary",clist)),
    m_filter_nav(new FilterNavigator("filter",clist)),
    m_regex_filter_nav(new RegexFilterNavigator("regex-filter",clist)),
    m_synth_nav(new SynthNavigator("synth",clist)),
    m_regex_synth_nav(new RegexSynthNavigator("regex-synth",clist)),
    m_enabled(false),
    m_change_listener(clist),
    m_mutex(Mutex::eMutexTypeRecursive),
    m_name(name)
{}

bool
FormatCategory::Get (ValueObject& valobj,
                     lldb::SummaryFormatSP& entry,
                     lldb::DynamicValueType use_dynamic,
                     uint32_t* reason)
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
FormatCategory::Get(ValueObject& valobj,
                    lldb::SyntheticChildrenSP& entry,
                    lldb::DynamicValueType use_dynamic,
                    uint32_t* reason)
{
    if (!IsEnabled())
        return false;
    SyntheticFilter::SharedPointer filter;
    SyntheticScriptProvider::SharedPointer synth;
    bool regex_filter, regex_synth;
    uint32_t reason_filter;
    uint32_t reason_synth;
    
    bool pick_synth = false;
    
    // first find both Filter and Synth, and then check which is most recent
    
    if (!GetFilterNavigator()->Get(valobj, filter, use_dynamic, &reason_filter))
        regex_filter = GetRegexFilterNavigator()->Get(valobj, filter, use_dynamic, &reason_filter);
    
    if (!GetSyntheticNavigator()->Get(valobj, synth, use_dynamic, &reason_synth))
        regex_synth = GetRegexSyntheticNavigator()->Get(valobj, synth, use_dynamic, &reason_synth);
    
    if (!filter.get() && !synth.get())
        return false;
    
    else if (!filter.get() && synth.get())
        pick_synth = true;
    
    else if (filter.get() && !synth.get())
        pick_synth = false;
    
    else /*if (filter.get() && synth.get())*/
    {
        if (filter->m_my_revision > synth->m_my_revision)
            pick_synth = false;
        else
            pick_synth = true;
    }
    
    if (pick_synth)
    {
        if (regex_synth && reason)
            *reason |= lldb::eFormatterChoiceCriterionRegularExpressionFilter;
        entry = synth;
        return true;
    }
    else
    {
        if (regex_filter && reason)
            *reason |= lldb::eFormatterChoiceCriterionRegularExpressionFilter;
        entry = filter;
        return true;
    }
}

void
FormatCategory::Clear (FormatCategoryItems items)
{
    if ( (items & eFormatCategoryItemSummary) == eFormatCategoryItemSummary )
        m_summary_nav->Clear();
    if ( (items & eFormatCategoryItemRegexSummary) == eFormatCategoryItemRegexSummary )
        m_regex_summary_nav->Clear();
    if ( (items & eFormatCategoryItemFilter)  == eFormatCategoryItemFilter )
        m_filter_nav->Clear();
    if ( (items & eFormatCategoryItemRegexFilter) == eFormatCategoryItemRegexFilter )
        m_regex_filter_nav->Clear();
    if ( (items & eFormatCategoryItemSynth)  == eFormatCategoryItemSynth )
        m_synth_nav->Clear();
    if ( (items & eFormatCategoryItemRegexSynth) == eFormatCategoryItemRegexSynth )
        m_regex_synth_nav->Clear();
}

bool
FormatCategory::Delete (ConstString name,
                        FormatCategoryItems items)
{
    bool success = false;
    if ( (items & eFormatCategoryItemSummary) == eFormatCategoryItemSummary )
        success = m_summary_nav->Delete(name) || success;
    if ( (items & eFormatCategoryItemRegexSummary) == eFormatCategoryItemRegexSummary )
        success = m_regex_summary_nav->Delete(name) || success;
    if ( (items & eFormatCategoryItemFilter)  == eFormatCategoryItemFilter )
        success = m_filter_nav->Delete(name) || success;
    if ( (items & eFormatCategoryItemRegexFilter) == eFormatCategoryItemRegexFilter )
        success = m_regex_filter_nav->Delete(name) || success;
    if ( (items & eFormatCategoryItemSynth)  == eFormatCategoryItemSynth )
        success = m_synth_nav->Delete(name) || success;
    if ( (items & eFormatCategoryItemRegexSynth) == eFormatCategoryItemRegexSynth )
        success = m_regex_synth_nav->Delete(name) || success;
    return success;
}

uint32_t
FormatCategory::GetCount (FormatCategoryItems items)
{
    uint32_t count = 0;
    if ( (items & eFormatCategoryItemSummary) == eFormatCategoryItemSummary )
        count += m_summary_nav->GetCount();
    if ( (items & eFormatCategoryItemRegexSummary) == eFormatCategoryItemRegexSummary )
        count += m_regex_summary_nav->GetCount();
    if ( (items & eFormatCategoryItemFilter)  == eFormatCategoryItemFilter )
        count += m_filter_nav->GetCount();
    if ( (items & eFormatCategoryItemRegexFilter) == eFormatCategoryItemRegexFilter )
        count += m_regex_filter_nav->GetCount();
    if ( (items & eFormatCategoryItemSynth)  == eFormatCategoryItemSynth )
        count += m_synth_nav->GetCount();
    if ( (items & eFormatCategoryItemRegexSynth) == eFormatCategoryItemRegexSynth )
        count += m_regex_synth_nav->GetCount();
    return count;
}

bool
FormatCategory::AnyMatches(ConstString type_name,
                           FormatCategoryItems items,
                           bool only_enabled,
                           const char** matching_category,
                           FormatCategoryItems* matching_type)
{
    if (!IsEnabled() && only_enabled)
        return false;
    
    lldb::SummaryFormatSP summary;
    SyntheticFilter::SharedPointer filter;
    SyntheticScriptProvider::SharedPointer synth;
    
    if ( (items & eFormatCategoryItemSummary) == eFormatCategoryItemSummary )
    {
        if (m_summary_nav->Get(type_name, summary))
        {
            if (matching_category)
                *matching_category = m_name.c_str();
            if (matching_type)
                *matching_type = eFormatCategoryItemSummary;
            return true;
        }
    }
    if ( (items & eFormatCategoryItemRegexSummary) == eFormatCategoryItemRegexSummary )
    {
        if (m_regex_summary_nav->Get(type_name, summary))
        {
            if (matching_category)
                *matching_category = m_name.c_str();
            if (matching_type)
                *matching_type = eFormatCategoryItemRegexSummary;
            return true;
        }
    }
    if ( (items & eFormatCategoryItemFilter)  == eFormatCategoryItemFilter )
    {
        if (m_filter_nav->Get(type_name, filter))
        {
            if (matching_category)
                *matching_category = m_name.c_str();
            if (matching_type)
                *matching_type = eFormatCategoryItemFilter;
            return true;
        }
    }
    if ( (items & eFormatCategoryItemRegexFilter) == eFormatCategoryItemRegexFilter )
    {
        if (m_regex_filter_nav->Get(type_name, filter))
        {
            if (matching_category)
                *matching_category = m_name.c_str();
            if (matching_type)
                *matching_type = eFormatCategoryItemRegexFilter;
            return true;
        }
    }
    if ( (items & eFormatCategoryItemSynth)  == eFormatCategoryItemSynth )
    {
        if (m_synth_nav->Get(type_name, synth))
        {
            if (matching_category)
                *matching_category = m_name.c_str();
            if (matching_type)
                *matching_type = eFormatCategoryItemSynth;
            return true;
        }
    }
    if ( (items & eFormatCategoryItemRegexSynth) == eFormatCategoryItemRegexSynth )
    {
        if (m_regex_synth_nav->Get(type_name, synth))
        {
            if (matching_category)
                *matching_category = m_name.c_str();
            if (matching_type)
                *matching_type = eFormatCategoryItemRegexSynth;
            return true;
        }
    }
    return false;
}

bool
CategoryMap::AnyMatches (ConstString type_name,
                                    FormatCategory::FormatCategoryItems items,
                                    bool only_enabled,
                                    const char** matching_category,
                                    FormatCategory::FormatCategoryItems* matching_type)
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

bool
CategoryMap::Get (ValueObject& valobj,
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
CategoryMap::Get (ValueObject& valobj,
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

void
CategoryMap::LoopThrough(CallbackType callback, void* param)
{
    if (callback)
    {
        Mutex::Locker(m_map_mutex);
        
        // loop through enabled categories in respective order
        {
            ActiveCategoriesIterator begin, end = m_active_categories.end();
            for (begin = m_active_categories.begin(); begin != end; begin++)
            {
                lldb::FormatCategorySP category = *begin;
                ConstString type = ConstString(category->GetName().c_str());
                if (!callback(param, category))
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
                if (!callback(param, pos->second))
                    break;
            }
        }
    }
}

lldb::FormatCategorySP
FormatManager::GetCategory (const ConstString& category_name,
                         bool can_create)
{
    if (!category_name)
        return GetCategory(m_default_category_name);
    lldb::FormatCategorySP category;
    if (m_categories_map.Get(category_name, category))
        return category;
    
    if (!can_create)
        return lldb::FormatCategorySP();
    
    m_categories_map.Add(category_name,lldb::FormatCategorySP(new FormatCategory(this, category_name.AsCString())));
    return GetCategory(category_name);
}

lldb::Format
FormatManager::GetSingleItemFormat(lldb::Format vector_format)
{
    switch(vector_format)
    {
        case eFormatVectorOfChar:
            return eFormatCharArray;
            
        case eFormatVectorOfSInt8:
        case eFormatVectorOfSInt16:
        case eFormatVectorOfSInt32:
        case eFormatVectorOfSInt64:
            return eFormatDecimal;
            
        case eFormatVectorOfUInt8:
        case eFormatVectorOfUInt16:
        case eFormatVectorOfUInt32:
        case eFormatVectorOfUInt64:
        case eFormatVectorOfUInt128:
            return eFormatHex;
            
        case eFormatVectorOfFloat32:
        case eFormatVectorOfFloat64:
            return eFormatFloat;
            
        default:
            return lldb::eFormatInvalid;
    }
}

ConstString
FormatManager::GetValidTypeName (const ConstString& type)
{
    return ::GetValidTypeName_Impl(type);
}

FormatManager::FormatManager() : 
    m_value_nav("format",this),
    m_named_summaries_map(this),
    m_last_revision(0),
    m_categories_map(this),
    m_default_category_name(ConstString("default")),
    m_system_category_name(ConstString("system")), 
    m_gnu_cpp_category_name(ConstString("gnu-libstdc++"))
{
    
    // add some default stuff
    // most formats, summaries, ... actually belong to the users' lldbinit file rather than here
    lldb::SummaryFormatSP string_format(new StringSummaryFormat(false,
                                                                       true,
                                                                       false,
                                                                       true,
                                                                       false,
                                                                       false,
                                                                       "${var%s}"));
    
    
    lldb::SummaryFormatSP string_array_format(new StringSummaryFormat(false,
                                                                             true,
                                                                             false,
                                                                             false,
                                                                             false,
                                                                             false,
                                                                             "${var%s}"));
    
    lldb::RegularExpressionSP any_size_char_arr(new RegularExpression("char \\[[0-9]+\\]"));
    
    
    FormatCategory::SharedPointer sys_category_sp = GetCategory(m_system_category_name);
    
    sys_category_sp->GetSummaryNavigator()->Add(ConstString("char *"), string_format);
    sys_category_sp->GetSummaryNavigator()->Add(ConstString("const char *"), string_format);
    sys_category_sp->GetRegexSummaryNavigator()->Add(any_size_char_arr, string_array_format);
    
    GetCategory(m_default_category_name); // this call is there to force LLDB into creating an empty "default" category
    
    // WARNING: temporary code!!
    // The platform should be responsible for initializing its own formatters
    // (e.g. to handle versioning, different runtime libraries, ...)
    // Currently, basic formatters for std:: objects as implemented by
    // the GNU libstdc++ are defined regardless, and enabled by default
    // This is going to be moved to some platform-dependent location
    // (in the meanwhile, these formatters should work for Mac OS X & Linux)
    lldb::SummaryFormatSP std_string_summary_sp(new StringSummaryFormat(true,
                                                                        false,
                                                                        false,
                                                                        true,
                                                                        true,
                                                                        false,
                                                                        "${var._M_dataplus._M_p}"));
    
    FormatCategory::SharedPointer gnu_category_sp = GetCategory(m_gnu_cpp_category_name);
    
    gnu_category_sp->GetSummaryNavigator()->Add(ConstString("std::string"),
                                                std_string_summary_sp);
    gnu_category_sp->GetSummaryNavigator()->Add(ConstString("std::basic_string<char>"),
                                                std_string_summary_sp);
    gnu_category_sp->GetSummaryNavigator()->Add(ConstString("std::basic_string<char,std::char_traits<char>,std::allocator<char> >"),
                                                std_string_summary_sp);
    
    gnu_category_sp->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("^(std::)?vector<.+>$")),
                                     SyntheticChildrenSP(new SyntheticScriptProvider(true,
                                                                                     false,
                                                                                     false,
                                                                                     "gnu_libstdcpp.StdVectorSynthProvider")));
    gnu_category_sp->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("^(std::)?map<.+> >$")),
                                     SyntheticChildrenSP(new SyntheticScriptProvider(true,
                                                                                     false,
                                                                                     false,
                                                                                     "gnu_libstdcpp.StdMapSynthProvider")));
    gnu_category_sp->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("^(std::)?list<.+>$")),
                                     SyntheticChildrenSP(new SyntheticScriptProvider(true,
                                                                                     false,
                                                                                     false,
                                                                                     "gnu_libstdcpp.StdListSynthProvider")));
    
    // DO NOT change the order of these calls, unless you WANT a change in the priority of these categories
    EnableCategory(m_system_category_name);
    EnableCategory(m_gnu_cpp_category_name);
    EnableCategory(m_default_category_name);
    
}