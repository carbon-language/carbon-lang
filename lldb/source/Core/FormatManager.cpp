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
    { eFormatDecimal        , 'i'   , "signed decimal"      },
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
    
    if ( (items & eSummary) == eSummary )
    {
        if (m_summary_nav->Get(type_name, summary))
        {
            if (matching_category)
                *matching_category = m_name.c_str();
            if (matching_type)
                *matching_type = eSummary;
            return true;
        }
    }
    if ( (items & eRegexSummary) == eRegexSummary )
    {
        if (m_regex_summary_nav->Get(type_name, summary))
        {
            if (matching_category)
                *matching_category = m_name.c_str();
            if (matching_type)
                *matching_type = eRegexSummary;
            return true;
        }
    }
    if ( (items & eFilter)  == eFilter )
    {
        if (m_filter_nav->Get(type_name, filter))
        {
            if (matching_category)
                *matching_category = m_name.c_str();
            if (matching_type)
                *matching_type = eFilter;
            return true;
        }
    }
    if ( (items & eRegexFilter) == eRegexFilter )
    {
        if (m_regex_filter_nav->Get(type_name, filter))
        {
            if (matching_category)
                *matching_category = m_name.c_str();
            if (matching_type)
                *matching_type = eRegexFilter;
            return true;
        }
    }
    if ( (items & eSynth)  == eSynth )
    {
        if (m_synth_nav->Get(type_name, synth))
        {
            if (matching_category)
                *matching_category = m_name.c_str();
            if (matching_type)
                *matching_type = eSynth;
            return true;
        }
    }
    if ( (items & eRegexSynth) == eRegexSynth )
    {
        if (m_regex_synth_nav->Get(type_name, synth))
        {
            if (matching_category)
                *matching_category = m_name.c_str();
            if (matching_type)
                *matching_type = eRegexSynth;
            return true;
        }
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
    m_default_cs(ConstString("default")),
    m_system_cs(ConstString("system")), 
    m_gnu_stdcpp_cs(ConstString("gnu-libstdc++"))
{
    
    // build default categories
    
    m_default_category_name = m_default_cs.GetCString();
    m_system_category_name = m_system_cs.GetCString();
    m_gnu_cpp_category_name = m_gnu_stdcpp_cs.AsCString();
    
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
    
    
    Category(m_system_category_name)->GetSummaryNavigator()->Add(ConstString("char *"), string_format);
    Category(m_system_category_name)->GetSummaryNavigator()->Add(ConstString("const char *"), string_format);
    Category(m_system_category_name)->GetRegexSummaryNavigator()->Add(any_size_char_arr, string_array_format);
    
    Category(m_default_category_name); // this call is there to force LLDB into creating an empty "default" category
    
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
    Category(m_gnu_cpp_category_name)->GetSummaryNavigator()->Add(ConstString("std::string"),
                                                      std_string_summary_sp);
    Category(m_gnu_cpp_category_name)->GetSummaryNavigator()->Add(ConstString("std::basic_string<char>"),
                                                      std_string_summary_sp);
    Category(m_gnu_cpp_category_name)->GetSummaryNavigator()->Add(ConstString("std::basic_string<char,std::char_traits<char>,std::allocator<char> >"),
                                                      std_string_summary_sp);
    
    Category(m_gnu_cpp_category_name)->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("std::vector<")),
                                     SyntheticChildrenSP(new SyntheticScriptProvider(true,
                                                                                     false,
                                                                                     false,
                                                                                     "StdVectorSynthProvider")));
    Category(m_gnu_cpp_category_name)->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("std::map<")),
                                     SyntheticChildrenSP(new SyntheticScriptProvider(true,
                                                                                     false,
                                                                                     false,
                                                                                     "StdMapSynthProvider")));
    Category(m_gnu_cpp_category_name)->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("std::list<")),
                                     SyntheticChildrenSP(new SyntheticScriptProvider(true,
                                                                                     false,
                                                                                     false,
                                                                                     "StdListSynthProvider")));
    
    // DO NOT change the order of these calls, unless you WANT a change in the priority of these categories
    EnableCategory(m_system_category_name);
    EnableCategory(m_gnu_cpp_category_name);
    EnableCategory(m_default_category_name);
    
}

static FormatManager&
GetFormatManager()
{
    static FormatManager g_format_manager;
    return g_format_manager;
}

void
DataVisualization::ForceUpdate()
{
    GetFormatManager().Changed();
}

uint32_t
DataVisualization::GetCurrentRevision ()
{
    return GetFormatManager().GetCurrentRevision();
}

bool
DataVisualization::ValueFormats::Get(ValueObject& valobj, lldb::DynamicValueType use_dynamic, lldb::ValueFormatSP &entry)
{
    return GetFormatManager().Value().Get(valobj,entry, use_dynamic);
}

void
DataVisualization::ValueFormats::Add(const ConstString &type, const lldb::ValueFormatSP &entry)
{
    GetFormatManager().Value().Add(FormatManager::GetValidTypeName(type),entry);
}

bool
DataVisualization::ValueFormats::Delete(const ConstString &type)
{
    return GetFormatManager().Value().Delete(type);
}

void
DataVisualization::ValueFormats::Clear()
{
    GetFormatManager().Value().Clear();
}

void
DataVisualization::ValueFormats::LoopThrough(ValueFormat::ValueCallback callback, void* callback_baton)
{
    GetFormatManager().Value().LoopThrough(callback, callback_baton);
}

uint32_t
DataVisualization::ValueFormats::GetCount()
{
    return GetFormatManager().Value().GetCount();
}

bool
DataVisualization::GetSummaryFormat(ValueObject& valobj,
                                       lldb::DynamicValueType use_dynamic,
                                       lldb::SummaryFormatSP& entry)
{
    return GetFormatManager().Get(valobj, entry, use_dynamic);
}
bool
DataVisualization::GetSyntheticChildren(ValueObject& valobj,
                                           lldb::DynamicValueType use_dynamic,
                                           lldb::SyntheticChildrenSP& entry)
{
    return GetFormatManager().Get(valobj, entry, use_dynamic);
}

bool
DataVisualization::AnyMatches(ConstString type_name,
                                 FormatCategory::FormatCategoryItems items,
                                 bool only_enabled,
                                 const char** matching_category,
                                 FormatCategory::FormatCategoryItems* matching_type)
{
    return GetFormatManager().AnyMatches(type_name,
                                         items,
                                         only_enabled,
                                         matching_category,
                                         matching_type);
}

bool
DataVisualization::Categories::Get(const ConstString &category, lldb::FormatCategorySP &entry)
{
    entry = GetFormatManager().Category(category.GetCString());
    return true;
}

void
DataVisualization::Categories::Add(const ConstString &category)
{
    GetFormatManager().Category(category.GetCString());
}

bool
DataVisualization::Categories::Delete(const ConstString &category)
{
    GetFormatManager().DisableCategory(category.GetCString());
    return GetFormatManager().Categories().Delete(category.GetCString());
}

void
DataVisualization::Categories::Clear()
{
    GetFormatManager().Categories().Clear();
}

void
DataVisualization::Categories::Clear(ConstString &category)
{
    GetFormatManager().Category(category.GetCString())->ClearSummaries();
}

void
DataVisualization::Categories::Enable(ConstString& category)
{
    if (GetFormatManager().Category(category.GetCString())->IsEnabled() == false)
        GetFormatManager().EnableCategory(category.GetCString());
    else
    {
        GetFormatManager().DisableCategory(category.GetCString());
        GetFormatManager().EnableCategory(category.GetCString());
    }
}

void
DataVisualization::Categories::Disable(ConstString& category)
{
    if (GetFormatManager().Category(category.GetCString())->IsEnabled() == true)
        GetFormatManager().DisableCategory(category.GetCString());
}

void
DataVisualization::Categories::LoopThrough(FormatManager::CategoryCallback callback, void* callback_baton)
{
    GetFormatManager().LoopThroughCategories(callback, callback_baton);
}

uint32_t
DataVisualization::Categories::GetCount()
{
    return GetFormatManager().Categories().GetCount();
}

bool
DataVisualization::NamedSummaryFormats::Get(const ConstString &type, lldb::SummaryFormatSP &entry)
{
    return GetFormatManager().NamedSummary().Get(type,entry);
}

void
DataVisualization::NamedSummaryFormats::Add(const ConstString &type, const lldb::SummaryFormatSP &entry)
{
    GetFormatManager().NamedSummary().Add(FormatManager::GetValidTypeName(type),entry);
}

bool
DataVisualization::NamedSummaryFormats::Delete(const ConstString &type)
{
    return GetFormatManager().NamedSummary().Delete(type);
}

void
DataVisualization::NamedSummaryFormats::Clear()
{
    GetFormatManager().NamedSummary().Clear();
}

void
DataVisualization::NamedSummaryFormats::LoopThrough(SummaryFormat::SummaryCallback callback, void* callback_baton)
{
    GetFormatManager().NamedSummary().LoopThrough(callback, callback_baton);
}

uint32_t
DataVisualization::NamedSummaryFormats::GetCount()
{
    return GetFormatManager().NamedSummary().GetCount();
}