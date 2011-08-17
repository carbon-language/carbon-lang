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

template<>
bool
FormatNavigator<lldb::RegularExpressionSP, SummaryFormat>::Get(ConstString key, SummaryFormat::SharedPointer& value)
{
    Mutex::Locker(m_format_map.mutex());
    MapIterator pos, end = m_format_map.map().end();
    for (pos = m_format_map.map().begin(); pos != end; pos++)
    {
        lldb::RegularExpressionSP regex = pos->first;
        if (regex->Execute(key.AsCString()))
        {
            value = pos->second;
            return true;
        }
    }
    return false;
}

template<>
bool
FormatNavigator<lldb::RegularExpressionSP, SummaryFormat>::Delete(ConstString type)
{
    Mutex::Locker(m_format_map.mutex());
    MapIterator pos, end = m_format_map.map().end();
    for (pos = m_format_map.map().begin(); pos != end; pos++)
    {
        lldb::RegularExpressionSP regex = pos->first;
        if ( ::strcmp(type.AsCString(),regex->GetText()) == 0)
        {
            m_format_map.map().erase(pos);
            if (m_format_map.listener)
                m_format_map.listener->Changed();
            return true;
        }
    }
    return false;
}

template<>
bool
FormatNavigator<lldb::RegularExpressionSP, SyntheticFilter>::Get(ConstString key, SyntheticFilter::SharedPointer& value)
{
    Mutex::Locker(m_format_map.mutex());
    MapIterator pos, end = m_format_map.map().end();
    for (pos = m_format_map.map().begin(); pos != end; pos++)
    {
        lldb::RegularExpressionSP regex = pos->first;
        if (regex->Execute(key.AsCString()))
        {
            value = pos->second;
            return true;
        }
    }
    return false;
}

template<>
bool
FormatNavigator<lldb::RegularExpressionSP, SyntheticFilter>::Delete(ConstString type)
{
    Mutex::Locker(m_format_map.mutex());
    MapIterator pos, end = m_format_map.map().end();
    for (pos = m_format_map.map().begin(); pos != end; pos++)
    {
        lldb::RegularExpressionSP regex = pos->first;
        if ( ::strcmp(type.AsCString(),regex->GetText()) == 0)
        {
            m_format_map.map().erase(pos);
            if (m_format_map.listener)
                m_format_map.listener->Changed();
            return true;
        }
    }
    return false;
}

template<>
bool
FormatNavigator<lldb::RegularExpressionSP, SyntheticScriptProvider>::Get(ConstString key, SyntheticFilter::SharedPointer& value)
{
    Mutex::Locker(m_format_map.mutex());
    MapIterator pos, end = m_format_map.map().end();
    for (pos = m_format_map.map().begin(); pos != end; pos++)
    {
        lldb::RegularExpressionSP regex = pos->first;
        if (regex->Execute(key.AsCString()))
        {
            value = pos->second;
            return true;
        }
    }
    return false;
}

template<>
bool
FormatNavigator<lldb::RegularExpressionSP, SyntheticScriptProvider>::Delete(ConstString type)
{
    Mutex::Locker(m_format_map.mutex());
    MapIterator pos, end = m_format_map.map().end();
    for (pos = m_format_map.map().begin(); pos != end; pos++)
    {
        lldb::RegularExpressionSP regex = pos->first;
        if ( ::strcmp(type.AsCString(),regex->GetText()) == 0)
        {
            m_format_map.map().erase(pos);
            if (m_format_map.listener)
                m_format_map.listener->Changed();
            return true;
        }
    }
    return false;
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
    
    
    Category(m_system_category_name)->Summary()->Add(ConstString("char *"), string_format);
    Category(m_system_category_name)->Summary()->Add(ConstString("const char *"), string_format);
    Category(m_system_category_name)->RegexSummary()->Add(any_size_char_arr, string_array_format);
    
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
    Category(m_gnu_cpp_category_name)->Summary()->Add(ConstString("std::string"),
                                                      std_string_summary_sp);
    Category(m_gnu_cpp_category_name)->Summary()->Add(ConstString("std::basic_string<char>"),
                                                      std_string_summary_sp);
    Category(m_gnu_cpp_category_name)->Summary()->Add(ConstString("std::basic_string<char,std::char_traits<char>,std::allocator<char> >"),
                                                      std_string_summary_sp);
    
    Category(m_gnu_cpp_category_name)->RegexSynth()->Add(RegularExpressionSP(new RegularExpression("std::vector<")),
                                     SyntheticChildrenSP(new SyntheticScriptProvider(true,
                                                                                     false,
                                                                                     false,
                                                                                     "StdVectorSynthProvider")));
    Category(m_gnu_cpp_category_name)->RegexSynth()->Add(RegularExpressionSP(new RegularExpression("std::map<")),
                                     SyntheticChildrenSP(new SyntheticScriptProvider(true,
                                                                                     false,
                                                                                     false,
                                                                                     "StdMapSynthProvider")));
    Category(m_gnu_cpp_category_name)->RegexSynth()->Add(RegularExpressionSP(new RegularExpression("std::list<")),
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
GetFormatManager() {
    static FormatManager g_format_manager;
    return g_format_manager;
}

void
DataVisualization::ForceUpdate()
{
    GetFormatManager().Changed();
}

bool
DataVisualization::ValueFormats::Get(ValueObject& valobj, lldb::DynamicValueType use_dynamic, ValueFormat::SharedPointer &entry)
{
    return GetFormatManager().Value().Get(valobj,entry, use_dynamic);
}

void
DataVisualization::ValueFormats::Add(const ConstString &type, const ValueFormat::SharedPointer &entry)
{
    GetFormatManager().Value().Add(type,entry);
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
DataVisualization::ValueFormats::GetCurrentRevision()
{
    return GetFormatManager().GetCurrentRevision();
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
DataVisualization::Categories::GetCurrentRevision()
{
    return GetFormatManager().GetCurrentRevision();
}

uint32_t
DataVisualization::Categories::GetCount()
{
    return GetFormatManager().Categories().GetCount();
}

bool
DataVisualization::NamedSummaryFormats::Get(const ConstString &type, SummaryFormat::SharedPointer &entry)
{
    return GetFormatManager().NamedSummary().Get(type,entry);
}

void
DataVisualization::NamedSummaryFormats::Add(const ConstString &type, const SummaryFormat::SharedPointer &entry)
{
    GetFormatManager().NamedSummary().Add(type,entry);
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
DataVisualization::NamedSummaryFormats::GetCurrentRevision()
{
    return GetFormatManager().GetCurrentRevision();
}

uint32_t
DataVisualization::NamedSummaryFormats::GetCount()
{
    return GetFormatManager().NamedSummary().GetCount();
}