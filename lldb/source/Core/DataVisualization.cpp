//===-- DataVisualization.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DataVisualization.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/Core/Debugger.h"

using namespace lldb;
using namespace lldb_private;

static FormatManager&
GetFormatManager()
{
    static FormatManager g_format_manager;
    return g_format_manager;
}

void
DataVisualization::ForceUpdate ()
{
    GetFormatManager().Changed();
}

uint32_t
DataVisualization::GetCurrentRevision ()
{
    return GetFormatManager().GetCurrentRevision();
}

bool
DataVisualization::ValueFormats::Get (ValueObject& valobj, lldb::DynamicValueType use_dynamic, lldb::ValueFormatSP &entry)
{
    return GetFormatManager().GetValueNavigator().Get(valobj,entry, use_dynamic);
}

void
DataVisualization::ValueFormats::Add (const ConstString &type, const lldb::ValueFormatSP &entry)
{
    GetFormatManager().GetValueNavigator().Add(FormatManager::GetValidTypeName(type),entry);
}

bool
DataVisualization::ValueFormats::Delete (const ConstString &type)
{
    return GetFormatManager().GetValueNavigator().Delete(type);
}

void
DataVisualization::ValueFormats::Clear ()
{
    GetFormatManager().GetValueNavigator().Clear();
}

void
DataVisualization::ValueFormats::LoopThrough (ValueFormat::ValueCallback callback, void* callback_baton)
{
    GetFormatManager().GetValueNavigator().LoopThrough(callback, callback_baton);
}

uint32_t
DataVisualization::ValueFormats::GetCount ()
{
    return GetFormatManager().GetValueNavigator().GetCount();
}

bool
DataVisualization::GetSummaryFormat (ValueObject& valobj,
                                     lldb::DynamicValueType use_dynamic,
                                     lldb::SummaryFormatSP& entry)
{
    return GetFormatManager().Get(valobj, entry, use_dynamic);
}
bool
DataVisualization::GetSyntheticChildren (ValueObject& valobj,
                                         lldb::DynamicValueType use_dynamic,
                                         lldb::SyntheticChildrenSP& entry)
{
    return GetFormatManager().Get(valobj, entry, use_dynamic);
}

bool
DataVisualization::AnyMatches (ConstString type_name,
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
DataVisualization::Categories::Get (const ConstString &category, lldb::FormatCategorySP &entry)
{
    entry = GetFormatManager().GetCategory(category);
    return true;
}

void
DataVisualization::Categories::Add (const ConstString &category)
{
    GetFormatManager().GetCategory(category);
}

bool
DataVisualization::Categories::Delete (const ConstString &category)
{
    GetFormatManager().DisableCategory(category);
    return GetFormatManager().DeleteCategory(category);
}

void
DataVisualization::Categories::Clear ()
{
    GetFormatManager().ClearCategories();
}

void
DataVisualization::Categories::Clear (ConstString &category)
{
    GetFormatManager().GetCategory(category)->Clear(eFormatCategoryItemSummary | eFormatCategoryItemRegexSummary);
}

void
DataVisualization::Categories::Enable (ConstString& category)
{
    if (GetFormatManager().GetCategory(category)->IsEnabled() == false)
        GetFormatManager().EnableCategory(category);
    else
    {
        GetFormatManager().DisableCategory(category);
        GetFormatManager().EnableCategory(category);
    }
}

void
DataVisualization::Categories::Disable (ConstString& category)
{
    if (GetFormatManager().GetCategory(category)->IsEnabled() == true)
        GetFormatManager().DisableCategory(category);
}

void
DataVisualization::Categories::LoopThrough (FormatManager::CategoryCallback callback, void* callback_baton)
{
    GetFormatManager().LoopThroughCategories(callback, callback_baton);
}

uint32_t
DataVisualization::Categories::GetCount ()
{
    return GetFormatManager().GetCategoriesCount();
}

bool
DataVisualization::NamedSummaryFormats::Get (const ConstString &type, lldb::SummaryFormatSP &entry)
{
    return GetFormatManager().GetNamedSummaryNavigator().Get(type,entry);
}

void
DataVisualization::NamedSummaryFormats::Add (const ConstString &type, const lldb::SummaryFormatSP &entry)
{
    GetFormatManager().GetNamedSummaryNavigator().Add(FormatManager::GetValidTypeName(type),entry);
}

bool
DataVisualization::NamedSummaryFormats::Delete (const ConstString &type)
{
    return GetFormatManager().GetNamedSummaryNavigator().Delete(type);
}

void
DataVisualization::NamedSummaryFormats::Clear ()
{
    GetFormatManager().GetNamedSummaryNavigator().Clear();
}

void
DataVisualization::NamedSummaryFormats::LoopThrough (SummaryFormat::SummaryCallback callback, void* callback_baton)
{
    GetFormatManager().GetNamedSummaryNavigator().LoopThrough(callback, callback_baton);
}

uint32_t
DataVisualization::NamedSummaryFormats::GetCount ()
{
    return GetFormatManager().GetNamedSummaryNavigator().GetCount();
}
