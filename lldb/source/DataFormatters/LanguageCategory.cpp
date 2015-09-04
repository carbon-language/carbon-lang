//===-- LanguageCategory.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/DataFormatters/LanguageCategory.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/DataFormatters/FormatManager.h"
#include "lldb/DataFormatters/TypeCategory.h"
#include "lldb/DataFormatters/TypeFormat.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/DataFormatters/TypeValidator.h"
#include "lldb/Target/Language.h"

using namespace lldb;
using namespace lldb_private;

LanguageCategory::LanguageCategory (lldb::LanguageType lang_type) :
    m_category_sp(),
    m_format_cache()
{
    if (Language* language_plugin = Language::FindPlugin(lang_type))
        m_category_sp = language_plugin->GetFormatters();
    if (m_category_sp)
        m_category_sp->Enable(true, 1);
}

bool
LanguageCategory::Get (ValueObject& valobj,
                       lldb::DynamicValueType dynamic,
                       FormattersMatchVector matches,
                       lldb::TypeFormatImplSP& format_sp)
{
    if (!m_category_sp)
        return false;
    
    if (!m_category_sp->IsEnabled())
        return false;

    ConstString type_name = FormatManager::GetTypeForCache(valobj, dynamic);
    if (type_name)
    {
        if (m_format_cache.GetFormat(type_name, format_sp))
            return true;
    }
    bool result = m_category_sp->Get(valobj, matches, format_sp);
    if (type_name && (!format_sp || !format_sp->NonCacheable()))
    {
        m_format_cache.SetFormat(type_name, format_sp);
    }
    return result;
}

bool
LanguageCategory::Get (ValueObject& valobj,
                       lldb::DynamicValueType dynamic,
                       FormattersMatchVector matches,
                       lldb::TypeSummaryImplSP& format_sp)
{
    if (!m_category_sp)
        return false;
    
    if (!m_category_sp->IsEnabled())
        return false;

    ConstString type_name = FormatManager::GetTypeForCache(valobj, dynamic);
    if (type_name)
    {
        if (m_format_cache.GetSummary(type_name, format_sp))
            return true;
    }
    bool result = m_category_sp->Get(valobj, matches, format_sp);
    if (type_name && (!format_sp || !format_sp->NonCacheable()))
    {
        m_format_cache.SetSummary(type_name, format_sp);
    }
    return result;
}

bool
LanguageCategory::Get (ValueObject& valobj,
                       lldb::DynamicValueType dynamic,
                       FormattersMatchVector matches,
                       lldb::SyntheticChildrenSP& format_sp)
{
    if (!m_category_sp)
        return false;
    
    if (!m_category_sp->IsEnabled())
        return false;

    ConstString type_name = FormatManager::GetTypeForCache(valobj, dynamic);
    if (type_name)
    {
        if (m_format_cache.GetSynthetic(type_name, format_sp))
            return true;
    }
    bool result = m_category_sp->Get(valobj, matches, format_sp);
    if (type_name && (!format_sp || !format_sp->NonCacheable()))
    {
        m_format_cache.SetSynthetic(type_name, format_sp);
    }
    return result;
}

bool
LanguageCategory::Get (ValueObject& valobj,
                       lldb::DynamicValueType dynamic,
                       FormattersMatchVector matches,
                       lldb::TypeValidatorImplSP& format_sp)
{
    if (!m_category_sp)
        return false;

    if (!m_category_sp->IsEnabled())
        return false;

    ConstString type_name = FormatManager::GetTypeForCache(valobj, dynamic);
    if (type_name)
    {
        if (m_format_cache.GetValidator(type_name, format_sp))
            return true;
    }
    bool result = m_category_sp->Get(valobj, matches, format_sp);
    if (type_name && (!format_sp || !format_sp->NonCacheable()))
    {
        m_format_cache.SetValidator(type_name, format_sp);
    }
    return result;
}

lldb::TypeCategoryImplSP
LanguageCategory::GetCategory () const
{
    return m_category_sp;
}

void
LanguageCategory::Enable ()
{
    if (m_category_sp)
        m_category_sp->Enable(true, TypeCategoryMap::Default);
}

void
LanguageCategory::Disable ()
{
    if (m_category_sp)
        m_category_sp->Disable();
}
