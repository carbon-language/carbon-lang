//===-- LanguageCategory.h----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_LanguageCategory_h_
#define lldb_LanguageCategory_h_

// C Includes
// C++ Includes

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"
#include "lldb/DataFormatters/FormatCache.h"
#include "lldb/DataFormatters/FormatClasses.h"

#include <memory>

namespace lldb_private {

class LanguageCategory
{
public:
    typedef std::unique_ptr<LanguageCategory> UniquePointer;
    
    LanguageCategory (lldb::LanguageType lang_type);
    
    bool
    Get (ValueObject& valobj,
         lldb::DynamicValueType dynamic,
         FormattersMatchVector matches,
         lldb::TypeFormatImplSP& format_sp);

    bool
    Get (ValueObject& valobj,
         lldb::DynamicValueType dynamic,
         FormattersMatchVector matches,
         lldb::TypeSummaryImplSP& format_sp);

    bool
    Get (ValueObject& valobj,
         lldb::DynamicValueType dynamic,
         FormattersMatchVector matches,
         lldb::SyntheticChildrenSP& format_sp);

    bool
    Get (ValueObject& valobj,
         lldb::DynamicValueType dynamic,
         FormattersMatchVector matches,
         lldb::TypeValidatorImplSP& format_sp);

    bool
    GetHardcoded (ValueObject& valobj,
                  lldb::DynamicValueType use_dynamic,
                  FormatManager& fmt_mgr,
                  lldb::TypeFormatImplSP& format_sp);

    bool
    GetHardcoded (ValueObject& valobj,
                  lldb::DynamicValueType use_dynamic,
                  FormatManager& fmt_mgr,
                  lldb::TypeSummaryImplSP& format_sp);
    
    bool
    GetHardcoded (ValueObject& valobj,
                  lldb::DynamicValueType use_dynamic,
                  FormatManager& fmt_mgr,
                  lldb::SyntheticChildrenSP& format_sp);
    
    bool
    GetHardcoded (ValueObject& valobj,
                  lldb::DynamicValueType use_dynamic,
                  FormatManager& fmt_mgr,
                  lldb::TypeValidatorImplSP& format_sp);
    
    lldb::TypeCategoryImplSP
    GetCategory () const;
    
    void
    Enable ();
    
    void
    Disable ();
    
    bool
    IsEnabled ();
    
private:
    lldb::TypeCategoryImplSP m_category_sp;
    
    HardcodedFormatters::HardcodedFormatFinder m_hardcoded_formats;
    HardcodedFormatters::HardcodedSummaryFinder m_hardcoded_summaries;
    HardcodedFormatters::HardcodedSyntheticFinder m_hardcoded_synthetics;
    HardcodedFormatters::HardcodedValidatorFinder m_hardcoded_validators;
    
    lldb_private::FormatCache m_format_cache;
    
    bool m_enabled;
};
    
} // namespace lldb_private

#endif // lldb_LanguageCategory_h_
