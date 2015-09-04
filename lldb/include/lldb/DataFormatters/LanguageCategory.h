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
    
    lldb::TypeCategoryImplSP
    GetCategory () const;
    
    void
    Enable ();
    
    void
    Disable ();
    
private:
    lldb::TypeCategoryImplSP m_category_sp;
    lldb_private::FormatCache m_format_cache;
};
    
} // namespace lldb_private

#endif // lldb_LanguageCategory_h_
