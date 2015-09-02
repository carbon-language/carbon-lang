//===-- FormattersHelpers.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_FormattersHelpers_h_
#define lldb_FormattersHelpers_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-forward.h"
#include "lldb/lldb-enumerations.h"

#include "lldb/DataFormatters/TypeCategory.h"
#include "lldb/DataFormatters/TypeFormat.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"

namespace lldb_private {
    namespace formatters {
        void
        AddFormat (TypeCategoryImpl::SharedPointer category_sp,
                   lldb::Format format,
                   ConstString type_name,
                   TypeFormatImpl::Flags flags,
                   bool regex = false);
        
        void
        AddStringSummary(TypeCategoryImpl::SharedPointer category_sp,
                         const char* string,
                         ConstString type_name,
                         TypeSummaryImpl::Flags flags,
                         bool regex = false);
        
        void
        AddOneLineSummary (TypeCategoryImpl::SharedPointer category_sp,
                           ConstString type_name,
                           TypeSummaryImpl::Flags flags,
                           bool regex = false);

#ifndef LLDB_DISABLE_PYTHON
        void
        AddCXXSummary (TypeCategoryImpl::SharedPointer category_sp,
                       CXXFunctionSummaryFormat::Callback funct,
                       const char* description,
                       ConstString type_name,
                       TypeSummaryImpl::Flags flags,
                       bool regex = false);

        void
        AddCXXSynthetic  (TypeCategoryImpl::SharedPointer category_sp,
                          CXXSyntheticChildren::CreateFrontEndCallback generator,
                          const char* description,
                          ConstString type_name,
                          ScriptedSyntheticChildren::Flags flags,
                          bool regex = false);

        void
        AddFilter  (TypeCategoryImpl::SharedPointer category_sp,
                    std::vector<std::string> children,
                    const char* description,
                    ConstString type_name,
                    ScriptedSyntheticChildren::Flags flags,
                    bool regex = false);
#endif

    } // namespace formatters
} // namespace lldb_private

#endif	// lldb_FormattersHelpers_h_
