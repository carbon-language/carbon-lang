//===-- LibStdCpp.h ---------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_LibStdCpp_h_
#define liblldb_LibStdCpp_h_

#include "lldb/Core/Stream.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"

namespace lldb_private {
    namespace formatters
    {
        SyntheticChildrenFrontEnd* LibstdcppMapIteratorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        SyntheticChildrenFrontEnd* LibStdcppVectorIteratorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
    } // namespace formatters
} // namespace lldb_private

#endif // liblldb_LibStdCpp_h_
