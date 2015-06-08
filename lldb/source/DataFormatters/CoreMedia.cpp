//===-- CoreMedia.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/DataFormatters/CXXFormatterFunctions.h"

#include "lldb/Core/Flags.h"
#include "lldb/Symbol/ClangASTContext.h"

#include <inttypes.h>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

bool
lldb_private::formatters::CMTimeSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
{
    ClangASTContext *ast_ctx = ClangASTContext::GetASTContext(valobj.GetClangType().GetASTContext());
    if (!ast_ctx)
        return false;
    
    // fetch children by offset to compensate for potential lack of debug info
    auto int64_ty = ast_ctx->GetIntTypeFromBitSize(64, true);
    auto int32_ty = ast_ctx->GetIntTypeFromBitSize(32, true);
    
    auto value_sp(valobj.GetSyntheticChildAtOffset(0, int64_ty, true));
    auto timescale_sp(valobj.GetSyntheticChildAtOffset(8, int32_ty, true));
    auto flags_sp(valobj.GetSyntheticChildAtOffset(12, int32_ty, true));
    
    if (!value_sp || !timescale_sp || !flags_sp)
        return false;
    
    auto value = value_sp->GetValueAsUnsigned(0);
    auto timescale = (int32_t)timescale_sp->GetValueAsUnsigned(0); // the timescale specifies the fraction of a second each unit in the numerator occupies
    auto flags = Flags(flags_sp->GetValueAsUnsigned(0) & 0x00000000000000FF); // the flags I need sit in the LSB
    
    const unsigned int FlagPositiveInf = 4;
    const unsigned int FlagNegativeInf = 8;
    const unsigned int FlagIndefinite = 16;
    
    if (flags.AnySet(FlagIndefinite))
    {
        stream.Printf("indefinite");
        return true;
    }
    
    if (flags.AnySet(FlagPositiveInf))
    {
        stream.Printf("+oo");
        return true;
    }
    
    if (flags.AnySet(FlagNegativeInf))
    {
        stream.Printf("-oo");
        return true;
    }
    
    if (timescale == 0)
        return false;
    
    switch (timescale)
    {
        case 0:
            return false;
        case 1:
            stream.Printf("%" PRId64 " seconds", value);
            return true;
        case 2:
            stream.Printf("%" PRId64 " half seconds", value);
            return true;
        case 3:
            stream.Printf("%" PRId64 " third%sof a second", value, value == 1 ? " " : "s ");
            return true;
        default:
            stream.Printf("%" PRId64 " %" PRId32 "th%sof a second", value, timescale, value == 1 ? " " : "s ");
            return true;
    }
}
