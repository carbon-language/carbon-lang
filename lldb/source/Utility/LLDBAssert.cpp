//===--------------------- LLDBAssert.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/Host.h"

using namespace lldb_private;

void
lldb_private::lldb_assert (int expression,
                           const char* expr_text,
                           const char* func,
                           const char* file,
                           unsigned int line)
{
    if (expression)
        ;
    else
    {
        StreamString stream;
        stream.Printf("Assertion failed: (%s), function %s, file %s, line %u\n",
                      expr_text,
                      func,
                      file,
                      line);
        stream.Printf("backtrace leading to the failure:\n");
        Host::Backtrace(stream, 1000);
        stream.Printf("please file a bug report against lldb reporting this failure log, and as many details as possible\n");
        printf("%s\n", stream.GetData());
    }
}
