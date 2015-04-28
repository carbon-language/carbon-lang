//===-- SWIG Interface for SBLanguageRuntime --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBLanguageRuntime
{
public:
    static lldb::LanguageType
    GetLanguageTypeFromString (const char *string);
    
    static const char *
    GetNameForLanguageType (lldb::LanguageType language);
};

} // namespace lldb
