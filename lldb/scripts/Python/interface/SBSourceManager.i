//===-- SWIG Interface for SBSourceManager ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBSourceManager
{
public:
    SBSourceManager (const lldb::SBSourceManager &rhs);
    
    ~SBSourceManager();

    size_t
    DisplaySourceLinesWithLineNumbers (const lldb::SBFileSpec &file,
                                       uint32_t line,
                                       uint32_t context_before,
                                       uint32_t context_after,
                                       const char* current_line_cstr,
                                       lldb::SBStream &s);
};

} // namespace lldb
