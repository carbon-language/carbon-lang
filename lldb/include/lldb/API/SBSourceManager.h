//===-- SBSourceManager.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBSourceManager_h_
#define LLDB_SBSourceManager_h_

#include "lldb/API/SBDefines.h"

#include <stdio.h>

namespace lldb {

class SBSourceManager
{
public:
    ~SBSourceManager();

    size_t
    DisplaySourceLinesWithLineNumbers (const lldb::SBFileSpec &file,
                                       uint32_t line,
                                       uint32_t context_before,
                                       uint32_t context_after,
                                       const char* current_line_cstr,
                                       FILE *f);


protected:
    friend class SBCommandInterpreter;
    friend class SBDebugger;

    SBSourceManager(lldb_private::SourceManager &source_manager);

    lldb_private::SourceManager &
    GetLLDBManager ();

private:

    lldb_private::SourceManager &m_source_manager;
};

} // namespace lldb

#endif  // LLDB_SBSourceManager_h_
