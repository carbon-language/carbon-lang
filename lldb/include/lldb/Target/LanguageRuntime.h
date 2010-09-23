//===-- LanguageRuntime.h ---------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_LanguageRuntime_h_
#define liblldb_LanguageRuntime_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/PluginInterface.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class LanguageRuntime :
    public PluginInterface
{
public:
    virtual
    ~LanguageRuntime();
    
    static LanguageRuntime* 
    FindPlugin (Process *process, lldb::LanguageType language);
    
    virtual lldb::LanguageType
    GetLanguageType () const = 0;
    
protected:
    //------------------------------------------------------------------
    // Classes that inherit from LanguageRuntime can see and modify these
    //------------------------------------------------------------------
    LanguageRuntime(Process *process);
private:
    Process *m_process_ptr;
    DISALLOW_COPY_AND_ASSIGN (LanguageRuntime);
};

} // namespace lldb_private

#endif  // liblldb_LanguageRuntime_h_
