//===-- MemoryHistoryASan.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_MemoryHistoryASan_h_
#define liblldb_MemoryHistoryASan_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/MemoryHistory.h"
#include "lldb/Target/Process.h"

namespace lldb_private {

class MemoryHistoryASan : public lldb_private::MemoryHistory
{
public:
    
    static lldb::MemoryHistorySP
    CreateInstance (const lldb::ProcessSP &process_sp);
    
    static void
    Initialize();
    
    static void
    Terminate();
    
    static lldb_private::ConstString
    GetPluginNameStatic();
    
    virtual
    ~MemoryHistoryASan () {}
    
    virtual lldb_private::ConstString
    GetPluginName() { return GetPluginNameStatic(); }
    
    virtual uint32_t
    GetPluginVersion() { return 1; }
    
    virtual lldb_private::HistoryThreads
    GetHistoryThreads(lldb::addr_t address);
    
private:
    
    MemoryHistoryASan(const lldb::ProcessSP &process_sp);
    
    lldb::ProcessSP m_process_sp;
    
};

} // namespace lldb_private
    
#endif  // liblldb_MemoryHistoryASan_h_
