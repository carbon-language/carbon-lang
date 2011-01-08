//===-- Symtab.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#ifndef liblldb_UnwindTable_h
#define liblldb_UnwindTable_h

#include <map>

#include "lldb/lldb-private.h"

namespace lldb_private {

// A class which holds all the FuncUnwinders objects for a given ObjectFile.
// The UnwindTable is populated with FuncUnwinders objects lazily during
// the debug session.

class UnwindTable
{
public:
    UnwindTable(ObjectFile& objfile);
    ~UnwindTable();

    lldb_private::DWARFCallFrameInfo *
    GetEHFrameInfo ();

    lldb::FuncUnwindersSP
    GetFuncUnwindersContainingAddress (const Address& addr, SymbolContext &sc);

private:
    void
    Dump (Stream &s);
    
    void Initialize ();

    typedef std::map<lldb::addr_t, lldb::FuncUnwindersSP> collection;
    typedef collection::iterator iterator;
    typedef collection::const_iterator const_iterator;

    ObjectFile&         m_object_file;
    collection          m_unwinds;

    bool                m_initialized;  // delay some initialization until ObjectFile is set up

    UnwindAssemblyProfiler* m_assembly_profiler;

    DWARFCallFrameInfo* m_eh_frame;
};

} // namespace lldb_private

#endif  // liblldb_UnwindTable_h
