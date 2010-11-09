//===-- UnwindTable.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/UnwindTable.h"

#include <stdio.h>

#include "lldb/lldb-forward.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/FuncUnwinders.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/DWARFCallFrameInfo.h"
#include "lldb/Utility/UnwindAssemblyProfiler.h"

// There is one UnwindTable object per ObjectFile.
// It contains a list of Unwind objects -- one per function, populated lazily -- for the ObjectFile.
// Each Unwind object has multiple UnwindPlans for different scenarios.

using namespace lldb;
using namespace lldb_private;

UnwindTable::UnwindTable (ObjectFile& objfile) : m_object_file(objfile), 
                                                 m_unwinds(),
                                                 m_initialized(false),
                                                 m_eh_frame(NULL),
                                                 m_assembly_profiler(NULL)
{
}

// We can't do some of this initialization when the ObjectFile is running its ctor; delay doing it
// until needed for something.

void
UnwindTable::initialize ()
{
    if (m_initialized)
        return;

    SectionList* sl = m_object_file.GetSectionList ();
    if (sl)
    {
        SectionSP sect = sl->FindSectionByType (eSectionTypeEHFrame, true);
        if (sect.get())
        {
            m_eh_frame = new DWARFCallFrameInfo(m_object_file, sect, eRegisterKindGCC, true);
        }
    }
    
    ArchSpec arch;
    ConstString str;
    m_object_file.GetTargetTriple (str);
    arch.SetArchFromTargetTriple (str.GetCString());
    m_assembly_profiler = UnwindAssemblyProfiler::FindPlugin (arch);

    m_initialized = true;
}

UnwindTable::~UnwindTable ()
{
    if (m_eh_frame)
        delete m_eh_frame;
}

FuncUnwindersSP
UnwindTable::GetFuncUnwindersContainingAddress (const Address& addr, SymbolContext &sc)
{
    FuncUnwindersSP no_unwind_found;

    initialize();

    // Create a FuncUnwinders object for the binary search below
    AddressRange search_range(addr, 1);
    FuncUnwindersSP search_unwind(new FuncUnwinders (*this, NULL, search_range));

    const_iterator idx;
    idx = std::lower_bound (m_unwinds.begin(), m_unwinds.end(), search_unwind);

    bool found_match = true;
    if (m_unwinds.size() == 0)
    {
        found_match = false;
    }
    else if (idx == m_unwinds.end())
    {
        --idx;
    }
    if (idx != m_unwinds.begin() && (*idx)->GetFunctionStartAddress().GetOffset() != addr.GetOffset())
    {
       --idx;
    }
    if (found_match && (*idx)->ContainsAddress (addr))
    {
        return *idx;
    }

    AddressRange range;
    if (!sc.GetAddressRange(eSymbolContextFunction | eSymbolContextSymbol, range) || !range.GetBaseAddress().IsValid())
    {
        // Does the eh_frame unwind info has a function bounds for this addr?
        if (m_eh_frame == NULL || !m_eh_frame->GetAddressRange (addr, range))
        {
            return no_unwind_found;
        }
    }

    FuncUnwindersSP unw(new FuncUnwinders(*this, m_assembly_profiler, range));
    m_unwinds.push_back (unw);
    std::sort (m_unwinds.begin(), m_unwinds.end());
    return unw;
}

DWARFCallFrameInfo *
UnwindTable::GetEHFrameInfo ()
{
    initialize();
    return m_eh_frame;
}
