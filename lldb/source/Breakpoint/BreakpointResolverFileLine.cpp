//===-- BreakpointResolverFileLine.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/BreakpointResolverFileLine.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/lldb-private-log.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// BreakpointResolverFileLine:
//----------------------------------------------------------------------
BreakpointResolverFileLine::BreakpointResolverFileLine
(
    Breakpoint *bkpt,
    const FileSpec &file_spec,
    uint32_t line_no,
    bool check_inlines
) :
    BreakpointResolver (bkpt, BreakpointResolver::FileLineResolver),
    m_file_spec (file_spec),
    m_line_number (line_no),
    m_inlines (check_inlines)
{
}

BreakpointResolverFileLine::~BreakpointResolverFileLine ()
{
}

Searcher::CallbackReturn
BreakpointResolverFileLine::SearchCallback
(
    SearchFilter &filter,
    SymbolContext &context,
    Address *addr,
    bool containing
)
{
    SymbolContextList sc_list;
    uint32_t sc_list_size;
    CompileUnit *cu = context.comp_unit;

    assert (m_breakpoint != NULL);
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));

    sc_list_size = cu->ResolveSymbolContext (m_file_spec, m_line_number, m_inlines, false, eSymbolContextEverything, sc_list);
    for (uint32_t i = 0; i < sc_list_size; i++)
    {
        SymbolContext sc;
        if (sc_list.GetContextAtIndex(i, sc))
        {
            Address line_start = sc.line_entry.range.GetBaseAddress();
            if (line_start.IsValid())
            {
                BreakpointLocationSP bp_loc_sp (m_breakpoint->AddLocation(line_start));
                if (log && bp_loc_sp && !m_breakpoint->IsInternal())
                {
                    StreamString s;
                    bp_loc_sp->GetDescription (&s, lldb::eDescriptionLevelVerbose);
                    log->Printf ("Added location: %s\n", s.GetData());
                }
            }
            else
            {
                if (log)
                    log->Printf ("error: Unable to set breakpoint at file address 0x%llx for %s:%d\n",
                                 line_start.GetFileAddress(),
                                 m_file_spec.GetFilename().AsCString("<Unknown>"),
                                 m_line_number);
            }
        }
        else
        {
#if 0
            s << "error: Breakpoint at '" << pos->c_str() << "' isn't resolved yet: \n";
            if (sc.line_entry.address.Dump(&s, Address::DumpStyleSectionNameOffset))
                s.EOL();
            if (sc.line_entry.address.Dump(&s, Address::DumpStyleSectionPointerOffset))
                s.EOL();
            if (sc.line_entry.address.Dump(&s, Address::DumpStyleFileAddress))
                s.EOL();
            if (sc.line_entry.address.Dump(&s, Address::DumpStyleLoadAddress))
                s.EOL();
#endif
        }
    }
    return Searcher::eCallbackReturnContinue;
}

Searcher::Depth
BreakpointResolverFileLine::GetDepth()
{
    return Searcher::eDepthCompUnit;
}

void
BreakpointResolverFileLine::GetDescription (Stream *s)
{
    s->Printf ("file ='%s', line = %u", m_file_spec.GetFilename().AsCString(), m_line_number);
}

void
BreakpointResolverFileLine::Dump (Stream *s) const
{

}

