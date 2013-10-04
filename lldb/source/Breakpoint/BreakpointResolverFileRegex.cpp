//===-- BreakpointResolverFileRegex.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/BreakpointResolverFileRegex.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Target/Target.h"
#include "lldb/lldb-private-log.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// BreakpointResolverFileRegex:
//----------------------------------------------------------------------
BreakpointResolverFileRegex::BreakpointResolverFileRegex
(
    Breakpoint *bkpt,
    RegularExpression &regex
) :
    BreakpointResolver (bkpt, BreakpointResolver::FileLineResolver),
    m_regex (regex)
{
}

BreakpointResolverFileRegex::~BreakpointResolverFileRegex ()
{
}

Searcher::CallbackReturn
BreakpointResolverFileRegex::SearchCallback
(
    SearchFilter &filter,
    SymbolContext &context,
    Address *addr,
    bool containing
)
{

    assert (m_breakpoint != NULL);
    if (!context.target_sp)
        return eCallbackReturnContinue;

    CompileUnit *cu = context.comp_unit;
    FileSpec cu_file_spec = *(static_cast<FileSpec *>(cu));
    std::vector<uint32_t> line_matches;
    context.target_sp->GetSourceManager().FindLinesMatchingRegex(cu_file_spec, m_regex, 1, UINT32_MAX, line_matches);
    
    uint32_t num_matches = line_matches.size();
    for (uint32_t i = 0; i < num_matches; i++)
    {
        SymbolContextList sc_list;
        const bool search_inlines = false;
        const bool exact = false;
        
        cu->ResolveSymbolContext (cu_file_spec, line_matches[i], search_inlines, exact, eSymbolContextEverything, sc_list);
        const bool skip_prologue = true;
        
        BreakpointResolver::SetSCMatchesByLine (filter, sc_list, skip_prologue, m_regex.GetText());
    }
    assert (m_breakpoint != NULL);        

    return Searcher::eCallbackReturnContinue;
}

Searcher::Depth
BreakpointResolverFileRegex::GetDepth()
{
    return Searcher::eDepthCompUnit;
}

void
BreakpointResolverFileRegex::GetDescription (Stream *s)
{
    s->Printf ("source regex = \"%s\"", m_regex.GetText());
}

void
BreakpointResolverFileRegex::Dump (Stream *s) const
{

}

