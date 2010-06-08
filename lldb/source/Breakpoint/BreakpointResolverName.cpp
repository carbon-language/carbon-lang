//===-- BreakpointResolverName.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/BreakpointResolverName.h"

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

BreakpointResolverName::BreakpointResolverName
(
    Breakpoint *bkpt,
    const char *func_name,
    Breakpoint::MatchType type
) :
    BreakpointResolver (bkpt),
    m_func_name (func_name),
    m_class_name (NULL),
    m_regex (),
    m_match_type (type)
{
    if (m_match_type == Breakpoint::Regexp)
    {
        if (!m_regex.Compile (m_func_name.AsCString()))
        {
            Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS);

            if (log)
                log->Warning ("function name regexp: \"%s\" did not compile.", m_func_name.AsCString());
        }
    }
}

BreakpointResolverName::BreakpointResolverName
(
    Breakpoint *bkpt,
    RegularExpression &func_regex
) :
    BreakpointResolver (bkpt),
    m_func_name (NULL),
    m_class_name (NULL),
    m_regex (func_regex),
    m_match_type (Breakpoint::Regexp)
{

}

BreakpointResolverName::BreakpointResolverName
(
    Breakpoint *bkpt,
    const char *class_name,
    const char *method,
    Breakpoint::MatchType type
) :
    BreakpointResolver (bkpt),
    m_func_name (method),
    m_class_name (class_name),
    m_regex (),
    m_match_type (type)
{

}

BreakpointResolverName::~BreakpointResolverName ()
{
}

// FIXME: Right now we look at the module level, and call the module's "FindFunctions".
// Greg says he will add function tables, maybe at the CompileUnit level to accelerate function
// lookup.  At that point, we should switch the depth to CompileUnit, and look in these tables.

Searcher::CallbackReturn
BreakpointResolverName::SearchCallback
(
    SearchFilter &filter,
    SymbolContext &context,
    Address *addr,
    bool containing
)
{
    SymbolContextList func_list;
    SymbolContextList sym_list;

    bool skip_prologue = true;
    uint32_t i;
    bool new_location;
    SymbolContext sc;
    Address break_addr;
    assert (m_breakpoint != NULL);

    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS);

    if (m_class_name)
    {
        if (log)
            log->Warning ("Class/method function specification not supported yet.\n");
        return Searcher::eCallbackReturnStop;
    }

    switch (m_match_type)
    {
      case Breakpoint::Exact:
        if (context.module_sp)
        {
            context.module_sp->FindSymbolsWithNameAndType (m_func_name, eSymbolTypeCode, sym_list);
            context.module_sp->FindFunctions (m_func_name, false, func_list);
        }
        break;
      case Breakpoint::Regexp:
        if (context.module_sp)
        {
            context.module_sp->FindSymbolsMatchingRegExAndType (m_regex, eSymbolTypeCode, sym_list);
            context.module_sp->FindFunctions (m_regex, true, func_list);
        }
        break;
      case Breakpoint::Glob:
        if (log)
            log->Warning ("glob is not supported yet.");
        break;
    }

    // Remove any duplicates between the funcion list and the symbol list
    if (func_list.GetSize())
    {
        for (i = 0; i < func_list.GetSize(); i++)
        {
            if (func_list.GetContextAtIndex(i, sc) == false)
                continue;

            if (sc.function == NULL)
                continue;
            uint32_t j = 0;
            while (j < sym_list.GetSize())
            {
                SymbolContext symbol_sc;
                if (sym_list.GetContextAtIndex(j, symbol_sc))
                {
                    if (symbol_sc.symbol && symbol_sc.symbol->GetAddressRangePtr())
                    {
                        if (sc.function->GetAddressRange().GetBaseAddress() == symbol_sc.symbol->GetAddressRangePtr()->GetBaseAddress())
                        {
                            sym_list.RemoveContextAtIndex(j);
                            continue;   // Don't increment j
                        }
                    }
                }

                j++;
            }
        }

        for (i = 0; i < func_list.GetSize(); i++)
        {
            if (func_list.GetContextAtIndex(i, sc))
            {
                if (sc.function)
                {
                    break_addr = sc.function->GetAddressRange().GetBaseAddress();
                    if (skip_prologue)
                    {
                        const uint32_t prologue_byte_size = sc.function->GetPrologueByteSize();
                        if (prologue_byte_size)
                            break_addr.SetOffset(break_addr.GetOffset() + prologue_byte_size);
                    }

                    if (filter.AddressPasses(break_addr))
                    {
                        BreakpointLocationSP bp_loc_sp (m_breakpoint->AddLocation(break_addr, &new_location));
                        if (bp_loc_sp && new_location && !m_breakpoint->IsInternal())
                        {
                            if (log)
                            {
                                StreamString s;
                                bp_loc_sp->GetDescription(&s, lldb::eDescriptionLevelVerbose);
                                log->Printf ("Added location: %s\n", s.GetData());
                            }
                        }
                    }
                }
            }
        }
    }

    for (i = 0; i < sym_list.GetSize(); i++)
    {
        if (sym_list.GetContextAtIndex(i, sc))
        {
            if (sc.symbol && sc.symbol->GetAddressRangePtr())
            {
                break_addr = sc.symbol->GetAddressRangePtr()->GetBaseAddress();

                if (skip_prologue)
                {
                    const uint32_t prologue_byte_size = sc.symbol->GetPrologueByteSize();
                    if (prologue_byte_size)
                        break_addr.SetOffset(break_addr.GetOffset() + prologue_byte_size);
                }

                if (filter.AddressPasses(break_addr))
                {
                    BreakpointLocationSP bp_loc_sp (m_breakpoint->AddLocation(break_addr, &new_location));
                    if (bp_loc_sp && new_location && !m_breakpoint->IsInternal())
                    {
                        StreamString s;
                        bp_loc_sp->GetDescription(&s, lldb::eDescriptionLevelVerbose);
                        if (log) 
                            log->Printf ("Added location: %s\n", s.GetData());
                    }
                }
            }
        }
    }
    return Searcher::eCallbackReturnContinue;
}

Searcher::Depth
BreakpointResolverName::GetDepth()
{
    return Searcher::eDepthModule;
}

void
BreakpointResolverName::GetDescription (Stream *s)
{
    s->PutCString("Breakpoint by name: ");

    if (m_match_type == Breakpoint::Regexp)
        s->Printf("'%s' (regular expression)", m_regex.GetText());
    else
        s->Printf("'%s'", m_func_name.AsCString());
}

void
BreakpointResolverName::Dump (Stream *s) const
{

}

