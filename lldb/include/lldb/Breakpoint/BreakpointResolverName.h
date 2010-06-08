//===-- BreakpointResolverName.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_BreakpointResolverName_h_
#define liblldb_BreakpointResolverName_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointResolver.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class BreakpointResolverName BreakpointResolverName.h "lldb/Breakpoint/BreakpointResolverName.h"
/// @brief This class sets breakpoints on a given function name, either by exact match
/// or by regular expression.
//----------------------------------------------------------------------

class BreakpointResolverName:
    public BreakpointResolver
{
public:

    BreakpointResolverName (Breakpoint *bkpt,
                        const char *func_name,
                        Breakpoint::MatchType type = Breakpoint::Exact);

    // Creates a function breakpoint by regular expression.  Takes over control of the lifespan of func_regex.
    BreakpointResolverName (Breakpoint *bkpt,
                        RegularExpression &func_regex);

    BreakpointResolverName (Breakpoint *bkpt,
                        const char *class_name,
                        const char *method,
                        Breakpoint::MatchType type);

    virtual
    ~BreakpointResolverName ();

    virtual Searcher::CallbackReturn
    SearchCallback (SearchFilter &filter,
                    SymbolContext &context,
                    Address *addr,
                    bool containing);

    virtual Searcher::Depth
    GetDepth ();

    virtual void
    GetDescription (Stream *s);

    virtual void
    Dump (Stream *s) const;

protected:
    ConstString m_func_name;
    ConstString m_class_name;  // FIXME: Not used yet.  The idea would be to stop on methods of this class.
    RegularExpression m_regex;
    Breakpoint::MatchType m_match_type;

private:
    DISALLOW_COPY_AND_ASSIGN(BreakpointResolverName);
};

} // namespace lldb_private

#endif  // liblldb_BreakpointResolverName_h_
