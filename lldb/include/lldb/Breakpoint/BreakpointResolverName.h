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
                            const char *name,
                            uint32_t name_type_mask,
                            Breakpoint::MatchType type,
                            bool skip_prologue);

    // Creates a function breakpoint by regular expression.  Takes over control of the lifespan of func_regex.
    BreakpointResolverName (Breakpoint *bkpt,
                            RegularExpression &func_regex,
                            bool skip_prologue);

    BreakpointResolverName (Breakpoint *bkpt,
                            const char *class_name,
                            const char *method,
                            Breakpoint::MatchType type,
                            bool skip_prologue);

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

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const BreakpointResolverName *) { return true; }
    static inline bool classof(const BreakpointResolver *V) {
        return V->getResolverID() == BreakpointResolver::NameResolver;
    }

protected:
    ConstString m_func_name;
    // "m_basename_filter" is used to filter results after searching for
    // "m_func_name" first. This is used when we are asked to set a breakpoint 
    // at "foo::bar::baz" (C++ function in namespace or in a class). For 
    // "foo::bar::baz" we will place "baz" into m_func_name and search for all
    // matching basename and methods that match "baz", then we will filter the
    // results by checking if the demangled name contains "m_basename_filter"
    // which would be set to "foo::bar::baz".
    std::string m_basename_filter;  
    uint32_t m_func_name_type_mask;  // See FunctionNameType
    ConstString m_class_name;  // FIXME: Not used yet.  The idea would be to stop on methods of this class.
    RegularExpression m_regex;
    Breakpoint::MatchType m_match_type;
    bool m_skip_prologue;

private:
    DISALLOW_COPY_AND_ASSIGN(BreakpointResolverName);
};

} // namespace lldb_private

#endif  // liblldb_BreakpointResolverName_h_
