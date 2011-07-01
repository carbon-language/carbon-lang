//===-- SBBlock.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBBlock_h_
#define LLDB_SBBlock_h_

#include "lldb/API/SBDefines.h"

namespace lldb {

#ifdef SWIG
%feature("docstring",
         "Represents a lexical block. SBFunction contains SBBlock(s)."
         ) SBBlock;
#endif
class SBBlock
{
#ifdef SWIG
    %feature("autodoc", "1");
#endif
public:

    SBBlock ();

    SBBlock (const lldb::SBBlock &rhs);

    ~SBBlock ();

#ifndef SWIG
    const lldb::SBBlock &
    operator = (const lldb::SBBlock &rhs);
#endif

    bool
    IsInlined () const;

    bool
    IsValid () const;

    const char *
    GetInlinedName () const;

    lldb::SBFileSpec
    GetInlinedCallSiteFile () const;

    uint32_t 
    GetInlinedCallSiteLine () const;

    uint32_t
    GetInlinedCallSiteColumn () const;

    lldb::SBBlock
    GetParent ();
    
    lldb::SBBlock
    GetSibling ();
    
    lldb::SBBlock
    GetFirstChild ();

    bool
    GetDescription (lldb::SBStream &description);

private:
    friend class SBFrame;
    friend class SBSymbolContext;

#ifndef SWIG

    const lldb_private::Block *
    get () const;

    void
    reset (lldb_private::Block *lldb_object_ptr);

    SBBlock (lldb_private::Block *lldb_object_ptr);

    void
    AppendVariables (bool can_create, bool get_parent_variables, lldb_private::VariableList *var_list);

#endif

    lldb_private::Block *m_opaque_ptr;
};


} // namespace lldb

#endif // LLDB_SBBlock_h_
