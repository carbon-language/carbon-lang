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

#ifdef SWIG
    %feature("docstring",
             "Does this block represent an inlined function?"
             ) IsInlined;
#endif
    bool
    IsInlined () const;

    bool
    IsValid () const;

#ifdef SWIG
    %feature("autodoc",
             "Get the function name if this block represents an inlined function;\n"
             "otherwise, return None."
             ) GetInlinedName;
#endif
    const char *
    GetInlinedName () const;

#ifdef SWIG
    %feature("docstring",
             "Get the call site file if this block represents an inlined function;\n"
             "otherwise, return an invalid file spec."
             ) GetInlinedCallSiteFile;
#endif
    lldb::SBFileSpec
    GetInlinedCallSiteFile () const;

#ifdef SWIG
    %feature("docstring",
             "Get the call site line if this block represents an inlined function;\n"
             "otherwise, return 0."
             ) GetInlinedCallSiteLine;
#endif
    uint32_t 
    GetInlinedCallSiteLine () const;

#ifdef SWIG
    %feature("docstring",
             "Get the call site column if this block represents an inlined function;\n"
             "otherwise, return 0."
             ) GetInlinedCallSiteColumn;
#endif
    uint32_t
    GetInlinedCallSiteColumn () const;

#ifdef SWIG
    %feature("docstring", "Get the parent block.") GetParent;
#endif
    lldb::SBBlock
    GetParent ();
    
#ifdef SWIG
    %feature("docstring", "Get the sibling block for this block.") GetSibling;
#endif
    lldb::SBBlock
    GetSibling ();
    
#ifdef SWIG
    %feature("docstring", "Get the first child block.") GetFirstChild;
#endif
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
