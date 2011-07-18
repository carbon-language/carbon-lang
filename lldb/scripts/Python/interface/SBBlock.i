//===-- SWIG Interface for SBBlock ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a lexical block. SBFunction contains SBBlock(s)."
) SBBlock;
class SBBlock
{
public:

    SBBlock ();

    SBBlock (const lldb::SBBlock &rhs);

    ~SBBlock ();

    %feature("docstring",
    "Does this block represent an inlined function?"
    ) IsInlined;
    bool
    IsInlined () const;

    bool
    IsValid () const;

    %feature("docstring", "
    Get the function name if this block represents an inlined function;
    otherwise, return None.
    ") GetInlinedName;
    const char *
    GetInlinedName () const;

    %feature("docstring", "
    Get the call site file if this block represents an inlined function;
    otherwise, return an invalid file spec.
    ") GetInlinedCallSiteFile;
    lldb::SBFileSpec
    GetInlinedCallSiteFile () const;

    %feature("docstring", "
    Get the call site line if this block represents an inlined function;
    otherwise, return 0.
    ") GetInlinedCallSiteLine;
    uint32_t 
    GetInlinedCallSiteLine () const;

    %feature("docstring", "
    Get the call site column if this block represents an inlined function;
    otherwise, return 0.
    ") GetInlinedCallSiteColumn;
    uint32_t
    GetInlinedCallSiteColumn () const;

    %feature("docstring", "Get the parent block.") GetParent;
    lldb::SBBlock
    GetParent ();
    
    %feature("docstring", "Get the sibling block for this block.") GetSibling;
    lldb::SBBlock
    GetSibling ();
    
    %feature("docstring", "Get the first child block.") GetFirstChild;
    lldb::SBBlock
    GetFirstChild ();

    bool
    GetDescription (lldb::SBStream &description);
};

} // namespace lldb
