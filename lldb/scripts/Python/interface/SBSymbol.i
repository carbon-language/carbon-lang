//===-- SWIG Interface for SBSymbol -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents the symbol possibly associated with a stack frame.
SBModule contains SBSymbol(s). SBSymbol can also be retrived from SBFrame.

See also SBModule and SBFrame."
) SBSymbol;
class SBSymbol
{
public:

    SBSymbol ();

    ~SBSymbol ();

    SBSymbol (const lldb::SBSymbol &rhs);

    bool
    IsValid () const;


    const char *
    GetName() const;

    const char *
    GetMangledName () const;

    lldb::SBInstructionList
    GetInstructions (lldb::SBTarget target);

    SBAddress
    GetStartAddress ();
    
    SBAddress
    GetEndAddress ();
    
    uint32_t
    GetPrologueByteSize ();

    SymbolType
    GetType ();

    bool
    GetDescription (lldb::SBStream &description);
};

} // namespace lldb
