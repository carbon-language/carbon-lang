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

    bool
    IsExternal();

    bool
    IsSynthetic();

    %pythoncode %{
        def get_instructions_from_current_target (self):
            return self.GetInstructions (target)
        
        __swig_getmethods__["name"] = GetName
        if _newclass: x = property(GetName, None)
        
        __swig_getmethods__["mangled"] = GetMangledName
        if _newclass: x = property(GetMangledName, None)
        
        __swig_getmethods__["type"] = GetType
        if _newclass: x = property(GetType, None)
        
        __swig_getmethods__["addr"] = GetStartAddress
        if _newclass: x = property(GetStartAddress, None)
        
        __swig_getmethods__["end_addr"] = GetEndAddress
        if _newclass: x = property(GetEndAddress, None)
        
        __swig_getmethods__["prologue_size"] = GetPrologueByteSize
        if _newclass: x = property(GetPrologueByteSize, None)
        
        __swig_getmethods__["instructions"] = get_instructions_from_current_target
        if _newclass: x = property(get_instructions_from_current_target, None)

        __swig_getmethods__["external"] = IsExternal
        if _newclass: x = property(IsExternal, None)

        __swig_getmethods__["synthetic"] = IsSynthetic
        if _newclass: x = property(IsSynthetic, None)

        
    %}

};

} // namespace lldb
