//===-- SWIG Interface for SBFunction ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a generic function, which can be inlined or not.

For example (from test/lldbutil.py, but slightly modified for doc purpose),

        ...

        frame = thread.GetFrameAtIndex(i)
        addr = frame.GetPCAddress()
        load_addr = addr.GetLoadAddress(target)
        function = frame.GetFunction()
        mod_name = frame.GetModule().GetFileSpec().GetFilename()

        if not function:
            # No debug info for 'function'.
            symbol = frame.GetSymbol()
            file_addr = addr.GetFileAddress()
            start_addr = symbol.GetStartAddress().GetFileAddress()
            symbol_name = symbol.GetName()
            symbol_offset = file_addr - start_addr
            print >> output, '  frame #{num}: {addr:#016x} {mod}`{symbol} + {offset}'.format(
                num=i, addr=load_addr, mod=mod_name, symbol=symbol_name, offset=symbol_offset)
        else:
            # Debug info is available for 'function'.
            func_name = frame.GetFunctionName()
            file_name = frame.GetLineEntry().GetFileSpec().GetFilename()
            line_num = frame.GetLineEntry().GetLine()
            print >> output, '  frame #{num}: {addr:#016x} {mod}`{func} at {file}:{line} {args}'.format(
                num=i, addr=load_addr, mod=mod_name,
                func='%s [inlined]' % func_name] if frame.IsInlined() else func_name,
                file=file_name, line=line_num, args=get_args_as_string(frame, showFuncName=False))

        ...
") SBFunction;
class SBFunction
{
public:

    SBFunction ();

    SBFunction (const lldb::SBFunction &rhs);

    ~SBFunction ();

    bool
    IsValid () const;

    const char *
    GetName() const;

    const char *
    GetMangledName () const;

    lldb::SBInstructionList
    GetInstructions (lldb::SBTarget target);

    lldb::SBAddress
    GetStartAddress ();

    lldb::SBAddress
    GetEndAddress ();

    uint32_t
    GetPrologueByteSize ();

    lldb::SBType
    GetType ();

    lldb::SBBlock
    GetBlock ();

    bool
    GetDescription (lldb::SBStream &description);
    
    %pythoncode %{
        def get_instructions_from_current_target (self):
            return self.GetInstructions (target)

        __swig_getmethods__["addr"] = GetStartAddress
        if _newclass: x = property(GetStartAddress, None)

        __swig_getmethods__["block"] = GetBlock
        if _newclass: x = property(GetBlock, None)

        __swig_getmethods__["end_addr"] = GetEndAddress
        if _newclass: x = property(GetEndAddress, None)
        
        __swig_getmethods__["instructions"] = get_instructions_from_current_target
        if _newclass: x = property(get_instructions_from_current_target, None)

        __swig_getmethods__["mangled"] = GetMangledName
        if _newclass: x = property(GetMangledName, None)

        __swig_getmethods__["name"] = GetName
        if _newclass: x = property(GetName, None)

        __swig_getmethods__["prologue_size"] = GetPrologueByteSize
        if _newclass: x = property(GetPrologueByteSize, None)

        __swig_getmethods__["type"] = GetType
        if _newclass: x = property(GetType, None)
    %}

};

} // namespace lldb
