//===-- SWIG Interface for SBFrame ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents one of the stack frames associated with a thread.
SBThread contains SBFrame(s). For example (from test/lldbutil.py),

def print_stacktrace(thread, string_buffer = False):
    '''Prints a simple stack trace of this thread.'''

    ...

    for i in range(depth):
        frame = thread.GetFrameAtIndex(i)
        function = frame.GetFunction()

        load_addr = addrs[i].GetLoadAddress(target)
        if not function:
            file_addr = addrs[i].GetFileAddress()
            start_addr = frame.GetSymbol().GetStartAddress().GetFileAddress()
            symbol_offset = file_addr - start_addr
            print >> output, '  frame #{num}: {addr:#016x} {mod}`{symbol} + {offset}'.format(
                num=i, addr=load_addr, mod=mods[i], symbol=symbols[i], offset=symbol_offset)
        else:
            print >> output, '  frame #{num}: {addr:#016x} {mod}`{func} at {file}:{line} {args}'.format(
                num=i, addr=load_addr, mod=mods[i],
                func='%s [inlined]' % funcs[i] if frame.IsInlined() else funcs[i],
                file=files[i], line=lines[i],
                args=get_args_as_string(frame, showFuncName=False) if not frame.IsInlined() else '()')

    ...
"
) SBFrame;
class SBFrame
{
public:
    SBFrame ();

    SBFrame (const lldb::SBFrame &rhs);
    
   ~SBFrame();

    bool
    IsValid() const;

    uint32_t
    GetFrameID () const;

    lldb::addr_t
    GetPC () const;

    bool
    SetPC (lldb::addr_t new_pc);

    lldb::addr_t
    GetSP () const;

    lldb::addr_t
    GetFP () const;

    lldb::SBAddress
    GetPCAddress () const;

    lldb::SBSymbolContext
    GetSymbolContext (uint32_t resolve_scope) const;

    lldb::SBModule
    GetModule () const;

    lldb::SBCompileUnit
    GetCompileUnit () const;

    lldb::SBFunction
    GetFunction () const;

    lldb::SBSymbol
    GetSymbol () const;

    %feature("docstring", "
    /// Gets the deepest block that contains the frame PC.
    ///
    /// See also GetFrameBlock().
    ") GetBlock;
    lldb::SBBlock
    GetBlock () const;

    %feature("docstring", "
    /// Get the appropriate function name for this frame. Inlined functions in
    /// LLDB are represented by Blocks that have inlined function information, so
    /// just looking at the SBFunction or SBSymbol for a frame isn't enough.
    /// This function will return the appriopriate function, symbol or inlined
    /// function name for the frame.
    ///
    /// This function returns:
    /// - the name of the inlined function (if there is one)
    /// - the name of the concrete function (if there is one)
    /// - the name of the symbol (if there is one)
    /// - NULL
    ///
    /// See also IsInlined().
    ") GetFunctionName;
    const char *
    GetFunctionName();

    %feature("docstring", "
    /// Return true if this frame represents an inlined function.
    ///
    /// See also GetFunctionName().
    ") IsInlined;
    bool
    IsInlined();
    
    %feature("docstring", "
    /// The version that doesn't supply a 'use_dynamic' value will use the
    /// target's default.
    ") EvaluateExpression;
    lldb::SBValue
    EvaluateExpression (const char *expr);    

    lldb::SBValue
    EvaluateExpression (const char *expr, lldb::DynamicValueType use_dynamic);

    %feature("docstring", "
    /// Gets the lexical block that defines the stack frame. Another way to think
    /// of this is it will return the block that contains all of the variables
    /// for a stack frame. Inlined functions are represented as SBBlock objects
    /// that have inlined function information: the name of the inlined function,
    /// where it was called from. The block that is returned will be the first 
    /// block at or above the block for the PC (SBFrame::GetBlock()) that defines
    /// the scope of the frame. When a function contains no inlined functions,
    /// this will be the top most lexical block that defines the function. 
    /// When a function has inlined functions and the PC is currently
    /// in one of those inlined functions, this method will return the inlined
    /// block that defines this frame. If the PC isn't currently in an inlined
    /// function, the lexical block that defines the function is returned.
    ") GetFrameBlock;
    lldb::SBBlock
    GetFrameBlock () const;

    lldb::SBLineEntry
    GetLineEntry () const;

    lldb::SBThread
    GetThread () const;

    const char *
    Disassemble () const;

    void
    Clear();

#ifndef SWIG
    bool
    operator == (const lldb::SBFrame &rhs) const;

    bool
    operator != (const lldb::SBFrame &rhs) const;

#endif

    %feature("docstring", "
    /// The version that doesn't supply a 'use_dynamic' value will use the
    /// target's default.
    ") GetVariables;
    lldb::SBValueList
    GetVariables (bool arguments,
                  bool locals,
                  bool statics,
                  bool in_scope_only);

    lldb::SBValueList
    GetVariables (bool arguments,
                  bool locals,
                  bool statics,
                  bool in_scope_only,
                  lldb::DynamicValueType  use_dynamic);

    lldb::SBValueList
    GetRegisters ();

    %feature("docstring", "
    /// The version that doesn't supply a 'use_dynamic' value will use the
    /// target's default.
    ") FindVariable;
    lldb::SBValue
    FindVariable (const char *var_name);

    lldb::SBValue
    FindVariable (const char *var_name, lldb::DynamicValueType use_dynamic);

    %feature("docstring", "
    /// Find variables, register sets, registers, or persistent variables using
    /// the frame as the scope.
    ///
    /// The version that doesn't supply a 'use_dynamic' value will use the
    /// target's default.
    ") FindValue;
    lldb::SBValue
    FindValue (const char *name, ValueType value_type);

    lldb::SBValue
    FindValue (const char *name, ValueType value_type, lldb::DynamicValueType use_dynamic);

    bool
    GetDescription (lldb::SBStream &description);

};

} // namespace lldb
