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

And,

    for frame in thread:
        print frame

See also SBThread."
) SBFrame;
class SBFrame
{
public:
    SBFrame ();

    SBFrame (const lldb::SBFrame &rhs);
    
   ~SBFrame();

    bool
    IsEqual (const lldb::SBFrame &rhs) const;

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

    lldb::SBValue
    EvaluateExpression (const char *expr, lldb::DynamicValueType use_dynamic, bool unwind_on_error);

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
    /// Get a lldb.SBValue for a variable path. 
    ///
    /// Variable paths can include access to pointer or instance members:
    ///     rect_ptr->origin.y
    ///     pt.x
    /// Pointer dereferences:
    ///     *this->foo_ptr
    ///     **argv
    /// Address of:
    ///     &pt
    ///     &my_array[3].x
    /// Array accesses and treating pointers as arrays:
    ///     int_array[1]
    ///     pt_ptr[22].x
    ///
    /// Unlike EvaluateExpression() which returns lldb.SBValue objects
    /// with constant copies of the values at the time of evaluation,
    /// the result of this function is a value that will continue to
    /// track the current value of the value as execution progresses
    /// in the current frame.
    ") GetValueForVariablePath;
    lldb::SBValue
    GetValueForVariablePath (const char *var_path);
             
    lldb::SBValue
    GetValueForVariablePath (const char *var_path, lldb::DynamicValueType use_dynamic);

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
    
    %pythoncode %{
        def get_all_variables(self):
            return self.GetVariables(True,True,True,True)

        def get_arguments(self):
            return self.GetVariables(True,False,False,False)

        def get_locals(self):
            return self.GetVariables(False,True,False,False)

        def get_statics(self):
            return self.GetVariables(False,False,True,False)

        def var(self, var_expr_path):
            '''Calls through to lldb.SBFrame.GetValueForVariablePath() and returns 
            a value that represents the variable expression path'''
            return self.GetValueForVariablePath(var_expr_path)

        __swig_getmethods__["pc"] = GetPC
        __swig_setmethods__["pc"] = SetPC
        if _newclass: x = property(GetPC, SetPC)

        __swig_getmethods__["addr"] = GetPCAddress
        if _newclass: x = property(GetPCAddress, None)

        __swig_getmethods__["fp"] = GetFP
        if _newclass: x = property(GetFP, None)

        __swig_getmethods__["sp"] = GetSP
        if _newclass: x = property(GetSP, None)

        __swig_getmethods__["module"] = GetModule
        if _newclass: x = property(GetModule, None)

        __swig_getmethods__["compile_unit"] = GetCompileUnit
        if _newclass: x = property(GetCompileUnit, None)

        __swig_getmethods__["function"] = GetFunction
        if _newclass: x = property(GetFunction, None)

        __swig_getmethods__["symbol"] = GetSymbol
        if _newclass: x = property(GetSymbol, None)

        __swig_getmethods__["block"] = GetBlock
        if _newclass: x = property(GetBlock, None)

        __swig_getmethods__["is_inlined"] = IsInlined
        if _newclass: x = property(IsInlined, None)

        __swig_getmethods__["name"] = GetFunctionName
        if _newclass: x = property(GetFunctionName, None)

        __swig_getmethods__["line_entry"] = GetLineEntry
        if _newclass: x = property(GetLineEntry, None)

        __swig_getmethods__["thread"] = GetThread
        if _newclass: x = property(GetThread, None)

        __swig_getmethods__["disassembly"] = Disassemble
        if _newclass: x = property(Disassemble, None)

        __swig_getmethods__["idx"] = GetFrameID
        if _newclass: x = property(GetFrameID, None)

        __swig_getmethods__["variables"] = get_all_variables
        if _newclass: x = property(get_all_variables, None)

        __swig_getmethods__["vars"] = get_all_variables
        if _newclass: x = property(get_all_variables, None)

        __swig_getmethods__["locals"] = get_locals
        if _newclass: x = property(get_locals, None)

        __swig_getmethods__["args"] = get_arguments
        if _newclass: x = property(get_arguments, None)

        __swig_getmethods__["arguments"] = get_arguments
        if _newclass: x = property(get_arguments, None)

        __swig_getmethods__["statics"] = get_statics
        if _newclass: x = property(get_statics, None)

        __swig_getmethods__["registers"] = GetRegisters
        if _newclass: x = property(GetRegisters, None)

        __swig_getmethods__["regs"] = GetRegisters
        if _newclass: x = property(GetRegisters, None)

    %}
};

} // namespace lldb
