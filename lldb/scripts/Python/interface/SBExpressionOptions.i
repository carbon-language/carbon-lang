//===-- SWIG interface for SBExpressionOptions -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"A container for options to use when evaluating expressions."
) SBExpressionOptions;

class SBExpressionOptions
{
friend class SBFrame;
friend class SBValue;

public:
    SBExpressionOptions();

    SBExpressionOptions (const lldb::SBExpressionOptions &rhs);
    
    SBExpressionOptions (bool coerce_to_id,
                         bool unwind_on_error,
                         bool keep_in_memory,
                         bool run_others,
                         DynamicValueType use_dynamic,
                         uint32_t timeout_usec);

    ~SBExpressionOptions();

    bool
    DoesCoerceToId () const;
    
    %feature("docstring",
    "Sets whether to coerce the expression result to ObjC id type after evaluation."
    ) SetCoerceToId;
    void
    SetCoerceToId (bool coerce = true);
    
    bool
    DoesUnwindOnError () const;
    
    %feature("docstring",
    "Sets whether to unwind the expression stack on error."
    ) SetUnwindOnError;
    void
    SetUnwindOnError (bool unwind = false);
    
    bool
    DoesKeepInMemory () const;
    
    %feature("docstring",
    "Sets whether to keep the expression result in the target program's memory - forced to true when creating SBValues."
    ) SetKeepInMemory;
    void
    SetKeepInMemory (bool keep = true);

    lldb::DynamicValueType
    GetUseDynamic () const;
    
    %feature("docstring",
    "Sets whether to cast the expression result to its dynamic type."
    ) SetUseDynamic;
    void
    SetUseDynamic (lldb::DynamicValueType dynamic = lldb::eDynamicCanRunTarget);
    
    uint32_t
    GetTimeoutUsec () const;
    
    %feature("docstring",
    "Sets the duration we will wait before cancelling expression evaluation.  0 means wait forever."
    ) SetTimeoutUsec;
    void
    SetTimeoutUsec (uint32_t timeout = 0);
    
    bool
    GetRunOthers () const;
    
    %feature("docstring",
    "Sets whether to run all threads if the expression does not complete on one thread."
    ) SetRunOthers;
    void
    SetRunOthers (bool run_others = true);

protected:

    SBExpressionOptions (lldb_private::EvaluateExpressionOptions &expression_options);

    lldb_private::EvaluateExpressionOptions *
    get () const;

    lldb_private::EvaluateExpressionOptions &
    ref () const;

private:
    // This auto_pointer is made in the constructor and is always valid.
    mutable std::auto_ptr<lldb_private::EvaluateExpressionOptions> m_opaque_ap;
};

} // namespace lldb
