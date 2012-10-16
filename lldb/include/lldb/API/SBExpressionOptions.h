//===-- SBEvent.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBExpressionOptions_h_
#define LLDB_SBExpressionOptions_h_

#include "lldb/API/SBDefines.h"

#include <memory>
#include <vector>

namespace lldb {


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

    const SBExpressionOptions &
    operator = (const lldb::SBExpressionOptions &rhs);

    bool
    DoesCoerceToId () const;
    
    void
    SetCoerceToId (bool coerce = true);
    
    bool
    DoesUnwindOnError () const;
    
    void
    SetUnwindOnError (bool unwind = false);
    
    bool
    DoesKeepInMemory () const;
    
    void
    SetKeepInMemory (bool keep = true);

    lldb::DynamicValueType
    GetUseDynamic () const;
    
    void
    SetUseDynamic (lldb::DynamicValueType dynamic = lldb::eDynamicCanRunTarget);
    
    uint32_t
    GetTimeoutUsec () const;
    
    void
    SetTimeoutUsec (uint32_t timeout = 0);
    
    bool
    GetRunOthers () const;
    
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

#endif  // LLDB_SBExpressionOptions_h_
