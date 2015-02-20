//===-- OptionValueBoolean.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionValueChar_h_
#define liblldb_OptionValueChar_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/OptionValue.h"

namespace lldb_private {

class OptionValueChar : public OptionValue
{
public:
    OptionValueChar (char value) :
        OptionValue(),
        m_current_value (value),
        m_default_value (value)
    {
    }
    OptionValueChar (char current_value,
                     char default_value) :
        OptionValue(),
        m_current_value (current_value),
        m_default_value (default_value)
    {
    }
    
    virtual 
    ~OptionValueChar()
    {
    }
    
    //---------------------------------------------------------------------
    // Virtual subclass pure virtual overrides
    //---------------------------------------------------------------------
    
    virtual OptionValue::Type
    GetType () const
    {
        return eTypeChar;
    }
    
    virtual void
    DumpValue (const ExecutionContext *exe_ctx, Stream &strm, uint32_t dump_mask);
    
    virtual Error
    SetValueFromString (llvm::StringRef value,
                         VarSetOperationType op = eVarSetOperationAssign);
    
    virtual bool
    Clear ()
    {
        m_current_value = m_default_value;
        m_value_was_set = false;
        return true;
    }

    //---------------------------------------------------------------------
    // Subclass specific functions
    //---------------------------------------------------------------------
    
    const char &
    operator = (char c)
    {
        m_current_value = c;
        return m_current_value;
    }

    char
    GetCurrentValue() const
    {
        return m_current_value;
    }
    
    char
    GetDefaultValue() const
    {
        return m_default_value;
    }
    
    void
    SetCurrentValue (char value)
    {
        m_current_value = value;
    }
    
    void
    SetDefaultValue (char value)
    {
        m_default_value = value;
    }
    
    virtual lldb::OptionValueSP
    DeepCopy () const;

protected:
    char m_current_value;
    char m_default_value;
};

} // namespace lldb_private

#endif  // liblldb_OptionValueChar_h_
