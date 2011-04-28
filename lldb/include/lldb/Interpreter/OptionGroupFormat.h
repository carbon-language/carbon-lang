//===-- OptionGroupFormat.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionGroupFormat_h_
#define liblldb_OptionGroupFormat_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/NamedOptionValue.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// OptionGroupFormat
//-------------------------------------------------------------------------

class OptionGroupFormat : public OptionGroup
{
public:
    
    OptionGroupFormat (lldb::Format default_format, 
                       uint32_t default_byte_size,
                       bool byte_size_prefix_ok);
    
    virtual
    ~OptionGroupFormat ();

    
    virtual uint32_t
    GetNumDefinitions ();
    
    virtual const OptionDefinition*
    GetDefinitions ();
    
    virtual Error
    SetOptionValue (CommandInterpreter &interpreter,
                    uint32_t option_idx,
                    const char *option_value);
    
    virtual void
    OptionParsingStarting (CommandInterpreter &interpreter);
    
    lldb::Format
    GetFormat () const
    {
        return m_format.GetCurrentValue();
    }

    uint32_t
    GetByteSize() const
    {
        return m_format.GetCurrentByteSize();
    }
    
protected:

    OptionValueFormat m_format;
};

} // namespace lldb_private

#endif  // liblldb_OptionGroupFormat_h_
