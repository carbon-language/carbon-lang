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
    static const uint32_t OPTION_GROUP_FORMAT = LLDB_OPT_SET_1;
    static const uint32_t OPTION_GROUP_SIZE   = LLDB_OPT_SET_2;
    static const uint32_t OPTION_GROUP_COUNT  = LLDB_OPT_SET_3;
    
    OptionGroupFormat (lldb::Format default_format, 
                       uint64_t default_byte_size = UINT64_MAX,  // Pass UINT64_MAX to disable the "--size" option
                       uint64_t default_count = UINT64_MAX);     // Pass UINT64_MAX to disable the "--count" option
    
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

    OptionValueFormat &
    GetFormatValue()
    {
        return m_format;
    }
    
    const OptionValueFormat &
    GetFormatValue() const
    {
        return m_format;
    }
    
    OptionValueUInt64  &
    GetByteSizeValue()
    {
        return m_byte_size;
    }

    const OptionValueUInt64  &
    GetByteSizeValue() const 
    {
        return m_byte_size;
    }

    OptionValueUInt64  &
    GetCountValue()
    {
        return m_count;
    }

    const OptionValueUInt64  &
    GetCountValue() const
    {
        return m_count;
    }
    

protected:

    lldb::Format
    SetFormatUsingGDBFormatLetter (char format_letter);

    uint32_t
    SetByteSizeUsingGDBSizeLetter (char size_letter);

    OptionValueFormat m_format;
    OptionValueUInt64 m_byte_size;
    OptionValueUInt64 m_count;
    char m_prev_gdb_format;
    char m_prev_gdb_size;
};

} // namespace lldb_private

#endif  // liblldb_OptionGroupFormat_h_
