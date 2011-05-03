//===-- OptionGroupBoolean.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionGroupBoolean_h_
#define liblldb_OptionGroupBoolean_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/NamedOptionValue.h"

namespace lldb_private {
    //-------------------------------------------------------------------------
    // OptionGroupBoolean
    //-------------------------------------------------------------------------
    
    class OptionGroupBoolean : public OptionGroup
    {
    public:
        
        OptionGroupBoolean (uint32_t usage_mask,
                            bool required,
                            const char *long_option, 
                            char short_option,
                            uint32_t completion_type,
                            lldb::CommandArgumentType argument_type,
                            const char *usage_text,
                            bool default_value);
        
        virtual
        ~OptionGroupBoolean ();
        
        
        virtual uint32_t
        GetNumDefinitions ()
        {
            return 1;
        }
        
        virtual const OptionDefinition*
        GetDefinitions ()
        {
            return &m_option_definition;
        }
        
        virtual Error
        SetOptionValue (CommandInterpreter &interpreter,
                        uint32_t option_idx,
                        const char *option_value);
        
        virtual void
        OptionParsingStarting (CommandInterpreter &interpreter);
        
        OptionValueBoolean &
        GetOptionValue ()
        {
            return m_value;
        }
        
        const OptionValueBoolean &
        GetOptionValue () const
        {
            return m_value;
        }
        
    protected:
        OptionValueBoolean m_value;
        OptionDefinition m_option_definition;
        
    };
    
} // namespace lldb_private

#endif  // liblldb_OptionGroupBoolean_h_
