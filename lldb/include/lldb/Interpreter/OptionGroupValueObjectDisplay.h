//===-- OptionGroupValueObjectDisplay.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionGroupValueObjectDisplay_h_
#define liblldb_OptionGroupValueObjectDisplay_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Options.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// OptionGroupValueObjectDisplay
//-------------------------------------------------------------------------

class OptionGroupValueObjectDisplay : public OptionGroup
{
public:
    
    OptionGroupValueObjectDisplay ();
    
    virtual
    ~OptionGroupValueObjectDisplay ();

    
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
    
    bool show_types;
    uint32_t no_summary_depth;
    bool show_location;
    bool flat_output;
    bool use_objc;
    uint32_t max_depth;
    uint32_t ptr_depth;
    lldb::DynamicValueType use_dynamic;
};

} // namespace lldb_private

#endif  // liblldb_OptionGroupValueObjectDisplay_h_
