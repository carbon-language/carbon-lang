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
#include "lldb/Core/ValueObject.h"
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
    
    bool
    AnyOptionWasSet () const
    {
        return show_types == true ||
               no_summary_depth  != 0 ||
               show_location == true ||
               flat_output == true ||
               use_objc == true ||
               max_depth != UINT32_MAX ||
               ptr_depth != 0 ||
               use_synth == false ||
               be_raw == true ||
               ignore_cap == true ||
               run_validator == true;
    }
    
    DumpValueObjectOptions
    GetAsDumpOptions (LanguageRuntimeDescriptionDisplayVerbosity lang_descr_verbosity = eLanguageRuntimeDescriptionDisplayVerbosityFull,
                      lldb::Format format = lldb::eFormatDefault,
                      lldb::TypeSummaryImplSP summary_sp = lldb::TypeSummaryImplSP());

    bool show_types : 1,
         show_location : 1,
         flat_output : 1,
         use_objc : 1,
         use_synth : 1,
         be_raw : 1,
         ignore_cap : 1,
         run_validator : 1;
    
    uint32_t no_summary_depth;
    uint32_t max_depth;
    uint32_t ptr_depth;
    lldb::DynamicValueType use_dynamic;
};

} // namespace lldb_private

#endif  // liblldb_OptionGroupValueObjectDisplay_h_
