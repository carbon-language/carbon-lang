//===-- CommandObjectPlatform.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectPlatform_h_
#define liblldb_CommandObjectPlatform_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/Options.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectPlatform
//-------------------------------------------------------------------------

class CommandObjectPlatform : public CommandObjectMultiword
{
public:
    CommandObjectPlatform(CommandInterpreter &interpreter);

    virtual
    ~CommandObjectPlatform();

    private:
    DISALLOW_COPY_AND_ASSIGN (CommandObjectPlatform);
};

    
//-------------------------------------------------------------------------
// PlatformOptionGroup
//
// Make platform options available to to any other command in case they 
// need them. The "file" command needs them, and by exposing them we can
// reuse the platform command options for any command, we can keep things
// consistent.
//-------------------------------------------------------------------------
class PlatformOptionGroup : public OptionGroup
{
public:
    
    PlatformOptionGroup (bool include_platform_option) :
        platform_name (),
        os_version_major (UINT32_MAX),
        os_version_minor (UINT32_MAX),
        os_version_update (UINT32_MAX),
        m_include_platform_option (include_platform_option)
    {
    }
    
    virtual
    ~PlatformOptionGroup ()
    {
    }
    
    virtual uint32_t
    GetNumDefinitions ();
    
    virtual const OptionDefinition*
    GetDefinitions ();
    
    virtual Error
    SetOptionValue (CommandInterpreter &interpreter,
                    uint32_t option_idx,
                    const char *option_value);
    
    lldb::PlatformSP 
    CreatePlatformWithOptions (CommandInterpreter &interpreter, 
                               bool select,
                               Error &error);

    virtual void
    OptionParsingStarting (CommandInterpreter &interpreter);
        
    // Instance variables to hold the values for command options.
    
    std::string platform_name;
    uint32_t os_version_major;
    uint32_t os_version_minor;
    uint32_t os_version_update;
protected:
    bool m_include_platform_option;
};

    
    
} // namespace lldb_private

#endif  // liblldb_CommandObjectPlatform_h_
