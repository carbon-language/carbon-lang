//===-- CommandObjectFile.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectFile_h_
#define liblldb_CommandObjectFile_h_

// C Includes
// C++ Includes
#include <vector>
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Interpreter/CommandObject.h"
#include "CommandObjectPlatform.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectFile
//-------------------------------------------------------------------------

class FileOptionGroup : public OptionGroup
{
public:
    
    FileOptionGroup ();
    
    virtual
    ~FileOptionGroup ();

    
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
    GetArchitecture (Platform *platform, ArchSpec &arch);

    std::string m_arch_str; // Save the arch triple in case a platform is specified after the architecture
};

class CommandObjectFile : public CommandObject
{
public:

    CommandObjectFile (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectFile ();

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result);

    virtual Options *
    GetOptions ();

    
    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches);
    

private:
    OptionGroupOptions m_option_group;
    FileOptionGroup m_file_options;
    PlatformOptionGroup m_platform_options;
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectFile_h_
