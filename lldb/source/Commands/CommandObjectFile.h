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
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectFile
//-------------------------------------------------------------------------

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

    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter);

        virtual
        ~CommandOptions ();

        virtual Error
        SetOptionValue (int option_idx, const char *option_arg);

        void
        ResetOptionValues ();

        const OptionDefinition*
        GetDefinitions ();

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        ArchSpec m_arch;
    };
    
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
    CommandOptions m_options;

};

} // namespace lldb_private

#endif  // liblldb_CommandObjectFile_h_
