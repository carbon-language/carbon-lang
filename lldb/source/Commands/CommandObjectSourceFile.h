//===-- CommandObjectSourceFile.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectSourceFile_h_
#define liblldb_CommandObjectSourceFile_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Core/Options.h"
#include "lldb/Core/FileSpec.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectSourceFile
//-------------------------------------------------------------------------

class CommandObjectSourceFile : public CommandObject
{
public:
    class CommandOptions : public Options
    {
    public:

        CommandOptions ();

        virtual
        ~CommandOptions ();

        virtual Error
        SetOptionValue (int option_idx, const char *option_arg);

        void
        ResetOptionValues ();

        const lldb::OptionDefinition*
        GetDefinitions ();

        // Options table: Required for subclasses of Options.

        static lldb::OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
        FileSpec file_spec;
        std::string file_name;
        uint32_t start_line;
        uint32_t num_lines;
    };

    CommandObjectSourceFile ();

    virtual
    ~CommandObjectSourceFile ();

    virtual bool
    Execute (Args& command,
             CommandContext *context,
             CommandInterpreter *interpreter,
             CommandReturnObject &result);

    virtual
    Options *
    GetOptions ();

protected:
    CommandOptions m_options;
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectSourceFile_h_
