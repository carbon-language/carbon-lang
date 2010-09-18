//===-- CommandObjectCall.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectCall_h_
#define liblldb_CommandObjectCall_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/Language.h"

namespace lldb_private {

class CommandObjectCall : public CommandObject
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
        Language  language;
        lldb::Encoding  encoding;
        lldb::Format    format;
        bool        debug;
        bool        show_types;
        bool        show_summary;
        bool        noexecute;
        bool        use_abi;
    };

    CommandObjectCall ();

    virtual
    ~CommandObjectCall ();

    virtual
    Options *
    GetOptions ();


    virtual bool
    Execute (Args& command,
             CommandReturnObject &result);

    virtual bool
    WantsRawCommandString() { return false; }

protected:

    CommandOptions m_options;
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectCall_h_
