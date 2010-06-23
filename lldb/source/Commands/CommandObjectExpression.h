//===-- CommandObjectExpression.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectExpression_h_
#define liblldb_CommandObjectExpression_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/Language.h"
#include "lldb/Target/ExecutionContext.h"

namespace lldb_private {

class CommandObjectExpression : public CommandObject
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
    };

    CommandObjectExpression ();

    virtual
    ~CommandObjectExpression ();

    virtual
    Options *
    GetOptions ();


    virtual bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result);

    virtual bool
    WantsRawCommandString() { return true; }

    virtual bool
    ExecuteRawCommandString (CommandInterpreter &interpreter,
                             const char *command,
                             CommandReturnObject &result);

protected:

    static size_t
    MultiLineExpressionCallback (void *baton, 
                                 InputReader &reader, 
                                 lldb::InputReaderAction notification,
                                 const char *bytes, 
                                 size_t bytes_len);

    bool
    EvaluateExpression (const char *expr, 
                        bool bare,
                        Stream &output_stream, 
                        Stream &error_stream);

    CommandOptions m_options;
    ExecutionContext m_exe_ctx;
    uint32_t m_expr_line_count;
    std::string m_expr_lines; // Multi-line expression support
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectExpression_h_
