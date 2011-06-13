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

        CommandOptions (CommandInterpreter &interpreter);

        virtual
        ~CommandOptions ();

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg);

        void
        OptionParsingStarting ();

        const OptionDefinition*
        GetDefinitions ();

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];
        //Language  language;
        lldb::Encoding  encoding;
        lldb::Format    format;
        bool        debug;
        bool        print_object;
        LazyBool    use_dynamic;
        bool        unwind_on_error;
        bool        show_types;
        bool        show_summary;
    };

    CommandObjectExpression (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectExpression ();

    virtual
    Options *
    GetOptions ();


    virtual bool
    Execute (Args& command,
             CommandReturnObject &result);

    virtual bool
    WantsRawCommandString() { return true; }

    virtual bool
    ExecuteRawCommandString (const char *command,
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
                        Stream *output_stream,
                        Stream *error_stream,
                        CommandReturnObject *result = NULL);

    CommandOptions m_options;
    ExecutionContext m_exe_ctx;
    uint32_t m_expr_line_count;
    std::string m_expr_lines; // Multi-line expression support
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectExpression_h_
