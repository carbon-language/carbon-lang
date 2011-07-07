//===-- OptionGroupVariable.cpp -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupVariable.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Target.h"
#include "lldb/Interpreter/CommandInterpreter.h"

using namespace lldb;
using namespace lldb_private;

static OptionDefinition
g_option_table[] =
{
    { LLDB_OPT_SET_1, false, "no-args",         'a', no_argument,       NULL, 0, eArgTypeNone,    "Omit function arguments."},
    { LLDB_OPT_SET_1, false, "no-locals",       'l', no_argument,       NULL, 0, eArgTypeNone,    "Omit local variables."},
    { LLDB_OPT_SET_1, false, "show-globals",    'g', no_argument,       NULL, 0, eArgTypeNone,    "Show the current frame source file global and static variables."},
    { LLDB_OPT_SET_1, false, "show-declaration",'c', no_argument,       NULL, 0, eArgTypeNone,    "Show variable declaration information (source file and line where the variable was declared)."},
    { LLDB_OPT_SET_1, false, "format",          'f', required_argument, NULL, 0, eArgTypeExprFormat,  "Specify the format that the variable output should use."},
    { LLDB_OPT_SET_1, false, "regex",           'r', no_argument,       NULL, 0, eArgTypeRegularExpression,    "The <variable-name> argument for name lookups are regular expressions."},
    { LLDB_OPT_SET_1, false, "scope",           's', no_argument,       NULL, 0, eArgTypeNone,    "Show variable scope (argument, local, global, static)."}
};


OptionGroupVariable::OptionGroupVariable (bool show_frame_options) :
    OptionGroup(),
    include_frame_options (show_frame_options)
{
}

OptionGroupVariable::~OptionGroupVariable ()
{
}

Error
OptionGroupVariable::SetOptionValue (CommandInterpreter &interpreter,
                                     uint32_t option_idx, 
                                     const char *option_arg)
{
    Error error;
    if (!include_frame_options)
        option_idx += 3;
    char short_option = (char) g_option_table[option_idx].short_option;
    switch (short_option)
    {
        case 'r':   use_regex    = true;  break;
        case 'a':   show_args    = false; break;
        case 'l':   show_locals  = false; break;
        case 'g':   show_globals = true;  break;
        case 'c':   show_decl    = true;  break;
        case 'f':   error = Args::StringToFormat(option_arg, format, NULL); break;
        case 's':
            show_scope = true;
            break;
            
        default:
            error.SetErrorStringWithFormat("Invalid short option character '%c'.\n", short_option);
            break;
    }
    
    return error;
}

void
OptionGroupVariable::OptionParsingStarting (CommandInterpreter &interpreter)
{
    show_args     = true;   // Frame option only
    show_locals   = true;   // Frame option only
    show_globals  = false;  // Frame option only
    show_decl     = false;
    format        = lldb::eFormatDefault;
    use_regex     = false;
    show_scope    = false;
}


const OptionDefinition*
OptionGroupVariable::GetDefinitions ()
{
    // Show the "--no-args", "--no-locals" and "--show-globals" 
    // options if we are showing frame specific options
    if (include_frame_options)
        return g_option_table;

    // Skip the "--no-args", "--no-locals" and "--show-globals" 
    // options if we are not showing frame specific options (globals only)
    return &g_option_table[3];
}

uint32_t
OptionGroupVariable::GetNumDefinitions ()
{
    if (include_frame_options)
        return 7;
    else
        return 4;
}


