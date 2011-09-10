//===-- OptionGroupFormat.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupFormat.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Utility/Utils.h"

using namespace lldb;
using namespace lldb_private;

OptionGroupFormat::OptionGroupFormat(lldb::Format default_format,
                                     uint32_t default_byte_size,
                                     bool byte_size_prefix_ok) :
    m_format (default_format, 
              default_format,
              default_byte_size,
              default_byte_size,
              byte_size_prefix_ok)
{
}

OptionGroupFormat::~OptionGroupFormat ()
{
}

static OptionDefinition 
g_option_table[] =
{
    { LLDB_OPT_SET_1 , false, "format", 'f', required_argument, NULL, 0, eArgTypeFormat , "Specify a format to be used for display."},
};

uint32_t
OptionGroupFormat::GetNumDefinitions ()
{
    return arraysize(g_option_table);
}

const OptionDefinition *
OptionGroupFormat::GetDefinitions ()
{
    return g_option_table;
}

Error
OptionGroupFormat::SetOptionValue (CommandInterpreter &interpreter,
                                   uint32_t option_idx,
                                   const char *option_arg)
{
    Error error;
    char short_option = (char) g_option_table[option_idx].short_option;

    switch (short_option)
    {
        case 'f':
            error = m_format.SetValueFromCString (option_arg);
            break;

        default:
            error.SetErrorStringWithFormat ("Unrecognized option '%c'.\n", short_option);
            break;
    }

    return error;
}

void
OptionGroupFormat::OptionParsingStarting (CommandInterpreter &interpreter)
{
    m_format.Clear();
}

