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

OptionGroupFormat::OptionGroupFormat (lldb::Format default_format,
                                      uint64_t default_byte_size,
                                      uint64_t default_count) :
    m_format (default_format, default_format),
    m_byte_size (default_byte_size, default_byte_size),
    m_count (default_count, default_count)
{
}

OptionGroupFormat::~OptionGroupFormat ()
{
}

static OptionDefinition 
g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "format",'f', required_argument, NULL, 0, eArgTypeFormat  , "Specify a format to be used for display."},
{ LLDB_OPT_SET_2, false, "size"  ,'s', required_argument, NULL, 0, eArgTypeByteSize, "The size in bytes to use when displaying with the selected format."},
{ LLDB_OPT_SET_3, false, "count" ,'c', required_argument, NULL, 0, eArgTypeCount   , "The number of total items to display."},
};

uint32_t
OptionGroupFormat::GetNumDefinitions ()
{
    if (m_byte_size.GetDefaultValue() < UINT64_MAX)
    {
        if (m_count.GetDefaultValue() < UINT64_MAX)
            return 3;
        else
            return 2;
    }
    return 1;
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

        case 'c':
            if (m_count.GetDefaultValue() == 0)
            {
                error.SetErrorString ("--count option is disabled");
            }
            else
            {
                error = m_count.SetValueFromCString (option_arg);
                if (m_count.GetCurrentValue() == 0)
                    error.SetErrorStringWithFormat("invalid --count option value '%s'", option_arg);
            }
            break;
            
        case 's':
            if (m_byte_size.GetDefaultValue() == 0)
            {
                error.SetErrorString ("--size option is disabled");
            }
            else
            {
                error = m_byte_size.SetValueFromCString (option_arg);
                if (m_byte_size.GetCurrentValue() == 0)
                    error.SetErrorStringWithFormat("invalid --size option value '%s'", option_arg);
            }
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
    m_byte_size.Clear();
    m_count.Clear();
}
