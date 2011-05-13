//===-- OptionGroupUUID.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupUUID.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb;
using namespace lldb_private;

OptionGroupUUID::OptionGroupUUID() :
    m_uuid ()
{
}

OptionGroupUUID::~OptionGroupUUID ()
{
}

static OptionDefinition
g_option_table[] =
{
{ LLDB_OPT_SET_1 , false, "uuid", 'u', required_argument, NULL, 0, eArgTypeNone, "A module UUID value."},
};

const uint32_t k_num_file_options = sizeof(g_option_table)/sizeof(OptionDefinition);

uint32_t
OptionGroupUUID::GetNumDefinitions ()
{
    return k_num_file_options;
}

const OptionDefinition *
OptionGroupUUID::GetDefinitions ()
{
    return g_option_table;
}

Error
OptionGroupUUID::SetOptionValue (CommandInterpreter &interpreter,
                                         uint32_t option_idx,
                                         const char *option_arg)
{
    Error error;
    char short_option = (char) g_option_table[option_idx].short_option;

    switch (short_option)
    {
        case 'u':
            error = m_uuid.SetValueFromCString (option_arg);
            break;

        default:
            error.SetErrorStringWithFormat ("Unrecognized option '%c'.\n", short_option);
            break;
    }

    return error;
}

void
OptionGroupUUID::OptionParsingStarting (CommandInterpreter &interpreter)
{
    m_uuid.Clear();
}
