//===-- OptionGroupWatchpoint.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupWatchpoint.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-enumerations.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Utility/Utils.h"

using namespace lldb;
using namespace lldb_private;

static OptionEnumValueElement g_watch_mode[] =
{
    { OptionGroupWatchpoint::eWatchRead,      "read",       "Watch for read"},
    { OptionGroupWatchpoint::eWatchWrite,     "write",      "Watch for write"},
    { OptionGroupWatchpoint::eWatchReadWrite, "read_write", "Watch for read/write"},
    { 0, NULL, NULL }
};

// if you add any options here, remember to update the counters in OptionGroupWatchpoint::GetNumDefinitions()
static OptionDefinition
g_option_table[] =
{
    { LLDB_OPT_SET_1, false, "watch", 'w', required_argument, g_watch_mode, 0, eArgTypeWatchMode, "Determine how to watch a memory location (read, write, or read/write)."}
};


OptionGroupWatchpoint::OptionGroupWatchpoint () :
    OptionGroup()
{
}

OptionGroupWatchpoint::~OptionGroupWatchpoint ()
{
}

Error
OptionGroupWatchpoint::SetOptionValue (CommandInterpreter &interpreter,
                                       uint32_t option_idx, 
                                       const char *option_arg)
{
    Error error;
    char short_option = (char) g_option_table[option_idx].short_option;
    switch (short_option)
    {
        case 'w': {
            watch_variable = false;
            OptionEnumValueElement *enum_values = g_option_table[option_idx].enum_values;
            watch_mode = (WatchMode) Args::StringToOptionEnum(option_arg, enum_values, 0, &watch_variable);
            break;
        }
        default:
            error.SetErrorStringWithFormat("Invalid short option character '%c'.\n", short_option);
            break;
    }
    
    return error;
}

void
OptionGroupWatchpoint::OptionParsingStarting (CommandInterpreter &interpreter)
{
    watch_variable = false;
    watch_mode     = eWatchInvalid;
}


const OptionDefinition*
OptionGroupWatchpoint::GetDefinitions ()
{
    return g_option_table;
}

uint32_t
OptionGroupWatchpoint::GetNumDefinitions ()
{
    return arraysize(g_option_table);
}
