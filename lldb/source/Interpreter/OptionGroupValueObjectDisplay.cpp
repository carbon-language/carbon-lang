//===-- OptionGroupValueObjectDisplay.cpp -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OptionGroupValueObjectDisplay.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb;
using namespace lldb_private;

OptionGroupValueObjectDisplay::OptionGroupValueObjectDisplay()
{
}

OptionGroupValueObjectDisplay::~OptionGroupValueObjectDisplay ()
{
}

static OptionDefinition
g_option_table[] =
{
    { LLDB_OPT_SET_1, false, "depth",           'D', required_argument, NULL, 0, eArgTypeCount,     "Set the max recurse depth when dumping aggregate types (default is infinity)."},
    { LLDB_OPT_SET_1, false, "flat",            'F', no_argument,       NULL, 0, eArgTypeNone,      "Display results in a flat format that uses expression paths for each variable or member."},
    { LLDB_OPT_SET_1, false, "location",        'L', no_argument,       NULL, 0, eArgTypeNone,      "Show variable location information."},
    { LLDB_OPT_SET_1, false, "objc",            'O', no_argument,       NULL, 0, eArgTypeNone,      "Print as an Objective-C object."},
    { LLDB_OPT_SET_1, false, "ptr-depth",       'P', required_argument, NULL, 0, eArgTypeCount,     "The number of pointers to be traversed when dumping values (default is zero)."},
    { LLDB_OPT_SET_1, false, "show-types",      'T', no_argument,       NULL, 0, eArgTypeNone,      "Show variable types when dumping values."},
    { LLDB_OPT_SET_1, false, "no-summary",      'Y', no_argument,       NULL, 0, eArgTypeNone,      "Omit summary information."},
    { 0, false, NULL, 0, 0, NULL, NULL, eArgTypeNone, NULL }
};

const uint32_t k_num_file_options = sizeof(g_option_table)/sizeof(OptionDefinition);

uint32_t
OptionGroupValueObjectDisplay::GetNumDefinitions ()
{
    return k_num_file_options;
}

const OptionDefinition *
OptionGroupValueObjectDisplay::GetDefinitions ()
{
    return g_option_table;
}


Error
OptionGroupValueObjectDisplay::SetOptionValue (CommandInterpreter &interpreter,
                                               uint32_t option_idx,
                                               const char *option_arg)
{
    Error error;
    char short_option = (char) g_option_table[option_idx].short_option;
    bool success = false;

    switch (short_option)
    {
        case 'T':   show_types   = true;  break;
        case 'Y':   show_summary = false; break;
        case 'L':   show_location= true;  break;
        case 'F':   flat_output  = true;  break;
        case 'O':   use_objc = true;      break;
        case 'D':
            max_depth = Args::StringToUInt32 (option_arg, UINT32_MAX, 0, &success);
            if (!success)
                error.SetErrorStringWithFormat("Invalid max depth '%s'.\n", option_arg);
            break;
            
        case 'P':
            ptr_depth = Args::StringToUInt32 (option_arg, 0, 0, &success);
            if (!success)
                error.SetErrorStringWithFormat("Invalid pointer depth '%s'.\n", option_arg);
            break;
            
        default:
            error.SetErrorStringWithFormat ("Unrecognized option '%c'.\n", short_option);
            break;
    }

    return error;
}

void
OptionGroupValueObjectDisplay::OptionParsingStarting (CommandInterpreter &interpreter)
{
    show_types    = false;
    show_summary  = true;
    show_location = false;
    flat_output   = false;
    use_objc      = false;
    max_depth     = UINT32_MAX;
    ptr_depth     = 0;
}

