//===-- OptionGroupValueObjectDisplay.cpp -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupValueObjectDisplay.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Target.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Utility/Utils.h"

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
    { LLDB_OPT_SET_1, false, "dynamic-type",     'd', required_argument, TargetInstanceSettings::g_dynamic_value_types, 
                                                                              0, eArgTypeNone,      "Show the object as its full dynamic type, not its static type, if available."},
    { LLDB_OPT_SET_1, false, "synthetic-type",   'S', required_argument, NULL, 0, eArgTypeBoolean,   "Show the object obeying its synthetic provider, if available."},
    { LLDB_OPT_SET_1, false, "depth",            'D', required_argument, NULL, 0, eArgTypeCount,     "Set the max recurse depth when dumping aggregate types (default is infinity)."},
    { LLDB_OPT_SET_1, false, "flat",             'F', no_argument,       NULL, 0, eArgTypeNone,      "Display results in a flat format that uses expression paths for each variable or member."},
    { LLDB_OPT_SET_1, false, "location",         'L', no_argument,       NULL, 0, eArgTypeNone,      "Show variable location information."},
    { LLDB_OPT_SET_1, false, "objc",             'O', no_argument,       NULL, 0, eArgTypeNone,      "Print as an Objective-C object."},
    { LLDB_OPT_SET_1, false, "ptr-depth",        'P', required_argument, NULL, 0, eArgTypeCount,     "The number of pointers to be traversed when dumping values (default is zero)."},
    { LLDB_OPT_SET_1, false, "show-types",       'T', no_argument,       NULL, 0, eArgTypeNone,      "Show variable types when dumping values."},
    { LLDB_OPT_SET_1, false, "no-summary-depth", 'Y', optional_argument, NULL, 0, eArgTypeCount,     "Set a depth for omitting summary information (default is 1)."},
    { LLDB_OPT_SET_1, false, "raw-output",       'R', no_argument,       NULL, 0, eArgTypeNone,      "Don't use formatting options."},
    { LLDB_OPT_SET_1, false, "show-all-children",'A', no_argument,       NULL, 0, eArgTypeNone,      "Ignore the upper bound on the number of children to show."},
    { 0, false, NULL, 0, 0, NULL, NULL, eArgTypeNone, NULL }
};

uint32_t
OptionGroupValueObjectDisplay::GetNumDefinitions ()
{
    return arraysize(g_option_table);
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
        case 'd':
            {
                bool success;
                int32_t result;
                result = Args::StringToOptionEnum (option_arg, TargetInstanceSettings::g_dynamic_value_types, 2, &success);
                if (!success)
                    error.SetErrorStringWithFormat("Invalid dynamic value setting: \"%s\".\n", option_arg);
                else
                    use_dynamic = (lldb::DynamicValueType) result;
            }
            break;
        case 'T':   show_types   = true;  break;
        case 'L':   show_location= true;  break;
        case 'F':   flat_output  = true;  break;
        case 'O':   use_objc     = true;  break;
        case 'R':   be_raw       = true;  break;
        case 'A':   ignore_cap   = true;  break;
            
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
            
        case 'Y':
            if (option_arg)
            {
                no_summary_depth = Args::StringToUInt32 (option_arg, 0, 0, &success);
                if (!success)
                    error.SetErrorStringWithFormat("Invalid pointer depth '%s'.\n", option_arg);
            }
            else
                no_summary_depth = 1;
            break;
            
        case 'S':
            use_synth = Args::StringToBoolean(option_arg, true, &success);
            if (!success)
                error.SetErrorStringWithFormat("Invalid synthetic-type '%s'.\n", option_arg);
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
    show_types        = false;
    no_summary_depth  = 0;
    show_location     = false;
    flat_output       = false;
    use_objc          = false;
    max_depth         = UINT32_MAX;
    ptr_depth         = 0;
    use_synth         = true;
    be_raw            = false;
    ignore_cap        = false;
    
    Target *target = interpreter.GetExecutionContext().target;
    if (target != NULL)
        use_dynamic = target->GetPreferDynamicValue();
    else
    {
        // If we don't have any targets, then dynamic values won't do us much good.
        use_dynamic = lldb::eNoDynamicValues;
    }
}
