//====-- UserSettingsController.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <string.h>
#include <algorithm>

#include "lldb/Core/UserSettingsController.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/OptionValueString.h"

using namespace lldb;
using namespace lldb_private;

//static void
//DumpSettingEntry (CommandInterpreter &interpreter, 
//                  Stream &strm,
//                  const uint32_t max_len, 
//                  const SettingEntry &entry)
//{
//    StreamString description;
//
//    if (entry.description)
//        description.Printf ("%s", entry.description);
//    
//    if (entry.default_value && entry.default_value[0])
//        description.Printf (" (default: %s)", entry.default_value);
//    
//    interpreter.OutputFormattedHelpText (strm, 
//                                         entry.var_name, 
//                                         "--", 
//                                         description.GetData(), 
//                                         max_len);
//    
//    if (entry.enum_values && entry.enum_values[0].string_value)
//    {
//        interpreter.OutputFormattedHelpText (strm, 
//                                             "", 
//                                             "  ", 
//                                             "Enumeration values:", 
//                                             max_len);
//        for (uint32_t enum_idx=0; entry.enum_values[enum_idx].string_value != NULL; ++enum_idx)
//        {
//            description.Clear();
//            if (entry.enum_values[enum_idx].usage)
//                description.Printf ("%s = %s", 
//                                    entry.enum_values[enum_idx].string_value,
//                                    entry.enum_values[enum_idx].usage);
//            else
//                description.Printf ("%s", entry.enum_values[enum_idx].string_value);
//            interpreter.OutputFormattedHelpText (strm, 
//                                                 "", 
//                                                 "  ", 
//                                                 description.GetData(), 
//                                                 max_len);
//        }
//    }
//}

lldb::OptionValueSP
Properties::GetPropertyValue (const ExecutionContext *exe_ctx,
                              const char *path,
                              bool will_modify,
                              Error &error) const
{
    OptionValuePropertiesSP properties_sp (GetValueProperties ());
    if (properties_sp)
        return properties_sp->GetSubValue(exe_ctx, path, will_modify, error);
    return lldb::OptionValueSP();
}

Error
Properties::SetPropertyValue (const ExecutionContext *exe_ctx,
                              VarSetOperationType op,
                              const char *path,
                              const char *value)
{
    OptionValuePropertiesSP properties_sp (GetValueProperties ());
    if (properties_sp)
        return properties_sp->SetSubValue(exe_ctx, op, path, value);
    Error error;
    error.SetErrorString ("no properties");
    return error;
}

void
Properties::DumpAllPropertyValues (const ExecutionContext *exe_ctx, Stream &strm, uint32_t dump_mask)
{
    OptionValuePropertiesSP properties_sp (GetValueProperties ());
    if (properties_sp)
        return properties_sp->DumpValue (exe_ctx, strm, dump_mask);
}

void
Properties::DumpAllDescriptions (CommandInterpreter &interpreter,
                                 Stream &strm) const
{
    strm.PutCString("Top level variables:\n\n");

    OptionValuePropertiesSP properties_sp (GetValueProperties ());
    if (properties_sp)
        return properties_sp->DumpAllDescriptions (interpreter, strm);
}



Error
Properties::DumpPropertyValue (const ExecutionContext *exe_ctx, Stream &strm, const char *property_path, uint32_t dump_mask)
{
    OptionValuePropertiesSP properties_sp (GetValueProperties ());
    if (properties_sp)
    {
        return properties_sp->DumpPropertyValue (exe_ctx,
                                                 strm,
                                                 property_path,
                                                 dump_mask);
    }
    Error error;
    error.SetErrorString("empty property list");
    return error;
}

size_t
Properties::Apropos (const char *keyword, std::vector<const Property *> &matching_properties) const
{
    OptionValuePropertiesSP properties_sp (GetValueProperties ());
    if (properties_sp)
    {
        properties_sp->Apropos (keyword, matching_properties);
    }
    return matching_properties.size();
}
