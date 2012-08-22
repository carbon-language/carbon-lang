//===-- OptionValueBoolean.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionValueBoolean.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Stream.h"
#include "lldb/Interpreter/Args.h"

using namespace lldb;
using namespace lldb_private;

void
OptionValueBoolean::DumpValue (const ExecutionContext *exe_ctx, Stream &strm, uint32_t dump_mask)
{
    if (dump_mask & eDumpOptionType)
        strm.Printf ("(%s)", GetTypeAsCString ());
//    if (dump_mask & eDumpOptionName)
//        DumpQualifiedName (strm);
    if (dump_mask & eDumpOptionValue)
    {
        if (dump_mask & eDumpOptionType)
            strm.PutCString (" = ");
        strm.PutCString (m_current_value ? "true" : "false");
    }
}

Error
OptionValueBoolean::SetValueFromCString (const char *value_cstr,
                                         VarSetOperationType op)
{
    Error error;
    switch (op)
    {
    case eVarSetOperationClear:
        Clear();
        break;

    case eVarSetOperationReplace:
    case eVarSetOperationAssign:
        {
            bool success = false;
            bool value = Args::StringToBoolean(value_cstr, false, &success);
            if (success)
            {
                m_value_was_set = true;
                m_current_value = value;
            }
            else
            {
                if (value_cstr == NULL)
                    error.SetErrorString ("invalid boolean string value: NULL");
                else if (value_cstr[0] == '\0')
                    error.SetErrorString ("invalid boolean string value <empty>");
                else
                    error.SetErrorStringWithFormat ("invalid boolean string value: '%s'", value_cstr);
            }
        }
        break;

    case eVarSetOperationInsertBefore:
    case eVarSetOperationInsertAfter:
    case eVarSetOperationRemove:
    case eVarSetOperationAppend:
    case eVarSetOperationInvalid:
        error = OptionValue::SetValueFromCString (value_cstr, op);
        break;
    }
    return error;
}

lldb::OptionValueSP
OptionValueBoolean::DeepCopy () const
{
    return OptionValueSP(new OptionValueBoolean(*this));
}


