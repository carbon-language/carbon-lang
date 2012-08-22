//===-- OptionValueUUID.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionValueUUID.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Stream.h"

using namespace lldb;
using namespace lldb_private;

void
OptionValueUUID::DumpValue (const ExecutionContext *exe_ctx, Stream &strm, uint32_t dump_mask)
{
    if (dump_mask & eDumpOptionType)
        strm.Printf ("(%s)", GetTypeAsCString ());
    if (dump_mask & eDumpOptionValue)
    {
        if (dump_mask & eDumpOptionType)
            strm.PutCString (" = ");
        m_uuid.Dump (&strm);
    }
}

Error
OptionValueUUID::SetValueFromCString (const char *value_cstr,
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
                if (m_uuid.SetfromCString(value_cstr) == 0)
                    error.SetErrorStringWithFormat ("invalid uuid string value '%s'", value_cstr);
                else
                    m_value_was_set = true;
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
OptionValueUUID::DeepCopy () const
{
    return OptionValueSP(new OptionValueUUID(*this));
}
