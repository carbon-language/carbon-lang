//===-- OptionValueFormatEntity.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionValueFormatEntity.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Module.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StringList.h"
#include "lldb/Interpreter/CommandInterpreter.h"
using namespace lldb;
using namespace lldb_private;


OptionValueFormatEntity::OptionValueFormatEntity (const char *default_format) :
    OptionValue(),
    m_current_format (),
    m_default_format (),
    m_current_entry (),
    m_default_entry ()
{
    if (default_format && default_format[0])
    {
        llvm::StringRef default_format_str(default_format);
        Error error = FormatEntity::Parse(default_format_str, m_default_entry);
        if (error.Success())
        {
            m_default_format = default_format;
            m_current_format = default_format;
            m_current_entry = m_default_entry;
        }
    }
}

bool
OptionValueFormatEntity::Clear ()
{
    m_current_entry = m_default_entry;
    m_current_format = m_default_format;
    m_value_was_set = false;
    return true;
}


void
OptionValueFormatEntity::DumpValue (const ExecutionContext *exe_ctx, Stream &strm, uint32_t dump_mask)
{
    if (dump_mask & eDumpOptionType)
        strm.Printf ("(%s)", GetTypeAsCString ());
    if (dump_mask & eDumpOptionValue)
    {
        if (dump_mask & eDumpOptionType)
            strm.PutCString (" = \"");
        strm << m_current_format.c_str() << '"';
    }
}

Error
OptionValueFormatEntity::SetValueFromString (llvm::StringRef value_str,
                                      VarSetOperationType op)
{
    Error error;
    switch (op)
    {
        case eVarSetOperationClear:
            Clear();
            NotifyValueChanged();
            break;
            
        case eVarSetOperationReplace:
        case eVarSetOperationAssign:
            {
                FormatEntity::Entry entry;
                error = FormatEntity::Parse(value_str, entry);
                if (error.Success())
                {
                    m_current_entry = std::move(entry);
                    m_current_format = value_str;
                    m_value_was_set = true;
                    NotifyValueChanged();
                }
            }
            break;
            
        case eVarSetOperationInsertBefore:
        case eVarSetOperationInsertAfter:
        case eVarSetOperationRemove:
        case eVarSetOperationAppend:
        case eVarSetOperationInvalid:
            error = OptionValue::SetValueFromString (value_str, op);
            break;
    }
    return error;
}

lldb::OptionValueSP
OptionValueFormatEntity::DeepCopy () const
{
    return OptionValueSP(new OptionValueFormatEntity(*this));
}

size_t
OptionValueFormatEntity::AutoComplete (CommandInterpreter &interpreter,
                                       const char *s,
                                       int match_start_point,
                                       int max_return_elements,
                                       bool &word_complete,
                                       StringList &matches)
{
    return FormatEntity::AutoComplete (s, match_start_point, max_return_elements, word_complete, matches);
}

