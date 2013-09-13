//===-- OptionValueFileSpec.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/Interpreter/OptionValueFileSpec.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/State.h"
#include "lldb/DataFormatters/FormatManager.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandCompletions.h"

using namespace lldb;
using namespace lldb_private;


OptionValueFileSpec::OptionValueFileSpec () :
    OptionValue(),
    m_current_value (),
    m_default_value (),
    m_data_sp(),
    m_completion_mask (CommandCompletions::eDiskFileCompletion)
{
}

OptionValueFileSpec::OptionValueFileSpec (const FileSpec &value) :
    OptionValue(),
    m_current_value (value),
    m_default_value (value),
    m_data_sp(),
    m_completion_mask (CommandCompletions::eDiskFileCompletion)
{
}

OptionValueFileSpec::OptionValueFileSpec (const FileSpec &current_value,
                                          const FileSpec &default_value) :
    OptionValue(),
    m_current_value (current_value),
    m_default_value (default_value),
    m_data_sp(),
    m_completion_mask (CommandCompletions::eDiskFileCompletion)
{
}

void
OptionValueFileSpec::DumpValue (const ExecutionContext *exe_ctx, Stream &strm, uint32_t dump_mask)
{
    if (dump_mask & eDumpOptionType)
        strm.Printf ("(%s)", GetTypeAsCString ());
    if (dump_mask & eDumpOptionValue)
    {
        if (dump_mask & eDumpOptionType)
            strm.PutCString (" = ");

        if (m_current_value)
        {
            strm << '"' << m_current_value.GetPath().c_str() << '"';
        }
    }
}

Error
OptionValueFileSpec::SetValueFromCString (const char *value_cstr,
                                          VarSetOperationType op)
{
    Error error;
    switch (op)
    {
    case eVarSetOperationClear:
        Clear ();
        break;
        
    case eVarSetOperationReplace:
    case eVarSetOperationAssign:
        if (value_cstr && value_cstr[0])
        {
            // The setting value may have whitespace, double-quotes, or single-quotes around the file
            // path to indicate that internal spaces are not word breaks.  Strip off any ws & quotes
            // from the start and end of the file path - we aren't doing any word // breaking here so 
            // the quoting is unnecessary.  NB this will cause a problem if someone tries to specify
            // a file path that legitimately begins or ends with a " or ' character, or whitespace.
            std::string filepath(value_cstr);
            auto prefix_chars_to_trim = filepath.find_first_not_of ("\"' \t");
            if (prefix_chars_to_trim != std::string::npos && prefix_chars_to_trim > 0)
                filepath.erase(0, prefix_chars_to_trim);
            auto suffix_chars_to_trim = filepath.find_last_not_of ("\"' \t");
            if (suffix_chars_to_trim != std::string::npos && suffix_chars_to_trim < filepath.size())
                filepath.erase (suffix_chars_to_trim + 1);

            m_value_was_set = true;
            m_current_value.SetFile(filepath.c_str(), true);
        }
        else
        {
            error.SetErrorString("invalid value string");
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
OptionValueFileSpec::DeepCopy () const
{
    return OptionValueSP(new OptionValueFileSpec(*this));
}


size_t
OptionValueFileSpec::AutoComplete (CommandInterpreter &interpreter,
                                   const char *s,
                                   int match_start_point,
                                   int max_return_elements,
                                   bool &word_complete,
                                   StringList &matches)
{
    word_complete = false;
    matches.Clear();
    CommandCompletions::InvokeCommonCompletionCallbacks (interpreter,
                                                         m_completion_mask,
                                                         s,
                                                         match_start_point,
                                                         max_return_elements,
                                                         NULL,
                                                         word_complete,
                                                         matches);
    return matches.GetSize();
}



const lldb::DataBufferSP &
OptionValueFileSpec::GetFileContents(bool null_terminate)
{
    if (!m_data_sp && m_current_value)
    {
        if (null_terminate)
            m_data_sp = m_current_value.ReadFileContentsAsCString();
        else
            m_data_sp = m_current_value.ReadFileContents();
    }
    return m_data_sp;
}


