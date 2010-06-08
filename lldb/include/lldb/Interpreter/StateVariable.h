//===-- StateVariable.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_InterpreterStateVariable_h_
#define liblldb_InterpreterStateVariable_h_

// C Includes
// C++ Includes
#include <string>
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/Args.h"

namespace lldb_private {

class StateVariable
{
public:

    // NOTE:  If you add more types to this enumeration list, you need to check for them and "do the right thing"
    // in CommandObjectSet::Execute.
    typedef enum
    {
        eTypeBoolean,
        eTypeInteger,
        eTypeString,
        eTypeStringArray
    } Type;


    typedef bool (*Callback) (CommandInterpreter *,
                              void *,
                              CommandReturnObject &);

    StateVariable (const char *name,
                   const char *value,
                   bool can_append = false,
                   const char *help_text = "",
                   Callback func_ptr = NULL);

    StateVariable (const char *name,
                   bool value,
                   const char *help_text = "",
                   Callback func_ptr = NULL);

    StateVariable (const char *name,
                   int value,
                   const char *help_text = "",
                   Callback func_ptr = NULL);

    StateVariable (const char *name,
                   const Args *value,
                   const char *help_text = "",
                   Callback func_ptr = NULL);

    virtual
    ~StateVariable ();


    const char *
    GetName () const;

    Type
    GetType () const;

    int
    GetIntValue () const;

    bool
    GetBoolValue () const;

    const char *
    GetStringValue () const;

    Args &
    GetArgs ();

    const Args &
    GetArgs () const;

    const char *
    GetHelp () const;

    void
    SetHelp (const char *);

    void
    AppendVariableInformation (CommandReturnObject &result);

    void
    SetStringValue (const char *);

    void
    SetIntValue (int);

    void
    SetBoolValue (bool);

    void
    ArrayAppendValue (const char *);

    void
    ArrayClearValues ();

    void
    AppendStringValue (const char *new_string);

    bool
    VerifyValue (CommandInterpreter *interpreter,
                 void *data,
                 CommandReturnObject &result);

    bool
    HasVerifyFunction ();

    static bool
    VerifyScriptLanguage (CommandInterpreter *interpreter,
                          void *data,
                          CommandReturnObject &result);

    static bool
    BroadcastPromptChange (CommandInterpreter *interpreter,
                           void *data,
                           CommandReturnObject &result);

private:
    std::string m_name;
    Type m_type;
    int m_int_value;
    Args m_string_values;
    std::string m_help_text;
    Callback m_verification_func_ptr;
};


} // namespace lldb_private

#endif  // liblldb_InterpreterStateVariable_h_
