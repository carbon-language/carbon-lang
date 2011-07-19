//===-- CommandObjectType.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectType_h_
#define liblldb_CommandObjectType_h_

// C Includes
// C++ Includes


// Other libraries and framework includes
// Project includes

#include "lldb/lldb-types.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/Options.h"

namespace lldb_private {

class ScriptAddOptions
{
    
public:
    
    bool m_skip_pointers;
    bool m_skip_references;
    bool m_cascade;
    StringList m_target_types;
    StringList m_user_source;
    
    bool m_no_children;
    bool m_no_value;
    bool m_one_liner;
    bool m_regex;
    
    bool m_is_system;
    
    ConstString* m_name;
    
    const char* m_category;
    
    ScriptAddOptions(bool sptr,
                     bool sref,
                     bool casc,
                     bool noch,
                     bool novl,
                     bool onel,
                     bool regx,
                     bool syst,
                     ConstString* name,
                     const char* catg) :
    m_skip_pointers(sptr),
    m_skip_references(sref),
    m_cascade(casc),
    m_target_types(),
    m_user_source(),
    m_no_children(noch),
    m_no_value(novl),
    m_one_liner(onel),
    m_regex(regx),
    m_is_system(syst),
    m_name(name),
    m_category(catg)
    {
    }
    
    typedef lldb::SharedPtr<ScriptAddOptions>::Type SharedPointer;
    
};
    
class CommandObjectType : public CommandObjectMultiword
{
public:
    CommandObjectType (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectType ();
};

class CommandObjectTypeSummaryAdd : public CommandObject
{
    
private:
    
    class CommandOptions : public Options
    {
    public:
        
        CommandOptions (CommandInterpreter &interpreter) :
        Options (interpreter)
        {
        }
        
        virtual
        ~CommandOptions (){}
        
        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg);
        
        void
        OptionParsingStarting ();
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        
        // Options table: Required for subclasses of Options.
        
        static OptionDefinition g_option_table[];
        
        // Instance variables to hold the values for command options.
        
        bool m_cascade;
        bool m_no_children;
        bool m_no_value;
        bool m_one_liner;
        bool m_skip_references;
        bool m_skip_pointers;
        bool m_regex;
        std::string m_format_string;
        ConstString* m_name;
        std::string m_python_script;
        std::string m_python_function;
        bool m_is_add_script;
        bool m_is_system;
        const char* m_category;
    };
    
    CommandOptions m_options;
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }
    
    void
    CollectPythonScript(ScriptAddOptions *options,
                        CommandReturnObject &result);
    
    bool
    Execute_ScriptSummary (Args& command, CommandReturnObject &result);

    bool
    Execute_StringSummary (Args& command, CommandReturnObject &result);
    
public:
    
    enum SummaryFormatType
    {
        eRegularSummary,
        eRegexSummary,
        eNamedSummary,
    };

    CommandObjectTypeSummaryAdd (CommandInterpreter &interpreter);
    
    ~CommandObjectTypeSummaryAdd ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result);
    
    static bool
    AddSummary(const ConstString& type_name,
               lldb::SummaryFormatSP entry,
               SummaryFormatType type,
               const char* category,
               Error* error = NULL);
};


} // namespace lldb_private

#endif  // liblldb_CommandObjectType_h_
