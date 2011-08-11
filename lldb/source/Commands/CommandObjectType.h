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
    
    ConstString* m_name;
    
    std::string m_category;
    
    ScriptAddOptions(bool sptr,
                     bool sref,
                     bool casc,
                     bool noch,
                     bool novl,
                     bool onel,
                     bool regx,
                     ConstString* name,
                     std::string catg) :
    m_skip_pointers(sptr),
    m_skip_references(sref),
    m_cascade(casc),
    m_target_types(),
    m_user_source(),
    m_no_children(noch),
    m_no_value(novl),
    m_one_liner(onel),
    m_regex(regx),
    m_name(name),
    m_category(catg)
    {
    }
    
    typedef lldb::SharedPtr<ScriptAddOptions>::Type SharedPointer;
    
};
    
class SynthAddOptions
{
    
public:
    
    bool m_skip_pointers;
    bool m_skip_references;
    bool m_cascade;
    bool m_regex;
    StringList m_user_source;
    StringList m_target_types;

    std::string m_category;
    
    SynthAddOptions(bool sptr,
                     bool sref,
                     bool casc,
                     bool regx,
                     std::string catg) :
    m_skip_pointers(sptr),
    m_skip_references(sref),
    m_cascade(casc),
    m_regex(regx),
    m_user_source(),
    m_target_types(),
    m_category(catg)
    {
    }
    
    typedef lldb::SharedPtr<SynthAddOptions>::Type SharedPointer;
    
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
        std::string m_category;
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
               std::string category,
               Error* error = NULL);
};
    
class CommandObjectTypeSynthAdd : public CommandObject
{
    
private:
    
    class CommandOptions : public Options
    {
        typedef std::vector<std::string> option_vector;
    public:
        
        CommandOptions (CommandInterpreter &interpreter) :
        Options (interpreter)
        {
        }
        
        virtual
        ~CommandOptions (){}
        
        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;
            bool success;
            
            switch (short_option)
            {
                case 'C':
                    m_cascade = Args::StringToBoolean(option_arg, true, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("Invalid value for cascade: %s.\n", option_arg);
                    break;
                case 'c':
                    m_expr_paths.push_back(option_arg);
                    has_child_list = true;
                    break;
                case 'P':
                    handwrite_python = true;
                    break;
                case 'l':
                    m_class_name = std::string(option_arg);
                    is_class_based = true;
                    break;
                case 'p':
                    m_skip_pointers = true;
                    break;
                case 'r':
                    m_skip_references = true;
                    break;
                case 'w':
                    m_category = std::string(option_arg);
                    break;
                case 'x':
                    m_regex = true;
                    break;
                default:
                    error.SetErrorStringWithFormat ("Unrecognized option '%c'.\n", short_option);
                    break;
            }
            
            return error;
        }
        
        void
        OptionParsingStarting ()
        {
            m_cascade = true;
            m_class_name = "";
            m_skip_pointers = false;
            m_skip_references = false;
            m_category = "default";
            m_expr_paths.clear();
            is_class_based = false;
            handwrite_python = false;
            has_child_list = false;
            m_regex = false;
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        
        // Options table: Required for subclasses of Options.
        
        static OptionDefinition g_option_table[];
        
        // Instance variables to hold the values for command options.
        
        bool m_cascade;
        bool m_skip_references;
        bool m_skip_pointers;
        std::string m_class_name;
        bool m_input_python;
        option_vector m_expr_paths;
        std::string m_category;
        
        bool is_class_based;
        
        bool handwrite_python;
        
        bool has_child_list;
        
        bool m_regex;
        
        typedef option_vector::iterator ExpressionPathsIterator;
    };
    
    CommandOptions m_options;
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }
    
    void
    CollectPythonScript (SynthAddOptions *options,
                         CommandReturnObject &result);    
    bool
    Execute_HandwritePython (Args& command, CommandReturnObject &result);    
    bool
    Execute_ChildrenList (Args& command, CommandReturnObject &result);    
    bool
    Execute_PythonClass (Args& command, CommandReturnObject &result);
    bool
    Execute (Args& command, CommandReturnObject &result);
    
public:
    
    enum SynthFormatType
    {
        eRegularSynth,
        eRegexSynth,
    };
    
    CommandObjectTypeSynthAdd (CommandInterpreter &interpreter);
    
    ~CommandObjectTypeSynthAdd ()
    {
    }
    
    static bool
    AddSynth(const ConstString& type_name,
             lldb::SyntheticChildrenSP entry,
             SynthFormatType type,
             std::string category_name,
             Error* error);
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectType_h_
