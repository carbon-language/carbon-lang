//===-- CommandObjectFrame.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectFrame.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Host/Host.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/OptionGroupValueObjectDisplay.h"
#include "lldb/Interpreter/OptionGroupVariable.h"
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

#pragma mark CommandObjectFrameInfo

//-------------------------------------------------------------------------
// CommandObjectFrameInfo
//-------------------------------------------------------------------------

class CommandObjectFrameInfo : public CommandObject
{
public:

    CommandObjectFrameInfo (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "frame info",
                       "List information about the currently selected frame in the current thread.",
                       "frame info",
                       eFlagProcessMustBeLaunched | eFlagProcessMustBePaused)
    {
    }

    ~CommandObjectFrameInfo ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        ExecutionContext exe_ctx(m_interpreter.GetExecutionContext());
        if (exe_ctx.frame)
        {
            exe_ctx.frame->DumpUsingSettingsFormat (&result.GetOutputStream());
            result.GetOutputStream().EOL();
            result.SetStatus (eReturnStatusSuccessFinishResult);
        }
        else
        {
            result.AppendError ("no current frame");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

#pragma mark CommandObjectFrameSelect

//-------------------------------------------------------------------------
// CommandObjectFrameSelect
//-------------------------------------------------------------------------

class CommandObjectFrameSelect : public CommandObject
{
public:

   class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter) :
            Options(interpreter)
        {
            OptionParsingStarting ();
        }

        virtual
        ~CommandOptions ()
        {
        }

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            bool success = false;
            char short_option = (char) m_getopt_table[option_idx].val;
            switch (short_option)
            {
            case 'r':   
                relative_frame_offset = Args::StringToSInt32 (option_arg, INT32_MIN, 0, &success);
                if (!success)
                    error.SetErrorStringWithFormat ("invalid frame offset argument '%s'.\n", option_arg);
                break;

            default:
                error.SetErrorStringWithFormat ("Invalid short option character '%c'.\n", short_option);
                break;
            }

            return error;
        }

        void
        OptionParsingStarting ()
        {
            relative_frame_offset = INT32_MIN;
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];
        int32_t relative_frame_offset;
    };
    
    CommandObjectFrameSelect (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "frame select",
                       "Select a frame by index from within the current thread and make it the current frame.",
                       NULL,
                       eFlagProcessMustBeLaunched | eFlagProcessMustBePaused),
        m_options (interpreter)
    {
        CommandArgumentEntry arg;
        CommandArgumentData index_arg;

        // Define the first (and only) variant of this arg.
        index_arg.arg_type = eArgTypeFrameIndex;
        index_arg.arg_repetition = eArgRepeatOptional;

        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (index_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    ~CommandObjectFrameSelect ()
    {
    }

    virtual
    Options *
    GetOptions ()
    {
        return &m_options;
    }


    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        ExecutionContext exe_ctx (m_interpreter.GetExecutionContext());
        if (exe_ctx.thread)
        {
            const uint32_t num_frames = exe_ctx.thread->GetStackFrameCount();
            uint32_t frame_idx = UINT32_MAX;
            if (m_options.relative_frame_offset != INT32_MIN)
            {
                // The one and only argument is a signed relative frame index
                frame_idx = exe_ctx.thread->GetSelectedFrameIndex ();
                if (frame_idx == UINT32_MAX)
                    frame_idx = 0;
                
                if (m_options.relative_frame_offset < 0)
                {
                    if (frame_idx >= -m_options.relative_frame_offset)
                        frame_idx += m_options.relative_frame_offset;
                    else
                        frame_idx = 0;
                }
                else if (m_options.relative_frame_offset > 0)
                {
                    if (num_frames - frame_idx > m_options.relative_frame_offset)
                        frame_idx += m_options.relative_frame_offset;
                    else
                        frame_idx = num_frames - 1;
                }
            }
            else 
            {
                if (command.GetArgumentCount() == 1)
                {
                    const char *frame_idx_cstr = command.GetArgumentAtIndex(0);
                    frame_idx = Args::StringToUInt32 (frame_idx_cstr, UINT32_MAX, 0);
                }
                else
                {
                    result.AppendError ("invalid arguments.\n");
                    m_options.GenerateOptionUsage (result.GetErrorStream(), this);
                }
            }
                
            if (frame_idx < num_frames)
            {
                exe_ctx.thread->SetSelectedFrameByIndex (frame_idx);
                exe_ctx.frame = exe_ctx.thread->GetSelectedFrame ().get();

                if (exe_ctx.frame)
                {
                    bool already_shown = false;
                    SymbolContext frame_sc(exe_ctx.frame->GetSymbolContext(eSymbolContextLineEntry));
                    if (m_interpreter.GetDebugger().GetUseExternalEditor() && frame_sc.line_entry.file && frame_sc.line_entry.line != 0)
                    {
                        already_shown = Host::OpenFileInExternalEditor (frame_sc.line_entry.file, frame_sc.line_entry.line);
                    }

                    bool show_frame_info = true;
                    bool show_source = !already_shown;
                    uint32_t source_lines_before = 3;
                    uint32_t source_lines_after = 3;
                    if (exe_ctx.frame->GetStatus(result.GetOutputStream(),
                                                 show_frame_info,
                                                 show_source,
                                                 source_lines_before,
                                                 source_lines_after))
                    {
                        result.SetStatus (eReturnStatusSuccessFinishResult);
                        return result.Succeeded();
                    }
                }
            }
            result.AppendErrorWithFormat ("Frame index (%u) out of range.\n", frame_idx);
        }
        else
        {
            result.AppendError ("no current thread");
        }
        result.SetStatus (eReturnStatusFailed);
        return false;
    }
protected:

    CommandOptions m_options;
};

OptionDefinition
CommandObjectFrameSelect::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "relative", 'r', required_argument, NULL, 0, eArgTypeOffset, "A relative frame index offset from the current frame index."},
{ 0, false, NULL, 0, 0, NULL, NULL, eArgTypeNone, NULL }
};

#pragma mark CommandObjectFrameVariable
//----------------------------------------------------------------------
// List images with associated information
//----------------------------------------------------------------------
class CommandObjectFrameVariable : public CommandObject
{
public:

    class OptionGroupFrameVariable : public OptionGroup
    {
    public:

        OptionGroupFrameVariable ()
        {
        }

        virtual
        ~OptionGroupFrameVariable ()
        {
        }

        virtual uint32_t
        GetNumDefinitions ();

        virtual const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
    
        virtual Error
        SetOptionValue (CommandInterpreter &interpreter,
                        uint32_t option_idx, 
                        const char *option_arg)
        {
            Error error;
            char short_option = (char) g_option_table[option_idx].short_option;
            switch (short_option)
            {
            case 'r':   use_regex    = true;  break;
            case 'a':   show_args    = false; break;
            case 'l':   show_locals  = false; break;
            case 'g':   show_globals = true;  break;
            case 'c':   show_decl    = true;  break;
            case 'f':   error = Args::StringToFormat(option_arg, format, NULL); break;
            case 'G':
                globals.push_back(ConstString (option_arg));
                break;
            case 's':
                show_scope = true;
                break;

            default:
                error.SetErrorStringWithFormat("Invalid short option character '%c'.\n", short_option);
                break;
            }

            return error;
        }

        virtual void
        OptionParsingStarting (CommandInterpreter &interpreter)
        {
            show_args     = true;
            show_decl     = false;
            format        = eFormatDefault;
            show_globals  = false;
            show_locals   = true;
            use_regex     = false;
            show_scope    = false;
            globals.clear();
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];
        
        bool use_regex:1,
             show_args:1,
             show_locals:1,
             show_globals:1,
             show_scope:1,
             show_decl:1;
        lldb::Format format; // The format to use when dumping variables or children of variables
        std::vector<ConstString> globals;
        // Instance variables to hold the values for command options.
    };

    CommandObjectFrameVariable (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "frame variable",
                       "Show frame variables. All argument and local variables "
                       "that are in scope will be shown when no arguments are given. "
                       "If any arguments are specified, they can be names of "
                       "argument, local, file static and file global variables. "
                       "Children of aggregate variables can be specified such as "
                       "'var->child.x'.",
                       NULL,
                       eFlagProcessMustBeLaunched | eFlagProcessMustBePaused),
        m_option_group (interpreter),
        m_option_variable(true), // Include the frame specific options by passing "true"
        m_varobj_options()
    {
        CommandArgumentEntry arg;
        CommandArgumentData var_name_arg;
        
        // Define the first (and only) variant of this arg.
        var_name_arg.arg_type = eArgTypeVarName;
        var_name_arg.arg_repetition = eArgRepeatStar;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (var_name_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
        
        m_option_group.Append (&m_option_variable, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);
        m_option_group.Append (&m_varobj_options, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);
        m_option_group.Finalize();
    }

    virtual
    ~CommandObjectFrameVariable ()
    {
    }

    virtual
    Options *
    GetOptions ()
    {
        return &m_option_group;
    }


    virtual bool
    Execute
    (
        Args& command,
        CommandReturnObject &result
    )
    {
        ExecutionContext exe_ctx(m_interpreter.GetExecutionContext());
        if (exe_ctx.frame == NULL)
        {
            result.AppendError ("you must be stopped in a valid stack frame to view frame variables.");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else
        {
            Stream &s = result.GetOutputStream();

            bool get_file_globals = true;
            VariableList *variable_list = exe_ctx.frame->GetVariableList (get_file_globals);

            VariableSP var_sp;
            ValueObjectSP valobj_sp;

            const char *name_cstr = NULL;
            size_t idx;

            if (variable_list)
            {
                if (command.GetArgumentCount() > 0)
                {
                    VariableList regex_var_list;

                    // If we have any args to the variable command, we will make
                    // variable objects from them...
                    for (idx = 0; (name_cstr = command.GetArgumentAtIndex(idx)) != NULL; ++idx)
                    {
                        uint32_t ptr_depth = m_varobj_options.ptr_depth;
                        
                        if (m_option_variable.use_regex)
                        {
                            const uint32_t regex_start_index = regex_var_list.GetSize();
                            RegularExpression regex (name_cstr);
                            if (regex.Compile(name_cstr))
                            {
                                size_t num_matches = 0;
                                const size_t num_new_regex_vars = variable_list->AppendVariablesIfUnique(regex, 
                                                                                                         regex_var_list, 
                                                                                                         num_matches);
                                if (num_new_regex_vars > 0)
                                {
                                    for (uint32_t regex_idx = regex_start_index, end_index = regex_var_list.GetSize(); 
                                         regex_idx < end_index;
                                         ++regex_idx)
                                    {
                                        var_sp = regex_var_list.GetVariableAtIndex (regex_idx);
                                        if (var_sp)
                                        {
                                            valobj_sp = exe_ctx.frame->GetValueObjectForFrameVariable (var_sp, m_varobj_options.use_dynamic);
                                            if (valobj_sp)
                                            {                                        
                                                if (m_option_variable.format != eFormatDefault)
                                                    valobj_sp->SetFormat (m_option_variable.format);
                                                
                                                if (m_option_variable.show_decl && var_sp->GetDeclaration ().GetFile())
                                                {
                                                    var_sp->GetDeclaration ().DumpStopContext (&s, false);
                                                    s.PutCString (": ");
                                                }
                                                
                                                ValueObject::DumpValueObject (result.GetOutputStream(), 
                                                                              valobj_sp.get(), 
                                                                              var_sp->GetName().AsCString(), 
                                                                              m_varobj_options.ptr_depth, 
                                                                              0, 
                                                                              m_varobj_options.max_depth, 
                                                                              m_varobj_options.show_types,
                                                                              m_varobj_options.show_location,
                                                                              m_varobj_options.use_objc,
                                                                              m_varobj_options.use_dynamic, 
                                                                              false,
                                                                              m_varobj_options.flat_output);                                        
                                            }
                                        }
                                    }
                                }
                                else if (num_matches == 0)
                                {
                                    result.GetErrorStream().Printf ("error: no variables matched the regular expression '%s'.\n", name_cstr);
                                }
                            }
                            else
                            {
                                char regex_error[1024];
                                if (regex.GetErrorAsCString(regex_error, sizeof(regex_error)))
                                    result.GetErrorStream().Printf ("error: %s\n", regex_error);
                                else
                                    result.GetErrorStream().Printf ("error: unkown regex error when compiling '%s'\n", name_cstr);
                            }
                        }
                        else
                        {
                            Error error;
                            uint32_t expr_path_options = StackFrame::eExpressionPathOptionCheckPtrVsMember;
                            lldb::VariableSP var_sp;
                            valobj_sp = exe_ctx.frame->GetValueForVariableExpressionPath (name_cstr, 
                                                                                          m_varobj_options.use_dynamic, 
                                                                                          expr_path_options,
                                                                                          var_sp,
                                                                                          error);
                            if (valobj_sp)
                            {
                                if (m_option_variable.format != eFormatDefault)
                                    valobj_sp->SetFormat (m_option_variable.format);
                                if (m_option_variable.show_decl && var_sp && var_sp->GetDeclaration ().GetFile())
                                {
                                    var_sp->GetDeclaration ().DumpStopContext (&s, false);
                                    s.PutCString (": ");
                                }
                                ValueObject::DumpValueObject (result.GetOutputStream(), 
                                                              valobj_sp.get(), 
                                                              valobj_sp->GetParent() ? name_cstr : NULL, 
                                                              ptr_depth, 
                                                              0, 
                                                              m_varobj_options.max_depth, 
                                                              m_varobj_options.show_types,
                                                              m_varobj_options.show_location,
                                                              m_varobj_options.use_objc,
                                                              m_varobj_options.use_dynamic,
                                                              false,
                                                              m_varobj_options.flat_output);
                            }
                            else
                            {
                                const char *error_cstr = error.AsCString(NULL);
                                if (error_cstr)
                                    result.GetErrorStream().Printf("error: %s\n", error_cstr);
                                else
                                    result.GetErrorStream().Printf ("error: unable to find any variable expression path that matches '%s'\n", name_cstr);
                            }
                        }
                    }
                }
                else
                {
                    const uint32_t num_variables = variable_list->GetSize();
        
                    if (num_variables > 0)
                    {
                        for (uint32_t i=0; i<num_variables; i++)
                        {
                            var_sp = variable_list->GetVariableAtIndex(i);
                            
                            bool dump_variable = true;
                            
                            switch (var_sp->GetScope())
                            {
                            case eValueTypeVariableGlobal:
                                dump_variable = m_option_variable.show_globals;
                                if (dump_variable && m_option_variable.show_scope)
                                    s.PutCString("GLOBAL: ");
                                break;

                            case eValueTypeVariableStatic:
                                dump_variable = m_option_variable.show_globals;
                                if (dump_variable && m_option_variable.show_scope)
                                    s.PutCString("STATIC: ");
                                break;
                                
                            case eValueTypeVariableArgument:
                                dump_variable = m_option_variable.show_args;
                                if (dump_variable && m_option_variable.show_scope)
                                    s.PutCString("   ARG: ");
                                break;
                                
                            case eValueTypeVariableLocal:
                                dump_variable = m_option_variable.show_locals;
                                if (dump_variable && m_option_variable.show_scope)
                                    s.PutCString(" LOCAL: ");
                                break;

                            default:
                                break;
                            }
                            
                            if (dump_variable)
                            {

                                // Use the variable object code to make sure we are
                                // using the same APIs as the the public API will be
                                // using...
                                valobj_sp = exe_ctx.frame->GetValueObjectForFrameVariable (var_sp, 
                                                                                           m_varobj_options.use_dynamic);
                                if (valobj_sp)
                                {
                                    if (m_option_variable.format != eFormatDefault)
                                        valobj_sp->SetFormat (m_option_variable.format);
                                    
                                    // When dumping all variables, don't print any variables
                                    // that are not in scope to avoid extra unneeded output
                                    if (valobj_sp->IsInScope ())
                                    {
                                        if (m_option_variable.show_decl && var_sp->GetDeclaration ().GetFile())
                                        {
                                            var_sp->GetDeclaration ().DumpStopContext (&s, false);
                                            s.PutCString (": ");
                                        }
                                        ValueObject::DumpValueObject (result.GetOutputStream(), 
                                                                      valobj_sp.get(), 
                                                                      name_cstr, 
                                                                      m_varobj_options.ptr_depth, 
                                                                      0, 
                                                                      m_varobj_options.max_depth, 
                                                                      m_varobj_options.show_types,
                                                                      m_varobj_options.show_location,
                                                                      m_varobj_options.use_objc,
                                                                      m_varobj_options.use_dynamic, 
                                                                      false,
                                                                      m_varobj_options.flat_output);                                        
                                    }
                                }
                            }
                        }
                    }
                }
                result.SetStatus (eReturnStatusSuccessFinishResult);
            }
        }
        return result.Succeeded();
    }
protected:

    OptionGroupOptions m_option_group;
    OptionGroupVariable m_option_variable;
    OptionGroupValueObjectDisplay m_varobj_options;
};


#pragma mark CommandObjectMultiwordFrame

//-------------------------------------------------------------------------
// CommandObjectMultiwordFrame
//-------------------------------------------------------------------------

CommandObjectMultiwordFrame::CommandObjectMultiwordFrame (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "frame",
                            "A set of commands for operating on the current thread's frames.",
                            "frame <subcommand> [<subcommand-options>]")
{
    LoadSubCommand ("info",   CommandObjectSP (new CommandObjectFrameInfo (interpreter)));
    LoadSubCommand ("select", CommandObjectSP (new CommandObjectFrameSelect (interpreter)));
    LoadSubCommand ("variable", CommandObjectSP (new CommandObjectFrameVariable (interpreter)));
}

CommandObjectMultiwordFrame::~CommandObjectMultiwordFrame ()
{
}

