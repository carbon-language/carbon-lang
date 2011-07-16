//===-- CommandObjectTarget.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectTarget.h"

// C Includes
#include <errno.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/InputReader.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/State.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/OptionGroupArchitecture.h"
#include "lldb/Interpreter/OptionGroupFile.h"
#include "lldb/Interpreter/OptionGroupVariable.h"
#include "lldb/Interpreter/OptionGroupPlatform.h"
#include "lldb/Interpreter/OptionGroupUInt64.h"
#include "lldb/Interpreter/OptionGroupUUID.h"
#include "lldb/Interpreter/OptionGroupValueObjectDisplay.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadSpec.h"

using namespace lldb;
using namespace lldb_private;



static void
DumpTargetInfo (uint32_t target_idx, Target *target, const char *prefix_cstr, bool show_stopped_process_status, Stream &strm)
{
    const ArchSpec &target_arch = target->GetArchitecture();
    
    ModuleSP exe_module_sp (target->GetExecutableModule ());
    char exe_path[PATH_MAX];
    bool exe_valid = false;
    if (exe_module_sp)
        exe_valid = exe_module_sp->GetFileSpec().GetPath (exe_path, sizeof(exe_path));
    
    if (!exe_valid)
        ::strcpy (exe_path, "<none>");
    
    strm.Printf ("%starget #%u: %s", prefix_cstr ? prefix_cstr : "", target_idx, exe_path);
    
    uint32_t properties = 0;
    if (target_arch.IsValid())
    {
        strm.Printf ("%sarch=%s", properties++ > 0 ? ", " : " ( ", target_arch.GetTriple().str().c_str());
        properties++;
    }
    PlatformSP platform_sp (target->GetPlatform());
    if (platform_sp)
        strm.Printf ("%splatform=%s", properties++ > 0 ? ", " : " ( ", platform_sp->GetName());
    
    ProcessSP process_sp (target->GetProcessSP());
    bool show_process_status = false;
    if (process_sp)
    {
        lldb::pid_t pid = process_sp->GetID();
        StateType state = process_sp->GetState();
        if (show_stopped_process_status)
            show_process_status = StateIsStoppedState(state);
        const char *state_cstr = StateAsCString (state);
        if (pid != LLDB_INVALID_PROCESS_ID)
            strm.Printf ("%spid=%i", properties++ > 0 ? ", " : " ( ", pid);
        strm.Printf ("%sstate=%s", properties++ > 0 ? ", " : " ( ", state_cstr);
    }
    if (properties > 0)
        strm.PutCString (" )\n");
    else
        strm.EOL();
    if (show_process_status)
    {
        const bool only_threads_with_stop_reason = true;
        const uint32_t start_frame = 0;
        const uint32_t num_frames = 1;
        const uint32_t num_frames_with_source = 1;
        process_sp->GetStatus (strm);
        process_sp->GetThreadStatus (strm, 
                                     only_threads_with_stop_reason, 
                                     start_frame,
                                     num_frames,
                                     num_frames_with_source);

    }
}

static uint32_t
DumpTargetList (TargetList &target_list, bool show_stopped_process_status, Stream &strm)
{
    const uint32_t num_targets = target_list.GetNumTargets();
    if (num_targets)
    {        
        TargetSP selected_target_sp (target_list.GetSelectedTarget());
        strm.PutCString ("Current targets:\n");
        for (uint32_t i=0; i<num_targets; ++i)
        {
            TargetSP target_sp (target_list.GetTargetAtIndex (i));
            if (target_sp)
            {
                bool is_selected = target_sp.get() == selected_target_sp.get();
                DumpTargetInfo (i, 
                                target_sp.get(), 
                                is_selected ? "* " : "  ", 
                                show_stopped_process_status,
                                strm);
            }
        }
    }
    return num_targets;
}
#pragma mark CommandObjectTargetCreate

//-------------------------------------------------------------------------
// "target create"
//-------------------------------------------------------------------------

class CommandObjectTargetCreate : public CommandObject
{
public:
    CommandObjectTargetCreate(CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target create",
                       "Create a target using the argument as the main executable.",
                       NULL),
        m_option_group (interpreter),
        m_arch_option (),
        m_platform_options(true) // Do include the "--platform" option in the platform settings by passing true
    {
        CommandArgumentEntry arg;
        CommandArgumentData file_arg;
    
        // Define the first (and only) variant of this arg.
            file_arg.arg_type = eArgTypeFilename;
        file_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (file_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
        
        m_option_group.Append (&m_arch_option, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);
        m_option_group.Append (&m_platform_options, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);
        m_option_group.Finalize();
    }

    ~CommandObjectTargetCreate ()
    {
    }

    Options *
    GetOptions ()
    {
        return &m_option_group;
    }

    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        const int argc = command.GetArgumentCount();
        if (argc == 1)
        {
            const char *file_path = command.GetArgumentAtIndex(0);
            Timer scoped_timer(__PRETTY_FUNCTION__, "(lldb) target create '%s'", file_path);
            FileSpec file_spec (file_path, true);
            
            bool select = true;
            PlatformSP platform_sp;
            
            Error error;
            
            if (m_platform_options.PlatformWasSpecified ())
            {
                platform_sp = m_platform_options.CreatePlatformWithOptions(m_interpreter, select, error);
                if (!platform_sp)
                {
                    result.AppendError(error.AsCString());
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            ArchSpec file_arch;
            
            const char *arch_cstr = m_arch_option.GetArchitectureName();
            if (arch_cstr)
            {        
                if (!platform_sp)
                    platform_sp = m_interpreter.GetDebugger().GetPlatformList().GetSelectedPlatform();
                if (!m_arch_option.GetArchitecture(platform_sp.get(), file_arch))
                {
                    result.AppendErrorWithFormat("invalid architecture '%s'\n", arch_cstr);
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            
            if (! file_spec.Exists() && !file_spec.ResolveExecutableLocation())
            {
                result.AppendErrorWithFormat ("File '%s' does not exist.\n", file_path);
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
            
            TargetSP target_sp;
            Debugger &debugger = m_interpreter.GetDebugger();
            error = debugger.GetTargetList().CreateTarget (debugger, file_spec, file_arch, true, target_sp);
            
            if (target_sp)
            {
                debugger.GetTargetList().SetSelectedTarget(target_sp.get());
                result.AppendMessageWithFormat ("Current executable set to '%s' (%s).\n", file_path, target_sp->GetArchitecture().GetArchitectureName());
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
            }
            else
            {
                result.AppendError(error.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else
        {
            result.AppendErrorWithFormat("'%s' takes exactly one executable path argument.\n", m_cmd_name.c_str());
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
        
    }

    int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex(cursor_index));
        completion_str.erase (cursor_char_position);
        
        CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter, 
                                                             CommandCompletions::eDiskFileCompletion,
                                                             completion_str.c_str(),
                                                             match_start_point,
                                                             max_return_elements,
                                                             NULL,
                                                             word_complete,
                                                             matches);
        return matches.GetSize();
    }
private:
    OptionGroupOptions m_option_group;
    OptionGroupArchitecture m_arch_option;
    OptionGroupPlatform m_platform_options;

};

#pragma mark CommandObjectTargetList

//----------------------------------------------------------------------
// "target list"
//----------------------------------------------------------------------

class CommandObjectTargetList : public CommandObject
{
public:
    CommandObjectTargetList (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target list",
                       "List all current targets in the current debug session.",
                       NULL,
                       0)
    {
    }
    
    virtual
    ~CommandObjectTargetList ()
    {
    }
    
    virtual bool
    Execute (Args& args, CommandReturnObject &result)
    {
        if (args.GetArgumentCount() == 0)
        {
            Stream &strm = result.GetOutputStream();
            
            bool show_stopped_process_status = false;
            if (DumpTargetList (m_interpreter.GetDebugger().GetTargetList(), show_stopped_process_status, strm) == 0)
            {
                strm.PutCString ("No targets.\n");
            }
            result.SetStatus (eReturnStatusSuccessFinishResult);
        }
        else
        {
            result.AppendError ("the 'target list' command takes no arguments\n");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};


#pragma mark CommandObjectTargetSelect

//----------------------------------------------------------------------
// "target select"
//----------------------------------------------------------------------

class CommandObjectTargetSelect : public CommandObject
{
public:
    CommandObjectTargetSelect (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target select",
                       "Select a target as the current target by target index.",
                       NULL,
                       0)
    {
    }
    
    virtual
    ~CommandObjectTargetSelect ()
    {
    }
    
    virtual bool
    Execute (Args& args, CommandReturnObject &result)
    {
        if (args.GetArgumentCount() == 1)
        {
            bool success = false;
            const char *target_idx_arg = args.GetArgumentAtIndex(0);
            uint32_t target_idx = Args::StringToUInt32 (target_idx_arg, UINT32_MAX, 0, &success);
            if (success)
            {
                TargetList &target_list = m_interpreter.GetDebugger().GetTargetList();
                const uint32_t num_targets = target_list.GetNumTargets();
                if (target_idx < num_targets)
                {        
                    TargetSP target_sp (target_list.GetTargetAtIndex (target_idx));
                    if (target_sp)
                    {
                        Stream &strm = result.GetOutputStream();
                        target_list.SetSelectedTarget (target_sp.get());
                        bool show_stopped_process_status = false;
                        DumpTargetList (target_list, show_stopped_process_status, strm);
                        result.SetStatus (eReturnStatusSuccessFinishResult);
                    }
                    else
                    {
                        result.AppendErrorWithFormat ("target #%u is NULL in target list\n", target_idx);
                        result.SetStatus (eReturnStatusFailed);
                    }
                }
                else
                {
                    result.AppendErrorWithFormat ("index %u is out of range, valid target indexes are 0 - %u\n", 
                                                  target_idx,
                                                  num_targets - 1);
                    result.SetStatus (eReturnStatusFailed);
                }
            }
            else
            {
                result.AppendErrorWithFormat("invalid index string value '%s'\n", target_idx_arg);
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else
        {
            result.AppendError ("'target select' takes a single argument: a target index\n");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};


#pragma mark CommandObjectTargetVariable

//----------------------------------------------------------------------
// "target variable"
//----------------------------------------------------------------------

class CommandObjectTargetVariable : public CommandObject
{
public:
    CommandObjectTargetVariable (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target variable",
                       "Read global variable(s) prior to running your binary.",
                       NULL,
                       0),
        m_option_group (interpreter),
        m_option_variable (false), // Don't include frame options
        m_option_compile_units    (LLDB_OPT_SET_1, false, "file", 'f', 0, eArgTypePath, "A basename or fullpath to a file that contains global variables. This option can be specified multiple times."),
        m_option_shared_libraries (LLDB_OPT_SET_1, false, "shlib",'s', 0, eArgTypePath, "A basename or fullpath to a shared library to use in the search for global variables. This option can be specified multiple times."),
        m_varobj_options()
    {
        m_option_group.Append (&m_varobj_options, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);
        m_option_group.Append (&m_option_variable, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);
        m_option_group.Append (&m_option_compile_units, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);   
        m_option_group.Append (&m_option_shared_libraries, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);   
        m_option_group.Finalize();
    }
    
    virtual
    ~CommandObjectTargetVariable ()
    {
    }

    void
    DumpValueObject (Stream &s, VariableSP &var_sp, ValueObjectSP &valobj_sp, const char *root_name)
    {
        if (m_option_variable.format != eFormatDefault)
            valobj_sp->SetFormat (m_option_variable.format);
        
        switch (var_sp->GetScope())
        {
            case eValueTypeVariableGlobal:
                if (m_option_variable.show_scope)
                    s.PutCString("GLOBAL: ");
                break;
                
            case eValueTypeVariableStatic:
                if (m_option_variable.show_scope)
                    s.PutCString("STATIC: ");
                break;
                
            case eValueTypeVariableArgument:
                if (m_option_variable.show_scope)
                    s.PutCString("   ARG: ");
                break;
                
            case eValueTypeVariableLocal:
                if (m_option_variable.show_scope)
                    s.PutCString(" LOCAL: ");
                break;
                
            default:
                break;
        }
        
        if (m_option_variable.show_decl)
        {
            bool show_fullpaths = false;
            bool show_module = true;
            if (var_sp->DumpDeclaration(&s, show_fullpaths, show_module))
                s.PutCString (": ");
        }
        
        const Format format = m_option_variable.format;
        if (format != eFormatDefault)
            valobj_sp->SetFormat (format);
        
        ValueObject::DumpValueObject (s, 
                                      valobj_sp.get(), 
                                      root_name,
                                      m_varobj_options.ptr_depth, 
                                      0, 
                                      m_varobj_options.max_depth, 
                                      m_varobj_options.show_types,
                                      m_varobj_options.show_location,
                                      m_varobj_options.use_objc,
                                      m_varobj_options.use_dynamic,
                                      false,
                                      m_varobj_options.flat_output,
                                      m_varobj_options.no_summary_depth);                                        

    }
    
    
    static uint32_t GetVariableCallback (void *baton, 
                                         const char *name,
                                         VariableList &variable_list)
    {
        Target *target = static_cast<Target *>(baton);
        if (target)
        {
            return target->GetImages().FindGlobalVariables (ConstString(name),
                                                            true, 
                                                            UINT32_MAX, 
                                                            variable_list);
        }
        return 0;
    }
    


    virtual bool
    Execute (Args& args, CommandReturnObject &result)
    {
        ExecutionContext exe_ctx (m_interpreter.GetExecutionContext());

        if (exe_ctx.target)
        {
            const size_t argc = args.GetArgumentCount();
            if (argc > 0)
            {
                Stream &s = result.GetOutputStream();

                for (size_t idx = 0; idx < argc; ++idx)
                {
                    VariableList variable_list;
                    ValueObjectList valobj_list;

                    const char *arg = args.GetArgumentAtIndex(idx);
                    uint32_t matches = 0;
                    bool use_var_name = false;
                    if (m_option_variable.use_regex)
                    {
                        RegularExpression regex(arg);
                        if (!regex.IsValid ())
                        {
                            result.GetErrorStream().Printf ("error: invalid regular expression: '%s'\n", arg);
                            result.SetStatus (eReturnStatusFailed);
                            return false;
                        }
                        use_var_name = true;
                        matches = exe_ctx.target->GetImages().FindGlobalVariables (regex,
                                                                                   true, 
                                                                                   UINT32_MAX, 
                                                                                   variable_list);
                    }
                    else
                    {
                        Error error (Variable::GetValuesForVariableExpressionPath (arg,
                                                                                   exe_ctx.GetBestExecutionContextScope(),
                                                                                   GetVariableCallback,
                                                                                   exe_ctx.target,
                                                                                   variable_list,
                                                                                   valobj_list));
                        
//                        matches = exe_ctx.target->GetImages().FindGlobalVariables (ConstString(arg),
//                                                                                   true, 
//                                                                                   UINT32_MAX, 
//                                                                                   variable_list);
                        matches = variable_list.GetSize();
                    }
                    
                    if (matches == 0)
                    {
                        result.GetErrorStream().Printf ("error: can't find global variable '%s'\n", arg);
                        result.SetStatus (eReturnStatusFailed);
                        return false;
                    }
                    else
                    {
                        for (uint32_t global_idx=0; global_idx<matches; ++global_idx)
                        {
                            VariableSP var_sp (variable_list.GetVariableAtIndex(global_idx));
                            if (var_sp)
                            {
                                ValueObjectSP valobj_sp (valobj_list.GetValueObjectAtIndex(global_idx));
                                if (!valobj_sp)
                                    valobj_sp = ValueObjectVariable::Create (exe_ctx.GetBestExecutionContextScope(), var_sp);
                                
                                if (valobj_sp)
                                    DumpValueObject (s, var_sp, valobj_sp, use_var_name ? var_sp->GetName().GetCString() : arg);
                            }
                        }
                    }
                }
            }
            else
            {
                result.AppendError ("'target variable' takes one or more global variable names as arguments\n");
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else
        {
            result.AppendError ("invalid target, create a debug target using the 'target create' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        return result.Succeeded();
    }
    
    Options *
    GetOptions ()
    {
        return &m_option_group;
    }
    
protected:
    OptionGroupOptions m_option_group;
    OptionGroupVariable m_option_variable;
    OptionGroupFileList m_option_compile_units;
    OptionGroupFileList m_option_shared_libraries;
    OptionGroupValueObjectDisplay m_varobj_options;

};


#pragma mark CommandObjectTargetModulesSearchPathsAdd

class CommandObjectTargetModulesSearchPathsAdd : public CommandObject
{
public:

    CommandObjectTargetModulesSearchPathsAdd (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target modules search-paths add",
                       "Add new image search paths substitution pairs to the current target.",
                       NULL)
    {
        CommandArgumentEntry arg;
        CommandArgumentData old_prefix_arg;
        CommandArgumentData new_prefix_arg;
        
        // Define the first variant of this arg pair.
        old_prefix_arg.arg_type = eArgTypeOldPathPrefix;
        old_prefix_arg.arg_repetition = eArgRepeatPairPlus;
        
        // Define the first variant of this arg pair.
        new_prefix_arg.arg_type = eArgTypeNewPathPrefix;
        new_prefix_arg.arg_repetition = eArgRepeatPairPlus;
        
        // There are two required arguments that must always occur together, i.e. an argument "pair".  Because they
        // must always occur together, they are treated as two variants of one argument rather than two independent
        // arguments.  Push them both into the first argument position for m_arguments...

        arg.push_back (old_prefix_arg);
        arg.push_back (new_prefix_arg);

        m_arguments.push_back (arg);
    }

    ~CommandObjectTargetModulesSearchPathsAdd ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            uint32_t argc = command.GetArgumentCount();
            if (argc & 1)
            {
                result.AppendError ("add requires an even number of arguments\n");
                result.SetStatus (eReturnStatusFailed);
            }
            else
            {
                for (uint32_t i=0; i<argc; i+=2)
                {
                    const char *from = command.GetArgumentAtIndex(i);
                    const char *to = command.GetArgumentAtIndex(i+1);
                    
                    if (from[0] && to[0])
                    {
                        bool last_pair = ((argc - i) == 2);
                        target->GetImageSearchPathList().Append (ConstString(from),
                                                                 ConstString(to),
                                                                 last_pair); // Notify if this is the last pair
                        result.SetStatus (eReturnStatusSuccessFinishNoResult);
                    }
                    else
                    {
                        if (from[0])
                            result.AppendError ("<path-prefix> can't be empty\n");
                        else
                            result.AppendError ("<new-path-prefix> can't be empty\n");
                        result.SetStatus (eReturnStatusFailed);
                    }
                }
            }
        }
        else
        {
            result.AppendError ("invalid target\n");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

#pragma mark CommandObjectTargetModulesSearchPathsClear

class CommandObjectTargetModulesSearchPathsClear : public CommandObject
{
public:

    CommandObjectTargetModulesSearchPathsClear (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target modules search-paths clear",
                       "Clear all current image search path substitution pairs from the current target.",
                       "target modules search-paths clear")
    {
    }

    ~CommandObjectTargetModulesSearchPathsClear ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            bool notify = true;
            target->GetImageSearchPathList().Clear(notify);
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            result.AppendError ("invalid target\n");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

#pragma mark CommandObjectTargetModulesSearchPathsInsert

class CommandObjectTargetModulesSearchPathsInsert : public CommandObject
{
public:

    CommandObjectTargetModulesSearchPathsInsert (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target modules search-paths insert",
                       "Insert a new image search path substitution pair into the current target at the specified index.",
                       NULL)
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentData index_arg;
        CommandArgumentData old_prefix_arg;
        CommandArgumentData new_prefix_arg;
        
        // Define the first and only variant of this arg.
        index_arg.arg_type = eArgTypeIndex;
        index_arg.arg_repetition = eArgRepeatPlain;

        // Put the one and only variant into the first arg for m_arguments:
        arg1.push_back (index_arg);

        // Define the first variant of this arg pair.
        old_prefix_arg.arg_type = eArgTypeOldPathPrefix;
        old_prefix_arg.arg_repetition = eArgRepeatPairPlus;
        
        // Define the first variant of this arg pair.
        new_prefix_arg.arg_type = eArgTypeNewPathPrefix;
        new_prefix_arg.arg_repetition = eArgRepeatPairPlus;
        
        // There are two required arguments that must always occur together, i.e. an argument "pair".  Because they
        // must always occur together, they are treated as two variants of one argument rather than two independent
        // arguments.  Push them both into the same argument position for m_arguments...

        arg2.push_back (old_prefix_arg);
        arg2.push_back (new_prefix_arg);

        // Add arguments to m_arguments.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
    }

    ~CommandObjectTargetModulesSearchPathsInsert ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            uint32_t argc = command.GetArgumentCount();
            // check for at least 3 arguments and an odd nubmer of parameters
            if (argc >= 3 && argc & 1)
            {
                bool success = false;

                uint32_t insert_idx = Args::StringToUInt32(command.GetArgumentAtIndex(0), UINT32_MAX, 0, &success);

                if (!success)
                {
                    result.AppendErrorWithFormat("<index> parameter is not an integer: '%s'.\n", command.GetArgumentAtIndex(0));
                    result.SetStatus (eReturnStatusFailed);
                    return result.Succeeded();
                }

                // shift off the index
                command.Shift();
                argc = command.GetArgumentCount();

                for (uint32_t i=0; i<argc; i+=2, ++insert_idx)
                {
                    const char *from = command.GetArgumentAtIndex(i);
                    const char *to = command.GetArgumentAtIndex(i+1);
                    
                    if (from[0] && to[0])
                    {
                        bool last_pair = ((argc - i) == 2);
                        target->GetImageSearchPathList().Insert (ConstString(from),
                                                                 ConstString(to),
                                                                 insert_idx,
                                                                 last_pair);
                        result.SetStatus (eReturnStatusSuccessFinishNoResult);
                    }
                    else
                    {
                        if (from[0])
                            result.AppendError ("<path-prefix> can't be empty\n");
                        else
                            result.AppendError ("<new-path-prefix> can't be empty\n");
                        result.SetStatus (eReturnStatusFailed);
                        return false;
                    }
                }
            }
            else
            {
                result.AppendError ("insert requires at least three arguments\n");
                result.SetStatus (eReturnStatusFailed);
                return result.Succeeded();
            }

        }
        else
        {
            result.AppendError ("invalid target\n");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};


#pragma mark CommandObjectTargetModulesSearchPathsList


class CommandObjectTargetModulesSearchPathsList : public CommandObject
{
public:

    CommandObjectTargetModulesSearchPathsList (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target modules search-paths list",
                       "List all current image search path substitution pairs in the current target.",
                       "target modules search-paths list")
    {
    }

    ~CommandObjectTargetModulesSearchPathsList ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            if (command.GetArgumentCount() != 0)
            {
                result.AppendError ("list takes no arguments\n");
                result.SetStatus (eReturnStatusFailed);
                return result.Succeeded();
            }

            target->GetImageSearchPathList().Dump(&result.GetOutputStream());
            result.SetStatus (eReturnStatusSuccessFinishResult);
        }
        else
        {
            result.AppendError ("invalid target\n");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

#pragma mark CommandObjectTargetModulesSearchPathsQuery

class CommandObjectTargetModulesSearchPathsQuery : public CommandObject
{
public:

    CommandObjectTargetModulesSearchPathsQuery (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "target modules search-paths query",
                   "Transform a path using the first applicable image search path.",
                   NULL)
    {
        CommandArgumentEntry arg;
        CommandArgumentData path_arg;
        
        // Define the first (and only) variant of this arg.
        path_arg.arg_type = eArgTypePath;
        path_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (path_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    ~CommandObjectTargetModulesSearchPathsQuery ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            if (command.GetArgumentCount() != 1)
            {
                result.AppendError ("query requires one argument\n");
                result.SetStatus (eReturnStatusFailed);
                return result.Succeeded();
            }

            ConstString orig(command.GetArgumentAtIndex(0));
            ConstString transformed;
            if (target->GetImageSearchPathList().RemapPath(orig, transformed))
                result.GetOutputStream().Printf("%s\n", transformed.GetCString());
            else
                result.GetOutputStream().Printf("%s\n", orig.GetCString());

            result.SetStatus (eReturnStatusSuccessFinishResult);
        }
        else
        {
            result.AppendError ("invalid target\n");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

//----------------------------------------------------------------------
// Static Helper functions
//----------------------------------------------------------------------
static void
DumpModuleArchitecture (Stream &strm, Module *module, bool full_triple, uint32_t width)
{
    if (module)
    {
        const char *arch_cstr;
        if (full_triple)
            arch_cstr = module->GetArchitecture().GetTriple().str().c_str();
        else
            arch_cstr = module->GetArchitecture().GetArchitectureName();
        if (width)
            strm.Printf("%-*s", width, arch_cstr);
        else
            strm.PutCString(arch_cstr);
    }
}

static void
DumpModuleUUID (Stream &strm, Module *module)
{
    module->GetUUID().Dump (&strm);
}

static uint32_t
DumpCompileUnitLineTable
(
 CommandInterpreter &interpreter,
 Stream &strm,
 Module *module,
 const FileSpec &file_spec,
 bool load_addresses
 )
{
    uint32_t num_matches = 0;
    if (module)
    {
        SymbolContextList sc_list;
        num_matches = module->ResolveSymbolContextsForFileSpec (file_spec,
                                                                0,
                                                                false,
                                                                eSymbolContextCompUnit,
                                                                sc_list);
        
        for (uint32_t i=0; i<num_matches; ++i)
        {
            SymbolContext sc;
            if (sc_list.GetContextAtIndex(i, sc))
            {
                if (i > 0)
                    strm << "\n\n";
                
                strm << "Line table for " << *static_cast<FileSpec*> (sc.comp_unit) << " in `"
                << module->GetFileSpec().GetFilename() << "\n";
                LineTable *line_table = sc.comp_unit->GetLineTable();
                if (line_table)
                    line_table->GetDescription (&strm, 
                                                interpreter.GetExecutionContext().target, 
                                                lldb::eDescriptionLevelBrief);
                else
                    strm << "No line table";
            }
        }
    }
    return num_matches;
}

static void
DumpFullpath (Stream &strm, const FileSpec *file_spec_ptr, uint32_t width)
{
    if (file_spec_ptr)
    {
        if (width > 0)
        {
            char fullpath[PATH_MAX];
            if (file_spec_ptr->GetPath(fullpath, sizeof(fullpath)))
            {
                strm.Printf("%-*s", width, fullpath);
                return;
            }
        }
        else
        {
            file_spec_ptr->Dump(&strm);
            return;
        }
    }
    // Keep the width spacing correct if things go wrong...
    if (width > 0)
        strm.Printf("%-*s", width, "");
}

static void
DumpDirectory (Stream &strm, const FileSpec *file_spec_ptr, uint32_t width)
{
    if (file_spec_ptr)
    {
        if (width > 0)
            strm.Printf("%-*s", width, file_spec_ptr->GetDirectory().AsCString(""));
        else
            file_spec_ptr->GetDirectory().Dump(&strm);
        return;
    }
    // Keep the width spacing correct if things go wrong...
    if (width > 0)
        strm.Printf("%-*s", width, "");
}

static void
DumpBasename (Stream &strm, const FileSpec *file_spec_ptr, uint32_t width)
{
    if (file_spec_ptr)
    {
        if (width > 0)
            strm.Printf("%-*s", width, file_spec_ptr->GetFilename().AsCString(""));
        else
            file_spec_ptr->GetFilename().Dump(&strm);
        return;
    }
    // Keep the width spacing correct if things go wrong...
    if (width > 0)
        strm.Printf("%-*s", width, "");
}


static void
DumpModuleSymtab (CommandInterpreter &interpreter, Stream &strm, Module *module, SortOrder sort_order)
{
    if (module)
    {
        ObjectFile *objfile = module->GetObjectFile ();
        if (objfile)
        {
            Symtab *symtab = objfile->GetSymtab();
            if (symtab)
                symtab->Dump(&strm, interpreter.GetExecutionContext().target, sort_order);
        }
    }
}

static void
DumpModuleSections (CommandInterpreter &interpreter, Stream &strm, Module *module)
{
    if (module)
    {
        ObjectFile *objfile = module->GetObjectFile ();
        if (objfile)
        {
            SectionList *section_list = objfile->GetSectionList();
            if (section_list)
            {
                strm.PutCString ("Sections for '");
                strm << module->GetFileSpec();
                if (module->GetObjectName())
                    strm << '(' << module->GetObjectName() << ')';
                strm.Printf ("' (%s):\n", module->GetArchitecture().GetArchitectureName());
                strm.IndentMore();
                section_list->Dump(&strm, interpreter.GetExecutionContext().target, true, UINT32_MAX);
                strm.IndentLess();
            }
        }
    }
}

static bool
DumpModuleSymbolVendor (Stream &strm, Module *module)
{
    if (module)
    {
        SymbolVendor *symbol_vendor = module->GetSymbolVendor(true);
        if (symbol_vendor)
        {
            symbol_vendor->Dump(&strm);
            return true;
        }
    }
    return false;
}

static bool
LookupAddressInModule 
(
 CommandInterpreter &interpreter, 
 Stream &strm, 
 Module *module, 
 uint32_t resolve_mask, 
 lldb::addr_t raw_addr, 
 lldb::addr_t offset,
 bool verbose
 )
{
    if (module)
    {
        lldb::addr_t addr = raw_addr - offset;
        Address so_addr;
        SymbolContext sc;
        Target *target = interpreter.GetExecutionContext().target;
        if (target && !target->GetSectionLoadList().IsEmpty())
        {
            if (!target->GetSectionLoadList().ResolveLoadAddress (addr, so_addr))
                return false;
            else if (so_addr.GetModule() != module)
                return false;
        }
        else
        {
            if (!module->ResolveFileAddress (addr, so_addr))
                return false;
        }
        
        // If an offset was given, print out the address we ended up looking up
        if (offset)
            strm.Printf("File Address: 0x%llx\n", addr);
        
        ExecutionContextScope *exe_scope = interpreter.GetExecutionContext().GetBestExecutionContextScope();
        strm.IndentMore();
        strm.Indent ("    Address: ");
        so_addr.Dump (&strm, exe_scope, Address::DumpStyleSectionNameOffset);
        strm.EOL();
        strm.Indent ("    Summary: ");
        const uint32_t save_indent = strm.GetIndentLevel ();
        strm.SetIndentLevel (save_indent + 11);
        so_addr.Dump (&strm, exe_scope, Address::DumpStyleResolvedDescription);
        strm.SetIndentLevel (save_indent);
        strm.EOL();
        // Print out detailed address information when verbose is enabled
        if (verbose)
        {
            if (so_addr.Dump (&strm, exe_scope, Address::DumpStyleDetailedSymbolContext))
                strm.EOL();
        }
        strm.IndentLess();
        return true;
    }
    
    return false;
}

static uint32_t
LookupSymbolInModule (CommandInterpreter &interpreter, Stream &strm, Module *module, const char *name, bool name_is_regex)
{
    if (module)
    {
        SymbolContext sc;
        
        ObjectFile *objfile = module->GetObjectFile ();
        if (objfile)
        {
            Symtab *symtab = objfile->GetSymtab();
            if (symtab)
            {
                uint32_t i;
                std::vector<uint32_t> match_indexes;
                ConstString symbol_name (name);
                uint32_t num_matches = 0;
                if (name_is_regex)
                {
                    RegularExpression name_regexp(name);
                    num_matches = symtab->AppendSymbolIndexesMatchingRegExAndType (name_regexp, 
                                                                                   eSymbolTypeAny,
                                                                                   match_indexes);
                }
                else
                {
                    num_matches = symtab->AppendSymbolIndexesWithName (symbol_name, match_indexes);
                }
                
                
                if (num_matches > 0)
                {
                    strm.Indent ();
                    strm.Printf("%u symbols match %s'%s' in ", num_matches,
                                name_is_regex ? "the regular expression " : "", name);
                    DumpFullpath (strm, &module->GetFileSpec(), 0);
                    strm.PutCString(":\n");
                    strm.IndentMore ();
                    Symtab::DumpSymbolHeader (&strm);
                    for (i=0; i < num_matches; ++i)
                    {
                        Symbol *symbol = symtab->SymbolAtIndex(match_indexes[i]);
                        strm.Indent ();
                        symbol->Dump (&strm, interpreter.GetExecutionContext().target, i);
                    }
                    strm.IndentLess ();
                    return num_matches;
                }
            }
        }
    }
    return 0;
}


static void
DumpSymbolContextList (CommandInterpreter &interpreter, Stream &strm, SymbolContextList &sc_list, bool prepend_addr, bool verbose)
{
    strm.IndentMore ();
    uint32_t i;
    const uint32_t num_matches = sc_list.GetSize();
    
    for (i=0; i<num_matches; ++i)
    {
        SymbolContext sc;
        if (sc_list.GetContextAtIndex(i, sc))
        {
            strm.Indent();
            ExecutionContextScope *exe_scope = interpreter.GetExecutionContext().GetBestExecutionContextScope ();
            
            if (prepend_addr)
            {
                if (sc.line_entry.range.GetBaseAddress().IsValid())
                {
                    sc.line_entry.range.GetBaseAddress().Dump (&strm, 
                                                               exe_scope,
                                                               Address::DumpStyleLoadAddress, 
                                                               Address::DumpStyleModuleWithFileAddress);
                    strm.PutCString(" in ");
                }
            }
            sc.DumpStopContext(&strm, 
                               exe_scope, 
                               sc.line_entry.range.GetBaseAddress(), 
                               true, 
                               true, 
                               false);
            strm.EOL();
            if (verbose)
            {
                if (sc.line_entry.range.GetBaseAddress().IsValid())
                {
                    if (sc.line_entry.range.GetBaseAddress().Dump (&strm, 
                                                                   exe_scope, 
                                                                   Address::DumpStyleDetailedSymbolContext))
                        strm.PutCString("\n\n");
                }
                else if (sc.function->GetAddressRange().GetBaseAddress().IsValid())
                {
                    if (sc.function->GetAddressRange().GetBaseAddress().Dump (&strm, 
                                                                              exe_scope, 
                                                                              Address::DumpStyleDetailedSymbolContext))
                        strm.PutCString("\n\n");
                }
            }
        }
    }
    strm.IndentLess ();
}

static uint32_t
LookupFunctionInModule (CommandInterpreter &interpreter, Stream &strm, Module *module, const char *name, bool name_is_regex, bool verbose)
{
    if (module && name && name[0])
    {
        SymbolContextList sc_list;
        const bool include_symbols = false;
        const bool append = true;
        uint32_t num_matches = 0;
        if (name_is_regex)
        {
            RegularExpression function_name_regex (name);
            num_matches = module->FindFunctions (function_name_regex, 
                                                 include_symbols,
                                                 append, 
                                                 sc_list);
        }
        else
        {
            ConstString function_name (name);
            num_matches = module->FindFunctions (function_name, 
                                                 eFunctionNameTypeBase | eFunctionNameTypeFull | eFunctionNameTypeMethod | eFunctionNameTypeSelector, 
                                                 include_symbols,
                                                 append, 
                                                 sc_list);
        }
        
        if (num_matches)
        {
            strm.Indent ();
            strm.Printf("%u match%s found in ", num_matches, num_matches > 1 ? "es" : "");
            DumpFullpath (strm, &module->GetFileSpec(), 0);
            strm.PutCString(":\n");
            DumpSymbolContextList (interpreter, strm, sc_list, true, verbose);
        }
        return num_matches;
    }
    return 0;
}

static uint32_t
LookupTypeInModule (CommandInterpreter &interpreter, 
                    Stream &strm, 
                    Module *module, 
                    const char *name_cstr, 
                    bool name_is_regex)
{
    if (module && name_cstr && name_cstr[0])
    {
        SymbolContextList sc_list;
        
        SymbolVendor *symbol_vendor = module->GetSymbolVendor();
        if (symbol_vendor)
        {
            TypeList type_list;
            uint32_t num_matches = 0;
            SymbolContext sc;
            //            if (name_is_regex)
            //            {
            //                RegularExpression name_regex (name_cstr);
            //                num_matches = symbol_vendor->FindFunctions(sc, name_regex, true, UINT32_MAX, type_list);
            //            }
            //            else
            //            {
            ConstString name(name_cstr);
            num_matches = symbol_vendor->FindTypes(sc, name, true, UINT32_MAX, type_list);
            //            }
            
            if (num_matches)
            {
                strm.Indent ();
                strm.Printf("%u match%s found in ", num_matches, num_matches > 1 ? "es" : "");
                DumpFullpath (strm, &module->GetFileSpec(), 0);
                strm.PutCString(":\n");
                const uint32_t num_types = type_list.GetSize();
                for (uint32_t i=0; i<num_types; ++i)
                {
                    TypeSP type_sp (type_list.GetTypeAtIndex(i));
                    if (type_sp)
                    {
                        // Resolve the clang type so that any forward references
                        // to types that haven't yet been parsed will get parsed.
                        type_sp->GetClangFullType ();
                        type_sp->GetDescription (&strm, eDescriptionLevelFull, true);
                    }
                    strm.EOL();
                }
            }
            return num_matches;
        }
    }
    return 0;
}

static uint32_t
LookupFileAndLineInModule (CommandInterpreter &interpreter, 
                           Stream &strm, 
                           Module *module, 
                           const FileSpec &file_spec, 
                           uint32_t line, 
                           bool check_inlines,
                           bool verbose)
{
    if (module && file_spec)
    {
        SymbolContextList sc_list;
        const uint32_t num_matches = module->ResolveSymbolContextsForFileSpec(file_spec, line, check_inlines,
                                                                              eSymbolContextEverything, sc_list);
        if (num_matches > 0)
        {
            strm.Indent ();
            strm.Printf("%u match%s found in ", num_matches, num_matches > 1 ? "es" : "");
            strm << file_spec;
            if (line > 0)
                strm.Printf (":%u", line);
            strm << " in ";
            DumpFullpath (strm, &module->GetFileSpec(), 0);
            strm.PutCString(":\n");
            DumpSymbolContextList (interpreter, strm, sc_list, true, verbose);
            return num_matches;
        }
    }
    return 0;
    
}

#pragma mark CommandObjectTargetModulesModuleAutoComplete

//----------------------------------------------------------------------
// A base command object class that can auto complete with module file
// paths
//----------------------------------------------------------------------

class CommandObjectTargetModulesModuleAutoComplete : public CommandObject
{
public:
    
    CommandObjectTargetModulesModuleAutoComplete (CommandInterpreter &interpreter,
                                      const char *name,
                                      const char *help,
                                      const char *syntax) :
    CommandObject (interpreter, name, help, syntax)
    {
        CommandArgumentEntry arg;
        CommandArgumentData file_arg;
        
        // Define the first (and only) variant of this arg.
        file_arg.arg_type = eArgTypeFilename;
        file_arg.arg_repetition = eArgRepeatStar;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (file_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }
    
    virtual
    ~CommandObjectTargetModulesModuleAutoComplete ()
    {
    }
    
    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        // Arguments are the standard module completer.
        std::string completion_str (input.GetArgumentAtIndex(cursor_index));
        completion_str.erase (cursor_char_position);
        
        CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                             CommandCompletions::eModuleCompletion,
                                                             completion_str.c_str(),
                                                             match_start_point,
                                                             max_return_elements,
                                                             NULL,
                                                             word_complete,
                                                             matches);
        return matches.GetSize();
    }
};

#pragma mark CommandObjectTargetModulesSourceFileAutoComplete

//----------------------------------------------------------------------
// A base command object class that can auto complete with module source
// file paths
//----------------------------------------------------------------------

class CommandObjectTargetModulesSourceFileAutoComplete : public CommandObject
{
public:
    
    CommandObjectTargetModulesSourceFileAutoComplete (CommandInterpreter &interpreter,
                                          const char *name,
                                          const char *help,
                                          const char *syntax) :
    CommandObject (interpreter, name, help, syntax)
    {
        CommandArgumentEntry arg;
        CommandArgumentData source_file_arg;
        
        // Define the first (and only) variant of this arg.
        source_file_arg.arg_type = eArgTypeSourceFile;
        source_file_arg.arg_repetition = eArgRepeatPlus;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (source_file_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }
    
    virtual
    ~CommandObjectTargetModulesSourceFileAutoComplete ()
    {
    }
    
    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        // Arguments are the standard source file completer.
        std::string completion_str (input.GetArgumentAtIndex(cursor_index));
        completion_str.erase (cursor_char_position);
        
        CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter, 
                                                             CommandCompletions::eSourceFileCompletion,
                                                             completion_str.c_str(),
                                                             match_start_point,
                                                             max_return_elements,
                                                             NULL,
                                                             word_complete,
                                                             matches);
        return matches.GetSize();
    }
};


#pragma mark CommandObjectTargetModulesDumpSymtab


class CommandObjectTargetModulesDumpSymtab : public CommandObjectTargetModulesModuleAutoComplete
{
public:
    CommandObjectTargetModulesDumpSymtab (CommandInterpreter &interpreter) :
    CommandObjectTargetModulesModuleAutoComplete (interpreter,
                                      "target modules dump symtab",
                                      "Dump the symbol table from one or more target modules.",
                                      NULL),
    m_options (interpreter)
    {
    }
    
    virtual
    ~CommandObjectTargetModulesDumpSymtab ()
    {
    }
    
    virtual bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("invalid target, create a debug target using the 'target create' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else
        {
            uint32_t num_dumped = 0;
            
            uint32_t addr_byte_size = target->GetArchitecture().GetAddressByteSize();
            result.GetOutputStream().SetAddressByteSize(addr_byte_size);
            result.GetErrorStream().SetAddressByteSize(addr_byte_size);
            
            if (command.GetArgumentCount() == 0)
            {
                // Dump all sections for all modules images
                const uint32_t num_modules = target->GetImages().GetSize();
                if (num_modules > 0)
                {
                    result.GetOutputStream().Printf("Dumping symbol table for %u modules.\n", num_modules);
                    for (uint32_t image_idx = 0;  image_idx<num_modules; ++image_idx)
                    {
                        if (num_dumped > 0)
                        {
                            result.GetOutputStream().EOL();
                            result.GetOutputStream().EOL();
                        }
                        num_dumped++;
                        DumpModuleSymtab (m_interpreter, result.GetOutputStream(), target->GetImages().GetModulePointerAtIndex(image_idx), m_options.m_sort_order);
                    }
                }
                else
                {
                    result.AppendError ("the target has no associated executable images");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            else
            {
                // Dump specified images (by basename or fullpath)
                const char *arg_cstr;
                for (int arg_idx = 0; (arg_cstr = command.GetArgumentAtIndex(arg_idx)) != NULL; ++arg_idx)
                {
                    FileSpec image_file(arg_cstr, false);
                    ModuleList matching_modules;
                    size_t num_matching_modules = target->GetImages().FindModules(&image_file, NULL, NULL, NULL, matching_modules);
                    
                    // Not found in our module list for our target, check the main
                    // shared module list in case it is a extra file used somewhere
                    // else
                    if (num_matching_modules == 0)
                        num_matching_modules = ModuleList::FindSharedModules (image_file, 
                                                                              target->GetArchitecture(), 
                                                                              NULL, 
                                                                              NULL, 
                                                                              matching_modules);
                    
                    if (num_matching_modules > 0)
                    {
                        for (size_t i=0; i<num_matching_modules; ++i)
                        {
                            Module *image_module = matching_modules.GetModulePointerAtIndex(i);
                            if (image_module)
                            {
                                if (num_dumped > 0)
                                {
                                    result.GetOutputStream().EOL();
                                    result.GetOutputStream().EOL();
                                }
                                num_dumped++;
                                DumpModuleSymtab (m_interpreter, result.GetOutputStream(), image_module, m_options.m_sort_order);
                            }
                        }
                    }
                    else
                        result.AppendWarningWithFormat("Unable to find an image that matches '%s'.\n", arg_cstr);
                }
            }
            
            if (num_dumped > 0)
                result.SetStatus (eReturnStatusSuccessFinishResult);
            else
            {
                result.AppendError ("no matching executable images found");
                result.SetStatus (eReturnStatusFailed);
            }
        }
        return result.Succeeded();
    }
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }
    
    class CommandOptions : public Options
    {
    public:
        
        CommandOptions (CommandInterpreter &interpreter) :
        Options(interpreter),
        m_sort_order (eSortOrderNone)
        {
        }
        
        virtual
        ~CommandOptions ()
        {
        }
        
        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;
            
            switch (short_option)
            {
                case 's':
                {
                    bool found_one = false;
                    m_sort_order = (SortOrder) Args::StringToOptionEnum (option_arg, 
                                                                         g_option_table[option_idx].enum_values, 
                                                                         eSortOrderNone,
                                                                         &found_one);
                    if (!found_one)
                        error.SetErrorStringWithFormat("Invalid enumeration value '%s' for option '%c'.\n", 
                                                       option_arg, 
                                                       short_option);
                }
                    break;
                    
                default:
                    error.SetErrorStringWithFormat("Invalid short option character '%c'.\n", short_option);
                    break;
                    
            }
            return error;
        }
        
        void
        OptionParsingStarting ()
        {
            m_sort_order = eSortOrderNone;
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        
        // Options table: Required for subclasses of Options.
        static OptionDefinition g_option_table[];
        
        SortOrder m_sort_order;
    };
    
protected:
    
    CommandOptions m_options;
};

static OptionEnumValueElement
g_sort_option_enumeration[4] =
{
    { eSortOrderNone,       "none",     "No sorting, use the original symbol table order."},
    { eSortOrderByAddress,  "address",  "Sort output by symbol address."},
    { eSortOrderByName,     "name",     "Sort output by symbol name."},
    { 0,                    NULL,       NULL }
};


OptionDefinition
CommandObjectTargetModulesDumpSymtab::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_1, false, "sort", 's', required_argument, g_sort_option_enumeration, 0, eArgTypeSortOrder, "Supply a sort order when dumping the symbol table."},
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

#pragma mark CommandObjectTargetModulesDumpSections

//----------------------------------------------------------------------
// Image section dumping command
//----------------------------------------------------------------------

class CommandObjectTargetModulesDumpSections : public CommandObjectTargetModulesModuleAutoComplete
{
public:
    CommandObjectTargetModulesDumpSections (CommandInterpreter &interpreter) :
    CommandObjectTargetModulesModuleAutoComplete (interpreter,
                                      "target modules dump sections",
                                      "Dump the sections from one or more target modules.",
                                      //"target modules dump sections [<file1> ...]")
                                      NULL)
    {
    }
    
    virtual
    ~CommandObjectTargetModulesDumpSections ()
    {
    }
    
    virtual bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("invalid target, create a debug target using the 'target create' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else
        {
            uint32_t num_dumped = 0;
            
            uint32_t addr_byte_size = target->GetArchitecture().GetAddressByteSize();
            result.GetOutputStream().SetAddressByteSize(addr_byte_size);
            result.GetErrorStream().SetAddressByteSize(addr_byte_size);
            
            if (command.GetArgumentCount() == 0)
            {
                // Dump all sections for all modules images
                const uint32_t num_modules = target->GetImages().GetSize();
                if (num_modules > 0)
                {
                    result.GetOutputStream().Printf("Dumping sections for %u modules.\n", num_modules);
                    for (uint32_t image_idx = 0;  image_idx<num_modules; ++image_idx)
                    {
                        num_dumped++;
                        DumpModuleSections (m_interpreter, result.GetOutputStream(), target->GetImages().GetModulePointerAtIndex(image_idx));
                    }
                }
                else
                {
                    result.AppendError ("the target has no associated executable images");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            else
            {
                // Dump specified images (by basename or fullpath)
                const char *arg_cstr;
                for (int arg_idx = 0; (arg_cstr = command.GetArgumentAtIndex(arg_idx)) != NULL; ++arg_idx)
                {
                    FileSpec image_file(arg_cstr, false);
                    ModuleList matching_modules;
                    size_t num_matching_modules = target->GetImages().FindModules(&image_file, NULL, NULL, NULL, matching_modules);
                    
                    // Not found in our module list for our target, check the main
                    // shared module list in case it is a extra file used somewhere
                    // else
                    if (num_matching_modules == 0)
                        num_matching_modules = ModuleList::FindSharedModules (image_file, 
                                                                              target->GetArchitecture(), 
                                                                              NULL, 
                                                                              NULL, 
                                                                              matching_modules);
                    
                    if (num_matching_modules > 0)
                    {
                        for (size_t i=0; i<num_matching_modules; ++i)
                        {
                            Module * image_module = matching_modules.GetModulePointerAtIndex(i);
                            if (image_module)
                            {
                                num_dumped++;
                                DumpModuleSections (m_interpreter, result.GetOutputStream(), image_module);
                            }
                        }
                    }
                    else
                        result.AppendWarningWithFormat("Unable to find an image that matches '%s'.\n", arg_cstr);
                }
            }
            
            if (num_dumped > 0)
                result.SetStatus (eReturnStatusSuccessFinishResult);
            else
            {
                result.AppendError ("no matching executable images found");
                result.SetStatus (eReturnStatusFailed);
            }
        }
        return result.Succeeded();
    }
};


#pragma mark CommandObjectTargetModulesDumpSymfile

//----------------------------------------------------------------------
// Image debug symbol dumping command
//----------------------------------------------------------------------

class CommandObjectTargetModulesDumpSymfile : public CommandObjectTargetModulesModuleAutoComplete
{
public:
    CommandObjectTargetModulesDumpSymfile (CommandInterpreter &interpreter) :
    CommandObjectTargetModulesModuleAutoComplete (interpreter,
                                      "target modules dump symfile",
                                      "Dump the debug symbol file for one or more target modules.",
                                      //"target modules dump symfile [<file1> ...]")
                                      NULL)
    {
    }
    
    virtual
    ~CommandObjectTargetModulesDumpSymfile ()
    {
    }
    
    virtual bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("invalid target, create a debug target using the 'target create' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else
        {
            uint32_t num_dumped = 0;
            
            uint32_t addr_byte_size = target->GetArchitecture().GetAddressByteSize();
            result.GetOutputStream().SetAddressByteSize(addr_byte_size);
            result.GetErrorStream().SetAddressByteSize(addr_byte_size);
            
            if (command.GetArgumentCount() == 0)
            {
                // Dump all sections for all modules images
                const uint32_t num_modules = target->GetImages().GetSize();
                if (num_modules > 0)
                {
                    result.GetOutputStream().Printf("Dumping debug symbols for %u modules.\n", num_modules);
                    for (uint32_t image_idx = 0;  image_idx<num_modules; ++image_idx)
                    {
                        if (DumpModuleSymbolVendor (result.GetOutputStream(), target->GetImages().GetModulePointerAtIndex(image_idx)))
                            num_dumped++;
                    }
                }
                else
                {
                    result.AppendError ("the target has no associated executable images");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            else
            {
                // Dump specified images (by basename or fullpath)
                const char *arg_cstr;
                for (int arg_idx = 0; (arg_cstr = command.GetArgumentAtIndex(arg_idx)) != NULL; ++arg_idx)
                {
                    FileSpec image_file(arg_cstr, false);
                    ModuleList matching_modules;
                    size_t num_matching_modules = target->GetImages().FindModules(&image_file, NULL, NULL, NULL, matching_modules);
                    
                    // Not found in our module list for our target, check the main
                    // shared module list in case it is a extra file used somewhere
                    // else
                    if (num_matching_modules == 0)
                        num_matching_modules = ModuleList::FindSharedModules (image_file, 
                                                                              target->GetArchitecture(), 
                                                                              NULL, 
                                                                              NULL, 
                                                                              matching_modules);
                    
                    if (num_matching_modules > 0)
                    {
                        for (size_t i=0; i<num_matching_modules; ++i)
                        {
                            Module * image_module = matching_modules.GetModulePointerAtIndex(i);
                            if (image_module)
                            {
                                if (DumpModuleSymbolVendor (result.GetOutputStream(), image_module))
                                    num_dumped++;
                            }
                        }
                    }
                    else
                        result.AppendWarningWithFormat("Unable to find an image that matches '%s'.\n", arg_cstr);
                }
            }
            
            if (num_dumped > 0)
                result.SetStatus (eReturnStatusSuccessFinishResult);
            else
            {
                result.AppendError ("no matching executable images found");
                result.SetStatus (eReturnStatusFailed);
            }
        }
        return result.Succeeded();
    }
};


#pragma mark CommandObjectTargetModulesDumpLineTable

//----------------------------------------------------------------------
// Image debug line table dumping command
//----------------------------------------------------------------------

class CommandObjectTargetModulesDumpLineTable : public CommandObjectTargetModulesSourceFileAutoComplete
{
public:
    CommandObjectTargetModulesDumpLineTable (CommandInterpreter &interpreter) :
    CommandObjectTargetModulesSourceFileAutoComplete (interpreter,
                                          "target modules dump line-table",
                                          "Dump the debug symbol file for one or more target modules.",
                                          NULL)
    {
    }
    
    virtual
    ~CommandObjectTargetModulesDumpLineTable ()
    {
    }
    
    virtual bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("invalid target, create a debug target using the 'target create' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else
        {
            ExecutionContext exe_ctx(m_interpreter.GetExecutionContext());
            uint32_t total_num_dumped = 0;
            
            uint32_t addr_byte_size = target->GetArchitecture().GetAddressByteSize();
            result.GetOutputStream().SetAddressByteSize(addr_byte_size);
            result.GetErrorStream().SetAddressByteSize(addr_byte_size);
            
            if (command.GetArgumentCount() == 0)
            {
                result.AppendErrorWithFormat ("\nSyntax: %s\n", m_cmd_syntax.c_str());
                result.SetStatus (eReturnStatusFailed);
            }
            else
            {
                // Dump specified images (by basename or fullpath)
                const char *arg_cstr;
                for (int arg_idx = 0; (arg_cstr = command.GetArgumentAtIndex(arg_idx)) != NULL; ++arg_idx)
                {
                    FileSpec file_spec(arg_cstr, false);
                    const uint32_t num_modules = target->GetImages().GetSize();
                    if (num_modules > 0)
                    {
                        uint32_t num_dumped = 0;
                        for (uint32_t i = 0; i<num_modules; ++i)
                        {
                            if (DumpCompileUnitLineTable (m_interpreter,
                                                          result.GetOutputStream(),
                                                          target->GetImages().GetModulePointerAtIndex(i),
                                                          file_spec,
                                                          exe_ctx.process != NULL && exe_ctx.process->IsAlive()))
                                num_dumped++;
                        }
                        if (num_dumped == 0)
                            result.AppendWarningWithFormat ("No source filenames matched '%s'.\n", arg_cstr);
                        else
                            total_num_dumped += num_dumped;
                    }
                }
            }
            
            if (total_num_dumped > 0)
                result.SetStatus (eReturnStatusSuccessFinishResult);
            else
            {
                result.AppendError ("no source filenames matched any command arguments");
                result.SetStatus (eReturnStatusFailed);
            }
        }
        return result.Succeeded();
    }
};


#pragma mark CommandObjectTargetModulesDump

//----------------------------------------------------------------------
// Dump multi-word command for target modules
//----------------------------------------------------------------------

class CommandObjectTargetModulesDump : public CommandObjectMultiword
{
public:
    
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    CommandObjectTargetModulesDump(CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter, 
                            "target modules dump",
                            "A set of commands for dumping information about one or more target modules.",
                            "target modules dump [symtab|sections|symfile|line-table] [<file1> <file2> ...]")
    {
        LoadSubCommand ("symtab",      CommandObjectSP (new CommandObjectTargetModulesDumpSymtab (interpreter)));
        LoadSubCommand ("sections",    CommandObjectSP (new CommandObjectTargetModulesDumpSections (interpreter)));
        LoadSubCommand ("symfile",     CommandObjectSP (new CommandObjectTargetModulesDumpSymfile (interpreter)));
        LoadSubCommand ("line-table",  CommandObjectSP (new CommandObjectTargetModulesDumpLineTable (interpreter)));
    }
    
    virtual
    ~CommandObjectTargetModulesDump()
    {
    }
};

class CommandObjectTargetModulesAdd : public CommandObject
{
public:
    CommandObjectTargetModulesAdd (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "target modules add",
                   "Add a new module to the current target's modules.",
                   "target modules add [<module>]")
    {
    }
    
    virtual
    ~CommandObjectTargetModulesAdd ()
    {
    }
    
    virtual bool
    Execute (Args& args,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("invalid target, create a debug target using the 'target create' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else
        {
            const size_t argc = args.GetArgumentCount();
            if (argc == 0)
            {
                result.AppendError ("one or more executable image paths must be specified");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
            else
            {
                for (size_t i=0; i<argc; ++i)
                {
                    const char *path = args.GetArgumentAtIndex(i);
                    if (path)
                    {
                        FileSpec file_spec(path, true);
                        ArchSpec arch;
                        if (file_spec.Exists())
                        {
                            ModuleSP module_sp (target->GetSharedModule(file_spec, arch));
                            if (!module_sp)
                            {
                                result.AppendError ("one or more executable image paths must be specified");
                                result.SetStatus (eReturnStatusFailed);
                                return false;
                            }
                        }
                        else
                        {
                            char resolved_path[PATH_MAX];
                            result.SetStatus (eReturnStatusFailed);
                            if (file_spec.GetPath (resolved_path, sizeof(resolved_path)))
                            {
                                if (strcmp (resolved_path, path) != 0)
                                {
                                    result.AppendErrorWithFormat ("invalid module path '%s' with resolved path '%s'\n", path, resolved_path);
                                    break;
                                }
                            }
                            result.AppendErrorWithFormat ("invalid module path '%s'\n", path);
                            break;
                        }
                    }
                }
            }
        }
        return result.Succeeded();
    }

    int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex(cursor_index));
        completion_str.erase (cursor_char_position);
        
        CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter, 
                                                             CommandCompletions::eDiskFileCompletion,
                                                             completion_str.c_str(),
                                                             match_start_point,
                                                             max_return_elements,
                                                             NULL,
                                                             word_complete,
                                                             matches);
        return matches.GetSize();
    }

};

class CommandObjectTargetModulesLoad : public CommandObjectTargetModulesModuleAutoComplete
{
public:
    CommandObjectTargetModulesLoad (CommandInterpreter &interpreter) : 
        CommandObjectTargetModulesModuleAutoComplete (interpreter,
                                                      "target modules load",
                                                      "Set the load addresses for one or more sections in a target module.",
                                                      "target modules load [--file <module> --uuid <uuid>] <sect-name> <address> [<sect-name> <address> ....]"),
        m_option_group (interpreter),
        m_file_option (LLDB_OPT_SET_1, false, "file", 'f', 0, eArgTypePath, "Fullpath or basename for module to load."),
        m_slide_option(LLDB_OPT_SET_1, false, "slide", 's', 0, eArgTypeOffset, "Set the load address for all sections to be the virtual address in the file plus the offset.", 0)
    {
        m_option_group.Append (&m_uuid_option_group, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);
        m_option_group.Append (&m_file_option, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);
        m_option_group.Append (&m_slide_option, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1); 
        m_option_group.Finalize();
    }
    
    virtual
    ~CommandObjectTargetModulesLoad ()
    {
    }
    
    virtual bool
    Execute (Args& args,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("invalid target, create a debug target using the 'target create' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else
        {
            const size_t argc = args.GetArgumentCount();
            const FileSpec *file_ptr = NULL;
            const UUID *uuid_ptr = NULL;
            if (m_file_option.GetOptionValue().OptionWasSet())
                file_ptr = &m_file_option.GetOptionValue().GetCurrentValue();
            
            if (m_uuid_option_group.GetOptionValue().OptionWasSet())
                uuid_ptr = &m_uuid_option_group.GetOptionValue().GetCurrentValue();

            if (file_ptr || uuid_ptr)
            {
                
                ModuleList matching_modules;
                const size_t num_matches = target->GetImages().FindModules (file_ptr,   // File spec to match (can be NULL to match by UUID only)
                                                                            NULL,       // Architecture
                                                                            uuid_ptr,   // UUID to match (can be NULL to not match on UUID)
                                                                            NULL,       // Object name
                                                                            matching_modules);

                char path[PATH_MAX];
                if (num_matches == 1)
                {
                    Module *module = matching_modules.GetModulePointerAtIndex(0);
                    if (module)
                    {
                        ObjectFile *objfile = module->GetObjectFile();
                        if (objfile)
                        {
                            SectionList *section_list = objfile->GetSectionList();
                            if (section_list)
                            {
                                if (argc == 0)
                                {
                                    if (m_slide_option.GetOptionValue().OptionWasSet())
                                    {
                                        Module *module = matching_modules.GetModulePointerAtIndex(0);
                                        if (module)
                                        {
                                            ObjectFile *objfile = module->GetObjectFile();
                                            if (objfile)
                                            {
                                                SectionList *section_list = objfile->GetSectionList();
                                                if (section_list)
                                                {
                                                    const size_t num_sections = section_list->GetSize();
                                                    const addr_t slide = m_slide_option.GetOptionValue().GetCurrentValue();
                                                    for (size_t sect_idx = 0; sect_idx < num_sections; ++sect_idx)
                                                    {
                                                        SectionSP section_sp (section_list->GetSectionAtIndex(sect_idx));
                                                        if (section_sp)
                                                            target->GetSectionLoadList().SetSectionLoadAddress (section_sp.get(), section_sp->GetFileAddress() + slide);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        result.AppendError ("one or more section name + load address pair must be specified");
                                        result.SetStatus (eReturnStatusFailed);
                                        return false;
                                    }
                                }
                                else
                                {
                                    if (m_slide_option.GetOptionValue().OptionWasSet())
                                    {
                                        result.AppendError ("The \"--slide <offset>\" option can't be used in conjunction with setting section load addresses.\n");
                                        result.SetStatus (eReturnStatusFailed);
                                        return false;
                                    }

                                    for (size_t i=0; i<argc; i += 2)
                                    {
                                        const char *sect_name = args.GetArgumentAtIndex(i);
                                        const char *load_addr_cstr = args.GetArgumentAtIndex(i+1);
                                        if (sect_name && load_addr_cstr)
                                        {
                                            ConstString const_sect_name(sect_name);
                                            bool success = false;
                                            addr_t load_addr = Args::StringToUInt64(load_addr_cstr, LLDB_INVALID_ADDRESS, 0, &success);
                                            if (success)
                                            {
                                                SectionSP section_sp (section_list->FindSectionByName(const_sect_name));
                                                if (section_sp)
                                                {
                                                    target->GetSectionLoadList().SetSectionLoadAddress (section_sp.get(), load_addr);
                                                    result.AppendMessageWithFormat("section '%s' loaded at 0x%llx\n", sect_name, load_addr);
                                                }
                                                else
                                                {
                                                    result.AppendErrorWithFormat ("no section found that matches the section name '%s'\n", sect_name);
                                                    result.SetStatus (eReturnStatusFailed);
                                                    break;
                                                }
                                            }
                                            else
                                            {
                                                result.AppendErrorWithFormat ("invalid load address string '%s'\n", load_addr_cstr);
                                                result.SetStatus (eReturnStatusFailed);
                                                break;
                                            }
                                        }
                                        else
                                        {
                                            if (sect_name)
                                                result.AppendError ("section names must be followed by a load address.\n");
                                            else
                                                result.AppendError ("one or more section name + load address pair must be specified.\n");
                                            result.SetStatus (eReturnStatusFailed);
                                            break;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                module->GetFileSpec().GetPath (path, sizeof(path));
                                result.AppendErrorWithFormat ("no sections in object file '%s'\n", path);
                                result.SetStatus (eReturnStatusFailed);
                            }
                        }
                        else
                        {
                            module->GetFileSpec().GetPath (path, sizeof(path));
                            result.AppendErrorWithFormat ("no object file for module '%s'\n", path);
                            result.SetStatus (eReturnStatusFailed);
                        }
                    }
                    else
                    {
                        module->GetFileSpec().GetPath (path, sizeof(path));
                        result.AppendErrorWithFormat ("invalid module '%s'.\n", path);
                        result.SetStatus (eReturnStatusFailed);
                    }
                }
                else
                {
                    char uuid_cstr[64];
                    if (file_ptr)
                        file_ptr->GetPath (path, sizeof(path));
                    else
                        path[0] = '\0';

                    if (uuid_ptr)
                        uuid_ptr->GetAsCString(uuid_cstr, sizeof(uuid_cstr));
                    else
                        uuid_cstr[0] = '\0';
                    if (num_matches > 1)
                    {
                        result.AppendErrorWithFormat ("multiple modules match%s%s%s%s:\n", 
                                                      path[0] ? " file=" : "", 
                                                      path,
                                                      uuid_cstr[0] ? " uuid=" : "", 
                                                      uuid_cstr);
                        for (size_t i=0; i<num_matches; ++i)
                        {
                            if (matching_modules.GetModulePointerAtIndex(i)->GetFileSpec().GetPath (path, sizeof(path)))
                                result.AppendMessageWithFormat("%s\n", path);
                        }
                    }
                    else
                    {
                        result.AppendErrorWithFormat ("no modules were found  that match%s%s%s%s.\n", 
                                                      path[0] ? " file=" : "", 
                                                      path,
                                                      uuid_cstr[0] ? " uuid=" : "", 
                                                      uuid_cstr);
                    }
                    result.SetStatus (eReturnStatusFailed);
                }
            }
            else
            {
                result.AppendError ("either the \"--file <module>\" or the \"--uuid <uuid>\" option must be specified.\n");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }
        return result.Succeeded();
    }    
    
    virtual Options *
    GetOptions ()
    {
        return &m_option_group;
    }
    
protected:
    OptionGroupOptions m_option_group;
    OptionGroupUUID m_uuid_option_group;
    OptionGroupFile m_file_option;
    OptionGroupUInt64 m_slide_option;
};

//----------------------------------------------------------------------
// List images with associated information
//----------------------------------------------------------------------
class CommandObjectTargetModulesList : public CommandObject
{
public:
    
    class CommandOptions : public Options
    {
    public:
        
        CommandOptions (CommandInterpreter &interpreter) :
        Options(interpreter),
        m_format_array()
        {
        }
        
        virtual
        ~CommandOptions ()
        {
        }
        
        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            char short_option = (char) m_getopt_table[option_idx].val;
            uint32_t width = 0;
            if (option_arg)
                width = strtoul (option_arg, NULL, 0);
            m_format_array.push_back(std::make_pair(short_option, width));
            Error error;
            return error;
        }
        
        void
        OptionParsingStarting ()
        {
            m_format_array.clear();
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        
        // Options table: Required for subclasses of Options.
        
        static OptionDefinition g_option_table[];
        
        // Instance variables to hold the values for command options.
        typedef std::vector< std::pair<char, uint32_t> > FormatWidthCollection;
        FormatWidthCollection m_format_array;
    };
    
    CommandObjectTargetModulesList (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "target modules list",
                   "List current executable and dependent shared library images.",
                   "target modules list [<cmd-options>]"),
        m_options (interpreter)
    {
    }
    
    virtual
    ~CommandObjectTargetModulesList ()
    {
    }
    
    virtual
    Options *
    GetOptions ()
    {
        return &m_options;
    }
    
    virtual bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("invalid target, create a debug target using the 'target create' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else
        {
            uint32_t addr_byte_size = target->GetArchitecture().GetAddressByteSize();
            result.GetOutputStream().SetAddressByteSize(addr_byte_size);
            result.GetErrorStream().SetAddressByteSize(addr_byte_size);
            // Dump all sections for all modules images
            const uint32_t num_modules = target->GetImages().GetSize();
            if (num_modules > 0)
            {
                Stream &strm = result.GetOutputStream();
                
                for (uint32_t image_idx = 0; image_idx<num_modules; ++image_idx)
                {
                    Module *module = target->GetImages().GetModulePointerAtIndex(image_idx);
                    strm.Printf("[%3u] ", image_idx);
                    
                    bool dump_object_name = false;
                    if (m_options.m_format_array.empty())
                    {
                        DumpFullpath(strm, &module->GetFileSpec(), 0);
                        dump_object_name = true;
                    }
                    else
                    {
                        const size_t num_entries = m_options.m_format_array.size();
                        for (size_t i=0; i<num_entries; ++i)
                        {
                            if (i > 0)
                                strm.PutChar(' ');
                            char format_char = m_options.m_format_array[i].first;
                            uint32_t width = m_options.m_format_array[i].second;
                            switch (format_char)
                            {
                                case 'a':
                                    DumpModuleArchitecture (strm, module, false, width);
                                    break;
                                    
                                case 't':
                                    DumpModuleArchitecture (strm, module, true, width);
                                    break;
                                    
                                case 'f':
                                    DumpFullpath (strm, &module->GetFileSpec(), width);
                                    dump_object_name = true;
                                    break;
                                    
                                case 'd':
                                    DumpDirectory (strm, &module->GetFileSpec(), width);
                                    break;
                                    
                                case 'b':
                                    DumpBasename (strm, &module->GetFileSpec(), width);
                                    dump_object_name = true;
                                    break;
                                    
                                case 's':
                                case 'S':
                                {
                                    SymbolVendor *symbol_vendor = module->GetSymbolVendor();
                                    if (symbol_vendor)
                                    {
                                        SymbolFile *symbol_file = symbol_vendor->GetSymbolFile();
                                        if (symbol_file)
                                        {
                                            if (format_char == 'S')
                                                DumpBasename(strm, &symbol_file->GetObjectFile()->GetFileSpec(), width);
                                            else
                                                DumpFullpath (strm, &symbol_file->GetObjectFile()->GetFileSpec(), width);
                                            dump_object_name = true;
                                            break;
                                        }
                                    }
                                    strm.Printf("%.*s", width, "<NONE>");
                                }
                                    break;
                                    
                                case 'u':
                                    DumpModuleUUID(strm, module);
                                    break;
                                    
                                default:
                                    break;
                            }
                            
                        }
                    }
                    if (dump_object_name)
                    {
                        const char *object_name = module->GetObjectName().GetCString();
                        if (object_name)
                            strm.Printf ("(%s)", object_name);
                    }
                    strm.EOL();
                }
                result.SetStatus (eReturnStatusSuccessFinishResult);
            }
            else
            {
                result.AppendError ("the target has no associated executable images");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }
        return result.Succeeded();
    }
protected:
    
    CommandOptions m_options;
};

OptionDefinition
CommandObjectTargetModulesList::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_1, false, "arch",       'a', optional_argument, NULL, 0, eArgTypeWidth,   "Display the architecture when listing images."},
    { LLDB_OPT_SET_1, false, "triple",     't', optional_argument, NULL, 0, eArgTypeWidth,   "Display the triple when listing images."},
    { LLDB_OPT_SET_1, false, "uuid",       'u', no_argument,       NULL, 0, eArgTypeNone,    "Display the UUID when listing images."},
    { LLDB_OPT_SET_1, false, "fullpath",   'f', optional_argument, NULL, 0, eArgTypeWidth,   "Display the fullpath to the image object file."},
    { LLDB_OPT_SET_1, false, "directory",  'd', optional_argument, NULL, 0, eArgTypeWidth,   "Display the directory with optional width for the image object file."},
    { LLDB_OPT_SET_1, false, "basename",   'b', optional_argument, NULL, 0, eArgTypeWidth,   "Display the basename with optional width for the image object file."},
    { LLDB_OPT_SET_1, false, "symfile",    's', optional_argument, NULL, 0, eArgTypeWidth,   "Display the fullpath to the image symbol file with optional width."},
    { LLDB_OPT_SET_1, false, "symfile-basename", 'S', optional_argument, NULL, 0, eArgTypeWidth,   "Display the basename to the image symbol file with optional width."},
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};



//----------------------------------------------------------------------
// Lookup information in images
//----------------------------------------------------------------------
class CommandObjectTargetModulesLookup : public CommandObject
{
public:
    
    enum
    {
        eLookupTypeInvalid = -1,
        eLookupTypeAddress = 0,
        eLookupTypeSymbol,
        eLookupTypeFileLine,    // Line is optional
        eLookupTypeFunction,
        eLookupTypeType,
        kNumLookupTypes
    };
    
    class CommandOptions : public Options
    {
    public:
        
        CommandOptions (CommandInterpreter &interpreter) :
        Options(interpreter)
        {
            OptionParsingStarting();
        }
        
        virtual
        ~CommandOptions ()
        {
        }
        
        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            
            char short_option = (char) m_getopt_table[option_idx].val;
            
            switch (short_option)
            {
                case 'a':
                    m_type = eLookupTypeAddress;
                    m_addr = Args::StringToUInt64(option_arg, LLDB_INVALID_ADDRESS);
                    if (m_addr == LLDB_INVALID_ADDRESS)
                        error.SetErrorStringWithFormat ("Invalid address string '%s'.\n", option_arg);
                    break;
                    
                case 'o':
                    m_offset = Args::StringToUInt64(option_arg, LLDB_INVALID_ADDRESS);
                    if (m_offset == LLDB_INVALID_ADDRESS)
                        error.SetErrorStringWithFormat ("Invalid offset string '%s'.\n", option_arg);
                    break;
                    
                case 's':
                    m_str = option_arg;
                    m_type = eLookupTypeSymbol;
                    break;
                    
                case 'f':
                    m_file.SetFile (option_arg, false);
                    m_type = eLookupTypeFileLine;
                    break;
                    
                case 'i':
                    m_check_inlines = false;
                    break;
                    
                case 'l':
                    m_line_number = Args::StringToUInt32(option_arg, UINT32_MAX);
                    if (m_line_number == UINT32_MAX)
                        error.SetErrorStringWithFormat ("Invalid line number string '%s'.\n", option_arg);
                    else if (m_line_number == 0)
                        error.SetErrorString ("Zero is an invalid line number.");
                    m_type = eLookupTypeFileLine;
                    break;
                    
                case 'n':
                    m_str = option_arg;
                    m_type = eLookupTypeFunction;
                    break;
                    
                case 't':
                    m_str = option_arg;
                    m_type = eLookupTypeType;
                    break;
                    
                case 'v':
                    m_verbose = 1;
                    break;
                    
                case 'r':
                    m_use_regex = true;
                    break;
            }
            
            return error;
        }
        
        void
        OptionParsingStarting ()
        {
            m_type = eLookupTypeInvalid;
            m_str.clear();
            m_file.Clear();
            m_addr = LLDB_INVALID_ADDRESS;
            m_offset = 0;
            m_line_number = 0;
            m_use_regex = false;
            m_check_inlines = true;
            m_verbose = false;
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        
        // Options table: Required for subclasses of Options.
        
        static OptionDefinition g_option_table[];
        int             m_type;         // Should be a eLookupTypeXXX enum after parsing options
        std::string     m_str;          // Holds name lookup
        FileSpec        m_file;         // Files for file lookups
        lldb::addr_t    m_addr;         // Holds the address to lookup
        lldb::addr_t    m_offset;       // Subtract this offset from m_addr before doing lookups.
        uint32_t        m_line_number;  // Line number for file+line lookups
        bool            m_use_regex;    // Name lookups in m_str are regular expressions.
        bool            m_check_inlines;// Check for inline entries when looking up by file/line.
        bool            m_verbose;      // Enable verbose lookup info
        
    };
    
    CommandObjectTargetModulesLookup (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "target modules lookup",
                   "Look up information within executable and dependent shared library images.",
                   NULL),
    m_options (interpreter)
    {
        CommandArgumentEntry arg;
        CommandArgumentData file_arg;
        
        // Define the first (and only) variant of this arg.
        file_arg.arg_type = eArgTypeFilename;
        file_arg.arg_repetition = eArgRepeatStar;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (file_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }
    
    virtual
    ~CommandObjectTargetModulesLookup ()
    {
    }
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }
    
    
    bool
    LookupInModule (CommandInterpreter &interpreter, Module *module, CommandReturnObject &result, bool &syntax_error)
    {
        switch (m_options.m_type)
        {
            case eLookupTypeAddress:
                if (m_options.m_addr != LLDB_INVALID_ADDRESS)
                {
                    if (LookupAddressInModule (m_interpreter, 
                                               result.GetOutputStream(), 
                                               module, 
                                               eSymbolContextEverything, 
                                               m_options.m_addr, 
                                               m_options.m_offset,
                                               m_options.m_verbose))
                    {
                        result.SetStatus(eReturnStatusSuccessFinishResult);
                        return true;
                    }
                }
                break;
                
            case eLookupTypeSymbol:
                if (!m_options.m_str.empty())
                {
                    if (LookupSymbolInModule (m_interpreter, result.GetOutputStream(), module, m_options.m_str.c_str(), m_options.m_use_regex))
                    {
                        result.SetStatus(eReturnStatusSuccessFinishResult);
                        return true;
                    }
                }
                break;
                
            case eLookupTypeFileLine:
                if (m_options.m_file)
                {
                    
                    if (LookupFileAndLineInModule (m_interpreter,
                                                   result.GetOutputStream(),
                                                   module,
                                                   m_options.m_file,
                                                   m_options.m_line_number,
                                                   m_options.m_check_inlines,
                                                   m_options.m_verbose))
                    {
                        result.SetStatus(eReturnStatusSuccessFinishResult);
                        return true;
                    }
                }
                break;
                
            case eLookupTypeFunction:
                if (!m_options.m_str.empty())
                {
                    if (LookupFunctionInModule (m_interpreter,
                                                result.GetOutputStream(),
                                                module,
                                                m_options.m_str.c_str(),
                                                m_options.m_use_regex,
                                                m_options.m_verbose))
                    {
                        result.SetStatus(eReturnStatusSuccessFinishResult);
                        return true;
                    }
                }
                break;
                
            case eLookupTypeType:
                if (!m_options.m_str.empty())
                {
                    if (LookupTypeInModule (m_interpreter,
                                            result.GetOutputStream(),
                                            module,
                                            m_options.m_str.c_str(),
                                            m_options.m_use_regex))
                    {
                        result.SetStatus(eReturnStatusSuccessFinishResult);
                        return true;
                    }
                }
                break;
                
            default:
                m_options.GenerateOptionUsage (result.GetErrorStream(), this);
                syntax_error = true;
                break;
        }
        
        result.SetStatus (eReturnStatusFailed);
        return false;
    }
    
    virtual bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("invalid target, create a debug target using the 'target create' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else
        {
            bool syntax_error = false;
            uint32_t i;
            uint32_t num_successful_lookups = 0;
            uint32_t addr_byte_size = target->GetArchitecture().GetAddressByteSize();
            result.GetOutputStream().SetAddressByteSize(addr_byte_size);
            result.GetErrorStream().SetAddressByteSize(addr_byte_size);
            // Dump all sections for all modules images
            
            if (command.GetArgumentCount() == 0)
            {
                // Dump all sections for all modules images
                const uint32_t num_modules = target->GetImages().GetSize();
                if (num_modules > 0)
                {
                    for (i = 0; i<num_modules && syntax_error == false; ++i)
                    {
                        if (LookupInModule (m_interpreter, target->GetImages().GetModulePointerAtIndex(i), result, syntax_error))
                        {
                            result.GetOutputStream().EOL();
                            num_successful_lookups++;
                        }
                    }
                }
                else
                {
                    result.AppendError ("the target has no associated executable images");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            else
            {
                // Dump specified images (by basename or fullpath)
                const char *arg_cstr;
                for (i = 0; (arg_cstr = command.GetArgumentAtIndex(i)) != NULL && syntax_error == false; ++i)
                {
                    FileSpec image_file(arg_cstr, false);
                    ModuleList matching_modules;
                    size_t num_matching_modules = target->GetImages().FindModules(&image_file, NULL, NULL, NULL, matching_modules);
                    
                    // Not found in our module list for our target, check the main
                    // shared module list in case it is a extra file used somewhere
                    // else
                    if (num_matching_modules == 0)
                        num_matching_modules = ModuleList::FindSharedModules (image_file, 
                                                                              target->GetArchitecture(), 
                                                                              NULL, 
                                                                              NULL, 
                                                                              matching_modules);
                    
                    if (num_matching_modules > 0)
                    {
                        for (size_t j=0; j<num_matching_modules; ++j)
                        {
                            Module * image_module = matching_modules.GetModulePointerAtIndex(j);
                            if (image_module)
                            {
                                if (LookupInModule (m_interpreter, image_module, result, syntax_error))
                                {
                                    result.GetOutputStream().EOL();
                                    num_successful_lookups++;
                                }
                            }
                        }
                    }
                    else
                        result.AppendWarningWithFormat("Unable to find an image that matches '%s'.\n", arg_cstr);
                }
            }
            
            if (num_successful_lookups > 0)
                result.SetStatus (eReturnStatusSuccessFinishResult);
            else
                result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
protected:
    
    CommandOptions m_options;
};

OptionDefinition
CommandObjectTargetModulesLookup::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_1,   true,  "address",    'a', required_argument, NULL, 0, eArgTypeAddress,      "Lookup an address in one or more target modules."},
    { LLDB_OPT_SET_1,   false, "offset",     'o', required_argument, NULL, 0, eArgTypeOffset,       "When looking up an address subtract <offset> from any addresses before doing the lookup."},
    { LLDB_OPT_SET_2| LLDB_OPT_SET_4  
      /* FIXME: re-enable this for types when the LookupTypeInModule actually uses the regex option: | LLDB_OPT_SET_5 */ ,
                        false, "regex",      'r', no_argument,       NULL, 0, eArgTypeNone,         "The <name> argument for name lookups are regular expressions."},
    { LLDB_OPT_SET_2,   true,  "symbol",     's', required_argument, NULL, 0, eArgTypeSymbol,       "Lookup a symbol by name in the symbol tables in one or more target modules."},
    { LLDB_OPT_SET_3,   true,  "file",       'f', required_argument, NULL, 0, eArgTypeFilename,     "Lookup a file by fullpath or basename in one or more target modules."},
    { LLDB_OPT_SET_3,   false, "line",       'l', required_argument, NULL, 0, eArgTypeLineNum,      "Lookup a line number in a file (must be used in conjunction with --file)."},
    { LLDB_OPT_SET_3,   false, "no-inlines", 'i', no_argument,       NULL, 0, eArgTypeNone,         "Check inline line entries (must be used in conjunction with --file)."},
    { LLDB_OPT_SET_4,   true,  "function",   'n', required_argument, NULL, 0, eArgTypeFunctionName, "Lookup a function by name in the debug symbols in one or more target modules."},
    { LLDB_OPT_SET_5,   true,  "type",       't', required_argument, NULL, 0, eArgTypeName,         "Lookup a type by name in the debug symbols in one or more target modules."},
    { LLDB_OPT_SET_ALL, false, "verbose",    'v', no_argument,       NULL, 0, eArgTypeNone,         "Enable verbose lookup information."},
    { 0, false, NULL,           0, 0,                 NULL, 0, eArgTypeNone, NULL }
};


#pragma mark CommandObjectMultiwordImageSearchPaths

//-------------------------------------------------------------------------
// CommandObjectMultiwordImageSearchPaths
//-------------------------------------------------------------------------

class CommandObjectTargetModulesImageSearchPaths : public CommandObjectMultiword
{
public:
    
    CommandObjectTargetModulesImageSearchPaths (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter, 
                            "target modules search-paths",
                            "A set of commands for operating on debugger target image search paths.",
                            "target modules search-paths <subcommand> [<subcommand-options>]")
    {
        LoadSubCommand ("add",     CommandObjectSP (new CommandObjectTargetModulesSearchPathsAdd (interpreter)));
        LoadSubCommand ("clear",   CommandObjectSP (new CommandObjectTargetModulesSearchPathsClear (interpreter)));
        LoadSubCommand ("insert",  CommandObjectSP (new CommandObjectTargetModulesSearchPathsInsert (interpreter)));
        LoadSubCommand ("list",    CommandObjectSP (new CommandObjectTargetModulesSearchPathsList (interpreter)));
        LoadSubCommand ("query",   CommandObjectSP (new CommandObjectTargetModulesSearchPathsQuery (interpreter)));
    }
    
    ~CommandObjectTargetModulesImageSearchPaths()
    {
    }
};



#pragma mark CommandObjectTargetModules

//-------------------------------------------------------------------------
// CommandObjectTargetModules
//-------------------------------------------------------------------------

class CommandObjectTargetModules : public CommandObjectMultiword
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    CommandObjectTargetModules(CommandInterpreter &interpreter) :
        CommandObjectMultiword (interpreter,
                                "target modules",
                                "A set of commands for accessing information for one or more target modules.",
                                "target modules <sub-command> ...")
    {
        LoadSubCommand ("add",          CommandObjectSP (new CommandObjectTargetModulesAdd (interpreter)));
        LoadSubCommand ("load",         CommandObjectSP (new CommandObjectTargetModulesLoad (interpreter)));
        //LoadSubCommand ("unload",       CommandObjectSP (new CommandObjectTargetModulesUnload (interpreter)));
        LoadSubCommand ("dump",         CommandObjectSP (new CommandObjectTargetModulesDump (interpreter)));
        LoadSubCommand ("list",         CommandObjectSP (new CommandObjectTargetModulesList (interpreter)));
        LoadSubCommand ("lookup",       CommandObjectSP (new CommandObjectTargetModulesLookup (interpreter)));
        LoadSubCommand ("search-paths", CommandObjectSP (new CommandObjectTargetModulesImageSearchPaths (interpreter)));

    }
    virtual
    ~CommandObjectTargetModules()
    {
    }
    
private:
    //------------------------------------------------------------------
    // For CommandObjectTargetModules only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (CommandObjectTargetModules);
};


#pragma mark CommandObjectTargetStopHookAdd

//-------------------------------------------------------------------------
// CommandObjectTargetStopHookAdd
//-------------------------------------------------------------------------

class CommandObjectTargetStopHookAdd : public CommandObject
{
public:

    class CommandOptions : public Options
    {
    public:
        CommandOptions (CommandInterpreter &interpreter) :
            Options(interpreter),
            m_line_start(0),
            m_line_end (UINT_MAX),
            m_func_name_type_mask (eFunctionNameTypeAuto),
            m_sym_ctx_specified (false),
            m_thread_specified (false),
            m_use_one_liner (false),
            m_one_liner()
        {
        }
        
        ~CommandOptions () {}
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;
            bool success;

            switch (short_option)
            {
                case 'c':
                    m_class_name = option_arg;
                    m_sym_ctx_specified = true;
                break;
                
                case 'e':
                    m_line_end = Args::StringToUInt32 (option_arg, UINT_MAX, 0, &success);
                    if (!success)
                    {
                        error.SetErrorStringWithFormat ("Invalid end line number: \"%s\".", option_arg);
                        break;
                    }
                    m_sym_ctx_specified = true;
                break;
                
                case 'l':
                    m_line_start = Args::StringToUInt32 (option_arg, 0, 0, &success);
                    if (!success)
                    {
                        error.SetErrorStringWithFormat ("Invalid start line number: \"%s\".", option_arg);
                        break;
                    }
                    m_sym_ctx_specified = true;
                break;
                
                case 'n':
                    m_function_name = option_arg;
                    m_func_name_type_mask |= eFunctionNameTypeAuto;
                    m_sym_ctx_specified = true;
                break;
                
                case 'f':
                    m_file_name = option_arg;
                    m_sym_ctx_specified = true;
                break;
                case 's':
                    m_module_name = option_arg;
                    m_sym_ctx_specified = true;
                break;
                case 't' :
                {
                    m_thread_id = Args::StringToUInt64(option_arg, LLDB_INVALID_THREAD_ID, 0);
                    if (m_thread_id == LLDB_INVALID_THREAD_ID)
                       error.SetErrorStringWithFormat ("Invalid thread id string '%s'.\n", option_arg);
                    m_thread_specified = true;
                }
                break;
                case 'T':
                    m_thread_name = option_arg;
                    m_thread_specified = true;
                break;
                case 'q':
                    m_queue_name = option_arg;
                    m_thread_specified = true;
                    break;
                case 'x':
                {
                    m_thread_index = Args::StringToUInt32(option_arg, UINT32_MAX, 0);
                    if (m_thread_id == UINT32_MAX)
                       error.SetErrorStringWithFormat ("Invalid thread index string '%s'.\n", option_arg);
                    m_thread_specified = true;
                }
                break;
                case 'o':
                    m_use_one_liner = true;
                    m_one_liner = option_arg;
                break;
                default:
                    error.SetErrorStringWithFormat ("Unrecognized option %c.");
                break;
            }
            return error;
        }

        void
        OptionParsingStarting ()
        {
            m_class_name.clear();
            m_function_name.clear();
            m_line_start = 0;
            m_line_end = UINT_MAX;
            m_file_name.clear();
            m_module_name.clear();
            m_func_name_type_mask = eFunctionNameTypeAuto;
            m_thread_id = LLDB_INVALID_THREAD_ID;
            m_thread_index = UINT32_MAX;
            m_thread_name.clear();
            m_queue_name.clear();

            m_sym_ctx_specified = false;
            m_thread_specified = false;

            m_use_one_liner = false;
            m_one_liner.clear();
        }

        
        static OptionDefinition g_option_table[];
        
        std::string m_class_name;
        std::string m_function_name;
        uint32_t    m_line_start;
        uint32_t    m_line_end;
        std::string m_file_name;
        std::string m_module_name;
        uint32_t m_func_name_type_mask;  // A pick from lldb::FunctionNameType.
        lldb::tid_t m_thread_id;
        uint32_t m_thread_index;
        std::string m_thread_name;
        std::string m_queue_name;
        bool        m_sym_ctx_specified;
        bool        m_thread_specified;
        // Instance variables to hold the values for one_liner options.
        bool m_use_one_liner;
        std::string m_one_liner;
    };

    Options *
    GetOptions ()
    {
        return &m_options;
    }

    CommandObjectTargetStopHookAdd (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target stop-hook add ",
                       "Add a hook to be executed when the target stops.",
                       "target stop-hook add"),
        m_options (interpreter)
    {
    }

    ~CommandObjectTargetStopHookAdd ()
    {
    }

    static size_t 
    ReadCommandsCallbackFunction (void *baton, 
                                  InputReader &reader, 
                                  lldb::InputReaderAction notification,
                                  const char *bytes, 
                                  size_t bytes_len)
    {
        StreamSP out_stream = reader.GetDebugger().GetAsyncOutputStream();
        Target::StopHook *new_stop_hook = ((Target::StopHook *) baton);
        static bool got_interrupted;
        bool batch_mode = reader.GetDebugger().GetCommandInterpreter().GetBatchCommandMode();

        switch (notification)
        {
        case eInputReaderActivate:
            if (!batch_mode)
            {
                out_stream->Printf ("%s\n", "Enter your stop hook command(s).  Type 'DONE' to end.");
                if (reader.GetPrompt())
                    out_stream->Printf ("%s", reader.GetPrompt());
                out_stream->Flush();
            }
            got_interrupted = false;
            break;

        case eInputReaderDeactivate:
            break;

        case eInputReaderReactivate:
            if (reader.GetPrompt() && !batch_mode)
            {
                out_stream->Printf ("%s", reader.GetPrompt());
                out_stream->Flush();
            }
            got_interrupted = false;
            break;

        case eInputReaderAsynchronousOutputWritten:
            break;
            
        case eInputReaderGotToken:
            if (bytes && bytes_len && baton)
            {
                StringList *commands = new_stop_hook->GetCommandPointer();
                if (commands)
                {
                    commands->AppendString (bytes, bytes_len); 
                }
            }
            if (!reader.IsDone() && reader.GetPrompt() && !batch_mode)
            {
                out_stream->Printf ("%s", reader.GetPrompt());
                out_stream->Flush();
            }
            break;
            
        case eInputReaderInterrupt:
            {
                // Finish, and cancel the stop hook.
                new_stop_hook->GetTarget()->RemoveStopHookByID(new_stop_hook->GetID());
                if (!batch_mode)
                {
                    out_stream->Printf ("Stop hook cancelled.\n");
                    out_stream->Flush();
                }
                
                reader.SetIsDone (true);
            }
            got_interrupted = true;
            break;
            
        case eInputReaderEndOfFile:
            reader.SetIsDone (true);
            break;
            
        case eInputReaderDone:
            if (!got_interrupted && !batch_mode)
            {
                out_stream->Printf ("Stop hook #%d added.\n", new_stop_hook->GetID());
                out_stream->Flush();
            }
            break;
        }

        return bytes_len;
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            Target::StopHookSP new_hook_sp;
            target->AddStopHook (new_hook_sp);

            //  First step, make the specifier.
            std::auto_ptr<SymbolContextSpecifier> specifier_ap;
            if (m_options.m_sym_ctx_specified)
            {
                specifier_ap.reset(new SymbolContextSpecifier(m_interpreter.GetDebugger().GetSelectedTarget()));
                
                if (!m_options.m_module_name.empty())
                {
                    specifier_ap->AddSpecification (m_options.m_module_name.c_str(), SymbolContextSpecifier::eModuleSpecified);
                }
                
                if (!m_options.m_class_name.empty())
                {
                    specifier_ap->AddSpecification (m_options.m_class_name.c_str(), SymbolContextSpecifier::eClassOrNamespaceSpecified);
                }
                
                if (!m_options.m_file_name.empty())
                {
                    specifier_ap->AddSpecification (m_options.m_file_name.c_str(), SymbolContextSpecifier::eFileSpecified);
                }
                
                if (m_options.m_line_start != 0)
                {
                    specifier_ap->AddLineSpecification (m_options.m_line_start, SymbolContextSpecifier::eLineStartSpecified);
                }
                
                if (m_options.m_line_end != UINT_MAX)
                {
                    specifier_ap->AddLineSpecification (m_options.m_line_end, SymbolContextSpecifier::eLineEndSpecified);
                }
                
                if (!m_options.m_function_name.empty())
                {
                    specifier_ap->AddSpecification (m_options.m_function_name.c_str(), SymbolContextSpecifier::eFunctionSpecified);
                }
            }
            
            if (specifier_ap.get())
                new_hook_sp->SetSpecifier (specifier_ap.release());

            // Next see if any of the thread options have been entered:
            
            if (m_options.m_thread_specified)
            {
                ThreadSpec *thread_spec = new ThreadSpec();
                
                if (m_options.m_thread_id != LLDB_INVALID_THREAD_ID)
                {
                    thread_spec->SetTID (m_options.m_thread_id);
                }
                
                if (m_options.m_thread_index != UINT32_MAX)
                    thread_spec->SetIndex (m_options.m_thread_index);
                
                if (!m_options.m_thread_name.empty())
                    thread_spec->SetName (m_options.m_thread_name.c_str());
                
                if (!m_options.m_queue_name.empty())
                    thread_spec->SetQueueName (m_options.m_queue_name.c_str());
                    
                new_hook_sp->SetThreadSpecifier (thread_spec);
            
            }
            if (m_options.m_use_one_liner)
            {
                // Use one-liner.
                new_hook_sp->GetCommandPointer()->AppendString (m_options.m_one_liner.c_str());
                result.AppendMessageWithFormat("Stop hook #%d added.\n", new_hook_sp->GetID());
            }
            else
            {
                // Otherwise gather up the command list, we'll push an input reader and suck the data from that directly into
                // the new stop hook's command string.
                InputReaderSP reader_sp (new InputReader(m_interpreter.GetDebugger()));
                if (!reader_sp)
                {
                    result.AppendError("out of memory\n");
                    result.SetStatus (eReturnStatusFailed);
                    target->RemoveStopHookByID (new_hook_sp->GetID());
                    return false;
                }
                
                Error err (reader_sp->Initialize (CommandObjectTargetStopHookAdd::ReadCommandsCallbackFunction,
                                                  new_hook_sp.get(), // baton
                                                  eInputReaderGranularityLine,  // token size, to pass to callback function
                                                  "DONE",                       // end token
                                                  "> ",                         // prompt
                                                  true));                       // echo input
                if (!err.Success())
                {
                    result.AppendError (err.AsCString());
                    result.SetStatus (eReturnStatusFailed);
                    target->RemoveStopHookByID (new_hook_sp->GetID());
                    return false;
                }
                m_interpreter.GetDebugger().PushInputReader (reader_sp);
            }
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            result.AppendError ("invalid target\n");
            result.SetStatus (eReturnStatusFailed);
        }
        
        return result.Succeeded();
    }
private:
    CommandOptions m_options;
};

OptionDefinition
CommandObjectTargetStopHookAdd::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "one-liner", 'o', required_argument, NULL, NULL, eArgTypeOneLiner,
        "Specify a one-line breakpoint command inline. Be sure to surround it with quotes." },
    { LLDB_OPT_SET_ALL, false, "shlib", 's', required_argument, NULL, CommandCompletions::eModuleCompletion, eArgTypeShlibName,
        "Set the module within which the stop-hook is to be run."},
    { LLDB_OPT_SET_ALL, false, "thread-index", 'x', required_argument, NULL, NULL, eArgTypeThreadIndex,
        "The stop hook is run only for the thread whose index matches this argument."},
    { LLDB_OPT_SET_ALL, false, "thread-id", 't', required_argument, NULL, NULL, eArgTypeThreadID,
        "The stop hook is run only for the thread whose TID matches this argument."},
    { LLDB_OPT_SET_ALL, false, "thread-name", 'T', required_argument, NULL, NULL, eArgTypeThreadName,
        "The stop hook is run only for the thread whose thread name matches this argument."},
    { LLDB_OPT_SET_ALL, false, "queue-name", 'q', required_argument, NULL, NULL, eArgTypeQueueName,
        "The stop hook is run only for threads in the queue whose name is given by this argument."},
    { LLDB_OPT_SET_1, false, "file", 'f', required_argument, NULL, CommandCompletions::eSourceFileCompletion, eArgTypeFilename,
        "Specify the source file within which the stop-hook is to be run." },
    { LLDB_OPT_SET_1, false, "start-line", 'l', required_argument, NULL, 0, eArgTypeLineNum,
        "Set the start of the line range for which the stop-hook is to be run."},
    { LLDB_OPT_SET_1, false, "end-line", 'e', required_argument, NULL, 0, eArgTypeLineNum,
        "Set the end of the line range for which the stop-hook is to be run."},
    { LLDB_OPT_SET_2, false, "classname", 'c', required_argument, NULL, NULL, eArgTypeClassName,
        "Specify the class within which the stop-hook is to be run." },
    { LLDB_OPT_SET_3, false, "name", 'n', required_argument, NULL, CommandCompletions::eSymbolCompletion, eArgTypeFunctionName,
        "Set the function name within which the stop hook will be run." },
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

#pragma mark CommandObjectTargetStopHookDelete

//-------------------------------------------------------------------------
// CommandObjectTargetStopHookDelete
//-------------------------------------------------------------------------

class CommandObjectTargetStopHookDelete : public CommandObject
{
public:

    CommandObjectTargetStopHookDelete (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target stop-hook delete [<id>]",
                       "Delete a stop-hook.",
                       "target stop-hook delete")
    {
    }

    ~CommandObjectTargetStopHookDelete ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            // FIXME: see if we can use the breakpoint id style parser?
            size_t num_args = command.GetArgumentCount();
            if (num_args == 0)
            {
                if (!m_interpreter.Confirm ("Delete all stop hooks?", true))
                {
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                else
                {
                    target->RemoveAllStopHooks();
                }
            }
            else
            {
                bool success;
                for (size_t i = 0; i < num_args; i++)
                {
                    lldb::user_id_t user_id = Args::StringToUInt32 (command.GetArgumentAtIndex(i), 0, 0, &success);
                    if (!success)
                    {
                        result.AppendErrorWithFormat ("invalid stop hook id: \"%s\".\n", command.GetArgumentAtIndex(i));
                        result.SetStatus(eReturnStatusFailed);
                        return false;
                    }
                    success = target->RemoveStopHookByID (user_id);
                    if (!success)
                    {
                        result.AppendErrorWithFormat ("unknown stop hook id: \"%s\".\n", command.GetArgumentAtIndex(i));
                        result.SetStatus(eReturnStatusFailed);
                        return false;
                    }
                }
            }
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            result.AppendError ("invalid target\n");
            result.SetStatus (eReturnStatusFailed);
        }
        
        return result.Succeeded();
    }
};
#pragma mark CommandObjectTargetStopHookEnableDisable

//-------------------------------------------------------------------------
// CommandObjectTargetStopHookEnableDisable
//-------------------------------------------------------------------------

class CommandObjectTargetStopHookEnableDisable : public CommandObject
{
public:

    CommandObjectTargetStopHookEnableDisable (CommandInterpreter &interpreter, bool enable, const char *name, const char *help, const char *syntax) :
        CommandObject (interpreter,
                       name,
                       help,
                       syntax),
        m_enable (enable)
    {
    }

    ~CommandObjectTargetStopHookEnableDisable ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            // FIXME: see if we can use the breakpoint id style parser?
            size_t num_args = command.GetArgumentCount();
            bool success;
            
            if (num_args == 0)
            {
                target->SetAllStopHooksActiveState (m_enable);
            }
            else
            {
                for (size_t i = 0; i < num_args; i++)
                {
                    lldb::user_id_t user_id = Args::StringToUInt32 (command.GetArgumentAtIndex(i), 0, 0, &success);
                    if (!success)
                    {
                        result.AppendErrorWithFormat ("invalid stop hook id: \"%s\".\n", command.GetArgumentAtIndex(i));
                        result.SetStatus(eReturnStatusFailed);
                        return false;
                    }
                    success = target->SetStopHookActiveStateByID (user_id, m_enable);
                    if (!success)
                    {
                        result.AppendErrorWithFormat ("unknown stop hook id: \"%s\".\n", command.GetArgumentAtIndex(i));
                        result.SetStatus(eReturnStatusFailed);
                        return false;
                    }
                }
            }
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            result.AppendError ("invalid target\n");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
private:
    bool m_enable;
};

#pragma mark CommandObjectTargetStopHookList

//-------------------------------------------------------------------------
// CommandObjectTargetStopHookList
//-------------------------------------------------------------------------

class CommandObjectTargetStopHookList : public CommandObject
{
public:

    CommandObjectTargetStopHookList (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target stop-hook list [<type>]",
                       "List all stop-hooks.",
                       "target stop-hook list")
    {
    }

    ~CommandObjectTargetStopHookList ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            bool notify = true;
            target->GetImageSearchPathList().Clear(notify);
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            result.AppendError ("invalid target\n");
            result.SetStatus (eReturnStatusFailed);
        }
        
        size_t num_hooks = target->GetNumStopHooks ();
        if (num_hooks == 0)
        {
            result.GetOutputStream().PutCString ("No stop hooks.\n");
        }
        else
        {
            for (size_t i = 0; i < num_hooks; i++)
            {
                Target::StopHookSP this_hook = target->GetStopHookAtIndex (i);
                if (i > 0)
                    result.GetOutputStream().PutCString ("\n");
                this_hook->GetDescription (&(result.GetOutputStream()), eDescriptionLevelFull);
            }
        }
        return result.Succeeded();
    }
};

#pragma mark CommandObjectMultiwordTargetStopHooks
//-------------------------------------------------------------------------
// CommandObjectMultiwordTargetStopHooks
//-------------------------------------------------------------------------

class CommandObjectMultiwordTargetStopHooks : public CommandObjectMultiword
{
public:

    CommandObjectMultiwordTargetStopHooks (CommandInterpreter &interpreter) :
        CommandObjectMultiword (interpreter, 
                                "target stop-hook",
                                "A set of commands for operating on debugger target stop-hooks.",
                                "target stop-hook <subcommand> [<subcommand-options>]")
    {
        LoadSubCommand ("add",      CommandObjectSP (new CommandObjectTargetStopHookAdd (interpreter)));
        LoadSubCommand ("delete",   CommandObjectSP (new CommandObjectTargetStopHookDelete (interpreter)));
        LoadSubCommand ("disable",  CommandObjectSP (new CommandObjectTargetStopHookEnableDisable (interpreter, 
                                                                                                   false, 
                                                                                                   "target stop-hook disable [<id>]",
                                                                                                   "Disable a stop-hook.",
                                                                                                   "target stop-hook disable")));
        LoadSubCommand ("enable",   CommandObjectSP (new CommandObjectTargetStopHookEnableDisable (interpreter, 
                                                                                                   true, 
                                                                                                   "target stop-hook enable [<id>]",
                                                                                                   "Enable a stop-hook.",
                                                                                                   "target stop-hook enable")));
        LoadSubCommand ("list",     CommandObjectSP (new CommandObjectTargetStopHookList (interpreter)));
    }

    ~CommandObjectMultiwordTargetStopHooks()
    {
    }
};



#pragma mark CommandObjectMultiwordTarget

//-------------------------------------------------------------------------
// CommandObjectMultiwordTarget
//-------------------------------------------------------------------------

CommandObjectMultiwordTarget::CommandObjectMultiwordTarget (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "target",
                            "A set of commands for operating on debugger targets.",
                            "target <subcommand> [<subcommand-options>]")
{
    
    LoadSubCommand ("create",    CommandObjectSP (new CommandObjectTargetCreate (interpreter)));
    LoadSubCommand ("list",      CommandObjectSP (new CommandObjectTargetList   (interpreter)));
    LoadSubCommand ("select",    CommandObjectSP (new CommandObjectTargetSelect (interpreter)));
    LoadSubCommand ("stop-hook", CommandObjectSP (new CommandObjectMultiwordTargetStopHooks (interpreter)));
    LoadSubCommand ("modules",   CommandObjectSP (new CommandObjectTargetModules (interpreter)));
    LoadSubCommand ("variable",  CommandObjectSP (new CommandObjectTargetVariable (interpreter)));
}

CommandObjectMultiwordTarget::~CommandObjectMultiwordTarget ()
{
}


