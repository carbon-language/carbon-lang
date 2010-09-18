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
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Options.h"
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

#include "CommandObjectThread.h"

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
        ExecutionContext exe_ctx(m_interpreter.GetDebugger().GetExecutionContext());
        if (exe_ctx.frame)
        {
            exe_ctx.frame->Dump (&result.GetOutputStream(), true, false);
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

    CommandObjectFrameSelect (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "frame select",
                       "Select a frame by index from within the current thread and make it the current frame.",
                       "frame select <frame-index>",
                       eFlagProcessMustBeLaunched | eFlagProcessMustBePaused)
    {
    }

    ~CommandObjectFrameSelect ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        ExecutionContext exe_ctx (m_interpreter.GetDebugger().GetExecutionContext());
        if (exe_ctx.thread)
        {
            if (command.GetArgumentCount() == 1)
            {
                const char *frame_idx_cstr = command.GetArgumentAtIndex(0);

                const uint32_t num_frames = exe_ctx.thread->GetStackFrameCount();
                const uint32_t frame_idx = Args::StringToUInt32 (frame_idx_cstr, UINT32_MAX, 0);
                if (frame_idx < num_frames)
                {
                    exe_ctx.thread->SetSelectedFrameByIndex (frame_idx);
                    exe_ctx.frame = exe_ctx.thread->GetSelectedFrame ().get();

                    if (exe_ctx.frame)
                    {
                        bool already_shown = false;
                        SymbolContext frame_sc(exe_ctx.frame->GetSymbolContext(eSymbolContextLineEntry));
                        if (m_interpreter.GetDebugger().UseExternalEditor() && frame_sc.line_entry.file && frame_sc.line_entry.line != 0)
                        {
                            already_shown = Host::OpenFileInExternalEditor (frame_sc.line_entry.file, frame_sc.line_entry.line);
                        }

                        if (DisplayFrameForExecutionContext (exe_ctx.thread,
                                                             exe_ctx.frame,
                                                             m_interpreter,
                                                             result.GetOutputStream(),
                                                             true,
                                                             !already_shown,
                                                             3,
                                                             3))
                        {
                            result.SetStatus (eReturnStatusSuccessFinishResult);
                            return result.Succeeded();
                        }
                    }
                }
                if (frame_idx == UINT32_MAX)
                    result.AppendErrorWithFormat ("Invalid frame index: %s.\n", frame_idx_cstr);
                else
                    result.AppendErrorWithFormat ("Frame index (%u) out of range.\n", frame_idx);
            }
            else
            {
                result.AppendError ("invalid arguments");
                result.AppendErrorWithFormat ("Usage: %s\n", m_cmd_syntax.c_str());
            }
        }
        else
        {
            result.AppendError ("no current thread");
        }
        result.SetStatus (eReturnStatusFailed);
        return false;
    }
};

#pragma mark CommandObjectFrameVariable
//----------------------------------------------------------------------
// List images with associated information
//----------------------------------------------------------------------
class CommandObjectFrameVariable : public CommandObject
{
public:

    class CommandOptions : public Options
    {
    public:

        CommandOptions () :
            Options()
        {
            ResetOptionValues ();
        }

        virtual
        ~CommandOptions ()
        {
        }

        virtual Error
        SetOptionValue (int option_idx, const char *option_arg)
        {
            Error error;
            bool success;
            char short_option = (char) m_getopt_table[option_idx].val;
            switch (short_option)
            {
            case 'o':   use_objc     = true;  break;
            case 'n':   name = option_arg;    break;
            case 'r':   use_regex    = true;  break;
            case 'a':   show_args    = false; break;
            case 'l':   show_locals  = false; break;
            case 'g':   show_globals = true;  break;
            case 't':   show_types   = false; break;
            case 'y':   show_summary = false; break;
            case 'L':   show_location= true;  break;
            case 'c':   show_decl    = true;  break;
            case 'D':   debug        = true;  break;
            case 'd':
                max_depth = Args::StringToUInt32 (option_arg, UINT32_MAX, 0, &success);
                if (!success)
                    error.SetErrorStringWithFormat("Invalid max depth '%s'.\n", option_arg);
                break;

            case 'p':
                ptr_depth = Args::StringToUInt32 (option_arg, 0, 0, &success);
                if (!success)
                    error.SetErrorStringWithFormat("Invalid pointer depth '%s'.\n", option_arg);
                break;

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

        void
        ResetOptionValues ()
        {
            Options::ResetOptionValues();

            name.clear();
            use_objc      = false;
            use_regex     = false;
            show_args     = true;
            show_locals   = true;
            show_globals  = false;
            show_types    = true;
            show_scope    = false;
            show_summary  = true;
            show_location = false;
            show_decl     = false;
            debug         = false;
            max_depth     = UINT32_MAX;
            ptr_depth     = 0;
            globals.clear();
        }

        const lldb::OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static lldb::OptionDefinition g_option_table[];
        std::string name;
        bool use_objc:1,
             use_regex:1,
             show_args:1,
             show_locals:1,
             show_globals:1,
             show_types:1,
             show_scope:1,
             show_summary:1,
             show_location:1,
             show_decl:1,
             debug:1;
        uint32_t max_depth; // The depth to print when dumping concrete (not pointers) aggreate values
        uint32_t ptr_depth; // The default depth that is dumped when we find pointers
        std::vector<ConstString> globals;
        // Instance variables to hold the values for command options.
    };

    CommandObjectFrameVariable (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "frame variable",
                       "Show frame variables. All argument and local variables "
                       "that are in scope will be shown when no arguments are given. "
                       "If any arguments are specified, they can be names of "
                       "argument, local, file static and file global variables."
                       "Children of aggregate variables can be specified such as "
                       "'var->child.x'.",
                       "frame variable [<cmd-options>] [<var-name1> [<var-name2>...]]")
    {
    }

    virtual
    ~CommandObjectFrameVariable ()
    {
    }

    virtual
    Options *
    GetOptions ()
    {
        return &m_options;
    }

    void
    DumpValueObject (CommandReturnObject &result,
                     ExecutionContextScope *exe_scope,
                     ValueObject *valobj,
                     const char *root_valobj_name,
                     uint32_t ptr_depth,
                     uint32_t curr_depth,
                     uint32_t max_depth,
                     bool use_objc,
                     bool scope_already_checked)
    {
        if (valobj)
        {
            Stream &s = result.GetOutputStream();

            //const char *loc_cstr = valobj->GetLocationAsCString();
            if (m_options.show_location)
            {
                s.Printf("%s: ", valobj->GetLocationAsCString(exe_scope));
            }
            if (m_options.debug)
                s.Printf ("%p ValueObject{%u} ", valobj, valobj->GetID());

            s.Indent();

            if (m_options.show_types)
                s.Printf("(%s) ", valobj->GetTypeName().AsCString());

            const char *name_cstr = root_valobj_name ? root_valobj_name : valobj->GetName().AsCString("");
            s.Printf ("%s = ", name_cstr);

            if (!scope_already_checked && !valobj->IsInScope(exe_scope->CalculateStackFrame()))
            {
                s.PutCString("error: out of scope");
                return;
            }
            
            const char *val_cstr = valobj->GetValueAsCString(exe_scope);
            const char *err_cstr = valobj->GetError().AsCString();

            if (err_cstr)
            {
                s.Printf ("error: %s", err_cstr);
            }
            else
            {
                const char *sum_cstr = valobj->GetSummaryAsCString(exe_scope);

                const bool is_aggregate = ClangASTContext::IsAggregateType (valobj->GetOpaqueClangQualType());

                if (val_cstr)
                    s.PutCString(val_cstr);

                if (sum_cstr)
                    s.Printf(" %s", sum_cstr);
                
                if (use_objc)
                {
                    const char *object_desc = valobj->GetObjectDescription(exe_scope);
                    if (object_desc)
                        s.Printf("\n%s\n", object_desc);
                    else
                        s.Printf ("No description available.\n");
                    return;
                }


                if (curr_depth < max_depth)
                {
                    if (is_aggregate)
                        s.PutChar('{');

                    bool is_ptr_or_ref = ClangASTContext::IsPointerOrReferenceType (valobj->GetOpaqueClangQualType());
                    
                    if (is_ptr_or_ref && ptr_depth == 0)
                        return;

                    const uint32_t num_children = valobj->GetNumChildren();
                    if (num_children)
                    {
                        s.IndentMore();
                        for (uint32_t idx=0; idx<num_children; ++idx)
                        {
                            ValueObjectSP child_sp(valobj->GetChildAtIndex(idx, true));
                            if (child_sp.get())
                            {
                                s.EOL();
                                DumpValueObject (result,
                                                 exe_scope,
                                                 child_sp.get(),
                                                 NULL,
                                                 is_ptr_or_ref ? ptr_depth - 1 : ptr_depth,
                                                 curr_depth + 1,
                                                 max_depth,
                                                 false,
                                                 true);
                                if (idx + 1 < num_children)
                                    s.PutChar(',');
                            }
                        }
                        s.IndentLess();
                    }
                    if (is_aggregate)
                    {
                        s.EOL();
                        s.Indent("}");
                    }
                }
                else
                {
                    if (is_aggregate)
                    {
                        s.PutCString("{...}");
                    }
                }

            }
        }
    }

    virtual bool
    Execute
    (
        Args& command,
        CommandReturnObject &result
    )
    {
        ExecutionContext exe_ctx(m_interpreter.GetDebugger().GetExecutionContext());
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
            //ValueObjectList &valobj_list = exe_ctx.frame->GetValueObjectList();
            const char *name_cstr = NULL;
            size_t idx;
            if (!m_options.globals.empty())
            {
                uint32_t fail_count = 0;
                if (exe_ctx.target)
                {
                    const size_t num_globals = m_options.globals.size();
                    for (idx = 0; idx < num_globals; ++idx)
                    {
                        VariableList global_var_list;
                        const uint32_t num_matching_globals = exe_ctx.target->GetImages().FindGlobalVariables (m_options.globals[idx], true, UINT32_MAX, global_var_list);

                        if (num_matching_globals == 0)
                        {
                            ++fail_count;
                            result.GetErrorStream().Printf ("error: can't find global variable '%s'\n", m_options.globals[idx].AsCString());
                        }
                        else
                        {
                            for (uint32_t global_idx=0; global_idx<num_matching_globals; ++global_idx)
                            {
                                var_sp = global_var_list.GetVariableAtIndex(global_idx);
                                if (var_sp)
                                {
                                    valobj_sp = exe_ctx.frame->GetValueObjectForFrameVariable (var_sp);
                                    if (!valobj_sp)
                                        valobj_sp = exe_ctx.frame->TrackGlobalVariable (var_sp);

                                    if (valobj_sp)
                                    {
                                        if (m_options.show_decl && var_sp->GetDeclaration ().GetFile())
                                        {
                                            var_sp->GetDeclaration ().DumpStopContext (&s, false);
                                            s.PutCString (": ");
                                        }

                                        DumpValueObject (result, 
                                                         exe_ctx.frame, 
                                                         valobj_sp.get(), 
                                                         name_cstr, 
                                                         m_options.ptr_depth, 
                                                         0, 
                                                         m_options.max_depth, 
                                                         m_options.use_objc, 
                                                         false);
                                        
                                        s.EOL();
                                    }
                                }
                            }
                        }
                    }
                }
                if (fail_count)
                    result.SetStatus (eReturnStatusFailed);
            }
            else if (variable_list)
            {
                if (command.GetArgumentCount() > 0)
                {
                    // If we have any args to the variable command, we will make
                    // variable objects from them...
                    for (idx = 0; (name_cstr = command.GetArgumentAtIndex(idx)) != NULL; ++idx)
                    {
                        uint32_t ptr_depth = m_options.ptr_depth;
                        // If first character is a '*', then show pointer contents
                        if (name_cstr[0] == '*')
                        {
                            ++ptr_depth;
                            name_cstr++; // Skip the '*'
                        }

                        std::string var_path (name_cstr);
                        size_t separator_idx = var_path.find_first_of(".-[");

                        ConstString name_const_string;
                        if (separator_idx == std::string::npos)
                            name_const_string.SetCString (var_path.c_str());
                        else
                            name_const_string.SetCStringWithLength (var_path.c_str(), separator_idx);

                        var_sp = variable_list->FindVariable(name_const_string);
                        if (var_sp)
                        {
                            valobj_sp = exe_ctx.frame->GetValueObjectForFrameVariable (var_sp);

                            var_path.erase (0, name_const_string.GetLength ());
                            // We are dumping at least one child
                            while (separator_idx != std::string::npos)
                            {
                                // Calculate the next separator index ahead of time
                                ValueObjectSP child_valobj_sp;
                                const char separator_type = var_path[0];
                                switch (separator_type)
                                {

                                case '-':
                                    if (var_path.size() >= 2 && var_path[1] != '>')
                                    {
                                        result.GetErrorStream().Printf ("error: invalid character in variable path starting at '%s'\n",
                                                                        var_path.c_str());
                                        var_path.clear();
                                        valobj_sp.reset();
                                        break;
                                    }
                                    var_path.erase (0, 1); // Remove the '-'
                                    // Fall through
                                case '.':
                                    {
                                        var_path.erase (0, 1); // Remove the '.' or '>'
                                        separator_idx = var_path.find_first_of(".-[");
                                        ConstString child_name;
                                        if (separator_idx == std::string::npos)
                                            child_name.SetCString (var_path.c_str());
                                        else
                                            child_name.SetCStringWithLength(var_path.c_str(), separator_idx);

                                        child_valobj_sp = valobj_sp->GetChildMemberWithName (child_name, true);
                                        if (!child_valobj_sp)
                                        {
                                            result.GetErrorStream().Printf ("error: can't find child of '%s' named '%s'\n",
                                                                            valobj_sp->GetName().AsCString(),
                                                                            child_name.GetCString());
                                            var_path.clear();
                                            valobj_sp.reset();
                                            break;
                                        }
                                        // Remove the child name from the path
                                        var_path.erase(0, child_name.GetLength());
                                    }
                                    break;

                                case '[':
                                    // Array member access, or treating pointer as an array
                                    if (var_path.size() > 2) // Need at least two brackets and a number
                                    {
                                        char *end = NULL;
                                        int32_t child_index = ::strtol (&var_path[1], &end, 0);
                                        if (end && *end == ']')
                                        {

                                            if (valobj_sp->IsPointerType ())
                                            {
                                                child_valobj_sp = valobj_sp->GetSyntheticArrayMemberFromPointer (child_index, true);
                                            }
                                            else
                                            {
                                                child_valobj_sp = valobj_sp->GetChildAtIndex (child_index, true);
                                            }

                                            if (!child_valobj_sp)
                                            {
                                                result.GetErrorStream().Printf ("error: invalid array index %u in '%s'\n",
                                                                                child_index,
                                                                                valobj_sp->GetName().AsCString());
                                                var_path.clear();
                                                valobj_sp.reset();
                                                break;
                                            }

                                            // Erase the array member specification '[%i]' where %i is the array index
                                            var_path.erase(0, (end - var_path.c_str()) + 1);
                                            separator_idx = var_path.find_first_of(".-[");

                                            // Break out early from the switch since we were able to find the child member
                                            break;
                                        }
                                    }
                                    result.GetErrorStream().Printf ("error: invalid array member specification for '%s' starting at '%s'\n",
                                                                    valobj_sp->GetName().AsCString(),
                                                                    var_path.c_str());
                                    var_path.clear();
                                    valobj_sp.reset();
                                    break;

                                    break;

                                default:
                                    result.GetErrorStream().Printf ("error: invalid character in variable path starting at '%s'\n",
                                                                        var_path.c_str());
                                    var_path.clear();
                                    valobj_sp.reset();
                                    separator_idx = std::string::npos;
                                    break;
                                }

                                if (child_valobj_sp)
                                    valobj_sp = child_valobj_sp;

                                if (var_path.empty())
                                    break;

                            }

                            if (valobj_sp)
                            {
                                if (m_options.show_decl && var_sp->GetDeclaration ().GetFile())
                                {
                                    var_sp->GetDeclaration ().DumpStopContext (&s, false);
                                    s.PutCString (": ");
                                }

                                DumpValueObject (result, 
                                                 exe_ctx.frame, 
                                                 valobj_sp.get(), 
                                                 name_cstr, 
                                                 ptr_depth, 
                                                 0, 
                                                 m_options.max_depth, 
                                                 m_options.use_objc,
                                                 false);

                                s.EOL();
                            }
                        }
                        else
                        {
                            result.GetErrorStream().Printf ("error: unable to find any variables named '%s'\n", name_cstr);
                            var_path.clear();
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
                            VariableSP var_sp (variable_list->GetVariableAtIndex(i));
                            bool dump_variable = true;
                            
                            switch (var_sp->GetScope())
                            {
                            case eValueTypeVariableGlobal:
                                dump_variable = m_options.show_globals;
                                if (dump_variable && m_options.show_scope)
                                    s.PutCString("GLOBAL: ");
                                break;

                            case eValueTypeVariableStatic:
                                dump_variable = m_options.show_globals;
                                if (dump_variable && m_options.show_scope)
                                    s.PutCString("STATIC: ");
                                break;
                                
                            case eValueTypeVariableArgument:
                                dump_variable = m_options.show_args;
                                if (dump_variable && m_options.show_scope)
                                    s.PutCString("   ARG: ");
                                break;
                                
                            case eValueTypeVariableLocal:
                                dump_variable = m_options.show_locals;
                                if (dump_variable && m_options.show_scope)
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
                                valobj_sp = exe_ctx.frame->GetValueObjectForFrameVariable (var_sp);
                                if (valobj_sp)
                                {
                                    // When dumping all variables, don't print any variables
                                    // that are not in scope to avoid extra unneeded output
                                    if (valobj_sp->IsInScope (exe_ctx.frame))
                                    {
                                        if (m_options.show_decl && var_sp->GetDeclaration ().GetFile())
                                        {
                                            var_sp->GetDeclaration ().DumpStopContext (&s, false);
                                            s.PutCString (": ");
                                        }
                                        DumpValueObject (result, 
                                                         exe_ctx.frame, 
                                                         valobj_sp.get(), 
                                                         name_cstr, 
                                                         m_options.ptr_depth, 
                                                         0, 
                                                         m_options.max_depth, 
                                                         m_options.use_objc,
                                                         true);

                                        s.EOL();
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

    CommandOptions m_options;
};

lldb::OptionDefinition
CommandObjectFrameVariable::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "debug",      'D', no_argument,       NULL, 0, NULL,        "Enable verbose debug information."},
{ LLDB_OPT_SET_1, false, "depth",      'd', required_argument, NULL, 0, "<count>",   "Set the max recurse depth when dumping aggregate types (default is infinity)."},
{ LLDB_OPT_SET_1, false, "show-globals",'g', no_argument,      NULL, 0, NULL,        "Show the current frame source file global and static variables."},
{ LLDB_OPT_SET_1, false, "find-global",'G', required_argument, NULL, 0, "<name>",    "Find a global variable by name (which might not be in the current stack frame source file)."},
{ LLDB_OPT_SET_1, false, "location",   'L', no_argument,       NULL, 0, NULL,        "Show variable location information."},
{ LLDB_OPT_SET_1, false, "show-declaration", 'c', no_argument, NULL, 0, NULL,        "Show variable declaration information (source file and line where the variable was declared)."},
{ LLDB_OPT_SET_1, false, "name",       'n', required_argument, NULL, 0, "<name>",    "Lookup a variable by name or regex (--regex) for the current execution context."},
{ LLDB_OPT_SET_1, false, "no-args",    'a', no_argument,       NULL, 0, NULL,        "Omit function arguments."},
{ LLDB_OPT_SET_1, false, "no-locals",  'l', no_argument,       NULL, 0, NULL,        "Omit local variables."},
{ LLDB_OPT_SET_1, false, "no-types",   't', no_argument,       NULL, 0, NULL,        "Omit variable type names."},
{ LLDB_OPT_SET_1, false, "no-summary", 'y', no_argument,       NULL, 0, NULL,        "Omit summary information."},
{ LLDB_OPT_SET_1, false, "scope",      's', no_argument,       NULL, 0, NULL,        "Show variable scope (argument, local, global, static)."},
{ LLDB_OPT_SET_1, false, "objc",       'o', no_argument,       NULL, 0, NULL,        "When looking up a variable by name (--name), print as an Objective-C object."},
{ LLDB_OPT_SET_1, false, "ptr-depth",  'p', required_argument, NULL, 0, "<count>",   "The number of pointers to be traversed when dumping values (default is zero)."},
{ LLDB_OPT_SET_1, false, "regex",      'r', no_argument,       NULL, 0, NULL,        "The <name> argument for name lookups are regular expressions."},
{ 0, false, NULL, 0, 0, NULL, NULL, NULL, NULL }
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

