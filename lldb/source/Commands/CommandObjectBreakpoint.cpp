//===-- CommandObjectBreakpoint.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectBreakpoint.h"
#include "CommandObjectBreakpointCommand.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointIDList.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Target.h"
#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadSpec.h"

using namespace lldb;
using namespace lldb_private;

static void
AddBreakpointDescription (StreamString *s, Breakpoint *bp, lldb::DescriptionLevel level)
{
    s->IndentMore();
    bp->GetDescription (s, level, true);
    s->IndentLess();
    s->EOL();
}

//-------------------------------------------------------------------------
// CommandObjectBreakpointSet::CommandOptions
//-------------------------------------------------------------------------
#pragma mark Set::CommandOptions

CommandObjectBreakpointSet::CommandOptions::CommandOptions() :
    Options (),
    m_filename (),
    m_line_num (0),
    m_column (0),
    m_ignore_inlines (false),
    m_func_name (),
    m_func_name_type_mask (0),
    m_func_regexp (),
    m_modules (),
    m_load_addr(),
    m_ignore_count (0),
    m_thread_id(LLDB_INVALID_THREAD_ID),
    m_thread_index (UINT32_MAX),
    m_thread_name(),
    m_queue_name()
{
}

CommandObjectBreakpointSet::CommandOptions::~CommandOptions ()
{
}

lldb::OptionDefinition
CommandObjectBreakpointSet::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "shlib", 's', required_argument, NULL, CommandCompletions::eModuleCompletion, "<shlib-name>",
        "Set the breakpoint only in this shared library (can use this option multiple times for multiple shlibs)."},

    { LLDB_OPT_SET_ALL, false, "ignore_count", 'k', required_argument,   NULL, 0, "<n>",
        "Set the number of times this breakpoint is sKipped before stopping." },

    { LLDB_OPT_SET_ALL, false, "thread_index", 'x', required_argument, NULL, NULL, "<thread_index>",
        "The breakpoint stops only for the thread whose indeX matches this argument."},

    { LLDB_OPT_SET_ALL, false, "thread_id", 't', required_argument, NULL, NULL, "<thread_id>",
        "The breakpoint stops only for the thread whose TID matches this argument."},

    { LLDB_OPT_SET_ALL, false, "thread_name", 'T', required_argument, NULL, NULL, "<thread_name>",
        "The breakpoint stops only for the thread whose thread name matches this argument."},

    { LLDB_OPT_SET_ALL, false, "queue_name", 'q', required_argument, NULL, NULL, "<queue_name>",
        "The breakpoint stops only for threads in the queue whose name is given by this argument."},

    { LLDB_OPT_SET_1, false, "file", 'f', required_argument, NULL, CommandCompletions::eSourceFileCompletion, "<filename>",
        "Set the breakpoint by source location in this particular file."},

    { LLDB_OPT_SET_1, true, "line", 'l', required_argument, NULL, 0, "<linenum>",
        "Set the breakpoint by source location at this particular line."},

    // Comment out this option for the moment, as we don't actually use it, but will in the future.
    // This way users won't see it, but the infrastructure is left in place.
    //    { 0, false, "column",     'c', required_argument, NULL, "<column>",
    //    "Set the breakpoint by source location at this particular column."},

    { LLDB_OPT_SET_2, true, "address", 'a', required_argument, NULL, 0, "<address>",
        "Set the breakpoint by address, at the specified address."},

    { LLDB_OPT_SET_3, true, "name", 'n', required_argument, NULL, CommandCompletions::eSymbolCompletion, "<name>",
        "Set the breakpoint by function name." },

    { LLDB_OPT_SET_3, false, "basename", 'b', no_argument, NULL, 0, NULL,
        "Used in conjuction with --name <name> to search function basenames." },

    { LLDB_OPT_SET_3, false, "fullname", 'F', no_argument, NULL, 0, NULL,
        "Used in conjuction with --name <name> to search fully qualified function names. For C++ this means namespaces and all arguemnts, and for Objective C this means a full function prototype with class and selector." },

    { LLDB_OPT_SET_3, false, "selector", 'S', no_argument, NULL, 0, NULL,
        "Used in conjuction with --name <name> to search objective C selector names." },

    { LLDB_OPT_SET_3, false, "method", 'm', no_argument, NULL, 0, NULL,
        "Used in conjuction with --name <name> to search objective C selector C++ method names." },

    { LLDB_OPT_SET_4, true, "func_regex", 'r', required_argument, NULL, 0, "<regular-expression>",
        "Set the breakpoint by function name, evaluating a regular-expression to find the function name(s)." },

    { 0, false, NULL, 0, 0, NULL, 0, NULL, NULL }
};

const lldb::OptionDefinition*
CommandObjectBreakpointSet::CommandOptions::GetDefinitions ()
{
    return g_option_table;
}

Error
CommandObjectBreakpointSet::CommandOptions::SetOptionValue (int option_idx, const char *option_arg)
{
    Error error;
    char short_option = (char) m_getopt_table[option_idx].val;

    switch (short_option)
    {
        case 'a':
            m_load_addr = Args::StringToUInt64(optarg, LLDB_INVALID_ADDRESS, 0);
            if (m_load_addr == LLDB_INVALID_ADDRESS)
                m_load_addr = Args::StringToUInt64(optarg, LLDB_INVALID_ADDRESS, 16);

            if (m_load_addr == LLDB_INVALID_ADDRESS)
                error.SetErrorStringWithFormat ("Invalid address string '%s'.\n", optarg);
            break;

        case 'c':
            m_column = Args::StringToUInt32 (option_arg, 0);
            break;

        case 'f':
            m_filename = option_arg;
            break;

        case 'l':
            m_line_num = Args::StringToUInt32 (option_arg, 0);
            break;

        case 'n':
            m_func_name = option_arg;
            break;

        case 'b':
            m_func_name_type_mask |= eFunctionNameTypeBase;
            break;

        case 'F':
            m_func_name_type_mask |= eFunctionNameTypeFull;
            break;

        case 'S':
            m_func_name_type_mask |= eFunctionNameTypeSelector;
            break;

        case 'm':
            m_func_name_type_mask |= eFunctionNameTypeMethod;
            break;

        case 'r':
            m_func_regexp = option_arg;
            break;

        case 's':
            {
                m_modules.push_back (std::string (option_arg));
                break;
            }
        case 'k':
        {
            m_ignore_count = Args::StringToUInt32(optarg, UINT32_MAX, 0);
            if (m_ignore_count == UINT32_MAX)
               error.SetErrorStringWithFormat ("Invalid ignore count '%s'.\n", optarg);
        }
        break;
        case 't' :
        {
            m_thread_id = Args::StringToUInt64(optarg, LLDB_INVALID_THREAD_ID, 0);
            if (m_thread_id == LLDB_INVALID_THREAD_ID)
               error.SetErrorStringWithFormat ("Invalid thread id string '%s'.\n", optarg);
        }
        break;
        case 'T':
            m_thread_name = option_arg;
            break;
        case 'q':
            m_queue_name = option_arg;
            break;
        case 'x':
        {
            m_thread_index = Args::StringToUInt32(optarg, UINT32_MAX, 0);
            if (m_thread_id == UINT32_MAX)
               error.SetErrorStringWithFormat ("Invalid thread index string '%s'.\n", optarg);
            
        }
        break;
        default:
            error.SetErrorStringWithFormat ("Unrecognized option '%c'.\n", short_option);
            break;
    }

    return error;
}

void
CommandObjectBreakpointSet::CommandOptions::ResetOptionValues ()
{
    Options::ResetOptionValues();

    m_filename.clear();
    m_line_num = 0;
    m_column = 0;
    m_func_name.clear();
    m_func_name_type_mask = 0;
    m_func_regexp.clear();
    m_load_addr = LLDB_INVALID_ADDRESS;
    m_modules.clear();
    m_ignore_count = 0;
    m_thread_id = LLDB_INVALID_THREAD_ID;
    m_thread_index = UINT32_MAX;
    m_thread_name.clear();
    m_queue_name.clear();
}

//-------------------------------------------------------------------------
// CommandObjectBreakpointSet
//-------------------------------------------------------------------------
#pragma mark Set

CommandObjectBreakpointSet::CommandObjectBreakpointSet () :
    CommandObject ("breakpoint set", "Sets a breakpoint or set of breakpoints in the executable.", 
                   "breakpoint set <cmd-options>")
{
}

CommandObjectBreakpointSet::~CommandObjectBreakpointSet ()
{
}

Options *
CommandObjectBreakpointSet::GetOptions ()
{
    return &m_options;
}

bool
CommandObjectBreakpointSet::Execute
(
    CommandInterpreter &interpreter,
    Args& command,
    CommandReturnObject &result
)
{
    Target *target = interpreter.GetDebugger().GetSelectedTarget().get();
    if (target == NULL)
    {
        result.AppendError ("Invalid target, set executable file using 'file' command.");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    // The following are the various types of breakpoints that could be set:
    //   1).  -f -l -p  [-s -g]   (setting breakpoint by source location)
    //   2).  -a  [-s -g]         (setting breakpoint by address)
    //   3).  -n  [-s -g]         (setting breakpoint by function name)
    //   4).  -r  [-s -g]         (setting breakpoint by function name regular expression)

    BreakpointSetType break_type = eSetTypeInvalid;

    if (m_options.m_line_num != 0)
        break_type = eSetTypeFileAndLine;
    else if (m_options.m_load_addr != LLDB_INVALID_ADDRESS)
        break_type = eSetTypeAddress;
    else if (!m_options.m_func_name.empty())
        break_type = eSetTypeFunctionName;
    else if  (!m_options.m_func_regexp.empty())
        break_type = eSetTypeFunctionRegexp;

    ModuleSP module_sp = target->GetExecutableModule();
    Breakpoint *bp = NULL;
    FileSpec module;
    bool use_module = false;
    int num_modules = m_options.m_modules.size();

    if ((num_modules > 0) && (break_type != eSetTypeAddress))
        use_module = true;
     
    switch (break_type)
    {
        case eSetTypeFileAndLine: // Breakpoint by source position
        {
            FileSpec file;
            if (m_options.m_filename.empty())
            {
                StackFrame *cur_frame = interpreter.GetDebugger().GetExecutionContext().frame;
                if (cur_frame == NULL)
                {
                    result.AppendError ("Attempting to set breakpoint by line number alone with no selected frame.");
                    result.SetStatus (eReturnStatusFailed);
                    break;
                }
                else if (!cur_frame->HasDebugInformation())
                {
                    result.AppendError ("Attempting to set breakpoint by line number alone but selected frame has no debug info.");
                    result.SetStatus (eReturnStatusFailed);
                    break;
                }
                else
                {
                    const SymbolContext &context = cur_frame->GetSymbolContext(true);
                    if (context.line_entry.file)
                    {
                        file = context.line_entry.file;
                    }
                    else if (context.comp_unit != NULL)
                    {    file = context.comp_unit;
                    }
                    else
                    {
                        result.AppendError ("Attempting to set breakpoint by line number alone but can't find the file for the selected frame.");
                        result.SetStatus (eReturnStatusFailed);
                        break;
                    }
                }
            }
            else
            {
                file.SetFile(m_options.m_filename.c_str());
            }

            if (use_module)
            {
                for (int i = 0; i < num_modules; ++i)
                {
                    module.SetFile(m_options.m_modules[i].c_str());
                    bp = target->CreateBreakpoint (&module,
                                                   file,
                                                   m_options.m_line_num,
                                                   m_options.m_ignore_inlines).get();
                    if (bp)
                    {
                        StreamString &output_stream = result.GetOutputStream();
                        output_stream.Printf ("Breakpoint created: ");
                        bp->GetDescription(&output_stream, lldb::eDescriptionLevelBrief);
                        output_stream.EOL();
                        result.SetStatus (eReturnStatusSuccessFinishResult);
                    }
                    else
                    {
                        result.AppendErrorWithFormat("Breakpoint creation failed: No breakpoint created in module '%s'.\n",
                                                    m_options.m_modules[i].c_str());
                        result.SetStatus (eReturnStatusFailed);
                    }
                }
            }
            else
                bp = target->CreateBreakpoint (NULL,
                                               file,
                                               m_options.m_line_num,
                                               m_options.m_ignore_inlines).get();
        }
        break;
        case eSetTypeAddress: // Breakpoint by address
            bp = target->CreateBreakpoint (m_options.m_load_addr, false).get();
            break;

        case eSetTypeFunctionName: // Breakpoint by function name
            {
                uint32_t name_type_mask = m_options.m_func_name_type_mask;
                
                if (name_type_mask == 0)
                {
                
                    if (m_options.m_func_name.find('(') != std::string::npos ||
                        m_options.m_func_name.find("-[") == 0 ||
                        m_options.m_func_name.find("+[") == 0)
                        name_type_mask |= eFunctionNameTypeFull;
                    else
                        name_type_mask |= eFunctionNameTypeBase;
                }
                    
                
                if (use_module)
                {
                    for (int i = 0; i < num_modules; ++i)
                    {
                        module.SetFile(m_options.m_modules[i].c_str());
                        bp = target->CreateBreakpoint (&module, m_options.m_func_name.c_str(), name_type_mask, Breakpoint::Exact).get();
                        if (bp)
                        {
                            StreamString &output_stream = result.GetOutputStream();
                            output_stream.Printf ("Breakpoint created: ");
                            bp->GetDescription(&output_stream, lldb::eDescriptionLevelBrief);
                            output_stream.EOL();
                            result.SetStatus (eReturnStatusSuccessFinishResult);
                        }
                        else
                        {
                            result.AppendErrorWithFormat("Breakpoint creation failed: No breakpoint created in module '%s'.\n",
                                                        m_options.m_modules[i].c_str());
                            result.SetStatus (eReturnStatusFailed);
                        }
                    }
                }
                else
                    bp = target->CreateBreakpoint (NULL, m_options.m_func_name.c_str(), name_type_mask, Breakpoint::Exact).get();
            }
            break;

        case eSetTypeFunctionRegexp: // Breakpoint by regular expression function name
            {
                RegularExpression regexp(m_options.m_func_regexp.c_str());
                if (use_module)
                {
                    for (int i = 0; i < num_modules; ++i)
                    {
                        module.SetFile(m_options.m_modules[i].c_str());
                        bp = target->CreateBreakpoint (&module, regexp).get();
                        if (bp)
                        {
                            StreamString &output_stream = result.GetOutputStream();
                            output_stream.Printf ("Breakpoint created: ");
                            bp->GetDescription(&output_stream, lldb::eDescriptionLevelBrief);
                            output_stream.EOL();
                            result.SetStatus (eReturnStatusSuccessFinishResult);
                        }
                        else
                        {
                            result.AppendErrorWithFormat("Breakpoint creation failed: No breakpoint created in module '%s'.\n",
                                                        m_options.m_modules[i].c_str());
                            result.SetStatus (eReturnStatusFailed);
                        }
                    }
                }
                else
                    bp = target->CreateBreakpoint (NULL, regexp).get();
            }
            break;

        default:
            break;
    }

    // Now set the various options that were passed in:
    if (bp)
    {
        if (m_options.m_thread_id != LLDB_INVALID_THREAD_ID)
            bp->SetThreadID (m_options.m_thread_id);
            
        if (m_options.m_thread_index != UINT32_MAX)
            bp->GetOptions()->GetThreadSpec()->SetIndex(m_options.m_thread_index);
        
        if (!m_options.m_thread_name.empty())
            bp->GetOptions()->GetThreadSpec()->SetName(m_options.m_thread_name.c_str());
        
        if (!m_options.m_queue_name.empty())
            bp->GetOptions()->GetThreadSpec()->SetQueueName(m_options.m_queue_name.c_str());
            
        if (m_options.m_ignore_count != 0)
            bp->GetOptions()->SetIgnoreCount(m_options.m_ignore_count);
    }
    
    if (bp && !use_module)
    {
        StreamString &output_stream = result.GetOutputStream();
        output_stream.Printf ("Breakpoint created: ");
        bp->GetDescription(&output_stream, lldb::eDescriptionLevelBrief);
        output_stream.EOL();
        result.SetStatus (eReturnStatusSuccessFinishResult);
    }
    else if (!bp)
    {
        result.AppendError ("Breakpoint creation failed: No breakpoint created.");
        result.SetStatus (eReturnStatusFailed);
    }

    return result.Succeeded();
}

//-------------------------------------------------------------------------
// CommandObjectMultiwordBreakpoint
//-------------------------------------------------------------------------
#pragma mark MultiwordBreakpoint

CommandObjectMultiwordBreakpoint::CommandObjectMultiwordBreakpoint (CommandInterpreter &interpreter) :
    CommandObjectMultiword ("breakpoint",
                              "A set of commands for operating on breakpoints.",
                              "breakpoint <command> [<command-options>]")
{
    bool status;

    CommandObjectSP list_command_object (new CommandObjectBreakpointList ());
    CommandObjectSP delete_command_object (new CommandObjectBreakpointDelete ());
    CommandObjectSP enable_command_object (new CommandObjectBreakpointEnable ());
    CommandObjectSP disable_command_object (new CommandObjectBreakpointDisable ());
    CommandObjectSP set_command_object (new CommandObjectBreakpointSet ());
    CommandObjectSP command_command_object (new CommandObjectBreakpointCommand (interpreter));
    CommandObjectSP modify_command_object (new CommandObjectBreakpointModify());

    command_command_object->SetCommandName ("breakpoint command");
    enable_command_object->SetCommandName("breakpoint enable");
    disable_command_object->SetCommandName("breakpoint disable");
    list_command_object->SetCommandName ("breakpoint list");
    modify_command_object->SetCommandName ("breakpoint modify");
    set_command_object->SetCommandName("breakpoint set");

    status = LoadSubCommand (interpreter, "list",       list_command_object);
    status = LoadSubCommand (interpreter, "enable",     enable_command_object);
    status = LoadSubCommand (interpreter, "disable",    disable_command_object);
    status = LoadSubCommand (interpreter, "delete",     delete_command_object);
    status = LoadSubCommand (interpreter, "set",        set_command_object);
    status = LoadSubCommand (interpreter, "command",    command_command_object);
    status = LoadSubCommand (interpreter, "modify",     modify_command_object);
}

CommandObjectMultiwordBreakpoint::~CommandObjectMultiwordBreakpoint ()
{
}

void
CommandObjectMultiwordBreakpoint::VerifyBreakpointIDs (Args &args, Target *target, CommandReturnObject &result,
                                                         BreakpointIDList *valid_ids)
{
    // args can be strings representing 1). integers (for breakpoint ids)
    //                                  2). the full breakpoint & location canonical representation
    //                                  3). the word "to" or a hyphen, representing a range (in which case there
    //                                      had *better* be an entry both before & after of one of the first two types.

    Args temp_args;

    // Create a new Args variable to use; copy any non-breakpoint-id-ranges stuff directly from the old ARGS to
    // the new TEMP_ARGS.  Do not copy breakpoint id range strings over; instead generate a list of strings for
    // all the breakpoint ids in the range, and shove all of those breakpoint id strings into TEMP_ARGS.

    BreakpointIDList::FindAndReplaceIDRanges (args, target, result, temp_args);

    // NOW, convert the list of breakpoint id strings in TEMP_ARGS into an actual BreakpointIDList:

    valid_ids->InsertStringArray (temp_args.GetConstArgumentVector(), temp_args.GetArgumentCount(), result);

    // At this point,  all of the breakpoint ids that the user passed in have been converted to breakpoint IDs
    // and put into valid_ids.

    if (result.Succeeded())
    {
        // Now that we've converted everything from args into a list of breakpoint ids, go through our tentative list
        // of breakpoint id's and verify that they correspond to valid/currently set breakpoints.

        const size_t count = valid_ids->GetSize();
        for (size_t i = 0; i < count; ++i)
        {
            BreakpointID cur_bp_id = valid_ids->GetBreakpointIDAtIndex (i);
            Breakpoint *breakpoint = target->GetBreakpointByID (cur_bp_id.GetBreakpointID()).get();
            if (breakpoint != NULL)
            {
                int num_locations = breakpoint->GetNumLocations();
                if (cur_bp_id.GetLocationID() > num_locations)
                {
                    StreamString id_str;
                    BreakpointID::GetCanonicalReference (&id_str, 
                                                         cur_bp_id.GetBreakpointID(),
                                                         cur_bp_id.GetLocationID());
                    i = valid_ids->GetSize() + 1;
                    result.AppendErrorWithFormat ("'%s' is not a currently valid breakpoint/location id.\n",
                                                 id_str.GetData());
                    result.SetStatus (eReturnStatusFailed);
                }
            }
            else
            {
                i = valid_ids->GetSize() + 1;
                result.AppendErrorWithFormat ("'%d' is not a currently valid breakpoint id.\n", cur_bp_id.GetBreakpointID());
                result.SetStatus (eReturnStatusFailed);
            }
        }
    }
}

//-------------------------------------------------------------------------
// CommandObjectBreakpointList::Options
//-------------------------------------------------------------------------
#pragma mark List::CommandOptions

CommandObjectBreakpointList::CommandOptions::CommandOptions() :
    Options (),
    m_level (lldb::eDescriptionLevelFull)  // Breakpoint List defaults to brief descriptions
{
}

CommandObjectBreakpointList::CommandOptions::~CommandOptions ()
{
}

lldb::OptionDefinition
CommandObjectBreakpointList::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "internal", 'i', no_argument, NULL, 0, NULL,
        "Show debugger internal breakpoints" },

    { LLDB_OPT_SET_1, false, "brief",    'b', no_argument, NULL, 0, NULL,
        "Give a brief description of the breakpoint (no location info)."},

    // FIXME: We need to add an "internal" command, and then add this sort of thing to it.
    // But I need to see it for now, and don't want to wait.
    { LLDB_OPT_SET_2, false, "full",    'f', no_argument, NULL, 0, NULL,
        "Give a full description of the breakpoint and its locations."},

    { LLDB_OPT_SET_3, false, "verbose", 'v', no_argument, NULL, 0, NULL,
        "Explain everything we know about the breakpoint (for debugging debugger bugs)." },

    { 0, false, NULL, 0, 0, NULL, 0, NULL, NULL }
};

const lldb::OptionDefinition*
CommandObjectBreakpointList::CommandOptions::GetDefinitions ()
{
    return g_option_table;
}

Error
CommandObjectBreakpointList::CommandOptions::SetOptionValue (int option_idx, const char *option_arg)
{
    Error error;
    char short_option = (char) m_getopt_table[option_idx].val;

    switch (short_option)
    {
        case 'b':
            m_level = lldb::eDescriptionLevelBrief;
            break;
        case 'f':
            m_level = lldb::eDescriptionLevelFull;
            break;
        case 'v':
            m_level = lldb::eDescriptionLevelVerbose;
            break;
        case 'i':
            m_internal = true;
            break;
        default:
            error.SetErrorStringWithFormat ("Unrecognized option '%c'.\n", short_option);
            break;
    }

    return error;
}

void
CommandObjectBreakpointList::CommandOptions::ResetOptionValues ()
{
    Options::ResetOptionValues();

    m_level = lldb::eDescriptionLevelFull;
    m_internal = false;
}

//-------------------------------------------------------------------------
// CommandObjectBreakpointList
//-------------------------------------------------------------------------
#pragma mark List

CommandObjectBreakpointList::CommandObjectBreakpointList () :
    CommandObject ("breakpoint list",
                     "List some or all breakpoints at configurable levels of detail.",
                     "breakpoint list [<breakpoint-id>]")
{
}

CommandObjectBreakpointList::~CommandObjectBreakpointList ()
{
}

Options *
CommandObjectBreakpointList::GetOptions ()
{
    return &m_options;
}

bool
CommandObjectBreakpointList::Execute
(
    CommandInterpreter &interpreter,
    Args& args,
    CommandReturnObject &result
)
{
    Target *target = interpreter.GetDebugger().GetSelectedTarget().get();
    if (target == NULL)
    {
        result.AppendError ("Invalid target, set executable file using 'file' command.");
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        return true;
    }

    const BreakpointList &breakpoints = target->GetBreakpointList(m_options.m_internal);
    Mutex::Locker locker;
    target->GetBreakpointList(m_options.m_internal).GetListMutex(locker);

    size_t num_breakpoints = breakpoints.GetSize();

    if (num_breakpoints == 0)
    {
        result.AppendMessage ("No breakpoints currently set.");
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        return true;
    }

    StreamString &output_stream = result.GetOutputStream();

    if (args.GetArgumentCount() == 0)
    {
        // No breakpoint selected; show info about all currently set breakpoints.
        result.AppendMessage ("Current breakpoints:");
        for (size_t i = 0; i < num_breakpoints; ++i)
        {
            Breakpoint *breakpoint = breakpoints.GetBreakpointAtIndex (i).get();
            AddBreakpointDescription (&output_stream, breakpoint, m_options.m_level);
        }
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
    }
    else
    {
        // Particular breakpoints selected; show info about that breakpoint.
        BreakpointIDList valid_bp_ids;
        CommandObjectMultiwordBreakpoint::VerifyBreakpointIDs (args, target, result, &valid_bp_ids);

        if (result.Succeeded())
        {
            for (size_t i = 0; i < valid_bp_ids.GetSize(); ++i)
            {
                BreakpointID cur_bp_id = valid_bp_ids.GetBreakpointIDAtIndex (i);
                Breakpoint *breakpoint = target->GetBreakpointByID (cur_bp_id.GetBreakpointID()).get();
                AddBreakpointDescription (&output_stream, breakpoint, m_options.m_level);
            }
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            result.AppendError ("Invalid breakpoint id.");
            result.SetStatus (eReturnStatusFailed);
        }
    }

    return result.Succeeded();
}

//-------------------------------------------------------------------------
// CommandObjectBreakpointEnable
//-------------------------------------------------------------------------
#pragma mark Enable

CommandObjectBreakpointEnable::CommandObjectBreakpointEnable () :
    CommandObject ("enable",
                     "Enables the specified disabled breakpoint(s).  If no breakpoints are specified, enables all of them.",
                     "breakpoint enable [<breakpoint-id> | <breakpoint-id-list>]")
{
    // This command object can either be called via 'enable' or 'breakpoint enable'.  Because it has two different
    // potential invocation methods, we need to be a little tricky about generating the syntax string.
    //StreamString tmp_string;
    //tmp_string.Printf ("%s <breakpoint-id>", GetCommandName());
    //m_cmd_syntax.assign (tmp_string.GetData(), tmp_string.GetSize());
}


CommandObjectBreakpointEnable::~CommandObjectBreakpointEnable ()
{
}


bool
CommandObjectBreakpointEnable::Execute 
(
    CommandInterpreter &interpreter,
    Args& args, 
    CommandReturnObject &result
)
{
    Target *target = interpreter.GetDebugger().GetSelectedTarget().get();
    if (target == NULL)
    {
        result.AppendError ("Invalid target, set executable file using 'file' command.");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    Mutex::Locker locker;
    target->GetBreakpointList().GetListMutex(locker);

    const BreakpointList &breakpoints = target->GetBreakpointList();

    size_t num_breakpoints = breakpoints.GetSize();

    if (num_breakpoints == 0)
    {
        result.AppendError ("No breakpoints exist to be enabled.");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    if (args.GetArgumentCount() == 0)
    {
        // No breakpoint selected; enable all currently set breakpoints.
        target->EnableAllBreakpoints ();
        result.AppendMessageWithFormat ("All breakpoints enabled. (%d breakpoints)\n", num_breakpoints);
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
    }
    else
    {
        // Particular breakpoint selected; enable that breakpoint.
        BreakpointIDList valid_bp_ids;
        CommandObjectMultiwordBreakpoint::VerifyBreakpointIDs (args, target, result, &valid_bp_ids);

        if (result.Succeeded())
        {
            int enable_count = 0;
            int loc_count = 0;
            const size_t count = valid_bp_ids.GetSize();
            for (size_t i = 0; i < count; ++i)
            {
                BreakpointID cur_bp_id = valid_bp_ids.GetBreakpointIDAtIndex (i);

                if (cur_bp_id.GetBreakpointID() != LLDB_INVALID_BREAK_ID)
                {
                    Breakpoint *breakpoint = target->GetBreakpointByID (cur_bp_id.GetBreakpointID()).get();
                    if (cur_bp_id.GetLocationID() != LLDB_INVALID_BREAK_ID)
                    {
                        BreakpointLocation *location = breakpoint->FindLocationByID (cur_bp_id.GetLocationID()).get();
                        if (location)
                        {
                            location->SetEnabled (true);
                            ++loc_count;
                        }
                    }
                    else
                    {
                        breakpoint->SetEnabled (true);
                        ++enable_count;
                    }
                }
            }
            result.AppendMessageWithFormat ("%d breakpoints enabled.\n", enable_count + loc_count);
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
    }

    return result.Succeeded();
}

//-------------------------------------------------------------------------
// CommandObjectBreakpointDisable
//-------------------------------------------------------------------------
#pragma mark Disable

CommandObjectBreakpointDisable::CommandObjectBreakpointDisable () :
    CommandObject ("disable",
                   "Disables the specified breakpoint(s) without removing it/them.  If no breakpoints are specified, disables them all.",
                   "disable [<breakpoint-id> | <breakpoint-id-list>]")
{
    // This command object can either be called via 'enable' or 'breakpoint enable'.  Because it has two different
    // potential invocation methods, we need to be a little tricky about generating the syntax string.
    //StreamString tmp_string;
    //tmp_string.Printf ("%s <breakpoint-id>", GetCommandName());
    //m_cmd_syntax.assign(tmp_string.GetData(), tmp_string.GetSize());
}

CommandObjectBreakpointDisable::~CommandObjectBreakpointDisable ()
{
}

bool
CommandObjectBreakpointDisable::Execute
(
    CommandInterpreter &interpreter,
    Args& args, 
    CommandReturnObject &result
)
{
    Target *target = interpreter.GetDebugger().GetSelectedTarget().get();
    if (target == NULL)
    {
        result.AppendError ("Invalid target, set executable file using 'file' command.");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    Mutex::Locker locker;
    target->GetBreakpointList().GetListMutex(locker);

    const BreakpointList &breakpoints = target->GetBreakpointList();
    size_t num_breakpoints = breakpoints.GetSize();

    if (num_breakpoints == 0)
    {
        result.AppendError ("No breakpoints exist to be disabled.");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    if (args.GetArgumentCount() == 0)
    {
        // No breakpoint selected; disable all currently set breakpoints.
        target->DisableAllBreakpoints ();
        result.AppendMessageWithFormat ("All breakpoints disabled. (%d breakpoints)\n", num_breakpoints);
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
    }
    else
    {
        // Particular breakpoint selected; disable that breakpoint.
        BreakpointIDList valid_bp_ids;

        CommandObjectMultiwordBreakpoint::VerifyBreakpointIDs (args, target, result, &valid_bp_ids);

        if (result.Succeeded())
        {
            int disable_count = 0;
            int loc_count = 0;
            const size_t count = valid_bp_ids.GetSize();
            for (size_t i = 0; i < count; ++i)
            {
                BreakpointID cur_bp_id = valid_bp_ids.GetBreakpointIDAtIndex (i);

                if (cur_bp_id.GetBreakpointID() != LLDB_INVALID_BREAK_ID)
                {
                    Breakpoint *breakpoint = target->GetBreakpointByID (cur_bp_id.GetBreakpointID()).get();
                    if (cur_bp_id.GetLocationID() != LLDB_INVALID_BREAK_ID)
                    {
                        BreakpointLocation *location = breakpoint->FindLocationByID (cur_bp_id.GetLocationID()).get();
                        if (location)
                        {
                            location->SetEnabled (false);
                            ++loc_count;
                        }
                    }
                    else
                    {
                        breakpoint->SetEnabled (false);
                        ++disable_count;
                    }
                }
            }
            result.AppendMessageWithFormat ("%d breakpoints disabled.\n", disable_count + loc_count);
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
    }

    return result.Succeeded();
}

//-------------------------------------------------------------------------
// CommandObjectBreakpointDelete
//-------------------------------------------------------------------------
#pragma mark Delete

CommandObjectBreakpointDelete::CommandObjectBreakpointDelete() :
    CommandObject ("breakpoint delete",
                   "Delete the specified breakpoint(s).  If no breakpoints are specified, deletes them all.",
                   "breakpoint delete [<breakpoint-id> | <breakpoint-id-list>]")
{
}


CommandObjectBreakpointDelete::~CommandObjectBreakpointDelete ()
{
}

bool
CommandObjectBreakpointDelete::Execute 
(
    CommandInterpreter &interpreter,
    Args& args, 
    CommandReturnObject &result
)
{
    Target *target = interpreter.GetDebugger().GetSelectedTarget().get();
    if (target == NULL)
    {
        result.AppendError ("Invalid target, set executable file using 'file' command.");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    Mutex::Locker locker;
    target->GetBreakpointList().GetListMutex(locker);
    
    const BreakpointList &breakpoints = target->GetBreakpointList();

    size_t num_breakpoints = breakpoints.GetSize();

    if (num_breakpoints == 0)
    {
        result.AppendError ("No breakpoints exist to be deleted.");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    if (args.GetArgumentCount() == 0)
    {
        // No breakpoint selected; disable all currently set breakpoints.
        if (args.GetArgumentCount() != 0)
        {
            result.AppendErrorWithFormat ("Specify breakpoints to delete with the -i option.\n");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        target->RemoveAllBreakpoints ();
        result.AppendMessageWithFormat ("All breakpoints removed. (%d breakpoints)\n", num_breakpoints);
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
    }
    else
    {
        // Particular breakpoint selected; disable that breakpoint.
        BreakpointIDList valid_bp_ids;
        CommandObjectMultiwordBreakpoint::VerifyBreakpointIDs (args, target, result, &valid_bp_ids);

        if (result.Succeeded())
        {
            int delete_count = 0;
            int disable_count = 0;
            const size_t count = valid_bp_ids.GetSize();
            for (size_t i = 0; i < count; ++i)
            {
                BreakpointID cur_bp_id = valid_bp_ids.GetBreakpointIDAtIndex (i);

                if (cur_bp_id.GetBreakpointID() != LLDB_INVALID_BREAK_ID)
                {
                    if (cur_bp_id.GetLocationID() != LLDB_INVALID_BREAK_ID)
                    {
                        Breakpoint *breakpoint = target->GetBreakpointByID (cur_bp_id.GetBreakpointID()).get();
                        BreakpointLocation *location = breakpoint->FindLocationByID (cur_bp_id.GetLocationID()).get();
                        // It makes no sense to try to delete individual locations, so we disable them instead.
                        if (location)
                        {
                            location->SetEnabled (false);
                            ++disable_count;
                        }
                    }
                    else
                    {
                        target->RemoveBreakpointByID (cur_bp_id.GetBreakpointID());
                        ++delete_count;
                    }
                }
            }
            result.AppendMessageWithFormat ("%d breakpoints deleted; %d breakpoint locations disabled.\n",
                                           delete_count, disable_count);
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
    }
    return result.Succeeded();
}

//-------------------------------------------------------------------------
// CommandObjectBreakpointModify::CommandOptions
//-------------------------------------------------------------------------
#pragma mark Modify::CommandOptions

CommandObjectBreakpointModify::CommandOptions::CommandOptions() :
    Options (),
    m_ignore_count (0),
    m_thread_id(LLDB_INVALID_THREAD_ID),
    m_thread_index (UINT32_MAX),
    m_thread_name(),
    m_queue_name(),
    m_enable_passed (false),
    m_enable_value (false),
    m_name_passed (false),
    m_queue_passed (false)
{
}

CommandObjectBreakpointModify::CommandOptions::~CommandOptions ()
{
}

lldb::OptionDefinition
CommandObjectBreakpointModify::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "ignore_count", 'k', required_argument,   NULL, 0, NULL,
        "Set the number of times this breakpoint is sKipped before stopping." },

    { LLDB_OPT_SET_ALL, false, "thread_index",       'x', required_argument, NULL, NULL, "<thread_index>",
        "The breakpoint stops only for the thread whose indeX matches this argument."},

    { LLDB_OPT_SET_ALL, false, "thread_id",       't', required_argument, NULL, NULL, "<thread_id>",
        "The breakpoint stops only for the thread whose TID matches this argument."},

    { LLDB_OPT_SET_ALL, false, "thread_name",       'T', required_argument, NULL, NULL, "<thread_name>",
        "The breakpoint stops only for the thread whose thread name matches this argument."},

    { LLDB_OPT_SET_ALL, false, "queue_name",       'q', required_argument, NULL, NULL, "<queue_name>",
        "The breakpoint stops only for threads in the queue whose name is given by this argument."},

    { LLDB_OPT_SET_1, false, "enable",       'e', no_argument, NULL, NULL, NULL,
        "Enable the breakpoint."},

    { LLDB_OPT_SET_2, false, "disable",       'd', no_argument, NULL, NULL, NULL,
        "Disable the breakpoint."},


    { 0, false, NULL, 0, 0, NULL, 0, NULL, NULL }
};

const lldb::OptionDefinition*
CommandObjectBreakpointModify::CommandOptions::GetDefinitions ()
{
    return g_option_table;
}

Error
CommandObjectBreakpointModify::CommandOptions::SetOptionValue (int option_idx, const char *option_arg)
{
    Error error;
    char short_option = (char) m_getopt_table[option_idx].val;

    switch (short_option)
    {
        case 'd':
            m_enable_passed = true;
            m_enable_value = false;
            break;
        case 'e':
            m_enable_passed = true;
            m_enable_value = true;
            break;
        case 'k':
        {
            m_ignore_count = Args::StringToUInt32(optarg, UINT32_MAX, 0);
            if (m_ignore_count == UINT32_MAX)
               error.SetErrorStringWithFormat ("Invalid ignore count '%s'.\n", optarg);
        }
        break;
        case 't' :
        {
            m_thread_id = Args::StringToUInt64(optarg, LLDB_INVALID_THREAD_ID, 0);
            if (m_thread_id == LLDB_INVALID_THREAD_ID)
               error.SetErrorStringWithFormat ("Invalid thread id string '%s'.\n", optarg);
        }
        break;
        case 'T':
            if (option_arg != NULL)
                m_thread_name = option_arg;
            else
                m_thread_name.clear();
            m_name_passed = true;
            break;
        case 'q':
            if (option_arg != NULL)
                m_queue_name = option_arg;
            else
                m_queue_name.clear();
            m_queue_passed = true;
            break;
        case 'x':
        {
            m_thread_index = Args::StringToUInt32 (optarg, UINT32_MAX, 0);
            if (m_thread_id == UINT32_MAX)
               error.SetErrorStringWithFormat ("Invalid thread index string '%s'.\n", optarg);
            
        }
        break;
        default:
            error.SetErrorStringWithFormat ("Unrecognized option '%c'.\n", short_option);
            break;
    }

    return error;
}

void
CommandObjectBreakpointModify::CommandOptions::ResetOptionValues ()
{
    Options::ResetOptionValues();

    m_ignore_count = 0;
    m_thread_id = LLDB_INVALID_THREAD_ID;
    m_thread_index = UINT32_MAX;
    m_thread_name.clear();
    m_queue_name.clear();
    m_enable_passed = false;
    m_queue_passed = false;
    m_name_passed = false;
}

//-------------------------------------------------------------------------
// CommandObjectBreakpointModify
//-------------------------------------------------------------------------
#pragma mark Modify

CommandObjectBreakpointModify::CommandObjectBreakpointModify () :
    CommandObject ("breakpoint modify", "Modifys the options on a breakpoint or set of breakpoints in the executable.", 
                   "breakpoint modify <cmd-options> break-id [break-id ...]")
{
}

CommandObjectBreakpointModify::~CommandObjectBreakpointModify ()
{
}

Options *
CommandObjectBreakpointModify::GetOptions ()
{
    return &m_options;
}

bool
CommandObjectBreakpointModify::Execute
(
    CommandInterpreter &interpreter,
    Args& command,
    CommandReturnObject &result
)
{
    if (command.GetArgumentCount() == 0)
    {
        result.AppendError ("No breakpoints specified.");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    Target *target = interpreter.GetDebugger().GetSelectedTarget().get();
    if (target == NULL)
    {
        result.AppendError ("Invalid target, set executable file using 'file' command.");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    Mutex::Locker locker;
    target->GetBreakpointList().GetListMutex(locker);
    
    BreakpointIDList valid_bp_ids;

    CommandObjectMultiwordBreakpoint::VerifyBreakpointIDs (command, target, result, &valid_bp_ids);

    if (result.Succeeded())
    {
        const size_t count = valid_bp_ids.GetSize(); 
        for (size_t i = 0; i < count; ++i)
        {
            BreakpointID cur_bp_id = valid_bp_ids.GetBreakpointIDAtIndex (i);

            if (cur_bp_id.GetBreakpointID() != LLDB_INVALID_BREAK_ID)
            {
                Breakpoint *bp = target->GetBreakpointByID (cur_bp_id.GetBreakpointID()).get();
                if (cur_bp_id.GetLocationID() != LLDB_INVALID_BREAK_ID)
                {
                    BreakpointLocation *location = bp->FindLocationByID (cur_bp_id.GetLocationID()).get();
                    if (location)
                    {
                        if (m_options.m_thread_id != LLDB_INVALID_THREAD_ID)
                            location->SetThreadID (m_options.m_thread_id);
                            
                        if (m_options.m_thread_index != UINT32_MAX)
                            location->GetLocationOptions()->GetThreadSpec()->SetIndex(m_options.m_thread_index);
                        
                        if (m_options.m_name_passed)
                            location->GetLocationOptions()->GetThreadSpec()->SetName(m_options.m_thread_name.c_str());
                        
                        if (m_options.m_queue_passed)
                            location->GetLocationOptions()->GetThreadSpec()->SetQueueName(m_options.m_queue_name.c_str());
                            
                        if (m_options.m_ignore_count != 0)
                            location->GetLocationOptions()->SetIgnoreCount(m_options.m_ignore_count);
                            
                        if (m_options.m_enable_passed)
                            location->SetEnabled (m_options.m_enable_value);
                    }
                }
                else
                {
                    if (m_options.m_thread_id != LLDB_INVALID_THREAD_ID)
                        bp->SetThreadID (m_options.m_thread_id);
                        
                    if (m_options.m_thread_index != UINT32_MAX)
                        bp->GetOptions()->GetThreadSpec()->SetIndex(m_options.m_thread_index);
                    
                    if (m_options.m_name_passed)
                        bp->GetOptions()->GetThreadSpec()->SetName(m_options.m_thread_name.c_str());
                    
                    if (m_options.m_queue_passed)
                        bp->GetOptions()->GetThreadSpec()->SetQueueName(m_options.m_queue_name.c_str());
                        
                    if (m_options.m_ignore_count != 0)
                        bp->GetOptions()->SetIgnoreCount(m_options.m_ignore_count);
                        
                    if (m_options.m_enable_passed)
                        bp->SetEnabled (m_options.m_enable_value);

                }
            }
        }
    }
    
    return result.Succeeded();
}


