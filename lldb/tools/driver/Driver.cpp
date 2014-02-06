//===-- Driver.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <fcntl.h>

#include <string>

#include <thread>
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBCommunication.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBHostOS.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBProcess.h"

using namespace lldb;

static void reset_stdin_termios ();
static bool g_old_stdin_termios_is_valid = false;
static struct termios g_old_stdin_termios;

static char *g_debugger_name =  (char *) "";
static Driver *g_driver = NULL;

// In the Driver::MainLoop, we change the terminal settings.  This function is
// added as an atexit handler to make sure we clean them up.
static void
reset_stdin_termios ()
{
    if (g_old_stdin_termios_is_valid)
    {
        g_old_stdin_termios_is_valid = false;
        ::tcsetattr (STDIN_FILENO, TCSANOW, &g_old_stdin_termios);
    }
}

typedef struct
{
    uint32_t usage_mask;                     // Used to mark options that can be used together.  If (1 << n & usage_mask) != 0
                                             // then this option belongs to option set n.
    bool required;                           // This option is required (in the current usage level)
    const char * long_option;                // Full name for this option.
    int short_option;                        // Single character for this option.
    int option_has_arg;                      // no_argument, required_argument or optional_argument
    uint32_t completion_type;                // Cookie the option class can use to do define the argument completion.
    lldb::CommandArgumentType argument_type; // Type of argument this option takes
    const char *  usage_text;                // Full text explaining what this options does and what (if any) argument to
                                             // pass it.
} OptionDefinition;

#define LLDB_3_TO_5 LLDB_OPT_SET_3|LLDB_OPT_SET_4|LLDB_OPT_SET_5
#define LLDB_4_TO_5 LLDB_OPT_SET_4|LLDB_OPT_SET_5

static OptionDefinition g_options[] =
{
    { LLDB_OPT_SET_1,    true , "help"           , 'h', no_argument      , 0,  eArgTypeNone,
        "Prints out the usage information for the LLDB debugger." },
    { LLDB_OPT_SET_2,    true , "version"        , 'v', no_argument      , 0,  eArgTypeNone,
        "Prints out the current version number of the LLDB debugger." },
    { LLDB_OPT_SET_3,    true , "arch"           , 'a', required_argument, 0,  eArgTypeArchitecture,
        "Tells the debugger to use the specified architecture when starting and running the program.  <architecture> must "
        "be one of the architectures for which the program was compiled." },
    { LLDB_OPT_SET_3,    true , "file"           , 'f', required_argument, 0,  eArgTypeFilename,
        "Tells the debugger to use the file <filename> as the program to be debugged." },
    { LLDB_OPT_SET_3,    false, "core"           , 'c', required_argument, 0,  eArgTypeFilename,
        "Tells the debugger to use the fullpath to <path> as the core file." },
    { LLDB_OPT_SET_5,    true , "attach-pid"     , 'p', required_argument, 0,  eArgTypePid,
        "Tells the debugger to attach to a process with the given pid." },
    { LLDB_OPT_SET_4,    true , "attach-name"    , 'n', required_argument, 0,  eArgTypeProcessName,
        "Tells the debugger to attach to a process with the given name." },
    { LLDB_OPT_SET_4,    true , "wait-for"       , 'w', no_argument      , 0,  eArgTypeNone,
        "Tells the debugger to wait for a process with the given pid or name to launch before attaching." },
    { LLDB_3_TO_5,       false, "source"         , 's', required_argument, 0,  eArgTypeFilename,
        "Tells the debugger to read in and execute the lldb commands in the given file, after any file provided on the command line has been loaded." },
    { LLDB_3_TO_5,       false, "one-line"         , 'o', required_argument, 0,  eArgTypeNone,
        "Tells the debugger to execute this one-line lldb command after any file provided on the command line has been loaded." },
    { LLDB_3_TO_5,       false, "source-before-file"         , 'S', required_argument, 0,  eArgTypeFilename,
        "Tells the debugger to read in and execute the lldb commands in the given file, before any file provided on the command line has been loaded." },
    { LLDB_3_TO_5,       false, "one-line-before-file"         , 'O', required_argument, 0,  eArgTypeNone,
        "Tells the debugger to execute this one-line lldb command before any file provided on the command line has been loaded." },
    { LLDB_3_TO_5,       false, "source-quietly"          , 'Q', no_argument      , 0,  eArgTypeNone,
        "Tells the debugger suppress output from commands provided in the -s, -S, -O and -o commands." },
    { LLDB_3_TO_5,       false, "editor"         , 'e', no_argument      , 0,  eArgTypeNone,
        "Tells the debugger to open source files using the host's \"external editor\" mechanism." },
    { LLDB_3_TO_5,       false, "no-lldbinit"    , 'x', no_argument      , 0,  eArgTypeNone,
        "Do not automatically parse any '.lldbinit' files." },
    { LLDB_3_TO_5,       false, "no-use-colors"  , 'X', no_argument      , 0,  eArgTypeNone,
        "Do not use colors." },
    { LLDB_OPT_SET_6,    true , "python-path"    , 'P', no_argument      , 0,  eArgTypeNone,
        "Prints out the path to the lldb.py file for this version of lldb." },
    { LLDB_3_TO_5,       false, "script-language", 'l', required_argument, 0,  eArgTypeScriptLang,
        "Tells the debugger to use the specified scripting language for user-defined scripts, rather than the default.  "
        "Valid scripting languages that can be specified include Python, Perl, Ruby and Tcl.  Currently only the Python "
        "extensions have been implemented." },
    { LLDB_3_TO_5,       false, "debug"          , 'd', no_argument      , 0,  eArgTypeNone,
        "Tells the debugger to print out extra information for debugging itself." },
    { 0,                 false, NULL             , 0  , 0                , 0,  eArgTypeNone,         NULL }
};

static const uint32_t last_option_set_with_args = 2;

Driver::Driver () :
    SBBroadcaster ("Driver"),
    m_debugger (SBDebugger::Create(false)),
    m_option_data ()
{
    // We want to be able to handle CTRL+D in the terminal to have it terminate
    // certain input
    m_debugger.SetCloseInputOnEOF (false);
    g_debugger_name = (char *) m_debugger.GetInstanceName();
    if (g_debugger_name == NULL)
        g_debugger_name = (char *) "";
    g_driver = this;
}

Driver::~Driver ()
{
    g_driver = NULL;
    g_debugger_name = NULL;
}


// This function takes INDENT, which tells how many spaces to output at the front
// of each line; TEXT, which is the text that is to be output. It outputs the 
// text, on multiple lines if necessary, to RESULT, with INDENT spaces at the 
// front of each line.  It breaks lines on spaces, tabs or newlines, shortening 
// the line if necessary to not break in the middle of a word. It assumes that 
// each output line should contain a maximum of OUTPUT_MAX_COLUMNS characters.

void
OutputFormattedUsageText (FILE *out, int indent, const char *text, int output_max_columns)
{
    int len = strlen (text);
    std::string text_string (text);

    // Force indentation to be reasonable.
    if (indent >= output_max_columns)
        indent = 0;

    // Will it all fit on one line?

    if (len + indent < output_max_columns)
        // Output as a single line
        fprintf (out, "%*s%s\n", indent, "", text);
    else
    {
        // We need to break it up into multiple lines.
        int text_width = output_max_columns - indent - 1;
        int start = 0;
        int end = start;
        int final_end = len;
        int sub_len;

        while (end < final_end)
        {
              // Dont start the 'text' on a space, since we're already outputting the indentation.
              while ((start < final_end) && (text[start] == ' '))
                  start++;

              end = start + text_width;
              if (end > final_end)
                  end = final_end;
              else
              {
                  // If we're not at the end of the text, make sure we break the line on white space.
                  while (end > start
                         && text[end] != ' ' && text[end] != '\t' && text[end] != '\n')
                      end--;
              }
              sub_len = end - start;
              std::string substring = text_string.substr (start, sub_len);
              fprintf (out, "%*s%s\n", indent, "", substring.c_str());
              start = end + 1;
        }
    }
}

void
ShowUsage (FILE *out, OptionDefinition *option_table, Driver::OptionData data)
{
    uint32_t screen_width = 80;
    uint32_t indent_level = 0;
    const char *name = "lldb";
    
    fprintf (out, "\nUsage:\n\n");

    indent_level += 2;


    // First, show each usage level set of options, e.g. <cmd> [options-for-level-0]
    //                                                   <cmd> [options-for-level-1]
    //                                                   etc.

    uint32_t num_options;
    uint32_t num_option_sets = 0;
    
    for (num_options = 0; option_table[num_options].long_option != NULL; ++num_options)
    {
        uint32_t this_usage_mask = option_table[num_options].usage_mask;
        if (this_usage_mask == LLDB_OPT_SET_ALL)
        {
            if (num_option_sets == 0)
                num_option_sets = 1;
        }
        else
        {
            for (uint32_t j = 0; j < LLDB_MAX_NUM_OPTION_SETS; j++)
            {
                if (this_usage_mask & 1 << j)
                {
                    if (num_option_sets <= j)
                        num_option_sets = j + 1;
                }
            }
        }
    }

    for (uint32_t opt_set = 0; opt_set < num_option_sets; opt_set++)
    {
        uint32_t opt_set_mask;
        
        opt_set_mask = 1 << opt_set;
        
        if (opt_set > 0)
            fprintf (out, "\n");
        fprintf (out, "%*s%s", indent_level, "", name);
        bool is_help_line = false;
        
        for (uint32_t i = 0; i < num_options; ++i)
        {
            if (option_table[i].usage_mask & opt_set_mask)
            {
                CommandArgumentType arg_type = option_table[i].argument_type;
                const char *arg_name = SBCommandInterpreter::GetArgumentTypeAsCString (arg_type);
                // This is a bit of a hack, but there's no way to say certain options don't have arguments yet...
                // so we do it by hand here.
                if (option_table[i].short_option == 'h')
                    is_help_line = true;
                    
                if (option_table[i].required)
                {
                    if (option_table[i].option_has_arg == required_argument)
                        fprintf (out, " -%c <%s>", option_table[i].short_option, arg_name);
                    else if (option_table[i].option_has_arg == optional_argument)
                        fprintf (out, " -%c [<%s>]", option_table[i].short_option, arg_name);
                    else
                        fprintf (out, " -%c", option_table[i].short_option);
                }
                else
                {
                    if (option_table[i].option_has_arg == required_argument)
                        fprintf (out, " [-%c <%s>]", option_table[i].short_option, arg_name);
                    else if (option_table[i].option_has_arg == optional_argument)
                        fprintf (out, " [-%c [<%s>]]", option_table[i].short_option, arg_name);
                    else
                        fprintf (out, " [-%c]", option_table[i].short_option);
                }
            }
        }
        if (!is_help_line && (opt_set <= last_option_set_with_args))
            fprintf (out, " [[--] <PROGRAM-ARG-1> [<PROGRAM_ARG-2> ...]]");
    }

    fprintf (out, "\n\n");

    // Now print out all the detailed information about the various options:  long form, short form and help text:
    //   -- long_name <argument>
    //   - short <argument>
    //   help text

    // This variable is used to keep track of which options' info we've printed out, because some options can be in
    // more than one usage level, but we only want to print the long form of its information once.

    Driver::OptionData::OptionSet options_seen;
    Driver::OptionData::OptionSet::iterator pos;

    indent_level += 5;

    for (uint32_t i = 0; i < num_options; ++i)
    {
        // Only print this option if we haven't already seen it.
        pos = options_seen.find (option_table[i].short_option);
        if (pos == options_seen.end())
        {
            CommandArgumentType arg_type = option_table[i].argument_type;
            const char *arg_name = SBCommandInterpreter::GetArgumentTypeAsCString (arg_type);

            options_seen.insert (option_table[i].short_option);
            fprintf (out, "%*s-%c ", indent_level, "", option_table[i].short_option);
            if (arg_type != eArgTypeNone)
                fprintf (out, "<%s>", arg_name);
            fprintf (out, "\n");
            fprintf (out, "%*s--%s ", indent_level, "", option_table[i].long_option);
            if (arg_type != eArgTypeNone)
                fprintf (out, "<%s>", arg_name);
            fprintf (out, "\n");
            indent_level += 5;
            OutputFormattedUsageText (out, indent_level, option_table[i].usage_text, screen_width);
            indent_level -= 5;
            fprintf (out, "\n");
        }
    }

    indent_level -= 5;
    
    fprintf (out, "\n%*sNotes:\n",
             indent_level, "");
    indent_level += 5;
    
    fprintf (out, "\n%*sMultiple \"-s\" and \"-o\" options can be provided.  They will be processed from left to right in order, "
                  "\n%*swith the source files and commands interleaved.  The same is true of the \"-S\" and \"-O\" options."
                  "\n%*sThe before file and after file sets can intermixed freely, the command parser will sort them out."
                  "\n%*sThe order of the file specifiers (\"-c\", \"-f\", etc.) is not significant in this regard.\n\n",
             indent_level, "", 
             indent_level, "", 
             indent_level, "",
             indent_level, "");
    
    fprintf (out, "\n%*sIf you don't provide -f then the first argument will be the file to be debugged"
                  "\n%*swhich means that '%s -- <filename> [<ARG1> [<ARG2>]]' also works."
                  "\n%*sBut remember to end the options with \"--\" if any of your arguments have a \"-\" in them.\n\n",
             indent_level, "", 
             indent_level, "",
             name, 
             indent_level, "");
}

void
BuildGetOptTable (OptionDefinition *expanded_option_table, std::vector<struct option> &getopt_table, 
                  uint32_t num_options)
{
    if (num_options == 0)
        return;

    uint32_t i;
    uint32_t j;
    std::bitset<256> option_seen;

    getopt_table.resize (num_options + 1);

    for (i = 0, j = 0; i < num_options; ++i)
    {
        char short_opt = expanded_option_table[i].short_option;
        
        if (option_seen.test(short_opt) == false)
        {
            getopt_table[j].name    = expanded_option_table[i].long_option;
            getopt_table[j].has_arg = expanded_option_table[i].option_has_arg;
            getopt_table[j].flag    = NULL;
            getopt_table[j].val     = expanded_option_table[i].short_option;
            option_seen.set(short_opt);
            ++j;
        }
    }

    getopt_table[j].name    = NULL;
    getopt_table[j].has_arg = 0;
    getopt_table[j].flag    = NULL;
    getopt_table[j].val     = 0;

}

Driver::OptionData::OptionData () :
    m_args(),
    m_script_lang (lldb::eScriptLanguageDefault),
    m_core_file (),
    m_crash_log (),
    m_initial_commands (),
    m_after_file_commands (),
    m_debug_mode (false),
    m_source_quietly(false),
    m_print_version (false),
    m_print_python_path (false),
    m_print_help (false),
    m_wait_for(false),
    m_process_name(),
    m_process_pid(LLDB_INVALID_PROCESS_ID),
    m_use_external_editor(false),
    m_seen_options()
{
}

Driver::OptionData::~OptionData ()
{
}

void
Driver::OptionData::Clear ()
{
    m_args.clear ();
    m_script_lang = lldb::eScriptLanguageDefault;
    m_initial_commands.clear ();
    m_after_file_commands.clear ();
    m_debug_mode = false;
    m_source_quietly = false;
    m_print_help = false;
    m_print_version = false;
    m_print_python_path = false;
    m_use_external_editor = false;
    m_wait_for = false;
    m_process_name.erase();
    m_process_pid = LLDB_INVALID_PROCESS_ID;
}

void
Driver::OptionData::AddInitialCommand (const char *command, bool before_file, bool is_file, SBError &error)
{
    std::vector<std::pair<bool, std::string> > *command_set;
    if (before_file)
        command_set = &(m_initial_commands);
    else
        command_set = &(m_after_file_commands);

    if (is_file)
    {
        SBFileSpec file(command);
        if (file.Exists())
            command_set->push_back (std::pair<bool, std::string> (true, optarg));
        else if (file.ResolveExecutableLocation())
        {
            char final_path[PATH_MAX];
            file.GetPath (final_path, sizeof(final_path));
            std::string path_str (final_path);
            command_set->push_back (std::pair<bool, std::string> (true, path_str));
        }
        else
            error.SetErrorStringWithFormat("file specified in --source (-s) option doesn't exist: '%s'", optarg);
    }
    else
        command_set->push_back (std::pair<bool, std::string> (false, optarg));
}

void
Driver::ResetOptionValues ()
{
    m_option_data.Clear ();
}

const char *
Driver::GetFilename() const
{
    if (m_option_data.m_args.empty())
        return NULL;
    return m_option_data.m_args.front().c_str();
}

const char *
Driver::GetCrashLogFilename() const
{
    if (m_option_data.m_crash_log.empty())
        return NULL;
    return m_option_data.m_crash_log.c_str();
}

lldb::ScriptLanguage
Driver::GetScriptLanguage() const
{
    return m_option_data.m_script_lang;
}

void
Driver::ExecuteInitialCommands (bool before_file)
{
    size_t num_commands;
    std::vector<std::pair<bool, std::string> > *command_set;
    if (before_file)
        command_set = &(m_option_data.m_initial_commands);
    else
        command_set = &(m_option_data.m_after_file_commands);
    
    num_commands = command_set->size();
    SBCommandReturnObject result;
    bool old_async = GetDebugger().GetAsync();
    GetDebugger().SetAsync(false);
    for (size_t idx = 0; idx < num_commands; idx++)
    {
        bool is_file = (*command_set)[idx].first;
        const char *command = (*command_set)[idx].second.c_str();
        char command_string[PATH_MAX * 2];
        const bool dump_stream_only_if_no_immediate = true;
        const char *executed_command = command;
        if (is_file)
        {
            ::snprintf (command_string, sizeof(command_string), "command source -s %i '%s'", m_option_data.m_source_quietly, command);
            executed_command = command_string;
        }
        
        m_debugger.GetCommandInterpreter().HandleCommand (executed_command, result, false);
        if (!m_option_data.m_source_quietly || result.Succeeded() == false)
        {
            const size_t output_size = result.GetOutputSize();
            if (output_size > 0)
            {
                const char *cstr = result.GetOutput(dump_stream_only_if_no_immediate);
                if (cstr)
                    printf ("%s", cstr);
            }
            const size_t error_size = result.GetErrorSize();
            if (error_size > 0)
            {
                const char *cstr = result.GetError(dump_stream_only_if_no_immediate);
                if (cstr)
                    printf ("%s", cstr);
            }
        }
        
        if (result.Succeeded() == false)
        {
            const char *type = before_file ? "before file" : "after_file";
            if (is_file)
                ::fprintf(stderr, "Aborting %s command execution, command file: '%s' failed.\n", type, command);
            else
                ::fprintf(stderr, "Aborting %s command execution, command: '%s' failed.\n", type, command);
            break;
        }
        result.Clear();
    }
    GetDebugger().SetAsync(old_async);
}

bool
Driver::GetDebugMode() const
{
    return m_option_data.m_debug_mode;
}


// Check the arguments that were passed to this program to make sure they are valid and to get their
// argument values (if any).  Return a boolean value indicating whether or not to start up the full
// debugger (i.e. the Command Interpreter) or not.  Return FALSE if the arguments were invalid OR
// if the user only wanted help or version information.

SBError
Driver::ParseArgs (int argc, const char *argv[], FILE *out_fh, bool &exiting)
{
    ResetOptionValues ();

    SBCommandReturnObject result;

    SBError error;
    std::string option_string;
    struct option *long_options = NULL;
    std::vector<struct option> long_options_vector;
    uint32_t num_options;

    for (num_options = 0; g_options[num_options].long_option != NULL; ++num_options)
        /* Do Nothing. */;

    if (num_options == 0)
    {
        if (argc > 1)
            error.SetErrorStringWithFormat ("invalid number of options");
        return error;
    }

    BuildGetOptTable (g_options, long_options_vector, num_options);

    if (long_options_vector.empty())
        long_options = NULL;
    else
        long_options = &long_options_vector.front();

    if (long_options == NULL)
    {
        error.SetErrorStringWithFormat ("invalid long options");
        return error;
    }

    // Build the option_string argument for call to getopt_long_only.

    for (int i = 0; long_options[i].name != NULL; ++i)
    {
        if (long_options[i].flag == NULL)
        {
            option_string.push_back ((char) long_options[i].val);
            switch (long_options[i].has_arg)
            {
                default:
                case no_argument:
                    break;
                case required_argument:
                    option_string.push_back (':');
                    break;
                case optional_argument:
                    option_string.append ("::");
                    break;
            }
        }
    }

    // This is kind of a pain, but since we make the debugger in the Driver's constructor, we can't
    // know at that point whether we should read in init files yet.  So we don't read them in in the
    // Driver constructor, then set the flags back to "read them in" here, and then if we see the
    // "-n" flag, we'll turn it off again.  Finally we have to read them in by hand later in the
    // main loop.
    
    m_debugger.SkipLLDBInitFiles (false);
    m_debugger.SkipAppInitFiles (false);

    // Prepare for & make calls to getopt_long_only.
#if __GLIBC__
    optind = 0;
#else
    optreset = 1;
    optind = 1;
#endif
    int val;
    while (1)
    {
        int long_options_index = -1;
        val = ::getopt_long_only (argc, const_cast<char **>(argv), option_string.c_str(), long_options, &long_options_index);

        if (val == -1)
            break;
        else if (val == '?')
        {
            m_option_data.m_print_help = true;
            error.SetErrorStringWithFormat ("unknown or ambiguous option");
            break;
        }
        else if (val == 0)
            continue;
        else
        {
            m_option_data.m_seen_options.insert ((char) val);
            if (long_options_index == -1)
            {
                for (int i = 0;
                     long_options[i].name || long_options[i].has_arg || long_options[i].flag || long_options[i].val;
                     ++i)
                {
                    if (long_options[i].val == val)
                    {
                        long_options_index = i;
                        break;
                    }
                }
            }

            if (long_options_index >= 0)
            {
                const int short_option = g_options[long_options_index].short_option;

                switch (short_option)
                {
                    case 'h':
                        m_option_data.m_print_help = true;
                        break;

                    case 'v':
                        m_option_data.m_print_version = true;
                        break;

                    case 'P':
                        m_option_data.m_print_python_path = true;
                        break;

                    case 'c':
                        {
                            SBFileSpec file(optarg);
                            if (file.Exists())
                            {
                                m_option_data.m_core_file = optarg;
                            }
                            else
                                error.SetErrorStringWithFormat("file specified in --core (-c) option doesn't exist: '%s'", optarg);
                        }
                        break;
                    
                    case 'e':
                        m_option_data.m_use_external_editor = true;
                        break;

                    case 'x':
                        m_debugger.SkipLLDBInitFiles (true);
                        m_debugger.SkipAppInitFiles (true);
                        break;

                    case 'X':
                        m_debugger.SetUseColor (false);
                        break;

                    case 'f':
                        {
                            SBFileSpec file(optarg);
                            if (file.Exists())
                            {
                                m_option_data.m_args.push_back (optarg);
                            }
                            else if (file.ResolveExecutableLocation())
                            {
                                char path[PATH_MAX];
                                file.GetPath (path, sizeof(path));
                                m_option_data.m_args.push_back (path);
                            }
                            else
                                error.SetErrorStringWithFormat("file specified in --file (-f) option doesn't exist: '%s'", optarg);
                        }
                        break;

                    case 'a':
                        if (!m_debugger.SetDefaultArchitecture (optarg))
                            error.SetErrorStringWithFormat("invalid architecture in the -a or --arch option: '%s'", optarg);
                        break;

                    case 'l':
                        m_option_data.m_script_lang = m_debugger.GetScriptingLanguage (optarg);
                        break;

                    case 'd':
                        m_option_data.m_debug_mode = true;
                        break;

                    case 'Q':
                        m_option_data.m_source_quietly = true;
                        break;

                    case 'n':
                        m_option_data.m_process_name = optarg;
                        break;
                    
                    case 'w':
                        m_option_data.m_wait_for = true;
                        break;
                        
                    case 'p':
                        {
                            char *remainder;
                            m_option_data.m_process_pid = strtol (optarg, &remainder, 0);
                            if (remainder == optarg || *remainder != '\0')
                                error.SetErrorStringWithFormat ("Could not convert process PID: \"%s\" into a pid.",
                                                                optarg);
                        }
                        break;
                    case 's':
                        m_option_data.AddInitialCommand(optarg, false, true, error);
                        break;
                    case 'o':
                        m_option_data.AddInitialCommand(optarg, false, false, error);
                        break;
                    case 'S':
                        m_option_data.AddInitialCommand(optarg, true, true, error);
                        break;
                    case 'O':
                        m_option_data.AddInitialCommand(optarg, true, false, error);
                        break;
                    default:
                        m_option_data.m_print_help = true;
                        error.SetErrorStringWithFormat ("unrecognized option %c", short_option);
                        break;
                }
            }
            else
            {
                error.SetErrorStringWithFormat ("invalid option with value %i", val);
            }
            if (error.Fail())
            {
                return error;
            }
        }
    }
    
    if (error.Fail() || m_option_data.m_print_help)
    {
        ShowUsage (out_fh, g_options, m_option_data);
        exiting = true;
    }
    else if (m_option_data.m_print_version)
    {
        ::fprintf (out_fh, "%s\n", m_debugger.GetVersionString());
        exiting = true;
    }
    else if (m_option_data.m_print_python_path)
    {
        SBFileSpec python_file_spec = SBHostOS::GetLLDBPythonPath();
        if (python_file_spec.IsValid())
        {
            char python_path[PATH_MAX];
            size_t num_chars = python_file_spec.GetPath(python_path, PATH_MAX);
            if (num_chars < PATH_MAX)
            {
                ::fprintf (out_fh, "%s\n", python_path);
            }
            else
                ::fprintf (out_fh, "<PATH TOO LONG>\n");
        }
        else
            ::fprintf (out_fh, "<COULD NOT FIND PATH>\n");
        exiting = true;
    }
    else if (m_option_data.m_process_name.empty() && m_option_data.m_process_pid == LLDB_INVALID_PROCESS_ID)
    {
        // Any arguments that are left over after option parsing are for
        // the program. If a file was specified with -f then the filename
        // is already in the m_option_data.m_args array, and any remaining args
        // are arguments for the inferior program. If no file was specified with
        // -f, then what is left is the program name followed by any arguments.

        // Skip any options we consumed with getopt_long_only
        argc -= optind;
        argv += optind;

        if (argc > 0)
        {
            for (int arg_idx=0; arg_idx<argc; ++arg_idx)
            {
                const char *arg = argv[arg_idx];
                if (arg)
                    m_option_data.m_args.push_back (arg);
            }
        }
        
    }
    else
    {
        // Skip any options we consumed with getopt_long_only
        argc -= optind;
        //argv += optind; // Commented out to keep static analyzer happy

        if (argc > 0)
            ::fprintf (out_fh, "Warning: program arguments are ignored when attaching.\n");
    }

    return error;
}

void
Driver::MainLoop ()
{
    if (::tcgetattr(STDIN_FILENO, &g_old_stdin_termios) == 0)
    {
        g_old_stdin_termios_is_valid = true;
        atexit (reset_stdin_termios);
    }

    ::setbuf (stdin, NULL);
    ::setbuf (stdout, NULL);

    m_debugger.SetErrorFileHandle (stderr, false);
    m_debugger.SetOutputFileHandle (stdout, false);
    m_debugger.SetInputFileHandle (stdin, true);
    
    m_debugger.SetUseExternalEditor(m_option_data.m_use_external_editor);

    struct winsize window_size;
    if (isatty (STDIN_FILENO)
        && ::ioctl (STDIN_FILENO, TIOCGWINSZ, &window_size) == 0)
    {
        if (window_size.ws_col > 0)
            m_debugger.SetTerminalWidth (window_size.ws_col);
    }

    SBCommandInterpreter sb_interpreter = m_debugger.GetCommandInterpreter();
    
    // Before we handle any options from the command line, we parse the
    // .lldbinit file in the user's home directory.
    SBCommandReturnObject result;
    sb_interpreter.SourceInitFileInHomeDirectory(result);
    if (GetDebugMode())
    {
        result.PutError (m_debugger.GetErrorFileHandle());
        result.PutOutput (m_debugger.GetOutputFileHandle());
    }

    // Now we handle options we got from the command line
    // First source in the commands specified to be run before the file arguments are processed.
    ExecuteInitialCommands(true);
    
    // Was there a core file specified?
    std::string core_file_spec("");
    if (!m_option_data.m_core_file.empty())
        core_file_spec.append("--core ").append(m_option_data.m_core_file);
    
    char command_string[PATH_MAX * 2];
    const size_t num_args = m_option_data.m_args.size();
    if (num_args > 0)
    {
        char arch_name[64];
        if (m_debugger.GetDefaultArchitecture (arch_name, sizeof (arch_name)))
            ::snprintf (command_string,
                        sizeof (command_string),
                        "target create --arch=%s %s \"%s\"",
                        arch_name,
                        core_file_spec.c_str(),
                        m_option_data.m_args[0].c_str());
        else
            ::snprintf (command_string,
                        sizeof(command_string),
                        "target create %s \"%s\"",
                        core_file_spec.c_str(),
                        m_option_data.m_args[0].c_str());
        
        m_debugger.HandleCommand (command_string);
        
        if (num_args > 1)
        {
            m_debugger.HandleCommand ("settings clear target.run-args");
            char arg_cstr[1024];
            for (size_t arg_idx = 1; arg_idx < num_args; ++arg_idx)
            {
                ::snprintf (arg_cstr,
                            sizeof(arg_cstr),
                            "settings append target.run-args \"%s\"",
                            m_option_data.m_args[arg_idx].c_str());
                m_debugger.HandleCommand (arg_cstr);
            }
        }
    }
    else if (!core_file_spec.empty())
    {
        ::snprintf (command_string,
                    sizeof(command_string),
                    "target create %s",
                    core_file_spec.c_str());
        m_debugger.HandleCommand (command_string);;
    }
    else if (!m_option_data.m_process_name.empty())
    {
        ::snprintf (command_string, 
                    sizeof(command_string), 
                    "process attach --name '%s'%s", 
                    m_option_data.m_process_name.c_str(), 
                    m_option_data.m_wait_for ? " --waitfor" : "");
        m_debugger.HandleCommand (command_string);
    }
    else if (LLDB_INVALID_PROCESS_ID != m_option_data.m_process_pid)
    {
        ::snprintf (command_string, 
                    sizeof(command_string), 
                    "process attach --pid %" PRIu64, 
                    m_option_data.m_process_pid);
        m_debugger.HandleCommand (command_string);
    }

    ExecuteInitialCommands(false);

    // Now that all option parsing is done, we try and parse the .lldbinit
    // file in the current working directory
    sb_interpreter.SourceInitFileInCurrentWorkingDirectory (result);
    if (GetDebugMode())
    {
        result.PutError(m_debugger.GetErrorFileHandle());
        result.PutOutput(m_debugger.GetOutputFileHandle());
    }
    
    bool handle_events = true;
    bool spawn_thread = false;
    m_debugger.RunCommandInterpreter(handle_events, spawn_thread);
    
    reset_stdin_termios();
    fclose (stdin);
    
    SBDebugger::Destroy (m_debugger);
}


void
Driver::ResizeWindow (unsigned short col)
{
    GetDebugger().SetTerminalWidth (col);
}

void
sigwinch_handler (int signo)
{
    struct winsize window_size;
    if (isatty (STDIN_FILENO)
        && ::ioctl (STDIN_FILENO, TIOCGWINSZ, &window_size) == 0)
    {
        if ((window_size.ws_col > 0) && g_driver != NULL)
        {
            g_driver->ResizeWindow (window_size.ws_col);
        }
    }
}

void
sigint_handler (int signo)
{
	static bool g_interrupt_sent = false;
    if (g_driver)
	{
		if (!g_interrupt_sent)
		{
			g_interrupt_sent = true;
        	g_driver->GetDebugger().DispatchInputInterrupt();
			g_interrupt_sent = false;
			return;
		}
	}
    
	exit (signo);
}

void
sigtstp_handler (int signo)
{
    g_driver->GetDebugger().SaveInputTerminalState();
    signal (signo, SIG_DFL);
    kill (getpid(), signo);
    signal (signo, sigtstp_handler);
}

void
sigcont_handler (int signo)
{
    g_driver->GetDebugger().RestoreInputTerminalState();
    signal (signo, SIG_DFL);
    kill (getpid(), signo);
    signal (signo, sigcont_handler);
}

int
main (int argc, char const *argv[], const char *envp[])
{
    SBDebugger::Initialize();
    
    SBHostOS::ThreadCreated ("<lldb.driver.main-thread>");

    signal (SIGPIPE, SIG_IGN);
    signal (SIGWINCH, sigwinch_handler);
    signal (SIGINT, sigint_handler);
    signal (SIGTSTP, sigtstp_handler);
    signal (SIGCONT, sigcont_handler);

    // Create a scope for driver so that the driver object will destroy itself
    // before SBDebugger::Terminate() is called.
    {
        Driver driver;

        bool exiting = false;
        SBError error (driver.ParseArgs (argc, argv, stdout, exiting));
        if (error.Fail())
        {
            const char *error_cstr = error.GetCString ();
            if (error_cstr)
                ::fprintf (stderr, "error: %s\n", error_cstr);
        }
        else if (!exiting)
        {
            driver.MainLoop ();
        }
    }

    SBDebugger::Terminate();
    return 0;
}
