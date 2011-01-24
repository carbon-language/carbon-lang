//===-- CommandObjectLog.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectLog.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private-log.h"

#include "lldb/Interpreter/Args.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/FileSpec.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/Timer.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolVendor.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;


static LogChannelSP
GetLogChannelPluginForChannel (const char *channel)
{
    std::string log_channel_plugin_name(channel);
    log_channel_plugin_name += LogChannel::GetPluginSuffix();
    LogChannelSP log_channel_sp (LogChannel::FindPlugin (log_channel_plugin_name.c_str()));
    return log_channel_sp;
}


class CommandObjectLogEnable : public CommandObject
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    CommandObjectLogEnable(CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "log enable",
                       "Enable logging for a single log channel.",
                        NULL)
    {

        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentData channel_arg;
        CommandArgumentData category_arg;
        
        // Define the first (and only) variant of this arg.
        channel_arg.arg_type = eArgTypeLogChannel;
        channel_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (channel_arg);
        
        category_arg.arg_type = eArgTypeLogCategory;
        category_arg.arg_repetition = eArgRepeatPlus;

        arg2.push_back (category_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
    }

    virtual
    ~CommandObjectLogEnable()
    {
    }

    Options *
    GetOptions ()
    {
        return &m_options;
    }

    virtual bool
    Execute (Args& args,
             CommandReturnObject &result)
    {
        if (args.GetArgumentCount() < 1)
        {
            result.AppendErrorWithFormat("Usage: %s\n", m_cmd_syntax.c_str());
        }
        else
        {
            Log::Callbacks log_callbacks;

            std::string channel(args.GetArgumentAtIndex(0));
            args.Shift ();  // Shift off the channel
            StreamSP log_stream_sp;
            StreamFile *log_file_ptr = NULL;  // This will get put in the log_stream_sp, no need to free it.
            if (m_options.log_file.empty())
            {
                log_file_ptr = new StreamFile(m_interpreter.GetDebugger().GetOutputFileHandle());
                log_stream_sp.reset(log_file_ptr);
            }
            else
            {
                LogStreamMap::iterator pos = m_log_streams.find(m_options.log_file);
                if (pos == m_log_streams.end())
                {
                    log_file_ptr = new StreamFile (m_options.log_file.c_str(), "w");
                    log_stream_sp.reset (log_file_ptr);
                    m_log_streams[m_options.log_file] = log_stream_sp;
                }
                else
                    log_stream_sp = pos->second;
            }
            assert (log_stream_sp.get());
            
            // If we ended up making a StreamFile for log output, line buffer it.
            if (log_file_ptr != NULL)
                log_file_ptr->SetLineBuffered();
                
            uint32_t log_options = m_options.log_options;
            if (log_options == 0)
                log_options = LLDB_LOG_OPTION_PREPEND_THREAD_NAME | LLDB_LOG_OPTION_THREADSAFE;
            if (Log::GetLogChannelCallbacks (channel.c_str(), log_callbacks))
            {
                log_callbacks.enable (log_stream_sp, log_options, args, &result.GetErrorStream());
                result.SetStatus(eReturnStatusSuccessFinishNoResult);
            }
            else
            {
                LogChannelSP log_channel_sp (GetLogChannelPluginForChannel(channel.c_str()));
                if (log_channel_sp)
                {
                    if (log_channel_sp->Enable (log_stream_sp, log_options, &result.GetErrorStream(), args))
                    {
                        result.SetStatus (eReturnStatusSuccessFinishNoResult);
                    }
                    else
                    {
                        result.AppendErrorWithFormat("Invalid log channel '%s'.\n", channel.c_str());
                        result.SetStatus (eReturnStatusFailed);
                    }
                }
                else
                {
                    result.AppendErrorWithFormat("Invalid log channel '%s'.\n", channel.c_str());
                    result.SetStatus (eReturnStatusFailed);
                }
            }
        }
        return result.Succeeded();
    }


    class CommandOptions : public Options
    {
    public:

        CommandOptions () :
            Options (),
            log_file (),
            log_options (0)
        {
        }


        virtual
        ~CommandOptions ()
        {
        }

        virtual Error
        SetOptionValue (int option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;

            switch (short_option)
            {
            case 'f':  log_file = option_arg;                                 break;
            case 't':  log_options |= LLDB_LOG_OPTION_THREADSAFE;             break;
            case 'v':  log_options |= LLDB_LOG_OPTION_VERBOSE;                break;
            case 'g':  log_options |= LLDB_LOG_OPTION_DEBUG;                  break;
            case 's':  log_options |= LLDB_LOG_OPTION_PREPEND_SEQUENCE;       break;
            case 'T':  log_options |= LLDB_LOG_OPTION_PREPEND_TIMESTAMP;      break;
            case 'p':  log_options |= LLDB_LOG_OPTION_PREPEND_PROC_AND_THREAD;break;
            case 'n':  log_options |= LLDB_LOG_OPTION_PREPEND_THREAD_NAME;    break;
            default:
                error.SetErrorStringWithFormat ("Unrecognized option '%c'\n", short_option);
                break;
            }

            return error;
        }

        void
        ResetOptionValues ()
        {
            Options::ResetOptionValues();
            log_file.clear();
            log_options = 0;
        }

        const lldb::OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static lldb::OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        std::string log_file;
        uint32_t log_options;
    };

protected:
    typedef std::map<std::string, StreamSP> LogStreamMap;
    CommandOptions m_options;
    LogStreamMap m_log_streams;
};

lldb::OptionDefinition
CommandObjectLogEnable::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "file",       'f', required_argument, NULL, 0, eArgTypeFilename,   "Set the destination file to log to."},
{ LLDB_OPT_SET_1, false, "threadsafe", 't', no_argument,       NULL, 0, eArgTypeNone,        "Enable thread safe logging to avoid interweaved log lines." },
{ LLDB_OPT_SET_1, false, "verbose",    'v', no_argument,       NULL, 0, eArgTypeNone,       "Enable verbose logging." },
{ LLDB_OPT_SET_1, false, "debug",      'g', no_argument,       NULL, 0, eArgTypeNone,       "Enable debug logging." },
{ LLDB_OPT_SET_1, false, "sequence",   's', no_argument,       NULL, 0, eArgTypeNone,       "Prepend all log lines with an increasing integer sequence id." },
{ LLDB_OPT_SET_1, false, "timestamp",  'T', no_argument,       NULL, 0, eArgTypeNone,       "Prepend all log lines with a timestamp." },
{ LLDB_OPT_SET_1, false, "pid-tid",    'p', no_argument,       NULL, 0, eArgTypeNone,       "Prepend all log lines with the process and thread ID that generates the log line." },
{ LLDB_OPT_SET_1, false, "thread-name",'n', no_argument,       NULL, 0, eArgTypeNone,       "Prepend all log lines with the thread name for the thread that generates the log line." },
{ 0, false, NULL,                       0,  0,                 NULL, 0, eArgTypeNone,       NULL }
};

class CommandObjectLogDisable : public CommandObject
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    CommandObjectLogDisable(CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "log disable",
                       "Disable one or more log channel categories.",
                       NULL)
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentData channel_arg;
        CommandArgumentData category_arg;
        
        // Define the first (and only) variant of this arg.
        channel_arg.arg_type = eArgTypeLogChannel;
        channel_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (channel_arg);
        
        category_arg.arg_type = eArgTypeLogCategory;
        category_arg.arg_repetition = eArgRepeatPlus;

        arg2.push_back (category_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
    }

    virtual
    ~CommandObjectLogDisable()
    {
    }

    virtual bool
    Execute (Args& args,
             CommandReturnObject &result)
    {
        const size_t argc = args.GetArgumentCount();
        if (argc == 0)
        {
            result.AppendErrorWithFormat("Usage: %s\n", m_cmd_syntax.c_str());
        }
        else
        {
            Log::Callbacks log_callbacks;

            std::string channel(args.GetArgumentAtIndex(0));
            args.Shift ();  // Shift off the channel
            if (Log::GetLogChannelCallbacks (channel.c_str(), log_callbacks))
            {
                log_callbacks.disable (args, &result.GetErrorStream());
                result.SetStatus(eReturnStatusSuccessFinishNoResult);
            }
            else if (channel == "all")
            {
                Log::DisableAllLogChannels(&result.GetErrorStream());
            }
            else
            {
                LogChannelSP log_channel_sp (GetLogChannelPluginForChannel(channel.c_str()));
                if (log_channel_sp)
                {
                    log_channel_sp->Disable(args, &result.GetErrorStream());
                    result.SetStatus(eReturnStatusSuccessFinishNoResult);
                }
                else
                    result.AppendErrorWithFormat("Invalid log channel '%s'.\n", args.GetArgumentAtIndex(0));
            }
        }
        return result.Succeeded();
    }
};

class CommandObjectLogList : public CommandObject
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    CommandObjectLogList(CommandInterpreter &interpreter) :
        CommandObject (interpreter, 
                       "log list",
                       "List the log categories for one or more log channels.  If none specified, lists them all.",
                       NULL)
    {
        CommandArgumentEntry arg;
        CommandArgumentData channel_arg;
        
        // Define the first (and only) variant of this arg.
        channel_arg.arg_type = eArgTypeLogChannel;
        channel_arg.arg_repetition = eArgRepeatStar;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (channel_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    virtual
    ~CommandObjectLogList()
    {
    }

    virtual bool
    Execute (Args& args,
             CommandReturnObject &result)
    {
        const size_t argc = args.GetArgumentCount();
        if (argc == 0)
        {
            Log::ListAllLogChannels (&result.GetOutputStream());
            result.SetStatus(eReturnStatusSuccessFinishResult);
        }
        else
        {
            for (size_t i=0; i<argc; ++i)
            {
                Log::Callbacks log_callbacks;

                std::string channel(args.GetArgumentAtIndex(i));
                if (Log::GetLogChannelCallbacks (channel.c_str(), log_callbacks))
                {
                    log_callbacks.list_categories (&result.GetOutputStream());
                    result.SetStatus(eReturnStatusSuccessFinishResult);
                }
                else if (channel == "all")
                {
                    Log::ListAllLogChannels (&result.GetOutputStream());
                    result.SetStatus(eReturnStatusSuccessFinishResult);
                }
                else
                {
                    LogChannelSP log_channel_sp (GetLogChannelPluginForChannel(channel.c_str()));
                    if (log_channel_sp)
                    {
                        log_channel_sp->ListCategories(&result.GetOutputStream());
                        result.SetStatus(eReturnStatusSuccessFinishNoResult);
                    }
                    else
                        result.AppendErrorWithFormat("Invalid log channel '%s'.\n", args.GetArgumentAtIndex(0));
                }
            }
        }
        return result.Succeeded();
    }
};

class CommandObjectLogTimer : public CommandObject
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    CommandObjectLogTimer(CommandInterpreter &interpreter) :
        CommandObject (interpreter, 
                       "log timers",
                       "Enable, disable, dump, and reset LLDB internal performance timers.",
                       "log timers < enable <depth> | disable | dump | increment <bool> | reset >")
    {
    }

    virtual
    ~CommandObjectLogTimer()
    {
    }

    virtual bool
    Execute (Args& args,
             CommandReturnObject &result)
    {
        const size_t argc = args.GetArgumentCount();
        result.SetStatus(eReturnStatusFailed);

        if (argc == 1)
        {
            const char *sub_command = args.GetArgumentAtIndex(0);

            if (strcasecmp(sub_command, "enable") == 0)
            {
                Timer::SetDisplayDepth (UINT32_MAX);
                result.SetStatus(eReturnStatusSuccessFinishNoResult);
            }
            else if (strcasecmp(sub_command, "disable") == 0)
            {
                Timer::DumpCategoryTimes (&result.GetOutputStream());
                Timer::SetDisplayDepth (0);
                result.SetStatus(eReturnStatusSuccessFinishResult);
            }
            else if (strcasecmp(sub_command, "dump") == 0)
            {
                Timer::DumpCategoryTimes (&result.GetOutputStream());
                result.SetStatus(eReturnStatusSuccessFinishResult);
            }
            else if (strcasecmp(sub_command, "reset") == 0)
            {
                Timer::ResetCategoryTimes ();
                result.SetStatus(eReturnStatusSuccessFinishResult);
            }

        }
        else if (argc == 2)
        {
            const char *sub_command = args.GetArgumentAtIndex(0);

            if (strcasecmp(sub_command, "enable") == 0)
            {
                bool success;
                uint32_t depth = Args::StringToUInt32(args.GetArgumentAtIndex(1), 0, 0, &success);
                if (success)
                {
                    Timer::SetDisplayDepth (depth);
                    result.SetStatus(eReturnStatusSuccessFinishNoResult);
                }
                else
                    result.AppendError("Could not convert enable depth to an unsigned integer.");
            }
            if (strcasecmp(sub_command, "increment") == 0)
            {
                bool success;
                bool increment = Args::StringToBoolean(args.GetArgumentAtIndex(1), false, &success);
                if (success)
                {
                    Timer::SetQuiet (!increment);
                    result.SetStatus(eReturnStatusSuccessFinishNoResult);
                }
                else
                    result.AppendError("Could not convert increment value to boolean.");
            }
        }
        
        if (!result.Succeeded())
        {
            result.AppendError("Missing subcommand");
            result.AppendErrorWithFormat("Usage: %s\n", m_cmd_syntax.c_str());
        }
        return result.Succeeded();
    }
};

//----------------------------------------------------------------------
// CommandObjectLog constructor
//----------------------------------------------------------------------
CommandObjectLog::CommandObjectLog(CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "log",
                            "A set of commands for operating on logs.",
                            "log <command> [<command-options>]")
{
    LoadSubCommand ("enable",  CommandObjectSP (new CommandObjectLogEnable (interpreter)));
    LoadSubCommand ("disable", CommandObjectSP (new CommandObjectLogDisable (interpreter)));
    LoadSubCommand ("list",    CommandObjectSP (new CommandObjectLogList (interpreter)));
    LoadSubCommand ("timers",  CommandObjectSP (new CommandObjectLogTimer (interpreter)));
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
CommandObjectLog::~CommandObjectLog()
{
}




