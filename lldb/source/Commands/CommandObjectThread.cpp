//===-- CommandObjectThread.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectThread.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/State.h"
#include "lldb/Core/SourceManager.h"

#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanStepInstruction.h"
#include "lldb/Target/ThreadPlanStepOut.h"
#include "lldb/Target/ThreadPlanStepRange.h"
#include "lldb/Target/ThreadPlanStepInRange.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/LineEntry.h"

using namespace lldb;
using namespace lldb_private;


bool
lldb_private::DisplayThreadInfo
(
    CommandInterpreter &interpreter,
    Stream &strm,
    Thread *thread,
    bool only_threads_with_stop_reason,
    bool show_source
)
{
    if (thread)
    {
        if (only_threads_with_stop_reason)
        {
            if (thread->GetStopInfo() == NULL)
                return false;
        }

        strm.Indent();
        strm.Printf("%c ", thread->GetProcess().GetThreadList().GetSelectedThread().get() == thread ? '*' : ' ');

        // Show one frame with only the first showing source
        if (show_source)
        {
            bool already_shown = false;
            StackFrameSP frame_sp = thread->GetStackFrameAtIndex(0);
            SymbolContext frame_sc(frame_sp->GetSymbolContext (eSymbolContextLineEntry));
            if (interpreter.GetDebugger().UseExternalEditor() && frame_sc.line_entry.file && frame_sc.line_entry.line != 0)
            {
                already_shown = Host::OpenFileInExternalEditor (frame_sc.line_entry.file, frame_sc.line_entry.line);
            }
            
            DisplayFramesForExecutionContext (thread,
                                              interpreter,
                                              strm,
                                              0,    // Start at first frame
                                              1,    // Number of frames to show
                                              false,// Don't show the frame info since we already displayed most of it above...
                                              !already_shown,    // Show source for the first frame
                                              3,    // lines of source context before
                                              3);   // lines of source context after
        }
        else
        {
            thread->DumpInfo (strm,
                              true, // Dump the stop reason?
                              true, // Dump the thread name?
                              true, // Dump the queue name?
                              0);   // Display context info for stack frame zero

            strm.EOL();
        }

        return true;
    }
    return false;
}

size_t
lldb_private::DisplayThreadsInfo
(
    CommandInterpreter &interpreter,
    ExecutionContext *exe_ctx,
    CommandReturnObject &result,
    bool only_threads_with_stop_reason,
    bool show_source
)
{
    StreamString strm;

    size_t num_thread_infos_dumped = 0;

    if (!exe_ctx->process)
        return 0;

    const size_t num_threads = exe_ctx->process->GetThreadList().GetSize();
    if (num_threads > 0)
    {

        for (uint32_t i = 0; i < num_threads; i++)
        {
            Thread *thread = exe_ctx->process->GetThreadList().GetThreadAtIndex(i).get();
            if (thread)
            {
                if (DisplayThreadInfo (interpreter,
                                       strm,
                                       thread,
                                       only_threads_with_stop_reason,
                                       show_source))
                    ++num_thread_infos_dumped;
            }
        }
    }

    if (num_thread_infos_dumped > 0)
    {
        if (num_thread_infos_dumped < num_threads)
            result.GetOutputStream().Printf("%u of %u threads stopped with reasons:\n", num_thread_infos_dumped, num_threads);

        result.GetOutputStream().GetString().append(strm.GetString());
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
    }
    return num_thread_infos_dumped;
}


size_t
lldb_private::DisplayFramesForExecutionContext
(
    Thread *thread,
    CommandInterpreter &interpreter,
    Stream& strm,
    uint32_t first_frame,
    uint32_t num_frames,
    bool show_frame_info,
    uint32_t num_frames_with_source,
    uint32_t source_lines_before,
    uint32_t source_lines_after
)
{
    if (thread == NULL)
        return 0;

    size_t num_frames_displayed = 0;

    if (num_frames == 0)
        return 0;
    
    thread->DumpInfo (strm,
                      true,     // Dump the stop reason?
                      true,     // Dump the thread name?
                      true,     // Dump the queue name?
                      num_frames > 1 ? UINT32_MAX : first_frame);  // Dump info for the first stack frame if we are showing only on frame
    strm.EOL();
    strm.IndentMore();

    StackFrameSP frame_sp;
    uint32_t frame_idx = 0;
    uint32_t last_frame;
    
    // Don't let the last frame wrap around...
    if (num_frames == UINT32_MAX)
        last_frame = UINT32_MAX;
    else
        last_frame = first_frame + num_frames;
    
    for (frame_idx = first_frame; frame_idx < last_frame; ++frame_idx)
    {
        frame_sp = thread->GetStackFrameAtIndex (frame_idx);
        if (frame_sp.get() == NULL)
            break;

        if (DisplayFrameForExecutionContext (thread,
                                             frame_sp.get(),
                                             interpreter,
                                             strm,
                                             show_frame_info,
                                             num_frames_with_source > first_frame - frame_idx,
                                             source_lines_before,
                                             source_lines_after) == false)
            break;

        ++num_frames_displayed;
    }

    strm.IndentLess();
    return num_frames_displayed;
}

bool
lldb_private::DisplayFrameForExecutionContext
(
    Thread *thread,
    StackFrame *frame,
    CommandInterpreter &interpreter,
    Stream& strm,
    bool show_frame_info,
    bool show_source,
    uint32_t source_lines_before,
    uint32_t source_lines_after
)
{
    // thread and frame must be filled in prior to calling this function
    if (thread && frame)
    {
        if (show_frame_info)
        {
            strm.Indent();
            frame->Dump (&strm, true);
            strm.EOL();
        }

        SymbolContext sc (frame->GetSymbolContext(eSymbolContextCompUnit | eSymbolContextLineEntry));

        if (show_source && sc.comp_unit && sc.line_entry.IsValid())
        {
            interpreter.GetDebugger().GetSourceManager().DisplaySourceLinesWithLineNumbers (
                    sc.line_entry.file,
                    sc.line_entry.line,
                    3,
                    3,
                    "->",
                    &strm);

        }
        return true;
    }
    return false;
}


//-------------------------------------------------------------------------
// CommandObjectThreadBacktrace
//-------------------------------------------------------------------------

class CommandObjectThreadBacktrace : public CommandObject
{
public:

    class CommandOptions : public Options
    {
    public:

        CommandOptions () :
            Options()
        {
            // Keep default values of all options in one place: ResetOptionValues ()
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
            char short_option = (char) m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 'c':
                {
                    bool success;
                    int32_t input_count =  Args::StringToSInt32 (option_arg, -1, 0, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("Invalid integer value for option '%c'.\n", short_option);
                    if (input_count < -1)
                        m_count = UINT32_MAX;
                    else
                        m_count = input_count;
                }
                break;
                case 's':
                {
                    bool success;
                    m_start =  Args::StringToUInt32 (option_arg, 0, 0, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("Invalid integer value for option '%c'.\n", short_option);
                }
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
            m_count = -1;
            m_start = 0;
        }

        const lldb::OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static lldb::OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
        uint32_t m_count;
        uint32_t m_start;
    };

    CommandObjectThreadBacktrace () :
        CommandObject ("thread backtrace",
                       "Shows the stack for one or more threads.  If no threads are specified, shows the currently selected thread.  \nUse the thread-index \"all\" to see all threads.",
                       "thread backtrace [<thread-index>] ...",
                       eFlagProcessMustBeLaunched | eFlagProcessMustBePaused),
        m_options()
    {
    }

    ~CommandObjectThreadBacktrace()
    {
    }

    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }

    virtual bool
    Execute
    (
        CommandInterpreter &interpreter,
        Args& command,
        CommandReturnObject &result
    )
    {

        bool show_frame_info = true;
        uint32_t num_frames_with_source = 0; // Don't show any frames with source when backtracing
        
        result.SetStatus (eReturnStatusSuccessFinishResult);
        
        if (command.GetArgumentCount() == 0)
        {
            ExecutionContext exe_ctx(interpreter.GetDebugger().GetExecutionContext());
            if (exe_ctx.thread)
            {
                if (DisplayFramesForExecutionContext (exe_ctx.thread,
                                                      interpreter,
                                                      result.GetOutputStream(),
                                                      m_options.m_start,
                                                      m_options.m_count,
                                                      show_frame_info,
                                                      num_frames_with_source,
                                                      3,
                                                      3))
                {
                    result.SetStatus (eReturnStatusSuccessFinishResult);
                }
            }
            else
            {
                result.AppendError ("invalid thread");
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else if (command.GetArgumentCount() == 1 && ::strcmp (command.GetArgumentAtIndex(0), "all") == 0)
        {
            Process *process = interpreter.GetDebugger().GetExecutionContext().process;
            uint32_t num_threads = process->GetThreadList().GetSize();
            for (uint32_t i = 0; i < num_threads; i++)
            {
                ThreadSP thread_sp = process->GetThreadList().GetThreadAtIndex(i);
                if (!DisplayFramesForExecutionContext (thread_sp.get(),
                                                      interpreter,
                                                      result.GetOutputStream(),
                                                      m_options.m_start,
                                                      m_options.m_count,
                                                      show_frame_info,
                                                      num_frames_with_source,
                                                      3,
                                                      3))
                {
                    result.AppendErrorWithFormat ("error displaying backtrace for thread: \"%d\"\n", i);
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                if (i < num_threads - 1)
                    result.AppendMessage("");
            }
        }
        else
        {
            uint32_t num_args = command.GetArgumentCount();
            Process *process = interpreter.GetDebugger().GetExecutionContext().process;
            std::vector<ThreadSP> thread_sps;

            for (uint32_t i = 0; i < num_args; i++)
            {
                bool success;
                
                uint32_t thread_idx = Args::StringToUInt32(command.GetArgumentAtIndex(i), 0, 0, &success);
                if (!success)
                {
                    result.AppendErrorWithFormat ("invalid thread specification: \"%s\"\n", command.GetArgumentAtIndex(i));
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                
                thread_sps.push_back(process->GetThreadList().FindThreadByIndexID(thread_idx));
                
                if (!thread_sps[i])
                {
                    result.AppendErrorWithFormat ("no thread with index: \"%s\"\n", command.GetArgumentAtIndex(i));
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                
            }
            
            for (uint32_t i = 0; i < num_args; i++)
            {
                if (!DisplayFramesForExecutionContext (thread_sps[i].get(),
                                                      interpreter,
                                                      result.GetOutputStream(),
                                                      m_options.m_start,
                                                      m_options.m_count,
                                                      show_frame_info,
                                                      num_frames_with_source,
                                                      3,
                                                      3))
                {
                    result.AppendErrorWithFormat ("error displaying backtrace for thread: \"%s\"\n", command.GetArgumentAtIndex(i));
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                
                if (i < num_args - 1)
                    result.AppendMessage("");
            }
        }
        return result.Succeeded();
    }
protected:
    CommandOptions m_options;
};

lldb::OptionDefinition
CommandObjectThreadBacktrace::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "count", 'c', required_argument, NULL,               0, "<count>", "How many frames to display (-1 for all)"},
{ LLDB_OPT_SET_1, false, "start",       's', required_argument, NULL, 0, "<start>",       "Where to start the backtrace"},
{ 0, false, NULL, 0, 0, NULL, 0, NULL, NULL }
};

enum StepScope
{
    eStepScopeSource,
    eStepScopeInstruction
};

class CommandObjectThreadStepWithTypeAndScope : public CommandObject
{
public:

    class CommandOptions : public Options
    {
    public:

        CommandOptions () :
            Options()
        {
            // Keep default values of all options in one place: ResetOptionValues ()
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
            char short_option = (char) m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 'a':
                {
                    bool success;
                    m_avoid_no_debug =  Args::StringToBoolean (option_arg, true, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("Invalid boolean value for option '%c'.\n", short_option);
                }
                break;
                case 'm':
                {
                    bool found_one = false;
                    OptionEnumValueElement *enum_values = g_option_table[option_idx].enum_values; 
                    m_run_mode = (lldb::RunMode) Args::StringToOptionEnum(option_arg, enum_values, eOnlyDuringStepping, &found_one);
                    if (!found_one)
                        error.SetErrorStringWithFormat("Invalid enumeration value for option '%c'.\n", short_option);
                }
                break;
                case 'r':
                {
                    m_avoid_regexp.clear();
                    m_avoid_regexp.assign(option_arg);
                }
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
            m_avoid_no_debug = true;
            m_run_mode = eOnlyDuringStepping;
            m_avoid_regexp.clear();
        }

        const lldb::OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static lldb::OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
        bool m_avoid_no_debug;
        RunMode m_run_mode;
        std::string m_avoid_regexp;
    };

    CommandObjectThreadStepWithTypeAndScope (const char *name,
                         const char *help,
                         const char *syntax,
                         uint32_t flags,
                         StepType step_type,
                         StepScope step_scope) :
        CommandObject (name, help, syntax, flags),
        m_step_type (step_type),
        m_step_scope (step_scope),
        m_options ()
    {
    }

    virtual
    ~CommandObjectThreadStepWithTypeAndScope ()
    {
    }

    virtual
    Options *
    GetOptions ()
    {
        return &m_options;
    }

    virtual bool
    Execute 
    (
        CommandInterpreter &interpreter,
        Args& command,
        CommandReturnObject &result
    )
    {
        Process *process = interpreter.GetDebugger().GetExecutionContext().process;
        bool synchronous_execution = interpreter.GetSynchronous();

        if (process == NULL)
        {
            result.AppendError ("need a valid process to step");
            result.SetStatus (eReturnStatusFailed);

        }
        else
        {
            const uint32_t num_threads = process->GetThreadList().GetSize();
            Thread *thread = NULL;

            if (command.GetArgumentCount() == 0)
            {
                thread = process->GetThreadList().GetSelectedThread().get();
                if (thread == NULL)
                {
                    result.AppendError ("no selected thread in process");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            else
            {
                const char *thread_idx_cstr = command.GetArgumentAtIndex(0);
                uint32_t step_thread_idx = Args::StringToUInt32 (thread_idx_cstr, LLDB_INVALID_INDEX32);
                if (step_thread_idx == LLDB_INVALID_INDEX32)
                {
                    result.AppendErrorWithFormat ("Invalid thread index '%s'.\n", thread_idx_cstr);
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                thread = process->GetThreadList().FindThreadByIndexID(step_thread_idx).get();
                if (thread == NULL)
                {
                    result.AppendErrorWithFormat ("Thread index %u is out of range (valid values are 0 - %u).\n", 
                                                  step_thread_idx, 0, num_threads);
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }

            const bool abort_other_plans = false;
            const lldb::RunMode stop_other_threads = m_options.m_run_mode;
            
            // This is a bit unfortunate, but not all the commands in this command object support
            // only while stepping, so I use the bool for them.
            bool bool_stop_other_threads;
            if (m_options.m_run_mode == eAllThreads)
                bool_stop_other_threads = false;
            else
                bool_stop_other_threads = true;

            if (m_step_type == eStepTypeInto)
            {
                StackFrame *frame = thread->GetStackFrameAtIndex(0).get();
                ThreadPlan *new_plan;

                if (frame->HasDebugInformation ())
                {
                    new_plan = thread->QueueThreadPlanForStepRange (abort_other_plans, m_step_type, 
                                                                    frame->GetSymbolContext(eSymbolContextEverything).line_entry.range, 
                                                                    frame->GetSymbolContext(eSymbolContextEverything), 
                                                                    stop_other_threads,
                                                                    m_options.m_avoid_no_debug);
                    if (new_plan && !m_options.m_avoid_regexp.empty())
                    {
                        ThreadPlanStepInRange *step_in_range_plan = static_cast<ThreadPlanStepInRange *> (new_plan);
                        step_in_range_plan->SetAvoidRegexp(m_options.m_avoid_regexp.c_str());
                    }
                }
                else
                    new_plan = thread->QueueThreadPlanForStepSingleInstruction (false, abort_other_plans, bool_stop_other_threads);

                process->GetThreadList().SetSelectedThreadByID (thread->GetID());
                process->Resume ();
            }
            else if (m_step_type == eStepTypeOver)
            {
                StackFrame *frame = thread->GetStackFrameAtIndex(0).get();
                ThreadPlan *new_plan;

                if (frame->HasDebugInformation())
                    new_plan = thread->QueueThreadPlanForStepRange (abort_other_plans, 
                                                                    m_step_type, 
                                                                    frame->GetSymbolContext(eSymbolContextEverything).line_entry.range, 
                                                                    frame->GetSymbolContext(eSymbolContextEverything), 
                                                                    stop_other_threads,
                                                                    false);
                else
                    new_plan = thread->QueueThreadPlanForStepSingleInstruction (true, 
                                                                                abort_other_plans, 
                                                                                bool_stop_other_threads);

                // FIXME: This will keep the step plan on the thread stack when we hit a breakpoint while stepping over.
                // Maybe there should be a parameter to control this.
                new_plan->SetOkayToDiscard(false);

                process->GetThreadList().SetSelectedThreadByID (thread->GetID());
                process->Resume ();
            }
            else if (m_step_type == eStepTypeTrace)
            {
                thread->QueueThreadPlanForStepSingleInstruction (false, abort_other_plans, bool_stop_other_threads);
                process->GetThreadList().SetSelectedThreadByID (thread->GetID());
                process->Resume ();
            }
            else if (m_step_type == eStepTypeTraceOver)
            {
                thread->QueueThreadPlanForStepSingleInstruction (true, abort_other_plans, bool_stop_other_threads);
                process->GetThreadList().SetSelectedThreadByID (thread->GetID());
                process->Resume ();
            }
            else if (m_step_type == eStepTypeOut)
            {
                ThreadPlan *new_plan;

                new_plan = thread->QueueThreadPlanForStepOut (abort_other_plans, NULL, false, bool_stop_other_threads, eVoteYes, eVoteNoOpinion);
                // FIXME: This will keep the step plan on the thread stack when we hit a breakpoint while stepping over.
                // Maybe there should be a parameter to control this.
                new_plan->SetOkayToDiscard(false);

                process->GetThreadList().SetSelectedThreadByID (thread->GetID());
                process->Resume ();
            }
            else
            {
                result.AppendError ("step type is not supported");
                result.SetStatus (eReturnStatusFailed);
            }
            if (synchronous_execution)
            {
                StateType state = process->WaitForProcessToStop (NULL);
                
                //EventSP event_sp;
                //StateType state = process->WaitForStateChangedEvents (NULL, event_sp);
                //while (! StateIsStoppedState (state))
                //  {
                //    state = process->WaitForStateChangedEvents (NULL, event_sp);
                //  }
                process->GetThreadList().SetSelectedThreadByID (thread->GetID());
                result.SetDidChangeProcessState (true);
                result.AppendMessageWithFormat ("Process %i %s\n", process->GetID(), StateAsCString (state));
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
            }
        }
        return result.Succeeded();
    }

protected:
    StepType m_step_type;
    StepScope m_step_scope;
    CommandOptions m_options;
};

static lldb::OptionEnumValueElement
g_tri_running_mode[] =
{
{ eOnlyThisThread,     "thisThread",    "Run only this thread"},
{ eAllThreads,         "allThreads",    "Run all threads"},
{ eOnlyDuringStepping, "whileStepping", "Run only this thread while stepping"},
{ 0, NULL, NULL }
};

static lldb::OptionEnumValueElement
g_duo_running_mode[] =
{
{ eOnlyThisThread,     "thisThread",    "Run only this thread"},
{ eAllThreads,         "allThreads",    "Run all threads"},
{ 0, NULL, NULL }
};

lldb::OptionDefinition
CommandObjectThreadStepWithTypeAndScope::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "avoid_no_debug", 'a', required_argument, NULL,               0, "<avoid_no_debug>", "Should step-in step over functions with no debug information"},
{ LLDB_OPT_SET_1, false, "run_mode",       'm', required_argument, g_tri_running_mode, 0, "<run_mode>",       "Determine how to run other threads while stepping this one"},
{ LLDB_OPT_SET_1, false, "regexp_to_avoid",'r', required_argument, NULL, 0, "<avoid_regexp>",       "Should step-in step over functions matching this regexp"},
{ 0, false, NULL, 0, 0, NULL, 0, NULL, NULL }
};


//-------------------------------------------------------------------------
// CommandObjectThreadContinue
//-------------------------------------------------------------------------

class CommandObjectThreadContinue : public CommandObject
{
public:

    CommandObjectThreadContinue () :
        CommandObject ("thread continue",
                       "Continues execution of one or more threads in an active process.",
                       "thread continue <thread-index> [<thread-index> ...]",
                       eFlagProcessMustBeLaunched | eFlagProcessMustBePaused)
    {
    }


    virtual
    ~CommandObjectThreadContinue ()
    {
    }

    virtual bool
    Execute
    (
        CommandInterpreter &interpreter,
        Args& command,
        CommandReturnObject &result
    )
    {
        bool synchronous_execution = interpreter.GetSynchronous ();

        if (!interpreter.GetDebugger().GetSelectedTarget().get())
        {
            result.AppendError ("invalid target, set executable file using 'file' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        Process *process = interpreter.GetDebugger().GetExecutionContext().process;
        if (process == NULL)
        {
            result.AppendError ("no process exists. Cannot continue");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        StateType state = process->GetState();
        if ((state == eStateCrashed) || (state == eStateStopped) || (state == eStateSuspended))
        {
            const uint32_t num_threads = process->GetThreadList().GetSize();
            uint32_t idx;
            const size_t argc = command.GetArgumentCount();
            if (argc > 0)
            {
                std::vector<uint32_t> resume_thread_indexes;
                for (uint32_t i=0; i<argc; ++i)
                {
                    idx = Args::StringToUInt32 (command.GetArgumentAtIndex(0), LLDB_INVALID_INDEX32);
                    if (idx < num_threads)
                        resume_thread_indexes.push_back(idx);
                    else
                        result.AppendWarningWithFormat("Thread index %u out of range.\n", idx);
                }

                if (resume_thread_indexes.empty())
                {
                    result.AppendError ("no valid thread indexes were specified");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                else
                {
                    result.AppendMessage ("Resuming thread ");
                    for (idx=0; idx<num_threads; ++idx)
                    {
                        Thread *thread = process->GetThreadList().GetThreadAtIndex(idx).get();
                        if (find(resume_thread_indexes.begin(), resume_thread_indexes.end(), idx) != resume_thread_indexes.end())
                        {
                            result.AppendMessageWithFormat ("%u ", idx);
                            thread->SetResumeState (eStateRunning);
                        }
                        else
                        {
                            thread->SetResumeState (eStateSuspended);
                        }
                    }
                    result.AppendMessageWithFormat ("in process %i\n", process->GetID());
                }
            }
            else
            {
                Thread *current_thread = process->GetThreadList().GetSelectedThread().get();
                if (current_thread == NULL)
                {
                    result.AppendError ("the process doesn't have a current thread");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                // Set the actions that the threads should each take when resuming
                for (idx=0; idx<num_threads; ++idx)
                {
                    Thread *thread = process->GetThreadList().GetThreadAtIndex(idx).get();
                    if (thread == current_thread)
                    {
                        result.AppendMessageWithFormat ("Resuming thread 0x%4.4x in process %i\n", thread->GetID(), process->GetID());
                        thread->SetResumeState (eStateRunning);
                    }
                    else
                    {
                        thread->SetResumeState (eStateSuspended);
                    }
                }
            }

            Error error (process->Resume());
            if (error.Success())
            {
                result.AppendMessageWithFormat ("Resuming process %i\n", process->GetID());
                if (synchronous_execution)
                {
                    state = process->WaitForProcessToStop (NULL);

                    result.SetDidChangeProcessState (true);
                    result.AppendMessageWithFormat ("Process %i %s\n", process->GetID(), StateAsCString (state));
                    result.SetStatus (eReturnStatusSuccessFinishNoResult);
                }
                else
                {
                    result.SetStatus (eReturnStatusSuccessContinuingNoResult);
                }
            }
            else
            {
                result.AppendErrorWithFormat("Failed to resume process: %s\n", error.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else
        {
            result.AppendErrorWithFormat ("Process cannot be continued from its current state (%s).\n",
                                          StateAsCString(state));
            result.SetStatus (eReturnStatusFailed);
        }

        return result.Succeeded();
    }

};

//-------------------------------------------------------------------------
// CommandObjectThreadUntil
//-------------------------------------------------------------------------

class CommandObjectThreadUntil : public CommandObject
{
public:

    class CommandOptions : public Options
    {
    public:
        uint32_t m_thread_idx;
        uint32_t m_frame_idx;

        CommandOptions () :
            Options(),
            m_thread_idx(LLDB_INVALID_THREAD_ID),
            m_frame_idx(LLDB_INVALID_FRAME_ID)
        {
            // Keep default values of all options in one place: ResetOptionValues ()
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
            char short_option = (char) m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 't':
                {
                    m_thread_idx = Args::StringToUInt32 (option_arg, LLDB_INVALID_INDEX32);
                    if (m_thread_idx == LLDB_INVALID_INDEX32)
                    {
                        error.SetErrorStringWithFormat ("Invalid thread index '%s'.\n", option_arg);
                    }
                }
                break;
                case 'f':
                {
                    m_frame_idx = Args::StringToUInt32 (option_arg, LLDB_INVALID_FRAME_ID);
                    if (m_frame_idx == LLDB_INVALID_FRAME_ID)
                    {
                        error.SetErrorStringWithFormat ("Invalid frame index '%s'.\n", option_arg);
                    }
                }
                break;
                case 'm':
                {
                    bool found_one = false;
                    OptionEnumValueElement *enum_values = g_option_table[option_idx].enum_values; 
                    lldb::RunMode run_mode = (lldb::RunMode) Args::StringToOptionEnum(option_arg, enum_values, eOnlyDuringStepping, &found_one);

                    if (!found_one)
                        error.SetErrorStringWithFormat("Invalid enumeration value for option '%c'.\n", short_option);
                    else if (run_mode == eAllThreads)
                        m_stop_others = false;
                    else
                        m_stop_others = true;
        
                }
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
            m_thread_idx = LLDB_INVALID_THREAD_ID;
            m_frame_idx = 0;
            m_stop_others = false;
        }

        const lldb::OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        uint32_t m_step_thread_idx;
        bool m_stop_others;

        // Options table: Required for subclasses of Options.

        static lldb::OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
    };

    CommandObjectThreadUntil () :
        CommandObject ("thread until",
                       "Runs the current or specified thread until it reaches a given line number or leaves the current function.",
                       "thread until [<cmd-options>] <line-number>",
                       eFlagProcessMustBeLaunched | eFlagProcessMustBePaused),
        m_options ()
    {
    }


    virtual
    ~CommandObjectThreadUntil ()
    {
    }

    virtual
    Options *
    GetOptions ()
    {
        return &m_options;
    }

    virtual bool
    Execute 
    (
        CommandInterpreter &interpreter,
        Args& command,
        CommandReturnObject &result
    )
    {
        bool synchronous_execution = interpreter.GetSynchronous ();

        if (!interpreter.GetDebugger().GetSelectedTarget().get())
        {
            result.AppendError ("invalid target, set executable file using 'file' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        Process *process = interpreter.GetDebugger().GetExecutionContext().process;
        if (process == NULL)
        {
            result.AppendError ("need a valid process to step");
            result.SetStatus (eReturnStatusFailed);

        }
        else
        {
            Thread *thread = NULL;
            uint32_t line_number;

            if (command.GetArgumentCount() != 1)
            {
                result.AppendErrorWithFormat ("No line number provided:\n%s", GetSyntax());
                result.SetStatus (eReturnStatusFailed);
                return false;
            }

            line_number = Args::StringToUInt32 (command.GetArgumentAtIndex(0), UINT32_MAX);
            if (line_number == UINT32_MAX)
            {
                result.AppendErrorWithFormat ("Invalid line number: '%s'.\n", command.GetArgumentAtIndex(0));
                result.SetStatus (eReturnStatusFailed);
                return false;
            }

            if (m_options.m_thread_idx == LLDB_INVALID_THREAD_ID)
            {
                thread = process->GetThreadList().GetSelectedThread().get();
            }
            else
            {
                thread = process->GetThreadList().GetThreadAtIndex(m_options.m_thread_idx).get();
            }

            if (thread == NULL)
            {
                const uint32_t num_threads = process->GetThreadList().GetSize();
                result.AppendErrorWithFormat ("Thread index %u is out of range (valid values are 0 - %u).\n", m_options.m_thread_idx, 0, num_threads);
                result.SetStatus (eReturnStatusFailed);
                return false;
            }

            const bool abort_other_plans = true;

            StackFrame *frame = thread->GetStackFrameAtIndex(m_options.m_frame_idx).get();
            if (frame == NULL)
            {

                result.AppendErrorWithFormat ("Frame index %u is out of range for thread %u.\n", m_options.m_frame_idx, m_options.m_thread_idx);
                result.SetStatus (eReturnStatusFailed);
                return false;
            }

            ThreadPlan *new_plan;

            if (frame->HasDebugInformation ())
            {
                // Finally we got here...  Translate the given line number to a bunch of addresses:
                SymbolContext sc(frame->GetSymbolContext (eSymbolContextCompUnit));
                LineTable *line_table = NULL;
                if (sc.comp_unit)
                    line_table = sc.comp_unit->GetLineTable();

                if (line_table == NULL)
                {
                    result.AppendErrorWithFormat ("Failed to resolve the line table for frame %u of thread index %u.\n",
                                                 m_options.m_frame_idx, m_options.m_thread_idx);
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }

                LineEntry function_start;
                uint32_t index_ptr = 0, end_ptr;
                std::vector<addr_t> address_list;

                // Find the beginning & end index of the
                AddressRange fun_addr_range = sc.function->GetAddressRange();
                Address fun_start_addr = fun_addr_range.GetBaseAddress();
                line_table->FindLineEntryByAddress (fun_start_addr, function_start, &index_ptr);

                Address fun_end_addr(fun_start_addr.GetSection(), fun_start_addr.GetOffset() + fun_addr_range.GetByteSize());
                line_table->FindLineEntryByAddress (fun_end_addr, function_start, &end_ptr);

                while (index_ptr <= end_ptr)
                {
                    LineEntry line_entry;
                    index_ptr = sc.comp_unit->FindLineEntry(index_ptr, line_number, sc.comp_unit, &line_entry);
                    if (index_ptr == UINT32_MAX)
                        break;

                    addr_t address = line_entry.range.GetBaseAddress().GetLoadAddress(process);
                    if (address != LLDB_INVALID_ADDRESS)
                        address_list.push_back (address);
                    index_ptr++;
                }

                new_plan = thread->QueueThreadPlanForStepUntil (abort_other_plans, &address_list.front(), address_list.size(), m_options.m_stop_others);
                new_plan->SetOkayToDiscard(false);
            }
            else
            {
                result.AppendErrorWithFormat ("Frame index %u of thread %u has no debug information.\n", m_options.m_frame_idx, m_options.m_thread_idx);
                result.SetStatus (eReturnStatusFailed);
                return false;

            }

            process->GetThreadList().SetSelectedThreadByID (m_options.m_thread_idx);
            Error error (process->Resume ());
            if (error.Success())
            {
                result.AppendMessageWithFormat ("Resuming process %i\n", process->GetID());
                if (synchronous_execution)
                {
                    StateType state = process->WaitForProcessToStop (NULL);

                    result.SetDidChangeProcessState (true);
                    result.AppendMessageWithFormat ("Process %i %s\n", process->GetID(), StateAsCString (state));
                    result.SetStatus (eReturnStatusSuccessFinishNoResult);
                }
                else
                {
                    result.SetStatus (eReturnStatusSuccessContinuingNoResult);
                }
            }
            else
            {
                result.AppendErrorWithFormat("Failed to resume process: %s.\n", error.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }

        }
        return result.Succeeded();
    }
protected:
    CommandOptions m_options;

};

lldb::OptionDefinition
CommandObjectThreadUntil::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "frame",   'f', required_argument, NULL,               0, "<frame>",   "Frame index for until operation - defaults to 0"},
{ LLDB_OPT_SET_1, false, "thread",  't', required_argument, NULL,               0, "<thread>",  "Thread index for the thread for until operation"},
{ LLDB_OPT_SET_1, false, "run_mode",'m', required_argument, g_duo_running_mode, 0, "<run_mode>","Determine how to run other threads while stepping this one"},
{ 0, false, NULL, 0, 0, NULL, 0, NULL, NULL }
};


//-------------------------------------------------------------------------
// CommandObjectThreadSelect
//-------------------------------------------------------------------------

class CommandObjectThreadSelect : public CommandObject
{
public:

    CommandObjectThreadSelect () :
        CommandObject ("thread select",
                         "Selects a threads as the currently active thread.",
                         "thread select <thread-index>",
                         eFlagProcessMustBeLaunched | eFlagProcessMustBePaused)
    {
    }


    virtual
    ~CommandObjectThreadSelect ()
    {
    }

    virtual bool
    Execute 
    (
        CommandInterpreter &interpreter,
        Args& command,
        CommandReturnObject &result
    )
    {
        Process *process = interpreter.GetDebugger().GetExecutionContext().process;
        if (process == NULL)
        {
            result.AppendError ("no process");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else if (command.GetArgumentCount() != 1)
        {
            result.AppendErrorWithFormat("'%s' takes exactly one thread index argument:\nUsage: \n", m_cmd_name.c_str(), m_cmd_syntax.c_str());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        uint32_t index_id = Args::StringToUInt32(command.GetArgumentAtIndex(0), 0, 0);

        Thread *new_thread = process->GetThreadList().FindThreadByIndexID(index_id).get();
        if (new_thread == NULL)
        {
            result.AppendErrorWithFormat ("Invalid thread #%s.\n", command.GetArgumentAtIndex(0));
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        process->GetThreadList().SetSelectedThreadByID(new_thread->GetID());
        
        DisplayThreadInfo (interpreter,
                           result.GetOutputStream(),
                           new_thread,
                           false,
                           true);

        return result.Succeeded();
    }

};


//-------------------------------------------------------------------------
// CommandObjectThreadList
//-------------------------------------------------------------------------

class CommandObjectThreadList : public CommandObject
{
public:


    CommandObjectThreadList ():
        CommandObject ("thread list",
                       "Shows a summary of all current threads in a process.",
                       "thread list",
                       eFlagProcessMustBeLaunched | eFlagProcessMustBePaused)
    {
    }

    ~CommandObjectThreadList()
    {
    }

    bool
    Execute
    (
        CommandInterpreter &interpreter,
        Args& command,
        CommandReturnObject &result
    )
    {
        StreamString &strm = result.GetOutputStream();
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        ExecutionContext exe_ctx(interpreter.GetDebugger().GetExecutionContext());
        if (exe_ctx.process)
        {
            const StateType state = exe_ctx.process->GetState();

            if (StateIsStoppedState(state))
            {
                if (state == eStateExited)
                {
                    int exit_status = exe_ctx.process->GetExitStatus();
                    const char *exit_description = exe_ctx.process->GetExitDescription();
                    strm.Printf ("Process %d exited with status = %i (0x%8.8x) %s\n",
                                          exe_ctx.process->GetID(),
                                          exit_status,
                                          exit_status,
                                          exit_description ? exit_description : "");
                }
                else
                {
                    strm.Printf ("Process %d state is %s\n", exe_ctx.process->GetID(), StateAsCString (state));
                    if (exe_ctx.thread == NULL)
                        exe_ctx.thread = exe_ctx.process->GetThreadList().GetThreadAtIndex(0).get();
                    if (exe_ctx.thread != NULL)
                    {
                        DisplayThreadsInfo (interpreter, &exe_ctx, result, false, false);
                    }
                    else
                    {
                        result.AppendError ("no valid thread found in current process");
                        result.SetStatus (eReturnStatusFailed);
                    }
                }
            }
            else
            {
                result.AppendError ("process is currently running");
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else
        {
            result.AppendError ("no current location or status available");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectMultiwordThread
//-------------------------------------------------------------------------

CommandObjectMultiwordThread::CommandObjectMultiwordThread (CommandInterpreter &interpreter) :
    CommandObjectMultiword ("thread",
                            "A set of commands for operating on one or more thread within a running process.",
                            "thread <subcommand> [<subcommand-options>]")
{
    LoadSubCommand (interpreter, "backtrace",  CommandObjectSP (new CommandObjectThreadBacktrace ()));
    LoadSubCommand (interpreter, "continue",   CommandObjectSP (new CommandObjectThreadContinue ()));
    LoadSubCommand (interpreter, "list",       CommandObjectSP (new CommandObjectThreadList ()));
    LoadSubCommand (interpreter, "select",     CommandObjectSP (new CommandObjectThreadSelect ()));
    LoadSubCommand (interpreter, "until",      CommandObjectSP (new CommandObjectThreadUntil ()));
    LoadSubCommand (interpreter, "step-in",    CommandObjectSP (new CommandObjectThreadStepWithTypeAndScope (
                                                    "thread step-in",
                                                     "Source level single step in in specified thread (current thread, if none specified).",
                                                     "thread step-in [<thread-id>]",
                                                     eFlagProcessMustBeLaunched | eFlagProcessMustBePaused,
                                                     eStepTypeInto,
                                                     eStepScopeSource)));
    
    LoadSubCommand (interpreter, "step-out",    CommandObjectSP (new CommandObjectThreadStepWithTypeAndScope ("thread step-out",
                                                                                      "Source level single step out in specified thread (current thread, if none specified).",
                                                                                      "thread step-out [<thread-id>]",
                                                                                      eFlagProcessMustBeLaunched | eFlagProcessMustBePaused,
                                                                                      eStepTypeOut,
                                                                                      eStepScopeSource)));

    LoadSubCommand (interpreter, "step-over",   CommandObjectSP (new CommandObjectThreadStepWithTypeAndScope ("thread step-over",
                                                                                      "Source level single step over in specified thread (current thread, if none specified).",
                                                                                      "thread step-over [<thread-id>]",
                                                                                      eFlagProcessMustBeLaunched | eFlagProcessMustBePaused,
                                                                                      eStepTypeOver,
                                                                                      eStepScopeSource)));

    LoadSubCommand (interpreter, "step-inst",   CommandObjectSP (new CommandObjectThreadStepWithTypeAndScope ("thread step-inst",
                                                                                      "Single step one instruction in specified thread (current thread, if none specified).",
                                                                                      "thread step-inst [<thread-id>]",
                                                                                      eFlagProcessMustBeLaunched | eFlagProcessMustBePaused,
                                                                                      eStepTypeTrace,
                                                                                      eStepScopeInstruction)));

    LoadSubCommand (interpreter, "step-inst-over", CommandObjectSP (new CommandObjectThreadStepWithTypeAndScope ("thread step-inst-over",
                                                                                      "Single step one instruction in specified thread (current thread, if none specified), stepping over calls.",
                                                                                      "thread step-inst-over [<thread-id>]",
                                                                                      eFlagProcessMustBeLaunched | eFlagProcessMustBePaused,
                                                                                      eStepTypeTraceOver,
                                                                                      eStepScopeInstruction)));
}

CommandObjectMultiwordThread::~CommandObjectMultiwordThread ()
{
}


