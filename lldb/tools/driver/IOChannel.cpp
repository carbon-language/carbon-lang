//===-- IOChannel.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IOChannel.h"

#include <map>

#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBHostOS.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBStringList.h"

#include <string.h>
#include <limits.h>

using namespace lldb;

typedef std::map<EditLine *, std::string> PromptMap;
const char *g_default_prompt = "(lldb) ";
PromptMap g_prompt_map;

static const char*
el_prompt(EditLine *el)
{
    PromptMap::const_iterator pos = g_prompt_map.find (el);
    if (pos == g_prompt_map.end())
        return g_default_prompt;
    return pos->second.c_str();
}

const char *
IOChannel::GetPrompt ()
{
    PromptMap::const_iterator pos = g_prompt_map.find (m_edit_line);
    if (pos == g_prompt_map.end())
        return g_default_prompt;
    return pos->second.c_str();
}

unsigned char
IOChannel::ElCompletionFn (EditLine *e, int ch)
{
    IOChannel *io_channel;
    if (el_get(e, EL_CLIENTDATA, &io_channel) == 0)
    {
        return io_channel->HandleCompletion (e, ch);
    }
    else
    {
        return CC_ERROR;
    }
}

unsigned char
IOChannel::HandleCompletion (EditLine *e, int ch)
{
    assert (e == m_edit_line);

    const LineInfo *line_info  = el_line(m_edit_line);
    SBStringList completions;
    int page_size = 40;

    int num_completions = m_driver->GetDebugger().GetCommandInterpreter().HandleCompletion (line_info->buffer,
                                                                                            line_info->cursor,
                                                                                            line_info->lastchar,
                                                                                            0,
                                                                                            -1,
                                                                                            completions);
    
    if (num_completions == -1)
    {
        el_insertstr (m_edit_line, m_completion_key);
        return CC_REDISPLAY;
    }

    // If we get a longer match display that first.
    const char *completion_str = completions.GetStringAtIndex(0);
    if (completion_str != NULL && *completion_str != '\0')
    {
        el_insertstr (m_edit_line, completion_str);
        return CC_REDISPLAY;
    }

    if (num_completions > 1)
    {
        const char *comment = "\nAvailable completions:";

        int num_elements = num_completions + 1;
        OutWrite(comment,  strlen (comment));
        if (num_completions < page_size)
        {
            for (int i = 1; i < num_elements; i++)
            {
                completion_str = completions.GetStringAtIndex(i);
                OutWrite("\n\t", 2);
                OutWrite(completion_str, strlen (completion_str));
            }
            OutWrite ("\n", 1);
        }
        else
        {
            int cur_pos = 1;
            char reply;
            int got_char;
            while (cur_pos < num_elements)
            {
                int endpoint = cur_pos + page_size;
                if (endpoint > num_elements)
                    endpoint = num_elements;
                for (; cur_pos < endpoint; cur_pos++)
                {
                    completion_str = completions.GetStringAtIndex(cur_pos);
                    OutWrite("\n\t", 2);
                    OutWrite(completion_str, strlen (completion_str));
                }

                if (cur_pos >= num_elements)
                {
                    OutWrite("\n", 1);
                    break;
                }

                OutWrite("\nMore (Y/n/a): ", strlen ("\nMore (Y/n/a): "));
                reply = 'n';
                got_char = el_getc(m_edit_line, &reply);
                if (got_char == -1 || reply == 'n')
                    break;
                if (reply == 'a')
                    page_size = num_elements - cur_pos;
            }
        }

    }

    if (num_completions == 0)
        return CC_REFRESH_BEEP;
    else
        return CC_REDISPLAY;
}

IOChannel::IOChannel
(
    FILE *in,
    FILE *out,
    FILE *err,
    Driver *driver
) :
    SBBroadcaster ("IOChannel"),
    m_driver (driver),
    m_read_thread (LLDB_INVALID_HOST_THREAD),
    m_read_thread_should_exit (false),
    m_out_file (out),
    m_err_file (err),
    m_command_queue (),
    m_completion_key ("\t"),
    m_edit_line (::el_init (SBHostOS::GetProgramFileSpec().GetFilename(), in, out, err)),
    m_history (history_init()),
    m_history_event(),
    m_getting_command (false)
{
    assert (m_edit_line);
    ::el_set (m_edit_line, EL_PROMPT, el_prompt);
    ::el_set (m_edit_line, EL_EDITOR, "emacs");
    ::el_set (m_edit_line, EL_HIST, history, m_history);

    // Source $PWD/.editrc then $HOME/.editrc
    ::el_source (m_edit_line, NULL);

    el_set(m_edit_line, EL_ADDFN, "lldb_complete",
            "LLDB completion function",
            IOChannel::ElCompletionFn);
    el_set(m_edit_line, EL_BIND, m_completion_key, "lldb_complete", NULL);
    el_set (m_edit_line, EL_CLIENTDATA, this);

    assert (m_history);
    ::history (m_history, &m_history_event, H_SETSIZE, 800);
    ::history (m_history, &m_history_event, H_SETUNIQUE, 1);
    // Load history
    HistorySaveLoad (false);
}

IOChannel::~IOChannel ()
{
    // Save history
    HistorySaveLoad (true);

    if (m_history != NULL)
    {
        ::history_end (m_history);
        m_history = NULL;
    }

    if (m_edit_line != NULL)
    {
        ::el_end (m_edit_line);
        m_edit_line = NULL;
    }
}

void
IOChannel::HistorySaveLoad (bool save)
{
    if (m_history != NULL)
    {
        char history_path[PATH_MAX];
        ::snprintf (history_path, sizeof(history_path), "~/.%s-history", SBHostOS::GetProgramFileSpec().GetFilename());
        if ((size_t)SBFileSpec::ResolvePath (history_path, history_path, sizeof(history_path)) < sizeof(history_path) - 1)
        {
            const char *path_ptr = history_path;
            if (save)
                ::history (m_history, &m_history_event, H_SAVE, path_ptr);
            else
                ::history (m_history, &m_history_event, H_LOAD, path_ptr);
        }
    }
}

bool
IOChannel::LibeditGetInput (std::string &new_line)
{
    if (m_edit_line != NULL)
    {
        int line_len = 0;
        const char *line = ::el_gets (m_edit_line, &line_len);
        if (line)
        {
            // strip any newlines off the end of the string...
            while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r'))
                --line_len;
            if (line_len > 0)
            {
                ::history (m_history, &m_history_event, H_ENTER, line);
                new_line.assign (line, line_len);   // Omit the newline
            }
            else
            {
                // Someone just hit ENTER, return the empty string
                new_line.clear();
            }
            // Return true to indicate success even if a string is empty
            return true;
        }
    }
    // Return false to indicate failure. This can happen when the file handle
    // is closed (EOF).
    new_line.clear();
    return false;
}

void *
IOChannel::IOReadThread (void *ptr)
{
    IOChannel *myself = static_cast<IOChannel *> (ptr);
    myself->Run();
    return NULL;
}

void
IOChannel::Run ()
{
    SBListener listener("IOChannel::Run");
    std::string new_line;

    SBBroadcaster interpreter_broadcaster (m_driver->GetDebugger().GetCommandInterpreter().GetBroadcaster());
    listener.StartListeningForEvents (interpreter_broadcaster,
                                      SBCommandInterpreter::eBroadcastBitResetPrompt |
                                      SBCommandInterpreter::eBroadcastBitThreadShouldExit |
                                      SBCommandInterpreter::eBroadcastBitQuitCommandReceived);

    listener.StartListeningForEvents (*this,
                                      IOChannel::eBroadcastBitThreadShouldExit);

    listener.StartListeningForEvents (*m_driver,
                                      Driver::eBroadcastBitReadyForInput |
                                      Driver::eBroadcastBitThreadShouldExit);

    // Let anyone know that the IO channel is up and listening and ready for events
    BroadcastEventByType (eBroadcastBitThreadDidStart);
    bool done = false;
    while (!done)
    {
        SBEvent event;

        listener.WaitForEvent (UINT32_MAX, event);
        if (!event.IsValid())
            continue;

        const uint32_t event_type = event.GetType();

        if (event.GetBroadcaster().IsValid())
        {
            if (event.BroadcasterMatchesPtr (m_driver))
            {
                if (event_type & Driver::eBroadcastBitReadyForInput)
                {
                    std::string line;

                    if (CommandQueueIsEmpty())
                    {
                        if (LibeditGetInput(line) == false)
                        {
                            // EOF or some other file error occurred
                            done = true;
                            continue;
                        }
                    }
                    else
                    {
                        GetCommandFromQueue (line);
                    }

                    // TO BE DONE: FIGURE OUT WHICH COMMANDS SHOULD NOT BE REPEATED IF USER PRESSES PLAIN 'RETURN'
                    // AND TAKE CARE OF THAT HERE.

                    SBEvent line_event(IOChannel::eBroadcastBitHasUserInput,
                             line.c_str(),
                             line.size());
                    BroadcastEvent (line_event);
                }
                else if (event_type & Driver::eBroadcastBitThreadShouldExit)
                {
                    done = true;
                    break;
                }
            }
            else if (event.BroadcasterMatchesRef (interpreter_broadcaster))
            {
                switch (event_type)
                {
                case SBCommandInterpreter::eBroadcastBitResetPrompt:
                    {
                        const char *new_prompt = SBEvent::GetCStringFromEvent (event);
                        if (new_prompt)
                            g_prompt_map[m_edit_line] = new_prompt;
                    }
                    break;

                case SBCommandInterpreter::eBroadcastBitThreadShouldExit:
                case SBCommandInterpreter::eBroadcastBitQuitCommandReceived:
                    done = true;
                    break;
                }
            }
            else if (event.BroadcasterMatchesPtr (this))
            {
                if (event_type & IOChannel::eBroadcastBitThreadShouldExit)
                {
                    done = true;
                    break;
                }
            }
        }
    }
    BroadcastEventByType (IOChannel::eBroadcastBitThreadDidExit);
    m_driver = NULL;
    m_read_thread = NULL;
}

bool
IOChannel::Start ()
{
    if (m_read_thread != LLDB_INVALID_HOST_THREAD)
        return true;

    m_read_thread = SBHostOS::ThreadCreate ("<lldb.driver.commandline_io>", IOChannel::IOReadThread, this,
                                            NULL);

    return (m_read_thread != LLDB_INVALID_HOST_THREAD);
}

bool
IOChannel::Stop ()
{
    if (m_read_thread == LLDB_INVALID_HOST_THREAD)
        return true;

    BroadcastEventByType (eBroadcastBitThreadShouldExit);

    // Don't call Host::ThreadCancel since el_gets won't respond to this
    // function call -- the thread will just die and all local variables in
    // IOChannel::Run() won't get destructed down which is bad since there is
    // a local listener holding onto broadcasters... To ensure proper shutdown,
    // a ^D (control-D) sequence (0x04) should be written to other end of the
    // the "in" file handle that was passed into the contructor as closing the
    // file handle doesn't seem to make el_gets() exit....
    return SBHostOS::ThreadJoin (m_read_thread, NULL, NULL);
}

void
IOChannel::RefreshPrompt ()
{
    ::el_set (m_edit_line, EL_REFRESH);
}

void
IOChannel::OutWrite (const char *buffer, size_t len)
{
    if (len == 0)
        return;
    ::fwrite (buffer, 1, len, m_out_file);
}

void
IOChannel::ErrWrite (const char *buffer, size_t len)
{
    if (len == 0)
        return;
    ::fwrite (buffer, 1, len, m_err_file);
}

void
IOChannel::AddCommandToQueue (const char *command)
{
    m_command_queue.push (std::string(command));
}

bool
IOChannel::GetCommandFromQueue (std::string &cmd)
{
    if (m_command_queue.empty())
        return false;
    cmd.swap(m_command_queue.front());
    m_command_queue.pop ();
    return true;
}

int
IOChannel::CommandQueueSize () const
{
    return m_command_queue.size();
}

void
IOChannel::ClearCommandQueue ()
{
    while (!m_command_queue.empty())
        m_command_queue.pop();
}

bool
IOChannel::CommandQueueIsEmpty () const
{
    return m_command_queue.empty();
}
