//===-- Editline.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "lldb/Host/Editline.h"

#include "lldb/Core/Error.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/StringList.h"
#include "lldb/Host/Host.h"

#include <limits.h>

using namespace lldb;
using namespace lldb_private;

namespace lldb_private {
    typedef std::weak_ptr<EditlineHistory> EditlineHistoryWP;
    
    
    // EditlineHistory objects are sometimes shared between multiple
    // Editline instances with the same program name. This class allows
    // multiple editline instances to
    //
    
    class EditlineHistory
    {
    private:
        // Use static GetHistory() function to get a EditlineHistorySP to one of these objects
        EditlineHistory(const std::string &prefix, uint32_t size, bool unique_entries) :
            m_history (NULL),
            m_event (),
            m_prefix (prefix),
            m_path ()
        {
            m_history = ::history_init();
            ::history (m_history, &m_event, H_SETSIZE, size);
            if (unique_entries)
                ::history (m_history, &m_event, H_SETUNIQUE, 1);
        }
        
        const char *
        GetHistoryFilePath()
        {
            if (m_path.empty() && m_history && !m_prefix.empty())
            {
                char history_path[PATH_MAX];
                ::snprintf (history_path, sizeof(history_path), "~/.%s-history", m_prefix.c_str());
                m_path = std::move(FileSpec(history_path, true).GetPath());
            }
            if (m_path.empty())
                return NULL;
            return m_path.c_str();
        }
        
    public:
        
        ~EditlineHistory()
        {
            Save ();
            
            if (m_history)
            {
                ::history_end (m_history);
                m_history = NULL;
            }
        }

        static EditlineHistorySP
        GetHistory (const std::string &prefix)
        {
            typedef std::map<std::string, EditlineHistoryWP> WeakHistoryMap;
            static Mutex g_mutex(Mutex::eMutexTypeRecursive);
            static WeakHistoryMap g_weak_map;
            Mutex::Locker locker (g_mutex);
            WeakHistoryMap::const_iterator pos = g_weak_map.find (prefix);
            EditlineHistorySP history_sp;
            if (pos != g_weak_map.end())
            {
                history_sp = pos->second.lock();
                if (history_sp)
                    return history_sp;
                g_weak_map.erase(pos);
            }
            history_sp.reset(new EditlineHistory(prefix, 800, true));
            g_weak_map[prefix] = history_sp;
            return history_sp;
        }
        
        bool IsValid() const
        {
            return m_history != NULL;
        }
        
        ::History *
        GetHistoryPtr ()
        {
            return m_history;
        }
        
        void
        Enter (const char *line_cstr)
        {
            if (m_history)
                ::history (m_history, &m_event, H_ENTER, line_cstr);
        }
        
        bool
        Load ()
        {
            if (m_history)
            {
                const char *path = GetHistoryFilePath();
                if (path)
                {
                    ::history (m_history, &m_event, H_LOAD, path);
                    return true;
                }
            }
            return false;
        }
        
        bool
        Save ()
        {
            if (m_history)
            {
                const char *path = GetHistoryFilePath();
                if (path)
                {
                    ::history (m_history, &m_event, H_SAVE, path);
                    return true;
                }
            }
            return false;
        }
        
    protected:
        ::History *m_history;       // The history object
        ::HistEvent m_event;// The history event needed to contain all history events
        std::string m_prefix;       // The prefix name (usually the editline program name) to use when loading/saving history
        std::string m_path;         // Path to the history file
    };
}


static const char k_prompt_escape_char = '\1';

Editline::Editline (const char *prog,       // prog can't be NULL
                    const char *prompt,     // can be NULL for no prompt
                    bool configure_for_multiline,
                    FILE *fin,
                    FILE *fout,
                    FILE *ferr) :
    m_editline (NULL),
    m_history_sp (),
    m_prompt (),
    m_lines_prompt (),
    m_getc_buffer (),
    m_getc_mutex (Mutex::eMutexTypeNormal),
    m_getc_cond (),
//    m_gets_mutex (Mutex::eMutexTypeNormal),
    m_completion_callback (NULL),
    m_completion_callback_baton (NULL),
    m_line_complete_callback (NULL),
    m_line_complete_callback_baton (NULL),
    m_lines_command (Command::None),
    m_line_offset (0),
    m_lines_curr_line (0),
    m_lines_max_line (0),
    m_file (fileno(fin), false),
    m_prompt_with_line_numbers (false),
    m_getting_line (false),
    m_got_eof (false),
    m_interrupted (false)
{
    if (prog && prog[0])
    {
        m_editline = ::el_init(prog, fin, fout, ferr);
        
        // Get a shared history instance
        m_history_sp = EditlineHistory::GetHistory(prog);
    }
    else
    {
        m_editline = ::el_init("lldb-tmp", fin, fout, ferr);
    }
    
    if (prompt && prompt[0])
        SetPrompt (prompt);

    //::el_set (m_editline, EL_BIND, "^[[A", NULL); // Print binding for up arrow key
    //::el_set (m_editline, EL_BIND, "^[[B", NULL); // Print binding for up down key

    assert (m_editline);
    ::el_set (m_editline, EL_CLIENTDATA, this);

    // only defined for newer versions of editline
#ifdef EL_PROMPT_ESC
    ::el_set (m_editline, EL_PROMPT_ESC, GetPromptCallback, k_prompt_escape_char);
#else
    // fall back on old prompt setting code
    ::el_set (m_editline, EL_PROMPT, GetPromptCallback);
#endif
    ::el_set (m_editline, EL_EDITOR, "emacs");
    if (m_history_sp && m_history_sp->IsValid())
    {
        ::el_set (m_editline, EL_HIST, history, m_history_sp->GetHistoryPtr());
    }
    ::el_set (m_editline, EL_ADDFN, "lldb-complete", "Editline completion function", Editline::CallbackComplete);
    // Keep old "lldb_complete" mapping for older clients that used this in their .editrc. editline also
    // has a bad bug where if you have a bind command that tries to bind to a function name that doesn't
    // exist, it will corrupt the heap and probably crash your process later.
    ::el_set (m_editline, EL_ADDFN, "lldb_complete", "Editline completion function", Editline::CallbackComplete);
    ::el_set (m_editline, EL_ADDFN, "lldb-edit-prev-line", "Editline edit prev line", Editline::CallbackEditPrevLine);
    ::el_set (m_editline, EL_ADDFN, "lldb-edit-next-line", "Editline edit next line", Editline::CallbackEditNextLine);

    ::el_set (m_editline, EL_BIND, "^r", "em-inc-search-prev", NULL); // Cycle through backwards search, entering string
    ::el_set (m_editline, EL_BIND, "^w", "ed-delete-prev-word", NULL); // Delete previous word, behave like bash does.
    ::el_set (m_editline, EL_BIND, "\033[3~", "ed-delete-next-char", NULL); // Fix the delete key.
    ::el_set (m_editline, EL_BIND, "\t", "lldb-complete", NULL); // Bind TAB to be auto complete
    
    if (configure_for_multiline)
    {
        // Use escape sequences for control characters due to bugs in editline
        // where "-k up" and "-k down" don't always work.
        ::el_set (m_editline, EL_BIND, "^[[A", "lldb-edit-prev-line", NULL); // Map up arrow
        ::el_set (m_editline, EL_BIND, "^[[B", "lldb-edit-next-line", NULL); // Map down arrow
        // Bindings for next/prev history
        ::el_set (m_editline, EL_BIND, "^P", "ed-prev-history", NULL); // Map up arrow
        ::el_set (m_editline, EL_BIND, "^N", "ed-next-history", NULL); // Map down arrow
    }
    else
    {
        // Use escape sequences for control characters due to bugs in editline
        // where "-k up" and "-k down" don't always work.
        ::el_set (m_editline, EL_BIND, "^[[A", "ed-prev-history", NULL); // Map up arrow
        ::el_set (m_editline, EL_BIND, "^[[B", "ed-next-history", NULL); // Map down arrow
    }
    
    // Source $PWD/.editrc then $HOME/.editrc
    ::el_source (m_editline, NULL);
 
    // Always read through our callback function so we don't read
    // stuff we aren't supposed to. This also stops the extra echoing
    // that can happen when you have more input than editline can handle
    // at once.
    SetGetCharCallback(GetCharFromInputFileCallback);

    LoadHistory();
}

Editline::~Editline()
{
    // EditlineHistory objects are sometimes shared between multiple
    // Editline instances with the same program name. So just release
    // our shared pointer and if we are the last owner, it will save the
    // history to the history save file automatically.
    m_history_sp.reset();
    
    // Disable edit mode to stop the terminal from flushing all input
    // during the call to el_end() since we expect to have multiple editline
    // instances in this program.
    ::el_set (m_editline, EL_EDITMODE, 0);

    ::el_end(m_editline);
    m_editline = NULL;
}

void
Editline::SetGetCharCallback (GetCharCallbackType callback)
{
    ::el_set (m_editline, EL_GETCFN, callback);
}

bool
Editline::LoadHistory ()
{
    if (m_history_sp)
        return m_history_sp->Load();
    return false;
}

bool
Editline::SaveHistory ()
{
    if (m_history_sp)
        return m_history_sp->Save();
    return false;
}


Error
Editline::PrivateGetLine(std::string &line)
{
    Error error;
    if (m_interrupted)
    {
        error.SetErrorString("interrupted");
        return error;
    }
    
    line.clear();
    if (m_editline != NULL)
    {
        int line_len = 0;
        // Call el_gets to prompt the user and read the user's input.
        const char *line_cstr = ::el_gets (m_editline, &line_len);
        
        static int save_errno = (line_len < 0) ? errno : 0;
        
        if (save_errno != 0)
        {
            error.SetError(save_errno, eErrorTypePOSIX);
        }
        else if (line_cstr)
        {
            // Decrement the length so we don't have newline characters in "line" for when
            // we assign the cstr into the std::string
            llvm::StringRef line_ref (line_cstr);
            line_ref = line_ref.rtrim("\n\r");
            
            if (!line_ref.empty() && !m_interrupted)
            {
                // We didn't strip the newlines, we just adjusted the length, and
                // we want to add the history item with the newlines
                if (m_history_sp)
                    m_history_sp->Enter(line_cstr);
                
                // Copy the part of the c string that we want (removing the newline chars)
                line = std::move(line_ref.str());
            }
        }
    }
    else
    {
        error.SetErrorString("the EditLine instance has been deleted");
    }
    return error;
}


Error
Editline::GetLine(std::string &line, bool &interrupted)
{
    Error error;
    interrupted = false;
    line.clear();

    // Set arrow key bindings for up and down arrows for single line
    // mode where up and down arrows do prev/next history
    m_interrupted = false;

    if (!m_got_eof)
    {
        if (m_getting_line)
        {
            error.SetErrorString("already getting a line");
            return error;
        }
        if (m_lines_curr_line > 0)
        {
            error.SetErrorString("already getting lines");
            return error;
        }
        m_getting_line = true;
        error = PrivateGetLine(line);
        m_getting_line = false;
    }

    interrupted = m_interrupted;

    if (m_got_eof && line.empty())
    {
        // Only set the error if we didn't get an error back from PrivateGetLine()
        if (error.Success())
            error.SetErrorString("end of file");
    }

    return error;
}

size_t
Editline::Push (const char *bytes, size_t len)
{
    if (m_editline)
    {
        // Must NULL terminate the string for el_push() so we stick it
        // into a std::string first
        ::el_push(m_editline,
                  const_cast<char*>(std::string (bytes, len).c_str()));
        return len;
    }
    return 0;
}


Error
Editline::GetLines(const std::string &end_line, StringList &lines, bool &interrupted)
{
    Error error;
    interrupted = false;
    if (m_getting_line)
    {
        error.SetErrorString("already getting a line");
        return error;
    }
    if (m_lines_curr_line > 0)
    {
        error.SetErrorString("already getting lines");
        return error;
    }
    
    // Set arrow key bindings for up and down arrows for multiple line
    // mode where up and down arrows do edit prev/next line
    m_interrupted = false;

    LineStatus line_status = LineStatus::Success;

    lines.Clear();

    FILE *out_file = GetOutputFile();
    FILE *err_file = GetErrorFile();
    m_lines_curr_line = 1;
    while (line_status != LineStatus::Done)
    {
        const uint32_t line_idx = m_lines_curr_line-1;
        if (line_idx >= lines.GetSize())
            lines.SetSize(m_lines_curr_line);
        m_lines_max_line = lines.GetSize();
        m_lines_command = Command::None;
        assert(line_idx < m_lines_max_line);
        std::string &line = lines[line_idx];
        error = PrivateGetLine(line);
        if (error.Fail())
        {
            line_status = LineStatus::Error;
        }
        else if (m_interrupted)
        {
            interrupted = true;
            line_status = LineStatus::Done;
        }
        else
        {
            switch (m_lines_command)
            {
                case Command::None:
                    if (m_line_complete_callback)
                    {
                        line_status = m_line_complete_callback (this,
                                                                lines,
                                                                line_idx,
                                                                error,
                                                                m_line_complete_callback_baton);
                    }
                    else if (line == end_line)
                    {
                        line_status = LineStatus::Done;
                    }

                    if (line_status == LineStatus::Success)
                    {
                        ++m_lines_curr_line;
                        // If we already have content for the next line because
                        // we were editing previous lines, then populate the line
                        // with the appropriate contents
                        if (line_idx+1 < lines.GetSize() && !lines[line_idx+1].empty())
                            ::el_push (m_editline,
                                       const_cast<char*>(lines[line_idx+1].c_str()));
                    }
                    else if (line_status == LineStatus::Error)
                    {
                        // Clear to end of line ("ESC[K"), then print the error,
                        // then go to the next line ("\n") and then move cursor up
                        // two lines ("ESC[2A").
                        fprintf (err_file, "\033[Kerror: %s\n\033[2A", error.AsCString());
                    }
                    break;
                case Command::EditPrevLine:
                    if (m_lines_curr_line > 1)
                    {
                        //::fprintf (out_file, "\033[1A\033[%uD\033[2K", (uint32_t)(m_lines_prompt.size() + lines[line_idx].size())); // Make cursor go up a line and clear that line
                        ::fprintf (out_file, "\033[1A\033[1000D\033[2K");
                        if (!lines[line_idx-1].empty())
                            ::el_push (m_editline,
                                       const_cast<char*>(lines[line_idx-1].c_str()));
                        --m_lines_curr_line;
                    }
                    break;
                case Command::EditNextLine:
                    // Allow the down arrow to create a new line
                    ++m_lines_curr_line;
                    //::fprintf (out_file, "\033[1B\033[%uD\033[2K", (uint32_t)(m_lines_prompt.size() + lines[line_idx].size()));
                    ::fprintf (out_file, "\033[1B\033[1000D\033[2K");
                    if (line_idx+1 < lines.GetSize() && !lines[line_idx+1].empty())
                        ::el_push (m_editline,
                                   const_cast<char*>(lines[line_idx+1].c_str()));
                    break;
            }
        }
    }
    m_lines_curr_line = 0;
    m_lines_command = Command::None;

    // If we have a callback, call it one more time to let the
    // user know the lines are complete
    if (m_line_complete_callback && !interrupted)
        m_line_complete_callback (this,
                                  lines,
                                  UINT32_MAX,
                                  error,
                                  m_line_complete_callback_baton);

    return error;
}

unsigned char
Editline::HandleCompletion (int ch)
{
    if (m_completion_callback == NULL)
        return CC_ERROR;

    const LineInfo *line_info  = ::el_line(m_editline);
    StringList completions;
    int page_size = 40;
        
    const int num_completions = m_completion_callback (line_info->buffer,
                                                       line_info->cursor,
                                                       line_info->lastchar,
                                                       0,     // Don't skip any matches (start at match zero)
                                                       -1,    // Get all the matches
                                                       completions,
                                                       m_completion_callback_baton);
    
    FILE *out_file = GetOutputFile();

//    if (num_completions == -1)
//    {
//        ::el_insertstr (m_editline, m_completion_key);
//        return CC_REDISPLAY;
//    }
//    else
    if (num_completions == -2)
    {
        // Replace the entire line with the first string...
        ::el_deletestr (m_editline, line_info->cursor - line_info->buffer);
        ::el_insertstr (m_editline, completions.GetStringAtIndex(0));
        return CC_REDISPLAY;
    }
    
    // If we get a longer match display that first.
    const char *completion_str = completions.GetStringAtIndex(0);
    if (completion_str != NULL && *completion_str != '\0')
    {
        el_insertstr (m_editline, completion_str);
        return CC_REDISPLAY;
    }
    
    if (num_completions > 1)
    {
        int num_elements = num_completions + 1;
        ::fprintf (out_file, "\nAvailable completions:");
        if (num_completions < page_size)
        {
            for (int i = 1; i < num_elements; i++)
            {
                completion_str = completions.GetStringAtIndex(i);
                ::fprintf (out_file, "\n\t%s", completion_str);
            }
            ::fprintf (out_file, "\n");
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
                    ::fprintf (out_file, "\n\t%s", completion_str);
                }
                
                if (cur_pos >= num_elements)
                {
                    ::fprintf (out_file, "\n");
                    break;
                }
                
                ::fprintf (out_file, "\nMore (Y/n/a): ");
                reply = 'n';
                got_char = el_getc(m_editline, &reply);
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

Editline *
Editline::GetClientData (::EditLine *e)
{
    Editline *editline = NULL;
    if (e && ::el_get(e, EL_CLIENTDATA, &editline) == 0)
        return editline;
    return NULL;
}

FILE *
Editline::GetInputFile ()
{
    return GetFilePointer (m_editline, 0);
}

FILE *
Editline::GetOutputFile ()
{
    return GetFilePointer (m_editline, 1);
}

FILE *
Editline::GetErrorFile ()
{
    return GetFilePointer (m_editline, 2);
}

const char *
Editline::GetPrompt()
{
    if (m_prompt_with_line_numbers && m_lines_curr_line > 0)
    {
        StreamString strm;
        strm.Printf("%3u: ", m_lines_curr_line);
        m_lines_prompt = std::move(strm.GetString());
        return m_lines_prompt.c_str();
    }
    else
    {
        return m_prompt.c_str();
    }
}

void
Editline::SetPrompt (const char *p)
{
    if (p && p[0])
        m_prompt = p;
    else
        m_prompt.clear();
    size_t start_pos = 0;
    size_t escape_pos;
    while ((escape_pos = m_prompt.find('\033', start_pos)) != std::string::npos)
    {
        m_prompt.insert(escape_pos, 1, k_prompt_escape_char);
        start_pos += 2;
    }
}

FILE *
Editline::GetFilePointer (::EditLine *e, int fd)
{
    FILE *file_ptr = NULL;
    if (e && ::el_get(e, EL_GETFP, fd, &file_ptr) == 0)
        return file_ptr;
    return NULL;
}

unsigned char
Editline::CallbackEditPrevLine (::EditLine *e, int ch)
{
    Editline *editline = GetClientData (e);
    if (editline->m_lines_curr_line > 1)
    {
        editline->m_lines_command = Command::EditPrevLine;
        return CC_NEWLINE;
    }
    return CC_ERROR;
}
unsigned char
Editline::CallbackEditNextLine (::EditLine *e, int ch)
{
    Editline *editline = GetClientData (e);
    if (editline->m_lines_curr_line < editline->m_lines_max_line)
    {
        editline->m_lines_command = Command::EditNextLine;
        return CC_NEWLINE;
    }
    return CC_ERROR;
}

unsigned char
Editline::CallbackComplete (::EditLine *e, int ch)
{
    Editline *editline = GetClientData (e);
    if (editline)
        return editline->HandleCompletion (ch);
    return CC_ERROR;
}

const char *
Editline::GetPromptCallback (::EditLine *e)
{
    Editline *editline = GetClientData (e);
    if (editline)
        return editline->GetPrompt();
    return "";
}

size_t
Editline::SetInputBuffer (const char *c, size_t len)
{
    if (c && len > 0)
    {
        Mutex::Locker locker(m_getc_mutex);
        SetGetCharCallback(GetCharInputBufferCallback);
        m_getc_buffer.append(c, len);
        m_getc_cond.Broadcast();
    }
    return len;
}

int
Editline::GetChar (char *c)
{
    Mutex::Locker locker(m_getc_mutex);
    if (m_getc_buffer.empty())
        m_getc_cond.Wait(m_getc_mutex);
    if (m_getc_buffer.empty())
        return 0;
    *c = m_getc_buffer[0];
    m_getc_buffer.erase(0,1);
    return 1;
}

int
Editline::GetCharInputBufferCallback (EditLine *e, char *c)
{
    Editline *editline = GetClientData (e);
    if (editline)
        return editline->GetChar(c);
    return 0;
}

int
Editline::GetCharFromInputFileCallback (EditLine *e, char *c)
{
    Editline *editline = GetClientData (e);
    if (editline && editline->m_got_eof == false)
    {
        FILE *f = editline->GetInputFile();
        if (f == NULL)
        {
            editline->m_got_eof = true;
            return 0;
        }
        
        
        while (1)
        {
            lldb::ConnectionStatus status = eConnectionStatusSuccess;
            char ch = 0;
            if (editline->m_file.Read(&ch, 1, UINT32_MAX, status, NULL))
            {
                if (ch == '\x04')
                {
                    // Only turn a CTRL+D into a EOF if we receive the
                    // CTRL+D an empty line, otherwise it will forward
                    // delete the character at the cursor
                    const LineInfo *line_info = ::el_line(e);
                    if (line_info != NULL &&
                        line_info->buffer == line_info->cursor &&
                        line_info->cursor == line_info->lastchar)
                    {
                        ch = EOF;
                    }
                }
            
                if (ch == EOF)
                {
                    editline->m_got_eof = true;
                    break;
                }
                else
                {
                    *c = ch;
                    return 1;
                }
            }
            else
            {
                switch (status)
                {
                    case eConnectionStatusInterrupted:
                        editline->m_interrupted = true;
                        *c = '\n';
                        return 1;

                    case eConnectionStatusSuccess:         // Success
                        break;
                        
                    case eConnectionStatusError:           // Check GetError() for details
                    case eConnectionStatusTimedOut:        // Request timed out
                    case eConnectionStatusEndOfFile:       // End-of-file encountered
                    case eConnectionStatusNoConnection:    // No connection
                    case eConnectionStatusLostConnection:  // Lost connection while connected to a valid connection
                        editline->m_got_eof = true;
                        break;
                }
            }
        }
    }
    return 0;
}

void
Editline::Hide ()
{
    FILE *out_file = GetOutputFile();
    if (out_file)
    {
        const LineInfo *line_info  = ::el_line(m_editline);
        if (line_info)
            ::fprintf (out_file, "\033[%uD\033[K", (uint32_t)(strlen(GetPrompt()) + line_info->cursor - line_info->buffer));
    }
}


void
Editline::Refresh()
{
    ::el_set (m_editline, EL_REFRESH);
}

bool
Editline::Interrupt ()
{
    m_interrupted = true;
    if (m_getting_line || m_lines_curr_line > 0)
        return m_file.InterruptRead();
    return false; // Interrupt not handled as we weren't getting a line or lines
}
