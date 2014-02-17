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

static const char k_prompt_escape_char = '\1';

Editline::Editline (const char *prog,       // prog can't be NULL
                    const char *prompt,     // can be NULL for no prompt
                    FILE *fin,
                    FILE *fout,
                    FILE *ferr) :
    m_editline (NULL),
    m_history (NULL),
    m_history_event (),
    m_program (),
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
    m_lines_curr_line (0),
    m_lines_max_line (0),
    m_prompt_with_line_numbers (false),
    m_getting_line (false),
    m_got_eof (false),
    m_interrupted (false)
{
    if (prog && prog[0])
    {
        m_program = prog;
        m_editline = ::el_init(prog, fin, fout, ferr);
        m_history = ::history_init();
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
    if (m_history)
    {
        ::el_set (m_editline, EL_HIST, history, m_history);
    }
    ::el_set (m_editline, EL_ADDFN, "lldb-complete", "Editline completion function", Editline::CallbackComplete);
    ::el_set (m_editline, EL_ADDFN, "lldb-edit-prev-line", "Editline edit prev line", Editline::CallbackEditPrevLine);
    ::el_set (m_editline, EL_ADDFN, "lldb-edit-next-line", "Editline edit next line", Editline::CallbackEditNextLine);

    ::el_set (m_editline, EL_BIND, "^r", "em-inc-search-prev", NULL); // Cycle through backwards search, entering string
    ::el_set (m_editline, EL_BIND, "^w", "ed-delete-prev-word", NULL); // Delete previous word, behave like bash does.
    ::el_set (m_editline, EL_BIND, "\033[3~", "ed-delete-next-char", NULL); // Fix the delete key.
    ::el_set (m_editline, EL_BIND, "\t", "lldb-complete", NULL); // Bind TAB to be autocompelte
    
    // Source $PWD/.editrc then $HOME/.editrc
    ::el_source (m_editline, NULL);
 
    if (m_history)
    {
        ::history (m_history, &m_history_event, H_SETSIZE, 800);
        ::history (m_history, &m_history_event, H_SETUNIQUE, 1);
    }

    // Always read through our callback function so we don't read
    // stuff we aren't supposed to. This also stops the extra echoing
    // that can happen when you have more input than editline can handle
    // at once.
    SetGetCharCallback(GetCharFromInputFileCallback);

    LoadHistory();
}

Editline::~Editline()
{
    SaveHistory();

    if (m_history)
    {
        ::history_end (m_history);
        m_history = NULL;
    }

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

FileSpec
Editline::GetHistoryFile()
{
    char history_path[PATH_MAX];
    ::snprintf (history_path, sizeof(history_path), "~/.%s-history", m_program.c_str());
    return FileSpec(history_path, true);
}

bool
Editline::LoadHistory ()
{
    if (m_history)
    {
        FileSpec history_file(GetHistoryFile());
        if (history_file.Exists())
            ::history (m_history, &m_history_event, H_LOAD, history_file.GetPath().c_str());
        return true;
    }
    return false;
}

bool
Editline::SaveHistory ()
{
    if (m_history)
    {
        std::string history_path = GetHistoryFile().GetPath();
        ::history (m_history, &m_history_event, H_SAVE, history_path.c_str());
        return true;
    }
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
        const char *line_cstr = NULL;
        // Call el_gets to prompt the user and read the user's input.
//        {
//            // Make sure we know when we are in el_gets() by using a mutex
//            Mutex::Locker locker (m_gets_mutex);
            line_cstr = ::el_gets (m_editline, &line_len);
//        }
        
        static int save_errno = (line_len < 0) ? errno : 0;
        
        if (save_errno != 0)
        {
            error.SetError(save_errno, eErrorTypePOSIX);
        }
        else if (line_cstr)
        {
            // Decrement the length so we don't have newline characters in "line" for when
            // we assign the cstr into the std::string
            while (line_len > 0 &&
                   (line_cstr[line_len - 1] == '\n' ||
                    line_cstr[line_len - 1] == '\r'))
                --line_len;
            
            if (line_len > 0)
            {
                // We didn't strip the newlines, we just adjusted the length, and
                // we want to add the history item with the newlines
                if (m_history)
                    ::history (m_history, &m_history_event, H_ENTER, line_cstr);
                
                // Copy the part of the c string that we want (removing the newline chars)
                line.assign(line_cstr, line_len);
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
Editline::GetLine(std::string &line)
{
    Error error;
    line.clear();

    // Set arrow key bindings for up and down arrows for single line
    // mode where up and down arrows do prev/next history
    ::el_set (m_editline, EL_BIND, "^[[A", "ed-prev-history", NULL); // Map up arrow
    ::el_set (m_editline, EL_BIND, "^[[B", "ed-next-history", NULL); // Map down arrow
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
Editline::GetLines(const std::string &end_line, StringList &lines)
{
    Error error;
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
    ::el_set (m_editline, EL_BIND, "^[[A", "lldb-edit-prev-line", NULL); // Map up arrow
    ::el_set (m_editline, EL_BIND, "^[[B", "lldb-edit-next-line", NULL); // Map down arrow
    ::el_set (m_editline, EL_BIND, "^b", "ed-prev-history", NULL);
    ::el_set (m_editline, EL_BIND, "^n", "ed-next-history", NULL);
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
    if (m_line_complete_callback)
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
        char ch = ::fgetc(editline->GetInputFile());
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
        }
        else
        {
            *c = ch;
            return 1;
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

void
Editline::Interrupt ()
{
    m_interrupted = true;
    if (m_getting_line || m_lines_curr_line > 0)
        el_insertstr(m_editline, "\n"); // True to force the line to complete itself so we get exit from el_gets()
}
