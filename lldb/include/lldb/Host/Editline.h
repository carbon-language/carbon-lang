//===-- Editline.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Editline_h_
#define liblldb_Editline_h_
#if defined(__cplusplus)

#include "lldb/lldb-private.h"

#include <stdio.h>
#if defined(_WIN32)
#include "lldb/Host/windows/editlinewin.h"
#else
#if !defined(__ANDROID_NDK__)
#include <histedit.h>
#endif
#endif

#include <string>
#include <vector>

#include "lldb/Host/Condition.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Host/Predicate.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Editline Editline.h "lldb/Host/Editline.h"
/// @brief A class that encapsulates editline functionality.
//----------------------------------------------------------------------
class EditlineHistory;
    
typedef std::shared_ptr<EditlineHistory> EditlineHistorySP;
    
class Editline
{
public:
    typedef LineStatus (*LineCompletedCallbackType) (
        Editline *editline,
        StringList &lines,
        uint32_t line_idx,
        Error &error,
        void *baton);
    
    typedef int (*CompleteCallbackType) (
        const char *current_line,
        const char *cursor,
        const char *last_char,
        int skip_first_n_matches,
        int max_matches,
        StringList &matches,
        void *baton);

    typedef int (*GetCharCallbackType) (
        ::EditLine *,
        char *c);
    
    Editline(const char *prog,  // Used for the history file and for editrc program name
             const char *prompt,
             bool configure_for_multiline,             
             FILE *fin,
             FILE *fout,
             FILE *ferr);

    ~Editline();

    Error
    GetLine (std::string &line,
             bool &interrupted);

    Error
    GetLines (const std::string &end_line,
              StringList &lines,
              bool &interrupted);

    bool
    LoadHistory ();
    
    bool
    SaveHistory ();
    
    FILE *
    GetInputFile ();
    
    FILE *
    GetOutputFile ();
    
    FILE *
    GetErrorFile ();

    bool
    GettingLine () const
    {
        return m_getting_line;
    }

    void
    Hide ();

    void
    Refresh();

    bool
    Interrupt ();

    void
    SetAutoCompleteCallback (CompleteCallbackType callback,
                             void *baton)
    {
        m_completion_callback = callback;
        m_completion_callback_baton = baton;
    }

    void
    SetLineCompleteCallback (LineCompletedCallbackType callback,
                             void *baton)
    {
        m_line_complete_callback = callback;
        m_line_complete_callback_baton = baton;
    }

    size_t
    Push (const char *bytes, size_t len);
    
    static int
    GetCharFromInputFileCallback (::EditLine *e, char *c);

    void
    SetGetCharCallback (GetCharCallbackType callback);
    
    const char *
    GetPrompt();
    
    void
    SetPrompt (const char *p);
    
    void
    ShowLineNumbers (bool enable, uint32_t line_offset)
    {
        m_prompt_with_line_numbers = enable;
        m_line_offset = line_offset;
    }
    
private:

    Error
    PrivateGetLine(std::string &line);
    
    unsigned char
    HandleCompletion (int ch);

    static unsigned char
    CallbackEditPrevLine (::EditLine *e, int ch);
    
    static unsigned char
    CallbackEditNextLine (::EditLine *e, int ch);
    
    static unsigned char
    CallbackComplete (::EditLine *e, int ch);

    static const char *
    GetPromptCallback (::EditLine *e);

    static Editline *
    GetClientData (::EditLine *e);
    
    static FILE *
    GetFilePointer (::EditLine *e, int fd);

    enum class Command
    {
        None = 0,
        EditPrevLine,
        EditNextLine,
    };
    ::EditLine *m_editline;
    EditlineHistorySP m_history_sp;
    std::string m_prompt;
    std::string m_lines_prompt;
    lldb_private::Predicate<bool> m_getting_char;
    CompleteCallbackType m_completion_callback;
    void *m_completion_callback_baton;
    LineCompletedCallbackType m_line_complete_callback;
    void *m_line_complete_callback_baton;
    Command m_lines_command;
    uint32_t m_line_offset;
    uint32_t m_lines_curr_line;
    uint32_t m_lines_max_line;
    ConnectionFileDescriptor m_file;
    bool m_prompt_with_line_numbers;
    bool m_getting_line;
    bool m_got_eof;    // Set to true when we detect EOF
    bool m_interrupted;
    
    DISALLOW_COPY_AND_ASSIGN(Editline);
};

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // liblldb_Host_h_
