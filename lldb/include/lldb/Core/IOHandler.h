//===-- IOHandler.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_IOHandler_h_
#define liblldb_IOHandler_h_

#include <string.h>

#include <stack>

#include "lldb/lldb-public.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/StringList.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Host/Mutex.h"

namespace curses
{
    class Application;
    typedef std::unique_ptr<Application> ApplicationAP;
}

namespace lldb_private {

    class IOHandler
    {
    public:
        IOHandler (Debugger &debugger);

        IOHandler (Debugger &debugger,
                   const lldb::StreamFileSP &input_sp,
                   const lldb::StreamFileSP &output_sp,
                   const lldb::StreamFileSP &error_sp,
                   uint32_t flags);

        virtual
        ~IOHandler ();

        // Each IOHandler gets to run until it is done. It should read data
        // from the "in" and place output into "out" and "err and return
        // when done.
        virtual void
        Run () = 0;

        // Hide any characters that have been displayed so far so async
        // output can be displayed. Refresh() will be called after the
        // output has been displayed.
        virtual void
        Hide () = 0;
        
        // Called when the async output has been received in order to update
        // the input reader (refresh the prompt and redisplay any current
        // line(s) that are being edited
        virtual void
        Refresh () = 0;

        virtual void
        Interrupt () = 0;
        
        virtual void
        GotEOF() = 0;
        
        virtual bool
        IsActive ()
        {
            return m_active && !m_done;
        }

        virtual void
        SetIsDone (bool b)
        {
            m_done = b;
        }

        virtual bool
        GetIsDone ()
        {
            return m_done;
        }

        virtual void
        Activate ()
        {
            m_active = true;
        }
        
        virtual void
        Deactivate ()
        {
            m_active = false;
        }

        virtual const char *
        GetPrompt ()
        {
            // Prompt support isn't mandatory
            return NULL;
        }
        
        virtual bool
        SetPrompt (const char *prompt)
        {
            // Prompt support isn't mandatory
            return false;
        }
        
        virtual ConstString
        GetControlSequence (char ch)
        {
            return ConstString();
        }
        
        int
        GetInputFD();
        
        int
        GetOutputFD();
        
        int
        GetErrorFD();

        FILE *
        GetInputFILE();
        
        FILE *
        GetOutputFILE();
        
        FILE *
        GetErrorFILE();

        lldb::StreamFileSP &
        GetInputStreamFile();
        
        lldb::StreamFileSP &
        GetOutputStreamFile();
        
        lldb::StreamFileSP &
        GetErrorStreamFile();

        Debugger &
        GetDebugger()
        {
            return m_debugger;
        }

        void *
        GetUserData ()
        {
            return m_user_data;
        }

        void
        SetUserData (void *user_data)
        {
            m_user_data = user_data;
        }

        Flags &
        GetFlags ()
        {
            return m_flags;
        }

        const Flags &
        GetFlags () const
        {
            return m_flags;
        }

        //------------------------------------------------------------------
        /// Check if the input is being supplied interactively by a user
        ///
        /// This will return true if the input stream is a terminal (tty or
        /// pty) and can cause IO handlers to do different things (like
        /// for a comfirmation when deleting all breakpoints).
        //------------------------------------------------------------------
        bool
        GetIsInteractive ();

        //------------------------------------------------------------------
        /// Check if the input is coming from a real terminal.
        ///
        /// A real terminal has a valid size with a certain number of rows
        /// and colums. If this function returns true, then terminal escape
        /// sequences are expected to work (cursor movement escape sequences,
        /// clearning lines, etc).
        //------------------------------------------------------------------
        bool
        GetIsRealTerminal ();

    protected:
        Debugger &m_debugger;
        lldb::StreamFileSP m_input_sp;
        lldb::StreamFileSP m_output_sp;
        lldb::StreamFileSP m_error_sp;
        Flags m_flags;
        void *m_user_data;
        bool m_done;
        bool m_active;

    private:
        DISALLOW_COPY_AND_ASSIGN (IOHandler);
    };

    
    //------------------------------------------------------------------
    /// A delegate class for use with IOHandler subclasses.
    ///
    /// The IOHandler delegate is designed to be mixed into classes so
    /// they can use an IOHandler subclass to fetch input and notify the
    /// object that inherits from this delegate class when a token is
    /// received.
    //------------------------------------------------------------------
    class IOHandlerDelegate
    {
    public:
        enum class Completion {
            None,
            LLDBCommand,
            Expression
        };
        
        IOHandlerDelegate (Completion completion = Completion::None) :
            m_completion(completion),
            m_io_handler_done (false)
        {
        }
        
        virtual
        ~IOHandlerDelegate()
        {
        }
        
        virtual void
        IOHandlerActivated (IOHandler &io_handler)
        {
        }
        
        virtual int
        IOHandlerComplete (IOHandler &io_handler,
                           const char *current_line,
                           const char *cursor,
                           const char *last_char,
                           int skip_first_n_matches,
                           int max_matches,
                           StringList &matches);
        
        //------------------------------------------------------------------
        /// Called when a line or lines have been retrieved.
        ///
        /// This funtion can handle the current line and possibly call
        /// IOHandler::SetIsDone(true) when the IO handler is done like when
        /// "quit" is entered as a command, of when an empty line is
        /// received. It is up to the delegate to determine when a line
        /// should cause a IOHandler to exit.
        //------------------------------------------------------------------
        virtual void
        IOHandlerInputComplete (IOHandler &io_handler, std::string &data) = 0;
        
        //------------------------------------------------------------------
        /// Called when a line in \a lines has been updated when doing
        /// multi-line input.
        ///
        /// @return
        ///     Return an enumeration to indicate the status of the current
        ///     line:
        ///         Success - The line is good and should be added to the
        ///                   multiple lines
        ///         Error - There is an error with the current line and it
        ///                 need to be re-edited before it is acceptable
        ///         Done - The lines collection is complete and ready to be
        ///                returned.
        //------------------------------------------------------------------
        virtual LineStatus
        IOHandlerLinesUpdated (IOHandler &io_handler,
                               StringList &lines,
                               uint32_t line_idx,
                               Error &error)
        {
            return LineStatus::Done; // Stop getting lines on the first line that is updated
            // subclasses should do something more intelligent here.
            // This function will not be called on IOHandler objects
            // that are getting single lines.
        }
        
        
        virtual ConstString
        GetControlSequence (char ch)
        {
            return ConstString();
        }
        
    protected:
        Completion m_completion; // Support for common builtin completions
        bool m_io_handler_done;
    };

    //----------------------------------------------------------------------
    // IOHandlerDelegateMultiline
    //
    // A IOHandlerDelegate that handles terminating multi-line input when
    // the last line is equal to "end_line" which is specified in the
    // constructor.
    //----------------------------------------------------------------------
    class IOHandlerDelegateMultiline :
        public IOHandlerDelegate
    {
    public:
        IOHandlerDelegateMultiline (const char *end_line,
                                    Completion completion = Completion::None) :
            IOHandlerDelegate (completion),
            m_end_line((end_line && end_line[0]) ? end_line : "")
        {
        }
        
        virtual
        ~IOHandlerDelegateMultiline ()
        {
        }
        
        virtual ConstString
        GetControlSequence (char ch)
        {
            if (ch == 'd')
                return ConstString (m_end_line + "\n");
            return ConstString();
        }

        virtual LineStatus
        IOHandlerLinesUpdated (IOHandler &io_handler,
                               StringList &lines,
                               uint32_t line_idx,
                               Error &error)
        {
            if (line_idx == UINT32_MAX)
            {
                // Remove the last empty line from "lines" so it doesn't appear
                // in our final expression and return true to indicate we are done
                // getting lines
                lines.PopBack();
                return LineStatus::Done;
            }
            else if (line_idx + 1 == lines.GetSize())
            {
                // The last line was edited, if this line is empty, then we are done
                // getting our multiple lines.
                if (lines[line_idx] == m_end_line)
                    return LineStatus::Done;
            }
            return LineStatus::Success;
        }
    protected:
        const std::string m_end_line;
    };
    
    
    class IOHandlerEditline : public IOHandler
    {
    public:
        IOHandlerEditline (Debugger &debugger,
                           const char *editline_name, // Used for saving history files
                           const char *prompt,
                           bool multi_line,
                           IOHandlerDelegate &delegate);

        IOHandlerEditline (Debugger &debugger,
                           const lldb::StreamFileSP &input_sp,
                           const lldb::StreamFileSP &output_sp,
                           const lldb::StreamFileSP &error_sp,
                           uint32_t flags,
                           const char *editline_name, // Used for saving history files
                           const char *prompt,
                           bool multi_line,
                           IOHandlerDelegate &delegate);
        
        virtual
        ~IOHandlerEditline ();
        
        virtual void
        Run ();
        
        virtual void
        Hide ();

        virtual void
        Refresh ();

        virtual void
        Interrupt ();
        
        virtual void
        GotEOF();
        
        virtual void
        Activate ()
        {
            IOHandler::Activate();
            m_delegate.IOHandlerActivated(*this);
        }

        virtual ConstString
        GetControlSequence (char ch)
        {
            return m_delegate.GetControlSequence (ch);
        }

        virtual const char *
        GetPrompt ();
        
        virtual bool
        SetPrompt (const char *prompt);

        bool
        GetLine (std::string &line);
        
        bool
        GetLines (StringList &lines);

    private:
        static LineStatus
        LineCompletedCallback (Editline *editline,
                               StringList &lines,
                               uint32_t line_idx,
                               Error &error,
                               void *baton);

        static int AutoCompleteCallback (const char *current_line,
                                         const char *cursor,
                                         const char *last_char,
                                         int skip_first_n_matches,
                                         int max_matches,
                                         StringList &matches,
                                         void *baton);

    protected:
        std::unique_ptr<Editline> m_editline_ap;
        IOHandlerDelegate &m_delegate;
        std::string m_prompt;
        bool m_multi_line;        
    };
    
    class IOHandlerConfirm :
        public IOHandlerEditline,
        public IOHandlerDelegate
    {
    public:
        IOHandlerConfirm (Debugger &debugger,
                          const char *prompt,
                          bool default_response);
        
        virtual
        ~IOHandlerConfirm ();
                
        bool
        GetResponse () const
        {
            return m_user_response;
        }
        
        virtual int
        IOHandlerComplete (IOHandler &io_handler,
                           const char *current_line,
                           const char *cursor,
                           const char *last_char,
                           int skip_first_n_matches,
                           int max_matches,
                           StringList &matches);
        
        virtual void
        IOHandlerInputComplete (IOHandler &io_handler, std::string &data);

    protected:
        const bool m_default_response;
        bool m_user_response;
    };

    class IOHandlerCursesGUI :
        public IOHandler
    {
    public:
        IOHandlerCursesGUI (Debugger &debugger);
        
        virtual
        ~IOHandlerCursesGUI ();
        
        virtual void
        Run ();
        
        virtual void
        Hide ();
        
        virtual void
        Refresh ();
        
        virtual void
        Interrupt ();
        
        virtual void
        GotEOF();
        
        virtual void
        Activate ();
        
        virtual void
        Deactivate ();

    protected:
        curses::ApplicationAP m_app_ap;
    };

    class IOHandlerCursesValueObjectList :
        public IOHandler
    {
    public:
        IOHandlerCursesValueObjectList (Debugger &debugger, ValueObjectList &valobj_list);
        
        virtual
        ~IOHandlerCursesValueObjectList ();
        
        virtual void
        Run ();
        
        virtual void
        Hide ();
        
        virtual void
        Refresh ();
        
        virtual void
        Interrupt ();
        
        virtual void
        GotEOF();
    protected:
        ValueObjectList m_valobj_list;
    };

    class IOHandlerStack
    {
    public:
        
        IOHandlerStack () :
            m_stack(),
            m_mutex(Mutex::eMutexTypeRecursive),
            m_top (NULL)
        {
        }
        
        ~IOHandlerStack ()
        {
        }
        
        size_t
        GetSize () const
        {
            Mutex::Locker locker (m_mutex);
            return m_stack.size();
        }
        
        void
        Push (const lldb::IOHandlerSP& sp)
        {
            if (sp)
            {
                Mutex::Locker locker (m_mutex);
                m_stack.push (sp);
                // Set m_top the non-locking IsTop() call
                m_top = sp.get();
            }
        }
        
        bool
        IsEmpty () const
        {
            Mutex::Locker locker (m_mutex);
            return m_stack.empty();
        }
        
        lldb::IOHandlerSP
        Top ()
        {
            lldb::IOHandlerSP sp;
            {
                Mutex::Locker locker (m_mutex);
                if (!m_stack.empty())
                    sp = m_stack.top();
            }
            return sp;
        }
        
        void
        Pop ()
        {
            Mutex::Locker locker (m_mutex);
            if (!m_stack.empty())
                m_stack.pop();
            // Set m_top the non-locking IsTop() call
            if (m_stack.empty())
                m_top = NULL;
            else
                m_top = m_stack.top().get();
        }

        Mutex &
        GetMutex()
        {
            return m_mutex;
        }
      
        bool
        IsTop (const lldb::IOHandlerSP &io_handler_sp) const
        {
            return m_top == io_handler_sp.get();
        }

        ConstString
        GetTopIOHandlerControlSequence (char ch)
        {
            if (m_top)
                return m_top->GetControlSequence(ch);
            return ConstString();
        }

    protected:
        
        std::stack<lldb::IOHandlerSP> m_stack;
        mutable Mutex m_mutex;
        IOHandler *m_top;
        
    private:
        
        DISALLOW_COPY_AND_ASSIGN (IOHandlerStack);
    };

} // namespace lldb_private

#endif // #ifndef liblldb_IOHandler_h_
