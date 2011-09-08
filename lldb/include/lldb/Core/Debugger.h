//===-- Debugger.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Debugger_h_
#define liblldb_Debugger_h_
#if defined(__cplusplus)


#include <stdint.h>
#include <unistd.h>

#include <stack>

#include "lldb/lldb-public.h"
#include "lldb/Core/Communication.h"
#include "lldb/Core/FormatManager.h"
#include "lldb/Core/InputReaderStack.h"
#include "lldb/Core/Listener.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Core/UserID.h"
#include "lldb/Core/UserSettingsController.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/TargetList.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Debugger Debugger.h "lldb/Core/Debugger.h"
/// @brief A class to manage flag bits.
///
/// Provides a global root objects for the debugger core.
//----------------------------------------------------------------------
    

class DebuggerInstanceSettings : public InstanceSettings
{
public:
    
    DebuggerInstanceSettings (UserSettingsController &owner, bool live_instance = true, const char *name = NULL);

    DebuggerInstanceSettings (const DebuggerInstanceSettings &rhs);

    virtual
    ~DebuggerInstanceSettings ();

    DebuggerInstanceSettings&
    operator= (const DebuggerInstanceSettings &rhs);

    void
    UpdateInstanceSettingsVariable (const ConstString &var_name,
                                    const char *index_value,
                                    const char *value,
                                    const ConstString &instance_name,
                                    const SettingEntry &entry,
                                    VarSetOperationType op,
                                    Error &err,
                                    bool pending);

    bool
    GetInstanceSettingsValue (const SettingEntry &entry,
                              const ConstString &var_name,
                              StringList &value,
                              Error *err);

    uint32_t
    GetTerminalWidth () const
    {
        return m_term_width;
    }

    void
    SetTerminalWidth (uint32_t term_width)
    {
        m_term_width = term_width;
    }
    
    const char *
    GetPrompt() const
    {
        return m_prompt.c_str();
    }

    void
    SetPrompt(const char *p)
    {
        if (p)
            m_prompt.assign (p);
        else
            m_prompt.assign ("(lldb) ");
        BroadcastPromptChange (m_instance_name, m_prompt.c_str());
    }

    const char *
    GetFrameFormat() const
    {
        return m_frame_format.c_str();
    }

    bool
    SetFrameFormat(const char *frame_format)
    {
        if (frame_format && frame_format[0])
        {
            m_frame_format.assign (frame_format);
            return true;
        }
        return false;
    }

    const char *
    GetThreadFormat() const
    {
        return m_thread_format.c_str();
    }

    bool
    SetThreadFormat(const char *thread_format)
    {
        if (thread_format && thread_format[0])
        {
            m_thread_format.assign (thread_format);
            return true;
        }
        return false;
    }

    lldb::ScriptLanguage 
    GetScriptLanguage() const
    {
        return m_script_lang;
    }

    void
    SetScriptLanguage (lldb::ScriptLanguage script_lang)
    {
        m_script_lang = script_lang;
    }

    bool
    GetUseExternalEditor () const
    {
        return m_use_external_editor;
    }

    bool
    SetUseExternalEditor (bool use_external_editor_p)
    {
        bool old_value = m_use_external_editor;
        m_use_external_editor = use_external_editor_p;
        return old_value;
    }
    
    bool 
    GetAutoConfirm () const
    {
        return m_auto_confirm_on;
    }
    
    void
    SetAutoConfirm (bool auto_confirm_on) 
    {
        m_auto_confirm_on = auto_confirm_on;
    }
        
protected:

    void
    CopyInstanceSettings (const lldb::InstanceSettingsSP &new_settings,
                          bool pending);

    bool
    BroadcastPromptChange (const ConstString &instance_name, const char *new_prompt);

    bool
    ValidTermWidthValue (const char *value, Error err);

    const ConstString
    CreateInstanceName ();

    static const ConstString &
    PromptVarName ();

    static const ConstString &
    GetFrameFormatName ();

    static const ConstString &
    GetThreadFormatName ();

    static const ConstString &
    ScriptLangVarName ();
  
    static const ConstString &
    TermWidthVarName ();
  
    static const ConstString &
    UseExternalEditorVarName ();
    
    static const ConstString &
    AutoConfirmName ();

private:

    uint32_t m_term_width;
    std::string m_prompt;
    std::string m_frame_format;
    std::string m_thread_format;
    lldb::ScriptLanguage m_script_lang;
    bool m_use_external_editor;
    bool m_auto_confirm_on;
};



class Debugger :
    public UserID,
    public DebuggerInstanceSettings
{
public:

    class SettingsController : public UserSettingsController
    {
    public:

        SettingsController ();

        virtual
        ~SettingsController ();

        static SettingEntry global_settings_table[];
        static SettingEntry instance_settings_table[];

    protected:

        lldb::InstanceSettingsSP
        CreateInstanceSettings (const char *instance_name);

    private:

        // Class-wide settings.

        DISALLOW_COPY_AND_ASSIGN (SettingsController);
    };

    static lldb::UserSettingsControllerSP &
    GetSettingsController ();

    static lldb::DebuggerSP
    CreateInstance ();

    static lldb::TargetSP
    FindTargetWithProcessID (lldb::pid_t pid);

    static void
    Initialize ();
    
    static void 
    Terminate ();
    
    static void
    SettingsInitialize ();
    
    static void
    SettingsTerminate ();
    
    static void
    Destroy (lldb::DebuggerSP &debugger_sp);

    ~Debugger ();

    lldb::DebuggerSP
    GetSP ();

    bool
    GetAsyncExecution ();

    void
    SetAsyncExecution (bool async);

    File &
    GetInputFile ()
    {
        return m_input_file.GetFile();
    }

    File &
    GetOutputFile ()
    {
        return m_output_file.GetFile();
    }

    File &
    GetErrorFile ()
    {
        return m_error_file.GetFile();
    }
    
    void
    SetInputFileHandle (FILE *fh, bool tranfer_ownership);

    void
    SetOutputFileHandle (FILE *fh, bool tranfer_ownership);

    void
    SetErrorFileHandle (FILE *fh, bool tranfer_ownership);

    Stream&
    GetOutputStream ()
    {
        return m_output_file;
    }

    Stream&
    GetErrorStream ()
    {
        return m_error_file;
    }

    lldb::StreamSP
    GetAsyncOutputStream ();
    
    lldb::StreamSP
    GetAsyncErrorStream ();
    
    CommandInterpreter &
    GetCommandInterpreter ()
    {
        assert (m_command_interpreter_ap.get());
        return *m_command_interpreter_ap;
    }

    Listener &
    GetListener ()
    {
        return m_listener;
    }

    SourceManager &
    GetSourceManager ()
    {
        lldb::TargetSP selected_target = GetSelectedTarget();
        if (selected_target)
            return selected_target->GetSourceManager();
        else
            return m_source_manager;
    }

    lldb::TargetSP
    GetSelectedTarget ()
    {
        return m_target_list.GetSelectedTarget ();
    }

    ExecutionContext
    GetSelectedExecutionContext();
    //------------------------------------------------------------------
    /// Get accessor for the target list.
    ///
    /// The target list is part of the global debugger object. This
    /// the single debugger shared instance to control where targets
    /// get created and to allow for tracking and searching for targets
    /// based on certain criteria.
    ///
    /// @return
    ///     A global shared target list.
    //------------------------------------------------------------------
    TargetList &
    GetTargetList ()
    {
        return m_target_list;
    }

    PlatformList &
    GetPlatformList ()
    {
        return m_platform_list;
    }

    void
    DispatchInputInterrupt ();

    void
    DispatchInputEndOfFile ();

    void
    DispatchInput (const char *bytes, size_t bytes_len);

    void
    WriteToDefaultReader (const char *bytes, size_t bytes_len);

    void
    PushInputReader (const lldb::InputReaderSP& reader_sp);

    bool
    PopInputReader (const lldb::InputReaderSP& reader_sp);

    void
    NotifyTopInputReader (lldb::InputReaderAction notification);

    bool
    InputReaderIsTopReader (const lldb::InputReaderSP& reader_sp);
    
    static lldb::DebuggerSP
    FindDebuggerWithID (lldb::user_id_t id);
    
    static lldb::DebuggerSP
    FindDebuggerWithInstanceName (const ConstString &instance_name);

    static bool
    FormatPrompt (const char *format,
                  const SymbolContext *sc,
                  const ExecutionContext *exe_ctx,
                  const Address *addr,
                  Stream &s,
                  const char **end,
                  ValueObject* valobj = NULL);


    void
    CleanUpInputReaders ();

    static int
    TestDebuggerRefCount ();

    bool
    GetCloseInputOnEOF () const;
    
    void
    SetCloseInputOnEOF (bool b);

protected:

    static void
    DispatchInputCallback (void *baton, const void *bytes, size_t bytes_len);

    lldb::InputReaderSP
    GetCurrentInputReader ();
    
    void
    ActivateInputReader (const lldb::InputReaderSP &reader_sp);

    bool
    CheckIfTopInputReaderIsDone ();
    
    void
    DisconnectInput()
    {
        m_input_comm.Clear ();
    }

    Communication m_input_comm;
    StreamFile m_input_file;
    StreamFile m_output_file;
    StreamFile m_error_file;
    TargetList m_target_list;
    PlatformList m_platform_list;
    Listener m_listener;
    SourceManager m_source_manager;    // This is a scratch source manager that we return if we have no targets.
    std::auto_ptr<CommandInterpreter> m_command_interpreter_ap;

    InputReaderStack m_input_reader_stack;
    std::string m_input_reader_data;

private:

    // Use Debugger::CreateInstance() to get a shared pointer to a new
    // debugger object
    Debugger ();

    DISALLOW_COPY_AND_ASSIGN (Debugger);
    
};

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // liblldb_Debugger_h_
