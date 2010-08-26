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

#include "lldb/Core/Communication.h"
#include "lldb/Core/Listener.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Core/UserID.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/TargetList.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Debugger Debugger.h "lldb/Core/Debugger.h"
/// @brief A class to manage flag bits.
///
/// Provides a global root objects for the debugger core.
//----------------------------------------------------------------------
class Debugger :
    public UserID
{
public:

    static lldb::DebuggerSP
    CreateInstance ();

    static lldb::TargetSP
    FindTargetWithProcessID (lldb::pid_t pid);

    static void
    Initialize ();
    
    static void 
    Terminate ();

    ~Debugger ();

    lldb::DebuggerSP
    GetSP ();

    bool
    GetAsyncExecution ();

    void
    SetAsyncExecution (bool async);

    void
    SetInputFileHandle (FILE *fh, bool tranfer_ownership);

    void
    SetOutputFileHandle (FILE *fh, bool tranfer_ownership);

    void
    SetErrorFileHandle (FILE *fh, bool tranfer_ownership);

    FILE *
    GetInputFileHandle ();

    FILE *
    GetOutputFileHandle ();

    FILE *
    GetErrorFileHandle ();

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

    CommandInterpreter &
    GetCommandInterpreter ();

    Listener &
    GetListener ();

    SourceManager &
    GetSourceManager ();

    lldb::TargetSP
    GetSelectedTarget ();

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
    TargetList&
    GetTargetList ();

    void
    DispatchInput (const char *bytes, size_t bytes_len);

    void
    WriteToDefaultReader (const char *bytes, size_t bytes_len);

    void
    PushInputReader (const lldb::InputReaderSP& reader_sp);

    bool
    PopInputReader (const lldb::InputReaderSP& reader_sp);

    ExecutionContext &
    GetExecutionContext()
    {
        return m_exe_ctx;
    }


    void
    UpdateExecutionContext (ExecutionContext *override_context);

    static lldb::DebuggerSP
    FindDebuggerWithID (lldb::user_id_t id);

protected:

    static void
    DispatchInputCallback (void *baton, const void *bytes, size_t bytes_len);

    void
    ActivateInputReader (const lldb::InputReaderSP &reader_sp);

    bool
    CheckIfTopInputReaderIsDone ();
    
    void
    DisconnectInput();

    Communication m_input_comm;
    StreamFile m_input_file;
    StreamFile m_output_file;
    StreamFile m_error_file;
    TargetList m_target_list;
    Listener m_listener;
    SourceManager m_source_manager;
    std::auto_ptr<CommandInterpreter> m_command_interpreter_ap;
    ExecutionContext m_exe_ctx;

    std::stack<lldb::InputReaderSP> m_input_readers;
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
