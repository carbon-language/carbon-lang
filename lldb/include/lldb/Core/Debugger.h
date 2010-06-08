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

#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Core/Communication.h"
#include "lldb/Core/Listener.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Target/TargetList.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Debugger Debugger.h "lldb/Core/Debugger.h"
/// @brief A class to manage flag bits.
///
/// Provides a global root objects for the debugger core.
//----------------------------------------------------------------------
class Debugger
{
public:

    static void
    Initialize ();
    
    static void 
    Terminate ();

    static Debugger &
    GetSharedInstance ();
    
    ~Debugger ();

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
    GetCurrentTarget ();

    ExecutionContext
    GetCurrentExecutionContext();
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

protected:

    static void
    DispatchInputCallback (void *baton, const void *bytes, size_t bytes_len);

    void
    ActivateInputReader (const lldb::InputReaderSP &reader_sp);

    bool
    CheckIfTopInputReaderIsDone ();
    
    void
    DisconnectInput();

    bool m_async_execution;
    Communication m_input_comm;
    StreamFile m_input_file;
    StreamFile m_output_file;
    StreamFile m_error_file;
    TargetList m_target_list;
    Listener m_listener;
    SourceManager m_source_manager;
    CommandInterpreter m_command_interpreter;

    std::stack<lldb::InputReaderSP> m_input_readers;
    std::string m_input_reader_data;
    
    typedef std::tr1::shared_ptr<Debugger> DebuggerSP;

    static DebuggerSP &
    GetDebuggerSP();
    
    static int g_shared_debugger_refcount;
    static bool g_in_terminate;

private:
    Debugger ();    // Access the single global instance of this class using Debugger::GetSharedInstance();

    DISALLOW_COPY_AND_ASSIGN (Debugger);
};

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // liblldb_Debugger_h_
