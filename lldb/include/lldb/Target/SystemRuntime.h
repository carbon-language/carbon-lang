//===-- SystemRuntime.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SystemRuntime_h_
#define liblldb_SystemRuntime_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include <vector>

#include "lldb/lldb-public.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/lldb-private.h"


namespace lldb_private {

//----------------------------------------------------------------------
/// @class SystemRuntime SystemRuntime.h "lldb/Target/SystemRuntime.h"
/// @brief A plug-in interface definition class for system runtimes.
///
/// The system runtime plugins can collect information from the system
/// libraries during a Process' lifetime and provide information about
/// how objects/threads were originated.
///
/// For instance, a system runtime plugin use a breakpoint when threads
/// are created to record the backtrace of where that thread was created.
/// Later, when backtracing the created thread, it could extend the backtrace
/// to show where it was originally created from.  
///
/// The plugin will insert its own breakpoint when Created and start collecting
/// information.  Later when it comes time to augment a Thread, it can be
/// asked to provide that information.
///
//----------------------------------------------------------------------

class SystemRuntime :
    public PluginInterface
{
public:
    //------------------------------------------------------------------
    /// Find a system runtime plugin for a given process.
    ///
    /// Scans the installed SystemRuntime plugins and tries to find
    /// an instance that can be used to track image changes in \a
    /// process.
    ///
    /// @param[in] process
    ///     The process for which to try and locate a system runtime
    ///     plugin instance.
    //------------------------------------------------------------------
    static SystemRuntime* 
    FindPlugin (Process *process);

    //------------------------------------------------------------------
    /// Construct with a process.
    // -----------------------------------------------------------------
    SystemRuntime(lldb_private::Process *process);

    //------------------------------------------------------------------
    /// Destructor.
    ///
    /// The destructor is virtual since this class is designed to be
    /// inherited by the plug-in instance.
    //------------------------------------------------------------------
    virtual
    ~SystemRuntime();

    //------------------------------------------------------------------
    /// Called after attaching to a process.
    ///
    /// Allow the SystemRuntime plugin to execute some code after attaching
    /// to a process. 
    //------------------------------------------------------------------
    virtual void
    DidAttach ();

    //------------------------------------------------------------------
    /// Called after launching a process.
    ///
    /// Allow the SystemRuntime plugin to execute some code after launching
    /// a process. 
    //------------------------------------------------------------------
    virtual void
    DidLaunch();

    //------------------------------------------------------------------
    /// Called when modules have been loaded in the process.
    ///
    /// Allow the SystemRuntime plugin to enable logging features in the
    /// system runtime libraries.
    //------------------------------------------------------------------
    virtual void
    ModulesDidLoad(lldb_private::ModuleList &module_list);


    //------------------------------------------------------------------
    /// Return a list of thread origin extended backtraces that may 
    /// be available.
    ///
    /// A System Runtime may be able to provide a backtrace of when this
    /// thread was originally created.  Furthermore, it may be able to 
    /// provide that extended backtrace for different styles of creation.
    /// On a system with both pthreads and libdispatch, aka Grand Central 
    /// Dispatch, queues, the system runtime may be able to provide the
    /// pthread creation of the thread and it may also be able to provide
    /// the backtrace of when this GCD queue work block was enqueued.
    /// The caller may request these different origins by name.
    ///
    /// The names will be provided in the order that they are most likely
    /// to be requested.  For instance, a most natural order may be to 
    /// request the GCD libdispatch queue origin.  If there is none, then 
    /// request the pthread origin.
    ///
    /// @return
    ///   A vector of ConstStrings with names like "pthread" or "libdispatch".
    ///   An empty vector may be returned if no thread origin extended 
    ///   backtrace capabilities are available.
    //------------------------------------------------------------------
    virtual const std::vector<ConstString> &
    GetExtendedBacktraceTypes ();

    //------------------------------------------------------------------
    /// Return a Thread which shows the origin of this thread's creation.
    ///
    /// This likely returns a HistoryThread which shows how thread was
    /// originally created (e.g. "pthread" type), or how the work that
    /// is currently executing on it was originally enqueued (e.g. 
    /// "libdispatch" type).
    ///
    /// There may be a chain of thread-origins; it may be informative to
    /// the end user to query the returned ThreadSP for its origins as 
    /// well.
    ///
    /// @param [in] thread
    ///   The thread to examine.
    ///
    /// @param [in] type
    ///   The type of thread origin being requested.  The types supported
    ///   are returned from SystemRuntime::GetExtendedBacktraceTypes.
    ///
    /// @return
    ///   A ThreadSP which will have a StackList of frames.  This Thread will
    ///   not appear in the Process' list of current threads.  Normal thread 
    ///   operations like stepping will not be available.  This is a historical
    ///   view thread and may be only useful for showing a backtrace.
    ///
    ///   An empty ThreadSP will be returned if no thread origin is available.
    //------------------------------------------------------------------
    virtual lldb::ThreadSP
    GetExtendedBacktraceThread (lldb::ThreadSP thread, ConstString type);

protected:
    //------------------------------------------------------------------
    // Member variables.
    //------------------------------------------------------------------
    Process *m_process;

    std::vector<ConstString> m_types;

private:
    DISALLOW_COPY_AND_ASSIGN (SystemRuntime);
};

} // namespace lldb_private

#endif  // liblldb_SystemRuntime_h_
