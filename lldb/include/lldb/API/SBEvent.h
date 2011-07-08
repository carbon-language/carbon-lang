//===-- SBEvent.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBEvent_h_
#define LLDB_SBEvent_h_

#include "lldb/API/SBDefines.h"

#include <stdio.h>
#include <vector>


namespace lldb {

class SBBroadcaster;

#ifdef SWIG
%feature("docstring",
"API clients can register to receive events.

For example, check out the following output:

Try wait for event...
Event description: 0x103d0bb70 Event: broadcaster = 0x1009c8410, type = 0x00000001, data = { process = 0x1009c8400 (pid = 21528), state = running}
Event data flavor: Process::ProcessEventData
Process state: running

Try wait for event...
Event description: 0x103a700a0 Event: broadcaster = 0x1009c8410, type = 0x00000001, data = { process = 0x1009c8400 (pid = 21528), state = stopped}
Event data flavor: Process::ProcessEventData
Process state: stopped

Try wait for event...
Event description: 0x103d0d4a0 Event: broadcaster = 0x1009c8410, type = 0x00000001, data = { process = 0x1009c8400 (pid = 21528), state = exited}
Event data flavor: Process::ProcessEventData
Process state: exited

Try wait for event...
timeout occurred waiting for event...

from test/python_api/event/TestEventspy:

    def do_listen_for_and_print_event(self):
        '''Create a listener and use SBEvent API to print the events received.'''
        exe = os.path.join(os.getcwd(), 'a.out')

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        # Get a handle on the process's broadcaster.
        broadcaster = process.GetBroadcaster()

        # Create an empty event object.
        event = lldb.SBEvent()

        # Create a listener object and register with the broadcaster.
        listener = lldb.SBListener('my listener')
        rc = broadcaster.AddListener(listener, lldb.SBProcess.eBroadcastBitStateChanged)
        self.assertTrue(rc, 'AddListener successfully retruns')

        traceOn = self.TraceOn()
        if traceOn:
            lldbutil.print_stacktraces(process)

        # Create MyListeningThread class to wait for any kind of event.
        import threading
        class MyListeningThread(threading.Thread):
            def run(self):
                count = 0
                # Let's only try at most 4 times to retrieve any kind of event.
                # After that, the thread exits.
                while not count > 3:
                    if traceOn:
                        print 'Try wait for event...'
                    if listener.WaitForEventForBroadcasterWithType(5,
                                                                   broadcaster,
                                                                   lldb.SBProcess.eBroadcastBitStateChanged,
                                                                   event):
                        if traceOn:
                            desc = lldbutil.get_description(event)
                            print 'Event description:', desc
                            print 'Event data flavor:', event.GetDataFlavor()
                            print 'Process state:', lldbutil.state_type_to_str(process.GetState())
                            print
                    else:
                        if traceOn:
                            print 'timeout occurred waiting for event...'
                    count = count + 1
                return

        # Let's start the listening thread to retrieve the events.
        my_thread = MyListeningThread()
        my_thread.start()

        # Use Python API to continue the process.  The listening thread should be
        # able to receive the state changed events.
        process.Continue()

        # Use Python API to kill the process.  The listening thread should be
        # able to receive the state changed event, too.
        process.Kill()

        # Wait until the 'MyListeningThread' terminates.
        my_thread.join()
"
         ) SBEvent;
#endif
class SBEvent
{
#ifdef SWIG
    %feature("autodoc", "1");
#endif
public:
    SBEvent();

    SBEvent (const lldb::SBEvent &rhs);
    
    // Make an event that contains a C string.
#ifdef SWIG
    %feature("autodoc", "__init__(self, int type, str data) -> SBEvent") SBEvent;
#endif
    SBEvent (uint32_t event, const char *cstr, uint32_t cstr_len);

    ~SBEvent();

#ifndef SWIG
    const SBEvent &
    operator = (const lldb::SBEvent &rhs);
#endif

    bool
    IsValid() const;

    const char *
    GetDataFlavor ();

    uint32_t
    GetType () const;

    lldb::SBBroadcaster
    GetBroadcaster () const;

#ifndef SWIG
    bool
    BroadcasterMatchesPtr (const lldb::SBBroadcaster *broadcaster);
#endif

    bool
    BroadcasterMatchesRef (const lldb::SBBroadcaster &broadcaster);

    void
    Clear();

    static const char *
    GetCStringFromEvent (const lldb::SBEvent &event);

#ifndef SWIG
    bool
    GetDescription (lldb::SBStream &description);
#endif

    bool
    GetDescription (lldb::SBStream &description) const;

protected:
    friend class SBListener;
    friend class SBBroadcaster;
    friend class SBBreakpoint;
    friend class SBDebugger;
    friend class SBProcess;

    SBEvent (lldb::EventSP &event_sp);

#ifndef SWIG

    lldb::EventSP &
    GetSP () const;

    void
    reset (lldb::EventSP &event_sp);

    void
    reset (lldb_private::Event* event);

    lldb_private::Event *
    get () const;

#endif

private:

    mutable lldb::EventSP m_event_sp;
    mutable lldb_private::Event *m_opaque_ptr;
};

} // namespace lldb

#endif  // LLDB_SBEvent_h_
