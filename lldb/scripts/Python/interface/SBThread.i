//===-- SWIG Interface for SBThread -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a thread of execution. SBProcess contains SBThread(s).

SBThread supports frame iteration. For example (from test/python_api/
lldbutil/iter/TestLLDBIterator.py),

        from lldbutil import print_stacktrace
        stopped_due_to_breakpoint = False
        for thread in process:
            if self.TraceOn():
                print_stacktrace(thread)
            ID = thread.GetThreadID()
            if thread.GetStopReason() == lldb.eStopReasonBreakpoint:
                stopped_due_to_breakpoint = True
            for frame in thread:
                self.assertTrue(frame.GetThread().GetThreadID() == ID)
                if self.TraceOn():
                    print frame

        self.assertTrue(stopped_due_to_breakpoint)

See also SBProcess and SBFrame."
) SBThread;
class SBThread
{
public:
    SBThread ();

    SBThread (const lldb::SBThread &thread);

   ~SBThread();

    bool
    IsValid() const;

    void
    Clear ();

    lldb::StopReason
    GetStopReason();

    %feature("docstring", "
    /// Get the number of words associated with the stop reason.
    /// See also GetStopReasonDataAtIndex().
    ") GetStopReasonDataCount;
    size_t
    GetStopReasonDataCount();

    %feature("docstring", "
    //--------------------------------------------------------------------------
    /// Get information associated with a stop reason.
    ///
    /// Breakpoint stop reasons will have data that consists of pairs of 
    /// breakpoint IDs followed by the breakpoint location IDs (they always come
    /// in pairs).
    ///
    /// Stop Reason              Count Data Type
    /// ======================== ===== =========================================
    /// eStopReasonNone          0
    /// eStopReasonTrace         0
    /// eStopReasonBreakpoint    N     duple: {breakpoint id, location id}
    /// eStopReasonWatchpoint    1     watchpoint id
    /// eStopReasonSignal        1     unix signal number
    /// eStopReasonException     N     exception data
    /// eStopReasonPlanComplete  0
    //--------------------------------------------------------------------------
    ") GetStopReasonDataAtIndex;
    uint64_t
    GetStopReasonDataAtIndex(uint32_t idx);

    %feature("autodoc", "
    Pass only an (int)length and expect to get a Python string describing the
    stop reason.
    ") GetStopDescription;
    size_t
    GetStopDescription (char *dst, size_t dst_len);

    SBValue
    GetStopReturnValue ();

    lldb::tid_t
    GetThreadID () const;

    uint32_t
    GetIndexID () const;

    const char *
    GetName () const;

    const char *
    GetQueueName() const;

    void
    StepOver (lldb::RunMode stop_other_threads = lldb::eOnlyDuringStepping);

    void
    StepInto (lldb::RunMode stop_other_threads = lldb::eOnlyDuringStepping);

    void
    StepOut ();

    void
    StepOutOfFrame (lldb::SBFrame &frame);

    void
    StepInstruction(bool step_over);

    SBError
    StepOverUntil (lldb::SBFrame &frame, 
                   lldb::SBFileSpec &file_spec, 
                   uint32_t line);

    void
    RunToAddress (lldb::addr_t addr);

    %feature("docstring", "
    //--------------------------------------------------------------------------
    /// LLDB currently supports process centric debugging which means when any
    /// thread in a process stops, all other threads are stopped. The Suspend()
    /// call here tells our process to suspend a thread and not let it run when
    /// the other threads in a process are allowed to run. So when 
    /// SBProcess::Continue() is called, any threads that aren't suspended will
    /// be allowed to run. If any of the SBThread functions for stepping are 
    /// called (StepOver, StepInto, StepOut, StepInstruction, RunToAddres), the
    /// thread will now be allowed to run and these funtions will simply return.
    ///
    /// Eventually we plan to add support for thread centric debugging where
    /// each thread is controlled individually and each thread would broadcast
    /// its state, but we haven't implemented this yet.
    /// 
    /// Likewise the SBThread::Resume() call will again allow the thread to run
    /// when the process is continued.
    ///
    /// Suspend() and Resume() functions are not currently reference counted, if
    /// anyone has the need for them to be reference counted, please let us
    /// know.
    //--------------------------------------------------------------------------
    ") Suspend;
    bool
    Suspend();
    
    bool
    Resume ();
    
    bool
    IsSuspended();

    uint32_t
    GetNumFrames ();

    lldb::SBFrame
    GetFrameAtIndex (uint32_t idx);

    lldb::SBFrame
    GetSelectedFrame ();

    lldb::SBFrame
    SetSelectedFrame (uint32_t frame_idx);

    lldb::SBProcess
    GetProcess ();

    bool
    GetDescription (lldb::SBStream &description) const;
    
    %pythoncode %{
        class frames_access(object):
            '''A helper object that will lazily hand out frames for a thread when supplied an index.'''
            def __init__(self, sbthread):
                self.sbthread = sbthread

            def __len__(self):
                if self.sbthread:
                    return int(self.sbthread.GetNumFrames())
                return 0
            
            def __getitem__(self, key):
                if type(key) is int and key < self.sbthread.GetNumFrames():
                    return self.sbthread.GetFrameAtIndex(key)
                return None
        
        def get_frames_access_object(self):
            '''An accessor function that returns a frames_access() object which allows lazy frame access from a lldb.SBThread object.'''
            return self.frames_access (self)

        def get_thread_frames(self):
            '''An accessor function that returns a list() that contains all frames in a lldb.SBThread object.'''
            frames = []
            for frame in self:
                frames.append(frame)
            return frames
        
        __swig_getmethods__["id"] = GetThreadID
        if _newclass: x = property(GetThreadID, None)

        __swig_getmethods__["idx"] = GetIndexID
        if _newclass: x = property(GetIndexID, None)

        __swig_getmethods__["return_value"] = GetStopReturnValue
        if _newclass: x = property(GetStopReturnValue, None)

        __swig_getmethods__["process"] = GetProcess
        if _newclass: x = property(GetProcess, None)

        __swig_getmethods__["num_frames"] = GetNumFrames
        if _newclass: x = property(GetNumFrames, None)

        __swig_getmethods__["frames"] = get_thread_frames
        if _newclass: x = property(get_thread_frames, None)

        __swig_getmethods__["frame"] = get_frames_access_object
        if _newclass: x = property(get_frames_access_object, None)

        __swig_getmethods__["name"] = GetName
        if _newclass: x = property(GetName, None)

        __swig_getmethods__["queue"] = GetQueueName
        if _newclass: x = property(GetQueueName, None)

        __swig_getmethods__["stop_reason"] = GetStopReason
        if _newclass: x = property(GetStopReason, None)

        __swig_getmethods__["is_suspended"] = IsSuspended
        if _newclass: x = property(IsSuspended, None)
    %}

};

} // namespace lldb
