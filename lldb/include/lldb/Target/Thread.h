//===-- Thread.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Thread_h_
#define liblldb_Thread_h_

#include "lldb/lldb-private.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Core/UserID.h"
#include "lldb/Core/UserSettingsController.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/StackFrameList.h"

#define LLDB_THREAD_MAX_STOP_EXC_DATA 8

namespace lldb_private {

class ThreadInstanceSettings : public InstanceSettings
{
public:

    ThreadInstanceSettings (UserSettingsController &owner, bool live_instance = true, const char *name = NULL);
  
    ThreadInstanceSettings (const ThreadInstanceSettings &rhs);

    virtual
    ~ThreadInstanceSettings ();
  
    ThreadInstanceSettings&
    operator= (const ThreadInstanceSettings &rhs);
  

    void
    UpdateInstanceSettingsVariable (const ConstString &var_name,
                                    const char *index_value,
                                    const char *value,
                                    const ConstString &instance_name,
                                    const SettingEntry &entry,
                                    lldb::VarSetOperationType op,
                                    Error &err,
                                    bool pending);

    bool
    GetInstanceSettingsValue (const SettingEntry &entry,
                              const ConstString &var_name,
                              StringList &value,
                              Error *err);

    RegularExpression *
    GetSymbolsToAvoidRegexp()
    {
        return m_avoid_regexp_ap.get();
    }

    static const ConstString &
    StepAvoidRegexpVarName ();
    
    bool
    GetTraceEnabledState()
    {
        return m_trace_enabled;
    }
    static const ConstString &
    GetTraceThreadVarName ();

protected:

    void
    CopyInstanceSettings (const lldb::InstanceSettingsSP &new_settings,
                          bool pending);

    const ConstString
    CreateInstanceName ();

private:

    std::auto_ptr<RegularExpression> m_avoid_regexp_ap;
    bool m_trace_enabled;
};

class Thread :
    public UserID,
    public ExecutionContextScope,
    public ThreadInstanceSettings
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

    class RegisterCheckpoint
    {
    public:

        RegisterCheckpoint() :
            m_stack_id (),
            m_data_sp ()
        {
        }

        RegisterCheckpoint (const StackID &stack_id) :
            m_stack_id (stack_id),
            m_data_sp ()
        {
        }

        ~RegisterCheckpoint()
        {
        }

        const StackID &
        GetStackID()
        {
            return m_stack_id;
        }

        void
        SetStackID (const StackID &stack_id)
        {
            m_stack_id = stack_id;
        }

        lldb::DataBufferSP &
        GetData()
        {
            return m_data_sp;
        }

        const lldb::DataBufferSP &
        GetData() const
        {
            return m_data_sp;
        }

    protected:
        StackID m_stack_id;
        lldb::DataBufferSP m_data_sp;
    };

    void
    UpdateInstanceName ();

    static void
    Initialize ();

    static void
    Terminate ();

    static lldb::UserSettingsControllerSP &
    GetSettingsController ();

    Thread (Process &process, lldb::tid_t tid);
    virtual ~Thread();

    Process &
    GetProcess() { return m_process; }

    const Process &
    GetProcess() const { return m_process; }

    int
    GetResumeSignal () const;

    void
    SetResumeSignal (int signal);

    lldb::StateType
    GetState() const;

    lldb::ThreadSP
    GetSP ();

    void
    SetState (lldb::StateType state);

    lldb::StateType
    GetResumeState () const;

    void
    SetResumeState (lldb::StateType state);

    // This function is called on all the threads before "WillResume" in case
    // a thread needs to change its state before the ThreadList polls all the
    // threads to figure out which ones actually will get to run and how.
    void
    SetupForResume ();
    
    // Override this to do platform specific tasks before resume, but always
    // call the Thread::WillResume at the end of your work.

    virtual bool
    WillResume (lldb::StateType resume_state);

    // This clears generic thread state after a resume.  If you subclass this,
    // be sure to call it.
    virtual void
    DidResume ();

    virtual void
    RefreshStateAfterStop() = 0;

    void
    WillStop ();

    bool
    ShouldStop (Event *event_ptr);

    lldb::Vote
    ShouldReportStop (Event *event_ptr);

    lldb::Vote
    ShouldReportRun (Event *event_ptr);
    
    // Return whether this thread matches the specification in ThreadSpec.  This is a virtual
    // method because at some point we may extend the thread spec with a platform specific
    // dictionary of attributes, which then only the platform specific Thread implementation
    // would know how to match.  For now, this just calls through to the ThreadSpec's 
    // ThreadPassesBasicTests method.
    virtual bool
    MatchesSpec (const ThreadSpec *spec);

    lldb::StopInfoSP
    GetStopInfo ();

    bool
    ThreadStoppedForAReason ();

    static const char *
    RunModeAsCString (lldb::RunMode mode);

    static const char *
    StopReasonAsCString (lldb::StopReason reason);

    virtual const char *
    GetInfo () = 0;

    virtual const char *
    GetName ()
    {
        return NULL;
    }

    virtual const char *
    GetQueueName ()
    {
        return NULL;
    }

    virtual uint32_t
    GetStackFrameCount();

    virtual lldb::StackFrameSP
    GetStackFrameAtIndex (uint32_t idx);
    
    virtual lldb::StackFrameSP
    GetFrameWithConcreteFrameIndex (uint32_t unwind_idx);

    uint32_t
    GetSelectedFrameIndex ();

    lldb::StackFrameSP
    GetSelectedFrame ();

    uint32_t
    SetSelectedFrame (lldb_private::StackFrame *frame);

    void
    SetSelectedFrameByIndex (uint32_t frame_idx);

    virtual lldb::RegisterContextSP
    GetRegisterContext () = 0;

    virtual bool
    SaveFrameZeroState (RegisterCheckpoint &checkpoint) = 0;

    virtual bool
    RestoreSaveFrameZero (const RegisterCheckpoint &checkpoint) = 0;

    virtual lldb::RegisterContextSP
    CreateRegisterContextForFrame (StackFrame *frame) = 0;
    
    virtual void
    ClearStackFrames ();

    void
    DumpUsingSettingsFormat (Stream &strm, uint32_t frame_idx);

    //------------------------------------------------------------------
    // Thread Plan Providers:
    // This section provides the basic thread plans that the Process control
    // machinery uses to run the target.  ThreadPlan.h provides more details on
    // how this mechanism works.
    // The thread provides accessors to a set of plans that perform basic operations.
    // The idea is that particular Platform plugins can override these methods to
    // provide the implementation of these basic operations appropriate to their
    // environment.
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    /// Queues the base plan for a thread.
    /// The version returned by Process does some things that are useful,
    /// like handle breakpoints and signals, so if you return a plugin specific
    /// one you probably want to call through to the Process one for anything
    /// your plugin doesn't explicitly handle.
    ///
    /// @param[in] abort_other_plans
    ///    \b true if we discard the currently queued plans and replace them with this one.
    ///    Otherwise this plan will go on the end of the plan stack.
    ///
    /// @return
    ///     A pointer to the newly queued thread plan, or NULL if the plan could not be queued.
    //------------------------------------------------------------------
    virtual ThreadPlan *
    QueueFundamentalPlan (bool abort_other_plans);

    //------------------------------------------------------------------
    /// Queues the plan used to step over a breakpoint at the current PC of \a thread.
    /// The default version returned by Process handles trap based breakpoints, and
    /// will disable the breakpoint, single step over it, then re-enable it.
    ///
    /// @param[in] abort_other_plans
    ///    \b true if we discard the currently queued plans and replace them with this one.
    ///    Otherwise this plan will go on the end of the plan stack.
    ///
    /// @return
    ///     A pointer to the newly queued thread plan, or NULL if the plan could not be queued.
    //------------------------------------------------------------------
    virtual ThreadPlan *
    QueueThreadPlanForStepOverBreakpointPlan (bool abort_other_plans);

    //------------------------------------------------------------------
    /// Queues the plan used to step one instruction from the current PC of \a thread.
    ///
    /// @param[in] step_over
    ///    \b true if we step over calls to functions, false if we step in.
    ///
    /// @param[in] abort_other_plans
    ///    \b true if we discard the currently queued plans and replace them with this one.
    ///    Otherwise this plan will go on the end of the plan stack.
    ///
    /// @param[in] stop_other_threads
    ///    \b true if we will stop other threads while we single step this one.
    ///
    /// @return
    ///     A pointer to the newly queued thread plan, or NULL if the plan could not be queued.
    //------------------------------------------------------------------
    virtual ThreadPlan *
    QueueThreadPlanForStepSingleInstruction (bool step_over,
                                             bool abort_other_plans,
                                             bool stop_other_threads);

    //------------------------------------------------------------------
    /// Queues the plan used to step through an address range, stepping into or over
    /// function calls depending on the value of StepType.
    ///
    /// @param[in] abort_other_plans
    ///    \b true if we discard the currently queued plans and replace them with this one.
    ///    Otherwise this plan will go on the end of the plan stack.
    ///
    /// @param[in] type
    ///    Type of step to do, only eStepTypeInto and eStepTypeOver are supported by this plan.
    ///
    /// @param[in] range
    ///    The address range to step through.
    ///
    /// @param[in] addr_context
    ///    When dealing with stepping through inlined functions the current PC is not enough information to know
    ///    what "step" means.  For instance a series of nested inline functions might start at the same address.
    //     The \a addr_context provides the current symbol context the step
    ///    is supposed to be out of.
    //   FIXME: Currently unused.
    ///
    /// @param[in] stop_other_threads
    ///    \b true if we will stop other threads while we single step this one.
    ///
    /// @return
    ///     A pointer to the newly queued thread plan, or NULL if the plan could not be queued.
    //------------------------------------------------------------------
    virtual ThreadPlan *
    QueueThreadPlanForStepRange (bool abort_other_plans,
                                 lldb::StepType type,
                                 const AddressRange &range,
                                 const SymbolContext &addr_context,
                                 lldb::RunMode stop_other_threads,
                                 bool avoid_code_without_debug_info);

    //------------------------------------------------------------------
    /// Queue the plan used to step out of the function at the current PC of
    /// \a thread.
    ///
    /// @param[in] abort_other_plans
    ///    \b true if we discard the currently queued plans and replace them with this one.
    ///    Otherwise this plan will go on the end of the plan stack.
    ///
    /// @param[in] addr_context
    ///    When dealing with stepping through inlined functions the current PC is not enough information to know
    ///    what "step" means.  For instance a series of nested inline functions might start at the same address.
    //     The \a addr_context provides the current symbol context the step
    ///    is supposed to be out of.
    //   FIXME: Currently unused.
    ///
    /// @param[in] first_insn
    ///     \b true if this is the first instruction of a function.
    ///
    /// @param[in] stop_other_threads
    ///    \b true if we will stop other threads while we single step this one.
    ///
    /// @param[in] stop_vote
    /// @param[in] run_vote
    ///    See standard meanings for the stop & run votes in ThreadPlan.h.
    ///
    /// @return
    ///     A pointer to the newly queued thread plan, or NULL if the plan could not be queued.
    //------------------------------------------------------------------
    virtual ThreadPlan *
    QueueThreadPlanForStepOut (bool abort_other_plans,
                               SymbolContext *addr_context,
                               bool first_insn,
                               bool stop_other_threads,
                               lldb::Vote stop_vote = lldb::eVoteYes,
                               lldb::Vote run_vote = lldb::eVoteNoOpinion);

    //------------------------------------------------------------------
    /// Gets the plan used to step through the code that steps from a function
    /// call site at the current PC into the actual function call.
    ///
    /// @param[in] abort_other_plans
    ///    \b true if we discard the currently queued plans and replace them with this one.
    ///    Otherwise this plan will go on the end of the plan stack.
    ///
    /// @param[in] stop_other_threads
    ///    \b true if we will stop other threads while we single step this one.
    ///
    /// @return
    ///     A pointer to the newly queued thread plan, or NULL if the plan could not be queued.
    //------------------------------------------------------------------
    virtual ThreadPlan *
    QueueThreadPlanForStepThrough (bool abort_other_plans,
                                   bool stop_other_threads);

    //------------------------------------------------------------------
    /// Gets the plan used to continue from the current PC.
    /// This is a simple plan, mostly useful as a backstop when you are continuing
    /// for some particular purpose.
    ///
    /// @param[in] abort_other_plans
    ///    \b true if we discard the currently queued plans and replace them with this one.
    ///    Otherwise this plan will go on the end of the plan stack.
    ///
    /// @param[in] target_addr
    ///    The address to which we're running.
    ///
    /// @param[in] stop_other_threads
    ///    \b true if we will stop other threads while we single step this one.
    ///
    /// @return
    ///     A pointer to the newly queued thread plan, or NULL if the plan could not be queued.
    //------------------------------------------------------------------
    virtual ThreadPlan *
    QueueThreadPlanForRunToAddress (bool abort_other_plans,
                                    Address &target_addr,
                                    bool stop_other_threads);

    virtual ThreadPlan *
    QueueThreadPlanForStepUntil (bool abort_other_plans,
                               lldb::addr_t *address_list,
                               size_t num_addresses,
                               bool stop_others);

    virtual ThreadPlan *
    QueueThreadPlanForCallFunction (bool abort_other_plans,
                                    Address& function,
                                    lldb::addr_t arg,
                                    bool stop_other_threads,
                                    bool discard_on_error = false);
                                            
    //------------------------------------------------------------------
    // Thread Plan accessors:
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    /// Gets the plan which will execute next on the plan stack.
    ///
    /// @return
    ///     A pointer to the next executed plan.
    //------------------------------------------------------------------
    ThreadPlan *
    GetCurrentPlan ();

    //------------------------------------------------------------------
    /// Gets the inner-most plan that was popped off the plan stack in the
    /// most recent stop.  Useful for printing the stop reason accurately.
    ///
    /// @return
    ///     A pointer to the last completed plan.
    //------------------------------------------------------------------
    lldb::ThreadPlanSP
    GetCompletedPlan ();

    //------------------------------------------------------------------
    ///  Checks whether the given plan is in the completed plans for this
    ///  stop.
    ///
    /// @param[in] plan
    ///     Pointer to the plan you're checking.
    ///
    /// @return
    ///     Returns true if the input plan is in the completed plan stack,
    ///     false otherwise.
    //------------------------------------------------------------------
    bool
    IsThreadPlanDone (ThreadPlan *plan);

    //------------------------------------------------------------------
    ///  Checks whether the given plan is in the discarded plans for this
    ///  stop.
    ///
    /// @param[in] plan
    ///     Pointer to the plan you're checking.
    ///
    /// @return
    ///     Returns true if the input plan is in the discarded plan stack,
    ///     false otherwise.
    //------------------------------------------------------------------
    bool
    WasThreadPlanDiscarded (ThreadPlan *plan);

    //------------------------------------------------------------------
    /// Queues a generic thread plan.
    ///
    /// @param[in] plan_sp
    ///    The plan to queue.
    ///
    /// @param[in] abort_other_plans
    ///    \b true if we discard the currently queued plans and replace them with this one.
    ///    Otherwise this plan will go on the end of the plan stack.
    ///
    /// @return
    ///     A pointer to the last completed plan.
    //------------------------------------------------------------------
    void
    QueueThreadPlan (lldb::ThreadPlanSP &plan_sp, bool abort_other_plans);


    //------------------------------------------------------------------
    /// Discards the plans queued on the plan stack of the current thread.  This is
    /// arbitrated by the "Master" ThreadPlans, using the "OkayToDiscard" call.
    //  But if \a force is true, all thread plans are discarded.
    //------------------------------------------------------------------
    void
    DiscardThreadPlans (bool force);

    //------------------------------------------------------------------
    /// Discards the plans queued on the plan stack of the current thread up to and
    /// including up_to_plan_sp.
    //
    // @param[in] up_to_plan_sp
    //   Discard all plans up to and including this one.
    //------------------------------------------------------------------
    void
    DiscardThreadPlansUpToPlan (lldb::ThreadPlanSP &up_to_plan_sp);

    //------------------------------------------------------------------
    /// Prints the current plan stack.
    ///
    /// @param[in] s
    ///    The stream to which to dump the plan stack info.
    ///
    //------------------------------------------------------------------
    void
    DumpThreadPlans (Stream *s) const;
    
    void
    EnableTracer (bool value, bool single_step);
    
    void
    SetTracer (lldb::ThreadPlanTracerSP &tracer_sp);
    
    //------------------------------------------------------------------
    /// The regular expression returned determines symbols that this
    /// thread won't stop in during "step-in" operations.
    ///
    /// @return
    ///    A pointer to a regular expression to compare against symbols,
    ///    or NULL if all symbols are allowed.
    ///
    //------------------------------------------------------------------
    RegularExpression *
    GetSymbolsToAvoidRegexp()
    {
        return ThreadInstanceSettings::GetSymbolsToAvoidRegexp();
    }

    // Get the thread index ID. The index ID that is guaranteed to not be
    // re-used by a process. They start at 1 and increase with each new thread.
    // This allows easy command line access by a unique ID that is easier to
    // type than the actual system thread ID.
    uint32_t
    GetIndexID () const;
    
    //------------------------------------------------------------------
    // lldb::ExecutionContextScope pure virtual functions
    //------------------------------------------------------------------
    virtual Target *
    CalculateTarget ();

    virtual Process *
    CalculateProcess ();

    virtual Thread *
    CalculateThread ();

    virtual StackFrame *
    CalculateStackFrame ();

    virtual void
    CalculateExecutionContext (ExecutionContext &exe_ctx);
    
    lldb::StackFrameSP
    GetStackFrameSPForStackFramePtr (StackFrame *stack_frame_ptr);

protected:

    friend class ThreadPlan;
    friend class StackFrameList;
    
    // This is necessary to make sure thread assets get destroyed while the thread is still in good shape
    // to call virtual thread methods.  This must be called by classes that derive from Thread in their destructor.
    virtual void DestroyThread ();

    void
    PushPlan (lldb::ThreadPlanSP &plan_sp);

    void
    PopPlan ();

    void
    DiscardPlan ();

    ThreadPlan *GetPreviousPlan (ThreadPlan *plan);

    virtual lldb::StopInfoSP
    GetPrivateStopReason () = 0;

    typedef std::vector<lldb::ThreadPlanSP> plan_stack;

    virtual lldb_private::Unwind *
    GetUnwinder () = 0;

    StackFrameList &
    GetStackFrameList ();

    void
    SetStopInfo (lldb::StopInfoSP stop_info_sp)
    {
        m_actual_stop_info_sp = stop_info_sp;
    }

    //------------------------------------------------------------------
    // Classes that inherit from Process can see and modify these
    //------------------------------------------------------------------
    Process &           m_process;          ///< The process that owns this thread.
    lldb::StopInfoSP    m_actual_stop_info_sp;     ///< The private stop reason for this thread
    const uint32_t      m_index_id;         ///< A unique 1 based index assigned to each thread for easy UI/command line access.
    lldb::RegisterContextSP   m_reg_context_sp;   ///< The register context for this thread's current register state.
    lldb::StateType     m_state;            ///< The state of our process.
    mutable Mutex       m_state_mutex;      ///< Multithreaded protection for m_state.
    plan_stack          m_plan_stack;       ///< The stack of plans this thread is executing.
    plan_stack          m_completed_plan_stack;  ///< Plans that have been completed by this stop.  They get deleted when the thread resumes.
    plan_stack          m_discarded_plan_stack;  ///< Plans that have been discarded by this stop.  They get deleted when the thread resumes.
    std::auto_ptr<StackFrameList> m_curr_frames_ap; ///< The stack frames that get lazily populated after a thread stops.
    lldb::StackFrameListSP m_prev_frames_sp; ///< The previous stack frames from the last time this thread stopped.
    int                 m_resume_signal;    ///< The signal that should be used when continuing this thread.
    lldb::StateType     m_resume_state;     ///< The state that indicates what this thread should do when the process is resumed.
    std::auto_ptr<lldb_private::Unwind> m_unwinder_ap;
    bool                m_destroy_called;    // This is used internally to make sure derived Thread classes call DestroyThread.
private:
    //------------------------------------------------------------------
    // For Thread only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (Thread);

};

} // namespace lldb_private

#endif  // liblldb_Thread_h_
