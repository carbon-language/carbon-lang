//===-- AppleThreadPlanStepThroughObjCTrampoline.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_AppleThreadPlanStepThroughObjCTrampoline_h_
#define lldb_AppleThreadPlanStepThroughObjCTrampoline_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-types.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/Target/ThreadPlan.h"
#include "AppleObjCTrampolineHandler.h"

namespace lldb_private 
{

class AppleThreadPlanStepThroughObjCTrampoline : public ThreadPlan
{
public:
	//------------------------------------------------------------------
	// Constructors and Destructors
	//------------------------------------------------------------------
	AppleThreadPlanStepThroughObjCTrampoline(Thread &thread, 
                                        AppleObjCTrampolineHandler *trampoline_handler, 
                                        lldb::addr_t args_addr, 
                                        lldb::addr_t object_ptr, 
                                        lldb::addr_t class_ptr, 
                                        lldb::addr_t sel_ptr, 
                                        bool stop_others);
    
	virtual ~AppleThreadPlanStepThroughObjCTrampoline();

    virtual void
    GetDescription (Stream *s,
                    lldb::DescriptionLevel level);
                    
    virtual bool
    ValidatePlan (Stream *error);

    virtual bool
    PlanExplainsStop ();


    virtual lldb::StateType
    GetPlanRunState ();

    virtual bool
    ShouldStop (Event *event_ptr);

    // The base class MischiefManaged does some cleanup - so you have to call it
    // in your MischiefManaged derived class.
    virtual bool
    MischiefManaged ();
    
    virtual void
    DidPush();
    
    virtual bool
    WillStop();



protected:
	//------------------------------------------------------------------
	// Classes that inherit from AppleThreadPlanStepThroughObjCTrampoline can see and modify these
	//------------------------------------------------------------------
	
private:
	//------------------------------------------------------------------
	// For AppleThreadPlanStepThroughObjCTrampoline only
	//------------------------------------------------------------------
    bool m_stop_others;
    lldb::addr_t m_object_ptr;
    lldb::addr_t m_class_ptr;
    lldb::addr_t m_sel_ptr;

    ThreadPlanSP m_func_sp;       // This is the function call plan.  We fill it at start, then set it
                                  // to NULL when this plan is done.  That way we know to go to:
    lldb::addr_t m_args_addr;     // Stores the address for our step through function result structure.
    ThreadPlanSP m_run_to_sp;     // The plan that runs to the target.
    AppleObjCTrampolineHandler *m_objc_trampoline_handler;
    ClangFunction *m_impl_function;  // This is a pointer to a impl function that 
                                     // is owned by the client that pushes this plan.
};

} // namespace lldb_private

#endif	// lldb_AppleThreadPlanStepThroughObjCTrampoline_h_
