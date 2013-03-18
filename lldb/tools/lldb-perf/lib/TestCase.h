//
//  TestCase.h
//  PerfTestDriver
//
//  Created by Enrico Granata on 3/7/13.
//  Copyright (c) 2013 Apple Inc. All rights reserved.
//

#ifndef __PerfTestDriver__TestCase__
#define __PerfTestDriver__TestCase__

#include "lldb/API/LLDB.h"
#include "Measurement.h"

namespace lldb_perf
{
class TestCase
{
public:
    TestCase();
    
    struct ActionWanted
	{
		enum class Type
		{
			eNext,
			eContinue,
            eFinish,
			eKill
		} type;
		lldb::SBThread thread;
        
        ActionWanted () :
            type (Type::eContinue),
            thread ()
        {
        }
        
        void
        Continue()
        {
            type = Type::eContinue;
            thread = lldb::SBThread();
        }
        
        void
        Next (lldb::SBThread t)
        {
            type = Type::eNext;
            thread = t;
        }

        void
        Finish (lldb::SBThread t)
        {
            type = Type::eFinish;
            thread = t;
        }
        
        void
        Kill ()
        {
            type = Type::eKill;
            thread = lldb::SBThread();
        }
	};
    
    virtual
    ~TestCase ()
    {}
    
	virtual void
	Setup (int argc, const char** argv);
    
	virtual void
	TestStep (int counter, ActionWanted &next_action) = 0;
	
	bool
	Launch (const char** args, const char* cwd);
	
	void
	Loop();
    
    void
    SetVerbose (bool);
    
    bool
    GetVerbose ();
    
    virtual void
    Results () = 0;
    
    template <typename G,typename A>
    Measurement<G,A> CreateMeasurement (A a, const char* name = NULL, const char* description = NULL)
    {
        return Measurement<G,A> (a,name, description);
    }
    
    template <typename A>
    TimeMeasurement<A> CreateTimeMeasurement (A a, const char* name = NULL, const char* description = NULL)
    {
        return TimeMeasurement<A> (a,name, description);
    }
    
    static void
    Run (TestCase& test, int argc, const char** argv);
    
protected:
    lldb::SBDebugger m_debugger;
	lldb::SBTarget m_target;
	lldb::SBProcess m_process;
	lldb::SBThread m_thread;
	lldb::SBListener m_listener;
    bool m_verbose;
};
}

#endif /* defined(__PerfTestDriver__TestCase__) */
