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

using namespace lldb;

namespace lldb { namespace perf
{
class TestCase
{
public:
    TestCase();
    
    struct ActionWanted
	{
		enum class Type
		{
			eAWNext,
			eAWContinue,
            eAWFinish,
			eAWKill
		} type;
		SBThread thread;
	};
    
    virtual
    ~TestCase ()
    {}
    
	virtual void
	Setup (int argc, const char** argv);
    
	virtual ActionWanted
	TestStep (int counter) = 0;
	
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
	SBDebugger m_debugger;
	SBTarget m_target;
	SBProcess m_process;
	SBThread m_thread;
	SBListener m_listener;
    bool m_verbose;
};
} }

#endif /* defined(__PerfTestDriver__TestCase__) */
