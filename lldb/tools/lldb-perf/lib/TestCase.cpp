//
//  TestCase.cpp
//  PerfTestDriver
//
//  Created by Enrico Granata on 3/7/13.
//  Copyright (c) 2013 Apple Inc. All rights reserved.
//

#include "TestCase.h"
#include "Xcode.h"

using namespace lldb::perf;

TestCase::TestCase () :
m_debugger(),
m_target(),
m_process(),
m_thread(),
m_listener(),
m_verbose(false)
{}

void
TestCase::Setup (int argc, const char** argv)
{
	SBDebugger::Initialize();
	SBHostOS::ThreadCreated ("<lldb-tester.app.main>");
	m_debugger = SBDebugger::Create(false);
	m_listener = m_debugger.GetListener();
}

bool
TestCase::Launch (const char** args, const char* cwd)
{
	m_process = m_target.LaunchSimple(args,NULL,cwd);
	m_process.GetBroadcaster().AddListener(m_listener, SBProcess::eBroadcastBitStateChanged | SBProcess::eBroadcastBitInterrupt);
	return m_process.IsValid ();
}

void
TestCase::SetVerbose (bool b)
{
    m_verbose = b;
}

bool
TestCase::GetVerbose ()
{
    return m_verbose;
}

void
TestCase::Loop ()
{
	int step = 0;
	SBEvent evt;
	while (true)
	{
		m_listener.WaitForEvent (UINT32_MAX,evt);
		StateType state = SBProcess::GetStateFromEvent (evt);
		if (m_verbose)
			printf("event = %s\n",SBDebugger::StateAsCString(state));
		if (SBProcess::GetRestartedFromEvent(evt))
			continue;
		switch (state)
		{
			case eStateInvalid:
			case eStateDetached:
			case eStateCrashed:
			case eStateUnloaded:
				break;
			case eStateExited:
				return;
			case eStateConnected:
			case eStateAttaching:
			case eStateLaunching:
			case eStateRunning:
			case eStateStepping:
				continue;
			case eStateStopped:
			case eStateSuspended:
			{
				bool fatal = false;
				for (auto thread_index = 0; thread_index < m_process.GetNumThreads(); thread_index++)
				{
					SBThread thread(m_process.GetThreadAtIndex(thread_index));
					SBFrame frame(thread.GetFrameAtIndex(0));
					StopReason stop_reason = thread.GetStopReason();
					if (m_verbose) printf("tid = 0x%llx pc = 0x%llx ",thread.GetThreadID(),frame.GetPC());
					switch (stop_reason)
					{
				        case eStopReasonNone:
                            if (m_verbose) printf("none\n");
                            break;
                            
				        case eStopReasonTrace:
                            if (m_verbose) printf("trace\n");
                            break;
                            
				        case eStopReasonPlanComplete:
                            if (m_verbose) printf("plan complete\n");
                            break;
				        case eStopReasonThreadExiting:
                            if (m_verbose) printf("thread exiting\n");
                            break;
				        case eStopReasonExec:
                            if (m_verbose) printf("exec\n");
                            break;
						case eStopReasonInvalid:
                            if (m_verbose) printf("invalid\n");
                            break;
			        	case eStopReasonException:
                            if (m_verbose) printf("exception\n");
                            fatal = true;
                            break;
				        case eStopReasonBreakpoint:
                            if (m_verbose) printf("breakpoint id = %lld.%lld\n",thread.GetStopReasonDataAtIndex(0),thread.GetStopReasonDataAtIndex(1));
                            break;
				        case eStopReasonWatchpoint:
                            if (m_verbose) printf("watchpoint id = %lld\n",thread.GetStopReasonDataAtIndex(0));
                            break;
				        case eStopReasonSignal:
                            if (m_verbose) printf("signal %d\n",(int)thread.GetStopReasonDataAtIndex(0));
                            break;
					}
				}
				if (fatal)
				{
					if (m_verbose) Xcode::RunCommand(m_debugger,"bt all",true);
					exit(1);
				}
				if (m_verbose)
					printf("RUNNING STEP %d\n",step);
				auto action = TestStep(step);
				step++;
				switch (action.type)
				{
					case ActionWanted::Type::eAWContinue:
						m_debugger.HandleCommand("continue");
						break;
                    case ActionWanted::Type::eAWFinish:
                        if (action.thread.IsValid() == false)
                        {
                            if (m_verbose) Xcode::RunCommand(m_debugger,"bt all",true);
                            if (m_verbose) printf("[finish invalid] I am gonna die at step %d\n",step);
                            exit(501);
                        }
                        m_process.SetSelectedThread(action.thread);
                        m_debugger.HandleCommand("finish");
						break;
					case ActionWanted::Type::eAWNext:
                        if (action.thread.IsValid() == false)
                        {
                            if (m_verbose) Xcode::RunCommand(m_debugger,"bt all",true);
                            if (m_verbose) printf("[next invalid] I am gonna die at step %d\n",step);
                            exit(500);
                        }
                        m_process.SetSelectedThread(action.thread);
                        m_debugger.HandleCommand("next");
						break;
					case ActionWanted::Type::eAWKill:
						if (m_verbose) printf("I want to die\n");
						m_process.Kill();
						return;
				}
			}
		}
	}
	if (GetVerbose()) printf("I am gonna die at step %d\n",step);
}

void
TestCase::Run (TestCase& test, int argc, const char** argv)
{
    test.Setup(argc, argv);
    test.Loop();
    test.Results();
}
