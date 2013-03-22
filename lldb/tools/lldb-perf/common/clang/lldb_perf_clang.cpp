//===-- lldb_perf_clang.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb-perf/lib/Timer.h"
#include "lldb-perf/lib/Metric.h"
#include "lldb-perf/lib/Measurement.h"
#include "lldb-perf/lib/Results.h"
#include "lldb-perf/lib/TestCase.h"
#include "lldb-perf/lib/Xcode.h"
#include <iostream>
#include <unistd.h>
#include <fstream>

using namespace lldb_perf;

class ClangTest : public TestCase
{
public:
    ClangTest () :
        TestCase(),
        m_set_bp_main_by_name(CreateTimeMeasurement([this] () -> void
            {
                m_target.BreakpointCreateByName("main");
                m_target.BreakpointCreateByName("malloc");
            }, "breakpoint1-relative-time", "Elapsed time to set a breakpoint at main by name, run and hit the breakpoint.")),
        m_delta_memory("breakpoint1-memory-delta", "Memory increase that occurs due to setting a breakpoint at main by name.")
    {
    }

    virtual
    ~ClangTest ()
    {
    }
    
    virtual bool
	Setup (int argc, const char** argv)
    {
        SetVerbose(true);
        m_app_path.assign(argv[1]);
        m_out_path.assign(argv[2]);
        return true;
    }
    
    void
    DoTest ()
    {
    }
    
	virtual void
	TestStep (int counter, ActionWanted &next_action)
    {
        switch (counter)
        {
            case 0:
                {
                    m_total_memory.Start();
                    m_target = m_debugger.CreateTarget(m_app_path.c_str());
                    const char *clang_argv[] = { "clang --version", NULL };
                    m_delta_memory.Start();
                    m_set_bp_main_by_name();
                    m_delta_memory.Stop();
                    SBLaunchInfo launch_info(clang_argv);
                    Launch (launch_info);
                }
                break;
            case 1:
                next_action.StepOver(m_thread);
                break;
            case 2:
                next_action.StepOver(m_thread);
                break;
            case 3:
                next_action.StepOver(m_thread);
                break;
            default:
                m_total_memory.Stop();
                next_action.Kill();
                break;
        }
    }
    
    void
    WriteResults (Results &results)
    {
        Results::Dictionary& results_dict = results.GetDictionary();
        
        m_set_bp_main_by_name.WriteAverageValue(results);
        m_delta_memory.WriteAverageValue(results);

        results_dict.Add ("breakpoint1-memory-total",
                          "The total memory that the current process is using after setting the first breakpoint.",
                          m_total_memory.GetStopValue().GetResult(NULL, NULL));
        
        results.Write(m_out_path.c_str());
    }
    
private:
    // C++ formatters
    TimeMeasurement<std::function<void()>> m_set_bp_main_by_name;
    MemoryMeasurement<std::function<void()>> m_delta_memory;
    MemoryGauge m_total_memory;
    std::string m_app_path;
    std::string m_out_path;

};

// argv[1] == path to app
// argv[2] == path to result
int main(int argc, const char * argv[])
{
    ClangTest test;
    test.SetVerbose(true);
    TestCase::Run(test, argc, argv);
    return 0;
}

