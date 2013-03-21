//===-- lldb_perf_clang.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CoreFoundation/CoreFoundation.h>

#include "lldb-perf/lib/Timer.h"
#include "lldb-perf/lib/Metric.h"
#include "lldb-perf/lib/Measurement.h"
#include "lldb-perf/lib/TestCase.h"
#include "lldb-perf/lib/Xcode.h"

#include <iostream>
#include <unistd.h>
#include <fstream>

using namespace lldb_perf;

class ClangTest : public TestCase
{
public:
    ClangTest () : TestCase()
    {
        m_set_bp_main_by_name = CreateTimeMeasurement([this] () -> void {
            m_target.BreakpointCreateByName("main");
        }, "break at \"main\"", "time set a breakpoint at main by name, run and hit the breakpoint");
    }

    virtual
    ~ClangTest ()
    {
    }
    
    virtual bool
	Setup (int argc, const char** argv)
    {
        m_app_path.assign(argv[1]);
        m_out_path.assign(argv[2]);
        m_target = m_debugger.CreateTarget(m_app_path.c_str());
        m_set_bp_main_by_name();
        const char *clang_argv[] = { "clang --version", NULL };
        SBLaunchInfo launch_info(clang_argv);
        return Launch (launch_info);
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
                m_target.BreakpointCreateByLocation("fmts_tester.mm", 68);
                next_action.Continue();
                break;
            case 1:
                DoTest ();
                next_action.Continue();
                break;
            case 2:
                DoTest ();
                next_action.Continue();
                break;
            case 3:
                DoTest ();
                next_action.Continue();
                break;
            case 4:
                DoTest ();
                next_action.Continue();
                break;
            case 5:
                DoTest ();
                next_action.Continue();
                break;
            case 6:
                DoTest ();
                next_action.Continue();
                break;
            case 7:
                DoTest ();
                next_action.Continue();
                break;
            case 8:
                DoTest ();
                next_action.Continue();
                break;
            case 9:
                DoTest ();
                next_action.Continue();
                break;
            case 10:
                DoTest ();
                next_action.Continue();
                break;
            default:
                next_action.Kill();
                break;
        }
    }
    
    void
    Results ()
    {
        CFCMutableArray array;
        m_set_bp_main_by_name.Write(array);

        CFDataRef xmlData = CFPropertyListCreateData(kCFAllocatorDefault, array.get(), kCFPropertyListXMLFormat_v1_0, 0, NULL);
        
        CFURLRef file = CFURLCreateFromFileSystemRepresentation(NULL, (const UInt8*)m_out_path.c_str(), m_out_path.size(), FALSE);
        
        CFURLWriteDataAndPropertiesToResource(file,xmlData,NULL,NULL);
    }
    
private:
    // C++ formatters
    TimeMeasurement<std::function<void()>> m_set_bp_main_by_name;
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

