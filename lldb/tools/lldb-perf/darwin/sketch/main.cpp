//
//  main.cpp
//  PerfTestDriver
//
//  Created by Enrico Granata on 3/6/13.
//  Copyright (c) 2013 Apple Inc. All rights reserved.
//

#include <CoreFoundation/CoreFoundation.h>

#include "lldb-perf/lib/Timer.h"
#include "lldb-perf/lib/Metric.h"
#include "lldb-perf/lib/Measurement.h"
#include "lldb-perf/lib/TestCase.h"
#include "lldb-perf/lib/Xcode.h"

#include <iostream>
#include <unistd.h>
#include <fstream>

using namespace lldb::perf;

class SketchTest : public TestCase
{
public:
    SketchTest () :
    m_fetch_frames_measurement ([this] (SBProcess process) -> void {
        Xcode::FetchFrames (process,false,false);
    }, "fetch-frames"),
    m_file_line_bp_measurement([] (SBTarget target,const char* file, uint32_t line) -> void {
        Xcode::CreateFileLineBreakpoint(target, file, line);
    }, "file-line-bkpt"),
    m_fetch_modules_measurement ([] (SBTarget target) -> void {
        Xcode::FetchModules(target);
    }, "fetch-modules"),
    m_fetch_vars_measurement([this] (SBProcess process, int depth) -> void {
        auto threads_count = process.GetNumThreads();
        for (size_t thread_num = 0; thread_num < threads_count; thread_num++)
        {
            SBThread thread(process.GetThreadAtIndex(thread_num));
            SBFrame frame(thread.GetFrameAtIndex(0));
            Xcode::FetchVariables(frame,depth,GetVerbose());
            
        }
    }, "fetch-vars"),
    m_run_expr_measurement([this] (SBFrame frame, const char* expr) -> void {
        SBValue value(frame.EvaluateExpression(expr, lldb::eDynamicCanRunTarget));
        Xcode::FetchVariable(value,0,GetVerbose());
    }, "run-expr")
    {}
    
    virtual
    ~SketchTest ()
    {
    }
    
    virtual void
	Setup (int argc, const char** argv)
    {
        m_app_path.assign(argv[1]); // "~/perf/Small_ObjC/Sketch/build/Debug/Sketch.app"
        m_doc_path.assign(argv[2]); // "/Volumes/work/egranata/perf/Small_ObjC/TesterApp/foobar.sketch2";
        m_out_path.assign(argv[3]);
        TestCase::Setup(argc,argv);
        m_target = m_debugger.CreateTarget(m_app_path.c_str());
        const char* file_arg = m_doc_path.c_str(); 
        const char* persist_arg = "-ApplePersistenceIgnoreState";
        const char* persist_skip = "YES";
        const char* empty = nullptr;
        const char* args[] = {file_arg,persist_arg,persist_skip,empty};
        m_file_line_bp_measurement(m_target, "SKTDocument.m",245);
        m_file_line_bp_measurement(m_target, "SKTDocument.m",283);
        m_file_line_bp_measurement(m_target, "SKTText.m",326);
        
        Launch (args,".");
    }
    
    void
    DoTest ()
    {
        m_fetch_frames_measurement(m_process);
        m_fetch_modules_measurement(m_target);
        m_fetch_vars_measurement(m_process,1);
    }
    
	virtual ActionWanted
	TestStep (int counter)
    {
#define STEP(n) if (counter == n)
#define NEXT(s) return TestCase::ActionWanted{TestCase::ActionWanted::Type::eAWNext,SelectMyThread(s)}
#define FINISH(s) return TestCase::ActionWanted{TestCase::ActionWanted::Type::eAWFinish,SelectMyThread(s)}
#define CONT return TestCase::ActionWanted{TestCase::ActionWanted::Type::eAWContinue,SBThread()}
#define KILL return TestCase::ActionWanted{TestCase::ActionWanted::Type::eAWKill,SBThread()}
        STEP(0) {
            DoTest ();
            m_file_line_bp_measurement(m_target, "SKTDocument.m",254);
            CONT;
        }
        STEP(1) {
            DoTest ();
            SBThread thread(SelectMyThread("SKTDocument.m"));
            m_run_expr_measurement(thread.GetFrameAtIndex(0),"properties");
            m_run_expr_measurement(thread.GetFrameAtIndex(0),"[properties description]");
            m_run_expr_measurement(thread.GetFrameAtIndex(0),"typeName");
            m_run_expr_measurement(thread.GetFrameAtIndex(0),"data");
            m_run_expr_measurement(thread.GetFrameAtIndex(0),"[data description]");
            CONT;
        }
        STEP(2) {
            DoTest ();
            CONT;
        }
        STEP(3) {
            DoTest ();
            NEXT("SKTText.m");
        }
        STEP(4) {
            DoTest ();
            SBThread thread(SelectMyThread("SKTText.m"));
            m_run_expr_measurement(thread.GetFrameAtIndex(0),"layoutManager");
            m_run_expr_measurement(thread.GetFrameAtIndex(0),"contents");
            NEXT("SKTText.m");
        }
        STEP(5) {
            DoTest ();
            NEXT("SKTText.m");
        }
        STEP(6) {
            DoTest ();
            NEXT("SKTText.m");
        }
        STEP(7) {
            DoTest ();
            SBThread thread(SelectMyThread("SKTText.m"));
            m_run_expr_measurement(thread.GetFrameAtIndex(0),"@\"an NSString\"");
            m_run_expr_measurement(thread.GetFrameAtIndex(0),"[(id)@\"an NSString\" description]");
            m_run_expr_measurement(thread.GetFrameAtIndex(0),"@[@1,@2,@3]");
            FINISH("SKTText.m");
        }
        STEP(8) {
            DoTest ();
            SBThread thread(SelectMyThread("SKTGraphicView.m"));
            m_run_expr_measurement(thread.GetFrameAtIndex(0),"[graphics description]");
            m_run_expr_measurement(thread.GetFrameAtIndex(0),"[selectionIndexes description]");
            m_run_expr_measurement(thread.GetFrameAtIndex(0),"(BOOL)NSIntersectsRect(rect, graphicDrawingBounds)");
            KILL;
        }
        KILL;
#undef STEP
#undef NEXT
#undef CONT
#undef KILL
    }
    
    void
    Results ()
    {
        auto ff_metric = m_fetch_frames_measurement.metric();
        auto fl_metric = m_file_line_bp_measurement.metric();
        auto md_metric = m_fetch_modules_measurement.metric();
        auto fv_metric = m_fetch_vars_measurement.metric();
        auto xp_metric = m_run_expr_measurement.metric();
        
        CFCMutableArray array;
        ff_metric.Write(array);
        fl_metric.Write(array);
        md_metric.Write(array);
        fv_metric.Write(array);
        xp_metric.Write(array);

        CFDataRef xmlData = CFPropertyListCreateData(kCFAllocatorDefault, array.get(), kCFPropertyListXMLFormat_v1_0, 0, NULL);
        
        CFURLRef file = CFURLCreateFromFileSystemRepresentation(NULL, (const UInt8*)m_out_path.c_str(), m_out_path.size(), FALSE);
        
        CFURLWriteDataAndPropertiesToResource(file,xmlData,NULL,NULL);
    }
    
private:
    Measurement<lldb::perf::TimeGauge, std::function<void(SBProcess)>> m_fetch_frames_measurement;
    Measurement<lldb::perf::TimeGauge, std::function<void(SBTarget, const char*, uint32_t)>> m_file_line_bp_measurement;
    Measurement<lldb::perf::TimeGauge, std::function<void(SBTarget)>> m_fetch_modules_measurement;
    Measurement<lldb::perf::TimeGauge, std::function<void(SBProcess,int)>> m_fetch_vars_measurement;
    Measurement<lldb::perf::TimeGauge, std::function<void(SBFrame,const char*)>> m_run_expr_measurement;
    
    SBThread
	SelectMyThread (const char* file_name)
	{
		auto threads_count = m_process.GetNumThreads();
		for (auto thread_num = 0; thread_num < threads_count; thread_num++)
		{
			SBThread thread(m_process.GetThreadAtIndex(thread_num));
			auto local_file_name = thread.GetFrameAtIndex(0).GetCompileUnit().GetFileSpec().GetFilename();
			if (!local_file_name)
				continue;
			if (strcmp(local_file_name,file_name))
				continue;
			return thread;
		}
		Xcode::RunCommand(m_debugger,"bt all",true);
		assert(false);
	}
    std::string m_app_path;
    std::string m_doc_path;
    std::string m_out_path;
};

// argv[1] == path to app
// argv[2] == path to document
// argv[3] == path to result
int main(int argc, const char * argv[])
{
    SketchTest skt;
    TestCase::Run(skt,argc,argv);
    return 0;
}

