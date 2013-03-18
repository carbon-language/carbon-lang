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

using namespace lldb_perf;

class FormattersTest : public TestCase
{
public:
    FormattersTest () : TestCase()
    {
        m_dump_std_vector_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,1,false);
        }, "std-vector", "time to dump an std::vector");
        m_dump_std_list_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,1,false);
        }, "std-list", "time to dump an std::list");
        m_dump_std_map_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,1,false);
        }, "std-map", "time to dump an std::map");
        m_dump_std_string_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,1,false);
        }, "std-string", "time to dump an std::string");
        
        m_dump_nsstring_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,0,false);
        }, "ns-string", "time to dump an NSString");
        
        m_dump_nsarray_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,1,false);
        }, "ns-array", "time to dump an NSArray");
        
        m_dump_nsdictionary_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,1,false);
        }, "ns-dictionary", "time to dump an NSDictionary");
        
        m_dump_nsset_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,1,false);
        }, "ns-set", "time to dump an NSSet");
        
        m_dump_nsbundle_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,1,false);
        }, "ns-bundle", "time to dump an NSBundle");
        
        m_dump_nsdate_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,0,false);
        }, "ns-date", "time to dump an NSDate");
    }

    virtual
    ~FormattersTest ()
    {
    }
    
    virtual void
	Setup (int argc, const char** argv)
    {
        m_app_path.assign(argv[1]);
        m_out_path.assign(argv[2]);
        m_target = m_debugger.CreateTarget(m_app_path.c_str());
        m_target.BreakpointCreateByName("main");
        
        Launch (NULL,".");
    }
    
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
    
    void
    DoTest ()
    {
        SBThread thread_main(SelectMyThread("fmts_tester.mm"));
        SBFrame frame_zero(thread_main.GetFrameAtIndex(0));
        
        m_dump_nsarray_measurement(frame_zero.FindVariable("nsarray", lldb::eDynamicCanRunTarget));
        m_dump_nsarray_measurement(frame_zero.FindVariable("nsmutablearray", lldb::eDynamicCanRunTarget));

        m_dump_nsdictionary_measurement(frame_zero.FindVariable("nsdictionary", lldb::eDynamicCanRunTarget));
        m_dump_nsdictionary_measurement(frame_zero.FindVariable("nsmutabledictionary", lldb::eDynamicCanRunTarget));
        
        m_dump_nsstring_measurement(frame_zero.FindVariable("str0", lldb::eDynamicCanRunTarget));
        m_dump_nsstring_measurement(frame_zero.FindVariable("str1", lldb::eDynamicCanRunTarget));
        m_dump_nsstring_measurement(frame_zero.FindVariable("str2", lldb::eDynamicCanRunTarget));
        m_dump_nsstring_measurement(frame_zero.FindVariable("str3", lldb::eDynamicCanRunTarget));
        m_dump_nsstring_measurement(frame_zero.FindVariable("str4", lldb::eDynamicCanRunTarget));
        
        m_dump_nsdate_measurement(frame_zero.FindVariable("me", lldb::eDynamicCanRunTarget));
        m_dump_nsdate_measurement(frame_zero.FindVariable("cutie", lldb::eDynamicCanRunTarget));
        m_dump_nsdate_measurement(frame_zero.FindVariable("mom", lldb::eDynamicCanRunTarget));
        m_dump_nsdate_measurement(frame_zero.FindVariable("dad", lldb::eDynamicCanRunTarget));
        m_dump_nsdate_measurement(frame_zero.FindVariable("today", lldb::eDynamicCanRunTarget));
        
        m_dump_nsbundle_measurement(frame_zero.FindVariable("bundles", lldb::eDynamicCanRunTarget));
        m_dump_nsbundle_measurement(frame_zero.FindVariable("frameworks", lldb::eDynamicCanRunTarget));
        
        m_dump_nsset_measurement(frame_zero.FindVariable("nsset", lldb::eDynamicCanRunTarget));
        m_dump_nsset_measurement(frame_zero.FindVariable("nsmutableset", lldb::eDynamicCanRunTarget));
        
        m_dump_std_vector_measurement(frame_zero.FindVariable("vector", lldb::eDynamicCanRunTarget));
        m_dump_std_list_measurement(frame_zero.FindVariable("list", lldb::eDynamicCanRunTarget));
        m_dump_std_map_measurement(frame_zero.FindVariable("map", lldb::eDynamicCanRunTarget));

        m_dump_std_string_measurement(frame_zero.FindVariable("sstr0", lldb::eDynamicCanRunTarget));
        m_dump_std_string_measurement(frame_zero.FindVariable("sstr1", lldb::eDynamicCanRunTarget));
        m_dump_std_string_measurement(frame_zero.FindVariable("sstr2", lldb::eDynamicCanRunTarget));
        m_dump_std_string_measurement(frame_zero.FindVariable("sstr3", lldb::eDynamicCanRunTarget));
        m_dump_std_string_measurement(frame_zero.FindVariable("sstr4", lldb::eDynamicCanRunTarget));
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
        m_dump_std_vector_measurement.Write(array);
        m_dump_std_list_measurement.Write(array);
        m_dump_std_map_measurement.Write(array);
        m_dump_std_string_measurement.Write(array);

        m_dump_nsstring_measurement.Write(array);
        m_dump_nsarray_measurement.Write(array);
        m_dump_nsdictionary_measurement.Write(array);
        m_dump_nsset_measurement.Write(array);
        m_dump_nsbundle_measurement.Write(array);
        m_dump_nsdate_measurement.Write(array);

        CFDataRef xmlData = CFPropertyListCreateData(kCFAllocatorDefault, array.get(), kCFPropertyListXMLFormat_v1_0, 0, NULL);
        
        CFURLRef file = CFURLCreateFromFileSystemRepresentation(NULL, (const UInt8*)m_out_path.c_str(), m_out_path.size(), FALSE);
        
        CFURLWriteDataAndPropertiesToResource(file,xmlData,NULL,NULL);
    }
    
private:
    // C++ formatters
    TimeMeasurement<std::function<void(SBValue)>> m_dump_std_vector_measurement;
    TimeMeasurement<std::function<void(SBValue)>> m_dump_std_list_measurement;
    TimeMeasurement<std::function<void(SBValue)>> m_dump_std_map_measurement;
    TimeMeasurement<std::function<void(SBValue)>> m_dump_std_string_measurement;

    // Cocoa formatters
    TimeMeasurement<std::function<void(SBValue)>> m_dump_nsstring_measurement;
    TimeMeasurement<std::function<void(SBValue)>> m_dump_nsarray_measurement;
    TimeMeasurement<std::function<void(SBValue)>> m_dump_nsdictionary_measurement;
    TimeMeasurement<std::function<void(SBValue)>> m_dump_nsset_measurement;
    TimeMeasurement<std::function<void(SBValue)>> m_dump_nsbundle_measurement;
    TimeMeasurement<std::function<void(SBValue)>> m_dump_nsdate_measurement;

    // useful files
    std::string m_app_path;
    std::string m_out_path;
};

// argv[1] == path to app
// argv[2] == path to result
int main(int argc, const char * argv[])
{
    FormattersTest frmtest;
    frmtest.SetVerbose(true);
    TestCase::Run(frmtest,argc,argv);
    return 0;
}

