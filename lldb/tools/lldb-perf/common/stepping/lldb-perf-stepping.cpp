#include <CoreFoundation/CoreFoundation.h>

#include "lldb-perf/lib/Timer.h"
#include "lldb-perf/lib/Metric.h"
#include "lldb-perf/lib/Measurement.h"
#include "lldb-perf/lib/TestCase.h"
#include "lldb-perf/lib/Xcode.h"

#include <unistd.h>
#include <string>

using namespace lldb_perf;

class StepTest : public TestCase
{
public:
    StepTest() :
        m_do_one_step_over_measurement (std::function<void(StepTest &, int)>(&StepTest::DoOneStep))
    {
    }
    
    virtual
    ~StepTest() {}
    
    virtual void
    Setup (int argc, const char **argv)
    {
        m_app_path.assign(argv[1]);
        m_out_path.assign(argv[2]);
        TestCase::Setup (argc, argv);
        
        m_target = m_debugger.CreateTarget(m_app_path.c_str());
        const char* file_arg = m_app_path.c_str();
        const char* empty = nullptr;
        const char* args[] = {file_arg, empty};
        
        Launch (args,".");
    }

private:
    void
    DoOneStep (int sequence)
    {
    
    }
    
    TimeMeasurement<std::function<void(StepTest &, int)> > m_do_one_step_over_measurement;
    std::string m_app_path;
    std::string m_out_path;
    

};

// argv[1] == path to app
// argv[2] == path to result
int main(int argc, const char * argv[])
{
    if (argc != 3)
    {
        printf ("Wrong number of arguments, should be \"path to app\", \"path to result.\"\n");
        return -1;
    }
    
    StepTest skt;
    TestCase::Run(skt,argc,argv);
    return 0;
}
