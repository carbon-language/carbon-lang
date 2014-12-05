//===-- DynamicLoaderWindows.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_DynamicLoaderWindows_H_
#define liblldb_Plugins_Process_Windows_DynamicLoaderWindows_H_

#include "lldb/lldb-forward.h"
#include "lldb/Target/DynamicLoader.h"

namespace lldb_private
{

class DynamicLoaderWindows : public DynamicLoader
{
  public:
    DynamicLoaderWindows(Process *process);
    virtual ~DynamicLoaderWindows();

    static void Initialize();
    static void Terminate();
    static ConstString GetPluginNameStatic();
    static const char *GetPluginDescriptionStatic();

    static DynamicLoader *CreateInstance(Process *process, bool force);

    void DidAttach () override;
    void DidLaunch () override;
    Error CanLoadImage () override;
    lldb::ThreadPlanSP GetStepThroughTrampolinePlan(Thread &thread, bool stop) override;

    virtual ConstString GetPluginName() override;
    virtual uint32_t GetPluginVersion() override;
};

}

#endif