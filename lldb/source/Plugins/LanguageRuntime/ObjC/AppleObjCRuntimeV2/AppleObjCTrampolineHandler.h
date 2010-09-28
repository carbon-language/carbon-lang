//===-- AppleObjCTrampolineHandler.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_AppleObjCTrampolineHandler_h_
#define lldb_AppleObjCTrampolineHandler_h_

// C Includes
// C++ Includes
#include <map>
#include <string>
// Other libraries and framework includes
// Project includes
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Host/Mutex.h"


namespace lldb_private
{
using namespace lldb;
  
class AppleObjCTrampolineHandler {
public:
    
    AppleObjCTrampolineHandler (ProcessSP process_sp, ModuleSP objc_module_sp);
    
    ~AppleObjCTrampolineHandler() {}
    
    static bool ModuleIsObjCLibrary (const ModuleSP &module_sp);
        
    ThreadPlanSP
    GetStepThroughDispatchPlan (Thread &thread, bool stop_others);
    
    ClangFunction *
    GetLookupImplementationWrapperFunction ();
    
    
    struct DispatchFunction {
    public:
        typedef enum 
        {
            eFixUpNone,
            eFixUpFixed,
            eFixUpToFix
        } FixUpState;
                
        const char *name;
        bool stret_return;
        bool is_super;
        FixUpState fixedup;
    };
    
private:
    static const DispatchFunction g_dispatch_functions[];
    
    typedef std::map<lldb::addr_t, int> MsgsendMap; // This table maps an dispatch fn address to the index in g_dispatch_functions
    MsgsendMap m_msgSend_map;
    ProcessSP m_process_sp;
    ModuleSP m_objc_module_sp;
    lldb::addr_t get_impl_addr;
    std::auto_ptr<ClangFunction> m_impl_function;
    Mutex m_impl_function_mutex;
    lldb::addr_t m_impl_fn_addr;
    lldb::addr_t m_impl_stret_fn_addr;
    
     
};

}  // using namespace lldb_private

#endif	// lldb_AppleObjCTrampolineHandler_h_
