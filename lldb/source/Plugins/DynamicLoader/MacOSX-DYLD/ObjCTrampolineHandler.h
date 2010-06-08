//===-- ObjCTrampolineHandler.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_ObjCTrampolineHandler_h_
#define lldb_ObjCTrampolineHandler_h_

// C Includes
// C++ Includes
#include <map>
#include <string>
// Other libraries and framework includes
// Project includes
#include "lldb.h"
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Host/Mutex.h"


namespace lldb_private
{
using namespace lldb;
  
class ObjCTrampolineHandler {
public:
    
    ObjCTrampolineHandler (ProcessSP process_sp, ModuleSP objc_module_sp);
    
    ~ObjCTrampolineHandler() {}
    
    static bool ModuleIsObjCLibrary (const ModuleSP &module_sp);
        
    ThreadPlanSP
    GetStepThroughDispatchPlan (Thread &thread, bool stop_others);
    
    void
    AddToCache (lldb::addr_t class_addr, lldb::addr_t sel, lldb::addr_t impl_addr);
    
    lldb::addr_t
    LookupInCache (lldb::addr_t class_addr, lldb::addr_t sel);
    
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
    
     
    // We keep a map of <Class,Selector>->Implementation so we don't have to call the resolver
    // function over and over.
    
    // FIXME: We need to watch for the loading of Protocols, and flush the cache for any
    // class that we see so changed.
    
    struct ClassAndSel
    {
        ClassAndSel()
        {
            sel_addr = LLDB_INVALID_ADDRESS;
            class_addr = LLDB_INVALID_ADDRESS;
        }
        ClassAndSel (lldb::addr_t in_sel_addr, lldb::addr_t in_class_addr) :
            class_addr (in_class_addr),
            sel_addr(in_sel_addr)
        {
        }
        bool operator== (const ClassAndSel &rhs)
        {
            if (class_addr == rhs.class_addr
                && sel_addr == rhs.sel_addr)
                return true;
            else
                return false;
        }
        
        bool operator< (const ClassAndSel &rhs) const
        {
            if (class_addr < rhs.class_addr)
                return true;
            else if (class_addr > rhs.class_addr)
                return false;
            else
            {
                if (sel_addr < rhs.sel_addr)
                    return true;
                else
                    return false;
            }
        }
        
        lldb::addr_t class_addr;
        lldb::addr_t sel_addr;
    };

    typedef std::map<ClassAndSel,lldb::addr_t> MsgImplMap;
    MsgImplMap m_impl_cache;
    
};

};  // using namespace lldb_private

#endif	// lldb_ObjCTrampolineHandler_h_
