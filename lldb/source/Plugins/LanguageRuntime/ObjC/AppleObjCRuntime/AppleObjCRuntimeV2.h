//===-- AppleObjCRuntimeV2.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_AppleObjCRuntimeV2_h_
#define liblldb_AppleObjCRuntimeV2_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Core/ValueObject.h"
#include "AppleObjCRuntime.h"
#include "AppleObjCTrampolineHandler.h"
#include "AppleThreadPlanStepThroughObjCTrampoline.h"

namespace lldb_private {

class AppleObjCRuntimeV2 :
        public AppleObjCRuntime
{
public:
    virtual ~AppleObjCRuntimeV2() { }
    
    // These are generic runtime functions:
    virtual bool
    GetDynamicTypeAndAddress (ValueObject &in_value, 
                              lldb::DynamicValueType use_dynamic, 
                              TypeAndOrName &class_type_or_name, 
                              Address &address);
    
    virtual ClangUtilityFunction *
    CreateObjectChecker (const char *);


    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();
    
    static void
    Terminate();
    
    static lldb_private::LanguageRuntime *
    CreateInstance (Process *process, lldb::LanguageType language);
    
    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();
    
    virtual const char *
    GetShortPluginName();
    
    virtual uint32_t
    GetPluginVersion();
    
    virtual void
    SetExceptionBreakpoints ();

    virtual ObjCRuntimeVersions
    GetRuntimeVersion ()
    {
        return eAppleObjC_V2;
    }

    virtual size_t
    GetByteOffsetForIvar (ClangASTType &parent_qual_type, const char *ivar_name);

    
protected:
    
private:
    AppleObjCRuntimeV2(Process *process, ModuleSP &objc_module_sp);
    
    bool RunFunctionToFindClassName (lldb::addr_t class_addr, Thread *thread, char *name_dst, size_t max_name_len);
    
    bool                                m_has_object_getClass;
    std::auto_ptr<ClangFunction>        m_get_class_name_function;
    std::auto_ptr<ClangUtilityFunction> m_get_class_name_code;
    lldb::addr_t                        m_get_class_name_args;
    Mutex                               m_get_class_name_args_mutex;
    
    static const char *g_find_class_name_function_name;
    static const char *g_find_class_name_function_body;
    static const char *g_objc_class_symbol_prefix;
    static const char *g_objc_class_data_section_name;
};
    
} // namespace lldb_private

#endif  // liblldb_AppleObjCRuntimeV2_h_
