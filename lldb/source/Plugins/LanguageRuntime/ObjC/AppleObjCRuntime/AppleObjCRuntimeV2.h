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

#include <map>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "AppleObjCRuntime.h"

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
    
    virtual ObjCRuntimeVersions
    GetRuntimeVersion ()
    {
        return eAppleObjC_V2;
    }

    virtual size_t
    GetByteOffsetForIvar (ClangASTType &parent_qual_type, const char *ivar_name);
    
    virtual void
    UpdateISAToDescriptorMap_Impl();
    
    virtual bool
    IsValidISA (ObjCLanguageRuntime::ObjCISA isa)
    {
        return (isa != 0);
    }
    
    // none of these are valid ISAs - we use them to infer the type
    // of tagged pointers - if we have something meaningful to say
    // we report an actual type - otherwise, we just say tagged
    // there is no connection between the values here and the tagged pointers map
    static const ObjCLanguageRuntime::ObjCISA g_objc_Tagged_ISA = 1;

    static const ObjCLanguageRuntime::ObjCISA g_objc_Tagged_ISA_NSAtom = 2;
    static const ObjCLanguageRuntime::ObjCISA g_objc_Tagged_ISA_NSNumber = 3;
    static const ObjCLanguageRuntime::ObjCISA g_objc_Tagged_ISA_NSDateTS = 4;
    static const ObjCLanguageRuntime::ObjCISA g_objc_Tagged_ISA_NSManagedObject = 5;
    static const ObjCLanguageRuntime::ObjCISA g_objc_Tagged_ISA_NSDate = 6;

    
    virtual ObjCLanguageRuntime::ObjCISA
    GetISA(ValueObject& valobj);
    
    virtual ConstString
    GetActualTypeName(ObjCLanguageRuntime::ObjCISA isa);
    
    virtual ClassDescriptorSP
    GetClassDescriptor (ValueObject& in_value);
    
    virtual ClassDescriptorSP
    CreateClassDescriptor (ObjCISA isa);
    
    virtual TypeVendor *
    GetTypeVendor();
    
    lldb::ProcessSP
    GetProcessSP ()
    {
        return m_process_wp.lock();
    }
    
protected:
    virtual lldb::BreakpointResolverSP
    CreateExceptionResolver (Breakpoint *bkpt, bool catch_bp, bool throw_bp);

private:
    
    AppleObjCRuntimeV2 (Process *process,
                        const lldb::ModuleSP &objc_module_sp);
    
    bool
    IsTaggedPointer(lldb::addr_t ptr);
    
    bool RunFunctionToFindClassName (lldb::addr_t class_addr, Thread *thread, char *name_dst, size_t max_name_len);
    
    bool                                m_has_object_getClass;
    std::auto_ptr<ClangFunction>        m_get_class_name_function;
    std::auto_ptr<ClangUtilityFunction> m_get_class_name_code;
    lldb::addr_t                        m_get_class_name_args;
    Mutex                               m_get_class_name_args_mutex;
    
    std::auto_ptr<TypeVendor>           m_type_vendor_ap;
    lldb::ProcessWP                     m_process_wp; // used by class descriptors to lazily fill their own data
    
    static const char *g_find_class_name_function_name;
    static const char *g_find_class_name_function_body;
};
    
} // namespace lldb_private

#endif  // liblldb_AppleObjCRuntimeV2_h_
