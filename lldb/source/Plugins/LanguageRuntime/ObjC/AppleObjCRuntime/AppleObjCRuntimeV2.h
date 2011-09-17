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
    
    virtual void
    SetExceptionBreakpoints ();

    virtual ObjCRuntimeVersions
    GetRuntimeVersion ()
    {
        return eAppleObjC_V2;
    }

    virtual size_t
    GetByteOffsetForIvar (ClangASTType &parent_qual_type, const char *ivar_name);
    
    virtual bool
    IsValidISA (ObjCLanguageRuntime::ObjCISA isa)
    {
        return (isa != 0);
    }
    
    // this is not a valid ISA in the sense that no valid
    // class pointer can live at address 1. we use it to refer to
    // tagged types, where the ISA must be dynamically determined
    static const ObjCLanguageRuntime::ObjCISA g_objc_Tagged_ISA = 1;
    
    virtual ObjCLanguageRuntime::ObjCISA
    GetISA(ValueObject& valobj);   
    
    virtual ConstString
    GetActualTypeName(ObjCLanguageRuntime::ObjCISA isa);
    
    virtual ObjCLanguageRuntime::ObjCISA
    GetParentClass(ObjCLanguageRuntime::ObjCISA isa);
    
protected:
    
private:
    
    typedef std::map<ObjCLanguageRuntime::ObjCISA, ConstString> ISAToNameCache;
    typedef std::map<ObjCLanguageRuntime::ObjCISA, ObjCLanguageRuntime::ObjCISA> ISAToParentCache;
    
    typedef ISAToNameCache::iterator ISAToNameIterator;
    typedef ISAToParentCache::iterator ISAToParentIterator;
    
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
    
    ISAToNameCache                      m_isa_to_name_cache;
    ISAToParentCache                    m_isa_to_parent_cache;
    
    static const char *g_find_class_name_function_name;
    static const char *g_find_class_name_function_body;
    static const char *g_objc_class_symbol_prefix;
    static const char *g_objc_class_data_section_name;
};
    
} // namespace lldb_private

#endif  // liblldb_AppleObjCRuntimeV2_h_
