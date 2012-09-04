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
    
    class ClassDescriptorV2 : public ObjCLanguageRuntime::ClassDescriptor
    {
    public:
        ClassDescriptorV2 (ValueObject &isa_pointer);
        ClassDescriptorV2 (ObjCISA isa, lldb::ProcessSP process);
        
        virtual ConstString
        GetClassName ()
        {
            return m_name;
        }
        
        virtual ClassDescriptorSP
        GetSuperclass ();
        
        virtual bool
        IsValid ()
        {
            return m_valid;
        }
        
        virtual bool
        IsTagged ()
        {
            return false;   // we use a special class for tagged descriptors
        }
        
        virtual uint64_t
        GetInstanceSize ()
        {
            return m_instance_size;
        }
        
        virtual ObjCISA
        GetISA ()
        {
            return m_isa;
        }
        
        virtual
        ~ClassDescriptorV2 ()
        {}
        
    protected:
        virtual bool
        CheckPointer (lldb::addr_t value,
                      uint32_t ptr_size)
        {
            if (ptr_size != 8)
                return true;
            return ((value & 0xFFFF800000000000) == 0);
        }
        
        void
        Initialize (ObjCISA isa, lldb::ProcessSP process_sp);
        
    private:
        ConstString m_name;
        ObjCISA m_isa;
        ObjCISA m_parent_isa;
        bool m_valid;
        lldb::ProcessWP m_process_wp;
        uint64_t m_instance_size;
    };
    
    class ClassDescriptorV2Tagged : public ObjCLanguageRuntime::ClassDescriptor
    {
    public:
        ClassDescriptorV2Tagged (ValueObject &isa_pointer);
        
        virtual ConstString
        GetClassName ()
        {
            return m_name;
        }
        
        virtual ClassDescriptorSP
        GetSuperclass ()
        {
            // tagged pointers can represent a class that has a superclass, but since that information is not
            // stored in the object itself, we would have to query the runtime to discover the hierarchy
            // for the time being, we skip this step in the interest of static discovery
            return ClassDescriptorSP(new ObjCLanguageRuntime::ClassDescriptor_Invalid());
        }
        
        virtual bool
        IsValid ()
        {
            return m_valid;
        }
        
        virtual bool
        IsKVO ()
        {
            return false; // tagged pointers are not KVO'ed
        }
        
        virtual bool
        IsCFType ()
        {
            return false; // tagged pointers are not CF objects
        }
        
        virtual bool
        IsTagged ()
        {
            return true;   // we use this class to describe tagged pointers
        }
        
        virtual uint64_t
        GetInstanceSize ()
        {
            return (IsValid() ? m_pointer_size : 0);
        }
        
        virtual ObjCISA
        GetISA ()
        {
            return 0; // tagged pointers have no ISA
        }

        virtual uint64_t
        GetClassBits ()
        {
            return (IsValid() ? m_class_bits : 0);
        }
        
        // these calls are not part of any formal tagged pointers specification
        virtual uint64_t
        GetValueBits ()
        {
            return (IsValid() ? m_value_bits : 0);
        }
        
        virtual uint64_t
        GetInfoBits ()
        {
            return (IsValid() ? m_info_bits : 0);
        }
        
        virtual
        ~ClassDescriptorV2Tagged ()
        {}
        
    protected:
        // TODO make this into a smarter OS version detector
        LazyBool
        IsLion (lldb::TargetSP &target_sp);
        
    private:
        ConstString m_name;
        uint8_t m_pointer_size;
        bool m_valid;
        uint64_t m_class_bits;
        uint64_t m_info_bits;
        uint64_t m_value_bits;
    };
    
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
    
    virtual ClassDescriptorSP
    GetClassDescriptor (ValueObject& in_value);
    
    virtual ClassDescriptorSP
    GetClassDescriptor (ObjCISA isa);
    
    virtual SymbolVendor *
    GetSymbolVendor();
    
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
    
    std::auto_ptr<SymbolVendor>         m_symbol_vendor_ap;
    
    static const char *g_find_class_name_function_name;
    static const char *g_find_class_name_function_body;
};
    
} // namespace lldb_private

#endif  // liblldb_AppleObjCRuntimeV2_h_
