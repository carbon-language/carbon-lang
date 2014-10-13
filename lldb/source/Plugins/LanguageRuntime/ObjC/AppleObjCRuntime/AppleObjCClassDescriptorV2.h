//===-- AppleObjCClassDescriptorV2.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_AppleObjCClassDescriptorV2_h_
#define liblldb_AppleObjCClassDescriptorV2_h_

// C Includes
// C++ Includes
// Other libraries and framework includes

// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "AppleObjCRuntimeV2.h"

namespace lldb_private {

class ClassDescriptorV2 : public ObjCLanguageRuntime::ClassDescriptor
{
public:
    friend class lldb_private::AppleObjCRuntimeV2;
    
private:
    // The constructor should only be invoked by the runtime as it builds its caches
    // or populates them.  A ClassDescriptorV2 should only ever exist in a cache.
    ClassDescriptorV2 (AppleObjCRuntimeV2 &runtime, ObjCLanguageRuntime::ObjCISA isa, const char *name) :
        m_runtime (runtime),
        m_objc_class_ptr (isa),
        m_name (name),
        m_ivars_storage()
    {
    }
    
public:
    virtual ConstString
    GetClassName ();
    
    virtual ObjCLanguageRuntime::ClassDescriptorSP
    GetSuperclass ();
    
    virtual ObjCLanguageRuntime::ClassDescriptorSP
    GetMetaclass () const;
    
    virtual bool
    IsValid ()
    {
        return true;    // any Objective-C v2 runtime class descriptor we vend is valid
    }
    
    // a custom descriptor is used for tagged pointers
    virtual bool
    GetTaggedPointerInfo (uint64_t* info_bits = NULL,
                          uint64_t* value_bits = NULL,
                          uint64_t* payload = NULL)
    {
        return false;
    }
    
    virtual uint64_t
    GetInstanceSize ();
    
    virtual ObjCLanguageRuntime::ObjCISA
    GetISA ()
    {
        return m_objc_class_ptr;
    }
    
    virtual bool
    Describe (std::function <void (ObjCLanguageRuntime::ObjCISA)> const &superclass_func,
              std::function <bool (const char *, const char *)> const &instance_method_func,
              std::function <bool (const char *, const char *)> const &class_method_func,
              std::function <bool (const char *, const char *, lldb::addr_t, uint64_t)> const &ivar_func) const;
    
    virtual
    ~ClassDescriptorV2 ()
    {
    }
    
    virtual size_t
    GetNumIVars ()
    {
        GetIVarInformation();
        return m_ivars_storage.size();
    }
    
    virtual iVarDescriptor
    GetIVarAtIndex (size_t idx)
    {
        if (idx >= GetNumIVars())
            return iVarDescriptor();
        return m_ivars_storage[idx];
    }
    
protected:
    void
    GetIVarInformation ();
    
private:
    static const uint32_t RW_REALIZED = (1 << 31);
    
    struct objc_class_t {
        ObjCLanguageRuntime::ObjCISA    m_isa;              // The class's metaclass.
        ObjCLanguageRuntime::ObjCISA    m_superclass;
        lldb::addr_t                    m_cache_ptr;
        lldb::addr_t                    m_vtable_ptr;
        lldb::addr_t                    m_data_ptr;
        uint8_t                         m_flags;
        
        objc_class_t () :
        m_isa (0),
        m_superclass (0),
        m_cache_ptr (0),
        m_vtable_ptr (0),
        m_data_ptr (0),
        m_flags (0)
        {
        }
        
        void
        Clear()
        {
            m_isa = 0;
            m_superclass = 0;
            m_cache_ptr = 0;
            m_vtable_ptr = 0;
            m_data_ptr = 0;
            m_flags = 0;
        }
        
        bool
        Read(Process *process, lldb::addr_t addr);
    };
    
    struct class_ro_t {
        uint32_t                        m_flags;
        uint32_t                        m_instanceStart;
        uint32_t                        m_instanceSize;
        uint32_t                        m_reserved;
        
        lldb::addr_t                    m_ivarLayout_ptr;
        lldb::addr_t                    m_name_ptr;
        lldb::addr_t                    m_baseMethods_ptr;
        lldb::addr_t                    m_baseProtocols_ptr;
        lldb::addr_t                    m_ivars_ptr;
        
        lldb::addr_t                    m_weakIvarLayout_ptr;
        lldb::addr_t                    m_baseProperties_ptr;
        
        std::string                     m_name;
        
        bool
        Read(Process *process, lldb::addr_t addr);
    };
    
    struct class_rw_t {
        uint32_t                        m_flags;
        uint32_t                        m_version;
        
        lldb::addr_t                    m_ro_ptr;
        union {
            lldb::addr_t                m_method_list_ptr;
            lldb::addr_t                m_method_lists_ptr;
        };
        lldb::addr_t                    m_properties_ptr;
        lldb::addr_t                    m_protocols_ptr;
        
        ObjCLanguageRuntime::ObjCISA    m_firstSubclass;
        ObjCLanguageRuntime::ObjCISA    m_nextSiblingClass;
        
        bool
        Read(Process *process, lldb::addr_t addr);
    };
    
    struct method_list_t
    {
        uint32_t        m_entsize;
        uint32_t        m_count;
        lldb::addr_t    m_first_ptr;
        
        bool
        Read(Process *process, lldb::addr_t addr);
    };
    
    struct method_t
    {
        lldb::addr_t    m_name_ptr;
        lldb::addr_t    m_types_ptr;
        lldb::addr_t    m_imp_ptr;
        
        std::string     m_name;
        std::string     m_types;
        
        static size_t GetSize(Process *process)
        {
            size_t ptr_size = process->GetAddressByteSize();
            
            return ptr_size     // SEL name;
            + ptr_size   // const char *types;
            + ptr_size;  // IMP imp;
        }
        
        bool
        Read(Process *process, lldb::addr_t addr);
    };
    
    struct ivar_list_t
    {
        uint32_t        m_entsize;
        uint32_t        m_count;
        lldb::addr_t    m_first_ptr;
        
        bool Read(Process *process, lldb::addr_t addr);
    };
    
    struct ivar_t
    {
        lldb::addr_t    m_offset_ptr;
        lldb::addr_t    m_name_ptr;
        lldb::addr_t    m_type_ptr;
        uint32_t        m_alignment;
        uint32_t        m_size;
        
        std::string     m_name;
        std::string     m_type;
        
        static size_t GetSize(Process *process)
        {
            size_t ptr_size = process->GetAddressByteSize();
            
            return ptr_size             // uintptr_t *offset;
            + ptr_size             // const char *name;
            + ptr_size             // const char *type;
            + sizeof(uint32_t)     // uint32_t alignment;
            + sizeof(uint32_t);    // uint32_t size;
        }
        
        bool
        Read(Process *process, lldb::addr_t addr);
    };
    
    class iVarsStorage
    {
    public:
        iVarsStorage ();
        
        size_t
        size ();
        
        iVarDescriptor&
        operator[] (size_t idx);
        
        void
        fill (AppleObjCRuntimeV2& runtime, ClassDescriptorV2& descriptor);
        
    private:
        bool m_filled;
        std::vector<iVarDescriptor> m_ivars;
        Mutex m_mutex;
    };
    
    bool
    Read_objc_class (Process* process, std::unique_ptr<objc_class_t> &objc_class) const;
    
    bool
    Read_class_row (Process* process, const objc_class_t &objc_class, std::unique_ptr<class_ro_t> &class_ro, std::unique_ptr<class_rw_t> &class_rw) const;
    
    AppleObjCRuntimeV2 &m_runtime;          // The runtime, so we can read information lazily.
    lldb::addr_t        m_objc_class_ptr;   // The address of the objc_class_t.  (I.e., objects of this class type have this as their ISA)
    ConstString         m_name;             // May be NULL
    iVarsStorage        m_ivars_storage;
};

// tagged pointer descriptor
class ClassDescriptorV2Tagged : public ObjCLanguageRuntime::ClassDescriptor
{
public:
    ClassDescriptorV2Tagged (ConstString class_name,
                             uint64_t payload)
    {
        m_name = class_name;
        if (!m_name)
        {
            m_valid = false;
            return;
        }
        m_valid = true;
        m_payload = payload;
        m_info_bits = (m_payload & 0xF0ULL) >> 4;
        m_value_bits = (m_payload & ~0x0000000000000000FFULL) >> 8;
    }
    
    ClassDescriptorV2Tagged (ObjCLanguageRuntime::ClassDescriptorSP actual_class_sp,
                             uint64_t payload)
    {
        if (!actual_class_sp)
        {
            m_valid = false;
            return;
        }
        m_name = actual_class_sp->GetClassName();
        if (!m_name)
        {
            m_valid = false;
            return;
        }
        m_valid = true;
        m_payload = payload;
        m_info_bits = (m_payload & 0x0FULL);
        m_value_bits = (m_payload & ~0x0FULL) >> 4;
    }
    
    virtual ConstString
    GetClassName ()
    {
        return m_name;
    }
    
    virtual ObjCLanguageRuntime::ClassDescriptorSP
    GetSuperclass ()
    {
        // tagged pointers can represent a class that has a superclass, but since that information is not
        // stored in the object itself, we would have to query the runtime to discover the hierarchy
        // for the time being, we skip this step in the interest of static discovery
        return ObjCLanguageRuntime::ClassDescriptorSP();
    }
    
    virtual ObjCLanguageRuntime::ClassDescriptorSP
    GetMetaclass () const
    {
        return ObjCLanguageRuntime::ClassDescriptorSP();
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
    GetTaggedPointerInfo (uint64_t* info_bits = NULL,
                          uint64_t* value_bits = NULL,
                          uint64_t* payload = NULL)
    {
        if (info_bits)
            *info_bits = GetInfoBits();
        if (value_bits)
            *value_bits = GetValueBits();
        if (payload)
            *payload = GetPayload();
        return true;
    }
    
    virtual uint64_t
    GetInstanceSize ()
    {
        return (IsValid() ? m_pointer_size : 0);
    }
    
    virtual ObjCLanguageRuntime::ObjCISA
    GetISA ()
    {
        return 0; // tagged pointers have no ISA
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
    
    virtual uint64_t
    GetPayload ()
    {
        return (IsValid() ? m_payload : 0);
    }
    
    virtual
    ~ClassDescriptorV2Tagged ()
    {}
    
private:
    ConstString m_name;
    uint8_t m_pointer_size;
    bool m_valid;
    uint64_t m_info_bits;
    uint64_t m_value_bits;
    uint64_t m_payload;
    
};
    
} // namespace lldb_private

#endif  // liblldb_AppleObjCRuntime_h_

