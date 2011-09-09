//===-- SBType.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBType_h_
#define LLDB_SBType_h_

#include "lldb/API/SBDefines.h"

namespace lldb {

class SBTypeList;    

class SBTypeMember
{
public:
    SBTypeMember ();
    
    SBTypeMember (const lldb::SBTypeMember& rhs);
    
    ~SBTypeMember();
    
#ifndef SWIG
    lldb::SBTypeMember&
    operator = (const lldb::SBTypeMember& rhs);
#endif

    bool
    IsValid() const;
    
    const char *
    GetName ();
    
    lldb::SBType
    GetType ();
    
    uint64_t
    GetOffsetByteSize();
    
protected:
    friend class SBType;

#ifndef SWIG
    void
    reset (lldb_private::TypeMemberImpl *);

    lldb_private::TypeMemberImpl &
    ref ();

    const lldb_private::TypeMemberImpl &
    ref () const;
#endif

    std::auto_ptr<lldb_private::TypeMemberImpl> m_opaque_ap;
};

class SBType
{
public:

    SBType();

    SBType (const lldb::SBType &rhs);

    ~SBType ();

    bool
    IsValid() const;
    
    size_t
    GetByteSize();

    bool
    IsPointerType();
    
    bool
    IsReferenceType();
    
    lldb::SBType
    GetPointerType();
    
    lldb::SBType
    GetPointeeType();
    
    lldb::SBType
    GetReferenceType();
    
    lldb::SBType
    GetDereferencedType();
    
    lldb::SBType
    GetBasicType(lldb::BasicType type);
    
    uint32_t
    GetNumberOfFields ();
    
    uint32_t
    GetNumberOfDirectBaseClasses ();

    uint32_t
    GetNumberOfVirtualBaseClasses ();
    
    lldb::SBTypeMember
    GetFieldAtIndex (uint32_t idx);
    
    lldb::SBTypeMember
    GetDirectBaseClassAtIndex (uint32_t idx);

    lldb::SBTypeMember
    GetVirtualBaseClassAtIndex (uint32_t idx);

    const char*
    GetName();
    
    lldb::TypeClass
    GetTypeClass ();

    // DEPRECATED: but needed for Xcode right now
    static bool
    IsPointerType (void * clang_type);
        
protected:
    
#ifndef SWIG
    lldb::SBType &
    operator = (const lldb::SBType &rhs);
    
    bool
    operator == (lldb::SBType &rhs);
    
    bool
    operator != (lldb::SBType &rhs);
    
    lldb_private::TypeImpl &
    ref ();
    
    const lldb_private::TypeImpl &
    ref () const;
    
    void
    reset(const lldb::TypeImplSP &type_impl_sp);
#endif
    

    lldb::TypeImplSP m_opaque_sp;
    
    friend class SBModule;
    friend class SBTarget;
    friend class SBValue;
    friend class SBTypeMember;
    friend class SBTypeList;
        
    SBType (const lldb_private::ClangASTType &);
    SBType (const lldb::TypeSP &);
    SBType (const lldb::TypeImplSP &);
    
};
    
class SBTypeList
{
public:
    SBTypeList();
    
    SBTypeList(const lldb::SBTypeList& rhs);
    
    ~SBTypeList();

    lldb::SBTypeList&
    operator = (const lldb::SBTypeList& rhs);
    
    bool
    IsValid();

    void
    Append (lldb::SBType type);
    
    lldb::SBType
    GetTypeAtIndex (uint32_t index);
    
    uint32_t
    GetSize();
    
    
private:
    std::auto_ptr<lldb_private::TypeListImpl> m_opaque_ap;
};
    

} // namespace lldb

#endif // LLDB_SBType_h_
