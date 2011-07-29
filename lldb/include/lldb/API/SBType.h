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
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/Type.h"

namespace lldb {

class SBTypeList;    

class SBType
{
public:

    SBType (const SBType &rhs);

    ~SBType ();

#ifndef SWIG
    const lldb::SBType &
    operator = (const lldb::SBType &rhs);
    
    bool
    operator == (const lldb::SBType &rhs) const;
    
    bool
    operator != (const lldb::SBType &rhs) const;
    
    lldb_private::TypeImpl &
    ref ();
    
    const lldb_private::TypeImpl &
    ref () const;
    
#endif
    
    bool
    IsValid() const;
    
    size_t
    GetByteSize() const;

    bool
    IsPointerType() const;
    
    bool
    IsReferenceType() const;
    
    SBType
    GetPointerType() const;
    
    SBType
    GetPointeeType() const;
    
    SBType
    GetReferenceType() const;
    
    SBType
    GetDereferencedType() const;
    
    SBType
    GetBasicType(lldb::BasicType type) const;
        
    const char*
    GetName();
        
protected:
    std::auto_ptr<lldb_private::TypeImpl> m_opaque_ap;
    
    friend class SBModule;
    friend class SBTarget;
    friend class SBValue;
    friend class SBTypeList;
        
    SBType (clang::ASTContext*, clang_type_t);
    SBType (lldb_private::ClangASTType type);
    SBType (lldb::TypeSP type);
    SBType (lldb_private::TypeImpl impl);
    SBType();
    
};
    
class SBTypeList
{
public:
    SBTypeList();
    
    SBTypeList(const SBTypeList& rhs);
    
    SBTypeList&
    operator = (const SBTypeList& rhs);
    
    void
    AppendType(SBType type);
    
    SBType
    GetTypeAtIndex(int index) const;
    
    int
    GetSize() const;
    
    ~SBTypeList();
    
private:
    std::auto_ptr<lldb_private::TypeListImpl> m_opaque_ap;
};

} // namespace lldb

#endif // LLDB_SBType_h_
