//===-- SWIG Interface for SBType -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

    class SBType
    {
        public:
                
        SBType (const SBType &rhs);
        
        ~SBType ();
                
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
    };
    
    class SBTypeList
    {
        public:
        SBTypeList();
        
        void
        AppendType(SBType type);
        
        SBType
        GetTypeAtIndex(int index);
        
        int
        GetSize();
        
        ~SBTypeList();
        
        private:
        std::auto_ptr<SBTypeListImpl> m_content;
    };

} // namespace lldb
