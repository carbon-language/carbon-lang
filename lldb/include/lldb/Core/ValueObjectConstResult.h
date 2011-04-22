//===-- ValueObjectConstResult.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectConstResult_h_
#define liblldb_ValueObjectConstResult_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ValueObject.h"

namespace lldb_private {

//----------------------------------------------------------------------
// A child of another ValueObject.
//----------------------------------------------------------------------
class ValueObjectConstResult : public ValueObject
{
public:
    static lldb::ValueObjectSP
    Create (ExecutionContextScope *exe_scope,
            lldb::ByteOrder byte_order, 
            uint32_t addr_byte_size);

    static lldb::ValueObjectSP
    Create (ExecutionContextScope *exe_scope,
            clang::ASTContext *clang_ast,
            void *clang_type,
            const ConstString &name,
            const DataExtractor &data);

    static lldb::ValueObjectSP
    Create (ExecutionContextScope *exe_scope,
            clang::ASTContext *clang_ast,
            void *clang_type,
            const ConstString &name,
            const lldb::DataBufferSP &result_data_sp,
            lldb::ByteOrder byte_order, 
            uint8_t addr_size);

    static lldb::ValueObjectSP
    Create (ExecutionContextScope *exe_scope,
            clang::ASTContext *clang_ast,
            void *clang_type,
            const ConstString &name,
            lldb::addr_t address,
            AddressType address_type,
            uint8_t addr_byte_size);

    // When an expression fails to evaluate, we return an error
    static lldb::ValueObjectSP
    Create (ExecutionContextScope *exe_scope,
            const Error& error);

    virtual ~ValueObjectConstResult();

    virtual size_t
    GetByteSize();

    virtual clang::ASTContext *
    GetClangAST ();

    virtual lldb::clang_type_t
    GetClangType ();

    virtual lldb::ValueType
    GetValueType() const;

    virtual uint32_t
    CalculateNumChildren();

    virtual ConstString
    GetTypeName();

    virtual bool
    IsInScope ();

    virtual bool
    SetClangAST (clang::ASTContext *ast)
    {
        m_clang_ast = ast;
        return true;
    }

    void
    SetByteSize (size_t size);

protected:
    virtual bool
    UpdateValue ();

    virtual void
    CalculateDynamicValue () {} // CalculateDynamicValue doesn't change the dynamic value, since this can get
                                // called at any time and you can't reliably fetch the dynamic value at any time.
                                // If we want to have dynamic values for ConstResults, then we'll need to make them
                                // up when we make the const result & stuff them in by hand.
                                    
    clang::ASTContext *m_clang_ast; // The clang AST that the clang type comes from
    ConstString m_type_name;
    uint32_t m_byte_size;

private:
    ValueObjectConstResult (ExecutionContextScope *exe_scope,
                            lldb::ByteOrder byte_order, 
                            uint32_t addr_byte_size);

    ValueObjectConstResult (ExecutionContextScope *exe_scope,
                            clang::ASTContext *clang_ast,
                            void *clang_type,
                            const ConstString &name,
                            const DataExtractor &data);

    ValueObjectConstResult (ExecutionContextScope *exe_scope,
                            clang::ASTContext *clang_ast,
                            void *clang_type,
                            const ConstString &name,
                            const lldb::DataBufferSP &result_data_sp,
                            lldb::ByteOrder byte_order, 
                            uint8_t addr_size);

    ValueObjectConstResult (ExecutionContextScope *exe_scope,
                            clang::ASTContext *clang_ast,
                            void *clang_type,
                            const ConstString &name,
                            lldb::addr_t address,
                            AddressType address_type,
                            uint8_t addr_byte_size);

    ValueObjectConstResult (ExecutionContextScope *exe_scope,
                            const Error& error);

    DISALLOW_COPY_AND_ASSIGN (ValueObjectConstResult);
};

} // namespace lldb_private

#endif  // liblldb_ValueObjectConstResult_h_
