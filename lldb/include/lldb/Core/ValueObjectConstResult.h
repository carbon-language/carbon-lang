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
    ValueObjectConstResult (lldb::ByteOrder byte_order, 
                            uint32_t addr_byte_size);

    ValueObjectConstResult (clang::ASTContext *clang_ast,
                            void *clang_type,
                            const ConstString &name,
                            const DataExtractor &data);

    ValueObjectConstResult (clang::ASTContext *clang_ast,
                            void *clang_type,
                            const ConstString &name,
                            const lldb::DataBufferSP &result_data_sp,
                            lldb::ByteOrder byte_order, 
                            uint8_t addr_size);

    ValueObjectConstResult (clang::ASTContext *clang_ast,
                            void *clang_type,
                            const ConstString &name,
                            lldb::addr_t address,
                            lldb::AddressType address_type,
                            uint8_t addr_byte_size);

    // When an expression fails to evaluate, we return an error
    ValueObjectConstResult (const Error& error);

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

    virtual void
    UpdateValue (ExecutionContextScope *exe_scope);

    virtual bool
    IsInScope (StackFrame *frame);

    virtual bool
    SetClangAST (clang::ASTContext *ast)
    {
        m_clang_ast = ast;
        return true;
    }

    void
    SetByteSize (size_t size);

protected:
    clang::ASTContext *m_clang_ast; // The clang AST that the clang type comes from
    ConstString m_type_name;
    uint32_t m_byte_size;

private:
    DISALLOW_COPY_AND_ASSIGN (ValueObjectConstResult);
};

} // namespace lldb_private

#endif  // liblldb_ValueObjectConstResult_h_
