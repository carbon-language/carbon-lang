//===-- ValueObjectConstResult.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectChild_h_
#define liblldb_ValueObjectChild_h_

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
    ValueObjectConstResult (clang::ASTContext *clang_ast,
                            void *clang_type,
                            const ConstString &name,
                            const lldb::DataBufferSP &result_data_sp,
                            lldb::ByteOrder byte_order, 
                            uint8_t addr_size);


    // When an expression fails to evaluate, we return an error
    ValueObjectConstResult (const Error& error);

    virtual ~ValueObjectConstResult();

    virtual size_t
    GetByteSize();

    virtual clang::ASTContext *
    GetClangAST ();

    virtual void *
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

protected:
    clang::ASTContext *m_clang_ast; // The clang AST that the clang type comes from
    ConstString m_type_name;

private:
    DISALLOW_COPY_AND_ASSIGN (ValueObjectConstResult);
};

} // namespace lldb_private

#endif  // liblldb_ValueObjectChild_h_
