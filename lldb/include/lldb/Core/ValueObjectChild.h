//===-- ValueObjectChild.h --------------------------------------*- C++ -*-===//
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
class ValueObjectChild : public ValueObject
{
public:
    ValueObjectChild (ValueObject *parent,
                      clang::ASTContext *clang_ast,
                      void *clang_type,
                      const ConstString &name,
                      uint32_t byte_size,
                      int32_t byte_offset,
                      uint32_t bitfield_bit_size,
                      uint32_t bitfield_bit_offset);


    virtual ~ValueObjectChild();

    virtual size_t
    GetByteSize();

    virtual off_t
    GetByteOffset();

    virtual uint32_t
    GetBitfieldBitSize();

    virtual uint32_t
    GetBitfieldBitOffset();

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
    ValueObject* m_parent;  // The parent value object of which this is a child of.
    clang::ASTContext *m_clang_ast; // The clang AST that the clang type comes from
    void *m_clang_type; // The type of the child in question within the parent (m_parent_sp)
    ConstString m_type_name;
    uint32_t m_byte_size;
    int32_t m_byte_offset;
    uint32_t m_bitfield_bit_size;
    uint32_t m_bitfield_bit_offset;

    uint32_t
    GetByteOffset() const;
//
//  void
//  ReadValueFromMemory (ValueObject* parent, lldb::addr_t address);

private:
    DISALLOW_COPY_AND_ASSIGN (ValueObjectChild);
};

} // namespace lldb_private

#endif  // liblldb_ValueObjectChild_h_
