//===-- Value.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Value_h_
#define liblldb_Value_h_

// C Includes
// C++ Includes
#include <vector>
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Scalar.h"

namespace lldb_private {

class Value
{
public:

    // Values Less than zero are an error, greater than or equal to zero
    // returns what the Scalar result is.
    enum ValueType
    {
                                        // m_value contains...
                                        // ============================
        eValueTypeScalar,               // raw scalar value
        eValueTypeFileAddress,          // file address value
        eValueTypeLoadAddress,          // load address value
        eValueTypeHostAddress           // host address value (for memory in the process that is using liblldb)
    };

    enum ContextType                    // Type that describes Value::m_context
    {
                                        // m_context contains...
                                        // ====================
        eContextTypeInvalid,            // undefined
        eContextTypeClangType,          // void * (an opaque clang::QualType * that can be fed to "static QualType QualType::getFromOpaquePtr(void *)")
        eContextTypeRegisterInfo,       // lldb::RegisterInfo *
        eContextTypeLLDBType,           // lldb_private::Type *
        eContextTypeVariable,           // lldb_private::Variable *
        eContextTypeValue               // Value * (making this a proxy value.  Used when putting locals on the DWARF expression parser stack)
    };

    Value();
    Value(const Scalar& scalar);
    Value(int v);
    Value(unsigned int v);
    Value(long v);
    Value(unsigned long v);
    Value(long long v);
    Value(unsigned long long v);
    Value(float v);
    Value(double v);
    Value(long double v);
    Value(const uint8_t *bytes, int len);
    Value(const Value &v);
    
    Value *
    CreateProxy();
    
    Value *
    GetProxyTarget();

    void *
    GetClangType();

    ValueType
    GetValueType() const;

    lldb::AddressType
    GetValueAddressType () const;

    ContextType
    GetContextType() const;

    void
    SetValueType (ValueType value_type);

    void
    ClearContext ();

    void
    SetContext (ContextType context_type, void *p);

    lldb::RegisterInfo *
    GetRegisterInfo();

    Type *
    GetType();

    Scalar &
    ResolveValue (ExecutionContext *exe_ctx, clang::ASTContext *ast_context);

    Scalar &
    GetScalar();
    
    void
    ResizeData(int len);

    bool
    ValueOf(ExecutionContext *exe_ctx, clang::ASTContext *ast_context);

    Variable *
    GetVariable();

    void
    Dump (Stream* strm);

    lldb::Format
    GetValueDefaultFormat ();

    size_t
    GetValueByteSize (clang::ASTContext *ast_context, Error *error_ptr);

    Error
    GetValueAsData (ExecutionContext *exe_ctx, clang::ASTContext *ast_context, DataExtractor &data, uint32_t data_offset);

    static const char *
    GetValueTypeAsCString (ValueType context_type);

    static const char *
    GetContextTypeAsCString (ContextType context_type);

    bool
    GetData (DataExtractor &data);

protected:
    Scalar          m_value;
    ValueType       m_value_type;
    void *          m_context;
    ContextType     m_context_type;
    DataBufferHeap  m_data_buffer;
};

class ValueList
{
public:
    ValueList () {}
    ValueList (const ValueList &rhs);

    ~ValueList () {}

    const ValueList & operator= (const ValueList &rhs);

    // void InsertValue (Value *value, size_t idx);
    void PushValue (const Value &value);

    size_t GetSize ();
    Value *GetValueAtIndex(size_t idx);
    void Clear();

protected:

private:
    typedef std::vector<Value> collection;

    collection m_values;
};

} // namespace lldb_private

#endif  // liblldb_Value_h_
