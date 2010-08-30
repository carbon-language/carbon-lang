//===-- ClangExpressionVariable.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/ClangExpressionVariable.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "clang/AST/ASTContext.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/Value.h"

using namespace lldb_private;
using namespace clang;

ClangExpressionVariable::ClangExpressionVariable()
{
    m_name = "";
    m_user_type = TypeFromUser(NULL, NULL);
    m_parser_vars.reset(NULL);
    m_jit_vars.reset(NULL);
    m_data_vars.reset(NULL);
}

void ClangExpressionVariable::DisableDataVars()
{
    if (m_data_vars.get() && m_data_vars->m_data)
        delete m_data_vars->m_data;
    m_data_vars.reset();
}

Error
ClangExpressionVariable::Print (Stream &output_stream,
                                ExecutionContext &exe_ctx,
                                lldb::Format format,
                                bool show_types,
                                bool show_summary,
                                bool verbose)
{
    Error err;
    
    if (!m_data_vars.get() || !m_data_vars->m_data)
    {
        err.SetErrorToGenericError();
        err.SetErrorStringWithFormat("Variable doesn't contain a value");
        return err;
    }
    
    Value val;
    
    clang::ASTContext *ast_context = m_user_type.GetASTContext();
    
    val.SetContext (Value::eContextTypeOpaqueClangQualType, m_user_type.GetOpaqueQualType ());
    val.SetValueType (Value::eValueTypeHostAddress);
    val.GetScalar() = (uint64_t)m_data_vars->m_data->GetBytes ();
    
    val.ResolveValue (&exe_ctx, ast_context);
    
    if (val.GetContextType () == Value::eContextTypeInvalid &&
        val.GetValueType () == Value::eValueTypeScalar &&
        format == lldb::eFormatDefault)
    {
        // The expression result is just a scalar with no special formatting
        val.GetScalar ().GetValue (&output_stream, show_types);
        output_stream.EOL ();
        return err;
    }
    
    // The expression result is more complex and requires special handling
    DataExtractor data;
    Error expr_error = val.GetValueAsData (&exe_ctx, ast_context, data, 0);
    
    if (!expr_error.Success ())
    {
        err.SetErrorToGenericError ();
        err.SetErrorStringWithFormat ("Couldn't resolve variable value: %s", expr_error.AsCString ());
        return err;
    }
    
    if (format == lldb::eFormatDefault)
        format = val.GetValueDefaultFormat ();
    
    void *clang_type = val.GetValueOpaqueClangQualType ();
    
    output_stream.Printf("%s = ", m_name.c_str());
    
    if (clang_type)
    {
        if (show_types)
            output_stream.Printf("(%s) ", ClangASTType::GetClangTypeName (clang_type).GetCString());
        
        ClangASTType::DumpValue (ast_context,               // The ASTContext that the clang type belongs to
                                 clang_type,                // The opaque clang type we want to dump that value of
                                 &exe_ctx,                  // The execution context for memory and variable access
                                 &output_stream,            // Stream to dump to
                                 format,                    // Format to use when dumping
                                 data,                      // A buffer containing the bytes for the clang type
                                 0,                         // Byte offset within "data" where value is
                                 data.GetByteSize (),       // Size in bytes of the value we are dumping
                                 0,                         // Bitfield bit size
                                 0,                         // Bitfield bit offset
                                 show_types,                // Show types?
                                 show_summary,              // Show summary?
                                 verbose,                   // Debug logging output?
                                 UINT32_MAX);               // Depth to dump in case this is an aggregate type
    }
    else
    {
        data.Dump (&output_stream,          // Stream to dump to
                   0,                       // Byte offset within "data"
                   format,                  // Format to use when dumping
                   data.GetByteSize (),     // Size in bytes of each item we are dumping
                   1,                       // Number of items to dump
                   UINT32_MAX,              // Number of items per line
                   LLDB_INVALID_ADDRESS,    // Invalid address, don't show any offset/address context
                   0,                       // Bitfield bit size
                   0);                      // Bitfield bit offset
    }
    
    output_stream.EOL();
    
    return err;
}

ClangExpressionVariable::ClangExpressionVariable(const ClangExpressionVariable &cev) :
    m_name(cev.m_name),
    m_user_type(cev.m_user_type),
    m_store(cev.m_store),
    m_index(cev.m_index)
{
    if (cev.m_parser_vars.get())
    {
        m_parser_vars.reset(new struct ParserVars);
        *m_parser_vars.get() = *cev.m_parser_vars.get();
    }
    
    if (cev.m_jit_vars.get())
    {
        m_jit_vars.reset(new struct JITVars);
        *m_jit_vars.get() = *cev.m_jit_vars.get();
    }
    
    if (cev.m_data_vars.get())
    {
        m_data_vars.reset(new struct DataVars);
        *m_data_vars.get() = *cev.m_data_vars.get();
    }
}

bool
ClangExpressionVariable::PointValueAtData(Value &value)
{
    if (!m_data_vars.get())
        return false;
    
    value.SetContext(Value::eContextTypeOpaqueClangQualType, m_user_type.GetOpaqueQualType());
    value.SetValueType(Value::eValueTypeHostAddress);
    value.GetScalar() = (uint64_t)m_data_vars->m_data->GetBytes();
    
    return true;
}
