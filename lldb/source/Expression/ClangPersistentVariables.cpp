//===-- ClangPersistentVariables.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangPersistentVariables.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Value.h"

using namespace lldb_private;

Error
ClangPersistentVariable::Print (Stream &output_stream,
                                ExecutionContext &exe_ctx,
                                lldb::Format format,
                                bool show_types,
                                bool show_summary,
                                bool verbose)
{
    Error err;
    
    Value val;
    
    clang::ASTContext *ast_context = m_user_type.GetASTContext();
    
    val.SetContext (Value::eContextTypeOpaqueClangQualType, m_user_type.GetOpaqueQualType ());
    val.SetValueType (Value::eValueTypeHostAddress);
    val.GetScalar() = (uint64_t)Data ();
    
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
    
    // The expression result is more complext and requires special handling
    DataExtractor data;
    Error expr_error = val.GetValueAsData (&exe_ctx, ast_context, data, 0);
    
    if (!expr_error.Success ())
    {
        err.SetErrorToGenericError ();
        err.SetErrorStringWithFormat ("Couldn't resolve result value: %s", expr_error.AsCString ());
        return err;
    }
    
    if (format == lldb::eFormatDefault)
        format = val.GetValueDefaultFormat ();
    
    void *clang_type = val.GetValueOpaqueClangQualType ();
    
    output_stream.Printf("%s = ", m_name.AsCString("<anonymous>"));
    
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

ClangPersistentVariables::ClangPersistentVariables () :
    m_variables(),
    m_result_counter(0)
{
}

ClangPersistentVariable *
ClangPersistentVariables::CreateVariable (ConstString name, 
                                          TypeFromUser user_type)
{    
    ClangPersistentVariable new_var(name, user_type);
    
    if (m_variables.find(name) != m_variables.end())
        return NULL;
    
    m_variables[name] = new_var;
    
    return &m_variables[name];
}

ClangPersistentVariable *
ClangPersistentVariables::GetVariable (ConstString name)
{    
    if (m_variables.find(name) == m_variables.end())
        return NULL;
    
    return &m_variables[name];
}

void
ClangPersistentVariables::GetNextResultName (std::string &name)
{
    StreamString s;
    s.Printf("$%llu", m_result_counter);
    
    m_result_counter++;
    
    name = s.GetString();
}
