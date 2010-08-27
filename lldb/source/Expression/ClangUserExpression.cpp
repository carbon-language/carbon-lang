//===-- ClangUserExpression.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <stdio.h>
#if HAVE_SYS_TYPES_H
#  include <sys/types.h>
#endif

// C++ Includes
#include <cstdlib>
#include <string>
#include <map>

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Expression/ClangExpressionDeclMap.h"
#include "lldb/Expression/ClangExpressionParser.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Expression/ASTResultSynthesizer.h"
#include "lldb/Expression/ClangUserExpression.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Target.h"

using namespace lldb_private;

ClangUserExpression::ClangUserExpression (const char *expr) :
    m_expr_text(expr),
    m_jit_addr(LLDB_INVALID_ADDRESS)
{
    StreamString m_transformed_stream;
    
    m_transformed_stream.Printf("extern \"C\" void %s(void *___clang_arg) { %s; }\n",
                                FunctionName(),
                                m_expr_text.c_str());
    
    m_transformed_text = m_transformed_stream.GetData();
}

ClangUserExpression::~ClangUserExpression ()
{
}

clang::ASTConsumer *
ClangUserExpression::ASTTransformer (clang::ASTConsumer *passthrough)
{
    return new ASTResultSynthesizer(passthrough);
}

bool 
ClangUserExpression::Parse (Stream &error_stream, ExecutionContext &exe_ctx)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
    
    ////////////////////////////////////
    // Set up the target and compiler
    //
    
    Target *target = exe_ctx.target;
    
    if (!target)
    {
        error_stream.PutCString ("error: invalid target\n");
        return false;
    }
    
    ConstString target_triple;
    
    target->GetTargetTriple (target_triple);
    
    if (!target_triple)
        target_triple = Host::GetTargetTriple ();
    
    if (!target_triple)
    {
        error_stream.PutCString ("error: invalid target triple\n");
        return false;
    }
        
    //////////////////////////
    // Parse the expression
    //
    
    m_expr_decl_map.reset(new ClangExpressionDeclMap(&exe_ctx));
    
    ClangExpressionParser parser(target_triple.GetCString(), *this);
    
    unsigned num_errors = parser.Parse (error_stream);
    
    if (num_errors)
    {
        error_stream.Printf ("error: %d errors parsing expression\n", num_errors);
        return false;
    }
    
    ///////////////////////////////////////////////
    // Convert the output of the parser to DWARF
    //

    m_dwarf_opcodes.reset(new StreamString);
    m_dwarf_opcodes->SetByteOrder (lldb::eByteOrderHost);
    m_dwarf_opcodes->GetFlags ().Set (Stream::eBinary);
    
    m_local_variables.reset(new ClangExpressionVariableStore());
            
    Error dwarf_error = parser.MakeDWARF ();
    
    if (dwarf_error.Success())
    {
        if (log)
            log->Printf("Code can be interpreted.");
        
        return true;
    }
    
    //////////////////////////////////
    // JIT the output of the parser
    //
    
    m_dwarf_opcodes.reset();
    
    lldb::addr_t jit_end;
    
    Error jit_error = parser.MakeJIT (m_jit_addr, jit_end, exe_ctx);
    
    if (jit_error.Success())
    {
        if (log)
        {
            log->Printf("Code can be run in the target.");
            
            StreamString disassembly_stream;
            
            Error err = parser.DisassembleFunction(disassembly_stream, exe_ctx);
            
            if (!err.Success())
            {
                log->Printf("Couldn't disassemble function : %s", err.AsCString("unknown error"));
            }
            else
            {
                log->Printf("Function disassembly:\n%s", disassembly_stream.GetData());
            }
        }
        
        return true;
    }
    else
    {
        error_stream.Printf ("error: expression can't be interpreted or run\n", num_errors);
        return false;
    }
}

bool
ClangUserExpression::Execute (Stream &error_stream,
                              ExecutionContext &exe_ctx,
                              ClangExpressionVariable *&result)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);

    if (m_dwarf_opcodes.get())
    {
        // TODO execute the JITted opcodes
        
        error_stream.Printf("We don't currently support executing DWARF expressions");
        
        return false;
    }
    else if (m_jit_addr != LLDB_INVALID_ADDRESS)
    {
        lldb::addr_t struct_address;
        
        Error materialize_error;
        
        if (!m_expr_decl_map->Materialize(&exe_ctx, struct_address, materialize_error))
        {
            error_stream.Printf("Couldn't materialize struct: %s\n", materialize_error.AsCString("unknown error"));
            return false;
        }
        
        if (log)
        {
            log->Printf("Function address  : 0x%llx", (uint64_t)m_jit_addr);
            log->Printf("Structure address : 0x%llx", (uint64_t)struct_address);
                    
            StreamString args;
            
            Error dump_error;
            
            if (!m_expr_decl_map->DumpMaterializedStruct(&exe_ctx, args, dump_error))
            {
                log->Printf("Couldn't extract variable values : %s", dump_error.AsCString("unknown error"));
            }
            else
            {
                log->Printf("Structure contents:\n%s", args.GetData());
            }
        }
        
        ClangFunction::ExecutionResults execution_result = 
        ClangFunction::ExecuteFunction (exe_ctx, m_jit_addr, struct_address, true, true, 10000, error_stream);
        
        if (execution_result != ClangFunction::eExecutionCompleted)
        {
            const char *result_name;
            
            switch (execution_result)
            {
                case ClangFunction::eExecutionCompleted:
                    result_name = "eExecutionCompleted";
                    break;
                case ClangFunction::eExecutionDiscarded:
                    result_name = "eExecutionDiscarded";
                    break;
                case ClangFunction::eExecutionInterrupted:
                    result_name = "eExecutionInterrupted";
                    break;
                case ClangFunction::eExecutionSetupError:
                    result_name = "eExecutionSetupError";
                    break;
                case ClangFunction::eExecutionTimedOut:
                    result_name = "eExecutionTimedOut";
                    break;
            }
            
            error_stream.Printf ("Couldn't execute function; result was %s\n", result_name);
            return false;
        }
        
        Error expr_error;
        
        if (!m_expr_decl_map->Dematerialize(&exe_ctx, result, expr_error))
        {
            error_stream.Printf ("Couldn't dematerialize struct : %s\n", expr_error.AsCString("unknown error"));
            return false;
        }
        
        return true;
    }
    else
    {
        error_stream.Printf("Expression can't be run; neither DWARF nor a JIT compiled function are present");
        return false;
    }
}

StreamString &
ClangUserExpression::DwarfOpcodeStream ()
{
    if (!m_dwarf_opcodes.get())
        m_dwarf_opcodes.reset(new StreamString());
    
    return *m_dwarf_opcodes.get();
}
