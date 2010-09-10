//===-- ObjCObjectPrinter.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/StreamString.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

#include "lldb/Target/ObjCObjectPrinter.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// ObjCObjectPrinter constructor
//----------------------------------------------------------------------
ObjCObjectPrinter::ObjCObjectPrinter (Process &process) :
    m_process(process)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ObjCObjectPrinter::~ObjCObjectPrinter ()
{
}

bool
ObjCObjectPrinter::PrintObject (Stream &str, Value &object_ptr, ExecutionContext &exe_ctx)
{
    if (!exe_ctx.process)
        return false;
    
    const Address *function_address = GetPrintForDebuggerAddr();

    if (!function_address)
        return false;
    
    const char *target_triple = exe_ctx.process->GetTargetTriple().GetCString();
    ClangASTContext *ast_context = exe_ctx.target->GetScratchClangASTContext();
    
    void *return_qualtype = ast_context->GetCStringType(true);
    Value ret;
    ret.SetContext(Value::eContextTypeOpaqueClangQualType, return_qualtype);
    
    ValueList arg_value_list;
    arg_value_list.PushValue(object_ptr);
    
    ClangFunction func(target_triple, ast_context, return_qualtype, *function_address, arg_value_list);
    StreamString error_stream;
    
    lldb::addr_t wrapper_struct_addr = LLDB_INVALID_ADDRESS;
    func.InsertFunction(exe_ctx, wrapper_struct_addr, error_stream);
    // FIXME: Check result of ExecuteFunction.
    ClangFunction::ExecutionResults results 
        = func.ExecuteFunction(exe_ctx, &wrapper_struct_addr, error_stream, true, 1000, true, ret);
    if (results != ClangFunction::eExecutionCompleted)
    {
        str.Printf("Error evaluating Print Object function: %d.\n", results);
        return false;
    }
       
    addr_t result_ptr = ret.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
    
    // poor man's strcpy
    
    Error error;
    std::vector<char> desc;
    while (1)
    {
        char byte = '\0';
        if (exe_ctx.process->ReadMemory(result_ptr + desc.size(), &byte, 1, error) != 1)
            break;
        
        desc.push_back(byte);

        if (byte == '\0')
            break;
    }
    
    if (!desc.empty())
    {
        str.PutCString(&desc.front());
        return true;
    }
    return false;
}

Address *
ObjCObjectPrinter::GetPrintForDebuggerAddr()
{
    if (!m_PrintForDebugger_addr.get())
    {
        ModuleList &modules = m_process.GetTarget().GetImages();
        
        SymbolContextList contexts;
        SymbolContext context;
        
        if((!modules.FindSymbolsWithNameAndType(ConstString ("_NSPrintForDebugger"), eSymbolTypeCode, contexts)) &&
           (!modules.FindSymbolsWithNameAndType(ConstString ("_CFPrintForDebugger"), eSymbolTypeCode, contexts)))
            return NULL;
        
        contexts.GetContextAtIndex(0, context);
        
        m_PrintForDebugger_addr.reset(new Address(context.symbol->GetValue()));
    }
    
    return m_PrintForDebugger_addr.get();
}

