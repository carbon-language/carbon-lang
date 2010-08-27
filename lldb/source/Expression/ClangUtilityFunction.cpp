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

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Stream.h"
#include "lldb/Expression/ClangExpressionParser.h"
#include "lldb/Expression/ClangUtilityFunction.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Target.h"

using namespace lldb_private;

//------------------------------------------------------------------
/// Constructor
///
/// @param[in] text
///     The text of the function.  Must be a full translation unit.
///
/// @param[in] name
///     The name of the function, as used in the text.
//------------------------------------------------------------------
ClangUtilityFunction::ClangUtilityFunction (const char *text, 
                                            const char *name) :
    m_function_text(text),
    m_function_name(name)
{
}

//------------------------------------------------------------------
/// Install the utility function into a process
///
/// @param[in] error_stream
///     A stream to print parse errors and warnings to.
///
/// @param[in] exe_ctx
///     The execution context to install the utility function to.
///
/// @return
///     True on success (no errors); false otherwise.
//------------------------------------------------------------------
bool
ClangUtilityFunction::Install (Stream &error_stream,
                               ExecutionContext &exe_ctx)
{    
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
        
    ClangExpressionParser parser(target_triple.GetCString(), *this);
    
    unsigned num_errors = parser.Parse (error_stream);
    
    if (num_errors)
    {
        error_stream.Printf ("error: %d errors parsing expression\n", num_errors);
        return false;
    }
    
    //////////////////////////////////
    // JIT the output of the parser
    //
        
    Error jit_error = parser.MakeJIT (m_jit_begin, m_jit_end, exe_ctx);
    
    if (jit_error.Success())
    {
        return true;
    }
    else
    {
        error_stream.Printf ("error: expression can't be interpreted or run\n", num_errors);
        return false;
    }
}


