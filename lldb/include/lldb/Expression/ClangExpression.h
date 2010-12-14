//===-- ClangExpression.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangExpression_h_
#define liblldb_ClangExpression_h_

// C Includes
// C++ Includes
#include <string>
#include <map>
#include <memory>
#include <vector>

// Other libraries and framework includes
// Project includes

#include "lldb/lldb-forward.h"
#include "lldb/lldb-private.h"
#include "lldb/Core/ClangForward.h"
#include "llvm/ExecutionEngine/JITMemoryManager.h"

namespace lldb_private {

class RecordingMemoryManager;

//----------------------------------------------------------------------
/// @class ClangExpression ClangExpression.h "lldb/Expression/ClangExpression.h"
/// @brief Encapsulates a single expression for use with Clang
///
/// LLDB uses expressions for various purposes, notably to call functions
/// and as a backend for the expr command.  ClangExpression encapsulates
/// the objects needed to parse and interpret or JIT an expression.  It
/// uses the Clang parser to produce LLVM IR from the expression.
//----------------------------------------------------------------------
class ClangExpression
{
public:
    //------------------------------------------------------------------
    /// Destructor
    //------------------------------------------------------------------
    virtual ~ClangExpression ()
    {
    }
    
    //------------------------------------------------------------------
    /// Return the string that the parser should parse.  Must be a full
    /// translation unit.
    //------------------------------------------------------------------
    virtual const char *
    Text () = 0;
    
    //------------------------------------------------------------------
    /// Return the function name that should be used for executing the
    /// expression.  Text() should contain the definition of this
    /// function.
    //------------------------------------------------------------------
    virtual const char *
    FunctionName () = 0;
    
    //------------------------------------------------------------------
    /// Return the object that the parser should use when resolving external
    /// values.  May be NULL if everything should be self-contained.
    //------------------------------------------------------------------
    virtual ClangExpressionDeclMap *
    DeclMap () = 0;
    
    //------------------------------------------------------------------
    /// Return the object that the parser should use when registering
    /// local variables.  May be NULL if the Expression doesn't care.
    //------------------------------------------------------------------
    virtual ClangExpressionVariableList *
    LocalVariables () = 0;
    
    //------------------------------------------------------------------
    /// Return the object that the parser should allow to access ASTs.
    /// May be NULL if the ASTs do not need to be transformed.
    ///
    /// @param[in] passthrough
    ///     The ASTConsumer that the returned transformer should send
    ///     the ASTs to after transformation.
    //------------------------------------------------------------------
    virtual clang::ASTConsumer *
    ASTTransformer (clang::ASTConsumer *passthrough) = 0;
    
    //------------------------------------------------------------------
    /// Return the stream that the parser should use to write DWARF
    /// opcodes.
    //------------------------------------------------------------------
    virtual StreamString &
    DwarfOpcodeStream () = 0;
    
    //------------------------------------------------------------------
    /// Flags
    //------------------------------------------------------------------
    
    //------------------------------------------------------------------
    /// Return true if validation code should be inserted into the
    /// expression.
    //------------------------------------------------------------------
    virtual bool
    NeedsValidation () = 0;
    
    //------------------------------------------------------------------
    /// Return true if external variables in the expression should be
    /// resolved.
    //------------------------------------------------------------------
    virtual bool
    NeedsVariableResolution () = 0;
};

} // namespace lldb_private

#endif  // liblldb_ClangExpression_h_
