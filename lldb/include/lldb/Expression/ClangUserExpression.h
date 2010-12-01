//===-- ClangUserExpression.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangUserExpression_h_
#define liblldb_ClangUserExpression_h_

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
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ClangExpressionVariable.h"
#include "lldb/Symbol/TaggedASTType.h"
#include "lldb/Target/Process.h"

#include "llvm/ExecutionEngine/JITMemoryManager.h"

namespace lldb_private 
{

//----------------------------------------------------------------------
/// @class ClangUserExpression ClangUserExpression.h "lldb/Expression/ClangUserExpression.h"
/// @brief Encapsulates a single expression for use with Clang
///
/// LLDB uses expressions for various purposes, notably to call functions
/// and as a backend for the expr command.  ClangUserExpression encapsulates
/// the objects needed to parse and interpret or JIT an expression.  It
/// uses the Clang parser to produce LLVM IR from the expression.
//----------------------------------------------------------------------
class ClangUserExpression : public ClangExpression
{
public:
    typedef lldb::SharedPtr<ClangUserExpression>::Type ClangUserExpressionSP;

    //------------------------------------------------------------------
    /// Constructor
    ///
    /// @param[in] expr
    ///     The expression to parse.
    ///
    /// @param[in] expr_prefix
    ///     If non-NULL, a C string containing translation-unit level
    ///     definitions to be included when the expression is parsed.
    //------------------------------------------------------------------
    ClangUserExpression (const char *expr,
                         const char *expr_prefix);
    
    //------------------------------------------------------------------
    /// Destructor
    //------------------------------------------------------------------
    virtual 
    ~ClangUserExpression ();
    
    //------------------------------------------------------------------
    /// Parse the expression
    ///
    /// @param[in] error_stream
    ///     A stream to print parse errors and warnings to.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to use when looking up entities that
    ///     are needed for parsing (locations of functions, types of
    ///     variables, persistent variables, etc.)
    ///
    /// @param[in] desired_type
    ///     The type that the expression should be coerced to.  If NULL,
    ///     inferred from the expression itself.
    ///
    /// @return
    ///     True on success (no errors); false otherwise.
    //------------------------------------------------------------------
    bool
    Parse (Stream &error_stream, 
           ExecutionContext &exe_ctx,
           TypeFromUser desired_type);
    
    //------------------------------------------------------------------
    /// Execute the parsed expression
    ///
    /// @param[in] error_stream
    ///     A stream to print errors to.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to use when looking up entities that
    ///     are needed for parsing (locations of variables, etc.)
    ///
    /// @param[in] discard_on_error
    ///     If true, and the execution stops before completion, we unwind the
    ///     function call, and return the program state to what it was before the
    ///     execution.  If false, we leave the program in the stopped state.
    /// @param[in] shared_ptr_to_me
    ///     This is a shared pointer to this ClangUserExpression.  This is
    ///     needed because Execute can push a thread plan that will hold onto
    ///     the ClangUserExpression for an unbounded period of time.  So you
    ///     need to give the thread plan a reference to this object that can 
    ///     keep it alive.
    /// 
    /// @param[in] result
    ///     A pointer to direct at the persistent variable in which the
    ///     expression's result is stored.
    ///
    /// @return
    ///     A Process::Execution results value.
    //------------------------------------------------------------------
    Process::ExecutionResults
    Execute (Stream &error_stream,
             ExecutionContext &exe_ctx,
             bool discard_on_error,
             ClangUserExpressionSP &shared_ptr_to_me,
             ClangExpressionVariable *&result);
             
    ThreadPlan *
    GetThreadPlanToExecuteJITExpression (Stream &error_stream,
                                         ExecutionContext &exe_ctx);
    bool
    FinalizeJITExecution (Stream &error_stream,
                          ExecutionContext &exe_ctx,
                          ClangExpressionVariable *&result);
    
    //------------------------------------------------------------------
    /// Return the string that the parser should parse.  Must be a full
    /// translation unit.
    //------------------------------------------------------------------
    const char *
    Text ()
    {
        return m_transformed_text.c_str();
    }
    
    //------------------------------------------------------------------
    /// Return the string that the user typed.
    //------------------------------------------------------------------
    const char *
    GetUserText ()
    {
        return m_expr_text.c_str();
    }
    
    //------------------------------------------------------------------
    /// Return the function name that should be used for executing the
    /// expression.  Text() should contain the definition of this
    /// function.
    //------------------------------------------------------------------
    const char *
    FunctionName ()
    {
        return "$__lldb_expr";
    }
    
    //------------------------------------------------------------------
    /// Return the object that the parser should use when resolving external
    /// values.  May be NULL if everything should be self-contained.
    //------------------------------------------------------------------
    ClangExpressionDeclMap *
    DeclMap ()
    {
        return m_expr_decl_map.get();
    }
    
    //------------------------------------------------------------------
    /// Return the object that the parser should use when registering
    /// local variables.  May be NULL if the Expression doesn't care.
    //------------------------------------------------------------------
    ClangExpressionVariableStore *
    LocalVariables ()
    {
        return m_local_variables.get();
    }
    
    //------------------------------------------------------------------
    /// Return the object that the parser should allow to access ASTs.
    /// May be NULL if the ASTs do not need to be transformed.
    ///
    /// @param[in] passthrough
    ///     The ASTConsumer that the returned transformer should send
    ///     the ASTs to after transformation.
    //------------------------------------------------------------------
    clang::ASTConsumer *
    ASTTransformer (clang::ASTConsumer *passthrough);
    
    //------------------------------------------------------------------
    /// Return the stream that the parser should use to write DWARF
    /// opcodes.
    //------------------------------------------------------------------
    StreamString &
    DwarfOpcodeStream ();
    
    //------------------------------------------------------------------
    /// Return true if validation code should be inserted into the
    /// expression.
    //------------------------------------------------------------------
    bool
    NeedsValidation ()
    {
        return true;
    }
    
    //------------------------------------------------------------------
    /// Return true if external variables in the expression should be
    /// resolved.
    //------------------------------------------------------------------
    bool
    NeedsVariableResolution ()
    {
        return true;
    }

    //------------------------------------------------------------------
    /// Evaluate one expression and return its result.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to use when evaluating the expression.
    ///
    /// @param[in] expr_cstr
    ///     A C string containing the expression to be evaluated.
    ///
    /// @param[in] expr_prefix
    ///     If non-NULL, a C string containing translation-unit level
    ///     definitions to be included when the expression is parsed.
    ///
    /// @param[in/out] result_valobj_sp
    ///      If execution is successful, the result valobj is placed here.
    ///
    /// @result
    ///      A Process::ExecutionResults value.  eExecutionCompleted for success.
    //------------------------------------------------------------------
    static Process::ExecutionResults
    Evaluate (ExecutionContext &exe_ctx, 
              bool discard_on_error,
              const char *expr_cstr,
              const char *expr_prefix,
              lldb::ValueObjectSP &result_valobj_sp);

private:
    //------------------------------------------------------------------
    /// Populate m_cplusplus and m_objetivec based on the environment.
    //------------------------------------------------------------------
    void
    ScanContext(ExecutionContext &exe_ctx);

    bool
    PrepareToExecuteJITExpression (Stream &error_stream,
                                   ExecutionContext &exe_ctx,
                                   lldb::addr_t &struct_address,
                                   lldb::addr_t &object_ptr);
    
    std::string                                 m_expr_text;            ///< The text of the expression, as typed by the user
    std::string                                 m_expr_prefix;          ///< The text of the translation-level definitions, as provided by the user
    std::string                                 m_transformed_text;     ///< The text of the expression, as send to the parser
    TypeFromUser                                m_desired_type;         ///< The type to coerce the expression's result to.  If NULL, inferred from the expression.
    
    std::auto_ptr<ClangExpressionDeclMap>       m_expr_decl_map;        ///< The map to use when parsing and materializing the expression.
    std::auto_ptr<ClangExpressionVariableStore> m_local_variables;      ///< The local expression variables, if the expression is DWARF.
    std::auto_ptr<StreamString>                 m_dwarf_opcodes;        ///< The DWARF opcodes for the expression.  May be NULL.
    lldb::addr_t                                m_jit_addr;             ///< The address of the JITted code.  LLDB_INVALID_ADDRESS if invalid.
    
    bool                                        m_cplusplus;            ///< True if the expression is compiled as a C++ member function (true if it was parsed when exe_ctx was in a C++ method).
    bool                                        m_objectivec;           ///< True if the expression is compiled as an Objective-C method (true if it was parsed when exe_ctx was in an Objective-C method).
    bool                                        m_needs_object_ptr;     ///< True if "this" or "self" must be looked up and passed in.  False if the expression doesn't really use them and they can be NULL.
    bool                                        m_const_object;         ///< True if "this" is const.
};
    
} // namespace lldb_private

#endif  // liblldb_ClangUserExpression_h_
