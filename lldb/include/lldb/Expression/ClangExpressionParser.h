//===-- ClangExpressionParser.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangExpressionParser_h_
#define liblldb_ClangExpressionParser_h_

#include "lldb/lldb-include.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/Error.h"

#include <string>
#include <vector>

namespace llvm
{
    class ExecutionEngine;
}

namespace lldb_private
{

class Process;
class RecordingMemoryManager;
    
//----------------------------------------------------------------------
/// @class ClangExpressionParser ClangExpressionParser.h "lldb/Expression/ClangExpressionParser.h"
/// @brief Encapsulates an instance of Clang that can parse expressions.
///
/// ClangExpressionParser is responsible for preparing an instance of
/// ClangExpression for execution.  ClangExpressionParser uses ClangExpression
/// as a glorified parameter list, performing the required parsing and
/// conversion to formats (DWARF bytecode, or JIT compiled machine code)
/// that can be executed.
//----------------------------------------------------------------------
class ClangExpressionParser
{
public:
    //------------------------------------------------------------------
    /// Constructor
    ///
    /// Initializes class variabes.
    ///
    /// @param[in] target_triple
    ///     The LLVM-friendly target triple for use in initializing the
    ///     compiler.
    ///
    /// @param[in process
    ///     If non-NULL, the process to customize the expression for
    ///     (e.g., by tuning Objective-C runtime support).  May be NULL.
    ///
    /// @param[in] expr
    ///     The expression to be parsed.
    //------------------------------------------------------------------
    ClangExpressionParser (const char *target_triple,
                           Process *process,
                           ClangExpression &expr);
    
    //------------------------------------------------------------------
    /// Destructor
    //------------------------------------------------------------------
    ~ClangExpressionParser ();
    
    //------------------------------------------------------------------
    /// Parse a single expression and convert it to IR using Clang.  Don't
    /// wrap the expression in anything at all.
    ///
    /// @param[in] stream
    ///     The stream to print errors to.
    ///
    /// @return
    ///     The number of errors encountered during parsing.  0 means
    ///     success.
    //------------------------------------------------------------------
    unsigned
    Parse (Stream &stream);
    
    //------------------------------------------------------------------
    /// Convert the IR for an already-parsed expression to DWARF if possible.
    ///
    /// @param[in] dwarf_opcode_strm
    ///     The stream to place the resulting DWARF code into.
    ///
    /// @return
    ///     An error code indicating the success or failure of the operation.
    ///     Test with Success().
    //------------------------------------------------------------------
    Error
    MakeDWARF ();
    
    //------------------------------------------------------------------
    /// JIT-compile the IR for an already-parsed expression.
    ///
    /// @param[out] func_addr
    ///     The address to which the function has been written.
    ///
    /// @param[out] func_end
    ///     The end of the function's allocated memory region.  (func_addr
    ///     and func_end do not delimit an allocated region; the allocated
    ///     region may begin before func_addr.)
    ///
    /// @param[in] exe_ctx
    ///     The execution context to write the function into.
    ///
    /// @param[out] const_result
    ///     If non-NULL, the result of the expression is constant, and the
    ///     expression has no side effects, this is set to the result of the 
    ///     expression.  
    ///
    /// @return
    ///     An error code indicating the success or failure of the operation.
    ///     Test with Success().
    //------------------------------------------------------------------
    Error
    MakeJIT (lldb::addr_t &func_addr,
             lldb::addr_t &func_end,
             ExecutionContext &exe_ctx,
             lldb::ClangExpressionVariableSP *const_result = NULL);
    
    //------------------------------------------------------------------
    /// Disassemble the machine code for a JITted function from the target 
    /// process's memory and print the result to a stream.
    ///
    /// @param[in] stream
    ///     The stream to print disassembly to.
    ///
    /// @param[in] exc_context
    ///     The execution context to get the machine code from.
    ///
    /// @return
    ///     The error generated.  If .Success() is true, disassembly succeeded.
    //------------------------------------------------------------------
    Error
    DisassembleFunction (Stream &stream, ExecutionContext &exc_context);
    
private:
    //----------------------------------------------------------------------
    /// @class JittedFunction ClangExpressionParser.h "lldb/Expression/ClangExpressionParser.h"
    /// @brief Encapsulates a single function that has been generated by the JIT.
    ///
    /// Functions that have been generated by the JIT are first resident in the
    /// local process, and then placed in the target process.  JittedFunction
    /// represents a function possibly resident in both.
    //----------------------------------------------------------------------
    struct JittedFunction {
        std::string m_name;             ///< The function's name
        lldb::addr_t m_local_addr;      ///< The address of the function in LLDB's memory
        lldb::addr_t m_remote_addr;     ///< The address of the function in the target's memory
        
        //------------------------------------------------------------------
        /// Constructor
        ///
        /// Initializes class variabes.
        ///
        /// @param[in] name
        ///     The name of the function.
        ///
        /// @param[in] local_addr
        ///     The address of the function in LLDB, or LLDB_INVALID_ADDRESS if
        ///     it is not present in LLDB's memory.
        ///
        /// @param[in] remote_addr
        ///     The address of the function in the target, or LLDB_INVALID_ADDRESS
        ///     if it is not present in the target's memory.
        //------------------------------------------------------------------
        JittedFunction (const char *name,
                        lldb::addr_t local_addr = LLDB_INVALID_ADDRESS,
                        lldb::addr_t remote_addr = LLDB_INVALID_ADDRESS) :
            m_name (name),
            m_local_addr (local_addr),
            m_remote_addr (remote_addr)
        {
        }
    };
    
    ClangExpression                            &m_expr;                 ///< The expression to be parsed
    
    std::string                                 m_target_triple;        ///< The target triple used to initialize LLVM
    std::auto_ptr<clang::FileManager>           m_file_manager;         ///< The Clang file manager object used by the compiler
    std::auto_ptr<clang::CompilerInstance>      m_compiler;             ///< The Clang compiler used to parse expressions into IR
    std::auto_ptr<clang::Builtin::Context>      m_builtin_context;      ///< Context for Clang built-ins
    std::auto_ptr<clang::SelectorTable>         m_selector_table;       ///< Selector table for Objective-C methods
    std::auto_ptr<clang::ASTContext>            m_ast_context;          ///< The AST context used to hold types and names for the parser
    std::auto_ptr<clang::CodeGenerator>         m_code_generator;       ///< [owned by the Execution Engine] The Clang object that generates IR
    RecordingMemoryManager                     *m_jit_mm;               ///< The memory manager for the LLVM JIT
    std::auto_ptr<llvm::ExecutionEngine>        m_execution_engine;     ///< The LLVM JIT
    std::vector<JittedFunction>                 m_jitted_functions;     ///< A vector of all functions that have been JITted into machine code (just one, if ParseExpression() was called)
};
    
}

#endif  // liblldb_ClangExpressionParser_h_
