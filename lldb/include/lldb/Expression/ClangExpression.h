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

namespace llvm
{
    class ExecutionEngine;
    class StringRef;
}

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
    /// Constructor
    ///
    /// Initializes class variabes.
    ///
    /// @param[in] target_triple
    ///     The LLVM-friendly target triple for use in initializing the
    ///     compiler.
    ///
    /// @param[in] expr_decl_map
    ///     The object that looks up externally-defined names in LLDB's
    ///     debug information.
    //------------------------------------------------------------------
    ClangExpression(const char *target_triple,
                    ClangExpressionDeclMap *expr_decl_map);

    //------------------------------------------------------------------
    /// Destructor
    //------------------------------------------------------------------
    ~ClangExpression();
    
    //------------------------------------------------------------------
    /// Parse a single expression and convert it to IR using Clang.  Wrap
    /// the expression in a function with signature void ___clang_expr(void*).
    ///
    /// @param[in] expr_text
    ///     The text of the expression to be parsed.
    ///
    /// @param[in] stream
    ///     The stream to print errors to.
    ///
    /// @param[in] add_result_var
    ///     True if a special result variable should be generated for
    ///     the expression.
    ///
    /// @return
    ///     The number of errors encountered during parsing.  0 means
    ///     success.
    //------------------------------------------------------------------
    unsigned
    ParseExpression (const char *expr_text, 
                     Stream &stream, 
                     bool add_result_var = false);

    //------------------------------------------------------------------
    /// Parse a single expression and convert it to IR using Clang.  Don't
    /// wrap the expression in anything at all.
    ///
    /// @param[in] expr_text
    ///     The text of the expression to be parsed.
    ///
    /// @param[in] stream
    ///     The stream to print errors to.
    ///
    /// @param[in] add_result_var
    ///     True if a special result variable should be generated for
    ///     the expression.
    ///
    /// @return
    ///     The number of errors encountered during parsing.  0 means
    ///     success.
    //------------------------------------------------------------------
    unsigned
    ParseBareExpression (llvm::StringRef expr_text, 
                         Stream &stream, 
                         bool add_result_var = false);
    
    //------------------------------------------------------------------
    /// Convert the IR for an already-parsed expression to DWARF if possible.
    ///
    /// @param[in] expr_local_variable_list
    ///     The list of local variables the expression uses, with types, for
    ///     use by the DWARF parser.
    ///
    /// @param[in] dwarf_opcode_strm
    ///     The stream to place the resulting DWARF code into.
    ///
    /// @return
    ///     True on success; false on failure.  On failure, it may be appropriate
    ///     to call PrepareIRForTarget().
    //------------------------------------------------------------------
    bool
    ConvertIRToDWARF (ClangExpressionVariableList &excpr_local_variable_list,
                      StreamString &dwarf_opcode_strm);
    
    //------------------------------------------------------------------
    /// Prepare the IR for an already-parsed expression for execution in the
    /// target process by (among other things) making all externally-defined
    /// variables point to offsets from the void* argument.
    ///
    /// @return
    ///     True on success; false on failure.  On failure, this expression
    ///     cannot be executed by LLDB.
    //------------------------------------------------------------------
    bool
    PrepareIRForTarget ();

    //------------------------------------------------------------------
    /// Use the JIT to compile an already-prepared expression from IR into
    /// machine code, but keep the code in the current process for now.
    ///
    /// @param[in] func_name
    ///     The name of the function to be JITted.  By default, the function
    ///     wrapped by ParseExpression().
    ///
    /// @return
    ///     True on success; false otherwise.
    //------------------------------------------------------------------
    bool
    JITFunction (const char *func_name = "___clang_expr");

    //------------------------------------------------------------------
    /// Write the machine code generated by the JIT into the target's memory.
    ///
    /// @param[in] exc_context
    ///     The execution context that the JITted code must be copied into.
    ///
    /// @return
    ///     True on success; false otherwise.
    //------------------------------------------------------------------
    bool
    WriteJITCode (const ExecutionContext &exc_context);

    //------------------------------------------------------------------
    /// Write the machine code generated by the JIT into the target process.
    ///
    /// @param[in] func_name
    ///     The name of the function whose address is being requested.
    ///     By default, the function wrapped by ParseExpression().
    ///
    /// @return
    ///     True on success; false otherwise.
    //------------------------------------------------------------------
    lldb::addr_t
    GetFunctionAddress (const char *func_name = "___clang_expr");
    
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
    /// @param[in] func_name
    ///     The name of the function to be disassembled.  By default, the
    ///     function wrapped by ParseExpression().
    ///
    /// @return
    ///     The error generated.  If .Success() is true, disassembly succeeded.
    //------------------------------------------------------------------
    Error
    DisassembleFunction (Stream &stream, ExecutionContext &exc_context, const char *func_name = "___clang_expr");

    //------------------------------------------------------------------
    /// Return the Clang compiler instance being used by this expression.
    //------------------------------------------------------------------
    clang::CompilerInstance *
    GetCompilerInstance ()
    {
        return m_clang_ap.get();
    }

    //------------------------------------------------------------------
    /// Return the AST context being used by this expression.
    //------------------------------------------------------------------
    clang::ASTContext *
    GetASTContext ();

    //------------------------------------------------------------------
    /// Return the mutex being used to serialize access to Clang.
    //------------------------------------------------------------------
    static Mutex &
    GetClangMutex ();
protected:
    //------------------------------------------------------------------
    // Classes that inherit from ClangExpression can see and modify these
    //------------------------------------------------------------------

    //----------------------------------------------------------------------
    /// @class JittedFunction ClangExpression.h "lldb/Expression/ClangExpression.h"
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
            m_remote_addr (remote_addr) {}
    };

    std::string m_target_triple;                                ///< The target triple used to initialize LLVM
    ClangExpressionDeclMap *m_decl_map;                         ///< The class used to look up entities defined in the debug info
    std::auto_ptr<clang::CompilerInstance> m_clang_ap;          ///< The Clang compiler used to parse expressions into IR
    clang::CodeGenerator *m_code_generator_ptr;                 ///< [owned by the Execution Engine] The Clang object that generates IR
    RecordingMemoryManager *m_jit_mm_ptr;                       ///< [owned by the Execution Engine] The memory manager that allocates code pages on the JIT's behalf
    std::auto_ptr<llvm::ExecutionEngine> m_execution_engine;    ///< The LLVM JIT
    std::vector<JittedFunction> m_jitted_functions;             ///< A vector of all functions that have been JITted into machine code (just one, if ParseExpression() was called)
private:
    //------------------------------------------------------------------
    /// Initialize m_clang_ap to a compiler instance with all the options
    /// required by the expression parser.
    ///
    /// @return
    ///     True on success; false otherwise.
    //------------------------------------------------------------------
    bool CreateCompilerInstance();
        
    //------------------------------------------------------------------
    // For ClangExpression only
    //------------------------------------------------------------------
    ClangExpression(const ClangExpression&);
    const ClangExpression& operator=(const ClangExpression&);
};

} // namespace lldb_private

#endif  // liblldb_ClangExpression_h_
