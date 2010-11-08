//===-- IRForTarget.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_IRForTarget_h_
#define liblldb_IRForTarget_h_

#include "llvm/Pass.h"

namespace llvm {
    class BasicBlock;
    class CallInst;
    class Constant;
    class Function;
    class Instruction;
    class Module;
    class Value;
}

namespace lldb_private {
    class ClangExpressionDeclMap;
}

//----------------------------------------------------------------------
/// @class IRForTarget IRForTarget.h "lldb/Expression/IRForTarget.h"
/// @brief Transforms the IR for a function to run in the target
///
/// Once an expression has been parsed and converted to IR, it can run
/// in two contexts: interpreted by LLDB as a DWARF location expression,
/// or compiled by the JIT and inserted into the target process for
/// execution.
///
/// IRForTarget makes the second possible, by applying a series of
/// transformations to the IR which make it relocatable.  These
/// transformations are discussed in more detail next to their relevant
/// functions.
//----------------------------------------------------------------------
class IRForTarget : public llvm::ModulePass
{
public:
    //------------------------------------------------------------------
    /// Constructor
    ///
    /// @param[in] decl_map
    ///     The list of externally-referenced variables for the expression,
    ///     for use in looking up globals and allocating the argument
    ///     struct.  See the documentation for ClangExpressionDeclMap.
    ///
    /// @param[in] func_name
    ///     The name of the function to prepare for execution in the target.
    ///
    /// @param[in] resolve_vars
    ///     True if the external variable references (including persistent
    ///     variables) should be resolved.  If not, only external functions
    ///     are resolved.
    //------------------------------------------------------------------
    IRForTarget(lldb_private::ClangExpressionDeclMap *decl_map,
                bool resolve_vars,
                const char* func_name = "$__lldb_expr");
    
    //------------------------------------------------------------------
    /// Destructor
    //------------------------------------------------------------------
    ~IRForTarget();
    
    //------------------------------------------------------------------
    /// Run this IR transformer on a single module
    ///
    /// @param[in] M
    ///     The module to run on.  This module is searched for the function
    ///     $__lldb_expr, and that function is passed to the passes one by 
    ///     one.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool runOnModule(llvm::Module &M);
    
    //------------------------------------------------------------------
    /// Interface stub
    //------------------------------------------------------------------
    void assignPassManager(llvm::PMStack &PMS,
                           llvm::PassManagerType T = llvm::PMT_ModulePassManager);
    
    //------------------------------------------------------------------
    /// Returns PMT_ModulePassManager
    //------------------------------------------------------------------
    llvm::PassManagerType getPotentialPassManagerType() const;
private:
    //------------------------------------------------------------------
    /// A function-level pass to take the generated global value
    /// $__lldb_expr_result and make it into a persistent variable.
    /// Also see ASTResultSynthesizer.
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    /// The top-level pass implementation
    ///
    /// @param[in] M
    ///     The module currently being processed.
    ///
    /// @param[in] F
    ///     The function currently being processed.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool createResultVariable(llvm::Module &M,
                              llvm::Function &F);

    //------------------------------------------------------------------
    /// A basic block-level pass to find all Objective-C method calls and
    /// rewrite them to use sel_registerName instead of statically allocated
    /// selectors.  The reason is that the selectors are created on the
    /// assumption that the Objective-C runtime will scan the appropriate
    /// section and prepare them.  This doesn't happen when code is copied
    /// into the target, though, and there's no easy way to induce the
    /// runtime to scan them.  So instead we get our selectors from
    /// sel_registerName.
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    /// Replace a single selector reference
    ///
    /// @param[in] selector_load
    ///     The load of the statically-allocated selector.
    ///
    /// @param[in] M
    ///     The module containing the load.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool RewriteObjCSelector(llvm::Instruction* selector_load,
                             llvm::Module &M);
    
    //------------------------------------------------------------------
    /// The top-level pass implementation
    ///
    /// @param[in] M
    ///     The module currently being processed.
    ///
    /// @param[in] BB
    ///     The basic block currently being processed.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool rewriteObjCSelectors(llvm::Module &M, 
                              llvm::BasicBlock &BB);
    
    //------------------------------------------------------------------
    /// A basic block-level pass to find all newly-declared persistent
    /// variables and register them with the ClangExprDeclMap.  This 
    /// allows them to be materialized and dematerialized like normal
    /// external variables.  Before transformation, these persistent
    /// variables look like normal locals, so they have an allocation.
    /// This pass excises these allocations and makes references look
    /// like external references where they will be resolved -- like all
    /// other external references -- by resolveExternals().
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    /// Handle a single allocation of a persistent variable
    ///
    /// @param[in] persistent_alloc
    ///     The allocation of the persistent variable.
    ///
    /// @param[in] M
    ///     The module currently being processed.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool RewritePersistentAlloc(llvm::Instruction *persistent_alloc,
                                llvm::Module &M);
    
    //------------------------------------------------------------------
    /// The top-level pass implementation
    ///
    /// @param[in] M
    ///     The module currently being processed.
    ///
    /// @param[in] BB
    ///     The basic block currently being processed.
    //------------------------------------------------------------------
    bool rewritePersistentAllocs(llvm::Module &M,
                                 llvm::BasicBlock &BB);
    
    //------------------------------------------------------------------
    /// A function-level pass to find all external variables and functions 
    /// used in the IR.  Each found external variable is added to the 
    /// struct, and each external function is resolved in place, its call
    /// replaced with a call to a function pointer whose value is the 
    /// address of the function in the target process.
    //------------------------------------------------------------------
    
    //------------------------------------------------------------------
    /// Handle a single externally-defined variable
    ///
    /// @param[in] M
    ///     The module currently being processed.
    ///
    /// @param[in] V
    ///     The variable.
    ///
    /// @param[in] Store
    ///     True if the access is a store.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool MaybeHandleVariable(llvm::Module &M, 
                             llvm::Value *V);
    
    //------------------------------------------------------------------
    /// Handle all the arguments to a function call
    ///
    /// @param[in] M
    ///     The module currently being processed.
    ///
    /// @param[in] C
    ///     The call instruction.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool MaybeHandleCallArguments(llvm::Module &M,
                                  llvm::CallInst *C);
    
    //------------------------------------------------------------------
    /// Handle a single external function call
    ///
    /// @param[in] M
    ///     The module currently being processed.
    ///
    /// @param[in] C
    ///     The call instruction.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool MaybeHandleCall(llvm::Module &M,
                         llvm::CallInst *C);
    
    //------------------------------------------------------------------
    /// Resolve calls to external functions
    ///
    /// @param[in] M
    ///     The module currently being processed.
    ///
    /// @param[in] BB
    ///     The basic block currently being processed.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool resolveCalls(llvm::Module &M,
                      llvm::BasicBlock &BB);
    
    //------------------------------------------------------------------
    /// The top-level pass implementation
    ///
    /// @param[in] M
    ///     The module currently being processed.
    ///
    /// @param[in] BB
    ///     The function currently being processed.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool resolveExternals(llvm::Module &M,
                          llvm::Function &F);
    
    //------------------------------------------------------------------
    /// A basic block-level pass to excise guard variables from the code.
    /// The result for the function is passed through Clang as a static
    /// variable.  Static variables normally have guard variables to
    /// ensure that they are only initialized once.  
    //------------------------------------------------------------------
    
    //------------------------------------------------------------------
    /// The top-level pass implementation
    ///
    /// @param[in] M
    ///     The module currently being processed.
    ///
    /// @param[in] BB
    ///     The basic block currently being processed.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool removeGuards(llvm::Module &M,
                      llvm::BasicBlock &BB);
    
    //------------------------------------------------------------------
    /// A function-level pass to make all external variable references
    /// point at the correct offsets from the void* passed into the
    /// function.  ClangExpressionDeclMap::DoStructLayout() must be called
    /// beforehand, so that the offsets are valid.
    //------------------------------------------------------------------
    
    //------------------------------------------------------------------
    /// The top-level pass implementation
    ///
    /// @param[in] M
    ///     The module currently being processed.
    ///
    /// @param[in] F
    ///     The function currently being processed.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool replaceVariables(llvm::Module &M,
                          llvm::Function &F);
    
    /// Flags
    bool                                    m_resolve_vars;         ///< True if external variable references and persistent variable references should be resolved
    
    std::string                             m_func_name;            ///< The name of the function to translate
    lldb_private::ClangExpressionDeclMap   *m_decl_map;             ///< The DeclMap containing the Decls 
    llvm::Constant                         *m_sel_registerName;     ///< The address of the function sel_registerName, cast to the appropriate function pointer type
};

#endif
