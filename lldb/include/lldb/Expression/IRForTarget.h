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

#include "lldb/lldb-include.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Symbol/TaggedASTType.h"
#include "llvm/Pass.h"

namespace llvm {
    class BasicBlock;
    class CallInst;
    class Constant;
    class Function;
    class GlobalVariable;
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
                lldb::ClangExpressionVariableSP *const_result,
                const char* func_name = "$__lldb_expr");
    
    //------------------------------------------------------------------
    /// Destructor
    //------------------------------------------------------------------
    ~IRForTarget();
    
    //------------------------------------------------------------------
    /// Run this IR transformer on a single module
    ///
    /// Implementation of the llvm::ModulePass::runOnModule() function.
    ///
    /// @param[in] llvm_module
    ///     The module to run on.  This module is searched for the function
    ///     $__lldb_expr, and that function is passed to the passes one by 
    ///     one.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    virtual bool 
    runOnModule (llvm::Module &llvm_module);
    
    //------------------------------------------------------------------
    /// Interface stub
    ///
    /// Implementation of the llvm::ModulePass::assignPassManager() 
    /// function.
    //------------------------------------------------------------------
    virtual void
    assignPassManager (llvm::PMStack &pass_mgr_stack,
                       llvm::PassManagerType pass_mgr_type = llvm::PMT_ModulePassManager);
    
    //------------------------------------------------------------------
    /// Returns PMT_ModulePassManager
    ///
    /// Implementation of the llvm::ModulePass::getPotentialPassManagerType() 
    /// function.
    //------------------------------------------------------------------
    virtual llvm::PassManagerType 
    getPotentialPassManagerType() const;

private:
    //------------------------------------------------------------------
    /// A function-level pass to check whether the function has side
    /// effects.
    //------------------------------------------------------------------
    
    //------------------------------------------------------------------
    /// The top-level pass implementation
    ///
    /// @param[in] llvm_module
    ///     The module currently being processed.
    ///
    /// @param[in] llvm_function
    ///     The function currently being processed.
    ///
    /// @return
    ///     True if the function has side effects (or if this cannot
    ///     be determined); false otherwise.
    //------------------------------------------------------------------
    bool 
    HasSideEffects (llvm::Module &llvm_module,
                    llvm::Function &llvm_function);
    
    //------------------------------------------------------------------
    /// A function-level pass to take the generated global value
    /// $__lldb_expr_result and make it into a persistent variable.
    /// Also see ASTResultSynthesizer.
    //------------------------------------------------------------------
    
    //------------------------------------------------------------------
    /// Set the constant result variable m_const_result to the provided
    /// constant, assuming it can be evaluated.  The result variable
    /// will be reset to NULL later if the expression has side effects.
    ///
    /// @param[in] initializer
    ///     The constant initializer for the variable.
    ///
    /// @param[in] name
    ///     The name of the result variable.
    ///
    /// @param[in] type
    ///     The Clang type of the result variable.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------    
    void 
    MaybeSetConstantResult (llvm::Constant *initializer,
                            const lldb_private::ConstString &name,
                            lldb_private::TypeFromParser type);
    
    //------------------------------------------------------------------
    /// The top-level pass implementation
    ///
    /// @param[in] llvm_module
    ///     The module currently being processed.
    ///
    /// @param[in] llvm_function
    ///     The function currently being processed.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool 
    CreateResultVariable (llvm::Module &llvm_module,
                          llvm::Function &llvm_function);
    
    //------------------------------------------------------------------
    /// A function-level pass to find Objective-C constant strings and
    /// transform them to calls to CFStringCreateWithBytes.
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    /// Rewrite a single Objective-C constant string.
    ///
    /// @param[in] llvm_module
    ///     The module currently being processed.
    ///
    /// @param[in] NSStr
    ///     The constant NSString to be transformed
    ///
    /// @param[in] CStr
    ///     The constant C string inside the NSString.  This will be
    ///     passed as the bytes argument to CFStringCreateWithBytes.
    ///
    /// @param[in] FirstEntryInstruction
    ///     An instruction early in the execution of the function.
    ///     When this function synthesizes a call to 
    ///     CFStringCreateWithBytes, it places the call before this
    ///     instruction.  The instruction should come before all 
    ///     uses of the NSString.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool 
    RewriteObjCConstString (llvm::Module &llvm_module,
                            llvm::GlobalVariable *NSStr,
                            llvm::GlobalVariable *CStr,
                            llvm::Instruction *FirstEntryInstruction);    
    
    //------------------------------------------------------------------
    /// The top-level pass implementation
    ///
    /// @param[in] llvm_module
    ///     The module currently being processed.
    ///
    /// @param[in] llvm_function
    ///     The function currently being processed.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool 
    RewriteObjCConstStrings (llvm::Module &llvm_module,
                             llvm::Function &llvm_function);

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
    /// @param[in] llvm_module
    ///     The module containing the load.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool 
    RewriteObjCSelector (llvm::Instruction* selector_load,
                         llvm::Module &llvm_module);
    
    //------------------------------------------------------------------
    /// The top-level pass implementation
    ///
    /// @param[in] llvm_module
    ///     The module currently being processed.
    ///
    /// @param[in] basic_block
    ///     The basic block currently being processed.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool 
    RewriteObjCSelectors (llvm::Module &llvm_module, 
                          llvm::BasicBlock &basic_block);
    
    //------------------------------------------------------------------
    /// A basic block-level pass to find all newly-declared persistent
    /// variables and register them with the ClangExprDeclMap.  This 
    /// allows them to be materialized and dematerialized like normal
    /// external variables.  Before transformation, these persistent
    /// variables look like normal locals, so they have an allocation.
    /// This pass excises these allocations and makes references look
    /// like external references where they will be resolved -- like all
    /// other external references -- by ResolveExternals().
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    /// Handle a single allocation of a persistent variable
    ///
    /// @param[in] persistent_alloc
    ///     The allocation of the persistent variable.
    ///
    /// @param[in] llvm_module
    ///     The module currently being processed.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool 
    RewritePersistentAlloc (llvm::Instruction *persistent_alloc,
                            llvm::Module &llvm_module);
    
    //------------------------------------------------------------------
    /// The top-level pass implementation
    ///
    /// @param[in] llvm_module
    ///     The module currently being processed.
    ///
    /// @param[in] basic_block
    ///     The basic block currently being processed.
    //------------------------------------------------------------------
    bool 
    RewritePersistentAllocs (llvm::Module &llvm_module,
                             llvm::BasicBlock &basic_block);
    
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
    /// @param[in] llvm_module
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
    bool 
    MaybeHandleVariable (llvm::Module &llvm_module, 
                         llvm::Value *value);
    
    //------------------------------------------------------------------
    /// Handle all the arguments to a function call
    ///
    /// @param[in] llvm_module
    ///     The module currently being processed.
    ///
    /// @param[in] C
    ///     The call instruction.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool 
    MaybeHandleCallArguments (llvm::Module &llvm_module,
                              llvm::CallInst *call_inst);
    
    //------------------------------------------------------------------
    /// Handle a single external function call
    ///
    /// @param[in] llvm_module
    ///     The module currently being processed.
    ///
    /// @param[in] C
    ///     The call instruction.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool 
    MaybeHandleCall (llvm::Module &llvm_module,
                     llvm::CallInst *C);
    
    //------------------------------------------------------------------
    /// Resolve calls to external functions
    ///
    /// @param[in] llvm_module
    ///     The module currently being processed.
    ///
    /// @param[in] basic_block
    ///     The basic block currently being processed.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool 
    ResolveCalls (llvm::Module &llvm_module,
                  llvm::BasicBlock &basic_block);
    
    //------------------------------------------------------------------
    /// The top-level pass implementation
    ///
    /// @param[in] llvm_module
    ///     The module currently being processed.
    ///
    /// @param[in] basic_block
    ///     The function currently being processed.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool 
    ResolveExternals (llvm::Module &llvm_module,
                      llvm::Function &llvm_function);
    
    //------------------------------------------------------------------
    /// A basic block-level pass to excise guard variables from the code.
    /// The result for the function is passed through Clang as a static
    /// variable.  Static variables normally have guard variables to
    /// ensure that they are only initialized once.  
    //------------------------------------------------------------------
    
    //------------------------------------------------------------------
    /// The top-level pass implementation
    ///
    /// @param[in] llvm_module
    ///     The module currently being processed.
    ///
    /// @param[in] basic_block
    ///     The basic block currently being processed.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool 
    RemoveGuards (llvm::Module &llvm_module,
                  llvm::BasicBlock &basic_block);
    
    //------------------------------------------------------------------
    /// A function-level pass to make all external variable references
    /// point at the correct offsets from the void* passed into the
    /// function.  ClangExpressionDeclMap::DoStructLayout() must be called
    /// beforehand, so that the offsets are valid.
    //------------------------------------------------------------------
    
    //------------------------------------------------------------------
    /// The top-level pass implementation
    ///
    /// @param[in] llvm_module
    ///     The module currently being processed.
    ///
    /// @param[in] llvm_function
    ///     The function currently being processed.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool 
    ReplaceVariables (llvm::Module &llvm_module,
                      llvm::Function &llvm_function);
    
    /// Flags
    bool                                    m_resolve_vars;             ///< True if external variable references and persistent variable references should be resolved
    std::string                             m_func_name;                ///< The name of the function to translate
    lldb_private::ConstString               m_result_name;              ///< The name of the result variable ($0, $1, ...)
    lldb_private::ClangExpressionDeclMap   *m_decl_map;                 ///< The DeclMap containing the Decls 
    llvm::Constant                         *m_CFStringCreateWithBytes;  ///< The address of the function CFStringCreateWithBytes, cast to the appropriate function pointer type
    llvm::Constant                         *m_sel_registerName;         ///< The address of the function sel_registerName, cast to the appropriate function pointer type
    lldb::ClangExpressionVariableSP        *m_const_result;             ///< If non-NULL, this value should be set to the return value of the expression if it is constant and the expression has no side effects
    
    bool                                    m_has_side_effects;         ///< True if the function's result cannot be simply determined statically
    bool                                    m_result_is_pointer;        ///< True if the function's result in the AST is a pointer (see comments in ASTResultSynthesizer::SynthesizeBodyResult)
    
private:
    //------------------------------------------------------------------
    /// UnfoldConstant operates on a constant [Old] which has just been 
    /// replaced with a value [New].  We assume that new_value has 
    /// been properly placed early in the function, in front of the 
    /// first instruction in the entry basic block 
    /// [FirstEntryInstruction].  
    ///
    /// UnfoldConstant reads through the uses of Old and replaces Old 
    /// in those uses with New.  Where those uses are constants, the 
    /// function generates new instructions to compute the result of the 
    /// new, non-constant expression and places them before 
    /// FirstEntryInstruction.  These instructions replace the constant
    /// uses, so UnfoldConstant calls itself recursively for those.
    ///
    /// @param[in] llvm_module
    ///     The module currently being processed.
    ///
    /// @param[in] llvm_function
    ///     The function currently being processed.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    static bool 
    UnfoldConstant (llvm::Constant *old_constant, 
                    llvm::Value *new_constant, 
                    llvm::Instruction *first_entry_inst);
};

#endif
