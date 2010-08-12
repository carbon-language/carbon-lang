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

class ClangExpression
{
public:

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    ClangExpression(const char *target_triple,
                    ClangExpressionDeclMap *expr_decl_map);

    ~ClangExpression();

    unsigned Compile();

    unsigned
    ParseExpression (const char *expr_text, 
                     Stream &stream, 
                     bool add_result_var = false);

    unsigned
    ParseBareExpression (llvm::StringRef expr_text, 
                         Stream &stream, 
                         bool add_result_var = false);
    
    bool
    ConvertIRToDWARF (ClangExpressionVariableList &excpr_local_variable_list,
                      StreamString &dwarf_opcode_strm);
    
    bool
    PrepareIRForTarget (ClangExpressionVariableList &excpr_local_variable_list);

    bool
    JITFunction (const ExecutionContext &exc_context, const char *func_name);

    bool
    WriteJITCode (const ExecutionContext &exc_context);

    lldb::addr_t
    GetFunctionAddress (const char *name);
    
    Error
    DisassembleFunction (Stream &stream, ExecutionContext &exc_context, const char *name);

    clang::CompilerInstance *
    GetCompilerInstance ()
    {
        return m_clang_ap.get();
    }

    clang::ASTContext *
    GetASTContext ();

    static Mutex &
    GetClangMutex ();
protected:

    // This class is a pass-through for the default JIT memory manager,
    // which just records the memory regions that were handed out so we
    // can copy them into the target later on.


    //------------------------------------------------------------------
    // Classes that inherit from ClangExpression can see and modify these
    //------------------------------------------------------------------

    struct JittedFunction {
        std::string m_name;
        lldb::addr_t m_local_addr;
        lldb::addr_t m_remote_addr;

        JittedFunction (const char *name,
                           lldb::addr_t local_addr = LLDB_INVALID_ADDRESS,
                           lldb::addr_t remote_addr = LLDB_INVALID_ADDRESS) :
            m_name (name),
            m_local_addr (local_addr),
            m_remote_addr (remote_addr) {}
    };

    std::string m_target_triple;
    ClangExpressionDeclMap *m_decl_map;
    std::auto_ptr<clang::CompilerInstance> m_clang_ap;
    clang::CodeGenerator *m_code_generator_ptr;  // This will be deleted by the Execution Engine.
    RecordingMemoryManager *m_jit_mm_ptr;        // This will be deleted by the Execution Engine.
    std::auto_ptr<llvm::ExecutionEngine> m_execution_engine;
    std::vector<JittedFunction> m_jitted_functions;
private:
    
    bool CreateCompilerInstance(bool &IsAST);
        
    //------------------------------------------------------------------
    // For ClangExpression only
    //------------------------------------------------------------------
    ClangExpression(const ClangExpression&);
    const ClangExpression& operator=(const ClangExpression&);
};

} // namespace lldb_private

#endif  // liblldb_ClangExpression_h_
