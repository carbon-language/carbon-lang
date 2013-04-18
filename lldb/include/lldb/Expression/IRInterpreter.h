//===-- IRInterpreter.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_IRInterpreter_h_
#define liblldb_IRInterpreter_h_

#include "lldb/lldb-public.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/TaggedASTType.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Pass.h"

namespace llvm {
    class Function;
    class Module;
}

namespace lldb_private {

class ClangExpressionDeclMap;
class IRMemoryMap;
    
}

//----------------------------------------------------------------------
/// @class IRInterpreter IRInterpreter.h "lldb/Expression/IRInterpreter.h"
/// @brief Attempt to interpret the function's code if it does not require
///        running the target.
///
/// In some cases, the IR for an expression can be evaluated entirely
/// in the debugger, manipulating variables but not executing any code
/// in the target.  The IRInterpreter attempts to do this.
//----------------------------------------------------------------------
class IRInterpreter
{
public:
    //------------------------------------------------------------------
    /// Run the IR interpreter on a single function
    ///
    /// @param[in] result
    ///     This variable is populated with the return value of the
    ///     function, if it could be interpreted completely.
    ///
    /// @param[in] result_name
    ///     The name of the result in the IR.  If this name got a
    ///     value written to it as part of execution, then that value
    ///     will be used to create the result variable.
    ///
    /// @param[in] result_type
    ///     The type of the result.
    ///
    /// @param[in] llvm_function
    ///     The function to interpret.
    ///
    /// @param[in] llvm_module
    ///     The module containing the function.
    ///
    /// @param[in] error
    ///     If the expression fails to interpret, a reason why.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    static bool
    maybeRunOnFunction (lldb_private::ClangExpressionDeclMap *decl_map,
                        lldb_private::IRMemoryMap &memory_map,
                        lldb_private::Stream *error_stream,
                        lldb::ClangExpressionVariableSP &result,
                        const lldb_private::ConstString &result_name,
                        lldb_private::TypeFromParser result_type,
                        llvm::Function &llvm_function,
                        llvm::Module &llvm_module,
                        lldb_private::Error &err);
    
    // new api
    
    static bool
    CanInterpret (llvm::Module &module,
                  llvm::Function &function,
                  lldb_private::Error &error);
    
    static bool
    Interpret (llvm::Module &module,
               llvm::Function &function,
               llvm::ArrayRef<lldb::addr_t> args,
               lldb_private::IRMemoryMap &memory_map,
               lldb_private::Error &error);
                  
private:   
    static bool
    supportsFunction (llvm::Function &llvm_function,
                      lldb_private::Error &err);
    
    static bool
    runOnFunction (lldb_private::ClangExpressionDeclMap *decl_map,
                   lldb_private::IRMemoryMap &memory_map,
                   lldb_private::Stream *error_stream,
                   lldb::ClangExpressionVariableSP &result,
                   const lldb_private::ConstString &result_name,
                   lldb_private::TypeFromParser result_type,
                   llvm::Function &llvm_function,
                   llvm::Module &llvm_module,
                   lldb_private::Error &err);
};

#endif
