//===-- ClangExpressionVariable.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangExpressionVariable_h_
#define liblldb_ClangExpressionVariable_h_

// C Includes
#include <signal.h>
#include <stdint.h>

// C++ Includes
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/Value.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class ClangExpressionVariableList ClangExpressionVariable.h "lldb/Expression/ClangExpressionVariable.h"
/// @brief Manages local variables that the expression interpreter uses.
///
/// The DWARF interpreter, when interpreting expressions, occasionally
/// needs to interact with chunks of memory corresponding to local variable 
/// values.  These locals are distinct from the externally-defined values
/// handled by ClangExpressionDeclMap, and do not persist between expressions
/// so they are not handled by ClangPersistentVariables.  They are kept in a 
/// list, which is encapsulated in ClangEpxressionVariableList.
//----------------------------------------------------------------------
class ClangExpressionVariableList
{
public:
    //----------------------------------------------------------------------
    /// Constructor
    //----------------------------------------------------------------------
    ClangExpressionVariableList();
    
    //----------------------------------------------------------------------
    /// Destructor
    //----------------------------------------------------------------------
    ~ClangExpressionVariableList();

    //----------------------------------------------------------------------
    /// Get or create the chunk of data corresponding to a given VarDecl.
    ///
    /// @param[in] var_decl
    ///     The Decl for which a chunk of memory is to be allocated.
    ///
    /// @param[out] idx
    ///     The index of the Decl in the list of variables.
    ///
    /// @param[in] can_create
    ///     True if the memory should be created if necessary.
    ///
    /// @return
    ///     A Value for the allocated memory.  NULL if the Decl couldn't be 
    ///     found and can_create was false, or if some error occurred during
    ///     allocation.
    //----------------------------------------------------------------------
    Value *
    GetVariableForVarDecl (const clang::VarDecl *var_decl, 
                           uint32_t& idx, 
                           bool can_create);

    //----------------------------------------------------------------------
    /// Get the chunk of data corresponding to a given index into the list.
    ///
    /// @param[in] idx
    ///     The index of the Decl in the list of variables.
    ///
    /// @return
    ///     The value at the given index, or NULL if there is none.
    //----------------------------------------------------------------------
    Value *
    GetVariableAtIndex (uint32_t idx);
private:
    //----------------------------------------------------------------------
    /// @class ClangExpressionVariable ClangExpressionVariable.h "lldb/Expression/ClangExpressionVariable.h"
    /// @brief Manages one local variable for the expression interpreter.
    ///
    /// The expression interpreter uses specially-created Values to hold its
    /// temporary locals.  These Values contain data buffers holding enough
    /// space to contain a variable of the appropriate type.  The VarDecls
    /// are only used while creating the list and generating the DWARF code for
    /// an expression; when interpreting the DWARF, the variables are identified
    /// only by their index into the list of variables.
    //----------------------------------------------------------------------
    struct ClangExpressionVariable
    {
        const clang::VarDecl    *m_var_decl;    ///< The VarDecl corresponding to the parsed local.
        Value                   *m_value;       ///< The LLDB Value containing the data for the local.
    };
    
    typedef std::vector<ClangExpressionVariable> Variables;
    Variables m_variables;                                  ///< The list of variables used by the expression.
};

} // namespace lldb_private

#endif  // liblldb_ClangExpressionVariable_h_
