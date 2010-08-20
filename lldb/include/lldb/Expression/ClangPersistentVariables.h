//===-- ClangPersistentVariables.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangPersistentVariables_h_
#define liblldb_ClangPersistentVariables_h_

#include "lldb/Expression/ClangExpressionVariable.h"

namespace lldb_private
{

//----------------------------------------------------------------------
/// @class ClangPersistentVariables ClangPersistentVariables.h "lldb/Expression/ClangPersistentVariables.h"
/// @brief Manages persistent values that need to be preserved between expression invocations.
///
/// A list of variables that can be accessed and updated by any expression.  See
/// ClangPersistentVariable for more discussion.  Also provides an increasing,
/// 0-based counter for naming result variables.
//----------------------------------------------------------------------
class ClangPersistentVariables : public ClangExpressionVariableStore
{
public:
    //----------------------------------------------------------------------
    /// Return the next entry in the sequence of strings "$0", "$1", ... for use
    /// naming result variables.
    ///
    /// @param[in] name
    ///     A string to place the variable name in.
    //----------------------------------------------------------------------
    void GetNextResultName (std::string &name);
    
    //----------------------------------------------------------------------
    /// Constructor
    //----------------------------------------------------------------------
    ClangPersistentVariables ();

    bool CreatePersistentVariable(const char   *name,
                                  TypeFromUser  user_type);
private:
    uint64_t                        m_result_counter;   ///< The counter used by GetNextResultName().
};

}

#endif
