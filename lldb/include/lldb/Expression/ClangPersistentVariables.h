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

#include "lldb/lldb-forward-rtti.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Symbol/TaggedASTType.h"

#include <map>
#include <string>

namespace lldb_private
{

//----------------------------------------------------------------------
/// @class ClangPersistentVariable ClangPersistentVariables.h "lldb/Expression/ClangPersistentVariables.h"
/// @brief Encapsulates a persistent value that need to be preserved between expression invocations.
///
/// Although expressions can define truly local variables, frequently the user
/// wants to create variables whose values persist between invocations of the
/// expression.  These variables are also created each time an expression returns
/// a result.  The ClangPersistentVariable class encapsulates such a variable,
/// which contains data and a type.
//----------------------------------------------------------------------
class ClangPersistentVariable 
{
    friend class ClangPersistentVariables;
public:
    //----------------------------------------------------------------------
    /// Constructor
    //----------------------------------------------------------------------
    ClangPersistentVariable () :
        m_name(),
        m_user_type(),
        m_data()
    {
    }
    
    //----------------------------------------------------------------------
    /// Copy constructor
    ///
    /// @param[in] pv
    ///     The persistent variable to make a copy of.
    //----------------------------------------------------------------------
    ClangPersistentVariable (const ClangPersistentVariable &pv) :
        m_name(pv.m_name),
        m_user_type(pv.m_user_type),
        m_data(pv.m_data)
    {
    }
    
    //----------------------------------------------------------------------
    /// Assignment operator
    //----------------------------------------------------------------------
    ClangPersistentVariable &operator=(const ClangPersistentVariable &pv)
    {
        m_name = pv.m_name;
        m_user_type = pv.m_user_type;
        m_data = pv.m_data;
        return *this;
    }
    
    //----------------------------------------------------------------------
    /// Return the number of bytes required to store the variable
    //----------------------------------------------------------------------
    size_t Size ()
    {
        return (m_user_type.GetClangTypeBitWidth () + 7) / 8;
    }
    
    //----------------------------------------------------------------------
    /// Return the variable's contents, in local memory but stored according
    /// to the target's byte order
    //----------------------------------------------------------------------
    uint8_t *Data ()
    {
        return m_data->GetBytes();
    }
    
    //----------------------------------------------------------------------
    /// Return the variable's contents, in local memory but stored in a form
    /// (byte order, etc.) appropriate for copying into the target's memory
    //----------------------------------------------------------------------
    TypeFromUser Type ()
    {
        return m_user_type;
    }
    
    //----------------------------------------------------------------------
    /// Pretty-print the variable
    ///
    /// @param[in] output_stream
    ///     The stream to pretty-print on.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to use when resolving the contents of the
    ///     variable.
    ///
    /// @param[in] format
    ///     The format to print the variable in
    ///
    /// @param[in] show_types
    ///     If true, print the type of the variable
    ///
    /// @param[in] show_summary
    ///     If true, print a summary of the variable's type
    ///
    /// @param[in] verbose
    ///     If true, be verbose in printing the value of the variable
    ///
    /// @return
    ///     An Error describing the result of the operation.  If Error::Success()
    ///     returns true, the pretty printing completed successfully.
    //----------------------------------------------------------------------
    Error Print(Stream &output_stream,
                ExecutionContext &exe_ctx,
                lldb::Format format,
                bool show_types,
                bool show_summary,
                bool verbose);
private:
    //----------------------------------------------------------------------
    /// Constructor
    ///
    /// @param[in] name
    ///     The name of the variable, usually of the form $foo.
    ///
    /// @param[in] user_type
    ///     The type of the variable, in an AST context that will survive
    ///     as long as the variable.
    //----------------------------------------------------------------------
    ClangPersistentVariable (ConstString name, TypeFromUser user_type)
    {
        m_name = name;
        m_user_type = user_type;
        m_data = lldb::DataBufferSP(new DataBufferHeap(Size(), 0));
    }
    
    ConstString         m_name;         ///< The name of the variable, usually $foo.
    TypeFromUser        m_user_type;    ///< The type of the variable.  Must be valid as long as the variable exists.
    lldb::DataBufferSP  m_data;         ///< A shared pointer to the variable's data.  This is a shared pointer so the variable object can move around without excess copying.
};

//----------------------------------------------------------------------
/// @class ClangPersistentVariables ClangPersistentVariables.h "lldb/Expression/ClangPersistentVariables.h"
/// @brief Manages persistent values that need to be preserved between expression invocations.
///
/// A list of variables that can be accessed and updated by any expression.  See
/// ClangPersistentVariable for more discussion.  Also provides an increasing,
/// 0-based counter for naming result variables.
//----------------------------------------------------------------------
class ClangPersistentVariables
{
public:
    //----------------------------------------------------------------------
    /// Create a single named persistent variable
    ///
    /// @param[in] name
    ///     The desired name for the newly-created variable.
    ///
    /// @param[in] user_type
    ///     The desired type for the variable, in a context that will survive
    ///     as long as ClangPersistentVariables.
    ///
    /// @return
    ///     The newly-created persistent variable or NULL if a variable with the
    ///     same name already exists.
    //----------------------------------------------------------------------
    ClangPersistentVariable *CreateVariable (ConstString name, TypeFromUser user_type);
    
    //----------------------------------------------------------------------
    /// Finds a persistent variable in the list.
    ///
    /// @param[in] name
    ///     The name of the requested variable.
    ///
    /// @return
    ///     The variable requested, or NULL if that variable is not in the list.
    //----------------------------------------------------------------------
    ClangPersistentVariable *GetVariable (ConstString name);
    
    //----------------------------------------------------------------------
    /// Return the next entry in the sequence of strings "$0", "$1", ... for use
    /// naming result variables.
    ///
    /// @param[in] name
    ///     A string to place the variable name in.
    //----------------------------------------------------------------------
    void GetNextResultName(std::string &name);
    
    //----------------------------------------------------------------------
    /// Constructor
    //----------------------------------------------------------------------
    ClangPersistentVariables ();
private:
    typedef std::map <ConstString, ClangPersistentVariable>     PVarMap;
    typedef PVarMap::iterator                                   PVarIterator;
    
    PVarMap                                 m_variables;        ///< The backing store for the list of variables.
    uint64_t                                m_result_counter;   ///< The counter used by GetNextResultName().
};

}

#endif
