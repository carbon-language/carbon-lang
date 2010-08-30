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
#include <string>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/ClangForward.h"
#include "lldb/Symbol/TaggedASTType.h"

namespace llvm {
    class Value;
}

namespace lldb_private {

class ClangExpressionVariableStore;
class DataBufferHeap;
class ExecutionContext;
class Stream;
class Value;

//----------------------------------------------------------------------
/// @class ClangExpressionVariable ClangExpressionVariable.h "lldb/Expression/ClangExpressionVariable.h"
/// @brief Encapsulates one variable for the expression parser.
///
/// The expression parser uses variables in three different contexts:
///
/// First, it stores persistent variables along with the process for use
/// in expressions.  These persistent variables contain their own data
/// and are typed.
///
/// Second, in an interpreted expression, it stores the local variables
/// for the expression along with the expression.  These variables
/// contain their own data and are typed.
///
/// Third, in a JIT-compiled expression, it stores the variables that
/// the expression needs to have materialized and dematerialized at each
/// execution.  These do not contain their own data but are named and
/// typed.
///
/// This class supports all of these use cases using simple type
/// polymorphism, and provides necessary support methods.  Its interface
/// is RTTI-neutral.
//----------------------------------------------------------------------
struct ClangExpressionVariable
{
    ClangExpressionVariable();
    
    ClangExpressionVariable(const ClangExpressionVariable &cev);
    
    //----------------------------------------------------------------------
    /// If the variable contains its own data, make a Value point at it
    ///
    /// @param[in] value
    ///     The value to point at the data.
    ///
    /// @return
    ///     True on success; false otherwise (in particular, if this variable
    ///     does not contain its own data).
    //----------------------------------------------------------------------
    bool
    PointValueAtData(Value &value);
    
    //----------------------------------------------------------------------
    /// The following values should stay valid for the life of the variable
    //----------------------------------------------------------------------
    std::string             m_name;         ///< The name of the variable
    TypeFromUser            m_user_type;    ///< The type of the variable according to some LLDB context; NULL if the type hasn't yet been migrated to one
    
    //----------------------------------------------------------------------
    /// The following values indicate where the variable originally came from
    //----------------------------------------------------------------------
    ClangExpressionVariableStore   *m_store;    ///< The store containing the variable
    uint64_t                        m_index;    ///< The index of the variable in the store
    
    //----------------------------------------------------------------------
    /// The following values should not live beyond parsing
    //----------------------------------------------------------------------
    struct ParserVars {
        TypeFromParser          m_parser_type;  ///< The type of the variable according to the parser
        const clang::NamedDecl *m_named_decl;   ///< The Decl corresponding to this variable
        llvm::Value            *m_llvm_value;   ///< The IR value corresponding to this variable; usually a GlobalValue
        lldb_private::Value    *m_lldb_value;   ///< The value found in LLDB for this variable
    };
    std::auto_ptr<ParserVars> m_parser_vars;
    
    //----------------------------------------------------------------------
    /// Make this variable usable by the parser by allocating space for
    /// parser-specific variables
    //----------------------------------------------------------------------
    void EnableParserVars()
    {
        if (!m_parser_vars.get())
            m_parser_vars.reset(new struct ParserVars);
    }
    
    //----------------------------------------------------------------------
    /// Deallocate parser-specific variables
    //----------------------------------------------------------------------
    void DisableParserVars()
    {
        m_parser_vars.reset();
    }
    
    //----------------------------------------------------------------------
    /// The following values are valid if the variable is used by JIT code
    //----------------------------------------------------------------------
    struct JITVars {
        off_t   m_alignment;    ///< The required alignment of the variable, in bytes
        size_t  m_size;         ///< The space required for the variable, in bytes
        off_t   m_offset;       ///< The offset of the variable in the struct, in bytes
    };
    std::auto_ptr<JITVars> m_jit_vars;
    
    //----------------------------------------------------------------------
    /// Make this variable usable for materializing for the JIT by allocating 
    /// space for JIT-specific variables
    //----------------------------------------------------------------------
    void EnableJITVars()
    {
        if (!m_jit_vars.get())
            m_jit_vars.reset(new struct JITVars);
    }
    
    //----------------------------------------------------------------------
    /// Deallocate JIT-specific variables
    //----------------------------------------------------------------------
    void DisableJITVars()
    {
        m_jit_vars.reset();
    }
    
    //----------------------------------------------------------------------
    /// The following values are valid if the value contains its own data
    //----------------------------------------------------------------------
    struct DataVars {
        lldb_private::DataBufferHeap   *m_data; ///< The heap area allocated to contain this variable's data.  Responsibility for deleting this falls to whoever uses the variable last
    };
    std::auto_ptr<DataVars> m_data_vars;
    
    //----------------------------------------------------------------------
    /// Make this variable usable for storing its data internally by
    /// allocating data-specific variables
    //----------------------------------------------------------------------
    void EnableDataVars()
    {
        if (!m_jit_vars.get())
            m_data_vars.reset(new struct DataVars);
    }
    
    //----------------------------------------------------------------------
    /// Deallocate data-specific variables
    //----------------------------------------------------------------------
    void DisableDataVars();
    
    //----------------------------------------------------------------------
    /// Return the variable's size in bytes
    //----------------------------------------------------------------------
    size_t Size ()
    {
        return (m_user_type.GetClangTypeBitWidth () + 7) / 8;
    }
    
    //----------------------------------------------------------------------
    /// Pretty-print the variable, assuming it contains its own data
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
};

//----------------------------------------------------------------------
/// @class ClangExpressionVariableListBase ClangExpressionVariable.h "lldb/Expression/ClangExpressionVariable.h"
/// @brief Manages variables that the expression parser uses.
///
/// The expression parser uses variable lists in various contexts, as
/// discuessed at ClangExpressionVariable.  This abstract class contains
/// the basic functions for managing a list of variables.  Its subclasses
/// store pointers to variables or variables, depending on whether they
/// are backing stores or merely transient repositories.
//----------------------------------------------------------------------
class ClangExpressionVariableListBase
{
public:
    //----------------------------------------------------------------------
    /// Return the number of variables in the list
    //----------------------------------------------------------------------
    virtual uint64_t Size() = 0;
    
    //----------------------------------------------------------------------
    /// Return the variable at the given index in the list
    //----------------------------------------------------------------------
    virtual ClangExpressionVariable &VariableAtIndex(uint64_t index) = 0;
    
    //----------------------------------------------------------------------
    /// Add a new variable and return its index
    //----------------------------------------------------------------------
    virtual uint64_t AddVariable(ClangExpressionVariable& var) = 0;
    
    //----------------------------------------------------------------------
    /// Finds a variable by name in the list.
    ///
    /// @param[in] name
    ///     The name of the requested variable.
    ///
    /// @return
    ///     The variable requested, or NULL if that variable is not in the list.
    //----------------------------------------------------------------------
    ClangExpressionVariable *GetVariable (const char *name)
    {
        for (uint64_t index = 0, size = Size(); index < size; ++index)
        {
            ClangExpressionVariable &candidate (VariableAtIndex(index));
            if (!candidate.m_name.compare(name))
                return &candidate;
        }
        return NULL;
    }
    
    //----------------------------------------------------------------------
    /// Finds a variable by NamedDecl in the list.
    ///
    /// @param[in] name
    ///     The name of the requested variable.
    ///
    /// @return
    ///     The variable requested, or NULL if that variable is not in the list.
    //----------------------------------------------------------------------
    ClangExpressionVariable *GetVariable (const clang::NamedDecl *decl)
    {
        for (uint64_t index = 0, size = Size(); index < size; ++index)
        {
            ClangExpressionVariable &candidate (VariableAtIndex(index));
            if (candidate.m_parser_vars.get() && 
                candidate.m_parser_vars->m_named_decl == decl)
                return &candidate;
        }
        return NULL;
    }
};
    
//----------------------------------------------------------------------
/// @class ClangExpressionVariableListBase ClangExpressionVariable.h "lldb/Expression/ClangExpressionVariable.h"
/// @brief A list of variable references.
///
/// This class stores variables internally, acting as the permanent store.
//----------------------------------------------------------------------
class ClangExpressionVariableStore : public ClangExpressionVariableListBase
{
public:
    //----------------------------------------------------------------------
    /// Implementation of methods in ClangExpressionVariableListBase
    //----------------------------------------------------------------------
    uint64_t Size()
    {
        return m_variables.size();
    }
    
    ClangExpressionVariable &VariableAtIndex(uint64_t index)
    {
        return m_variables[index];
    }
    
    uint64_t AddVariable(ClangExpressionVariable &var)
    {
        m_variables.push_back(var);
        return m_variables.size() - 1;
    }
    
    //----------------------------------------------------------------------
    /// Create a new variable in the list and return its index
    //----------------------------------------------------------------------
    uint64_t CreateVariable()
    {
        uint64_t index = m_variables.size();
        
        m_variables.push_back(ClangExpressionVariable());
        m_variables[index].m_store = this;
        m_variables[index].m_index = index;
        
        return index;
    }
private:
    std::vector <ClangExpressionVariable> m_variables;
};
    
//----------------------------------------------------------------------
/// @class ClangExpressionVariableListBase ClangExpressionVariable.h "lldb/Expression/ClangExpressionVariable.h"
/// @brief A list of variable references.
///
/// This class stores references to variables stored elsewhere.
//----------------------------------------------------------------------
class ClangExpressionVariableList : public ClangExpressionVariableListBase
{
public:
    //----------------------------------------------------------------------
    /// Implementation of methods in ClangExpressionVariableListBase
    //----------------------------------------------------------------------
    uint64_t Size()
    {
        return m_references.size();
    }
    
    ClangExpressionVariable &VariableAtIndex(uint64_t index)
    {
        return m_references[index].first->VariableAtIndex(m_references[index].second);
    }
    
    uint64_t AddVariable(ClangExpressionVariable &var)
    {
        m_references.push_back(ClangExpressionVariableRef(var.m_store, var.m_index));
        return m_references.size() - 1;
    }
private:
    typedef std::pair <ClangExpressionVariableStore *, uint64_t>
    ClangExpressionVariableRef;
    
    std::vector <ClangExpressionVariableRef> m_references;
};

} // namespace lldb_private

#endif  // liblldb_ClangExpressionVariable_h_
