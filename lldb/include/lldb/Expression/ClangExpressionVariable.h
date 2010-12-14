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
#include "lldb/lldb-include.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Symbol/TaggedASTType.h"

namespace llvm {
    class Value;
}

namespace lldb_private {

class ClangExpressionVariableList;
class ValueObjectConstResult;

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
class ClangExpressionVariable
{
public:
    ClangExpressionVariable(lldb::ByteOrder byte_order, uint32_t addr_byte_size);

    ClangExpressionVariable(const lldb::ValueObjectSP &valobj_sp);

    //----------------------------------------------------------------------
    /// If the variable contains its own data, make a Value point at it.
    /// If \a exe_ctx in not NULL, the value will be resolved in with
    /// that execution context.
    ///
    /// @param[in] value
    ///     The value to point at the data.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to use to resolve \a value.
    ///
    /// @return
    ///     True on success; false otherwise (in particular, if this variable
    ///     does not contain its own data).
    //----------------------------------------------------------------------
    bool
    PointValueAtData(Value &value, ExecutionContext *exe_ctx);
    
    lldb::ValueObjectSP
    GetValueObject();

    //----------------------------------------------------------------------
    /// The following values should not live beyond parsing
    //----------------------------------------------------------------------
    struct ParserVars {

        ParserVars() :
            m_parser_type(),
            m_named_decl (NULL),
            m_llvm_value (NULL),
            m_lldb_value (NULL)
        {
        }

        TypeFromParser          m_parser_type;  ///< The type of the variable according to the parser
        const clang::NamedDecl *m_named_decl;   ///< The Decl corresponding to this variable
        llvm::Value            *m_llvm_value;   ///< The IR value corresponding to this variable; usually a GlobalValue
        lldb_private::Value    *m_lldb_value;   ///< The value found in LLDB for this variable
    };
    //----------------------------------------------------------------------
    /// Make this variable usable by the parser by allocating space for
    /// parser-specific variables
    //----------------------------------------------------------------------
    void 
    EnableParserVars()
    {
        if (!m_parser_vars.get())
            m_parser_vars.reset(new struct ParserVars);
    }
    
    //----------------------------------------------------------------------
    /// Deallocate parser-specific variables
    //----------------------------------------------------------------------
    void
    DisableParserVars()
    {
        m_parser_vars.reset();
    }
    
    //----------------------------------------------------------------------
    /// The following values are valid if the variable is used by JIT code
    //----------------------------------------------------------------------
    struct JITVars {
        JITVars () :
            m_alignment (0),
            m_size (0),
            m_offset (0)
        {
        }

        off_t   m_alignment;    ///< The required alignment of the variable, in bytes
        size_t  m_size;         ///< The space required for the variable, in bytes
        off_t   m_offset;       ///< The offset of the variable in the struct, in bytes
    };

    //----------------------------------------------------------------------
    /// Make this variable usable for materializing for the JIT by allocating 
    /// space for JIT-specific variables
    //----------------------------------------------------------------------
    void 
    EnableJITVars()
    {
        if (!m_jit_vars.get())
            m_jit_vars.reset(new struct JITVars);
    }
    
    //----------------------------------------------------------------------
    /// Deallocate JIT-specific variables
    //----------------------------------------------------------------------
    void 
    DisableJITVars()
    {
        m_jit_vars.reset();
    }
        
    //----------------------------------------------------------------------
    /// Return the variable's size in bytes
    //----------------------------------------------------------------------
    size_t 
    GetByteSize ();

    const ConstString &
    GetName();

    lldb::RegisterInfo *
    GetRegisterInfo();
    
    void
    SetRegisterInfo (const lldb::RegisterInfo *reg_info);

    lldb::clang_type_t
    GetClangType ();
    
    void
    SetClangType (lldb::clang_type_t);

    clang::ASTContext *
    GetClangAST ();
    
    void
    SetClangAST (clang::ASTContext *ast);

    TypeFromUser
    GetTypeFromUser ();

    uint8_t *
    GetValueBytes ();
    
    void
    SetName (const ConstString &name);

    void
    ValueUpdated ();


    typedef lldb::SharedPtr<ValueObjectConstResult>::Type ValueObjectConstResultSP;

    //----------------------------------------------------------------------
    /// Members
    //----------------------------------------------------------------------
    std::auto_ptr<ParserVars> m_parser_vars;
    std::auto_ptr<JITVars> m_jit_vars;
    //ValueObjectConstResultSP m_valojb_sp;
    lldb::ValueObjectSP m_valojb_sp;

private:
    DISALLOW_COPY_AND_ASSIGN (ClangExpressionVariable);
};

//----------------------------------------------------------------------
/// @class ClangExpressionVariableListBase ClangExpressionVariable.h "lldb/Expression/ClangExpressionVariable.h"
/// @brief A list of variable references.
///
/// This class stores variables internally, acting as the permanent store.
//----------------------------------------------------------------------
class ClangExpressionVariableList
{
public:
    //----------------------------------------------------------------------
    /// Implementation of methods in ClangExpressionVariableListBase
    //----------------------------------------------------------------------
    virtual size_t 
    GetSize()
    {
        return m_variables.size();
    }
    
    virtual lldb::ClangExpressionVariableSP
    GetVariableAtIndex(size_t index)
    {
        lldb::ClangExpressionVariableSP var_sp;
        if (index < m_variables.size())
            var_sp = m_variables[index];
        return var_sp;
    }
    
    virtual size_t
    AddVariable (const lldb::ClangExpressionVariableSP &var_sp)
    {
        m_variables.push_back(var_sp);
        return m_variables.size() - 1;
    }

    //----------------------------------------------------------------------
    /// Finds a variable by name in the list.
    ///
    /// @param[in] name
    ///     The name of the requested variable.
    ///
    /// @return
    ///     The variable requested, or NULL if that variable is not in the list.
    //----------------------------------------------------------------------
    lldb::ClangExpressionVariableSP
    GetVariable (const ConstString &name)
    {
        lldb::ClangExpressionVariableSP var_sp;
        for (size_t index = 0, size = GetSize(); index < size; ++index)
        {
            var_sp = GetVariableAtIndex(index);
            if (var_sp->GetName() == name)
                return var_sp;
        }
        var_sp.reset();
        return var_sp;
    }

    lldb::ClangExpressionVariableSP
    GetVariable (const char *name)
    {
        lldb::ClangExpressionVariableSP var_sp;
        if (name && name[0])
        {
            for (size_t index = 0, size = GetSize(); index < size; ++index)
            {
                var_sp = GetVariableAtIndex(index);
                const char *var_name_cstr = var_sp->GetName().GetCString();
                if (::strcmp (var_name_cstr, name) == 0)
                    return var_sp;
            }
            var_sp.reset();
        }
        return var_sp;
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
    lldb::ClangExpressionVariableSP
    GetVariable (const clang::NamedDecl *decl)
    {
        lldb::ClangExpressionVariableSP var_sp;
        for (size_t index = 0, size = GetSize(); index < size; ++index)
        {
            var_sp = GetVariableAtIndex(index);
            if (var_sp->m_parser_vars.get() && var_sp->m_parser_vars->m_named_decl == decl)
                return var_sp;
        }
        var_sp.reset();
        return var_sp;
    }

    //----------------------------------------------------------------------
    /// Create a new variable in the list and return its index
    //----------------------------------------------------------------------
    lldb::ClangExpressionVariableSP
    CreateVariable (lldb::ByteOrder byte_order, uint32_t addr_byte_size)
    {
        lldb::ClangExpressionVariableSP var_sp(new ClangExpressionVariable(byte_order, addr_byte_size));
        m_variables.push_back(var_sp);
        return var_sp;
    }

    lldb::ClangExpressionVariableSP
    CreateVariable(const lldb::ValueObjectSP &valobj_sp)
    {
        lldb::ClangExpressionVariableSP var_sp(new ClangExpressionVariable(valobj_sp));
        m_variables.push_back(var_sp);
        return var_sp;
    }
    
    

    lldb::ClangExpressionVariableSP
    CreateVariable (const ConstString &name, 
                    const TypeFromUser& user_type,
                    lldb::ByteOrder byte_order, 
                    uint32_t addr_byte_size)
    {
        lldb::ClangExpressionVariableSP var_sp(new ClangExpressionVariable(byte_order, addr_byte_size));
        var_sp->SetName (name);
        var_sp->SetClangType (user_type.GetOpaqueQualType());
        var_sp->SetClangAST (user_type.GetASTContext());
        m_variables.push_back(var_sp);
        return var_sp;
    }
    
private:
    std::vector <lldb::ClangExpressionVariableSP> m_variables;
};


} // namespace lldb_private

#endif  // liblldb_ClangExpressionVariable_h_
