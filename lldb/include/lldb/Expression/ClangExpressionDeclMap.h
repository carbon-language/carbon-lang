//===-- ClangExpressionDeclMap.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangExpressionDeclMap_h_
#define liblldb_ClangExpressionDeclMap_h_

// C Includes
#include <signal.h>
#include <stdint.h>

// C++ Includes
#include <vector>

// Other libraries and framework includes
// Project includes
#include "llvm/ADT/DenseMap.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/Value.h"
#include "lldb/Expression/ClangExpressionVariable.h"
#include "lldb/Symbol/TaggedASTType.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ExecutionContext.h"

namespace llvm {
    class Type;
    class Value;
}

namespace lldb_private {

class ClangExpressionVariables;
class ClangPersistentVariables;
class Error;
class Function;
class NameSearchContext;
class Variable;

//----------------------------------------------------------------------
/// @class ClangExpressionDeclMap ClangExpressionDeclMap.h "lldb/Expression/ClangExpressionDeclMap.h"
/// @brief Manages named entities that are defined in LLDB's debug information.
///
/// The Clang parser uses the ClangASTSource as an interface to request named
/// entities from outside an expression.  The ClangASTSource reports back, listing
/// all possible objects corresponding to a particular name.  But it in turn
/// relies on ClangExpressionDeclMap, which performs several important functions.
///
/// First, it records what variables and functions were looked up and what Decls
/// were returned for them.
///
/// Second, it constructs a struct on behalf of IRForTarget, recording which 
/// variables should be placed where and relaying this information back so that 
/// IRForTarget can generate context-independent code.
///
/// Third, it "materializes" this struct on behalf of the expression command,
/// finding the current values of each variable and placing them into the
/// struct so that it can be passed to the JITted version of the IR.
///
/// Fourth and finally, it "dematerializes" the struct after the JITted code has
/// has executed, placing the new values back where it found the old ones.
//----------------------------------------------------------------------
class ClangExpressionDeclMap
{
public:
    //------------------------------------------------------------------
    /// Constructor
    ///
    /// Initializes class variabes.
    //------------------------------------------------------------------
    ClangExpressionDeclMap();
    
    //------------------------------------------------------------------
    /// Destructor
    //------------------------------------------------------------------
    ~ClangExpressionDeclMap();
    
    //------------------------------------------------------------------
    /// Enable the state needed for parsing and IR transformation.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to use when finding types for variables.
    ///     Also used to find a "scratch" AST context to store result types.
    //------------------------------------------------------------------
    void WillParse(ExecutionContext &exe_ctx);
    
    //------------------------------------------------------------------
    /// Disable the state needed for parsing and IR transformation.
    //------------------------------------------------------------------
    void DidParse();
    
    //------------------------------------------------------------------
    /// [Used by IRForTarget] Get a new result variable name of the form
    ///     $n, where n is a natural number starting with 0.
    ///
    /// @param[in] name
    ///     The std::string to place the name into.
    //------------------------------------------------------------------
    const ConstString &
    GetPersistentResultName ();
    
    clang::NamespaceDecl *
    AddNamespace (NameSearchContext &context, 
                  const ClangNamespaceDecl &namespace_decl);

    //------------------------------------------------------------------
    /// [Used by IRForTarget] Add a variable to the list of persistent
    ///     variables for the process.
    ///
    /// @param[in] decl
    ///     The Clang declaration for the persistent variable, used for
    ///     lookup during parsing.
    ///
    /// @param[in] name
    ///     The name of the persistent variable, usually $something.
    ///
    /// @param[in] type
    ///     The type of the variable, in the Clang parser's context.
    ///
    /// @return
    ///     True on success; false otherwise.
    //------------------------------------------------------------------
    bool AddPersistentVariable (const clang::NamedDecl *decl,
                                const ConstString &name, 
                                TypeFromParser type);
    
    //------------------------------------------------------------------
    /// [Used by IRForTarget] Add a variable to the struct that needs to
    ///     be materialized each time the expression runs.
    ///
    /// @param[in] decl
    ///     The Clang declaration for the variable.
    ///
    /// @param[in] name
    ///     The name of the variable.
    ///
    /// @param[in] value
    ///     The LLVM IR value for this variable.
    ///
    /// @param[in] size
    ///     The size of the variable in bytes.
    ///
    /// @param[in] alignment
    ///     The required alignment of the variable in bytes.
    ///
    /// @return
    ///     True on success; false otherwise.
    //------------------------------------------------------------------
    bool AddValueToStruct (const clang::NamedDecl *decl,
                           const ConstString &name,
                           llvm::Value *value,
                           size_t size,
                           off_t alignment);
    
    //------------------------------------------------------------------
    /// [Used by IRForTarget] Finalize the struct, laying out the position 
    /// of each object in it.
    ///
    /// @return
    ///     True on success; false otherwise.
    //------------------------------------------------------------------
    bool DoStructLayout ();
    
    //------------------------------------------------------------------
    /// [Used by IRForTarget] Get general information about the laid-out
    /// struct after DoStructLayout() has been called.
    ///
    /// @param[out] num_elements
    ///     The number of elements in the struct.
    ///
    /// @param[out] size
    ///     The size of the struct, in bytes.
    ///
    /// @param[out] alignment
    ///     The alignment of the struct, in bytes.
    ///
    /// @return
    ///     True if the information could be retrieved; false otherwise.
    //------------------------------------------------------------------
    bool GetStructInfo (uint32_t &num_elements,
                        size_t &size,
                        off_t &alignment);
    
    //------------------------------------------------------------------
    /// [Used by IRForTarget] Get specific information about one field
    /// of the laid-out struct after DoStructLayout() has been called.
    ///
    /// @param[out] decl
    ///     The parsed Decl for the field, as generated by ClangASTSource
    ///     on ClangExpressionDeclMap's behalf.  In the case of the result
    ///     value, this will have the name $__lldb_result even if the
    ///     result value ends up having the name $1.  This is an
    ///     implementation detail of IRForTarget.
    ///
    /// @param[out] value
    ///     The IR value for the field (usually a GlobalVariable).  In
    ///     the case of the result value, this will have the correct
    ///     name ($1, for instance).  This is an implementation detail
    ///     of IRForTarget.
    ///
    /// @param[out] offset
    ///     The offset of the field from the beginning of the struct.
    ///     As long as the struct is aligned according to its required
    ///     alignment, this offset will align the field correctly.
    ///
    /// @param[out] name
    ///     The name of the field as used in materialization.
    ///
    /// @param[in] index
    ///     The index of the field about which information is requested.
    ///
    /// @return
    ///     True if the information could be retrieved; false otherwise.
    //------------------------------------------------------------------
    bool GetStructElement (const clang::NamedDecl *&decl,
                           llvm::Value *&value,
                           off_t &offset,
                           ConstString &name,
                           uint32_t index);
    
    //------------------------------------------------------------------
    /// [Used by IRForTarget] Get information about a function given its
    /// Decl.
    ///
    /// @param[in] decl
    ///     The parsed Decl for the Function, as generated by ClangASTSource
    ///     on ClangExpressionDeclMap's behalf.
    ///
    /// @param[out] value
    ///     A pointer to the address where a Value for the function's address
    ///     can be stored.  IRForTarget typically places a ConstantExpr here.
    ///
    /// @param[out] ptr
    ///     The absolute address of the function in the target.
    ///
    /// @return
    ///     True if the information could be retrieved; false otherwise.
    //------------------------------------------------------------------
    bool GetFunctionInfo (const clang::NamedDecl *decl, 
                          llvm::Value**& value, 
                          uint64_t &ptr);
    
    //------------------------------------------------------------------
    /// [Used by IRForTarget] Get the address of a function given nothing
    /// but its name.  Some functions are needed but didn't get Decls made
    /// during parsing -- specifically, sel_registerName is never called
    /// in the generated IR but we need to call it nonetheless.
    ///
    /// @param[in] name
    ///     The name of the function.  
    ///
    /// @param[out] ptr
    ///     The absolute address of the function in the target.
    ///
    /// @return
    ///     True if the address could be retrieved; false otherwise.
    //------------------------------------------------------------------
    bool GetFunctionAddress (const ConstString &name,
                             uint64_t &ptr);
    
    //------------------------------------------------------------------
    /// [Used by CommandObjectExpression] Materialize the entire struct
    /// at a given address, which should be aligned as specified by 
    /// GetStructInfo().
    ///
    /// @param[in] exe_ctx
    ///     The execution context at which to dump the struct.
    ///
    /// @param[in] struct_address
    ///     The address at which the struct should be written.
    ///
    /// @param[in] error
    ///     An Error to populate with any messages related to
    ///     materializing the struct.
    ///
    /// @return
    ///     True on success; false otherwise.
    //------------------------------------------------------------------
    bool Materialize(ExecutionContext &exe_ctx,
                     lldb::addr_t &struct_address,
                     Error &error);
    
    //------------------------------------------------------------------
    /// [Used by CommandObjectExpression] Get the "this" pointer
    /// from a given execution context.
    ///
    /// @param[out] object_ptr
    ///     The this pointer.
    ///
    /// @param[in] exe_ctx
    ///     The execution context at which to dump the struct.
    ///
    /// @param[in] error
    ///     An Error to populate with any messages related to
    ///     finding the "this" pointer.
    ///
    /// @return
    ///     True on success; false otherwise.
    //------------------------------------------------------------------
    bool GetObjectPointer(lldb::addr_t &object_ptr,
                          ExecutionContext &exe_ctx,
                          Error &error);
    
    //------------------------------------------------------------------
    /// [Used by CommandObjectExpression] Pretty-print a materialized
    /// struct, which must have been materialized by Materialize(),
    /// byte for byte on a given stream.
    ///
    /// @param[in] exe_ctx
    ///     The execution context from which to read the struct.
    ///
    /// @param[in] s
    ///     The stream on which to write the pretty-printed output.
    ///
    /// @param[in] error
    ///     An Error to populate with any messages related to
    ///     pretty-printing the struct.
    ///
    /// @return
    ///     True on success; false otherwise.
    //------------------------------------------------------------------
    bool DumpMaterializedStruct(ExecutionContext &exe_ctx,
                                Stream &s,
                                Error &error);
    
    //------------------------------------------------------------------
    /// [Used by CommandObjectExpression] Deaterialize the entire struct.
    ///
    /// @param[in] exe_ctx
    ///     The execution context from which to read the struct.
    ///
    /// @param[out] result
    ///     A ClangExpressionVariable containing the result of the
    ///     expression, for potential re-use.
    ///
    /// @param[in] error
    ///     An Error to populate with any messages related to
    ///     dematerializing the struct.
    ///
    /// @return
    ///     True on success; false otherwise.
    //------------------------------------------------------------------
    bool Dematerialize(ExecutionContext &exe_ctx,
                       ClangExpressionVariable *&result,
                       Error &error);
    
    //------------------------------------------------------------------
    /// [Used by ClangASTSource] Find all entities matching a given name,
    /// using a NameSearchContext to make Decls for them.
    ///
    /// @param[in] context
    ///     The NameSearchContext that can construct Decls for this name.
    ///
    /// @param[in] name
    ///     The name as a plain C string.  The NameSearchContext contains 
    ///     a DeclarationName for the name so at first the name may seem
    ///     redundant, but ClangExpressionDeclMap operates in RTTI land so 
    ///     it can't access DeclarationName.
    ///
    /// @return
    ///     True on success; false otherwise.
    //------------------------------------------------------------------
    void GetDecls (NameSearchContext &context,
                   const ConstString &name);
    
    //------------------------------------------------------------------
    /// [Used by ClangASTSource] Report whether a $__lldb variable has
    /// been searched for yet.  This is the trigger for beginning to 
    /// actually look for externally-defined names.  (Names that come
    /// before this are typically the names of built-ins that don't need
    /// to be looked up.)
    ///
    /// @return
    ///     True if a $__lldb variable has been found.
    //------------------------------------------------------------------
    bool
    GetLookupsEnabled ()
    {
        assert(m_parser_vars.get());
        return m_parser_vars->m_enable_lookups;
    }
    
    //------------------------------------------------------------------
    /// [Used by ClangASTSource] Indicate that a $__lldb variable has
    /// been found.
    //------------------------------------------------------------------
    void
    SetLookupsEnabled ()
    {
        assert(m_parser_vars.get());
        m_parser_vars->m_enable_lookups = true;
    }

private:
    ClangExpressionVariableStore    m_found_entities;           ///< All entities that were looked up for the parser.
    ClangExpressionVariableList     m_struct_members;           ///< All entities that need to be placed in the struct.
    
    //----------------------------------------------------------------------
    /// The following values should not live beyond parsing
    //----------------------------------------------------------------------
    struct ParserVars {
        ParserVars() :
            m_exe_ctx(NULL),
            m_sym_ctx(),
            m_persistent_vars(NULL),
            m_enable_lookups(false),
            m_ignore_lookups(false)
        {
        }
        
        ExecutionContext           *m_exe_ctx;          ///< The execution context to use when parsing.
        SymbolContext               m_sym_ctx;          ///< The symbol context to use in finding variables and types.
        ClangPersistentVariables   *m_persistent_vars;  ///< The persistent variables for the process.
        bool                        m_enable_lookups;   ///< Set to true during parsing if we have found the first "$__lldb" name.
        bool                        m_ignore_lookups;   ///< True during an import when we should be ignoring type lookups.
    };
    
    std::auto_ptr<ParserVars> m_parser_vars;
    
    //----------------------------------------------------------------------
    /// Activate parser-specific variables
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
    /// The following values contain layout information for the materialized
    /// struct, but are not specific to a single materialization
    //----------------------------------------------------------------------
    struct StructVars {
        StructVars() :
            m_struct_alignment(0),
            m_struct_size(0),
            m_struct_laid_out(false),
            m_result_name(),
            m_object_pointer_type(NULL, NULL)
        {
        }
        
        off_t                       m_struct_alignment;         ///< The alignment of the struct in bytes.
        size_t                      m_struct_size;              ///< The size of the struct in bytes.
        bool                        m_struct_laid_out;          ///< True if the struct has been laid out and the layout is valid (that is, no new fields have been added since).
        ConstString                 m_result_name;              ///< The name of the result variable ($1, for example)
        TypeFromUser                m_object_pointer_type;      ///< The type of the "this" variable, if one exists
    };
    
    std::auto_ptr<StructVars> m_struct_vars;
    
    //----------------------------------------------------------------------
    /// Activate struct variables
    //----------------------------------------------------------------------
    void EnableStructVars()
    {
        if (!m_struct_vars.get())
            m_struct_vars.reset(new struct StructVars);
    }
    
    //----------------------------------------------------------------------
    /// Deallocate struct variables
    //----------------------------------------------------------------------
    void DisableStructVars()
    {
        m_struct_vars.reset();
    }
    
    //----------------------------------------------------------------------
    /// The following values refer to a specific materialization of the
    /// structure in a process
    //----------------------------------------------------------------------
    struct MaterialVars {
        MaterialVars() :
            m_allocated_area(NULL),
            m_materialized_location(NULL)
        {
        }
        
        Process                    *m_process;                  ///< The process that the struct is materialized into.
        lldb::addr_t                m_allocated_area;           ///< The base of the memory allocated for the struct.  Starts on a potentially unaligned address and may therefore be larger than the struct.
        lldb::addr_t                m_materialized_location;    ///< The address at which the struct is placed.  Falls inside the allocated area.
    };
    
    std::auto_ptr<MaterialVars> m_material_vars;
    
    //----------------------------------------------------------------------
    /// Activate materialization-specific variables
    //----------------------------------------------------------------------
    void EnableMaterialVars()
    {
        if (!m_material_vars.get())
            m_material_vars.reset(new struct MaterialVars);
    }
    
    //----------------------------------------------------------------------
    /// Deallocate materialization-specific variables
    //----------------------------------------------------------------------
    void DisableMaterialVars()
    {
        m_material_vars.reset();
    }
    
    //------------------------------------------------------------------
    /// Given a stack frame, find a variable that matches the given name and 
    /// type.  We need this for expression re-use; we may not always get the
    /// same lldb::Variable back, and we want the expression to work wherever 
    /// it can.  Returns the variable defined in the tightest scope.
    ///
    /// @param[in] frame
    ///     The stack frame to use as a basis for finding the variable.
    ///
    /// @param[in] name
    ///     The name as a plain C string.
    ///
    /// @param[in] type
    ///     The required type for the variable.  This function may be called
    ///     during parsing, in which case we don't know its type; hence the
    ///     default.
    ///
    /// @return
    ///     The LLDB Variable found, or NULL if none was found.
    //------------------------------------------------------------------
    Variable *FindVariableInScope(StackFrame &frame,
                                  const ConstString &name,
                                  TypeFromUser *type = NULL);
    
    //------------------------------------------------------------------
    /// Get the value of a variable in a given execution context and return
    /// the associated Types if needed.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to look for the variable in.
    ///
    /// @param[in] var
    ///     The variable to evaluate.
    ///
    /// @param[in] parser_ast_context
    ///     The AST context of the parser, to store the found type in.
    ///
    /// @param[out] found_type
    ///     The type of the found value, as it was found in the user process.
    ///     This is only useful when the variable is being inspected on behalf
    ///     of the parser, hence the default.
    ///
    /// @param[out] parser_type
    ///     The type of the found value, as it was copied into the parser's
    ///     AST context.  This is only useful when the variable is being
    ///     inspected on behalf of the parser, hence the default.
    ///
    /// @param[in] decl
    ///     The Decl to be looked up.
    ///
    /// @return
    ///     The LLDB Value for the variable.
    //------------------------------------------------------------------
    Value *GetVariableValue(ExecutionContext &exe_ctx,
                            Variable *var,
                            clang::ASTContext *parser_ast_context,
                            TypeFromUser *found_type = NULL,
                            TypeFromParser *parser_type = NULL);
    
    //------------------------------------------------------------------
    /// Use the NameSearchContext to generate a Decl for the given LLDB
    /// Variable, and put it in the Tuple list.
    ///
    /// @param[in] context
    ///     The NameSearchContext to use when constructing the Decl.
    ///
    /// @param[in] var
    ///     The LLDB Variable that needs a Decl.
    //------------------------------------------------------------------
    void AddOneVariable(NameSearchContext &context, 
                        Variable *var);
    
    //------------------------------------------------------------------
    /// Use the NameSearchContext to generate a Decl for the given
    /// persistent variable, and put it in the Tuple list.
    ///
    /// @param[in] context
    ///     The NameSearchContext to use when constructing the Decl.
    ///
    /// @param[in] pvar
    ///     The persistent variable that needs a Decl.
    //------------------------------------------------------------------
    void AddOneVariable(NameSearchContext &context, ClangExpressionVariable *pvar);
    
    //------------------------------------------------------------------
    /// Use the NameSearchContext to generate a Decl for the given
    /// function.  (Functions are not placed in the Tuple list.)  Can
    /// handle both fully typed functions and generic functions.
    ///
    /// @param[in] context
    ///     The NameSearchContext to use when constructing the Decl.
    ///
    /// @param[in] fun
    ///     The Function that needs to be created.  If non-NULL, this is
    ///     a fully-typed function.
    ///
    /// @param[in] sym
    ///     The Symbol that corresponds to a function that needs to be 
    ///     created with generic type (unitptr_t foo(...)).
    //------------------------------------------------------------------
    void AddOneFunction(NameSearchContext &context, Function *fun, Symbol *sym);
    
    //------------------------------------------------------------------
    /// Use the NameSearchContext to generate a Decl for the given
    /// register.
    ///
    /// @param[in] context
    ///     The NameSearchContext to use when constructing the Decl.
    ///
    /// @param[in] reg_info
    ///     The information corresponding to that register.
    //------------------------------------------------------------------
    void AddOneRegister(NameSearchContext &context, const lldb::RegisterInfo *reg_info);
    
    //------------------------------------------------------------------
    /// Use the NameSearchContext to generate a Decl for the given
    /// type.  (Types are not placed in the Tuple list.)
    ///
    /// @param[in] context
    ///     The NameSearchContext to use when constructing the Decl.
    ///
    /// @param[in] type
    ///     The type that needs to be created.
    ///
    /// @param[in] add_method
    ///     True if a method with signature void $__lldb_expr(void*)
    ///     should be added to the C++ class type passed in
    //------------------------------------------------------------------
    void AddOneType(NameSearchContext &context, 
                    TypeFromUser &type, 
                    bool add_method = false);
    
    //------------------------------------------------------------------
    /// Actually do the task of materializing or dematerializing the struct.
    /// Since both tasks are very similar, although ClangExpressionDeclMap
    /// exposes two functions to the outside, both call DoMaterialize.
    ///
    /// @param[in] dematerialize
    ///     True if the struct is to be dematerialized; false if it is to
    ///     be materialized.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to use.
    ///
    /// @param[out] result
    ///     If the struct is being dematerialized, a pointer into which the
    ///     location of the result persistent variable is placed.  If not,
    ///     NULL.
    ///
    /// @param[in] err
    ///     An Error to populate with any messages related to
    ///     (de)materializing the struct.
    ///
    /// @return
    ///     True on success; false otherwise.
    //------------------------------------------------------------------
    bool DoMaterialize (bool dematerialize,
                        ExecutionContext &exe_ctx,
                        ClangExpressionVariable **result,
                        Error &err);
    
    //------------------------------------------------------------------
    /// Clean up the state required to dematerialize the variable.
    //------------------------------------------------------------------
    void DidDematerialize ();

    //------------------------------------------------------------------
    /// Actually do the task of materializing or dematerializing a persistent
    /// variable.
    ///
    /// @param[in] dematerialize
    ///     True if the variable is to be dematerialized; false if it is to
    ///     be materialized.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to use.
    ///
    /// @param[in] name
    ///     The name of the persistent variable.
    ///
    /// @param[in] addr
    ///     The address at which to materialize the variable.
    ///
    /// @param[in] err
    ///     An Error to populate with any messages related to
    ///     (de)materializing the persistent variable.
    ///
    /// @return
    ///     True on success; false otherwise.
    //------------------------------------------------------------------
    bool DoMaterializeOnePersistentVariable(bool dematerialize,
                                            ExecutionContext &exe_ctx,
                                            const ConstString &name,
                                            lldb::addr_t addr,
                                            Error &err);
    
    //------------------------------------------------------------------
    /// Actually do the task of materializing or dematerializing a 
    /// variable.
    ///
    /// @param[in] dematerialize
    ///     True if the variable is to be dematerialized; false if it is to
    ///     be materialized.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to use.
    ///
    /// @param[in] sym_ctx
    ///     The symbol context to use (for looking the variable up).
    ///
    /// @param[in] name
    ///     The name of the variable (for looking the variable up).
    ///
    /// @param[in] type
    ///     The required type of the variable (for looking the variable up).
    ///
    /// @param[in] addr
    ///     The address at which to materialize the variable.
    ///
    /// @param[in] err
    ///     An Error to populate with any messages related to
    ///     (de)materializing the persistent variable.
    ///
    /// @return
    ///     True on success; false otherwise.
    //------------------------------------------------------------------
    bool DoMaterializeOneVariable(bool dematerialize,
                                  ExecutionContext &exe_ctx,
                                  const SymbolContext &sym_ctx,
                                  const ConstString &name,
                                  TypeFromUser type,
                                  lldb::addr_t addr, 
                                  Error &err);
    
    //------------------------------------------------------------------
    /// Actually do the task of materializing or dematerializing a 
    /// register variable.
    ///
    /// @param[in] dematerialize
    ///     True if the variable is to be dematerialized; false if it is to
    ///     be materialized.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to use.
    ///
    /// @param[in] reg_ctx
    ///     The register context to use.
    ///
    /// @param[in] reg_info
    ///     The information for the register to read/write.
    ///
    /// @param[in] addr
    ///     The address at which to materialize the variable.
    ///
    /// @param[in] err
    ///     An Error to populate with any messages related to
    ///     (de)materializing the persistent variable.
    ///
    /// @return
    ///     True on success; false otherwise.
    //------------------------------------------------------------------
    bool DoMaterializeOneRegister(bool dematerialize,
                                  ExecutionContext &exe_ctx,
                                  RegisterContext &reg_ctx,
                                  const lldb::RegisterInfo &reg_info,
                                  lldb::addr_t addr, 
                                  Error &err);
    
    //------------------------------------------------------------------
    /// A wrapper for ClangASTContext::CopyType that sets a flag that
    /// indicates that we should not respond to queries during import.
    ///
    /// @param[in] dest_context
    ///     The target AST context, typically the parser's AST context.
    ///
    /// @param[in] source_context
    ///     The source AST context, typically the AST context of whatever
    ///     symbol file the type was found in.
    ///
    /// @param[in] clang_type
    ///     The source type.
    ///
    /// @return
    ///     The imported type.
    //------------------------------------------------------------------
    void *GuardedCopyType (clang::ASTContext *dest_context, 
                           clang::ASTContext *source_context,
                           void *clang_type);
};
    
} // namespace lldb_private

#endif  // liblldb_ClangExpressionDeclMap_h_
