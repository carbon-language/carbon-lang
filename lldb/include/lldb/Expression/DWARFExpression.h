//===-- DWARFExpression.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DWARFExpression_h_
#define liblldb_DWARFExpression_h_

#include "lldb/lldb-private.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Scalar.h"

class ClangExpressionVariable;
class ClangExpressionVariableList;

namespace lldb_private {

class ClangExpressionDeclMap;

//----------------------------------------------------------------------
/// @class DWARFExpression DWARFExpression.h "lldb/Expression/DWARFExpression.h"
/// @brief Encapsulates a DWARF location expression and interprets it.
///
/// DWARF location expressions are used in two ways by LLDB.  The first 
/// use is to find entities specified in the debug information, since
/// their locations are specified in precisely this language.  The second
/// is to interpret expressions without having to run the target in cases
/// where the overhead from copying JIT-compiled code into the target is
/// too high or where the target cannot be run.  This class encapsulates
/// a single DWARF location expression or a location list and interprets
/// it.
//----------------------------------------------------------------------
class DWARFExpression
{
public:
    //------------------------------------------------------------------
    /// Constructor
    //------------------------------------------------------------------
    DWARFExpression();

    //------------------------------------------------------------------
    /// Constructor
    ///
    /// @param[in] data
    ///     A data extractor configured to read the DWARF location expression's
    ///     bytecode.
    ///
    /// @param[in] data_offset
    ///     The offset of the location expression in the extractor.
    ///
    /// @param[in] data_length
    ///     The byte length of the location expression.
    ///
    /// @param[in] loclist_base_addr_ptr
    ///     If non-NULL, the address of the location list in the target 
    ///     process's .debug_loc section.
    //------------------------------------------------------------------
    DWARFExpression(const DataExtractor& data,
                    uint32_t data_offset,
                    uint32_t data_length,
                    const Address* loclist_base_addr_ptr);

    //------------------------------------------------------------------
    /// Copy constructor
    //------------------------------------------------------------------
    DWARFExpression(const DWARFExpression& rhs);

    //------------------------------------------------------------------
    /// Destructor
    //------------------------------------------------------------------
    virtual
    ~DWARFExpression();

    //------------------------------------------------------------------
    /// Print the description of the expression to a stream
    ///
    /// @param[in] s
    ///     The stream to print to.
    ///
    /// @param[in] level
    ///     The level of verbosity to use.
    //------------------------------------------------------------------
    void
    GetDescription (Stream *s, lldb::DescriptionLevel level) const;

    //------------------------------------------------------------------
    /// Return true if the location expression contains data
    //------------------------------------------------------------------
    bool
    IsValid() const;

    //------------------------------------------------------------------
    /// Return true if a location list was provided
    //------------------------------------------------------------------
    bool
    IsLocationList() const;

    //------------------------------------------------------------------
    /// Search for a load address in the location list
    ///
    /// @param[in] process
    ///     The process to use when resolving the load address
    ///
    /// @param[in] addr
    ///     The address to resolve
    ///
    /// @return
    ///     True if IsLocationList() is true and the address was found;
    ///     false otherwise.
    //------------------------------------------------------------------
    bool
    LocationListContainsLoadAddress (Process* process, const Address &addr) const;

    bool
    LocationListContainsLoadAddress (Process* process, lldb::addr_t load_addr) const;
    
    //------------------------------------------------------------------
    /// Make the expression parser read its location information from a
    /// given data source.  Does not change the offset and length
    ///
    /// @param[in] data
    ///     A data extractor configured to read the DWARF location expression's
    ///     bytecode.
    ///
    /// @param[in] loclist_base_addr_ptr
    ///     If non-NULL, the address of the location list in the target 
    ///     process's .debug_loc section.
    //------------------------------------------------------------------
    void
    SetOpcodeData(const DataExtractor& data, const Address* loclist_base_addr_ptr);

    //------------------------------------------------------------------
    /// Make the expression parser read its location information from a
    /// given data source
    ///
    /// @param[in] data
    ///     A data extractor configured to read the DWARF location expression's
    ///     bytecode.
    ///
    /// @param[in] data_offset
    ///     The offset of the location expression in the extractor.
    ///
    /// @param[in] data_length
    ///     The byte length of the location expression.
    ///
    /// @param[in] loclist_base_addr_ptr
    ///     If non-NULL, the address of the location list in the target 
    ///     process's .debug_loc section.
    //------------------------------------------------------------------
    void
    SetOpcodeData(const DataExtractor& data, uint32_t data_offset, uint32_t data_length, const Address* loclist_base_addr_ptr);

    //------------------------------------------------------------------
    /// Make the expression parser refer to a location list
    ///
    /// @param[in] base_addr
    ///     The address of the location list in the target process's .debug_loc
    ///     section.
    //------------------------------------------------------------------
    void
    SetLocationListBaseAddress(Address& base_addr);

    //------------------------------------------------------------------
    /// Return the call-frame-info style register kind
    //------------------------------------------------------------------
    int
    GetRegisterKind ();

    //------------------------------------------------------------------
    /// Set the call-frame-info style register kind
    ///
    /// @param[in] reg_kind
    ///     The register kind.
    //------------------------------------------------------------------
    void
    SetRegisterKind (int reg_kind);

    //------------------------------------------------------------------
    /// Wrapper for the static evaluate function that accepts an
    /// ExecutionContextScope instead of an ExecutionContext and uses
    /// member variables to populate many operands
    //------------------------------------------------------------------
    bool
    Evaluate (ExecutionContextScope *exe_scope,
              clang::ASTContext *ast_context,
              const Value* initial_value_ptr,
              Value& result,
              Error *error_ptr) const;

    //------------------------------------------------------------------
    /// Wrapper for the static evaluate function that uses member 
    /// variables to populate many operands
    //------------------------------------------------------------------
    bool
    Evaluate (ExecutionContext *exe_ctx,
              clang::ASTContext *ast_context,
              const Value* initial_value_ptr,
              Value& result,
              Error *error_ptr) const;

    //------------------------------------------------------------------
    /// Evaluate a DWARF location expression in a particular context
    ///
    /// @param[in] exe_ctx
    ///     The execution context in which to evaluate the location
    ///     expression.  The location expression may access the target's
    ///     memory, especially if it comes from the expression parser.
    ///
    /// @param[in] ast_context
    ///     The context in which to interpret types.
    ///
    /// @param[in] opcodes
    ///     This is a static method so the opcodes need to be provided
    ///     explicitly.
    ///
    /// @param[in] expr_locals
    ///     If the location expression was produced by the expression parser,
    ///     the list of local variables referenced by the DWARF expression.
    ///     This list should already have been populated during parsing;
    ///     the DWARF expression refers to variables by index.  Can be NULL if
    ///     the location expression uses no locals.
    ///
    /// @param[in] decl_map
    ///     If the location expression was produced by the expression parser,
    ///     the list of external variables referenced by the location 
    ///     expression.  Can be NULL if the location expression uses no
    ///     external variables.
    ///
    /// @param[in] offset
    ///     The offset of the location expression in the data extractor.
    ///
    /// @param[in] length
    ///     The length in bytes of the location expression.
    ///
    /// @param[in] reg_set
    ///     The call-frame-info style register kind.
    ///
    /// @param[in] initial_value_ptr
    ///     A value to put on top of the interpreter stack before evaluating
    ///     the expression, if the expression is parametrized.  Can be NULL.
    ///
    /// @param[in] result
    ///     A value into which the result of evaluating the expression is
    ///     to be placed.
    ///
    /// @param[in] error_ptr
    ///     If non-NULL, used to report errors in expression evaluation.
    ///
    /// @return
    ///     True on success; false otherwise.  If error_ptr is non-NULL,
    ///     details of the failure are provided through it.
    //------------------------------------------------------------------
    static bool
    Evaluate (ExecutionContext *exe_ctx,
              clang::ASTContext *ast_context,
              const DataExtractor& opcodes,
              ClangExpressionVariableList *expr_locals,
              ClangExpressionDeclMap *decl_map,
              const uint32_t offset,
              const uint32_t length,
              const uint32_t reg_set,
              const Value* initial_value_ptr,
              Value& result,
              Error *error_ptr);

    //------------------------------------------------------------------
    /// Loads a ClangExpressionVariableList into the object
    ///
    /// @param[in] locals
    ///     If non-NULL, the list of locals used by this expression.
    ///     See Evaluate().
    //------------------------------------------------------------------
    void
    SetExpressionLocalVariableList (ClangExpressionVariableList *locals);
    
    //------------------------------------------------------------------
    /// Loads a ClangExpressionDeclMap into the object
    ///
    /// @param[in] locals
    ///     If non-NULL, the list of external variables used by this 
    ///     expression.  See Evaluate().
    //------------------------------------------------------------------
    void
    SetExpressionDeclMap (ClangExpressionDeclMap *decl_map);

protected:
    //------------------------------------------------------------------
    /// Pretty-prints the location expression to a stream
    ///
    /// @param[in] stream
    ///     The stream to use for pretty-printing.
    ///
    /// @param[in] offset
    ///     The offset into the data buffer of the opcodes to be printed.
    ///
    /// @param[in] length
    ///     The length in bytes of the opcodes to be printed.
    ///
    /// @param[in] level
    ///     The level of detail to use in pretty-printing.
    //------------------------------------------------------------------
    void
    DumpLocation(Stream *s, 
                 uint32_t offset, 
                 uint32_t length, 
                 lldb::DescriptionLevel level) const;
    
    //------------------------------------------------------------------
    /// Classes that inherit from DWARFExpression can see and modify these
    //------------------------------------------------------------------
    
    DataExtractor m_data;                       ///< A data extractor capable of reading opcode bytes
    int m_reg_kind;                             ///< One of the defines that starts with LLDB_REGKIND_
    Address m_loclist_base_addr;                ///< Base address needed for location lists
    ClangExpressionVariableList *m_expr_locals; ///< The locals used by this expression.  See Evaluate()
    ClangExpressionDeclMap *m_decl_map;         ///< The external variables used by this expression.  See Evaluate()
};

} // namespace lldb_private

#endif  // liblldb_DWARFExpression_h_
