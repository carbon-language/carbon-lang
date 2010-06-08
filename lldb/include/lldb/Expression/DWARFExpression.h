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
// A class designed to evaluate the DWARF expression opcodes. We will
// likely augment its abilities to handle features not supported by
// the DWARF expression engine.
//----------------------------------------------------------------------
class DWARFExpression
{
public:

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    DWARFExpression();

    DWARFExpression(const DataExtractor& data,
                    uint32_t data_offset,
                    uint32_t data_length,
                    const Address* loclist_base_addr_ptr);

    DWARFExpression(const DWARFExpression& rhs);

    virtual
    ~DWARFExpression();

    void
    GetDescription (Stream *s, lldb::DescriptionLevel level) const;

    bool
    IsValid() const;

    bool
    IsLocationList() const;

    bool
    LocationListContainsLoadAddress (Process* process, const Address &addr) const;

    void
    SetOpcodeData(const DataExtractor& data, const Address* loclist_base_addr_ptr);

    void
    SetOpcodeData(const DataExtractor& data, uint32_t data_offset, uint32_t data_length, const Address* loclist_base_addr_ptr);

    void
    SetLocationListBaseAddress(Address& base_addr);

    int
    GetRegisterKind ();

    void
    SetRegisterKind (int reg_kind);

    bool
    Evaluate (ExecutionContextScope *exe_scope,
              clang::ASTContext *ast_context,
              const Value* initial_value_ptr,
              Value& result,
              Error *error_ptr) const;

    bool
    Evaluate (ExecutionContext *exe_ctx,
              clang::ASTContext *ast_context,
              const Value* initial_value_ptr,
              Value& result,
              Error *error_ptr) const;

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

    void
    SetExpressionLocalVariableList (ClangExpressionVariableList *locals);
    
    void
    SetExpressionDeclMap (ClangExpressionDeclMap *decl_map);

protected:

    void DumpLocation(Stream *s, uint32_t offset, uint32_t length, lldb::DescriptionLevel level) const;
    //------------------------------------------------------------------
    // Classes that inherit from DWARFExpression can see and modify these
    //------------------------------------------------------------------
    DataExtractor   m_data;
    int m_reg_kind; // One of the defines that starts with LLDB_REGKIND_
    Address m_loclist_base_addr; // Base address needed for location lists
    ClangExpressionVariableList *m_expr_locals;
    ClangExpressionDeclMap *m_decl_map;
};

} // namespace lldb_private

#endif  // liblldb_DWARFExpression_h_
