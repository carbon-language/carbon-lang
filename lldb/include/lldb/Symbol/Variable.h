//===-- Variable.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Variable_h_
#define liblldb_Variable_h_

#include <vector>

#include "lldb/lldb-private.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Symbol/Declaration.h"
#include "lldb/Core/UserID.h"

namespace lldb_private {

class Variable : public UserID
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    Variable(lldb::user_id_t uid,
             const ConstString& name, Type *type,
             lldb::ValueType scope,
             SymbolContextScope *owner_scope,
             Declaration* decl,
             const DWARFExpression& location,
             bool external,
             bool artificial);

    virtual
    ~Variable();

    void
    Dump(Stream *s, bool show_context) const;

    const Declaration&
    GetDeclaration() const
    {
        return m_declaration;
    }

    const ConstString&
    GetName() const
    {
        return m_name;
    }

    Type *
    GetType()
    {
        return m_type;
    }

    const Type *
    GetType() const
    {
        return m_type;
    }

    lldb::ValueType
    GetScope() const
    {
        return m_scope;
    }

    bool
    IsExternal() const
    {
        return m_external;
    }

    bool
    IsArtificial() const
    {
        return m_artificial;
    }

    DWARFExpression &
    LocationExpression()
    {
        return m_location;
    }

    const DWARFExpression &
    LocationExpression() const
    {
        return m_location;
    }

    size_t
    MemorySize() const;

    void
    CalculateSymbolContext (SymbolContext *sc);

    bool
    IsInScope (StackFrame *frame);

protected:
    ConstString m_name;                 // Name of the variable
    Type *m_type;                       // The type pointer of the variable (int, struct, class, etc)
    lldb::ValueType m_scope;            // global, parameter, local
    SymbolContextScope *m_owner_scope;  // The symbol file scope that this variable was defined in
    Declaration m_declaration;          // Declaration location for this item.
    DWARFExpression m_location;         // The location of this variable that can be fed to DWARFExpression::Evaluate()
    uint8_t m_external:1,               // Visible outside the containing compile unit?
            m_artificial:1;             // Non-zero if the variable is not explicitly declared in source
private:
    Variable(const Variable& rhs);
    Variable& operator=(const Variable& rhs);
};

} // namespace lldb_private

#endif  // liblldb_Variable_h_
