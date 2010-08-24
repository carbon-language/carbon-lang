//===-- Variable.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/Variable.h"

#include "lldb/Core/Stream.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Variable constructor
//----------------------------------------------------------------------
Variable::Variable(lldb::user_id_t uid,
                       const ConstString& name,
                       Type *type,
                       ValueType scope,
                       SymbolContextScope *context,
                       Declaration* decl_ptr,
                       const DWARFExpression& location,
                       bool external,
                       bool artificial) :
    UserID(uid),
    m_name(name),
    m_type(type),
    m_scope(scope),
    m_context(context),
    m_declaration(decl_ptr),
    m_location(location),
    m_external(external),
    m_artificial(artificial)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
Variable::~Variable()
{
}


void
Variable::Dump(Stream *s, bool show_context) const
{
    s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    *s << "Variable" << (const UserID&)*this;

    if (m_name)
        *s << ", name = \"" << m_name << "\"";

    if (m_type != NULL)
    {
        *s << ", type = " << (void*)m_type << " (";
        m_type->DumpTypeName(s);
        s->PutChar(')');
    }

    if (m_scope != eValueTypeInvalid)
    {
        s->PutCString(", scope = ");
        switch (m_scope)
        {
        case eValueTypeVariableGlobal:       s->PutCString(m_external ? "global" : "static"); break;
        case eValueTypeVariableArgument:    s->PutCString("parameter"); break;
        case eValueTypeVariableLocal:        s->PutCString("local"); break;
        default:            *s << "??? (" << m_scope << ')';
        }
    }

    if (show_context && m_context != NULL)
    {
        s->PutCString(", context = ( ");
        m_context->DumpSymbolContext(s);
        s->PutCString(" )");
    }

    m_declaration.Dump(s);

    if (m_location.IsValid())
    {
        s->PutCString(", location = ");
        m_location.GetDescription(s, lldb::eDescriptionLevelBrief);
    }

    if (m_external)
        s->PutCString(", external");

    if (m_artificial)
        s->PutCString(", artificial");

    s->EOL();
}


size_t
Variable::MemorySize() const
{
    return sizeof(Variable);
}


void
Variable::CalculateSymbolContext (SymbolContext *sc)
{
    if (m_context)
        m_context->CalculateSymbolContext(sc);
    else
        sc->Clear();
}


bool
Variable::IsInScope (StackFrame *frame)
{
    switch (m_scope)
    {
    case eValueTypeVariableGlobal:
        // Globals and statics are always in scope.
        return true;

    case eValueTypeVariableArgument:
    case eValueTypeVariableLocal:
        // Check if the location has a location list that describes the value
        // of the variable with address ranges and different locations for each
        // address range?
        if (m_location.IsLocationList())
        {
            // It is a location list. We just need to tell if the location
            // list contains the current address when converted to a load
            // address
            return m_location.LocationListContainsLoadAddress (&frame->GetThread().GetProcess(), frame->GetRegisterContext()->GetPC());
        }
        else
        {
            // We don't have a location list, we just need to see if the block
            // that this variable was defined in is currently
            Block *frame_block = frame->GetSymbolContext(eSymbolContextBlock).block;
            if (frame_block)
            {
                SymbolContext variable_sc;
                CalculateSymbolContext (&variable_sc);
                if (variable_sc.function && variable_sc.block)
                    return variable_sc.block->FindBlockByID(frame_block->GetID()) != NULL;
            }
        }
        break;

    default:
        break;
    }
    return false;
}

