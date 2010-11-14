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
#include "lldb/Core/RegularExpression.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Variable constructor
//----------------------------------------------------------------------
Variable::Variable 
(
    lldb::user_id_t uid,
    const char *name, 
    const char *mangled,   // The mangled variable name for variables in namespaces
    Type *type,
    ValueType scope,
    SymbolContextScope *context,
    Declaration* decl_ptr,
    const DWARFExpression& location,
    bool external,
    bool artificial
) :
    UserID(uid),
    m_name(name),
    m_mangled (mangled, true),
    m_type(type),
    m_scope(scope),
    m_owner_scope(context),
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


const ConstString&
Variable::GetName() const
{
    if (m_mangled)
        return m_mangled.GetName();
    return m_name;
}

bool
Variable::NameMatches (const RegularExpression& regex) const
{
    if (regex.Execute (m_name.AsCString()))
        return true;
    return m_mangled.NameMatches (regex);
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
        *s << ", type = {" << m_type->GetID() << "} " << (void*)m_type << " (";
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

    if (show_context && m_owner_scope != NULL)
    {
        s->PutCString(", context = ( ");
        m_owner_scope->DumpSymbolContext(s);
        s->PutCString(" )");
    }

    bool show_fullpaths = false;
    m_declaration.Dump(s, show_fullpaths);

    if (m_location.IsValid())
    {
        s->PutCString(", location = ");
        lldb::addr_t loclist_base_addr = LLDB_INVALID_ADDRESS;
        if (m_location.IsLocationList())
        {
            SymbolContext variable_sc;
            m_owner_scope->CalculateSymbolContext(&variable_sc);
            if (variable_sc.function)
                loclist_base_addr = variable_sc.function->GetAddressRange().GetBaseAddress().GetFileAddress();
        }
        m_location.GetDescription(s, lldb::eDescriptionLevelBrief, loclist_base_addr);
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
    if (m_owner_scope)
        m_owner_scope->CalculateSymbolContext(sc);
    else
        sc->Clear();
}


bool
Variable::IsInScope (StackFrame *frame)
{
    switch (m_scope)
    {
    case eValueTypeVariableGlobal:
    case eValueTypeVariableStatic:
        // Globals and statics are always in scope.
        return true;

    case eValueTypeVariableArgument:
    case eValueTypeVariableLocal:
        // Check if the location has a location list that describes the value
        // of the variable with address ranges and different locations for each
        // address range?
        if (m_location.IsLocationList())
        {
            SymbolContext sc;
            CalculateSymbolContext(&sc);
            
            // Currently we only support functions that have things with 
            // locations lists. If this expands, we will need to add support
            assert (sc.function);
            Target *target = &frame->GetThread().GetProcess().GetTarget();
            addr_t loclist_base_load_addr = sc.function->GetAddressRange().GetBaseAddress().GetLoadAddress (target);
            if (loclist_base_load_addr == LLDB_INVALID_ADDRESS)
                return false;
            // It is a location list. We just need to tell if the location
            // list contains the current address when converted to a load
            // address
            return m_location.LocationListContainsAddress (loclist_base_load_addr, frame->GetFrameCodeAddress().GetLoadAddress (target));
        }
        else
        {
            // We don't have a location list, we just need to see if the block
            // that this variable was defined in is currently
            Block *deepest_frame_block = frame->GetSymbolContext(eSymbolContextBlock).block;
            if (deepest_frame_block)
            {
                SymbolContext variable_sc;
                CalculateSymbolContext (&variable_sc);
                if (variable_sc.block == deepest_frame_block)
                    return true;

                return variable_sc.block->Contains (deepest_frame_block);
            }
        }
        break;

    default:
        assert (!"Unhandled case");
        break;
    }
    return false;
}

