//===-- Block.cpp -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/VariableList.h"

using namespace lldb;
using namespace lldb_private;

Block::Block(lldb::user_id_t uid) :
    UserID(uid),
    m_parent_scope (NULL),
    m_sibling (NULL),
    m_children (),
    m_ranges (),
    m_inlineInfoSP (),
    m_variables (),
    m_parsed_block_info (false),
    m_parsed_block_variables (false),
    m_parsed_child_blocks (false)
{
}

Block::~Block ()
{
}

void
Block::GetDescription(Stream *s, Function *function, lldb::DescriptionLevel level, Process *process) const
{
    size_t num_ranges = m_ranges.size();
    if (num_ranges)
    {
        
        addr_t base_addr = LLDB_INVALID_ADDRESS;
        if (process)
            base_addr = function->GetAddressRange().GetBaseAddress().GetLoadAddress(process);
        if (base_addr == LLDB_INVALID_ADDRESS)
            base_addr = function->GetAddressRange().GetBaseAddress().GetFileAddress();

        s->Printf("range%s = ", num_ranges > 1 ? "s" : "");
        std::vector<VMRange>::const_iterator pos, end = m_ranges.end();
        for (pos = m_ranges.begin(); pos != end; ++pos)
            pos->Dump(s, base_addr, 4);
    }
    *s << ", id = " << ((const UserID&)*this);

    if (m_inlineInfoSP.get() != NULL)
        m_inlineInfoSP->Dump(s);
}

void
Block::Dump(Stream *s, addr_t base_addr, int32_t depth, bool show_context) const
{
    if (depth < 0)
    {
        Block *parent = GetParent();
        if (parent)
        {
            // We have a depth that is less than zero, print our parent blocks
            // first
            parent->Dump(s, base_addr, depth + 1, show_context);
        }
    }

    s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    *s << "Block" << ((const UserID&)*this);
    const Block* parent_block = GetParent();
    if (parent_block)
    {
        s->Printf(", parent = {0x%8.8x}", parent_block->GetID());
    }
    if (m_inlineInfoSP.get() != NULL)
        m_inlineInfoSP->Dump(s);

    if (!m_ranges.empty())
    {
        *s << ", ranges =";
        std::vector<VMRange>::const_iterator pos;
        std::vector<VMRange>::const_iterator end = m_ranges.end();
        for (pos = m_ranges.begin(); pos != end; ++pos)
        {
            if (parent_block != NULL && parent_block->Contains(*pos) == false)
                *s << '!';
            else
                *s << ' ';
            pos->Dump(s, base_addr);
        }
    }
    s->EOL();

    if (depth > 0)
    {
        s->IndentMore();

        if (m_variables.get())
        {
            m_variables->Dump(s, show_context);
        }

        for (Block *child_block = GetFirstChild(); child_block != NULL; child_block = child_block->GetSibling())
        {
            child_block->Dump(s, base_addr, depth - 1, show_context);
        }

        s->IndentLess();
    }

}


Block *
Block::FindBlockByID (user_id_t block_id)
{
    if (block_id == GetID())
        return this;

    Block *matching_block = NULL;
    for (Block *child_block = GetFirstChild(); child_block != NULL; child_block = child_block->GetSibling())
    {
        matching_block = child_block->FindBlockByID (block_id);
        if (matching_block)
            break;
    }
    return matching_block;
}

void
Block::CalculateSymbolContext(SymbolContext* sc)
{
    if (m_parent_scope)
        m_parent_scope->CalculateSymbolContext(sc);
    sc->block = this;
}

void
Block::DumpStopContext (Stream *s, const SymbolContext *sc)
{
    Block* parent_block = GetParent();

    InlineFunctionInfo* inline_info = InlinedFunctionInfo ();
    if (inline_info)
    {
        const Declaration &call_site = inline_info->GetCallSite();
        if (sc)
        {
            // First frame, dump the first inline call site
//            if (call_site.IsValid())
//            {
//                s->PutCString(" at ");
//                call_site.DumpStopContext (s);
//            }
            s->PutCString (" [inlined]");
        }
        s->EOL();
        inline_info->DumpStopContext (s);
        if (sc == NULL)
        {
            if (call_site.IsValid())
            {
                s->PutCString(" at ");
                call_site.DumpStopContext (s);
            }
        }
    }

    if (sc)
    {
        // If we have any inlined functions, this will be the deepest most
        // inlined location
        if (sc->line_entry.IsValid())
        {
            s->PutCString(" at ");
            sc->line_entry.DumpStopContext (s);
        }
    }
    if (parent_block)
        parent_block->Block::DumpStopContext (s, NULL);
}


void
Block::DumpSymbolContext(Stream *s)
{
    SymbolContext sc;
    CalculateSymbolContext(&sc);
    if (sc.function)
        sc.function->DumpSymbolContext(s);
    s->Printf(", Block{0x%8.8x}", GetID());
}

bool
Block::Contains (addr_t range_offset) const
{
    return VMRange::ContainsValue(m_ranges, range_offset);
}

bool
Block::Contains (const VMRange& range) const
{
    return VMRange::ContainsRange(m_ranges, range);
}

Block *
Block::GetParent () const
{
    if (m_parent_scope)
    {
        SymbolContext sc;
        m_parent_scope->CalculateSymbolContext(&sc);
        if (sc.block)
            return sc.block;
    }
    return NULL;
}

void
Block::AddRange(addr_t start_offset, addr_t end_offset)
{
    m_ranges.resize(m_ranges.size()+1);
    m_ranges.back().Reset(start_offset, end_offset);
}

InlineFunctionInfo*
Block::InlinedFunctionInfo ()
{
    return m_inlineInfoSP.get();
}

const InlineFunctionInfo*
Block::InlinedFunctionInfo () const
{
    return m_inlineInfoSP.get();
}

// Return the current number of bytes that this object occupies in memory
size_t
Block::MemorySize() const
{
    size_t mem_size = sizeof(Block) + m_ranges.size() * sizeof(VMRange);
    if (m_inlineInfoSP.get())
        mem_size += m_inlineInfoSP->MemorySize();
    if (m_variables.get())
        mem_size += m_variables->MemorySize();
    return mem_size;

}

Block *
Block::GetFirstChild () const
{
    if (m_children.empty())
        return NULL;
    return m_children.front().get();
}

void
Block::AddChild(const BlockSP &child_block_sp)
{
    if (child_block_sp)
    {
        Block *block_needs_sibling = NULL;

        if (!m_children.empty())
            block_needs_sibling = m_children.back().get();

        child_block_sp->SetParentScope (this);
        m_children.push_back (child_block_sp);

        if (block_needs_sibling)
            block_needs_sibling->SetSibling (child_block_sp.get());
    }
}

void
Block::SetInlinedFunctionInfo(const char *name, const char *mangled, const Declaration *decl_ptr, const Declaration *call_decl_ptr)
{
    m_inlineInfoSP.reset(new InlineFunctionInfo(name, mangled, decl_ptr, call_decl_ptr));
}



VariableListSP
Block::GetVariableList (bool get_child_variables, bool can_create)
{
    VariableListSP variable_list_sp;
    if (m_parsed_block_variables == false)
    {
        if (m_variables.get() == NULL && can_create)
        {
            m_parsed_block_variables = true;
            SymbolContext sc;
            CalculateSymbolContext(&sc);
            assert(sc.module_sp);
            sc.module_sp->GetSymbolVendor()->ParseVariablesForContext(sc);
        }
    }

    if (m_variables.get())
    {
        variable_list_sp.reset(new VariableList());
        if (variable_list_sp.get())
            variable_list_sp->AddVariables(m_variables.get());

        if (get_child_variables)
        {
            Block *child_block = GetFirstChild();
            while (child_block)
            {
                VariableListSP child_block_variable_list(child_block->GetVariableList(get_child_variables, can_create));
                if (child_block_variable_list.get())
                    variable_list_sp->AddVariables(child_block_variable_list.get());
                child_block = child_block->GetSibling();
            }
        }
    }

    return variable_list_sp;
}

uint32_t
Block::AppendVariables (bool can_create, bool get_parent_variables, VariableList *variable_list)
{
    uint32_t num_variables_added = 0;
    VariableListSP variable_list_sp(GetVariableList(false, can_create));

    if (variable_list_sp.get())
    {
        num_variables_added = variable_list_sp->GetSize();
        variable_list->AddVariables(variable_list_sp.get());
    }

    if (get_parent_variables)
    {
        Block* parent_block = GetParent();
        if (parent_block)
            num_variables_added += parent_block->AppendVariables (can_create, get_parent_variables, variable_list);
    }
    return num_variables_added;
}


void
Block::SetVariableList(VariableListSP& variables)
{
    m_variables = variables;
}

void
Block::SetBlockInfoHasBeenParsed (bool b, bool set_children)
{
    m_parsed_block_info = b;
    if (set_children)
    {
        m_parsed_child_blocks = true;
        for (Block *child_block = GetFirstChild(); child_block != NULL; child_block = child_block->GetSibling())
            child_block->SetBlockInfoHasBeenParsed (b, true);
    }
}
