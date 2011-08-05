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
#include "lldb/Symbol/SymbolFile.h"
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
    m_variable_list_sp (),
    m_parsed_block_info (false),
    m_parsed_block_variables (false),
    m_parsed_child_blocks (false)
{
}

Block::~Block ()
{
}

void
Block::GetDescription(Stream *s, Function *function, lldb::DescriptionLevel level, Target *target) const
{
    *s << "id = " << ((const UserID&)*this);

    size_t num_ranges = m_ranges.size();
    if (num_ranges)
    {
        
        addr_t base_addr = LLDB_INVALID_ADDRESS;
        if (target)
            base_addr = function->GetAddressRange().GetBaseAddress().GetLoadAddress(target);
        if (base_addr == LLDB_INVALID_ADDRESS)
            base_addr = function->GetAddressRange().GetBaseAddress().GetFileAddress();

        s->Printf(", range%s = ", num_ranges > 1 ? "s" : "");
        std::vector<VMRange>::const_iterator pos, end = m_ranges.end();
        for (pos = m_ranges.begin(); pos != end; ++pos)
            pos->Dump(s, base_addr, 4);
    }

    if (m_inlineInfoSP.get() != NULL)
    {
        bool show_fullpaths = (level == eDescriptionLevelVerbose);
        m_inlineInfoSP->Dump(s, show_fullpaths);
    }
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
    {
        bool show_fullpaths = false;
        m_inlineInfoSP->Dump(s, show_fullpaths);
    }

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

        if (m_variable_list_sp.get())
        {
            m_variable_list_sp->Dump(s, show_context);
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
Block::CalculateSymbolContext (SymbolContext* sc)
{
    if (m_parent_scope)
        m_parent_scope->CalculateSymbolContext(sc);
    sc->block = this;
}

void
Block::DumpStopContext 
(
    Stream *s, 
    const SymbolContext *sc_ptr, 
    const Declaration *child_inline_call_site, 
    bool show_fullpaths,
    bool show_inline_blocks)
{
    const InlineFunctionInfo* inline_info = NULL;
    Block* inlined_block;
    if (sc_ptr)
        inlined_block = GetContainingInlinedBlock ();
    else
        inlined_block = GetInlinedParent();

    if (inlined_block)
        inline_info = inlined_block->GetInlinedFunctionInfo();
    const Declaration *inline_call_site = child_inline_call_site;
    if (inline_info)
    {
        inline_call_site = &inline_info->GetCallSite();
        if (sc_ptr)
        {
            // First frame in a frame with inlined functions
            s->PutCString (" [inlined]");
        }
        if (show_inline_blocks && child_inline_call_site)
            s->EOL();
        else
            s->PutChar(' ');
        
        if (sc_ptr == NULL)
            s->Indent();

        s->PutCString(inline_info->GetName ().AsCString());

        if (child_inline_call_site && child_inline_call_site->IsValid())
        {
            s->PutCString(" at ");
            child_inline_call_site->DumpStopContext (s, show_fullpaths);
        }
    }

    // The first call to this function from something that has a symbol
    // context will pass in a valid sc_ptr. Subsequent calls to this function
    // from this function for inline purposes will NULL out sc_ptr. So on the
    // first time through we dump the line table entry (which is always at the
    // deepest inline code block). And subsequent calls to this function we
    // will use hte inline call site information to print line numbers.
    if (sc_ptr)
    {
        // If we have any inlined functions, this will be the deepest most
        // inlined location
        if (sc_ptr->line_entry.IsValid())
        {
            s->PutCString(" at ");
            sc_ptr->line_entry.DumpStopContext (s, show_fullpaths);
        }
    }

    if (show_inline_blocks)
    {
        if (inlined_block)
        {
            inlined_block->Block::DumpStopContext (s, 
                                                   NULL, 
                                                   inline_call_site, 
                                                   show_fullpaths, 
                                                   show_inline_blocks);
        }
        else if (child_inline_call_site)
        {
            SymbolContext sc;
            CalculateSymbolContext(&sc);
            if (sc.function)
            {
                s->EOL();
                s->Indent (sc.function->GetMangled().GetName().AsCString());
                if (child_inline_call_site && child_inline_call_site->IsValid())
                {
                    s->PutCString(" at ");
                    child_inline_call_site->DumpStopContext (s, show_fullpaths);
                }
            }
        }
    }
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

void
Block::DumpAddressRanges (Stream *s, lldb::addr_t base_addr)
{
    if (!m_ranges.empty())
    {
        std::vector<VMRange>::const_iterator pos, end = m_ranges.end();
        for (pos = m_ranges.begin(); pos != end; ++pos)
            pos->Dump (s, base_addr);
    }
}

bool
Block::Contains (addr_t range_offset) const
{
    return VMRange::ContainsValue(m_ranges, range_offset);
}

bool
Block::Contains (const Block *block) const
{
    if (this == block)
        return false; // This block doesn't contain itself...
    
    // Walk the parent chain for "block" and see if any if them match this block
    const Block *block_parent;
    for (block_parent = block->GetParent();
         block_parent != NULL;
         block_parent = block_parent->GetParent())
    {
        if (this == block_parent)
            return true; // One of the parents of "block" is this object!
    }
    return false;
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

Block *
Block::GetContainingInlinedBlock ()
{
    if (GetInlinedFunctionInfo())
        return this;
    return GetInlinedParent ();
}

Block *
Block::GetInlinedParent ()
{
    Block *parent_block = GetParent ();
    if (parent_block)
    {
        if (parent_block->GetInlinedFunctionInfo())
            return parent_block;
        else
            return parent_block->GetInlinedParent();
    }
    return NULL;
}


bool
Block::GetRangeContainingOffset (const addr_t offset, VMRange &range)
{
    uint32_t range_idx = VMRange::FindRangeIndexThatContainsValue (m_ranges, offset);
    if (range_idx < m_ranges.size())
    {
        range = m_ranges[range_idx];
        return true;
    }
    range.Clear();
    return false;
}


bool
Block::GetRangeContainingAddress (const Address& addr, AddressRange &range)
{
    SymbolContext sc;
    CalculateSymbolContext(&sc);
    if (sc.function)
    {
        const AddressRange &func_range = sc.function->GetAddressRange();
        if (addr.GetSection() == func_range.GetBaseAddress().GetSection())
        {
            const addr_t addr_offset = addr.GetOffset();
            const addr_t func_offset = func_range.GetBaseAddress().GetOffset();
            if (addr_offset >= func_offset && addr_offset < func_offset + func_range.GetByteSize())
            {
                addr_t offset = addr_offset - func_offset;
                
                uint32_t range_idx = VMRange::FindRangeIndexThatContainsValue (m_ranges, offset);
                if (range_idx < m_ranges.size())
                {
                    range.GetBaseAddress() = func_range.GetBaseAddress();
                    range.GetBaseAddress().SetOffset(func_offset + m_ranges[range_idx].GetBaseAddress());
                    range.SetByteSize(m_ranges[range_idx].GetByteSize());
                    return true;
                }
            }
        }
    }
    range.Clear();
    return false;
}

bool
Block::GetRangeAtIndex (uint32_t range_idx, AddressRange &range)
{
    if (range_idx < m_ranges.size())
    {
        SymbolContext sc;
        CalculateSymbolContext(&sc);
        if (sc.function)
        {
            range.GetBaseAddress() = sc.function->GetAddressRange().GetBaseAddress();
            range.GetBaseAddress().Slide(m_ranges[range_idx].GetBaseAddress ());
            range.SetByteSize (m_ranges[range_idx].GetByteSize());
            return true;
        }
    }
    return false;
}

bool
Block::GetStartAddress (Address &addr)
{
    if (m_ranges.empty())
        return false;

    SymbolContext sc;
    CalculateSymbolContext(&sc);
    if (sc.function)
    {
        addr = sc.function->GetAddressRange().GetBaseAddress();
        addr.Slide(m_ranges.front().GetBaseAddress ());
        return true;
    }
    return false;
}

void
Block::AddRange(addr_t start_offset, addr_t end_offset)
{
    m_ranges.resize(m_ranges.size()+1);
    m_ranges.back().Reset(start_offset, end_offset);
}

// Return the current number of bytes that this object occupies in memory
size_t
Block::MemorySize() const
{
    size_t mem_size = sizeof(Block) + m_ranges.size() * sizeof(VMRange);
    if (m_inlineInfoSP.get())
        mem_size += m_inlineInfoSP->MemorySize();
    if (m_variable_list_sp.get())
        mem_size += m_variable_list_sp->MemorySize();
    return mem_size;

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
Block::GetBlockVariableList (bool can_create)
{
    if (m_parsed_block_variables == false)
    {
        if (m_variable_list_sp.get() == NULL && can_create)
        {
            m_parsed_block_variables = true;
            SymbolContext sc;
            CalculateSymbolContext(&sc);
            assert(sc.module_sp);
            sc.module_sp->GetSymbolVendor()->ParseVariablesForContext(sc);
        }
    }
    return m_variable_list_sp;
}

uint32_t
Block::AppendBlockVariables (bool can_create,
                             bool get_child_block_variables,
                             bool stop_if_child_block_is_inlined_function,
                             VariableList *variable_list)
{
    uint32_t num_variables_added = 0;
    VariableList *block_var_list = GetBlockVariableList (can_create).get();
    if (block_var_list)
    {
        num_variables_added += block_var_list->GetSize();
        variable_list->AddVariables (block_var_list);
    }
    
    if (get_child_block_variables)
    {
        for (Block *child_block = GetFirstChild(); 
             child_block != NULL; 
             child_block = child_block->GetSibling())
        {   
            if (stop_if_child_block_is_inlined_function == false || 
                child_block->GetInlinedFunctionInfo() == NULL)
            {
                num_variables_added += child_block->AppendBlockVariables (can_create,
                                                                          get_child_block_variables,
                                                                          stop_if_child_block_is_inlined_function,
                                                                          variable_list);
            }
        }
    }
    return num_variables_added;
}

uint32_t
Block::AppendVariables 
(
    bool can_create, 
    bool get_parent_variables, 
    bool stop_if_block_is_inlined_function,
    VariableList *variable_list
)
{
    uint32_t num_variables_added = 0;
    VariableListSP variable_list_sp(GetBlockVariableList(can_create));

    bool is_inlined_function = GetInlinedFunctionInfo() != NULL;
    if (variable_list_sp.get())
    {
        num_variables_added = variable_list_sp->GetSize();
        variable_list->AddVariables(variable_list_sp.get());
    }
    
    if (get_parent_variables)
    {
        if (stop_if_block_is_inlined_function && is_inlined_function)
            return num_variables_added;
            
        Block* parent_block = GetParent();
        if (parent_block)
            num_variables_added += parent_block->AppendVariables (can_create, get_parent_variables, stop_if_block_is_inlined_function, variable_list);
    }
    return num_variables_added;
}

clang::DeclContext *
Block::GetClangDeclContextForInlinedFunction()
{
    SymbolContext sc;
    
    CalculateSymbolContext (&sc);
    
    if (!sc.module_sp)
        return NULL;
    
    SymbolVendor *sym_vendor = sc.module_sp->GetSymbolVendor();
    
    if (!sym_vendor)
        return NULL;
    
    SymbolFile *sym_file = sym_vendor->GetSymbolFile();
    
    if (!sym_file)
        return NULL;
    
    return sym_file->GetClangDeclContextForTypeUID (sc, m_uid);
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

void
Block::SetDidParseVariables (bool b, bool set_children)
{
    m_parsed_block_variables = b;
    if (set_children)
    {
        for (Block *child_block = GetFirstChild(); child_block != NULL; child_block = child_block->GetSibling())
            child_block->SetDidParseVariables (b, true);
    }
}

