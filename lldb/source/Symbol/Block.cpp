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

Block::Block(user_id_t uid, uint32_t depth, BlockList* blocks) :
    UserID(uid),
    m_block_list(blocks),
    m_depth(depth),
    m_ranges(),
    m_inlineInfoSP(),
    m_variables()
{
}

Block::Block(const Block& rhs) :
    UserID(rhs),
    m_block_list(rhs.m_block_list),
    m_depth(rhs.m_depth),
    m_ranges(rhs.m_ranges),
    m_inlineInfoSP(rhs.m_inlineInfoSP),
    m_variables(rhs.m_variables)
{
}

const Block&
Block::operator= (const Block& rhs)
{
    if (this != &rhs)
    {
        UserID::operator= (rhs);
        m_block_list = rhs.m_block_list;
        m_depth = rhs.m_depth;
        m_ranges = rhs.m_ranges;
        m_inlineInfoSP = rhs.m_inlineInfoSP;
        m_variables = rhs.m_variables;
    }
    return *this;
}

Block::~Block ()
{
}

void
Block::GetDescription(Stream *s, lldb::DescriptionLevel level, Process *process) const
{
    size_t num_ranges = m_ranges.size();
    if (num_ranges)
    {
        
        addr_t base_addr = LLDB_INVALID_ADDRESS;
        if (process)
            base_addr = m_block_list->GetAddressRange().GetBaseAddress().GetLoadAddress(process);
        if (base_addr == LLDB_INVALID_ADDRESS)
            base_addr = m_block_list->GetAddressRange().GetBaseAddress().GetFileAddress();

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
        // We have a depth that is less than zero, print our parent blocks
        // first
        m_block_list->Dump(s, GetParentUID(), depth + 1, show_context);
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

        uint32_t blockID = m_block_list->GetFirstChild(GetID());
        while (blockID != Block::InvalidID)
        {
            m_block_list->Dump(s, blockID, depth - 1, show_context);

            blockID = m_block_list->GetSibling(blockID);
        }

        s->IndentLess();
    }

}


void
Block::CalculateSymbolContext(SymbolContext* sc)
{
    sc->block = this;
    m_block_list->GetFunction()->CalculateSymbolContext(sc);
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
    m_block_list->GetFunction()->DumpSymbolContext(s);
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



bool
BlockList::BlockContainsBlockWithID (const user_id_t block_id, const user_id_t find_block_id) const
{
    if (block_id == Block::InvalidID)
        return false;

    if (block_id == find_block_id)
        return true;
    else
    {
        user_id_t child_block_id = GetFirstChild(block_id);
        while (child_block_id != Block::InvalidID)
        {
            if (BlockContainsBlockWithID (child_block_id, find_block_id))
                return true;
            child_block_id = GetSibling(child_block_id);
        }
    }

    return false;
}

bool
Block::ContainsBlockWithID (user_id_t block_id) const
{
    return m_block_list->BlockContainsBlockWithID (GetID(), block_id);
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
Block::GetParent () const
{
    return m_block_list->GetBlockByID (m_block_list->GetParent(GetID()));
}

Block *
Block::GetSibling () const
{
    return m_block_list->GetBlockByID (m_block_list->GetSibling(GetID()));
}

Block *
Block::GetFirstChild () const
{
    return m_block_list->GetBlockByID (m_block_list->GetFirstChild(GetID()));
}

user_id_t
Block::GetParentUID() const
{
    return m_block_list->GetParent(GetID());
}

user_id_t
Block::GetSiblingUID() const
{
    return m_block_list->GetSibling(GetID());
}

user_id_t
Block::GetFirstChildUID() const
{
    return m_block_list->GetFirstChild(GetID());
}

user_id_t
Block::AddChild(user_id_t userID)
{
    return m_block_list->AddChild(GetID(), userID);
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
    if (m_variables.get() == NULL && can_create)
    {
        SymbolContext sc;
        CalculateSymbolContext(&sc);
        assert(sc.module_sp);
        sc.module_sp->GetSymbolVendor()->ParseVariablesForContext(sc);
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

uint32_t
Block::Depth () const
{
    return m_depth;
}

BlockList::BlockList(Function *function, const AddressRange& range) :
    m_function(function),
    m_range(range),
    m_blocks()
{
}

BlockList::~BlockList()
{
}

AddressRange &
BlockList::GetAddressRange()
{
    return m_range;
}

const AddressRange &
BlockList::GetAddressRange() const
{
    return m_range;
}

void
BlockList::Dump(Stream *s, user_id_t blockID, uint32_t depth, bool show_context) const
{
    const Block* block = GetBlockByID(blockID);
    if (block)
        block->Dump(s, m_range.GetBaseAddress().GetFileAddress(), depth, show_context);
}

Function *
BlockList::GetFunction()
{
    return m_function;
}


const Function *
BlockList::GetFunction() const
{
    return m_function;
}

user_id_t
BlockList::GetParent(user_id_t blockID) const
{
    collection::const_iterator end = m_blocks.end();
    collection::const_iterator begin = m_blocks.begin();
    collection::const_iterator pos = std::find_if(begin, end, UserID::IDMatches(blockID));

    if (pos != end && pos != begin && pos->Depth() > 0)
    {
        const uint32_t parent_depth = pos->Depth() - 1;

        while (--pos >= begin)
        {
            if (pos->Depth() == parent_depth)
                return pos->GetID();
        }
    }
    return Block::InvalidID;
}

user_id_t
BlockList::GetSibling(user_id_t blockID) const
{
    collection::const_iterator end = m_blocks.end();
    collection::const_iterator pos = std::find_if(m_blocks.begin(), end, UserID::IDMatches(blockID));

    if (pos != end)
    {
        const uint32_t sibling_depth = pos->Depth();
        while (++pos != end)
        {
            uint32_t depth = pos->Depth();
            if (depth == sibling_depth)
                return pos->GetID();
            if (depth < sibling_depth)
                break;
        }
    }
    return Block::InvalidID;
}

user_id_t
BlockList::GetFirstChild(user_id_t blockID) const
{
    if (!m_blocks.empty())
    {
        if (blockID == Block::RootID)
        {
            return m_blocks.front().GetID();
        }
        else
        {
            collection::const_iterator end = m_blocks.end();
            collection::const_iterator pos = std::find_if(m_blocks.begin(), end, UserID::IDMatches(blockID));

            if (pos != end)
            {
                collection::const_iterator child_pos = pos + 1;
                if (child_pos != end)
                {
                    if (child_pos->Depth() == pos->Depth() + 1)
                        return child_pos->GetID();
                }
            }
        }
    }
    return Block::InvalidID;
}


// Return the current number of bytes that this object occupies in memory
size_t
BlockList::MemorySize() const
{
    size_t mem_size = sizeof(BlockList);

    collection::const_iterator pos, end = m_blocks.end();
    for (pos = m_blocks.begin(); pos != end; ++pos)
        mem_size += pos->MemorySize();  // Each block can vary in size

    return mem_size;

}

user_id_t
BlockList::AddChild (user_id_t parentID, user_id_t childID)
{
    bool added = false;
    if (parentID == Block::RootID)
    {
        assert(m_blocks.empty());
        Block block(childID, 0, this);
        m_blocks.push_back(block);
        added = true;
    }
    else
    {
        collection::iterator end = m_blocks.end();
        collection::iterator parent_pos = std::find_if(m_blocks.begin(), end, UserID::IDMatches(parentID));
        assert(parent_pos != end);
        if (parent_pos != end)
        {
            const uint32_t parent_sibling_depth = parent_pos->Depth();

            collection::iterator insert_pos = parent_pos;
            collection::iterator prev_sibling = end;
            while (++insert_pos != end)
            {
                if (insert_pos->Depth() <= parent_sibling_depth)
                    break;
            }

            Block child_block(childID, parent_pos->Depth() + 1, this);
            collection::iterator child_pos = m_blocks.insert(insert_pos, child_block);
            added = true;
        }
    }
    if (added)
        return childID;
    return Block::InvalidID;
}

const Block *
BlockList::GetBlockByID(user_id_t blockID) const
{
    if (m_blocks.empty() || blockID == Block::InvalidID)
        return NULL;

    if (blockID == Block::RootID)
        blockID = m_blocks.front().GetID();

    collection::const_iterator end = m_blocks.end();
    collection::const_iterator pos = std::find_if(m_blocks.begin(), end, UserID::IDMatches(blockID));
    if (pos != end)
        return &(*pos);
    return NULL;
}

Block *
BlockList::GetBlockByID(user_id_t blockID)
{
    if (m_blocks.empty() || blockID == Block::InvalidID)
        return NULL;

    if (blockID == Block::RootID)
        blockID = m_blocks.front().GetID();

    collection::iterator end = m_blocks.end();
    collection::iterator pos = std::find_if(m_blocks.begin(), end, UserID::IDMatches(blockID));
    if (pos != end)
        return &(*pos);
    return NULL;
}

bool
BlockList::AddRange(user_id_t blockID, addr_t start_offset, addr_t end_offset)
{
    Block *block = GetBlockByID(blockID);

    if (block)
    {
        block->AddRange(start_offset, end_offset);
        return true;
    }
    return false;
}
//
//const Block *
//BlockList::FindDeepestBlockForAddress (const Address &addr)
//{
//    if (m_range.Contains(addr))
//    {
//        addr_t block_offset = addr.GetFileAddress() - m_range.GetBaseAddress().GetFileAddress();
//        collection::const_iterator pos, end = m_blocks.end();
//        collection::const_iterator deepest_match_pos = end;
//        for (pos = m_blocks.begin(); pos != end; ++pos)
//        {
//            if (pos->Contains (block_offset))
//            {
//                if (deepest_match_pos == end || deepest_match_pos->Depth() < pos->Depth())
//                    deepest_match_pos = pos;
//            }
//        }
//        if (deepest_match_pos != end)
//            return &(*deepest_match_pos);
//    }
//    return NULL;
//}
//
bool
BlockList::SetInlinedFunctionInfo(user_id_t blockID, const char *name, const char *mangled, const Declaration *decl_ptr, const Declaration *call_decl_ptr)
{
    Block *block = GetBlockByID(blockID);

    if (block)
    {
        block->SetInlinedFunctionInfo(name, mangled, decl_ptr, call_decl_ptr);
        return true;
    }
    return false;
}

VariableListSP
BlockList::GetVariableList(user_id_t blockID, bool get_child_variables, bool can_create)
{
    VariableListSP variable_list_sp;
    Block *block = GetBlockByID(blockID);
    if (block)
        variable_list_sp = block->GetVariableList(get_child_variables, can_create);
    return variable_list_sp;
}

bool
BlockList::IsEmpty() const
{
    return m_blocks.empty();
}



bool
BlockList::SetVariableList(user_id_t blockID, VariableListSP& variables)
{
    Block *block = GetBlockByID(blockID);
    if (block)
    {
        block->SetVariableList(variables);
        return true;
    }
    return false;

}
