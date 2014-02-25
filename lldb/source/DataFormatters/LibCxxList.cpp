//===-- LibCxxList.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/DataFormatters/CXXFormatterFunctions.h"

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Host/Endian.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

class ListEntry
{
public:
    ListEntry () {}
    ListEntry (ValueObjectSP entry_sp) : m_entry_sp(entry_sp) {}
    ListEntry (const ListEntry& rhs) : m_entry_sp(rhs.m_entry_sp) {}
    ListEntry (ValueObject* entry) : m_entry_sp(entry ? entry->GetSP() : ValueObjectSP()) {}
    
    ListEntry
    next ()
    {
        if (!m_entry_sp)
            return ListEntry();
        return ListEntry(m_entry_sp->GetChildMemberWithName(ConstString("__next_"), true));
    }
    
    ListEntry
    prev ()
    {
        if (!m_entry_sp)
            return ListEntry();
        return ListEntry(m_entry_sp->GetChildMemberWithName(ConstString("__prev_"), true));
    }
    
    uint64_t
    value ()
    {
        if (!m_entry_sp)
            return 0;
        return m_entry_sp->GetValueAsUnsigned(0);
    }

    bool
    null()
    {
        return (value() == 0);
    }
    
    explicit operator bool ()
    {
        return GetEntry().get() != nullptr && null() == false;
    }
    
    ValueObjectSP
    GetEntry ()
    {
        return m_entry_sp;
    }
    
    void
    SetEntry (ValueObjectSP entry)
    {
        m_entry_sp = entry;
    }
    
    bool
    operator == (const ListEntry& rhs) const
    {
        return (rhs.m_entry_sp.get() == m_entry_sp.get());
    }
    
private:
    ValueObjectSP m_entry_sp;
};

class ListIterator
{
public:
    ListIterator () {}
    ListIterator (ListEntry entry) : m_entry(entry) {}
    ListIterator (ValueObjectSP entry) : m_entry(entry) {}
    ListIterator (const ListIterator& rhs) : m_entry(rhs.m_entry) {}
    ListIterator (ValueObject* entry) : m_entry(entry) {}

    ValueObjectSP
    value ()
    {
        return m_entry.GetEntry();
    }
    
    ValueObjectSP
    advance (size_t count)
    {
        if (count == 0)
            return m_entry.GetEntry();
        if (count == 1)
        {
            next ();
            return m_entry.GetEntry();
        }
        while (count > 0)
        {
            next ();
            count--;
            if (m_entry.null())
                return lldb::ValueObjectSP();
        }
        return m_entry.GetEntry();
    }
    
    bool
    operator == (const ListIterator& rhs) const
    {
        return (rhs.m_entry == m_entry);
    }
    
protected:
    void
    next ()
    {
        m_entry = m_entry.next();
    }
    
    void
    prev ()
    {
        m_entry = m_entry.prev();
    }
private:
    ListEntry m_entry;
};

lldb_private::formatters::LibcxxStdListSyntheticFrontEnd::LibcxxStdListSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get()),
m_list_capping_size(0),
m_node_address(),
m_head(NULL),
m_tail(NULL),
m_element_type(),
m_count(UINT32_MAX),
m_children()
{
    if (valobj_sp)
        Update();
}

bool
lldb_private::formatters::LibcxxStdListSyntheticFrontEnd::HasLoop()
{
    if (g_use_loop_detect == false)
        return false;
    // don't bother checking for a loop if we won't actually need to jump nodes
    if (m_count < 2)
        return false;
    auto steps_left = m_count;
    ListEntry slow(m_head);
    ListEntry fast(m_head);
    while (steps_left-- > 0)
    {
        slow = slow.next();
        fast = fast.next();
        if (fast.next())
            fast = fast.next().next();
        else
            fast = nullptr;
        if (!slow || !fast)
            return false;
        if (slow == fast)
            return true;
    }
    return false;
}

size_t
lldb_private::formatters::LibcxxStdListSyntheticFrontEnd::CalculateNumChildren ()
{
    if (m_count != UINT32_MAX)
        return m_count;
    if (!m_head || !m_tail || m_node_address == 0)
        return 0;
    ValueObjectSP size_alloc(m_backend.GetChildMemberWithName(ConstString("__size_alloc_"), true));
    if (size_alloc)
    {
        ValueObjectSP first(size_alloc->GetChildMemberWithName(ConstString("__first_"), true));
        if (first)
        {
            m_count = first->GetValueAsUnsigned(UINT32_MAX);
        }
    }
    if (m_count != UINT32_MAX)
    {
        if (!HasLoop())
            return m_count;
        return m_count = 0;
    }
    else
    {
        uint64_t next_val = m_head->GetValueAsUnsigned(0);
        uint64_t prev_val = m_tail->GetValueAsUnsigned(0);
        if (next_val == 0 || prev_val == 0)
            return 0;
        if (next_val == m_node_address)
            return 0;
        if (next_val == prev_val)
            return 1;
        if (HasLoop())
            return 0;
        uint64_t size = 2;
        ListEntry current(m_head);
        while (current.next() && current.next().value() != m_node_address)
        {
            size++;
            current = current.next();
            if (size > m_list_capping_size)
                break;
        }
        return m_count = (size-1);
    }
}

lldb::ValueObjectSP
lldb_private::formatters::LibcxxStdListSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    if (idx >= CalculateNumChildren())
        return lldb::ValueObjectSP();
    
    if (!m_head || !m_tail || m_node_address == 0)
        return lldb::ValueObjectSP();
    
    auto cached = m_children.find(idx);
    if (cached != m_children.end())
        return cached->second;
    
    ListIterator current(m_head);
    ValueObjectSP current_sp(current.advance(idx));
    if (!current_sp)
        return lldb::ValueObjectSP();
    current_sp = current_sp->GetChildMemberWithName(ConstString("__value_"), true);
    if (!current_sp)
        return lldb::ValueObjectSP();
    // we need to copy current_sp into a new object otherwise we will end up with all items named __value_
    DataExtractor data;
    current_sp->GetData(data);
    StreamString name;
    name.Printf("[%zu]",idx);
    return (m_children[idx] = ValueObject::CreateValueObjectFromData(name.GetData(), data, m_backend.GetExecutionContextRef(), m_element_type));
}

bool
lldb_private::formatters::LibcxxStdListSyntheticFrontEnd::Update()
{
    m_head = m_tail = NULL;
    m_node_address = 0;
    m_count = UINT32_MAX;
    Error err;
    ValueObjectSP backend_addr(m_backend.AddressOf(err));
    m_list_capping_size = 0;
    if (m_backend.GetTargetSP())
        m_list_capping_size = m_backend.GetTargetSP()->GetMaximumNumberOfChildrenToDisplay();
    if (m_list_capping_size == 0)
        m_list_capping_size = 255;
    if (err.Fail() || backend_addr.get() == NULL)
        return false;
    m_node_address = backend_addr->GetValueAsUnsigned(0);
    if (!m_node_address || m_node_address == LLDB_INVALID_ADDRESS)
        return false;
    ValueObjectSP impl_sp(m_backend.GetChildMemberWithName(ConstString("__end_"),true));
    if (!impl_sp)
        return false;
    ClangASTType list_type = m_backend.GetClangType();
    if (list_type.IsReferenceType())
        list_type = list_type.GetNonReferenceType();

    if (list_type.GetNumTemplateArguments() == 0)
        return false;
    lldb::TemplateArgumentKind kind;
    m_element_type = list_type.GetTemplateArgument(0, kind);
    m_head = impl_sp->GetChildMemberWithName(ConstString("__next_"), true).get();
    m_tail = impl_sp->GetChildMemberWithName(ConstString("__prev_"), true).get();
    return false;
}

bool
lldb_private::formatters::LibcxxStdListSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::LibcxxStdListSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    return ExtractIndexFromString(name.GetCString());
}

lldb_private::formatters::LibcxxStdListSyntheticFrontEnd::~LibcxxStdListSyntheticFrontEnd ()
{}

SyntheticChildrenFrontEnd*
lldb_private::formatters::LibcxxStdListSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    if (!valobj_sp)
        return NULL;
    return (new LibcxxStdListSyntheticFrontEnd(valobj_sp));
}

