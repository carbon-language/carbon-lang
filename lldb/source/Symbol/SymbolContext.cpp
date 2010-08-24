//===-- SymbolContext.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/SymbolContext.h"

#include "lldb/Core/Module.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

SymbolContext::SymbolContext() :
    target_sp   (),
    module_sp   (),
    comp_unit   (NULL),
    function    (NULL),
    block       (NULL),
    line_entry  (),
    symbol      (NULL)
{
}

SymbolContext::SymbolContext(const ModuleSP& m, CompileUnit *cu, Function *f, Block *b, LineEntry *le, Symbol *s) :
    target_sp   (),
    module_sp   (m),
    comp_unit   (cu),
    function    (f),
    block       (b),
    line_entry  (),
    symbol      (s)
{
    if (le)
        line_entry = *le;
}

SymbolContext::SymbolContext(const TargetSP &t, const ModuleSP& m, CompileUnit *cu, Function *f, Block *b, LineEntry *le, Symbol *s) :
    target_sp   (t),
    module_sp   (m),
    comp_unit   (cu),
    function    (f),
    block       (b),
    line_entry  (),
    symbol      (s)
{
    if (le)
        line_entry = *le;
}

SymbolContext::SymbolContext(const SymbolContext& rhs) :
    target_sp   (rhs.target_sp),
    module_sp   (rhs.module_sp),
    comp_unit   (rhs.comp_unit),
    function    (rhs.function),
    block       (rhs.block),
    line_entry  (rhs.line_entry),
    symbol      (rhs.symbol)
{
}


SymbolContext::SymbolContext (SymbolContextScope *sc_scope) :
    target_sp   (),
    module_sp   (),
    comp_unit   (NULL),
    function    (NULL),
    block       (NULL),
    line_entry  (),
    symbol      (NULL)
{
    sc_scope->CalculateSymbolContext (this);
}

const SymbolContext&
SymbolContext::operator= (const SymbolContext& rhs)
{
    if (this != &rhs)
    {
        target_sp   = rhs.target_sp;
        module_sp   = rhs.module_sp;
        comp_unit   = rhs.comp_unit;
        function    = rhs.function;
        block       = rhs.block;
        line_entry  = rhs.line_entry;
        symbol      = rhs.symbol;
    }
    return *this;
}

void
SymbolContext::Clear()
{
    target_sp.reset();
    module_sp.reset();
    comp_unit   = NULL;
    function    = NULL;
    block       = NULL;
    line_entry.Clear();
    symbol      = NULL;
}

void
SymbolContext::DumpStopContext
(
    Stream *s,
    ExecutionContextScope *exe_scope,
    const Address &addr,
    bool show_module,
    bool show_inlined_frames
) const
{
    if (show_module && module_sp)
    {
        *s << module_sp->GetFileSpec().GetFilename() << '`';
    }

    if (function != NULL)
    {
        if (function->GetMangled().GetName())
            function->GetMangled().GetName().Dump(s);


        if (show_inlined_frames && block)
        {
            InlineFunctionInfo *inline_info = block->InlinedFunctionInfo();
            if (inline_info == NULL)
            {
                Block *parent_inline_block = block->GetInlinedParent();
                if (parent_inline_block)
                    inline_info = parent_inline_block->InlinedFunctionInfo();
            }

            if (inline_info)
            {
                s->PutCString(" [inlined] ");
                inline_info->GetName().Dump(s);
                
                if (line_entry.IsValid())
                {
                    s->PutCString(" at ");
                    line_entry.DumpStopContext(s);
                }
                return;
            }
        }
        const addr_t function_offset = addr.GetOffset() - function->GetAddressRange().GetBaseAddress().GetOffset();
        if (function_offset)
            s->Printf(" + %llu", function_offset);

        if (block != NULL)
        {
            s->IndentMore();
            block->DumpStopContext(s, this);
            s->IndentLess();
        }
        else
        {
            if (line_entry.IsValid())
            {
                s->PutCString(" at ");
                if (line_entry.DumpStopContext(s))
                    return;
            }
        }
    }
    else if (symbol != NULL)
    {
        symbol->GetMangled().GetName().Dump(s);

        if (symbol->GetAddressRangePtr())
        {
            const addr_t symbol_offset = addr.GetOffset() - symbol->GetAddressRangePtr()->GetBaseAddress().GetOffset();
            if (symbol_offset)
                s->Printf(" + %llu", symbol_offset);
        }
    }
    else
    {
        addr.Dump(s, exe_scope, Address::DumpStyleModuleWithFileAddress);
    }
}

void
SymbolContext::GetDescription(Stream *s, lldb::DescriptionLevel level, Process *process) const
{
    if (module_sp)
    {
        s->Indent("     Module: \"");
        module_sp->GetFileSpec().Dump(s);
        s->PutChar('"');
        s->EOL();
    }

    if (comp_unit != NULL)
    {
        s->Indent("CompileUnit: ");
        comp_unit->GetDescription (s, level);
        s->EOL();
    }

    if (function != NULL)
    {
        s->Indent("   Function: ");
        function->GetDescription (s, level, process);
        s->EOL();

        Type *func_type = function->GetType();
        if (func_type)
        {
            s->Indent("   FuncType: ");
            func_type->GetDescription (s, level, false);
            s->EOL();
        }
    }

    if (block != NULL)
    {
        std::vector<Block *> blocks;
        blocks.push_back (block);
        Block *parent_block = block->GetParent();
        
        while (parent_block)
        {
            blocks.push_back (parent_block);
            parent_block = parent_block->GetParent();
        }
        std::vector<Block *>::reverse_iterator pos;        
        std::vector<Block *>::reverse_iterator begin = blocks.rbegin();
        std::vector<Block *>::reverse_iterator end = blocks.rend();
        for (pos = begin; pos != end; ++pos)
        {
            if (pos == begin)
                s->Indent("     Blocks: ");
            else
                s->Indent("             ");
            (*pos)->GetDescription(s, function, level, process);
            s->EOL();
        }
    }

    if (line_entry.IsValid())
    {
        s->Indent("  LineEntry: ");
        line_entry.GetDescription (s, level, comp_unit, process, false);
        s->EOL();
    }

    if (symbol != NULL)
    {
        s->Indent("     Symbol: ");
        symbol->GetDescription(s, level, process);
        s->EOL();
    }
}

uint32_t
SymbolContext::GetResolvedMask () const
{
    uint32_t resolved_mask = 0;
    if (target_sp)              resolved_mask |= eSymbolContextTarget;
    if (module_sp)              resolved_mask |= eSymbolContextModule;
    if (comp_unit)              resolved_mask |= eSymbolContextCompUnit;
    if (function)               resolved_mask |= eSymbolContextFunction;
    if (block)                  resolved_mask |= eSymbolContextBlock;
    if (line_entry.IsValid())   resolved_mask |= eSymbolContextLineEntry;
    if (symbol)                 resolved_mask |= eSymbolContextSymbol;
    return resolved_mask;
}


void
SymbolContext::Dump(Stream *s, Process *process) const
{
    *s << (void *)this << ": ";
    s->Indent();
    s->PutCString("SymbolContext");
    s->IndentMore();
    s->EOL();
    s->IndentMore();
    s->Indent();
    *s << "Module       = " << (void *)module_sp.get() << ' ';
    if (module_sp)
        module_sp->GetFileSpec().Dump(s);
    s->EOL();
    s->Indent();
    *s << "CompileUnit  = " << (void *)comp_unit;
    if (comp_unit != NULL)
        *s << " {0x" << comp_unit->GetID() << "} " << *(dynamic_cast<FileSpec*> (comp_unit));
    s->EOL();
    s->Indent();
    *s << "Function     = " << (void *)function;
    if (function != NULL)
    {
        *s << " {0x" << function->GetID() << "} " << function->GetType()->GetName() << ", address-range = ";
        function->GetAddressRange().Dump(s, process, Address::DumpStyleLoadAddress, Address::DumpStyleModuleWithFileAddress);
        s->EOL();
        s->Indent();
        Type* func_type = function->GetType();
        if (func_type)
        {
            *s << "        Type = ";
            func_type->Dump (s, false);
        }
    }
    s->EOL();
    s->Indent();
    *s << "Block        = " << (void *)block;
    if (block != NULL)
        *s << " {0x" << block->GetID() << '}';
    // Dump the block and pass it a negative depth to we print all the parent blocks
    //if (block != NULL)
    //  block->Dump(s, function->GetFileAddress(), INT_MIN);
    s->EOL();
    s->Indent();
    *s << "LineEntry    = ";
    line_entry.Dump (s, process, true, Address::DumpStyleLoadAddress, Address::DumpStyleModuleWithFileAddress, true);
    s->EOL();
    s->Indent();
    *s << "Symbol       = " << (void *)symbol;
    if (symbol != NULL && symbol->GetMangled())
        *s << ' ' << symbol->GetMangled().GetName().AsCString();
    s->EOL();
    s->IndentLess();
    s->IndentLess();
}

bool
lldb_private::operator== (const SymbolContext& lhs, const SymbolContext& rhs)
{
    return lhs.target_sp.get() == rhs.target_sp.get() &&
           lhs.module_sp.get() == rhs.module_sp.get() &&
           lhs.comp_unit    == rhs.comp_unit &&
           lhs.function     == rhs.function &&
           LineEntry::Compare(lhs.line_entry, rhs.line_entry) == 0 &&
           lhs.symbol       == rhs.symbol;
}

bool
lldb_private::operator!= (const SymbolContext& lhs, const SymbolContext& rhs)
{
    return lhs.target_sp.get() != rhs.target_sp.get() ||
           lhs.module_sp.get() != rhs.module_sp.get() ||
           lhs.comp_unit    != rhs.comp_unit ||
           lhs.function     != rhs.function ||
           LineEntry::Compare(lhs.line_entry, rhs.line_entry) != 0 ||
           lhs.symbol       != rhs.symbol;
}

bool
SymbolContext::GetAddressRange (uint32_t scope, AddressRange &range) const
{
    if ((scope & eSymbolContextLineEntry) && line_entry.IsValid())
    {
        range = line_entry.range;
        return true;
    }
    else if ((scope & eSymbolContextFunction) && function != NULL)
    {
        range = function->GetAddressRange();
        return true;
    }
    else if ((scope & eSymbolContextSymbol) && symbol != NULL && symbol->GetAddressRangePtr())
    {
        range = *symbol->GetAddressRangePtr();

        if (range.GetByteSize() == 0)
        {
            if (module_sp)
            {
                ObjectFile *objfile = module_sp->GetObjectFile();
                if (objfile)
                {
                    Symtab *symtab = objfile->GetSymtab();
                    if (symtab)
                        range.SetByteSize(symtab->CalculateSymbolSize (symbol));
                }
            }
        }
        return true;
    }
    range.Clear();
    return false;
}


size_t
SymbolContext::FindFunctionsByName (const ConstString &name, bool append, SymbolContextList &sc_list) const
{    
    if (!append)
        sc_list.Clear();
    
    if (function != NULL)
    {
        // FIXME: Look in the class of the current function, if it exists,
        // for methods matching name.
    }

    if (comp_unit != NULL)
    {
        // Make sure we've read in all the functions.  We should be able to check and see
        // if there's one by this name present before we do this...
        module_sp->GetSymbolVendor()->ParseCompileUnitFunctions(*this);
        uint32_t func_idx;
        lldb::FunctionSP func_sp;
        for (func_idx = 0; (func_sp = comp_unit->GetFunctionAtIndex(func_idx)) != NULL; ++func_idx)
        {
            if (func_sp->GetMangled().GetName() == name)
            {
                SymbolContext sym_ctx(target_sp,
                                      module_sp,
                                      comp_unit,
                                      func_sp.get());
                sc_list.Append(sym_ctx);
            }
        }
    }
    
    if (module_sp != NULL)
        module_sp->FindFunctions (name, eFunctionNameTypeBase | eFunctionNameTypeFull, true, sc_list);

    if (target_sp)
        target_sp->GetImages().FindFunctions (name, eFunctionNameTypeBase | eFunctionNameTypeFull, true, sc_list);

    return sc_list.GetSize();
}

lldb::VariableSP
SymbolContext::FindVariableByName (const char *name) const
{
    lldb::VariableSP return_value;
    return return_value;
}

lldb::TypeSP
SymbolContext::FindTypeByName (const ConstString &name) const
{
    lldb::TypeSP return_value;
        
    TypeList types;
    
    if (module_sp && module_sp->FindTypes (*this, name, false, 1, types))
        return types.GetTypeAtIndex(0);
    
    if (!return_value.get() && target_sp && target_sp->GetImages().FindTypes (*this, name, false, 1, types))
        return types.GetTypeAtIndex(0);
    
    return return_value;
}

//----------------------------------------------------------------------
//
//  SymbolContextList
//
//----------------------------------------------------------------------


SymbolContextList::SymbolContextList() :
    m_symbol_contexts()
{
}

SymbolContextList::~SymbolContextList()
{
}

void
SymbolContextList::Append(const SymbolContext& sc)
{
    m_symbol_contexts.push_back(sc);
}

void
SymbolContextList::Clear()
{
    m_symbol_contexts.clear();
}

void
SymbolContextList::Dump(Stream *s, Process *process) const
{

    *s << (void *)this << ": ";
    s->Indent();
    s->PutCString("SymbolContextList");
    s->EOL();
    s->IndentMore();

    collection::const_iterator pos, end = m_symbol_contexts.end();
    for (pos = m_symbol_contexts.begin(); pos != end; ++pos)
    {
        pos->Dump(s, process);
    }
    s->IndentLess();
}

bool
SymbolContextList::GetContextAtIndex(uint32_t idx, SymbolContext& sc) const
{
    if (idx < m_symbol_contexts.size())
    {
        sc = m_symbol_contexts[idx];
        return true;
    }
    return false;
}

bool
SymbolContextList::RemoveContextAtIndex (uint32_t idx)
{
    if (idx < m_symbol_contexts.size())
    {
        m_symbol_contexts.erase(m_symbol_contexts.begin() + idx);
        return true;
    }
    return false;
}

uint32_t
SymbolContextList::GetSize() const
{
    return m_symbol_contexts.size();
}
