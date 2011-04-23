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
#include "lldb/Interpreter/Args.h"
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
    bool show_fullpaths,
    bool show_module,
    bool show_inlined_frames
) const
{
    if (show_module && module_sp)
    {
        if (show_fullpaths)
            *s << module_sp->GetFileSpec();
        else
            *s << module_sp->GetFileSpec().GetFilename();
        s->PutChar('`');
    }

    if (function != NULL)
    {
        if (function->GetMangled().GetName())
            function->GetMangled().GetName().Dump(s);

        if (addr.IsValid())
        {
            const addr_t function_offset = addr.GetOffset() - function->GetAddressRange().GetBaseAddress().GetOffset();
            if (function_offset)
                s->Printf(" + %llu", function_offset);
        }

        if (block != NULL)
        {
            s->IndentMore();
            block->DumpStopContext (s, this, NULL, show_fullpaths, show_inlined_frames);
            s->IndentLess();
        }
        else
        {
            if (line_entry.IsValid())
            {
                s->PutCString(" at ");
                if (line_entry.DumpStopContext(s, show_fullpaths))
                    return;
            }
        }
    }
    else if (symbol != NULL)
    {
        symbol->GetMangled().GetName().Dump(s);

        if (addr.IsValid() && symbol->GetAddressRangePtr())
        {
            const addr_t symbol_offset = addr.GetOffset() - symbol->GetAddressRangePtr()->GetBaseAddress().GetOffset();
            if (symbol_offset)
                s->Printf(" + %llu", symbol_offset);
        }
    }
    else if (addr.IsValid())
    {
        addr.Dump(s, exe_scope, Address::DumpStyleModuleWithFileAddress);
    }
}

void
SymbolContext::GetDescription(Stream *s, lldb::DescriptionLevel level, Target *target) const
{
    if (module_sp)
    {
        s->Indent("     Module: file = \"");
        module_sp->GetFileSpec().Dump(s);
        *s << '"';
        if (module_sp->GetArchitecture().IsValid())
            s->Printf (", arch = \"%s\"", module_sp->GetArchitecture().GetArchitectureName());
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
        function->GetDescription (s, level, target);
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
            (*pos)->GetDescription(s, function, level, target);
            s->EOL();
        }
    }

    if (line_entry.IsValid())
    {
        s->Indent("  LineEntry: ");
        line_entry.GetDescription (s, level, comp_unit, target, false);
        s->EOL();
    }

    if (symbol != NULL)
    {
        s->Indent("     Symbol: ");
        symbol->GetDescription(s, level, target);
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
SymbolContext::Dump(Stream *s, Target *target) const
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
        *s << " {0x" << comp_unit->GetID() << "} " << *(static_cast<FileSpec*> (comp_unit));
    s->EOL();
    s->Indent();
    *s << "Function     = " << (void *)function;
    if (function != NULL)
    {
        *s << " {0x" << function->GetID() << "} " << function->GetType()->GetName() << ", address-range = ";
        function->GetAddressRange().Dump(s, target, Address::DumpStyleLoadAddress, Address::DumpStyleModuleWithFileAddress);
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
    line_entry.Dump (s, target, true, Address::DumpStyleLoadAddress, Address::DumpStyleModuleWithFileAddress, true);
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
    return  lhs.function == rhs.function
            && lhs.symbol == rhs.symbol 
            && lhs.module_sp.get() == rhs.module_sp.get()
            && lhs.comp_unit == rhs.comp_unit
            && lhs.target_sp.get() == rhs.target_sp.get() 
            && LineEntry::Compare(lhs.line_entry, rhs.line_entry) == 0;
}

bool
lldb_private::operator!= (const SymbolContext& lhs, const SymbolContext& rhs)
{
    return  lhs.function != rhs.function
            || lhs.symbol != rhs.symbol 
            || lhs.module_sp.get() != rhs.module_sp.get()
            || lhs.comp_unit != rhs.comp_unit
            || lhs.target_sp.get() != rhs.target_sp.get() 
            || LineEntry::Compare(lhs.line_entry, rhs.line_entry) != 0;
}

bool
SymbolContext::GetAddressRange (uint32_t scope, 
                                uint32_t range_idx, 
                                bool use_inline_block_range,
                                AddressRange &range) const
{
    if ((scope & eSymbolContextLineEntry) && line_entry.IsValid())
    {
        range = line_entry.range;
        return true;
    }
    
    if ((scope & eSymbolContextBlock) && (block != NULL))
    {
        if (use_inline_block_range)
        {
            Block *inline_block = block->GetContainingInlinedBlock();
            if (inline_block)
                return inline_block->GetRangeAtIndex (range_idx, range);
        }
        else
        {
            return block->GetRangeAtIndex (range_idx, range);
        }
    }

    if ((scope & eSymbolContextFunction) && (function != NULL))
    {
        if (range_idx == 0)
        {
            range = function->GetAddressRange();
            return true;
        }            
    } 
    
    if ((scope & eSymbolContextSymbol) && (symbol != NULL) && (symbol->GetAddressRangePtr() != NULL))
    {
        if (range_idx == 0)
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
    }
    range.Clear();
    return false;
}

ClangNamespaceDecl
SymbolContext::FindNamespace (const ConstString &name) const
{
    ClangNamespaceDecl namespace_decl;
    if (module_sp)
        namespace_decl = module_sp->GetSymbolVendor()->FindNamespace (*this, name);
    return namespace_decl;
}

size_t
SymbolContext::FindFunctionsByName (const ConstString &name, 
                                    bool include_symbols, 
                                    bool append, 
                                    SymbolContextList &sc_list) const
{    
    if (!append)
        sc_list.Clear();
    
    if (function != NULL)
    {
        // FIXME: Look in the class of the current function, if it exists,
        // for methods matching name.
    }

    if (module_sp != NULL)
        module_sp->FindFunctions (name, eFunctionNameTypeBase | eFunctionNameTypeFull, include_symbols, true, sc_list);

    if (target_sp)
        target_sp->GetImages().FindFunctions (name, eFunctionNameTypeBase | eFunctionNameTypeFull, include_symbols, true, sc_list);

    return sc_list.GetSize();
}

//lldb::VariableSP
//SymbolContext::FindVariableByName (const char *name) const
//{
//    lldb::VariableSP return_value;
//    return return_value;
//}

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
//  SymbolContextSpecifier
//
//----------------------------------------------------------------------

bool
SymbolContextSpecifier::AddLineSpecification (uint32_t line_no, SpecificationType type)
{
    bool return_value = true;
    switch (type)
    {
    case eNothingSpecified:
        Clear();
        break;
    case eLineStartSpecified:
        m_start_line = line_no;
        m_type |= eLineStartSpecified;
        break;
    case eLineEndSpecified:
        m_end_line = line_no;
        m_type |= eLineEndSpecified;
        break;
    default:
        return_value = false;
        break;
    }
    return return_value;
}

bool
SymbolContextSpecifier::AddSpecification (const char *spec_string, SpecificationType type)
{
    bool return_value = true;
    switch (type)
    {
    case eNothingSpecified:
        Clear();
        break;
    case eModuleSpecified:
        {
            // See if we can find the Module, if so stick it in the SymbolContext.
            FileSpec module_spec(spec_string, false);
            lldb::ModuleSP module_sp = m_target_sp->GetImages().FindFirstModuleForFileSpec (module_spec, NULL, NULL);
            m_type |= eModuleSpecified;
            if (module_sp)
                m_module_sp = module_sp;
            else
                m_module_spec.assign (spec_string);
        }
        break;
    case eFileSpecified:
        // CompUnits can't necessarily be resolved here, since an inlined function might show up in 
        // a number of CompUnits.  Instead we just convert to a FileSpec and store it away.
        m_file_spec_ap.reset (new FileSpec (spec_string, false));
        m_type |= eFileSpecified;
        break;
    case eLineStartSpecified:
        m_start_line = Args::StringToSInt32(spec_string, 0, 0, &return_value);
        if (return_value)
            m_type |= eLineStartSpecified;
        break;
    case eLineEndSpecified:
        m_end_line = Args::StringToSInt32(spec_string, 0, 0, &return_value);
        if (return_value)
            m_type |= eLineEndSpecified;
        break;
    case eFunctionSpecified:
        m_function_spec.assign(spec_string);
        m_type |= eFunctionSpecified;
        break;
    case eClassOrNamespaceSpecified:
        Clear();
        m_class_name.assign (spec_string);
        m_type = eClassOrNamespaceSpecified;
        break;
    case eAddressRangeSpecified:
        // Not specified yet...
        break;
    }
    
    return return_value;
}

void
SymbolContextSpecifier::Clear()
{
    m_module_spec.clear();
    m_file_spec_ap.reset();
    m_function_spec.clear();
    m_class_name.clear();
    m_start_line = 0;
    m_end_line = 0;
    m_address_range_ap.reset();
    
    m_type = eNothingSpecified;
}

bool
SymbolContextSpecifier::SymbolContextMatches(SymbolContext &sc)
{
    if (m_type == eNothingSpecified)
        return true;
        
    if (m_target_sp.get() != sc.target_sp.get())
        return false;
        
    if (m_type & eModuleSpecified)
    {
        if (sc.module_sp)
        {
            if (m_module_sp.get() != NULL)
            { 
                if (m_module_sp.get() != sc.module_sp.get())
                    return false;
            }
            else
            {
                FileSpec module_file_spec (m_module_spec.c_str(), false);
                if (!FileSpec::Equal (module_file_spec, sc.module_sp->GetFileSpec(), false))
                    return false;
            }
        }
    }
    if (m_type & eFileSpecified)
    {
        if (m_file_spec_ap.get())
        {
            // If we don't have a block or a comp_unit, then we aren't going to match a source file.
            if (sc.block == NULL && sc.comp_unit == NULL)
                return false;
                
            // Check if the block is present, and if so is it inlined:
            bool was_inlined = false;
            if (sc.block != NULL)
            {
                const InlineFunctionInfo *inline_info = sc.block->GetInlinedFunctionInfo();
                if (inline_info != NULL)
                {
                    was_inlined = true;
                    if (!FileSpec::Equal (inline_info->GetDeclaration().GetFile(), *(m_file_spec_ap.get()), false))
                        return false;
                }
            }
            
            // Next check the comp unit, but only if the SymbolContext was not inlined.
            if (!was_inlined && sc.comp_unit != NULL)
            {
                if (!FileSpec::Equal (*(sc.comp_unit), *(m_file_spec_ap.get()), false))
                    return false;
            }
        }
    }
    if (m_type & eLineStartSpecified 
        || m_type & eLineEndSpecified)
    {
        if (sc.line_entry.line < m_start_line || sc.line_entry.line > m_end_line)
            return false;
    }
    
    if (m_type & eFunctionSpecified)
    {
        // First check the current block, and if it is inlined, get the inlined function name:
        bool was_inlined = false;
        ConstString func_name(m_function_spec.c_str());
        
        if (sc.block != NULL)
        {
            const InlineFunctionInfo *inline_info = sc.block->GetInlinedFunctionInfo();
            if (inline_info != NULL)
            {
                was_inlined = true;
                const Mangled &name = inline_info->GetMangled();
                if (!name.NameMatches (func_name))
                    return false;
            }
        }
        //  If it wasn't inlined, check the name in the function or symbol:
        if (!was_inlined)
        {
            if (sc.function != NULL)
            {
                if (!sc.function->GetMangled().NameMatches(func_name))
                    return false;
            }
            else if (sc.symbol != NULL)
            {
                if (!sc.symbol->GetMangled().NameMatches(func_name))
                    return false;
            }
        }
        
            
    }
    
    return true;
}

bool
SymbolContextSpecifier::AddressMatches(lldb::addr_t addr)
{
    if (m_type & eAddressRangeSpecified)
    {
    
    }
    else
    {
        Address match_address (addr, NULL);
        SymbolContext sc;
        m_target_sp->GetImages().ResolveSymbolContextForAddress(match_address, eSymbolContextEverything, sc);
        return SymbolContextMatches(sc);
    }
    return true;
}

void
SymbolContextSpecifier::GetDescription (Stream *s, lldb::DescriptionLevel level) const
{
    char path_str[PATH_MAX + 1];

    if (m_type == eNothingSpecified)
    {
        s->Printf ("Nothing specified.\n");
    }
    
    if (m_type == eModuleSpecified)
    {
        s->Indent();
        if (m_module_sp)
        {
            m_module_sp->GetFileSpec().GetPath (path_str, PATH_MAX);
            s->Printf ("Module: %s\n", path_str);
        }
        else
            s->Printf ("Module: %s\n", m_module_spec.c_str());
    }
    
    if (m_type == eFileSpecified  && m_file_spec_ap.get() != NULL)
    {
        m_file_spec_ap->GetPath (path_str, PATH_MAX);
        s->Indent();
        s->Printf ("File: %s", path_str);
        if (m_type == eLineStartSpecified)
        {
            s->Printf (" from line %d", m_start_line);
            if (m_type == eLineEndSpecified)
                s->Printf ("to line %d", m_end_line);
            else
                s->Printf ("to end", m_end_line);
        }
        else if (m_type == eLineEndSpecified)
        {
            s->Printf (" from start to line %d", m_end_line);
        }
        s->Printf (".\n");
    }
    
    if (m_type == eLineStartSpecified)
    {
        s->Indent();
        s->Printf ("From line %d", m_start_line);
        if (m_type == eLineEndSpecified)
            s->Printf ("to line %d", m_end_line);
        else
            s->Printf ("to end", m_end_line);
        s->Printf (".\n");
    }
    else if (m_type == eLineEndSpecified)
    {
        s->Printf ("From start to line %d.\n", m_end_line);
    }
    
    if (m_type == eFunctionSpecified)
    {
        s->Indent();
        s->Printf ("Function: %s.\n", m_function_spec.c_str());
    }
    
    if (m_type == eClassOrNamespaceSpecified)
    {
        s->Indent();
        s->Printf ("Class name: %s.\n", m_class_name.c_str());
    }
    
    if (m_type == eAddressRangeSpecified && m_address_range_ap.get() != NULL)
    {
        s->Indent();
        s->PutCString ("Address range: ");
        m_address_range_ap->Dump (s, m_target_sp.get(), Address::DumpStyleLoadAddress, Address::DumpStyleFileAddress);
        s->PutCString ("\n");
    }
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

bool
SymbolContextList::AppendIfUnique (const SymbolContext& sc, bool merge_symbol_into_function)
{
    collection::iterator pos, end = m_symbol_contexts.end();
    for (pos = m_symbol_contexts.begin(); pos != end; ++pos)
    {
        if (*pos == sc)
            return false;
    }
    if (merge_symbol_into_function 
        && sc.symbol    != NULL
        && sc.comp_unit == NULL
        && sc.function  == NULL
        && sc.block     == NULL
        && sc.line_entry.IsValid() == false)
    {
        const AddressRange *symbol_range = sc.symbol->GetAddressRangePtr();
        if (symbol_range)
        {
            for (pos = m_symbol_contexts.begin(); pos != end; ++pos)
            {
                if (pos->function)
                {
                    if (pos->function->GetAddressRange().GetBaseAddress() == symbol_range->GetBaseAddress())
                    {
                        // Do we already have a function with this symbol?
                        if (pos->symbol == sc.symbol)
                            return false;
                        if (pos->symbol == NULL)
                        {
                            pos->symbol = sc.symbol;
                            return false;
                        }
                    }
                }
            }
        }
    }
    m_symbol_contexts.push_back(sc);
    return true;
}

void
SymbolContextList::Clear()
{
    m_symbol_contexts.clear();
}

void
SymbolContextList::Dump(Stream *s, Target *target) const
{

    *s << (void *)this << ": ";
    s->Indent();
    s->PutCString("SymbolContextList");
    s->EOL();
    s->IndentMore();

    collection::const_iterator pos, end = m_symbol_contexts.end();
    for (pos = m_symbol_contexts.begin(); pos != end; ++pos)
    {
        pos->Dump(s, target);
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

uint32_t
SymbolContextList::NumLineEntriesWithLine (uint32_t line) const
{
    uint32_t match_count = 0;
    const uint32_t size = m_symbol_contexts.size();
    for (uint32_t idx = 0; idx<size; ++idx)
    {
        if (m_symbol_contexts[idx].line_entry.line == line)
            ++match_count;
    }
    return match_count;
}

