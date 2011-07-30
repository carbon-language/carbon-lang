//===-- Function.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/Function.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "clang/AST/Type.h"
#include "clang/AST/CanonicalType.h"
#include "llvm/Support/Casting.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Basic function information is contained in the FunctionInfo class.
// It is designed to contain the name, linkage name, and declaration
// location.
//----------------------------------------------------------------------
FunctionInfo::FunctionInfo (const char *name, const Declaration *decl_ptr) :
    m_name(name),
    m_declaration(decl_ptr)
{
}


FunctionInfo::FunctionInfo (const ConstString& name, const Declaration *decl_ptr) :
    m_name(name),
    m_declaration(decl_ptr)
{
}


FunctionInfo::~FunctionInfo()
{
}

void
FunctionInfo::Dump(Stream *s, bool show_fullpaths) const
{
    if (m_name)
        *s << ", name = \"" << m_name << "\"";
    m_declaration.Dump(s, show_fullpaths);
}


int
FunctionInfo::Compare(const FunctionInfo& a, const FunctionInfo& b)
{
    int result = ConstString::Compare(a.GetName(), b.GetName());
    if (result)
        return result;

    return Declaration::Compare(a.m_declaration, b.m_declaration);
}


Declaration&
FunctionInfo::GetDeclaration()
{
    return m_declaration;
}

const Declaration&
FunctionInfo::GetDeclaration() const
{
    return m_declaration;
}

const ConstString&
FunctionInfo::GetName() const
{
    return m_name;
}

size_t
FunctionInfo::MemorySize() const
{
    return m_name.MemorySize() + m_declaration.MemorySize();
}


InlineFunctionInfo::InlineFunctionInfo
(
    const char *name,
    const char *mangled,
    const Declaration *decl_ptr,
    const Declaration *call_decl_ptr
) :
    FunctionInfo(name, decl_ptr),
    m_mangled(mangled, true),
    m_call_decl (call_decl_ptr)
{
}

InlineFunctionInfo::InlineFunctionInfo
(
    const ConstString& name,
    const Mangled &mangled,
    const Declaration *decl_ptr,
    const Declaration *call_decl_ptr
) :
    FunctionInfo(name, decl_ptr),
    m_mangled(mangled),
    m_call_decl (call_decl_ptr)
{
}

InlineFunctionInfo::~InlineFunctionInfo()
{
}

int
InlineFunctionInfo::Compare(const InlineFunctionInfo& a, const InlineFunctionInfo& b)
{

    int result = FunctionInfo::Compare(a, b);
    if (result)
        return result;
    // only compare the mangled names if both have them
    return Mangled::Compare(a.m_mangled, a.m_mangled);
}

void
InlineFunctionInfo::Dump(Stream *s, bool show_fullpaths) const
{
    FunctionInfo::Dump(s, show_fullpaths);
    if (m_mangled)
        m_mangled.Dump(s);
}

void
InlineFunctionInfo::DumpStopContext (Stream *s) const
{
//    s->Indent("[inlined] ");
    s->Indent();
    if (m_mangled)
        s->PutCString (m_mangled.GetName().AsCString());
    else
        s->PutCString (m_name.AsCString());
}


const ConstString &
InlineFunctionInfo::GetName () const
{
    if (m_mangled)
        return m_mangled.GetName();
    return m_name;
}


Declaration &
InlineFunctionInfo::GetCallSite ()
{
    return m_call_decl;
}

const Declaration &
InlineFunctionInfo::GetCallSite () const
{
    return m_call_decl;
}


Mangled&
InlineFunctionInfo::GetMangled()
{
    return m_mangled;
}

const Mangled&
InlineFunctionInfo::GetMangled() const
{
    return m_mangled;
}

size_t
InlineFunctionInfo::MemorySize() const
{
    return FunctionInfo::MemorySize() + m_mangled.MemorySize();
}

//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
Function::Function
(
    CompileUnit *comp_unit,
    lldb::user_id_t func_uid,
    lldb::user_id_t type_uid,
    const Mangled &mangled,
    Type * type,
    const AddressRange& range
) :
    UserID (func_uid),
    m_comp_unit (comp_unit),
    m_type_uid (type_uid),
    m_type (type),
    m_mangled (mangled),
    m_block (func_uid),
    m_range (range),
    m_frame_base (),
    m_flags (),
    m_prologue_byte_size (0)
{
    m_block.SetParentScope(this);
    assert(comp_unit != NULL);
}

Function::Function
(
    CompileUnit *comp_unit,
    lldb::user_id_t func_uid,
    lldb::user_id_t type_uid,
    const char *mangled,
    Type *type,
    const AddressRange &range
) :
    UserID (func_uid),
    m_comp_unit (comp_unit),
    m_type_uid (type_uid),
    m_type (type),
    m_mangled (mangled, true),
    m_block (func_uid),
    m_range (range),
    m_frame_base (),
    m_flags (),
    m_prologue_byte_size (0)
{
    m_block.SetParentScope(this);
    assert(comp_unit != NULL);
}


Function::~Function()
{
}

void
Function::GetStartLineSourceInfo (FileSpec &source_file, uint32_t &line_no)
{
    line_no = 0;
    source_file.Clear();
    
    if (m_comp_unit == NULL)
        return;
        
    if (m_type != NULL && m_type->GetDeclaration().GetLine() != 0)
    {
        source_file = m_type->GetDeclaration().GetFile();
        line_no = m_type->GetDeclaration().GetLine();
    }
    else 
    {
        LineTable *line_table = m_comp_unit->GetLineTable();
        if (line_table == NULL)
            return;
            
        LineEntry line_entry;
        if (line_table->FindLineEntryByAddress (GetAddressRange().GetBaseAddress(), line_entry, NULL))
        {
            line_no = line_entry.line;
            source_file = line_entry.file;
        }
    }
}

void
Function::GetEndLineSourceInfo (FileSpec &source_file, uint32_t &line_no)
{
    line_no = 0;
    source_file.Clear();
    
    // The -1 is kind of cheesy, but I want to get the last line entry for the given function, not the 
    // first entry of the next.
    Address scratch_addr(GetAddressRange().GetBaseAddress());
    scratch_addr.SetOffset (scratch_addr.GetOffset() + GetAddressRange().GetByteSize() - 1);
    
    LineTable *line_table = m_comp_unit->GetLineTable();
    if (line_table == NULL)
        return;
        
    LineEntry line_entry;
    if (line_table->FindLineEntryByAddress (scratch_addr, line_entry, NULL))
    {
        line_no = line_entry.line;
        source_file = line_entry.file;
    }
}

Block &
Function::GetBlock (bool can_create)
{
    if (!m_block.BlockInfoHasBeenParsed() && can_create)
    {
        SymbolContext sc;
        CalculateSymbolContext(&sc);
        if (sc.module_sp)
        {
            sc.module_sp->GetSymbolVendor()->ParseFunctionBlocks(sc);
        }
        else
        {
            ::fprintf (stderr, 
                       "unable to find module shared pointer for function '%s' in %s%s%s\n", 
                       GetName().GetCString(),
                       m_comp_unit->GetDirectory().GetCString(),
                       m_comp_unit->GetDirectory() ? "/" : "",
                       m_comp_unit->GetFilename().GetCString());
        }
        m_block.SetBlockInfoHasBeenParsed (true, true);
    }
    return m_block;
}

CompileUnit*
Function::GetCompileUnit()
{
    return m_comp_unit;
}

const CompileUnit*
Function::GetCompileUnit() const
{
    return m_comp_unit;
}


void
Function::GetDescription(Stream *s, lldb::DescriptionLevel level, Target *target)
{
    Type* func_type = GetType();
    *s << "id = " << (const UserID&)*this << ", name = \"" << func_type->GetName() << "\", range = ";
    
    Address::DumpStyle fallback_style;
    if (level == eDescriptionLevelVerbose)
        fallback_style = Address::DumpStyleModuleWithFileAddress;
    else
        fallback_style = Address::DumpStyleFileAddress;
    GetAddressRange().Dump(s, target, Address::DumpStyleLoadAddress, fallback_style);
}

void
Function::Dump(Stream *s, bool show_context) const
{
    s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    *s << "Function" << (const UserID&)*this;

    m_mangled.Dump(s);

//  FunctionInfo::Dump(s);
    if (m_type)
    {
        *s << ", type = " << (void*)m_type;
        /// << " (";
        ///m_type->DumpTypeName(s);
        ///s->PutChar(')');
    }
    else if (m_type_uid != LLDB_INVALID_UID)
        *s << ", type_uid = " << m_type_uid;

    s->EOL();
    // Dump the root object
    if (m_block.BlockInfoHasBeenParsed ())
        m_block.Dump(s, m_range.GetBaseAddress().GetFileAddress(), INT_MAX, show_context);
}


void
Function::CalculateSymbolContext(SymbolContext* sc)
{
    sc->function = this;
    m_comp_unit->CalculateSymbolContext(sc);
}

void
Function::DumpSymbolContext(Stream *s)
{
    m_comp_unit->DumpSymbolContext(s);
    s->Printf(", Function{0x%8.8x}", GetID());
}

size_t
Function::MemorySize () const
{
    size_t mem_size = sizeof(Function) + m_block.MemorySize();
    return mem_size;
}

Type*
Function::GetType()
{
    return m_type;
}

const Type*
Function::GetType() const
{
    return m_type;
}

clang_type_t
Function::GetReturnClangType ()
{
    clang::QualType clang_type (clang::QualType::getFromOpaquePtr(GetType()->GetClangFullType()));
    const clang::FunctionType *function_type = llvm::dyn_cast<clang::FunctionType> (clang_type);
    if (function_type)
        return function_type->getResultType().getAsOpaquePtr();
    return NULL;
}

int
Function::GetArgumentCount ()
{
    clang::QualType clang_type (clang::QualType::getFromOpaquePtr(GetType()->GetClangFullType()));
    assert (clang_type->isFunctionType());
    if (!clang_type->isFunctionProtoType())
        return -1;

    const clang::FunctionProtoType *function_proto_type = llvm::dyn_cast<clang::FunctionProtoType>(clang_type);
    if (function_proto_type != NULL)
        return function_proto_type->getNumArgs();

    return 0;
}

clang_type_t
Function::GetArgumentTypeAtIndex (size_t idx)
{
    clang::QualType clang_type (clang::QualType::getFromOpaquePtr(GetType()->GetClangFullType()));
    const clang::FunctionProtoType *function_proto_type = llvm::dyn_cast<clang::FunctionProtoType>(clang_type);
    if (function_proto_type)
    {
        unsigned num_args = function_proto_type->getNumArgs();
        if (idx >= num_args)
            return NULL;
        
        return (function_proto_type->arg_type_begin())[idx].getAsOpaquePtr();
    }
    return NULL;
}

bool
Function::IsVariadic ()
{
   const clang::Type *clang_type = static_cast<clang::QualType *>(GetType()->GetClangFullType())->getTypePtr();
   assert (clang_type->isFunctionType());
   if (!clang_type->isFunctionProtoType())
        return false;

    const clang::FunctionProtoType *function_proto_type = llvm::dyn_cast<clang::FunctionProtoType>(clang_type);
    if (function_proto_type)
        return function_proto_type->isVariadic();

    return false;
}

uint32_t
Function::GetPrologueByteSize ()
{
    if (m_prologue_byte_size == 0 && m_flags.IsClear(flagsCalculatedPrologueSize))
    {
        m_flags.Set(flagsCalculatedPrologueSize);
        LineTable* line_table = m_comp_unit->GetLineTable ();
        if (line_table)
        {
            LineEntry line_entry;
            if (line_table->FindLineEntryByAddress(GetAddressRange().GetBaseAddress(), line_entry))
            {
                // We need to take the delta of the end of the first line entry
                // as a file address and the start file address of the function
                // in case the first line entry doesn't start at the beginning 
                // of the function.
                const addr_t func_start_file_addr = m_range.GetBaseAddress().GetFileAddress();
                const addr_t line_entry_end_file_addr = line_entry.range.GetBaseAddress().GetFileAddress() + line_entry.range.GetByteSize();
                if (line_entry_end_file_addr > func_start_file_addr)
                    m_prologue_byte_size = line_entry_end_file_addr - func_start_file_addr;
            }
        }
    }
    return m_prologue_byte_size;
}



