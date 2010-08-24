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
FunctionInfo::Dump(Stream *s) const
{
    if (m_name)
        *s << ", name = \"" << m_name << "\"";
    m_declaration.Dump(s);
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
InlineFunctionInfo::Dump(Stream *s) const
{
    FunctionInfo::Dump(s);
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
        assert(sc.module_sp);
        sc.module_sp->GetSymbolVendor()->ParseFunctionBlocks(sc);
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
Function::GetDescription(Stream *s, lldb::DescriptionLevel level, Process *process)
{
    Type* func_type = GetType();
    *s << '"' << func_type->GetName() << "\", id = " << (const UserID&)*this;
    *s << ", range = ";
    GetAddressRange().Dump(s, process, Address::DumpStyleLoadAddress, Address::DumpStyleModuleWithFileAddress);
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

Type
Function::GetReturnType ()
{
    clang::QualType clang_type (clang::QualType::getFromOpaquePtr(GetType()->GetOpaqueClangQualType()));
    assert (clang_type->isFunctionType());
    clang::FunctionType *function_type = dyn_cast<clang::FunctionType> (clang_type);
    clang::QualType fun_return_qualtype = function_type->getResultType();

    const ConstString fun_return_name(ClangASTType::GetClangTypeName(fun_return_qualtype.getAsOpaquePtr()));

    SymbolContext sc;
    CalculateSymbolContext (&sc);
    // Null out everything below the CompUnit 'cause we don't actually know these.

    size_t bit_size = ClangASTType::GetClangTypeBitWidth ((GetType()->GetClangASTContext().getASTContext()), fun_return_qualtype.getAsOpaquePtr());
    Type return_type (0, GetType()->GetSymbolFile(), fun_return_name, bit_size, sc.comp_unit, 0, Type::eTypeUIDSynthetic, Declaration(), fun_return_qualtype.getAsOpaquePtr());
    return return_type;
}

int
Function::GetArgumentCount ()
{
    clang::QualType clang_type (clang::QualType::getFromOpaquePtr(GetType()->GetOpaqueClangQualType()));
    assert (clang_type->isFunctionType());
    if (!clang_type->isFunctionProtoType())
        return -1;

    const clang::FunctionProtoType *function_proto_type = dyn_cast<clang::FunctionProtoType>(clang_type);
    if (function_proto_type != NULL)
        return function_proto_type->getNumArgs();

    return 0;
}

const Type
Function::GetArgumentTypeAtIndex (size_t idx)
{
    clang::QualType clang_type (clang::QualType::getFromOpaquePtr(GetType()->GetOpaqueClangQualType()));
   assert (clang_type->isFunctionType());
   if (!clang_type->isFunctionProtoType())
        return Type();

    const clang::FunctionProtoType *function_proto_type = dyn_cast<clang::FunctionProtoType>(clang_type);
    if (function_proto_type != NULL)
    {
        unsigned num_args = function_proto_type->getNumArgs();
        if (idx >= num_args)
            return Type();
        clang::QualType arg_qualtype = (function_proto_type->arg_type_begin())[idx];

        const ConstString arg_return_name(ClangASTType::GetClangTypeName(arg_qualtype.getAsOpaquePtr()));
        SymbolContext sc;
        CalculateSymbolContext (&sc);
        // Null out everything below the CompUnit 'cause we don't actually know these.

        size_t bit_size = ClangASTType::GetClangTypeBitWidth ((GetType()->GetClangASTContext().getASTContext()), arg_qualtype.getAsOpaquePtr());
        Type arg_type (0, GetType()->GetSymbolFile(), arg_return_name, bit_size, sc.comp_unit, 0, Type::eTypeUIDSynthetic, Declaration(), arg_qualtype.getAsOpaquePtr());
        return arg_type;
    }

    return Type();
}

const char *
Function::GetArgumentNameAtIndex (size_t idx)
{
   clang::Type *clang_type = static_cast<clang::QualType *>(GetType()->GetOpaqueClangQualType())->getTypePtr();
   assert (clang_type->isFunctionType());
   if (!clang_type->isFunctionProtoType())
       return NULL;
    return NULL;
}

bool
Function::IsVariadic ()
{
   const clang::Type *clang_type = static_cast<clang::QualType *>(GetType()->GetOpaqueClangQualType())->getTypePtr();
   assert (clang_type->isFunctionType());
   if (!clang_type->isFunctionProtoType())
        return false;

    const clang::FunctionProtoType *function_proto_type = dyn_cast<clang::FunctionProtoType>(clang_type);
    if (function_proto_type != NULL)
    {
        return function_proto_type->isVariadic();
    }

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
                m_prologue_byte_size = line_entry.range.GetByteSize();
        }
    }
    return m_prologue_byte_size;
}



