//===-- ASTStructExtractor.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "stdlib.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Stmt.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "lldb/Core/Log.h"
#include "lldb/Expression/ASTStructExtractor.h"

using namespace llvm;
using namespace clang;
using namespace lldb_private;

ASTStructExtractor::ASTStructExtractor(ASTConsumer *passthrough,
                                       const char *struct_name,
                                       ClangFunction &function) :
    m_ast_context (NULL),
    m_passthrough (passthrough),
    m_passthrough_sema (NULL),
    m_sema (NULL),
    m_action (NULL),
    m_function (function),
    m_struct_name (struct_name)
{
    if (!m_passthrough)
        return;
    
    m_passthrough_sema = dyn_cast<SemaConsumer>(passthrough);
}

ASTStructExtractor::~ASTStructExtractor()
{
}

void
ASTStructExtractor::Initialize(ASTContext &Context) 
{
    m_ast_context = &Context;
    
    if (m_passthrough)
        m_passthrough->Initialize(Context);
}

void
ASTStructExtractor::ExtractFromFunctionDecl(FunctionDecl *F)
{
    DeclarationName struct_name(&m_ast_context->Idents.get(m_struct_name.c_str()));
    RecordDecl::lookup_result struct_lookup = F->lookup(struct_name);
    
    if (struct_lookup.first == struct_lookup.second)
        return;
    
    RecordDecl *struct_decl = dyn_cast<RecordDecl>(*(struct_lookup.first));
    
    if (!struct_decl)
        return;
    
    const ASTRecordLayout* struct_layout(&m_ast_context->getASTRecordLayout (struct_decl));
    
    if (!struct_layout)
        return;
    
    m_function.m_struct_size = struct_layout->getSize().getQuantity(); // TODO Store m_struct_size as CharUnits   
    m_function.m_return_offset = struct_layout->getFieldOffset(struct_layout->getFieldCount() - 1) / 8;
    m_function.m_return_size = struct_layout->getDataSize().getQuantity() - m_function.m_return_offset;
    
    for (unsigned field_index = 0, num_fields = struct_layout->getFieldCount();
         field_index < num_fields;
         ++field_index)
    {
        m_function.m_member_offsets.push_back(struct_layout->getFieldOffset(field_index) / 8);
    }
    
    m_function.m_struct_valid = true;
}

void
ASTStructExtractor::ExtractFromTopLevelDecl(Decl* D)
{
    LinkageSpecDecl *linkage_spec_decl = dyn_cast<LinkageSpecDecl>(D);
    
    if (linkage_spec_decl)
    {
        RecordDecl::decl_iterator decl_iterator;
        
        for (decl_iterator = linkage_spec_decl->decls_begin();
             decl_iterator != linkage_spec_decl->decls_end();
             ++decl_iterator)
        {
            ExtractFromTopLevelDecl(*decl_iterator);
        }
    }
    
    FunctionDecl *function_decl = dyn_cast<FunctionDecl>(D);
    
    if (m_ast_context &&
        function_decl &&
        !m_function.m_wrapper_function_name.compare(function_decl->getNameAsString().c_str()))
    {
        ExtractFromFunctionDecl(function_decl);
    }
}

void 
ASTStructExtractor::HandleTopLevelDecl(DeclGroupRef D)
{
    DeclGroupRef::iterator decl_iterator;
    
    for (decl_iterator = D.begin();
         decl_iterator != D.end();
         ++decl_iterator)
    {
        Decl *decl = *decl_iterator;
        
        ExtractFromTopLevelDecl(decl);
    }
    
    if (m_passthrough)
        m_passthrough->HandleTopLevelDecl(D);
}

void
ASTStructExtractor::HandleTranslationUnit(ASTContext &Ctx)
{    
    if (m_passthrough)
        m_passthrough->HandleTranslationUnit(Ctx);
}

void 
ASTStructExtractor::HandleTagDeclDefinition(TagDecl *D)
{
    if (m_passthrough)
        m_passthrough->HandleTagDeclDefinition(D);
}

void
ASTStructExtractor::CompleteTentativeDefinition(VarDecl *D)
{
    if (m_passthrough)
        m_passthrough->CompleteTentativeDefinition(D);
}

void 
ASTStructExtractor::HandleVTable(CXXRecordDecl *RD, bool DefinitionRequired) 
{
    if (m_passthrough)
        m_passthrough->HandleVTable(RD, DefinitionRequired);
}

void
ASTStructExtractor::PrintStats() 
{
    if (m_passthrough)
        m_passthrough->PrintStats();
}

void
ASTStructExtractor::InitializeSema(Sema &S)
{
    m_sema = &S;
    m_action = reinterpret_cast<Action*>(m_sema);
    
    if (m_passthrough_sema)
        m_passthrough_sema->InitializeSema(S);
}

void 
ASTStructExtractor::ForgetSema() 
{
    m_sema = NULL;
    m_action = NULL;
    
    if (m_passthrough_sema)
        m_passthrough_sema->ForgetSema();
}
