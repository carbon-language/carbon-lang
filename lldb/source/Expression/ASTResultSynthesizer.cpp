//===-- ASTResultSynthesizer.cpp --------------------------------*- C++ -*-===//
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
#include "clang/AST/Stmt.h"
#include "clang/Parse/Action.h"
#include "clang/Parse/Parser.h"
#include "clang/Parse/Scope.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "lldb/Core/Log.h"
#include "lldb/Expression/ASTResultSynthesizer.h"

using namespace llvm;
using namespace clang;
using namespace lldb_private;

ASTResultSynthesizer::ASTResultSynthesizer(ASTConsumer *passthrough) :
    m_ast_context (NULL),
    m_passthrough (passthrough),
    m_passthrough_sema (NULL),
    m_sema (NULL),
    m_action (NULL)
{
    if (!m_passthrough)
        return;
    
    m_passthrough_sema = dyn_cast<SemaConsumer>(passthrough);
}

ASTResultSynthesizer::~ASTResultSynthesizer()
{
}

void
ASTResultSynthesizer::Initialize(ASTContext &Context) 
{
    m_ast_context = &Context;
    
    if (m_passthrough)
        m_passthrough->Initialize(Context);
}

void
ASTResultSynthesizer::TransformTopLevelDecl(Decl* D)
{
    LinkageSpecDecl *linkage_spec_decl = dyn_cast<LinkageSpecDecl>(D);
    
    if (linkage_spec_decl)
    {
        RecordDecl::decl_iterator decl_iterator;
        
        for (decl_iterator = linkage_spec_decl->decls_begin();
             decl_iterator != linkage_spec_decl->decls_end();
             ++decl_iterator)
        {
            TransformTopLevelDecl(*decl_iterator);
        }
    }
    
    FunctionDecl *function_decl = dyn_cast<FunctionDecl>(D);
    
    if (m_ast_context &&
        function_decl &&
        !strcmp(function_decl->getNameAsCString(),
                "___clang_expr"))
    {
        SynthesizeResult(function_decl);
    }
}

void 
ASTResultSynthesizer::HandleTopLevelDecl(DeclGroupRef D)
{
    DeclGroupRef::iterator decl_iterator;
    
    for (decl_iterator = D.begin();
         decl_iterator != D.end();
         ++decl_iterator)
    {
        Decl *decl = *decl_iterator;
        
        TransformTopLevelDecl(decl);
    }
    
    if (m_passthrough)
        m_passthrough->HandleTopLevelDecl(D);
}

bool 
ASTResultSynthesizer::SynthesizeResult (FunctionDecl *FunDecl)
{
    ASTContext &Ctx(*m_ast_context);
    
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
    
    if (!m_sema)
        return false;

    FunctionDecl *function_decl = FunDecl;
    
    if (!function_decl)
        return false;
    
    Stmt *function_body = function_decl->getBody();
    CompoundStmt *compound_stmt = dyn_cast<CompoundStmt>(function_body);
    
    if (!compound_stmt)
        return false;
    
    if (compound_stmt->body_empty())
        return false;
    
    Stmt **last_stmt_ptr = compound_stmt->body_end() - 1;
    Stmt *last_stmt = *last_stmt_ptr;
    
    Expr *last_expr = dyn_cast<Expr>(last_stmt);
    
    if (!last_expr)
        // No auxiliary variable necessary; expression returns void
        return true;
    
    QualType expr_qual_type = last_expr->getType();
    clang::Type *expr_type = expr_qual_type.getTypePtr();
    
    if (!expr_type)
        return false;
    
    if (expr_type->isVoidType())
        return true;
    
    if (log)
    {
        std::string s = expr_qual_type.getAsString();
        
        log->Printf("Last statement's type: %s", s.c_str());
    }
    
    IdentifierInfo &result_id = Ctx.Idents.get("___clang_expr_result");
        
    clang::VarDecl *result_decl = VarDecl::Create(Ctx, 
                                                  function_decl, 
                                                  SourceLocation(), 
                                                  &result_id, 
                                                  expr_qual_type, 
                                                  NULL, 
                                                  VarDecl::Static, 
                                                  VarDecl::Static);
    
    if (!result_decl)
        return false;
    
    function_decl->addDecl(result_decl);
    
    ///////////////////////////////
    // call AddInitializerToDecl
    //
    
    Parser::DeclPtrTy result_decl_ptr;
    result_decl_ptr.set(result_decl);
    
    m_action->AddInitializerToDecl(result_decl_ptr, Parser::ExprArg(*m_action, last_expr));
    
    /////////////////////////////////
    // call ConvertDeclToDeclGroup
    //
    
    Parser::DeclGroupPtrTy result_decl_group_ptr;
    
    result_decl_group_ptr = m_action->ConvertDeclToDeclGroup(result_decl_ptr);
    
    ////////////////////////
    // call ActOnDeclStmt
    //
    
    Parser::OwningStmtResult result_initialization_stmt_result(m_action->ActOnDeclStmt(result_decl_group_ptr,
                                                                                       SourceLocation(),
                                                                                       SourceLocation()));
    
    ////////////////////////////////////////////////
    // replace the old statement with the new one
    //
    
    *last_stmt_ptr = reinterpret_cast<Stmt*>(result_initialization_stmt_result.take());

    if (log)
    {
        std::string s;
        raw_string_ostream os(s);
        
        function_decl->print(os);
        
        os.flush();
        
        log->Printf("Transformed function AST:\n%s", s.c_str());
    }
    
    return true;
}

void
ASTResultSynthesizer::HandleTranslationUnit(ASTContext &Ctx)
{    
    if (m_passthrough)
        m_passthrough->HandleTranslationUnit(Ctx);
}

void 
ASTResultSynthesizer::HandleTagDeclDefinition(TagDecl *D)
{
    if (m_passthrough)
        m_passthrough->HandleTagDeclDefinition(D);
}

void
ASTResultSynthesizer::CompleteTentativeDefinition(VarDecl *D)
{
    if (m_passthrough)
        m_passthrough->CompleteTentativeDefinition(D);
}

void 
ASTResultSynthesizer::HandleVTable(CXXRecordDecl *RD, bool DefinitionRequired) 
{
    if (m_passthrough)
        m_passthrough->HandleVTable(RD, DefinitionRequired);
}

void
ASTResultSynthesizer::PrintStats() 
{
    if (m_passthrough)
        m_passthrough->PrintStats();
}

void
ASTResultSynthesizer::InitializeSema(Sema &S)
{
    m_sema = &S;
    m_action = reinterpret_cast<Action*>(m_sema);
    
    if (m_passthrough_sema)
        m_passthrough_sema->InitializeSema(S);
}

void 
ASTResultSynthesizer::ForgetSema() 
{
    m_sema = NULL;
    m_action = NULL;
    
    if (m_passthrough_sema)
        m_passthrough_sema->ForgetSema();
}
