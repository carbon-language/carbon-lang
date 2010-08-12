//===-- ClangASTSource.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "clang/AST/ASTContext.h"
#include "lldb/Expression/ClangASTSource.h"
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ClangExpressionDeclMap.h"

using namespace clang;
using namespace lldb_private;

ClangASTSource::~ClangASTSource() {}

void ClangASTSource::StartTranslationUnit(ASTConsumer *Consumer) {
    // Tell Sema to ask us when looking into the translation unit's decl.
    Context.getTranslationUnitDecl()->setHasExternalVisibleStorage();
    Context.getTranslationUnitDecl()->setHasExternalLexicalStorage();
}

// These are only required for AST source that want to lazily load
// the declarations (or parts thereof) that they return.
Decl *ClangASTSource::GetExternalDecl(uint32_t) { return 0; }
Stmt *ClangASTSource::GetExternalDeclStmt(uint64_t) { return 0; }

// These are also optional, although it might help with ObjC
// debugging if we have respectable signatures.  But a more
// efficient interface (that didn't require scanning all files
// for method signatures!) might help.
Selector ClangASTSource::GetExternalSelector(uint32_t) { return Selector(); }
uint32_t ClangASTSource::GetNumExternalSelectors() { return 0; }

// The core lookup interface.
DeclContext::lookup_result ClangASTSource::FindExternalVisibleDeclsByName(const DeclContext *DC, DeclarationName Name) {
    switch (Name.getNameKind()) {
    // Normal identifiers.
    case DeclarationName::Identifier:
      break;
            
    // Operator names.  Not important for now.
    case DeclarationName::CXXOperatorName:
    case DeclarationName::CXXLiteralOperatorName:
      return DeclContext::lookup_result();
            
    // Using directives found in this context.
    // Tell Sema we didn't find any or we'll end up getting asked a *lot*.
    case DeclarationName::CXXUsingDirective:
      return SetNoExternalVisibleDeclsForName(DC, Name);
            
    // These aren't looked up like this.
    case DeclarationName::ObjCZeroArgSelector:
    case DeclarationName::ObjCOneArgSelector:
    case DeclarationName::ObjCMultiArgSelector:
      return DeclContext::lookup_result();

    // These aren't possible in the global context.
    case DeclarationName::CXXConstructorName:
    case DeclarationName::CXXDestructorName:
    case DeclarationName::CXXConversionFunctionName:
      return DeclContext::lookup_result();
    }
    
	llvm::SmallVector<NamedDecl*, 4> Decls;
    
    NameSearchContext NSC(*this, Decls, Name, DC);
    
    DeclMap.GetDecls(NSC, Name.getAsString().c_str());
    return SetExternalVisibleDeclsForName(DC, Name, Decls);
}

// This is used to support iterating through an entire lexical context,
// which isn't something the debugger should ever need to do.
bool ClangASTSource::FindExternalLexicalDecls(const DeclContext *DC, llvm::SmallVectorImpl<Decl*> &Decls) {
	// true is for error, that's good enough for me
	return true;
}

clang::ASTContext *NameSearchContext::GetASTContext() {
    return &ASTSource.Context;
}

clang::NamedDecl *NameSearchContext::AddVarDecl(void *type) {
    clang::NamedDecl *Decl = VarDecl::Create(ASTSource.Context, 
                                             const_cast<DeclContext*>(DC), 
                                             SourceLocation(), 
                                             Name.getAsIdentifierInfo(), 
                                             QualType::getFromOpaquePtr(type), 
                                             0, 
                                             VarDecl::Static, 
                                             VarDecl::Static);
    Decls.push_back(Decl);
    
    return Decl;
}

clang::NamedDecl *NameSearchContext::AddFunDecl(void *type) {
    clang::FunctionDecl *Decl = FunctionDecl::Create(ASTSource.Context,
                                                     const_cast<DeclContext*>(DC),
                                                     SourceLocation(),
                                                     Name.getAsIdentifierInfo(),
                                                     QualType::getFromOpaquePtr(type),
                                                     NULL,
                                                     FunctionDecl::Static,
                                                     FunctionDecl::Static,
                                                     false,
                                                     true);
    
    // We have to do more than just synthesize the FunctionDecl.  We have to
    // synthesize ParmVarDecls for all of the FunctionDecl's arguments.  To do
    // this, we raid the function's FunctionProtoType for types.
    
    QualType QT = QualType::getFromOpaquePtr(type);
    clang::Type *T = QT.getTypePtr();
    const FunctionProtoType *FPT = T->getAs<FunctionProtoType>();
    
    if (FPT)
    {        
        unsigned NumArgs = FPT->getNumArgs();
        unsigned ArgIndex;
        
        ParmVarDecl **ParmVarDecls = new ParmVarDecl*[NumArgs];
        
        for (ArgIndex = 0; ArgIndex < NumArgs; ++ArgIndex)
        {
            QualType ArgQT = FPT->getArgType(ArgIndex);
            
            ParmVarDecls[ArgIndex] = ParmVarDecl::Create(ASTSource.Context,
                                                         const_cast<DeclContext*>(DC),
                                                         SourceLocation(),
                                                         NULL,
                                                         ArgQT,
                                                         NULL,
                                                         ParmVarDecl::Static,
                                                         ParmVarDecl::Static,
                                                         NULL);
        }
        
        Decl->setParams(ParmVarDecls, NumArgs);
        
        delete [] ParmVarDecls;
    }
    
    Decls.push_back(Decl);
    
    return Decl;
}

clang::NamedDecl *NameSearchContext::AddGenericFunDecl()
{
    QualType generic_function_type(ASTSource.Context.getFunctionType(ASTSource.Context.getSizeType(),   // result
                                                                     NULL,                              // argument types
                                                                     0,                                 // number of arguments
                                                                     true,                              // variadic?
                                                                     0,                                 // type qualifiers
                                                                     false,                             // has exception specification?
                                                                     false,                             // has any exception specification?
                                                                     0,                                 // number of exceptions
                                                                     NULL,                              // exceptions
                                                                     FunctionType::ExtInfo()));         // defaults for noreturn, regparm, calling convention

    return AddFunDecl(generic_function_type.getAsOpaquePtr());
}

clang::NamedDecl *NameSearchContext::AddTypeDecl(void *type)
{
    QualType QT = QualType::getFromOpaquePtr(type);
    clang::Type *T = QT.getTypePtr();

    if (TagType *tag_type = dyn_cast<clang::TagType>(T))
    {
        TagDecl *tag_decl = tag_type->getDecl();
        
        Decls.push_back(tag_decl);
        
        return tag_decl;
    }
    else
    {
        return NULL;
    }
}
