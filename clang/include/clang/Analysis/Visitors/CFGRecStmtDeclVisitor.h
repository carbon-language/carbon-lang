//= CFGRecStmtDeclVisitor - Recursive visitor of CFG stmts/decls -*- C++ --*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the template class CFGRecStmtDeclVisitor, which extends
// CFGRecStmtVisitor by implementing (typed) visitation of decls.
//
// FIXME: This may not be fully complete.  We currently explore only subtypes
//        of ScopedDecl.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_CFG_REC_STMT_DECL_VISITOR_H
#define LLVM_CLANG_ANALYSIS_CFG_REC_STMT_DECL_VISITOR_H

#include "clang/Analysis/Visitors/CFGRecStmtVisitor.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"

#define DISPATCH_CASE(CASE,CLASS) \
case Decl::CASE: \
static_cast<ImplClass*>(this)->Visit##CLASS(static_cast<CLASS*>(D));\
break;

#define DEFAULT_DISPATCH(CLASS) void Visit##CLASS(CLASS* D) {}
#define DEFAULT_DISPATCH_VARDECL(CLASS) void Visit##CLASS(CLASS* D)\
  { static_cast<ImplClass*>(this)->VisitVarDecl(D); }

  
namespace clang {
template <typename ImplClass>
class CFGRecStmtDeclVisitor : public CFGRecStmtVisitor<ImplClass> {
public:  

  void VisitDeclRefExpr(DeclRefExpr* DR) {
    static_cast<ImplClass*>(this)->VisitDecl(DR->getDecl()); 
  }
  
  void VisitDeclStmt(DeclStmt* DS) {
    for (DeclStmt::decl_iterator DI = DS->decl_begin(), DE = DS->decl_end();
        DI != DE; ++DI) {
      Decl* D = *DI;
      static_cast<ImplClass*>(this)->VisitDecl(D); 
      // Visit the initializer.
      if (VarDecl* VD = dyn_cast<VarDecl>(D))
        if (Expr* I = VD->getInit())
          static_cast<ImplClass*>(this)->Visit(I);
    }
  }
    
  void VisitDecl(Decl* D) {
    switch (D->getKind()) {
        DISPATCH_CASE(Function,FunctionDecl)
        DISPATCH_CASE(Var,VarDecl)
        DISPATCH_CASE(ParmVar,ParmVarDecl)       // FIXME: (same)
        DISPATCH_CASE(OriginalParmVar,OriginalParmVarDecl) // FIXME: (same)
        DISPATCH_CASE(ImplicitParam,ImplicitParamDecl)
        DISPATCH_CASE(EnumConstant,EnumConstantDecl)
        DISPATCH_CASE(Typedef,TypedefDecl)
        DISPATCH_CASE(Record,RecordDecl)    // FIXME: Refine.  VisitStructDecl?
        DISPATCH_CASE(Enum,EnumDecl)
      default:
        assert(false && "Subtype of ScopedDecl not handled.");
    }
  }
  
  DEFAULT_DISPATCH(VarDecl)
  DEFAULT_DISPATCH(FunctionDecl)
  DEFAULT_DISPATCH_VARDECL(OriginalParmVarDecl)
  DEFAULT_DISPATCH_VARDECL(ParmVarDecl)
  DEFAULT_DISPATCH(ImplicitParamDecl)
  DEFAULT_DISPATCH(EnumConstantDecl)
  DEFAULT_DISPATCH(TypedefDecl)
  DEFAULT_DISPATCH(RecordDecl)
  DEFAULT_DISPATCH(EnumDecl)
  DEFAULT_DISPATCH(ObjCInterfaceDecl)
  DEFAULT_DISPATCH(ObjCClassDecl)
  DEFAULT_DISPATCH(ObjCMethodDecl)
  DEFAULT_DISPATCH(ObjCProtocolDecl)
  DEFAULT_DISPATCH(ObjCCategoryDecl)
};

} // end namespace clang

#undef DISPATCH_CASE
#undef DEFAULT_DISPATCH
#endif
