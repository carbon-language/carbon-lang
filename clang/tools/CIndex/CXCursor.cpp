//===- CXCursor.cpp - Routines for manipulating CXCursors -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines routines for manipulating CXCursors. It should be the
// only file that has internal knowledge of the encoding of the data in
// CXCursor.
//
//===----------------------------------------------------------------------===//

#include "CXCursor.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;

CXCursor cxcursor::MakeCXCursorInvalid(CXCursorKind K) {
  assert(K >= CXCursor_FirstInvalid && K <= CXCursor_LastInvalid);
  CXCursor C = { K, { 0, 0, 0 } };
  return C;
}

static CXCursorKind GetCursorKind(Decl *D) {
  switch (D->getKind()) {
    case Decl::Enum:               return CXCursor_EnumDecl;
    case Decl::EnumConstant:       return CXCursor_EnumConstantDecl;
    case Decl::Field:              return CXCursor_FieldDecl;
    case Decl::Function:  
      return CXCursor_FunctionDecl;
    case Decl::ObjCCategory:       return CXCursor_ObjCCategoryDecl;
    case Decl::ObjCCategoryImpl:   return CXCursor_ObjCCategoryImplDecl;
    case Decl::ObjCClass:
      // FIXME
      return CXCursor_UnexposedDecl;
    case Decl::ObjCForwardProtocol:
      // FIXME
      return CXCursor_UnexposedDecl;      
    case Decl::ObjCImplementation: return CXCursor_ObjCImplementationDecl;
    case Decl::ObjCInterface:      return CXCursor_ObjCInterfaceDecl;
    case Decl::ObjCIvar:           return CXCursor_ObjCIvarDecl; 
    case Decl::ObjCMethod:
      return cast<ObjCMethodDecl>(D)->isInstanceMethod()
              ? CXCursor_ObjCInstanceMethodDecl : CXCursor_ObjCClassMethodDecl;
    case Decl::ObjCProperty:       return CXCursor_ObjCPropertyDecl;
    case Decl::ObjCProtocol:       return CXCursor_ObjCProtocolDecl;
    case Decl::ParmVar:            return CXCursor_ParmDecl;
    case Decl::Typedef:            return CXCursor_TypedefDecl;
    case Decl::Var:                return CXCursor_VarDecl;
    default:
      if (TagDecl *TD = dyn_cast<TagDecl>(D)) {
        switch (TD->getTagKind()) {
          case TagDecl::TK_struct: return CXCursor_StructDecl;
          case TagDecl::TK_class:  return CXCursor_ClassDecl;
          case TagDecl::TK_union:  return CXCursor_UnionDecl;
          case TagDecl::TK_enum:   return CXCursor_EnumDecl;
        }
      }

      return CXCursor_UnexposedDecl;
  }
  
  llvm_unreachable("Invalid Decl");
  return CXCursor_NotImplemented;  
}

CXCursor cxcursor::MakeCXCursor(Decl *D) {
  CXCursor C = { GetCursorKind(D), { D, 0, 0 } };
  return C;
}

CXCursor cxcursor::MakeCXCursor(Stmt *S, Decl *Parent) {
  CXCursorKind K = CXCursor_NotImplemented;
  
  switch (S->getStmtClass()) {
  case Stmt::NoStmtClass:
    break;
      
  case Stmt::NullStmtClass:
  case Stmt::CompoundStmtClass:
  case Stmt::CaseStmtClass:
  case Stmt::DefaultStmtClass:
  case Stmt::LabelStmtClass:       
  case Stmt::IfStmtClass:          
  case Stmt::SwitchStmtClass:      
  case Stmt::WhileStmtClass:       
  case Stmt::DoStmtClass:          
  case Stmt::ForStmtClass:        
  case Stmt::GotoStmtClass:        
  case Stmt::IndirectGotoStmtClass:
  case Stmt::ContinueStmtClass:    
  case Stmt::BreakStmtClass:       
  case Stmt::ReturnStmtClass:      
  case Stmt::DeclStmtClass:        
  case Stmt::SwitchCaseClass:      
  case Stmt::AsmStmtClass:         
  case Stmt::ObjCAtTryStmtClass:        
  case Stmt::ObjCAtCatchStmtClass:      
  case Stmt::ObjCAtFinallyStmtClass:    
  case Stmt::ObjCAtThrowStmtClass:      
  case Stmt::ObjCAtSynchronizedStmtClass: 
  case Stmt::ObjCForCollectionStmtClass:
  case Stmt::CXXCatchStmtClass:
  case Stmt::CXXTryStmtClass:  
    K = CXCursor_UnexposedStmt;
    break;
      
  case Stmt::ExprClass:
  case Stmt::PredefinedExprClass:        
  case Stmt::IntegerLiteralClass:        
  case Stmt::FloatingLiteralClass:       
  case Stmt::ImaginaryLiteralClass:      
  case Stmt::StringLiteralClass:         
  case Stmt::CharacterLiteralClass:      
  case Stmt::ParenExprClass:             
  case Stmt::UnaryOperatorClass:         
  case Stmt::SizeOfAlignOfExprClass:     
  case Stmt::ArraySubscriptExprClass:    
  case Stmt::CastExprClass:              
  case Stmt::BinaryOperatorClass:        
  case Stmt::CompoundAssignOperatorClass:
  case Stmt::ConditionalOperatorClass:   
  case Stmt::ImplicitCastExprClass:
  case Stmt::ExplicitCastExprClass:
  case Stmt::CStyleCastExprClass:
  case Stmt::CompoundLiteralExprClass:   
  case Stmt::ExtVectorElementExprClass:  
  case Stmt::InitListExprClass:          
  case Stmt::DesignatedInitExprClass:    
  case Stmt::ImplicitValueInitExprClass: 
  case Stmt::ParenListExprClass:         
  case Stmt::VAArgExprClass:             
  case Stmt::AddrLabelExprClass:        
  case Stmt::StmtExprClass:             
  case Stmt::TypesCompatibleExprClass:  
  case Stmt::ChooseExprClass:           
  case Stmt::GNUNullExprClass:          
  case Stmt::CXXNamedCastExprClass:
  case Stmt::CXXStaticCastExprClass:      
  case Stmt::CXXDynamicCastExprClass:     
  case Stmt::CXXReinterpretCastExprClass: 
  case Stmt::CXXConstCastExprClass:       
  case Stmt::CXXFunctionalCastExprClass:
  case Stmt::CXXTypeidExprClass:          
  case Stmt::CXXBoolLiteralExprClass:     
  case Stmt::CXXNullPtrLiteralExprClass:  
  case Stmt::CXXThisExprClass:            
  case Stmt::CXXThrowExprClass:           
  case Stmt::CXXDefaultArgExprClass:      
  case Stmt::CXXZeroInitValueExprClass:   
  case Stmt::CXXNewExprClass:             
  case Stmt::CXXDeleteExprClass:          
  case Stmt::CXXPseudoDestructorExprClass:
  case Stmt::UnresolvedLookupExprClass:   
  case Stmt::UnaryTypeTraitExprClass:     
  case Stmt::DependentScopeDeclRefExprClass:  
  case Stmt::CXXBindTemporaryExprClass:   
  case Stmt::CXXExprWithTemporariesClass: 
  case Stmt::CXXUnresolvedConstructExprClass:
  case Stmt::CXXDependentScopeMemberExprClass:
  case Stmt::UnresolvedMemberExprClass:   
  case Stmt::ObjCStringLiteralClass:    
  case Stmt::ObjCEncodeExprClass:       
  case Stmt::ObjCSelectorExprClass:   
  case Stmt::ObjCProtocolExprClass:   
  case Stmt::ObjCImplicitSetterGetterRefExprClass: 
  case Stmt::ObjCSuperExprClass:     
  case Stmt::ObjCIsaExprClass:       
  case Stmt::ShuffleVectorExprClass: 
  case Stmt::BlockExprClass:  
    K = CXCursor_UnexposedExpr;
    break;
  case Stmt::DeclRefExprClass:           
  case Stmt::BlockDeclRefExprClass:
    // FIXME: UnresolvedLookupExpr?
    // FIXME: DependentScopeDeclRefExpr?
    K = CXCursor_DeclRefExpr;
    break;
      
  case Stmt::MemberExprClass:            
  case Stmt::ObjCIvarRefExprClass:    
  case Stmt::ObjCPropertyRefExprClass: 
    // FIXME: UnresolvedMemberExpr?
    // FIXME: CXXDependentScopeMemberExpr?
    K = CXCursor_MemberRefExpr;
    break;
      
  case Stmt::CallExprClass:              
  case Stmt::CXXOperatorCallExprClass:
  case Stmt::CXXMemberCallExprClass:
  case Stmt::CXXConstructExprClass:  
  case Stmt::CXXTemporaryObjectExprClass:
    // FIXME: CXXUnresolvedConstructExpr
    // FIXME: ObjCImplicitSetterGetterRefExpr?
    K = CXCursor_CallExpr;
    break;
      
  case Stmt::ObjCMessageExprClass:      
    K = CXCursor_ObjCMessageExpr;
    break;
  }
  
  CXCursor C = { K, { Parent, S, 0 } };
  return C;
}

CXCursor cxcursor::MakeCursorObjCSuperClassRef(ObjCInterfaceDecl *Super, 
                                         SourceLocation Loc) {
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_ObjCSuperClassRef, { Super, RawLoc, 0 } };
  return C;    
}

std::pair<ObjCInterfaceDecl *, SourceLocation> 
cxcursor::getCursorObjCSuperClassRef(CXCursor C) {
  assert(C.kind == CXCursor_ObjCSuperClassRef);
  return std::make_pair(static_cast<ObjCInterfaceDecl *>(C.data[0]),
           SourceLocation::getFromRawEncoding(
                                      reinterpret_cast<uintptr_t>(C.data[1])));
}

CXCursor cxcursor::MakeCursorObjCProtocolRef(ObjCProtocolDecl *Super, 
                                             SourceLocation Loc) {
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_ObjCProtocolRef, { Super, RawLoc, 0 } };
  return C;    
}

std::pair<ObjCProtocolDecl *, SourceLocation> 
cxcursor::getCursorObjCProtocolRef(CXCursor C) {
  assert(C.kind == CXCursor_ObjCProtocolRef);
  return std::make_pair(static_cast<ObjCProtocolDecl *>(C.data[0]),
           SourceLocation::getFromRawEncoding(
                                      reinterpret_cast<uintptr_t>(C.data[1])));
}

CXCursor cxcursor::MakeCursorObjCClassRef(ObjCInterfaceDecl *Class, 
                                         SourceLocation Loc) {
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_ObjCClassRef, { Class, RawLoc, 0 } };
  return C;    
}

std::pair<ObjCInterfaceDecl *, SourceLocation> 
cxcursor::getCursorObjCClassRef(CXCursor C) {
  assert(C.kind == CXCursor_ObjCClassRef);
  return std::make_pair(static_cast<ObjCInterfaceDecl *>(C.data[0]),
           SourceLocation::getFromRawEncoding(
                                      reinterpret_cast<uintptr_t>(C.data[1])));
}

Decl *cxcursor::getCursorDecl(CXCursor Cursor) {
  return (Decl *)Cursor.data[0];
}

Expr *cxcursor::getCursorExpr(CXCursor Cursor) {
  return dyn_cast_or_null<Expr>(getCursorStmt(Cursor));
}

Stmt *cxcursor::getCursorStmt(CXCursor Cursor) {
  if (Cursor.kind == CXCursor_ObjCSuperClassRef ||
      Cursor.kind == CXCursor_ObjCProtocolRef ||
      Cursor.kind == CXCursor_ObjCClassRef)
    return 0;

  return (Stmt *)Cursor.data[1];
}

ASTContext &cxcursor::getCursorContext(CXCursor Cursor) {
  switch (Cursor.kind) {
  case CXCursor_TypedefDecl:
  case CXCursor_StructDecl:
  case CXCursor_UnionDecl:
  case CXCursor_ClassDecl:
  case CXCursor_EnumDecl:
  case CXCursor_FieldDecl:
  case CXCursor_EnumConstantDecl:
  case CXCursor_FunctionDecl:
  case CXCursor_VarDecl:
  case CXCursor_ParmDecl:
  case CXCursor_ObjCInterfaceDecl:
  case CXCursor_ObjCCategoryDecl:
  case CXCursor_ObjCProtocolDecl:
  case CXCursor_ObjCPropertyDecl:
  case CXCursor_ObjCIvarDecl:
  case CXCursor_ObjCInstanceMethodDecl:
  case CXCursor_ObjCClassMethodDecl:
  case CXCursor_ObjCImplementationDecl:
  case CXCursor_ObjCCategoryImplDecl:
  case CXCursor_UnexposedDecl:
    return static_cast<Decl *>(Cursor.data[0])->getASTContext();

  case CXCursor_ObjCSuperClassRef:
  case CXCursor_ObjCProtocolRef:
  case CXCursor_ObjCClassRef:
    return static_cast<Decl *>(Cursor.data[0])->getASTContext();
    
  case CXCursor_InvalidFile:
  case CXCursor_NoDeclFound:
  case CXCursor_NotImplemented:
    llvm_unreachable("No context in an invalid cursor");
    break;

  case CXCursor_UnexposedExpr:
  case CXCursor_DeclRefExpr:
  case CXCursor_MemberRefExpr:
  case CXCursor_CallExpr:
  case CXCursor_ObjCMessageExpr:
  case CXCursor_UnexposedStmt:
    return static_cast<Decl *>(Cursor.data[0])->getASTContext();

  case CXCursor_TranslationUnit: {
    ASTUnit *CXXUnit = static_cast<ASTUnit *>(Cursor.data[0]);
    return CXXUnit->getASTContext();
  }
  }
  
  llvm_unreachable("No context available");
}

bool cxcursor::operator==(CXCursor X, CXCursor Y) {
  return X.kind == Y.kind && X.data[0] == Y.data[0] && X.data[1] == Y.data[1] &&
         X.data[2] == Y.data[2];
}
