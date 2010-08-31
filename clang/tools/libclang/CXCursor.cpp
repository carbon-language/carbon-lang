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
  assert(D && "Invalid arguments!");
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
    case Decl::CXXMethod:          return CXCursor_CXXMethod;
    case Decl::CXXConstructor:     return CXCursor_Constructor;
    case Decl::CXXDestructor:      return CXCursor_Destructor;
    case Decl::CXXConversion:      return CXCursor_ConversionFunction;
    case Decl::ObjCProperty:       return CXCursor_ObjCPropertyDecl;
    case Decl::ObjCProtocol:       return CXCursor_ObjCProtocolDecl;
    case Decl::ParmVar:            return CXCursor_ParmDecl;
    case Decl::Typedef:            return CXCursor_TypedefDecl;
    case Decl::Var:                return CXCursor_VarDecl;
    case Decl::Namespace:          return CXCursor_Namespace;
    case Decl::TemplateTypeParm:   return CXCursor_TemplateTypeParameter;
    case Decl::NonTypeTemplateParm:return CXCursor_NonTypeTemplateParameter;
    case Decl::TemplateTemplateParm:return CXCursor_TemplateTemplateParameter;
    case Decl::FunctionTemplate:   return CXCursor_FunctionTemplate;
    default:
      if (TagDecl *TD = dyn_cast<TagDecl>(D)) {
        switch (TD->getTagKind()) {
          case TTK_Struct: return CXCursor_StructDecl;
          case TTK_Class:  return CXCursor_ClassDecl;
          case TTK_Union:  return CXCursor_UnionDecl;
          case TTK_Enum:   return CXCursor_EnumDecl;
        }
      }

      return CXCursor_UnexposedDecl;
  }
  
  llvm_unreachable("Invalid Decl");
  return CXCursor_NotImplemented;  
}

static CXCursorKind GetCursorKind(const Attr *A) {
  assert(A && "Invalid arguments!");
  switch (A->getKind()) {
    default: break;
    case attr::IBAction: return CXCursor_IBActionAttr;
    case attr::IBOutlet: return CXCursor_IBOutletAttr;
    case attr::IBOutletCollection: return CXCursor_IBOutletCollectionAttr;
  }

  return CXCursor_UnexposedAttr;
}

CXCursor cxcursor::MakeCXCursor(const Attr *A, Decl *Parent, ASTUnit *TU) {
  assert(A && Parent && TU && "Invalid arguments!");
  CXCursor C = { GetCursorKind(A), { Parent, (void*)A, TU } };
  return C;
}

CXCursor cxcursor::MakeCXCursor(Decl *D, ASTUnit *TU) {
  assert(D && TU && "Invalid arguments!");
  CXCursor C = { GetCursorKind(D), { D, 0, TU } };
  return C;
}

CXCursor cxcursor::MakeCXCursor(Stmt *S, Decl *Parent, ASTUnit *TU) {
  assert(S && TU && "Invalid arguments!");
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
      
  case Stmt::PredefinedExprClass:        
  case Stmt::IntegerLiteralClass:        
  case Stmt::FloatingLiteralClass:       
  case Stmt::ImaginaryLiteralClass:      
  case Stmt::StringLiteralClass:         
  case Stmt::CharacterLiteralClass:      
  case Stmt::ParenExprClass:             
  case Stmt::UnaryOperatorClass:
  case Stmt::OffsetOfExprClass:         
  case Stmt::SizeOfAlignOfExprClass:     
  case Stmt::ArraySubscriptExprClass:    
  case Stmt::BinaryOperatorClass:        
  case Stmt::CompoundAssignOperatorClass:
  case Stmt::ConditionalOperatorClass:   
  case Stmt::ImplicitCastExprClass:
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
  case Stmt::CXXScalarValueInitExprClass:   
  case Stmt::CXXNewExprClass:             
  case Stmt::CXXDeleteExprClass:          
  case Stmt::CXXPseudoDestructorExprClass:
  case Stmt::UnresolvedLookupExprClass:   
  case Stmt::UnaryTypeTraitExprClass:     
  case Stmt::DependentScopeDeclRefExprClass:  
  case Stmt::CXXBindTemporaryExprClass:   
  case Stmt::CXXBindReferenceExprClass:   
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
  
  CXCursor C = { K, { Parent, S, TU } };
  return C;
}

CXCursor cxcursor::MakeCursorObjCSuperClassRef(ObjCInterfaceDecl *Super, 
                                               SourceLocation Loc, 
                                               ASTUnit *TU) {
  assert(Super && TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_ObjCSuperClassRef, { Super, RawLoc, TU } };
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
                                             SourceLocation Loc, 
                                             ASTUnit *TU) {
  assert(Super && TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_ObjCProtocolRef, { Super, RawLoc, TU } };
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
                                          SourceLocation Loc, 
                                          ASTUnit *TU) {
  // 'Class' can be null for invalid code.
  if (!Class)
    return MakeCXCursorInvalid(CXCursor_InvalidCode);
  assert(TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_ObjCClassRef, { Class, RawLoc, TU } };
  return C;    
}

std::pair<ObjCInterfaceDecl *, SourceLocation> 
cxcursor::getCursorObjCClassRef(CXCursor C) {
  assert(C.kind == CXCursor_ObjCClassRef);
  return std::make_pair(static_cast<ObjCInterfaceDecl *>(C.data[0]),
           SourceLocation::getFromRawEncoding(
                                      reinterpret_cast<uintptr_t>(C.data[1])));
}

CXCursor cxcursor::MakeCursorTypeRef(TypeDecl *Type, SourceLocation Loc, 
                                     ASTUnit *TU) {
  assert(Type && TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_TypeRef, { Type, RawLoc, TU } };
  return C;    
}

std::pair<TypeDecl *, SourceLocation> 
cxcursor::getCursorTypeRef(CXCursor C) {
  assert(C.kind == CXCursor_TypeRef);
  return std::make_pair(static_cast<TypeDecl *>(C.data[0]),
           SourceLocation::getFromRawEncoding(
                                      reinterpret_cast<uintptr_t>(C.data[1])));
}

CXCursor cxcursor::MakeCursorCXXBaseSpecifier(CXXBaseSpecifier *B, ASTUnit *TU){
  CXCursor C = { CXCursor_CXXBaseSpecifier, { B, 0, TU } };
  return C;  
}

CXXBaseSpecifier *cxcursor::getCursorCXXBaseSpecifier(CXCursor C) {
  assert(C.kind == CXCursor_CXXBaseSpecifier);
  return static_cast<CXXBaseSpecifier*>(C.data[0]);
}

CXCursor cxcursor::MakePreprocessingDirectiveCursor(SourceRange Range, 
                                                    ASTUnit *TU) {
  CXCursor C = { CXCursor_PreprocessingDirective, 
                 { reinterpret_cast<void *>(Range.getBegin().getRawEncoding()),
                   reinterpret_cast<void *>(Range.getEnd().getRawEncoding()),
                   TU }
               };
  return C;
}

SourceRange cxcursor::getCursorPreprocessingDirective(CXCursor C) {
  assert(C.kind == CXCursor_PreprocessingDirective);
  return SourceRange(SourceLocation::getFromRawEncoding(
                                      reinterpret_cast<uintptr_t> (C.data[0])),
                     SourceLocation::getFromRawEncoding(
                                      reinterpret_cast<uintptr_t> (C.data[1])));
}

CXCursor cxcursor::MakeMacroDefinitionCursor(MacroDefinition *MI, ASTUnit *TU) {
  CXCursor C = { CXCursor_MacroDefinition, { MI, 0, TU } };
  return C;
}

MacroDefinition *cxcursor::getCursorMacroDefinition(CXCursor C) {
  assert(C.kind == CXCursor_MacroDefinition);
  return static_cast<MacroDefinition *>(C.data[0]);
}

CXCursor cxcursor::MakeMacroInstantiationCursor(MacroInstantiation *MI, 
                                                ASTUnit *TU) {
  CXCursor C = { CXCursor_MacroInstantiation, { MI, 0, TU } };
  return C;
}

MacroInstantiation *cxcursor::getCursorMacroInstantiation(CXCursor C) {
  assert(C.kind == CXCursor_MacroInstantiation);
  return static_cast<MacroInstantiation *>(C.data[0]);
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

Attr *cxcursor::getCursorAttr(CXCursor Cursor) {
  return (Attr *)Cursor.data[1];
}

ASTContext &cxcursor::getCursorContext(CXCursor Cursor) {
  return getCursorASTUnit(Cursor)->getASTContext();
}

ASTUnit *cxcursor::getCursorASTUnit(CXCursor Cursor) {
  return static_cast<ASTUnit *>(Cursor.data[2]);
}

bool cxcursor::operator==(CXCursor X, CXCursor Y) {
  return X.kind == Y.kind && X.data[0] == Y.data[0] && X.data[1] == Y.data[1] &&
         X.data[2] == Y.data[2];
}
