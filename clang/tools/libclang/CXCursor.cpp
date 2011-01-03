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

#include "CXTranslationUnit.h"
#include "CXCursor.h"
#include "CXString.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang-c/Index.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace cxcursor;

CXCursor cxcursor::MakeCXCursorInvalid(CXCursorKind K) {
  assert(K >= CXCursor_FirstInvalid && K <= CXCursor_LastInvalid);
  CXCursor C = { K, { 0, 0, 0 } };
  return C;
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

CXCursor cxcursor::MakeCXCursor(const Attr *A, Decl *Parent,
                                CXTranslationUnit TU) {
  assert(A && Parent && TU && "Invalid arguments!");
  CXCursor C = { GetCursorKind(A), { Parent, (void*)A, TU } };
  return C;
}

CXCursor cxcursor::MakeCXCursor(Decl *D, CXTranslationUnit TU,
                                bool FirstInDeclGroup) {
  assert(D && TU && "Invalid arguments!");
  CXCursor C = { getCursorKindForDecl(D),
                 { D, (void*) (FirstInDeclGroup ? 1 : 0), TU }
               };
  return C;
}

CXCursor cxcursor::MakeCXCursor(Stmt *S, Decl *Parent,
                                CXTranslationUnit TU) {
  assert(S && TU && "Invalid arguments!");
  CXCursorKind K = CXCursor_NotImplemented;
  
  switch (S->getStmtClass()) {
  case Stmt::NoStmtClass:
    break;
      
  case Stmt::NullStmtClass:
  case Stmt::CompoundStmtClass:
  case Stmt::CaseStmtClass:
  case Stmt::DefaultStmtClass:
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
      
  case Stmt::LabelStmtClass:       
    K = CXCursor_LabelStmt;
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
  case Stmt::ChooseExprClass:           
  case Stmt::GNUNullExprClass:          
  case Stmt::CXXStaticCastExprClass:      
  case Stmt::CXXDynamicCastExprClass:     
  case Stmt::CXXReinterpretCastExprClass: 
  case Stmt::CXXConstCastExprClass:       
  case Stmt::CXXFunctionalCastExprClass:
  case Stmt::CXXTypeidExprClass:          
  case Stmt::CXXUuidofExprClass:          
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
  case Stmt::BinaryTypeTraitExprClass:     
  case Stmt::DependentScopeDeclRefExprClass:  
  case Stmt::CXXBindTemporaryExprClass:   
  case Stmt::ExprWithCleanupsClass: 
  case Stmt::CXXUnresolvedConstructExprClass:
  case Stmt::CXXDependentScopeMemberExprClass:
  case Stmt::UnresolvedMemberExprClass:   
  case Stmt::CXXNoexceptExprClass:
  case Stmt::ObjCStringLiteralClass:    
  case Stmt::ObjCEncodeExprClass:       
  case Stmt::ObjCSelectorExprClass:   
  case Stmt::ObjCProtocolExprClass:   
  case Stmt::ObjCIsaExprClass:       
  case Stmt::ShuffleVectorExprClass: 
  case Stmt::BlockExprClass:  
  case Stmt::OpaqueValueExprClass:
  case Stmt::PackExpansionExprClass:
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
                                               CXTranslationUnit TU) {
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
                                             CXTranslationUnit TU) {
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
                                          CXTranslationUnit TU) {
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
                                     CXTranslationUnit TU) {
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

CXCursor cxcursor::MakeCursorTemplateRef(TemplateDecl *Template, 
                                         SourceLocation Loc,
                                         CXTranslationUnit TU) {
  assert(Template && TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_TemplateRef, { Template, RawLoc, TU } };
  return C;    
}

std::pair<TemplateDecl *, SourceLocation> 
cxcursor::getCursorTemplateRef(CXCursor C) {
  assert(C.kind == CXCursor_TemplateRef);
  return std::make_pair(static_cast<TemplateDecl *>(C.data[0]),
                        SourceLocation::getFromRawEncoding(
                                       reinterpret_cast<uintptr_t>(C.data[1])));  
}

CXCursor cxcursor::MakeCursorNamespaceRef(NamedDecl *NS, SourceLocation Loc, 
                                          CXTranslationUnit TU) {
  
  assert(NS && (isa<NamespaceDecl>(NS) || isa<NamespaceAliasDecl>(NS)) && TU &&
         "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_NamespaceRef, { NS, RawLoc, TU } };
  return C;    
}

std::pair<NamedDecl *, SourceLocation> 
cxcursor::getCursorNamespaceRef(CXCursor C) {
  assert(C.kind == CXCursor_NamespaceRef);
  return std::make_pair(static_cast<NamedDecl *>(C.data[0]),
                        SourceLocation::getFromRawEncoding(
                                       reinterpret_cast<uintptr_t>(C.data[1])));  
}

CXCursor cxcursor::MakeCursorMemberRef(FieldDecl *Field, SourceLocation Loc, 
                                       CXTranslationUnit TU) {
  
  assert(Field && TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_MemberRef, { Field, RawLoc, TU } };
  return C;    
}

std::pair<FieldDecl *, SourceLocation> 
cxcursor::getCursorMemberRef(CXCursor C) {
  assert(C.kind == CXCursor_MemberRef);
  return std::make_pair(static_cast<FieldDecl *>(C.data[0]),
                        SourceLocation::getFromRawEncoding(
                                       reinterpret_cast<uintptr_t>(C.data[1])));  
}

CXCursor cxcursor::MakeCursorCXXBaseSpecifier(CXXBaseSpecifier *B,
                                              CXTranslationUnit TU){
  CXCursor C = { CXCursor_CXXBaseSpecifier, { B, 0, TU } };
  return C;  
}

CXXBaseSpecifier *cxcursor::getCursorCXXBaseSpecifier(CXCursor C) {
  assert(C.kind == CXCursor_CXXBaseSpecifier);
  return static_cast<CXXBaseSpecifier*>(C.data[0]);
}

CXCursor cxcursor::MakePreprocessingDirectiveCursor(SourceRange Range, 
                                                    CXTranslationUnit TU) {
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

CXCursor cxcursor::MakeMacroDefinitionCursor(MacroDefinition *MI,
                                             CXTranslationUnit TU) {
  CXCursor C = { CXCursor_MacroDefinition, { MI, 0, TU } };
  return C;
}

MacroDefinition *cxcursor::getCursorMacroDefinition(CXCursor C) {
  assert(C.kind == CXCursor_MacroDefinition);
  return static_cast<MacroDefinition *>(C.data[0]);
}

CXCursor cxcursor::MakeMacroInstantiationCursor(MacroInstantiation *MI, 
                                                CXTranslationUnit TU) {
  CXCursor C = { CXCursor_MacroInstantiation, { MI, 0, TU } };
  return C;
}

MacroInstantiation *cxcursor::getCursorMacroInstantiation(CXCursor C) {
  assert(C.kind == CXCursor_MacroInstantiation);
  return static_cast<MacroInstantiation *>(C.data[0]);
}

CXCursor cxcursor::MakeInclusionDirectiveCursor(InclusionDirective *ID, 
                                                CXTranslationUnit TU) {
  CXCursor C = { CXCursor_InclusionDirective, { ID, 0, TU } };
  return C;
}

InclusionDirective *cxcursor::getCursorInclusionDirective(CXCursor C) {
  assert(C.kind == CXCursor_InclusionDirective);
  return static_cast<InclusionDirective *>(C.data[0]);  
}

CXCursor cxcursor::MakeCursorLabelRef(LabelStmt *Label, SourceLocation Loc, 
                                      CXTranslationUnit TU) {
  
  assert(Label && TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_LabelRef, { Label, RawLoc, TU } };
  return C;    
}

std::pair<LabelStmt*, SourceLocation> 
cxcursor::getCursorLabelRef(CXCursor C) {
  assert(C.kind == CXCursor_LabelRef);
  return std::make_pair(static_cast<LabelStmt *>(C.data[0]),
                        SourceLocation::getFromRawEncoding(
                                       reinterpret_cast<uintptr_t>(C.data[1])));  
}

CXCursor cxcursor::MakeCursorOverloadedDeclRef(OverloadExpr *E, 
                                               CXTranslationUnit TU) {
  assert(E && TU && "Invalid arguments!");
  OverloadedDeclRefStorage Storage(E);
  void *RawLoc = reinterpret_cast<void *>(E->getNameLoc().getRawEncoding());
  CXCursor C = { 
                 CXCursor_OverloadedDeclRef, 
                 { Storage.getOpaqueValue(), RawLoc, TU } 
               };
  return C;    
}

CXCursor cxcursor::MakeCursorOverloadedDeclRef(Decl *D, 
                                               SourceLocation Loc,
                                               CXTranslationUnit TU) {
  assert(D && TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  OverloadedDeclRefStorage Storage(D);
  CXCursor C = { 
    CXCursor_OverloadedDeclRef, 
    { Storage.getOpaqueValue(), RawLoc, TU }
  };
  return C;    
}

CXCursor cxcursor::MakeCursorOverloadedDeclRef(TemplateName Name, 
                                               SourceLocation Loc,
                                               CXTranslationUnit TU) {
  assert(Name.getAsOverloadedTemplate() && TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  OverloadedDeclRefStorage Storage(Name.getAsOverloadedTemplate());
  CXCursor C = { 
    CXCursor_OverloadedDeclRef, 
    { Storage.getOpaqueValue(), RawLoc, TU } 
  };
  return C;    
}

std::pair<cxcursor::OverloadedDeclRefStorage, SourceLocation>
cxcursor::getCursorOverloadedDeclRef(CXCursor C) {
  assert(C.kind == CXCursor_OverloadedDeclRef);
  return std::make_pair(OverloadedDeclRefStorage::getFromOpaqueValue(C.data[0]),
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

Attr *cxcursor::getCursorAttr(CXCursor Cursor) {
  return (Attr *)Cursor.data[1];
}

ASTContext &cxcursor::getCursorContext(CXCursor Cursor) {
  return getCursorASTUnit(Cursor)->getASTContext();
}

ASTUnit *cxcursor::getCursorASTUnit(CXCursor Cursor) {
  return static_cast<ASTUnit *>(static_cast<CXTranslationUnit>(Cursor.data[2])
                                  ->TUData);
}

CXTranslationUnit cxcursor::getCursorTU(CXCursor Cursor) {
  return static_cast<CXTranslationUnit>(Cursor.data[2]);
}

bool cxcursor::operator==(CXCursor X, CXCursor Y) {
  return X.kind == Y.kind && X.data[0] == Y.data[0] && X.data[1] == Y.data[1] &&
         X.data[2] == Y.data[2];
}

// FIXME: Remove once we can model DeclGroups and their appropriate ranges
// properly in the ASTs.
bool cxcursor::isFirstInDeclGroup(CXCursor C) {
  assert(clang_isDeclaration(C.kind));
  return ((uintptr_t) (C.data[1])) != 0;
}

//===----------------------------------------------------------------------===//
// CXCursorSet.
//===----------------------------------------------------------------------===//

typedef llvm::DenseMap<CXCursor, unsigned> CXCursorSet_Impl;

static inline CXCursorSet packCXCursorSet(CXCursorSet_Impl *setImpl) {
  return (CXCursorSet) setImpl;
}
static inline CXCursorSet_Impl *unpackCXCursorSet(CXCursorSet set) {
  return (CXCursorSet_Impl*) set;
}
namespace llvm {
template<> struct DenseMapInfo<CXCursor> {
public:
  static inline CXCursor getEmptyKey() {
    return MakeCXCursorInvalid(CXCursor_InvalidFile);
  }
  static inline CXCursor getTombstoneKey() {
    return MakeCXCursorInvalid(CXCursor_NoDeclFound);
  }
  static inline unsigned getHashValue(const CXCursor &cursor) {
    return llvm::DenseMapInfo<std::pair<void*,void*> >
      ::getHashValue(std::make_pair(cursor.data[0], cursor.data[1]));
  }
  static inline bool isEqual(const CXCursor &x, const CXCursor &y) {
    return x.kind == y.kind &&
           x.data[0] == y.data[0] &&
           x.data[1] == y.data[1];
  }
};
}

extern "C" {
CXCursorSet clang_createCXCursorSet() {
  return packCXCursorSet(new CXCursorSet_Impl());
}

void clang_disposeCXCursorSet(CXCursorSet set) {
  delete unpackCXCursorSet(set);
}

unsigned clang_CXCursorSet_contains(CXCursorSet set, CXCursor cursor) {
  CXCursorSet_Impl *setImpl = unpackCXCursorSet(set);
  if (!setImpl)
    return 0;
  return setImpl->find(cursor) == setImpl->end();
}

unsigned clang_CXCursorSet_insert(CXCursorSet set, CXCursor cursor) {
  // Do not insert invalid cursors into the set.
  if (cursor.kind >= CXCursor_FirstInvalid &&
      cursor.kind <= CXCursor_LastInvalid)
    return 1;

  CXCursorSet_Impl *setImpl = unpackCXCursorSet(set);
  if (!setImpl)
    return 1;
  unsigned &entry = (*setImpl)[cursor];
  unsigned flag = entry == 0 ? 1 : 0;
  entry = 1;
  return flag;
}
} // end: extern "C"

