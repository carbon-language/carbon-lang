//===---- StmtProfile.cpp - Profile implementation for Stmt ASTs ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Stmt::Profile method, which builds a unique bit
// representation that identifies a statement/expression.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/Compiler.h"
using namespace clang;

namespace {
  class VISIBILITY_HIDDEN StmtProfiler : public StmtVisitor<StmtProfiler> {
    llvm::FoldingSetNodeID &ID;
    ASTContext &Context;
    bool Canonical;
    
  public:
    StmtProfiler(llvm::FoldingSetNodeID &ID, ASTContext &Context,
                 bool Canonical) 
      : ID(ID), Context(Context), Canonical(Canonical) { }
    
    void VisitStmt(Stmt *S);
    
#define STMT(Node, Base) void Visit##Node(Node *S);
#include "clang/AST/StmtNodes.def"
    
    /// \brief Visit a declaration that is referenced within an expression
    /// or statement.
    void VisitDecl(Decl *D);
    
    /// \brief Visit a type that is referenced within an expression or 
    /// statement.
    void VisitType(QualType T);
    
    /// \brief Visit a name that occurs within an expression or statement.
    void VisitName(DeclarationName Name);
    
    /// \brief Visit a nested-name-specifier that occurs within an expression
    /// or statement.
    void VisitNestedNameSpecifier(NestedNameSpecifier *NNS);
    
    /// \brief Visit a template name that occurs within an expression or
    /// statement.
    void VisitTemplateName(TemplateName Name);
    
    /// \brief Visit template arguments that occur within an expression or
    /// statement.
    void VisitTemplateArguments(const TemplateArgument *Args, unsigned NumArgs);
  };
}

void StmtProfiler::VisitStmt(Stmt *S) {
  ID.AddInteger(S->getStmtClass());
  for (Stmt::child_iterator C = S->child_begin(), CEnd = S->child_end();
       C != CEnd; ++C)
    Visit(*C);
}

void StmtProfiler::VisitDeclStmt(DeclStmt *S) {
  VisitStmt(S);
  for (DeclStmt::decl_iterator D = S->decl_begin(), DEnd = S->decl_end();
       D != DEnd; ++D)
    VisitDecl(*D);
}

void StmtProfiler::VisitNullStmt(NullStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitCompoundStmt(CompoundStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitSwitchCase(SwitchCase *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitCaseStmt(CaseStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitDefaultStmt(DefaultStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitLabelStmt(LabelStmt *S) {
  VisitStmt(S);
  VisitName(S->getID());
}

void StmtProfiler::VisitIfStmt(IfStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitSwitchStmt(SwitchStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitWhileStmt(WhileStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitDoStmt(DoStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitForStmt(ForStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitGotoStmt(GotoStmt *S) {
  VisitStmt(S);
  VisitName(S->getLabel()->getID());
}

void StmtProfiler::VisitIndirectGotoStmt(IndirectGotoStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitContinueStmt(ContinueStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitBreakStmt(BreakStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitReturnStmt(ReturnStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitAsmStmt(AsmStmt *S) {
  VisitStmt(S);
  ID.AddBoolean(S->isVolatile());
  ID.AddBoolean(S->isSimple());
  VisitStringLiteral(S->getAsmString());
  ID.AddInteger(S->getNumOutputs());
  for (unsigned I = 0, N = S->getNumOutputs(); I != N; ++I) {
    ID.AddString(S->getOutputName(I));
    VisitStringLiteral(S->getOutputConstraintLiteral(I));
  }
  ID.AddInteger(S->getNumInputs());
  for (unsigned I = 0, N = S->getNumInputs(); I != N; ++I) {
    ID.AddString(S->getInputName(I));
    VisitStringLiteral(S->getInputConstraintLiteral(I));
  }
  ID.AddInteger(S->getNumClobbers());
  for (unsigned I = 0, N = S->getNumClobbers(); I != N; ++I)
    VisitStringLiteral(S->getClobber(I));
}

void StmtProfiler::VisitCXXCatchStmt(CXXCatchStmt *S) {
  VisitStmt(S);
  VisitType(S->getCaughtType());
}

void StmtProfiler::VisitCXXTryStmt(CXXTryStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitObjCForCollectionStmt(ObjCForCollectionStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitObjCAtCatchStmt(ObjCAtCatchStmt *S) {
  VisitStmt(S);
  ID.AddBoolean(S->hasEllipsis());
  if (S->getCatchParamDecl())
    VisitType(S->getCatchParamDecl()->getType());
}

void StmtProfiler::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitObjCAtTryStmt(ObjCAtTryStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitObjCAtThrowStmt(ObjCAtThrowStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitExpr(Expr *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitDeclRefExpr(DeclRefExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getDecl());
}

void StmtProfiler::VisitPredefinedExpr(PredefinedExpr *S) {
  VisitExpr(S);
  ID.AddInteger(S->getIdentType());
}

void StmtProfiler::VisitIntegerLiteral(IntegerLiteral *S) {
  VisitExpr(S);
  S->getValue().Profile(ID);
}

void StmtProfiler::VisitCharacterLiteral(CharacterLiteral *S) {
  VisitExpr(S);
  ID.AddBoolean(S->isWide());
  ID.AddInteger(S->getValue());
}

void StmtProfiler::VisitFloatingLiteral(FloatingLiteral *S) {
  VisitExpr(S);
  S->getValue().Profile(ID);
  ID.AddBoolean(S->isExact());
}

void StmtProfiler::VisitImaginaryLiteral(ImaginaryLiteral *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitStringLiteral(StringLiteral *S) {
  VisitExpr(S);
  ID.AddString(S->getStrData(), S->getStrData() + S->getByteLength());
  ID.AddBoolean(S->isWide());
}

void StmtProfiler::VisitParenExpr(ParenExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitParenListExpr(ParenListExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitUnaryOperator(UnaryOperator *S) {
  VisitExpr(S);
  ID.AddInteger(S->getOpcode());
}

void StmtProfiler::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *S) {
  VisitExpr(S);
  ID.AddBoolean(S->isSizeOf());
  if (S->isArgumentType())
    VisitType(S->getArgumentType());
}

void StmtProfiler::VisitArraySubscriptExpr(ArraySubscriptExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitCallExpr(CallExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitMemberExpr(MemberExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getMemberDecl());
  VisitNestedNameSpecifier(S->getQualifier());
  ID.AddBoolean(S->isArrow());
}

void StmtProfiler::VisitCompoundLiteralExpr(CompoundLiteralExpr *S) {
  VisitExpr(S);
  ID.AddBoolean(S->isFileScope());
}

void StmtProfiler::VisitCastExpr(CastExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitImplicitCastExpr(ImplicitCastExpr *S) {
  VisitCastExpr(S);
  ID.AddBoolean(S->isLvalueCast());
}

void StmtProfiler::VisitExplicitCastExpr(ExplicitCastExpr *S) {
  VisitCastExpr(S);
  VisitType(S->getTypeAsWritten());
}

void StmtProfiler::VisitCStyleCastExpr(CStyleCastExpr *S) {
  VisitExplicitCastExpr(S);
}

void StmtProfiler::VisitBinaryOperator(BinaryOperator *S) {
  VisitExpr(S);
  ID.AddInteger(S->getOpcode());
}

void StmtProfiler::VisitCompoundAssignOperator(CompoundAssignOperator *S) {
  VisitBinaryOperator(S);
}

void StmtProfiler::VisitConditionalOperator(ConditionalOperator *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitAddrLabelExpr(AddrLabelExpr *S) {
  VisitExpr(S);
  VisitName(S->getLabel()->getID());
}

void StmtProfiler::VisitStmtExpr(StmtExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitTypesCompatibleExpr(TypesCompatibleExpr *S) {
  VisitExpr(S);
  VisitType(S->getArgType1());
  VisitType(S->getArgType2());
}

void StmtProfiler::VisitShuffleVectorExpr(ShuffleVectorExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitChooseExpr(ChooseExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitGNUNullExpr(GNUNullExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitVAArgExpr(VAArgExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitInitListExpr(InitListExpr *S) {
  if (S->getSyntacticForm()) {
    VisitInitListExpr(S->getSyntacticForm());
    return;
  }
  
  VisitExpr(S);
}

void StmtProfiler::VisitDesignatedInitExpr(DesignatedInitExpr *S) {
  VisitExpr(S);
  ID.AddBoolean(S->usesGNUSyntax());
  for (DesignatedInitExpr::designators_iterator D = S->designators_begin(),
                                             DEnd = S->designators_end();
       D != DEnd; ++D) {
    if (D->isFieldDesignator()) {
      ID.AddInteger(0);
      VisitName(D->getFieldName());
      continue;
    }
    
    if (D->isArrayDesignator()) {
      ID.AddInteger(1);
    } else {
      assert(D->isArrayRangeDesignator());
      ID.AddInteger(2);
    }
    ID.AddInteger(D->getFirstExprIndex());
  }
}

void StmtProfiler::VisitImplicitValueInitExpr(ImplicitValueInitExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitExtVectorElementExpr(ExtVectorElementExpr *S) {
  VisitExpr(S);
  VisitName(&S->getAccessor());
}

void StmtProfiler::VisitBlockExpr(BlockExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getBlockDecl());
}

void StmtProfiler::VisitBlockDeclRefExpr(BlockDeclRefExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getDecl());
  ID.AddBoolean(S->isByRef());
  ID.AddBoolean(S->isConstQualAdded());
}

void StmtProfiler::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *S) {
  VisitCallExpr(S);
  ID.AddInteger(S->getOperator());
}

void StmtProfiler::VisitCXXMemberCallExpr(CXXMemberCallExpr *S) {
  VisitCallExpr(S);
}

void StmtProfiler::VisitCXXNamedCastExpr(CXXNamedCastExpr *S) {
  VisitExplicitCastExpr(S);
}

void StmtProfiler::VisitCXXStaticCastExpr(CXXStaticCastExpr *S) {
  VisitCXXNamedCastExpr(S);
}

void StmtProfiler::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *S) {
  VisitCXXNamedCastExpr(S);
}

void StmtProfiler::VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *S) {
  VisitCXXNamedCastExpr(S);
}

void StmtProfiler::VisitCXXConstCastExpr(CXXConstCastExpr *S) {
  VisitCXXNamedCastExpr(S);
}

void StmtProfiler::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *S) {
  VisitExpr(S);
  ID.AddBoolean(S->getValue());
}

void StmtProfiler::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitCXXTypeidExpr(CXXTypeidExpr *S) {
  VisitExpr(S);
  if (S->isTypeOperand())
    VisitType(S->getTypeOperand());
}

void StmtProfiler::VisitCXXThisExpr(CXXThisExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitCXXThrowExpr(CXXThrowExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getParam());
}

void StmtProfiler::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *S) {
  VisitExpr(S);
  VisitDecl(
         const_cast<CXXDestructorDecl *>(S->getTemporary()->getDestructor()));
}

void StmtProfiler::VisitCXXConstructExpr(CXXConstructExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getConstructor());
  ID.AddBoolean(S->isElidable());
}

void StmtProfiler::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *S) {
  VisitExplicitCastExpr(S);
}

void StmtProfiler::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *S) {
  VisitCXXConstructExpr(S);
}

void StmtProfiler::VisitCXXZeroInitValueExpr(CXXZeroInitValueExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitCXXConditionDeclExpr(CXXConditionDeclExpr *S) {
  VisitDeclRefExpr(S);
}

void StmtProfiler::VisitCXXDeleteExpr(CXXDeleteExpr *S) {
  VisitExpr(S);
  ID.AddBoolean(S->isGlobalDelete());
  ID.AddBoolean(S->isArrayForm());
  VisitDecl(S->getOperatorDelete());
}


void StmtProfiler::VisitCXXNewExpr(CXXNewExpr *S) {
  VisitExpr(S);
  VisitType(S->getAllocatedType());
  VisitDecl(S->getOperatorNew());
  VisitDecl(S->getOperatorDelete());
  VisitDecl(S->getConstructor());
  ID.AddBoolean(S->isArray());
  ID.AddInteger(S->getNumPlacementArgs());
  ID.AddBoolean(S->isGlobalNew());
  ID.AddBoolean(S->isParenTypeId());
  ID.AddBoolean(S->hasInitializer());
  ID.AddInteger(S->getNumConstructorArgs());
}

void 
StmtProfiler::VisitUnresolvedFunctionNameExpr(UnresolvedFunctionNameExpr *S) {
  VisitExpr(S);
  VisitName(S->getName());
}

void StmtProfiler::VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *S) {
  VisitExpr(S);
  ID.AddInteger(S->getTrait());
  VisitType(S->getQueriedType());
}

void StmtProfiler::VisitQualifiedDeclRefExpr(QualifiedDeclRefExpr *S) {
  VisitDeclRefExpr(S);
  VisitNestedNameSpecifier(S->getQualifier());
}

void StmtProfiler::VisitUnresolvedDeclRefExpr(UnresolvedDeclRefExpr *S) {
  VisitExpr(S);
  VisitName(S->getDeclName());
  VisitNestedNameSpecifier(S->getQualifier());
  ID.AddBoolean(S->isAddressOfOperand());
}

void StmtProfiler::VisitTemplateIdRefExpr(TemplateIdRefExpr *S) {
  VisitExpr(S);
  VisitNestedNameSpecifier(S->getQualifier());
  VisitTemplateName(S->getTemplateName());
  VisitTemplateArguments(S->getTemplateArgs(), S->getNumTemplateArgs());
}

void StmtProfiler::VisitCXXExprWithTemporaries(CXXExprWithTemporaries *S) {
  VisitExpr(S);
  ID.AddBoolean(S->shouldDestroyTemporaries());
  for (unsigned I = 0, N = S->getNumTemporaries(); I != N; ++I)
    VisitDecl(
      const_cast<CXXDestructorDecl *>(S->getTemporary(I)->getDestructor()));
}

void 
StmtProfiler::VisitCXXUnresolvedConstructExpr(CXXUnresolvedConstructExpr *S) {
  VisitExpr(S);
  VisitType(S->getTypeAsWritten());
}

void StmtProfiler::VisitCXXUnresolvedMemberExpr(CXXUnresolvedMemberExpr *S) {
  VisitExpr(S);
  ID.AddBoolean(S->isArrow());
  VisitName(S->getMember());
}

void StmtProfiler::VisitObjCStringLiteral(ObjCStringLiteral *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitObjCEncodeExpr(ObjCEncodeExpr *S) {
  VisitExpr(S);
  VisitType(S->getEncodedType());
}

void StmtProfiler::VisitObjCSelectorExpr(ObjCSelectorExpr *S) {
  VisitExpr(S);
  VisitName(S->getSelector());
}

void StmtProfiler::VisitObjCProtocolExpr(ObjCProtocolExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getProtocol());
}

void StmtProfiler::VisitObjCIvarRefExpr(ObjCIvarRefExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getDecl());
  ID.AddBoolean(S->isArrow());
  ID.AddBoolean(S->isFreeIvar());
}

void StmtProfiler::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getProperty());
}

void StmtProfiler::VisitObjCImplicitSetterGetterRefExpr(
                                  ObjCImplicitSetterGetterRefExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getGetterMethod());
  VisitDecl(S->getSetterMethod());
  VisitDecl(S->getInterfaceDecl());
}

void StmtProfiler::VisitObjCMessageExpr(ObjCMessageExpr *S) {
  VisitExpr(S);
  VisitName(S->getSelector());
  VisitDecl(S->getMethodDecl());
}

void StmtProfiler::VisitObjCSuperExpr(ObjCSuperExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitObjCIsaExpr(ObjCIsaExpr *S) {
  VisitExpr(S);
  ID.AddBoolean(S->isArrow());
}

void StmtProfiler::VisitDecl(Decl *D) {
  ID.AddInteger(D? D->getKind() : 0);
  
  if (Canonical && D) {
    if (NonTypeTemplateParmDecl *NTTP = dyn_cast<NonTypeTemplateParmDecl>(D)) {
      ID.AddInteger(NTTP->getDepth());
      ID.AddInteger(NTTP->getIndex());
      VisitType(NTTP->getType());
      return;
    }
    
    if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(D)) {
      // The Itanium C++ ABI uses the type of a parameter when mangling
      // expressions that involve function parameters, so we will use the
      // parameter's type for establishing function parameter identity. That
      // way, our definition of "equivalent" (per C++ [temp.over.link]) 
      // matches the definition of "equivalent" used for name mangling.
      VisitType(Parm->getType());
      return;
    }
    
    if (TemplateTemplateParmDecl *TTP = dyn_cast<TemplateTemplateParmDecl>(D)) {
      ID.AddInteger(TTP->getDepth());
      ID.AddInteger(TTP->getIndex());
      return;
    }
    
    if (OverloadedFunctionDecl *Ovl = dyn_cast<OverloadedFunctionDecl>(D)) {
      // The Itanium C++ ABI mangles references to a set of overloaded 
      // functions using just the function name, so we do the same here.
      VisitName(Ovl->getDeclName());
      return;
    }
  }
  
  ID.AddPointer(D? D->getCanonicalDecl() : 0);
}

void StmtProfiler::VisitType(QualType T) {
  if (Canonical)
    T = Context.getCanonicalType(T);
  
  ID.AddPointer(T.getAsOpaquePtr());
}

void StmtProfiler::VisitName(DeclarationName Name) {
  ID.AddPointer(Name.getAsOpaquePtr());
}

void StmtProfiler::VisitNestedNameSpecifier(NestedNameSpecifier *NNS) {
  if (Canonical)
    NNS = Context.getCanonicalNestedNameSpecifier(NNS);
  ID.AddPointer(NNS);
}

void StmtProfiler::VisitTemplateName(TemplateName Name) {
  if (Canonical)
    Name = Context.getCanonicalTemplateName(Name);
  
  Name.Profile(ID);
}

void StmtProfiler::VisitTemplateArguments(const TemplateArgument *Args, 
                                          unsigned NumArgs) {
  ID.AddInteger(NumArgs);
  for (unsigned I = 0; I != NumArgs; ++I) {
    const TemplateArgument &Arg = Args[I];
    
    // Mostly repetitive with TemplateArgument::Profile!
    ID.AddInteger(Arg.getKind());
    switch (Arg.getKind()) {
      case TemplateArgument::Null:
        break;
        
      case TemplateArgument::Type:
        VisitType(Arg.getAsType());
        break;
        
      case TemplateArgument::Declaration:
        VisitDecl(Arg.getAsDecl());
        break;
        
      case TemplateArgument::Integral:
        Arg.getAsIntegral()->Profile(ID);
        VisitType(Arg.getIntegralType());
        break;
        
      case TemplateArgument::Expression:
        Visit(Arg.getAsExpr());
        break;
        
      case TemplateArgument::Pack:
        VisitTemplateArguments(Arg.pack_begin(), Arg.pack_size());
        break;
    }
  }
}

void Stmt::Profile(llvm::FoldingSetNodeID &ID, ASTContext &Context,
                   bool Canonical) {
  StmtProfiler Profiler(ID, Context, Canonical);
  Profiler.Visit(this);
}
