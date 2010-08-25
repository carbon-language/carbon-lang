//===--- ASTWriterStmt.cpp - Statement and Expression Serialization -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements serialization for Statements and Expressions.
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/ASTWriter.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/Bitcode/BitstreamWriter.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// Statement/expression serialization
//===----------------------------------------------------------------------===//

namespace clang {
  class ASTStmtWriter : public StmtVisitor<ASTStmtWriter, void> {
    ASTWriter &Writer;
    ASTWriter::RecordData &Record;

  public:
    serialization::StmtCode Code;

    ASTStmtWriter(ASTWriter &Writer, ASTWriter::RecordData &Record)
      : Writer(Writer), Record(Record) { }
    
    void
    AddExplicitTemplateArgumentList(const ExplicitTemplateArgumentList &Args);

    void VisitStmt(Stmt *S);
    void VisitNullStmt(NullStmt *S);
    void VisitCompoundStmt(CompoundStmt *S);
    void VisitSwitchCase(SwitchCase *S);
    void VisitCaseStmt(CaseStmt *S);
    void VisitDefaultStmt(DefaultStmt *S);
    void VisitLabelStmt(LabelStmt *S);
    void VisitIfStmt(IfStmt *S);
    void VisitSwitchStmt(SwitchStmt *S);
    void VisitWhileStmt(WhileStmt *S);
    void VisitDoStmt(DoStmt *S);
    void VisitForStmt(ForStmt *S);
    void VisitGotoStmt(GotoStmt *S);
    void VisitIndirectGotoStmt(IndirectGotoStmt *S);
    void VisitContinueStmt(ContinueStmt *S);
    void VisitBreakStmt(BreakStmt *S);
    void VisitReturnStmt(ReturnStmt *S);
    void VisitDeclStmt(DeclStmt *S);
    void VisitAsmStmt(AsmStmt *S);
    void VisitExpr(Expr *E);
    void VisitPredefinedExpr(PredefinedExpr *E);
    void VisitDeclRefExpr(DeclRefExpr *E);
    void VisitIntegerLiteral(IntegerLiteral *E);
    void VisitFloatingLiteral(FloatingLiteral *E);
    void VisitImaginaryLiteral(ImaginaryLiteral *E);
    void VisitStringLiteral(StringLiteral *E);
    void VisitCharacterLiteral(CharacterLiteral *E);
    void VisitParenExpr(ParenExpr *E);
    void VisitParenListExpr(ParenListExpr *E);
    void VisitUnaryOperator(UnaryOperator *E);
    void VisitOffsetOfExpr(OffsetOfExpr *E);
    void VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E);
    void VisitArraySubscriptExpr(ArraySubscriptExpr *E);
    void VisitCallExpr(CallExpr *E);
    void VisitMemberExpr(MemberExpr *E);
    void VisitCastExpr(CastExpr *E);
    void VisitBinaryOperator(BinaryOperator *E);
    void VisitCompoundAssignOperator(CompoundAssignOperator *E);
    void VisitConditionalOperator(ConditionalOperator *E);
    void VisitImplicitCastExpr(ImplicitCastExpr *E);
    void VisitExplicitCastExpr(ExplicitCastExpr *E);
    void VisitCStyleCastExpr(CStyleCastExpr *E);
    void VisitCompoundLiteralExpr(CompoundLiteralExpr *E);
    void VisitExtVectorElementExpr(ExtVectorElementExpr *E);
    void VisitInitListExpr(InitListExpr *E);
    void VisitDesignatedInitExpr(DesignatedInitExpr *E);
    void VisitImplicitValueInitExpr(ImplicitValueInitExpr *E);
    void VisitVAArgExpr(VAArgExpr *E);
    void VisitAddrLabelExpr(AddrLabelExpr *E);
    void VisitStmtExpr(StmtExpr *E);
    void VisitTypesCompatibleExpr(TypesCompatibleExpr *E);
    void VisitChooseExpr(ChooseExpr *E);
    void VisitGNUNullExpr(GNUNullExpr *E);
    void VisitShuffleVectorExpr(ShuffleVectorExpr *E);
    void VisitBlockExpr(BlockExpr *E);
    void VisitBlockDeclRefExpr(BlockDeclRefExpr *E);

    // Objective-C Expressions
    void VisitObjCStringLiteral(ObjCStringLiteral *E);
    void VisitObjCEncodeExpr(ObjCEncodeExpr *E);
    void VisitObjCSelectorExpr(ObjCSelectorExpr *E);
    void VisitObjCProtocolExpr(ObjCProtocolExpr *E);
    void VisitObjCIvarRefExpr(ObjCIvarRefExpr *E);
    void VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *E);
    void VisitObjCImplicitSetterGetterRefExpr(
                        ObjCImplicitSetterGetterRefExpr *E);
    void VisitObjCMessageExpr(ObjCMessageExpr *E);
    void VisitObjCSuperExpr(ObjCSuperExpr *E);
    void VisitObjCIsaExpr(ObjCIsaExpr *E);

    // Objective-C Statements
    void VisitObjCForCollectionStmt(ObjCForCollectionStmt *);
    void VisitObjCAtCatchStmt(ObjCAtCatchStmt *);
    void VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *);
    void VisitObjCAtTryStmt(ObjCAtTryStmt *);
    void VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *);
    void VisitObjCAtThrowStmt(ObjCAtThrowStmt *);

    // C++ Statements
    void VisitCXXCatchStmt(CXXCatchStmt *S);
    void VisitCXXTryStmt(CXXTryStmt *S);

    void VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E);
    void VisitCXXMemberCallExpr(CXXMemberCallExpr *E);
    void VisitCXXConstructExpr(CXXConstructExpr *E);
    void VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *E);
    void VisitCXXNamedCastExpr(CXXNamedCastExpr *E);
    void VisitCXXStaticCastExpr(CXXStaticCastExpr *E);
    void VisitCXXDynamicCastExpr(CXXDynamicCastExpr *E);
    void VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *E);
    void VisitCXXConstCastExpr(CXXConstCastExpr *E);
    void VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *E);
    void VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *E);
    void VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E);
    void VisitCXXTypeidExpr(CXXTypeidExpr *E);
    void VisitCXXThisExpr(CXXThisExpr *E);
    void VisitCXXThrowExpr(CXXThrowExpr *E);
    void VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E);
    void VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E);
    void VisitCXXBindReferenceExpr(CXXBindReferenceExpr *E);

    void VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E);
    void VisitCXXNewExpr(CXXNewExpr *E);
    void VisitCXXDeleteExpr(CXXDeleteExpr *E);
    void VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E);

    void VisitCXXExprWithTemporaries(CXXExprWithTemporaries *E);
    void VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *E);
    void VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *E);
    void VisitCXXUnresolvedConstructExpr(CXXUnresolvedConstructExpr *E);

    void VisitOverloadExpr(OverloadExpr *E);
    void VisitUnresolvedMemberExpr(UnresolvedMemberExpr *E);
    void VisitUnresolvedLookupExpr(UnresolvedLookupExpr *E);

    void VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *E);
  };
}

void ASTStmtWriter::
AddExplicitTemplateArgumentList(const ExplicitTemplateArgumentList &Args) {
  Writer.AddSourceLocation(Args.LAngleLoc, Record);
  Writer.AddSourceLocation(Args.RAngleLoc, Record);
  for (unsigned i=0; i != Args.NumTemplateArgs; ++i)
    Writer.AddTemplateArgumentLoc(Args.getTemplateArgs()[i], Record);
}

void ASTStmtWriter::VisitStmt(Stmt *S) {
}

void ASTStmtWriter::VisitNullStmt(NullStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getSemiLoc(), Record);
  Code = serialization::STMT_NULL;
}

void ASTStmtWriter::VisitCompoundStmt(CompoundStmt *S) {
  VisitStmt(S);
  Record.push_back(S->size());
  for (CompoundStmt::body_iterator CS = S->body_begin(), CSEnd = S->body_end();
       CS != CSEnd; ++CS)
    Writer.AddStmt(*CS);
  Writer.AddSourceLocation(S->getLBracLoc(), Record);
  Writer.AddSourceLocation(S->getRBracLoc(), Record);
  Code = serialization::STMT_COMPOUND;
}

void ASTStmtWriter::VisitSwitchCase(SwitchCase *S) {
  VisitStmt(S);
  Record.push_back(Writer.getSwitchCaseID(S));
}

void ASTStmtWriter::VisitCaseStmt(CaseStmt *S) {
  VisitSwitchCase(S);
  Writer.AddStmt(S->getLHS());
  Writer.AddStmt(S->getRHS());
  Writer.AddStmt(S->getSubStmt());
  Writer.AddSourceLocation(S->getCaseLoc(), Record);
  Writer.AddSourceLocation(S->getEllipsisLoc(), Record);
  Writer.AddSourceLocation(S->getColonLoc(), Record);
  Code = serialization::STMT_CASE;
}

void ASTStmtWriter::VisitDefaultStmt(DefaultStmt *S) {
  VisitSwitchCase(S);
  Writer.AddStmt(S->getSubStmt());
  Writer.AddSourceLocation(S->getDefaultLoc(), Record);
  Writer.AddSourceLocation(S->getColonLoc(), Record);
  Code = serialization::STMT_DEFAULT;
}

void ASTStmtWriter::VisitLabelStmt(LabelStmt *S) {
  VisitStmt(S);
  Writer.AddIdentifierRef(S->getID(), Record);
  Writer.AddStmt(S->getSubStmt());
  Writer.AddSourceLocation(S->getIdentLoc(), Record);
  Record.push_back(Writer.GetLabelID(S));
  Code = serialization::STMT_LABEL;
}

void ASTStmtWriter::VisitIfStmt(IfStmt *S) {
  VisitStmt(S);
  Writer.AddDeclRef(S->getConditionVariable(), Record);
  Writer.AddStmt(S->getCond());
  Writer.AddStmt(S->getThen());
  Writer.AddStmt(S->getElse());
  Writer.AddSourceLocation(S->getIfLoc(), Record);
  Writer.AddSourceLocation(S->getElseLoc(), Record);
  Code = serialization::STMT_IF;
}

void ASTStmtWriter::VisitSwitchStmt(SwitchStmt *S) {
  VisitStmt(S);
  Writer.AddDeclRef(S->getConditionVariable(), Record);
  Writer.AddStmt(S->getCond());
  Writer.AddStmt(S->getBody());
  Writer.AddSourceLocation(S->getSwitchLoc(), Record);
  for (SwitchCase *SC = S->getSwitchCaseList(); SC;
       SC = SC->getNextSwitchCase())
    Record.push_back(Writer.RecordSwitchCaseID(SC));
  Code = serialization::STMT_SWITCH;
}

void ASTStmtWriter::VisitWhileStmt(WhileStmt *S) {
  VisitStmt(S);
  Writer.AddDeclRef(S->getConditionVariable(), Record);
  Writer.AddStmt(S->getCond());
  Writer.AddStmt(S->getBody());
  Writer.AddSourceLocation(S->getWhileLoc(), Record);
  Code = serialization::STMT_WHILE;
}

void ASTStmtWriter::VisitDoStmt(DoStmt *S) {
  VisitStmt(S);
  Writer.AddStmt(S->getCond());
  Writer.AddStmt(S->getBody());
  Writer.AddSourceLocation(S->getDoLoc(), Record);
  Writer.AddSourceLocation(S->getWhileLoc(), Record);
  Writer.AddSourceLocation(S->getRParenLoc(), Record);
  Code = serialization::STMT_DO;
}

void ASTStmtWriter::VisitForStmt(ForStmt *S) {
  VisitStmt(S);
  Writer.AddStmt(S->getInit());
  Writer.AddStmt(S->getCond());
  Writer.AddDeclRef(S->getConditionVariable(), Record);
  Writer.AddStmt(S->getInc());
  Writer.AddStmt(S->getBody());
  Writer.AddSourceLocation(S->getForLoc(), Record);
  Writer.AddSourceLocation(S->getLParenLoc(), Record);
  Writer.AddSourceLocation(S->getRParenLoc(), Record);
  Code = serialization::STMT_FOR;
}

void ASTStmtWriter::VisitGotoStmt(GotoStmt *S) {
  VisitStmt(S);
  Record.push_back(Writer.GetLabelID(S->getLabel()));
  Writer.AddSourceLocation(S->getGotoLoc(), Record);
  Writer.AddSourceLocation(S->getLabelLoc(), Record);
  Code = serialization::STMT_GOTO;
}

void ASTStmtWriter::VisitIndirectGotoStmt(IndirectGotoStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getGotoLoc(), Record);
  Writer.AddSourceLocation(S->getStarLoc(), Record);
  Writer.AddStmt(S->getTarget());
  Code = serialization::STMT_INDIRECT_GOTO;
}

void ASTStmtWriter::VisitContinueStmt(ContinueStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getContinueLoc(), Record);
  Code = serialization::STMT_CONTINUE;
}

void ASTStmtWriter::VisitBreakStmt(BreakStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getBreakLoc(), Record);
  Code = serialization::STMT_BREAK;
}

void ASTStmtWriter::VisitReturnStmt(ReturnStmt *S) {
  VisitStmt(S);
  Writer.AddStmt(S->getRetValue());
  Writer.AddSourceLocation(S->getReturnLoc(), Record);
  Writer.AddDeclRef(S->getNRVOCandidate(), Record);
  Code = serialization::STMT_RETURN;
}

void ASTStmtWriter::VisitDeclStmt(DeclStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getStartLoc(), Record);
  Writer.AddSourceLocation(S->getEndLoc(), Record);
  DeclGroupRef DG = S->getDeclGroup();
  for (DeclGroupRef::iterator D = DG.begin(), DEnd = DG.end(); D != DEnd; ++D)
    Writer.AddDeclRef(*D, Record);
  Code = serialization::STMT_DECL;
}

void ASTStmtWriter::VisitAsmStmt(AsmStmt *S) {
  VisitStmt(S);
  Record.push_back(S->getNumOutputs());
  Record.push_back(S->getNumInputs());
  Record.push_back(S->getNumClobbers());
  Writer.AddSourceLocation(S->getAsmLoc(), Record);
  Writer.AddSourceLocation(S->getRParenLoc(), Record);
  Record.push_back(S->isVolatile());
  Record.push_back(S->isSimple());
  Record.push_back(S->isMSAsm());
  Writer.AddStmt(S->getAsmString());

  // Outputs
  for (unsigned I = 0, N = S->getNumOutputs(); I != N; ++I) {      
    Writer.AddIdentifierRef(S->getOutputIdentifier(I), Record);
    Writer.AddStmt(S->getOutputConstraintLiteral(I));
    Writer.AddStmt(S->getOutputExpr(I));
  }

  // Inputs
  for (unsigned I = 0, N = S->getNumInputs(); I != N; ++I) {
    Writer.AddIdentifierRef(S->getInputIdentifier(I), Record);
    Writer.AddStmt(S->getInputConstraintLiteral(I));
    Writer.AddStmt(S->getInputExpr(I));
  }

  // Clobbers
  for (unsigned I = 0, N = S->getNumClobbers(); I != N; ++I)
    Writer.AddStmt(S->getClobber(I));

  Code = serialization::STMT_ASM;
}

void ASTStmtWriter::VisitExpr(Expr *E) {
  VisitStmt(E);
  Writer.AddTypeRef(E->getType(), Record);
  Record.push_back(E->isTypeDependent());
  Record.push_back(E->isValueDependent());
}

void ASTStmtWriter::VisitPredefinedExpr(PredefinedExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Record.push_back(E->getIdentType()); // FIXME: stable encoding
  Code = serialization::EXPR_PREDEFINED;
}

void ASTStmtWriter::VisitDeclRefExpr(DeclRefExpr *E) {
  VisitExpr(E);

  Record.push_back(E->hasQualifier());
  unsigned NumTemplateArgs = E->getNumTemplateArgs();
  assert((NumTemplateArgs != 0) == E->hasExplicitTemplateArgs() &&
         "Template args list with no args ?");
  Record.push_back(NumTemplateArgs);

  if (E->hasQualifier()) {
    Writer.AddNestedNameSpecifier(E->getQualifier(), Record);
    Writer.AddSourceRange(E->getQualifierRange(), Record);
  }

  if (NumTemplateArgs)
    AddExplicitTemplateArgumentList(E->getExplicitTemplateArgs());

  Writer.AddDeclRef(E->getDecl(), Record);
  // FIXME: write DeclarationNameLoc.
  Writer.AddSourceLocation(E->getLocation(), Record);
  Code = serialization::EXPR_DECL_REF;
}

void ASTStmtWriter::VisitIntegerLiteral(IntegerLiteral *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Writer.AddAPInt(E->getValue(), Record);
  Code = serialization::EXPR_INTEGER_LITERAL;
}

void ASTStmtWriter::VisitFloatingLiteral(FloatingLiteral *E) {
  VisitExpr(E);
  Writer.AddAPFloat(E->getValue(), Record);
  Record.push_back(E->isExact());
  Writer.AddSourceLocation(E->getLocation(), Record);
  Code = serialization::EXPR_FLOATING_LITERAL;
}

void ASTStmtWriter::VisitImaginaryLiteral(ImaginaryLiteral *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getSubExpr());
  Code = serialization::EXPR_IMAGINARY_LITERAL;
}

void ASTStmtWriter::VisitStringLiteral(StringLiteral *E) {
  VisitExpr(E);
  Record.push_back(E->getByteLength());
  Record.push_back(E->getNumConcatenated());
  Record.push_back(E->isWide());
  // FIXME: String data should be stored as a blob at the end of the
  // StringLiteral. However, we can't do so now because we have no
  // provision for coping with abbreviations when we're jumping around
  // the AST file during deserialization.
  Record.append(E->getString().begin(), E->getString().end());
  for (unsigned I = 0, N = E->getNumConcatenated(); I != N; ++I)
    Writer.AddSourceLocation(E->getStrTokenLoc(I), Record);
  Code = serialization::EXPR_STRING_LITERAL;
}

void ASTStmtWriter::VisitCharacterLiteral(CharacterLiteral *E) {
  VisitExpr(E);
  Record.push_back(E->getValue());
  Writer.AddSourceLocation(E->getLocation(), Record);
  Record.push_back(E->isWide());
  Code = serialization::EXPR_CHARACTER_LITERAL;
}

void ASTStmtWriter::VisitParenExpr(ParenExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLParen(), Record);
  Writer.AddSourceLocation(E->getRParen(), Record);
  Writer.AddStmt(E->getSubExpr());
  Code = serialization::EXPR_PAREN;
}

void ASTStmtWriter::VisitParenListExpr(ParenListExpr *E) {
  VisitExpr(E);
  Record.push_back(E->NumExprs);
  for (unsigned i=0; i != E->NumExprs; ++i)
    Writer.AddStmt(E->Exprs[i]);
  Writer.AddSourceLocation(E->LParenLoc, Record);
  Writer.AddSourceLocation(E->RParenLoc, Record);
  Code = serialization::EXPR_PAREN_LIST;
}

void ASTStmtWriter::VisitUnaryOperator(UnaryOperator *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getSubExpr());
  Record.push_back(E->getOpcode()); // FIXME: stable encoding
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Code = serialization::EXPR_UNARY_OPERATOR;
}

void ASTStmtWriter::VisitOffsetOfExpr(OffsetOfExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumComponents());
  Record.push_back(E->getNumExpressions());
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Writer.AddTypeSourceInfo(E->getTypeSourceInfo(), Record);
  for (unsigned I = 0, N = E->getNumComponents(); I != N; ++I) {
    const OffsetOfExpr::OffsetOfNode &ON = E->getComponent(I);
    Record.push_back(ON.getKind()); // FIXME: Stable encoding
    Writer.AddSourceLocation(ON.getRange().getBegin(), Record);
    Writer.AddSourceLocation(ON.getRange().getEnd(), Record);
    switch (ON.getKind()) {
    case OffsetOfExpr::OffsetOfNode::Array:
      Record.push_back(ON.getArrayExprIndex());
      break;
        
    case OffsetOfExpr::OffsetOfNode::Field:
      Writer.AddDeclRef(ON.getField(), Record);
      break;
        
    case OffsetOfExpr::OffsetOfNode::Identifier:
      Writer.AddIdentifierRef(ON.getFieldName(), Record);
      break;
        
    case OffsetOfExpr::OffsetOfNode::Base:
      Writer.AddCXXBaseSpecifier(*ON.getBase(), Record);
      break;
    }
  }
  for (unsigned I = 0, N = E->getNumExpressions(); I != N; ++I)
    Writer.AddStmt(E->getIndexExpr(I));
  Code = serialization::EXPR_OFFSETOF;
}

void ASTStmtWriter::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E) {
  VisitExpr(E);
  Record.push_back(E->isSizeOf());
  if (E->isArgumentType())
    Writer.AddTypeSourceInfo(E->getArgumentTypeInfo(), Record);
  else {
    Record.push_back(0);
    Writer.AddStmt(E->getArgumentExpr());
  }
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_SIZEOF_ALIGN_OF;
}

void ASTStmtWriter::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getLHS());
  Writer.AddStmt(E->getRHS());
  Writer.AddSourceLocation(E->getRBracketLoc(), Record);
  Code = serialization::EXPR_ARRAY_SUBSCRIPT;
}

void ASTStmtWriter::VisitCallExpr(CallExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumArgs());
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Writer.AddStmt(E->getCallee());
  for (CallExpr::arg_iterator Arg = E->arg_begin(), ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg)
    Writer.AddStmt(*Arg);
  Code = serialization::EXPR_CALL;
}

void ASTStmtWriter::VisitMemberExpr(MemberExpr *E) {
  // Don't call VisitExpr, we'll write everything here.

  Record.push_back(E->hasQualifier());
  if (E->hasQualifier()) {
    Writer.AddNestedNameSpecifier(E->getQualifier(), Record);
    Writer.AddSourceRange(E->getQualifierRange(), Record);
  }

  unsigned NumTemplateArgs = E->getNumTemplateArgs();
  assert((NumTemplateArgs != 0) == E->hasExplicitTemplateArgs() &&
         "Template args list with no args ?");
  Record.push_back(NumTemplateArgs);
  if (NumTemplateArgs) {
    Writer.AddSourceLocation(E->getLAngleLoc(), Record);
    Writer.AddSourceLocation(E->getRAngleLoc(), Record);
    for (unsigned i=0; i != NumTemplateArgs; ++i)
      Writer.AddTemplateArgumentLoc(E->getTemplateArgs()[i], Record);
  }
  
  DeclAccessPair FoundDecl = E->getFoundDecl();
  Writer.AddDeclRef(FoundDecl.getDecl(), Record);
  Record.push_back(FoundDecl.getAccess());

  Writer.AddTypeRef(E->getType(), Record);
  Writer.AddStmt(E->getBase());
  Writer.AddDeclRef(E->getMemberDecl(), Record);
  // FIXME: write DeclarationNameLoc.
  Writer.AddSourceLocation(E->getMemberLoc(), Record);
  Record.push_back(E->isArrow());
  Code = serialization::EXPR_MEMBER;
}

void ASTStmtWriter::VisitObjCIsaExpr(ObjCIsaExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getBase());
  Writer.AddSourceLocation(E->getIsaMemberLoc(), Record);
  Record.push_back(E->isArrow());
  Code = serialization::EXPR_OBJC_ISA;
}

void ASTStmtWriter::VisitCastExpr(CastExpr *E) {
  VisitExpr(E);
  Record.push_back(E->path_size());
  Writer.AddStmt(E->getSubExpr());
  Record.push_back(E->getCastKind()); // FIXME: stable encoding

  for (CastExpr::path_iterator
         PI = E->path_begin(), PE = E->path_end(); PI != PE; ++PI)
    Writer.AddCXXBaseSpecifier(**PI, Record);
}

void ASTStmtWriter::VisitBinaryOperator(BinaryOperator *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getLHS());
  Writer.AddStmt(E->getRHS());
  Record.push_back(E->getOpcode()); // FIXME: stable encoding
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Code = serialization::EXPR_BINARY_OPERATOR;
}

void ASTStmtWriter::VisitCompoundAssignOperator(CompoundAssignOperator *E) {
  VisitBinaryOperator(E);
  Writer.AddTypeRef(E->getComputationLHSType(), Record);
  Writer.AddTypeRef(E->getComputationResultType(), Record);
  Code = serialization::EXPR_COMPOUND_ASSIGN_OPERATOR;
}

void ASTStmtWriter::VisitConditionalOperator(ConditionalOperator *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getCond());
  Writer.AddStmt(E->getLHS());
  Writer.AddStmt(E->getRHS());
  Writer.AddSourceLocation(E->getQuestionLoc(), Record);
  Writer.AddSourceLocation(E->getColonLoc(), Record);
  Code = serialization::EXPR_CONDITIONAL_OPERATOR;
}

void ASTStmtWriter::VisitImplicitCastExpr(ImplicitCastExpr *E) {
  VisitCastExpr(E);
  Record.push_back(E->getValueKind());
  Code = serialization::EXPR_IMPLICIT_CAST;
}

void ASTStmtWriter::VisitExplicitCastExpr(ExplicitCastExpr *E) {
  VisitCastExpr(E);
  Writer.AddTypeSourceInfo(E->getTypeInfoAsWritten(), Record);
}

void ASTStmtWriter::VisitCStyleCastExpr(CStyleCastExpr *E) {
  VisitExplicitCastExpr(E);
  Writer.AddSourceLocation(E->getLParenLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_CSTYLE_CAST;
}

void ASTStmtWriter::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLParenLoc(), Record);
  Writer.AddTypeSourceInfo(E->getTypeSourceInfo(), Record);
  Writer.AddStmt(E->getInitializer());
  Record.push_back(E->isFileScope());
  Code = serialization::EXPR_COMPOUND_LITERAL;
}

void ASTStmtWriter::VisitExtVectorElementExpr(ExtVectorElementExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getBase());
  Writer.AddIdentifierRef(&E->getAccessor(), Record);
  Writer.AddSourceLocation(E->getAccessorLoc(), Record);
  Code = serialization::EXPR_EXT_VECTOR_ELEMENT;
}

void ASTStmtWriter::VisitInitListExpr(InitListExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumInits());
  for (unsigned I = 0, N = E->getNumInits(); I != N; ++I)
    Writer.AddStmt(E->getInit(I));
  Writer.AddStmt(E->getSyntacticForm());
  Writer.AddSourceLocation(E->getLBraceLoc(), Record);
  Writer.AddSourceLocation(E->getRBraceLoc(), Record);
  Writer.AddDeclRef(E->getInitializedFieldInUnion(), Record);
  Record.push_back(E->hadArrayRangeDesignator());
  Code = serialization::EXPR_INIT_LIST;
}

void ASTStmtWriter::VisitDesignatedInitExpr(DesignatedInitExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumSubExprs());
  for (unsigned I = 0, N = E->getNumSubExprs(); I != N; ++I)
    Writer.AddStmt(E->getSubExpr(I));
  Writer.AddSourceLocation(E->getEqualOrColonLoc(), Record);
  Record.push_back(E->usesGNUSyntax());
  for (DesignatedInitExpr::designators_iterator D = E->designators_begin(),
                                             DEnd = E->designators_end();
       D != DEnd; ++D) {
    if (D->isFieldDesignator()) {
      if (FieldDecl *Field = D->getField()) {
        Record.push_back(serialization::DESIG_FIELD_DECL);
        Writer.AddDeclRef(Field, Record);
      } else {
        Record.push_back(serialization::DESIG_FIELD_NAME);
        Writer.AddIdentifierRef(D->getFieldName(), Record);
      }
      Writer.AddSourceLocation(D->getDotLoc(), Record);
      Writer.AddSourceLocation(D->getFieldLoc(), Record);
    } else if (D->isArrayDesignator()) {
      Record.push_back(serialization::DESIG_ARRAY);
      Record.push_back(D->getFirstExprIndex());
      Writer.AddSourceLocation(D->getLBracketLoc(), Record);
      Writer.AddSourceLocation(D->getRBracketLoc(), Record);
    } else {
      assert(D->isArrayRangeDesignator() && "Unknown designator");
      Record.push_back(serialization::DESIG_ARRAY_RANGE);
      Record.push_back(D->getFirstExprIndex());
      Writer.AddSourceLocation(D->getLBracketLoc(), Record);
      Writer.AddSourceLocation(D->getEllipsisLoc(), Record);
      Writer.AddSourceLocation(D->getRBracketLoc(), Record);
    }
  }
  Code = serialization::EXPR_DESIGNATED_INIT;
}

void ASTStmtWriter::VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
  VisitExpr(E);
  Code = serialization::EXPR_IMPLICIT_VALUE_INIT;
}

void ASTStmtWriter::VisitVAArgExpr(VAArgExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getSubExpr());
  Writer.AddTypeSourceInfo(E->getWrittenTypeInfo(), Record);
  Writer.AddSourceLocation(E->getBuiltinLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_VA_ARG;
}

void ASTStmtWriter::VisitAddrLabelExpr(AddrLabelExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getAmpAmpLoc(), Record);
  Writer.AddSourceLocation(E->getLabelLoc(), Record);
  Record.push_back(Writer.GetLabelID(E->getLabel()));
  Code = serialization::EXPR_ADDR_LABEL;
}

void ASTStmtWriter::VisitStmtExpr(StmtExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getSubStmt());
  Writer.AddSourceLocation(E->getLParenLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_STMT;
}

void ASTStmtWriter::VisitTypesCompatibleExpr(TypesCompatibleExpr *E) {
  VisitExpr(E);
  Writer.AddTypeSourceInfo(E->getArgTInfo1(), Record);
  Writer.AddTypeSourceInfo(E->getArgTInfo2(), Record);
  Writer.AddSourceLocation(E->getBuiltinLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_TYPES_COMPATIBLE;
}

void ASTStmtWriter::VisitChooseExpr(ChooseExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getCond());
  Writer.AddStmt(E->getLHS());
  Writer.AddStmt(E->getRHS());
  Writer.AddSourceLocation(E->getBuiltinLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_CHOOSE;
}

void ASTStmtWriter::VisitGNUNullExpr(GNUNullExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getTokenLocation(), Record);
  Code = serialization::EXPR_GNU_NULL;
}

void ASTStmtWriter::VisitShuffleVectorExpr(ShuffleVectorExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumSubExprs());
  for (unsigned I = 0, N = E->getNumSubExprs(); I != N; ++I)
    Writer.AddStmt(E->getExpr(I));
  Writer.AddSourceLocation(E->getBuiltinLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_SHUFFLE_VECTOR;
}

void ASTStmtWriter::VisitBlockExpr(BlockExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getBlockDecl(), Record);
  Record.push_back(E->hasBlockDeclRefExprs());
  Code = serialization::EXPR_BLOCK;
}

void ASTStmtWriter::VisitBlockDeclRefExpr(BlockDeclRefExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getDecl(), Record);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Record.push_back(E->isByRef());
  Record.push_back(E->isConstQualAdded());
  Writer.AddStmt(E->getCopyConstructorExpr());
  Code = serialization::EXPR_BLOCK_DECL_REF;
}

//===----------------------------------------------------------------------===//
// Objective-C Expressions and Statements.
//===----------------------------------------------------------------------===//

void ASTStmtWriter::VisitObjCStringLiteral(ObjCStringLiteral *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getString());
  Writer.AddSourceLocation(E->getAtLoc(), Record);
  Code = serialization::EXPR_OBJC_STRING_LITERAL;
}

void ASTStmtWriter::VisitObjCEncodeExpr(ObjCEncodeExpr *E) {
  VisitExpr(E);
  Writer.AddTypeSourceInfo(E->getEncodedTypeSourceInfo(), Record);
  Writer.AddSourceLocation(E->getAtLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_OBJC_ENCODE;
}

void ASTStmtWriter::VisitObjCSelectorExpr(ObjCSelectorExpr *E) {
  VisitExpr(E);
  Writer.AddSelectorRef(E->getSelector(), Record);
  Writer.AddSourceLocation(E->getAtLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_OBJC_SELECTOR_EXPR;
}

void ASTStmtWriter::VisitObjCProtocolExpr(ObjCProtocolExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getProtocol(), Record);
  Writer.AddSourceLocation(E->getAtLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_OBJC_PROTOCOL_EXPR;
}

void ASTStmtWriter::VisitObjCIvarRefExpr(ObjCIvarRefExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getDecl(), Record);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Writer.AddStmt(E->getBase());
  Record.push_back(E->isArrow());
  Record.push_back(E->isFreeIvar());
  Code = serialization::EXPR_OBJC_IVAR_REF_EXPR;
}

void ASTStmtWriter::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getProperty(), Record);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Writer.AddStmt(E->getBase());
  Code = serialization::EXPR_OBJC_PROPERTY_REF_EXPR;
}

void ASTStmtWriter::VisitObjCImplicitSetterGetterRefExpr(
                                  ObjCImplicitSetterGetterRefExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getGetterMethod(), Record);
  Writer.AddDeclRef(E->getSetterMethod(), Record);

  // NOTE: InterfaceDecl and Base are mutually exclusive.
  Writer.AddDeclRef(E->getInterfaceDecl(), Record);
  Writer.AddStmt(E->getBase());
  Writer.AddSourceLocation(E->getLocation(), Record);
  Writer.AddSourceLocation(E->getClassLoc(), Record);
  Code = serialization::EXPR_OBJC_KVC_REF_EXPR;
}

void ASTStmtWriter::VisitObjCMessageExpr(ObjCMessageExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumArgs());
  Record.push_back((unsigned)E->getReceiverKind()); // FIXME: stable encoding
  switch (E->getReceiverKind()) {
  case ObjCMessageExpr::Instance:
    Writer.AddStmt(E->getInstanceReceiver());
    break;

  case ObjCMessageExpr::Class:
    Writer.AddTypeSourceInfo(E->getClassReceiverTypeInfo(), Record);
    break;

  case ObjCMessageExpr::SuperClass:
  case ObjCMessageExpr::SuperInstance:
    Writer.AddTypeRef(E->getSuperType(), Record);
    Writer.AddSourceLocation(E->getSuperLoc(), Record);
    break;
  }

  if (E->getMethodDecl()) {
    Record.push_back(1);
    Writer.AddDeclRef(E->getMethodDecl(), Record);
  } else {
    Record.push_back(0);
    Writer.AddSelectorRef(E->getSelector(), Record);    
  }
    
  Writer.AddSourceLocation(E->getLeftLoc(), Record);
  Writer.AddSourceLocation(E->getRightLoc(), Record);

  for (CallExpr::arg_iterator Arg = E->arg_begin(), ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg)
    Writer.AddStmt(*Arg);
  Code = serialization::EXPR_OBJC_MESSAGE_EXPR;
}

void ASTStmtWriter::VisitObjCSuperExpr(ObjCSuperExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLoc(), Record);
  Code = serialization::EXPR_OBJC_SUPER_EXPR;
}

void ASTStmtWriter::VisitObjCForCollectionStmt(ObjCForCollectionStmt *S) {
  VisitStmt(S);
  Writer.AddStmt(S->getElement());
  Writer.AddStmt(S->getCollection());
  Writer.AddStmt(S->getBody());
  Writer.AddSourceLocation(S->getForLoc(), Record);
  Writer.AddSourceLocation(S->getRParenLoc(), Record);
  Code = serialization::STMT_OBJC_FOR_COLLECTION;
}

void ASTStmtWriter::VisitObjCAtCatchStmt(ObjCAtCatchStmt *S) {
  Writer.AddStmt(S->getCatchBody());
  Writer.AddDeclRef(S->getCatchParamDecl(), Record);
  Writer.AddSourceLocation(S->getAtCatchLoc(), Record);
  Writer.AddSourceLocation(S->getRParenLoc(), Record);
  Code = serialization::STMT_OBJC_CATCH;
}

void ASTStmtWriter::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *S) {
  Writer.AddStmt(S->getFinallyBody());
  Writer.AddSourceLocation(S->getAtFinallyLoc(), Record);
  Code = serialization::STMT_OBJC_FINALLY;
}

void ASTStmtWriter::VisitObjCAtTryStmt(ObjCAtTryStmt *S) {
  Record.push_back(S->getNumCatchStmts());
  Record.push_back(S->getFinallyStmt() != 0);
  Writer.AddStmt(S->getTryBody());
  for (unsigned I = 0, N = S->getNumCatchStmts(); I != N; ++I)
    Writer.AddStmt(S->getCatchStmt(I));
  if (S->getFinallyStmt())
    Writer.AddStmt(S->getFinallyStmt());
  Writer.AddSourceLocation(S->getAtTryLoc(), Record);
  Code = serialization::STMT_OBJC_AT_TRY;
}

void ASTStmtWriter::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *S) {
  Writer.AddStmt(S->getSynchExpr());
  Writer.AddStmt(S->getSynchBody());
  Writer.AddSourceLocation(S->getAtSynchronizedLoc(), Record);
  Code = serialization::STMT_OBJC_AT_SYNCHRONIZED;
}

void ASTStmtWriter::VisitObjCAtThrowStmt(ObjCAtThrowStmt *S) {
  Writer.AddStmt(S->getThrowExpr());
  Writer.AddSourceLocation(S->getThrowLoc(), Record);
  Code = serialization::STMT_OBJC_AT_THROW;
}

//===----------------------------------------------------------------------===//
// C++ Expressions and Statements.
//===----------------------------------------------------------------------===//

void ASTStmtWriter::VisitCXXCatchStmt(CXXCatchStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getCatchLoc(), Record);
  Writer.AddDeclRef(S->getExceptionDecl(), Record);
  Writer.AddStmt(S->getHandlerBlock());
  Code = serialization::STMT_CXX_CATCH;
}

void ASTStmtWriter::VisitCXXTryStmt(CXXTryStmt *S) {
  VisitStmt(S);
  Record.push_back(S->getNumHandlers());
  Writer.AddSourceLocation(S->getTryLoc(), Record);
  Writer.AddStmt(S->getTryBlock());
  for (unsigned i = 0, e = S->getNumHandlers(); i != e; ++i)
    Writer.AddStmt(S->getHandler(i));
  Code = serialization::STMT_CXX_TRY;
}

void ASTStmtWriter::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
  VisitCallExpr(E);
  Record.push_back(E->getOperator());
  Code = serialization::EXPR_CXX_OPERATOR_CALL;
}

void ASTStmtWriter::VisitCXXMemberCallExpr(CXXMemberCallExpr *E) {
  VisitCallExpr(E);
  Code = serialization::EXPR_CXX_MEMBER_CALL;
}

void ASTStmtWriter::VisitCXXConstructExpr(CXXConstructExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumArgs());
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
    Writer.AddStmt(E->getArg(I));
  Writer.AddDeclRef(E->getConstructor(), Record);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Record.push_back(E->isElidable());
  Record.push_back(E->requiresZeroInitialization());
  Record.push_back(E->getConstructionKind()); // FIXME: stable encoding
  Code = serialization::EXPR_CXX_CONSTRUCT;
}

void ASTStmtWriter::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *E) {
  VisitCXXConstructExpr(E);
  Writer.AddSourceLocation(E->getTypeBeginLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_CXX_TEMPORARY_OBJECT;
}

void ASTStmtWriter::VisitCXXNamedCastExpr(CXXNamedCastExpr *E) {
  VisitExplicitCastExpr(E);
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
}

void ASTStmtWriter::VisitCXXStaticCastExpr(CXXStaticCastExpr *E) {
  VisitCXXNamedCastExpr(E);
  Code = serialization::EXPR_CXX_STATIC_CAST;
}

void ASTStmtWriter::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *E) {
  VisitCXXNamedCastExpr(E);
  Code = serialization::EXPR_CXX_DYNAMIC_CAST;
}

void ASTStmtWriter::VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *E) {
  VisitCXXNamedCastExpr(E);
  Code = serialization::EXPR_CXX_REINTERPRET_CAST;
}

void ASTStmtWriter::VisitCXXConstCastExpr(CXXConstCastExpr *E) {
  VisitCXXNamedCastExpr(E);
  Code = serialization::EXPR_CXX_CONST_CAST;
}

void ASTStmtWriter::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *E) {
  VisitExplicitCastExpr(E);
  Writer.AddSourceLocation(E->getTypeBeginLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_CXX_FUNCTIONAL_CAST;
}

void ASTStmtWriter::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getValue());
  Writer.AddSourceLocation(E->getLocation(), Record);
  Code = serialization::EXPR_CXX_BOOL_LITERAL;
}

void ASTStmtWriter::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Code = serialization::EXPR_CXX_NULL_PTR_LITERAL;
}

void ASTStmtWriter::VisitCXXTypeidExpr(CXXTypeidExpr *E) {
  VisitExpr(E);
  Writer.AddSourceRange(E->getSourceRange(), Record);
  if (E->isTypeOperand()) {
    Writer.AddTypeSourceInfo(E->getTypeOperandSourceInfo(), Record);
    Code = serialization::EXPR_CXX_TYPEID_TYPE;
  } else {
    Writer.AddStmt(E->getExprOperand());
    Code = serialization::EXPR_CXX_TYPEID_EXPR;
  }
}

void ASTStmtWriter::VisitCXXThisExpr(CXXThisExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Record.push_back(E->isImplicit());
  Code = serialization::EXPR_CXX_THIS;
}

void ASTStmtWriter::VisitCXXThrowExpr(CXXThrowExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getThrowLoc(), Record);
  Writer.AddStmt(E->getSubExpr());
  Code = serialization::EXPR_CXX_THROW;
}

void ASTStmtWriter::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E) {
  VisitExpr(E);

  bool HasOtherExprStored = E->Param.getInt();
  // Store these first, the reader reads them before creation.
  Record.push_back(HasOtherExprStored);
  if (HasOtherExprStored)
    Writer.AddStmt(E->getExpr());
  Writer.AddDeclRef(E->getParam(), Record);
  Writer.AddSourceLocation(E->getUsedLocation(), Record);

  Code = serialization::EXPR_CXX_DEFAULT_ARG;
}

void ASTStmtWriter::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
  VisitExpr(E);
  Writer.AddCXXTemporary(E->getTemporary(), Record);
  Writer.AddStmt(E->getSubExpr());
  Code = serialization::EXPR_CXX_BIND_TEMPORARY;
}

void ASTStmtWriter::VisitCXXBindReferenceExpr(CXXBindReferenceExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getSubExpr());
  Record.push_back(E->extendsLifetime());
  Record.push_back(E->requiresTemporaryCopy());
  Code = serialization::EXPR_CXX_BIND_REFERENCE;
}

void ASTStmtWriter::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getTypeBeginLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_CXX_SCALAR_VALUE_INIT;
}

void ASTStmtWriter::VisitCXXNewExpr(CXXNewExpr *E) {
  VisitExpr(E);
  Record.push_back(E->isGlobalNew());
  Record.push_back(E->hasInitializer());
  Record.push_back(E->isArray());
  Record.push_back(E->getNumPlacementArgs());
  Record.push_back(E->getNumConstructorArgs());
  Writer.AddDeclRef(E->getOperatorNew(), Record);
  Writer.AddDeclRef(E->getOperatorDelete(), Record);
  Writer.AddDeclRef(E->getConstructor(), Record);
  Writer.AddSourceRange(E->getTypeIdParens(), Record);
  Writer.AddSourceLocation(E->getStartLoc(), Record);
  Writer.AddSourceLocation(E->getEndLoc(), Record);
  for (CXXNewExpr::arg_iterator I = E->raw_arg_begin(), e = E->raw_arg_end();
       I != e; ++I)
    Writer.AddStmt(*I);
  
  Code = serialization::EXPR_CXX_NEW;
}

void ASTStmtWriter::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
  VisitExpr(E);
  Record.push_back(E->isGlobalDelete());
  Record.push_back(E->isArrayForm());
  Writer.AddDeclRef(E->getOperatorDelete(), Record);
  Writer.AddStmt(E->getArgument());
  Writer.AddSourceLocation(E->getSourceRange().getBegin(), Record);
  
  Code = serialization::EXPR_CXX_DELETE;
}

void ASTStmtWriter::VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E) {
  VisitExpr(E);

  Writer.AddStmt(E->getBase());
  Record.push_back(E->isArrow());
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Writer.AddNestedNameSpecifier(E->getQualifier(), Record);
  Writer.AddSourceRange(E->getQualifierRange(), Record);
  Writer.AddTypeSourceInfo(E->getScopeTypeInfo(), Record);
  Writer.AddSourceLocation(E->getColonColonLoc(), Record);
  Writer.AddSourceLocation(E->getTildeLoc(), Record);

  // PseudoDestructorTypeStorage.
  Writer.AddIdentifierRef(E->getDestroyedTypeIdentifier(), Record);
  if (E->getDestroyedTypeIdentifier())
    Writer.AddSourceLocation(E->getDestroyedTypeLoc(), Record);
  else
    Writer.AddTypeSourceInfo(E->getDestroyedTypeInfo(), Record);

  Code = serialization::EXPR_CXX_PSEUDO_DESTRUCTOR;
}

void ASTStmtWriter::VisitCXXExprWithTemporaries(CXXExprWithTemporaries *E) {
  VisitExpr(E);
  Record.push_back(E->getNumTemporaries());
  for (unsigned i = 0, e = E->getNumTemporaries(); i != e; ++i)
    Writer.AddCXXTemporary(E->getTemporary(i), Record);
  
  Writer.AddStmt(E->getSubExpr());
  Code = serialization::EXPR_CXX_EXPR_WITH_TEMPORARIES;
}

void
ASTStmtWriter::VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *E){
  VisitExpr(E);
  
  // Don't emit anything here, NumTemplateArgs must be emitted first.

  if (E->hasExplicitTemplateArgs()) {
    const ExplicitTemplateArgumentList &Args = E->getExplicitTemplateArgs();
    assert(Args.NumTemplateArgs &&
           "Num of template args was zero! AST reading will mess up!");
    Record.push_back(Args.NumTemplateArgs);
    AddExplicitTemplateArgumentList(Args);
  } else {
    Record.push_back(0);
  }
  
  if (!E->isImplicitAccess())
    Writer.AddStmt(E->getBase());
  else
    Writer.AddStmt(0);
  Writer.AddTypeRef(E->getBaseType(), Record);
  Record.push_back(E->isArrow());
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Writer.AddNestedNameSpecifier(E->getQualifier(), Record);
  Writer.AddSourceRange(E->getQualifierRange(), Record);
  Writer.AddDeclRef(E->getFirstQualifierFoundInScope(), Record);
  // FIXME: write whole DeclarationNameInfo.
  Writer.AddDeclarationName(E->getMember(), Record);
  Writer.AddSourceLocation(E->getMemberLoc(), Record);
  Code = serialization::EXPR_CXX_DEPENDENT_SCOPE_MEMBER;
}

void
ASTStmtWriter::VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *E) {
  VisitExpr(E);
  
  // Don't emit anything here, NumTemplateArgs must be emitted first.

  if (E->hasExplicitTemplateArgs()) {
    const ExplicitTemplateArgumentList &Args = E->getExplicitTemplateArgs();
    assert(Args.NumTemplateArgs &&
           "Num of template args was zero! AST reading will mess up!");
    Record.push_back(Args.NumTemplateArgs);
    AddExplicitTemplateArgumentList(Args);
  } else {
    Record.push_back(0);
  }

  // FIXME: write whole DeclarationNameInfo.
  Writer.AddDeclarationName(E->getDeclName(), Record);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Writer.AddSourceRange(E->getQualifierRange(), Record);
  Writer.AddNestedNameSpecifier(E->getQualifier(), Record);
  Code = serialization::EXPR_CXX_DEPENDENT_SCOPE_DECL_REF;
}

void
ASTStmtWriter::VisitCXXUnresolvedConstructExpr(CXXUnresolvedConstructExpr *E) {
  VisitExpr(E);
  Record.push_back(E->arg_size());
  for (CXXUnresolvedConstructExpr::arg_iterator
         ArgI = E->arg_begin(), ArgE = E->arg_end(); ArgI != ArgE; ++ArgI)
    Writer.AddStmt(*ArgI);
  Writer.AddSourceLocation(E->getTypeBeginLoc(), Record);
  Writer.AddTypeRef(E->getTypeAsWritten(), Record);
  Writer.AddSourceLocation(E->getLParenLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_CXX_UNRESOLVED_CONSTRUCT;
}

void ASTStmtWriter::VisitOverloadExpr(OverloadExpr *E) {
  VisitExpr(E);
  
  // Don't emit anything here, NumTemplateArgs must be emitted first.

  if (E->hasExplicitTemplateArgs()) {
    const ExplicitTemplateArgumentList &Args = E->getExplicitTemplateArgs();
    assert(Args.NumTemplateArgs &&
           "Num of template args was zero! AST reading will mess up!");
    Record.push_back(Args.NumTemplateArgs);
    AddExplicitTemplateArgumentList(Args);
  } else {
    Record.push_back(0);
  }

  Record.push_back(E->getNumDecls());
  for (OverloadExpr::decls_iterator
         OvI = E->decls_begin(), OvE = E->decls_end(); OvI != OvE; ++OvI) {
    Writer.AddDeclRef(OvI.getDecl(), Record);
    Record.push_back(OvI.getAccess());
  }

  // FIXME: write whole DeclarationNameInfo.
  Writer.AddDeclarationName(E->getName(), Record);
  Writer.AddNestedNameSpecifier(E->getQualifier(), Record);
  Writer.AddSourceRange(E->getQualifierRange(), Record);
  Writer.AddSourceLocation(E->getNameLoc(), Record);
}

void ASTStmtWriter::VisitUnresolvedMemberExpr(UnresolvedMemberExpr *E) {
  VisitOverloadExpr(E);
  Record.push_back(E->isArrow());
  Record.push_back(E->hasUnresolvedUsing());
  Writer.AddStmt(!E->isImplicitAccess() ? E->getBase() : 0);
  Writer.AddTypeRef(E->getBaseType(), Record);
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Code = serialization::EXPR_CXX_UNRESOLVED_MEMBER;
}

void ASTStmtWriter::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *E) {
  VisitOverloadExpr(E);
  Record.push_back(E->requiresADL());
  Record.push_back(E->isOverloaded());
  Writer.AddDeclRef(E->getNamingClass(), Record);
  Code = serialization::EXPR_CXX_UNRESOLVED_LOOKUP;
}

void ASTStmtWriter::VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getTrait());
  Writer.AddSourceRange(E->getSourceRange(), Record);
  Writer.AddTypeRef(E->getQueriedType(), Record);
  Code = serialization::EXPR_CXX_UNARY_TYPE_TRAIT;
}

//===----------------------------------------------------------------------===//
// ASTWriter Implementation
//===----------------------------------------------------------------------===//

unsigned ASTWriter::RecordSwitchCaseID(SwitchCase *S) {
  assert(SwitchCaseIDs.find(S) == SwitchCaseIDs.end() &&
         "SwitchCase recorded twice");
  unsigned NextID = SwitchCaseIDs.size();
  SwitchCaseIDs[S] = NextID;
  return NextID;
}

unsigned ASTWriter::getSwitchCaseID(SwitchCase *S) {
  assert(SwitchCaseIDs.find(S) != SwitchCaseIDs.end() &&
         "SwitchCase hasn't been seen yet");
  return SwitchCaseIDs[S];
}

/// \brief Retrieve the ID for the given label statement, which may
/// or may not have been emitted yet.
unsigned ASTWriter::GetLabelID(LabelStmt *S) {
  std::map<LabelStmt *, unsigned>::iterator Pos = LabelIDs.find(S);
  if (Pos != LabelIDs.end())
    return Pos->second;

  unsigned NextID = LabelIDs.size();
  LabelIDs[S] = NextID;
  return NextID;
}

/// \brief Write the given substatement or subexpression to the
/// bitstream.
void ASTWriter::WriteSubStmt(Stmt *S) {
  RecordData Record;
  ASTStmtWriter Writer(*this, Record);
  ++NumStatements;
  
  if (!S) {
    Stream.EmitRecord(serialization::STMT_NULL_PTR, Record);
    return;
  }

  // Redirect ASTWriter::AddStmt to collect sub stmts.
  llvm::SmallVector<Stmt *, 16> SubStmts;
  CollectedStmts = &SubStmts;

  Writer.Code = serialization::STMT_NULL_PTR;
  Writer.Visit(S);
  
#ifndef NDEBUG
  if (Writer.Code == serialization::STMT_NULL_PTR) {
    SourceManager &SrcMgr
      = DeclIDs.begin()->first->getASTContext().getSourceManager();
    S->dump(SrcMgr);
    assert(0 && "Unhandled sub statement writing AST file");
  }
#endif

  // Revert ASTWriter::AddStmt.
  CollectedStmts = &StmtsToEmit;

  // Write the sub stmts in reverse order, last to first. When reading them back
  // we will read them in correct order by "pop"ing them from the Stmts stack.
  // This simplifies reading and allows to store a variable number of sub stmts
  // without knowing it in advance.
  while (!SubStmts.empty())
    WriteSubStmt(SubStmts.pop_back_val());
  
  Stream.EmitRecord(Writer.Code, Record);
}

/// \brief Flush all of the statements that have been added to the
/// queue via AddStmt().
void ASTWriter::FlushStmts() {
  RecordData Record;

  for (unsigned I = 0, N = StmtsToEmit.size(); I != N; ++I) {
    WriteSubStmt(StmtsToEmit[I]);
    
    assert(N == StmtsToEmit.size() &&
           "Substatement writen via AddStmt rather than WriteSubStmt!");

    // Note that we are at the end of a full expression. Any
    // expression records that follow this one are part of a different
    // expression.
    Stream.EmitRecord(serialization::STMT_STOP, Record);
  }

  StmtsToEmit.clear();
}
