//===--- PCHWriterStmt.cpp - Statement and Expression Serialization -------===//
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

#include "clang/Frontend/PCHWriter.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/Bitcode/BitstreamWriter.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// Statement/expression serialization
//===----------------------------------------------------------------------===//

namespace {
  class PCHStmtWriter : public StmtVisitor<PCHStmtWriter, void> {
    PCHWriter &Writer;
    PCHWriter::RecordData &Record;

  public:
    pch::StmtCode Code;

    PCHStmtWriter(PCHWriter &Writer, PCHWriter::RecordData &Record)
      : Writer(Writer), Record(Record) { }

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
    void VisitUnaryOperator(UnaryOperator *E);
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
    void VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E);
    void VisitCXXConstructExpr(CXXConstructExpr *E);
  };
}

void PCHStmtWriter::VisitStmt(Stmt *S) {
}

void PCHStmtWriter::VisitNullStmt(NullStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getSemiLoc(), Record);
  Code = pch::STMT_NULL;
}

void PCHStmtWriter::VisitCompoundStmt(CompoundStmt *S) {
  VisitStmt(S);
  Record.push_back(S->size());
  for (CompoundStmt::body_iterator CS = S->body_begin(), CSEnd = S->body_end();
       CS != CSEnd; ++CS)
    Writer.WriteSubStmt(*CS);
  Writer.AddSourceLocation(S->getLBracLoc(), Record);
  Writer.AddSourceLocation(S->getRBracLoc(), Record);
  Code = pch::STMT_COMPOUND;
}

void PCHStmtWriter::VisitSwitchCase(SwitchCase *S) {
  VisitStmt(S);
  Record.push_back(Writer.RecordSwitchCaseID(S));
}

void PCHStmtWriter::VisitCaseStmt(CaseStmt *S) {
  VisitSwitchCase(S);
  Writer.WriteSubStmt(S->getLHS());
  Writer.WriteSubStmt(S->getRHS());
  Writer.WriteSubStmt(S->getSubStmt());
  Writer.AddSourceLocation(S->getCaseLoc(), Record);
  Writer.AddSourceLocation(S->getEllipsisLoc(), Record);
  Writer.AddSourceLocation(S->getColonLoc(), Record);
  Code = pch::STMT_CASE;
}

void PCHStmtWriter::VisitDefaultStmt(DefaultStmt *S) {
  VisitSwitchCase(S);
  Writer.WriteSubStmt(S->getSubStmt());
  Writer.AddSourceLocation(S->getDefaultLoc(), Record);
  Writer.AddSourceLocation(S->getColonLoc(), Record);
  Code = pch::STMT_DEFAULT;
}

void PCHStmtWriter::VisitLabelStmt(LabelStmt *S) {
  VisitStmt(S);
  Writer.AddIdentifierRef(S->getID(), Record);
  Writer.WriteSubStmt(S->getSubStmt());
  Writer.AddSourceLocation(S->getIdentLoc(), Record);
  Record.push_back(Writer.GetLabelID(S));
  Code = pch::STMT_LABEL;
}

void PCHStmtWriter::VisitIfStmt(IfStmt *S) {
  VisitStmt(S);
  Writer.AddDeclRef(S->getConditionVariable(), Record);
  Writer.WriteSubStmt(S->getCond());
  Writer.WriteSubStmt(S->getThen());
  Writer.WriteSubStmt(S->getElse());
  Writer.AddSourceLocation(S->getIfLoc(), Record);
  Writer.AddSourceLocation(S->getElseLoc(), Record);
  Code = pch::STMT_IF;
}

void PCHStmtWriter::VisitSwitchStmt(SwitchStmt *S) {
  VisitStmt(S);
  Writer.AddDeclRef(S->getConditionVariable(), Record);
  Writer.WriteSubStmt(S->getCond());
  Writer.WriteSubStmt(S->getBody());
  Writer.AddSourceLocation(S->getSwitchLoc(), Record);
  for (SwitchCase *SC = S->getSwitchCaseList(); SC;
       SC = SC->getNextSwitchCase())
    Record.push_back(Writer.getSwitchCaseID(SC));
  Code = pch::STMT_SWITCH;
}

void PCHStmtWriter::VisitWhileStmt(WhileStmt *S) {
  VisitStmt(S);
  Writer.AddDeclRef(S->getConditionVariable(), Record);
  Writer.WriteSubStmt(S->getCond());
  Writer.WriteSubStmt(S->getBody());
  Writer.AddSourceLocation(S->getWhileLoc(), Record);
  Code = pch::STMT_WHILE;
}

void PCHStmtWriter::VisitDoStmt(DoStmt *S) {
  VisitStmt(S);
  Writer.WriteSubStmt(S->getCond());
  Writer.WriteSubStmt(S->getBody());
  Writer.AddSourceLocation(S->getDoLoc(), Record);
  Writer.AddSourceLocation(S->getWhileLoc(), Record);
  Writer.AddSourceLocation(S->getRParenLoc(), Record);
  Code = pch::STMT_DO;
}

void PCHStmtWriter::VisitForStmt(ForStmt *S) {
  VisitStmt(S);
  Writer.WriteSubStmt(S->getInit());
  Writer.WriteSubStmt(S->getCond());
  Writer.AddDeclRef(S->getConditionVariable(), Record);
  Writer.WriteSubStmt(S->getInc());
  Writer.WriteSubStmt(S->getBody());
  Writer.AddSourceLocation(S->getForLoc(), Record);
  Writer.AddSourceLocation(S->getLParenLoc(), Record);
  Writer.AddSourceLocation(S->getRParenLoc(), Record);
  Code = pch::STMT_FOR;
}

void PCHStmtWriter::VisitGotoStmt(GotoStmt *S) {
  VisitStmt(S);
  Record.push_back(Writer.GetLabelID(S->getLabel()));
  Writer.AddSourceLocation(S->getGotoLoc(), Record);
  Writer.AddSourceLocation(S->getLabelLoc(), Record);
  Code = pch::STMT_GOTO;
}

void PCHStmtWriter::VisitIndirectGotoStmt(IndirectGotoStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getGotoLoc(), Record);
  Writer.AddSourceLocation(S->getStarLoc(), Record);
  Writer.WriteSubStmt(S->getTarget());
  Code = pch::STMT_INDIRECT_GOTO;
}

void PCHStmtWriter::VisitContinueStmt(ContinueStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getContinueLoc(), Record);
  Code = pch::STMT_CONTINUE;
}

void PCHStmtWriter::VisitBreakStmt(BreakStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getBreakLoc(), Record);
  Code = pch::STMT_BREAK;
}

void PCHStmtWriter::VisitReturnStmt(ReturnStmt *S) {
  VisitStmt(S);
  Writer.WriteSubStmt(S->getRetValue());
  Writer.AddSourceLocation(S->getReturnLoc(), Record);
  Code = pch::STMT_RETURN;
}

void PCHStmtWriter::VisitDeclStmt(DeclStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getStartLoc(), Record);
  Writer.AddSourceLocation(S->getEndLoc(), Record);
  DeclGroupRef DG = S->getDeclGroup();
  for (DeclGroupRef::iterator D = DG.begin(), DEnd = DG.end(); D != DEnd; ++D)
    Writer.AddDeclRef(*D, Record);
  Code = pch::STMT_DECL;
}

void PCHStmtWriter::VisitAsmStmt(AsmStmt *S) {
  VisitStmt(S);
  Record.push_back(S->getNumOutputs());
  Record.push_back(S->getNumInputs());
  Record.push_back(S->getNumClobbers());
  Writer.AddSourceLocation(S->getAsmLoc(), Record);
  Writer.AddSourceLocation(S->getRParenLoc(), Record);
  Record.push_back(S->isVolatile());
  Record.push_back(S->isSimple());
  Writer.WriteSubStmt(S->getAsmString());

  // Outputs
  for (unsigned I = 0, N = S->getNumOutputs(); I != N; ++I) {
    Writer.AddString(S->getOutputName(I), Record);
    Writer.WriteSubStmt(S->getOutputConstraintLiteral(I));
    Writer.WriteSubStmt(S->getOutputExpr(I));
  }

  // Inputs
  for (unsigned I = 0, N = S->getNumInputs(); I != N; ++I) {
    Writer.AddString(S->getInputName(I), Record);
    Writer.WriteSubStmt(S->getInputConstraintLiteral(I));
    Writer.WriteSubStmt(S->getInputExpr(I));
  }

  // Clobbers
  for (unsigned I = 0, N = S->getNumClobbers(); I != N; ++I)
    Writer.WriteSubStmt(S->getClobber(I));

  Code = pch::STMT_ASM;
}

void PCHStmtWriter::VisitExpr(Expr *E) {
  VisitStmt(E);
  Writer.AddTypeRef(E->getType(), Record);
  Record.push_back(E->isTypeDependent());
  Record.push_back(E->isValueDependent());
}

void PCHStmtWriter::VisitPredefinedExpr(PredefinedExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Record.push_back(E->getIdentType()); // FIXME: stable encoding
  Code = pch::EXPR_PREDEFINED;
}

void PCHStmtWriter::VisitDeclRefExpr(DeclRefExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getDecl(), Record);
  Writer.AddSourceLocation(E->getLocation(), Record);
  // FIXME: write qualifier
  // FIXME: write explicit template arguments
  Code = pch::EXPR_DECL_REF;
}

void PCHStmtWriter::VisitIntegerLiteral(IntegerLiteral *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Writer.AddAPInt(E->getValue(), Record);
  Code = pch::EXPR_INTEGER_LITERAL;
}

void PCHStmtWriter::VisitFloatingLiteral(FloatingLiteral *E) {
  VisitExpr(E);
  Writer.AddAPFloat(E->getValue(), Record);
  Record.push_back(E->isExact());
  Writer.AddSourceLocation(E->getLocation(), Record);
  Code = pch::EXPR_FLOATING_LITERAL;
}

void PCHStmtWriter::VisitImaginaryLiteral(ImaginaryLiteral *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getSubExpr());
  Code = pch::EXPR_IMAGINARY_LITERAL;
}

void PCHStmtWriter::VisitStringLiteral(StringLiteral *E) {
  VisitExpr(E);
  Record.push_back(E->getByteLength());
  Record.push_back(E->getNumConcatenated());
  Record.push_back(E->isWide());
  // FIXME: String data should be stored as a blob at the end of the
  // StringLiteral. However, we can't do so now because we have no
  // provision for coping with abbreviations when we're jumping around
  // the PCH file during deserialization.
  Record.insert(Record.end(),
                E->getStrData(), E->getStrData() + E->getByteLength());
  for (unsigned I = 0, N = E->getNumConcatenated(); I != N; ++I)
    Writer.AddSourceLocation(E->getStrTokenLoc(I), Record);
  Code = pch::EXPR_STRING_LITERAL;
}

void PCHStmtWriter::VisitCharacterLiteral(CharacterLiteral *E) {
  VisitExpr(E);
  Record.push_back(E->getValue());
  Writer.AddSourceLocation(E->getLocation(), Record);
  Record.push_back(E->isWide());
  Code = pch::EXPR_CHARACTER_LITERAL;
}

void PCHStmtWriter::VisitParenExpr(ParenExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLParen(), Record);
  Writer.AddSourceLocation(E->getRParen(), Record);
  Writer.WriteSubStmt(E->getSubExpr());
  Code = pch::EXPR_PAREN;
}

void PCHStmtWriter::VisitUnaryOperator(UnaryOperator *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getSubExpr());
  Record.push_back(E->getOpcode()); // FIXME: stable encoding
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Code = pch::EXPR_UNARY_OPERATOR;
}

void PCHStmtWriter::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E) {
  VisitExpr(E);
  Record.push_back(E->isSizeOf());
  if (E->isArgumentType())
    Writer.AddDeclaratorInfo(E->getArgumentTypeInfo(), Record);
  else {
    Record.push_back(0);
    Writer.WriteSubStmt(E->getArgumentExpr());
  }
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_SIZEOF_ALIGN_OF;
}

void PCHStmtWriter::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getLHS());
  Writer.WriteSubStmt(E->getRHS());
  Writer.AddSourceLocation(E->getRBracketLoc(), Record);
  Code = pch::EXPR_ARRAY_SUBSCRIPT;
}

void PCHStmtWriter::VisitCallExpr(CallExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumArgs());
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Writer.WriteSubStmt(E->getCallee());
  for (CallExpr::arg_iterator Arg = E->arg_begin(), ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg)
    Writer.WriteSubStmt(*Arg);
  Code = pch::EXPR_CALL;
}

void PCHStmtWriter::VisitMemberExpr(MemberExpr *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getBase());
  Writer.AddDeclRef(E->getMemberDecl(), Record);
  Writer.AddSourceLocation(E->getMemberLoc(), Record);
  Record.push_back(E->isArrow());
  // FIXME: C++ nested-name-specifier
  // FIXME: C++ template argument list
  Code = pch::EXPR_MEMBER;
}

void PCHStmtWriter::VisitObjCIsaExpr(ObjCIsaExpr *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getBase());
  Writer.AddSourceLocation(E->getIsaMemberLoc(), Record);
  Record.push_back(E->isArrow());
  Code = pch::EXPR_OBJC_ISA;
}

void PCHStmtWriter::VisitCastExpr(CastExpr *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getSubExpr());
  Record.push_back(E->getCastKind()); // FIXME: stable encoding
}

void PCHStmtWriter::VisitBinaryOperator(BinaryOperator *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getLHS());
  Writer.WriteSubStmt(E->getRHS());
  Record.push_back(E->getOpcode()); // FIXME: stable encoding
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Code = pch::EXPR_BINARY_OPERATOR;
}

void PCHStmtWriter::VisitCompoundAssignOperator(CompoundAssignOperator *E) {
  VisitBinaryOperator(E);
  Writer.AddTypeRef(E->getComputationLHSType(), Record);
  Writer.AddTypeRef(E->getComputationResultType(), Record);
  Code = pch::EXPR_COMPOUND_ASSIGN_OPERATOR;
}

void PCHStmtWriter::VisitConditionalOperator(ConditionalOperator *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getCond());
  Writer.WriteSubStmt(E->getLHS());
  Writer.WriteSubStmt(E->getRHS());
  Writer.AddSourceLocation(E->getQuestionLoc(), Record);
  Writer.AddSourceLocation(E->getColonLoc(), Record);
  Code = pch::EXPR_CONDITIONAL_OPERATOR;
}

void PCHStmtWriter::VisitImplicitCastExpr(ImplicitCastExpr *E) {
  VisitCastExpr(E);
  Record.push_back(E->isLvalueCast());
  Code = pch::EXPR_IMPLICIT_CAST;
}

void PCHStmtWriter::VisitExplicitCastExpr(ExplicitCastExpr *E) {
  VisitCastExpr(E);
  Writer.AddTypeRef(E->getTypeAsWritten(), Record);
}

void PCHStmtWriter::VisitCStyleCastExpr(CStyleCastExpr *E) {
  VisitExplicitCastExpr(E);
  Writer.AddSourceLocation(E->getLParenLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_CSTYLE_CAST;
}

void PCHStmtWriter::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLParenLoc(), Record);
  Writer.WriteSubStmt(E->getInitializer());
  Record.push_back(E->isFileScope());
  Code = pch::EXPR_COMPOUND_LITERAL;
}

void PCHStmtWriter::VisitExtVectorElementExpr(ExtVectorElementExpr *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getBase());
  Writer.AddIdentifierRef(&E->getAccessor(), Record);
  Writer.AddSourceLocation(E->getAccessorLoc(), Record);
  Code = pch::EXPR_EXT_VECTOR_ELEMENT;
}

void PCHStmtWriter::VisitInitListExpr(InitListExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumInits());
  for (unsigned I = 0, N = E->getNumInits(); I != N; ++I)
    Writer.WriteSubStmt(E->getInit(I));
  Writer.WriteSubStmt(E->getSyntacticForm());
  Writer.AddSourceLocation(E->getLBraceLoc(), Record);
  Writer.AddSourceLocation(E->getRBraceLoc(), Record);
  Writer.AddDeclRef(E->getInitializedFieldInUnion(), Record);
  Record.push_back(E->hadArrayRangeDesignator());
  Code = pch::EXPR_INIT_LIST;
}

void PCHStmtWriter::VisitDesignatedInitExpr(DesignatedInitExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumSubExprs());
  for (unsigned I = 0, N = E->getNumSubExprs(); I != N; ++I)
    Writer.WriteSubStmt(E->getSubExpr(I));
  Writer.AddSourceLocation(E->getEqualOrColonLoc(), Record);
  Record.push_back(E->usesGNUSyntax());
  for (DesignatedInitExpr::designators_iterator D = E->designators_begin(),
                                             DEnd = E->designators_end();
       D != DEnd; ++D) {
    if (D->isFieldDesignator()) {
      if (FieldDecl *Field = D->getField()) {
        Record.push_back(pch::DESIG_FIELD_DECL);
        Writer.AddDeclRef(Field, Record);
      } else {
        Record.push_back(pch::DESIG_FIELD_NAME);
        Writer.AddIdentifierRef(D->getFieldName(), Record);
      }
      Writer.AddSourceLocation(D->getDotLoc(), Record);
      Writer.AddSourceLocation(D->getFieldLoc(), Record);
    } else if (D->isArrayDesignator()) {
      Record.push_back(pch::DESIG_ARRAY);
      Record.push_back(D->getFirstExprIndex());
      Writer.AddSourceLocation(D->getLBracketLoc(), Record);
      Writer.AddSourceLocation(D->getRBracketLoc(), Record);
    } else {
      assert(D->isArrayRangeDesignator() && "Unknown designator");
      Record.push_back(pch::DESIG_ARRAY_RANGE);
      Record.push_back(D->getFirstExprIndex());
      Writer.AddSourceLocation(D->getLBracketLoc(), Record);
      Writer.AddSourceLocation(D->getEllipsisLoc(), Record);
      Writer.AddSourceLocation(D->getRBracketLoc(), Record);
    }
  }
  Code = pch::EXPR_DESIGNATED_INIT;
}

void PCHStmtWriter::VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
  VisitExpr(E);
  Code = pch::EXPR_IMPLICIT_VALUE_INIT;
}

void PCHStmtWriter::VisitVAArgExpr(VAArgExpr *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getSubExpr());
  Writer.AddSourceLocation(E->getBuiltinLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_VA_ARG;
}

void PCHStmtWriter::VisitAddrLabelExpr(AddrLabelExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getAmpAmpLoc(), Record);
  Writer.AddSourceLocation(E->getLabelLoc(), Record);
  Record.push_back(Writer.GetLabelID(E->getLabel()));
  Code = pch::EXPR_ADDR_LABEL;
}

void PCHStmtWriter::VisitStmtExpr(StmtExpr *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getSubStmt());
  Writer.AddSourceLocation(E->getLParenLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_STMT;
}

void PCHStmtWriter::VisitTypesCompatibleExpr(TypesCompatibleExpr *E) {
  VisitExpr(E);
  Writer.AddTypeRef(E->getArgType1(), Record);
  Writer.AddTypeRef(E->getArgType2(), Record);
  Writer.AddSourceLocation(E->getBuiltinLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_TYPES_COMPATIBLE;
}

void PCHStmtWriter::VisitChooseExpr(ChooseExpr *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getCond());
  Writer.WriteSubStmt(E->getLHS());
  Writer.WriteSubStmt(E->getRHS());
  Writer.AddSourceLocation(E->getBuiltinLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_CHOOSE;
}

void PCHStmtWriter::VisitGNUNullExpr(GNUNullExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getTokenLocation(), Record);
  Code = pch::EXPR_GNU_NULL;
}

void PCHStmtWriter::VisitShuffleVectorExpr(ShuffleVectorExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumSubExprs());
  for (unsigned I = 0, N = E->getNumSubExprs(); I != N; ++I)
    Writer.WriteSubStmt(E->getExpr(I));
  Writer.AddSourceLocation(E->getBuiltinLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_SHUFFLE_VECTOR;
}

void PCHStmtWriter::VisitBlockExpr(BlockExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getBlockDecl(), Record);
  Record.push_back(E->hasBlockDeclRefExprs());
  Code = pch::EXPR_BLOCK;
}

void PCHStmtWriter::VisitBlockDeclRefExpr(BlockDeclRefExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getDecl(), Record);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Record.push_back(E->isByRef());
  Record.push_back(E->isConstQualAdded());
  Code = pch::EXPR_BLOCK_DECL_REF;
}

//===----------------------------------------------------------------------===//
// Objective-C Expressions and Statements.
//===----------------------------------------------------------------------===//

void PCHStmtWriter::VisitObjCStringLiteral(ObjCStringLiteral *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getString());
  Writer.AddSourceLocation(E->getAtLoc(), Record);
  Code = pch::EXPR_OBJC_STRING_LITERAL;
}

void PCHStmtWriter::VisitObjCEncodeExpr(ObjCEncodeExpr *E) {
  VisitExpr(E);
  Writer.AddTypeRef(E->getEncodedType(), Record);
  Writer.AddSourceLocation(E->getAtLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_OBJC_ENCODE;
}

void PCHStmtWriter::VisitObjCSelectorExpr(ObjCSelectorExpr *E) {
  VisitExpr(E);
  Writer.AddSelectorRef(E->getSelector(), Record);
  Writer.AddSourceLocation(E->getAtLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_OBJC_SELECTOR_EXPR;
}

void PCHStmtWriter::VisitObjCProtocolExpr(ObjCProtocolExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getProtocol(), Record);
  Writer.AddSourceLocation(E->getAtLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_OBJC_PROTOCOL_EXPR;
}

void PCHStmtWriter::VisitObjCIvarRefExpr(ObjCIvarRefExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getDecl(), Record);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Writer.WriteSubStmt(E->getBase());
  Record.push_back(E->isArrow());
  Record.push_back(E->isFreeIvar());
  Code = pch::EXPR_OBJC_IVAR_REF_EXPR;
}

void PCHStmtWriter::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getProperty(), Record);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Writer.WriteSubStmt(E->getBase());
  Code = pch::EXPR_OBJC_PROPERTY_REF_EXPR;
}

void PCHStmtWriter::VisitObjCImplicitSetterGetterRefExpr(
                                  ObjCImplicitSetterGetterRefExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getGetterMethod(), Record);
  Writer.AddDeclRef(E->getSetterMethod(), Record);

  // NOTE: InterfaceDecl and Base are mutually exclusive.
  Writer.AddDeclRef(E->getInterfaceDecl(), Record);
  Writer.WriteSubStmt(E->getBase());
  Writer.AddSourceLocation(E->getLocation(), Record);
  Writer.AddSourceLocation(E->getClassLoc(), Record);
  Code = pch::EXPR_OBJC_KVC_REF_EXPR;
}

void PCHStmtWriter::VisitObjCMessageExpr(ObjCMessageExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumArgs());
  Writer.AddSourceLocation(E->getLeftLoc(), Record);
  Writer.AddSourceLocation(E->getRightLoc(), Record);
  Writer.AddSelectorRef(E->getSelector(), Record);
  Writer.AddDeclRef(E->getMethodDecl(), Record); // optional
  Writer.WriteSubStmt(E->getReceiver());

  if (!E->getReceiver()) {
    ObjCMessageExpr::ClassInfo CI = E->getClassInfo();
    Writer.AddDeclRef(CI.first, Record);
    Writer.AddIdentifierRef(CI.second, Record);
  }

  for (CallExpr::arg_iterator Arg = E->arg_begin(), ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg)
    Writer.WriteSubStmt(*Arg);
  Code = pch::EXPR_OBJC_MESSAGE_EXPR;
}

void PCHStmtWriter::VisitObjCSuperExpr(ObjCSuperExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLoc(), Record);
  Code = pch::EXPR_OBJC_SUPER_EXPR;
}

void PCHStmtWriter::VisitObjCForCollectionStmt(ObjCForCollectionStmt *S) {
  VisitStmt(S);
  Writer.WriteSubStmt(S->getElement());
  Writer.WriteSubStmt(S->getCollection());
  Writer.WriteSubStmt(S->getBody());
  Writer.AddSourceLocation(S->getForLoc(), Record);
  Writer.AddSourceLocation(S->getRParenLoc(), Record);
  Code = pch::STMT_OBJC_FOR_COLLECTION;
}

void PCHStmtWriter::VisitObjCAtCatchStmt(ObjCAtCatchStmt *S) {
  Writer.WriteSubStmt(S->getCatchBody());
  Writer.WriteSubStmt(S->getNextCatchStmt());
  Writer.AddDeclRef(S->getCatchParamDecl(), Record);
  Writer.AddSourceLocation(S->getAtCatchLoc(), Record);
  Writer.AddSourceLocation(S->getRParenLoc(), Record);
  Code = pch::STMT_OBJC_CATCH;
}

void PCHStmtWriter::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *S) {
  Writer.WriteSubStmt(S->getFinallyBody());
  Writer.AddSourceLocation(S->getAtFinallyLoc(), Record);
  Code = pch::STMT_OBJC_FINALLY;
}

void PCHStmtWriter::VisitObjCAtTryStmt(ObjCAtTryStmt *S) {
  Writer.WriteSubStmt(S->getTryBody());
  Writer.WriteSubStmt(S->getCatchStmts());
  Writer.WriteSubStmt(S->getFinallyStmt());
  Writer.AddSourceLocation(S->getAtTryLoc(), Record);
  Code = pch::STMT_OBJC_AT_TRY;
}

void PCHStmtWriter::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *S) {
  Writer.WriteSubStmt(S->getSynchExpr());
  Writer.WriteSubStmt(S->getSynchBody());
  Writer.AddSourceLocation(S->getAtSynchronizedLoc(), Record);
  Code = pch::STMT_OBJC_AT_SYNCHRONIZED;
}

void PCHStmtWriter::VisitObjCAtThrowStmt(ObjCAtThrowStmt *S) {
  Writer.WriteSubStmt(S->getThrowExpr());
  Writer.AddSourceLocation(S->getThrowLoc(), Record);
  Code = pch::STMT_OBJC_AT_THROW;
}

//===----------------------------------------------------------------------===//
// C++ Expressions and Statements.
//===----------------------------------------------------------------------===//

void PCHStmtWriter::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
  VisitCallExpr(E);
  Record.push_back(E->getOperator());
  Code = pch::EXPR_CXX_OPERATOR_CALL;
}

void PCHStmtWriter::VisitCXXConstructExpr(CXXConstructExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getConstructor(), Record);
  Record.push_back(E->isElidable());
  Record.push_back(E->getNumArgs());
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
    Writer.WriteSubStmt(E->getArg(I));
  Code = pch::EXPR_CXX_CONSTRUCT;
}

//===----------------------------------------------------------------------===//
// PCHWriter Implementation
//===----------------------------------------------------------------------===//

unsigned PCHWriter::RecordSwitchCaseID(SwitchCase *S) {
  assert(SwitchCaseIDs.find(S) == SwitchCaseIDs.end() &&
         "SwitchCase recorded twice");
  unsigned NextID = SwitchCaseIDs.size();
  SwitchCaseIDs[S] = NextID;
  return NextID;
}

unsigned PCHWriter::getSwitchCaseID(SwitchCase *S) {
  assert(SwitchCaseIDs.find(S) != SwitchCaseIDs.end() &&
         "SwitchCase hasn't been seen yet");
  return SwitchCaseIDs[S];
}

/// \brief Retrieve the ID for the given label statement, which may
/// or may not have been emitted yet.
unsigned PCHWriter::GetLabelID(LabelStmt *S) {
  std::map<LabelStmt *, unsigned>::iterator Pos = LabelIDs.find(S);
  if (Pos != LabelIDs.end())
    return Pos->second;

  unsigned NextID = LabelIDs.size();
  LabelIDs[S] = NextID;
  return NextID;
}

/// \brief Write the given substatement or subexpression to the
/// bitstream.
void PCHWriter::WriteSubStmt(Stmt *S) {
  RecordData Record;
  PCHStmtWriter Writer(*this, Record);
  ++NumStatements;

  if (!S) {
    Stream.EmitRecord(pch::STMT_NULL_PTR, Record);
    return;
  }

  Writer.Code = pch::STMT_NULL_PTR;
  Writer.Visit(S);
  assert(Writer.Code != pch::STMT_NULL_PTR &&
         "Unhandled expression writing PCH file");
  Stream.EmitRecord(Writer.Code, Record);
}

/// \brief Flush all of the statements that have been added to the
/// queue via AddStmt().
void PCHWriter::FlushStmts() {
  RecordData Record;
  PCHStmtWriter Writer(*this, Record);

  for (unsigned I = 0, N = StmtsToEmit.size(); I != N; ++I) {
    ++NumStatements;
    Stmt *S = StmtsToEmit[I];

    if (!S) {
      Stream.EmitRecord(pch::STMT_NULL_PTR, Record);
      continue;
    }

    Writer.Code = pch::STMT_NULL_PTR;
    Writer.Visit(S);
    assert(Writer.Code != pch::STMT_NULL_PTR &&
           "Unhandled expression writing PCH file");
    Stream.EmitRecord(Writer.Code, Record);

    assert(N == StmtsToEmit.size() &&
           "Substatement writen via AddStmt rather than WriteSubStmt!");

    // Note that we are at the end of a full expression. Any
    // expression records that follow this one are part of a different
    // expression.
    Record.clear();
    Stream.EmitRecord(pch::STMT_STOP, Record);
  }

  StmtsToEmit.clear();
}
