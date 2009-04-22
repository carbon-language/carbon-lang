//===--- PCHReader.cpp - Precompiled Headers Reader -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PCHReader class, which reads a precompiled header.
//
//===----------------------------------------------------------------------===//
#include "clang/Frontend/PCHReader.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "../Sema/Sema.h" // FIXME: move Sema headers elsewhere
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/OnDiskHashTable.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/SourceManagerInternals.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>
#include <cstdio>

using namespace clang;

namespace {
  /// \brief Helper class that saves the current stream position and
  /// then restores it when destroyed.
  struct VISIBILITY_HIDDEN SavedStreamPosition {
    explicit SavedStreamPosition(llvm::BitstreamReader &Stream)
      : Stream(Stream), Offset(Stream.GetCurrentBitNo()) { }

    ~SavedStreamPosition() {
      Stream.JumpToBit(Offset);
    }

  private:
    llvm::BitstreamReader &Stream;
    uint64_t Offset;
  };
}

//===----------------------------------------------------------------------===//
// Declaration deserialization
//===----------------------------------------------------------------------===//
namespace {
  class VISIBILITY_HIDDEN PCHDeclReader 
    : public DeclVisitor<PCHDeclReader, void> {
    PCHReader &Reader;
    const PCHReader::RecordData &Record;
    unsigned &Idx;

  public:
    PCHDeclReader(PCHReader &Reader, const PCHReader::RecordData &Record,
                  unsigned &Idx)
      : Reader(Reader), Record(Record), Idx(Idx) { }

    void VisitDecl(Decl *D);
    void VisitTranslationUnitDecl(TranslationUnitDecl *TU);
    void VisitNamedDecl(NamedDecl *ND);
    void VisitTypeDecl(TypeDecl *TD);
    void VisitTypedefDecl(TypedefDecl *TD);
    void VisitTagDecl(TagDecl *TD);
    void VisitEnumDecl(EnumDecl *ED);
    void VisitRecordDecl(RecordDecl *RD);
    void VisitValueDecl(ValueDecl *VD);
    void VisitEnumConstantDecl(EnumConstantDecl *ECD);
    void VisitFunctionDecl(FunctionDecl *FD);
    void VisitFieldDecl(FieldDecl *FD);
    void VisitVarDecl(VarDecl *VD);
    void VisitParmVarDecl(ParmVarDecl *PD);
    void VisitOriginalParmVarDecl(OriginalParmVarDecl *PD);
    void VisitFileScopeAsmDecl(FileScopeAsmDecl *AD);
    void VisitBlockDecl(BlockDecl *BD);
    std::pair<uint64_t, uint64_t> VisitDeclContext(DeclContext *DC);
    void VisitObjCMethodDecl(ObjCMethodDecl *D);
    void VisitObjCContainerDecl(ObjCContainerDecl *D);
    void VisitObjCInterfaceDecl(ObjCInterfaceDecl *D);
    void VisitObjCIvarDecl(ObjCIvarDecl *D);
    void VisitObjCProtocolDecl(ObjCProtocolDecl *D);
    void VisitObjCAtDefsFieldDecl(ObjCAtDefsFieldDecl *D);
    void VisitObjCClassDecl(ObjCClassDecl *D);
    void VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *D);
    void VisitObjCCategoryDecl(ObjCCategoryDecl *D);
    void VisitObjCImplDecl(ObjCImplDecl *D);
    void VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D);
    void VisitObjCImplementationDecl(ObjCImplementationDecl *D);
    void VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *D);
    void VisitObjCPropertyDecl(ObjCPropertyDecl *D);
    void VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D);
  };
}

void PCHDeclReader::VisitDecl(Decl *D) {
  D->setDeclContext(cast_or_null<DeclContext>(Reader.GetDecl(Record[Idx++])));
  D->setLexicalDeclContext(
                     cast_or_null<DeclContext>(Reader.GetDecl(Record[Idx++])));
  D->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  D->setInvalidDecl(Record[Idx++]);
  if (Record[Idx++])
    D->addAttr(Reader.ReadAttributes());
  D->setImplicit(Record[Idx++]);
  D->setAccess((AccessSpecifier)Record[Idx++]);
}

void PCHDeclReader::VisitTranslationUnitDecl(TranslationUnitDecl *TU) {
  VisitDecl(TU);
}

void PCHDeclReader::VisitNamedDecl(NamedDecl *ND) {
  VisitDecl(ND);
  ND->setDeclName(Reader.ReadDeclarationName(Record, Idx));  
}

void PCHDeclReader::VisitTypeDecl(TypeDecl *TD) {
  VisitNamedDecl(TD);
  TD->setTypeForDecl(Reader.GetType(Record[Idx++]).getTypePtr());
}

void PCHDeclReader::VisitTypedefDecl(TypedefDecl *TD) {
  // Note that we cannot use VisitTypeDecl here, because we need to
  // set the underlying type of the typedef *before* we try to read
  // the type associated with the TypedefDecl.
  VisitNamedDecl(TD);
  TD->setUnderlyingType(Reader.GetType(Record[Idx + 1]));
  TD->setTypeForDecl(Reader.GetType(Record[Idx]).getTypePtr());
  Idx += 2;
}

void PCHDeclReader::VisitTagDecl(TagDecl *TD) {
  VisitTypeDecl(TD);
  TD->setTagKind((TagDecl::TagKind)Record[Idx++]);
  TD->setDefinition(Record[Idx++]);
  TD->setTypedefForAnonDecl(
                    cast_or_null<TypedefDecl>(Reader.GetDecl(Record[Idx++])));
}

void PCHDeclReader::VisitEnumDecl(EnumDecl *ED) {
  VisitTagDecl(ED);
  ED->setIntegerType(Reader.GetType(Record[Idx++]));
}

void PCHDeclReader::VisitRecordDecl(RecordDecl *RD) {
  VisitTagDecl(RD);
  RD->setHasFlexibleArrayMember(Record[Idx++]);
  RD->setAnonymousStructOrUnion(Record[Idx++]);
}

void PCHDeclReader::VisitValueDecl(ValueDecl *VD) {
  VisitNamedDecl(VD);
  VD->setType(Reader.GetType(Record[Idx++]));
}

void PCHDeclReader::VisitEnumConstantDecl(EnumConstantDecl *ECD) {
  VisitValueDecl(ECD);
  if (Record[Idx++])
    ECD->setInitExpr(Reader.ReadExpr());
  ECD->setInitVal(Reader.ReadAPSInt(Record, Idx));
}

void PCHDeclReader::VisitFunctionDecl(FunctionDecl *FD) {
  VisitValueDecl(FD);
  if (Record[Idx++])
    FD->setLazyBody(Reader.getStream().GetCurrentBitNo());
  FD->setPreviousDeclaration(
                   cast_or_null<FunctionDecl>(Reader.GetDecl(Record[Idx++])));
  FD->setStorageClass((FunctionDecl::StorageClass)Record[Idx++]);
  FD->setInline(Record[Idx++]);
  FD->setVirtual(Record[Idx++]);
  FD->setPure(Record[Idx++]);
  FD->setInheritedPrototype(Record[Idx++]);
  FD->setHasPrototype(Record[Idx++]);
  FD->setDeleted(Record[Idx++]);
  FD->setTypeSpecStartLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  unsigned NumParams = Record[Idx++];
  llvm::SmallVector<ParmVarDecl *, 16> Params;
  Params.reserve(NumParams);
  for (unsigned I = 0; I != NumParams; ++I)
    Params.push_back(cast<ParmVarDecl>(Reader.GetDecl(Record[Idx++])));
  FD->setParams(Reader.getContext(), &Params[0], NumParams);
}

void PCHDeclReader::VisitObjCMethodDecl(ObjCMethodDecl *MD) {
  VisitNamedDecl(MD);
  if (Record[Idx++]) {
    // In practice, this won't be executed (since method definitions
    // don't occur in header files).
    MD->setBody(cast<CompoundStmt>(Reader.GetStmt(Record[Idx++])));
    MD->setSelfDecl(cast<ImplicitParamDecl>(Reader.GetDecl(Record[Idx++])));
    MD->setCmdDecl(cast<ImplicitParamDecl>(Reader.GetDecl(Record[Idx++])));
  }
  MD->setInstanceMethod(Record[Idx++]);
  MD->setVariadic(Record[Idx++]);
  MD->setSynthesized(Record[Idx++]);
  MD->setDeclImplementation((ObjCMethodDecl::ImplementationControl)Record[Idx++]);
  MD->setObjCDeclQualifier((Decl::ObjCDeclQualifier)Record[Idx++]);
  MD->setResultType(Reader.GetType(Record[Idx++]));
  MD->setEndLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  unsigned NumParams = Record[Idx++];
  llvm::SmallVector<ParmVarDecl *, 16> Params;
  Params.reserve(NumParams);
  for (unsigned I = 0; I != NumParams; ++I)
    Params.push_back(cast<ParmVarDecl>(Reader.GetDecl(Record[Idx++])));
  MD->setMethodParams(Reader.getContext(), &Params[0], NumParams);
}

void PCHDeclReader::VisitObjCContainerDecl(ObjCContainerDecl *CD) {
  VisitNamedDecl(CD);
  CD->setAtEndLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void PCHDeclReader::VisitObjCInterfaceDecl(ObjCInterfaceDecl *ID) {
  VisitObjCContainerDecl(ID);
  ID->setTypeForDecl(Reader.GetType(Record[Idx++]).getTypePtr());
  ID->setSuperClass(cast_or_null<ObjCInterfaceDecl>
                       (Reader.GetDecl(Record[Idx++])));
  unsigned NumIvars = Record[Idx++];
  llvm::SmallVector<ObjCIvarDecl *, 16> IVars;
  IVars.reserve(NumIvars);
  for (unsigned I = 0; I != NumIvars; ++I)
    IVars.push_back(cast<ObjCIvarDecl>(Reader.GetDecl(Record[Idx++])));
  ID->setIVarList(&IVars[0], NumIvars, Reader.getContext());

  ID->setForwardDecl(Record[Idx++]);
  ID->setImplicitInterfaceDecl(Record[Idx++]);
  ID->setClassLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  ID->setSuperClassLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  ID->setAtEndLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  // FIXME: add protocols, categories.
}

void PCHDeclReader::VisitObjCIvarDecl(ObjCIvarDecl *IVD) {
  VisitFieldDecl(IVD);
  IVD->setAccessControl((ObjCIvarDecl::AccessControl)Record[Idx++]);
}

void PCHDeclReader::VisitObjCProtocolDecl(ObjCProtocolDecl *PD) {
  VisitObjCContainerDecl(PD);
  PD->setForwardDecl(Record[Idx++]);
  PD->setLocEnd(SourceLocation::getFromRawEncoding(Record[Idx++]));
  unsigned NumProtoRefs = Record[Idx++];
  llvm::SmallVector<ObjCProtocolDecl *, 16> ProtoRefs;
  ProtoRefs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoRefs.push_back(cast<ObjCProtocolDecl>(Reader.GetDecl(Record[Idx++])));
  PD->setProtocolList(&ProtoRefs[0], NumProtoRefs, Reader.getContext());
}

void PCHDeclReader::VisitObjCAtDefsFieldDecl(ObjCAtDefsFieldDecl *FD) {
  VisitFieldDecl(FD);
}

void PCHDeclReader::VisitObjCClassDecl(ObjCClassDecl *CD) {
  VisitDecl(CD);
  unsigned NumClassRefs = Record[Idx++];
  llvm::SmallVector<ObjCInterfaceDecl *, 16> ClassRefs;
  ClassRefs.reserve(NumClassRefs);
  for (unsigned I = 0; I != NumClassRefs; ++I)
    ClassRefs.push_back(cast<ObjCInterfaceDecl>(Reader.GetDecl(Record[Idx++])));
  CD->setClassList(Reader.getContext(), &ClassRefs[0], NumClassRefs);
}

void PCHDeclReader::VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *FPD) {
  VisitDecl(FPD);
  unsigned NumProtoRefs = Record[Idx++];
  llvm::SmallVector<ObjCProtocolDecl *, 16> ProtoRefs;
  ProtoRefs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoRefs.push_back(cast<ObjCProtocolDecl>(Reader.GetDecl(Record[Idx++])));
  FPD->setProtocolList(&ProtoRefs[0], NumProtoRefs, Reader.getContext());
}

void PCHDeclReader::VisitObjCCategoryDecl(ObjCCategoryDecl *CD) {
  VisitObjCContainerDecl(CD);
  CD->setClassInterface(cast<ObjCInterfaceDecl>(Reader.GetDecl(Record[Idx++])));
  unsigned NumProtoRefs = Record[Idx++];
  llvm::SmallVector<ObjCProtocolDecl *, 16> ProtoRefs;
  ProtoRefs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoRefs.push_back(cast<ObjCProtocolDecl>(Reader.GetDecl(Record[Idx++])));
  CD->setProtocolList(&ProtoRefs[0], NumProtoRefs, Reader.getContext());
  CD->setNextClassCategory(cast<ObjCCategoryDecl>(Reader.GetDecl(Record[Idx++])));
  CD->setLocEnd(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void PCHDeclReader::VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *CAD) {
  VisitNamedDecl(CAD);
  CAD->setClassInterface(cast<ObjCInterfaceDecl>(Reader.GetDecl(Record[Idx++])));
}

void PCHDeclReader::VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
  VisitNamedDecl(D);
  // FIXME: Implement.
}

void PCHDeclReader::VisitObjCImplDecl(ObjCImplDecl *D) {
  VisitDecl(D);
  // FIXME: Implement.
}

void PCHDeclReader::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
  VisitObjCImplDecl(D);
  // FIXME: Implement.
}

void PCHDeclReader::VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
  VisitObjCImplDecl(D);
  // FIXME: Implement.
}


void PCHDeclReader::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
  VisitDecl(D);
  // FIXME: Implement.
}

void PCHDeclReader::VisitFieldDecl(FieldDecl *FD) {
  VisitValueDecl(FD);
  FD->setMutable(Record[Idx++]);
  if (Record[Idx++])
    FD->setBitWidth(Reader.ReadExpr());
}

void PCHDeclReader::VisitVarDecl(VarDecl *VD) {
  VisitValueDecl(VD);
  VD->setStorageClass((VarDecl::StorageClass)Record[Idx++]);
  VD->setThreadSpecified(Record[Idx++]);
  VD->setCXXDirectInitializer(Record[Idx++]);
  VD->setDeclaredInCondition(Record[Idx++]);
  VD->setPreviousDeclaration(
                         cast_or_null<VarDecl>(Reader.GetDecl(Record[Idx++])));
  VD->setTypeSpecStartLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  if (Record[Idx++])
    VD->setInit(Reader.ReadExpr());
}

void PCHDeclReader::VisitParmVarDecl(ParmVarDecl *PD) {
  VisitVarDecl(PD);
  PD->setObjCDeclQualifier((Decl::ObjCDeclQualifier)Record[Idx++]);
  // FIXME: default argument (C++ only)
}

void PCHDeclReader::VisitOriginalParmVarDecl(OriginalParmVarDecl *PD) {
  VisitParmVarDecl(PD);
  PD->setOriginalType(Reader.GetType(Record[Idx++]));
}

void PCHDeclReader::VisitFileScopeAsmDecl(FileScopeAsmDecl *AD) {
  VisitDecl(AD);
  AD->setAsmString(cast<StringLiteral>(Reader.ReadExpr()));
}

void PCHDeclReader::VisitBlockDecl(BlockDecl *BD) {
  VisitDecl(BD);
  BD->setBody(cast_or_null<CompoundStmt>(Reader.ReadStmt()));
  unsigned NumParams = Record[Idx++];
  llvm::SmallVector<ParmVarDecl *, 16> Params;
  Params.reserve(NumParams);
  for (unsigned I = 0; I != NumParams; ++I)
    Params.push_back(cast<ParmVarDecl>(Reader.GetDecl(Record[Idx++])));
  BD->setParams(Reader.getContext(), &Params[0], NumParams);  
}

std::pair<uint64_t, uint64_t> 
PCHDeclReader::VisitDeclContext(DeclContext *DC) {
  uint64_t LexicalOffset = Record[Idx++];
  uint64_t VisibleOffset = Record[Idx++];
  return std::make_pair(LexicalOffset, VisibleOffset);
}

//===----------------------------------------------------------------------===//
// Statement/expression deserialization
//===----------------------------------------------------------------------===//
namespace {
  class VISIBILITY_HIDDEN PCHStmtReader 
    : public StmtVisitor<PCHStmtReader, unsigned> {
    PCHReader &Reader;
    const PCHReader::RecordData &Record;
    unsigned &Idx;
    llvm::SmallVectorImpl<Stmt *> &StmtStack;

  public:
    PCHStmtReader(PCHReader &Reader, const PCHReader::RecordData &Record,
                  unsigned &Idx, llvm::SmallVectorImpl<Stmt *> &StmtStack)
      : Reader(Reader), Record(Record), Idx(Idx), StmtStack(StmtStack) { }

    /// \brief The number of record fields required for the Stmt class
    /// itself.
    static const unsigned NumStmtFields = 0;

    /// \brief The number of record fields required for the Expr class
    /// itself.
    static const unsigned NumExprFields = NumStmtFields + 3;

    // Each of the Visit* functions reads in part of the expression
    // from the given record and the current expression stack, then
    // return the total number of operands that it read from the
    // expression stack.

    unsigned VisitStmt(Stmt *S);
    unsigned VisitNullStmt(NullStmt *S);
    unsigned VisitCompoundStmt(CompoundStmt *S);
    unsigned VisitSwitchCase(SwitchCase *S);
    unsigned VisitCaseStmt(CaseStmt *S);
    unsigned VisitDefaultStmt(DefaultStmt *S);
    unsigned VisitLabelStmt(LabelStmt *S);
    unsigned VisitIfStmt(IfStmt *S);
    unsigned VisitSwitchStmt(SwitchStmt *S);
    unsigned VisitWhileStmt(WhileStmt *S);
    unsigned VisitDoStmt(DoStmt *S);
    unsigned VisitForStmt(ForStmt *S);
    unsigned VisitGotoStmt(GotoStmt *S);
    unsigned VisitIndirectGotoStmt(IndirectGotoStmt *S);
    unsigned VisitContinueStmt(ContinueStmt *S);
    unsigned VisitBreakStmt(BreakStmt *S);
    unsigned VisitReturnStmt(ReturnStmt *S);
    unsigned VisitDeclStmt(DeclStmt *S);
    unsigned VisitAsmStmt(AsmStmt *S);
    unsigned VisitExpr(Expr *E);
    unsigned VisitPredefinedExpr(PredefinedExpr *E);
    unsigned VisitDeclRefExpr(DeclRefExpr *E);
    unsigned VisitIntegerLiteral(IntegerLiteral *E);
    unsigned VisitFloatingLiteral(FloatingLiteral *E);
    unsigned VisitImaginaryLiteral(ImaginaryLiteral *E);
    unsigned VisitStringLiteral(StringLiteral *E);
    unsigned VisitCharacterLiteral(CharacterLiteral *E);
    unsigned VisitParenExpr(ParenExpr *E);
    unsigned VisitUnaryOperator(UnaryOperator *E);
    unsigned VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E);
    unsigned VisitArraySubscriptExpr(ArraySubscriptExpr *E);
    unsigned VisitCallExpr(CallExpr *E);
    unsigned VisitMemberExpr(MemberExpr *E);
    unsigned VisitCastExpr(CastExpr *E);
    unsigned VisitBinaryOperator(BinaryOperator *E);
    unsigned VisitCompoundAssignOperator(CompoundAssignOperator *E);
    unsigned VisitConditionalOperator(ConditionalOperator *E);
    unsigned VisitImplicitCastExpr(ImplicitCastExpr *E);
    unsigned VisitExplicitCastExpr(ExplicitCastExpr *E);
    unsigned VisitCStyleCastExpr(CStyleCastExpr *E);
    unsigned VisitCompoundLiteralExpr(CompoundLiteralExpr *E);
    unsigned VisitExtVectorElementExpr(ExtVectorElementExpr *E);
    unsigned VisitInitListExpr(InitListExpr *E);
    unsigned VisitDesignatedInitExpr(DesignatedInitExpr *E);
    unsigned VisitImplicitValueInitExpr(ImplicitValueInitExpr *E);
    unsigned VisitVAArgExpr(VAArgExpr *E);
    unsigned VisitAddrLabelExpr(AddrLabelExpr *E);
    unsigned VisitStmtExpr(StmtExpr *E);
    unsigned VisitTypesCompatibleExpr(TypesCompatibleExpr *E);
    unsigned VisitChooseExpr(ChooseExpr *E);
    unsigned VisitGNUNullExpr(GNUNullExpr *E);
    unsigned VisitShuffleVectorExpr(ShuffleVectorExpr *E);
    unsigned VisitBlockExpr(BlockExpr *E);
    unsigned VisitBlockDeclRefExpr(BlockDeclRefExpr *E);
    unsigned VisitObjCStringLiteral(ObjCStringLiteral *E);
    unsigned VisitObjCEncodeExpr(ObjCEncodeExpr *E);
    unsigned VisitObjCSelectorExpr(ObjCSelectorExpr *E);
    unsigned VisitObjCProtocolExpr(ObjCProtocolExpr *E);
  };
}

unsigned PCHStmtReader::VisitStmt(Stmt *S) {
  assert(Idx == NumStmtFields && "Incorrect statement field count");
  return 0;
}

unsigned PCHStmtReader::VisitNullStmt(NullStmt *S) {
  VisitStmt(S);
  S->setSemiLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitCompoundStmt(CompoundStmt *S) {
  VisitStmt(S);
  unsigned NumStmts = Record[Idx++];
  S->setStmts(Reader.getContext(), 
              &StmtStack[StmtStack.size() - NumStmts], NumStmts);
  S->setLBracLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setRBracLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return NumStmts;
}

unsigned PCHStmtReader::VisitSwitchCase(SwitchCase *S) {
  VisitStmt(S);
  Reader.RecordSwitchCaseID(S, Record[Idx++]);
  return 0;
}

unsigned PCHStmtReader::VisitCaseStmt(CaseStmt *S) {
  VisitSwitchCase(S);
  S->setLHS(cast<Expr>(StmtStack[StmtStack.size() - 3]));
  S->setRHS(cast_or_null<Expr>(StmtStack[StmtStack.size() - 2]));
  S->setSubStmt(StmtStack.back());
  S->setCaseLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 3;
}

unsigned PCHStmtReader::VisitDefaultStmt(DefaultStmt *S) {
  VisitSwitchCase(S);
  S->setSubStmt(StmtStack.back());
  S->setDefaultLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 1;
}

unsigned PCHStmtReader::VisitLabelStmt(LabelStmt *S) {
  VisitStmt(S);
  S->setID(Reader.GetIdentifierInfo(Record, Idx));
  S->setSubStmt(StmtStack.back());
  S->setIdentLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  Reader.RecordLabelStmt(S, Record[Idx++]);
  return 1;
}

unsigned PCHStmtReader::VisitIfStmt(IfStmt *S) {
  VisitStmt(S);
  S->setCond(cast<Expr>(StmtStack[StmtStack.size() - 3]));
  S->setThen(StmtStack[StmtStack.size() - 2]);
  S->setElse(StmtStack[StmtStack.size() - 1]);
  S->setIfLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 3;
}

unsigned PCHStmtReader::VisitSwitchStmt(SwitchStmt *S) {
  VisitStmt(S);
  S->setCond(cast<Expr>(StmtStack[StmtStack.size() - 2]));
  S->setBody(StmtStack.back());
  S->setSwitchLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  SwitchCase *PrevSC = 0;
  for (unsigned N = Record.size(); Idx != N; ++Idx) {
    SwitchCase *SC = Reader.getSwitchCaseWithID(Record[Idx]);
    if (PrevSC)
      PrevSC->setNextSwitchCase(SC);
    else
      S->setSwitchCaseList(SC);
    PrevSC = SC;
  }
  return 2;
}

unsigned PCHStmtReader::VisitWhileStmt(WhileStmt *S) {
  VisitStmt(S);
  S->setCond(cast_or_null<Expr>(StmtStack[StmtStack.size() - 2]));
  S->setBody(StmtStack.back());
  S->setWhileLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 2;
}

unsigned PCHStmtReader::VisitDoStmt(DoStmt *S) {
  VisitStmt(S);
  S->setCond(cast_or_null<Expr>(StmtStack[StmtStack.size() - 2]));
  S->setBody(StmtStack.back());
  S->setDoLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 2;
}

unsigned PCHStmtReader::VisitForStmt(ForStmt *S) {
  VisitStmt(S);
  S->setInit(StmtStack[StmtStack.size() - 4]);
  S->setCond(cast_or_null<Expr>(StmtStack[StmtStack.size() - 3]));
  S->setInc(cast_or_null<Expr>(StmtStack[StmtStack.size() - 2]));
  S->setBody(StmtStack.back());
  S->setForLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 4;
}

unsigned PCHStmtReader::VisitGotoStmt(GotoStmt *S) {
  VisitStmt(S);
  Reader.SetLabelOf(S, Record[Idx++]);
  S->setGotoLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setLabelLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitIndirectGotoStmt(IndirectGotoStmt *S) {
  VisitStmt(S);
  S->setGotoLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setTarget(cast_or_null<Expr>(StmtStack.back()));
  return 1;
}

unsigned PCHStmtReader::VisitContinueStmt(ContinueStmt *S) {
  VisitStmt(S);
  S->setContinueLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitBreakStmt(BreakStmt *S) {
  VisitStmt(S);
  S->setBreakLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitReturnStmt(ReturnStmt *S) {
  VisitStmt(S);
  S->setRetValue(cast_or_null<Expr>(StmtStack.back()));
  S->setReturnLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 1;
}

unsigned PCHStmtReader::VisitDeclStmt(DeclStmt *S) {
  VisitStmt(S);
  S->setStartLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setEndLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));

  if (Idx + 1 == Record.size()) {
    // Single declaration
    S->setDeclGroup(DeclGroupRef(Reader.GetDecl(Record[Idx++])));
  } else {
    llvm::SmallVector<Decl *, 16> Decls;
    Decls.reserve(Record.size() - Idx);
    for (unsigned N = Record.size(); Idx != N; ++Idx)
      Decls.push_back(Reader.GetDecl(Record[Idx]));
    S->setDeclGroup(DeclGroupRef(DeclGroup::Create(Reader.getContext(),
                                                   &Decls[0], Decls.size())));
  }
  return 0;
}

unsigned PCHStmtReader::VisitAsmStmt(AsmStmt *S) {
  VisitStmt(S);
  unsigned NumOutputs = Record[Idx++];
  unsigned NumInputs = Record[Idx++];
  unsigned NumClobbers = Record[Idx++];
  S->setAsmLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setVolatile(Record[Idx++]);
  S->setSimple(Record[Idx++]);
  
  unsigned StackIdx 
    = StmtStack.size() - (NumOutputs*2 + NumInputs*2 + NumClobbers + 1);
  S->setAsmString(cast_or_null<StringLiteral>(StmtStack[StackIdx++]));

  // Outputs and inputs
  llvm::SmallVector<std::string, 16> Names;
  llvm::SmallVector<StringLiteral*, 16> Constraints;
  llvm::SmallVector<Stmt*, 16> Exprs;
  for (unsigned I = 0, N = NumOutputs + NumInputs; I != N; ++I) {
    Names.push_back(Reader.ReadString(Record, Idx));
    Constraints.push_back(cast_or_null<StringLiteral>(StmtStack[StackIdx++]));
    Exprs.push_back(StmtStack[StackIdx++]);
  }
  S->setOutputsAndInputs(NumOutputs, NumInputs,
                         &Names[0], &Constraints[0], &Exprs[0]);

  // Constraints
  llvm::SmallVector<StringLiteral*, 16> Clobbers;
  for (unsigned I = 0; I != NumClobbers; ++I)
    Clobbers.push_back(cast_or_null<StringLiteral>(StmtStack[StackIdx++]));
  S->setClobbers(&Clobbers[0], NumClobbers);

  assert(StackIdx == StmtStack.size() && "Error deserializing AsmStmt");
  return NumOutputs*2 + NumInputs*2 + NumClobbers + 1;
}

unsigned PCHStmtReader::VisitExpr(Expr *E) {
  VisitStmt(E);
  E->setType(Reader.GetType(Record[Idx++]));
  E->setTypeDependent(Record[Idx++]);
  E->setValueDependent(Record[Idx++]);
  assert(Idx == NumExprFields && "Incorrect expression field count");
  return 0;
}

unsigned PCHStmtReader::VisitPredefinedExpr(PredefinedExpr *E) {
  VisitExpr(E);
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setIdentType((PredefinedExpr::IdentType)Record[Idx++]);
  return 0;
}

unsigned PCHStmtReader::VisitDeclRefExpr(DeclRefExpr *E) {
  VisitExpr(E);
  E->setDecl(cast<NamedDecl>(Reader.GetDecl(Record[Idx++])));
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitIntegerLiteral(IntegerLiteral *E) {
  VisitExpr(E);
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setValue(Reader.ReadAPInt(Record, Idx));
  return 0;
}

unsigned PCHStmtReader::VisitFloatingLiteral(FloatingLiteral *E) {
  VisitExpr(E);
  E->setValue(Reader.ReadAPFloat(Record, Idx));
  E->setExact(Record[Idx++]);
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitImaginaryLiteral(ImaginaryLiteral *E) {
  VisitExpr(E);
  E->setSubExpr(cast<Expr>(StmtStack.back()));
  return 1;
}

unsigned PCHStmtReader::VisitStringLiteral(StringLiteral *E) {
  VisitExpr(E);
  unsigned Len = Record[Idx++];
  assert(Record[Idx] == E->getNumConcatenated() && 
         "Wrong number of concatenated tokens!");
  ++Idx;
  E->setWide(Record[Idx++]);

  // Read string data  
  llvm::SmallVector<char, 16> Str(&Record[Idx], &Record[Idx] + Len);
  E->setStrData(Reader.getContext(), &Str[0], Len);
  Idx += Len;

  // Read source locations
  for (unsigned I = 0, N = E->getNumConcatenated(); I != N; ++I)
    E->setStrTokenLoc(I, SourceLocation::getFromRawEncoding(Record[Idx++]));

  return 0;
}

unsigned PCHStmtReader::VisitCharacterLiteral(CharacterLiteral *E) {
  VisitExpr(E);
  E->setValue(Record[Idx++]);
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setWide(Record[Idx++]);
  return 0;
}

unsigned PCHStmtReader::VisitParenExpr(ParenExpr *E) {
  VisitExpr(E);
  E->setLParen(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParen(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setSubExpr(cast<Expr>(StmtStack.back()));
  return 1;
}

unsigned PCHStmtReader::VisitUnaryOperator(UnaryOperator *E) {
  VisitExpr(E);
  E->setSubExpr(cast<Expr>(StmtStack.back()));
  E->setOpcode((UnaryOperator::Opcode)Record[Idx++]);
  E->setOperatorLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 1;
}

unsigned PCHStmtReader::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E) {
  VisitExpr(E);
  E->setSizeof(Record[Idx++]);
  if (Record[Idx] == 0) {
    E->setArgument(cast<Expr>(StmtStack.back()));
    ++Idx;
  } else {
    E->setArgument(Reader.GetType(Record[Idx++]));
  }
  E->setOperatorLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return E->isArgumentType()? 0 : 1;
}

unsigned PCHStmtReader::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  VisitExpr(E);
  E->setLHS(cast<Expr>(StmtStack[StmtStack.size() - 2]));
  E->setRHS(cast<Expr>(StmtStack[StmtStack.size() - 2]));
  E->setRBracketLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 2;
}

unsigned PCHStmtReader::VisitCallExpr(CallExpr *E) {
  VisitExpr(E);
  E->setNumArgs(Reader.getContext(), Record[Idx++]);
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setCallee(cast<Expr>(StmtStack[StmtStack.size() - E->getNumArgs() - 1]));
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
    E->setArg(I, cast<Expr>(StmtStack[StmtStack.size() - N + I]));
  return E->getNumArgs() + 1;
}

unsigned PCHStmtReader::VisitMemberExpr(MemberExpr *E) {
  VisitExpr(E);
  E->setBase(cast<Expr>(StmtStack.back()));
  E->setMemberDecl(cast<NamedDecl>(Reader.GetDecl(Record[Idx++])));
  E->setMemberLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setArrow(Record[Idx++]);
  return 1;
}

unsigned PCHStmtReader::VisitCastExpr(CastExpr *E) {
  VisitExpr(E);
  E->setSubExpr(cast<Expr>(StmtStack.back()));
  return 1;
}

unsigned PCHStmtReader::VisitBinaryOperator(BinaryOperator *E) {
  VisitExpr(E);
  E->setLHS(cast<Expr>(StmtStack.end()[-2]));
  E->setRHS(cast<Expr>(StmtStack.end()[-1]));
  E->setOpcode((BinaryOperator::Opcode)Record[Idx++]);
  E->setOperatorLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 2;
}

unsigned PCHStmtReader::VisitCompoundAssignOperator(CompoundAssignOperator *E) {
  VisitBinaryOperator(E);
  E->setComputationLHSType(Reader.GetType(Record[Idx++]));
  E->setComputationResultType(Reader.GetType(Record[Idx++]));
  return 2;
}

unsigned PCHStmtReader::VisitConditionalOperator(ConditionalOperator *E) {
  VisitExpr(E);
  E->setCond(cast<Expr>(StmtStack[StmtStack.size() - 3]));
  E->setLHS(cast_or_null<Expr>(StmtStack[StmtStack.size() - 2]));
  E->setRHS(cast_or_null<Expr>(StmtStack[StmtStack.size() - 1]));
  return 3;
}

unsigned PCHStmtReader::VisitImplicitCastExpr(ImplicitCastExpr *E) {
  VisitCastExpr(E);
  E->setLvalueCast(Record[Idx++]);
  return 1;
}

unsigned PCHStmtReader::VisitExplicitCastExpr(ExplicitCastExpr *E) {
  VisitCastExpr(E);
  E->setTypeAsWritten(Reader.GetType(Record[Idx++]));
  return 1;
}

unsigned PCHStmtReader::VisitCStyleCastExpr(CStyleCastExpr *E) {
  VisitExplicitCastExpr(E);
  E->setLParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 1;
}

unsigned PCHStmtReader::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  VisitExpr(E);
  E->setLParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setInitializer(cast<Expr>(StmtStack.back()));
  E->setFileScope(Record[Idx++]);
  return 1;
}

unsigned PCHStmtReader::VisitExtVectorElementExpr(ExtVectorElementExpr *E) {
  VisitExpr(E);
  E->setBase(cast<Expr>(StmtStack.back()));
  E->setAccessor(Reader.GetIdentifierInfo(Record, Idx));
  E->setAccessorLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 1;
}

unsigned PCHStmtReader::VisitInitListExpr(InitListExpr *E) {
  VisitExpr(E);
  unsigned NumInits = Record[Idx++];
  E->reserveInits(NumInits);
  for (unsigned I = 0; I != NumInits; ++I)
    E->updateInit(I, 
                  cast<Expr>(StmtStack[StmtStack.size() - NumInits - 1 + I]));
  E->setSyntacticForm(cast_or_null<InitListExpr>(StmtStack.back()));
  E->setLBraceLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRBraceLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setInitializedFieldInUnion(
                      cast_or_null<FieldDecl>(Reader.GetDecl(Record[Idx++])));
  E->sawArrayRangeDesignator(Record[Idx++]);
  return NumInits + 1;
}

unsigned PCHStmtReader::VisitDesignatedInitExpr(DesignatedInitExpr *E) {
  typedef DesignatedInitExpr::Designator Designator;

  VisitExpr(E);
  unsigned NumSubExprs = Record[Idx++];
  assert(NumSubExprs == E->getNumSubExprs() && "Wrong number of subexprs");
  for (unsigned I = 0; I != NumSubExprs; ++I)
    E->setSubExpr(I, cast<Expr>(StmtStack[StmtStack.size() - NumSubExprs + I]));
  E->setEqualOrColonLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setGNUSyntax(Record[Idx++]);

  llvm::SmallVector<Designator, 4> Designators;
  while (Idx < Record.size()) {
    switch ((pch::DesignatorTypes)Record[Idx++]) {
    case pch::DESIG_FIELD_DECL: {
      FieldDecl *Field = cast<FieldDecl>(Reader.GetDecl(Record[Idx++]));
      SourceLocation DotLoc 
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      SourceLocation FieldLoc 
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      Designators.push_back(Designator(Field->getIdentifier(), DotLoc, 
                                       FieldLoc));
      Designators.back().setField(Field);
      break;
    }

    case pch::DESIG_FIELD_NAME: {
      const IdentifierInfo *Name = Reader.GetIdentifierInfo(Record, Idx);
      SourceLocation DotLoc 
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      SourceLocation FieldLoc 
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      Designators.push_back(Designator(Name, DotLoc, FieldLoc));
      break;
    }
      
    case pch::DESIG_ARRAY: {
      unsigned Index = Record[Idx++];
      SourceLocation LBracketLoc
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      SourceLocation RBracketLoc
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      Designators.push_back(Designator(Index, LBracketLoc, RBracketLoc));
      break;
    }

    case pch::DESIG_ARRAY_RANGE: {
      unsigned Index = Record[Idx++];
      SourceLocation LBracketLoc
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      SourceLocation EllipsisLoc
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      SourceLocation RBracketLoc
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      Designators.push_back(Designator(Index, LBracketLoc, EllipsisLoc,
                                       RBracketLoc));
      break;
    }
    }
  }
  E->setDesignators(&Designators[0], Designators.size());

  return NumSubExprs;
}

unsigned PCHStmtReader::VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
  VisitExpr(E);
  return 0;
}

unsigned PCHStmtReader::VisitVAArgExpr(VAArgExpr *E) {
  VisitExpr(E);
  E->setSubExpr(cast<Expr>(StmtStack.back()));
  E->setBuiltinLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 1;
}

unsigned PCHStmtReader::VisitAddrLabelExpr(AddrLabelExpr *E) {
  VisitExpr(E);
  E->setAmpAmpLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setLabelLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  Reader.SetLabelOf(E, Record[Idx++]);
  return 0;
}

unsigned PCHStmtReader::VisitStmtExpr(StmtExpr *E) {
  VisitExpr(E);
  E->setLParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setSubStmt(cast_or_null<CompoundStmt>(StmtStack.back()));
  return 1;
}

unsigned PCHStmtReader::VisitTypesCompatibleExpr(TypesCompatibleExpr *E) {
  VisitExpr(E);
  E->setArgType1(Reader.GetType(Record[Idx++]));
  E->setArgType2(Reader.GetType(Record[Idx++]));
  E->setBuiltinLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitChooseExpr(ChooseExpr *E) {
  VisitExpr(E);
  E->setCond(cast<Expr>(StmtStack[StmtStack.size() - 3]));
  E->setLHS(cast_or_null<Expr>(StmtStack[StmtStack.size() - 2]));
  E->setRHS(cast_or_null<Expr>(StmtStack[StmtStack.size() - 1]));
  E->setBuiltinLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 3;
}

unsigned PCHStmtReader::VisitGNUNullExpr(GNUNullExpr *E) {
  VisitExpr(E);
  E->setTokenLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitShuffleVectorExpr(ShuffleVectorExpr *E) {
  VisitExpr(E);
  unsigned NumExprs = Record[Idx++];
  E->setExprs((Expr **)&StmtStack[StmtStack.size() - NumExprs], NumExprs);
  E->setBuiltinLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return NumExprs;
}

unsigned PCHStmtReader::VisitBlockExpr(BlockExpr *E) {
  VisitExpr(E);
  E->setBlockDecl(cast_or_null<BlockDecl>(Reader.GetDecl(Record[Idx++])));
  E->setHasBlockDeclRefExprs(Record[Idx++]);
  return 0;
}

unsigned PCHStmtReader::VisitBlockDeclRefExpr(BlockDeclRefExpr *E) {
  VisitExpr(E);
  E->setDecl(cast<ValueDecl>(Reader.GetDecl(Record[Idx++])));
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setByRef(Record[Idx++]);
  return 0;
}

//===----------------------------------------------------------------------===//
// Objective-C Expressions and Statements

unsigned PCHStmtReader::VisitObjCStringLiteral(ObjCStringLiteral *E) {
  VisitExpr(E);
  E->setString(cast<StringLiteral>(StmtStack.back()));
  E->setAtLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 1;
}

unsigned PCHStmtReader::VisitObjCEncodeExpr(ObjCEncodeExpr *E) {
  VisitExpr(E);
  E->setEncodedType(Reader.GetType(Record[Idx++]));
  E->setAtLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitObjCSelectorExpr(ObjCSelectorExpr *E) {
  VisitExpr(E);
  // FIXME: Selectors.
  E->setAtLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitObjCProtocolExpr(ObjCProtocolExpr *E) {
  VisitExpr(E);
  E->setProtocol(cast<ObjCProtocolDecl>(Reader.GetDecl(Record[Idx++])));
  E->setAtLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}


//===----------------------------------------------------------------------===//
// PCH reader implementation
//===----------------------------------------------------------------------===//

namespace {
class VISIBILITY_HIDDEN PCHIdentifierLookupTrait {
  PCHReader &Reader;

  // If we know the IdentifierInfo in advance, it is here and we will
  // not build a new one. Used when deserializing information about an
  // identifier that was constructed before the PCH file was read.
  IdentifierInfo *KnownII;

public:
  typedef IdentifierInfo * data_type;

  typedef const std::pair<const char*, unsigned> external_key_type;

  typedef external_key_type internal_key_type;

  explicit PCHIdentifierLookupTrait(PCHReader &Reader, IdentifierInfo *II = 0) 
    : Reader(Reader), KnownII(II) { }
  
  static bool EqualKey(const internal_key_type& a,
                       const internal_key_type& b) {
    return (a.second == b.second) ? memcmp(a.first, b.first, a.second) == 0
                                  : false;
  }
  
  static unsigned ComputeHash(const internal_key_type& a) {
    return BernsteinHash(a.first, a.second);
  }
  
  // This hopefully will just get inlined and removed by the optimizer.
  static const internal_key_type&
  GetInternalKey(const external_key_type& x) { return x; }
  
  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char*& d) {
    using namespace clang::io;
    unsigned KeyLen = ReadUnalignedLE16(d);
    unsigned DataLen = ReadUnalignedLE16(d);
    return std::make_pair(KeyLen, DataLen);
  }
    
  static std::pair<const char*, unsigned>
  ReadKey(const unsigned char* d, unsigned n) {
    assert(n >= 2 && d[n-1] == '\0');
    return std::make_pair((const char*) d, n-1);
  }
    
  IdentifierInfo *ReadData(const internal_key_type& k, 
                           const unsigned char* d,
                           unsigned DataLen) {
    using namespace clang::io;
    uint32_t Bits = ReadUnalignedLE32(d); // FIXME: use these?
    bool CPlusPlusOperatorKeyword = Bits & 0x01;
    Bits >>= 1;
    bool Poisoned = Bits & 0x01;
    Bits >>= 1;
    bool ExtensionToken = Bits & 0x01;
    Bits >>= 1;
    bool hasMacroDefinition = Bits & 0x01;
    Bits >>= 1;
    unsigned ObjCOrBuiltinID = Bits & 0x3FF;
    Bits >>= 10;
    unsigned TokenID = Bits & 0xFF;
    Bits >>= 8;

    pch::IdentID ID = ReadUnalignedLE32(d);
    assert(Bits == 0 && "Extra bits in the identifier?");
    DataLen -= 8;

    // Build the IdentifierInfo itself and link the identifier ID with
    // the new IdentifierInfo.
    IdentifierInfo *II = KnownII;
    if (!II)
      II = &Reader.getIdentifierTable().CreateIdentifierInfo(
                                                 k.first, k.first + k.second);
    Reader.SetIdentifierInfo(ID, II);

    // Set or check the various bits in the IdentifierInfo structure.
    // FIXME: Load token IDs lazily, too?
    assert((unsigned)II->getTokenID() == TokenID && 
           "Incorrect token ID loaded"); 
    (void)TokenID;
    II->setObjCOrBuiltinID(ObjCOrBuiltinID);
    assert(II->isExtensionToken() == ExtensionToken && 
           "Incorrect extension token flag");
    (void)ExtensionToken;
    II->setIsPoisoned(Poisoned);
    assert(II->isCPlusPlusOperatorKeyword() == CPlusPlusOperatorKeyword &&
           "Incorrect C++ operator keyword flag");
    (void)CPlusPlusOperatorKeyword;

    // If this identifier is a macro, deserialize the macro
    // definition.
    if (hasMacroDefinition) {
      uint32_t Offset = ReadUnalignedLE64(d);
      Reader.ReadMacroRecord(Offset);
      DataLen -= 8;
    }

    // Read all of the declarations visible at global scope with this
    // name.
    Sema *SemaObj = Reader.getSema();
    while (DataLen > 0) {
      NamedDecl *D = cast<NamedDecl>(Reader.GetDecl(ReadUnalignedLE32(d)));

      if (SemaObj) {
        // Introduce this declaration into the translation-unit scope
        // and add it to the declaration chain for this identifier, so
        // that (unqualified) name lookup will find it.
        SemaObj->TUScope->AddDecl(Action::DeclPtrTy::make(D));
        SemaObj->IdResolver.AddDeclToIdentifierChain(II, D);
      } else {
        // Queue this declaration so that it will be added to the
        // translation unit scope and identifier's declaration chain
        // once a Sema object is known.
        // FIXME: This is a temporary hack. It will go away once we have
        // lazy deserialization of macros.
        Reader.TUDecls.push_back(D);
      }

      DataLen -= 4;
    }
    return II;
  }
};
  
} // end anonymous namespace  

/// \brief The on-disk hash table used to contain information about
/// all of the identifiers in the program.
typedef OnDiskChainedHashTable<PCHIdentifierLookupTrait> 
  PCHIdentifierLookupTable;

// FIXME: use the diagnostics machinery
static bool Error(const char *Str) {
  std::fprintf(stderr, "%s\n", Str);
  return true;
}

/// \brief Check the contents of the predefines buffer against the
/// contents of the predefines buffer used to build the PCH file.
///
/// The contents of the two predefines buffers should be the same. If
/// not, then some command-line option changed the preprocessor state
/// and we must reject the PCH file.
///
/// \param PCHPredef The start of the predefines buffer in the PCH
/// file.
///
/// \param PCHPredefLen The length of the predefines buffer in the PCH
/// file.
///
/// \param PCHBufferID The FileID for the PCH predefines buffer.
///
/// \returns true if there was a mismatch (in which case the PCH file
/// should be ignored), or false otherwise.
bool PCHReader::CheckPredefinesBuffer(const char *PCHPredef, 
                                      unsigned PCHPredefLen,
                                      FileID PCHBufferID) {
  const char *Predef = PP.getPredefines().c_str();
  unsigned PredefLen = PP.getPredefines().size();

  // If the two predefines buffers compare equal, we're done!.
  if (PredefLen == PCHPredefLen && 
      strncmp(Predef, PCHPredef, PCHPredefLen) == 0)
    return false;
  
  // The predefines buffers are different. Produce a reasonable
  // diagnostic showing where they are different.

  // The source locations (potentially in the two different predefines
  // buffers)
  SourceLocation Loc1, Loc2;
  SourceManager &SourceMgr = PP.getSourceManager();

  // Create a source buffer for our predefines string, so
  // that we can build a diagnostic that points into that
  // source buffer.
  FileID BufferID;
  if (Predef && Predef[0]) {
    llvm::MemoryBuffer *Buffer
      = llvm::MemoryBuffer::getMemBuffer(Predef, Predef + PredefLen,
                                         "<built-in>");
    BufferID = SourceMgr.createFileIDForMemBuffer(Buffer);
  }

  unsigned MinLen = std::min(PredefLen, PCHPredefLen);
  std::pair<const char *, const char *> Locations
    = std::mismatch(Predef, Predef + MinLen, PCHPredef); 
 
  if (Locations.first != Predef + MinLen) {
    // We found the location in the two buffers where there is a
    // difference. Form source locations to point there (in both
    // buffers).
    unsigned Offset = Locations.first - Predef;
    Loc1 = SourceMgr.getLocForStartOfFile(BufferID)
             .getFileLocWithOffset(Offset);
    Loc2 = SourceMgr.getLocForStartOfFile(PCHBufferID)
             .getFileLocWithOffset(Offset);
  } else if (PredefLen > PCHPredefLen) {
    Loc1 = SourceMgr.getLocForStartOfFile(BufferID)
             .getFileLocWithOffset(MinLen);
  } else {
    Loc1 = SourceMgr.getLocForStartOfFile(PCHBufferID)
             .getFileLocWithOffset(MinLen);
  }
  
  Diag(Loc1, diag::warn_pch_preprocessor);
  if (Loc2.isValid())
    Diag(Loc2, diag::note_predef_in_pch);
  Diag(diag::note_ignoring_pch) << FileName;
  return true;
}

/// \brief Read the line table in the source manager block.
/// \returns true if ther was an error.
static bool ParseLineTable(SourceManager &SourceMgr, 
                           llvm::SmallVectorImpl<uint64_t> &Record) {
  unsigned Idx = 0;
  LineTableInfo &LineTable = SourceMgr.getLineTable();

  // Parse the file names
  std::map<int, int> FileIDs;
  for (int I = 0, N = Record[Idx++]; I != N; ++I) {
    // Extract the file name
    unsigned FilenameLen = Record[Idx++];
    std::string Filename(&Record[Idx], &Record[Idx] + FilenameLen);
    Idx += FilenameLen;
    FileIDs[I] = LineTable.getLineTableFilenameID(Filename.c_str(), 
                                                  Filename.size());
  }

  // Parse the line entries
  std::vector<LineEntry> Entries;
  while (Idx < Record.size()) {
    int FID = FileIDs[Record[Idx++]];

    // Extract the line entries
    unsigned NumEntries = Record[Idx++];
    Entries.clear();
    Entries.reserve(NumEntries);
    for (unsigned I = 0; I != NumEntries; ++I) {
      unsigned FileOffset = Record[Idx++];
      unsigned LineNo = Record[Idx++];
      int FilenameID = Record[Idx++];
      SrcMgr::CharacteristicKind FileKind 
        = (SrcMgr::CharacteristicKind)Record[Idx++];
      unsigned IncludeOffset = Record[Idx++];
      Entries.push_back(LineEntry::get(FileOffset, LineNo, FilenameID,
                                       FileKind, IncludeOffset));
    }
    LineTable.AddEntry(FID, Entries);
  }

  return false;
}

/// \brief Read the source manager block
PCHReader::PCHReadResult PCHReader::ReadSourceManagerBlock() {
  using namespace SrcMgr;
  if (Stream.EnterSubBlock(pch::SOURCE_MANAGER_BLOCK_ID)) {
    Error("Malformed source manager block record");
    return Failure;
  }

  SourceManager &SourceMgr = Context.getSourceManager();
  RecordData Record;
  while (true) {
    unsigned Code = Stream.ReadCode();
    if (Code == llvm::bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd()) {
        Error("Error at end of Source Manager block");
        return Failure;
      }

      return Success;
    }
    
    if (Code == llvm::bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      Stream.ReadSubBlockID();
      if (Stream.SkipBlock()) {
        Error("Malformed block record");
        return Failure;
      }
      continue;
    }
    
    if (Code == llvm::bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }
    
    // Read a record.
    const char *BlobStart;
    unsigned BlobLen;
    Record.clear();
    switch (Stream.ReadRecord(Code, Record, &BlobStart, &BlobLen)) {
    default:  // Default behavior: ignore.
      break;

    case pch::SM_SLOC_FILE_ENTRY: {
      // FIXME: We would really like to delay the creation of this
      // FileEntry until it is actually required, e.g., when producing
      // a diagnostic with a source location in this file.
      const FileEntry *File 
        = PP.getFileManager().getFile(BlobStart, BlobStart + BlobLen);
      // FIXME: Error recovery if file cannot be found.
      FileID ID = SourceMgr.createFileID(File,
                                SourceLocation::getFromRawEncoding(Record[1]),
                                         (CharacteristicKind)Record[2]);
      if (Record[3])
        const_cast<SrcMgr::FileInfo&>(SourceMgr.getSLocEntry(ID).getFile())
          .setHasLineDirectives();
      break;
    }

    case pch::SM_SLOC_BUFFER_ENTRY: {
      const char *Name = BlobStart;
      unsigned Code = Stream.ReadCode();
      Record.clear();
      unsigned RecCode = Stream.ReadRecord(Code, Record, &BlobStart, &BlobLen);
      assert(RecCode == pch::SM_SLOC_BUFFER_BLOB && "Ill-formed PCH file");
      (void)RecCode;
      llvm::MemoryBuffer *Buffer
        = llvm::MemoryBuffer::getMemBuffer(BlobStart, 
                                           BlobStart + BlobLen - 1,
                                           Name);
      FileID BufferID = SourceMgr.createFileIDForMemBuffer(Buffer);

      if (strcmp(Name, "<built-in>") == 0
          && CheckPredefinesBuffer(BlobStart, BlobLen - 1, BufferID))
        return IgnorePCH;
      break;
    }

    case pch::SM_SLOC_INSTANTIATION_ENTRY: {
      SourceLocation SpellingLoc 
        = SourceLocation::getFromRawEncoding(Record[1]);
      SourceMgr.createInstantiationLoc(
                              SpellingLoc,
                              SourceLocation::getFromRawEncoding(Record[2]),
                              SourceLocation::getFromRawEncoding(Record[3]),
                              Record[4]);
      break;
    }

    case pch::SM_LINE_TABLE:
      if (ParseLineTable(SourceMgr, Record))
        return Failure;
      break;
    }
  }
}

void PCHReader::ReadMacroRecord(uint64_t Offset) {
  // Keep track of where we are in the stream, then jump back there
  // after reading this macro.
  SavedStreamPosition SavedPosition(Stream);

  Stream.JumpToBit(Offset);
  RecordData Record;
  llvm::SmallVector<IdentifierInfo*, 16> MacroArgs;
  MacroInfo *Macro = 0;
  while (true) {
    unsigned Code = Stream.ReadCode();
    switch (Code) {
    case llvm::bitc::END_BLOCK:
      return;

    case llvm::bitc::ENTER_SUBBLOCK:
      // No known subblocks, always skip them.
      Stream.ReadSubBlockID();
      if (Stream.SkipBlock()) {
        Error("Malformed block record");
        return;
      }
      continue;
    
    case llvm::bitc::DEFINE_ABBREV:
      Stream.ReadAbbrevRecord();
      continue;
    default: break;
    }

    // Read a record.
    Record.clear();
    pch::PreprocessorRecordTypes RecType =
      (pch::PreprocessorRecordTypes)Stream.ReadRecord(Code, Record);
    switch (RecType) {
    case pch::PP_COUNTER_VALUE:
      // Skip this record.
      break;

    case pch::PP_MACRO_OBJECT_LIKE:
    case pch::PP_MACRO_FUNCTION_LIKE: {
      // If we already have a macro, that means that we've hit the end
      // of the definition of the macro we were looking for. We're
      // done.
      if (Macro)
        return;

      IdentifierInfo *II = DecodeIdentifierInfo(Record[0]);
      if (II == 0) {
        Error("Macro must have a name");
        return;
      }
      SourceLocation Loc = SourceLocation::getFromRawEncoding(Record[1]);
      bool isUsed = Record[2];
      
      MacroInfo *MI = PP.AllocateMacroInfo(Loc);
      MI->setIsUsed(isUsed);
      
      if (RecType == pch::PP_MACRO_FUNCTION_LIKE) {
        // Decode function-like macro info.
        bool isC99VarArgs = Record[3];
        bool isGNUVarArgs = Record[4];
        MacroArgs.clear();
        unsigned NumArgs = Record[5];
        for (unsigned i = 0; i != NumArgs; ++i)
          MacroArgs.push_back(DecodeIdentifierInfo(Record[6+i]));

        // Install function-like macro info.
        MI->setIsFunctionLike();
        if (isC99VarArgs) MI->setIsC99Varargs();
        if (isGNUVarArgs) MI->setIsGNUVarargs();
        MI->setArgumentList(&MacroArgs[0], MacroArgs.size(),
                            PP.getPreprocessorAllocator());
      }

      // Finally, install the macro.
      PP.setMacroInfo(II, MI);

      // Remember that we saw this macro last so that we add the tokens that
      // form its body to it.
      Macro = MI;
      ++NumMacrosRead;
      break;
    }
        
    case pch::PP_TOKEN: {
      // If we see a TOKEN before a PP_MACRO_*, then the file is
      // erroneous, just pretend we didn't see this.
      if (Macro == 0) break;
      
      Token Tok;
      Tok.startToken();
      Tok.setLocation(SourceLocation::getFromRawEncoding(Record[0]));
      Tok.setLength(Record[1]);
      if (IdentifierInfo *II = DecodeIdentifierInfo(Record[2]))
        Tok.setIdentifierInfo(II);
      Tok.setKind((tok::TokenKind)Record[3]);
      Tok.setFlag((Token::TokenFlags)Record[4]);
      Macro->AddTokenToBody(Tok);
      break;
    }
    }
  }
}

bool PCHReader::ReadPreprocessorBlock() {
  if (Stream.EnterSubBlock(pch::PREPROCESSOR_BLOCK_ID))
    return Error("Malformed preprocessor block record");
  
  RecordData Record;
  while (true) {
    unsigned Code = Stream.ReadCode();
    switch (Code) {
    case llvm::bitc::END_BLOCK:
      if (Stream.ReadBlockEnd())
        return Error("Error at end of preprocessor block");
      return false;
    
    case llvm::bitc::ENTER_SUBBLOCK:
      // No known subblocks, always skip them.
      Stream.ReadSubBlockID();
      if (Stream.SkipBlock())
        return Error("Malformed block record");
      continue;
    
    case llvm::bitc::DEFINE_ABBREV:
      Stream.ReadAbbrevRecord();
      continue;
    default: break;
    }
    
    // Read a record.
    Record.clear();
    pch::PreprocessorRecordTypes RecType =
      (pch::PreprocessorRecordTypes)Stream.ReadRecord(Code, Record);
    switch (RecType) {
    default:  // Default behavior: ignore unknown records.
      break;
    case pch::PP_COUNTER_VALUE:
      if (!Record.empty())
        PP.setCounterValue(Record[0]);
      break;

    case pch::PP_MACRO_OBJECT_LIKE:
    case pch::PP_MACRO_FUNCTION_LIKE:
    case pch::PP_TOKEN:
      // Once we've hit a macro definition or a token, we're done.
      return false;
    }
  }
}

PCHReader::PCHReadResult 
PCHReader::ReadPCHBlock(uint64_t &PreprocessorBlockOffset) {
  if (Stream.EnterSubBlock(pch::PCH_BLOCK_ID)) {
    Error("Malformed block record");
    return Failure;
  }

  // Read all of the records and blocks for the PCH file.
  RecordData Record;
  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();
    if (Code == llvm::bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd()) {
        Error("Error at end of module block");
        return Failure;
      }

      return Success;
    }

    if (Code == llvm::bitc::ENTER_SUBBLOCK) {
      switch (Stream.ReadSubBlockID()) {
      case pch::DECLS_BLOCK_ID: // Skip decls block (lazily loaded)
      case pch::TYPES_BLOCK_ID: // Skip types block (lazily loaded)
      default:  // Skip unknown content.
        if (Stream.SkipBlock()) {
          Error("Malformed block record");
          return Failure;
        }
        break;

      case pch::PREPROCESSOR_BLOCK_ID:
        // Skip the preprocessor block for now, but remember where it is.  We
        // want to read it in after the identifier table.
        if (PreprocessorBlockOffset) {
          Error("Multiple preprocessor blocks found.");
          return Failure;
        }
        PreprocessorBlockOffset = Stream.GetCurrentBitNo();
        if (Stream.SkipBlock()) {
          Error("Malformed block record");
          return Failure;
        }
        break;
          
      case pch::SOURCE_MANAGER_BLOCK_ID:
        switch (ReadSourceManagerBlock()) {
        case Success:
          break;

        case Failure:
          Error("Malformed source manager block");
          return Failure;

        case IgnorePCH:
          return IgnorePCH;
        }
        break;
      }
      continue;
    }

    if (Code == llvm::bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }

    // Read and process a record.
    Record.clear();
    const char *BlobStart = 0;
    unsigned BlobLen = 0;
    switch ((pch::PCHRecordTypes)Stream.ReadRecord(Code, Record, 
                                                   &BlobStart, &BlobLen)) {
    default:  // Default behavior: ignore.
      break;

    case pch::TYPE_OFFSET:
      if (!TypeOffsets.empty()) {
        Error("Duplicate TYPE_OFFSET record in PCH file");
        return Failure;
      }
      TypeOffsets.swap(Record);
      TypeAlreadyLoaded.resize(TypeOffsets.size(), false);
      break;

    case pch::DECL_OFFSET:
      if (!DeclOffsets.empty()) {
        Error("Duplicate DECL_OFFSET record in PCH file");
        return Failure;
      }
      DeclOffsets.swap(Record);
      DeclAlreadyLoaded.resize(DeclOffsets.size(), false);
      break;

    case pch::LANGUAGE_OPTIONS:
      if (ParseLanguageOptions(Record))
        return IgnorePCH;
      break;

    case pch::TARGET_TRIPLE: {
      std::string TargetTriple(BlobStart, BlobLen);
      if (TargetTriple != Context.Target.getTargetTriple()) {
        Diag(diag::warn_pch_target_triple)
          << TargetTriple << Context.Target.getTargetTriple();
        Diag(diag::note_ignoring_pch) << FileName;
        return IgnorePCH;
      }
      break;
    }

    case pch::IDENTIFIER_TABLE:
      IdentifierTableData = BlobStart;
      IdentifierLookupTable 
        = PCHIdentifierLookupTable::Create(
                        (const unsigned char *)IdentifierTableData + Record[0],
                        (const unsigned char *)IdentifierTableData, 
                        PCHIdentifierLookupTrait(*this));
      // FIXME: What about any identifiers already placed into the
      // identifier table? Should we load decls with those names now?
      PP.getIdentifierTable().setExternalIdentifierLookup(this);
      break;

    case pch::IDENTIFIER_OFFSET:
      if (!IdentifierData.empty()) {
        Error("Duplicate IDENTIFIER_OFFSET record in PCH file");
        return Failure;
      }
      IdentifierData.swap(Record);
#ifndef NDEBUG
      for (unsigned I = 0, N = IdentifierData.size(); I != N; ++I) {
        if ((IdentifierData[I] & 0x01) == 0) {
          Error("Malformed identifier table in the precompiled header");
          return Failure;
        }
      }
#endif
      break;

    case pch::EXTERNAL_DEFINITIONS:
      if (!ExternalDefinitions.empty()) {
        Error("Duplicate EXTERNAL_DEFINITIONS record in PCH file");
        return Failure;
      }
      ExternalDefinitions.swap(Record);
      break;

    case pch::SPECIAL_TYPES:
      SpecialTypes.swap(Record);
      break;

    case pch::STATISTICS:
      TotalNumStatements = Record[0];
      TotalNumMacros = Record[1];
      break;

    }
  }

  Error("Premature end of bitstream");
  return Failure;
}

PCHReader::PCHReadResult PCHReader::ReadPCH(const std::string &FileName) {
  // Set the PCH file name.
  this->FileName = FileName;

  // Open the PCH file.
  std::string ErrStr;
  Buffer.reset(llvm::MemoryBuffer::getFile(FileName.c_str(), &ErrStr));
  if (!Buffer) {
    Error(ErrStr.c_str());
    return IgnorePCH;
  }

  // Initialize the stream
  Stream.init((const unsigned char *)Buffer->getBufferStart(), 
              (const unsigned char *)Buffer->getBufferEnd());

  // Sniff for the signature.
  if (Stream.Read(8) != 'C' ||
      Stream.Read(8) != 'P' ||
      Stream.Read(8) != 'C' ||
      Stream.Read(8) != 'H') {
    Error("Not a PCH file");
    return IgnorePCH;
  }

  // We expect a number of well-defined blocks, though we don't necessarily
  // need to understand them all.
  uint64_t PreprocessorBlockOffset = 0;
  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();
    
    if (Code != llvm::bitc::ENTER_SUBBLOCK) {
      Error("Invalid record at top-level");
      return Failure;
    }

    unsigned BlockID = Stream.ReadSubBlockID();

    // We only know the PCH subblock ID.
    switch (BlockID) {
    case llvm::bitc::BLOCKINFO_BLOCK_ID:
      if (Stream.ReadBlockInfoBlock()) {
        Error("Malformed BlockInfoBlock");
        return Failure;
      }
      break;
    case pch::PCH_BLOCK_ID:
      switch (ReadPCHBlock(PreprocessorBlockOffset)) {
      case Success:
        break;

      case Failure:
        return Failure;

      case IgnorePCH:
        // FIXME: We could consider reading through to the end of this
        // PCH block, skipping subblocks, to see if there are other
        // PCH blocks elsewhere.
        return IgnorePCH;
      }
      break;
    default:
      if (Stream.SkipBlock()) {
        Error("Malformed block record");
        return Failure;
      }
      break;
    }
  }  

  // Load the translation unit declaration
  ReadDeclRecord(DeclOffsets[0], 0);

  // Initialization of builtins and library builtins occurs before the
  // PCH file is read, so there may be some identifiers that were
  // loaded into the IdentifierTable before we intercepted the
  // creation of identifiers. Iterate through the list of known
  // identifiers and determine whether we have to establish
  // preprocessor definitions or top-level identifier declaration
  // chains for those identifiers.
  //
  // We copy the IdentifierInfo pointers to a small vector first,
  // since de-serializing declarations or macro definitions can add
  // new entries into the identifier table, invalidating the
  // iterators.
  llvm::SmallVector<IdentifierInfo *, 128> Identifiers;
  for (IdentifierTable::iterator Id = PP.getIdentifierTable().begin(),
                              IdEnd = PP.getIdentifierTable().end();
       Id != IdEnd; ++Id)
    Identifiers.push_back(Id->second);
  PCHIdentifierLookupTable *IdTable 
    = (PCHIdentifierLookupTable *)IdentifierLookupTable;
  for (unsigned I = 0, N = Identifiers.size(); I != N; ++I) {
    IdentifierInfo *II = Identifiers[I];
    // Look in the on-disk hash table for an entry for
    PCHIdentifierLookupTrait Info(*this, II);
    std::pair<const char*, unsigned> Key(II->getName(), II->getLength());
    PCHIdentifierLookupTable::iterator Pos = IdTable->find(Key, &Info);
    if (Pos == IdTable->end())
      continue;

    // Dereferencing the iterator has the effect of populating the
    // IdentifierInfo node with the various declarations it needs.
    (void)*Pos;
  }

  // Load the special types.
  Context.setBuiltinVaListType(
    GetType(SpecialTypes[pch::SPECIAL_TYPE_BUILTIN_VA_LIST]));

  // If we saw the preprocessor block, read it now.
  if (PreprocessorBlockOffset) {
    SavedStreamPosition SavedPos(Stream);
    Stream.JumpToBit(PreprocessorBlockOffset);
    if (ReadPreprocessorBlock()) {
      Error("Malformed preprocessor block");
      return Failure;
    }
  }

  return Success;
}

/// \brief Parse the record that corresponds to a LangOptions data
/// structure.
///
/// This routine compares the language options used to generate the
/// PCH file against the language options set for the current
/// compilation. For each option, we classify differences between the
/// two compiler states as either "benign" or "important". Benign
/// differences don't matter, and we accept them without complaint
/// (and without modifying the language options). Differences between
/// the states for important options cause the PCH file to be
/// unusable, so we emit a warning and return true to indicate that
/// there was an error.
///
/// \returns true if the PCH file is unacceptable, false otherwise.
bool PCHReader::ParseLanguageOptions(
                             const llvm::SmallVectorImpl<uint64_t> &Record) {
  const LangOptions &LangOpts = Context.getLangOptions();
#define PARSE_LANGOPT_BENIGN(Option) ++Idx
#define PARSE_LANGOPT_IMPORTANT(Option, DiagID)                 \
  if (Record[Idx] != LangOpts.Option) {                         \
    Diag(DiagID) << (unsigned)Record[Idx] << LangOpts.Option;   \
    Diag(diag::note_ignoring_pch) << FileName;                  \
    return true;                                                \
  }                                                             \
  ++Idx

  unsigned Idx = 0;
  PARSE_LANGOPT_BENIGN(Trigraphs);
  PARSE_LANGOPT_BENIGN(BCPLComment);
  PARSE_LANGOPT_BENIGN(DollarIdents);
  PARSE_LANGOPT_BENIGN(AsmPreprocessor);
  PARSE_LANGOPT_IMPORTANT(GNUMode, diag::warn_pch_gnu_extensions);
  PARSE_LANGOPT_BENIGN(ImplicitInt);
  PARSE_LANGOPT_BENIGN(Digraphs);
  PARSE_LANGOPT_BENIGN(HexFloats);
  PARSE_LANGOPT_IMPORTANT(C99, diag::warn_pch_c99);
  PARSE_LANGOPT_IMPORTANT(Microsoft, diag::warn_pch_microsoft_extensions);
  PARSE_LANGOPT_IMPORTANT(CPlusPlus, diag::warn_pch_cplusplus);
  PARSE_LANGOPT_IMPORTANT(CPlusPlus0x, diag::warn_pch_cplusplus0x);
  PARSE_LANGOPT_IMPORTANT(NoExtensions, diag::warn_pch_extensions);
  PARSE_LANGOPT_BENIGN(CXXOperatorName);
  PARSE_LANGOPT_IMPORTANT(ObjC1, diag::warn_pch_objective_c);
  PARSE_LANGOPT_IMPORTANT(ObjC2, diag::warn_pch_objective_c2);
  PARSE_LANGOPT_IMPORTANT(ObjCNonFragileABI, diag::warn_pch_nonfragile_abi);
  PARSE_LANGOPT_BENIGN(PascalStrings);
  PARSE_LANGOPT_BENIGN(Boolean);
  PARSE_LANGOPT_BENIGN(WritableStrings);
  PARSE_LANGOPT_IMPORTANT(LaxVectorConversions, 
                          diag::warn_pch_lax_vector_conversions);
  PARSE_LANGOPT_IMPORTANT(Exceptions, diag::warn_pch_exceptions);
  PARSE_LANGOPT_IMPORTANT(NeXTRuntime, diag::warn_pch_objc_runtime);
  PARSE_LANGOPT_IMPORTANT(Freestanding, diag::warn_pch_freestanding);
  PARSE_LANGOPT_IMPORTANT(NoBuiltin, diag::warn_pch_builtins);
  PARSE_LANGOPT_IMPORTANT(ThreadsafeStatics, 
                          diag::warn_pch_thread_safe_statics);
  PARSE_LANGOPT_IMPORTANT(Blocks, diag::warn_pch_blocks);
  PARSE_LANGOPT_BENIGN(EmitAllDecls);
  PARSE_LANGOPT_IMPORTANT(MathErrno, diag::warn_pch_math_errno);
  PARSE_LANGOPT_IMPORTANT(OverflowChecking, diag::warn_pch_overflow_checking);
  PARSE_LANGOPT_IMPORTANT(HeinousExtensions, 
                          diag::warn_pch_heinous_extensions);
  // FIXME: Most of the options below are benign if the macro wasn't
  // used. Unfortunately, this means that a PCH compiled without
  // optimization can't be used with optimization turned on, even
  // though the only thing that changes is whether __OPTIMIZE__ was
  // defined... but if __OPTIMIZE__ never showed up in the header, it
  // doesn't matter. We could consider making this some special kind
  // of check.
  PARSE_LANGOPT_IMPORTANT(Optimize, diag::warn_pch_optimize);
  PARSE_LANGOPT_IMPORTANT(OptimizeSize, diag::warn_pch_optimize_size);
  PARSE_LANGOPT_IMPORTANT(Static, diag::warn_pch_static);
  PARSE_LANGOPT_IMPORTANT(PICLevel, diag::warn_pch_pic_level);
  PARSE_LANGOPT_IMPORTANT(GNUInline, diag::warn_pch_gnu_inline);
  PARSE_LANGOPT_IMPORTANT(NoInline, diag::warn_pch_no_inline);
  if ((LangOpts.getGCMode() != 0) != (Record[Idx] != 0)) {
    Diag(diag::warn_pch_gc_mode) 
      << (unsigned)Record[Idx] << LangOpts.getGCMode();
    Diag(diag::note_ignoring_pch) << FileName;
    return true;
  }
  ++Idx;
  PARSE_LANGOPT_BENIGN(getVisibilityMode());
  PARSE_LANGOPT_BENIGN(InstantiationDepth);
#undef PARSE_LANGOPT_IRRELEVANT
#undef PARSE_LANGOPT_BENIGN

  return false;
}

/// \brief Read and return the type at the given offset.
///
/// This routine actually reads the record corresponding to the type
/// at the given offset in the bitstream. It is a helper routine for
/// GetType, which deals with reading type IDs.
QualType PCHReader::ReadTypeRecord(uint64_t Offset) {
  // Keep track of where we are in the stream, then jump back there
  // after reading this type.
  SavedStreamPosition SavedPosition(Stream);

  Stream.JumpToBit(Offset);
  RecordData Record;
  unsigned Code = Stream.ReadCode();
  switch ((pch::TypeCode)Stream.ReadRecord(Code, Record)) {
  case pch::TYPE_EXT_QUAL: {
    assert(Record.size() == 3 && 
           "Incorrect encoding of extended qualifier type");
    QualType Base = GetType(Record[0]);
    QualType::GCAttrTypes GCAttr = (QualType::GCAttrTypes)Record[1];
    unsigned AddressSpace = Record[2];
    
    QualType T = Base;
    if (GCAttr != QualType::GCNone)
      T = Context.getObjCGCQualType(T, GCAttr);
    if (AddressSpace)
      T = Context.getAddrSpaceQualType(T, AddressSpace);
    return T;
  }

  case pch::TYPE_FIXED_WIDTH_INT: {
    assert(Record.size() == 2 && "Incorrect encoding of fixed-width int type");
    return Context.getFixedWidthIntType(Record[0], Record[1]);
  }

  case pch::TYPE_COMPLEX: {
    assert(Record.size() == 1 && "Incorrect encoding of complex type");
    QualType ElemType = GetType(Record[0]);
    return Context.getComplexType(ElemType);
  }

  case pch::TYPE_POINTER: {
    assert(Record.size() == 1 && "Incorrect encoding of pointer type");
    QualType PointeeType = GetType(Record[0]);
    return Context.getPointerType(PointeeType);
  }

  case pch::TYPE_BLOCK_POINTER: {
    assert(Record.size() == 1 && "Incorrect encoding of block pointer type");
    QualType PointeeType = GetType(Record[0]);
    return Context.getBlockPointerType(PointeeType);
  }

  case pch::TYPE_LVALUE_REFERENCE: {
    assert(Record.size() == 1 && "Incorrect encoding of lvalue reference type");
    QualType PointeeType = GetType(Record[0]);
    return Context.getLValueReferenceType(PointeeType);
  }

  case pch::TYPE_RVALUE_REFERENCE: {
    assert(Record.size() == 1 && "Incorrect encoding of rvalue reference type");
    QualType PointeeType = GetType(Record[0]);
    return Context.getRValueReferenceType(PointeeType);
  }

  case pch::TYPE_MEMBER_POINTER: {
    assert(Record.size() == 1 && "Incorrect encoding of member pointer type");
    QualType PointeeType = GetType(Record[0]);
    QualType ClassType = GetType(Record[1]);
    return Context.getMemberPointerType(PointeeType, ClassType.getTypePtr());
  }

  case pch::TYPE_CONSTANT_ARRAY: {
    QualType ElementType = GetType(Record[0]);
    ArrayType::ArraySizeModifier ASM = (ArrayType::ArraySizeModifier)Record[1];
    unsigned IndexTypeQuals = Record[2];
    unsigned Idx = 3;
    llvm::APInt Size = ReadAPInt(Record, Idx);
    return Context.getConstantArrayType(ElementType, Size, ASM, IndexTypeQuals);
  }

  case pch::TYPE_INCOMPLETE_ARRAY: {
    QualType ElementType = GetType(Record[0]);
    ArrayType::ArraySizeModifier ASM = (ArrayType::ArraySizeModifier)Record[1];
    unsigned IndexTypeQuals = Record[2];
    return Context.getIncompleteArrayType(ElementType, ASM, IndexTypeQuals);
  }

  case pch::TYPE_VARIABLE_ARRAY: {
    QualType ElementType = GetType(Record[0]);
    ArrayType::ArraySizeModifier ASM = (ArrayType::ArraySizeModifier)Record[1];
    unsigned IndexTypeQuals = Record[2];
    return Context.getVariableArrayType(ElementType, ReadExpr(),
                                        ASM, IndexTypeQuals);
  }

  case pch::TYPE_VECTOR: {
    if (Record.size() != 2) {
      Error("Incorrect encoding of vector type in PCH file");
      return QualType();
    }

    QualType ElementType = GetType(Record[0]);
    unsigned NumElements = Record[1];
    return Context.getVectorType(ElementType, NumElements);
  }

  case pch::TYPE_EXT_VECTOR: {
    if (Record.size() != 2) {
      Error("Incorrect encoding of extended vector type in PCH file");
      return QualType();
    }

    QualType ElementType = GetType(Record[0]);
    unsigned NumElements = Record[1];
    return Context.getExtVectorType(ElementType, NumElements);
  }

  case pch::TYPE_FUNCTION_NO_PROTO: {
    if (Record.size() != 1) {
      Error("Incorrect encoding of no-proto function type");
      return QualType();
    }
    QualType ResultType = GetType(Record[0]);
    return Context.getFunctionNoProtoType(ResultType);
  }

  case pch::TYPE_FUNCTION_PROTO: {
    QualType ResultType = GetType(Record[0]);
    unsigned Idx = 1;
    unsigned NumParams = Record[Idx++];
    llvm::SmallVector<QualType, 16> ParamTypes;
    for (unsigned I = 0; I != NumParams; ++I)
      ParamTypes.push_back(GetType(Record[Idx++]));
    bool isVariadic = Record[Idx++];
    unsigned Quals = Record[Idx++];
    return Context.getFunctionType(ResultType, &ParamTypes[0], NumParams,
                                   isVariadic, Quals);
  }

  case pch::TYPE_TYPEDEF:
    assert(Record.size() == 1 && "Incorrect encoding of typedef type");
    return Context.getTypeDeclType(cast<TypedefDecl>(GetDecl(Record[0])));

  case pch::TYPE_TYPEOF_EXPR:
    return Context.getTypeOfExprType(ReadExpr());

  case pch::TYPE_TYPEOF: {
    if (Record.size() != 1) {
      Error("Incorrect encoding of typeof(type) in PCH file");
      return QualType();
    }
    QualType UnderlyingType = GetType(Record[0]);
    return Context.getTypeOfType(UnderlyingType);
  }
    
  case pch::TYPE_RECORD:
    assert(Record.size() == 1 && "Incorrect encoding of record type");
    return Context.getTypeDeclType(cast<RecordDecl>(GetDecl(Record[0])));

  case pch::TYPE_ENUM:
    assert(Record.size() == 1 && "Incorrect encoding of enum type");
    return Context.getTypeDeclType(cast<EnumDecl>(GetDecl(Record[0])));

  case pch::TYPE_OBJC_INTERFACE:
    assert(Record.size() == 1 && "Incorrect encoding of objc interface type");
    return Context.getObjCInterfaceType(
                                  cast<ObjCInterfaceDecl>(GetDecl(Record[0])));

  case pch::TYPE_OBJC_QUALIFIED_INTERFACE: {
    unsigned Idx = 0;
    ObjCInterfaceDecl *ItfD = cast<ObjCInterfaceDecl>(GetDecl(Record[Idx++]));
    unsigned NumProtos = Record[Idx++];
    llvm::SmallVector<ObjCProtocolDecl*, 4> Protos;
    for (unsigned I = 0; I != NumProtos; ++I)
      Protos.push_back(cast<ObjCProtocolDecl>(GetDecl(Record[Idx++])));
    return Context.getObjCQualifiedInterfaceType(ItfD, &Protos[0], NumProtos);
  }

  case pch::TYPE_OBJC_QUALIFIED_ID: {
    unsigned Idx = 0;
    unsigned NumProtos = Record[Idx++];
    llvm::SmallVector<ObjCProtocolDecl*, 4> Protos;
    for (unsigned I = 0; I != NumProtos; ++I)
      Protos.push_back(cast<ObjCProtocolDecl>(GetDecl(Record[Idx++])));
    return Context.getObjCQualifiedIdType(&Protos[0], NumProtos);
  }
  }
  // Suppress a GCC warning
  return QualType();
}

/// \brief Note that we have loaded the declaration with the given
/// Index.
/// 
/// This routine notes that this declaration has already been loaded,
/// so that future GetDecl calls will return this declaration rather
/// than trying to load a new declaration.
inline void PCHReader::LoadedDecl(unsigned Index, Decl *D) {
  assert(!DeclAlreadyLoaded[Index] && "Decl loaded twice?");
  DeclAlreadyLoaded[Index] = true;
  DeclOffsets[Index] = reinterpret_cast<uint64_t>(D);
}

/// \brief Read the declaration at the given offset from the PCH file.
Decl *PCHReader::ReadDeclRecord(uint64_t Offset, unsigned Index) {
  // Keep track of where we are in the stream, then jump back there
  // after reading this declaration.
  SavedStreamPosition SavedPosition(Stream);

  Decl *D = 0;
  Stream.JumpToBit(Offset);
  RecordData Record;
  unsigned Code = Stream.ReadCode();
  unsigned Idx = 0;
  PCHDeclReader Reader(*this, Record, Idx);

  switch ((pch::DeclCode)Stream.ReadRecord(Code, Record)) {
  case pch::DECL_ATTR:
  case pch::DECL_CONTEXT_LEXICAL:
  case pch::DECL_CONTEXT_VISIBLE:
    assert(false && "Record cannot be de-serialized with ReadDeclRecord");
    break;

  case pch::DECL_TRANSLATION_UNIT:
    assert(Index == 0 && "Translation unit must be at index 0");
    D = Context.getTranslationUnitDecl();
    break;

  case pch::DECL_TYPEDEF: {
    D = TypedefDecl::Create(Context, 0, SourceLocation(), 0, QualType());
    break;
  }

  case pch::DECL_ENUM: {
    D = EnumDecl::Create(Context, 0, SourceLocation(), 0, 0);
    break;
  }

  case pch::DECL_RECORD: {
    D = RecordDecl::Create(Context, TagDecl::TK_struct, 0, SourceLocation(), 
                           0, 0);
    break;
  }

  case pch::DECL_ENUM_CONSTANT: {
    D = EnumConstantDecl::Create(Context, 0, SourceLocation(), 0, QualType(),
                                 0, llvm::APSInt());
    break;
  }
  
  case pch::DECL_FUNCTION: {
    D = FunctionDecl::Create(Context, 0, SourceLocation(), DeclarationName(), 
                             QualType());
    break;
  }

  case pch::DECL_OBJC_METHOD: {
    D = ObjCMethodDecl::Create(Context, SourceLocation(), SourceLocation(), 
                               Selector(), QualType(), 0);
    break;
  }

  case pch::DECL_OBJC_INTERFACE: {
    D = ObjCInterfaceDecl::Create(Context, 0, SourceLocation(), 0);
    break;
  }

  case pch::DECL_OBJC_IVAR: {
    D = ObjCIvarDecl::Create(Context, 0, SourceLocation(), 0, QualType(),
                             ObjCIvarDecl::None);
    break;
  }

  case pch::DECL_OBJC_PROTOCOL: {
    D = ObjCProtocolDecl::Create(Context, 0, SourceLocation(), 0);
    break;
  }

  case pch::DECL_OBJC_AT_DEFS_FIELD: {
    D = ObjCAtDefsFieldDecl::Create(Context, 0, SourceLocation(), 0, 
                                    QualType(), 0);
    break;
  }

  case pch::DECL_OBJC_CLASS: {
    D = ObjCClassDecl::Create(Context, 0, SourceLocation());
    break;
  }

  case pch::DECL_OBJC_FORWARD_PROTOCOL: {
    D = ObjCForwardProtocolDecl::Create(Context, 0, SourceLocation());
    break;
  }

  case pch::DECL_OBJC_CATEGORY: {
    D = ObjCCategoryDecl::Create(Context, 0, SourceLocation(), 0);
    break;
  }

  case pch::DECL_OBJC_CATEGORY_IMPL: {
    // FIXME: Implement.
    break;
  }
  
  case pch::DECL_OBJC_IMPLEMENTATION: {
    // FIXME: Implement.
    break;
  }
  
  case pch::DECL_OBJC_COMPATIBLE_ALIAS: {
    // FIXME: Implement.
    break;
  }
  
  case pch::DECL_OBJC_PROPERTY: {
    // FIXME: Implement.
    break;
  }
  
  case pch::DECL_OBJC_PROPERTY_IMPL: {
    // FIXME: Implement.
    break;
  }

  case pch::DECL_FIELD: {
    D = FieldDecl::Create(Context, 0, SourceLocation(), 0, QualType(), 0, 
                          false);
    break;
  }

  case pch::DECL_VAR: {
    D = VarDecl::Create(Context, 0, SourceLocation(), 0, QualType(),
                        VarDecl::None, SourceLocation());
    break;
  }

  case pch::DECL_PARM_VAR: {
    D = ParmVarDecl::Create(Context, 0, SourceLocation(), 0, QualType(), 
                            VarDecl::None, 0);
    break;
  }

  case pch::DECL_ORIGINAL_PARM_VAR: {
    D = OriginalParmVarDecl::Create(Context, 0, SourceLocation(), 0,
                                    QualType(), QualType(), VarDecl::None, 
                                    0);
    break;
  }

  case pch::DECL_FILE_SCOPE_ASM: {
    D = FileScopeAsmDecl::Create(Context, 0, SourceLocation(), 0);
    break;
  }

  case pch::DECL_BLOCK: {
    D = BlockDecl::Create(Context, 0, SourceLocation());
    break;
  }
  }

  assert(D && "Unknown declaration reading PCH file");
  if (D) {
    LoadedDecl(Index, D);
    Reader.Visit(D);
  }

  // If this declaration is also a declaration context, get the
  // offsets for its tables of lexical and visible declarations.
  if (DeclContext *DC = dyn_cast<DeclContext>(D)) {
    std::pair<uint64_t, uint64_t> Offsets = Reader.VisitDeclContext(DC);
    if (Offsets.first || Offsets.second) {
      DC->setHasExternalLexicalStorage(Offsets.first != 0);
      DC->setHasExternalVisibleStorage(Offsets.second != 0);
      DeclContextOffsets[DC] = Offsets;
    }
  }
  assert(Idx == Record.size());

  if (Consumer) {
    // If we have deserialized a declaration that has a definition the
    // AST consumer might need to know about, notify the consumer
    // about that definition now.
    if (VarDecl *Var = dyn_cast<VarDecl>(D)) {
      if (Var->isFileVarDecl() && Var->getInit()) {
        DeclGroupRef DG(Var);
        Consumer->HandleTopLevelDecl(DG);
      }
    } else if (FunctionDecl *Func = dyn_cast<FunctionDecl>(D)) {
      if (Func->isThisDeclarationADefinition()) {
        DeclGroupRef DG(Func);
        Consumer->HandleTopLevelDecl(DG);
      }
    }
  }

  return D;
}

QualType PCHReader::GetType(pch::TypeID ID) {
  unsigned Quals = ID & 0x07; 
  unsigned Index = ID >> 3;

  if (Index < pch::NUM_PREDEF_TYPE_IDS) {
    QualType T;
    switch ((pch::PredefinedTypeIDs)Index) {
    case pch::PREDEF_TYPE_NULL_ID: return QualType();
    case pch::PREDEF_TYPE_VOID_ID: T = Context.VoidTy; break;
    case pch::PREDEF_TYPE_BOOL_ID: T = Context.BoolTy; break;

    case pch::PREDEF_TYPE_CHAR_U_ID:
    case pch::PREDEF_TYPE_CHAR_S_ID:
      // FIXME: Check that the signedness of CharTy is correct!
      T = Context.CharTy;
      break;

    case pch::PREDEF_TYPE_UCHAR_ID:      T = Context.UnsignedCharTy;     break;
    case pch::PREDEF_TYPE_USHORT_ID:     T = Context.UnsignedShortTy;    break;
    case pch::PREDEF_TYPE_UINT_ID:       T = Context.UnsignedIntTy;      break;
    case pch::PREDEF_TYPE_ULONG_ID:      T = Context.UnsignedLongTy;     break;
    case pch::PREDEF_TYPE_ULONGLONG_ID:  T = Context.UnsignedLongLongTy; break;
    case pch::PREDEF_TYPE_SCHAR_ID:      T = Context.SignedCharTy;       break;
    case pch::PREDEF_TYPE_WCHAR_ID:      T = Context.WCharTy;            break;
    case pch::PREDEF_TYPE_SHORT_ID:      T = Context.ShortTy;            break;
    case pch::PREDEF_TYPE_INT_ID:        T = Context.IntTy;              break;
    case pch::PREDEF_TYPE_LONG_ID:       T = Context.LongTy;             break;
    case pch::PREDEF_TYPE_LONGLONG_ID:   T = Context.LongLongTy;         break;
    case pch::PREDEF_TYPE_FLOAT_ID:      T = Context.FloatTy;            break;
    case pch::PREDEF_TYPE_DOUBLE_ID:     T = Context.DoubleTy;           break;
    case pch::PREDEF_TYPE_LONGDOUBLE_ID: T = Context.LongDoubleTy;       break;
    case pch::PREDEF_TYPE_OVERLOAD_ID:   T = Context.OverloadTy;         break;
    case pch::PREDEF_TYPE_DEPENDENT_ID:  T = Context.DependentTy;        break;
    }

    assert(!T.isNull() && "Unknown predefined type");
    return T.getQualifiedType(Quals);
  }

  Index -= pch::NUM_PREDEF_TYPE_IDS;
  if (!TypeAlreadyLoaded[Index]) {
    // Load the type from the PCH file.
    TypeOffsets[Index] = reinterpret_cast<uint64_t>(
                             ReadTypeRecord(TypeOffsets[Index]).getTypePtr());
    TypeAlreadyLoaded[Index] = true;
  }
    
  return QualType(reinterpret_cast<Type *>(TypeOffsets[Index]), Quals);
}

Decl *PCHReader::GetDecl(pch::DeclID ID) {
  if (ID == 0)
    return 0;

  unsigned Index = ID - 1;
  if (DeclAlreadyLoaded[Index])
    return reinterpret_cast<Decl *>(DeclOffsets[Index]);

  // Load the declaration from the PCH file.
  return ReadDeclRecord(DeclOffsets[Index], Index);
}

Stmt *PCHReader::GetStmt(uint64_t Offset) {
  // Keep track of where we are in the stream, then jump back there
  // after reading this declaration.
  SavedStreamPosition SavedPosition(Stream);

  Stream.JumpToBit(Offset);
  return ReadStmt();
}

bool PCHReader::ReadDeclsLexicallyInContext(DeclContext *DC,
                                  llvm::SmallVectorImpl<pch::DeclID> &Decls) {
  assert(DC->hasExternalLexicalStorage() && 
         "DeclContext has no lexical decls in storage");
  uint64_t Offset = DeclContextOffsets[DC].first;
  assert(Offset && "DeclContext has no lexical decls in storage");

  // Keep track of where we are in the stream, then jump back there
  // after reading this context.
  SavedStreamPosition SavedPosition(Stream);

  // Load the record containing all of the declarations lexically in
  // this context.
  Stream.JumpToBit(Offset);
  RecordData Record;
  unsigned Code = Stream.ReadCode();
  unsigned RecCode = Stream.ReadRecord(Code, Record);
  (void)RecCode;
  assert(RecCode == pch::DECL_CONTEXT_LEXICAL && "Expected lexical block");

  // Load all of the declaration IDs
  Decls.clear();
  Decls.insert(Decls.end(), Record.begin(), Record.end());
  return false;
}

bool PCHReader::ReadDeclsVisibleInContext(DeclContext *DC,
                           llvm::SmallVectorImpl<VisibleDeclaration> & Decls) {
  assert(DC->hasExternalVisibleStorage() && 
         "DeclContext has no visible decls in storage");
  uint64_t Offset = DeclContextOffsets[DC].second;
  assert(Offset && "DeclContext has no visible decls in storage");

  // Keep track of where we are in the stream, then jump back there
  // after reading this context.
  SavedStreamPosition SavedPosition(Stream);

  // Load the record containing all of the declarations visible in
  // this context.
  Stream.JumpToBit(Offset);
  RecordData Record;
  unsigned Code = Stream.ReadCode();
  unsigned RecCode = Stream.ReadRecord(Code, Record);
  (void)RecCode;
  assert(RecCode == pch::DECL_CONTEXT_VISIBLE && "Expected visible block");
  if (Record.size() == 0)
    return false;  

  Decls.clear();

  unsigned Idx = 0;
  while (Idx < Record.size()) {
    Decls.push_back(VisibleDeclaration());
    Decls.back().Name = ReadDeclarationName(Record, Idx);

    unsigned Size = Record[Idx++];
    llvm::SmallVector<unsigned, 4> & LoadedDecls
      = Decls.back().Declarations;
    LoadedDecls.reserve(Size);
    for (unsigned I = 0; I < Size; ++I)
      LoadedDecls.push_back(Record[Idx++]);
  }

  return false;
}

void PCHReader::StartTranslationUnit(ASTConsumer *Consumer) {
  this->Consumer = Consumer;

  if (!Consumer)
    return;

  for (unsigned I = 0, N = ExternalDefinitions.size(); I != N; ++I) {
    Decl *D = GetDecl(ExternalDefinitions[I]);
    DeclGroupRef DG(D);
    Consumer->HandleTopLevelDecl(DG);
  }
}

void PCHReader::PrintStats() {
  std::fprintf(stderr, "*** PCH Statistics:\n");

  unsigned NumTypesLoaded = std::count(TypeAlreadyLoaded.begin(),
                                       TypeAlreadyLoaded.end(),
                                       true);
  unsigned NumDeclsLoaded = std::count(DeclAlreadyLoaded.begin(),
                                       DeclAlreadyLoaded.end(),
                                       true);
  unsigned NumIdentifiersLoaded = 0;
  for (unsigned I = 0; I < IdentifierData.size(); ++I) {
    if ((IdentifierData[I] & 0x01) == 0)
      ++NumIdentifiersLoaded;
  }

  std::fprintf(stderr, "  %u/%u types read (%f%%)\n",
               NumTypesLoaded, (unsigned)TypeAlreadyLoaded.size(),
               ((float)NumTypesLoaded/TypeAlreadyLoaded.size() * 100));
  std::fprintf(stderr, "  %u/%u declarations read (%f%%)\n",
               NumDeclsLoaded, (unsigned)DeclAlreadyLoaded.size(),
               ((float)NumDeclsLoaded/DeclAlreadyLoaded.size() * 100));
  std::fprintf(stderr, "  %u/%u identifiers read (%f%%)\n",
               NumIdentifiersLoaded, (unsigned)IdentifierData.size(),
               ((float)NumIdentifiersLoaded/IdentifierData.size() * 100));
  std::fprintf(stderr, "  %u/%u statements read (%f%%)\n",
               NumStatementsRead, TotalNumStatements,
               ((float)NumStatementsRead/TotalNumStatements * 100));
  std::fprintf(stderr, "  %u/%u macros read (%f%%)\n",
               NumMacrosRead, TotalNumMacros,
               ((float)NumMacrosRead/TotalNumMacros * 100));
  std::fprintf(stderr, "\n");
}

void PCHReader::InitializeSema(Sema &S) {
  SemaObj = &S;
 
  // FIXME: this makes sure any declarations that were deserialized
  // "too early" still get added to the identifier's declaration
  // chains.
  for (unsigned I = 0, N = TUDecls.size(); I != N; ++I) {
    SemaObj->TUScope->AddDecl(Action::DeclPtrTy::make(TUDecls[I]));
    SemaObj->IdResolver.AddDecl(TUDecls[I]);
  }
  TUDecls.clear();
}

IdentifierInfo* PCHReader::get(const char *NameStart, const char *NameEnd) {
  // Try to find this name within our on-disk hash table
  PCHIdentifierLookupTable *IdTable 
    = (PCHIdentifierLookupTable *)IdentifierLookupTable;
  std::pair<const char*, unsigned> Key(NameStart, NameEnd - NameStart);
  PCHIdentifierLookupTable::iterator Pos = IdTable->find(Key);
  if (Pos == IdTable->end())
    return 0;

  // Dereferencing the iterator has the effect of building the
  // IdentifierInfo node and populating it with the various
  // declarations it needs.
  return *Pos;
}

void PCHReader::SetIdentifierInfo(unsigned ID, const IdentifierInfo *II) {
  assert(ID && "Non-zero identifier ID required");
  IdentifierData[ID - 1] = reinterpret_cast<uint64_t>(II);
}

IdentifierInfo *PCHReader::DecodeIdentifierInfo(unsigned ID) {
  if (ID == 0)
    return 0;
  
  if (!IdentifierTableData || IdentifierData.empty()) {
    Error("No identifier table in PCH file");
    return 0;
  }
  
  if (IdentifierData[ID - 1] & 0x01) {
    uint64_t Offset = IdentifierData[ID - 1] >> 1;
    IdentifierData[ID - 1] = reinterpret_cast<uint64_t>(
                               &Context.Idents.get(IdentifierTableData + Offset));
  }
  
  return reinterpret_cast<IdentifierInfo *>(IdentifierData[ID - 1]);
}

DeclarationName 
PCHReader::ReadDeclarationName(const RecordData &Record, unsigned &Idx) {
  DeclarationName::NameKind Kind = (DeclarationName::NameKind)Record[Idx++];
  switch (Kind) {
  case DeclarationName::Identifier:
    return DeclarationName(GetIdentifierInfo(Record, Idx));

  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    assert(false && "Unable to de-serialize Objective-C selectors");
    break;

  case DeclarationName::CXXConstructorName:
    return Context.DeclarationNames.getCXXConstructorName(
                                                      GetType(Record[Idx++]));

  case DeclarationName::CXXDestructorName:
    return Context.DeclarationNames.getCXXDestructorName(
                                                      GetType(Record[Idx++]));

  case DeclarationName::CXXConversionFunctionName:
    return Context.DeclarationNames.getCXXConversionFunctionName(
                                                      GetType(Record[Idx++]));

  case DeclarationName::CXXOperatorName:
    return Context.DeclarationNames.getCXXOperatorName(
                                       (OverloadedOperatorKind)Record[Idx++]);

  case DeclarationName::CXXUsingDirective:
    return DeclarationName::getUsingDirectiveName();
  }

  // Required to silence GCC warning
  return DeclarationName();
}

/// \brief Read an integral value
llvm::APInt PCHReader::ReadAPInt(const RecordData &Record, unsigned &Idx) {
  unsigned BitWidth = Record[Idx++];
  unsigned NumWords = llvm::APInt::getNumWords(BitWidth);
  llvm::APInt Result(BitWidth, NumWords, &Record[Idx]);
  Idx += NumWords;
  return Result;
}

/// \brief Read a signed integral value
llvm::APSInt PCHReader::ReadAPSInt(const RecordData &Record, unsigned &Idx) {
  bool isUnsigned = Record[Idx++];
  return llvm::APSInt(ReadAPInt(Record, Idx), isUnsigned);
}

/// \brief Read a floating-point value
llvm::APFloat PCHReader::ReadAPFloat(const RecordData &Record, unsigned &Idx) {
  return llvm::APFloat(ReadAPInt(Record, Idx));
}

// \brief Read a string
std::string PCHReader::ReadString(const RecordData &Record, unsigned &Idx) {
  unsigned Len = Record[Idx++];
  std::string Result(&Record[Idx], &Record[Idx] + Len);
  Idx += Len;
  return Result;
}

/// \brief Reads attributes from the current stream position.
Attr *PCHReader::ReadAttributes() {
  unsigned Code = Stream.ReadCode();
  assert(Code == llvm::bitc::UNABBREV_RECORD && 
         "Expected unabbreviated record"); (void)Code;
  
  RecordData Record;
  unsigned Idx = 0;
  unsigned RecCode = Stream.ReadRecord(Code, Record);
  assert(RecCode == pch::DECL_ATTR && "Expected attribute record"); 
  (void)RecCode;

#define SIMPLE_ATTR(Name)                       \
 case Attr::Name:                               \
   New = ::new (Context) Name##Attr();          \
   break

#define STRING_ATTR(Name)                                       \
 case Attr::Name:                                               \
   New = ::new (Context) Name##Attr(ReadString(Record, Idx));   \
   break

#define UNSIGNED_ATTR(Name)                             \
 case Attr::Name:                                       \
   New = ::new (Context) Name##Attr(Record[Idx++]);     \
   break

  Attr *Attrs = 0;
  while (Idx < Record.size()) {
    Attr *New = 0;
    Attr::Kind Kind = (Attr::Kind)Record[Idx++];
    bool IsInherited = Record[Idx++];

    switch (Kind) {
    STRING_ATTR(Alias);
    UNSIGNED_ATTR(Aligned);
    SIMPLE_ATTR(AlwaysInline);
    SIMPLE_ATTR(AnalyzerNoReturn);
    STRING_ATTR(Annotate);
    STRING_ATTR(AsmLabel);
    
    case Attr::Blocks:
      New = ::new (Context) BlocksAttr(
                                  (BlocksAttr::BlocksAttrTypes)Record[Idx++]);
      break;
      
    case Attr::Cleanup:
      New = ::new (Context) CleanupAttr(
                                  cast<FunctionDecl>(GetDecl(Record[Idx++])));
      break;

    SIMPLE_ATTR(Const);
    UNSIGNED_ATTR(Constructor);
    SIMPLE_ATTR(DLLExport);
    SIMPLE_ATTR(DLLImport);
    SIMPLE_ATTR(Deprecated);
    UNSIGNED_ATTR(Destructor);
    SIMPLE_ATTR(FastCall);
    
    case Attr::Format: {
      std::string Type = ReadString(Record, Idx);
      unsigned FormatIdx = Record[Idx++];
      unsigned FirstArg = Record[Idx++];
      New = ::new (Context) FormatAttr(Type, FormatIdx, FirstArg);
      break;
    }

    SIMPLE_ATTR(GNUInline);
    
    case Attr::IBOutletKind:
      New = ::new (Context) IBOutletAttr();
      break;

    SIMPLE_ATTR(NoReturn);
    SIMPLE_ATTR(NoThrow);
    SIMPLE_ATTR(Nodebug);
    SIMPLE_ATTR(Noinline);
    
    case Attr::NonNull: {
      unsigned Size = Record[Idx++];
      llvm::SmallVector<unsigned, 16> ArgNums;
      ArgNums.insert(ArgNums.end(), &Record[Idx], &Record[Idx] + Size);
      Idx += Size;
      New = ::new (Context) NonNullAttr(&ArgNums[0], Size);
      break;
    }

    SIMPLE_ATTR(ObjCException);
    SIMPLE_ATTR(ObjCNSObject);
    SIMPLE_ATTR(Overloadable);
    UNSIGNED_ATTR(Packed);
    SIMPLE_ATTR(Pure);
    UNSIGNED_ATTR(Regparm);
    STRING_ATTR(Section);
    SIMPLE_ATTR(StdCall);
    SIMPLE_ATTR(TransparentUnion);
    SIMPLE_ATTR(Unavailable);
    SIMPLE_ATTR(Unused);
    SIMPLE_ATTR(Used);
    
    case Attr::Visibility:
      New = ::new (Context) VisibilityAttr(
                              (VisibilityAttr::VisibilityTypes)Record[Idx++]);
      break;

    SIMPLE_ATTR(WarnUnusedResult);
    SIMPLE_ATTR(Weak);
    SIMPLE_ATTR(WeakImport);
    }

    assert(New && "Unable to decode attribute?");
    New->setInherited(IsInherited);
    New->setNext(Attrs);
    Attrs = New;
  }
#undef UNSIGNED_ATTR
#undef STRING_ATTR
#undef SIMPLE_ATTR

  // The list of attributes was built backwards. Reverse the list
  // before returning it.
  Attr *PrevAttr = 0, *NextAttr = 0;
  while (Attrs) {
    NextAttr = Attrs->getNext();
    Attrs->setNext(PrevAttr);
    PrevAttr = Attrs;
    Attrs = NextAttr;
  }

  return PrevAttr;
}

Stmt *PCHReader::ReadStmt() {
  // Within the bitstream, expressions are stored in Reverse Polish
  // Notation, with each of the subexpressions preceding the
  // expression they are stored in. To evaluate expressions, we
  // continue reading expressions and placing them on the stack, with
  // expressions having operands removing those operands from the
  // stack. Evaluation terminates when we see a STMT_STOP record, and
  // the single remaining expression on the stack is our result.
  RecordData Record;
  unsigned Idx;
  llvm::SmallVector<Stmt *, 16> StmtStack;
  PCHStmtReader Reader(*this, Record, Idx, StmtStack);
  Stmt::EmptyShell Empty;

  while (true) {
    unsigned Code = Stream.ReadCode();
    if (Code == llvm::bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd()) {
        Error("Error at end of Source Manager block");
        return 0;
      }
      break;
    }

    if (Code == llvm::bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      Stream.ReadSubBlockID();
      if (Stream.SkipBlock()) {
        Error("Malformed block record");
        return 0;
      }
      continue;
    }

    if (Code == llvm::bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }

    Stmt *S = 0;
    Idx = 0;
    Record.clear();
    bool Finished = false;
    switch ((pch::StmtCode)Stream.ReadRecord(Code, Record)) {
    case pch::STMT_STOP:
      Finished = true;
      break;

    case pch::STMT_NULL_PTR: 
      S = 0; 
      break;

    case pch::STMT_NULL:
      S = new (Context) NullStmt(Empty);
      break;

    case pch::STMT_COMPOUND:
      S = new (Context) CompoundStmt(Empty);
      break;

    case pch::STMT_CASE:
      S = new (Context) CaseStmt(Empty);
      break;

    case pch::STMT_DEFAULT:
      S = new (Context) DefaultStmt(Empty);
      break;

    case pch::STMT_LABEL:
      S = new (Context) LabelStmt(Empty);
      break;

    case pch::STMT_IF:
      S = new (Context) IfStmt(Empty);
      break;

    case pch::STMT_SWITCH:
      S = new (Context) SwitchStmt(Empty);
      break;

    case pch::STMT_WHILE:
      S = new (Context) WhileStmt(Empty);
      break;

    case pch::STMT_DO:
      S = new (Context) DoStmt(Empty);
      break;
      
    case pch::STMT_FOR:
      S = new (Context) ForStmt(Empty);
      break;

    case pch::STMT_GOTO:
      S = new (Context) GotoStmt(Empty);
      break;
      
    case pch::STMT_INDIRECT_GOTO:
      S = new (Context) IndirectGotoStmt(Empty);
      break;

    case pch::STMT_CONTINUE:
      S = new (Context) ContinueStmt(Empty);
      break;

    case pch::STMT_BREAK:
      S = new (Context) BreakStmt(Empty);
      break;

    case pch::STMT_RETURN:
      S = new (Context) ReturnStmt(Empty);
      break;

    case pch::STMT_DECL:
      S = new (Context) DeclStmt(Empty);
      break;

    case pch::STMT_ASM:
      S = new (Context) AsmStmt(Empty);
      break;

    case pch::EXPR_PREDEFINED:
      S = new (Context) PredefinedExpr(Empty);
      break;
      
    case pch::EXPR_DECL_REF: 
      S = new (Context) DeclRefExpr(Empty); 
      break;
      
    case pch::EXPR_INTEGER_LITERAL: 
      S = new (Context) IntegerLiteral(Empty);
      break;
      
    case pch::EXPR_FLOATING_LITERAL:
      S = new (Context) FloatingLiteral(Empty);
      break;
      
    case pch::EXPR_IMAGINARY_LITERAL:
      S = new (Context) ImaginaryLiteral(Empty);
      break;

    case pch::EXPR_STRING_LITERAL:
      S = StringLiteral::CreateEmpty(Context, 
                                     Record[PCHStmtReader::NumExprFields + 1]);
      break;

    case pch::EXPR_CHARACTER_LITERAL:
      S = new (Context) CharacterLiteral(Empty);
      break;

    case pch::EXPR_PAREN:
      S = new (Context) ParenExpr(Empty);
      break;

    case pch::EXPR_UNARY_OPERATOR:
      S = new (Context) UnaryOperator(Empty);
      break;

    case pch::EXPR_SIZEOF_ALIGN_OF:
      S = new (Context) SizeOfAlignOfExpr(Empty);
      break;

    case pch::EXPR_ARRAY_SUBSCRIPT:
      S = new (Context) ArraySubscriptExpr(Empty);
      break;

    case pch::EXPR_CALL:
      S = new (Context) CallExpr(Context, Empty);
      break;

    case pch::EXPR_MEMBER:
      S = new (Context) MemberExpr(Empty);
      break;

    case pch::EXPR_BINARY_OPERATOR:
      S = new (Context) BinaryOperator(Empty);
      break;

    case pch::EXPR_COMPOUND_ASSIGN_OPERATOR:
      S = new (Context) CompoundAssignOperator(Empty);
      break;

    case pch::EXPR_CONDITIONAL_OPERATOR:
      S = new (Context) ConditionalOperator(Empty);
      break;

    case pch::EXPR_IMPLICIT_CAST:
      S = new (Context) ImplicitCastExpr(Empty);
      break;

    case pch::EXPR_CSTYLE_CAST:
      S = new (Context) CStyleCastExpr(Empty);
      break;

    case pch::EXPR_COMPOUND_LITERAL:
      S = new (Context) CompoundLiteralExpr(Empty);
      break;

    case pch::EXPR_EXT_VECTOR_ELEMENT:
      S = new (Context) ExtVectorElementExpr(Empty);
      break;

    case pch::EXPR_INIT_LIST:
      S = new (Context) InitListExpr(Empty);
      break;

    case pch::EXPR_DESIGNATED_INIT:
      S = DesignatedInitExpr::CreateEmpty(Context, 
                                     Record[PCHStmtReader::NumExprFields] - 1);
     
      break;

    case pch::EXPR_IMPLICIT_VALUE_INIT:
      S = new (Context) ImplicitValueInitExpr(Empty);
      break;

    case pch::EXPR_VA_ARG:
      S = new (Context) VAArgExpr(Empty);
      break;

    case pch::EXPR_ADDR_LABEL:
      S = new (Context) AddrLabelExpr(Empty);
      break;

    case pch::EXPR_STMT:
      S = new (Context) StmtExpr(Empty);
      break;

    case pch::EXPR_TYPES_COMPATIBLE:
      S = new (Context) TypesCompatibleExpr(Empty);
      break;

    case pch::EXPR_CHOOSE:
      S = new (Context) ChooseExpr(Empty);
      break;

    case pch::EXPR_GNU_NULL:
      S = new (Context) GNUNullExpr(Empty);
      break;

    case pch::EXPR_SHUFFLE_VECTOR:
      S = new (Context) ShuffleVectorExpr(Empty);
      break;
      
    case pch::EXPR_BLOCK:
      S = new (Context) BlockExpr(Empty);
      break;

    case pch::EXPR_BLOCK_DECL_REF:
      S = new (Context) BlockDeclRefExpr(Empty);
      break;
        
    case pch::EXPR_OBJC_STRING_LITERAL:
      S = new (Context) ObjCStringLiteral(Empty);
      break;
    case pch::EXPR_OBJC_ENCODE:
      S = new (Context) ObjCEncodeExpr(Empty);
      break;
    case pch::EXPR_OBJC_SELECTOR_EXPR:
      S = new (Context) ObjCSelectorExpr(Empty);
      break;
    case pch::EXPR_OBJC_PROTOCOL_EXPR:
      S = new (Context) ObjCProtocolExpr(Empty);
      break;
    }

    // We hit a STMT_STOP, so we're done with this expression.
    if (Finished)
      break;

    ++NumStatementsRead;

    if (S) {
      unsigned NumSubStmts = Reader.Visit(S);
      while (NumSubStmts > 0) {
        StmtStack.pop_back();
        --NumSubStmts;
      }
    }

    assert(Idx == Record.size() && "Invalid deserialization of statement");
    StmtStack.push_back(S);
  }
  assert(StmtStack.size() == 1 && "Extra expressions on stack!");
  SwitchCaseStmts.clear();
  return StmtStack.back();
}

Expr *PCHReader::ReadExpr() {
  return dyn_cast_or_null<Expr>(ReadStmt());
}

DiagnosticBuilder PCHReader::Diag(unsigned DiagID) {
  return Diag(SourceLocation(), DiagID);
}

DiagnosticBuilder PCHReader::Diag(SourceLocation Loc, unsigned DiagID) {
  return PP.getDiagnostics().Report(FullSourceLoc(Loc,
                                                  Context.getSourceManager()),
                                    DiagID);
}

/// \brief Retrieve the identifier table associated with the
/// preprocessor.
IdentifierTable &PCHReader::getIdentifierTable() {
  return PP.getIdentifierTable();
}

/// \brief Record that the given ID maps to the given switch-case
/// statement.
void PCHReader::RecordSwitchCaseID(SwitchCase *SC, unsigned ID) {
  assert(SwitchCaseStmts[ID] == 0 && "Already have a SwitchCase with this ID");
  SwitchCaseStmts[ID] = SC;
}

/// \brief Retrieve the switch-case statement with the given ID.
SwitchCase *PCHReader::getSwitchCaseWithID(unsigned ID) {
  assert(SwitchCaseStmts[ID] != 0 && "No SwitchCase with this ID");
  return SwitchCaseStmts[ID];
}

/// \brief Record that the given label statement has been
/// deserialized and has the given ID.
void PCHReader::RecordLabelStmt(LabelStmt *S, unsigned ID) {
  assert(LabelStmts.find(ID) == LabelStmts.end() && 
         "Deserialized label twice");
  LabelStmts[ID] = S;

  // If we've already seen any goto statements that point to this
  // label, resolve them now.
  typedef std::multimap<unsigned, GotoStmt *>::iterator GotoIter;
  std::pair<GotoIter, GotoIter> Gotos = UnresolvedGotoStmts.equal_range(ID);
  for (GotoIter Goto = Gotos.first; Goto != Gotos.second; ++Goto)
    Goto->second->setLabel(S);
  UnresolvedGotoStmts.erase(Gotos.first, Gotos.second);

  // If we've already seen any address-label statements that point to
  // this label, resolve them now.
  typedef std::multimap<unsigned, AddrLabelExpr *>::iterator AddrLabelIter;
  std::pair<AddrLabelIter, AddrLabelIter> AddrLabels 
    = UnresolvedAddrLabelExprs.equal_range(ID);
  for (AddrLabelIter AddrLabel = AddrLabels.first; 
       AddrLabel != AddrLabels.second; ++AddrLabel)
    AddrLabel->second->setLabel(S);
  UnresolvedAddrLabelExprs.erase(AddrLabels.first, AddrLabels.second);
}

/// \brief Set the label of the given statement to the label
/// identified by ID.
///
/// Depending on the order in which the label and other statements
/// referencing that label occur, this operation may complete
/// immediately (updating the statement) or it may queue the
/// statement to be back-patched later.
void PCHReader::SetLabelOf(GotoStmt *S, unsigned ID) {
  std::map<unsigned, LabelStmt *>::iterator Label = LabelStmts.find(ID);
  if (Label != LabelStmts.end()) {
    // We've already seen this label, so set the label of the goto and
    // we're done.
    S->setLabel(Label->second);
  } else {
    // We haven't seen this label yet, so add this goto to the set of
    // unresolved goto statements.
    UnresolvedGotoStmts.insert(std::make_pair(ID, S));
  }
}

/// \brief Set the label of the given expression to the label
/// identified by ID.
///
/// Depending on the order in which the label and other statements
/// referencing that label occur, this operation may complete
/// immediately (updating the statement) or it may queue the
/// statement to be back-patched later.
void PCHReader::SetLabelOf(AddrLabelExpr *S, unsigned ID) {
  std::map<unsigned, LabelStmt *>::iterator Label = LabelStmts.find(ID);
  if (Label != LabelStmts.end()) {
    // We've already seen this label, so set the label of the
    // label-address expression and we're done.
    S->setLabel(Label->second);
  } else {
    // We haven't seen this label yet, so add this label-address
    // expression to the set of unresolved label-address expressions.
    UnresolvedAddrLabelExprs.insert(std::make_pair(ID, S));
  }
}
