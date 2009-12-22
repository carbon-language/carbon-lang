//===--- Sema.h - Semantic Analysis & AST Building --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Sema class, which performs semantic analysis and
// builds ASTs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_SEMA_H
#define LLVM_CLANG_AST_SEMA_H

#include "IdentifierResolver.h"
#include "CXXFieldCollector.h"
#include "SemaOverload.h"
#include "SemaTemplate.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/FullExpr.h"
#include "clang/Parse/Action.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/OwningPtr.h"
#include <deque>
#include <list>
#include <map>
#include <string>
#include <vector>

namespace llvm {
  class APSInt;
}

namespace clang {
  class ASTContext;
  class ASTConsumer;
  class CodeCompleteConsumer;
  class Preprocessor;
  class Decl;
  class DeclContext;
  class DeclSpec;
  class ExternalSemaSource;
  class NamedDecl;
  class Stmt;
  class Expr;
  class InitListExpr;
  class ParenListExpr;
  class DesignatedInitExpr;
  class CallExpr;
  class DeclRefExpr;
  class UnresolvedLookupExpr;
  class VarDecl;
  class ParmVarDecl;
  class TypedefDecl;
  class FunctionDecl;
  class QualType;
  class LangOptions;
  class Token;
  class IntegerLiteral;
  class StringLiteral;
  class ArrayType;
  class LabelStmt;
  class SwitchStmt;
  class CXXTryStmt;
  class ExtVectorType;
  class TypedefDecl;
  class TemplateDecl;
  class TemplateArgument;
  class TemplateArgumentLoc;
  class TemplateArgumentList;
  class TemplateParameterList;
  class TemplateTemplateParmDecl;
  class ClassTemplatePartialSpecializationDecl;
  class ClassTemplateDecl;
  class ObjCInterfaceDecl;
  class ObjCCompatibleAliasDecl;
  class ObjCProtocolDecl;
  class ObjCImplDecl;
  class ObjCImplementationDecl;
  class ObjCCategoryImplDecl;
  class ObjCCategoryDecl;
  class ObjCIvarDecl;
  class ObjCMethodDecl;
  class ObjCPropertyDecl;
  class ObjCContainerDecl;
  class FunctionProtoType;
  class CXXBasePaths;
  class CXXTemporary;
  class LookupResult;
  class InitializedEntity;
  class InitializationKind;
  class InitializationSequence;
  
/// BlockSemaInfo - When a block is being parsed, this contains information
/// about the block.  It is pointed to from Sema::CurBlock.
struct BlockSemaInfo {
  llvm::SmallVector<ParmVarDecl*, 8> Params;
  bool hasPrototype;
  bool isVariadic;
  bool hasBlockDeclRefExprs;

  BlockDecl *TheDecl;

  /// TheScope - This is the scope for the block itself, which contains
  /// arguments etc.
  Scope *TheScope;

  /// ReturnType - This will get set to block result type, by looking at
  /// return types, if any, in the block body.
  QualType ReturnType;

  /// LabelMap - This is a mapping from label identifiers to the LabelStmt for
  /// it (which acts like the label decl in some ways).  Forward referenced
  /// labels have a LabelStmt created for them with a null location & SubStmt.
  llvm::DenseMap<IdentifierInfo*, LabelStmt*> LabelMap;

  /// SwitchStack - This is the current set of active switch statements in the
  /// block.
  llvm::SmallVector<SwitchStmt*, 8> SwitchStack;

  /// SavedFunctionNeedsScopeChecking - This is the value of
  /// CurFunctionNeedsScopeChecking at the point when the block started.
  bool SavedFunctionNeedsScopeChecking;

  /// PrevBlockInfo - If this is nested inside another block, this points
  /// to the outer block.
  BlockSemaInfo *PrevBlockInfo;
};

/// \brief Holds a QualType and a TypeSourceInfo* that came out of a declarator
/// parsing.
///
/// LocInfoType is a "transient" type, only needed for passing to/from Parser
/// and Sema, when we want to preserve type source info for a parsed type.
/// It will not participate in the type system semantics in any way.
class LocInfoType : public Type {
  enum {
    // The last number that can fit in Type's TC.
    // Avoids conflict with an existing Type class.
    LocInfo = (1 << TypeClassBitSize) - 1
  };

  TypeSourceInfo *DeclInfo;

  LocInfoType(QualType ty, TypeSourceInfo *TInfo)
    : Type((TypeClass)LocInfo, ty, ty->isDependentType()), DeclInfo(TInfo) {
    assert(getTypeClass() == (TypeClass)LocInfo && "LocInfo didn't fit in TC?");
  }
  friend class Sema;

public:
  QualType getType() const { return getCanonicalTypeInternal(); }
  TypeSourceInfo *getTypeSourceInfo() const { return DeclInfo; }

  virtual void getAsStringInternal(std::string &Str,
                                   const PrintingPolicy &Policy) const;

  static bool classof(const Type *T) {
    return T->getTypeClass() == (TypeClass)LocInfo;
  }
  static bool classof(const LocInfoType *) { return true; }
};

/// Sema - This implements semantic analysis and AST building for C.
class Sema : public Action {
  Sema(const Sema&);           // DO NOT IMPLEMENT
  void operator=(const Sema&); // DO NOT IMPLEMENT
public:
  const LangOptions &LangOpts;
  Preprocessor &PP;
  ASTContext &Context;
  ASTConsumer &Consumer;
  Diagnostic &Diags;
  SourceManager &SourceMgr;

  /// \brief Source of additional semantic information.
  ExternalSemaSource *ExternalSource;

  /// \brief Code-completion consumer.
  CodeCompleteConsumer *CodeCompleter;

  /// CurContext - This is the current declaration context of parsing.
  DeclContext *CurContext;

  /// CurBlock - If inside of a block definition, this contains a pointer to
  /// the active block object that represents it.
  BlockSemaInfo *CurBlock;

  /// PackContext - Manages the stack for #pragma pack. An alignment
  /// of 0 indicates default alignment.
  void *PackContext; // Really a "PragmaPackStack*"

  /// FunctionLabelMap - This is a mapping from label identifiers to the
  /// LabelStmt for it (which acts like the label decl in some ways).  Forward
  /// referenced labels have a LabelStmt created for them with a null location &
  /// SubStmt.
  ///
  /// Note that this should always be accessed through getLabelMap() in order
  /// to handle blocks properly.
  llvm::DenseMap<IdentifierInfo*, LabelStmt*> FunctionLabelMap;

  /// FunctionSwitchStack - This is the current set of active switch statements
  /// in the top level function.  Clients should always use getSwitchStack() to
  /// handle the case when they are in a block.
  llvm::SmallVector<SwitchStmt*, 8> FunctionSwitchStack;

  /// ExprTemporaries - This is the stack of temporaries that are created by
  /// the current full expression.
  llvm::SmallVector<CXXTemporary*, 8> ExprTemporaries;

  /// CurFunctionNeedsScopeChecking - This is set to true when a function or
  /// ObjC method body contains a VLA or an ObjC try block, which introduce
  /// scopes that need to be checked for goto conditions.  If a function does
  /// not contain this, then it need not have the jump checker run on it.
  bool CurFunctionNeedsScopeChecking;

  /// ExtVectorDecls - This is a list all the extended vector types. This allows
  /// us to associate a raw vector type with one of the ext_vector type names.
  /// This is only necessary for issuing pretty diagnostics.
  llvm::SmallVector<TypedefDecl*, 24> ExtVectorDecls;

  /// FieldCollector - Collects CXXFieldDecls during parsing of C++ classes.
  llvm::OwningPtr<CXXFieldCollector> FieldCollector;

  typedef llvm::SmallPtrSet<const CXXRecordDecl*, 8> RecordDeclSetTy;

  /// PureVirtualClassDiagSet - a set of class declarations which we have
  /// emitted a list of pure virtual functions. Used to prevent emitting the
  /// same list more than once.
  llvm::OwningPtr<RecordDeclSetTy> PureVirtualClassDiagSet;

  /// \brief A mapping from external names to the most recent
  /// locally-scoped external declaration with that name.
  ///
  /// This map contains external declarations introduced in local
  /// scoped, e.g.,
  ///
  /// \code
  /// void f() {
  ///   void foo(int, int);
  /// }
  /// \endcode
  ///
  /// Here, the name "foo" will be associated with the declaration on
  /// "foo" within f. This name is not visible outside of
  /// "f". However, we still find it in two cases:
  ///
  ///   - If we are declaring another external with the name "foo", we
  ///     can find "foo" as a previous declaration, so that the types
  ///     of this external declaration can be checked for
  ///     compatibility.
  ///
  ///   - If we would implicitly declare "foo" (e.g., due to a call to
  ///     "foo" in C when no prototype or definition is visible), then
  ///     we find this declaration of "foo" and complain that it is
  ///     not visible.
  llvm::DenseMap<DeclarationName, NamedDecl *> LocallyScopedExternalDecls;

  /// \brief The set of tentative declarations seen so far in this
  /// translation unit for which no definition has been seen.
  ///
  /// The tentative declarations are indexed by the name of the
  /// declaration, and only the most recent tentative declaration for
  /// a given variable will be recorded here.
  llvm::DenseMap<DeclarationName, VarDecl *> TentativeDefinitions;
  std::vector<DeclarationName> TentativeDefinitionList;

  /// \brief The collection of delayed deprecation warnings.
  llvm::SmallVector<std::pair<SourceLocation,NamedDecl*>, 8>
    DelayedDeprecationWarnings;

  /// \brief The depth of the current ParsingDeclaration stack.
  /// If nonzero, we are currently parsing a declaration (and
  /// hence should delay deprecation warnings).
  unsigned ParsingDeclDepth;

  /// WeakUndeclaredIdentifiers - Identifiers contained in
  /// #pragma weak before declared. rare. may alias another
  /// identifier, declared or undeclared
  class WeakInfo {
    IdentifierInfo *alias;  // alias (optional)
    SourceLocation loc;     // for diagnostics
    bool used;              // identifier later declared?
  public:
    WeakInfo()
      : alias(0), loc(SourceLocation()), used(false) {}
    WeakInfo(IdentifierInfo *Alias, SourceLocation Loc)
      : alias(Alias), loc(Loc), used(false) {}
    inline IdentifierInfo * getAlias() const { return alias; }
    inline SourceLocation getLocation() const { return loc; }
    void setUsed(bool Used=true) { used = Used; }
    inline bool getUsed() { return used; }
    bool operator==(WeakInfo RHS) const {
      return alias == RHS.getAlias() && loc == RHS.getLocation();
    }
    bool operator!=(WeakInfo RHS) const { return !(*this == RHS); }
  };
  llvm::DenseMap<IdentifierInfo*,WeakInfo> WeakUndeclaredIdentifiers;

  /// WeakTopLevelDecl - Translation-unit scoped declarations generated by
  /// #pragma weak during processing of other Decls.
  /// I couldn't figure out a clean way to generate these in-line, so
  /// we store them here and handle separately -- which is a hack.
  /// It would be best to refactor this.
  llvm::SmallVector<Decl*,2> WeakTopLevelDecl;

  IdentifierResolver IdResolver;

  /// Translation Unit Scope - useful to Objective-C actions that need
  /// to lookup file scope declarations in the "ordinary" C decl namespace.
  /// For example, user-defined classes, built-in "id" type, etc.
  Scope *TUScope;

  /// \brief The C++ "std" namespace, where the standard library resides.
  NamespaceDecl *StdNamespace;

  /// \brief The C++ "std::bad_alloc" class, which is defined by the C++
  /// standard library.
  CXXRecordDecl *StdBadAlloc;
  
  /// A flag to remember whether the implicit forms of operator new and delete
  /// have been declared.
  bool GlobalNewDeleteDeclared;

  /// \brief The set of declarations that have been referenced within
  /// a potentially evaluated expression.
  typedef std::vector<std::pair<SourceLocation, Decl *> > 
    PotentiallyReferencedDecls;

  /// \brief A set of diagnostics that may be emitted.
  typedef std::vector<std::pair<SourceLocation, PartialDiagnostic> >
    PotentiallyEmittedDiagnostics;

  /// \brief Data structure used to record current or nested
  /// expression evaluation contexts.
  struct ExpressionEvaluationContextRecord {
    /// \brief The expression evaluation context.
    ExpressionEvaluationContext Context;

    /// \brief The number of temporaries that were active when we
    /// entered this expression evaluation context.
    unsigned NumTemporaries;

    /// \brief The set of declarations referenced within a
    /// potentially potentially-evaluated context.
    ///
    /// When leaving a potentially potentially-evaluated context, each
    /// of these elements will be as referenced if the corresponding
    /// potentially potentially evaluated expression is potentially
    /// evaluated.
    PotentiallyReferencedDecls *PotentiallyReferenced;

    /// \brief The set of diagnostics to emit should this potentially
    /// potentially-evaluated context become evaluated.
    PotentiallyEmittedDiagnostics *PotentiallyDiagnosed;

    ExpressionEvaluationContextRecord(ExpressionEvaluationContext Context,
                                      unsigned NumTemporaries) 
      : Context(Context), NumTemporaries(NumTemporaries), 
        PotentiallyReferenced(0), PotentiallyDiagnosed(0) { }

    void addReferencedDecl(SourceLocation Loc, Decl *Decl) {
      if (!PotentiallyReferenced)
        PotentiallyReferenced = new PotentiallyReferencedDecls;
      PotentiallyReferenced->push_back(std::make_pair(Loc, Decl));
    }

    void addDiagnostic(SourceLocation Loc, const PartialDiagnostic &PD) {
      if (!PotentiallyDiagnosed)
        PotentiallyDiagnosed = new PotentiallyEmittedDiagnostics;
      PotentiallyDiagnosed->push_back(std::make_pair(Loc, PD));
    }

    void Destroy() {
      delete PotentiallyReferenced;
      delete PotentiallyDiagnosed;
      PotentiallyReferenced = 0;
      PotentiallyDiagnosed = 0;
    }
  };

  /// A stack of expression evaluation contexts.
  llvm::SmallVector<ExpressionEvaluationContextRecord, 8> ExprEvalContexts;

  /// \brief Whether the code handled by Sema should be considered a
  /// complete translation unit or not.
  ///
  /// When true (which is generally the case), Sema will perform
  /// end-of-translation-unit semantic tasks (such as creating
  /// initializers for tentative definitions in C) once parsing has
  /// completed. This flag will be false when building PCH files,
  /// since a PCH file is by definition not a complete translation
  /// unit.
  bool CompleteTranslationUnit;

  llvm::BumpPtrAllocator BumpAlloc;

  /// \brief The number of SFINAE diagnostics that have been trapped.
  unsigned NumSFINAEErrors;

  typedef llvm::DenseMap<Selector, ObjCMethodList> MethodPool;

  /// Instance/Factory Method Pools - allows efficient lookup when typechecking
  /// messages to "id". We need to maintain a list, since selectors can have
  /// differing signatures across classes. In Cocoa, this happens to be
  /// extremely uncommon (only 1% of selectors are "overloaded").
  MethodPool InstanceMethodPool;
  MethodPool FactoryMethodPool;

  MethodPool::iterator ReadMethodPool(Selector Sel, bool isInstance);

  /// Private Helper predicate to check for 'self'.
  bool isSelfExpr(Expr *RExpr);
public:
  Sema(Preprocessor &pp, ASTContext &ctxt, ASTConsumer &consumer,
       bool CompleteTranslationUnit = true,
       CodeCompleteConsumer *CompletionConsumer = 0);
  ~Sema() {
    if (PackContext) FreePackedContext();
  }

  const LangOptions &getLangOptions() const { return LangOpts; }
  Diagnostic &getDiagnostics() const { return Diags; }
  SourceManager &getSourceManager() const { return SourceMgr; }

  /// \brief Helper class that creates diagnostics with optional
  /// template instantiation stacks.
  ///
  /// This class provides a wrapper around the basic DiagnosticBuilder
  /// class that emits diagnostics. SemaDiagnosticBuilder is
  /// responsible for emitting the diagnostic (as DiagnosticBuilder
  /// does) and, if the diagnostic comes from inside a template
  /// instantiation, printing the template instantiation stack as
  /// well.
  class SemaDiagnosticBuilder : public DiagnosticBuilder {
    Sema &SemaRef;
    unsigned DiagID;

  public:
    SemaDiagnosticBuilder(DiagnosticBuilder &DB, Sema &SemaRef, unsigned DiagID)
      : DiagnosticBuilder(DB), SemaRef(SemaRef), DiagID(DiagID) { }

    explicit SemaDiagnosticBuilder(Sema &SemaRef)
      : DiagnosticBuilder(DiagnosticBuilder::Suppress), SemaRef(SemaRef) { }

    ~SemaDiagnosticBuilder();
  };

  /// \brief Emit a diagnostic.
  SemaDiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID) {
    if (isSFINAEContext() && Diagnostic::isBuiltinSFINAEDiag(DiagID)) {
      // If we encountered an error during template argument
      // deduction, and that error is one of the SFINAE errors,
      // suppress the diagnostic.
      ++NumSFINAEErrors;
      Diags.setLastDiagnosticIgnored();
      return SemaDiagnosticBuilder(*this);
    }

    DiagnosticBuilder DB = Diags.Report(FullSourceLoc(Loc, SourceMgr), DiagID);
    return SemaDiagnosticBuilder(DB, *this, DiagID);
  }

  /// \brief Emit a partial diagnostic.
  SemaDiagnosticBuilder Diag(SourceLocation Loc, const PartialDiagnostic& PD);

  virtual void DeleteExpr(ExprTy *E);
  virtual void DeleteStmt(StmtTy *S);

  OwningExprResult Owned(Expr* E) {
    assert(!E || E->isRetained());
    return OwningExprResult(*this, E);
  }
  OwningExprResult Owned(ExprResult R) {
    if (R.isInvalid())
      return ExprError();
    assert(!R.get() || ((Expr*) R.get())->isRetained());
    return OwningExprResult(*this, R.get());
  }
  OwningStmtResult Owned(Stmt* S) {
    assert(!S || S->isRetained());
    return OwningStmtResult(*this, S);
  }

  virtual void ActOnEndOfTranslationUnit();

  /// getLabelMap() - Return the current label map.  If we're in a block, we
  /// return it.
  llvm::DenseMap<IdentifierInfo*, LabelStmt*> &getLabelMap() {
    return CurBlock ? CurBlock->LabelMap : FunctionLabelMap;
  }

  /// getSwitchStack - This is returns the switch stack for the current block or
  /// function.
  llvm::SmallVector<SwitchStmt*,8> &getSwitchStack() {
    return CurBlock ? CurBlock->SwitchStack : FunctionSwitchStack;
  }

  /// WeakTopLevelDeclDecls - access to #pragma weak-generated Decls
  llvm::SmallVector<Decl*,2> &WeakTopLevelDecls() { return WeakTopLevelDecl; }

  virtual void ActOnComment(SourceRange Comment);

  //===--------------------------------------------------------------------===//
  // Type Analysis / Processing: SemaType.cpp.
  //
  QualType adjustParameterType(QualType T);
  void ProcessTypeAttributeList(QualType &Result, const AttributeList *AL);
  QualType BuildPointerType(QualType T, unsigned Quals,
                            SourceLocation Loc, DeclarationName Entity);
  QualType BuildReferenceType(QualType T, bool LValueRef, unsigned Quals,
                              SourceLocation Loc, DeclarationName Entity);
  QualType BuildArrayType(QualType T, ArrayType::ArraySizeModifier ASM,
                          Expr *ArraySize, unsigned Quals,
                          SourceRange Brackets, DeclarationName Entity);
  QualType BuildExtVectorType(QualType T, ExprArg ArraySize,
                              SourceLocation AttrLoc);
  QualType BuildFunctionType(QualType T,
                             QualType *ParamTypes, unsigned NumParamTypes,
                             bool Variadic, unsigned Quals,
                             SourceLocation Loc, DeclarationName Entity);
  QualType BuildMemberPointerType(QualType T, QualType Class,
                                  unsigned Quals, SourceLocation Loc,
                                  DeclarationName Entity);
  QualType BuildBlockPointerType(QualType T, unsigned Quals,
                                 SourceLocation Loc, DeclarationName Entity);
  QualType GetTypeForDeclarator(Declarator &D, Scope *S,
                                TypeSourceInfo **TInfo = 0,
                                TagDecl **OwnedDecl = 0);
  TypeSourceInfo *GetTypeSourceInfoForDeclarator(Declarator &D, QualType T);
  /// \brief Create a LocInfoType to hold the given QualType and TypeSourceInfo.
  QualType CreateLocInfoType(QualType T, TypeSourceInfo *TInfo);
  DeclarationName GetNameForDeclarator(Declarator &D);
  DeclarationName GetNameFromUnqualifiedId(const UnqualifiedId &Name);
  static QualType GetTypeFromParser(TypeTy *Ty, TypeSourceInfo **TInfo = 0);
  bool CheckSpecifiedExceptionType(QualType T, const SourceRange &Range);
  bool CheckDistantExceptionSpec(QualType T);
  bool CheckEquivalentExceptionSpec(
      const FunctionProtoType *Old, SourceLocation OldLoc,
      const FunctionProtoType *New, SourceLocation NewLoc);
  bool CheckEquivalentExceptionSpec(
      const PartialDiagnostic &DiagID, const PartialDiagnostic & NoteID,
      const FunctionProtoType *Old, SourceLocation OldLoc,
      const FunctionProtoType *New, SourceLocation NewLoc);
  bool CheckExceptionSpecSubset(
      const PartialDiagnostic &DiagID, const PartialDiagnostic & NoteID,
      const FunctionProtoType *Superset, SourceLocation SuperLoc,
      const FunctionProtoType *Subset, SourceLocation SubLoc);
  bool CheckParamExceptionSpec(const PartialDiagnostic & NoteID,
      const FunctionProtoType *Target, SourceLocation TargetLoc,
      const FunctionProtoType *Source, SourceLocation SourceLoc);

  QualType ObjCGetTypeForMethodDefinition(DeclPtrTy D);

  bool UnwrapSimilarPointerTypes(QualType& T1, QualType& T2);

  virtual TypeResult ActOnTypeName(Scope *S, Declarator &D);

  bool RequireCompleteType(SourceLocation Loc, QualType T,
                           const PartialDiagnostic &PD,
                           std::pair<SourceLocation,
                                     PartialDiagnostic> Note =
                            std::make_pair(SourceLocation(), PDiag()));

  QualType getQualifiedNameType(const CXXScopeSpec &SS, QualType T);

  QualType BuildTypeofExprType(Expr *E);
  QualType BuildDecltypeType(Expr *E);

  //===--------------------------------------------------------------------===//
  // Symbol table / Decl tracking callbacks: SemaDecl.cpp.
  //

  /// getDeclName - Return a pretty name for the specified decl if possible, or
  /// an empty string if not.  This is used for pretty crash reporting.
  virtual std::string getDeclName(DeclPtrTy D);

  DeclGroupPtrTy ConvertDeclToDeclGroup(DeclPtrTy Ptr);

  virtual TypeTy *getTypeName(IdentifierInfo &II, SourceLocation NameLoc,
                              Scope *S, const CXXScopeSpec *SS,
                              bool isClassName = false,
                              TypeTy *ObjectType = 0);
  virtual DeclSpec::TST isTagName(IdentifierInfo &II, Scope *S);
  virtual bool DiagnoseUnknownTypeName(const IdentifierInfo &II, 
                                       SourceLocation IILoc,
                                       Scope *S,
                                       const CXXScopeSpec *SS,
                                       TypeTy *&SuggestedType);
  
  virtual DeclPtrTy ActOnDeclarator(Scope *S, Declarator &D) {
    return HandleDeclarator(S, D, MultiTemplateParamsArg(*this), false);
  }

  DeclPtrTy HandleDeclarator(Scope *S, Declarator &D,
                             MultiTemplateParamsArg TemplateParameterLists,
                             bool IsFunctionDefinition);
  void RegisterLocallyScopedExternCDecl(NamedDecl *ND,
                                        const LookupResult &Previous,
                                        Scope *S);
  void DiagnoseFunctionSpecifiers(Declarator& D);
  NamedDecl* ActOnTypedefDeclarator(Scope* S, Declarator& D, DeclContext* DC,
                                    QualType R, TypeSourceInfo *TInfo,
                                    LookupResult &Previous, bool &Redeclaration);
  NamedDecl* ActOnVariableDeclarator(Scope* S, Declarator& D, DeclContext* DC,
                                     QualType R, TypeSourceInfo *TInfo,
                                     LookupResult &Previous,
                                     MultiTemplateParamsArg TemplateParamLists,
                                     bool &Redeclaration);
  void CheckVariableDeclaration(VarDecl *NewVD, LookupResult &Previous,
                                bool &Redeclaration);
  NamedDecl* ActOnFunctionDeclarator(Scope* S, Declarator& D, DeclContext* DC,
                                     QualType R, TypeSourceInfo *TInfo,
                                     LookupResult &Previous,
                                     MultiTemplateParamsArg TemplateParamLists,
                                     bool IsFunctionDefinition,
                                     bool &Redeclaration);
  void AddOverriddenMethods(CXXRecordDecl *DC, CXXMethodDecl *MD);
  void CheckFunctionDeclaration(Scope *S,
                                FunctionDecl *NewFD, LookupResult &Previous,
                                bool IsExplicitSpecialization,
                                bool &Redeclaration,
                                bool &OverloadableAttrRequired);
  void CheckMain(FunctionDecl *FD);
  virtual DeclPtrTy ActOnParamDeclarator(Scope *S, Declarator &D);
  virtual void ActOnParamDefaultArgument(DeclPtrTy param,
                                         SourceLocation EqualLoc,
                                         ExprArg defarg);
  virtual void ActOnParamUnparsedDefaultArgument(DeclPtrTy param,
                                                 SourceLocation EqualLoc,
                                                 SourceLocation ArgLoc);
  virtual void ActOnParamDefaultArgumentError(DeclPtrTy param);
  bool SetParamDefaultArgument(ParmVarDecl *Param, ExprArg DefaultArg,
                               SourceLocation EqualLoc);


  // Contains the locations of the beginning of unparsed default
  // argument locations.
  llvm::DenseMap<ParmVarDecl *,SourceLocation> UnparsedDefaultArgLocs;

  virtual void AddInitializerToDecl(DeclPtrTy dcl, ExprArg init);
  void AddInitializerToDecl(DeclPtrTy dcl, ExprArg init, bool DirectInit);
  void ActOnUninitializedDecl(DeclPtrTy dcl, bool TypeContainsUndeducedAuto);
  virtual void SetDeclDeleted(DeclPtrTy dcl, SourceLocation DelLoc);
  virtual DeclGroupPtrTy FinalizeDeclaratorGroup(Scope *S, const DeclSpec &DS,
                                                 DeclPtrTy *Group,
                                                 unsigned NumDecls);
  virtual void ActOnFinishKNRParamDeclarations(Scope *S, Declarator &D,
                                               SourceLocation LocAfterDecls);
  virtual DeclPtrTy ActOnStartOfFunctionDef(Scope *S, Declarator &D);
  virtual DeclPtrTy ActOnStartOfFunctionDef(Scope *S, DeclPtrTy D);
  virtual void ActOnStartOfObjCMethodDef(Scope *S, DeclPtrTy D);

  virtual DeclPtrTy ActOnFinishFunctionBody(DeclPtrTy Decl, StmtArg Body);
  DeclPtrTy ActOnFinishFunctionBody(DeclPtrTy Decl, StmtArg Body,
                                    bool IsInstantiation);

  /// \brief Diagnose any unused parameters in the given sequence of
  /// ParmVarDecl pointers.
  template<typename InputIterator>
  void DiagnoseUnusedParameters(InputIterator Param, InputIterator ParamEnd) {
    for (; Param != ParamEnd; ++Param) {
      if (!(*Param)->isUsed() && (*Param)->getDeclName() &&
          !(*Param)->template hasAttr<UnusedAttr>())
        Diag((*Param)->getLocation(), diag::warn_unused_parameter)
          << (*Param)->getDeclName();
    }
  }

  void DiagnoseInvalidJumps(Stmt *Body);
  virtual DeclPtrTy ActOnFileScopeAsmDecl(SourceLocation Loc, ExprArg expr);

  /// Scope actions.
  virtual void ActOnPopScope(SourceLocation Loc, Scope *S);
  virtual void ActOnTranslationUnitScope(SourceLocation Loc, Scope *S);

  /// ParsedFreeStandingDeclSpec - This method is invoked when a declspec with
  /// no declarator (e.g. "struct foo;") is parsed.
  virtual DeclPtrTy ParsedFreeStandingDeclSpec(Scope *S, DeclSpec &DS);

  bool InjectAnonymousStructOrUnionMembers(Scope *S, DeclContext *Owner,
                                           RecordDecl *AnonRecord);
  virtual DeclPtrTy BuildAnonymousStructOrUnion(Scope *S, DeclSpec &DS,
                                                RecordDecl *Record);

  bool isAcceptableTagRedeclaration(const TagDecl *Previous,
                                    TagDecl::TagKind NewTag,
                                    SourceLocation NewTagLoc,
                                    const IdentifierInfo &Name);

  virtual DeclPtrTy ActOnTag(Scope *S, unsigned TagSpec, TagUseKind TUK,
                             SourceLocation KWLoc, const CXXScopeSpec &SS,
                             IdentifierInfo *Name, SourceLocation NameLoc,
                             AttributeList *Attr, AccessSpecifier AS,
                             MultiTemplateParamsArg TemplateParameterLists,
                             bool &OwnedDecl, bool &IsDependent);

  virtual TypeResult ActOnDependentTag(Scope *S,
                                       unsigned TagSpec,
                                       TagUseKind TUK,
                                       const CXXScopeSpec &SS,
                                       IdentifierInfo *Name,
                                       SourceLocation TagLoc,
                                       SourceLocation NameLoc);

  virtual void ActOnDefs(Scope *S, DeclPtrTy TagD, SourceLocation DeclStart,
                         IdentifierInfo *ClassName,
                         llvm::SmallVectorImpl<DeclPtrTy> &Decls);
  virtual DeclPtrTy ActOnField(Scope *S, DeclPtrTy TagD,
                               SourceLocation DeclStart,
                               Declarator &D, ExprTy *BitfieldWidth);

  FieldDecl *HandleField(Scope *S, RecordDecl *TagD, SourceLocation DeclStart,
                         Declarator &D, Expr *BitfieldWidth,
                         AccessSpecifier AS);

  FieldDecl *CheckFieldDecl(DeclarationName Name, QualType T,
                            TypeSourceInfo *TInfo,
                            RecordDecl *Record, SourceLocation Loc,
                            bool Mutable, Expr *BitfieldWidth,
                            SourceLocation TSSL,
                            AccessSpecifier AS, NamedDecl *PrevDecl,
                            Declarator *D = 0);

  enum CXXSpecialMember {
    CXXDefaultConstructor = 0,
    CXXCopyConstructor = 1,
    CXXCopyAssignment = 2,
    CXXDestructor = 3
  };
  void DiagnoseNontrivial(const RecordType* Record, CXXSpecialMember mem);

  virtual DeclPtrTy ActOnIvar(Scope *S, SourceLocation DeclStart,
                              DeclPtrTy IntfDecl,
                              Declarator &D, ExprTy *BitfieldWidth,
                              tok::ObjCKeywordKind visibility);

  // This is used for both record definitions and ObjC interface declarations.
  virtual void ActOnFields(Scope* S,
                           SourceLocation RecLoc, DeclPtrTy TagDecl,
                           DeclPtrTy *Fields, unsigned NumFields,
                           SourceLocation LBrac, SourceLocation RBrac,
                           AttributeList *AttrList);

  /// ActOnTagStartDefinition - Invoked when we have entered the
  /// scope of a tag's definition (e.g., for an enumeration, class,
  /// struct, or union).
  virtual void ActOnTagStartDefinition(Scope *S, DeclPtrTy TagDecl);

  /// ActOnStartCXXMemberDeclarations - Invoked when we have parsed a
  /// C++ record definition's base-specifiers clause and are starting its
  /// member declarations.
  virtual void ActOnStartCXXMemberDeclarations(Scope *S, DeclPtrTy TagDecl,
                                               SourceLocation LBraceLoc);

  /// ActOnTagFinishDefinition - Invoked once we have finished parsing
  /// the definition of a tag (enumeration, class, struct, or union).
  virtual void ActOnTagFinishDefinition(Scope *S, DeclPtrTy TagDecl,
                                        SourceLocation RBraceLoc);

  EnumConstantDecl *CheckEnumConstant(EnumDecl *Enum,
                                      EnumConstantDecl *LastEnumConst,
                                      SourceLocation IdLoc,
                                      IdentifierInfo *Id,
                                      ExprArg val);

  virtual DeclPtrTy ActOnEnumConstant(Scope *S, DeclPtrTy EnumDecl,
                                      DeclPtrTy LastEnumConstant,
                                      SourceLocation IdLoc, IdentifierInfo *Id,
                                      SourceLocation EqualLoc, ExprTy *Val);
  virtual void ActOnEnumBody(SourceLocation EnumLoc, SourceLocation LBraceLoc,
                             SourceLocation RBraceLoc, DeclPtrTy EnumDecl,
                             DeclPtrTy *Elements, unsigned NumElements,
                             Scope *S, AttributeList *Attr);

  DeclContext *getContainingDC(DeclContext *DC);

  /// Set the current declaration context until it gets popped.
  void PushDeclContext(Scope *S, DeclContext *DC);
  void PopDeclContext();

  /// EnterDeclaratorContext - Used when we must lookup names in the context
  /// of a declarator's nested name specifier.
  void EnterDeclaratorContext(Scope *S, DeclContext *DC);
  void ExitDeclaratorContext(Scope *S);

  DeclContext *getFunctionLevelDeclContext();

  /// getCurFunctionDecl - If inside of a function body, this returns a pointer
  /// to the function decl for the function being parsed.  If we're currently
  /// in a 'block', this returns the containing context.
  FunctionDecl *getCurFunctionDecl();

  /// getCurMethodDecl - If inside of a method body, this returns a pointer to
  /// the method decl for the method being parsed.  If we're currently
  /// in a 'block', this returns the containing context.
  ObjCMethodDecl *getCurMethodDecl();

  /// getCurFunctionOrMethodDecl - Return the Decl for the current ObjC method
  /// or C function we're in, otherwise return null.  If we're currently
  /// in a 'block', this returns the containing context.
  NamedDecl *getCurFunctionOrMethodDecl();

  /// Add this decl to the scope shadowed decl chains.
  void PushOnScopeChains(NamedDecl *D, Scope *S, bool AddToContext = true);

  /// isDeclInScope - If 'Ctx' is a function/method, isDeclInScope returns true
  /// if 'D' is in Scope 'S', otherwise 'S' is ignored and isDeclInScope returns
  /// true if 'D' belongs to the given declaration context.
  bool isDeclInScope(NamedDecl *&D, DeclContext *Ctx, Scope *S = 0);

  /// Finds the scope corresponding to the given decl context, if it
  /// happens to be an enclosing scope.  Otherwise return NULL.
  Scope *getScopeForDeclContext(Scope *S, DeclContext *DC) {
    DeclContext *TargetDC = DC->getPrimaryContext();
    do {
      if (DeclContext *ScopeDC = (DeclContext*) S->getEntity())
        if (ScopeDC->getPrimaryContext() == TargetDC)
          return S;
    } while ((S = S->getParent()));

    return NULL;
  }

  /// Subroutines of ActOnDeclarator().
  TypedefDecl *ParseTypedefDecl(Scope *S, Declarator &D, QualType T,
                                TypeSourceInfo *TInfo);
  void MergeTypeDefDecl(TypedefDecl *New, LookupResult &OldDecls);
  bool MergeFunctionDecl(FunctionDecl *New, Decl *Old);
  bool MergeCompatibleFunctionDecls(FunctionDecl *New, FunctionDecl *Old);
  void MergeVarDecl(VarDecl *New, LookupResult &OldDecls);
  bool MergeCXXFunctionDecl(FunctionDecl *New, FunctionDecl *Old);

  // AssignmentAction - This is used by all the assignment diagnostic functions
  // to represent what is actually causing the operation
  enum AssignmentAction {
    AA_Assigning,
    AA_Passing,
    AA_Returning,
    AA_Converting,
    AA_Initializing,
    AA_Sending,
    AA_Casting
  };
  
  /// C++ Overloading.
  enum OverloadKind {
    /// This is a legitimate overload: the existing declarations are
    /// functions or function templates with different signatures.
    Ovl_Overload,

    /// This is not an overload because the signature exactly matches
    /// an existing declaration.
    Ovl_Match,

    /// This is not an overload because the lookup results contain a
    /// non-function.
    Ovl_NonFunction
  };
  OverloadKind CheckOverload(FunctionDecl *New,
                             const LookupResult &OldDecls,
                             NamedDecl *&OldDecl);
  bool IsOverload(FunctionDecl *New, FunctionDecl *Old);

  ImplicitConversionSequence
  TryImplicitConversion(Expr* From, QualType ToType,
                        bool SuppressUserConversions,
                        bool AllowExplicit,
                        bool ForceRValue,
                        bool InOverloadResolution,
                        bool UserCast = false);
  bool IsStandardConversion(Expr *From, QualType ToType,
                            bool InOverloadResolution,
                            StandardConversionSequence& SCS);
  bool IsIntegralPromotion(Expr *From, QualType FromType, QualType ToType);
  bool IsFloatingPointPromotion(QualType FromType, QualType ToType);
  bool IsComplexPromotion(QualType FromType, QualType ToType);
  bool IsPointerConversion(Expr *From, QualType FromType, QualType ToType,
                           bool InOverloadResolution,
                           QualType& ConvertedType, bool &IncompatibleObjC);
  bool isObjCPointerConversion(QualType FromType, QualType ToType,
                               QualType& ConvertedType, bool &IncompatibleObjC);
  bool CheckPointerConversion(Expr *From, QualType ToType,
                              CastExpr::CastKind &Kind,
                              bool IgnoreBaseAccess);
  bool IsMemberPointerConversion(Expr *From, QualType FromType, QualType ToType,
                                 bool InOverloadResolution,
                                 QualType &ConvertedType);
  bool CheckMemberPointerConversion(Expr *From, QualType ToType,
                                    CastExpr::CastKind &Kind,
                                    bool IgnoreBaseAccess);
  bool IsQualificationConversion(QualType FromType, QualType ToType);
  OverloadingResult IsUserDefinedConversion(Expr *From, QualType ToType,
                               UserDefinedConversionSequence& User,
                               OverloadCandidateSet& Conversions,
                               bool AllowConversionFunctions,
                               bool AllowExplicit, bool ForceRValue,
                               bool UserCast = false);
  bool DiagnoseMultipleUserDefinedConversion(Expr *From, QualType ToType);
                                              

  ImplicitConversionSequence::CompareKind
  CompareImplicitConversionSequences(const ImplicitConversionSequence& ICS1,
                                     const ImplicitConversionSequence& ICS2);

  ImplicitConversionSequence::CompareKind
  CompareStandardConversionSequences(const StandardConversionSequence& SCS1,
                                     const StandardConversionSequence& SCS2);

  ImplicitConversionSequence::CompareKind
  CompareQualificationConversions(const StandardConversionSequence& SCS1,
                                  const StandardConversionSequence& SCS2);

  ImplicitConversionSequence::CompareKind
  CompareDerivedToBaseConversions(const StandardConversionSequence& SCS1,
                                  const StandardConversionSequence& SCS2);

  ImplicitConversionSequence
  TryCopyInitialization(Expr* From, QualType ToType,
                        bool SuppressUserConversions, bool ForceRValue,
                        bool InOverloadResolution);
  
  bool PerformCopyInitialization(Expr *&From, QualType ToType,
                                 AssignmentAction Action, bool Elidable = false);

  OwningExprResult PerformCopyInitialization(const InitializedEntity &Entity,
                                             SourceLocation EqualLoc,
                                             OwningExprResult Init);
  ImplicitConversionSequence
  TryObjectArgumentInitialization(QualType FromType, CXXMethodDecl *Method,
                                  CXXRecordDecl *ActingContext);
  bool PerformObjectArgumentInitialization(Expr *&From, CXXMethodDecl *Method);

  ImplicitConversionSequence TryContextuallyConvertToBool(Expr *From);
  bool PerformContextuallyConvertToBool(Expr *&From);

  bool PerformObjectMemberConversion(Expr *&From, NamedDecl *Member);

  // Members have to be NamespaceDecl* or TranslationUnitDecl*.
  // TODO: make this is a typesafe union.
  typedef llvm::SmallPtrSet<DeclContext   *, 16> AssociatedNamespaceSet;

  typedef llvm::SmallPtrSet<AnyFunctionDecl, 16> FunctionSet;
  typedef llvm::SmallPtrSet<CXXRecordDecl *, 16> AssociatedClassSet;

  void AddOverloadCandidate(FunctionDecl *Function,
                            Expr **Args, unsigned NumArgs,
                            OverloadCandidateSet& CandidateSet,
                            bool SuppressUserConversions = false,
                            bool ForceRValue = false,
                            bool PartialOverloading = false);
  void AddFunctionCandidates(const FunctionSet &Functions,
                             Expr **Args, unsigned NumArgs,
                             OverloadCandidateSet& CandidateSet,
                             bool SuppressUserConversions = false);
  void AddMethodCandidate(NamedDecl *Decl,
                          QualType ObjectType, Expr **Args, unsigned NumArgs,
                          OverloadCandidateSet& CandidateSet,
                          bool SuppressUserConversion = false,
                          bool ForceRValue = false);
  void AddMethodCandidate(CXXMethodDecl *Method, CXXRecordDecl *ActingContext,
                          QualType ObjectType, Expr **Args, unsigned NumArgs,
                          OverloadCandidateSet& CandidateSet,
                          bool SuppressUserConversions = false,
                          bool ForceRValue = false);
  void AddMethodTemplateCandidate(FunctionTemplateDecl *MethodTmpl,
                                  CXXRecordDecl *ActingContext,
                         const TemplateArgumentListInfo *ExplicitTemplateArgs,
                                  QualType ObjectType,
                                  Expr **Args, unsigned NumArgs,
                                  OverloadCandidateSet& CandidateSet,
                                  bool SuppressUserConversions = false,
                                  bool ForceRValue = false);
  void AddTemplateOverloadCandidate(FunctionTemplateDecl *FunctionTemplate,
                      const TemplateArgumentListInfo *ExplicitTemplateArgs,
                                    Expr **Args, unsigned NumArgs,
                                    OverloadCandidateSet& CandidateSet,
                                    bool SuppressUserConversions = false,
                                    bool ForceRValue = false);
  void AddConversionCandidate(CXXConversionDecl *Conversion,
                              CXXRecordDecl *ActingContext,
                              Expr *From, QualType ToType,
                              OverloadCandidateSet& CandidateSet);
  void AddTemplateConversionCandidate(FunctionTemplateDecl *FunctionTemplate,
                                      CXXRecordDecl *ActingContext,
                                      Expr *From, QualType ToType,
                                      OverloadCandidateSet &CandidateSet);
  void AddSurrogateCandidate(CXXConversionDecl *Conversion,
                             CXXRecordDecl *ActingContext,
                             const FunctionProtoType *Proto,
                             QualType ObjectTy, Expr **Args, unsigned NumArgs,
                             OverloadCandidateSet& CandidateSet);
  void AddOperatorCandidates(OverloadedOperatorKind Op, Scope *S,
                             SourceLocation OpLoc,
                             Expr **Args, unsigned NumArgs,
                             OverloadCandidateSet& CandidateSet,
                             SourceRange OpRange = SourceRange());
  void AddMemberOperatorCandidates(OverloadedOperatorKind Op,
                                   SourceLocation OpLoc,
                                   Expr **Args, unsigned NumArgs,
                                   OverloadCandidateSet& CandidateSet,
                                   SourceRange OpRange = SourceRange());
  void AddBuiltinCandidate(QualType ResultTy, QualType *ParamTys,
                           Expr **Args, unsigned NumArgs,
                           OverloadCandidateSet& CandidateSet,
                           bool IsAssignmentOperator = false,
                           unsigned NumContextualBoolArguments = 0);
  void AddBuiltinOperatorCandidates(OverloadedOperatorKind Op,
                                    SourceLocation OpLoc,
                                    Expr **Args, unsigned NumArgs,
                                    OverloadCandidateSet& CandidateSet);
  void AddArgumentDependentLookupCandidates(DeclarationName Name,
                                            Expr **Args, unsigned NumArgs,
                        const TemplateArgumentListInfo *ExplicitTemplateArgs,
                                            OverloadCandidateSet& CandidateSet,
                                            bool PartialOverloading = false);
  bool isBetterOverloadCandidate(const OverloadCandidate& Cand1,
                                 const OverloadCandidate& Cand2);
  OverloadingResult BestViableFunction(OverloadCandidateSet& CandidateSet,
                                       SourceLocation Loc,
                                       OverloadCandidateSet::iterator& Best);
  void PrintOverloadCandidates(OverloadCandidateSet& CandidateSet,
                         bool OnlyViable,
                         const char *Opc=0,
                         SourceLocation Loc=SourceLocation());

  FunctionDecl *ResolveAddressOfOverloadedFunction(Expr *From, QualType ToType,
                                                   bool Complain);
  FunctionDecl *ResolveSingleFunctionTemplateSpecialization(Expr *From);

  Expr *FixOverloadedFunctionReference(Expr *E, FunctionDecl *Fn);
  OwningExprResult FixOverloadedFunctionReference(OwningExprResult, 
                                                  FunctionDecl *Fn);

  void AddOverloadedCallCandidates(UnresolvedLookupExpr *ULE,
                                   Expr **Args, unsigned NumArgs,
                                   OverloadCandidateSet &CandidateSet,
                                   bool PartialOverloading = false);
    
  OwningExprResult BuildOverloadedCallExpr(Expr *Fn,
                                           UnresolvedLookupExpr *ULE,
                                           SourceLocation LParenLoc,
                                           Expr **Args, unsigned NumArgs,
                                           SourceLocation *CommaLocs,
                                           SourceLocation RParenLoc);

  OwningExprResult CreateOverloadedUnaryOp(SourceLocation OpLoc,
                                           unsigned Opc,
                                           FunctionSet &Functions,
                                           ExprArg input);

  OwningExprResult CreateOverloadedBinOp(SourceLocation OpLoc,
                                         unsigned Opc,
                                         FunctionSet &Functions,
                                         Expr *LHS, Expr *RHS);

  OwningExprResult CreateOverloadedArraySubscriptExpr(SourceLocation LLoc,
                                                      SourceLocation RLoc,
                                                      ExprArg Base,ExprArg Idx);

  OwningExprResult
  BuildCallToMemberFunction(Scope *S, Expr *MemExpr,
                            SourceLocation LParenLoc, Expr **Args,
                            unsigned NumArgs, SourceLocation *CommaLocs,
                            SourceLocation RParenLoc);
  ExprResult
  BuildCallToObjectOfClassType(Scope *S, Expr *Object, SourceLocation LParenLoc,
                               Expr **Args, unsigned NumArgs,
                               SourceLocation *CommaLocs,
                               SourceLocation RParenLoc);

  OwningExprResult BuildOverloadedArrowExpr(Scope *S, ExprArg Base,
                                            SourceLocation OpLoc);

  /// CheckCallReturnType - Checks that a call expression's return type is
  /// complete. Returns true on failure. The location passed in is the location
  /// that best represents the call.
  bool CheckCallReturnType(QualType ReturnType, SourceLocation Loc,
                           CallExpr *CE, FunctionDecl *FD);
                           
  /// Helpers for dealing with blocks and functions.
  void CheckFallThroughForFunctionDef(Decl *D, Stmt *Body);
  void CheckFallThroughForBlock(QualType BlockTy, Stmt *Body);
  bool CheckParmsForFunctionDef(FunctionDecl *FD);
  void CheckCXXDefaultArguments(FunctionDecl *FD);
  void CheckExtraCXXDefaultArguments(Declarator &D);
  enum ControlFlowKind { NeverFallThrough = 0, MaybeFallThrough = 1,
                         AlwaysFallThrough = 2, NeverFallThroughOrReturn = 3 };
  ControlFlowKind CheckFallThrough(Stmt *);

  Scope *getNonFieldDeclScope(Scope *S);

  /// \name Name lookup
  ///
  /// These routines provide name lookup that is used during semantic
  /// analysis to resolve the various kinds of names (identifiers,
  /// overloaded operator names, constructor names, etc.) into zero or
  /// more declarations within a particular scope. The major entry
  /// points are LookupName, which performs unqualified name lookup,
  /// and LookupQualifiedName, which performs qualified name lookup.
  ///
  /// All name lookup is performed based on some specific criteria,
  /// which specify what names will be visible to name lookup and how
  /// far name lookup should work. These criteria are important both
  /// for capturing language semantics (certain lookups will ignore
  /// certain names, for example) and for performance, since name
  /// lookup is often a bottleneck in the compilation of C++. Name
  /// lookup criteria is specified via the LookupCriteria enumeration.
  ///
  /// The results of name lookup can vary based on the kind of name
  /// lookup performed, the current language, and the translation
  /// unit. In C, for example, name lookup will either return nothing
  /// (no entity found) or a single declaration. In C++, name lookup
  /// can additionally refer to a set of overloaded functions or
  /// result in an ambiguity. All of the possible results of name
  /// lookup are captured by the LookupResult class, which provides
  /// the ability to distinguish among them.
  //@{

  /// @brief Describes the kind of name lookup to perform.
  enum LookupNameKind {
    /// Ordinary name lookup, which finds ordinary names (functions,
    /// variables, typedefs, etc.) in C and most kinds of names
    /// (functions, variables, members, types, etc.) in C++.
    LookupOrdinaryName = 0,
    /// Tag name lookup, which finds the names of enums, classes,
    /// structs, and unions.
    LookupTagName,
    /// Member name lookup, which finds the names of
    /// class/struct/union members.
    LookupMemberName,
    // Look up of an operator name (e.g., operator+) for use with
    // operator overloading. This lookup is similar to ordinary name
    // lookup, but will ignore any declarations that are class
    // members.
    LookupOperatorName,
    /// Look up of a name that precedes the '::' scope resolution
    /// operator in C++. This lookup completely ignores operator, object,
    /// function, and enumerator names (C++ [basic.lookup.qual]p1).
    LookupNestedNameSpecifierName,
    /// Look up a namespace name within a C++ using directive or
    /// namespace alias definition, ignoring non-namespace names (C++
    /// [basic.lookup.udir]p1).
    LookupNamespaceName,
    /// Look up all declarations in a scope with the given name,
    /// including resolved using declarations.  This is appropriate
    /// for checking redeclarations for a using declaration.
    LookupUsingDeclName,
    /// Look up an ordinary name that is going to be redeclared as a
    /// name with linkage. This lookup ignores any declarations that
    /// are outside of the current scope unless they have linkage. See
    /// C99 6.2.2p4-5 and C++ [basic.link]p6.
    LookupRedeclarationWithLinkage,
    /// Look up the name of an Objective-C protocol.
    LookupObjCProtocolName,
    /// Look up the name of an Objective-C implementation
    LookupObjCImplementationName
  };

  enum RedeclarationKind {
    NotForRedeclaration,
    ForRedeclaration
  };

private:
  bool CppLookupName(LookupResult &R, Scope *S);

public:
  /// \brief Look up a name, looking for a single declaration.  Return
  /// null if the results were absent, ambiguous, or overloaded.
  ///
  /// It is preferable to use the elaborated form and explicitly handle
  /// ambiguity and overloaded.
  NamedDecl *LookupSingleName(Scope *S, DeclarationName Name,
                              LookupNameKind NameKind,
                              RedeclarationKind Redecl
                                = NotForRedeclaration);
  bool LookupName(LookupResult &R, Scope *S,
                  bool AllowBuiltinCreation = false);
  bool LookupQualifiedName(LookupResult &R, DeclContext *LookupCtx);
  bool LookupParsedName(LookupResult &R, Scope *S, const CXXScopeSpec *SS,
                        bool AllowBuiltinCreation = false,
                        bool EnteringContext = false);

  ObjCProtocolDecl *LookupProtocol(IdentifierInfo *II);

  void LookupOverloadedOperatorName(OverloadedOperatorKind Op, Scope *S,
                                    QualType T1, QualType T2,
                                    FunctionSet &Functions);

  void ArgumentDependentLookup(DeclarationName Name, bool Operator,
                               Expr **Args, unsigned NumArgs,
                               FunctionSet &Functions);

  void FindAssociatedClassesAndNamespaces(Expr **Args, unsigned NumArgs,
                                   AssociatedNamespaceSet &AssociatedNamespaces,
                                   AssociatedClassSet &AssociatedClasses);

  bool DiagnoseAmbiguousLookup(LookupResult &Result);
  //@}

  ObjCInterfaceDecl *getObjCInterfaceDecl(IdentifierInfo *Id);
  NamedDecl *LazilyCreateBuiltin(IdentifierInfo *II, unsigned ID,
                                 Scope *S, bool ForRedeclaration,
                                 SourceLocation Loc);
  NamedDecl *ImplicitlyDefineFunction(SourceLocation Loc, IdentifierInfo &II,
                                      Scope *S);
  void AddKnownFunctionAttributes(FunctionDecl *FD);

  // More parsing and symbol table subroutines.

  // Decl attributes - this routine is the top level dispatcher.
  void ProcessDeclAttributes(Scope *S, Decl *D, const Declarator &PD);
  void ProcessDeclAttributeList(Scope *S, Decl *D, const AttributeList *AttrList);

  void WarnUndefinedMethod(SourceLocation ImpLoc, ObjCMethodDecl *method,
                           bool &IncompleteImpl);
  void WarnConflictingTypedMethods(ObjCMethodDecl *ImpMethod,
                                   ObjCMethodDecl *IntfMethod);

  bool isPropertyReadonly(ObjCPropertyDecl *PropertyDecl,
                          ObjCInterfaceDecl *IDecl);

  /// CheckProtocolMethodDefs - This routine checks unimplemented
  /// methods declared in protocol, and those referenced by it.
  /// \param IDecl - Used for checking for methods which may have been
  /// inherited.
  void CheckProtocolMethodDefs(SourceLocation ImpLoc,
                               ObjCProtocolDecl *PDecl,
                               bool& IncompleteImpl,
                               const llvm::DenseSet<Selector> &InsMap,
                               const llvm::DenseSet<Selector> &ClsMap,
                               ObjCInterfaceDecl *IDecl);

  /// CheckImplementationIvars - This routine checks if the instance variables
  /// listed in the implelementation match those listed in the interface.
  void CheckImplementationIvars(ObjCImplementationDecl *ImpDecl,
                                ObjCIvarDecl **Fields, unsigned nIvars,
                                SourceLocation Loc);

  /// ImplMethodsVsClassMethods - This is main routine to warn if any method
  /// remains unimplemented in the class or category @implementation.
  void ImplMethodsVsClassMethods(ObjCImplDecl* IMPDecl,
                                 ObjCContainerDecl* IDecl,
                                 bool IncompleteImpl = false);
  
  /// AtomicPropertySetterGetterRules - This routine enforces the rule (via
  /// warning) when atomic property has one but not the other user-declared
  /// setter or getter.
  void AtomicPropertySetterGetterRules(ObjCImplDecl* IMPDecl,
                                       ObjCContainerDecl* IDecl);

  /// MatchTwoMethodDeclarations - Checks if two methods' type match and returns
  /// true, or false, accordingly.
  bool MatchTwoMethodDeclarations(const ObjCMethodDecl *Method,
                                  const ObjCMethodDecl *PrevMethod,
                                  bool matchBasedOnSizeAndAlignment = false);

  /// MatchAllMethodDeclarations - Check methods declaraed in interface or
  /// or protocol against those declared in their implementations.
  void MatchAllMethodDeclarations(const llvm::DenseSet<Selector> &InsMap,
                                  const llvm::DenseSet<Selector> &ClsMap,
                                  llvm::DenseSet<Selector> &InsMapSeen,
                                  llvm::DenseSet<Selector> &ClsMapSeen,
                                  ObjCImplDecl* IMPDecl,
                                  ObjCContainerDecl* IDecl,
                                  bool &IncompleteImpl,
                                  bool ImmediateClass);

  /// AddInstanceMethodToGlobalPool - All instance methods in a translation
  /// unit are added to a global pool. This allows us to efficiently associate
  /// a selector with a method declaraation for purposes of typechecking
  /// messages sent to "id" (where the class of the object is unknown).
  void AddInstanceMethodToGlobalPool(ObjCMethodDecl *Method);

  /// LookupInstanceMethodInGlobalPool - Returns the method and warns if
  /// there are multiple signatures.
  ObjCMethodDecl *LookupInstanceMethodInGlobalPool(Selector Sel, SourceRange R,
                                                   bool warn=true);

  /// LookupFactoryMethodInGlobalPool - Returns the method and warns if
  /// there are multiple signatures.
  ObjCMethodDecl *LookupFactoryMethodInGlobalPool(Selector Sel, SourceRange R);

  /// AddFactoryMethodToGlobalPool - Same as above, but for factory methods.
  void AddFactoryMethodToGlobalPool(ObjCMethodDecl *Method);
  //===--------------------------------------------------------------------===//
  // Statement Parsing Callbacks: SemaStmt.cpp.
public:
  virtual OwningStmtResult ActOnExprStmt(FullExprArg Expr);

  virtual OwningStmtResult ActOnNullStmt(SourceLocation SemiLoc);
  virtual OwningStmtResult ActOnCompoundStmt(SourceLocation L, SourceLocation R,
                                             MultiStmtArg Elts,
                                             bool isStmtExpr);
  virtual OwningStmtResult ActOnDeclStmt(DeclGroupPtrTy Decl,
                                         SourceLocation StartLoc,
                                         SourceLocation EndLoc);
  virtual void ActOnForEachDeclStmt(DeclGroupPtrTy Decl);
  virtual OwningStmtResult ActOnCaseStmt(SourceLocation CaseLoc, ExprArg LHSVal,
                                    SourceLocation DotDotDotLoc, ExprArg RHSVal,
                                    SourceLocation ColonLoc);
  virtual void ActOnCaseStmtBody(StmtTy *CaseStmt, StmtArg SubStmt);

  virtual OwningStmtResult ActOnDefaultStmt(SourceLocation DefaultLoc,
                                            SourceLocation ColonLoc,
                                            StmtArg SubStmt, Scope *CurScope);
  virtual OwningStmtResult ActOnLabelStmt(SourceLocation IdentLoc,
                                          IdentifierInfo *II,
                                          SourceLocation ColonLoc,
                                          StmtArg SubStmt);
  virtual OwningStmtResult ActOnIfStmt(SourceLocation IfLoc,
                                       FullExprArg CondVal, DeclPtrTy CondVar,
                                       StmtArg ThenVal,
                                       SourceLocation ElseLoc, StmtArg ElseVal);
  virtual OwningStmtResult ActOnStartOfSwitchStmt(FullExprArg Cond, 
                                                  DeclPtrTy CondVar);
  virtual OwningStmtResult ActOnFinishSwitchStmt(SourceLocation SwitchLoc,
                                                 StmtArg Switch, StmtArg Body);
  virtual OwningStmtResult ActOnWhileStmt(SourceLocation WhileLoc,
                                          FullExprArg Cond,
                                          DeclPtrTy CondVar, StmtArg Body);
  virtual OwningStmtResult ActOnDoStmt(SourceLocation DoLoc, StmtArg Body,
                                       SourceLocation WhileLoc,
                                       SourceLocation CondLParen, ExprArg Cond,
                                       SourceLocation CondRParen);

  virtual OwningStmtResult ActOnForStmt(SourceLocation ForLoc,
                                        SourceLocation LParenLoc,
                                        StmtArg First, FullExprArg Second,
                                        DeclPtrTy SecondVar,
                                        FullExprArg Third, 
                                        SourceLocation RParenLoc,
                                        StmtArg Body);
  virtual OwningStmtResult ActOnObjCForCollectionStmt(SourceLocation ForColLoc,
                                       SourceLocation LParenLoc,
                                       StmtArg First, ExprArg Second,
                                       SourceLocation RParenLoc, StmtArg Body);

  virtual OwningStmtResult ActOnGotoStmt(SourceLocation GotoLoc,
                                         SourceLocation LabelLoc,
                                         IdentifierInfo *LabelII);
  virtual OwningStmtResult ActOnIndirectGotoStmt(SourceLocation GotoLoc,
                                                 SourceLocation StarLoc,
                                                 ExprArg DestExp);
  virtual OwningStmtResult ActOnContinueStmt(SourceLocation ContinueLoc,
                                             Scope *CurScope);
  virtual OwningStmtResult ActOnBreakStmt(SourceLocation GotoLoc,
                                          Scope *CurScope);

  virtual OwningStmtResult ActOnReturnStmt(SourceLocation ReturnLoc,
                                           ExprArg RetValExp);
  OwningStmtResult ActOnBlockReturnStmt(SourceLocation ReturnLoc,
                                        Expr *RetValExp);

  virtual OwningStmtResult ActOnAsmStmt(SourceLocation AsmLoc,
                                        bool IsSimple,
                                        bool IsVolatile,
                                        unsigned NumOutputs,
                                        unsigned NumInputs,
                                        std::string *Names,
                                        MultiExprArg Constraints,
                                        MultiExprArg Exprs,
                                        ExprArg AsmString,
                                        MultiExprArg Clobbers,
                                        SourceLocation RParenLoc);

  virtual OwningStmtResult ActOnObjCAtCatchStmt(SourceLocation AtLoc,
                                                SourceLocation RParen,
                                                DeclPtrTy Parm, StmtArg Body,
                                                StmtArg CatchList);

  virtual OwningStmtResult ActOnObjCAtFinallyStmt(SourceLocation AtLoc,
                                                  StmtArg Body);

  virtual OwningStmtResult ActOnObjCAtTryStmt(SourceLocation AtLoc,
                                              StmtArg Try,
                                              StmtArg Catch, StmtArg Finally);

  virtual OwningStmtResult ActOnObjCAtThrowStmt(SourceLocation AtLoc,
                                                ExprArg Throw,
                                                Scope *CurScope);
  virtual OwningStmtResult ActOnObjCAtSynchronizedStmt(SourceLocation AtLoc,
                                                       ExprArg SynchExpr,
                                                       StmtArg SynchBody);

  VarDecl *BuildExceptionDeclaration(Scope *S, QualType ExDeclType,
                                     TypeSourceInfo *TInfo,
                                     IdentifierInfo *Name,
                                     SourceLocation Loc,
                                     SourceRange Range);
  virtual DeclPtrTy ActOnExceptionDeclarator(Scope *S, Declarator &D);

  virtual OwningStmtResult ActOnCXXCatchBlock(SourceLocation CatchLoc,
                                              DeclPtrTy ExDecl,
                                              StmtArg HandlerBlock);
  virtual OwningStmtResult ActOnCXXTryBlock(SourceLocation TryLoc,
                                            StmtArg TryBlock,
                                            MultiStmtArg Handlers);
  void DiagnoseReturnInConstructorExceptionHandler(CXXTryStmt *TryBlock);

  /// DiagnoseUnusedExprResult - If the statement passed in is an expression
  /// whose result is unused, warn.
  void DiagnoseUnusedExprResult(const Stmt *S);

  ParsingDeclStackState PushParsingDeclaration();
  void PopParsingDeclaration(ParsingDeclStackState S, DeclPtrTy D);
  void EmitDeprecationWarning(NamedDecl *D, SourceLocation Loc);

  //===--------------------------------------------------------------------===//
  // Expression Parsing Callbacks: SemaExpr.cpp.

  bool DiagnoseUseOfDecl(NamedDecl *D, SourceLocation Loc);
  bool DiagnosePropertyAccessorMismatch(ObjCPropertyDecl *PD,
                                        ObjCMethodDecl *Getter,
                                        SourceLocation Loc);
  void DiagnoseSentinelCalls(NamedDecl *D, SourceLocation Loc,
                             Expr **Args, unsigned NumArgs);

  void CheckSignCompare(Expr *LHS, Expr *RHS, SourceLocation Loc,
                        const PartialDiagnostic &PD,
                        bool Equality = false);

  virtual void
  PushExpressionEvaluationContext(ExpressionEvaluationContext NewContext);

  virtual void PopExpressionEvaluationContext();

  void MarkDeclarationReferenced(SourceLocation Loc, Decl *D);
  bool DiagRuntimeBehavior(SourceLocation Loc, const PartialDiagnostic &PD);
  
  // Primary Expressions.
  virtual SourceRange getExprRange(ExprTy *E) const;

  virtual OwningExprResult ActOnIdExpression(Scope *S,
                                             const CXXScopeSpec &SS,
                                             UnqualifiedId &Name,
                                             bool HasTrailingLParen,
                                             bool IsAddressOfOperand);

  bool DiagnoseEmptyLookup(const CXXScopeSpec &SS, LookupResult &R);

  OwningExprResult LookupInObjCMethod(LookupResult &R,
                                      Scope *S,
                                      IdentifierInfo *II);

  OwningExprResult ActOnDependentIdExpression(const CXXScopeSpec &SS,
                                              DeclarationName Name,
                                              SourceLocation NameLoc,
                                              bool isAddressOfOperand,
                                const TemplateArgumentListInfo *TemplateArgs);
  
  OwningExprResult BuildDeclRefExpr(ValueDecl *D, QualType Ty,
                                    SourceLocation Loc,
                                    const CXXScopeSpec *SS = 0);
  VarDecl *BuildAnonymousStructUnionMemberPath(FieldDecl *Field,
                                    llvm::SmallVectorImpl<FieldDecl *> &Path);
  OwningExprResult
  BuildAnonymousStructUnionMemberReference(SourceLocation Loc,
                                           FieldDecl *Field,
                                           Expr *BaseObjectExpr = 0,
                                      SourceLocation OpLoc = SourceLocation());
  OwningExprResult BuildPossibleImplicitMemberExpr(const CXXScopeSpec &SS,
                                                   LookupResult &R,
                                const TemplateArgumentListInfo *TemplateArgs);
  OwningExprResult BuildImplicitMemberExpr(const CXXScopeSpec &SS,
                                           LookupResult &R,
                                const TemplateArgumentListInfo *TemplateArgs,
                                           bool IsDefiniteInstance);
  bool UseArgumentDependentLookup(const CXXScopeSpec &SS,
                                  const LookupResult &R,
                                  bool HasTrailingLParen);

  OwningExprResult BuildQualifiedDeclarationNameExpr(const CXXScopeSpec &SS,
                                                     DeclarationName Name,
                                                     SourceLocation NameLoc);
  OwningExprResult BuildDependentDeclRefExpr(const CXXScopeSpec &SS,
                                             DeclarationName Name,
                                             SourceLocation NameLoc,
                                const TemplateArgumentListInfo *TemplateArgs);

  OwningExprResult BuildDeclarationNameExpr(const CXXScopeSpec &SS,
                                            LookupResult &R,
                                            bool ADL);
  OwningExprResult BuildDeclarationNameExpr(const CXXScopeSpec &SS,
                                            SourceLocation Loc,
                                            NamedDecl *D);

  virtual OwningExprResult ActOnPredefinedExpr(SourceLocation Loc,
                                               tok::TokenKind Kind);
  virtual OwningExprResult ActOnNumericConstant(const Token &);
  virtual OwningExprResult ActOnCharacterConstant(const Token &);
  virtual OwningExprResult ActOnParenExpr(SourceLocation L, SourceLocation R,
                                          ExprArg Val);
  virtual OwningExprResult ActOnParenOrParenListExpr(SourceLocation L,
                                              SourceLocation R,
                                              MultiExprArg Val,
                                              TypeTy *TypeOfCast=0);

  /// ActOnStringLiteral - The specified tokens were lexed as pasted string
  /// fragments (e.g. "foo" "bar" L"baz").
  virtual OwningExprResult ActOnStringLiteral(const Token *Toks,
                                              unsigned NumToks);

  // Binary/Unary Operators.  'Tok' is the token for the operator.
  OwningExprResult CreateBuiltinUnaryOp(SourceLocation OpLoc,
                                        unsigned OpcIn,
                                        ExprArg InputArg);
  OwningExprResult BuildUnaryOp(Scope *S, SourceLocation OpLoc,
                                UnaryOperator::Opcode Opc, ExprArg input);
  virtual OwningExprResult ActOnUnaryOp(Scope *S, SourceLocation OpLoc,
                                        tok::TokenKind Op, ExprArg Input);

  OwningExprResult CreateSizeOfAlignOfExpr(TypeSourceInfo *T,
                                           SourceLocation OpLoc,
                                           bool isSizeOf, SourceRange R);
  OwningExprResult CreateSizeOfAlignOfExpr(Expr *E, SourceLocation OpLoc,
                                           bool isSizeOf, SourceRange R);
  virtual OwningExprResult
    ActOnSizeOfAlignOfExpr(SourceLocation OpLoc, bool isSizeof, bool isType,
                           void *TyOrEx, const SourceRange &ArgRange);

  bool CheckAlignOfExpr(Expr *E, SourceLocation OpLoc, const SourceRange &R);
  bool CheckSizeOfAlignOfOperand(QualType type, SourceLocation OpLoc,
                                 const SourceRange &R, bool isSizeof);

  virtual OwningExprResult ActOnPostfixUnaryOp(Scope *S, SourceLocation OpLoc,
                                               tok::TokenKind Kind,
                                               ExprArg Input);

  virtual OwningExprResult ActOnArraySubscriptExpr(Scope *S, ExprArg Base,
                                                   SourceLocation LLoc,
                                                   ExprArg Idx,
                                                   SourceLocation RLoc);
  OwningExprResult CreateBuiltinArraySubscriptExpr(ExprArg Base,
                                                   SourceLocation LLoc,
                                                   ExprArg Idx,
                                                   SourceLocation RLoc);

  OwningExprResult BuildMemberReferenceExpr(ExprArg Base,
                                            QualType BaseType,
                                            SourceLocation OpLoc,
                                            bool IsArrow,
                                            const CXXScopeSpec &SS,
                                            NamedDecl *FirstQualifierInScope,
                                            DeclarationName Name,
                                            SourceLocation NameLoc,
                                const TemplateArgumentListInfo *TemplateArgs);

  OwningExprResult BuildMemberReferenceExpr(ExprArg Base,
                                            QualType BaseType,
                                            SourceLocation OpLoc, bool IsArrow,
                                            const CXXScopeSpec &SS,
                                            LookupResult &R,
                                 const TemplateArgumentListInfo *TemplateArgs);

  OwningExprResult LookupMemberExpr(LookupResult &R, Expr *&Base,
                                    bool &IsArrow, SourceLocation OpLoc,
                                    const CXXScopeSpec &SS,
                                    NamedDecl *FirstQualifierInScope,
                                    DeclPtrTy ObjCImpDecl);

  bool CheckQualifiedMemberReference(Expr *BaseExpr, QualType BaseType,
                                     const CXXScopeSpec &SS,
                                     const LookupResult &R);

  OwningExprResult ActOnDependentMemberExpr(ExprArg Base,
                                            QualType BaseType,
                                            bool IsArrow,
                                            SourceLocation OpLoc,
                                            const CXXScopeSpec &SS,
                                            NamedDecl *FirstQualifierInScope,
                                            DeclarationName Name,
                                            SourceLocation NameLoc,
                               const TemplateArgumentListInfo *TemplateArgs);

  virtual OwningExprResult ActOnMemberAccessExpr(Scope *S, ExprArg Base,
                                                 SourceLocation OpLoc,
                                                 tok::TokenKind OpKind,
                                                 const CXXScopeSpec &SS,
                                                 UnqualifiedId &Member,
                                                 DeclPtrTy ObjCImpDecl,
                                                 bool HasTrailingLParen);
    
  virtual void ActOnDefaultCtorInitializers(DeclPtrTy CDtorDecl);
  bool ConvertArgumentsForCall(CallExpr *Call, Expr *Fn,
                               FunctionDecl *FDecl,
                               const FunctionProtoType *Proto,
                               Expr **Args, unsigned NumArgs,
                               SourceLocation RParenLoc);

  /// ActOnCallExpr - Handle a call to Fn with the specified array of arguments.
  /// This provides the location of the left/right parens and a list of comma
  /// locations.
  virtual OwningExprResult ActOnCallExpr(Scope *S, ExprArg Fn,
                                         SourceLocation LParenLoc,
                                         MultiExprArg Args,
                                         SourceLocation *CommaLocs,
                                         SourceLocation RParenLoc);
  OwningExprResult BuildResolvedCallExpr(Expr *Fn,
                                         NamedDecl *NDecl,
                                         SourceLocation LParenLoc,
                                         Expr **Args, unsigned NumArgs,
                                         SourceLocation RParenLoc);

  virtual OwningExprResult ActOnCastExpr(Scope *S, SourceLocation LParenLoc,
                                         TypeTy *Ty, SourceLocation RParenLoc,
                                         ExprArg Op);
  virtual bool TypeIsVectorType(TypeTy *Ty) {
    return GetTypeFromParser(Ty)->isVectorType();
  }

  OwningExprResult MaybeConvertParenListExprToParenExpr(Scope *S, ExprArg ME);
  OwningExprResult ActOnCastOfParenListExpr(Scope *S, SourceLocation LParenLoc,
                                            SourceLocation RParenLoc, ExprArg E,
                                            QualType Ty);

  virtual OwningExprResult ActOnCompoundLiteral(SourceLocation LParenLoc,
                                                TypeTy *Ty,
                                                SourceLocation RParenLoc,
                                                ExprArg Op);

  virtual OwningExprResult ActOnInitList(SourceLocation LParenLoc,
                                         MultiExprArg InitList,
                                         SourceLocation RParenLoc);

  virtual OwningExprResult ActOnDesignatedInitializer(Designation &Desig,
                                                      SourceLocation Loc,
                                                      bool GNUSyntax,
                                                      OwningExprResult Init);

  virtual OwningExprResult ActOnBinOp(Scope *S, SourceLocation TokLoc,
                                      tok::TokenKind Kind,
                                      ExprArg LHS, ExprArg RHS);
  OwningExprResult BuildBinOp(Scope *S, SourceLocation OpLoc,
                              BinaryOperator::Opcode Opc,
                              Expr *lhs, Expr *rhs);
  OwningExprResult CreateBuiltinBinOp(SourceLocation TokLoc,
                                      unsigned Opc, Expr *lhs, Expr *rhs);

  /// ActOnConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
  /// in the case of a the GNU conditional expr extension.
  virtual OwningExprResult ActOnConditionalOp(SourceLocation QuestionLoc,
                                              SourceLocation ColonLoc,
                                              ExprArg Cond, ExprArg LHS,
                                              ExprArg RHS);

  /// ActOnAddrLabel - Parse the GNU address of label extension: "&&foo".
  virtual OwningExprResult ActOnAddrLabel(SourceLocation OpLoc,
                                          SourceLocation LabLoc,
                                          IdentifierInfo *LabelII);

  virtual OwningExprResult ActOnStmtExpr(SourceLocation LPLoc, StmtArg SubStmt,
                                         SourceLocation RPLoc); // "({..})"

  /// __builtin_offsetof(type, a.b[123][456].c)
  virtual OwningExprResult ActOnBuiltinOffsetOf(Scope *S,
                                                SourceLocation BuiltinLoc,
                                                SourceLocation TypeLoc,
                                                TypeTy *Arg1,
                                                OffsetOfComponent *CompPtr,
                                                unsigned NumComponents,
                                                SourceLocation RParenLoc);

  // __builtin_types_compatible_p(type1, type2)
  virtual OwningExprResult ActOnTypesCompatibleExpr(SourceLocation BuiltinLoc,
                                                    TypeTy *arg1, TypeTy *arg2,
                                                    SourceLocation RPLoc);

  // __builtin_choose_expr(constExpr, expr1, expr2)
  virtual OwningExprResult ActOnChooseExpr(SourceLocation BuiltinLoc,
                                           ExprArg cond, ExprArg expr1,
                                           ExprArg expr2, SourceLocation RPLoc);

  // __builtin_va_arg(expr, type)
  virtual OwningExprResult ActOnVAArg(SourceLocation BuiltinLoc,
                                      ExprArg expr, TypeTy *type,
                                      SourceLocation RPLoc);

  // __null
  virtual OwningExprResult ActOnGNUNullExpr(SourceLocation TokenLoc);

  //===------------------------- "Block" Extension ------------------------===//

  /// ActOnBlockStart - This callback is invoked when a block literal is
  /// started.
  virtual void ActOnBlockStart(SourceLocation CaretLoc, Scope *CurScope);

  /// ActOnBlockArguments - This callback allows processing of block arguments.
  /// If there are no arguments, this is still invoked.
  virtual void ActOnBlockArguments(Declarator &ParamInfo, Scope *CurScope);

  /// ActOnBlockError - If there is an error parsing a block, this callback
  /// is invoked to pop the information about the block from the action impl.
  virtual void ActOnBlockError(SourceLocation CaretLoc, Scope *CurScope);

  /// ActOnBlockStmtExpr - This is called when the body of a block statement
  /// literal was successfully completed.  ^(int x){...}
  virtual OwningExprResult ActOnBlockStmtExpr(SourceLocation CaretLoc,
                                              StmtArg Body, Scope *CurScope);

  //===---------------------------- C++ Features --------------------------===//

  // Act on C++ namespaces
  virtual DeclPtrTy ActOnStartNamespaceDef(Scope *S, SourceLocation IdentLoc,
                                           IdentifierInfo *Ident,
                                           SourceLocation LBrace);
  virtual void ActOnFinishNamespaceDef(DeclPtrTy Dcl, SourceLocation RBrace);

  virtual DeclPtrTy ActOnUsingDirective(Scope *CurScope,
                                        SourceLocation UsingLoc,
                                        SourceLocation NamespcLoc,
                                        const CXXScopeSpec &SS,
                                        SourceLocation IdentLoc,
                                        IdentifierInfo *NamespcName,
                                        AttributeList *AttrList);

  void PushUsingDirective(Scope *S, UsingDirectiveDecl *UDir);

  virtual DeclPtrTy ActOnNamespaceAliasDef(Scope *CurScope,
                                           SourceLocation NamespaceLoc,
                                           SourceLocation AliasLoc,
                                           IdentifierInfo *Alias,
                                           const CXXScopeSpec &SS,
                                           SourceLocation IdentLoc,
                                           IdentifierInfo *Ident);

  void HideUsingShadowDecl(Scope *S, UsingShadowDecl *Shadow);
  bool CheckUsingShadowDecl(UsingDecl *UD, NamedDecl *Target,
                            const LookupResult &PreviousDecls);
  UsingShadowDecl *BuildUsingShadowDecl(Scope *S, UsingDecl *UD,
                                        NamedDecl *Target);

  bool CheckUsingDeclRedeclaration(SourceLocation UsingLoc,
                                   bool isTypeName,
                                   const CXXScopeSpec &SS,
                                   SourceLocation NameLoc,
                                   const LookupResult &Previous);
  bool CheckUsingDeclQualifier(SourceLocation UsingLoc,
                               const CXXScopeSpec &SS,
                               SourceLocation NameLoc);

  NamedDecl *BuildUsingDeclaration(Scope *S, AccessSpecifier AS,
                                   SourceLocation UsingLoc,
                                   const CXXScopeSpec &SS,
                                   SourceLocation IdentLoc,
                                   DeclarationName Name,
                                   AttributeList *AttrList,
                                   bool IsInstantiation,
                                   bool IsTypeName,
                                   SourceLocation TypenameLoc);

  virtual DeclPtrTy ActOnUsingDeclaration(Scope *CurScope,
                                          AccessSpecifier AS,
                                          bool HasUsingKeyword,
                                          SourceLocation UsingLoc,
                                          const CXXScopeSpec &SS,
                                          UnqualifiedId &Name,
                                          AttributeList *AttrList,
                                          bool IsTypeName,
                                          SourceLocation TypenameLoc);

  /// AddCXXDirectInitializerToDecl - This action is called immediately after
  /// ActOnDeclarator, when a C++ direct initializer is present.
  /// e.g: "int x(1);"
  virtual void AddCXXDirectInitializerToDecl(DeclPtrTy Dcl,
                                             SourceLocation LParenLoc,
                                             MultiExprArg Exprs,
                                             SourceLocation *CommaLocs,
                                             SourceLocation RParenLoc);

  /// InitializeVarWithConstructor - Creates an CXXConstructExpr
  /// and sets it as the initializer for the the passed in VarDecl.
  bool InitializeVarWithConstructor(VarDecl *VD,
                                    CXXConstructorDecl *Constructor,
                                    MultiExprArg Exprs);

  /// BuildCXXConstructExpr - Creates a complete call to a constructor,
  /// including handling of its default argument expressions.
  OwningExprResult BuildCXXConstructExpr(SourceLocation ConstructLoc,
                                         QualType DeclInitType,
                                         CXXConstructorDecl *Constructor,
                                         MultiExprArg Exprs,
                                         bool RequiresZeroInit = false);

  // FIXME: Can re remove this and have the above BuildCXXConstructExpr check if
  // the constructor can be elidable?
  OwningExprResult BuildCXXConstructExpr(SourceLocation ConstructLoc,
                                         QualType DeclInitType,
                                         CXXConstructorDecl *Constructor,
                                         bool Elidable,
                                         MultiExprArg Exprs,
                                         bool RequiresZeroInit = false);

  OwningExprResult BuildCXXTemporaryObjectExpr(CXXConstructorDecl *Cons,
                                               QualType writtenTy,
                                               SourceLocation tyBeginLoc,
                                               MultiExprArg Args,
                                               SourceLocation rParenLoc);

  OwningExprResult BuildCXXCastArgument(SourceLocation CastLoc,
                                        QualType Ty,
                                        CastExpr::CastKind Kind,
                                        CXXMethodDecl *Method,
                                        ExprArg Arg);

  /// BuildCXXDefaultArgExpr - Creates a CXXDefaultArgExpr, instantiating
  /// the default expr if needed.
  OwningExprResult BuildCXXDefaultArgExpr(SourceLocation CallLoc,
                                          FunctionDecl *FD,
                                          ParmVarDecl *Param);

  /// FinalizeVarWithDestructor - Prepare for calling destructor on the
  /// constructed variable.
  void FinalizeVarWithDestructor(VarDecl *VD, QualType DeclInitType);

  /// DefineImplicitDefaultConstructor - Checks for feasibility of
  /// defining this constructor as the default constructor.
  void DefineImplicitDefaultConstructor(SourceLocation CurrentLocation,
                                        CXXConstructorDecl *Constructor);

  /// DefineImplicitDestructor - Checks for feasibility of
  /// defining this destructor as the default destructor.
  void DefineImplicitDestructor(SourceLocation CurrentLocation,
                                        CXXDestructorDecl *Destructor);

  /// DefineImplicitCopyConstructor - Checks for feasibility of
  /// defining this constructor as the copy constructor.
  void DefineImplicitCopyConstructor(SourceLocation CurrentLocation,
                                     CXXConstructorDecl *Constructor,
                                     unsigned TypeQuals);

  /// DefineImplicitOverloadedAssign - Checks for feasibility of
  /// defining implicit this overloaded assignment operator.
  void DefineImplicitOverloadedAssign(SourceLocation CurrentLocation,
                                      CXXMethodDecl *MethodDecl);

  /// getAssignOperatorMethod - Returns the default copy assignmment operator
  /// for the class.
  CXXMethodDecl *getAssignOperatorMethod(SourceLocation CurrentLocation,
                                         ParmVarDecl *Decl,
                                         CXXRecordDecl *ClassDecl);

  /// MaybeBindToTemporary - If the passed in expression has a record type with
  /// a non-trivial destructor, this will return CXXBindTemporaryExpr. Otherwise
  /// it simply returns the passed in expression.
  OwningExprResult MaybeBindToTemporary(Expr *E);

  CXXConstructorDecl *
  TryInitializationByConstructor(QualType ClassType,
                                 Expr **Args, unsigned NumArgs,
                                 SourceLocation Loc,
                                 InitializationKind Kind);
  
  CXXConstructorDecl *
  PerformInitializationByConstructor(QualType ClassType,
                                     MultiExprArg ArgsPtr,
                                     SourceLocation Loc, SourceRange Range,
                                     DeclarationName InitEntity,
                                     InitializationKind Kind,
                       ASTOwningVector<&ActionBase::DeleteExpr> &ConvertedArgs);

  bool CompleteConstructorCall(CXXConstructorDecl *Constructor,
                               MultiExprArg ArgsPtr,
                               SourceLocation Loc,                                    
                      ASTOwningVector<&ActionBase::DeleteExpr> &ConvertedArgs);
    
  /// ActOnCXXNamedCast - Parse {dynamic,static,reinterpret,const}_cast's.
  virtual OwningExprResult ActOnCXXNamedCast(SourceLocation OpLoc,
                                             tok::TokenKind Kind,
                                             SourceLocation LAngleBracketLoc,
                                             TypeTy *Ty,
                                             SourceLocation RAngleBracketLoc,
                                             SourceLocation LParenLoc,
                                             ExprArg E,
                                             SourceLocation RParenLoc);

  /// ActOnCXXTypeid - Parse typeid( something ).
  virtual OwningExprResult ActOnCXXTypeid(SourceLocation OpLoc,
                                          SourceLocation LParenLoc, bool isType,
                                          void *TyOrExpr,
                                          SourceLocation RParenLoc);

  //// ActOnCXXThis -  Parse 'this' pointer.
  virtual OwningExprResult ActOnCXXThis(SourceLocation ThisLoc);

  /// ActOnCXXBoolLiteral - Parse {true,false} literals.
  virtual OwningExprResult ActOnCXXBoolLiteral(SourceLocation OpLoc,
                                               tok::TokenKind Kind);

  /// ActOnCXXNullPtrLiteral - Parse 'nullptr'.
  virtual OwningExprResult ActOnCXXNullPtrLiteral(SourceLocation Loc);

  //// ActOnCXXThrow -  Parse throw expressions.
  virtual OwningExprResult ActOnCXXThrow(SourceLocation OpLoc,
                                         ExprArg expr);
  bool CheckCXXThrowOperand(SourceLocation ThrowLoc, Expr *&E);

  /// ActOnCXXTypeConstructExpr - Parse construction of a specified type.
  /// Can be interpreted either as function-style casting ("int(x)")
  /// or class type construction ("ClassType(x,y,z)")
  /// or creation of a value-initialized type ("int()").
  virtual OwningExprResult ActOnCXXTypeConstructExpr(SourceRange TypeRange,
                                                     TypeTy *TypeRep,
                                                     SourceLocation LParenLoc,
                                                     MultiExprArg Exprs,
                                                     SourceLocation *CommaLocs,
                                                     SourceLocation RParenLoc);

  /// ActOnCXXNew - Parsed a C++ 'new' expression.
  virtual OwningExprResult ActOnCXXNew(SourceLocation StartLoc, bool UseGlobal,
                                       SourceLocation PlacementLParen,
                                       MultiExprArg PlacementArgs,
                                       SourceLocation PlacementRParen,
                                       bool ParenTypeId, Declarator &D,
                                       SourceLocation ConstructorLParen,
                                       MultiExprArg ConstructorArgs,
                                       SourceLocation ConstructorRParen);
  OwningExprResult BuildCXXNew(SourceLocation StartLoc, bool UseGlobal,
                               SourceLocation PlacementLParen,
                               MultiExprArg PlacementArgs,
                               SourceLocation PlacementRParen,
                               bool ParenTypeId,
                               QualType AllocType,
                               SourceLocation TypeLoc,
                               SourceRange TypeRange,
                               ExprArg ArraySize,
                               SourceLocation ConstructorLParen,
                               MultiExprArg ConstructorArgs,
                               SourceLocation ConstructorRParen);

  bool CheckAllocatedType(QualType AllocType, SourceLocation Loc,
                          SourceRange R);
  bool FindAllocationFunctions(SourceLocation StartLoc, SourceRange Range,
                               bool UseGlobal, QualType AllocType, bool IsArray,
                               Expr **PlaceArgs, unsigned NumPlaceArgs,
                               FunctionDecl *&OperatorNew,
                               FunctionDecl *&OperatorDelete);
  bool FindAllocationOverload(SourceLocation StartLoc, SourceRange Range,
                              DeclarationName Name, Expr** Args,
                              unsigned NumArgs, DeclContext *Ctx,
                              bool AllowMissing, FunctionDecl *&Operator);
  void DeclareGlobalNewDelete();
  void DeclareGlobalAllocationFunction(DeclarationName Name, QualType Return,
                                       QualType Argument,
                                       bool addMallocAttr = false);

  bool FindDeallocationFunction(SourceLocation StartLoc, CXXRecordDecl *RD, 
                                DeclarationName Name, FunctionDecl* &Operator);

  /// ActOnCXXDelete - Parsed a C++ 'delete' expression
  virtual OwningExprResult ActOnCXXDelete(SourceLocation StartLoc,
                                          bool UseGlobal, bool ArrayForm,
                                          ExprArg Operand);

  virtual DeclResult ActOnCXXConditionDeclaration(Scope *S,
                                                  Declarator &D);
  OwningExprResult CheckConditionVariable(VarDecl *ConditionVar);
                                          
  /// ActOnUnaryTypeTrait - Parsed one of the unary type trait support
  /// pseudo-functions.
  virtual OwningExprResult ActOnUnaryTypeTrait(UnaryTypeTrait OTT,
                                               SourceLocation KWLoc,
                                               SourceLocation LParen,
                                               TypeTy *Ty,
                                               SourceLocation RParen);

  virtual OwningExprResult ActOnStartCXXMemberReference(Scope *S,
                                                        ExprArg Base,
                                                        SourceLocation OpLoc,
                                                        tok::TokenKind OpKind,
                                                        TypeTy *&ObjectType);

  /// MaybeCreateCXXExprWithTemporaries - If the list of temporaries is
  /// non-empty, will create a new CXXExprWithTemporaries expression.
  /// Otherwise, just returs the passed in expression.
  Expr *MaybeCreateCXXExprWithTemporaries(Expr *SubExpr);

  FullExpr CreateFullExpr(Expr *SubExpr);
  
  virtual OwningExprResult ActOnFinishFullExpr(ExprArg Expr);

  bool RequireCompleteDeclContext(const CXXScopeSpec &SS);

  DeclContext *computeDeclContext(QualType T);
  DeclContext *computeDeclContext(const CXXScopeSpec &SS,
                                  bool EnteringContext = false);
  bool isDependentScopeSpecifier(const CXXScopeSpec &SS);
  CXXRecordDecl *getCurrentInstantiationOf(NestedNameSpecifier *NNS);
  bool isUnknownSpecialization(const CXXScopeSpec &SS);

  /// ActOnCXXGlobalScopeSpecifier - Return the object that represents the
  /// global scope ('::').
  virtual CXXScopeTy *ActOnCXXGlobalScopeSpecifier(Scope *S,
                                                   SourceLocation CCLoc);

  bool isAcceptableNestedNameSpecifier(NamedDecl *SD);
  NamedDecl *FindFirstQualifierInScope(Scope *S, NestedNameSpecifier *NNS);


  CXXScopeTy *BuildCXXNestedNameSpecifier(Scope *S,
                                          const CXXScopeSpec &SS,
                                          SourceLocation IdLoc,
                                          SourceLocation CCLoc,
                                          IdentifierInfo &II,
                                          QualType ObjectType,
                                          NamedDecl *ScopeLookupResult,
                                          bool EnteringContext,
                                          bool ErrorRecoveryLookup);

  virtual CXXScopeTy *ActOnCXXNestedNameSpecifier(Scope *S,
                                                  const CXXScopeSpec &SS,
                                                  SourceLocation IdLoc,
                                                  SourceLocation CCLoc,
                                                  IdentifierInfo &II,
                                                  TypeTy *ObjectType,
                                                  bool EnteringContext);

  virtual bool IsInvalidUnlessNestedName(Scope *S,
                                         const CXXScopeSpec &SS,
                                         IdentifierInfo &II,
                                         TypeTy *ObjectType,
                                         bool EnteringContext);
  
  /// ActOnCXXNestedNameSpecifier - Called during parsing of a
  /// nested-name-specifier that involves a template-id, e.g.,
  /// "foo::bar<int, float>::", and now we need to build a scope
  /// specifier. \p SS is empty or the previously parsed nested-name
  /// part ("foo::"), \p Type is the already-parsed class template
  /// specialization (or other template-id that names a type), \p
  /// TypeRange is the source range where the type is located, and \p
  /// CCLoc is the location of the trailing '::'.
  virtual CXXScopeTy *ActOnCXXNestedNameSpecifier(Scope *S,
                                                  const CXXScopeSpec &SS,
                                                  TypeTy *Type,
                                                  SourceRange TypeRange,
                                                  SourceLocation CCLoc);

  virtual bool ShouldEnterDeclaratorScope(Scope *S, const CXXScopeSpec &SS);

  /// ActOnCXXEnterDeclaratorScope - Called when a C++ scope specifier (global
  /// scope or nested-name-specifier) is parsed, part of a declarator-id.
  /// After this method is called, according to [C++ 3.4.3p3], names should be
  /// looked up in the declarator-id's scope, until the declarator is parsed and
  /// ActOnCXXExitDeclaratorScope is called.
  /// The 'SS' should be a non-empty valid CXXScopeSpec.
  virtual bool ActOnCXXEnterDeclaratorScope(Scope *S, const CXXScopeSpec &SS);

  /// ActOnCXXExitDeclaratorScope - Called when a declarator that previously
  /// invoked ActOnCXXEnterDeclaratorScope(), is finished. 'SS' is the same
  /// CXXScopeSpec that was passed to ActOnCXXEnterDeclaratorScope as well.
  /// Used to indicate that names should revert to being looked up in the
  /// defining scope.
  virtual void ActOnCXXExitDeclaratorScope(Scope *S, const CXXScopeSpec &SS);

  /// ActOnCXXEnterDeclInitializer - Invoked when we are about to parse an
  /// initializer for the declaration 'Dcl'.
  /// After this method is called, according to [C++ 3.4.1p13], if 'Dcl' is a
  /// static data member of class X, names should be looked up in the scope of
  /// class X.
  virtual void ActOnCXXEnterDeclInitializer(Scope *S, DeclPtrTy Dcl);

  /// ActOnCXXExitDeclInitializer - Invoked after we are finished parsing an
  /// initializer for the declaration 'Dcl'.
  virtual void ActOnCXXExitDeclInitializer(Scope *S, DeclPtrTy Dcl);

  // ParseObjCStringLiteral - Parse Objective-C string literals.
  virtual ExprResult ParseObjCStringLiteral(SourceLocation *AtLocs,
                                            ExprTy **Strings,
                                            unsigned NumStrings);

  Expr *BuildObjCEncodeExpression(SourceLocation AtLoc,
                                  QualType EncodedType,
                                  SourceLocation RParenLoc);
  CXXMemberCallExpr *BuildCXXMemberCallExpr(Expr *Exp, CXXMethodDecl *Method);

  virtual ExprResult ParseObjCEncodeExpression(SourceLocation AtLoc,
                                               SourceLocation EncodeLoc,
                                               SourceLocation LParenLoc,
                                               TypeTy *Ty,
                                               SourceLocation RParenLoc);

  // ParseObjCSelectorExpression - Build selector expression for @selector
  virtual ExprResult ParseObjCSelectorExpression(Selector Sel,
                                                 SourceLocation AtLoc,
                                                 SourceLocation SelLoc,
                                                 SourceLocation LParenLoc,
                                                 SourceLocation RParenLoc);

  // ParseObjCProtocolExpression - Build protocol expression for @protocol
  virtual ExprResult ParseObjCProtocolExpression(IdentifierInfo * ProtocolName,
                                                 SourceLocation AtLoc,
                                                 SourceLocation ProtoLoc,
                                                 SourceLocation LParenLoc,
                                                 SourceLocation RParenLoc);

  //===--------------------------------------------------------------------===//
  // C++ Declarations
  //
  virtual DeclPtrTy ActOnStartLinkageSpecification(Scope *S,
                                                   SourceLocation ExternLoc,
                                                   SourceLocation LangLoc,
                                                   const char *Lang,
                                                   unsigned StrSize,
                                                   SourceLocation LBraceLoc);
  virtual DeclPtrTy ActOnFinishLinkageSpecification(Scope *S,
                                                    DeclPtrTy LinkageSpec,
                                                    SourceLocation RBraceLoc);


  //===--------------------------------------------------------------------===//
  // C++ Classes
  //
  virtual bool isCurrentClassName(const IdentifierInfo &II, Scope *S,
                                  const CXXScopeSpec *SS);

  virtual DeclPtrTy ActOnCXXMemberDeclarator(Scope *S, AccessSpecifier AS,
                                             Declarator &D,
                                 MultiTemplateParamsArg TemplateParameterLists,
                                             ExprTy *BitfieldWidth,
                                             ExprTy *Init, bool IsDefinition,
                                             bool Deleted = false);

  virtual MemInitResult ActOnMemInitializer(DeclPtrTy ConstructorD,
                                            Scope *S,
                                            const CXXScopeSpec &SS,
                                            IdentifierInfo *MemberOrBase,
                                            TypeTy *TemplateTypeTy,
                                            SourceLocation IdLoc,
                                            SourceLocation LParenLoc,
                                            ExprTy **Args, unsigned NumArgs,
                                            SourceLocation *CommaLocs,
                                            SourceLocation RParenLoc);

  MemInitResult BuildMemberInitializer(FieldDecl *Member, Expr **Args,
                                       unsigned NumArgs, SourceLocation IdLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation RParenLoc);

  MemInitResult BuildBaseInitializer(QualType BaseType,
                                     TypeSourceInfo *BaseTInfo,
                                     Expr **Args, unsigned NumArgs,
                                     SourceLocation LParenLoc,
                                     SourceLocation RParenLoc,
                                     CXXRecordDecl *ClassDecl);

  bool SetBaseOrMemberInitializers(CXXConstructorDecl *Constructor,
                              CXXBaseOrMemberInitializer **Initializers,
                              unsigned NumInitializers,
                              bool IsImplicitConstructor);

  /// MarkBaseAndMemberDestructorsReferenced - Given a destructor decl,
  /// mark all its non-trivial member and base destructor declarations
  /// as referenced.
  void MarkBaseAndMemberDestructorsReferenced(CXXDestructorDecl *Destructor);

  /// ClassesWithUnmarkedVirtualMembers - Contains record decls whose virtual
  /// members might need to be marked as referenced. This is either done when
  /// the key function definition is emitted (this is handled by by 
  /// MaybeMarkVirtualMembersReferenced), or at the end of the translation unit
  /// (done by ProcessPendingClassesWithUnmarkedVirtualMembers).
  std::map<CXXRecordDecl *, SourceLocation> ClassesWithUnmarkedVirtualMembers;

  /// MaybeMarkVirtualMembersReferenced - If the passed in method is the
  /// key function of the record decl, will mark virtual member functions as 
  /// referenced.
  void MaybeMarkVirtualMembersReferenced(SourceLocation Loc, CXXMethodDecl *MD);
  
  /// MarkVirtualMembersReferenced - Will mark all virtual members of the given
  /// CXXRecordDecl referenced.
  void MarkVirtualMembersReferenced(SourceLocation Loc, CXXRecordDecl *RD);

  /// ProcessPendingClassesWithUnmarkedVirtualMembers - Will process classes 
  /// that might need to have their virtual members marked as referenced.
  /// Returns false if no work was done.
  bool ProcessPendingClassesWithUnmarkedVirtualMembers();
  
  void AddImplicitlyDeclaredMembersToClass(CXXRecordDecl *ClassDecl);

  virtual void ActOnMemInitializers(DeclPtrTy ConstructorDecl,
                                    SourceLocation ColonLoc,
                                    MemInitTy **MemInits, unsigned NumMemInits);

  void CheckCompletedCXXClass(CXXRecordDecl *Record);
  virtual void ActOnFinishCXXMemberSpecification(Scope* S, SourceLocation RLoc,
                                                 DeclPtrTy TagDecl,
                                                 SourceLocation LBrac,
                                                 SourceLocation RBrac);

  virtual void ActOnReenterTemplateScope(Scope *S, DeclPtrTy Template);
  virtual void ActOnStartDelayedMemberDeclarations(Scope *S,
                                                   DeclPtrTy Record);
  virtual void ActOnStartDelayedCXXMethodDeclaration(Scope *S,
                                                     DeclPtrTy Method);
  virtual void ActOnDelayedCXXMethodParameter(Scope *S, DeclPtrTy Param);
  virtual void ActOnFinishDelayedCXXMethodDeclaration(Scope *S,
                                                      DeclPtrTy Method);
  virtual void ActOnFinishDelayedMemberDeclarations(Scope *S,
                                                    DeclPtrTy Record);

  virtual DeclPtrTy ActOnStaticAssertDeclaration(SourceLocation AssertLoc,
                                                 ExprArg AssertExpr,
                                                 ExprArg AssertMessageExpr);

  DeclPtrTy ActOnFriendTypeDecl(Scope *S, const DeclSpec &DS,
                                MultiTemplateParamsArg TemplateParams);
  DeclPtrTy ActOnFriendFunctionDecl(Scope *S, Declarator &D, bool IsDefinition,
                                    MultiTemplateParamsArg TemplateParams);

  QualType CheckConstructorDeclarator(Declarator &D, QualType R,
                                      FunctionDecl::StorageClass& SC);
  void CheckConstructor(CXXConstructorDecl *Constructor);
  QualType CheckDestructorDeclarator(Declarator &D,
                                     FunctionDecl::StorageClass& SC);
  bool CheckDestructor(CXXDestructorDecl *Destructor);
  void CheckConversionDeclarator(Declarator &D, QualType &R,
                                 FunctionDecl::StorageClass& SC);
  DeclPtrTy ActOnConversionDeclarator(CXXConversionDecl *Conversion);

  bool isImplicitMemberReference(const LookupResult &R, QualType &ThisType);
  
  //===--------------------------------------------------------------------===//
  // C++ Derived Classes
  //

  /// ActOnBaseSpecifier - Parsed a base specifier
  CXXBaseSpecifier *CheckBaseSpecifier(CXXRecordDecl *Class,
                                       SourceRange SpecifierRange,
                                       bool Virtual, AccessSpecifier Access,
                                       QualType BaseType,
                                       SourceLocation BaseLoc);
  
  /// SetClassDeclAttributesFromBase - Copies class decl traits 
  /// (such as whether the class has a trivial constructor, 
  /// trivial destructor etc) from the given base class.
  void SetClassDeclAttributesFromBase(CXXRecordDecl *Class,
                                      const CXXRecordDecl *BaseClass,
                                      bool BaseIsVirtual);
  
  virtual BaseResult ActOnBaseSpecifier(DeclPtrTy classdecl,
                                        SourceRange SpecifierRange,
                                        bool Virtual, AccessSpecifier Access,
                                        TypeTy *basetype, SourceLocation
                                        BaseLoc);

  bool AttachBaseSpecifiers(CXXRecordDecl *Class, CXXBaseSpecifier **Bases,
                            unsigned NumBases);
  virtual void ActOnBaseSpecifiers(DeclPtrTy ClassDecl, BaseTy **Bases,
                                   unsigned NumBases);

  bool IsDerivedFrom(QualType Derived, QualType Base);
  bool IsDerivedFrom(QualType Derived, QualType Base, CXXBasePaths &Paths);
  
  bool CheckDerivedToBaseConversion(QualType Derived, QualType Base,
                                    SourceLocation Loc, SourceRange Range,
                                    bool IgnoreAccess = false);
  bool CheckDerivedToBaseConversion(QualType Derived, QualType Base,
                                    unsigned InaccessibleBaseID,
                                    unsigned AmbigiousBaseConvID,
                                    SourceLocation Loc, SourceRange Range,
                                    DeclarationName Name);

  std::string getAmbiguousPathsDisplayString(CXXBasePaths &Paths);

  /// CheckOverridingFunctionReturnType - Checks whether the return types are
  /// covariant, according to C++ [class.virtual]p5.
  bool CheckOverridingFunctionReturnType(const CXXMethodDecl *New,
                                         const CXXMethodDecl *Old);

  /// CheckOverridingFunctionExceptionSpec - Checks whether the exception
  /// spec is a subset of base spec.
  bool CheckOverridingFunctionExceptionSpec(const CXXMethodDecl *New,
                                            const CXXMethodDecl *Old);

  /// CheckOverridingFunctionAttributes - Checks whether attributes are
  /// incompatible or prevent overriding.
  bool CheckOverridingFunctionAttributes(const CXXMethodDecl *New,
                                         const CXXMethodDecl *Old);

  bool CheckPureMethod(CXXMethodDecl *Method, SourceRange InitRange);
  //===--------------------------------------------------------------------===//
  // C++ Access Control
  //

  bool SetMemberAccessSpecifier(NamedDecl *MemberDecl,
                                NamedDecl *PrevMemberDecl,
                                AccessSpecifier LexicalAS);

  const CXXBaseSpecifier *FindInaccessibleBase(QualType Derived, QualType Base,
                                               CXXBasePaths &Paths,
                                               bool NoPrivileges = false);

  bool CheckBaseClassAccess(QualType Derived, QualType Base,
                            unsigned InaccessibleBaseID,
                            CXXBasePaths& Paths, SourceLocation AccessLoc,
                            DeclarationName Name);


  enum AbstractDiagSelID {
    AbstractNone = -1,
    AbstractReturnType,
    AbstractParamType,
    AbstractVariableType,
    AbstractFieldType
  };

  bool RequireNonAbstractType(SourceLocation Loc, QualType T,
                              const PartialDiagnostic &PD,
                              const CXXRecordDecl *CurrentRD = 0);

  bool RequireNonAbstractType(SourceLocation Loc, QualType T, unsigned DiagID,
                              AbstractDiagSelID SelID = AbstractNone,
                              const CXXRecordDecl *CurrentRD = 0);

  //===--------------------------------------------------------------------===//
  // C++ Overloaded Operators [C++ 13.5]
  //

  bool CheckOverloadedOperatorDeclaration(FunctionDecl *FnDecl);

  //===--------------------------------------------------------------------===//
  // C++ Templates [C++ 14]
  //
  void LookupTemplateName(LookupResult &R, Scope *S, const CXXScopeSpec &SS,
                          QualType ObjectType, bool EnteringContext);

  virtual TemplateNameKind isTemplateName(Scope *S,
                                          const CXXScopeSpec &SS,
                                          UnqualifiedId &Name,
                                          TypeTy *ObjectType,
                                          bool EnteringContext,
                                          TemplateTy &Template);
  bool DiagnoseTemplateParameterShadow(SourceLocation Loc, Decl *PrevDecl);
  TemplateDecl *AdjustDeclIfTemplate(DeclPtrTy &Decl);

  virtual DeclPtrTy ActOnTypeParameter(Scope *S, bool Typename, bool Ellipsis,
                                       SourceLocation EllipsisLoc,
                                       SourceLocation KeyLoc,
                                       IdentifierInfo *ParamName,
                                       SourceLocation ParamNameLoc,
                                       unsigned Depth, unsigned Position);
  virtual void ActOnTypeParameterDefault(DeclPtrTy TypeParam,
                                         SourceLocation EqualLoc,
                                         SourceLocation DefaultLoc,
                                         TypeTy *Default);

  QualType CheckNonTypeTemplateParameterType(QualType T, SourceLocation Loc);
  virtual DeclPtrTy ActOnNonTypeTemplateParameter(Scope *S, Declarator &D,
                                                  unsigned Depth,
                                                  unsigned Position);
  virtual void ActOnNonTypeTemplateParameterDefault(DeclPtrTy TemplateParam,
                                                    SourceLocation EqualLoc,
                                                    ExprArg Default);
  virtual DeclPtrTy ActOnTemplateTemplateParameter(Scope *S,
                                                   SourceLocation TmpLoc,
                                                   TemplateParamsTy *Params,
                                                   IdentifierInfo *ParamName,
                                                   SourceLocation ParamNameLoc,
                                                   unsigned Depth,
                                                   unsigned Position);
  virtual void ActOnTemplateTemplateParameterDefault(DeclPtrTy TemplateParam,
                                                     SourceLocation EqualLoc,
                                        const ParsedTemplateArgument &Default);

  virtual TemplateParamsTy *
  ActOnTemplateParameterList(unsigned Depth,
                             SourceLocation ExportLoc,
                             SourceLocation TemplateLoc,
                             SourceLocation LAngleLoc,
                             DeclPtrTy *Params, unsigned NumParams,
                             SourceLocation RAngleLoc);

  /// \brief The context in which we are checking a template parameter
  /// list.
  enum TemplateParamListContext {
    TPC_ClassTemplate,
    TPC_FunctionTemplate,
    TPC_ClassTemplateMember,
    TPC_FriendFunctionTemplate
  };

  bool CheckTemplateParameterList(TemplateParameterList *NewParams,
                                  TemplateParameterList *OldParams,
                                  TemplateParamListContext TPC);
  TemplateParameterList *
  MatchTemplateParametersToScopeSpecifier(SourceLocation DeclStartLoc,
                                          const CXXScopeSpec &SS,
                                          TemplateParameterList **ParamLists,
                                          unsigned NumParamLists,
                                          bool &IsExplicitSpecialization);

  DeclResult CheckClassTemplate(Scope *S, unsigned TagSpec, TagUseKind TUK,
                                SourceLocation KWLoc, const CXXScopeSpec &SS,
                                IdentifierInfo *Name, SourceLocation NameLoc,
                                AttributeList *Attr,
                                TemplateParameterList *TemplateParams,
                                AccessSpecifier AS);

  void translateTemplateArguments(const ASTTemplateArgsPtr &In,
                                  TemplateArgumentListInfo &Out);
    
  QualType CheckTemplateIdType(TemplateName Template,
                               SourceLocation TemplateLoc,
                               const TemplateArgumentListInfo &TemplateArgs);

  virtual TypeResult
  ActOnTemplateIdType(TemplateTy Template, SourceLocation TemplateLoc,
                      SourceLocation LAngleLoc,
                      ASTTemplateArgsPtr TemplateArgs,
                      SourceLocation RAngleLoc);

  virtual TypeResult ActOnTagTemplateIdType(TypeResult Type,
                                            TagUseKind TUK,
                                            DeclSpec::TST TagSpec,
                                            SourceLocation TagLoc);

  OwningExprResult BuildTemplateIdExpr(const CXXScopeSpec &SS,
                                       LookupResult &R,
                                       bool RequiresADL,
                               const TemplateArgumentListInfo &TemplateArgs);
  OwningExprResult BuildQualifiedTemplateIdExpr(const CXXScopeSpec &SS,
                                                DeclarationName Name,
                                                SourceLocation NameLoc,
                               const TemplateArgumentListInfo &TemplateArgs);

  virtual TemplateTy ActOnDependentTemplateName(SourceLocation TemplateKWLoc,
                                                const CXXScopeSpec &SS,
                                                UnqualifiedId &Name,
                                                TypeTy *ObjectType,
                                                bool EnteringContext);

  bool CheckClassTemplatePartialSpecializationArgs(
                                        TemplateParameterList *TemplateParams,
                              const TemplateArgumentListBuilder &TemplateArgs,
                                        bool &MirrorsPrimaryTemplate);

  virtual DeclResult
  ActOnClassTemplateSpecialization(Scope *S, unsigned TagSpec, TagUseKind TUK,
                                   SourceLocation KWLoc,
                                   const CXXScopeSpec &SS,
                                   TemplateTy Template,
                                   SourceLocation TemplateNameLoc,
                                   SourceLocation LAngleLoc,
                                   ASTTemplateArgsPtr TemplateArgs,
                                   SourceLocation RAngleLoc,
                                   AttributeList *Attr,
                                 MultiTemplateParamsArg TemplateParameterLists);

  virtual DeclPtrTy ActOnTemplateDeclarator(Scope *S,
                                  MultiTemplateParamsArg TemplateParameterLists,
                                            Declarator &D);

  virtual DeclPtrTy ActOnStartOfFunctionTemplateDef(Scope *FnBodyScope,
                                                    MultiTemplateParamsArg TemplateParameterLists,
                                                    Declarator &D);

  bool
  CheckSpecializationInstantiationRedecl(SourceLocation NewLoc,
                                         TemplateSpecializationKind NewTSK,
                                         NamedDecl *PrevDecl,
                                         TemplateSpecializationKind PrevTSK,
                                         SourceLocation PrevPointOfInstantiation,
                                         bool &SuppressNew);
    
  bool CheckFunctionTemplateSpecialization(FunctionDecl *FD,
                        const TemplateArgumentListInfo *ExplicitTemplateArgs,
                                           LookupResult &Previous);
  bool CheckMemberSpecialization(NamedDecl *Member, LookupResult &Previous);
    
  virtual DeclResult
  ActOnExplicitInstantiation(Scope *S,
                             SourceLocation ExternLoc,
                             SourceLocation TemplateLoc,
                             unsigned TagSpec,
                             SourceLocation KWLoc,
                             const CXXScopeSpec &SS,
                             TemplateTy Template,
                             SourceLocation TemplateNameLoc,
                             SourceLocation LAngleLoc,
                             ASTTemplateArgsPtr TemplateArgs,
                             SourceLocation RAngleLoc,
                             AttributeList *Attr);

  virtual DeclResult
  ActOnExplicitInstantiation(Scope *S,
                             SourceLocation ExternLoc,
                             SourceLocation TemplateLoc,
                             unsigned TagSpec,
                             SourceLocation KWLoc,
                             const CXXScopeSpec &SS,
                             IdentifierInfo *Name,
                             SourceLocation NameLoc,
                             AttributeList *Attr);

  virtual DeclResult ActOnExplicitInstantiation(Scope *S,
                                                SourceLocation ExternLoc,
                                                SourceLocation TemplateLoc,
                                                Declarator &D);
    
  TemplateArgumentLoc 
  SubstDefaultTemplateArgumentIfAvailable(TemplateDecl *Template,
                                          SourceLocation TemplateLoc,
                                          SourceLocation RAngleLoc,
                                          Decl *Param,
                                      TemplateArgumentListBuilder &Converted);

  bool CheckTemplateArgument(NamedDecl *Param,
                             const TemplateArgumentLoc &Arg,
                             TemplateDecl *Template,
                             SourceLocation TemplateLoc,
                             SourceLocation RAngleLoc,
                             TemplateArgumentListBuilder &Converted);
  
  bool CheckTemplateArgumentList(TemplateDecl *Template,
                                 SourceLocation TemplateLoc,
                                 const TemplateArgumentListInfo &TemplateArgs,
                                 bool PartialTemplateArgs,
                                 TemplateArgumentListBuilder &Converted);

  bool CheckTemplateTypeArgument(TemplateTypeParmDecl *Param,
                                 const TemplateArgumentLoc &Arg,
                                 TemplateArgumentListBuilder &Converted);

  bool CheckTemplateArgument(TemplateTypeParmDecl *Param,
                             TypeSourceInfo *Arg);
  bool CheckTemplateArgumentAddressOfObjectOrFunction(Expr *Arg,
                                                      NamedDecl *&Entity);
  bool CheckTemplateArgumentPointerToMember(Expr *Arg, 
                                            TemplateArgument &Converted);
  bool CheckTemplateArgument(NonTypeTemplateParmDecl *Param,
                             QualType InstantiatedParamType, Expr *&Arg,
                             TemplateArgument &Converted);
  bool CheckTemplateArgument(TemplateTemplateParmDecl *Param, 
                             const TemplateArgumentLoc &Arg);
  
  /// \brief Enumeration describing how template parameter lists are compared
  /// for equality.
  enum TemplateParameterListEqualKind {
    /// \brief We are matching the template parameter lists of two templates
    /// that might be redeclarations.
    ///
    /// \code
    /// template<typename T> struct X;
    /// template<typename T> struct X;
    /// \endcode
    TPL_TemplateMatch,
    
    /// \brief We are matching the template parameter lists of two template
    /// template parameters as part of matching the template parameter lists
    /// of two templates that might be redeclarations.
    ///
    /// \code
    /// template<template<int I> class TT> struct X;
    /// template<template<int Value> class Other> struct X;
    /// \endcode
    TPL_TemplateTemplateParmMatch,
    
    /// \brief We are matching the template parameter lists of a template
    /// template argument against the template parameter lists of a template
    /// template parameter.
    ///
    /// \code
    /// template<template<int Value> class Metafun> struct X;
    /// template<int Value> struct integer_c;
    /// X<integer_c> xic;
    /// \endcode
    TPL_TemplateTemplateArgumentMatch
  };
  
  bool TemplateParameterListsAreEqual(TemplateParameterList *New,
                                      TemplateParameterList *Old,
                                      bool Complain,
                                      TemplateParameterListEqualKind Kind,
                                      SourceLocation TemplateArgLoc
                                        = SourceLocation());

  bool CheckTemplateDeclScope(Scope *S, TemplateParameterList *TemplateParams);

  /// \brief Called when the parser has parsed a C++ typename
  /// specifier, e.g., "typename T::type".
  ///
  /// \param TypenameLoc the location of the 'typename' keyword
  /// \param SS the nested-name-specifier following the typename (e.g., 'T::').
  /// \param II the identifier we're retrieving (e.g., 'type' in the example).
  /// \param IdLoc the location of the identifier.
  virtual TypeResult
  ActOnTypenameType(SourceLocation TypenameLoc, const CXXScopeSpec &SS,
                    const IdentifierInfo &II, SourceLocation IdLoc);

  /// \brief Called when the parser has parsed a C++ typename
  /// specifier that ends in a template-id, e.g.,
  /// "typename MetaFun::template apply<T1, T2>".
  ///
  /// \param TypenameLoc the location of the 'typename' keyword
  /// \param SS the nested-name-specifier following the typename (e.g., 'T::').
  /// \param TemplateLoc the location of the 'template' keyword, if any.
  /// \param Ty the type that the typename specifier refers to.
  virtual TypeResult
  ActOnTypenameType(SourceLocation TypenameLoc, const CXXScopeSpec &SS,
                    SourceLocation TemplateLoc, TypeTy *Ty);

  QualType CheckTypenameType(NestedNameSpecifier *NNS,
                             const IdentifierInfo &II,
                             SourceRange Range);

  QualType RebuildTypeInCurrentInstantiation(QualType T, SourceLocation Loc,
                                             DeclarationName Name);

  std::string
  getTemplateArgumentBindingsText(const TemplateParameterList *Params,
                                  const TemplateArgumentList &Args);

  std::string
  getTemplateArgumentBindingsText(const TemplateParameterList *Params,
                                  const TemplateArgument *Args,
                                  unsigned NumArgs);
  
  /// \brief Describes the result of template argument deduction.
  ///
  /// The TemplateDeductionResult enumeration describes the result of
  /// template argument deduction, as returned from
  /// DeduceTemplateArguments(). The separate TemplateDeductionInfo
  /// structure provides additional information about the results of
  /// template argument deduction, e.g., the deduced template argument
  /// list (if successful) or the specific template parameters or
  /// deduced arguments that were involved in the failure.
  enum TemplateDeductionResult {
    /// \brief Template argument deduction was successful.
    TDK_Success = 0,
    /// \brief Template argument deduction exceeded the maximum template
    /// instantiation depth (which has already been diagnosed).
    TDK_InstantiationDepth,
    /// \brief Template argument deduction did not deduce a value
    /// for every template parameter.
    TDK_Incomplete,
    /// \brief Template argument deduction produced inconsistent
    /// deduced values for the given template parameter.
    TDK_Inconsistent,
    /// \brief Template argument deduction failed due to inconsistent
    /// cv-qualifiers on a template parameter type that would
    /// otherwise be deduced, e.g., we tried to deduce T in "const T"
    /// but were given a non-const "X".
    TDK_InconsistentQuals,
    /// \brief Substitution of the deduced template argument values
    /// resulted in an error.
    TDK_SubstitutionFailure,
    /// \brief Substitution of the deduced template argument values
    /// into a non-deduced context produced a type or value that
    /// produces a type that does not match the original template
    /// arguments provided.
    TDK_NonDeducedMismatch,
    /// \brief When performing template argument deduction for a function
    /// template, there were too many call arguments.
    TDK_TooManyArguments,
    /// \brief When performing template argument deduction for a function
    /// template, there were too few call arguments.
    TDK_TooFewArguments,
    /// \brief The explicitly-specified template arguments were not valid
    /// template arguments for the given template.
    TDK_InvalidExplicitArguments,
    /// \brief The arguments included an overloaded function name that could
    /// not be resolved to a suitable function.
    TDK_FailedOverloadResolution
  };

  /// \brief Provides information about an attempted template argument
  /// deduction, whose success or failure was described by a
  /// TemplateDeductionResult value.
  class TemplateDeductionInfo {
    /// \brief The context in which the template arguments are stored.
    ASTContext &Context;

    /// \brief The deduced template argument list.
    ///
    TemplateArgumentList *Deduced;

    // do not implement these
    TemplateDeductionInfo(const TemplateDeductionInfo&);
    TemplateDeductionInfo &operator=(const TemplateDeductionInfo&);

  public:
    TemplateDeductionInfo(ASTContext &Context) : Context(Context), Deduced(0) { }

    ~TemplateDeductionInfo() {
      // FIXME: if (Deduced) Deduced->Destroy(Context);
    }

    /// \brief Take ownership of the deduced template argument list.
    TemplateArgumentList *take() {
      TemplateArgumentList *Result = Deduced;
      Deduced = 0;
      return Result;
    }

    /// \brief Provide a new template argument list that contains the
    /// results of template argument deduction.
    void reset(TemplateArgumentList *NewDeduced) {
      // FIXME: if (Deduced) Deduced->Destroy(Context);
      Deduced = NewDeduced;
    }

    /// \brief The template parameter to which a template argument
    /// deduction failure refers.
    ///
    /// Depending on the result of template argument deduction, this
    /// template parameter may have different meanings:
    ///
    ///   TDK_Incomplete: this is the first template parameter whose
    ///   corresponding template argument was not deduced.
    ///
    ///   TDK_Inconsistent: this is the template parameter for which
    ///   two different template argument values were deduced.
    TemplateParameter Param;

    /// \brief The first template argument to which the template
    /// argument deduction failure refers.
    ///
    /// Depending on the result of the template argument deduction,
    /// this template argument may have different meanings:
    ///
    ///   TDK_Inconsistent: this argument is the first value deduced
    ///   for the corresponding template parameter.
    ///
    ///   TDK_SubstitutionFailure: this argument is the template
    ///   argument we were instantiating when we encountered an error.
    ///
    ///   TDK_NonDeducedMismatch: this is the template argument
    ///   provided in the source code.
    TemplateArgument FirstArg;

    /// \brief The second template argument to which the template
    /// argument deduction failure refers.
    ///
    /// FIXME: Finish documenting this.
    TemplateArgument SecondArg;
  };

  TemplateDeductionResult
  DeduceTemplateArguments(ClassTemplatePartialSpecializationDecl *Partial,
                          const TemplateArgumentList &TemplateArgs,
                          TemplateDeductionInfo &Info);

  TemplateDeductionResult
  SubstituteExplicitTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                        const TemplateArgumentListInfo &ExplicitTemplateArgs,
                            llvm::SmallVectorImpl<TemplateArgument> &Deduced,
                                 llvm::SmallVectorImpl<QualType> &ParamTypes,
                                      QualType *FunctionType,
                                      TemplateDeductionInfo &Info);

  TemplateDeductionResult
  FinishTemplateArgumentDeduction(FunctionTemplateDecl *FunctionTemplate,
                             llvm::SmallVectorImpl<TemplateArgument> &Deduced,
                                  FunctionDecl *&Specialization,
                                  TemplateDeductionInfo &Info);

  TemplateDeductionResult
  DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                          const TemplateArgumentListInfo *ExplicitTemplateArgs,
                          Expr **Args, unsigned NumArgs,
                          FunctionDecl *&Specialization,
                          TemplateDeductionInfo &Info);

  TemplateDeductionResult
  DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                          const TemplateArgumentListInfo *ExplicitTemplateArgs,
                          QualType ArgFunctionType,
                          FunctionDecl *&Specialization,
                          TemplateDeductionInfo &Info);

  TemplateDeductionResult
  DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                          QualType ToType,
                          CXXConversionDecl *&Specialization,
                          TemplateDeductionInfo &Info);

  TemplateDeductionResult
  DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                          const TemplateArgumentListInfo *ExplicitTemplateArgs,
                          FunctionDecl *&Specialization,
                          TemplateDeductionInfo &Info);

  FunctionTemplateDecl *getMoreSpecializedTemplate(FunctionTemplateDecl *FT1,
                                                   FunctionTemplateDecl *FT2,
                                           TemplatePartialOrderingContext TPOC);
  FunctionDecl *getMostSpecialized(FunctionDecl **Specializations,
                                   unsigned NumSpecializations,
                                   TemplatePartialOrderingContext TPOC,
                                   SourceLocation Loc,
                                   const PartialDiagnostic &NoneDiag,
                                   const PartialDiagnostic &AmbigDiag,
                                   const PartialDiagnostic &CandidateDiag,
                                   unsigned *Index = 0);
                                   
  ClassTemplatePartialSpecializationDecl *
  getMoreSpecializedPartialSpecialization(
                                  ClassTemplatePartialSpecializationDecl *PS1,
                                  ClassTemplatePartialSpecializationDecl *PS2);
  
  void MarkUsedTemplateParameters(const TemplateArgumentList &TemplateArgs,
                                  bool OnlyDeduced,
                                  unsigned Depth,
                                  llvm::SmallVectorImpl<bool> &Used);
  void MarkDeducedTemplateParameters(FunctionTemplateDecl *FunctionTemplate,
                                     llvm::SmallVectorImpl<bool> &Deduced);
  
  //===--------------------------------------------------------------------===//
  // C++ Template Instantiation
  //

  MultiLevelTemplateArgumentList getTemplateInstantiationArgs(NamedDecl *D,
                                     const TemplateArgumentList *Innermost = 0);

  /// \brief A template instantiation that is currently in progress.
  struct ActiveTemplateInstantiation {
    /// \brief The kind of template instantiation we are performing
    enum InstantiationKind {
      /// We are instantiating a template declaration. The entity is
      /// the declaration we're instantiating (e.g., a CXXRecordDecl).
      TemplateInstantiation,

      /// We are instantiating a default argument for a template
      /// parameter. The Entity is the template, and
      /// TemplateArgs/NumTemplateArguments provides the template
      /// arguments as specified.
      /// FIXME: Use a TemplateArgumentList
      DefaultTemplateArgumentInstantiation,

      /// We are instantiating a default argument for a function.
      /// The Entity is the ParmVarDecl, and TemplateArgs/NumTemplateArgs
      /// provides the template arguments as specified.
      DefaultFunctionArgumentInstantiation,

      /// We are substituting explicit template arguments provided for
      /// a function template. The entity is a FunctionTemplateDecl.
      ExplicitTemplateArgumentSubstitution,

      /// We are substituting template argument determined as part of
      /// template argument deduction for either a class template
      /// partial specialization or a function template. The
      /// Entity is either a ClassTemplatePartialSpecializationDecl or
      /// a FunctionTemplateDecl.
      DeducedTemplateArgumentSubstitution,
      
      /// We are substituting prior template arguments into a new
      /// template parameter. The template parameter itself is either a
      /// NonTypeTemplateParmDecl or a TemplateTemplateParmDecl.
      PriorTemplateArgumentSubstitution,
      
      /// We are checking the validity of a default template argument that
      /// has been used when naming a template-id.
      DefaultTemplateArgumentChecking
    } Kind;

    /// \brief The point of instantiation within the source code.
    SourceLocation PointOfInstantiation;

    /// \brief The template in which we are performing the instantiation,
    /// for substitutions of prior template arguments.
    TemplateDecl *Template;
    
    /// \brief The entity that is being instantiated.
    uintptr_t Entity;

    /// \brief The list of template arguments we are substituting, if they
    /// are not part of the entity.
    const TemplateArgument *TemplateArgs;

    /// \brief The number of template arguments in TemplateArgs.
    unsigned NumTemplateArgs;

    /// \brief The source range that covers the construct that cause
    /// the instantiation, e.g., the template-id that causes a class
    /// template instantiation.
    SourceRange InstantiationRange;

    ActiveTemplateInstantiation()
      : Kind(TemplateInstantiation), Template(0), Entity(0), TemplateArgs(0), 
        NumTemplateArgs(0) {}

    /// \brief Determines whether this template is an actual instantiation
    /// that should be counted toward the maximum instantiation depth.
    bool isInstantiationRecord() const;
    
    friend bool operator==(const ActiveTemplateInstantiation &X,
                           const ActiveTemplateInstantiation &Y) {
      if (X.Kind != Y.Kind)
        return false;

      if (X.Entity != Y.Entity)
        return false;

      switch (X.Kind) {
      case TemplateInstantiation:
        return true;

      case PriorTemplateArgumentSubstitution:
      case DefaultTemplateArgumentChecking:
        if (X.Template != Y.Template)
          return false;
          
        // Fall through
          
      case DefaultTemplateArgumentInstantiation:
      case ExplicitTemplateArgumentSubstitution:
      case DeducedTemplateArgumentSubstitution:
      case DefaultFunctionArgumentInstantiation:
        return X.TemplateArgs == Y.TemplateArgs;

      }

      return true;
    }

    friend bool operator!=(const ActiveTemplateInstantiation &X,
                           const ActiveTemplateInstantiation &Y) {
      return !(X == Y);
    }
  };

  /// \brief List of active template instantiations.
  ///
  /// This vector is treated as a stack. As one template instantiation
  /// requires another template instantiation, additional
  /// instantiations are pushed onto the stack up to a
  /// user-configurable limit LangOptions::InstantiationDepth.
  llvm::SmallVector<ActiveTemplateInstantiation, 16>
    ActiveTemplateInstantiations;

  /// \brief The number of ActiveTemplateInstantiation entries in
  /// \c ActiveTemplateInstantiations that are not actual instantiations and,
  /// therefore, should not be counted as part of the instantiation depth.
  unsigned NonInstantiationEntries;
  
  /// \brief The last template from which a template instantiation
  /// error or warning was produced.
  ///
  /// This value is used to suppress printing of redundant template
  /// instantiation backtraces when there are multiple errors in the
  /// same instantiation. FIXME: Does this belong in Sema? It's tough
  /// to implement it anywhere else.
  ActiveTemplateInstantiation LastTemplateInstantiationErrorContext;

  /// \brief A stack object to be created when performing template
  /// instantiation.
  ///
  /// Construction of an object of type \c InstantiatingTemplate
  /// pushes the current instantiation onto the stack of active
  /// instantiations. If the size of this stack exceeds the maximum
  /// number of recursive template instantiations, construction
  /// produces an error and evaluates true.
  ///
  /// Destruction of this object will pop the named instantiation off
  /// the stack.
  struct InstantiatingTemplate {
    /// \brief Note that we are instantiating a class template,
    /// function template, or a member thereof.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          Decl *Entity,
                          SourceRange InstantiationRange = SourceRange());

    /// \brief Note that we are instantiating a default argument in a
    /// template-id.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          TemplateDecl *Template,
                          const TemplateArgument *TemplateArgs,
                          unsigned NumTemplateArgs,
                          SourceRange InstantiationRange = SourceRange());

    /// \brief Note that we are instantiating a default argument in a
    /// template-id.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          FunctionTemplateDecl *FunctionTemplate,
                          const TemplateArgument *TemplateArgs,
                          unsigned NumTemplateArgs,
                          ActiveTemplateInstantiation::InstantiationKind Kind,
                          SourceRange InstantiationRange = SourceRange());

    /// \brief Note that we are instantiating as part of template
    /// argument deduction for a class template partial
    /// specialization.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          ClassTemplatePartialSpecializationDecl *PartialSpec,
                          const TemplateArgument *TemplateArgs,
                          unsigned NumTemplateArgs,
                          SourceRange InstantiationRange = SourceRange());

    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          ParmVarDecl *Param,
                          const TemplateArgument *TemplateArgs,
                          unsigned NumTemplateArgs,
                          SourceRange InstantiationRange = SourceRange());

    /// \brief Note that we are substituting prior template arguments into a
    /// non-type or template template parameter.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          TemplateDecl *Template,
                          NonTypeTemplateParmDecl *Param,
                          const TemplateArgument *TemplateArgs,
                          unsigned NumTemplateArgs,
                          SourceRange InstantiationRange);

    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          TemplateDecl *Template,
                          TemplateTemplateParmDecl *Param,
                          const TemplateArgument *TemplateArgs,
                          unsigned NumTemplateArgs,
                          SourceRange InstantiationRange);
    
    /// \brief Note that we are checking the default template argument
    /// against the template parameter for a given template-id.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          TemplateDecl *Template,
                          NamedDecl *Param,
                          const TemplateArgument *TemplateArgs,
                          unsigned NumTemplateArgs,
                          SourceRange InstantiationRange);
    
    
    /// \brief Note that we have finished instantiating this template.
    void Clear();

    ~InstantiatingTemplate() { Clear(); }

    /// \brief Determines whether we have exceeded the maximum
    /// recursive template instantiations.
    operator bool() const { return Invalid; }

  private:
    Sema &SemaRef;
    bool Invalid;

    bool CheckInstantiationDepth(SourceLocation PointOfInstantiation,
                                 SourceRange InstantiationRange);

    InstantiatingTemplate(const InstantiatingTemplate&); // not implemented

    InstantiatingTemplate&
    operator=(const InstantiatingTemplate&); // not implemented
  };

  void PrintInstantiationStack();

  /// \brief Determines whether we are currently in a context where
  /// template argument substitution failures are not considered
  /// errors.
  ///
  /// When this routine returns true, the emission of most diagnostics
  /// will be suppressed and there will be no local error recovery.
  bool isSFINAEContext() const;

  /// \brief RAII class used to determine whether SFINAE has
  /// trapped any errors that occur during template argument
  /// deduction.
  class SFINAETrap {
    Sema &SemaRef;
    unsigned PrevSFINAEErrors;
  public:
    explicit SFINAETrap(Sema &SemaRef)
      : SemaRef(SemaRef), PrevSFINAEErrors(SemaRef.NumSFINAEErrors) { }

    ~SFINAETrap() { SemaRef.NumSFINAEErrors = PrevSFINAEErrors; }

    /// \brief Determine whether any SFINAE errors have been trapped.
    bool hasErrorOccurred() const {
      return SemaRef.NumSFINAEErrors > PrevSFINAEErrors;
    }
  };

  /// \brief A stack-allocated class that identifies which local
  /// variable declaration instantiations are present in this scope.
  ///
  /// A new instance of this class type will be created whenever we
  /// instantiate a new function declaration, which will have its own
  /// set of parameter declarations.
  class LocalInstantiationScope {
    /// \brief Reference to the semantic analysis that is performing
    /// this template instantiation.
    Sema &SemaRef;

    /// \brief A mapping from local declarations that occur
    /// within a template to their instantiations.
    ///
    /// This mapping is used during instantiation to keep track of,
    /// e.g., function parameter and variable declarations. For example,
    /// given:
    ///
    /// \code
    ///   template<typename T> T add(T x, T y) { return x + y; }
    /// \endcode
    ///
    /// when we instantiate add<int>, we will introduce a mapping from
    /// the ParmVarDecl for 'x' that occurs in the template to the
    /// instantiated ParmVarDecl for 'x'.
    llvm::DenseMap<const Decl *, Decl *> LocalDecls;

    /// \brief The outer scope, in which contains local variable
    /// definitions from some other instantiation (that may not be
    /// relevant to this particular scope).
    LocalInstantiationScope *Outer;

    // This class is non-copyable
    LocalInstantiationScope(const LocalInstantiationScope &);
    LocalInstantiationScope &operator=(const LocalInstantiationScope &);

  public:
    LocalInstantiationScope(Sema &SemaRef, bool CombineWithOuterScope = false)
      : SemaRef(SemaRef), Outer(SemaRef.CurrentInstantiationScope) {
      if (!CombineWithOuterScope)
        SemaRef.CurrentInstantiationScope = this;
      else
        assert(SemaRef.CurrentInstantiationScope && 
               "No outer instantiation scope?");
    }

    ~LocalInstantiationScope() {
      SemaRef.CurrentInstantiationScope = Outer;
    }

    Decl *getInstantiationOf(const Decl *D) {
      Decl *Result = LocalDecls[D];
      assert(Result && "declaration was not instantiated in this scope!");
      return Result;
    }

    VarDecl *getInstantiationOf(const VarDecl *Var) {
      return cast<VarDecl>(getInstantiationOf(cast<Decl>(Var)));
    }

    ParmVarDecl *getInstantiationOf(const ParmVarDecl *Var) {
      return cast<ParmVarDecl>(getInstantiationOf(cast<Decl>(Var)));
    }

    NonTypeTemplateParmDecl *getInstantiationOf(
                                          const NonTypeTemplateParmDecl *Var) {
      return cast<NonTypeTemplateParmDecl>(getInstantiationOf(cast<Decl>(Var)));
    }
    
    void InstantiatedLocal(const Decl *D, Decl *Inst) {
      Decl *&Stored = LocalDecls[D];
      assert(!Stored && "Already instantiated this local");
      Stored = Inst;
    }
  };

  /// \brief The current instantiation scope used to store local
  /// variables.
  LocalInstantiationScope *CurrentInstantiationScope;

  /// \brief An entity for which implicit template instantiation is required.
  ///
  /// The source location associated with the declaration is the first place in
  /// the source code where the declaration was "used". It is not necessarily
  /// the point of instantiation (which will be either before or after the
  /// namespace-scope declaration that triggered this implicit instantiation),
  /// However, it is the location that diagnostics should generally refer to,
  /// because users will need to know what code triggered the instantiation.
  typedef std::pair<ValueDecl *, SourceLocation> PendingImplicitInstantiation;

  /// \brief The queue of implicit template instantiations that are required
  /// but have not yet been performed.
  std::deque<PendingImplicitInstantiation> PendingImplicitInstantiations;

  void PerformPendingImplicitInstantiations();

  TypeSourceInfo *SubstType(TypeSourceInfo *T,
                            const MultiLevelTemplateArgumentList &TemplateArgs,
                            SourceLocation Loc, DeclarationName Entity);

  QualType SubstType(QualType T,
                     const MultiLevelTemplateArgumentList &TemplateArgs,
                     SourceLocation Loc, DeclarationName Entity);

  OwningExprResult SubstExpr(Expr *E,
                            const MultiLevelTemplateArgumentList &TemplateArgs);

  OwningStmtResult SubstStmt(Stmt *S,
                            const MultiLevelTemplateArgumentList &TemplateArgs);

  Decl *SubstDecl(Decl *D, DeclContext *Owner,
                  const MultiLevelTemplateArgumentList &TemplateArgs);

  bool
  SubstBaseSpecifiers(CXXRecordDecl *Instantiation,
                      CXXRecordDecl *Pattern,
                      const MultiLevelTemplateArgumentList &TemplateArgs);

  bool
  InstantiateClass(SourceLocation PointOfInstantiation,
                   CXXRecordDecl *Instantiation, CXXRecordDecl *Pattern,
                   const MultiLevelTemplateArgumentList &TemplateArgs,
                   TemplateSpecializationKind TSK,
                   bool Complain = true);

  bool
  InstantiateClassTemplateSpecialization(SourceLocation PointOfInstantiation,
                           ClassTemplateSpecializationDecl *ClassTemplateSpec,
                           TemplateSpecializationKind TSK,
                           bool Complain = true);

  void InstantiateClassMembers(SourceLocation PointOfInstantiation,
                               CXXRecordDecl *Instantiation,
                            const MultiLevelTemplateArgumentList &TemplateArgs,
                               TemplateSpecializationKind TSK);

  void InstantiateClassTemplateSpecializationMembers(
                                          SourceLocation PointOfInstantiation,
                           ClassTemplateSpecializationDecl *ClassTemplateSpec,
                                                TemplateSpecializationKind TSK);

  NestedNameSpecifier *
  SubstNestedNameSpecifier(NestedNameSpecifier *NNS,
                           SourceRange Range,
                           const MultiLevelTemplateArgumentList &TemplateArgs);

  TemplateName
  SubstTemplateName(TemplateName Name, SourceLocation Loc,
                    const MultiLevelTemplateArgumentList &TemplateArgs);
  bool Subst(const TemplateArgumentLoc &Arg, TemplateArgumentLoc &Result,
             const MultiLevelTemplateArgumentList &TemplateArgs);

  void InstantiateFunctionDefinition(SourceLocation PointOfInstantiation,
                                     FunctionDecl *Function,
                                     bool Recursive = false,
                                     bool DefinitionRequired = false);
  void InstantiateStaticDataMemberDefinition(
                                     SourceLocation PointOfInstantiation,
                                     VarDecl *Var,
                                     bool Recursive = false,
                                     bool DefinitionRequired = false);

  void InstantiateMemInitializers(CXXConstructorDecl *New,
                                  const CXXConstructorDecl *Tmpl,
                            const MultiLevelTemplateArgumentList &TemplateArgs);

  NamedDecl *FindInstantiatedDecl(NamedDecl *D,
                          const MultiLevelTemplateArgumentList &TemplateArgs);
  DeclContext *FindInstantiatedContext(DeclContext *DC,
                          const MultiLevelTemplateArgumentList &TemplateArgs);

  // Objective-C declarations.
  virtual DeclPtrTy ActOnStartClassInterface(SourceLocation AtInterfaceLoc,
                                             IdentifierInfo *ClassName,
                                             SourceLocation ClassLoc,
                                             IdentifierInfo *SuperName,
                                             SourceLocation SuperLoc,
                                             const DeclPtrTy *ProtoRefs,
                                             unsigned NumProtoRefs,
                                             SourceLocation EndProtoLoc,
                                             AttributeList *AttrList);

  virtual DeclPtrTy ActOnCompatiblityAlias(
                    SourceLocation AtCompatibilityAliasLoc,
                    IdentifierInfo *AliasName,  SourceLocation AliasLocation,
                    IdentifierInfo *ClassName, SourceLocation ClassLocation);

  void CheckForwardProtocolDeclarationForCircularDependency(
    IdentifierInfo *PName,
    SourceLocation &PLoc, SourceLocation PrevLoc,
    const ObjCList<ObjCProtocolDecl> &PList);

  virtual DeclPtrTy ActOnStartProtocolInterface(
                    SourceLocation AtProtoInterfaceLoc,
                    IdentifierInfo *ProtocolName, SourceLocation ProtocolLoc,
                    const DeclPtrTy *ProtoRefNames, unsigned NumProtoRefs,
                    SourceLocation EndProtoLoc,
                    AttributeList *AttrList);

  virtual DeclPtrTy ActOnStartCategoryInterface(SourceLocation AtInterfaceLoc,
                                                IdentifierInfo *ClassName,
                                                SourceLocation ClassLoc,
                                                IdentifierInfo *CategoryName,
                                                SourceLocation CategoryLoc,
                                                const DeclPtrTy *ProtoRefs,
                                                unsigned NumProtoRefs,
                                                SourceLocation EndProtoLoc);

  virtual DeclPtrTy ActOnStartClassImplementation(
                    SourceLocation AtClassImplLoc,
                    IdentifierInfo *ClassName, SourceLocation ClassLoc,
                    IdentifierInfo *SuperClassname,
                    SourceLocation SuperClassLoc);

  virtual DeclPtrTy ActOnStartCategoryImplementation(
                                                  SourceLocation AtCatImplLoc,
                                                  IdentifierInfo *ClassName,
                                                  SourceLocation ClassLoc,
                                                  IdentifierInfo *CatName,
                                                  SourceLocation CatLoc);

  virtual DeclPtrTy ActOnForwardClassDeclaration(SourceLocation Loc,
                                                 IdentifierInfo **IdentList,
                                                 SourceLocation *IdentLocs,
                                                 unsigned NumElts);

  virtual DeclPtrTy ActOnForwardProtocolDeclaration(SourceLocation AtProtocolLoc,
                                            const IdentifierLocPair *IdentList,
                                                  unsigned NumElts,
                                                  AttributeList *attrList);

  virtual void FindProtocolDeclaration(bool WarnOnDeclarations,
                                       const IdentifierLocPair *ProtocolId,
                                       unsigned NumProtocols,
                                   llvm::SmallVectorImpl<DeclPtrTy> &Protocols);

  /// Ensure attributes are consistent with type.
  /// \param [in, out] Attributes The attributes to check; they will
  /// be modified to be consistent with \arg PropertyTy.
  void CheckObjCPropertyAttributes(QualType PropertyTy,
                                   SourceLocation Loc,
                                   unsigned &Attributes);
  void ProcessPropertyDecl(ObjCPropertyDecl *property, ObjCContainerDecl *DC);
  void DiagnosePropertyMismatch(ObjCPropertyDecl *Property,
                                ObjCPropertyDecl *SuperProperty,
                                const IdentifierInfo *Name);
  void ComparePropertiesInBaseAndSuper(ObjCInterfaceDecl *IDecl);

  void CompareMethodParamsInBaseAndSuper(Decl *IDecl,
                                         ObjCMethodDecl *MethodDecl,
                                         bool IsInstance);

  void MergeProtocolPropertiesIntoClass(Decl *CDecl,
                                        DeclPtrTy MergeProtocols);

  void DiagnoseClassExtensionDupMethods(ObjCCategoryDecl *CAT,
                                        ObjCInterfaceDecl *ID);

  void MergeOneProtocolPropertiesIntoClass(Decl *CDecl,
                                           ObjCProtocolDecl *PDecl);

  virtual void ActOnAtEnd(SourceLocation AtEndLoc, DeclPtrTy classDecl,
                      DeclPtrTy *allMethods = 0, unsigned allNum = 0,
                      DeclPtrTy *allProperties = 0, unsigned pNum = 0,
                      DeclGroupPtrTy *allTUVars = 0, unsigned tuvNum = 0);

  virtual DeclPtrTy ActOnProperty(Scope *S, SourceLocation AtLoc,
                                  FieldDeclarator &FD, ObjCDeclSpec &ODS,
                                  Selector GetterSel, Selector SetterSel,
                                  DeclPtrTy ClassCategory,
                                  bool *OverridingProperty,
                                  tok::ObjCKeywordKind MethodImplKind);

  virtual DeclPtrTy ActOnPropertyImplDecl(SourceLocation AtLoc,
                                          SourceLocation PropertyLoc,
                                          bool ImplKind,DeclPtrTy ClassImplDecl,
                                          IdentifierInfo *PropertyId,
                                          IdentifierInfo *PropertyIvar);

  virtual DeclPtrTy ActOnMethodDeclaration(
    SourceLocation BeginLoc, // location of the + or -.
    SourceLocation EndLoc,   // location of the ; or {.
    tok::TokenKind MethodType,
    DeclPtrTy ClassDecl, ObjCDeclSpec &ReturnQT, TypeTy *ReturnType,
    Selector Sel,
    // optional arguments. The number of types/arguments is obtained
    // from the Sel.getNumArgs().
    ObjCArgInfo *ArgInfo,
    llvm::SmallVectorImpl<Declarator> &Cdecls,
    AttributeList *AttrList, tok::ObjCKeywordKind MethodImplKind,
    bool isVariadic = false);

  // Helper method for ActOnClassMethod/ActOnInstanceMethod.
  // Will search "local" class/category implementations for a method decl.
  // Will also search in class's root looking for instance method.
  // Returns 0 if no method is found.
  ObjCMethodDecl *LookupPrivateClassMethod(Selector Sel,
                                           ObjCInterfaceDecl *CDecl);
  ObjCMethodDecl *LookupPrivateInstanceMethod(Selector Sel,
                                              ObjCInterfaceDecl *ClassDecl);

  virtual OwningExprResult ActOnClassPropertyRefExpr(
    IdentifierInfo &receiverName,
    IdentifierInfo &propertyName,
    SourceLocation &receiverNameLoc,
    SourceLocation &propertyNameLoc);

  // ActOnClassMessage - used for both unary and keyword messages.
  // ArgExprs is optional - if it is present, the number of expressions
  // is obtained from NumArgs.
  virtual ExprResult ActOnClassMessage(
    Scope *S,
    IdentifierInfo *receivingClassName, Selector Sel, SourceLocation lbrac,
    SourceLocation receiverLoc, SourceLocation selectorLoc,SourceLocation rbrac,
    ExprTy **ArgExprs, unsigned NumArgs);

  // ActOnInstanceMessage - used for both unary and keyword messages.
  // ArgExprs is optional - if it is present, the number of expressions
  // is obtained from NumArgs.
  virtual ExprResult ActOnInstanceMessage(
    ExprTy *receiver, Selector Sel,
    SourceLocation lbrac, SourceLocation receiverLoc, SourceLocation rbrac,
    ExprTy **ArgExprs, unsigned NumArgs);

  /// ActOnPragmaPack - Called on well formed #pragma pack(...).
  virtual void ActOnPragmaPack(PragmaPackKind Kind,
                               IdentifierInfo *Name,
                               ExprTy *Alignment,
                               SourceLocation PragmaLoc,
                               SourceLocation LParenLoc,
                               SourceLocation RParenLoc);

  /// ActOnPragmaUnused - Called on well-formed '#pragma unused'.
  virtual void ActOnPragmaUnused(const Token *Identifiers,
                                 unsigned NumIdentifiers, Scope *curScope,
                                 SourceLocation PragmaLoc,
                                 SourceLocation LParenLoc,
                                 SourceLocation RParenLoc);

  NamedDecl *DeclClonePragmaWeak(NamedDecl *ND, IdentifierInfo *II);
  void DeclApplyPragmaWeak(Scope *S, NamedDecl *ND, WeakInfo &W);

  /// ActOnPragmaWeakID - Called on well formed #pragma weak ident.
  virtual void ActOnPragmaWeakID(IdentifierInfo* WeakName,
                                 SourceLocation PragmaLoc,
                                 SourceLocation WeakNameLoc);

  /// ActOnPragmaWeakAlias - Called on well formed #pragma weak ident = ident.
  virtual void ActOnPragmaWeakAlias(IdentifierInfo* WeakName,
                                    IdentifierInfo* AliasName,
                                    SourceLocation PragmaLoc,
                                    SourceLocation WeakNameLoc,
                                    SourceLocation AliasNameLoc);

  /// getPragmaPackAlignment() - Return the current alignment as specified by
  /// the current #pragma pack directive, or 0 if none is currently active.
  unsigned getPragmaPackAlignment() const;

  /// FreePackedContext - Deallocate and null out PackContext.
  void FreePackedContext();

  /// ImpCastExprToType - If Expr is not of type 'Type', insert an implicit
  /// cast.  If there is already an implicit cast, merge into the existing one.
  /// If isLvalue, the result of the cast is an lvalue.
  void ImpCastExprToType(Expr *&Expr, QualType Type, CastExpr::CastKind Kind,
                         bool isLvalue = false);

  // UsualUnaryConversions - promotes integers (C99 6.3.1.1p2) and converts
  // functions and arrays to their respective pointers (C99 6.3.2.1).
  Expr *UsualUnaryConversions(Expr *&expr);

  // DefaultFunctionArrayConversion - converts functions and arrays
  // to their respective pointers (C99 6.3.2.1).
  void DefaultFunctionArrayConversion(Expr *&expr);

  // DefaultArgumentPromotion (C99 6.5.2.2p6). Used for function calls that
  // do not have a prototype. Integer promotions are performed on each
  // argument, and arguments that have type float are promoted to double.
  void DefaultArgumentPromotion(Expr *&Expr);

  // Used for emitting the right warning by DefaultVariadicArgumentPromotion
  enum VariadicCallType {
    VariadicFunction,
    VariadicBlock,
    VariadicMethod,
    VariadicConstructor,
    VariadicDoesNotApply
  };

  /// GatherArgumentsForCall - Collector argument expressions for various
  /// form of call prototypes.
  bool GatherArgumentsForCall(SourceLocation CallLoc,
                              FunctionDecl *FDecl,
                              const FunctionProtoType *Proto,
                              unsigned FirstProtoArg,
                              Expr **Args, unsigned NumArgs,
                              llvm::SmallVector<Expr *, 8> &AllArgs,
                              VariadicCallType CallType = VariadicDoesNotApply);

  // DefaultVariadicArgumentPromotion - Like DefaultArgumentPromotion, but
  // will warn if the resulting type is not a POD type.
  bool DefaultVariadicArgumentPromotion(Expr *&Expr, VariadicCallType CT);

  // UsualArithmeticConversions - performs the UsualUnaryConversions on it's
  // operands and then handles various conversions that are common to binary
  // operators (C99 6.3.1.8). If both operands aren't arithmetic, this
  // routine returns the first non-arithmetic type found. The client is
  // responsible for emitting appropriate error diagnostics.
  QualType UsualArithmeticConversions(Expr *&lExpr, Expr *&rExpr,
                                      bool isCompAssign = false);

  /// AssignConvertType - All of the 'assignment' semantic checks return this
  /// enum to indicate whether the assignment was allowed.  These checks are
  /// done for simple assignments, as well as initialization, return from
  /// function, argument passing, etc.  The query is phrased in terms of a
  /// source and destination type.
  enum AssignConvertType {
    /// Compatible - the types are compatible according to the standard.
    Compatible,

    /// PointerToInt - The assignment converts a pointer to an int, which we
    /// accept as an extension.
    PointerToInt,

    /// IntToPointer - The assignment converts an int to a pointer, which we
    /// accept as an extension.
    IntToPointer,

    /// FunctionVoidPointer - The assignment is between a function pointer and
    /// void*, which the standard doesn't allow, but we accept as an extension.
    FunctionVoidPointer,

    /// IncompatiblePointer - The assignment is between two pointers types that
    /// are not compatible, but we accept them as an extension.
    IncompatiblePointer,

    /// IncompatiblePointer - The assignment is between two pointers types which
    /// point to integers which have a different sign, but are otherwise identical.
    /// This is a subset of the above, but broken out because it's by far the most
    /// common case of incompatible pointers.
    IncompatiblePointerSign,

    /// CompatiblePointerDiscardsQualifiers - The assignment discards
    /// c/v/r qualifiers, which we accept as an extension.
    CompatiblePointerDiscardsQualifiers,
    
    /// IncompatibleNestedPointerQualifiers - The assignment is between two
    /// nested pointer types, and the qualifiers other than the first two
    /// levels differ e.g. char ** -> const char **, but we accept them as an 
    /// extension. 
    IncompatibleNestedPointerQualifiers,

    /// IncompatibleVectors - The assignment is between two vector types that
    /// have the same size, which we accept as an extension.
    IncompatibleVectors,

    /// IntToBlockPointer - The assignment converts an int to a block
    /// pointer. We disallow this.
    IntToBlockPointer,

    /// IncompatibleBlockPointer - The assignment is between two block
    /// pointers types that are not compatible.
    IncompatibleBlockPointer,

    /// IncompatibleObjCQualifiedId - The assignment is between a qualified
    /// id type and something else (that is incompatible with it). For example,
    /// "id <XXX>" = "Foo *", where "Foo *" doesn't implement the XXX protocol.
    IncompatibleObjCQualifiedId,

    /// Incompatible - We reject this conversion outright, it is invalid to
    /// represent it in the AST.
    Incompatible
  };
  
  /// DiagnoseAssignmentResult - Emit a diagnostic, if required, for the
  /// assignment conversion type specified by ConvTy.  This returns true if the
  /// conversion was invalid or false if the conversion was accepted.
  bool DiagnoseAssignmentResult(AssignConvertType ConvTy,
                                SourceLocation Loc,
                                QualType DstType, QualType SrcType,
                                Expr *SrcExpr, AssignmentAction Action);

  /// CheckAssignmentConstraints - Perform type checking for assignment,
  /// argument passing, variable initialization, and function return values.
  /// This routine is only used by the following two methods. C99 6.5.16.
  AssignConvertType CheckAssignmentConstraints(QualType lhs, QualType rhs);

  // CheckSingleAssignmentConstraints - Currently used by
  // CheckAssignmentOperands, and ActOnReturnStmt. Prior to type checking,
  // this routine performs the default function/array converions.
  AssignConvertType CheckSingleAssignmentConstraints(QualType lhs,
                                                     Expr *&rExpr);

  // \brief If the lhs type is a transparent union, check whether we
  // can initialize the transparent union with the given expression.
  AssignConvertType CheckTransparentUnionArgumentConstraints(QualType lhs,
                                                             Expr *&rExpr);

  // Helper function for CheckAssignmentConstraints (C99 6.5.16.1p1)
  AssignConvertType CheckPointerTypesForAssignment(QualType lhsType,
                                                   QualType rhsType);

  AssignConvertType CheckObjCPointerTypesForAssignment(QualType lhsType,
                                                       QualType rhsType);

  // Helper function for CheckAssignmentConstraints involving two
  // block pointer types.
  AssignConvertType CheckBlockPointerTypesForAssignment(QualType lhsType,
                                                        QualType rhsType);

  bool IsStringLiteralToNonConstPointerConversion(Expr *From, QualType ToType);

  bool CheckExceptionSpecCompatibility(Expr *From, QualType ToType);

  bool PerformImplicitConversion(Expr *&From, QualType ToType,
                                 AssignmentAction Action,
                                 bool AllowExplicit = false,
                                 bool Elidable = false);
  bool PerformImplicitConversion(Expr *&From, QualType ToType,
                                 AssignmentAction Action,
                                 bool AllowExplicit,
                                 bool Elidable,
                                 ImplicitConversionSequence& ICS);
  bool PerformImplicitConversion(Expr *&From, QualType ToType,
                                 const ImplicitConversionSequence& ICS,
                                 AssignmentAction Action,
                                 bool IgnoreBaseAccess = false);
  bool PerformImplicitConversion(Expr *&From, QualType ToType,
                                 const StandardConversionSequence& SCS,
                                 AssignmentAction Action, bool IgnoreBaseAccess);
  
  bool BuildCXXDerivedToBaseExpr(Expr *&From, CastExpr::CastKind CastKind,
                                 const ImplicitConversionSequence& ICS);

  /// the following "Check" methods will return a valid/converted QualType
  /// or a null QualType (indicating an error diagnostic was issued).

  /// type checking binary operators (subroutines of CreateBuiltinBinOp).
  QualType InvalidOperands(SourceLocation l, Expr *&lex, Expr *&rex);
  QualType CheckPointerToMemberOperands( // C++ 5.5
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, bool isIndirect);
  QualType CheckMultiplyDivideOperands( // C99 6.5.5
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, bool isCompAssign = false);
  QualType CheckRemainderOperands( // C99 6.5.5
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, bool isCompAssign = false);
  QualType CheckAdditionOperands( // C99 6.5.6
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, QualType* CompLHSTy = 0);
  QualType CheckSubtractionOperands( // C99 6.5.6
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, QualType* CompLHSTy = 0);
  QualType CheckShiftOperands( // C99 6.5.7
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, bool isCompAssign = false);
  QualType CheckCompareOperands( // C99 6.5.8/9
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, unsigned Opc, bool isRelational);
  QualType CheckBitwiseOperands( // C99 6.5.[10...12]
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, bool isCompAssign = false);
  QualType CheckLogicalOperands( // C99 6.5.[13,14]
    Expr *&lex, Expr *&rex, SourceLocation OpLoc);
  // CheckAssignmentOperands is used for both simple and compound assignment.
  // For simple assignment, pass both expressions and a null converted type.
  // For compound assignment, pass both expressions and the converted type.
  QualType CheckAssignmentOperands( // C99 6.5.16.[1,2]
    Expr *lex, Expr *&rex, SourceLocation OpLoc, QualType convertedType);
  QualType CheckCommaOperands( // C99 6.5.17
    Expr *lex, Expr *&rex, SourceLocation OpLoc);
  QualType CheckConditionalOperands( // C99 6.5.15
    Expr *&cond, Expr *&lhs, Expr *&rhs, SourceLocation questionLoc);
  QualType CXXCheckConditionalOperands( // C++ 5.16
    Expr *&cond, Expr *&lhs, Expr *&rhs, SourceLocation questionLoc);
  QualType FindCompositePointerType(Expr *&E1, Expr *&E2); // C++ 5.9

  QualType FindCompositeObjCPointerType(Expr *&LHS, Expr *&RHS,
                                        SourceLocation questionLoc);
  
  /// type checking for vector binary operators.
  inline QualType CheckVectorOperands(SourceLocation l, Expr *&lex, Expr *&rex);
  inline QualType CheckVectorCompareOperands(Expr *&lex, Expr *&rx,
                                             SourceLocation l, bool isRel);

  /// type checking unary operators (subroutines of ActOnUnaryOp).
  /// C99 6.5.3.1, 6.5.3.2, 6.5.3.4
  QualType CheckIncrementDecrementOperand(Expr *op, SourceLocation OpLoc,
                                          bool isInc);
  QualType CheckAddressOfOperand(Expr *op, SourceLocation OpLoc);
  QualType CheckIndirectionOperand(Expr *op, SourceLocation OpLoc);
  QualType CheckRealImagOperand(Expr *&Op, SourceLocation OpLoc, bool isReal);

  /// type checking primary expressions.
  QualType CheckExtVectorComponent(QualType baseType, SourceLocation OpLoc,
                                   const IdentifierInfo *Comp,
                                   SourceLocation CmpLoc);

  /// type checking declaration initializers (C99 6.7.8)
  bool CheckInitList(const InitializedEntity &Entity,
                     InitListExpr *&InitList, QualType &DeclType);
  bool CheckForConstantInitializer(Expr *e, QualType t);

  // type checking C++ declaration initializers (C++ [dcl.init]).

  /// ReferenceCompareResult - Expresses the result of comparing two
  /// types (cv1 T1 and cv2 T2) to determine their compatibility for the
  /// purposes of initialization by reference (C++ [dcl.init.ref]p4).
  enum ReferenceCompareResult {
    /// Ref_Incompatible - The two types are incompatible, so direct
    /// reference binding is not possible.
    Ref_Incompatible = 0,
    /// Ref_Related - The two types are reference-related, which means
    /// that their unqualified forms (T1 and T2) are either the same
    /// or T1 is a base class of T2.
    Ref_Related,
    /// Ref_Compatible_With_Added_Qualification - The two types are
    /// reference-compatible with added qualification, meaning that
    /// they are reference-compatible and the qualifiers on T1 (cv1)
    /// are greater than the qualifiers on T2 (cv2).
    Ref_Compatible_With_Added_Qualification,
    /// Ref_Compatible - The two types are reference-compatible and
    /// have equivalent qualifiers (cv1 == cv2).
    Ref_Compatible
  };

  ReferenceCompareResult CompareReferenceRelationship(SourceLocation Loc,
                                                      QualType T1, QualType T2,
                                                      bool& DerivedToBase);

  bool CheckReferenceInit(Expr *&simpleInit_or_initList, QualType declType,
                          SourceLocation DeclLoc,
                          bool SuppressUserConversions,
                          bool AllowExplicit,
                          bool ForceRValue,
                          ImplicitConversionSequence *ICS = 0,
                          bool IgnoreBaseAccess = false);

  /// CheckCastTypes - Check type constraints for casting between types under
  /// C semantics, or forward to CXXCheckCStyleCast in C++.
  bool CheckCastTypes(SourceRange TyRange, QualType CastTy, Expr *&CastExpr,
                      CastExpr::CastKind &Kind,
                      CXXMethodDecl *& ConversionDecl,
                      bool FunctionalStyle = false);

  // CheckVectorCast - check type constraints for vectors.
  // Since vectors are an extension, there are no C standard reference for this.
  // We allow casting between vectors and integer datatypes of the same size.
  // returns true if the cast is invalid
  bool CheckVectorCast(SourceRange R, QualType VectorTy, QualType Ty,
                       CastExpr::CastKind &Kind);

  // CheckExtVectorCast - check type constraints for extended vectors.
  // Since vectors are an extension, there are no C standard reference for this.
  // We allow casting between vectors and integer datatypes of the same size,
  // or vectors and the element type of that vector.
  // returns true if the cast is invalid
  bool CheckExtVectorCast(SourceRange R, QualType VectorTy, Expr *&CastExpr,
                          CastExpr::CastKind &Kind);

  /// CXXCheckCStyleCast - Check constraints of a C-style or function-style
  /// cast under C++ semantics.
  bool CXXCheckCStyleCast(SourceRange R, QualType CastTy, Expr *&CastExpr,
                          CastExpr::CastKind &Kind, bool FunctionalStyle,
                          CXXMethodDecl *&ConversionDecl);

  /// CheckMessageArgumentTypes - Check types in an Obj-C message send.
  /// \param Method - May be null.
  /// \param [out] ReturnType - The return type of the send.
  /// \return true iff there were any incompatible types.
  bool CheckMessageArgumentTypes(Expr **Args, unsigned NumArgs, Selector Sel,
                                 ObjCMethodDecl *Method, bool isClassMessage,
                                 SourceLocation lbrac, SourceLocation rbrac,
                                 QualType &ReturnType);

  /// CheckBooleanCondition - Diagnose problems involving the use of
  /// the given expression as a boolean condition (e.g. in an if
  /// statement).  Also performs the standard function and array
  /// decays, possibly changing the input variable.
  ///
  /// \param Loc - A location associated with the condition, e.g. the
  /// 'if' keyword.
  /// \return true iff there were any errors
  bool CheckBooleanCondition(Expr *&CondExpr, SourceLocation Loc);

  /// DiagnoseAssignmentAsCondition - Given that an expression is
  /// being used as a boolean condition, warn if it's an assignment.
  void DiagnoseAssignmentAsCondition(Expr *E);

  /// CheckCXXBooleanCondition - Returns true if conversion to bool is invalid.
  bool CheckCXXBooleanCondition(Expr *&CondExpr);

  /// ConvertIntegerToTypeWarnOnOverflow - Convert the specified APInt to have
  /// the specified width and sign.  If an overflow occurs, detect it and emit
  /// the specified diagnostic.
  void ConvertIntegerToTypeWarnOnOverflow(llvm::APSInt &OldVal,
                                          unsigned NewWidth, bool NewSign,
                                          SourceLocation Loc, unsigned DiagID);

  /// Checks that the Objective-C declaration is declared in the global scope.
  /// Emits an error and marks the declaration as invalid if it's not declared
  /// in the global scope.
  bool CheckObjCDeclScope(Decl *D);

  void InitBuiltinVaListType();

  /// VerifyIntegerConstantExpression - verifies that an expression is an ICE,
  /// and reports the appropriate diagnostics. Returns false on success.
  /// Can optionally return the value of the expression.
  bool VerifyIntegerConstantExpression(const Expr *E, llvm::APSInt *Result = 0);

  /// VerifyBitField - verifies that a bit field expression is an ICE and has
  /// the correct width, and that the field type is valid.
  /// Returns false on success.
  /// Can optionally return whether the bit-field is of width 0
  bool VerifyBitField(SourceLocation FieldLoc, IdentifierInfo *FieldName,
                      QualType FieldTy, const Expr *BitWidth,
                      bool *ZeroWidth = 0);

  /// \name Code completion
  //@{
  virtual void CodeCompleteOrdinaryName(Scope *S);
  virtual void CodeCompleteMemberReferenceExpr(Scope *S, ExprTy *Base,
                                               SourceLocation OpLoc,
                                               bool IsArrow);
  virtual void CodeCompleteTag(Scope *S, unsigned TagSpec);
  virtual void CodeCompleteCase(Scope *S);
  virtual void CodeCompleteCall(Scope *S, ExprTy *Fn,
                                ExprTy **Args, unsigned NumArgs);
  virtual void CodeCompleteQualifiedId(Scope *S, const CXXScopeSpec &SS,
                                       bool EnteringContext);
  virtual void CodeCompleteUsing(Scope *S);
  virtual void CodeCompleteUsingDirective(Scope *S);
  virtual void CodeCompleteNamespaceDecl(Scope *S);
  virtual void CodeCompleteNamespaceAliasDecl(Scope *S);
  virtual void CodeCompleteOperatorName(Scope *S);
  
  virtual void CodeCompleteObjCAtDirective(Scope *S, DeclPtrTy ObjCImpDecl,
                                           bool InInterface);
  virtual void CodeCompleteObjCAtStatement(Scope *S);
  virtual void CodeCompleteObjCAtExpression(Scope *S);
  virtual void CodeCompleteObjCPropertyFlags(Scope *S, ObjCDeclSpec &ODS);
  virtual void CodeCompleteObjCPropertyGetter(Scope *S, DeclPtrTy ClassDecl,
                                              DeclPtrTy *Methods,
                                              unsigned NumMethods);
  virtual void CodeCompleteObjCPropertySetter(Scope *S, DeclPtrTy ClassDecl,
                                              DeclPtrTy *Methods,
                                              unsigned NumMethods);

  virtual void CodeCompleteObjCClassMessage(Scope *S, IdentifierInfo *FName,
                                            SourceLocation FNameLoc,
                                            IdentifierInfo **SelIdents, 
                                            unsigned NumSelIdents);
  virtual void CodeCompleteObjCInstanceMessage(Scope *S, ExprTy *Receiver,
                                               IdentifierInfo **SelIdents,
                                               unsigned NumSelIdents);
  virtual void CodeCompleteObjCProtocolReferences(IdentifierLocPair *Protocols,
                                                  unsigned NumProtocols);
  virtual void CodeCompleteObjCProtocolDecl(Scope *S);
  virtual void CodeCompleteObjCInterfaceDecl(Scope *S);
  virtual void CodeCompleteObjCSuperclass(Scope *S, 
                                          IdentifierInfo *ClassName);
  virtual void CodeCompleteObjCImplementationDecl(Scope *S);
  virtual void CodeCompleteObjCInterfaceCategory(Scope *S, 
                                                 IdentifierInfo *ClassName);
  virtual void CodeCompleteObjCImplementationCategory(Scope *S, 
                                                    IdentifierInfo *ClassName);
  virtual void CodeCompleteObjCPropertyDefinition(Scope *S, 
                                                  DeclPtrTy ObjCImpDecl);
  virtual void CodeCompleteObjCPropertySynthesizeIvar(Scope *S, 
                                                  IdentifierInfo *PropertyName,
                                                      DeclPtrTy ObjCImpDecl);
  //@}
  
  //===--------------------------------------------------------------------===//
  // Extra semantic analysis beyond the C type system
private:
  bool CheckFunctionCall(FunctionDecl *FDecl, CallExpr *TheCall);
  bool CheckBlockCall(NamedDecl *NDecl, CallExpr *TheCall);

  SourceLocation getLocationOfStringLiteralByte(const StringLiteral *SL,
                                                unsigned ByteNo) const;
  bool CheckablePrintfAttr(const FormatAttr *Format, CallExpr *TheCall);
  bool CheckObjCString(Expr *Arg);

  Action::OwningExprResult CheckBuiltinFunctionCall(unsigned BuiltinID,
                                                    CallExpr *TheCall);
  bool SemaBuiltinVAStart(CallExpr *TheCall);
  bool SemaBuiltinUnorderedCompare(CallExpr *TheCall);
  bool SemaBuiltinUnaryFP(CallExpr *TheCall);
  bool SemaBuiltinStackAddress(CallExpr *TheCall);

public:
  // Used by C++ template instantiation.
  Action::OwningExprResult SemaBuiltinShuffleVector(CallExpr *TheCall);

private:
  bool SemaBuiltinPrefetch(CallExpr *TheCall);
  bool SemaBuiltinObjectSize(CallExpr *TheCall);
  bool SemaBuiltinLongjmp(CallExpr *TheCall);
  bool SemaBuiltinAtomicOverloaded(CallExpr *TheCall);
  bool SemaBuiltinEHReturnDataRegNo(CallExpr *TheCall);
  bool SemaCheckStringLiteral(const Expr *E, const CallExpr *TheCall,
                              bool HasVAListArg, unsigned format_idx,
                              unsigned firstDataArg);
  void CheckPrintfString(const StringLiteral *FExpr, const Expr *OrigFormatExpr,
                         const CallExpr *TheCall, bool HasVAListArg,
                         unsigned format_idx, unsigned firstDataArg);
  void CheckNonNullArguments(const NonNullAttr *NonNull,
                             const CallExpr *TheCall);
  void CheckPrintfArguments(const CallExpr *TheCall, bool HasVAListArg,
                            unsigned format_idx, unsigned firstDataArg);
  void CheckReturnStackAddr(Expr *RetValExp, QualType lhsType,
                            SourceLocation ReturnLoc);
  void CheckFloatComparison(SourceLocation loc, Expr* lex, Expr* rex);
};

//===--------------------------------------------------------------------===//
// Typed version of Parser::ExprArg (smart pointer for wrapping Expr pointers).
template <typename T>
class ExprOwningPtr : public Action::ExprArg {
public:
  ExprOwningPtr(Sema *S, T *expr) : Action::ExprArg(*S, expr) {}

  void reset(T* p) { Action::ExprArg::operator=(p); }
  T* get() const { return static_cast<T*>(Action::ExprArg::get()); }
  T* take() { return static_cast<T*>(Action::ExprArg::take()); }
  T* release() { return take(); }

  T& operator*() const { return *get(); }
  T* operator->() const { return get(); }
};

}  // end namespace clang

#endif
