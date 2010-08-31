//===- CIndex.cpp - Clang-C Source Indexing Library -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the main API hooks in the Clang-C Source Indexing
// library.
//
//===----------------------------------------------------------------------===//

#include "CIndexer.h"
#include "CXCursor.h"
#include "CXType.h"
#include "CXSourceLocation.h"
#include "CIndexDiagnostic.h"

#include "clang/Basic/Version.h"

#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TypeLocVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Timer.h"
#include "llvm/System/Program.h"
#include "llvm/System/Signals.h"

// Needed to define L_TMPNAM on some systems.
#include <cstdio>

using namespace clang;
using namespace clang::cxcursor;
using namespace clang::cxstring;

//===----------------------------------------------------------------------===//
// Crash Reporting.
//===----------------------------------------------------------------------===//

#ifdef USE_CRASHTRACER
#include "clang/Analysis/Support/SaveAndRestore.h"
// Integrate with crash reporter.
static const char *__crashreporter_info__ = 0;
asm(".desc ___crashreporter_info__, 0x10");
#define NUM_CRASH_STRINGS 32
static unsigned crashtracer_counter = 0;
static unsigned crashtracer_counter_id[NUM_CRASH_STRINGS] = { 0 };
static const char *crashtracer_strings[NUM_CRASH_STRINGS] = { 0 };
static const char *agg_crashtracer_strings[NUM_CRASH_STRINGS] = { 0 };

static unsigned SetCrashTracerInfo(const char *str,
                                   llvm::SmallString<1024> &AggStr) {

  unsigned slot = 0;
  while (crashtracer_strings[slot]) {
    if (++slot == NUM_CRASH_STRINGS)
      slot = 0;
  }
  crashtracer_strings[slot] = str;
  crashtracer_counter_id[slot] = ++crashtracer_counter;

  // We need to create an aggregate string because multiple threads
  // may be in this method at one time.  The crash reporter string
  // will attempt to overapproximate the set of in-flight invocations
  // of this function.  Race conditions can still cause this goal
  // to not be achieved.
  {
    llvm::raw_svector_ostream Out(AggStr);
    for (unsigned i = 0; i < NUM_CRASH_STRINGS; ++i)
      if (crashtracer_strings[i]) Out << crashtracer_strings[i] << '\n';
  }
  __crashreporter_info__ = agg_crashtracer_strings[slot] =  AggStr.c_str();
  return slot;
}

static void ResetCrashTracerInfo(unsigned slot) {
  unsigned max_slot = 0;
  unsigned max_value = 0;

  crashtracer_strings[slot] = agg_crashtracer_strings[slot] = 0;

  for (unsigned i = 0 ; i < NUM_CRASH_STRINGS; ++i)
    if (agg_crashtracer_strings[i] &&
        crashtracer_counter_id[i] > max_value) {
      max_slot = i;
      max_value = crashtracer_counter_id[i];
    }

  __crashreporter_info__ = agg_crashtracer_strings[max_slot];
}

namespace {
class ArgsCrashTracerInfo {
  llvm::SmallString<1024> CrashString;
  llvm::SmallString<1024> AggregateString;
  unsigned crashtracerSlot;
public:
  ArgsCrashTracerInfo(llvm::SmallVectorImpl<const char*> &Args)
    : crashtracerSlot(0)
  {
    {
      llvm::raw_svector_ostream Out(CrashString);
      Out << "ClangCIndex [" << getClangFullVersion() << "]"
          << "[createTranslationUnitFromSourceFile]: clang";
      for (llvm::SmallVectorImpl<const char*>::iterator I=Args.begin(),
           E=Args.end(); I!=E; ++I)
        Out << ' ' << *I;
    }
    crashtracerSlot = SetCrashTracerInfo(CrashString.c_str(),
                                         AggregateString);
  }

  ~ArgsCrashTracerInfo() {
    ResetCrashTracerInfo(crashtracerSlot);
  }
};
}
#endif

/// \brief The result of comparing two source ranges.
enum RangeComparisonResult {
  /// \brief Either the ranges overlap or one of the ranges is invalid.
  RangeOverlap,

  /// \brief The first range ends before the second range starts.
  RangeBefore,

  /// \brief The first range starts after the second range ends.
  RangeAfter
};

/// \brief Compare two source ranges to determine their relative position in
/// the translation unit.
static RangeComparisonResult RangeCompare(SourceManager &SM,
                                          SourceRange R1,
                                          SourceRange R2) {
  assert(R1.isValid() && "First range is invalid?");
  assert(R2.isValid() && "Second range is invalid?");
  if (R1.getEnd() != R2.getBegin() &&
      SM.isBeforeInTranslationUnit(R1.getEnd(), R2.getBegin()))
    return RangeBefore;
  if (R2.getEnd() != R1.getBegin() &&
      SM.isBeforeInTranslationUnit(R2.getEnd(), R1.getBegin()))
    return RangeAfter;
  return RangeOverlap;
}

/// \brief Determine if a source location falls within, before, or after a
///   a given source range.
static RangeComparisonResult LocationCompare(SourceManager &SM,
                                             SourceLocation L, SourceRange R) {
  assert(R.isValid() && "First range is invalid?");
  assert(L.isValid() && "Second range is invalid?");
  if (L == R.getBegin() || L == R.getEnd())
    return RangeOverlap;
  if (SM.isBeforeInTranslationUnit(L, R.getBegin()))
    return RangeBefore;
  if (SM.isBeforeInTranslationUnit(R.getEnd(), L))
    return RangeAfter;
  return RangeOverlap;
}

/// \brief Translate a Clang source range into a CIndex source range.
///
/// Clang internally represents ranges where the end location points to the
/// start of the token at the end. However, for external clients it is more
/// useful to have a CXSourceRange be a proper half-open interval. This routine
/// does the appropriate translation.
CXSourceRange cxloc::translateSourceRange(const SourceManager &SM,
                                          const LangOptions &LangOpts,
                                          const CharSourceRange &R) {
  // We want the last character in this location, so we will adjust the
  // location accordingly.
  // FIXME: How do do this with a macro instantiation location?
  SourceLocation EndLoc = R.getEnd();
  if (R.isTokenRange() && !EndLoc.isInvalid() && EndLoc.isFileID()) {
    unsigned Length = Lexer::MeasureTokenLength(EndLoc, SM, LangOpts);
    EndLoc = EndLoc.getFileLocWithOffset(Length);
  }

  CXSourceRange Result = { { (void *)&SM, (void *)&LangOpts },
                           R.getBegin().getRawEncoding(),
                           EndLoc.getRawEncoding() };
  return Result;
}

//===----------------------------------------------------------------------===//
// Cursor visitor.
//===----------------------------------------------------------------------===//

namespace {

// Cursor visitor.
class CursorVisitor : public DeclVisitor<CursorVisitor, bool>,
                      public TypeLocVisitor<CursorVisitor, bool>,
                      public StmtVisitor<CursorVisitor, bool>
{
  /// \brief The translation unit we are traversing.
  ASTUnit *TU;

  /// \brief The parent cursor whose children we are traversing.
  CXCursor Parent;

  /// \brief The declaration that serves at the parent of any statement or
  /// expression nodes.
  Decl *StmtParent;

  /// \brief The visitor function.
  CXCursorVisitor Visitor;

  /// \brief The opaque client data, to be passed along to the visitor.
  CXClientData ClientData;

  // MaxPCHLevel - the maximum PCH level of declarations that we will pass on
  // to the visitor. Declarations with a PCH level greater than this value will
  // be suppressed.
  unsigned MaxPCHLevel;

  /// \brief When valid, a source range to which the cursor should restrict
  /// its search.
  SourceRange RegionOfInterest;

  using DeclVisitor<CursorVisitor, bool>::Visit;
  using TypeLocVisitor<CursorVisitor, bool>::Visit;
  using StmtVisitor<CursorVisitor, bool>::Visit;

  /// \brief Determine whether this particular source range comes before, comes
  /// after, or overlaps the region of interest.
  ///
  /// \param R a half-open source range retrieved from the abstract syntax tree.
  RangeComparisonResult CompareRegionOfInterest(SourceRange R);

  class SetParentRAII {
    CXCursor &Parent;
    Decl *&StmtParent;
    CXCursor OldParent;

  public:
    SetParentRAII(CXCursor &Parent, Decl *&StmtParent, CXCursor NewParent)
      : Parent(Parent), StmtParent(StmtParent), OldParent(Parent)
    {
      Parent = NewParent;
      if (clang_isDeclaration(Parent.kind))
        StmtParent = getCursorDecl(Parent);
    }

    ~SetParentRAII() {
      Parent = OldParent;
      if (clang_isDeclaration(Parent.kind))
        StmtParent = getCursorDecl(Parent);
    }
  };

public:
  CursorVisitor(ASTUnit *TU, CXCursorVisitor Visitor, CXClientData ClientData,
                unsigned MaxPCHLevel,
                SourceRange RegionOfInterest = SourceRange())
    : TU(TU), Visitor(Visitor), ClientData(ClientData),
      MaxPCHLevel(MaxPCHLevel), RegionOfInterest(RegionOfInterest)
  {
    Parent.kind = CXCursor_NoDeclFound;
    Parent.data[0] = 0;
    Parent.data[1] = 0;
    Parent.data[2] = 0;
    StmtParent = 0;
  }

  bool Visit(CXCursor Cursor, bool CheckedRegionOfInterest = false);
  
  std::pair<PreprocessingRecord::iterator, PreprocessingRecord::iterator>
    getPreprocessedEntities();

  bool VisitChildren(CXCursor Parent);

  // Declaration visitors
  bool VisitAttributes(Decl *D);
  bool VisitBlockDecl(BlockDecl *B);
  bool VisitCXXRecordDecl(CXXRecordDecl *D);
  bool VisitDeclContext(DeclContext *DC);
  bool VisitTranslationUnitDecl(TranslationUnitDecl *D);
  bool VisitTypedefDecl(TypedefDecl *D);
  bool VisitTagDecl(TagDecl *D);
  bool VisitClassTemplatePartialSpecializationDecl(
                                     ClassTemplatePartialSpecializationDecl *D);
  bool VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D);
  bool VisitEnumConstantDecl(EnumConstantDecl *D);
  bool VisitDeclaratorDecl(DeclaratorDecl *DD);
  bool VisitFunctionDecl(FunctionDecl *ND);
  bool VisitFieldDecl(FieldDecl *D);
  bool VisitVarDecl(VarDecl *);
  bool VisitFunctionTemplateDecl(FunctionTemplateDecl *D);
  bool VisitClassTemplateDecl(ClassTemplateDecl *D);
  bool VisitObjCMethodDecl(ObjCMethodDecl *ND);
  bool VisitObjCContainerDecl(ObjCContainerDecl *D);
  bool VisitObjCCategoryDecl(ObjCCategoryDecl *ND);
  bool VisitObjCProtocolDecl(ObjCProtocolDecl *PID);
  bool VisitObjCPropertyDecl(ObjCPropertyDecl *PD);
  bool VisitObjCInterfaceDecl(ObjCInterfaceDecl *D);
  bool VisitObjCImplDecl(ObjCImplDecl *D);
  bool VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D);
  bool VisitObjCImplementationDecl(ObjCImplementationDecl *D);
  // FIXME: ObjCPropertyDecl requires TypeSourceInfo, getter/setter locations,
  // etc.
  // FIXME: ObjCCompatibleAliasDecl requires aliased-class locations.
  bool VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *D);
  bool VisitObjCClassDecl(ObjCClassDecl *D);
  bool VisitLinkageSpecDecl(LinkageSpecDecl *D);
  bool VisitNamespaceDecl(NamespaceDecl *D);
 
  // Name visitor
  bool VisitDeclarationNameInfo(DeclarationNameInfo Name);
  
  // Template visitors
  bool VisitTemplateParameters(const TemplateParameterList *Params);
  bool VisitTemplateName(TemplateName Name, SourceLocation Loc);
  bool VisitTemplateArgumentLoc(const TemplateArgumentLoc &TAL);
  
  // Type visitors
  bool VisitQualifiedTypeLoc(QualifiedTypeLoc TL);
  bool VisitBuiltinTypeLoc(BuiltinTypeLoc TL);
  bool VisitTypedefTypeLoc(TypedefTypeLoc TL);
  bool VisitUnresolvedUsingTypeLoc(UnresolvedUsingTypeLoc TL);
  bool VisitTagTypeLoc(TagTypeLoc TL);
  bool VisitTemplateTypeParmTypeLoc(TemplateTypeParmTypeLoc TL);
  bool VisitObjCInterfaceTypeLoc(ObjCInterfaceTypeLoc TL);
  bool VisitObjCObjectTypeLoc(ObjCObjectTypeLoc TL);
  bool VisitObjCObjectPointerTypeLoc(ObjCObjectPointerTypeLoc TL);
  bool VisitPointerTypeLoc(PointerTypeLoc TL);
  bool VisitBlockPointerTypeLoc(BlockPointerTypeLoc TL);
  bool VisitMemberPointerTypeLoc(MemberPointerTypeLoc TL);
  bool VisitLValueReferenceTypeLoc(LValueReferenceTypeLoc TL);
  bool VisitRValueReferenceTypeLoc(RValueReferenceTypeLoc TL);
  bool VisitFunctionTypeLoc(FunctionTypeLoc TL, bool SkipResultType = false);
  bool VisitArrayTypeLoc(ArrayTypeLoc TL);
  bool VisitTemplateSpecializationTypeLoc(TemplateSpecializationTypeLoc TL);
  // FIXME: Implement visitors here when the unimplemented TypeLocs get
  // implemented
  bool VisitTypeOfExprTypeLoc(TypeOfExprTypeLoc TL);
  bool VisitTypeOfTypeLoc(TypeOfTypeLoc TL);

  // Statement visitors
  bool VisitStmt(Stmt *S);
  bool VisitDeclStmt(DeclStmt *S);
  // FIXME: LabelStmt label?
  bool VisitIfStmt(IfStmt *S);
  bool VisitSwitchStmt(SwitchStmt *S);
  bool VisitCaseStmt(CaseStmt *S);
  bool VisitWhileStmt(WhileStmt *S);
  bool VisitForStmt(ForStmt *S);
//  bool VisitSwitchCase(SwitchCase *S);

  // Expression visitors
  // FIXME: DeclRefExpr with template arguments, nested-name-specifier
  // FIXME: MemberExpr with template arguments, nested-name-specifier
  bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E);
  bool VisitBlockExpr(BlockExpr *B);
  bool VisitCompoundLiteralExpr(CompoundLiteralExpr *E);
  bool VisitExplicitCastExpr(ExplicitCastExpr *E);
  bool VisitObjCMessageExpr(ObjCMessageExpr *E);
  bool VisitObjCEncodeExpr(ObjCEncodeExpr *E);
  bool VisitOffsetOfExpr(OffsetOfExpr *E);
  bool VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E);
  // FIXME: AddrLabelExpr (once we have cursors for labels)
  bool VisitTypesCompatibleExpr(TypesCompatibleExpr *E);
  bool VisitVAArgExpr(VAArgExpr *E);
  // FIXME: InitListExpr (for the designators)
  // FIXME: DesignatedInitExpr
};

} // end anonymous namespace

static SourceRange getRawCursorExtent(CXCursor C);

RangeComparisonResult CursorVisitor::CompareRegionOfInterest(SourceRange R) {
  return RangeCompare(TU->getSourceManager(), R, RegionOfInterest);
}

/// \brief Visit the given cursor and, if requested by the visitor,
/// its children.
///
/// \param Cursor the cursor to visit.
///
/// \param CheckRegionOfInterest if true, then the caller already checked that
/// this cursor is within the region of interest.
///
/// \returns true if the visitation should be aborted, false if it
/// should continue.
bool CursorVisitor::Visit(CXCursor Cursor, bool CheckedRegionOfInterest) {
  if (clang_isInvalid(Cursor.kind))
    return false;

  if (clang_isDeclaration(Cursor.kind)) {
    Decl *D = getCursorDecl(Cursor);
    assert(D && "Invalid declaration cursor");
    if (D->getPCHLevel() > MaxPCHLevel)
      return false;

    if (D->isImplicit())
      return false;
  }

  // If we have a range of interest, and this cursor doesn't intersect with it,
  // we're done.
  if (RegionOfInterest.isValid() && !CheckedRegionOfInterest) {
    SourceRange Range = getRawCursorExtent(Cursor);
    if (Range.isInvalid() || CompareRegionOfInterest(Range))
      return false;
  }

  switch (Visitor(Cursor, Parent, ClientData)) {
  case CXChildVisit_Break:
    return true;

  case CXChildVisit_Continue:
    return false;

  case CXChildVisit_Recurse:
    return VisitChildren(Cursor);
  }

  return false;
}

std::pair<PreprocessingRecord::iterator, PreprocessingRecord::iterator>
CursorVisitor::getPreprocessedEntities() {
  PreprocessingRecord &PPRec
    = *TU->getPreprocessor().getPreprocessingRecord();
  
  bool OnlyLocalDecls
    = !TU->isMainFileAST() && TU->getOnlyLocalDecls();
  
  // There is no region of interest; we have to walk everything.
  if (RegionOfInterest.isInvalid())
    return std::make_pair(PPRec.begin(OnlyLocalDecls),
                          PPRec.end(OnlyLocalDecls));

  // Find the file in which the region of interest lands.
  SourceManager &SM = TU->getSourceManager();
  std::pair<FileID, unsigned> Begin
    = SM.getDecomposedInstantiationLoc(RegionOfInterest.getBegin());
  std::pair<FileID, unsigned> End
    = SM.getDecomposedInstantiationLoc(RegionOfInterest.getEnd());
  
  // The region of interest spans files; we have to walk everything.
  if (Begin.first != End.first)
    return std::make_pair(PPRec.begin(OnlyLocalDecls),
                          PPRec.end(OnlyLocalDecls));
    
  ASTUnit::PreprocessedEntitiesByFileMap &ByFileMap
    = TU->getPreprocessedEntitiesByFile();
  if (ByFileMap.empty()) {
    // Build the mapping from files to sets of preprocessed entities.
    for (PreprocessingRecord::iterator E = PPRec.begin(OnlyLocalDecls),
                                    EEnd = PPRec.end(OnlyLocalDecls);
         E != EEnd; ++E) {
      std::pair<FileID, unsigned> P
        = SM.getDecomposedInstantiationLoc((*E)->getSourceRange().getBegin());
      ByFileMap[P.first].push_back(*E);
    }
  }

  return std::make_pair(ByFileMap[Begin.first].begin(), 
                        ByFileMap[Begin.first].end());
}

/// \brief Visit the children of the given cursor.
///
/// \returns true if the visitation should be aborted, false if it
/// should continue.
bool CursorVisitor::VisitChildren(CXCursor Cursor) {
  if (clang_isReference(Cursor.kind)) {
    // By definition, references have no children.
    return false;
  }

  // Set the Parent field to Cursor, then back to its old value once we're
  // done.
  SetParentRAII SetParent(Parent, StmtParent, Cursor);

  if (clang_isDeclaration(Cursor.kind)) {
    Decl *D = getCursorDecl(Cursor);
    assert(D && "Invalid declaration cursor");
    return VisitAttributes(D) || Visit(D);
  }

  if (clang_isStatement(Cursor.kind))
    return Visit(getCursorStmt(Cursor));
  if (clang_isExpression(Cursor.kind))
    return Visit(getCursorExpr(Cursor));

  if (clang_isTranslationUnit(Cursor.kind)) {
    ASTUnit *CXXUnit = getCursorASTUnit(Cursor);
    if (!CXXUnit->isMainFileAST() && CXXUnit->getOnlyLocalDecls() &&
        RegionOfInterest.isInvalid()) {
      for (ASTUnit::top_level_iterator TL = CXXUnit->top_level_begin(),
                                    TLEnd = CXXUnit->top_level_end();
           TL != TLEnd; ++TL) {
        if (Visit(MakeCXCursor(*TL, CXXUnit), true))
          return true;
      }
    } else if (VisitDeclContext(
                            CXXUnit->getASTContext().getTranslationUnitDecl()))
      return true;

    // Walk the preprocessing record.
    if (CXXUnit->getPreprocessor().getPreprocessingRecord()) {
      // FIXME: Once we have the ability to deserialize a preprocessing record,
      // do so.
      PreprocessingRecord::iterator E, EEnd;
      for (llvm::tie(E, EEnd) = getPreprocessedEntities(); E != EEnd; ++E) {
        if (MacroInstantiation *MI = dyn_cast<MacroInstantiation>(*E)) {
          if (Visit(MakeMacroInstantiationCursor(MI, CXXUnit)))
            return true;
          
          continue;
        }
        
        if (MacroDefinition *MD = dyn_cast<MacroDefinition>(*E)) {
          if (Visit(MakeMacroDefinitionCursor(MD, CXXUnit)))
            return true;
          
          continue;
        }
      }
    }
    return false;
  }

  // Nothing to visit at the moment.
  return false;
}

bool CursorVisitor::VisitBlockDecl(BlockDecl *B) {
  if (Visit(B->getSignatureAsWritten()->getTypeLoc()))
    return true;

  if (Stmt *Body = B->getBody())
    return Visit(MakeCXCursor(Body, StmtParent, TU));

  return false;
}

bool CursorVisitor::VisitDeclContext(DeclContext *DC) {
  for (DeclContext::decl_iterator
       I = DC->decls_begin(), E = DC->decls_end(); I != E; ++I) {

    Decl *D = *I;
    if (D->getLexicalDeclContext() != DC)
      continue;

    CXCursor Cursor = MakeCXCursor(D, TU);

    if (RegionOfInterest.isValid()) {
      SourceRange Range = getRawCursorExtent(Cursor);
      if (Range.isInvalid())
        continue;

      switch (CompareRegionOfInterest(Range)) {
      case RangeBefore:
        // This declaration comes before the region of interest; skip it.
        continue;

      case RangeAfter:
        // This declaration comes after the region of interest; we're done.
        return false;

      case RangeOverlap:
        // This declaration overlaps the region of interest; visit it.
        break;
      }
    }

    if (Visit(Cursor, true))
      return true;
  }

  return false;
}

bool CursorVisitor::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
  llvm_unreachable("Translation units are visited directly by Visit()");
  return false;
}

bool CursorVisitor::VisitTypedefDecl(TypedefDecl *D) {
  if (TypeSourceInfo *TSInfo = D->getTypeSourceInfo())
    return Visit(TSInfo->getTypeLoc());

  return false;
}

bool CursorVisitor::VisitTagDecl(TagDecl *D) {
  return VisitDeclContext(D);
}

bool CursorVisitor::VisitClassTemplatePartialSpecializationDecl(
                                   ClassTemplatePartialSpecializationDecl *D) {
  // FIXME: Visit the "outer" template parameter lists on the TagDecl
  // before visiting these template parameters.
  if (VisitTemplateParameters(D->getTemplateParameters()))
    return true;

  // Visit the partial specialization arguments.
  const TemplateArgumentLoc *TemplateArgs = D->getTemplateArgsAsWritten();
  for (unsigned I = 0, N = D->getNumTemplateArgsAsWritten(); I != N; ++I)
    if (VisitTemplateArgumentLoc(TemplateArgs[I]))
      return true;
  
  return VisitCXXRecordDecl(D);
}

bool CursorVisitor::VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D) {
  // FIXME: Visit default argument
  return false;
}

bool CursorVisitor::VisitEnumConstantDecl(EnumConstantDecl *D) {
  if (Expr *Init = D->getInitExpr())
    return Visit(MakeCXCursor(Init, StmtParent, TU));
  return false;
}

bool CursorVisitor::VisitDeclaratorDecl(DeclaratorDecl *DD) {
  if (TypeSourceInfo *TSInfo = DD->getTypeSourceInfo())
    if (Visit(TSInfo->getTypeLoc()))
      return true;

  return false;
}

bool CursorVisitor::VisitFunctionDecl(FunctionDecl *ND) {
  if (TypeSourceInfo *TSInfo = ND->getTypeSourceInfo()) {
    // Visit the function declaration's syntactic components in the order
    // written. This requires a bit of work.
    TypeLoc TL = TSInfo->getTypeLoc();
    FunctionTypeLoc *FTL = dyn_cast<FunctionTypeLoc>(&TL);
    
    // If we have a function declared directly (without the use of a typedef),
    // visit just the return type. Otherwise, just visit the function's type
    // now.
    if ((FTL && !isa<CXXConversionDecl>(ND) && Visit(FTL->getResultLoc())) ||
        (!FTL && Visit(TL)))
      return true;
    
    // FIXME: Visit the nested-name-specifier, if present.
    
    // Visit the declaration name.
    if (VisitDeclarationNameInfo(ND->getNameInfo()))
      return true;
    
    // FIXME: Visit explicitly-specified template arguments!
    
    // Visit the function parameters, if we have a function type.
    if (FTL && VisitFunctionTypeLoc(*FTL, true))
      return true;
    
    // FIXME: Attributes?
  }
  
  if (ND->isThisDeclarationADefinition() &&
      Visit(MakeCXCursor(ND->getBody(), StmtParent, TU)))
    return true;

  return false;
}

bool CursorVisitor::VisitFieldDecl(FieldDecl *D) {
  if (VisitDeclaratorDecl(D))
    return true;

  if (Expr *BitWidth = D->getBitWidth())
    return Visit(MakeCXCursor(BitWidth, StmtParent, TU));

  return false;
}

bool CursorVisitor::VisitVarDecl(VarDecl *D) {
  if (VisitDeclaratorDecl(D))
    return true;

  if (Expr *Init = D->getInit())
    return Visit(MakeCXCursor(Init, StmtParent, TU));

  return false;
}

bool CursorVisitor::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
  // FIXME: Visit the "outer" template parameter lists on the FunctionDecl
  // before visiting these template parameters.
  if (VisitTemplateParameters(D->getTemplateParameters()))
    return true;
  
  return VisitFunctionDecl(D->getTemplatedDecl());
}

bool CursorVisitor::VisitClassTemplateDecl(ClassTemplateDecl *D) {
  // FIXME: Visit the "outer" template parameter lists on the TagDecl
  // before visiting these template parameters.
  if (VisitTemplateParameters(D->getTemplateParameters()))
    return true;
  
  return VisitCXXRecordDecl(D->getTemplatedDecl());
}

bool CursorVisitor::VisitObjCMethodDecl(ObjCMethodDecl *ND) {
  if (TypeSourceInfo *TSInfo = ND->getResultTypeSourceInfo())
    if (Visit(TSInfo->getTypeLoc()))
      return true;

  for (ObjCMethodDecl::param_iterator P = ND->param_begin(),
       PEnd = ND->param_end();
       P != PEnd; ++P) {
    if (Visit(MakeCXCursor(*P, TU)))
      return true;
  }

  if (ND->isThisDeclarationADefinition() &&
      Visit(MakeCXCursor(ND->getBody(), StmtParent, TU)))
    return true;

  return false;
}

bool CursorVisitor::VisitObjCContainerDecl(ObjCContainerDecl *D) {
  return VisitDeclContext(D);
}

bool CursorVisitor::VisitObjCCategoryDecl(ObjCCategoryDecl *ND) {
  if (Visit(MakeCursorObjCClassRef(ND->getClassInterface(), ND->getLocation(),
                                   TU)))
    return true;

  ObjCCategoryDecl::protocol_loc_iterator PL = ND->protocol_loc_begin();
  for (ObjCCategoryDecl::protocol_iterator I = ND->protocol_begin(),
         E = ND->protocol_end(); I != E; ++I, ++PL)
    if (Visit(MakeCursorObjCProtocolRef(*I, *PL, TU)))
      return true;

  return VisitObjCContainerDecl(ND);
}

bool CursorVisitor::VisitObjCProtocolDecl(ObjCProtocolDecl *PID) {
  ObjCProtocolDecl::protocol_loc_iterator PL = PID->protocol_loc_begin();
  for (ObjCProtocolDecl::protocol_iterator I = PID->protocol_begin(),
       E = PID->protocol_end(); I != E; ++I, ++PL)
    if (Visit(MakeCursorObjCProtocolRef(*I, *PL, TU)))
      return true;

  return VisitObjCContainerDecl(PID);
}

bool CursorVisitor::VisitObjCPropertyDecl(ObjCPropertyDecl *PD) {
  if (Visit(PD->getTypeSourceInfo()->getTypeLoc()))
    return true;

  // FIXME: This implements a workaround with @property declarations also being
  // installed in the DeclContext for the @interface.  Eventually this code
  // should be removed.
  ObjCCategoryDecl *CDecl = dyn_cast<ObjCCategoryDecl>(PD->getDeclContext());
  if (!CDecl || !CDecl->IsClassExtension())
    return false;

  ObjCInterfaceDecl *ID = CDecl->getClassInterface();
  if (!ID)
    return false;

  IdentifierInfo *PropertyId = PD->getIdentifier();
  ObjCPropertyDecl *prevDecl =
    ObjCPropertyDecl::findPropertyDecl(cast<DeclContext>(ID), PropertyId);

  if (!prevDecl)
    return false;

  // Visit synthesized methods since they will be skipped when visiting
  // the @interface.
  if (ObjCMethodDecl *MD = prevDecl->getGetterMethodDecl())
    if (MD->isSynthesized())
      if (Visit(MakeCXCursor(MD, TU)))
        return true;

  if (ObjCMethodDecl *MD = prevDecl->getSetterMethodDecl())
    if (MD->isSynthesized())
      if (Visit(MakeCXCursor(MD, TU)))
        return true;

  return false;
}

bool CursorVisitor::VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
  // Issue callbacks for super class.
  if (D->getSuperClass() &&
      Visit(MakeCursorObjCSuperClassRef(D->getSuperClass(),
                                        D->getSuperClassLoc(),
                                        TU)))
    return true;

  ObjCInterfaceDecl::protocol_loc_iterator PL = D->protocol_loc_begin();
  for (ObjCInterfaceDecl::protocol_iterator I = D->protocol_begin(),
         E = D->protocol_end(); I != E; ++I, ++PL)
    if (Visit(MakeCursorObjCProtocolRef(*I, *PL, TU)))
      return true;

  return VisitObjCContainerDecl(D);
}

bool CursorVisitor::VisitObjCImplDecl(ObjCImplDecl *D) {
  return VisitObjCContainerDecl(D);
}

bool CursorVisitor::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
  // 'ID' could be null when dealing with invalid code.
  if (ObjCInterfaceDecl *ID = D->getClassInterface())
    if (Visit(MakeCursorObjCClassRef(ID, D->getLocation(), TU)))
      return true;

  return VisitObjCImplDecl(D);
}

bool CursorVisitor::VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
#if 0
  // Issue callbacks for super class.
  // FIXME: No source location information!
  if (D->getSuperClass() &&
      Visit(MakeCursorObjCSuperClassRef(D->getSuperClass(),
                                        D->getSuperClassLoc(),
                                        TU)))
    return true;
#endif

  return VisitObjCImplDecl(D);
}

bool CursorVisitor::VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *D) {
  ObjCForwardProtocolDecl::protocol_loc_iterator PL = D->protocol_loc_begin();
  for (ObjCForwardProtocolDecl::protocol_iterator I = D->protocol_begin(),
                                                  E = D->protocol_end();
       I != E; ++I, ++PL)
    if (Visit(MakeCursorObjCProtocolRef(*I, *PL, TU)))
      return true;

  return false;
}

bool CursorVisitor::VisitObjCClassDecl(ObjCClassDecl *D) {
  for (ObjCClassDecl::iterator C = D->begin(), CEnd = D->end(); C != CEnd; ++C)
    if (Visit(MakeCursorObjCClassRef(C->getInterface(), C->getLocation(), TU)))
      return true;

  return false;
}

bool CursorVisitor::VisitNamespaceDecl(NamespaceDecl *D) {
  return VisitDeclContext(D);
}

bool CursorVisitor::VisitDeclarationNameInfo(DeclarationNameInfo Name) {
  switch (Name.getName().getNameKind()) {
  case clang::DeclarationName::Identifier:
  case clang::DeclarationName::CXXLiteralOperatorName:
  case clang::DeclarationName::CXXOperatorName:
  case clang::DeclarationName::CXXUsingDirective:
    return false;
      
  case clang::DeclarationName::CXXConstructorName:
  case clang::DeclarationName::CXXDestructorName:
  case clang::DeclarationName::CXXConversionFunctionName:
    if (TypeSourceInfo *TSInfo = Name.getNamedTypeInfo())
      return Visit(TSInfo->getTypeLoc());
    return false;

  case clang::DeclarationName::ObjCZeroArgSelector:
  case clang::DeclarationName::ObjCOneArgSelector:
  case clang::DeclarationName::ObjCMultiArgSelector:
    // FIXME: Per-identifier location info?
    return false;
  }
  
  return false;
}

bool CursorVisitor::VisitTemplateParameters(
                                          const TemplateParameterList *Params) {
  if (!Params)
    return false;
  
  for (TemplateParameterList::const_iterator P = Params->begin(),
                                          PEnd = Params->end();
       P != PEnd; ++P) {
    if (Visit(MakeCXCursor(*P, TU)))
      return true;
  }
  
  return false;
}

bool CursorVisitor::VisitTemplateName(TemplateName Name, SourceLocation Loc) {
  switch (Name.getKind()) {
  case TemplateName::Template:
    return Visit(MakeCursorTemplateRef(Name.getAsTemplateDecl(), Loc, TU));

  case TemplateName::OverloadedTemplate:
    // FIXME: We need a way to return multiple lookup results in a single
    // cursor.
    return false;

  case TemplateName::DependentTemplate:
    // FIXME: Visit nested-name-specifier.
    return false;
      
  case TemplateName::QualifiedTemplate:
    // FIXME: Visit nested-name-specifier.
    return Visit(MakeCursorTemplateRef(
                                  Name.getAsQualifiedTemplateName()->getDecl(), 
                                       Loc, TU));
  }
                 
  return false;
}

bool CursorVisitor::VisitTemplateArgumentLoc(const TemplateArgumentLoc &TAL) {
  switch (TAL.getArgument().getKind()) {
  case TemplateArgument::Null:
  case TemplateArgument::Integral:
    return false;
      
  case TemplateArgument::Pack:
    // FIXME: Implement when variadic templates come along.
    return false;

  case TemplateArgument::Type:
    if (TypeSourceInfo *TSInfo = TAL.getTypeSourceInfo())
      return Visit(TSInfo->getTypeLoc());
    return false;
      
  case TemplateArgument::Declaration:
    if (Expr *E = TAL.getSourceDeclExpression())
      return Visit(MakeCXCursor(E, StmtParent, TU));
    return false;
      
  case TemplateArgument::Expression:
    if (Expr *E = TAL.getSourceExpression())
      return Visit(MakeCXCursor(E, StmtParent, TU));
    return false;
  
  case TemplateArgument::Template:
    return VisitTemplateName(TAL.getArgument().getAsTemplate(), 
                             TAL.getTemplateNameLoc());
  }
  
  return false;
}

bool CursorVisitor::VisitLinkageSpecDecl(LinkageSpecDecl *D) {
  return VisitDeclContext(D);
}

bool CursorVisitor::VisitQualifiedTypeLoc(QualifiedTypeLoc TL) {
  return Visit(TL.getUnqualifiedLoc());
}

bool CursorVisitor::VisitBuiltinTypeLoc(BuiltinTypeLoc TL) {
  ASTContext &Context = TU->getASTContext();

  // Some builtin types (such as Objective-C's "id", "sel", and
  // "Class") have associated declarations. Create cursors for those.
  QualType VisitType;
  switch (TL.getType()->getAs<BuiltinType>()->getKind()) {
  case BuiltinType::Void:
  case BuiltinType::Bool:
  case BuiltinType::Char_U:
  case BuiltinType::UChar:
  case BuiltinType::Char16:
  case BuiltinType::Char32:
  case BuiltinType::UShort:
  case BuiltinType::UInt:
  case BuiltinType::ULong:
  case BuiltinType::ULongLong:
  case BuiltinType::UInt128:
  case BuiltinType::Char_S:
  case BuiltinType::SChar:
  case BuiltinType::WChar:
  case BuiltinType::Short:
  case BuiltinType::Int:
  case BuiltinType::Long:
  case BuiltinType::LongLong:
  case BuiltinType::Int128:
  case BuiltinType::Float:
  case BuiltinType::Double:
  case BuiltinType::LongDouble:
  case BuiltinType::NullPtr:
  case BuiltinType::Overload:
  case BuiltinType::Dependent:
    break;

  case BuiltinType::UndeducedAuto: // FIXME: Deserves a cursor?
    break;

  case BuiltinType::ObjCId:
    VisitType = Context.getObjCIdType();
    break;

  case BuiltinType::ObjCClass:
    VisitType = Context.getObjCClassType();
    break;

  case BuiltinType::ObjCSel:
    VisitType = Context.getObjCSelType();
    break;
  }

  if (!VisitType.isNull()) {
    if (const TypedefType *Typedef = VisitType->getAs<TypedefType>())
      return Visit(MakeCursorTypeRef(Typedef->getDecl(), TL.getBuiltinLoc(),
                                     TU));
  }

  return false;
}

bool CursorVisitor::VisitTypedefTypeLoc(TypedefTypeLoc TL) {
  return Visit(MakeCursorTypeRef(TL.getTypedefDecl(), TL.getNameLoc(), TU));
}

bool CursorVisitor::VisitUnresolvedUsingTypeLoc(UnresolvedUsingTypeLoc TL) {
  return Visit(MakeCursorTypeRef(TL.getDecl(), TL.getNameLoc(), TU));
}

bool CursorVisitor::VisitTagTypeLoc(TagTypeLoc TL) {
  return Visit(MakeCursorTypeRef(TL.getDecl(), TL.getNameLoc(), TU));
}

bool CursorVisitor::VisitTemplateTypeParmTypeLoc(TemplateTypeParmTypeLoc TL) {
  // FIXME: We can't visit the template template parameter, but there's
  // no context information with which we can match up the depth/index in the
  // type to the appropriate 
  return false;
}

bool CursorVisitor::VisitObjCInterfaceTypeLoc(ObjCInterfaceTypeLoc TL) {
  if (Visit(MakeCursorObjCClassRef(TL.getIFaceDecl(), TL.getNameLoc(), TU)))
    return true;

  return false;
}

bool CursorVisitor::VisitObjCObjectTypeLoc(ObjCObjectTypeLoc TL) {
  if (TL.hasBaseTypeAsWritten() && Visit(TL.getBaseLoc()))
    return true;

  for (unsigned I = 0, N = TL.getNumProtocols(); I != N; ++I) {
    if (Visit(MakeCursorObjCProtocolRef(TL.getProtocol(I), TL.getProtocolLoc(I),
                                        TU)))
      return true;
  }

  return false;
}

bool CursorVisitor::VisitObjCObjectPointerTypeLoc(ObjCObjectPointerTypeLoc TL) {
  return Visit(TL.getPointeeLoc());
}

bool CursorVisitor::VisitPointerTypeLoc(PointerTypeLoc TL) {
  return Visit(TL.getPointeeLoc());
}

bool CursorVisitor::VisitBlockPointerTypeLoc(BlockPointerTypeLoc TL) {
  return Visit(TL.getPointeeLoc());
}

bool CursorVisitor::VisitMemberPointerTypeLoc(MemberPointerTypeLoc TL) {
  return Visit(TL.getPointeeLoc());
}

bool CursorVisitor::VisitLValueReferenceTypeLoc(LValueReferenceTypeLoc TL) {
  return Visit(TL.getPointeeLoc());
}

bool CursorVisitor::VisitRValueReferenceTypeLoc(RValueReferenceTypeLoc TL) {
  return Visit(TL.getPointeeLoc());
}

bool CursorVisitor::VisitFunctionTypeLoc(FunctionTypeLoc TL, 
                                         bool SkipResultType) {
  if (!SkipResultType && Visit(TL.getResultLoc()))
    return true;

  for (unsigned I = 0, N = TL.getNumArgs(); I != N; ++I)
    if (Decl *D = TL.getArg(I))
      if (Visit(MakeCXCursor(D, TU)))
        return true;

  return false;
}

bool CursorVisitor::VisitArrayTypeLoc(ArrayTypeLoc TL) {
  if (Visit(TL.getElementLoc()))
    return true;

  if (Expr *Size = TL.getSizeExpr())
    return Visit(MakeCXCursor(Size, StmtParent, TU));

  return false;
}

bool CursorVisitor::VisitTemplateSpecializationTypeLoc(
                                             TemplateSpecializationTypeLoc TL) {
  // Visit the template name.
  if (VisitTemplateName(TL.getTypePtr()->getTemplateName(), 
                        TL.getTemplateNameLoc()))
    return true;
  
  // Visit the template arguments.
  for (unsigned I = 0, N = TL.getNumArgs(); I != N; ++I)
    if (VisitTemplateArgumentLoc(TL.getArgLoc(I)))
      return true;
  
  return false;
}

bool CursorVisitor::VisitTypeOfExprTypeLoc(TypeOfExprTypeLoc TL) {
  return Visit(MakeCXCursor(TL.getUnderlyingExpr(), StmtParent, TU));
}

bool CursorVisitor::VisitTypeOfTypeLoc(TypeOfTypeLoc TL) {
  if (TypeSourceInfo *TSInfo = TL.getUnderlyingTInfo())
    return Visit(TSInfo->getTypeLoc());

  return false;
}

bool CursorVisitor::VisitStmt(Stmt *S) {
  for (Stmt::child_iterator Child = S->child_begin(), ChildEnd = S->child_end();
       Child != ChildEnd; ++Child) {
    if (Stmt *C = *Child)
      if (Visit(MakeCXCursor(C, StmtParent, TU)))
        return true;
  }

  return false;
}

bool CursorVisitor::VisitCaseStmt(CaseStmt *S) {
  // Specially handle CaseStmts because they can be nested, e.g.:
  //
  //    case 1:
  //    case 2:
  //
  // In this case the second CaseStmt is the child of the first.  Walking
  // these recursively can blow out the stack.
  CXCursor Cursor = MakeCXCursor(S, StmtParent, TU);
  while (true) {
    // Set the Parent field to Cursor, then back to its old value once we're
    //   done.
    SetParentRAII SetParent(Parent, StmtParent, Cursor);

    if (Stmt *LHS = S->getLHS())
      if (Visit(MakeCXCursor(LHS, StmtParent, TU)))
        return true;
    if (Stmt *RHS = S->getRHS())
      if (Visit(MakeCXCursor(RHS, StmtParent, TU)))
        return true;
    if (Stmt *SubStmt = S->getSubStmt()) {
      if (!isa<CaseStmt>(SubStmt))
        return Visit(MakeCXCursor(SubStmt, StmtParent, TU));

      // Specially handle 'CaseStmt' so that we don't blow out the stack.
      CaseStmt *CS = cast<CaseStmt>(SubStmt);
      Cursor = MakeCXCursor(CS, StmtParent, TU);
      if (RegionOfInterest.isValid()) {
        SourceRange Range = CS->getSourceRange();
        if (Range.isInvalid() || CompareRegionOfInterest(Range))
          return false;
      }

      switch (Visitor(Cursor, Parent, ClientData)) {
        case CXChildVisit_Break: return true;
        case CXChildVisit_Continue: return false;
        case CXChildVisit_Recurse:
          // Perform tail-recursion manually.
          S = CS;
          continue;
      }
    }
    return false;
  }
}

bool CursorVisitor::VisitDeclStmt(DeclStmt *S) {
  for (DeclStmt::decl_iterator D = S->decl_begin(), DEnd = S->decl_end();
       D != DEnd; ++D) {
    if (*D && Visit(MakeCXCursor(*D, TU)))
      return true;
  }

  return false;
}

bool CursorVisitor::VisitIfStmt(IfStmt *S) {
  if (VarDecl *Var = S->getConditionVariable()) {
    if (Visit(MakeCXCursor(Var, TU)))
      return true;
  }

  if (S->getCond() && Visit(MakeCXCursor(S->getCond(), StmtParent, TU)))
    return true;
  if (S->getThen() && Visit(MakeCXCursor(S->getThen(), StmtParent, TU)))
    return true;
  if (S->getElse() && Visit(MakeCXCursor(S->getElse(), StmtParent, TU)))
    return true;

  return false;
}

bool CursorVisitor::VisitSwitchStmt(SwitchStmt *S) {
  if (VarDecl *Var = S->getConditionVariable()) {
    if (Visit(MakeCXCursor(Var, TU)))
      return true;
  }

  if (S->getCond() && Visit(MakeCXCursor(S->getCond(), StmtParent, TU)))
    return true;
  if (S->getBody() && Visit(MakeCXCursor(S->getBody(), StmtParent, TU)))
    return true;

  return false;
}

bool CursorVisitor::VisitWhileStmt(WhileStmt *S) {
  if (VarDecl *Var = S->getConditionVariable()) {
    if (Visit(MakeCXCursor(Var, TU)))
      return true;
  }

  if (S->getCond() && Visit(MakeCXCursor(S->getCond(), StmtParent, TU)))
    return true;
  if (S->getBody() && Visit(MakeCXCursor(S->getBody(), StmtParent, TU)))
    return true;

  return false;
}

bool CursorVisitor::VisitForStmt(ForStmt *S) {
  if (S->getInit() && Visit(MakeCXCursor(S->getInit(), StmtParent, TU)))
    return true;
  if (VarDecl *Var = S->getConditionVariable()) {
    if (Visit(MakeCXCursor(Var, TU)))
      return true;
  }

  if (S->getCond() && Visit(MakeCXCursor(S->getCond(), StmtParent, TU)))
    return true;
  if (S->getInc() && Visit(MakeCXCursor(S->getInc(), StmtParent, TU)))
    return true;
  if (S->getBody() && Visit(MakeCXCursor(S->getBody(), StmtParent, TU)))
    return true;

  return false;
}

bool CursorVisitor::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
  if (Visit(MakeCXCursor(E->getArg(0), StmtParent, TU)))
    return true;
  
  if (Visit(MakeCXCursor(E->getCallee(), StmtParent, TU)))
    return true;
  
  for (unsigned I = 1, N = E->getNumArgs(); I != N; ++I)
    if (Visit(MakeCXCursor(E->getArg(I), StmtParent, TU)))
      return true;
  
  return false;
}

bool CursorVisitor::VisitCXXRecordDecl(CXXRecordDecl *D) {
  if (D->isDefinition()) {
    for (CXXRecordDecl::base_class_iterator I = D->bases_begin(),
         E = D->bases_end(); I != E; ++I) {
      if (Visit(cxcursor::MakeCursorCXXBaseSpecifier(I, TU)))
        return true;
    }
  }

  return VisitTagDecl(D);
}


bool CursorVisitor::VisitBlockExpr(BlockExpr *B) {
  return Visit(B->getBlockDecl());
}

bool CursorVisitor::VisitOffsetOfExpr(OffsetOfExpr *E) {
  // FIXME: Visit fields as well?
  if (Visit(E->getTypeSourceInfo()->getTypeLoc()))
    return true;
  
  return VisitExpr(E);
}

bool CursorVisitor::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E) {
  if (E->isArgumentType()) {
    if (TypeSourceInfo *TSInfo = E->getArgumentTypeInfo())
      return Visit(TSInfo->getTypeLoc());

    return false;
  }

  return VisitExpr(E);
}

bool CursorVisitor::VisitExplicitCastExpr(ExplicitCastExpr *E) {
  if (TypeSourceInfo *TSInfo = E->getTypeInfoAsWritten())
    if (Visit(TSInfo->getTypeLoc()))
      return true;

  return VisitCastExpr(E);
}

bool CursorVisitor::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  if (TypeSourceInfo *TSInfo = E->getTypeSourceInfo())
    if (Visit(TSInfo->getTypeLoc()))
      return true;

  return VisitExpr(E);
}

bool CursorVisitor::VisitTypesCompatibleExpr(TypesCompatibleExpr *E) {
  return Visit(E->getArgTInfo1()->getTypeLoc()) || 
         Visit(E->getArgTInfo2()->getTypeLoc());
}

bool CursorVisitor::VisitVAArgExpr(VAArgExpr *E) {
  if (Visit(E->getWrittenTypeInfo()->getTypeLoc()))
    return true;
  
  return Visit(MakeCXCursor(E->getSubExpr(), StmtParent, TU));
}

bool CursorVisitor::VisitObjCMessageExpr(ObjCMessageExpr *E) {
  if (TypeSourceInfo *TSInfo = E->getClassReceiverTypeInfo())
    if (Visit(TSInfo->getTypeLoc()))
      return true;

  return VisitExpr(E);
}

bool CursorVisitor::VisitObjCEncodeExpr(ObjCEncodeExpr *E) {
  return Visit(E->getEncodedTypeSourceInfo()->getTypeLoc());
}


bool CursorVisitor::VisitAttributes(Decl *D) {
  for (AttrVec::const_iterator i = D->attr_begin(), e = D->attr_end();
       i != e; ++i)
    if (Visit(MakeCXCursor(*i, D, TU)))
        return true;

  return false;
}

extern "C" {
CXIndex clang_createIndex(int excludeDeclarationsFromPCH,
                          int displayDiagnostics) {
  // We use crash recovery to make some of our APIs more reliable, implicitly
  // enable it.
  llvm::CrashRecoveryContext::Enable();

  CIndexer *CIdxr = new CIndexer();
  if (excludeDeclarationsFromPCH)
    CIdxr->setOnlyLocalDecls();
  if (displayDiagnostics)
    CIdxr->setDisplayDiagnostics();
  return CIdxr;
}

void clang_disposeIndex(CXIndex CIdx) {
  if (CIdx)
    delete static_cast<CIndexer *>(CIdx);
  if (getenv("LIBCLANG_TIMING"))
    llvm::TimerGroup::printAll(llvm::errs());
}

void clang_setUseExternalASTGeneration(CXIndex CIdx, int value) {
  if (CIdx) {
    CIndexer *CXXIdx = static_cast<CIndexer *>(CIdx);
    CXXIdx->setUseExternalASTGeneration(value);
  }
}

CXTranslationUnit clang_createTranslationUnit(CXIndex CIdx,
                                              const char *ast_filename) {
  if (!CIdx)
    return 0;

  CIndexer *CXXIdx = static_cast<CIndexer *>(CIdx);

  llvm::IntrusiveRefCntPtr<Diagnostic> Diags;
  return ASTUnit::LoadFromASTFile(ast_filename, Diags,
                                  CXXIdx->getOnlyLocalDecls(),
                                  0, 0, true);
}

unsigned clang_defaultEditingTranslationUnitOptions() {
  return CXTranslationUnit_PrecompiledPreamble;
}
  
CXTranslationUnit
clang_createTranslationUnitFromSourceFile(CXIndex CIdx,
                                          const char *source_filename,
                                          int num_command_line_args,
                                          const char **command_line_args,
                                          unsigned num_unsaved_files,
                                          struct CXUnsavedFile *unsaved_files) {
  return clang_parseTranslationUnit(CIdx, source_filename,
                                    command_line_args, num_command_line_args,
                                    unsaved_files, num_unsaved_files,
                                 CXTranslationUnit_DetailedPreprocessingRecord);
}
  
struct ParseTranslationUnitInfo {
  CXIndex CIdx;
  const char *source_filename;
  const char **command_line_args;
  int num_command_line_args;
  struct CXUnsavedFile *unsaved_files;
  unsigned num_unsaved_files;
  unsigned options;
  CXTranslationUnit result;
};
static void clang_parseTranslationUnit_Impl(void *UserData) {
  ParseTranslationUnitInfo *PTUI =
    static_cast<ParseTranslationUnitInfo*>(UserData);
  CXIndex CIdx = PTUI->CIdx;
  const char *source_filename = PTUI->source_filename;
  const char **command_line_args = PTUI->command_line_args;
  int num_command_line_args = PTUI->num_command_line_args;
  struct CXUnsavedFile *unsaved_files = PTUI->unsaved_files;
  unsigned num_unsaved_files = PTUI->num_unsaved_files;
  unsigned options = PTUI->options;
  PTUI->result = 0;

  if (!CIdx)
    return;

  CIndexer *CXXIdx = static_cast<CIndexer *>(CIdx);

  bool PrecompilePreamble = options & CXTranslationUnit_PrecompiledPreamble;
  bool CompleteTranslationUnit
    = ((options & CXTranslationUnit_Incomplete) == 0);
  bool CacheCodeCompetionResults
    = options & CXTranslationUnit_CacheCompletionResults;
  
  // Configure the diagnostics.
  DiagnosticOptions DiagOpts;
  llvm::IntrusiveRefCntPtr<Diagnostic> Diags;
  Diags = CompilerInstance::createDiagnostics(DiagOpts, 0, 0);

  llvm::SmallVector<ASTUnit::RemappedFile, 4> RemappedFiles;
  for (unsigned I = 0; I != num_unsaved_files; ++I) {
    llvm::StringRef Data(unsaved_files[I].Contents, unsaved_files[I].Length);
    const llvm::MemoryBuffer *Buffer
      = llvm::MemoryBuffer::getMemBufferCopy(Data, unsaved_files[I].Filename);
    RemappedFiles.push_back(std::make_pair(unsaved_files[I].Filename,
                                           Buffer));
  }

  if (!CXXIdx->getUseExternalASTGeneration()) {
    llvm::SmallVector<const char *, 16> Args;

    // The 'source_filename' argument is optional.  If the caller does not
    // specify it then it is assumed that the source file is specified
    // in the actual argument list.
    if (source_filename)
      Args.push_back(source_filename);
    
    // Since the Clang C library is primarily used by batch tools dealing with
    // (often very broken) source code, where spell-checking can have a
    // significant negative impact on performance (particularly when 
    // precompiled headers are involved), we disable it by default.
    // Note that we place this argument early in the list, so that it can be
    // overridden by the caller with "-fspell-checking".
    Args.push_back("-fno-spell-checking");
    
    Args.insert(Args.end(), command_line_args,
                command_line_args + num_command_line_args);

    // Do we need the detailed preprocessing record?
    if (options & CXTranslationUnit_DetailedPreprocessingRecord) {
      Args.push_back("-Xclang");
      Args.push_back("-detailed-preprocessing-record");
    }
    
    unsigned NumErrors = Diags->getNumErrors();

#ifdef USE_CRASHTRACER
    ArgsCrashTracerInfo ACTI(Args);
#endif

    llvm::OwningPtr<ASTUnit> Unit(
      ASTUnit::LoadFromCommandLine(Args.data(), Args.data() + Args.size(),
                                   Diags,
                                   CXXIdx->getClangResourcesPath(),
                                   CXXIdx->getOnlyLocalDecls(),
                                   RemappedFiles.data(),
                                   RemappedFiles.size(),
                                   /*CaptureDiagnostics=*/true,
                                   PrecompilePreamble,
                                   CompleteTranslationUnit,
                                   CacheCodeCompetionResults));

    if (NumErrors != Diags->getNumErrors()) {
      // Make sure to check that 'Unit' is non-NULL.
      if (CXXIdx->getDisplayDiagnostics() && Unit.get()) {
        for (ASTUnit::stored_diag_iterator D = Unit->stored_diag_begin(), 
                                        DEnd = Unit->stored_diag_end();
             D != DEnd; ++D) {
          CXStoredDiagnostic Diag(*D, Unit->getASTContext().getLangOptions());
          CXString Msg = clang_formatDiagnostic(&Diag,
                                      clang_defaultDiagnosticDisplayOptions());
          fprintf(stderr, "%s\n", clang_getCString(Msg));
          clang_disposeString(Msg);
        }
#ifdef LLVM_ON_WIN32
        // On Windows, force a flush, since there may be multiple copies of
        // stderr and stdout in the file system, all with different buffers
        // but writing to the same device.
        fflush(stderr);
#endif
      }
    }

    PTUI->result = Unit.take();
    return;
  }

  // Build up the arguments for invoking 'clang'.
  std::vector<const char *> argv;

  // First add the complete path to the 'clang' executable.
  llvm::sys::Path ClangPath = static_cast<CIndexer *>(CIdx)->getClangPath();
  argv.push_back(ClangPath.c_str());

  // Add the '-emit-ast' option as our execution mode for 'clang'.
  argv.push_back("-emit-ast");

  // The 'source_filename' argument is optional.  If the caller does not
  // specify it then it is assumed that the source file is specified
  // in the actual argument list.
  if (source_filename)
    argv.push_back(source_filename);

  // Generate a temporary name for the AST file.
  argv.push_back("-o");
  char astTmpFile[L_tmpnam];
  argv.push_back(tmpnam(astTmpFile));
  
  // Since the Clang C library is primarily used by batch tools dealing with
  // (often very broken) source code, where spell-checking can have a
  // significant negative impact on performance (particularly when 
  // precompiled headers are involved), we disable it by default.
  // Note that we place this argument early in the list, so that it can be
  // overridden by the caller with "-fspell-checking".
  argv.push_back("-fno-spell-checking");

  // Remap any unsaved files to temporary files.
  std::vector<llvm::sys::Path> TemporaryFiles;
  std::vector<std::string> RemapArgs;
  if (RemapFiles(num_unsaved_files, unsaved_files, RemapArgs, TemporaryFiles))
    return;

  // The pointers into the elements of RemapArgs are stable because we
  // won't be adding anything to RemapArgs after this point.
  for (unsigned i = 0, e = RemapArgs.size(); i != e; ++i)
    argv.push_back(RemapArgs[i].c_str());

  // Process the compiler options, stripping off '-o', '-c', '-fsyntax-only'.
  for (int i = 0; i < num_command_line_args; ++i)
    if (const char *arg = command_line_args[i]) {
      if (strcmp(arg, "-o") == 0) {
        ++i; // Also skip the matching argument.
        continue;
      }
      if (strcmp(arg, "-emit-ast") == 0 ||
          strcmp(arg, "-c") == 0 ||
          strcmp(arg, "-fsyntax-only") == 0) {
        continue;
      }

      // Keep the argument.
      argv.push_back(arg);
    }

  // Generate a temporary name for the diagnostics file.
  char tmpFileResults[L_tmpnam];
  char *tmpResultsFileName = tmpnam(tmpFileResults);
  llvm::sys::Path DiagnosticsFile(tmpResultsFileName);
  TemporaryFiles.push_back(DiagnosticsFile);
  argv.push_back("-fdiagnostics-binary");

  // Do we need the detailed preprocessing record?
  if (options & CXTranslationUnit_DetailedPreprocessingRecord) {
    argv.push_back("-Xclang");
    argv.push_back("-detailed-preprocessing-record");
  }
  
  // Add the null terminator.
  argv.push_back(NULL);

  // Invoke 'clang'.
  llvm::sys::Path DevNull; // leave empty, causes redirection to /dev/null
                           // on Unix or NUL (Windows).
  std::string ErrMsg;
  const llvm::sys::Path *Redirects[] = { &DevNull, &DevNull, &DiagnosticsFile,
                                         NULL };
  llvm::sys::Program::ExecuteAndWait(ClangPath, &argv[0], /* env */ NULL,
      /* redirects */ &Redirects[0],
      /* secondsToWait */ 0, /* memoryLimits */ 0, &ErrMsg);

  if (!ErrMsg.empty()) {
    std::string AllArgs;
    for (std::vector<const char*>::iterator I = argv.begin(), E = argv.end();
         I != E; ++I) {
      AllArgs += ' ';
      if (*I)
        AllArgs += *I;
    }

    Diags->Report(diag::err_fe_invoking) << AllArgs << ErrMsg;
  }

  ASTUnit *ATU = ASTUnit::LoadFromASTFile(astTmpFile, Diags,
                                          CXXIdx->getOnlyLocalDecls(),
                                          RemappedFiles.data(),
                                          RemappedFiles.size(),
                                          /*CaptureDiagnostics=*/true);
  if (ATU) {
    LoadSerializedDiagnostics(DiagnosticsFile, 
                              num_unsaved_files, unsaved_files,
                              ATU->getFileManager(),
                              ATU->getSourceManager(),
                              ATU->getStoredDiagnostics());
  } else if (CXXIdx->getDisplayDiagnostics()) {
    // We failed to load the ASTUnit, but we can still deserialize the
    // diagnostics and emit them.
    FileManager FileMgr;
    Diagnostic Diag;
    SourceManager SourceMgr(Diag);
    // FIXME: Faked LangOpts!
    LangOptions LangOpts;
    llvm::SmallVector<StoredDiagnostic, 4> Diags;
    LoadSerializedDiagnostics(DiagnosticsFile, 
                              num_unsaved_files, unsaved_files,
                              FileMgr, SourceMgr, Diags);
    for (llvm::SmallVector<StoredDiagnostic, 4>::iterator D = Diags.begin(), 
                                                       DEnd = Diags.end();
         D != DEnd; ++D) {
      CXStoredDiagnostic Diag(*D, LangOpts);
      CXString Msg = clang_formatDiagnostic(&Diag,
                                      clang_defaultDiagnosticDisplayOptions());
      fprintf(stderr, "%s\n", clang_getCString(Msg));
      clang_disposeString(Msg);
    }
    
#ifdef LLVM_ON_WIN32
    // On Windows, force a flush, since there may be multiple copies of
    // stderr and stdout in the file system, all with different buffers
    // but writing to the same device.
    fflush(stderr);
#endif    
  }

  if (ATU) {
    // Make the translation unit responsible for destroying all temporary files.
    for (unsigned i = 0, e = TemporaryFiles.size(); i != e; ++i)
      ATU->addTemporaryFile(TemporaryFiles[i]);
    ATU->addTemporaryFile(llvm::sys::Path(ATU->getASTFileName()));
  } else {
    // Destroy all of the temporary files now; they can't be referenced any
    // longer.
    llvm::sys::Path(astTmpFile).eraseFromDisk();
    for (unsigned i = 0, e = TemporaryFiles.size(); i != e; ++i)
      TemporaryFiles[i].eraseFromDisk();
  }
  
  PTUI->result = ATU;
}
CXTranslationUnit clang_parseTranslationUnit(CXIndex CIdx,
                                             const char *source_filename,
                                             const char **command_line_args,
                                             int num_command_line_args,
                                             struct CXUnsavedFile *unsaved_files,
                                             unsigned num_unsaved_files,
                                             unsigned options) {
  ParseTranslationUnitInfo PTUI = { CIdx, source_filename, command_line_args,
                                    num_command_line_args, unsaved_files, num_unsaved_files,
                                    options, 0 };
  llvm::CrashRecoveryContext CRC;

  if (!CRC.RunSafely(clang_parseTranslationUnit_Impl, &PTUI)) {
    fprintf(stderr, "libclang: crash detected during parsing: {\n");
    fprintf(stderr, "  'source_filename' : '%s'\n", source_filename);
    fprintf(stderr, "  'command_line_args' : [");
    for (int i = 0; i != num_command_line_args; ++i) {
      if (i)
        fprintf(stderr, ", ");
      fprintf(stderr, "'%s'", command_line_args[i]);
    }
    fprintf(stderr, "],\n");
    fprintf(stderr, "  'unsaved_files' : [");
    for (unsigned i = 0; i != num_unsaved_files; ++i) {
      if (i)
        fprintf(stderr, ", ");
      fprintf(stderr, "('%s', '...', %ld)", unsaved_files[i].Filename,
              unsaved_files[i].Length);
    }
    fprintf(stderr, "],\n");
    fprintf(stderr, "  'options' : %d,\n", options);
    fprintf(stderr, "}\n");
    
    return 0;
  }

  return PTUI.result;
}

unsigned clang_defaultSaveOptions(CXTranslationUnit TU) {
  return CXSaveTranslationUnit_None;
}  
  
int clang_saveTranslationUnit(CXTranslationUnit TU, const char *FileName,
                              unsigned options) {
  if (!TU)
    return 1;
  
  return static_cast<ASTUnit *>(TU)->Save(FileName);
}

void clang_disposeTranslationUnit(CXTranslationUnit CTUnit) {
  if (CTUnit) {
    // If the translation unit has been marked as unsafe to free, just discard
    // it.
    if (static_cast<ASTUnit *>(CTUnit)->isUnsafeToFree())
      return;

    delete static_cast<ASTUnit *>(CTUnit);
  }
}

unsigned clang_defaultReparseOptions(CXTranslationUnit TU) {
  return CXReparse_None;
}

struct ReparseTranslationUnitInfo {
  CXTranslationUnit TU;
  unsigned num_unsaved_files;
  struct CXUnsavedFile *unsaved_files;
  unsigned options;
  int result;
};
static void clang_reparseTranslationUnit_Impl(void *UserData) {
  ReparseTranslationUnitInfo *RTUI =
    static_cast<ReparseTranslationUnitInfo*>(UserData);
  CXTranslationUnit TU = RTUI->TU;
  unsigned num_unsaved_files = RTUI->num_unsaved_files;
  struct CXUnsavedFile *unsaved_files = RTUI->unsaved_files;
  unsigned options = RTUI->options;
  (void) options;
  RTUI->result = 1;

  if (!TU)
    return;
  
  llvm::SmallVector<ASTUnit::RemappedFile, 4> RemappedFiles;
  for (unsigned I = 0; I != num_unsaved_files; ++I) {
    llvm::StringRef Data(unsaved_files[I].Contents, unsaved_files[I].Length);
    const llvm::MemoryBuffer *Buffer
      = llvm::MemoryBuffer::getMemBufferCopy(Data, unsaved_files[I].Filename);
    RemappedFiles.push_back(std::make_pair(unsaved_files[I].Filename,
                                           Buffer));
  }
  
  if (!static_cast<ASTUnit *>(TU)->Reparse(RemappedFiles.data(),
                                           RemappedFiles.size()))
      RTUI->result = 0;
}
int clang_reparseTranslationUnit(CXTranslationUnit TU,
                                 unsigned num_unsaved_files,
                                 struct CXUnsavedFile *unsaved_files,
                                 unsigned options) {
  ReparseTranslationUnitInfo RTUI = { TU, num_unsaved_files, unsaved_files,
                                      options, 0 };
  llvm::CrashRecoveryContext CRC;

  if (!CRC.RunSafely(clang_reparseTranslationUnit_Impl, &RTUI)) {
    fprintf(stderr, "libclang: crash detected during reparsing\n");
    static_cast<ASTUnit *>(TU)->setUnsafeToFree(true);
    return 1;
  }

  return RTUI.result;
}


CXString clang_getTranslationUnitSpelling(CXTranslationUnit CTUnit) {
  if (!CTUnit)
    return createCXString("");

  ASTUnit *CXXUnit = static_cast<ASTUnit *>(CTUnit);
  return createCXString(CXXUnit->getOriginalSourceFileName(), true);
}

CXCursor clang_getTranslationUnitCursor(CXTranslationUnit TU) {
  CXCursor Result = { CXCursor_TranslationUnit, { 0, 0, TU } };
  return Result;
}

} // end: extern "C"

//===----------------------------------------------------------------------===//
// CXSourceLocation and CXSourceRange Operations.
//===----------------------------------------------------------------------===//

extern "C" {
CXSourceLocation clang_getNullLocation() {
  CXSourceLocation Result = { { 0, 0 }, 0 };
  return Result;
}

unsigned clang_equalLocations(CXSourceLocation loc1, CXSourceLocation loc2) {
  return (loc1.ptr_data[0] == loc2.ptr_data[0] &&
          loc1.ptr_data[1] == loc2.ptr_data[1] &&
          loc1.int_data == loc2.int_data);
}

CXSourceLocation clang_getLocation(CXTranslationUnit tu,
                                   CXFile file,
                                   unsigned line,
                                   unsigned column) {
  if (!tu || !file)
    return clang_getNullLocation();
  
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(tu);
  SourceLocation SLoc
    = CXXUnit->getSourceManager().getLocation(
                                        static_cast<const FileEntry *>(file),
                                              line, column);

  return cxloc::translateSourceLocation(CXXUnit->getASTContext(), SLoc);
}

CXSourceRange clang_getNullRange() {
  CXSourceRange Result = { { 0, 0 }, 0, 0 };
  return Result;
}

CXSourceRange clang_getRange(CXSourceLocation begin, CXSourceLocation end) {
  if (begin.ptr_data[0] != end.ptr_data[0] ||
      begin.ptr_data[1] != end.ptr_data[1])
    return clang_getNullRange();

  CXSourceRange Result = { { begin.ptr_data[0], begin.ptr_data[1] },
                           begin.int_data, end.int_data };
  return Result;
}

void clang_getInstantiationLocation(CXSourceLocation location,
                                    CXFile *file,
                                    unsigned *line,
                                    unsigned *column,
                                    unsigned *offset) {
  SourceLocation Loc = SourceLocation::getFromRawEncoding(location.int_data);

  if (!location.ptr_data[0] || Loc.isInvalid()) {
    if (file)
      *file = 0;
    if (line)
      *line = 0;
    if (column)
      *column = 0;
    if (offset)
      *offset = 0;
    return;
  }

  const SourceManager &SM =
    *static_cast<const SourceManager*>(location.ptr_data[0]);
  SourceLocation InstLoc = SM.getInstantiationLoc(Loc);

  if (file)
    *file = (void *)SM.getFileEntryForID(SM.getFileID(InstLoc));
  if (line)
    *line = SM.getInstantiationLineNumber(InstLoc);
  if (column)
    *column = SM.getInstantiationColumnNumber(InstLoc);
  if (offset)
    *offset = SM.getDecomposedLoc(InstLoc).second;
}

CXSourceLocation clang_getRangeStart(CXSourceRange range) {
  CXSourceLocation Result = { { range.ptr_data[0], range.ptr_data[1] },
                              range.begin_int_data };
  return Result;
}

CXSourceLocation clang_getRangeEnd(CXSourceRange range) {
  CXSourceLocation Result = { { range.ptr_data[0], range.ptr_data[1] },
                              range.end_int_data };
  return Result;
}

} // end: extern "C"

//===----------------------------------------------------------------------===//
// CXFile Operations.
//===----------------------------------------------------------------------===//

extern "C" {
CXString clang_getFileName(CXFile SFile) {
  if (!SFile)
    return createCXString(NULL);

  FileEntry *FEnt = static_cast<FileEntry *>(SFile);
  return createCXString(FEnt->getName());
}

time_t clang_getFileTime(CXFile SFile) {
  if (!SFile)
    return 0;

  FileEntry *FEnt = static_cast<FileEntry *>(SFile);
  return FEnt->getModificationTime();
}

CXFile clang_getFile(CXTranslationUnit tu, const char *file_name) {
  if (!tu)
    return 0;

  ASTUnit *CXXUnit = static_cast<ASTUnit *>(tu);

  FileManager &FMgr = CXXUnit->getFileManager();
  const FileEntry *File = FMgr.getFile(file_name, file_name+strlen(file_name));
  return const_cast<FileEntry *>(File);
}

} // end: extern "C"

//===----------------------------------------------------------------------===//
// CXCursor Operations.
//===----------------------------------------------------------------------===//

static Decl *getDeclFromExpr(Stmt *E) {
  if (DeclRefExpr *RefExpr = dyn_cast<DeclRefExpr>(E))
    return RefExpr->getDecl();
  if (MemberExpr *ME = dyn_cast<MemberExpr>(E))
    return ME->getMemberDecl();
  if (ObjCIvarRefExpr *RE = dyn_cast<ObjCIvarRefExpr>(E))
    return RE->getDecl();

  if (CallExpr *CE = dyn_cast<CallExpr>(E))
    return getDeclFromExpr(CE->getCallee());
  if (CastExpr *CE = dyn_cast<CastExpr>(E))
    return getDeclFromExpr(CE->getSubExpr());
  if (ObjCMessageExpr *OME = dyn_cast<ObjCMessageExpr>(E))
    return OME->getMethodDecl();

  return 0;
}

static SourceLocation getLocationFromExpr(Expr *E) {
  if (ObjCMessageExpr *Msg = dyn_cast<ObjCMessageExpr>(E))
    return /*FIXME:*/Msg->getLeftLoc();
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E))
    return DRE->getLocation();
  if (MemberExpr *Member = dyn_cast<MemberExpr>(E))
    return Member->getMemberLoc();
  if (ObjCIvarRefExpr *Ivar = dyn_cast<ObjCIvarRefExpr>(E))
    return Ivar->getLocation();
  return E->getLocStart();
}

extern "C" {

unsigned clang_visitChildren(CXCursor parent,
                             CXCursorVisitor visitor,
                             CXClientData client_data) {
  ASTUnit *CXXUnit = getCursorASTUnit(parent);

  CursorVisitor CursorVis(CXXUnit, visitor, client_data, 
                          CXXUnit->getMaxPCHLevel());
  return CursorVis.VisitChildren(parent);
}

static CXString getDeclSpelling(Decl *D) {
  NamedDecl *ND = dyn_cast_or_null<NamedDecl>(D);
  if (!ND)
    return createCXString("");

  if (ObjCMethodDecl *OMD = dyn_cast<ObjCMethodDecl>(ND))
    return createCXString(OMD->getSelector().getAsString());

  if (ObjCCategoryImplDecl *CIMP = dyn_cast<ObjCCategoryImplDecl>(ND))
    // No, this isn't the same as the code below. getIdentifier() is non-virtual
    // and returns different names. NamedDecl returns the class name and
    // ObjCCategoryImplDecl returns the category name.
    return createCXString(CIMP->getIdentifier()->getNameStart());

  llvm::SmallString<1024> S;
  llvm::raw_svector_ostream os(S);
  ND->printName(os);
  
  return createCXString(os.str());
}

CXString clang_getCursorSpelling(CXCursor C) {
  if (clang_isTranslationUnit(C.kind))
    return clang_getTranslationUnitSpelling(C.data[2]);

  if (clang_isReference(C.kind)) {
    switch (C.kind) {
    case CXCursor_ObjCSuperClassRef: {
      ObjCInterfaceDecl *Super = getCursorObjCSuperClassRef(C).first;
      return createCXString(Super->getIdentifier()->getNameStart());
    }
    case CXCursor_ObjCClassRef: {
      ObjCInterfaceDecl *Class = getCursorObjCClassRef(C).first;
      return createCXString(Class->getIdentifier()->getNameStart());
    }
    case CXCursor_ObjCProtocolRef: {
      ObjCProtocolDecl *OID = getCursorObjCProtocolRef(C).first;
      assert(OID && "getCursorSpelling(): Missing protocol decl");
      return createCXString(OID->getIdentifier()->getNameStart());
    }
    case CXCursor_CXXBaseSpecifier: {
      CXXBaseSpecifier *B = getCursorCXXBaseSpecifier(C);
      return createCXString(B->getType().getAsString());
    }
    case CXCursor_TypeRef: {
      TypeDecl *Type = getCursorTypeRef(C).first;
      assert(Type && "Missing type decl");

      return createCXString(getCursorContext(C).getTypeDeclType(Type).
                              getAsString());
    }
    case CXCursor_TemplateRef: {
      TemplateDecl *Template = getCursorTemplateRef(C).first;
      assert(Template && "Missing type decl");
      
      return createCXString(Template->getNameAsString());
    }

    default:
      return createCXString("<not implemented>");
    }
  }

  if (clang_isExpression(C.kind)) {
    Decl *D = getDeclFromExpr(getCursorExpr(C));
    if (D)
      return getDeclSpelling(D);
    return createCXString("");
  }

  if (C.kind == CXCursor_MacroInstantiation)
    return createCXString(getCursorMacroInstantiation(C)->getName()
                                                           ->getNameStart());

  if (C.kind == CXCursor_MacroDefinition)
    return createCXString(getCursorMacroDefinition(C)->getName()
                                                           ->getNameStart());

  if (clang_isDeclaration(C.kind))
    return getDeclSpelling(getCursorDecl(C));

  return createCXString("");
}

CXString clang_getCursorKindSpelling(enum CXCursorKind Kind) {
  switch (Kind) {
  case CXCursor_FunctionDecl:
      return createCXString("FunctionDecl");
  case CXCursor_TypedefDecl:
      return createCXString("TypedefDecl");
  case CXCursor_EnumDecl:
      return createCXString("EnumDecl");
  case CXCursor_EnumConstantDecl:
      return createCXString("EnumConstantDecl");
  case CXCursor_StructDecl:
      return createCXString("StructDecl");
  case CXCursor_UnionDecl:
      return createCXString("UnionDecl");
  case CXCursor_ClassDecl:
      return createCXString("ClassDecl");
  case CXCursor_FieldDecl:
      return createCXString("FieldDecl");
  case CXCursor_VarDecl:
      return createCXString("VarDecl");
  case CXCursor_ParmDecl:
      return createCXString("ParmDecl");
  case CXCursor_ObjCInterfaceDecl:
      return createCXString("ObjCInterfaceDecl");
  case CXCursor_ObjCCategoryDecl:
      return createCXString("ObjCCategoryDecl");
  case CXCursor_ObjCProtocolDecl:
      return createCXString("ObjCProtocolDecl");
  case CXCursor_ObjCPropertyDecl:
      return createCXString("ObjCPropertyDecl");
  case CXCursor_ObjCIvarDecl:
      return createCXString("ObjCIvarDecl");
  case CXCursor_ObjCInstanceMethodDecl:
      return createCXString("ObjCInstanceMethodDecl");
  case CXCursor_ObjCClassMethodDecl:
      return createCXString("ObjCClassMethodDecl");
  case CXCursor_ObjCImplementationDecl:
      return createCXString("ObjCImplementationDecl");
  case CXCursor_ObjCCategoryImplDecl:
      return createCXString("ObjCCategoryImplDecl");
  case CXCursor_CXXMethod:
      return createCXString("CXXMethod");
  case CXCursor_UnexposedDecl:
      return createCXString("UnexposedDecl");
  case CXCursor_ObjCSuperClassRef:
      return createCXString("ObjCSuperClassRef");
  case CXCursor_ObjCProtocolRef:
      return createCXString("ObjCProtocolRef");
  case CXCursor_ObjCClassRef:
      return createCXString("ObjCClassRef");
  case CXCursor_TypeRef:
      return createCXString("TypeRef");
  case CXCursor_TemplateRef:
      return createCXString("TemplateRef");
  case CXCursor_UnexposedExpr:
      return createCXString("UnexposedExpr");
  case CXCursor_BlockExpr:
      return createCXString("BlockExpr");
  case CXCursor_DeclRefExpr:
      return createCXString("DeclRefExpr");
  case CXCursor_MemberRefExpr:
      return createCXString("MemberRefExpr");
  case CXCursor_CallExpr:
      return createCXString("CallExpr");
  case CXCursor_ObjCMessageExpr:
      return createCXString("ObjCMessageExpr");
  case CXCursor_UnexposedStmt:
      return createCXString("UnexposedStmt");
  case CXCursor_InvalidFile:
      return createCXString("InvalidFile");
  case CXCursor_InvalidCode:
    return createCXString("InvalidCode");
  case CXCursor_NoDeclFound:
      return createCXString("NoDeclFound");
  case CXCursor_NotImplemented:
      return createCXString("NotImplemented");
  case CXCursor_TranslationUnit:
      return createCXString("TranslationUnit");
  case CXCursor_UnexposedAttr:
      return createCXString("UnexposedAttr");
  case CXCursor_IBActionAttr:
      return createCXString("attribute(ibaction)");
  case CXCursor_IBOutletAttr:
     return createCXString("attribute(iboutlet)");
  case CXCursor_IBOutletCollectionAttr:
      return createCXString("attribute(iboutletcollection)");
  case CXCursor_PreprocessingDirective:
    return createCXString("preprocessing directive");
  case CXCursor_MacroDefinition:
    return createCXString("macro definition");
  case CXCursor_MacroInstantiation:
    return createCXString("macro instantiation");
  case CXCursor_Namespace:
    return createCXString("Namespace");
  case CXCursor_LinkageSpec:
    return createCXString("LinkageSpec");
  case CXCursor_CXXBaseSpecifier:
    return createCXString("C++ base class specifier");  
  case CXCursor_Constructor:
    return createCXString("CXXConstructor");
  case CXCursor_Destructor:
    return createCXString("CXXDestructor");
  case CXCursor_ConversionFunction:
    return createCXString("CXXConversion");
  case CXCursor_TemplateTypeParameter:
    return createCXString("TemplateTypeParameter");
  case CXCursor_NonTypeTemplateParameter:
    return createCXString("NonTypeTemplateParameter");
  case CXCursor_TemplateTemplateParameter:
    return createCXString("TemplateTemplateParameter");
  case CXCursor_FunctionTemplate:
    return createCXString("FunctionTemplate");
  case CXCursor_ClassTemplate:
    return createCXString("ClassTemplate");
  case CXCursor_ClassTemplatePartialSpecialization:
    return createCXString("ClassTemplatePartialSpecialization");
  }

  llvm_unreachable("Unhandled CXCursorKind");
  return createCXString(NULL);
}

enum CXChildVisitResult GetCursorVisitor(CXCursor cursor,
                                         CXCursor parent,
                                         CXClientData client_data) {
  CXCursor *BestCursor = static_cast<CXCursor *>(client_data);
  *BestCursor = cursor;
  return CXChildVisit_Recurse;
}

CXCursor clang_getCursor(CXTranslationUnit TU, CXSourceLocation Loc) {
  if (!TU)
    return clang_getNullCursor();

  ASTUnit *CXXUnit = static_cast<ASTUnit *>(TU);
  ASTUnit::ConcurrencyCheck Check(*CXXUnit);

  // Translate the given source location to make it point at the beginning of
  // the token under the cursor.
  SourceLocation SLoc = cxloc::translateSourceLocation(Loc);

  // Guard against an invalid SourceLocation, or we may assert in one
  // of the following calls.
  if (SLoc.isInvalid())
    return clang_getNullCursor();

  SLoc = Lexer::GetBeginningOfToken(SLoc, CXXUnit->getSourceManager(),
                                    CXXUnit->getASTContext().getLangOptions());
  
  CXCursor Result = MakeCXCursorInvalid(CXCursor_NoDeclFound);
  if (SLoc.isValid()) {
    // FIXME: Would be great to have a "hint" cursor, then walk from that
    // hint cursor upward until we find a cursor whose source range encloses
    // the region of interest, rather than starting from the translation unit.
    CXCursor Parent = clang_getTranslationUnitCursor(CXXUnit);
    CursorVisitor CursorVis(CXXUnit, GetCursorVisitor, &Result,
                            Decl::MaxPCHLevel, SourceLocation(SLoc));
    CursorVis.VisitChildren(Parent);
  }
  return Result;
}

CXCursor clang_getNullCursor(void) {
  return MakeCXCursorInvalid(CXCursor_InvalidFile);
}

unsigned clang_equalCursors(CXCursor X, CXCursor Y) {
  return X == Y;
}

unsigned clang_isInvalid(enum CXCursorKind K) {
  return K >= CXCursor_FirstInvalid && K <= CXCursor_LastInvalid;
}

unsigned clang_isDeclaration(enum CXCursorKind K) {
  return K >= CXCursor_FirstDecl && K <= CXCursor_LastDecl;
}

unsigned clang_isReference(enum CXCursorKind K) {
  return K >= CXCursor_FirstRef && K <= CXCursor_LastRef;
}

unsigned clang_isExpression(enum CXCursorKind K) {
  return K >= CXCursor_FirstExpr && K <= CXCursor_LastExpr;
}

unsigned clang_isStatement(enum CXCursorKind K) {
  return K >= CXCursor_FirstStmt && K <= CXCursor_LastStmt;
}

unsigned clang_isTranslationUnit(enum CXCursorKind K) {
  return K == CXCursor_TranslationUnit;
}

unsigned clang_isPreprocessing(enum CXCursorKind K) {
  return K >= CXCursor_FirstPreprocessing && K <= CXCursor_LastPreprocessing;
}
  
unsigned clang_isUnexposed(enum CXCursorKind K) {
  switch (K) {
    case CXCursor_UnexposedDecl:
    case CXCursor_UnexposedExpr:
    case CXCursor_UnexposedStmt:
    case CXCursor_UnexposedAttr:
      return true;
    default:
      return false;
  }
}

CXCursorKind clang_getCursorKind(CXCursor C) {
  return C.kind;
}

CXSourceLocation clang_getCursorLocation(CXCursor C) {
  if (clang_isReference(C.kind)) {
    switch (C.kind) {
    case CXCursor_ObjCSuperClassRef: {
      std::pair<ObjCInterfaceDecl *, SourceLocation> P
        = getCursorObjCSuperClassRef(C);
      return cxloc::translateSourceLocation(P.first->getASTContext(), P.second);
    }

    case CXCursor_ObjCProtocolRef: {
      std::pair<ObjCProtocolDecl *, SourceLocation> P
        = getCursorObjCProtocolRef(C);
      return cxloc::translateSourceLocation(P.first->getASTContext(), P.second);
    }

    case CXCursor_ObjCClassRef: {
      std::pair<ObjCInterfaceDecl *, SourceLocation> P
        = getCursorObjCClassRef(C);
      return cxloc::translateSourceLocation(P.first->getASTContext(), P.second);
    }

    case CXCursor_TypeRef: {
      std::pair<TypeDecl *, SourceLocation> P = getCursorTypeRef(C);
      return cxloc::translateSourceLocation(P.first->getASTContext(), P.second);
    }

    case CXCursor_TemplateRef: {
      std::pair<TemplateDecl *, SourceLocation> P = getCursorTemplateRef(C);
      return cxloc::translateSourceLocation(P.first->getASTContext(), P.second);
    }

    case CXCursor_CXXBaseSpecifier: {
      // FIXME: Figure out what location to return for a CXXBaseSpecifier.
      return clang_getNullLocation();
    }

    default:
      // FIXME: Need a way to enumerate all non-reference cases.
      llvm_unreachable("Missed a reference kind");
    }
  }

  if (clang_isExpression(C.kind))
    return cxloc::translateSourceLocation(getCursorContext(C),
                                   getLocationFromExpr(getCursorExpr(C)));

  if (C.kind == CXCursor_PreprocessingDirective) {
    SourceLocation L = cxcursor::getCursorPreprocessingDirective(C).getBegin();
    return cxloc::translateSourceLocation(getCursorContext(C), L);
  }

  if (C.kind == CXCursor_MacroInstantiation) {
    SourceLocation L
      = cxcursor::getCursorMacroInstantiation(C)->getSourceRange().getBegin();
    return cxloc::translateSourceLocation(getCursorContext(C), L);
  }

  if (C.kind == CXCursor_MacroDefinition) {
    SourceLocation L = cxcursor::getCursorMacroDefinition(C)->getLocation();
    return cxloc::translateSourceLocation(getCursorContext(C), L);
  }
  
  if (C.kind < CXCursor_FirstDecl || C.kind > CXCursor_LastDecl)
    return clang_getNullLocation();

  Decl *D = getCursorDecl(C);
  SourceLocation Loc = D->getLocation();
  if (ObjCInterfaceDecl *Class = dyn_cast<ObjCInterfaceDecl>(D))
    Loc = Class->getClassLoc();
  return cxloc::translateSourceLocation(getCursorContext(C), Loc);
}

} // end extern "C"

static SourceRange getRawCursorExtent(CXCursor C) {
  if (clang_isReference(C.kind)) {
    switch (C.kind) {
    case CXCursor_ObjCSuperClassRef:
      return  getCursorObjCSuperClassRef(C).second;

    case CXCursor_ObjCProtocolRef:
      return getCursorObjCProtocolRef(C).second;

    case CXCursor_ObjCClassRef:
      return getCursorObjCClassRef(C).second;

    case CXCursor_TypeRef:
      return getCursorTypeRef(C).second;

    case CXCursor_TemplateRef:
      return getCursorTemplateRef(C).second;

    case CXCursor_CXXBaseSpecifier:
      // FIXME: Figure out what source range to use for a CXBaseSpecifier.
      return SourceRange();

    default:
      // FIXME: Need a way to enumerate all non-reference cases.
      llvm_unreachable("Missed a reference kind");
    }
  }

  if (clang_isExpression(C.kind))
    return getCursorExpr(C)->getSourceRange();

  if (clang_isStatement(C.kind))
    return getCursorStmt(C)->getSourceRange();

  if (C.kind == CXCursor_PreprocessingDirective)
    return cxcursor::getCursorPreprocessingDirective(C);

  if (C.kind == CXCursor_MacroInstantiation)
    return cxcursor::getCursorMacroInstantiation(C)->getSourceRange();

  if (C.kind == CXCursor_MacroDefinition)
    return cxcursor::getCursorMacroDefinition(C)->getSourceRange();
  
  if (C.kind >= CXCursor_FirstDecl && C.kind <= CXCursor_LastDecl)
    return getCursorDecl(C)->getSourceRange();

  return SourceRange();
}

extern "C" {

CXSourceRange clang_getCursorExtent(CXCursor C) {
  SourceRange R = getRawCursorExtent(C);
  if (R.isInvalid())
    return clang_getNullRange();

  return cxloc::translateSourceRange(getCursorContext(C), R);
}

CXCursor clang_getCursorReferenced(CXCursor C) {
  if (clang_isInvalid(C.kind))
    return clang_getNullCursor();

  ASTUnit *CXXUnit = getCursorASTUnit(C);
  if (clang_isDeclaration(C.kind))
    return C;

  if (clang_isExpression(C.kind)) {
    Decl *D = getDeclFromExpr(getCursorExpr(C));
    if (D)
      return MakeCXCursor(D, CXXUnit);
    return clang_getNullCursor();
  }

  if (C.kind == CXCursor_MacroInstantiation) {
    if (MacroDefinition *Def = getCursorMacroInstantiation(C)->getDefinition())
      return MakeMacroDefinitionCursor(Def, CXXUnit);
  }

  if (!clang_isReference(C.kind))
    return clang_getNullCursor();

  switch (C.kind) {
    case CXCursor_ObjCSuperClassRef:
      return MakeCXCursor(getCursorObjCSuperClassRef(C).first, CXXUnit);

    case CXCursor_ObjCProtocolRef: {
      return MakeCXCursor(getCursorObjCProtocolRef(C).first, CXXUnit);

    case CXCursor_ObjCClassRef:
      return MakeCXCursor(getCursorObjCClassRef(C).first, CXXUnit);

    case CXCursor_TypeRef:
      return MakeCXCursor(getCursorTypeRef(C).first, CXXUnit);

    case CXCursor_TemplateRef:
      return MakeCXCursor(getCursorTemplateRef(C).first, CXXUnit);

    case CXCursor_CXXBaseSpecifier: {
      CXXBaseSpecifier *B = cxcursor::getCursorCXXBaseSpecifier(C);
      return clang_getTypeDeclaration(cxtype::MakeCXType(B->getType(),
                                                         CXXUnit));
    }

    default:
      // We would prefer to enumerate all non-reference cursor kinds here.
      llvm_unreachable("Unhandled reference cursor kind");
      break;
    }
  }

  return clang_getNullCursor();
}

CXCursor clang_getCursorDefinition(CXCursor C) {
  if (clang_isInvalid(C.kind))
    return clang_getNullCursor();

  ASTUnit *CXXUnit = getCursorASTUnit(C);

  bool WasReference = false;
  if (clang_isReference(C.kind) || clang_isExpression(C.kind)) {
    C = clang_getCursorReferenced(C);
    WasReference = true;
  }

  if (C.kind == CXCursor_MacroInstantiation)
    return clang_getCursorReferenced(C);

  if (!clang_isDeclaration(C.kind))
    return clang_getNullCursor();

  Decl *D = getCursorDecl(C);
  if (!D)
    return clang_getNullCursor();

  switch (D->getKind()) {
  // Declaration kinds that don't really separate the notions of
  // declaration and definition.
  case Decl::Namespace:
  case Decl::Typedef:
  case Decl::TemplateTypeParm:
  case Decl::EnumConstant:
  case Decl::Field:
  case Decl::ObjCIvar:
  case Decl::ObjCAtDefsField:
  case Decl::ImplicitParam:
  case Decl::ParmVar:
  case Decl::NonTypeTemplateParm:
  case Decl::TemplateTemplateParm:
  case Decl::ObjCCategoryImpl:
  case Decl::ObjCImplementation:
  case Decl::AccessSpec:
  case Decl::LinkageSpec:
  case Decl::ObjCPropertyImpl:
  case Decl::FileScopeAsm:
  case Decl::StaticAssert:
  case Decl::Block:
    return C;

  // Declaration kinds that don't make any sense here, but are
  // nonetheless harmless.
  case Decl::TranslationUnit:
    break;

  // Declaration kinds for which the definition is not resolvable.
  case Decl::UnresolvedUsingTypename:
  case Decl::UnresolvedUsingValue:
    break;

  case Decl::UsingDirective:
    return MakeCXCursor(cast<UsingDirectiveDecl>(D)->getNominatedNamespace(),
                        CXXUnit);

  case Decl::NamespaceAlias:
    return MakeCXCursor(cast<NamespaceAliasDecl>(D)->getNamespace(), CXXUnit);

  case Decl::Enum:
  case Decl::Record:
  case Decl::CXXRecord:
  case Decl::ClassTemplateSpecialization:
  case Decl::ClassTemplatePartialSpecialization:
    if (TagDecl *Def = cast<TagDecl>(D)->getDefinition())
      return MakeCXCursor(Def, CXXUnit);
    return clang_getNullCursor();

  case Decl::Function:
  case Decl::CXXMethod:
  case Decl::CXXConstructor:
  case Decl::CXXDestructor:
  case Decl::CXXConversion: {
    const FunctionDecl *Def = 0;
    if (cast<FunctionDecl>(D)->getBody(Def))
      return MakeCXCursor(const_cast<FunctionDecl *>(Def), CXXUnit);
    return clang_getNullCursor();
  }

  case Decl::Var: {
    // Ask the variable if it has a definition.
    if (VarDecl *Def = cast<VarDecl>(D)->getDefinition())
      return MakeCXCursor(Def, CXXUnit);
    return clang_getNullCursor();
  }

  case Decl::FunctionTemplate: {
    const FunctionDecl *Def = 0;
    if (cast<FunctionTemplateDecl>(D)->getTemplatedDecl()->getBody(Def))
      return MakeCXCursor(Def->getDescribedFunctionTemplate(), CXXUnit);
    return clang_getNullCursor();
  }

  case Decl::ClassTemplate: {
    if (RecordDecl *Def = cast<ClassTemplateDecl>(D)->getTemplatedDecl()
                                                            ->getDefinition())
      return MakeCXCursor(cast<CXXRecordDecl>(Def)->getDescribedClassTemplate(),
                          CXXUnit);
    return clang_getNullCursor();
  }

  case Decl::Using: {
    UsingDecl *Using = cast<UsingDecl>(D);
    CXCursor Def = clang_getNullCursor();
    for (UsingDecl::shadow_iterator S = Using->shadow_begin(),
                                 SEnd = Using->shadow_end();
         S != SEnd; ++S) {
      if (Def != clang_getNullCursor()) {
        // FIXME: We have no way to return multiple results.
        return clang_getNullCursor();
      }

      Def = clang_getCursorDefinition(MakeCXCursor((*S)->getTargetDecl(),
                                                   CXXUnit));
    }

    return Def;
  }

  case Decl::UsingShadow:
    return clang_getCursorDefinition(
                       MakeCXCursor(cast<UsingShadowDecl>(D)->getTargetDecl(),
                                    CXXUnit));

  case Decl::ObjCMethod: {
    ObjCMethodDecl *Method = cast<ObjCMethodDecl>(D);
    if (Method->isThisDeclarationADefinition())
      return C;

    // Dig out the method definition in the associated
    // @implementation, if we have it.
    // FIXME: The ASTs should make finding the definition easier.
    if (ObjCInterfaceDecl *Class
                       = dyn_cast<ObjCInterfaceDecl>(Method->getDeclContext()))
      if (ObjCImplementationDecl *ClassImpl = Class->getImplementation())
        if (ObjCMethodDecl *Def = ClassImpl->getMethod(Method->getSelector(),
                                                  Method->isInstanceMethod()))
          if (Def->isThisDeclarationADefinition())
            return MakeCXCursor(Def, CXXUnit);

    return clang_getNullCursor();
  }

  case Decl::ObjCCategory:
    if (ObjCCategoryImplDecl *Impl
                               = cast<ObjCCategoryDecl>(D)->getImplementation())
      return MakeCXCursor(Impl, CXXUnit);
    return clang_getNullCursor();

  case Decl::ObjCProtocol:
    if (!cast<ObjCProtocolDecl>(D)->isForwardDecl())
      return C;
    return clang_getNullCursor();

  case Decl::ObjCInterface:
    // There are two notions of a "definition" for an Objective-C
    // class: the interface and its implementation. When we resolved a
    // reference to an Objective-C class, produce the @interface as
    // the definition; when we were provided with the interface,
    // produce the @implementation as the definition.
    if (WasReference) {
      if (!cast<ObjCInterfaceDecl>(D)->isForwardDecl())
        return C;
    } else if (ObjCImplementationDecl *Impl
                              = cast<ObjCInterfaceDecl>(D)->getImplementation())
      return MakeCXCursor(Impl, CXXUnit);
    return clang_getNullCursor();

  case Decl::ObjCProperty:
    // FIXME: We don't really know where to find the
    // ObjCPropertyImplDecls that implement this property.
    return clang_getNullCursor();

  case Decl::ObjCCompatibleAlias:
    if (ObjCInterfaceDecl *Class
          = cast<ObjCCompatibleAliasDecl>(D)->getClassInterface())
      if (!Class->isForwardDecl())
        return MakeCXCursor(Class, CXXUnit);

    return clang_getNullCursor();

  case Decl::ObjCForwardProtocol: {
    ObjCForwardProtocolDecl *Forward = cast<ObjCForwardProtocolDecl>(D);
    if (Forward->protocol_size() == 1)
      return clang_getCursorDefinition(
                                     MakeCXCursor(*Forward->protocol_begin(),
                                                  CXXUnit));

    // FIXME: Cannot return multiple definitions.
    return clang_getNullCursor();
  }

  case Decl::ObjCClass: {
    ObjCClassDecl *Class = cast<ObjCClassDecl>(D);
    if (Class->size() == 1) {
      ObjCInterfaceDecl *IFace = Class->begin()->getInterface();
      if (!IFace->isForwardDecl())
        return MakeCXCursor(IFace, CXXUnit);
      return clang_getNullCursor();
    }

    // FIXME: Cannot return multiple definitions.
    return clang_getNullCursor();
  }

  case Decl::Friend:
    if (NamedDecl *Friend = cast<FriendDecl>(D)->getFriendDecl())
      return clang_getCursorDefinition(MakeCXCursor(Friend, CXXUnit));
    return clang_getNullCursor();

  case Decl::FriendTemplate:
    if (NamedDecl *Friend = cast<FriendTemplateDecl>(D)->getFriendDecl())
      return clang_getCursorDefinition(MakeCXCursor(Friend, CXXUnit));
    return clang_getNullCursor();
  }

  return clang_getNullCursor();
}

unsigned clang_isCursorDefinition(CXCursor C) {
  if (!clang_isDeclaration(C.kind))
    return 0;

  return clang_getCursorDefinition(C) == C;
}

void clang_getDefinitionSpellingAndExtent(CXCursor C,
                                          const char **startBuf,
                                          const char **endBuf,
                                          unsigned *startLine,
                                          unsigned *startColumn,
                                          unsigned *endLine,
                                          unsigned *endColumn) {
  assert(getCursorDecl(C) && "CXCursor has null decl");
  NamedDecl *ND = static_cast<NamedDecl *>(getCursorDecl(C));
  FunctionDecl *FD = dyn_cast<FunctionDecl>(ND);
  CompoundStmt *Body = dyn_cast<CompoundStmt>(FD->getBody());

  SourceManager &SM = FD->getASTContext().getSourceManager();
  *startBuf = SM.getCharacterData(Body->getLBracLoc());
  *endBuf = SM.getCharacterData(Body->getRBracLoc());
  *startLine = SM.getSpellingLineNumber(Body->getLBracLoc());
  *startColumn = SM.getSpellingColumnNumber(Body->getLBracLoc());
  *endLine = SM.getSpellingLineNumber(Body->getRBracLoc());
  *endColumn = SM.getSpellingColumnNumber(Body->getRBracLoc());
}

void clang_enableStackTraces(void) {
  llvm::sys::PrintStackTraceOnErrorSignal();
}

} // end: extern "C"

//===----------------------------------------------------------------------===//
// Token-based Operations.
//===----------------------------------------------------------------------===//

/* CXToken layout:
 *   int_data[0]: a CXTokenKind
 *   int_data[1]: starting token location
 *   int_data[2]: token length
 *   int_data[3]: reserved
 *   ptr_data: for identifiers and keywords, an IdentifierInfo*.
 *   otherwise unused.
 */
extern "C" {

CXTokenKind clang_getTokenKind(CXToken CXTok) {
  return static_cast<CXTokenKind>(CXTok.int_data[0]);
}

CXString clang_getTokenSpelling(CXTranslationUnit TU, CXToken CXTok) {
  switch (clang_getTokenKind(CXTok)) {
  case CXToken_Identifier:
  case CXToken_Keyword:
    // We know we have an IdentifierInfo*, so use that.
    return createCXString(static_cast<IdentifierInfo *>(CXTok.ptr_data)
                            ->getNameStart());

  case CXToken_Literal: {
    // We have stashed the starting pointer in the ptr_data field. Use it.
    const char *Text = static_cast<const char *>(CXTok.ptr_data);
    return createCXString(llvm::StringRef(Text, CXTok.int_data[2]));
  }

  case CXToken_Punctuation:
  case CXToken_Comment:
    break;
  }

  // We have to find the starting buffer pointer the hard way, by
  // deconstructing the source location.
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(TU);
  if (!CXXUnit)
    return createCXString("");

  SourceLocation Loc = SourceLocation::getFromRawEncoding(CXTok.int_data[1]);
  std::pair<FileID, unsigned> LocInfo
    = CXXUnit->getSourceManager().getDecomposedLoc(Loc);
  bool Invalid = false;
  llvm::StringRef Buffer
    = CXXUnit->getSourceManager().getBufferData(LocInfo.first, &Invalid);
  if (Invalid)
    return createCXString("");

  return createCXString(Buffer.substr(LocInfo.second, CXTok.int_data[2]));
}

CXSourceLocation clang_getTokenLocation(CXTranslationUnit TU, CXToken CXTok) {
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(TU);
  if (!CXXUnit)
    return clang_getNullLocation();

  return cxloc::translateSourceLocation(CXXUnit->getASTContext(),
                        SourceLocation::getFromRawEncoding(CXTok.int_data[1]));
}

CXSourceRange clang_getTokenExtent(CXTranslationUnit TU, CXToken CXTok) {
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(TU);
  if (!CXXUnit)
    return clang_getNullRange();

  return cxloc::translateSourceRange(CXXUnit->getASTContext(),
                        SourceLocation::getFromRawEncoding(CXTok.int_data[1]));
}

void clang_tokenize(CXTranslationUnit TU, CXSourceRange Range,
                    CXToken **Tokens, unsigned *NumTokens) {
  if (Tokens)
    *Tokens = 0;
  if (NumTokens)
    *NumTokens = 0;

  ASTUnit *CXXUnit = static_cast<ASTUnit *>(TU);
  if (!CXXUnit || !Tokens || !NumTokens)
    return;

  ASTUnit::ConcurrencyCheck Check(*CXXUnit);
  
  SourceRange R = cxloc::translateCXSourceRange(Range);
  if (R.isInvalid())
    return;

  SourceManager &SourceMgr = CXXUnit->getSourceManager();
  std::pair<FileID, unsigned> BeginLocInfo
    = SourceMgr.getDecomposedLoc(R.getBegin());
  std::pair<FileID, unsigned> EndLocInfo
    = SourceMgr.getDecomposedLoc(R.getEnd());

  // Cannot tokenize across files.
  if (BeginLocInfo.first != EndLocInfo.first)
    return;

  // Create a lexer
  bool Invalid = false;
  llvm::StringRef Buffer
    = SourceMgr.getBufferData(BeginLocInfo.first, &Invalid);
  if (Invalid)
    return;
  
  Lexer Lex(SourceMgr.getLocForStartOfFile(BeginLocInfo.first),
            CXXUnit->getASTContext().getLangOptions(),
            Buffer.begin(), Buffer.data() + BeginLocInfo.second, Buffer.end());
  Lex.SetCommentRetentionState(true);

  // Lex tokens until we hit the end of the range.
  const char *EffectiveBufferEnd = Buffer.data() + EndLocInfo.second;
  llvm::SmallVector<CXToken, 32> CXTokens;
  Token Tok;
  do {
    // Lex the next token
    Lex.LexFromRawLexer(Tok);
    if (Tok.is(tok::eof))
      break;

    // Initialize the CXToken.
    CXToken CXTok;

    //   - Common fields
    CXTok.int_data[1] = Tok.getLocation().getRawEncoding();
    CXTok.int_data[2] = Tok.getLength();
    CXTok.int_data[3] = 0;

    //   - Kind-specific fields
    if (Tok.isLiteral()) {
      CXTok.int_data[0] = CXToken_Literal;
      CXTok.ptr_data = (void *)Tok.getLiteralData();
    } else if (Tok.is(tok::identifier)) {
      // Lookup the identifier to determine whether we have a keyword.
      std::pair<FileID, unsigned> LocInfo
        = SourceMgr.getDecomposedLoc(Tok.getLocation());
      bool Invalid = false;
      llvm::StringRef Buf
        = CXXUnit->getSourceManager().getBufferData(LocInfo.first, &Invalid);
      if (Invalid)
        return;
      
      const char *StartPos = Buf.data() + LocInfo.second;
      IdentifierInfo *II
        = CXXUnit->getPreprocessor().LookUpIdentifierInfo(Tok, StartPos);

      if (II->getObjCKeywordID() != tok::objc_not_keyword) {
        CXTok.int_data[0] = CXToken_Keyword;
      }
      else {
        CXTok.int_data[0] = II->getTokenID() == tok::identifier?
                                CXToken_Identifier
                              : CXToken_Keyword;
      }
      CXTok.ptr_data = II;
    } else if (Tok.is(tok::comment)) {
      CXTok.int_data[0] = CXToken_Comment;
      CXTok.ptr_data = 0;
    } else {
      CXTok.int_data[0] = CXToken_Punctuation;
      CXTok.ptr_data = 0;
    }
    CXTokens.push_back(CXTok);
  } while (Lex.getBufferLocation() <= EffectiveBufferEnd);

  if (CXTokens.empty())
    return;

  *Tokens = (CXToken *)malloc(sizeof(CXToken) * CXTokens.size());
  memmove(*Tokens, CXTokens.data(), sizeof(CXToken) * CXTokens.size());
  *NumTokens = CXTokens.size();
}

void clang_disposeTokens(CXTranslationUnit TU,
                         CXToken *Tokens, unsigned NumTokens) {
  free(Tokens);
}

} // end: extern "C"

//===----------------------------------------------------------------------===//
// Token annotation APIs.
//===----------------------------------------------------------------------===//

typedef llvm::DenseMap<unsigned, CXCursor> AnnotateTokensData;
static enum CXChildVisitResult AnnotateTokensVisitor(CXCursor cursor,
                                                     CXCursor parent,
                                                     CXClientData client_data);
namespace {
class AnnotateTokensWorker {
  AnnotateTokensData &Annotated;
  CXToken *Tokens;
  CXCursor *Cursors;
  unsigned NumTokens;
  unsigned TokIdx;
  CursorVisitor AnnotateVis;
  SourceManager &SrcMgr;

  bool MoreTokens() const { return TokIdx < NumTokens; }
  unsigned NextToken() const { return TokIdx; }
  void AdvanceToken() { ++TokIdx; }
  SourceLocation GetTokenLoc(unsigned tokI) {
    return SourceLocation::getFromRawEncoding(Tokens[tokI].int_data[1]);
  }

public:
  AnnotateTokensWorker(AnnotateTokensData &annotated,
                       CXToken *tokens, CXCursor *cursors, unsigned numTokens,
                       ASTUnit *CXXUnit, SourceRange RegionOfInterest)
    : Annotated(annotated), Tokens(tokens), Cursors(cursors),
      NumTokens(numTokens), TokIdx(0),
      AnnotateVis(CXXUnit, AnnotateTokensVisitor, this,
                  Decl::MaxPCHLevel, RegionOfInterest),
      SrcMgr(CXXUnit->getSourceManager()) {}

  void VisitChildren(CXCursor C) { AnnotateVis.VisitChildren(C); }
  enum CXChildVisitResult Visit(CXCursor cursor, CXCursor parent);
  void AnnotateTokens(CXCursor parent);
};
}

void AnnotateTokensWorker::AnnotateTokens(CXCursor parent) {
  // Walk the AST within the region of interest, annotating tokens
  // along the way.
  VisitChildren(parent);

  for (unsigned I = 0 ; I < TokIdx ; ++I) {
    AnnotateTokensData::iterator Pos = Annotated.find(Tokens[I].int_data[1]);
    if (Pos != Annotated.end())
      Cursors[I] = Pos->second;
  }

  // Finish up annotating any tokens left.
  if (!MoreTokens())
    return;

  const CXCursor &C = clang_getNullCursor();
  for (unsigned I = TokIdx ; I < NumTokens ; ++I) {
    AnnotateTokensData::iterator Pos = Annotated.find(Tokens[I].int_data[1]);
    Cursors[I] = (Pos == Annotated.end()) ? C : Pos->second;
  }
}

enum CXChildVisitResult
AnnotateTokensWorker::Visit(CXCursor cursor, CXCursor parent) {
  CXSourceLocation Loc = clang_getCursorLocation(cursor);
  // We can always annotate a preprocessing directive/macro instantiation.
  if (clang_isPreprocessing(cursor.kind)) {
    Annotated[Loc.int_data] = cursor;
    return CXChildVisit_Recurse;
  }

  SourceRange cursorRange = getRawCursorExtent(cursor);
  
  if (cursorRange.isInvalid())
    return CXChildVisit_Continue;
  
  SourceLocation L = SourceLocation::getFromRawEncoding(Loc.int_data);

  // Adjust the annotated range based specific declarations.
  const enum CXCursorKind cursorK = clang_getCursorKind(cursor);
  if (cursorK >= CXCursor_FirstDecl && cursorK <= CXCursor_LastDecl) {
    Decl *D = cxcursor::getCursorDecl(cursor);
    // Don't visit synthesized ObjC methods, since they have no syntatic
    // representation in the source.
    if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
      if (MD->isSynthesized())
        return CXChildVisit_Continue;
    }
    if (const DeclaratorDecl *DD = dyn_cast<DeclaratorDecl>(D)) {
      if (TypeSourceInfo *TI = DD->getTypeSourceInfo()) {
        TypeLoc TL = TI->getTypeLoc();
        SourceLocation TLoc = TL.getSourceRange().getBegin();
        if (TLoc.isValid() && 
            SrcMgr.isBeforeInTranslationUnit(TLoc, L))
          cursorRange.setBegin(TLoc);
      }
    }
  }

  // If the location of the cursor occurs within a macro instantiation, record
  // the spelling location of the cursor in our annotation map.  We can then
  // paper over the token labelings during a post-processing step to try and
  // get cursor mappings for tokens that are the *arguments* of a macro
  // instantiation.
  if (L.isMacroID()) {
    unsigned rawEncoding = SrcMgr.getSpellingLoc(L).getRawEncoding();
    // Only invalidate the old annotation if it isn't part of a preprocessing
    // directive.  Here we assume that the default construction of CXCursor
    // results in CXCursor.kind being an initialized value (i.e., 0).  If
    // this isn't the case, we can fix by doing lookup + insertion.

    CXCursor &oldC = Annotated[rawEncoding];
    if (!clang_isPreprocessing(oldC.kind))
      oldC = cursor;
  }
  
  const enum CXCursorKind K = clang_getCursorKind(parent);
  const CXCursor updateC =
    (clang_isInvalid(K) || K == CXCursor_TranslationUnit)
     ? clang_getNullCursor() : parent;

  while (MoreTokens()) {
    const unsigned I = NextToken();
    SourceLocation TokLoc = GetTokenLoc(I);
    switch (LocationCompare(SrcMgr, TokLoc, cursorRange)) {
      case RangeBefore:
        Cursors[I] = updateC;
        AdvanceToken();
        continue;
      case RangeAfter:
      case RangeOverlap:
        break;
    }
    break;
  }

  // Visit children to get their cursor information.
  const unsigned BeforeChildren = NextToken();
  VisitChildren(cursor);
  const unsigned AfterChildren = NextToken();

  // Adjust 'Last' to the last token within the extent of the cursor.
  while (MoreTokens()) {
    const unsigned I = NextToken();
    SourceLocation TokLoc = GetTokenLoc(I);
    switch (LocationCompare(SrcMgr, TokLoc, cursorRange)) {
      case RangeBefore:
        assert(0 && "Infeasible");
      case RangeAfter:
        break;
      case RangeOverlap:
        Cursors[I] = updateC;
        AdvanceToken();
        continue;
    }
    break;
  }
  const unsigned Last = NextToken();
  
  // Scan the tokens that are at the beginning of the cursor, but are not
  // capture by the child cursors.

  // For AST elements within macros, rely on a post-annotate pass to
  // to correctly annotate the tokens with cursors.  Otherwise we can
  // get confusing results of having tokens that map to cursors that really
  // are expanded by an instantiation.
  if (L.isMacroID())
    cursor = clang_getNullCursor();

  for (unsigned I = BeforeChildren; I != AfterChildren; ++I) {
    if (!clang_isInvalid(clang_getCursorKind(Cursors[I])))
      break;
    Cursors[I] = cursor;
  }
  // Scan the tokens that are at the end of the cursor, but are not captured
  // but the child cursors.
  for (unsigned I = AfterChildren; I != Last; ++I)
    Cursors[I] = cursor;

  TokIdx = Last;
  return CXChildVisit_Continue;
}

static enum CXChildVisitResult AnnotateTokensVisitor(CXCursor cursor,
                                                     CXCursor parent,
                                                     CXClientData client_data) {
  return static_cast<AnnotateTokensWorker*>(client_data)->Visit(cursor, parent);
}

extern "C" {

void clang_annotateTokens(CXTranslationUnit TU,
                          CXToken *Tokens, unsigned NumTokens,
                          CXCursor *Cursors) {

  if (NumTokens == 0 || !Tokens || !Cursors)
    return;

  ASTUnit *CXXUnit = static_cast<ASTUnit *>(TU);
  if (!CXXUnit) {
    // Any token we don't specifically annotate will have a NULL cursor.
    const CXCursor &C = clang_getNullCursor();
    for (unsigned I = 0; I != NumTokens; ++I)
      Cursors[I] = C;
    return;
  }

  ASTUnit::ConcurrencyCheck Check(*CXXUnit);

  // Determine the region of interest, which contains all of the tokens.
  SourceRange RegionOfInterest;
  RegionOfInterest.setBegin(cxloc::translateSourceLocation(
                                        clang_getTokenLocation(TU, Tokens[0])));
  RegionOfInterest.setEnd(cxloc::translateSourceLocation(
                                clang_getTokenLocation(TU, 
                                                       Tokens[NumTokens - 1])));

  // A mapping from the source locations found when re-lexing or traversing the
  // region of interest to the corresponding cursors.
  AnnotateTokensData Annotated;

  // Relex the tokens within the source range to look for preprocessing
  // directives.
  SourceManager &SourceMgr = CXXUnit->getSourceManager();
  std::pair<FileID, unsigned> BeginLocInfo
    = SourceMgr.getDecomposedLoc(RegionOfInterest.getBegin());
  std::pair<FileID, unsigned> EndLocInfo
    = SourceMgr.getDecomposedLoc(RegionOfInterest.getEnd());

  llvm::StringRef Buffer;
  bool Invalid = false;
  if (BeginLocInfo.first == EndLocInfo.first &&
      ((Buffer = SourceMgr.getBufferData(BeginLocInfo.first, &Invalid)),true) &&
      !Invalid) {
    Lexer Lex(SourceMgr.getLocForStartOfFile(BeginLocInfo.first),
              CXXUnit->getASTContext().getLangOptions(),
              Buffer.begin(), Buffer.data() + BeginLocInfo.second,
              Buffer.end());
    Lex.SetCommentRetentionState(true);

    // Lex tokens in raw mode until we hit the end of the range, to avoid
    // entering #includes or expanding macros.
    while (true) {
      Token Tok;
      Lex.LexFromRawLexer(Tok);

    reprocess:
      if (Tok.is(tok::hash) && Tok.isAtStartOfLine()) {
        // We have found a preprocessing directive. Gobble it up so that we
        // don't see it while preprocessing these tokens later, but keep track of
        // all of the token locations inside this preprocessing directive so that
        // we can annotate them appropriately.
        //
        // FIXME: Some simple tests here could identify macro definitions and
        // #undefs, to provide specific cursor kinds for those.
        std::vector<SourceLocation> Locations;
        do {
          Locations.push_back(Tok.getLocation());
          Lex.LexFromRawLexer(Tok);
        } while (!Tok.isAtStartOfLine() && !Tok.is(tok::eof));

        using namespace cxcursor;
        CXCursor Cursor
          = MakePreprocessingDirectiveCursor(SourceRange(Locations.front(),
                                                         Locations.back()),
                                           CXXUnit);
        for (unsigned I = 0, N = Locations.size(); I != N; ++I) {
          Annotated[Locations[I].getRawEncoding()] = Cursor;
        }

        if (Tok.isAtStartOfLine())
          goto reprocess;

        continue;
      }

      if (Tok.is(tok::eof))
        break;
    }
  }

  // Annotate all of the source locations in the region of interest that map to
  // a specific cursor.
  AnnotateTokensWorker W(Annotated, Tokens, Cursors, NumTokens,
                         CXXUnit, RegionOfInterest);
  W.AnnotateTokens(clang_getTranslationUnitCursor(CXXUnit));
}
} // end: extern "C"

//===----------------------------------------------------------------------===//
// Operations for querying linkage of a cursor.
//===----------------------------------------------------------------------===//

extern "C" {
CXLinkageKind clang_getCursorLinkage(CXCursor cursor) {
  if (!clang_isDeclaration(cursor.kind))
    return CXLinkage_Invalid;

  Decl *D = cxcursor::getCursorDecl(cursor);
  if (NamedDecl *ND = dyn_cast_or_null<NamedDecl>(D))
    switch (ND->getLinkage()) {
      case NoLinkage: return CXLinkage_NoLinkage;
      case InternalLinkage: return CXLinkage_Internal;
      case UniqueExternalLinkage: return CXLinkage_UniqueExternal;
      case ExternalLinkage: return CXLinkage_External;
    };

  return CXLinkage_Invalid;
}
} // end: extern "C"

//===----------------------------------------------------------------------===//
// Operations for querying language of a cursor.
//===----------------------------------------------------------------------===//

static CXLanguageKind getDeclLanguage(const Decl *D) {
  switch (D->getKind()) {
    default:
      break;
    case Decl::ImplicitParam:
    case Decl::ObjCAtDefsField:
    case Decl::ObjCCategory:
    case Decl::ObjCCategoryImpl:
    case Decl::ObjCClass:
    case Decl::ObjCCompatibleAlias:
    case Decl::ObjCForwardProtocol:
    case Decl::ObjCImplementation:
    case Decl::ObjCInterface:
    case Decl::ObjCIvar:
    case Decl::ObjCMethod:
    case Decl::ObjCProperty:
    case Decl::ObjCPropertyImpl:
    case Decl::ObjCProtocol:
      return CXLanguage_ObjC;
    case Decl::CXXConstructor:
    case Decl::CXXConversion:
    case Decl::CXXDestructor:
    case Decl::CXXMethod:
    case Decl::CXXRecord:
    case Decl::ClassTemplate:
    case Decl::ClassTemplatePartialSpecialization:
    case Decl::ClassTemplateSpecialization:
    case Decl::Friend:
    case Decl::FriendTemplate:
    case Decl::FunctionTemplate:
    case Decl::LinkageSpec:
    case Decl::Namespace:
    case Decl::NamespaceAlias:
    case Decl::NonTypeTemplateParm:
    case Decl::StaticAssert:
    case Decl::TemplateTemplateParm:
    case Decl::TemplateTypeParm:
    case Decl::UnresolvedUsingTypename:
    case Decl::UnresolvedUsingValue:
    case Decl::Using:
    case Decl::UsingDirective:
    case Decl::UsingShadow:
      return CXLanguage_CPlusPlus;
  }

  return CXLanguage_C;
}

extern "C" {
  
enum CXAvailabilityKind clang_getCursorAvailability(CXCursor cursor) {
  if (clang_isDeclaration(cursor.kind))
    if (Decl *D = cxcursor::getCursorDecl(cursor)) {
      if (D->hasAttr<UnavailableAttr>() ||
          (isa<FunctionDecl>(D) && cast<FunctionDecl>(D)->isDeleted()))
        return CXAvailability_Available;
      
      if (D->hasAttr<DeprecatedAttr>())
        return CXAvailability_Deprecated;
    }
  
  return CXAvailability_Available;
}

CXLanguageKind clang_getCursorLanguage(CXCursor cursor) {
  if (clang_isDeclaration(cursor.kind))
    return getDeclLanguage(cxcursor::getCursorDecl(cursor));

  return CXLanguage_Invalid;
}
} // end: extern "C"


//===----------------------------------------------------------------------===//
// C++ AST instrospection.
//===----------------------------------------------------------------------===//

extern "C" {
unsigned clang_CXXMethod_isStatic(CXCursor C) {
  if (!clang_isDeclaration(C.kind))
    return 0;
  
  CXXMethodDecl *Method = 0;
  Decl *D = cxcursor::getCursorDecl(C);
  if (FunctionTemplateDecl *FunTmpl = dyn_cast_or_null<FunctionTemplateDecl>(D))
    Method = dyn_cast<CXXMethodDecl>(FunTmpl->getTemplatedDecl());
  else
    Method = dyn_cast_or_null<CXXMethodDecl>(D);
  return (Method && Method->isStatic()) ? 1 : 0;
}

} // end: extern "C"

//===----------------------------------------------------------------------===//
// Attribute introspection.
//===----------------------------------------------------------------------===//

extern "C" {
CXType clang_getIBOutletCollectionType(CXCursor C) {
  if (C.kind != CXCursor_IBOutletCollectionAttr)
    return cxtype::MakeCXType(QualType(), cxcursor::getCursorASTUnit(C));
  
  IBOutletCollectionAttr *A =
    cast<IBOutletCollectionAttr>(cxcursor::getCursorAttr(C));
  
  return cxtype::MakeCXType(A->getInterface(), cxcursor::getCursorASTUnit(C));  
}
} // end: extern "C"

//===----------------------------------------------------------------------===//
// CXString Operations.
//===----------------------------------------------------------------------===//

extern "C" {
const char *clang_getCString(CXString string) {
  return string.Spelling;
}

void clang_disposeString(CXString string) {
  if (string.MustFreeString && string.Spelling)
    free((void*)string.Spelling);
}

} // end: extern "C"

namespace clang { namespace cxstring {
CXString createCXString(const char *String, bool DupString){
  CXString Str;
  if (DupString) {
    Str.Spelling = strdup(String);
    Str.MustFreeString = 1;
  } else {
    Str.Spelling = String;
    Str.MustFreeString = 0;
  }
  return Str;
}

CXString createCXString(llvm::StringRef String, bool DupString) {
  CXString Result;
  if (DupString || (!String.empty() && String.data()[String.size()] != 0)) {
    char *Spelling = (char *)malloc(String.size() + 1);
    memmove(Spelling, String.data(), String.size());
    Spelling[String.size()] = 0;
    Result.Spelling = Spelling;
    Result.MustFreeString = 1;
  } else {
    Result.Spelling = String.data();
    Result.MustFreeString = 0;
  }
  return Result;
}
}}

//===----------------------------------------------------------------------===//
// Misc. utility functions.
//===----------------------------------------------------------------------===//

extern "C" {

CXString clang_getClangVersion() {
  return createCXString(getClangFullVersion());
}

} // end: extern "C"
