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

#include "clang/Basic/Version.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TypeLocVisitor.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/Program.h"

// Needed to define L_TMPNAM on some systems.
#include <cstdio>

using namespace clang;
using namespace clang::cxcursor;
using namespace idx;

//===----------------------------------------------------------------------===//
// Crash Reporting.
//===----------------------------------------------------------------------===//

#ifdef __APPLE__
#ifndef NDEBUG
#define USE_CRASHTRACER
#include "clang/Analysis/Support/SaveAndRestore.h"
// Integrate with crash reporter.
extern "C" const char *__crashreporter_info__;
#define NUM_CRASH_STRINGS 16
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
      Out << "ClangCIndex [createTranslationUnitFromSourceFile]: clang";
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
#endif

typedef llvm::PointerIntPair<ASTContext *, 1, bool> CXSourceLocationPtr;

/// \brief Translate a Clang source location into a CIndex source location.
static CXSourceLocation translateSourceLocation(ASTContext &Context,
                                                SourceLocation Loc,
                                                bool AtEnd = false) {
  CXSourceLocationPtr Ptr(&Context, AtEnd);
  CXSourceLocation Result = { Ptr.getOpaqueValue(), Loc.getRawEncoding() };
  return Result;
}

/// \brief Translate a Clang source range into a CIndex source range.
static CXSourceRange translateSourceRange(ASTContext &Context, SourceRange R) {
  CXSourceRange Result = { &Context, 
                           R.getBegin().getRawEncoding(),
                           R.getEnd().getRawEncoding() };
  return Result;
}

static SourceLocation translateSourceLocation(CXSourceLocation L) {
  return SourceLocation::getFromRawEncoding(L.int_data);
}

static SourceRange translateSourceRange(CXSourceRange R) {
  return SourceRange(SourceLocation::getFromRawEncoding(R.begin_int_data),
                     SourceLocation::getFromRawEncoding(R.end_int_data));
}

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
  if (SM.isBeforeInTranslationUnit(R1.getEnd(), R2.getBegin()))
    return RangeBefore;
  if (SM.isBeforeInTranslationUnit(R2.getEnd(), R1.getBegin()))
    return RangeAfter;
  return RangeOverlap;
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
  /// \param R a source range retrieved from the abstract syntax tree.
  RangeComparisonResult CompareRegionOfInterest(SourceRange R); 
  
  /// \brief Determine whether this particular source range comes before, comes 
  /// after, or overlaps the region of interest. 
  ///
  /// \param CXR a source range retrieved from a cursor.
  RangeComparisonResult CompareRegionOfInterest(CXSourceRange CXR); 
  
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
  bool VisitChildren(CXCursor Parent);
  
  // Declaration visitors
  bool VisitDeclContext(DeclContext *DC);
  bool VisitTranslationUnitDecl(TranslationUnitDecl *D);
  bool VisitTypedefDecl(TypedefDecl *D);
  bool VisitTagDecl(TagDecl *D);
  bool VisitEnumConstantDecl(EnumConstantDecl *D);
  bool VisitDeclaratorDecl(DeclaratorDecl *DD);
  bool VisitFunctionDecl(FunctionDecl *ND);
  bool VisitFieldDecl(FieldDecl *D);
  bool VisitVarDecl(VarDecl *);
  bool VisitObjCMethodDecl(ObjCMethodDecl *ND);
  bool VisitObjCContainerDecl(ObjCContainerDecl *D);
  bool VisitObjCCategoryDecl(ObjCCategoryDecl *ND);
  bool VisitObjCProtocolDecl(ObjCProtocolDecl *PID);
  bool VisitObjCInterfaceDecl(ObjCInterfaceDecl *D);
  bool VisitObjCImplDecl(ObjCImplDecl *D);
  bool VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D);
  bool VisitObjCImplementationDecl(ObjCImplementationDecl *D);
  // FIXME: ObjCPropertyDecl requires TypeSourceInfo, getter/setter locations,
  // etc.
  // FIXME: ObjCCompatibleAliasDecl requires aliased-class locations.
  bool VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *D);
  bool VisitObjCClassDecl(ObjCClassDecl *D);

  // Type visitors
  // FIXME: QualifiedTypeLoc doesn't provide any location information
  bool VisitBuiltinTypeLoc(BuiltinTypeLoc TL);
  bool VisitTypedefTypeLoc(TypedefTypeLoc TL);
  bool VisitUnresolvedUsingTypeLoc(UnresolvedUsingTypeLoc TL);
  bool VisitTagTypeLoc(TagTypeLoc TL);
  // FIXME: TemplateTypeParmTypeLoc doesn't provide any location information
  bool VisitObjCInterfaceTypeLoc(ObjCInterfaceTypeLoc TL);
  bool VisitObjCObjectPointerTypeLoc(ObjCObjectPointerTypeLoc TL);
  bool VisitPointerTypeLoc(PointerTypeLoc TL);
  bool VisitBlockPointerTypeLoc(BlockPointerTypeLoc TL);
  bool VisitMemberPointerTypeLoc(MemberPointerTypeLoc TL);
  bool VisitLValueReferenceTypeLoc(LValueReferenceTypeLoc TL);
  bool VisitRValueReferenceTypeLoc(RValueReferenceTypeLoc TL);
  bool VisitFunctionTypeLoc(FunctionTypeLoc TL);
  bool VisitArrayTypeLoc(ArrayTypeLoc TL);
  // FIXME: Implement for TemplateSpecializationTypeLoc
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
  
  // Expression visitors
  bool VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E);
  bool VisitExplicitCastExpr(ExplicitCastExpr *E);
  bool VisitCompoundLiteralExpr(CompoundLiteralExpr *E);
};
  
} // end anonymous namespace

RangeComparisonResult CursorVisitor::CompareRegionOfInterest(SourceRange R) {
  assert(RegionOfInterest.isValid() && "RangeCompare called with invalid range");
  if (R.isInvalid())
    return RangeOverlap;

  // Move the end of the input range to the end of the last token in that
  // range.
  R.setEnd(TU->getPreprocessor().getLocForEndOfToken(R.getEnd(), 1));
  return RangeCompare(TU->getSourceManager(), R, RegionOfInterest);
}

RangeComparisonResult CursorVisitor::CompareRegionOfInterest(CXSourceRange CXR) {
  return CompareRegionOfInterest(translateSourceRange(CXR));
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
    CXSourceRange Range = clang_getCursorExtent(Cursor);
    if (translateSourceRange(Range).isInvalid() || 
        CompareRegionOfInterest(Range))
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

  llvm_unreachable("Silly GCC, we can't get here");
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
  } SetParent(Parent, StmtParent, Cursor);
  
  if (clang_isDeclaration(Cursor.kind)) {
    Decl *D = getCursorDecl(Cursor);
    assert(D && "Invalid declaration cursor");
    return Visit(D);
  }
  
  if (clang_isStatement(Cursor.kind))
    return Visit(getCursorStmt(Cursor));
  if (clang_isExpression(Cursor.kind))
    return Visit(getCursorExpr(Cursor));
  
  if (clang_isTranslationUnit(Cursor.kind)) {
    ASTUnit *CXXUnit = getCursorASTUnit(Cursor);
    if (!CXXUnit->isMainFileAST() && CXXUnit->getOnlyLocalDecls() &&
        RegionOfInterest.isInvalid()) {
      const std::vector<Decl*> &TLDs = CXXUnit->getTopLevelDecls();
      for (std::vector<Decl*>::const_iterator it = TLDs.begin(),
           ie = TLDs.end(); it != ie; ++it) {
        if (Visit(MakeCXCursor(*it, CXXUnit), true))
          return true;
      }
    } else {
      return VisitDeclContext(
                            CXXUnit->getASTContext().getTranslationUnitDecl());
    }
    
    return false;
  }
  
  // Nothing to visit at the moment.
  return false;
}

bool CursorVisitor::VisitDeclContext(DeclContext *DC) {
  for (DeclContext::decl_iterator
       I = DC->decls_begin(), E = DC->decls_end(); I != E; ++I) {
    if (RegionOfInterest.isValid()) {
      SourceRange R = (*I)->getSourceRange();
      if (R.isInvalid())
        continue;
      
      switch (CompareRegionOfInterest(R)) {
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
    
    if (Visit(MakeCXCursor(*I, TU), true))
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
  if (VisitDeclaratorDecl(ND))
    return true;

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

bool CursorVisitor::VisitObjCMethodDecl(ObjCMethodDecl *ND) {
  // FIXME: We really need a TypeLoc covering Objective-C method declarations.
  // At the moment, we don't have information about locations in the return 
  // type.
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
  if (Visit(MakeCursorObjCClassRef(D->getCategoryDecl()->getClassInterface(), 
                                   D->getLocation(), TU)))
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

bool CursorVisitor::VisitObjCInterfaceTypeLoc(ObjCInterfaceTypeLoc TL) {
  if (Visit(MakeCursorObjCClassRef(TL.getIFaceDecl(), TL.getNameLoc(), TU)))
    return true;

  for (unsigned I = 0, N = TL.getNumProtocols(); I != N; ++I) {
    if (Visit(MakeCursorObjCProtocolRef(TL.getProtocol(I), TL.getProtocolLoc(I),
                                        TU)))
      return true;
  }

  return false;
}

bool CursorVisitor::VisitObjCObjectPointerTypeLoc(ObjCObjectPointerTypeLoc TL) {
  if (TL.hasBaseTypeAsWritten() && Visit(TL.getBaseTypeLoc()))
    return true;

  if (TL.hasProtocolsAsWritten()) {
    for (unsigned I = 0, N = TL.getNumProtocols(); I != N; ++I) {
      if (Visit(MakeCursorObjCProtocolRef(TL.getProtocol(I), 
                                          TL.getProtocolLoc(I),
                                          TU)))
        return true;
    }
  }
  
  return false;
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

bool CursorVisitor::VisitFunctionTypeLoc(FunctionTypeLoc TL) {
  if (Visit(TL.getResultLoc()))
    return true;

  for (unsigned I = 0, N = TL.getNumArgs(); I != N; ++I)
    if (Visit(MakeCXCursor(TL.getArg(I), TU)))
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
    if (Visit(MakeCXCursor(*Child, StmtParent, TU)))
      return true;
  }
  
  return false;
}

bool CursorVisitor::VisitDeclStmt(DeclStmt *S) {
  for (DeclStmt::decl_iterator D = S->decl_begin(), DEnd = S->decl_end();
       D != DEnd; ++D) {
    if (Visit(MakeCXCursor(*D, TU)))
      return true;
  }
  
  return false;
}

bool CursorVisitor::VisitIfStmt(IfStmt *S) {
  if (VarDecl *Var = S->getConditionVariable()) {
    if (Visit(MakeCXCursor(Var, TU)))
      return true;
  } else if (S->getCond() && Visit(MakeCXCursor(S->getCond(), StmtParent, TU)))
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
  } else if (S->getCond() && Visit(MakeCXCursor(S->getCond(), StmtParent, TU)))
    return true;

  if (S->getBody() && Visit(MakeCXCursor(S->getBody(), StmtParent, TU)))
    return true;
  
  return false;
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

CXString CIndexer::createCXString(const char *String, bool DupString){
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

extern "C" {
CXIndex clang_createIndex(int excludeDeclarationsFromPCH,
                          int displayDiagnostics) {
  CIndexer *CIdxr = new CIndexer();
  if (excludeDeclarationsFromPCH)
    CIdxr->setOnlyLocalDecls();
  if (displayDiagnostics)
    CIdxr->setDisplayDiagnostics();
  return CIdxr;
}

void clang_disposeIndex(CXIndex CIdx) {
  assert(CIdx && "Passed null CXIndex");
  delete static_cast<CIndexer *>(CIdx);
}

void clang_setUseExternalASTGeneration(CXIndex CIdx, int value) {
  assert(CIdx && "Passed null CXIndex");
  CIndexer *CXXIdx = static_cast<CIndexer *>(CIdx);
  CXXIdx->setUseExternalASTGeneration(value);
}

// FIXME: need to pass back error info.
CXTranslationUnit clang_createTranslationUnit(CXIndex CIdx,
                                              const char *ast_filename) {
  assert(CIdx && "Passed null CXIndex");
  CIndexer *CXXIdx = static_cast<CIndexer *>(CIdx);

  return ASTUnit::LoadFromPCHFile(ast_filename, CXXIdx->getDiags(),
                                  CXXIdx->getOnlyLocalDecls(),
                                  /* UseBumpAllocator = */ true);
}

CXTranslationUnit
clang_createTranslationUnitFromSourceFile(CXIndex CIdx,
                                          const char *source_filename,
                                          int num_command_line_args,
                                          const char **command_line_args,
                                          unsigned num_unsaved_files,
                                          struct CXUnsavedFile *unsaved_files) {
  assert(CIdx && "Passed null CXIndex");
  CIndexer *CXXIdx = static_cast<CIndexer *>(CIdx);

  llvm::SmallVector<ASTUnit::RemappedFile, 4> RemappedFiles;
  for (unsigned I = 0; I != num_unsaved_files; ++I) {
    const llvm::MemoryBuffer *Buffer 
    = llvm::MemoryBuffer::getMemBuffer(unsaved_files[I].Contents,
                                       unsaved_files[I].Contents + unsaved_files[I].Length,
                                       unsaved_files[I].Filename);
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
    Args.insert(Args.end(), command_line_args,
                command_line_args + num_command_line_args);

    unsigned NumErrors = CXXIdx->getDiags().getNumErrors();
    
#ifdef USE_CRASHTRACER
    ArgsCrashTracerInfo ACTI(Args);
#endif
    
    llvm::OwningPtr<ASTUnit> Unit(
      ASTUnit::LoadFromCommandLine(Args.data(), Args.data() + Args.size(),
                                   CXXIdx->getDiags(),
                                   CXXIdx->getClangResourcesPath(),
                                   CXXIdx->getOnlyLocalDecls(),
                                   /* UseBumpAllocator = */ true,
                                   RemappedFiles.data(),
                                   RemappedFiles.size()));
    
    // FIXME: Until we have broader testing, just drop the entire AST if we
    // encountered an error.
    if (NumErrors != CXXIdx->getDiags().getNumErrors())
      return 0;

    return Unit.take();
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

  // Remap any unsaved files to temporary files.
  std::vector<llvm::sys::Path> TemporaryFiles;
  std::vector<std::string> RemapArgs;
  if (RemapFiles(num_unsaved_files, unsaved_files, RemapArgs, TemporaryFiles))
    return 0;
  
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

  // Add the null terminator.
  argv.push_back(NULL);

  // Invoke 'clang'.
  llvm::sys::Path DevNull; // leave empty, causes redirection to /dev/null
                           // on Unix or NUL (Windows).
  std::string ErrMsg;
  const llvm::sys::Path *Redirects[] = { &DevNull, &DevNull, &DevNull, NULL };
  llvm::sys::Program::ExecuteAndWait(ClangPath, &argv[0], /* env */ NULL,
      /* redirects */ !CXXIdx->getDisplayDiagnostics() ? &Redirects[0] : NULL,
      /* secondsToWait */ 0, /* memoryLimits */ 0, &ErrMsg);

  if (CXXIdx->getDisplayDiagnostics() && !ErrMsg.empty()) {
    llvm::errs() << "clang_createTranslationUnitFromSourceFile: " << ErrMsg
                 << '\n' << "Arguments: \n";
    for (std::vector<const char*>::iterator I = argv.begin(), E = argv.end();
         I!=E; ++I) {
      if (*I)
        llvm::errs() << ' ' << *I << '\n';
    }
    llvm::errs() << '\n';
  }

  ASTUnit *ATU = ASTUnit::LoadFromPCHFile(astTmpFile, CXXIdx->getDiags(),
                                          CXXIdx->getOnlyLocalDecls(),
                                          /* UseBumpAllocator = */ true,
                                          RemappedFiles.data(),
                                          RemappedFiles.size());
  if (ATU)
    ATU->unlinkTemporaryFile();
  
  for (unsigned i = 0, e = TemporaryFiles.size(); i != e; ++i)
    TemporaryFiles[i].eraseFromDisk();
  
  return ATU;
}

void clang_disposeTranslationUnit(CXTranslationUnit CTUnit) {
  assert(CTUnit && "Passed null CXTranslationUnit");
  delete static_cast<ASTUnit *>(CTUnit);
}

CXString clang_getTranslationUnitSpelling(CXTranslationUnit CTUnit) {
  assert(CTUnit && "Passed null CXTranslationUnit");
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(CTUnit);
  return CIndexer::createCXString(CXXUnit->getOriginalSourceFileName().c_str(),
                                  true);
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
  CXSourceLocation Result = { 0, 0 };
  return Result;
}

unsigned clang_equalLocations(CXSourceLocation loc1, CXSourceLocation loc2) {
  return loc1.ptr_data == loc2.ptr_data && loc1.int_data == loc2.int_data;
}

CXSourceLocation clang_getLocation(CXTranslationUnit tu,
                                   CXFile file,
                                   unsigned line,
                                   unsigned column) {
  if (!tu)
    return clang_getNullLocation();
  
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(tu);
  SourceLocation SLoc
    = CXXUnit->getSourceManager().getLocation(
                                        static_cast<const FileEntry *>(file), 
                                              line, column);
  
  return translateSourceLocation(CXXUnit->getASTContext(), SLoc, false);
}

CXSourceRange clang_getRange(CXSourceLocation begin, CXSourceLocation end) {
  if (begin.ptr_data != end.ptr_data) {
    CXSourceRange Result = { 0, 0, 0 };
    return Result;
  }
  
  CXSourceRange Result = { begin.ptr_data, begin.int_data, end.int_data };
  return Result;
}

void clang_getInstantiationLocation(CXSourceLocation location,
                                    CXFile *file,
                                    unsigned *line,
                                    unsigned *column) {
  CXSourceLocationPtr Ptr
    = CXSourceLocationPtr::getFromOpaqueValue(location.ptr_data);
  SourceLocation Loc = SourceLocation::getFromRawEncoding(location.int_data);

  if (!Ptr.getPointer() || Loc.isInvalid()) {
    if (file)
      *file = 0;
    if (line)
      *line = 0;
    if (column)
      *column = 0;
    return;
  }

  // FIXME: This is largely copy-paste from
  ///TextDiagnosticPrinter::HighlightRange.  When it is clear that this is
  // what we want the two routines should be refactored.  
  ASTContext &Context = *Ptr.getPointer();
  SourceManager &SM = Context.getSourceManager();
  SourceLocation InstLoc = SM.getInstantiationLoc(Loc);
  
  if (Ptr.getInt()) {
    // We want the last character in this location, so we will adjust
    // the instantiation location accordingly.

    // If the location is from a macro instantiation, get the end of
    // the instantiation range.
    if (Loc.isMacroID())
      InstLoc = SM.getInstantiationRange(Loc).second;

    // Measure the length token we're pointing at, so we can adjust
    // the physical location in the file to point at the last
    // character.
    // FIXME: This won't cope with trigraphs or escaped newlines
    // well. For that, we actually need a preprocessor, which isn't
    // currently available here. Eventually, we'll switch the pointer
    // data of CXSourceLocation/CXSourceRange to a translation unit
    // (CXXUnit), so that the preprocessor will be available here. At
    // that point, we can use Preprocessor::getLocForEndOfToken().
    unsigned Length = Lexer::MeasureTokenLength(InstLoc, SM, 
                                                Context.getLangOptions());
    if (Length > 0)
      InstLoc = InstLoc.getFileLocWithOffset(Length - 1);
  }

  if (file)
    *file = (void *)SM.getFileEntryForID(SM.getFileID(InstLoc));
  if (line)
    *line = SM.getInstantiationLineNumber(InstLoc);
  if (column)
    *column = SM.getInstantiationColumnNumber(InstLoc);
}

CXSourceLocation clang_getRangeStart(CXSourceRange range) {
  CXSourceLocation Result = { range.ptr_data, range.begin_int_data };
  return Result;
}

CXSourceLocation clang_getRangeEnd(CXSourceRange range) {
  llvm::PointerIntPair<ASTContext *, 1, bool> Ptr;
  Ptr.setPointer(static_cast<ASTContext *>(range.ptr_data));
  Ptr.setInt(true);
  CXSourceLocation Result = { Ptr.getOpaqueValue(), range.end_int_data };
  return Result;
}

} // end: extern "C"

//===----------------------------------------------------------------------===//
// CXFile Operations.
//===----------------------------------------------------------------------===//

extern "C" {
const char *clang_getFileName(CXFile SFile) {
  if (!SFile)
    return 0;
  
  assert(SFile && "Passed null CXFile");
  FileEntry *FEnt = static_cast<FileEntry *>(SFile);
  return FEnt->getName();
}

time_t clang_getFileTime(CXFile SFile) {
  if (!SFile)
    return 0;
  
  assert(SFile && "Passed null CXFile");
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

extern "C" {
  
unsigned clang_visitChildren(CXCursor parent, 
                             CXCursorVisitor visitor,
                             CXClientData client_data) {
  ASTUnit *CXXUnit = getCursorASTUnit(parent);

  unsigned PCHLevel = Decl::MaxPCHLevel;
  
  // Set the PCHLevel to filter out unwanted decls if requested.
  if (CXXUnit->getOnlyLocalDecls()) {
    PCHLevel = 0;
    
    // If the main input was an AST, bump the level.
    if (CXXUnit->isMainFileAST())
      ++PCHLevel;
  }
  
  CursorVisitor CursorVis(CXXUnit, visitor, client_data, PCHLevel);
  return CursorVis.VisitChildren(parent);
}

static CXString getDeclSpelling(Decl *D) {
  NamedDecl *ND = dyn_cast_or_null<NamedDecl>(D);
  if (!ND)
    return CIndexer::createCXString("");
  
  if (ObjCMethodDecl *OMD = dyn_cast<ObjCMethodDecl>(ND))
    return CIndexer::createCXString(OMD->getSelector().getAsString().c_str(),
                                    true);
  
  if (ObjCCategoryImplDecl *CIMP = dyn_cast<ObjCCategoryImplDecl>(ND))
    // No, this isn't the same as the code below. getIdentifier() is non-virtual
    // and returns different names. NamedDecl returns the class name and
    // ObjCCategoryImplDecl returns the category name.
    return CIndexer::createCXString(CIMP->getIdentifier()->getNameStart());
  
  if (ND->getIdentifier())
    return CIndexer::createCXString(ND->getIdentifier()->getNameStart());
  
  return CIndexer::createCXString("");
}
    
CXString clang_getCursorSpelling(CXCursor C) {
  assert(getCursorDecl(C) && "CXCursor has null decl");
  if (clang_isTranslationUnit(C.kind))
    return clang_getTranslationUnitSpelling(C.data[2]);

  if (clang_isReference(C.kind)) {
    switch (C.kind) {
    case CXCursor_ObjCSuperClassRef: {
      ObjCInterfaceDecl *Super = getCursorObjCSuperClassRef(C).first;
      return CIndexer::createCXString(Super->getIdentifier()->getNameStart());
    }
    case CXCursor_ObjCClassRef: {
      ObjCInterfaceDecl *Class = getCursorObjCClassRef(C).first;
      return CIndexer::createCXString(Class->getIdentifier()->getNameStart());
    }
    case CXCursor_ObjCProtocolRef: {
      ObjCProtocolDecl *OID = getCursorObjCProtocolRef(C).first;
      assert(OID && "getCursorSpelling(): Missing protocol decl");
      return CIndexer::createCXString(OID->getIdentifier()->getNameStart());
    }
    case CXCursor_TypeRef: {
      TypeDecl *Type = getCursorTypeRef(C).first;
      assert(Type && "Missing type decl");

      return CIndexer::createCXString(
               getCursorContext(C).getTypeDeclType(Type).getAsString().c_str(),
                                      true);
    }

    default:
      return CIndexer::createCXString("<not implemented>");
    }
  }

  if (clang_isExpression(C.kind)) {
    Decl *D = getDeclFromExpr(getCursorExpr(C));
    if (D)
      return getDeclSpelling(D);
    return CIndexer::createCXString("");
  }

  return getDeclSpelling(getCursorDecl(C));
}

const char *clang_getCursorKindSpelling(enum CXCursorKind Kind) {
  switch (Kind) {
  case CXCursor_FunctionDecl: return "FunctionDecl";
  case CXCursor_TypedefDecl: return "TypedefDecl";
  case CXCursor_EnumDecl: return "EnumDecl";
  case CXCursor_EnumConstantDecl: return "EnumConstantDecl";
  case CXCursor_StructDecl: return "StructDecl";
  case CXCursor_UnionDecl: return "UnionDecl";
  case CXCursor_ClassDecl: return "ClassDecl";
  case CXCursor_FieldDecl: return "FieldDecl";
  case CXCursor_VarDecl: return "VarDecl";
  case CXCursor_ParmDecl: return "ParmDecl";
  case CXCursor_ObjCInterfaceDecl: return "ObjCInterfaceDecl";
  case CXCursor_ObjCCategoryDecl: return "ObjCCategoryDecl";
  case CXCursor_ObjCProtocolDecl: return "ObjCProtocolDecl";
  case CXCursor_ObjCPropertyDecl: return "ObjCPropertyDecl";
  case CXCursor_ObjCIvarDecl: return "ObjCIvarDecl";
  case CXCursor_ObjCInstanceMethodDecl: return "ObjCInstanceMethodDecl";
  case CXCursor_ObjCClassMethodDecl: return "ObjCClassMethodDecl";
  case CXCursor_ObjCImplementationDecl: return "ObjCImplementationDecl";
  case CXCursor_ObjCCategoryImplDecl: return "ObjCCategoryImplDecl";
  case CXCursor_UnexposedDecl: return "UnexposedDecl";
  case CXCursor_ObjCSuperClassRef: return "ObjCSuperClassRef";
  case CXCursor_ObjCProtocolRef: return "ObjCProtocolRef";
  case CXCursor_ObjCClassRef: return "ObjCClassRef";
  case CXCursor_TypeRef: return "TypeRef";
  case CXCursor_UnexposedExpr: return "UnexposedExpr";
  case CXCursor_DeclRefExpr: return "DeclRefExpr";
  case CXCursor_MemberRefExpr: return "MemberRefExpr";
  case CXCursor_CallExpr: return "CallExpr";
  case CXCursor_ObjCMessageExpr: return "ObjCMessageExpr";
  case CXCursor_UnexposedStmt: return "UnexposedStmt";
  case CXCursor_InvalidFile: return "InvalidFile";
  case CXCursor_NoDeclFound: return "NoDeclFound";
  case CXCursor_NotImplemented: return "NotImplemented";
  case CXCursor_TranslationUnit: return "TranslationUnit";
  }
  
  llvm_unreachable("Unhandled CXCursorKind");
  return NULL;
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

  SourceLocation SLoc = translateSourceLocation(Loc);
  CXCursor Result = MakeCXCursorInvalid(CXCursor_NoDeclFound);
  if (SLoc.isValid()) {
    SourceRange RegionOfInterest(SLoc, 
                       CXXUnit->getPreprocessor().getLocForEndOfToken(SLoc, 1));
    
    // FIXME: Would be great to have a "hint" cursor, then walk from that
    // hint cursor upward until we find a cursor whose source range encloses
    // the region of interest, rather than starting from the translation unit.
    CXCursor Parent = clang_getTranslationUnitCursor(CXXUnit);
    CursorVisitor CursorVis(CXXUnit, GetCursorVisitor, &Result, 
                            Decl::MaxPCHLevel, RegionOfInterest);
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

CXCursorKind clang_getCursorKind(CXCursor C) {
  return C.kind;
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

CXSourceLocation clang_getCursorLocation(CXCursor C) {
  if (clang_isReference(C.kind)) {
    switch (C.kind) {
    case CXCursor_ObjCSuperClassRef: {       
      std::pair<ObjCInterfaceDecl *, SourceLocation> P
        = getCursorObjCSuperClassRef(C);
      return translateSourceLocation(P.first->getASTContext(), P.second);
    }

    case CXCursor_ObjCProtocolRef: {       
      std::pair<ObjCProtocolDecl *, SourceLocation> P
        = getCursorObjCProtocolRef(C);
      return translateSourceLocation(P.first->getASTContext(), P.second);
    }

    case CXCursor_ObjCClassRef: {       
      std::pair<ObjCInterfaceDecl *, SourceLocation> P
        = getCursorObjCClassRef(C);
      return translateSourceLocation(P.first->getASTContext(), P.second);
    }

    case CXCursor_TypeRef: {       
      std::pair<TypeDecl *, SourceLocation> P = getCursorTypeRef(C);
      return translateSourceLocation(P.first->getASTContext(), P.second);
    }
      
    default:
      // FIXME: Need a way to enumerate all non-reference cases.
      llvm_unreachable("Missed a reference kind");
    }
  }

  if (clang_isExpression(C.kind))
    return translateSourceLocation(getCursorContext(C), 
                                   getLocationFromExpr(getCursorExpr(C)));

  if (!getCursorDecl(C)) {
    CXSourceLocation empty = { 0, 0 };
    return empty;
  }

  Decl *D = getCursorDecl(C);
  SourceLocation Loc = D->getLocation();
  if (ObjCInterfaceDecl *Class = dyn_cast<ObjCInterfaceDecl>(D))
    Loc = Class->getClassLoc();
  return translateSourceLocation(D->getASTContext(), Loc);
}

CXSourceRange clang_getCursorExtent(CXCursor C) {
  if (clang_isReference(C.kind)) {
    switch (C.kind) {
      case CXCursor_ObjCSuperClassRef: {       
        std::pair<ObjCInterfaceDecl *, SourceLocation> P
          = getCursorObjCSuperClassRef(C);
        return translateSourceRange(P.first->getASTContext(), P.second);
      }
        
      case CXCursor_ObjCProtocolRef: {       
        std::pair<ObjCProtocolDecl *, SourceLocation> P
          = getCursorObjCProtocolRef(C);
        return translateSourceRange(P.first->getASTContext(), P.second);
      }
        
      case CXCursor_ObjCClassRef: {       
        std::pair<ObjCInterfaceDecl *, SourceLocation> P
          = getCursorObjCClassRef(C);
        
        return translateSourceRange(P.first->getASTContext(), P.second);
      }

      case CXCursor_TypeRef: {       
        std::pair<TypeDecl *, SourceLocation> P = getCursorTypeRef(C);
        return translateSourceRange(P.first->getASTContext(), P.second);
      }
        
      default:
        // FIXME: Need a way to enumerate all non-reference cases.
        llvm_unreachable("Missed a reference kind");
    }
  }

  if (clang_isExpression(C.kind))
    return translateSourceRange(getCursorContext(C), 
                                getCursorExpr(C)->getSourceRange());

  if (clang_isStatement(C.kind))
    return translateSourceRange(getCursorContext(C), 
                                getCursorStmt(C)->getSourceRange());
  
  if (!getCursorDecl(C)) {
    CXSourceRange empty = { 0, 0, 0 };
    return empty;
  }
  
  Decl *D = getCursorDecl(C);
  return translateSourceRange(D->getASTContext(), D->getSourceRange());
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
  case Decl::LinkageSpec:
  case Decl::ObjCPropertyImpl:
  case Decl::FileScopeAsm:
  case Decl::StaticAssert:
  case Decl::Block:
    return C;

  // Declaration kinds that don't make any sense here, but are
  // nonetheless harmless.
  case Decl::TranslationUnit:
  case Decl::Template:
  case Decl::ObjCContainer:
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
    if (TagDecl *Def = cast<TagDecl>(D)->getDefinition(D->getASTContext()))
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
    VarDecl *Var = cast<VarDecl>(D);

    // Variables with initializers have definitions.
    const VarDecl *Def = 0;
    if (Var->getDefinition(Def))
      return MakeCXCursor(const_cast<VarDecl *>(Def), CXXUnit);

    // extern and private_extern variables are not definitions.
    if (Var->hasExternalStorage())
      return clang_getNullCursor();

    // In-line static data members do not have definitions.
    if (Var->isStaticDataMember() && !Var->isOutOfLine())
      return clang_getNullCursor();

    // All other variables are themselves definitions.
    return C;
  }
   
  case Decl::FunctionTemplate: {
    const FunctionDecl *Def = 0;
    if (cast<FunctionTemplateDecl>(D)->getTemplatedDecl()->getBody(Def))
      return MakeCXCursor(Def->getDescribedFunctionTemplate(), CXXUnit);
    return clang_getNullCursor();
  }
   
  case Decl::ClassTemplate: {
    if (RecordDecl *Def = cast<ClassTemplateDecl>(D)->getTemplatedDecl()
                                          ->getDefinition(D->getASTContext()))
      return MakeCXCursor(
                         cast<CXXRecordDecl>(Def)->getDescribedClassTemplate(), 
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

//===----------------------------------------------------------------------===//
// Misc. utility functions.
//===----------------------------------------------------------------------===//
  
extern "C" {

const char *clang_getClangVersion() {
  return getClangFullVersion();
}

} // end: extern "C"

