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

#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/Lexer.h"
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


//===----------------------------------------------------------------------===//
// Visitors.
//===----------------------------------------------------------------------===//

namespace {
  
// Cursor visitor.
class CursorVisitor : public DeclVisitor<CursorVisitor, bool> {
  CXCursor Parent;
  CXCursorVisitor Visitor;
  CXClientData ClientData;
  
  // MaxPCHLevel - the maximum PCH level of declarations that we will pass on
  // to the visitor. Declarations with a PCH level greater than this value will
  // be suppressed.
  unsigned MaxPCHLevel;
  
  using DeclVisitor<CursorVisitor, bool>::Visit;
  
public:
  CursorVisitor(CXCursorVisitor Visitor, CXClientData ClientData, 
                unsigned MaxPCHLevel)
    : Visitor(Visitor), ClientData(ClientData), MaxPCHLevel(MaxPCHLevel)
  {
    Parent.kind = CXCursor_NoDeclFound;
    Parent.data[0] = 0;
    Parent.data[1] = 0;
    Parent.data[2] = 0;
  }
  
  bool Visit(CXCursor Cursor);
  bool VisitChildren(CXCursor Parent);
  
  bool VisitDeclContext(DeclContext *DC);
  
  bool VisitTranslationUnitDecl(TranslationUnitDecl *D);
  bool VisitFunctionDecl(FunctionDecl *ND);
  bool VisitObjCCategoryDecl(ObjCCategoryDecl *ND);
  bool VisitObjCInterfaceDecl(ObjCInterfaceDecl *D);
  bool VisitObjCMethodDecl(ObjCMethodDecl *ND);
  bool VisitObjCProtocolDecl(ObjCProtocolDecl *PID);
  bool VisitTagDecl(TagDecl *D);
};
  
} // end anonymous namespace

/// \brief Visit the given cursor and, if requested by the visitor,
/// its children.
///
/// \returns true if the visitation should be aborted, false if it
/// should continue.
bool CursorVisitor::Visit(CXCursor Cursor) {
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
  // Set the Parent field to Cursor, then back to its old value once we're 
  // done.
  class SetParentRAII {
    CXCursor &Parent;
    CXCursor OldParent;
    
  public:
    SetParentRAII(CXCursor &Parent, CXCursor NewParent)
      : Parent(Parent), OldParent(Parent) 
    {
      Parent = NewParent;
    }
    
    ~SetParentRAII() {
      Parent = OldParent;
    }
  } SetParent(Parent, Cursor);
  
  if (clang_isDeclaration(Cursor.kind)) {
    Decl *D = getCursorDecl(Cursor);
    assert(D && "Invalid declaration cursor");
    return Visit(D);
  }
  
  if (clang_isTranslationUnit(Cursor.kind)) {
    ASTUnit *CXXUnit = static_cast<ASTUnit *>(Cursor.data[0]);
    if (!CXXUnit->isMainFileAST() && CXXUnit->getOnlyLocalDecls()) {
      const std::vector<Decl*> &TLDs = CXXUnit->getTopLevelDecls();
      for (std::vector<Decl*>::const_iterator it = TLDs.begin(),
           ie = TLDs.end(); it != ie; ++it) {
        if (Visit(MakeCXCursor(*it)))
          return true;
      }
    } else {
      return VisitDeclContext(
                            CXXUnit->getASTContext().getTranslationUnitDecl());
    }
    
    return false;
  }
    
  // Nothing to visit at the moment.
  // FIXME: Traverse statements, declarations, etc. here.
  return false;
}

bool CursorVisitor::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
  llvm_unreachable("Translation units are visited directly by Visit()");
  return false;
}

bool CursorVisitor::VisitDeclContext(DeclContext *DC) {
  for (DeclContext::decl_iterator
       I = DC->decls_begin(), E = DC->decls_end(); I != E; ++I) {
    if (Visit(MakeCXCursor(*I)))
      return true;
  }
  
  return false;
}

bool CursorVisitor::VisitFunctionDecl(FunctionDecl *ND) {
  // FIXME: This is wrong. We always want to visit the parameters and
  // the body, if available.
  if (ND->isThisDeclarationADefinition()) {
    return VisitDeclContext(ND);
    
#if 0
    // Not currently needed.
    CompoundStmt *Body = dyn_cast<CompoundStmt>(ND->getBody());
    CRefVisitor RVisit(CDecl, Callback, CData);
    RVisit.Visit(Body);
#endif
  }
  
  return false;
}

bool CursorVisitor::VisitObjCCategoryDecl(ObjCCategoryDecl *ND) {
  if (Visit(MakeCursorObjCClassRef(ND->getClassInterface(), ND->getLocation())))
    return true;
  
  ObjCCategoryDecl::protocol_loc_iterator PL = ND->protocol_loc_begin();
  for (ObjCCategoryDecl::protocol_iterator I = ND->protocol_begin(),
         E = ND->protocol_end(); I != E; ++I, ++PL)
    if (Visit(MakeCursorObjCProtocolRef(*I, *PL)))
      return true;
  
  return VisitDeclContext(ND);
}

bool CursorVisitor::VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
  // Issue callbacks for super class.
  if (D->getSuperClass() &&
      Visit(MakeCursorObjCSuperClassRef(D->getSuperClass(),
                                        D->getSuperClassLoc())))
    return true;
  
  ObjCInterfaceDecl::protocol_loc_iterator PL = D->protocol_loc_begin();
  for (ObjCInterfaceDecl::protocol_iterator I = D->protocol_begin(),
         E = D->protocol_end(); I != E; ++I, ++PL)
    if (Visit(MakeCursorObjCProtocolRef(*I, *PL)))
      return true;
  
  return VisitDeclContext(D);
}

bool CursorVisitor::VisitObjCMethodDecl(ObjCMethodDecl *ND) {
  // FIXME: Wrong in the same way that VisitFunctionDecl is wrong.
  if (ND->getBody())
    return VisitDeclContext(ND);
  
  return false;
}

bool CursorVisitor::VisitObjCProtocolDecl(ObjCProtocolDecl *PID) {
  ObjCProtocolDecl::protocol_loc_iterator PL = PID->protocol_loc_begin();
  for (ObjCProtocolDecl::protocol_iterator I = PID->protocol_begin(),
         E = PID->protocol_end(); I != E; ++I, ++PL)
    if (Visit(MakeCursorObjCProtocolRef(*I, *PL)))
      return true;
  
  return VisitDeclContext(PID);
}

bool CursorVisitor::VisitTagDecl(TagDecl *D) {
  return VisitDeclContext(D);
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
  CIndexer *CIdxr = new CIndexer(new Program());
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
                                          const char **command_line_args) {
  assert(CIdx && "Passed null CXIndex");
  CIndexer *CXXIdx = static_cast<CIndexer *>(CIdx);

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
                                   /* UseBumpAllocator = */ true));
    
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

  // Finally, we create the translation unit from the ast file.
  ASTUnit *ATU = static_cast<ASTUnit *>(
    clang_createTranslationUnit(CIdx, astTmpFile));
  if (ATU)
    ATU->unlinkTemporaryFile();
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
  CXCursor Result = { CXCursor_TranslationUnit, { TU, 0, 0 } };
  return Result;
}

struct LoadTranslationUnitData {
  CXTranslationUnit TU;
  CXTranslationUnitIterator Callback;
  CXClientData ClientData;
};
  
enum CXChildVisitResult LoadTranslationUnitVisitor(CXCursor cursor,
                                                   CXCursor parent,
                                                   CXClientData client_data) {
  LoadTranslationUnitData *Data
    = static_cast<LoadTranslationUnitData *>(client_data);
  Data->Callback(Data->TU, cursor, Data->ClientData);
  return CXChildVisit_Continue;
}
  
void clang_loadTranslationUnit(CXTranslationUnit CTUnit,
                               CXTranslationUnitIterator callback,
                               CXClientData CData) {
  assert(CTUnit && "Passed null CXTranslationUnit");
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(CTUnit);

  unsigned PCHLevel = Decl::MaxPCHLevel;

  // Set the PCHLevel to filter out unwanted decls if requested.
  if (CXXUnit->getOnlyLocalDecls()) {
    PCHLevel = 0;

    // If the main input was an AST, bump the level.
    if (CXXUnit->isMainFileAST())
      ++PCHLevel;
  }

  LoadTranslationUnitData Data = { CTUnit, callback, CData };
  
  CursorVisitor CurVisitor(&LoadTranslationUnitVisitor, &Data, PCHLevel);
  CurVisitor.VisitChildren(clang_getTranslationUnitCursor(CTUnit));
}

struct LoadDeclarationData {
  CXDeclIterator Callback;
  CXClientData ClientData;
};

CXChildVisitResult LoadDeclarationVisitor(CXCursor cursor, 
                                          CXCursor parent, 
                                          CXClientData client_data) {
  LoadDeclarationData *Data = static_cast<LoadDeclarationData *>(client_data);
  Data->Callback(clang_getCursorDecl(cursor), cursor, Data->ClientData);
  return CXChildVisit_Recurse;
}
  
void clang_loadDeclaration(CXDecl Dcl,
                           CXDeclIterator callback,
                           CXClientData CData) {
  assert(Dcl && "Passed null CXDecl");

  LoadDeclarationData Data = { callback, CData };
  CursorVisitor CurVisit(&LoadDeclarationVisitor, &Data, 
                         static_cast<Decl *>(Dcl)->getPCHLevel());
  CurVisit.VisitChildren(clang_getCursorFromDecl(Dcl));
}
} // end: extern "C"

//===----------------------------------------------------------------------===//
// CXSourceLocation and CXSourceRange Operations.
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// CXDecl Operations.
//===----------------------------------------------------------------------===//

extern "C" {
CXString clang_getDeclSpelling(CXDecl AnonDecl) {
  assert(AnonDecl && "Passed null CXDecl");
  Decl *D = static_cast<Decl *>(AnonDecl);
  NamedDecl *ND = dyn_cast<NamedDecl>(D);
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
  
unsigned clang_visitChildren(CXTranslationUnit tu,
                             CXCursor parent, 
                             CXCursorVisitor visitor,
                             CXClientData client_data) {
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(tu);

  unsigned PCHLevel = Decl::MaxPCHLevel;
  
  // Set the PCHLevel to filter out unwanted decls if requested.
  if (CXXUnit->getOnlyLocalDecls()) {
    PCHLevel = 0;
    
    // If the main input was an AST, bump the level.
    if (CXXUnit->isMainFileAST())
      ++PCHLevel;
  }
  
  CursorVisitor CursorVis(visitor, client_data, PCHLevel);
  return CursorVis.VisitChildren(parent);
}

CXString clang_getCursorSpelling(CXCursor C) {
  assert(getCursorDecl(C) && "CXCursor has null decl");
  if (clang_isTranslationUnit(C.kind))
    return clang_getTranslationUnitSpelling(C.data[0]);

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
    default:
      return CIndexer::createCXString("<not implemented>");
    }
  }

  if (clang_isExpression(C.kind)) {
    Decl *D = getDeclFromExpr(getCursorExpr(C));
    if (D)
      return clang_getDeclSpelling(D);
    return CIndexer::createCXString("");
  }

  return clang_getDeclSpelling(getCursorDecl(C));
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

CXCursor clang_getCursor(CXTranslationUnit CTUnit, const char *source_name,
                         unsigned line, unsigned column) {
  assert(CTUnit && "Passed null CXTranslationUnit");
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(CTUnit);

  FileManager &FMgr = CXXUnit->getFileManager();
  const FileEntry *File = FMgr.getFile(source_name,
                                       source_name+strlen(source_name));
  if (!File)
    return clang_getNullCursor();

  SourceLocation SLoc =
    CXXUnit->getSourceManager().getLocation(File, line, column);

  ASTLocation LastLoc = CXXUnit->getLastASTLocation();
  ASTLocation ALoc = ResolveLocationInAST(CXXUnit->getASTContext(), SLoc,
                                          &LastLoc);
  
  // FIXME: This doesn't look thread-safe.
  if (ALoc.isValid())
    CXXUnit->setLastASTLocation(ALoc);

  Decl *Dcl = ALoc.getParentDecl();
  if (ALoc.isNamedRef())
    Dcl = ALoc.AsNamedRef().ND;
  Stmt *Stm = ALoc.dyn_AsStmt();
  if (Dcl) {
    if (Stm)
      return MakeCXCursor(Stm, Dcl);

    if (ALoc.isNamedRef()) {
      if (ObjCInterfaceDecl *Class = dyn_cast<ObjCInterfaceDecl>(Dcl))
        return MakeCursorObjCClassRef(Class, ALoc.AsNamedRef().Loc);
      if (ObjCProtocolDecl *Proto = dyn_cast<ObjCProtocolDecl>(Dcl))
        return MakeCursorObjCProtocolRef(Proto, ALoc.AsNamedRef().Loc);
    }
    return MakeCXCursor(Dcl);
  }
  return MakeCXCursor(CXCursor_NoDeclFound, 0);
}

CXCursor clang_getNullCursor(void) {
  return MakeCXCursor(CXCursor_InvalidFile, 0);
}

unsigned clang_equalCursors(CXCursor X, CXCursor Y) {
  return X == Y;
}

CXCursor clang_getCursorFromDecl(CXDecl AnonDecl) {
  assert(AnonDecl && "Passed null CXDecl");
  return MakeCXCursor(static_cast<NamedDecl *>(AnonDecl));
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

CXDecl clang_getCursorDecl(CXCursor C) {
  if (clang_isDeclaration(C.kind))
    return getCursorDecl(C);

  if (clang_isReference(C.kind)) {
    if (getCursorStmt(C))
      return getDeclFromExpr(getCursorStmt(C));

    return getCursorDecl(C);
  }

  if (clang_isExpression(C.kind))
    return getDeclFromExpr(getCursorStmt(C));

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
        
      default:
        // FIXME: Need a way to enumerate all non-reference cases.
        llvm_unreachable("Missed a reference kind");
    }
  }

  if (clang_isExpression(C.kind))
    return translateSourceRange(getCursorContext(C), 
                                getCursorExpr(C)->getSourceRange());
  
  if (!getCursorDecl(C)) {
    CXSourceRange empty = { 0, 0, 0 };
    return empty;
  }
  
  Decl *D = getCursorDecl(C);
  return translateSourceRange(D->getASTContext(), D->getSourceRange());
}

CXCursor clang_getCursorReferenced(CXCursor C) {
  if (clang_isDeclaration(C.kind))
    return C;
  
  if (clang_isExpression(C.kind)) {
    Decl *D = getDeclFromExpr(getCursorExpr(C));
    if (D)
      return MakeCXCursor(D);
    return clang_getNullCursor();
  }

  if (!clang_isReference(C.kind))
    return clang_getNullCursor();
  
  switch (C.kind) {
    case CXCursor_ObjCSuperClassRef:
      return MakeCXCursor(getCursorObjCSuperClassRef(C).first);
      
    case CXCursor_ObjCProtocolRef: {       
      return MakeCXCursor(getCursorObjCProtocolRef(C).first);
      
    case CXCursor_ObjCClassRef:      
      return MakeCXCursor(getCursorObjCClassRef(C).first);
      
    default:
      // We would prefer to enumerate all non-reference cursor kinds here.
      llvm_unreachable("Unhandled reference cursor kind");
      break;
    }
  }
  
  return clang_getNullCursor();
}

CXCursor clang_getCursorDefinition(CXCursor C) {
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
    return MakeCXCursor(cast<UsingDirectiveDecl>(D)->getNominatedNamespace());

  case Decl::NamespaceAlias:
    return MakeCXCursor(cast<NamespaceAliasDecl>(D)->getNamespace());

  case Decl::Enum:
  case Decl::Record:
  case Decl::CXXRecord:
  case Decl::ClassTemplateSpecialization:
  case Decl::ClassTemplatePartialSpecialization:
    if (TagDecl *Def = cast<TagDecl>(D)->getDefinition(D->getASTContext()))
      return MakeCXCursor(Def);
    return clang_getNullCursor();

  case Decl::Function:
  case Decl::CXXMethod:
  case Decl::CXXConstructor:
  case Decl::CXXDestructor:
  case Decl::CXXConversion: {
    const FunctionDecl *Def = 0;
    if (cast<FunctionDecl>(D)->getBody(Def))
      return MakeCXCursor(const_cast<FunctionDecl *>(Def));
    return clang_getNullCursor();
  }

  case Decl::Var: {
    VarDecl *Var = cast<VarDecl>(D);

    // Variables with initializers have definitions.
    const VarDecl *Def = 0;
    if (Var->getDefinition(Def))
      return MakeCXCursor(const_cast<VarDecl *>(Def));

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
      return MakeCXCursor(Def->getDescribedFunctionTemplate());
    return clang_getNullCursor();
  }
   
  case Decl::ClassTemplate: {
    if (RecordDecl *Def = cast<ClassTemplateDecl>(D)->getTemplatedDecl()
                                          ->getDefinition(D->getASTContext()))
      return MakeCXCursor(
                         cast<CXXRecordDecl>(Def)->getDescribedClassTemplate());
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

      Def = clang_getCursorDefinition(MakeCXCursor((*S)->getTargetDecl()));
    }

    return Def;
  }

  case Decl::UsingShadow:
    return clang_getCursorDefinition(
                       MakeCXCursor(cast<UsingShadowDecl>(D)->getTargetDecl()));

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
            return MakeCXCursor(Def);

    return clang_getNullCursor();
  }

  case Decl::ObjCCategory:
    if (ObjCCategoryImplDecl *Impl
                               = cast<ObjCCategoryDecl>(D)->getImplementation())
      return MakeCXCursor(Impl);
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
      return MakeCXCursor(Impl);
    return clang_getNullCursor();
  
  case Decl::ObjCProperty:
    // FIXME: We don't really know where to find the
    // ObjCPropertyImplDecls that implement this property.
    return clang_getNullCursor();

  case Decl::ObjCCompatibleAlias:
    if (ObjCInterfaceDecl *Class
          = cast<ObjCCompatibleAliasDecl>(D)->getClassInterface())
      if (!Class->isForwardDecl())
        return MakeCXCursor(Class);
    
    return clang_getNullCursor();

  case Decl::ObjCForwardProtocol: {
    ObjCForwardProtocolDecl *Forward = cast<ObjCForwardProtocolDecl>(D);
    if (Forward->protocol_size() == 1)
      return clang_getCursorDefinition(
                                     MakeCXCursor(*Forward->protocol_begin()));

    // FIXME: Cannot return multiple definitions.
    return clang_getNullCursor();
  }

  case Decl::ObjCClass: {
    ObjCClassDecl *Class = cast<ObjCClassDecl>(D);
    if (Class->size() == 1) {
      ObjCInterfaceDecl *IFace = Class->begin()->getInterface();
      if (!IFace->isForwardDecl())
        return MakeCXCursor(IFace);
      return clang_getNullCursor();
    }

    // FIXME: Cannot return multiple definitions.
    return clang_getNullCursor();
  }

  case Decl::Friend:
    if (NamedDecl *Friend = cast<FriendDecl>(D)->getFriendDecl())
      return clang_getCursorDefinition(MakeCXCursor(Friend));
    return clang_getNullCursor();

  case Decl::FriendTemplate:
    if (NamedDecl *Friend = cast<FriendTemplateDecl>(D)->getFriendDecl())
      return clang_getCursorDefinition(MakeCXCursor(Friend));
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
