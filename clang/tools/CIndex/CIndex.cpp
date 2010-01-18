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

/// \brief Translate a Clang source location into a CIndex source location.
static CXSourceLocation translateSourceLocation(SourceManager &SourceMgr,
                                                SourceLocation Loc) {
  SourceLocation InstLoc = SourceMgr.getInstantiationLoc(Loc);
  if (InstLoc.isInvalid()) {
      CXSourceLocation Loc = { 0, 0, 0 };
      return Loc;
    }
 
  CXSourceLocation Result;
  Result.file 
    = (void*)SourceMgr.getFileEntryForID(SourceMgr.getFileID(InstLoc));
  Result.line = SourceMgr.getInstantiationLineNumber(InstLoc);
  Result.column = SourceMgr.getInstantiationColumnNumber(InstLoc);
  return Result;
}

//===----------------------------------------------------------------------===//
// Visitors.
//===----------------------------------------------------------------------===//

namespace {
static enum CXCursorKind TranslateDeclRefExpr(DeclRefExpr *DRE) {
  NamedDecl *D = DRE->getDecl();
  if (isa<VarDecl>(D))
    return CXCursor_VarRef;
  else if (isa<FunctionDecl>(D))
    return CXCursor_FunctionRef;
  else if (isa<EnumConstantDecl>(D))
    return CXCursor_EnumConstantRef;
  else
    return CXCursor_NotImplemented;
}

// Translation Unit Visitor.

class TUVisitor : public DeclVisitor<TUVisitor> {
public:
  typedef void (*Iterator)(void *, CXCursor, CXClientData);
private:
  void *Root; // CXDecl or CXTranslationUnit
  Iterator Callback; // CXTranslationUnitIterator or CXDeclIterator.
  CXClientData CData;

  // MaxPCHLevel - the maximum PCH level of declarations that we will pass on
  // to the visitor. Declarations with a PCH level greater than this value will
  // be suppressed.
  unsigned MaxPCHLevel;

  void Call(const CXCursor &C) {
    if (clang_isInvalid(C.kind))
      return;
    
    if (const Decl *D = getCursorDecl(C)) {
      // Filter any declarations that have a PCH level greater than what 
      // we allow.
      if (D->getPCHLevel() > MaxPCHLevel)
        return;

      // Filter any implicit declarations (since the source info will be bogus).
      if (D->isImplicit())
        return;
    }

    Callback(Root, C, CData);
  }

public:
  TUVisitor(void *root, Iterator cback, CXClientData D, unsigned MaxPCHLevel) :
    Root(root), Callback(cback), CData(D), MaxPCHLevel(MaxPCHLevel) {}

  void VisitDecl(Decl *D);
  void VisitDeclContext(DeclContext *DC);
  void VisitTranslationUnitDecl(TranslationUnitDecl *D);
};

void TUVisitor::VisitDecl(Decl *D) {
  Call(MakeCXCursor(D));
}
  
void TUVisitor::VisitDeclContext(DeclContext *DC) {
  for (DeclContext::decl_iterator I = DC->decls_begin(), E = DC->decls_end();
       I != E; ++I)
    Visit(*I);
}
    
void TUVisitor::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
  VisitDeclContext(dyn_cast<DeclContext>(D));
}

// Declaration visitor.
class CDeclVisitor : public DeclVisitor<CDeclVisitor> {
  CXDecl CDecl;
  CXDeclIterator Callback;
  CXClientData CData;

  // MaxPCHLevel - the maximum PCH level of declarations that we will pass on
  // to the visitor. Declarations with a PCH level greater than this value will
  // be suppressed.
  unsigned MaxPCHLevel;

  void Call(enum CXCursorKind CK, NamedDecl *ND) {
    // Disable the callback when the context is equal to the visiting decl.
    if (CDecl == ND && !clang_isReference(CK))
      return;

    // Filter any declarations that have a PCH level greater than what we allow.
    if (ND->getPCHLevel() > MaxPCHLevel)
      return;

    CXCursor C = { CK, { ND, 0, 0 } };
    Callback(CDecl, C, CData);
  }

public:
  CDeclVisitor(CXDecl C, CXDeclIterator cback, CXClientData D,
               unsigned MaxPCHLevel) :
    CDecl(C), Callback(cback), CData(D), MaxPCHLevel(MaxPCHLevel) {}

  void VisitDeclContext(DeclContext *DC);
  void VisitEnumConstantDecl(EnumConstantDecl *ND);
  void VisitFieldDecl(FieldDecl *ND);
  void VisitFunctionDecl(FunctionDecl *ND);
  void VisitObjCCategoryDecl(ObjCCategoryDecl *ND);
  void VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D);
  void VisitObjCImplementationDecl(ObjCImplementationDecl *D);
  void VisitObjCInterfaceDecl(ObjCInterfaceDecl *D);
  void VisitObjCIvarDecl(ObjCIvarDecl *ND);
  void VisitObjCMethodDecl(ObjCMethodDecl *ND);
  void VisitObjCPropertyDecl(ObjCPropertyDecl *ND);
  void VisitObjCProtocolDecl(ObjCProtocolDecl *PID);
  void VisitParmVarDecl(ParmVarDecl *ND);
  void VisitTagDecl(TagDecl *D);
  void VisitVarDecl(VarDecl *ND);
};
} // end anonymous namespace

void CDeclVisitor::VisitDeclContext(DeclContext *DC) {
  for (DeclContext::decl_iterator
       I = DC->decls_begin(), E = DC->decls_end(); I != E; ++I)
    Visit(*I);
}

void CDeclVisitor::VisitEnumConstantDecl(EnumConstantDecl *ND) {
  Call(CXCursor_EnumConstantDecl, ND);
}

void CDeclVisitor::VisitFieldDecl(FieldDecl *ND) {
  Call(CXCursor_FieldDecl, ND);
}

void CDeclVisitor::VisitFunctionDecl(FunctionDecl *ND) {
  if (ND->isThisDeclarationADefinition()) {
    VisitDeclContext(dyn_cast<DeclContext>(ND));
#if 0
    // Not currently needed.
    CompoundStmt *Body = dyn_cast<CompoundStmt>(ND->getBody());
    CRefVisitor RVisit(CDecl, Callback, CData);
    RVisit.Visit(Body);
#endif
  }
}  

void CDeclVisitor::VisitObjCCategoryDecl(ObjCCategoryDecl *ND) {
  // Issue callbacks for the containing class.
  Callback(CDecl, 
           MakeCursorObjCClassRef(ND->getClassInterface(), ND->getLocation()),
           CData);
  ObjCCategoryDecl::protocol_loc_iterator PL = ND->protocol_loc_begin();
  for (ObjCCategoryDecl::protocol_iterator I = ND->protocol_begin(),
         E = ND->protocol_end(); I != E; ++I, ++PL)
    Callback(CDecl, MakeCursorObjCProtocolRef(*I, *PL), CData);
  VisitDeclContext(dyn_cast<DeclContext>(ND));
}

void CDeclVisitor::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
  VisitDeclContext(dyn_cast<DeclContext>(D));
}

void CDeclVisitor::VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
  VisitDeclContext(dyn_cast<DeclContext>(D));
}

void CDeclVisitor::VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
  // Issue callbacks for super class.
  if (D->getSuperClass())
    Callback(CDecl, 
             MakeCursorObjCSuperClassRef(D->getSuperClass(),
                                         D->getSuperClassLoc()), 
             CData);
  
  ObjCInterfaceDecl::protocol_loc_iterator PL = D->protocol_loc_begin();
  for (ObjCInterfaceDecl::protocol_iterator I = D->protocol_begin(),
         E = D->protocol_end(); I != E; ++I, ++PL)
    Callback(CDecl, MakeCursorObjCProtocolRef(*I, *PL), CData);
  VisitDeclContext(dyn_cast<DeclContext>(D));
}

void CDeclVisitor::VisitObjCIvarDecl(ObjCIvarDecl *ND) {
  Call(CXCursor_ObjCIvarDecl, ND);
}

void CDeclVisitor::VisitObjCMethodDecl(ObjCMethodDecl *ND) {
  if (ND->getBody()) {
    Call(ND->isInstanceMethod() ? CXCursor_ObjCInstanceMethodDefn
         : CXCursor_ObjCClassMethodDefn, ND);
    VisitDeclContext(dyn_cast<DeclContext>(ND));
  } else
    Call(ND->isInstanceMethod() ? CXCursor_ObjCInstanceMethodDecl
         : CXCursor_ObjCClassMethodDecl, ND);
}

void CDeclVisitor::VisitObjCPropertyDecl(ObjCPropertyDecl *ND) {
  Call(CXCursor_ObjCPropertyDecl, ND);
}

void CDeclVisitor::VisitObjCProtocolDecl(ObjCProtocolDecl *PID) {
  ObjCProtocolDecl::protocol_loc_iterator PL = PID->protocol_loc_begin();
  for (ObjCProtocolDecl::protocol_iterator I = PID->protocol_begin(),
         E = PID->protocol_end(); I != E; ++I, ++PL)
    Callback(CDecl, MakeCursorObjCProtocolRef(*I, *PL), CData);
  
  VisitDeclContext(dyn_cast<DeclContext>(PID));
}

void CDeclVisitor::VisitParmVarDecl(ParmVarDecl *ND) {
  Call(CXCursor_ParmDecl, ND);
}

void CDeclVisitor::VisitTagDecl(TagDecl *D) {
  VisitDeclContext(dyn_cast<DeclContext>(D));
}

void CDeclVisitor::VisitVarDecl(VarDecl *ND) {
  Call(CXCursor_VarDecl, ND);
}

static SourceLocation getLocationFromCursor(CXCursor C,
                                            SourceManager &SourceMgr,
                                            NamedDecl *ND) {
  if (clang_isReference(C.kind)) {
    
    if (Decl *D = getCursorReferringDecl(C))
      return D->getLocation();
    
    switch (C.kind) {
    case CXCursor_ObjCClassRef:
      return getCursorObjCClassRef(C).second;
    case CXCursor_ObjCSuperClassRef:
      return getCursorObjCSuperClassRef(C).second;
    case CXCursor_ObjCProtocolRef:
      return getCursorObjCProtocolRef(C).second;
    case CXCursor_ObjCSelectorRef: {
      ObjCMessageExpr *OME = dyn_cast<ObjCMessageExpr>(getCursorStmt(C));
      assert(OME && "getLocationFromCursor(): Missing message expr");
      return OME->getLeftLoc(); /* FIXME: should be a range */
    }
    case CXCursor_VarRef:
    case CXCursor_FunctionRef:
    case CXCursor_EnumConstantRef: {
      DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(getCursorStmt(C));
      assert(DRE && "getLocationFromCursor(): Missing decl ref expr");
      return DRE->getLocation();
    }
    default:
      return SourceLocation();
    }
  } else { // We have a declaration or a definition.
    SourceLocation SLoc;
    switch (ND->getKind()) {
    case Decl::ObjCInterface: {
      SLoc = dyn_cast<ObjCInterfaceDecl>(ND)->getClassLoc();
      break;
    }
    case Decl::ObjCProtocol: {
      SLoc = ND->getLocation(); /* FIXME: need to get the name location. */
      break;
    }
    default: {
      SLoc = ND->getLocation();
      break;
    }
    }
    if (SLoc.isInvalid())
      return SourceLocation();
    return SourceMgr.getSpellingLoc(SLoc); // handles macro instantiations.
  }
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

void clang_loadTranslationUnit(CXTranslationUnit CTUnit,
                               CXTranslationUnitIterator callback,
                               CXClientData CData) {
  assert(CTUnit && "Passed null CXTranslationUnit");
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(CTUnit);
  ASTContext &Ctx = CXXUnit->getASTContext();

  unsigned PCHLevel = Decl::MaxPCHLevel;

  // Set the PCHLevel to filter out unwanted decls if requested.
  if (CXXUnit->getOnlyLocalDecls()) {
    PCHLevel = 0;

    // If the main input was an AST, bump the level.
    if (CXXUnit->isMainFileAST())
      ++PCHLevel;
  }

  TUVisitor DVisit(CTUnit, callback, CData, PCHLevel);

  // If using a non-AST based ASTUnit, iterate over the stored list of top-level
  // decls.
  if (!CXXUnit->isMainFileAST() && CXXUnit->getOnlyLocalDecls()) {
    const std::vector<Decl*> &TLDs = CXXUnit->getTopLevelDecls();
    for (std::vector<Decl*>::const_iterator it = TLDs.begin(),
           ie = TLDs.end(); it != ie; ++it) {
      DVisit.Visit(*it);
    }
  } else
    DVisit.Visit(Ctx.getTranslationUnitDecl());
}

void clang_loadDeclaration(CXDecl Dcl,
                           CXDeclIterator callback,
                           CXClientData CData) {
  assert(Dcl && "Passed null CXDecl");

  CDeclVisitor DVisit(Dcl, callback, CData,
                      static_cast<Decl *>(Dcl)->getPCHLevel());
  DVisit.Visit(static_cast<Decl *>(Dcl));
}
} // end: extern "C"

//===----------------------------------------------------------------------===//
// CXDecl Operations.
//===----------------------------------------------------------------------===//

static const FileEntry *getFileEntryFromSourceLocation(SourceManager &SMgr,
                                                       SourceLocation SLoc) {
  FileID FID;
  if (SLoc.isFileID())
    FID = SMgr.getFileID(SLoc);
  else
    FID = SMgr.getDecomposedSpellingLoc(SLoc).first;
  return SMgr.getFileEntryForID(FID);
}

extern "C" {
CXString clang_getDeclSpelling(CXDecl AnonDecl) {
  assert(AnonDecl && "Passed null CXDecl");
  NamedDecl *ND = static_cast<NamedDecl *>(AnonDecl);

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

unsigned clang_getDeclLine(CXDecl AnonDecl) {
  assert(AnonDecl && "Passed null CXDecl");
  NamedDecl *ND = static_cast<NamedDecl *>(AnonDecl);
  SourceManager &SourceMgr = ND->getASTContext().getSourceManager();
  return SourceMgr.getSpellingLineNumber(ND->getLocation());
}

unsigned clang_getDeclColumn(CXDecl AnonDecl) {
  assert(AnonDecl && "Passed null CXDecl");
  NamedDecl *ND = static_cast<NamedDecl *>(AnonDecl);
  SourceManager &SourceMgr = ND->getASTContext().getSourceManager();
  return SourceMgr.getSpellingColumnNumber(ND->getLocation());
}
  
CXSourceRange clang_getDeclExtent(CXDecl AnonDecl) {
  assert(AnonDecl && "Passed null CXDecl");
  NamedDecl *ND = static_cast<NamedDecl *>(AnonDecl);
  SourceManager &SM = ND->getASTContext().getSourceManager();
  SourceRange R = ND->getSourceRange();

  SourceLocation Begin = SM.getInstantiationLoc(R.getBegin());
  SourceLocation End = SM.getInstantiationLoc(R.getEnd());

  if (!Begin.isValid()) {
    CXSourceRange extent = { { 0, 0, 0 }, { 0, 0, 0 } };
    return extent;
  }
  
  // FIXME: This is largely copy-paste from
  ///TextDiagnosticPrinter::HighlightRange.  When it is clear that this is
  // what we want the two routines should be refactored.
  
  // If the End location and the start location are the same and are a macro
  // location, then the range was something that came from a macro expansion
  // or _Pragma.  If this is an object-like macro, the best we can do is to
  // get the range.  If this is a function-like macro, we'd also like to
  // get the arguments.
  if (Begin == End && R.getEnd().isMacroID())
    End = SM.getInstantiationRange(R.getEnd()).second;

  assert(SM.getFileID(Begin) == SM.getFileID(End));
  unsigned StartLineNo = SM.getInstantiationLineNumber(Begin);  
  unsigned EndLineNo = SM.getInstantiationLineNumber(End);
  
  // Compute the column number of the start.  Keep the column based at 1.
  unsigned StartColNo = SM.getInstantiationColumnNumber(Begin);
  
  // Compute the column number of the end.
  unsigned EndColNo = SM.getInstantiationColumnNumber(End);
  if (EndColNo) {
    // Offset the end column by 1 so that we point to the last character
    // in the last token.
    --EndColNo;
    
    // Add in the length of the token, so that we cover multi-char tokens.
    ASTContext &Ctx = ND->getTranslationUnitDecl()->getASTContext();
    const LangOptions &LOpts = Ctx.getLangOptions();

    EndColNo += Lexer::MeasureTokenLength(End, SM, LOpts);
  }

  // Package up the line/column data and return to the caller.
  const FileEntry *FEntry = SM.getFileEntryForID(SM.getFileID(Begin));
  CXSourceRange extent = { { (void *)FEntry, StartLineNo, StartColNo },
                           { (void *)FEntry, EndLineNo, EndColNo } };
  return extent;  
}

const char *clang_getDeclSource(CXDecl AnonDecl) {
  assert(AnonDecl && "Passed null CXDecl");
  FileEntry *FEnt = static_cast<FileEntry *>(clang_getDeclSourceFile(AnonDecl));
  assert(FEnt && "Cannot find FileEntry for Decl");
  return clang_getFileName(FEnt);
}


CXFile clang_getDeclSourceFile(CXDecl AnonDecl) {
  assert(AnonDecl && "Passed null CXDecl");
  NamedDecl *ND = static_cast<NamedDecl *>(AnonDecl);
  SourceManager &SourceMgr = ND->getASTContext().getSourceManager();
  return (void *)getFileEntryFromSourceLocation(SourceMgr, ND->getLocation());
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
CXString clang_getCursorSpelling(CXCursor C) {
  assert(getCursorDecl(C) && "CXCursor has null decl");
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
      assert(OID && "getLocationFromCursor(): Missing protocol decl");
      return CIndexer::createCXString(OID->getIdentifier()->getNameStart());
    }
    case CXCursor_ObjCSelectorRef: {
      ObjCMessageExpr *OME = dyn_cast<ObjCMessageExpr>(getCursorStmt(C));
      assert(OME && "getLocationFromCursor(): Missing message expr");
      return CIndexer::createCXString(OME->getSelector().getAsString().c_str(),
                                      true);
    }
    case CXCursor_VarRef:
    case CXCursor_FunctionRef:
    case CXCursor_EnumConstantRef: {
      DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(getCursorStmt(C));
      assert(DRE && "getLocationFromCursor(): Missing decl ref expr");
      return CIndexer::createCXString(DRE->getDecl()->getIdentifier()
                                      ->getNameStart());
    }
    default:
      return CIndexer::createCXString("<not implemented>");
    }
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
  case CXCursor_FunctionDefn: return "FunctionDefn";
  case CXCursor_VarDecl: return "VarDecl";
  case CXCursor_ParmDecl: return "ParmDecl";
  case CXCursor_ObjCInterfaceDecl: return "ObjCInterfaceDecl";
  case CXCursor_ObjCCategoryDecl: return "ObjCCategoryDecl";
  case CXCursor_ObjCProtocolDecl: return "ObjCProtocolDecl";
  case CXCursor_ObjCPropertyDecl: return "ObjCPropertyDecl";
  case CXCursor_ObjCIvarDecl: return "ObjCIvarDecl";
  case CXCursor_ObjCIvarRef: return "ObjCIvarRef";
  case CXCursor_ObjCInstanceMethodDecl: return "ObjCInstanceMethodDecl";
  case CXCursor_ObjCClassMethodDecl: return "ObjCClassMethodDecl";
  case CXCursor_ObjCInstanceMethodDefn: return "ObjCInstanceMethodDefn";
  case CXCursor_ObjCCategoryDefn: return "ObjCCategoryDefn";
  case CXCursor_ObjCClassMethodDefn: return "ObjCClassMethodDefn";
  case CXCursor_ObjCClassDefn: return "ObjCClassDefn";
  case CXCursor_ObjCSuperClassRef: return "ObjCSuperClassRef";
  case CXCursor_ObjCProtocolRef: return "ObjCProtocolRef";
  case CXCursor_ObjCClassRef: return "ObjCClassRef";
  case CXCursor_ObjCSelectorRef: return "ObjCSelectorRef";

  case CXCursor_VarRef: return "VarRef";
  case CXCursor_FunctionRef: return "FunctionRef";
  case CXCursor_EnumConstantRef: return "EnumConstantRef";
  case CXCursor_MemberRef: return "MemberRef";

  case CXCursor_InvalidFile: return "InvalidFile";
  case CXCursor_NoDeclFound: return "NoDeclFound";
  case CXCursor_NotImplemented: return "NotImplemented";
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
    if (Stm) {
      if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Stm))
        return MakeCXCursor(TranslateDeclRefExpr(DRE), Dcl, Stm);
      else if (ObjCMessageExpr *MExp = dyn_cast<ObjCMessageExpr>(Stm))
        return MakeCXCursor(CXCursor_ObjCSelectorRef, Dcl, MExp);
      // Fall through...treat as a decl, not a ref.
    }
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

unsigned clang_isDefinition(enum CXCursorKind K) {
  return K >= CXCursor_FirstDefn && K <= CXCursor_LastDefn;
}

CXCursorKind clang_getCursorKind(CXCursor C) {
  return C.kind;
}

CXDecl clang_getCursorDecl(CXCursor C) {
  if (clang_isDeclaration(C.kind) || clang_isDefinition(C.kind))
    return getCursorDecl(C);

  if (clang_isReference(C.kind)) {
    if (getCursorStmt(C))
      return getDeclFromExpr(getCursorStmt(C));

    return getCursorDecl(C);
  }
  return 0;
}

unsigned clang_getCursorLine(CXCursor C) {
  return clang_getCursorLocation(C).line;
}
  
unsigned clang_getCursorColumn(CXCursor C) {
  return clang_getCursorLocation(C).column;
}

const char *clang_getCursorSource(CXCursor C) {
  return clang_getFileName(clang_getCursorLocation(C).file);
}

CXFile clang_getCursorSourceFile(CXCursor C) {
  return clang_getCursorLocation(C).file;
}

CXSourceLocation clang_getCursorLocation(CXCursor C) {
  if (clang_isReference(C.kind)) {
    // FIXME: Return the location of the reference, not of the underlying
    // declaration (which may not even exist!).
  }
  
  if (!getCursorDecl(C)) {
    CXSourceLocation empty = { 0, 0, 0 };
    return empty;
  }

  NamedDecl *ND = static_cast<NamedDecl *>(getCursorDecl(C));
  SourceManager &SM = ND->getASTContext().getSourceManager();
  
  return translateSourceLocation(SM, getLocationFromCursor(C, SM, ND));
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
