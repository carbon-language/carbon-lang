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

#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/Program.h"

// Needed to define L_TMPNAM on some systems.
#include <cstdio>

using namespace clang;
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

#if 0
// Will be useful one day.
class CRefVisitor : public StmtVisitor<CRefVisitor> {
  CXDecl CDecl;
  CXDeclIterator Callback;
  CXClientData CData;

  void Call(enum CXCursorKind CK, Stmt *SRef) {
    CXCursor C = { CK, CDecl, SRef };
    Callback(CDecl, C, CData);
  }

public:
  CRefVisitor(CXDecl C, CXDeclIterator cback, CXClientData D) :
    CDecl(C), Callback(cback), CData(D) {}

  void VisitStmt(Stmt *S) {
    for (Stmt::child_iterator C = S->child_begin(), CEnd = S->child_end();
         C != CEnd; ++C)
      Visit(*C);
  }
  void VisitDeclRefExpr(DeclRefExpr *Node) {
    Call(TranslateDeclRefExpr(Node), Node);
  }
  void VisitMemberExpr(MemberExpr *Node) {
    Call(CXCursor_MemberRef, Node);
  }
  void VisitObjCMessageExpr(ObjCMessageExpr *Node) {
    Call(CXCursor_ObjCSelectorRef, Node);
  }
  void VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node) {
    Call(CXCursor_ObjCIvarRef, Node);
  }
};
#endif

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

  void Call(enum CXCursorKind CK, NamedDecl *ND) {
    // Filter any declarations that have a PCH level greater than what we allow.
    if (ND->getPCHLevel() > MaxPCHLevel)
      return;

    // Filter any implicit declarations (since the source info will be bogus).
    if (ND->isImplicit())
      return;

    CXCursor C = { CK, ND, 0, 0 };
    Callback(Root, C, CData);
  }

public:
  TUVisitor(void *root, Iterator cback, CXClientData D, unsigned MaxPCHLevel) :
    Root(root), Callback(cback), CData(D), MaxPCHLevel(MaxPCHLevel) {}

  void VisitDeclContext(DeclContext *DC);
  void VisitFunctionDecl(FunctionDecl *ND);
  void VisitObjCCategoryDecl(ObjCCategoryDecl *ND);
  void VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *ND);
  void VisitObjCImplementationDecl(ObjCImplementationDecl *ND);
  void VisitObjCInterfaceDecl(ObjCInterfaceDecl *ND);
  void VisitObjCProtocolDecl(ObjCProtocolDecl *ND);
  void VisitTagDecl(TagDecl *ND);
  void VisitTranslationUnitDecl(TranslationUnitDecl *D);
  void VisitTypedefDecl(TypedefDecl *ND);
  void VisitVarDecl(VarDecl *ND);
};

void TUVisitor::VisitDeclContext(DeclContext *DC) {
  for (DeclContext::decl_iterator I = DC->decls_begin(), E = DC->decls_end();
       I != E; ++I)
    Visit(*I);
}
  
void TUVisitor::VisitFunctionDecl(FunctionDecl *ND) {
  Call(ND->isThisDeclarationADefinition() ? CXCursor_FunctionDefn
       : CXCursor_FunctionDecl, ND);
}
  
void TUVisitor::VisitObjCCategoryDecl(ObjCCategoryDecl *ND) {
  Call(CXCursor_ObjCCategoryDecl, ND);
}

void TUVisitor::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *ND) {
  Call(CXCursor_ObjCCategoryDefn, ND);
}

void TUVisitor::VisitObjCImplementationDecl(ObjCImplementationDecl *ND) {
  Call(CXCursor_ObjCClassDefn, ND);
}
  
void TUVisitor::VisitObjCInterfaceDecl(ObjCInterfaceDecl *ND) {
  Call(CXCursor_ObjCInterfaceDecl, ND);
}  

void TUVisitor::VisitObjCProtocolDecl(ObjCProtocolDecl *ND) {
  Call(CXCursor_ObjCProtocolDecl, ND);
}
  
void TUVisitor::VisitTagDecl(TagDecl *ND) {
  switch (ND->getTagKind()) {
    case TagDecl::TK_struct:
      Call(CXCursor_StructDecl, ND);
      break;
    case TagDecl::TK_class:
      Call(CXCursor_ClassDecl, ND);
      break;
    case TagDecl::TK_union:
      Call(CXCursor_UnionDecl, ND);
      break;
    case TagDecl::TK_enum:
      Call(CXCursor_EnumDecl, ND);
      break;
  }
}
  
void TUVisitor::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
  VisitDeclContext(dyn_cast<DeclContext>(D));
}
  
void TUVisitor::VisitTypedefDecl(TypedefDecl *ND) {
  Call(CXCursor_TypedefDecl, ND);
}

void TUVisitor::VisitVarDecl(VarDecl *ND) {
  Call(CXCursor_VarDecl, ND);
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

    CXCursor C = { CK, ND, 0, 0 };
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
  Call(CXCursor_ObjCClassRef, ND);
  // FIXME: Issue callbacks for protocol refs.
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
    Call(CXCursor_ObjCSuperClassRef, D);
  
  for (ObjCProtocolDecl::protocol_iterator I = D->protocol_begin(),
       E = D->protocol_end(); I != E; ++I)
    Call(CXCursor_ObjCProtocolRef, *I);
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
  for (ObjCProtocolDecl::protocol_iterator I = PID->protocol_begin(),
       E = PID->protocol_end(); I != E; ++I)
    Call(CXCursor_ObjCProtocolRef, *I);
  
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
    switch (C.kind) {
    case CXCursor_ObjCClassRef: {
      if (isa<ObjCInterfaceDecl>(ND)) {
        // FIXME: This is a hack (storing the parent decl in the stmt slot).
        NamedDecl *parentDecl = static_cast<NamedDecl *>(C.stmt);
        return parentDecl->getLocation();
      }
      ObjCCategoryDecl *OID = dyn_cast<ObjCCategoryDecl>(ND);
      assert(OID && "clang_getCursorLine(): Missing category decl");
      return OID->getClassInterface()->getLocation();
    }
    case CXCursor_ObjCSuperClassRef: {
      ObjCInterfaceDecl *OID = dyn_cast<ObjCInterfaceDecl>(ND);
      assert(OID && "clang_getCursorLine(): Missing interface decl");
      return OID->getSuperClassLoc();
    }
    case CXCursor_ObjCProtocolRef: {
      ObjCProtocolDecl *OID = dyn_cast<ObjCProtocolDecl>(ND);
      assert(OID && "clang_getCursorLine(): Missing protocol decl");
      return OID->getLocation();
    }
    case CXCursor_ObjCSelectorRef: {
      ObjCMessageExpr *OME = dyn_cast<ObjCMessageExpr>(
        static_cast<Stmt *>(C.stmt));
      assert(OME && "clang_getCursorLine(): Missing message expr");
      return OME->getLeftLoc(); /* FIXME: should be a range */
    }
    case CXCursor_VarRef:
    case CXCursor_FunctionRef:
    case CXCursor_EnumConstantRef: {
      DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(
        static_cast<Stmt *>(C.stmt));
      assert(DRE && "clang_getCursorLine(): Missing decl ref expr");
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
  
CXDeclExtent clang_getDeclExtent(CXDecl AnonDecl) {
  assert(AnonDecl && "Passed null CXDecl");
  NamedDecl *ND = static_cast<NamedDecl *>(AnonDecl);
  SourceManager &SM = ND->getASTContext().getSourceManager();
  SourceRange R = ND->getSourceRange();

  SourceLocation Begin = SM.getInstantiationLoc(R.getBegin());
  SourceLocation End = SM.getInstantiationLoc(R.getEnd());

  if (!Begin.isValid()) {
    CXDeclExtent extent = { { 0, 0 }, { 0, 0 } };
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
  CXDeclExtent extent = { { StartLineNo, StartColNo },
                          { EndLineNo, EndColNo } };
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
  assert(SFile && "Passed null CXFile");
  FileEntry *FEnt = static_cast<FileEntry *>(SFile);
  return FEnt->getName();
}

time_t clang_getFileTime(CXFile SFile) {
  assert(SFile && "Passed null CXFile");
  FileEntry *FEnt = static_cast<FileEntry *>(SFile);
  return FEnt->getModificationTime();
}
} // end: extern "C"

//===----------------------------------------------------------------------===//
// CXCursor Operations.
//===----------------------------------------------------------------------===//

static enum CXCursorKind TranslateKind(Decl *D) {
  switch (D->getKind()) {
    case Decl::Function: return CXCursor_FunctionDecl;
    case Decl::Typedef: return CXCursor_TypedefDecl;
    case Decl::Enum: return CXCursor_EnumDecl;
    case Decl::EnumConstant: return CXCursor_EnumConstantDecl;
    case Decl::Record: return CXCursor_StructDecl; // FIXME: union/class
    case Decl::Field: return CXCursor_FieldDecl;
    case Decl::Var: return CXCursor_VarDecl;
    case Decl::ParmVar: return CXCursor_ParmDecl;
    case Decl::ObjCInterface: return CXCursor_ObjCInterfaceDecl;
    case Decl::ObjCCategory: return CXCursor_ObjCCategoryDecl;
    case Decl::ObjCProtocol: return CXCursor_ObjCProtocolDecl;
    case Decl::ObjCMethod: {
      ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D);
      if (MD->isInstanceMethod())
        return CXCursor_ObjCInstanceMethodDecl;
      return CXCursor_ObjCClassMethodDecl;
    }
    default: break;
  }
  return CXCursor_NotImplemented;
}


static CXCursor MakeCXCursor(CXCursorKind K, Decl *D) {
  CXCursor C = { K, D, 0, 0 };
  return C;  
}

static CXCursor MakeCXCursor(CXCursorKind K, Decl *D, Stmt *S) {
  assert(clang_isReference(K));
  CXCursor C = { K, D, S, 0 };
  return C;  
}

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
  assert(C.decl && "CXCursor has null decl");
  NamedDecl *ND = static_cast<NamedDecl *>(C.decl);

  if (clang_isReference(C.kind)) {
    switch (C.kind) {
    case CXCursor_ObjCSuperClassRef: {
      ObjCInterfaceDecl *OID = dyn_cast<ObjCInterfaceDecl>(ND);
      assert(OID && "clang_getCursorLine(): Missing interface decl");
      return CIndexer::createCXString(OID->getSuperClass()->getIdentifier()
                                      ->getNameStart());
    }
    case CXCursor_ObjCClassRef: {
      if (ObjCInterfaceDecl *OID = dyn_cast<ObjCInterfaceDecl>(ND))
        return CIndexer::createCXString(OID->getIdentifier()->getNameStart());

      ObjCCategoryDecl *OCD = dyn_cast<ObjCCategoryDecl>(ND);
      assert(OCD && "clang_getCursorLine(): Missing category decl");
      return CIndexer::createCXString(OCD->getClassInterface()->getIdentifier()
                            ->getNameStart());
    }
    case CXCursor_ObjCProtocolRef: {
      ObjCProtocolDecl *OID = dyn_cast<ObjCProtocolDecl>(ND);
      assert(OID && "clang_getCursorLine(): Missing protocol decl");
      return CIndexer::createCXString(OID->getIdentifier()->getNameStart());
    }
    case CXCursor_ObjCSelectorRef: {
      ObjCMessageExpr *OME = dyn_cast<ObjCMessageExpr>(
        static_cast<Stmt *>(C.stmt));
      assert(OME && "clang_getCursorLine(): Missing message expr");
      return CIndexer::createCXString(OME->getSelector().getAsString().c_str(),
                                      true);
    }
    case CXCursor_VarRef:
    case CXCursor_FunctionRef:
    case CXCursor_EnumConstantRef: {
      DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(
        static_cast<Stmt *>(C.stmt));
      assert(DRE && "clang_getCursorLine(): Missing decl ref expr");
      return CIndexer::createCXString(DRE->getDecl()->getIdentifier()
                                      ->getNameStart());
    }
    default:
      return CIndexer::createCXString("<not implemented>");
    }
  }
  return clang_getDeclSpelling(C.decl);
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
  case CXCursor_FunctionDefn: return "FunctionDefn";
  case CXCursor_ObjCInstanceMethodDefn: return "ObjCInstanceMethodDefn";
  case CXCursor_ObjCClassMethodDefn: return "ObjCClassMethodDefn";
  case CXCursor_ObjCClassDefn: return "ObjCClassDefn";
  case CXCursor_ObjCCategoryDefn: return "ObjCCategoryDefn";
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
  default: return "<not implemented>";
  }
}

CXCursor clang_getCursor(CXTranslationUnit CTUnit, const char *source_name,
                         unsigned line, unsigned column) {
  assert(CTUnit && "Passed null CXTranslationUnit");
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(CTUnit);

  FileManager &FMgr = CXXUnit->getFileManager();
  const FileEntry *File = FMgr.getFile(source_name,
                                       source_name+strlen(source_name));
  if (!File) {
    CXCursor C = { CXCursor_InvalidFile, 0, 0, 0 };
    return C;
  }
  SourceLocation SLoc =
    CXXUnit->getSourceManager().getLocation(File, line, column);

  ASTLocation LastLoc = CXXUnit->getLastASTLocation();

  ASTLocation ALoc = ResolveLocationInAST(CXXUnit->getASTContext(), SLoc,
                                          &LastLoc);
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
      if (isa<ObjCInterfaceDecl>(Dcl)) {
        CXCursor C = { CXCursor_ObjCClassRef, Dcl, ALoc.getParentDecl(), 0 };
        return C;
      }
      if (isa<ObjCProtocolDecl>(Dcl)) {
        CXCursor C = { CXCursor_ObjCProtocolRef, Dcl, ALoc.getParentDecl(), 0 };
        return C;
      }
    }
    return MakeCXCursor(TranslateKind(Dcl), Dcl);
  }
  return MakeCXCursor(CXCursor_NoDeclFound, 0);
}

CXCursor clang_getNullCursor(void) {
  return MakeCXCursor(CXCursor_InvalidFile, 0);
}

unsigned clang_equalCursors(CXCursor X, CXCursor Y) {
  return X.kind == Y.kind && X.decl == Y.decl && X.stmt == Y.stmt &&
         X.referringDecl == Y.referringDecl;
}

CXCursor clang_getCursorFromDecl(CXDecl AnonDecl) {
  assert(AnonDecl && "Passed null CXDecl");
  NamedDecl *ND = static_cast<NamedDecl *>(AnonDecl);
  return MakeCXCursor(TranslateKind(ND), ND);
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
  if (clang_isDeclaration(C.kind))
    return C.decl;

  if (clang_isReference(C.kind)) {
    if (C.stmt) {
      if (C.kind == CXCursor_ObjCClassRef ||
          C.kind == CXCursor_ObjCProtocolRef)
        return static_cast<Stmt *>(C.stmt);
      else
        return getDeclFromExpr(static_cast<Stmt *>(C.stmt));
    } else
      return C.decl;
  }
  return 0;
}

unsigned clang_getCursorLine(CXCursor C) {
  assert(C.decl && "CXCursor has null decl");
  NamedDecl *ND = static_cast<NamedDecl *>(C.decl);
  SourceManager &SourceMgr = ND->getASTContext().getSourceManager();

  SourceLocation SLoc = getLocationFromCursor(C, SourceMgr, ND);
  return SourceMgr.getSpellingLineNumber(SLoc);
}
  
unsigned clang_getCursorColumn(CXCursor C) {
  assert(C.decl && "CXCursor has null decl");
  NamedDecl *ND = static_cast<NamedDecl *>(C.decl);
  SourceManager &SourceMgr = ND->getASTContext().getSourceManager();
  
  SourceLocation SLoc = getLocationFromCursor(C, SourceMgr, ND);
  return SourceMgr.getSpellingColumnNumber(SLoc);
}

const char *clang_getCursorSource(CXCursor C) {
  assert(C.decl && "CXCursor has null decl");
  NamedDecl *ND = static_cast<NamedDecl *>(C.decl);
  SourceManager &SourceMgr = ND->getASTContext().getSourceManager();
  
  SourceLocation SLoc = getLocationFromCursor(C, SourceMgr, ND);
  
  if (SLoc.isFileID()) {
    const char *bufferName = SourceMgr.getBufferName(SLoc);
    return bufferName[0] == '<' ? NULL : bufferName;
  }
  
  // Retrieve the file in which the macro was instantiated, then provide that
  // buffer name.
  // FIXME: Do we want to give specific macro-instantiation information?
  const llvm::MemoryBuffer *Buffer
  = SourceMgr.getBuffer(SourceMgr.getDecomposedSpellingLoc(SLoc).first);
  if (!Buffer)
    return 0;
  
  return Buffer->getBufferIdentifier();
}

CXFile clang_getCursorSourceFile(CXCursor C) {
  assert(C.decl && "CXCursor has null decl");
  NamedDecl *ND = static_cast<NamedDecl *>(C.decl);
  SourceManager &SourceMgr = ND->getASTContext().getSourceManager();
  
  return (void *)
  getFileEntryFromSourceLocation(SourceMgr, getLocationFromCursor(C,SourceMgr,
                                                                  ND));
}

void clang_getDefinitionSpellingAndExtent(CXCursor C,
                                          const char **startBuf,
                                          const char **endBuf,
                                          unsigned *startLine,
                                          unsigned *startColumn,
                                          unsigned *endLine,
                                          unsigned *endColumn) {
  assert(C.decl && "CXCursor has null decl");
  NamedDecl *ND = static_cast<NamedDecl *>(C.decl);
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
