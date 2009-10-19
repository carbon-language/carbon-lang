//===- CIndex.cpp - Clang-C Source Indexing Library -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Clang-C Source Indexing library.
//
//===----------------------------------------------------------------------===//

#include "clang-c/Index.h"
#include "clang/Index/Program.h"
#include "clang/Index/Indexer.h"
#include "clang/Index/ASTLocation.h"
#include "clang/Index/Utils.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/ASTUnit.h"
#include "llvm/Config/config.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/Path.h"
#include "llvm/System/Program.h"
#include <cstdio>
#ifdef LLVM_ON_WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <vector>

using namespace clang;
using namespace idx;

namespace {

static enum CXCursorKind TranslateDeclRefExpr(DeclRefExpr *DRE) 
{
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
  CXTranslationUnit TUnit;
  CXTranslationUnitIterator Callback;
  CXClientData CData;
  
  // MaxPCHLevel - the maximum PCH level of declarations that we will pass on
  // to the visitor. Declarations with a PCH level greater than this value will
  // be suppressed.
  unsigned MaxPCHLevel;
  
  void Call(enum CXCursorKind CK, NamedDecl *ND) {
    // Filter any declarations that have a PCH level greater than what we allow.
    if (ND->getPCHLevel() > MaxPCHLevel)
      return;
    
    CXCursor C = { CK, ND, 0 };
    Callback(TUnit, C, CData);
  }
public:
  TUVisitor(CXTranslationUnit CTU, 
            CXTranslationUnitIterator cback, CXClientData D,
            unsigned MaxPCHLevel) : 
    TUnit(CTU), Callback(cback), CData(D), MaxPCHLevel(MaxPCHLevel) {}
  
  void VisitTranslationUnitDecl(TranslationUnitDecl *D) {
    VisitDeclContext(dyn_cast<DeclContext>(D));
  }
  void VisitDeclContext(DeclContext *DC) {
    for (DeclContext::decl_iterator
           I = DC->decls_begin(), E = DC->decls_end(); I != E; ++I)
      Visit(*I);
  }
  void VisitTypedefDecl(TypedefDecl *ND) { 
    Call(CXCursor_TypedefDecl, ND); 
  }
  void VisitTagDecl(TagDecl *ND) {
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
  void VisitVarDecl(VarDecl *ND) {
    Call(CXCursor_VarDecl, ND);
  }
  void VisitFunctionDecl(FunctionDecl *ND) {
    Call(ND->isThisDeclarationADefinition() ? CXCursor_FunctionDefn
                                            : CXCursor_FunctionDecl, ND);
  }
  void VisitObjCInterfaceDecl(ObjCInterfaceDecl *ND) {
    Call(CXCursor_ObjCInterfaceDecl, ND);
  }
  void VisitObjCCategoryDecl(ObjCCategoryDecl *ND) {
    Call(CXCursor_ObjCCategoryDecl, ND);
  }
  void VisitObjCProtocolDecl(ObjCProtocolDecl *ND) {
    Call(CXCursor_ObjCProtocolDecl, ND);
  }
  void VisitObjCImplementationDecl(ObjCImplementationDecl *ND) {
    Call(CXCursor_ObjCClassDefn, ND);
  }
  void VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *ND) {
    Call(CXCursor_ObjCCategoryDefn, ND);
  }
};

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
    
    CXCursor C = { CK, ND, 0 };
    Callback(CDecl, C, CData);
  }
public:
  CDeclVisitor(CXDecl C, CXDeclIterator cback, CXClientData D, 
               unsigned MaxPCHLevel) : 
    CDecl(C), Callback(cback), CData(D), MaxPCHLevel(MaxPCHLevel) {}
    
  void VisitObjCCategoryDecl(ObjCCategoryDecl *ND) {
    // Issue callbacks for the containing class.
    Call(CXCursor_ObjCClassRef, ND);
    // FIXME: Issue callbacks for protocol refs.
    VisitDeclContext(dyn_cast<DeclContext>(ND));
  }
  void VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
    // Issue callbacks for super class.
    if (D->getSuperClass())
      Call(CXCursor_ObjCSuperClassRef, D);

    for (ObjCProtocolDecl::protocol_iterator I = D->protocol_begin(), 
         E = D->protocol_end(); I != E; ++I)
      Call(CXCursor_ObjCProtocolRef, *I);
    VisitDeclContext(dyn_cast<DeclContext>(D));
  }
  void VisitObjCProtocolDecl(ObjCProtocolDecl *PID) {
    for (ObjCProtocolDecl::protocol_iterator I = PID->protocol_begin(), 
         E = PID->protocol_end(); I != E; ++I)
      Call(CXCursor_ObjCProtocolRef, *I);
      
    VisitDeclContext(dyn_cast<DeclContext>(PID));
  }
  void VisitTagDecl(TagDecl *D) {
    VisitDeclContext(dyn_cast<DeclContext>(D));
  }
  void VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
    VisitDeclContext(dyn_cast<DeclContext>(D));
  }
  void VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
    VisitDeclContext(dyn_cast<DeclContext>(D));
  }
  void VisitDeclContext(DeclContext *DC) {
    for (DeclContext::decl_iterator
           I = DC->decls_begin(), E = DC->decls_end(); I != E; ++I)
      Visit(*I);
  }
  void VisitEnumConstantDecl(EnumConstantDecl *ND) {
    Call(CXCursor_EnumConstantDecl, ND);
  }
  void VisitFieldDecl(FieldDecl *ND) {
    Call(CXCursor_FieldDecl, ND);
  }
  void VisitVarDecl(VarDecl *ND) {
    Call(CXCursor_VarDecl, ND);
  }
  void VisitParmVarDecl(ParmVarDecl *ND) {
    Call(CXCursor_ParmDecl, ND);
  }
  void VisitObjCPropertyDecl(ObjCPropertyDecl *ND) {
    Call(CXCursor_ObjCPropertyDecl, ND);
  }
  void VisitObjCIvarDecl(ObjCIvarDecl *ND) {
    Call(CXCursor_ObjCIvarDecl, ND);
  }
  void VisitFunctionDecl(FunctionDecl *ND) {
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
  void VisitObjCMethodDecl(ObjCMethodDecl *ND) {
    if (ND->getBody()) {
      Call(ND->isInstanceMethod() ? CXCursor_ObjCInstanceMethodDefn
                                  : CXCursor_ObjCClassMethodDefn, ND);
      VisitDeclContext(dyn_cast<DeclContext>(ND));
    } else
      Call(ND->isInstanceMethod() ? CXCursor_ObjCInstanceMethodDecl
                                  : CXCursor_ObjCClassMethodDecl, ND);
  }
};

class CIndexer : public Indexer {
public:  
  explicit CIndexer(Program *prog) : Indexer(*prog), OnlyLocalDecls(false) {}

  virtual ~CIndexer() { delete &getProgram(); }

  /// \brief Whether we only want to see "local" declarations (that did not
  /// come from a previous precompiled header). If false, we want to see all
  /// declarations.
  bool getOnlyLocalDecls() const { return OnlyLocalDecls; }
  void setOnlyLocalDecls(bool Local = true) { OnlyLocalDecls = Local; }

  /// \brief Get the path of the clang binary.
  const llvm::sys::Path& getClangPath();
private:
  bool OnlyLocalDecls;
  llvm::sys::Path ClangPath;
};

const llvm::sys::Path& CIndexer::getClangPath() {
  // Did we already compute the path?
  if (!ClangPath.empty())
    return ClangPath;

  // Find the location where this library lives (libCIndex.dylib).
#ifdef LLVM_ON_WIN32
  MEMORY_BASIC_INFORMATION mbi;
  char path[MAX_PATH];
  VirtualQuery((void *)(uintptr_t)clang_createTranslationUnit, &mbi,
               sizeof(mbi));
  GetModuleFileNameA((HINSTANCE)mbi.AllocationBase, path, MAX_PATH);

  llvm::sys::Path CIndexPath(path);
#else
  // This silly cast below avoids a C++ warning.
  Dl_info info;
  if (dladdr((void *)(uintptr_t)clang_createTranslationUnit, &info) == 0)
    assert(0 && "Call to dladdr() failed");

  llvm::sys::Path CIndexPath(info.dli_fname);
#endif

  // We now have the CIndex directory, locate clang relative to it.
  CIndexPath.eraseComponent();
  CIndexPath.eraseComponent();
  CIndexPath.appendComponent("bin");
  CIndexPath.appendComponent("clang");

  // Cache our result.
  ClangPath = CIndexPath;
  return ClangPath;
}

}

extern "C" {

CXIndex clang_createIndex() 
{  
  return new CIndexer(new Program());
}

void clang_disposeIndex(CXIndex CIdx)
{
  assert(CIdx && "Passed null CXIndex");
  delete static_cast<CIndexer *>(CIdx);
}

// FIXME: need to pass back error info.
CXTranslationUnit clang_createTranslationUnit(
  CXIndex CIdx, const char *ast_filename) 
{
  assert(CIdx && "Passed null CXIndex");
  CIndexer *CXXIdx = static_cast<CIndexer *>(CIdx);
  std::string astName(ast_filename);
  std::string ErrMsg;
  
  return ASTUnit::LoadFromPCHFile(astName, CXXIdx->getDiagnostics(),
                                  CXXIdx->getFileManager(), &ErrMsg,
                                  CXXIdx->getOnlyLocalDecls(),
                                  /* UseBumpAllocator = */ true);
}

CXTranslationUnit clang_createTranslationUnitFromSourceFile(
  CXIndex CIdx, 
  const char *source_filename,
  int num_command_line_args, const char **command_line_args) 
{
  // Build up the arguments for involing clang.
  llvm::sys::Path ClangPath = static_cast<CIndexer *>(CIdx)->getClangPath();
  std::vector<const char *> argv;
  argv.push_back(ClangPath.c_str());
  argv.push_back("-emit-ast");
  argv.push_back(source_filename);
  argv.push_back("-o");
  // Generate a temporary name for the AST file.
  char astTmpFile[L_tmpnam];
  argv.push_back(tmpnam(astTmpFile));
  for (int i = 0; i < num_command_line_args; i++)
    argv.push_back(command_line_args[i]);
  argv.push_back(NULL);

  // Generate the AST file in a separate process.
  llvm::sys::Program::ExecuteAndWait(ClangPath, &argv[0]);

  // Finally, we create the translation unit from the ast file.
  ASTUnit *ATU = static_cast<ASTUnit *>(
                   clang_createTranslationUnit(CIdx, astTmpFile));
  ATU->unlinkTemporaryFile();
  return ATU;
}

void clang_disposeTranslationUnit(
  CXTranslationUnit CTUnit) 
{
  assert(CTUnit && "Passed null CXTranslationUnit");
  delete static_cast<ASTUnit *>(CTUnit);
}

void clang_wantOnlyLocalDeclarations(CXIndex CIdx) {
  static_cast<CIndexer *>(CIdx)->setOnlyLocalDecls(true);
}
  
const char *clang_getTranslationUnitSpelling(CXTranslationUnit CTUnit)
{
  assert(CTUnit && "Passed null CXTranslationUnit");
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(CTUnit);
  return CXXUnit->getOriginalSourceFileName().c_str();
}

void clang_loadTranslationUnit(CXTranslationUnit CTUnit, 
                               CXTranslationUnitIterator callback,
                               CXClientData CData)
{
  assert(CTUnit && "Passed null CXTranslationUnit");
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(CTUnit);
  ASTContext &Ctx = CXXUnit->getASTContext();
  
  TUVisitor DVisit(CTUnit, callback, CData, 
                   CXXUnit->getOnlyLocalDecls()? 1 : Decl::MaxPCHLevel);
  DVisit.Visit(Ctx.getTranslationUnitDecl());
}

void clang_loadDeclaration(CXDecl Dcl, 
                           CXDeclIterator callback, 
                           CXClientData CData)
{
  assert(Dcl && "Passed null CXDecl");
  
  CDeclVisitor DVisit(Dcl, callback, CData,
                      static_cast<Decl *>(Dcl)->getPCHLevel());
  DVisit.Visit(static_cast<Decl *>(Dcl));
}

// Some notes on CXEntity:
//
// - Since the 'ordinary' namespace includes functions, data, typedefs, 
// ObjC interfaces, thecurrent algorithm is a bit naive (resulting in one 
// entity for 2 different types). For example:
//
// module1.m: @interface Foo @end Foo *x;
// module2.m: void Foo(int);
//
// - Since the unique name spans translation units, static data/functions 
// within a CXTranslationUnit are *not* currently represented by entities.
// As a result, there will be no entity for the following:
//
// module.m: static void Foo() { }
//


const char *clang_getDeclarationName(CXEntity)
{
  return "";
}
const char *clang_getURI(CXEntity)
{
  return "";
}

CXEntity clang_getEntity(const char *URI)
{
  return 0;
}

//
// CXDecl Operations.
//
CXEntity clang_getEntityFromDecl(CXDecl)
{
  return 0;
}
const char *clang_getDeclSpelling(CXDecl AnonDecl)
{
  assert(AnonDecl && "Passed null CXDecl");
  NamedDecl *ND = static_cast<NamedDecl *>(AnonDecl);
  
  if (ObjCMethodDecl *OMD = dyn_cast<ObjCMethodDecl>(ND)) {
    return OMD->getSelector().getAsString().c_str();
  }    
  if (ND->getIdentifier())
    return ND->getIdentifier()->getNameStart();
  else 
    return "";
}

unsigned clang_getDeclLine(CXDecl AnonDecl)
{
  assert(AnonDecl && "Passed null CXDecl");
  NamedDecl *ND = static_cast<NamedDecl *>(AnonDecl);
  SourceManager &SourceMgr = ND->getASTContext().getSourceManager();
  return SourceMgr.getSpellingLineNumber(ND->getLocation());
}

unsigned clang_getDeclColumn(CXDecl AnonDecl)
{
  assert(AnonDecl && "Passed null CXDecl");
  NamedDecl *ND = static_cast<NamedDecl *>(AnonDecl);
  SourceManager &SourceMgr = ND->getASTContext().getSourceManager();
  return SourceMgr.getSpellingColumnNumber(ND->getLocation());
}

const char *clang_getDeclSource(CXDecl AnonDecl) 
{
  assert(AnonDecl && "Passed null CXDecl");
  NamedDecl *ND = static_cast<NamedDecl *>(AnonDecl);
  SourceManager &SourceMgr = ND->getASTContext().getSourceManager();
  return SourceMgr.getBufferName(ND->getLocation());
}

const char *clang_getCursorSpelling(CXCursor C)
{
  assert(C.decl && "CXCursor has null decl");
  NamedDecl *ND = static_cast<NamedDecl *>(C.decl);
  
  if (clang_isReference(C.kind)) {
    switch (C.kind) {
      case CXCursor_ObjCSuperClassRef: 
        {
        ObjCInterfaceDecl *OID = dyn_cast<ObjCInterfaceDecl>(ND);
        assert(OID && "clang_getCursorLine(): Missing interface decl");
        return OID->getSuperClass()->getIdentifier()->getNameStart();
        }
      case CXCursor_ObjCClassRef: 
        {
        if (ObjCInterfaceDecl *OID = dyn_cast<ObjCInterfaceDecl>(ND)) {
          return OID->getIdentifier()->getNameStart();
        }
        ObjCCategoryDecl *OID = dyn_cast<ObjCCategoryDecl>(ND);
        assert(OID && "clang_getCursorLine(): Missing category decl");
        return OID->getClassInterface()->getIdentifier()->getNameStart();
        }
      case CXCursor_ObjCProtocolRef: 
        {
        ObjCProtocolDecl *OID = dyn_cast<ObjCProtocolDecl>(ND);
        assert(OID && "clang_getCursorLine(): Missing protocol decl");
        return OID->getIdentifier()->getNameStart();
        }
      case CXCursor_ObjCSelectorRef:
        {
        ObjCMessageExpr *OME = dyn_cast<ObjCMessageExpr>(
                                 static_cast<Stmt *>(C.stmt));
        assert(OME && "clang_getCursorLine(): Missing message expr");
        return OME->getSelector().getAsString().c_str();
        }
      case CXCursor_VarRef:
      case CXCursor_FunctionRef:
      case CXCursor_EnumConstantRef:
        {
        DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(
                                 static_cast<Stmt *>(C.stmt));
        assert(DRE && "clang_getCursorLine(): Missing decl ref expr");
        return DRE->getDecl()->getIdentifier()->getNameStart();
        }
      default:
        return "<not implemented>";
    }
  }
  return clang_getDeclSpelling(C.decl);
}

const char *clang_getCursorKindSpelling(enum CXCursorKind Kind)
{
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
//
// CXCursor Operations.
//
CXCursor clang_getCursor(CXTranslationUnit CTUnit, const char *source_name, 
                         unsigned line, unsigned column)
{
  assert(CTUnit && "Passed null CXTranslationUnit");
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(CTUnit);
  
  FileManager &FMgr = CXXUnit->getFileManager();
  const FileEntry *File = FMgr.getFile(source_name, 
                                       source_name+strlen(source_name));  
  if (!File) {
    CXCursor C = { CXCursor_InvalidFile, 0, 0 };
    return C;
  }
  SourceLocation SLoc = 
    CXXUnit->getSourceManager().getLocation(File, line, column);
                                                                
  ASTLocation ALoc = ResolveLocationInAST(CXXUnit->getASTContext(), SLoc);
  
  Decl *Dcl = ALoc.getParentDecl();
  if (ALoc.isNamedRef())
    Dcl = ALoc.AsNamedRef().ND;
  Stmt *Stm = ALoc.dyn_AsStmt();
  if (Dcl) {
    if (Stm) {
      if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Stm)) {
        CXCursor C = { TranslateDeclRefExpr(DRE), Dcl, Stm };
        return C;
      } else if (ObjCMessageExpr *MExp = dyn_cast<ObjCMessageExpr>(Stm)) {
        CXCursor C = { CXCursor_ObjCSelectorRef, Dcl, MExp };
        return C;
      } 
      // Fall through...treat as a decl, not a ref.
    }
    if (ALoc.isNamedRef()) {
      if (isa<ObjCInterfaceDecl>(Dcl)) {
        CXCursor C = { CXCursor_ObjCClassRef, Dcl, ALoc.getParentDecl() };
        return C;
      }
      if (isa<ObjCProtocolDecl>(Dcl)) {
        CXCursor C = { CXCursor_ObjCProtocolRef, Dcl, ALoc.getParentDecl() };
        return C;
      }
    }
	  CXCursor C = { TranslateKind(Dcl), Dcl, 0 };
    return C;
  }
  CXCursor C = { CXCursor_NoDeclFound, 0, 0 };
  return C;
}

CXCursor clang_getCursorFromDecl(CXDecl AnonDecl)
{
  assert(AnonDecl && "Passed null CXDecl");
  NamedDecl *ND = static_cast<NamedDecl *>(AnonDecl);
  
  CXCursor C = { TranslateKind(ND), ND, 0 };
  return C;
}

unsigned clang_isInvalid(enum CXCursorKind K)
{
  return K >= CXCursor_FirstInvalid && K <= CXCursor_LastInvalid;
}

unsigned clang_isDeclaration(enum CXCursorKind K)
{
  return K >= CXCursor_FirstDecl && K <= CXCursor_LastDecl;
}

unsigned clang_isReference(enum CXCursorKind K)
{
  return K >= CXCursor_FirstRef && K <= CXCursor_LastRef;
}

unsigned clang_isDefinition(enum CXCursorKind K)
{
  return K >= CXCursor_FirstDefn && K <= CXCursor_LastDefn;
}

CXCursorKind clang_getCursorKind(CXCursor C)
{
  return C.kind;
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

CXDecl clang_getCursorDecl(CXCursor C) 
{
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

  
static SourceLocation getLocationFromCursor(CXCursor C, 
                                            SourceManager &SourceMgr,
                                            NamedDecl *ND) {
  if (clang_isReference(C.kind)) {
    switch (C.kind) {
      case CXCursor_ObjCClassRef: 
        {
        if (isa<ObjCInterfaceDecl>(ND)) {
          // FIXME: This is a hack (storing the parent decl in the stmt slot).
          NamedDecl *parentDecl = static_cast<NamedDecl *>(C.stmt);
          return parentDecl->getLocation();
        }
        ObjCCategoryDecl *OID = dyn_cast<ObjCCategoryDecl>(ND);
        assert(OID && "clang_getCursorLine(): Missing category decl");
        return OID->getClassInterface()->getLocation();
        }
      case CXCursor_ObjCSuperClassRef: 
        {
        ObjCInterfaceDecl *OID = dyn_cast<ObjCInterfaceDecl>(ND);
        assert(OID && "clang_getCursorLine(): Missing interface decl");
        return OID->getSuperClassLoc();
        }
      case CXCursor_ObjCProtocolRef: 
        {
        ObjCProtocolDecl *OID = dyn_cast<ObjCProtocolDecl>(ND);
        assert(OID && "clang_getCursorLine(): Missing protocol decl");
        return OID->getLocation();
        }
      case CXCursor_ObjCSelectorRef:
        {
        ObjCMessageExpr *OME = dyn_cast<ObjCMessageExpr>(
                                 static_cast<Stmt *>(C.stmt));
        assert(OME && "clang_getCursorLine(): Missing message expr");
        return OME->getLeftLoc(); /* FIXME: should be a range */
        }
      case CXCursor_VarRef:
      case CXCursor_FunctionRef:
      case CXCursor_EnumConstantRef:
        {
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
      case Decl::ObjCInterface: 
        {
        SLoc = dyn_cast<ObjCInterfaceDecl>(ND)->getClassLoc();
        break;
        }
      case Decl::ObjCProtocol: 
        {
        SLoc = ND->getLocation(); /* FIXME: need to get the name location. */
        break;
        }
      default: 
        {
        SLoc = ND->getLocation();
        break;
        }
    }
    if (SLoc.isInvalid())
      return SourceLocation();
    return SourceMgr.getSpellingLoc(SLoc); // handles macro instantiations.
  }
}

unsigned clang_getCursorLine(CXCursor C)
{
  assert(C.decl && "CXCursor has null decl");
  NamedDecl *ND = static_cast<NamedDecl *>(C.decl);
  SourceManager &SourceMgr = ND->getASTContext().getSourceManager();
  
  SourceLocation SLoc = getLocationFromCursor(C, SourceMgr, ND);
  return SourceMgr.getSpellingLineNumber(SLoc);
}

unsigned clang_getCursorColumn(CXCursor C)
{
  assert(C.decl && "CXCursor has null decl");
  NamedDecl *ND = static_cast<NamedDecl *>(C.decl);
  SourceManager &SourceMgr = ND->getASTContext().getSourceManager();
  
  SourceLocation SLoc = getLocationFromCursor(C, SourceMgr, ND);
  return SourceMgr.getSpellingColumnNumber(SLoc);
}
const char *clang_getCursorSource(CXCursor C) 
{
  assert(C.decl && "CXCursor has null decl");
  NamedDecl *ND = static_cast<NamedDecl *>(C.decl);
  SourceManager &SourceMgr = ND->getASTContext().getSourceManager();
  
  SourceLocation SLoc = getLocationFromCursor(C, SourceMgr, ND);
  if (SLoc.isFileID())
    return SourceMgr.getBufferName(SLoc);

  // Retrieve the file in which the macro was instantiated, then provide that
  // buffer name.
  // FIXME: Do we want to give specific macro-instantiation information?
  const llvm::MemoryBuffer *Buffer 
    = SourceMgr.getBuffer(SourceMgr.getDecomposedSpellingLoc(SLoc).first);
  if (!Buffer)
    return 0;

  return Buffer->getBufferIdentifier();
}

void clang_getDefinitionSpellingAndExtent(CXCursor C, 
                                          const char **startBuf,
                                          const char **endBuf,
                                          unsigned *startLine,
                                          unsigned *startColumn,
                                          unsigned *endLine,
                                          unsigned *endColumn) 
{
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


} // end extern "C"
