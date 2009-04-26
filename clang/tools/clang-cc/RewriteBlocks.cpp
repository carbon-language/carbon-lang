//===--- RewriteBlocks.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Hacks and fun related to the closure rewriter.
//
//===----------------------------------------------------------------------===//

#include "ASTConsumers.h"
#include "clang/Rewrite/Rewriter.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <sstream>

using namespace clang;
using llvm::utostr;

namespace {

class RewriteBlocks : public ASTConsumer {
  Rewriter Rewrite;
  Diagnostic &Diags;
  const LangOptions &LangOpts;
  unsigned RewriteFailedDiag;

  ASTContext *Context;
  SourceManager *SM;
  FileID MainFileID;
  const char *MainFileStart, *MainFileEnd;

  // Block expressions.
  llvm::SmallVector<BlockExpr *, 32> Blocks;
  llvm::SmallVector<BlockDeclRefExpr *, 32> BlockDeclRefs;
  llvm::DenseMap<BlockDeclRefExpr *, CallExpr *> BlockCallExprs;
  
  // Block related declarations.
  llvm::SmallPtrSet<ValueDecl *, 8> BlockByCopyDecls;
  llvm::SmallPtrSet<ValueDecl *, 8> BlockByRefDecls;
  llvm::SmallPtrSet<ValueDecl *, 8> ImportedBlockDecls;

  llvm::DenseMap<BlockExpr *, std::string> RewrittenBlockExprs;
  
  // The function/method we are rewriting.
  FunctionDecl *CurFunctionDef;
  ObjCMethodDecl *CurMethodDef;
  
  bool IsHeader;
  std::string InFileName;
  std::string OutFileName;
  
  std::string Preamble;
public:
  RewriteBlocks(std::string inFile, std::string outFile, Diagnostic &D, 
                const LangOptions &LOpts);
  ~RewriteBlocks() {
    // Get the buffer corresponding to MainFileID.  
    // If we haven't changed it, then we are done.
    if (const RewriteBuffer *RewriteBuf = 
        Rewrite.getRewriteBufferFor(MainFileID)) {
      std::string S(RewriteBuf->begin(), RewriteBuf->end());
      printf("%s\n", S.c_str());
    } else {
      printf("No changes\n");
    }
  }
  
  void Initialize(ASTContext &context);

  void InsertText(SourceLocation Loc, const char *StrData, unsigned StrLen);
  void ReplaceText(SourceLocation Start, unsigned OrigLength,
                   const char *NewStr, unsigned NewLength);

  // Top Level Driver code.
  virtual void HandleTopLevelDecl(DeclGroupRef D) {
    for (DeclGroupRef::iterator I = D.begin(), E = D.end(); I != E; ++I)
      HandleTopLevelSingleDecl(*I);
  }
  void HandleTopLevelSingleDecl(Decl *D);
  void HandleDeclInMainFile(Decl *D);
  
  // Top level 
  Stmt *RewriteFunctionBody(Stmt *S);
  void InsertBlockLiteralsWithinFunction(FunctionDecl *FD);
  void InsertBlockLiteralsWithinMethod(ObjCMethodDecl *MD);
  
  // Block specific rewrite rules.
  std::string SynthesizeBlockInitExpr(BlockExpr *Exp, VarDecl *VD=0);
  
  void RewriteBlockCall(CallExpr *Exp);
  void RewriteBlockPointerDecl(NamedDecl *VD);
  void RewriteBlockDeclRefExpr(BlockDeclRefExpr *VD);
  void RewriteBlockPointerFunctionArgs(FunctionDecl *FD);
  
  std::string SynthesizeBlockHelperFuncs(BlockExpr *CE, int i, 
                                    const char *funcName, std::string Tag);
  std::string SynthesizeBlockFunc(BlockExpr *CE, int i, 
                                    const char *funcName, std::string Tag);
  std::string SynthesizeBlockImpl(BlockExpr *CE, std::string Tag, 
                                  bool hasCopyDisposeHelpers);
  std::string SynthesizeBlockCall(CallExpr *Exp);
  void SynthesizeBlockLiterals(SourceLocation FunLocStart,
                                 const char *FunName);
  
  void CollectBlockDeclRefInfo(BlockExpr *Exp);
  void GetBlockCallExprs(Stmt *S);
  void GetBlockDeclRefExprs(Stmt *S);
  
  // We avoid calling Type::isBlockPointerType(), since it operates on the
  // canonical type. We only care if the top-level type is a closure pointer.
  bool isBlockPointerType(QualType T) { return isa<BlockPointerType>(T); }
  
  // FIXME: This predicate seems like it would be useful to add to ASTContext.
  bool isObjCType(QualType T) {
    if (!LangOpts.ObjC1 && !LangOpts.ObjC2)
      return false;
      
    QualType OCT = Context->getCanonicalType(T).getUnqualifiedType();
    
    if (OCT == Context->getCanonicalType(Context->getObjCIdType()) ||
        OCT == Context->getCanonicalType(Context->getObjCClassType()))
      return true;
      
    if (const PointerType *PT = OCT->getAsPointerType()) {
      if (isa<ObjCInterfaceType>(PT->getPointeeType()) || 
          isa<ObjCQualifiedIdType>(PT->getPointeeType()))
        return true;
    }
    return false;
  }
  // ObjC rewrite methods.
  void RewriteInterfaceDecl(ObjCInterfaceDecl *ClassDecl);
  void RewriteCategoryDecl(ObjCCategoryDecl *CatDecl);
  void RewriteProtocolDecl(ObjCProtocolDecl *PDecl);
  void RewriteMethodDecl(ObjCMethodDecl *MDecl);

  void RewriteFunctionProtoType(QualType funcType, NamedDecl *D);
  void CheckFunctionPointerDecl(QualType dType, NamedDecl *ND);
  void RewriteCastExpr(CastExpr *CE);
  
  bool PointerTypeTakesAnyBlockArguments(QualType QT);
  void GetExtentOfArgList(const char *Name, const char *&LParen, const char *&RParen);
};
  
}

static bool IsHeaderFile(const std::string &Filename) {
  std::string::size_type DotPos = Filename.rfind('.');
  
  if (DotPos == std::string::npos) {
    // no file extension
    return false; 
  }
  
  std::string Ext = std::string(Filename.begin()+DotPos+1, Filename.end());
  // C header: .h
  // C++ header: .hh or .H;
  return Ext == "h" || Ext == "hh" || Ext == "H";
}    

RewriteBlocks::RewriteBlocks(std::string inFile, std::string outFile, 
                             Diagnostic &D, const LangOptions &LOpts) : 
  Diags(D), LangOpts(LOpts) {
  IsHeader = IsHeaderFile(inFile);
  InFileName = inFile;
  OutFileName = outFile;
  CurFunctionDef = 0;
  CurMethodDef = 0;
  RewriteFailedDiag = Diags.getCustomDiagID(Diagnostic::Warning, 
                                            "rewriting failed");
}

ASTConsumer *clang::CreateBlockRewriter(const std::string& InFile,
                                        const std::string& OutFile,
                                        Diagnostic &Diags,
                                        const LangOptions &LangOpts) {
  return new RewriteBlocks(InFile, OutFile, Diags, LangOpts);
}

void RewriteBlocks::Initialize(ASTContext &context) {
  Context = &context;
  SM = &Context->getSourceManager();
  
  // Get the ID and start/end of the main file.
  MainFileID = SM->getMainFileID();
  const llvm::MemoryBuffer *MainBuf = SM->getBuffer(MainFileID);
  MainFileStart = MainBuf->getBufferStart();
  MainFileEnd = MainBuf->getBufferEnd();
  
  Rewrite.setSourceMgr(Context->getSourceManager(), LangOpts);
  
  if (IsHeader)
    Preamble = "#pragma once\n";
  Preamble += "#ifndef BLOCK_IMPL\n";
  Preamble += "#define BLOCK_IMPL\n";
  Preamble += "struct __block_impl {\n";
  Preamble += "  void *isa;\n";
  Preamble += "  int Flags;\n";
  Preamble += "  int Size;\n";
  Preamble += "  void *FuncPtr;\n";
  Preamble += "};\n";
  Preamble += "enum {\n";
  Preamble += "  BLOCK_HAS_COPY_DISPOSE = (1<<25),\n";
  Preamble += "  BLOCK_IS_GLOBAL = (1<<28)\n";
  Preamble += "};\n";
  if (LangOpts.Microsoft) 
    Preamble += "#define __OBJC_RW_EXTERN extern \"C\" __declspec(dllimport)\n";
  else
    Preamble += "#define __OBJC_RW_EXTERN extern\n";
  Preamble += "// Runtime copy/destroy helper functions\n";
  Preamble += "__OBJC_RW_EXTERN void _Block_copy_assign(void *, void *);\n";
  Preamble += "__OBJC_RW_EXTERN void _Block_byref_assign_copy(void *, void *);\n";
  Preamble += "__OBJC_RW_EXTERN void _Block_destroy(void *);\n";
  Preamble += "__OBJC_RW_EXTERN void _Block_byref_release(void *);\n";
  Preamble += "__OBJC_RW_EXTERN void *_NSConcreteGlobalBlock;\n";
  Preamble += "__OBJC_RW_EXTERN void *_NSConcreteStackBlock;\n";
  Preamble += "#endif\n";
  
  InsertText(SM->getLocForStartOfFile(MainFileID), 
             Preamble.c_str(), Preamble.size());
}

void RewriteBlocks::InsertText(SourceLocation Loc, const char *StrData, 
                                 unsigned StrLen)
{
  if (!Rewrite.InsertText(Loc, StrData, StrLen))
    return;
  Diags.Report(Context->getFullLoc(Loc), RewriteFailedDiag);
}

void RewriteBlocks::ReplaceText(SourceLocation Start, unsigned OrigLength,
                                  const char *NewStr, unsigned NewLength) {
  if (!Rewrite.ReplaceText(Start, OrigLength, NewStr, NewLength))
    return;
  Diags.Report(Context->getFullLoc(Start), RewriteFailedDiag);
}

void RewriteBlocks::RewriteMethodDecl(ObjCMethodDecl *Method) {
  bool haveBlockPtrs = false;
  for (ObjCMethodDecl::param_iterator I = Method->param_begin(), 
       E = Method->param_end(); I != E; ++I)
    if (isBlockPointerType((*I)->getType()))
      haveBlockPtrs = true;
      
  if (!haveBlockPtrs)
    return;
    
  // Do a fuzzy rewrite.
  // We have 1 or more arguments that have closure pointers.
  SourceLocation Loc = Method->getLocStart();
  SourceLocation LocEnd = Method->getLocEnd();
  const char *startBuf = SM->getCharacterData(Loc);
  const char *endBuf = SM->getCharacterData(LocEnd);

  const char *methodPtr = startBuf;
  std::string Tag = "struct __block_impl *";
  
  while (*methodPtr++ && (methodPtr != endBuf)) {
    switch (*methodPtr) {
      case ':':
        methodPtr++;
        if (*methodPtr == '(') {
          const char *scanType = ++methodPtr;
          bool foundBlockPointer = false;
          unsigned parenCount = 1;
          
          while (parenCount) {
            switch (*scanType) {
              case '(': 
                parenCount++; 
                break;
              case ')': 
                parenCount--;
                break;
              case '^':
                foundBlockPointer = true;
                break;
            }
            scanType++;
          }
          if (foundBlockPointer) {
            // advance the location to startArgList.
            Loc = Loc.getFileLocWithOffset(methodPtr-startBuf);
            assert((Loc.isValid()) && "Invalid Loc");
            ReplaceText(Loc, scanType-methodPtr-1, Tag.c_str(), Tag.size());
            
            // Advance startBuf. Since the underlying buffer has changed,
            // it's very important to advance startBuf (so we can correctly
            // compute a relative Loc the next time around).
            startBuf = methodPtr;
          }
          // Advance the method ptr to the end of the type.
          methodPtr = scanType;
        }
        break;
    }
  }
  return;
}

void RewriteBlocks::RewriteInterfaceDecl(ObjCInterfaceDecl *ClassDecl) {
  for (ObjCInterfaceDecl::instmeth_iterator 
         I = ClassDecl->instmeth_begin(*Context), 
         E = ClassDecl->instmeth_end(*Context); 
       I != E; ++I)
    RewriteMethodDecl(*I);
  for (ObjCInterfaceDecl::classmeth_iterator 
         I = ClassDecl->classmeth_begin(*Context), 
         E = ClassDecl->classmeth_end(*Context);
       I != E; ++I)
    RewriteMethodDecl(*I);
}

void RewriteBlocks::RewriteCategoryDecl(ObjCCategoryDecl *CatDecl) {
  for (ObjCCategoryDecl::instmeth_iterator 
         I = CatDecl->instmeth_begin(*Context), 
         E = CatDecl->instmeth_end(*Context); 
       I != E; ++I)
    RewriteMethodDecl(*I);
  for (ObjCCategoryDecl::classmeth_iterator 
         I = CatDecl->classmeth_begin(*Context), 
         E = CatDecl->classmeth_end(*Context);
       I != E; ++I)
    RewriteMethodDecl(*I);
}

void RewriteBlocks::RewriteProtocolDecl(ObjCProtocolDecl *PDecl) {
  for (ObjCProtocolDecl::instmeth_iterator 
         I = PDecl->instmeth_begin(*Context), 
         E = PDecl->instmeth_end(*Context); 
       I != E; ++I)
    RewriteMethodDecl(*I);
  for (ObjCProtocolDecl::classmeth_iterator 
         I = PDecl->classmeth_begin(*Context), 
         E = PDecl->classmeth_end(*Context); 
       I != E; ++I)
    RewriteMethodDecl(*I);
}

//===----------------------------------------------------------------------===//
// Top Level Driver Code
//===----------------------------------------------------------------------===//

void RewriteBlocks::HandleTopLevelSingleDecl(Decl *D) {
  // Two cases: either the decl could be in the main file, or it could be in a
  // #included file.  If the former, rewrite it now.  If the later, check to see
  // if we rewrote the #include/#import.
  SourceLocation Loc = D->getLocation();
  Loc = SM->getInstantiationLoc(Loc);
  
  // If this is for a builtin, ignore it.
  if (Loc.isInvalid()) return;
  
  if (ObjCInterfaceDecl *MD = dyn_cast<ObjCInterfaceDecl>(D))
    RewriteInterfaceDecl(MD);
  else if (ObjCCategoryDecl *CD = dyn_cast<ObjCCategoryDecl>(D))
    RewriteCategoryDecl(CD);
  else if (ObjCProtocolDecl *PD = dyn_cast<ObjCProtocolDecl>(D))
    RewriteProtocolDecl(PD);

  // If we have a decl in the main file, see if we should rewrite it.
  if (SM->isFromMainFile(Loc))
    HandleDeclInMainFile(D);
  return;
}

std::string RewriteBlocks::SynthesizeBlockFunc(BlockExpr *CE, int i,
                                                   const char *funcName,
                                                   std::string Tag) {
  const FunctionType *AFT = CE->getFunctionType();
  QualType RT = AFT->getResultType();
  std::string StructRef = "struct " + Tag;
  std::string S = "static " + RT.getAsString() + " __" +
                  funcName + "_" + "block_func_" + utostr(i);

  BlockDecl *BD = CE->getBlockDecl();
  
  if (isa<FunctionNoProtoType>(AFT)) {
    S += "()";
  } else if (BD->param_empty()) {
    S += "(" + StructRef + " *__cself)";
  } else {
    const FunctionProtoType *FT = cast<FunctionProtoType>(AFT);
    assert(FT && "SynthesizeBlockFunc: No function proto");
    S += '(';
    // first add the implicit argument.
    S += StructRef + " *__cself, ";
    std::string ParamStr;
    for (BlockDecl::param_iterator AI = BD->param_begin(),
         E = BD->param_end(); AI != E; ++AI) {
      if (AI != BD->param_begin()) S += ", ";
      ParamStr = (*AI)->getNameAsString();
      (*AI)->getType().getAsStringInternal(ParamStr);
      S += ParamStr;
    }
    if (FT->isVariadic()) {
      if (!BD->param_empty()) S += ", ";
      S += "...";
    }
    S += ')';
  }
  S += " {\n";
  
  // Create local declarations to avoid rewriting all closure decl ref exprs.
  // First, emit a declaration for all "by ref" decls.
  for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByRefDecls.begin(), 
       E = BlockByRefDecls.end(); I != E; ++I) {
    S += "  ";
    std::string Name = (*I)->getNameAsString();
    Context->getPointerType((*I)->getType()).getAsStringInternal(Name);
    S += Name + " = __cself->" + (*I)->getNameAsString() + "; // bound by ref\n";
  }    
  // Next, emit a declaration for all "by copy" declarations.
  for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByCopyDecls.begin(), 
       E = BlockByCopyDecls.end(); I != E; ++I) {
    S += "  ";
    std::string Name = (*I)->getNameAsString();
    // Handle nested closure invocation. For example:
    //
    //   void (^myImportedClosure)(void);
    //   myImportedClosure  = ^(void) { setGlobalInt(x + y); };
    // 
    //   void (^anotherClosure)(void);
    //   anotherClosure = ^(void) {
    //     myImportedClosure(); // import and invoke the closure
    //   };
    //
    if (isBlockPointerType((*I)->getType()))
      S += "struct __block_impl *";
    else
      (*I)->getType().getAsStringInternal(Name);
    S += Name + " = __cself->" + (*I)->getNameAsString() + "; // bound by copy\n";
  }
  std::string RewrittenStr = RewrittenBlockExprs[CE];
  const char *cstr = RewrittenStr.c_str();
  while (*cstr++ != '{') ;
  S += cstr;
  S += "\n";
  return S;
}

std::string RewriteBlocks::SynthesizeBlockHelperFuncs(BlockExpr *CE, int i,
                                                   const char *funcName,
                                                   std::string Tag) {
  std::string StructRef = "struct " + Tag;
  std::string S = "static void __";
  
  S += funcName;
  S += "_block_copy_" + utostr(i);
  S += "(" + StructRef;
  S += "*dst, " + StructRef;
  S += "*src) {";
  for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = ImportedBlockDecls.begin(), 
      E = ImportedBlockDecls.end(); I != E; ++I) {
    S += "_Block_copy_assign(&dst->";
    S += (*I)->getNameAsString();
    S += ", src->";
    S += (*I)->getNameAsString();
    S += ");}";
  }
  S += "\nstatic void __";
  S += funcName;
  S += "_block_dispose_" + utostr(i);
  S += "(" + StructRef;
  S += "*src) {";
  for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = ImportedBlockDecls.begin(), 
      E = ImportedBlockDecls.end(); I != E; ++I) {
    S += "_Block_destroy(src->";
    S += (*I)->getNameAsString();
    S += ");";
  }
  S += "}\n";  
  return S;
}

std::string RewriteBlocks::SynthesizeBlockImpl(BlockExpr *CE, std::string Tag,
                                               bool hasCopyDisposeHelpers) {
  std::string S = "struct " + Tag;
  std::string Constructor = "  " + Tag;
  
  S += " {\n  struct __block_impl impl;\n";
  
  if (hasCopyDisposeHelpers)
    S += "  void *copy;\n  void *dispose;\n";
    
  Constructor += "(void *fp";
  
  if (hasCopyDisposeHelpers)
    Constructor += ", void *copyHelp, void *disposeHelp";
    
  if (BlockDeclRefs.size()) {
    // Output all "by copy" declarations.
    for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByCopyDecls.begin(), 
         E = BlockByCopyDecls.end(); I != E; ++I) {
      S += "  ";
      std::string FieldName = (*I)->getNameAsString();
      std::string ArgName = "_" + FieldName;
      // Handle nested closure invocation. For example:
      //
      //   void (^myImportedBlock)(void);
      //   myImportedBlock  = ^(void) { setGlobalInt(x + y); };
      // 
      //   void (^anotherBlock)(void);
      //   anotherBlock = ^(void) {
      //     myImportedBlock(); // import and invoke the closure
      //   };
      //
      if (isBlockPointerType((*I)->getType())) {
        S += "struct __block_impl *";
        Constructor += ", void *" + ArgName;
      } else {
        (*I)->getType().getAsStringInternal(FieldName);
        (*I)->getType().getAsStringInternal(ArgName);
        Constructor += ", " + ArgName;
      }
      S += FieldName + ";\n";
    }
    // Output all "by ref" declarations.
    for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByRefDecls.begin(), 
         E = BlockByRefDecls.end(); I != E; ++I) {
      S += "  ";
      std::string FieldName = (*I)->getNameAsString();
      std::string ArgName = "_" + FieldName;
      // Handle nested closure invocation. For example:
      //
      //   void (^myImportedBlock)(void);
      //   myImportedBlock  = ^(void) { setGlobalInt(x + y); };
      // 
      //   void (^anotherBlock)(void);
      //   anotherBlock = ^(void) {
      //     myImportedBlock(); // import and invoke the closure
      //   };
      //
      if (isBlockPointerType((*I)->getType())) {
        S += "struct __block_impl *";
        Constructor += ", void *" + ArgName;
      } else {
        Context->getPointerType((*I)->getType()).getAsStringInternal(FieldName);
        Context->getPointerType((*I)->getType()).getAsStringInternal(ArgName);
        Constructor += ", " + ArgName;
      }
      S += FieldName + "; // by ref\n";
    }
    // Finish writing the constructor.
    // FIXME: handle NSConcreteGlobalBlock.
    Constructor += ", int flags=0) {\n";
    Constructor += "    impl.isa = 0/*&_NSConcreteStackBlock*/;\n    impl.Size = sizeof(";
    Constructor += Tag + ");\n    impl.Flags = flags;\n    impl.FuncPtr = fp;\n";
    
    if (hasCopyDisposeHelpers)
      Constructor += "    copy = copyHelp;\n    dispose = disposeHelp;\n";
      
    // Initialize all "by copy" arguments.
    for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByCopyDecls.begin(), 
         E = BlockByCopyDecls.end(); I != E; ++I) {
      std::string Name = (*I)->getNameAsString();
      Constructor += "    ";
      if (isBlockPointerType((*I)->getType()))
        Constructor += Name + " = (struct __block_impl *)_";
      else
        Constructor += Name + " = _";
      Constructor += Name + ";\n";
    }
    // Initialize all "by ref" arguments.
    for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByRefDecls.begin(), 
         E = BlockByRefDecls.end(); I != E; ++I) {
      std::string Name = (*I)->getNameAsString();
      Constructor += "    ";
      if (isBlockPointerType((*I)->getType()))
        Constructor += Name + " = (struct __block_impl *)_";
      else
        Constructor += Name + " = _";
      Constructor += Name + ";\n";
    }
  } else {
    // Finish writing the constructor.
    // FIXME: handle NSConcreteGlobalBlock.
    Constructor += ", int flags=0) {\n";
    Constructor += "    impl.isa = 0/*&_NSConcreteStackBlock*/;\n    impl.Size = sizeof(";
    Constructor += Tag + ");\n    impl.Flags = flags;\n    impl.FuncPtr = fp;\n";
    if (hasCopyDisposeHelpers)
      Constructor += "    copy = copyHelp;\n    dispose = disposeHelp;\n";
  }
  Constructor += "  ";
  Constructor += "}\n";
  S += Constructor;
  S += "};\n";
  return S;
}

void RewriteBlocks::SynthesizeBlockLiterals(SourceLocation FunLocStart,
                                                const char *FunName) {
  // Insert closures that were part of the function.
  for (unsigned i = 0; i < Blocks.size(); i++) {

    CollectBlockDeclRefInfo(Blocks[i]);

    std::string Tag = "__" + std::string(FunName) + "_block_impl_" + utostr(i);
                      
    std::string CI = SynthesizeBlockImpl(Blocks[i], Tag, 
                                         ImportedBlockDecls.size() > 0);

    InsertText(FunLocStart, CI.c_str(), CI.size());

    std::string CF = SynthesizeBlockFunc(Blocks[i], i, FunName, Tag);
    
    InsertText(FunLocStart, CF.c_str(), CF.size());

    if (ImportedBlockDecls.size()) {
      std::string HF = SynthesizeBlockHelperFuncs(Blocks[i], i, FunName, Tag);
      InsertText(FunLocStart, HF.c_str(), HF.size());
    }
    
    BlockDeclRefs.clear();
    BlockByRefDecls.clear();
    BlockByCopyDecls.clear();
    BlockCallExprs.clear();
    ImportedBlockDecls.clear();
  }
  Blocks.clear();
  RewrittenBlockExprs.clear();
}

void RewriteBlocks::InsertBlockLiteralsWithinFunction(FunctionDecl *FD) {
  SourceLocation FunLocStart = FD->getTypeSpecStartLoc();
  const char *FuncName = FD->getNameAsCString();
  
  SynthesizeBlockLiterals(FunLocStart, FuncName);
}

void RewriteBlocks::InsertBlockLiteralsWithinMethod(ObjCMethodDecl *MD) {
  SourceLocation FunLocStart = MD->getLocStart();
  std::string FuncName = MD->getSelector().getAsString();
  // Convert colons to underscores.
  std::string::size_type loc = 0;
  while ((loc = FuncName.find(":", loc)) != std::string::npos)
    FuncName.replace(loc, 1, "_");
  
  SynthesizeBlockLiterals(FunLocStart, FuncName.c_str());
}

void RewriteBlocks::GetBlockDeclRefExprs(Stmt *S) {
  for (Stmt::child_iterator CI = S->child_begin(), E = S->child_end();
       CI != E; ++CI)
    if (*CI) {
      if (BlockExpr *CBE = dyn_cast<BlockExpr>(*CI))
        GetBlockDeclRefExprs(CBE->getBody());
      else
        GetBlockDeclRefExprs(*CI);
    }
  // Handle specific things.
  if (BlockDeclRefExpr *CDRE = dyn_cast<BlockDeclRefExpr>(S))
    // FIXME: Handle enums.
    if (!isa<FunctionDecl>(CDRE->getDecl()))
      BlockDeclRefs.push_back(CDRE);
  return;
}

void RewriteBlocks::GetBlockCallExprs(Stmt *S) {
  for (Stmt::child_iterator CI = S->child_begin(), E = S->child_end();
       CI != E; ++CI)
    if (*CI) {
      if (BlockExpr *CBE = dyn_cast<BlockExpr>(*CI))
        GetBlockCallExprs(CBE->getBody());
      else
        GetBlockCallExprs(*CI);
    }
      
  if (CallExpr *CE = dyn_cast<CallExpr>(S)) {
    if (CE->getCallee()->getType()->isBlockPointerType()) {
      BlockCallExprs[dyn_cast<BlockDeclRefExpr>(CE->getCallee())] = CE;
    }
  }
  return;
}

std::string RewriteBlocks::SynthesizeBlockCall(CallExpr *Exp) {
  // Navigate to relevant type information.
  const char *closureName = 0;
  const BlockPointerType *CPT = 0;
  
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Exp->getCallee())) {
    closureName = DRE->getDecl()->getNameAsCString();
    CPT = DRE->getType()->getAsBlockPointerType();
  } else if (BlockDeclRefExpr *CDRE = dyn_cast<BlockDeclRefExpr>(Exp->getCallee())) {
    closureName = CDRE->getDecl()->getNameAsCString();
    CPT = CDRE->getType()->getAsBlockPointerType();
  } else if (MemberExpr *MExpr = dyn_cast<MemberExpr>(Exp->getCallee())) {
    closureName = MExpr->getMemberDecl()->getNameAsCString();
    CPT = MExpr->getType()->getAsBlockPointerType();
  } else {
    assert(1 && "RewriteBlockClass: Bad type");
  }
  assert(CPT && "RewriteBlockClass: Bad type");
  const FunctionType *FT = CPT->getPointeeType()->getAsFunctionType();
  assert(FT && "RewriteBlockClass: Bad type");
  const FunctionProtoType *FTP = dyn_cast<FunctionProtoType>(FT);
  // FTP will be null for closures that don't take arguments.
  
  // Build a closure call - start with a paren expr to enforce precedence.
  std::string BlockCall = "(";

  // Synthesize the cast.  
  BlockCall += "(" + Exp->getType().getAsString() + "(*)";
  BlockCall += "(struct __block_impl *";
  if (FTP) {
    for (FunctionProtoType::arg_type_iterator I = FTP->arg_type_begin(), 
         E = FTP->arg_type_end(); I && (I != E); ++I)
      BlockCall += ", " + (*I).getAsString();
  }
  BlockCall += "))"; // close the argument list and paren expression.
  
  // Invoke the closure. We need to cast it since the declaration type is
  // bogus (it's a function pointer type)
  BlockCall += "((struct __block_impl *)";
  std::string closureExprBufStr;
  llvm::raw_string_ostream closureExprBuf(closureExprBufStr);
  Exp->getCallee()->printPretty(closureExprBuf);
  BlockCall += closureExprBuf.str();
  BlockCall += ")->FuncPtr)";
  
  // Add the arguments.
  BlockCall += "((struct __block_impl *)";
  BlockCall += closureExprBuf.str();
  for (CallExpr::arg_iterator I = Exp->arg_begin(), 
       E = Exp->arg_end(); I != E; ++I) {
    std::string syncExprBufS;
    llvm::raw_string_ostream Buf(syncExprBufS);
    (*I)->printPretty(Buf);
    BlockCall += ", " + Buf.str();
  }
  return BlockCall;
}

void RewriteBlocks::RewriteBlockCall(CallExpr *Exp) {
  std::string BlockCall = SynthesizeBlockCall(Exp);
  
  const char *startBuf = SM->getCharacterData(Exp->getLocStart());
  const char *endBuf = SM->getCharacterData(Exp->getLocEnd());

  ReplaceText(Exp->getLocStart(), endBuf-startBuf, 
              BlockCall.c_str(), BlockCall.size());
}

void RewriteBlocks::RewriteBlockDeclRefExpr(BlockDeclRefExpr *BDRE) {
  // FIXME: Add more elaborate code generation required by the ABI.
  InsertText(BDRE->getLocStart(), "*", 1);
}

void RewriteBlocks::RewriteCastExpr(CastExpr *CE) {
  SourceLocation LocStart = CE->getLocStart();
  SourceLocation LocEnd = CE->getLocEnd();
  
  if (!Rewriter::isRewritable(LocStart) || !Rewriter::isRewritable(LocEnd))
    return;
    
  const char *startBuf = SM->getCharacterData(LocStart);
  const char *endBuf = SM->getCharacterData(LocEnd);
  
  // advance the location to startArgList.
  const char *argPtr = startBuf;
  
  while (*argPtr++ && (argPtr < endBuf)) {
    switch (*argPtr) {
      case '^': 
        // Replace the '^' with '*'.
        LocStart = LocStart.getFileLocWithOffset(argPtr-startBuf);
        ReplaceText(LocStart, 1, "*", 1);
        break;
    }
  }
  return;
}

void RewriteBlocks::RewriteBlockPointerFunctionArgs(FunctionDecl *FD) {
  SourceLocation DeclLoc = FD->getLocation();
  unsigned parenCount = 0;
  
  // We have 1 or more arguments that have closure pointers.
  const char *startBuf = SM->getCharacterData(DeclLoc);
  const char *startArgList = strchr(startBuf, '(');
  
  assert((*startArgList == '(') && "Rewriter fuzzy parser confused");
  
  parenCount++;
  // advance the location to startArgList.
  DeclLoc = DeclLoc.getFileLocWithOffset(startArgList-startBuf);
  assert((DeclLoc.isValid()) && "Invalid DeclLoc");
  
  const char *argPtr = startArgList;
  
  while (*argPtr++ && parenCount) {
    switch (*argPtr) {
      case '^': 
        // Replace the '^' with '*'.
        DeclLoc = DeclLoc.getFileLocWithOffset(argPtr-startArgList);
        ReplaceText(DeclLoc, 1, "*", 1);
        break;
      case '(': 
        parenCount++; 
        break;
      case ')': 
        parenCount--;
        break;
    }
  }
  return;
}

bool RewriteBlocks::PointerTypeTakesAnyBlockArguments(QualType QT) {
  const FunctionProtoType *FTP;
  const PointerType *PT = QT->getAsPointerType();
  if (PT) {
    FTP = PT->getPointeeType()->getAsFunctionProtoType();
  } else {
    const BlockPointerType *BPT = QT->getAsBlockPointerType();
    assert(BPT && "BlockPointerTypeTakeAnyBlockArguments(): not a block pointer type");
    FTP = BPT->getPointeeType()->getAsFunctionProtoType();
  }
  if (FTP) {
    for (FunctionProtoType::arg_type_iterator I = FTP->arg_type_begin(), 
         E = FTP->arg_type_end(); I != E; ++I)
      if (isBlockPointerType(*I))
        return true;
  }
  return false;
}

void RewriteBlocks::GetExtentOfArgList(const char *Name, 
                                       const char *&LParen, const char *&RParen) {
  const char *argPtr = strchr(Name, '(');
  assert((*argPtr == '(') && "Rewriter fuzzy parser confused");
  
  LParen = argPtr; // output the start.
  argPtr++; // skip past the left paren.
  unsigned parenCount = 1;
  
  while (*argPtr && parenCount) {
    switch (*argPtr) {
      case '(': parenCount++; break;
      case ')': parenCount--; break;
      default: break;
    }
    if (parenCount) argPtr++;
  }
  assert((*argPtr == ')') && "Rewriter fuzzy parser confused");
  RParen = argPtr; // output the end
}

void RewriteBlocks::RewriteBlockPointerDecl(NamedDecl *ND) {
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(ND)) {
    RewriteBlockPointerFunctionArgs(FD);
    return;
  } 
  // Handle Variables and Typedefs.
  SourceLocation DeclLoc = ND->getLocation();
  QualType DeclT;
  if (VarDecl *VD = dyn_cast<VarDecl>(ND))
    DeclT = VD->getType();
  else if (TypedefDecl *TDD = dyn_cast<TypedefDecl>(ND))
    DeclT = TDD->getUnderlyingType();
  else if (FieldDecl *FD = dyn_cast<FieldDecl>(ND))
    DeclT = FD->getType();
  else 
    assert(0 && "RewriteBlockPointerDecl(): Decl type not yet handled");
    
  const char *startBuf = SM->getCharacterData(DeclLoc);
  const char *endBuf = startBuf;
  // scan backward (from the decl location) for the end of the previous decl.
  while (*startBuf != '^' && *startBuf != ';' && startBuf != MainFileStart)
    startBuf--;
    
  // *startBuf != '^' if we are dealing with a pointer to function that
  // may take block argument types (which will be handled below).
  if (*startBuf == '^') {
    // Replace the '^' with '*', computing a negative offset.
    DeclLoc = DeclLoc.getFileLocWithOffset(startBuf-endBuf);
    ReplaceText(DeclLoc, 1, "*", 1);
  }
  if (PointerTypeTakesAnyBlockArguments(DeclT)) {
    // Replace the '^' with '*' for arguments.
    DeclLoc = ND->getLocation();
    startBuf = SM->getCharacterData(DeclLoc);
    const char *argListBegin, *argListEnd;
    GetExtentOfArgList(startBuf, argListBegin, argListEnd);
    while (argListBegin < argListEnd) {
      if (*argListBegin == '^') {
        SourceLocation CaretLoc = DeclLoc.getFileLocWithOffset(argListBegin-startBuf);
        ReplaceText(CaretLoc, 1, "*", 1);
      }
      argListBegin++;
    }
  }
  return;
}

void RewriteBlocks::CollectBlockDeclRefInfo(BlockExpr *Exp) {  
  // Add initializers for any closure decl refs.
  GetBlockDeclRefExprs(Exp->getBody());
  if (BlockDeclRefs.size()) {
    // Unique all "by copy" declarations.
    for (unsigned i = 0; i < BlockDeclRefs.size(); i++)
      if (!BlockDeclRefs[i]->isByRef())
        BlockByCopyDecls.insert(BlockDeclRefs[i]->getDecl());
    // Unique all "by ref" declarations.
    for (unsigned i = 0; i < BlockDeclRefs.size(); i++)
      if (BlockDeclRefs[i]->isByRef()) {
        BlockByRefDecls.insert(BlockDeclRefs[i]->getDecl());
      }
    // Find any imported blocks...they will need special attention.
    for (unsigned i = 0; i < BlockDeclRefs.size(); i++)
      if (isBlockPointerType(BlockDeclRefs[i]->getType())) {
        GetBlockCallExprs(Blocks[i]);
        ImportedBlockDecls.insert(BlockDeclRefs[i]->getDecl());
      }
  }
}

std::string RewriteBlocks::SynthesizeBlockInitExpr(BlockExpr *Exp, VarDecl *VD) {
  Blocks.push_back(Exp);

  CollectBlockDeclRefInfo(Exp);
  std::string FuncName;
  
  if (CurFunctionDef)
    FuncName = std::string(CurFunctionDef->getNameAsString());
  else if (CurMethodDef) {
    FuncName = CurMethodDef->getSelector().getAsString();
    // Convert colons to underscores.
    std::string::size_type loc = 0;
    while ((loc = FuncName.find(":", loc)) != std::string::npos)
      FuncName.replace(loc, 1, "_");
  } else if (VD)
    FuncName = std::string(VD->getNameAsString());
    
  std::string BlockNumber = utostr(Blocks.size()-1);
  
  std::string Tag = "__" + FuncName + "_block_impl_" + BlockNumber;
  std::string Func = "__" + FuncName + "_block_func_" + BlockNumber;
  
  std::string FunkTypeStr;
  
  // Get a pointer to the function type so we can cast appropriately.
  Context->getPointerType(QualType(Exp->getFunctionType(),0)).getAsStringInternal(FunkTypeStr);
  
  // Rewrite the closure block with a compound literal. The first cast is
  // to prevent warnings from the C compiler.
  std::string Init = "(" + FunkTypeStr;
  
  Init += ")&" + Tag;
  
  // Initialize the block function.
  Init += "((void*)" + Func;
  
  if (ImportedBlockDecls.size()) {
    std::string Buf = "__" + FuncName + "_block_copy_" + BlockNumber;
    Init += ",(void*)" + Buf;
    Buf = "__" + FuncName + "_block_dispose_" + BlockNumber;
    Init += ",(void*)" + Buf;
  }
  // Add initializers for any closure decl refs.
  if (BlockDeclRefs.size()) {
    // Output all "by copy" declarations.
    for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByCopyDecls.begin(), 
         E = BlockByCopyDecls.end(); I != E; ++I) {
      Init += ",";
      if (isObjCType((*I)->getType())) {
        Init += "[[";
        Init += (*I)->getNameAsString();
        Init += " retain] autorelease]";
      } else if (isBlockPointerType((*I)->getType())) {
        Init += "(void *)";
        Init += (*I)->getNameAsString();
      } else {
        Init += (*I)->getNameAsString();
      }
    }
    // Output all "by ref" declarations.
    for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByRefDecls.begin(), 
         E = BlockByRefDecls.end(); I != E; ++I) {
      Init += ",&";
      Init += (*I)->getNameAsString();
    }
  }
  Init += ")";
  BlockDeclRefs.clear();
  BlockByRefDecls.clear();
  BlockByCopyDecls.clear();
  ImportedBlockDecls.clear();

  return Init;
}

//===----------------------------------------------------------------------===//
// Function Body / Expression rewriting
//===----------------------------------------------------------------------===//

Stmt *RewriteBlocks::RewriteFunctionBody(Stmt *S) {
  // Start by rewriting all children.
  for (Stmt::child_iterator CI = S->child_begin(), E = S->child_end();
       CI != E; ++CI)
    if (*CI) {
      if (BlockExpr *CBE = dyn_cast<BlockExpr>(*CI)) {
        Stmt *newStmt = RewriteFunctionBody(CBE->getBody());
        if (newStmt) 
          *CI = newStmt;
          
        // We've just rewritten the block body in place.
        // Now we snarf the rewritten text and stash it away for later use.
        std::string S = Rewrite.getRewritenText(CBE->getSourceRange());
        RewrittenBlockExprs[CBE] = S;
        std::string Init = SynthesizeBlockInitExpr(CBE);
        // Do the rewrite, using S.size() which contains the rewritten size.
        ReplaceText(CBE->getLocStart(), S.size(), Init.c_str(), Init.size());
      } else {
        Stmt *newStmt = RewriteFunctionBody(*CI);
        if (newStmt) 
          *CI = newStmt;
      }
    }
  // Handle specific things.
  if (CallExpr *CE = dyn_cast<CallExpr>(S)) {
    if (CE->getCallee()->getType()->isBlockPointerType())
      RewriteBlockCall(CE);
  }
  if (CastExpr *CE = dyn_cast<CastExpr>(S)) {
    RewriteCastExpr(CE);
  }
  if (DeclStmt *DS = dyn_cast<DeclStmt>(S)) {
    for (DeclStmt::decl_iterator DI = DS->decl_begin(), DE = DS->decl_end();
         DI != DE; ++DI) {
      
      Decl *SD = *DI;
      if (ValueDecl *ND = dyn_cast<ValueDecl>(SD)) {
        if (isBlockPointerType(ND->getType()))
          RewriteBlockPointerDecl(ND);
        else if (ND->getType()->isFunctionPointerType()) 
          CheckFunctionPointerDecl(ND->getType(), ND);
      }
      if (TypedefDecl *TD = dyn_cast<TypedefDecl>(SD)) {
        if (isBlockPointerType(TD->getUnderlyingType()))
          RewriteBlockPointerDecl(TD);
        else if (TD->getUnderlyingType()->isFunctionPointerType()) 
          CheckFunctionPointerDecl(TD->getUnderlyingType(), TD);
      }
    }
  }
  // Handle specific things.
  if (BlockDeclRefExpr *BDRE = dyn_cast<BlockDeclRefExpr>(S)) {
    if (BDRE->isByRef())
      RewriteBlockDeclRefExpr(BDRE);
  }
  // Return this stmt unmodified.
  return S;
}

void RewriteBlocks::RewriteFunctionProtoType(QualType funcType, NamedDecl *D) {    
  if (FunctionProtoType *fproto = dyn_cast<FunctionProtoType>(funcType)) {
    for (FunctionProtoType::arg_type_iterator I = fproto->arg_type_begin(), 
         E = fproto->arg_type_end(); I && (I != E); ++I)
      if (isBlockPointerType(*I)) {
        // All the args are checked/rewritten. Don't call twice!
        RewriteBlockPointerDecl(D);
        break;
      }
  }
}

void RewriteBlocks::CheckFunctionPointerDecl(QualType funcType, NamedDecl *ND) {
  const PointerType *PT = funcType->getAsPointerType();
  if (PT && PointerTypeTakesAnyBlockArguments(funcType))
    RewriteFunctionProtoType(PT->getPointeeType(), ND);
}

/// HandleDeclInMainFile - This is called for each top-level decl defined in the
/// main file of the input.
void RewriteBlocks::HandleDeclInMainFile(Decl *D) {
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // Since function prototypes don't have ParmDecl's, we check the function
    // prototype. This enables us to rewrite function declarations and
    // definitions using the same code.
    RewriteFunctionProtoType(FD->getType(), FD);

    // FIXME: Handle CXXTryStmt
    if (CompoundStmt *Body = FD->getCompoundBody(*Context)) {
      CurFunctionDef = FD;
      FD->setBody(cast_or_null<CompoundStmt>(RewriteFunctionBody(Body)));
      // This synthesizes and inserts the block "impl" struct, invoke function,
      // and any copy/dispose helper functions.
      InsertBlockLiteralsWithinFunction(FD);
      CurFunctionDef = 0;
    } 
    return;
  }
  if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
    RewriteMethodDecl(MD);
    if (Stmt *Body = MD->getBody(*Context)) {
      CurMethodDef = MD;
      RewriteFunctionBody(Body);
      InsertBlockLiteralsWithinMethod(MD);
      CurMethodDef = 0;
    }
  }
  if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
    if (isBlockPointerType(VD->getType())) {
      RewriteBlockPointerDecl(VD);
      if (VD->getInit()) {
        if (BlockExpr *CBE = dyn_cast<BlockExpr>(VD->getInit())) {
          RewriteFunctionBody(CBE->getBody(*Context));

          // We've just rewritten the block body in place.
          // Now we snarf the rewritten text and stash it away for later use.
          std::string S = Rewrite.getRewritenText(CBE->getSourceRange());
          RewrittenBlockExprs[CBE] = S;
          std::string Init = SynthesizeBlockInitExpr(CBE, VD);
          // Do the rewrite, using S.size() which contains the rewritten size.
          ReplaceText(CBE->getLocStart(), S.size(), Init.c_str(), Init.size());
          SynthesizeBlockLiterals(VD->getTypeSpecStartLoc(), 
                                  VD->getNameAsCString());
        } else if (CastExpr *CE = dyn_cast<CastExpr>(VD->getInit())) {
          RewriteCastExpr(CE);
        }
      }
    } else if (VD->getType()->isFunctionPointerType()) {
      CheckFunctionPointerDecl(VD->getType(), VD);
      if (VD->getInit()) {
        if (CastExpr *CE = dyn_cast<CastExpr>(VD->getInit())) {
          RewriteCastExpr(CE);
        }
      }
    }
    return;
  }
  if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
    if (isBlockPointerType(TD->getUnderlyingType()))
      RewriteBlockPointerDecl(TD);
    else if (TD->getUnderlyingType()->isFunctionPointerType()) 
      CheckFunctionPointerDecl(TD->getUnderlyingType(), TD);
    return;
  }
  if (RecordDecl *RD = dyn_cast<RecordDecl>(D)) {
    if (RD->isDefinition()) {
      for (RecordDecl::field_iterator i = RD->field_begin(*Context), 
             e = RD->field_end(*Context); i != e; ++i) {
        FieldDecl *FD = *i;
        if (isBlockPointerType(FD->getType()))
          RewriteBlockPointerDecl(FD);
      }
    }
    return;
  }
}
