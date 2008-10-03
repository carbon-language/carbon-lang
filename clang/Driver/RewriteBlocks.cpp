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
  unsigned NoNestedBlockCalls;

  ASTContext *Context;
  SourceManager *SM;
  unsigned MainFileID;
  const char *MainFileStart, *MainFileEnd;

  // Block expressions.
  llvm::SmallVector<BlockExpr *, 32> Blocks;
  llvm::SmallVector<BlockDeclRefExpr *, 32> BlockDeclRefs;
  llvm::DenseMap<BlockDeclRefExpr *, CallExpr *> BlockCallExprs;
  
  // Block related declarations.
  llvm::SmallPtrSet<ValueDecl *, 8> BlockByCopyDecls;
  llvm::SmallPtrSet<ValueDecl *, 8> BlockByRefDecls;
  llvm::SmallPtrSet<ValueDecl *, 8> ImportedBlockDecls;
  
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
  virtual void HandleTopLevelDecl(Decl *D);
  void HandleDeclInMainFile(Decl *D);
  
  // Top level 
  Stmt *RewriteFunctionBody(Stmt *S);
  void InsertBlockLiteralsWithinFunction(FunctionDecl *FD);
  void InsertBlockLiteralsWithinMethod(ObjCMethodDecl *MD);
  
  // Block specific rewrite rules.
  void RewriteBlockExpr(BlockExpr *Exp, VarDecl *VD=0);
  
  void RewriteBlockCall(CallExpr *Exp);
  void RewriteBlockPointerDecl(NamedDecl *VD);
  void RewriteBlockPointerFunctionArgs(FunctionDecl *FD);
  
  std::string SynthesizeBlockHelperFuncs(BlockExpr *CE, int i, 
                                    const char *funcName, std::string Tag);
  std::string SynthesizeBlockFunc(BlockExpr *CE, int i, 
                                    const char *funcName, std::string Tag);
  std::string SynthesizeBlockImpl(BlockExpr *CE, std::string Tag);
  std::string SynthesizeBlockCall(CallExpr *Exp);
  void SynthesizeBlockLiterals(SourceLocation FunLocStart,
                                 const char *FunName);
  
  void GetBlockDeclRefExprs(Stmt *S);
  void GetBlockCallExprs(Stmt *S);
  
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
  
  bool BlockPointerTypeTakesAnyBlockArguments(QualType QT);
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
  NoNestedBlockCalls = Diags.getCustomDiagID(Diagnostic::Warning, 
    "Rewrite support for closure calls nested within closure blocks is incomplete");
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
  
  Rewrite.setSourceMgr(Context->getSourceManager());
  
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
  
  InsertText(SourceLocation::getFileLoc(MainFileID, 0), 
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
  for (ObjCInterfaceDecl::instmeth_iterator I = ClassDecl->instmeth_begin(), 
       E = ClassDecl->instmeth_end(); I != E; ++I)
    RewriteMethodDecl(*I);
  for (ObjCInterfaceDecl::classmeth_iterator I = ClassDecl->classmeth_begin(), 
       E = ClassDecl->classmeth_end(); I != E; ++I)
    RewriteMethodDecl(*I);
}

void RewriteBlocks::RewriteCategoryDecl(ObjCCategoryDecl *CatDecl) {
  for (ObjCCategoryDecl::instmeth_iterator I = CatDecl->instmeth_begin(), 
       E = CatDecl->instmeth_end(); I != E; ++I)
    RewriteMethodDecl(*I);
  for (ObjCCategoryDecl::classmeth_iterator I = CatDecl->classmeth_begin(), 
       E = CatDecl->classmeth_end(); I != E; ++I)
    RewriteMethodDecl(*I);
}

void RewriteBlocks::RewriteProtocolDecl(ObjCProtocolDecl *PDecl) {
  for (ObjCProtocolDecl::instmeth_iterator I = PDecl->instmeth_begin(), 
       E = PDecl->instmeth_end(); I != E; ++I)
    RewriteMethodDecl(*I);
  for (ObjCProtocolDecl::classmeth_iterator I = PDecl->classmeth_begin(), 
       E = PDecl->classmeth_end(); I != E; ++I)
    RewriteMethodDecl(*I);
}

//===----------------------------------------------------------------------===//
// Top Level Driver Code
//===----------------------------------------------------------------------===//

void RewriteBlocks::HandleTopLevelDecl(Decl *D) {
  // Two cases: either the decl could be in the main file, or it could be in a
  // #included file.  If the former, rewrite it now.  If the later, check to see
  // if we rewrote the #include/#import.
  SourceLocation Loc = D->getLocation();
  Loc = SM->getLogicalLoc(Loc);
  
  // If this is for a builtin, ignore it.
  if (Loc.isInvalid()) return;
  
  if (ObjCInterfaceDecl *MD = dyn_cast<ObjCInterfaceDecl>(D))
    RewriteInterfaceDecl(MD);
  else if (ObjCCategoryDecl *CD = dyn_cast<ObjCCategoryDecl>(D))
    RewriteCategoryDecl(CD);
  else if (ObjCProtocolDecl *PD = dyn_cast<ObjCProtocolDecl>(D))
    RewriteProtocolDecl(PD);

  // If we have a decl in the main file, see if we should rewrite it.
  if (SM->getDecomposedFileLoc(Loc).first == MainFileID)
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

  if (isa<FunctionTypeNoProto>(AFT)) {
    S += "()";
  } else if (CE->arg_empty()) {
    S += "(" + StructRef + " *__cself)";
  } else {
    const FunctionTypeProto *FT = cast<FunctionTypeProto>(AFT);
    assert(FT && "SynthesizeBlockFunc: No function proto");
    S += '(';
    // first add the implicit argument.
    S += StructRef + " *__cself, ";
    std::string ParamStr;
    for (BlockExpr::arg_iterator AI = CE->arg_begin(),
         E = CE->arg_end(); AI != E; ++AI) {
      if (AI != CE->arg_begin()) S += ", ";
      ParamStr = (*AI)->getName();
      (*AI)->getType().getAsStringInternal(ParamStr);
      S += ParamStr;
    }
    if (FT->isVariadic()) {
      if (!CE->arg_empty()) S += ", ";
      S += "...";
    }
    S += ')';
  }
  S += " {\n";
  
  bool haveByRefDecls = false;
  
  // Create local declarations to avoid rewriting all closure decl ref exprs.
  // First, emit a declaration for all "by ref" decls.
  for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByRefDecls.begin(), 
       E = BlockByRefDecls.end(); I != E; ++I) {
    // Note: It is not possible to have "by ref" closure pointer decls.
    haveByRefDecls = true;
    S += "  ";
    std::string Name = (*I)->getName();
    Context->getPointerType((*I)->getType()).getAsStringInternal(Name);
    S += Name + " = __cself->" + (*I)->getName() + "; // bound by ref\n";
  }    
  // Next, emit a declaration for all "by copy" declarations.
  for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByCopyDecls.begin(), 
       E = BlockByCopyDecls.end(); I != E; ++I) {
    S += "  ";
    std::string Name = (*I)->getName();
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
    S += Name + " = __cself->" + (*I)->getName() + "; // bound by copy\n";
  }    
  if (BlockExpr *CBE = dyn_cast<BlockExpr>(CE)) {
    std::string BodyBuf;
    
    SourceLocation BodyLocStart = CBE->getBody()->getLocStart();
    SourceLocation BodyLocEnd = CBE->getBody()->getLocEnd();
    const char *BodyStartBuf = SM->getCharacterData(BodyLocStart);
    const char *BodyEndBuf = SM->getCharacterData(BodyLocEnd);
    
    BodyBuf.append(BodyStartBuf, BodyEndBuf-BodyStartBuf+1);
    
    //fprintf(stderr, "BodyBuf=>%s\n", BodyBuf.c_str());
    if (BlockDeclRefs.size()) {
      unsigned int nCharsAdded = 0;
      for (unsigned i = 0; i < BlockDeclRefs.size(); i++) {
        if (BlockDeclRefs[i]->isByRef()) {
          // Add a level of indirection! The code below assumes
          // the closure decl refs/locations are in strictly ascending
          // order. The traversal performed by GetBlockDeclRefExprs()
          // currently does this. FIXME: Wrap the *x with parens,
          // just in case x is a more complex expression, like x->member,
          // which needs to be rewritten to (*x)->member.
          SourceLocation StarLoc = BlockDeclRefs[i]->getLocStart();
          const char *StarBuf = SM->getCharacterData(StarLoc);
          BodyBuf.insert(StarBuf-BodyStartBuf+nCharsAdded, 1, '*');
          // Get a fresh buffer, the insert might have caused it to grow.
          BodyStartBuf = SM->getCharacterData(BodyLocStart);
          nCharsAdded++;
        } else if (isBlockPointerType(BlockDeclRefs[i]->getType())) {
          Diags.Report(NoNestedBlockCalls);

          GetBlockCallExprs(CE);
          ImportedBlockDecls.insert(BlockDeclRefs[i]->getDecl());
          
          // Rewrite the closure in place.
          // The character based equivalent of RewriteBlockCall().
          // Need to get the CallExpr associated with this BlockDeclRef.
          std::string BlockCall = SynthesizeBlockCall(BlockCallExprs[BlockDeclRefs[i]]);
          
          // FIXME: this is still incomplete.
          SourceLocation CallLocStart = BlockCallExprs[BlockDeclRefs[i]]->getLocStart();
          SourceLocation CallLocEnd = BlockCallExprs[BlockDeclRefs[i]]->getLocEnd();
          const char *CallStart = SM->getCharacterData(CallLocStart) + nCharsAdded;
          const char *CallEnd = SM->getCharacterData(CallLocEnd);
          unsigned CallBytes = CallEnd-CallStart;
          //fprintf(stderr, "BlockCall=>%s CallStart=%d\n", BlockCall.c_str(),CallStart);
          BodyBuf.replace(CallStart-BodyStartBuf, CallBytes, BlockCall.c_str());
          nCharsAdded += CallBytes;
        }
      }
    }
    if (haveByRefDecls) {
      // Remove |...|.
      //const char *firstBarPtr = strchr(BodyStartBuf, '|');
      //const char *secondBarPtr = strchr(firstBarPtr+1, '|');
      //BodyBuf.replace(firstBarPtr-BodyStartBuf, secondBarPtr-firstBarPtr+1, "");
    } 
    S += "  ";
    S += BodyBuf;
  }
  S += "\n}\n";
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
    S += (*I)->getName();
    S += ", src->";
    S += (*I)->getName();
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
    S += (*I)->getName();
    S += ");";
  }
  S += "}\n";  
  return S;
}

std::string RewriteBlocks::SynthesizeBlockImpl(BlockExpr *CE, std::string Tag) {
  std::string S = "struct " + Tag;
  std::string Constructor = "  " + Tag;
  
  S += " {\n  struct __block_impl impl;\n";
  Constructor += "(void *fp";
  
  GetBlockDeclRefExprs(CE);
  if (BlockDeclRefs.size()) {
    // Unique all "by copy" declarations.
    for (unsigned i = 0; i < BlockDeclRefs.size(); i++)
      if (!BlockDeclRefs[i]->isByRef())
        BlockByCopyDecls.insert(BlockDeclRefs[i]->getDecl());
    // Unique all "by ref" declarations.
    for (unsigned i = 0; i < BlockDeclRefs.size(); i++)
      if (BlockDeclRefs[i]->isByRef())
        BlockByRefDecls.insert(BlockDeclRefs[i]->getDecl());
        
    // Output all "by copy" declarations.
    for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByCopyDecls.begin(), 
         E = BlockByCopyDecls.end(); I != E; ++I) {
      S += "  ";
      std::string FieldName = (*I)->getName();
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
      std::string FieldName = (*I)->getName();
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
    
    // Initialize all "by copy" arguments.
    for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByCopyDecls.begin(), 
         E = BlockByCopyDecls.end(); I != E; ++I) {
      std::string Name = (*I)->getName();
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
      std::string Name = (*I)->getName();
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
  
    std::string Tag = "__" + std::string(FunName) + "_block_impl_" + utostr(i);
                      
    std::string CI = SynthesizeBlockImpl(Blocks[i], Tag);

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
}

void RewriteBlocks::InsertBlockLiteralsWithinFunction(FunctionDecl *FD) {
  SourceLocation FunLocStart = FD->getTypeSpecStartLoc();
  const char *FuncName = FD->getName();
  
  SynthesizeBlockLiterals(FunLocStart, FuncName);
}

void RewriteBlocks::InsertBlockLiteralsWithinMethod(ObjCMethodDecl *MD) {
  SourceLocation FunLocStart = MD->getLocStart();
  std::string FuncName = std::string(MD->getSelector().getName());
  // Convert colons to underscores.
  std::string::size_type loc = 0;
  while ((loc = FuncName.find(":", loc)) != std::string::npos)
    FuncName.replace(loc, 1, "_");
  
  SynthesizeBlockLiterals(FunLocStart, FuncName.c_str());
}

/// HandleDeclInMainFile - This is called for each top-level decl defined in the
/// main file of the input.
void RewriteBlocks::HandleDeclInMainFile(Decl *D) {
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
  
    // Since function prototypes don't have ParmDecl's, we check the function
    // prototype. This enables us to rewrite function declarations and
    // definitions using the same code.
    QualType funcType = FD->getType();
    
    if (FunctionTypeProto *fproto = dyn_cast<FunctionTypeProto>(funcType)) {
      for (FunctionTypeProto::arg_type_iterator I = fproto->arg_type_begin(), 
           E = fproto->arg_type_end(); I && (I != E); ++I)
        if (isBlockPointerType(*I)) {
          // All the args are checked/rewritten. Don't call twice!
          RewriteBlockPointerDecl(FD);
          break;
        }
    }
    if (Stmt *Body = FD->getBody()) {
      CurFunctionDef = FD;
      FD->setBody(RewriteFunctionBody(Body));
      InsertBlockLiteralsWithinFunction(FD);
      CurFunctionDef = 0;
    } 
    return;
  }
  if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
    RewriteMethodDecl(MD);
    if (Stmt *Body = MD->getBody()) {
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
        if (BlockExpr *BExp = dyn_cast<BlockExpr>(VD->getInit())) {
          RewriteBlockExpr(BExp, VD);
          SynthesizeBlockLiterals(VD->getTypeSpecStartLoc(), VD->getName());
        }
      }
    }
    return;
  }
  if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
    if (isBlockPointerType(TD->getUnderlyingType()))
      RewriteBlockPointerDecl(TD);
    return;
  }
  if (RecordDecl *RD = dyn_cast<RecordDecl>(D)) {
    if (RD->isDefinition()) {
      for (RecordDecl::field_const_iterator i = RD->field_begin(), 
             e = RD->field_end(); i != e; ++i) {
        FieldDecl *FD = *i;
        if (isBlockPointerType(FD->getType()))
          RewriteBlockPointerDecl(FD);
      }
    }
    return;
  }
}

void RewriteBlocks::GetBlockDeclRefExprs(Stmt *S) {
  for (Stmt::child_iterator CI = S->child_begin(), E = S->child_end();
       CI != E; ++CI)
    if (*CI) 
      GetBlockDeclRefExprs(*CI);
      
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
    if (*CI) 
      GetBlockCallExprs(*CI);
      
  if (CallExpr *CE = dyn_cast<CallExpr>(S)) {
    if (CE->getCallee()->getType()->isBlockPointerType()) {
      BlockCallExprs[dyn_cast<BlockDeclRefExpr>(CE->getCallee())] = CE;
    }
  }
  return;
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
        // We intentionally avoid rewritting the contents of a closure block
        // expr. InsertBlockLiteralsWithinFunction() will rewrite the body.
        RewriteBlockExpr(CBE);
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
  if (DeclStmt *DS = dyn_cast<DeclStmt>(S)) {
    ScopedDecl *SD = DS->getDecl();
    if (ValueDecl *ND = dyn_cast<ValueDecl>(SD)) {
      if (isBlockPointerType(ND->getType()))
        RewriteBlockPointerDecl(ND);
    }
    if (TypedefDecl *TD = dyn_cast<TypedefDecl>(SD)) {
      if (isBlockPointerType(TD->getUnderlyingType()))
        RewriteBlockPointerDecl(TD);
    }
  }
  // Return this stmt unmodified.
  return S;
}

std::string RewriteBlocks::SynthesizeBlockCall(CallExpr *Exp) {
  // Navigate to relevant type information.
  const char *closureName = 0;
  const BlockPointerType *CPT = 0;
  
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Exp->getCallee())) {
    closureName = DRE->getDecl()->getName();
    CPT = DRE->getType()->getAsBlockPointerType();
  } else if (BlockDeclRefExpr *CDRE = dyn_cast<BlockDeclRefExpr>(Exp->getCallee())) {
    closureName = CDRE->getDecl()->getName();
    CPT = CDRE->getType()->getAsBlockPointerType();
  } else if (MemberExpr *MExpr = dyn_cast<MemberExpr>(Exp->getCallee())) {
    closureName = MExpr->getMemberDecl()->getName();
    CPT = MExpr->getType()->getAsBlockPointerType();
  } else {
    assert(1 && "RewriteBlockClass: Bad type");
  }
  assert(CPT && "RewriteBlockClass: Bad type");
  const FunctionType *FT = CPT->getPointeeType()->getAsFunctionType();
  assert(FT && "RewriteBlockClass: Bad type");
  const FunctionTypeProto *FTP = dyn_cast<FunctionTypeProto>(FT);
  // FTP will be null for closures that don't take arguments.
  
  // Build a closure call - start with a paren expr to enforce precedence.
  std::string BlockCall = "(";

  // Synthesize the cast.  
  BlockCall += "(" + Exp->getType().getAsString() + "(*)";
  BlockCall += "(struct __block_impl *";
  if (FTP) {
    for (FunctionTypeProto::arg_type_iterator I = FTP->arg_type_begin(), 
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
  BlockCall += closureName;
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

void RewriteBlocks::RewriteBlockPointerFunctionArgs(FunctionDecl *FD) {
  SourceLocation DeclLoc = FD->getLocation();
  unsigned parenCount = 0, nArgs = 0;
  
  // We have 1 or more arguments that have closure pointers.
  const char *startBuf = SM->getCharacterData(DeclLoc);
  const char *startArgList = strchr(startBuf, '(');
  
  assert((*startArgList == '(') && "Rewriter fuzzy parser confused");
  
  parenCount++;
  // advance the location to startArgList.
  DeclLoc = DeclLoc.getFileLocWithOffset(startArgList-startBuf+1);
  assert((DeclLoc.isValid()) && "Invalid DeclLoc");
  
  const char *topLevelCommaCursor = 0;
  const char *argPtr = startArgList;
  bool scannedBlockDecl = false;
  std::string Tag = "struct __block_impl *";
  
  while (*argPtr++ && parenCount) {
    switch (*argPtr) {
      case '^': 
        scannedBlockDecl = true; 
        break;
      case '(': 
        parenCount++; 
        break;
      case ')': 
        parenCount--;
        if (parenCount == 0) {
          if (scannedBlockDecl) {
            // If we are rewriting a definition, don't forget the arg name.
            if (FD->getBody())
              Tag += FD->getParamDecl(nArgs)->getName();
            // The last argument is a closure pointer decl, rewrite it!
            if (topLevelCommaCursor)
              ReplaceText(DeclLoc, argPtr-topLevelCommaCursor-2, Tag.c_str(), Tag.size());
            else
              ReplaceText(DeclLoc, argPtr-startArgList-1, Tag.c_str(), Tag.size());
            scannedBlockDecl = false; // reset.
          }
          nArgs++;
        }
        break;
      case ',':
        if (parenCount == 1) {
          // Make sure the function takes more than one argument.
          assert((FD->getNumParams() > 1) && "Rewriter fuzzy parser confused");
          if (scannedBlockDecl) {
            // If we are rewriting a definition, don't forget the arg name.
            if (FD->getBody())
              Tag += FD->getParamDecl(nArgs)->getName();
            // The current argument is a closure pointer decl, rewrite it!
            if (topLevelCommaCursor)
              ReplaceText(DeclLoc, argPtr-topLevelCommaCursor-1, Tag.c_str(), Tag.size());
            else
              ReplaceText(DeclLoc, argPtr-startArgList-1, Tag.c_str(), Tag.size());
            scannedBlockDecl = false;
          }
          nArgs++;
          // advance the location to topLevelCommaCursor.
          if (topLevelCommaCursor)
            DeclLoc = DeclLoc.getFileLocWithOffset(argPtr-topLevelCommaCursor);
          else
            DeclLoc = DeclLoc.getFileLocWithOffset(argPtr-startArgList+1);
          topLevelCommaCursor = argPtr;
          assert((DeclLoc.isValid()) && "Invalid DeclLoc");
        }
        break;
    }
  }
  return;
}

bool RewriteBlocks::BlockPointerTypeTakesAnyBlockArguments(QualType QT) {
  const BlockPointerType *BPT = QT->getAsBlockPointerType();
  assert(BPT && "BlockPointerTypeTakeAnyBlockArguments(): not a block pointer type");
  const FunctionTypeProto *FTP = BPT->getPointeeType()->getAsFunctionTypeProto();
  if (FTP) {
    for (FunctionTypeProto::arg_type_iterator I = FTP->arg_type_begin(), 
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
  assert((*startBuf == '^') && 
         "RewriteBlockPointerDecl() scan error: no caret");
  // Replace the '^' with '*', computing a negative offset.
  DeclLoc = DeclLoc.getFileLocWithOffset(startBuf-endBuf);
  ReplaceText(DeclLoc, 1, "*", 1);
  
  if (BlockPointerTypeTakesAnyBlockArguments(DeclT)) {
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

void RewriteBlocks::RewriteBlockExpr(BlockExpr *Exp, VarDecl *VD) {
  Blocks.push_back(Exp);
  bool haveByRefDecls = false;

  // Add initializers for any closure decl refs.
  GetBlockDeclRefExprs(Exp);
  if (BlockDeclRefs.size()) {
    // Unique all "by copy" declarations.
    for (unsigned i = 0; i < BlockDeclRefs.size(); i++)
      if (!BlockDeclRefs[i]->isByRef())
        BlockByCopyDecls.insert(BlockDeclRefs[i]->getDecl());
    // Unique all "by ref" declarations.
    for (unsigned i = 0; i < BlockDeclRefs.size(); i++)
      if (BlockDeclRefs[i]->isByRef()) {
        haveByRefDecls = true;
        BlockByRefDecls.insert(BlockDeclRefs[i]->getDecl());
      }
  }
  std::string FuncName;
  
  if (CurFunctionDef)
    FuncName = std::string(CurFunctionDef->getName());
  else if (CurMethodDef) {
    FuncName = std::string(CurMethodDef->getSelector().getName());
    // Convert colons to underscores.
    std::string::size_type loc = 0;
    while ((loc = FuncName.find(":", loc)) != std::string::npos)
      FuncName.replace(loc, 1, "_");
  } else if (VD)
    FuncName = std::string(VD->getName());
    
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
  
  // Add initializers for any closure decl refs.
  if (BlockDeclRefs.size()) {
    // Output all "by copy" declarations.
    for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByCopyDecls.begin(), 
         E = BlockByCopyDecls.end(); I != E; ++I) {
      Init += ",";
      if (isObjCType((*I)->getType())) {
        Init += "[[";
        Init += (*I)->getName();
        Init += " retain] autorelease]";
      } else if (isBlockPointerType((*I)->getType())) {
        Init += "(void *)";
        Init += (*I)->getName();
      } else {
        Init += (*I)->getName();
      }
    }
    // Output all "by ref" declarations.
    for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByRefDecls.begin(), 
         E = BlockByRefDecls.end(); I != E; ++I) {
      Init += ",&";
      Init += (*I)->getName();
    }
  }
  Init += ")";
  BlockDeclRefs.clear();
  BlockByRefDecls.clear();
  BlockByCopyDecls.clear();
  ImportedBlockDecls.clear();

  // Do the rewrite.
  const char *startBuf = SM->getCharacterData(Exp->getLocStart());
  const char *endBuf = SM->getCharacterData(Exp->getLocEnd());
  ReplaceText(Exp->getLocStart(), endBuf-startBuf+1, Init.c_str(), Init.size());
  return;
}
