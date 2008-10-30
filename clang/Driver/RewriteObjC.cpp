//===--- RewriteObjC.cpp - Playground for the code rewriter ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Hacks and fun related to the code rewriter.
//
//===----------------------------------------------------------------------===//

#include "ASTConsumers.h"
#include "clang/Rewrite/Rewriter.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/TranslationUnit.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"
using namespace clang;
using llvm::utostr;

static llvm::cl::opt<bool>
SilenceRewriteMacroWarning("Wno-rewrite-macros", llvm::cl::init(false),
                           llvm::cl::desc("Silence ObjC rewriting warnings"));

namespace {
  class RewriteObjC : public ASTConsumer {
    Rewriter Rewrite;
    Diagnostic &Diags;
    const LangOptions &LangOpts;
    unsigned RewriteFailedDiag;
    
    ASTContext *Context;
    SourceManager *SM;
    TranslationUnitDecl *TUDecl;
    unsigned MainFileID;
    const char *MainFileStart, *MainFileEnd;
    SourceLocation LastIncLoc;
    
    llvm::SmallVector<ObjCImplementationDecl *, 8> ClassImplementation;
    llvm::SmallVector<ObjCCategoryImplDecl *, 8> CategoryImplementation;
    llvm::SmallPtrSet<ObjCInterfaceDecl*, 8> ObjCSynthesizedStructs;
    llvm::SmallPtrSet<ObjCProtocolDecl*, 8> ObjCSynthesizedProtocols;
    llvm::SmallPtrSet<ObjCInterfaceDecl*, 8> ObjCForwardDecls;
    llvm::DenseMap<ObjCMethodDecl*, std::string> MethodInternalNames;
    llvm::SmallVector<Stmt *, 32> Stmts;
    llvm::SmallVector<int, 8> ObjCBcLabelNo;
    
    unsigned NumObjCStringLiterals;
    
    FunctionDecl *MsgSendFunctionDecl;
    FunctionDecl *MsgSendSuperFunctionDecl;
    FunctionDecl *MsgSendStretFunctionDecl;
    FunctionDecl *MsgSendSuperStretFunctionDecl;
    FunctionDecl *MsgSendFpretFunctionDecl;
    FunctionDecl *GetClassFunctionDecl;
    FunctionDecl *GetMetaClassFunctionDecl;
    FunctionDecl *SelGetUidFunctionDecl;
    FunctionDecl *CFStringFunctionDecl;
    FunctionDecl *GetProtocolFunctionDecl;
    FunctionDecl *SuperContructorFunctionDecl;
      
    // ObjC string constant support.
    VarDecl *ConstantStringClassReference;
    RecordDecl *NSStringRecord;
    
    // ObjC foreach break/continue generation support.
    int BcLabelCount;
    
    // Needed for super.
    ObjCMethodDecl *CurMethodDef;
    RecordDecl *SuperStructDecl;
    RecordDecl *ConstantStringDecl;
    
    // Needed for header files being rewritten
    bool IsHeader;
    
    std::string InFileName;
    std::string OutFileName;
     
    std::string Preamble;

    // Block expressions.
    llvm::SmallVector<BlockExpr *, 32> Blocks;
    llvm::SmallVector<BlockDeclRefExpr *, 32> BlockDeclRefs;
    llvm::DenseMap<BlockDeclRefExpr *, CallExpr *> BlockCallExprs;
    
    // Block related declarations.
    llvm::SmallPtrSet<ValueDecl *, 8> BlockByCopyDecls;
    llvm::SmallPtrSet<ValueDecl *, 8> BlockByRefDecls;
    llvm::SmallPtrSet<ValueDecl *, 8> ImportedBlockDecls;

    llvm::DenseMap<BlockExpr *, std::string> RewrittenBlockExprs;

    FunctionDecl *CurFunctionDef;
    VarDecl *GlobalVarDecl;
    
    static const int OBJC_ABI_VERSION =7 ;
  public:
    virtual void Initialize(ASTContext &context);

    virtual void InitializeTU(TranslationUnit &TU) {
      TU.SetOwnsDecls(false);
      Initialize(TU.getContext());
    }
    

    // Top Level Driver code.
    virtual void HandleTopLevelDecl(Decl *D);
    void HandleDeclInMainFile(Decl *D);
    RewriteObjC(std::string inFile, std::string outFile,
                Diagnostic &D, const LangOptions &LOpts);

    ~RewriteObjC() {}
    
    virtual void HandleTranslationUnit(TranslationUnit& TU);
    
    void ReplaceStmt(Stmt *Old, Stmt *New) {
      // If replacement succeeded or warning disabled return with no warning.
      if (!Rewrite.ReplaceStmt(Old, New) || SilenceRewriteMacroWarning)
        return;

      SourceRange Range = Old->getSourceRange();
      Diags.Report(Context->getFullLoc(Old->getLocStart()), RewriteFailedDiag,
                   0, 0, &Range, 1);
    }
    
    void InsertText(SourceLocation Loc, const char *StrData, unsigned StrLen,
                    bool InsertAfter = true) {
      // If insertion succeeded or warning disabled return with no warning.
      if (!Rewrite.InsertText(Loc, StrData, StrLen, InsertAfter) ||
          SilenceRewriteMacroWarning)
        return;
      
      Diags.Report(Context->getFullLoc(Loc), RewriteFailedDiag);
    }
    
    void RemoveText(SourceLocation Loc, unsigned StrLen) {
      // If removal succeeded or warning disabled return with no warning.
      if (!Rewrite.RemoveText(Loc, StrLen) || SilenceRewriteMacroWarning)
        return;
      
      Diags.Report(Context->getFullLoc(Loc), RewriteFailedDiag);
    }

    void ReplaceText(SourceLocation Start, unsigned OrigLength,
                     const char *NewStr, unsigned NewLength) {
      // If removal succeeded or warning disabled return with no warning.
      if (!Rewrite.ReplaceText(Start, OrigLength, NewStr, NewLength) ||
          SilenceRewriteMacroWarning)
        return;
      
      Diags.Report(Context->getFullLoc(Start), RewriteFailedDiag);
    }
    
    // Syntactic Rewriting.
    void RewritePrologue(SourceLocation Loc);
    void RewriteInclude();
    void RewriteTabs();
    void RewriteForwardClassDecl(ObjCClassDecl *Dcl);
    void RewriteInterfaceDecl(ObjCInterfaceDecl *Dcl);
    void RewriteImplementationDecl(NamedDecl *Dcl);
    void RewriteObjCMethodDecl(ObjCMethodDecl *MDecl, std::string &ResultStr);
    void RewriteCategoryDecl(ObjCCategoryDecl *Dcl);
    void RewriteProtocolDecl(ObjCProtocolDecl *Dcl);
    void RewriteForwardProtocolDecl(ObjCForwardProtocolDecl *Dcl);
    void RewriteMethodDeclaration(ObjCMethodDecl *Method);
    void RewriteProperties(unsigned nProperties, ObjCPropertyDecl **Properties);
    void RewriteFunctionDecl(FunctionDecl *FD);
    void RewriteObjCQualifiedInterfaceTypes(Decl *Dcl);
    void RewriteObjCQualifiedInterfaceTypes(Expr *E);
    bool needToScanForQualifiers(QualType T);
    ObjCInterfaceDecl *isSuperReceiver(Expr *recExpr);
    QualType getSuperStructType();
    QualType getConstantStringStructType();
    bool BufferContainsPPDirectives(const char *startBuf, const char *endBuf);
    
    // Expression Rewriting.
    Stmt *RewriteFunctionBodyOrGlobalInitializer(Stmt *S);
    Stmt *RewriteAtEncode(ObjCEncodeExpr *Exp);
    Stmt *RewriteObjCIvarRefExpr(ObjCIvarRefExpr *IV, SourceLocation OrigStart);
    Stmt *RewriteAtSelector(ObjCSelectorExpr *Exp);
    Stmt *RewriteMessageExpr(ObjCMessageExpr *Exp);
    Stmt *RewriteObjCStringLiteral(ObjCStringLiteral *Exp);
    Stmt *RewriteObjCProtocolExpr(ObjCProtocolExpr *Exp);
    Stmt *RewriteObjCTryStmt(ObjCAtTryStmt *S);
    Stmt *RewriteObjCSynchronizedStmt(ObjCAtSynchronizedStmt *S);
    Stmt *RewriteObjCCatchStmt(ObjCAtCatchStmt *S);
    Stmt *RewriteObjCFinallyStmt(ObjCAtFinallyStmt *S);
    Stmt *RewriteObjCThrowStmt(ObjCAtThrowStmt *S);
    Stmt *RewriteObjCForCollectionStmt(ObjCForCollectionStmt *S,
                                       SourceLocation OrigEnd);
    CallExpr *SynthesizeCallToFunctionDecl(FunctionDecl *FD, 
                                           Expr **args, unsigned nargs);
    Stmt *SynthMessageExpr(ObjCMessageExpr *Exp);
    Stmt *RewriteBreakStmt(BreakStmt *S);
    Stmt *RewriteContinueStmt(ContinueStmt *S);
    void SynthCountByEnumWithState(std::string &buf);
    
    void SynthMsgSendFunctionDecl();
    void SynthMsgSendSuperFunctionDecl();
    void SynthMsgSendStretFunctionDecl();
    void SynthMsgSendFpretFunctionDecl();
    void SynthMsgSendSuperStretFunctionDecl();
    void SynthGetClassFunctionDecl();
    void SynthGetMetaClassFunctionDecl();
    void SynthSelGetUidFunctionDecl();
    void SynthGetProtocolFunctionDecl();
    void SynthSuperContructorFunctionDecl();
      
    // Metadata emission.
    void RewriteObjCClassMetaData(ObjCImplementationDecl *IDecl,
                                  std::string &Result);
    
    void RewriteObjCCategoryImplDecl(ObjCCategoryImplDecl *CDecl,
                                     std::string &Result);
    
    typedef ObjCCategoryImplDecl::instmeth_iterator instmeth_iterator;
    void RewriteObjCMethodsMetaData(instmeth_iterator MethodBegin,
                                    instmeth_iterator MethodEnd,
                                    bool IsInstanceMethod,
                                    const char *prefix,
                                    const char *ClassName,
                                    std::string &Result);
    
    void RewriteObjCProtocolsMetaData(const ObjCList<ObjCProtocolDecl>
                                                       &Protocols,
                                      const char *prefix,
                                      const char *ClassName,
                                      std::string &Result);
    void SynthesizeObjCInternalStruct(ObjCInterfaceDecl *CDecl,
                                      std::string &Result);
    void SynthesizeIvarOffsetComputation(ObjCImplementationDecl *IDecl, 
                                         ObjCIvarDecl *ivar, 
                                         std::string &Result);
    void RewriteImplementations(std::string &Result);
    
    // Block rewriting.
    void RewriteBlocksInFunctionTypeProto(QualType funcType, NamedDecl *D);  
    void CheckFunctionPointerDecl(QualType dType, NamedDecl *ND);
    
    void InsertBlockLiteralsWithinFunction(FunctionDecl *FD);
    void InsertBlockLiteralsWithinMethod(ObjCMethodDecl *MD);
    
    // Block specific rewrite rules.    
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
    Stmt *SynthesizeBlockCall(CallExpr *Exp);
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
    bool PointerTypeTakesAnyBlockArguments(QualType QT);
    void GetExtentOfArgList(const char *Name, const char *&LParen, const char *&RParen);
    void RewriteCastExpr(CastExpr *CE);
    
    FunctionDecl *SynthBlockInitFunctionDecl(const char *name);
    Stmt *SynthBlockInitExpr(BlockExpr *Exp);
  };
}

void RewriteObjC::RewriteBlocksInFunctionTypeProto(QualType funcType, 
                                                   NamedDecl *D) {    
  if (FunctionTypeProto *fproto = dyn_cast<FunctionTypeProto>(funcType)) {
    for (FunctionTypeProto::arg_type_iterator I = fproto->arg_type_begin(), 
         E = fproto->arg_type_end(); I && (I != E); ++I)
      if (isBlockPointerType(*I)) {
        // All the args are checked/rewritten. Don't call twice!
        RewriteBlockPointerDecl(D);
        break;
      }
  }
}

void RewriteObjC::CheckFunctionPointerDecl(QualType funcType, NamedDecl *ND) {
  const PointerType *PT = funcType->getAsPointerType();
  if (PT && PointerTypeTakesAnyBlockArguments(funcType))
    RewriteBlocksInFunctionTypeProto(PT->getPointeeType(), ND);
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

RewriteObjC::RewriteObjC(std::string inFile, std::string outFile,
                         Diagnostic &D, const LangOptions &LOpts)
      : Diags(D), LangOpts(LOpts) {
  IsHeader = IsHeaderFile(inFile);
  InFileName = inFile;
  OutFileName = outFile;
  RewriteFailedDiag = Diags.getCustomDiagID(Diagnostic::Warning, 
               "rewriting sub-expression within a macro (may not be correct)");
}

ASTConsumer *clang::CreateCodeRewriterTest(const std::string& InFile,
                                           const std::string& OutFile,
                                           Diagnostic &Diags, 
                                           const LangOptions &LOpts) {
  return new RewriteObjC(InFile, OutFile, Diags, LOpts);
}

void RewriteObjC::Initialize(ASTContext &context) {
  Context = &context;
  SM = &Context->getSourceManager();
  TUDecl = Context->getTranslationUnitDecl();
  MsgSendFunctionDecl = 0;
  MsgSendSuperFunctionDecl = 0;
  MsgSendStretFunctionDecl = 0;
  MsgSendSuperStretFunctionDecl = 0;
  MsgSendFpretFunctionDecl = 0;
  GetClassFunctionDecl = 0;
  GetMetaClassFunctionDecl = 0;
  SelGetUidFunctionDecl = 0;
  CFStringFunctionDecl = 0;
  GetProtocolFunctionDecl = 0;
  ConstantStringClassReference = 0;
  NSStringRecord = 0;
  CurMethodDef = 0;
  CurFunctionDef = 0;
  SuperStructDecl = 0;
  ConstantStringDecl = 0;
  BcLabelCount = 0;
  SuperContructorFunctionDecl = 0;
  NumObjCStringLiterals = 0;
  
  // Get the ID and start/end of the main file.
  MainFileID = SM->getMainFileID();
  const llvm::MemoryBuffer *MainBuf = SM->getBuffer(MainFileID);
  MainFileStart = MainBuf->getBufferStart();
  MainFileEnd = MainBuf->getBufferEnd();
     
  Rewrite.setSourceMgr(Context->getSourceManager());
  
  // declaring objc_selector outside the parameter list removes a silly
  // scope related warning...
  if (IsHeader)
    Preamble = "#pragma once\n";
  Preamble += "struct objc_selector; struct objc_class;\n";
  Preamble += "#ifndef OBJC_SUPER\n";
  Preamble += "struct objc_super { struct objc_object *object; ";
  Preamble += "struct objc_object *superClass; ";
  if (LangOpts.Microsoft) {
    // Add a constructor for creating temporary objects.
    Preamble += "objc_super(struct objc_object *o, struct objc_object *s) : ";
    Preamble += "object(o), superClass(s) {} ";
  }
  Preamble += "};\n";
  Preamble += "#define OBJC_SUPER\n";
  Preamble += "#endif\n";
  Preamble += "#ifndef _REWRITER_typedef_Protocol\n";
  Preamble += "typedef struct objc_object Protocol;\n";
  Preamble += "#define _REWRITER_typedef_Protocol\n";
  Preamble += "#endif\n";
  if (LangOpts.Microsoft) 
    Preamble += "#define __OBJC_RW_EXTERN extern \"C\" __declspec(dllimport)\n";
  else
    Preamble += "#define __OBJC_RW_EXTERN extern\n";
  Preamble += "__OBJC_RW_EXTERN struct objc_object *objc_msgSend";
  Preamble += "(struct objc_object *, struct objc_selector *, ...);\n";
  Preamble += "__OBJC_RW_EXTERN struct objc_object *objc_msgSendSuper";
  Preamble += "(struct objc_super *, struct objc_selector *, ...);\n";
  Preamble += "__OBJC_RW_EXTERN struct objc_object *objc_msgSend_stret";
  Preamble += "(struct objc_object *, struct objc_selector *, ...);\n";
  Preamble += "__OBJC_RW_EXTERN struct objc_object *objc_msgSendSuper_stret";
  Preamble += "(struct objc_super *, struct objc_selector *, ...);\n";
  Preamble += "__OBJC_RW_EXTERN double objc_msgSend_fpret";
  Preamble += "(struct objc_object *, struct objc_selector *, ...);\n";
  Preamble += "__OBJC_RW_EXTERN struct objc_object *objc_getClass";
  Preamble += "(const char *);\n";
  Preamble += "__OBJC_RW_EXTERN struct objc_object *objc_getMetaClass";
  Preamble += "(const char *);\n";
  Preamble += "__OBJC_RW_EXTERN void objc_exception_throw(struct objc_object *);\n";
  Preamble += "__OBJC_RW_EXTERN void objc_exception_try_enter(void *);\n";
  Preamble += "__OBJC_RW_EXTERN void objc_exception_try_exit(void *);\n";
  Preamble += "__OBJC_RW_EXTERN struct objc_object *objc_exception_extract(void *);\n";
  Preamble += "__OBJC_RW_EXTERN int objc_exception_match";
  Preamble += "(struct objc_class *, struct objc_object *);\n";
  // @synchronized hooks.
  Preamble += "__OBJC_RW_EXTERN void objc_sync_enter(struct objc_object *);\n";
  Preamble += "__OBJC_RW_EXTERN void objc_sync_exit(struct objc_object *);\n";
  Preamble += "__OBJC_RW_EXTERN Protocol *objc_getProtocol(const char *);\n";
  Preamble += "#ifndef __FASTENUMERATIONSTATE\n";
  Preamble += "struct __objcFastEnumerationState {\n\t";
  Preamble += "unsigned long state;\n\t";
  Preamble += "void **itemsPtr;\n\t";
  Preamble += "unsigned long *mutationsPtr;\n\t";
  Preamble += "unsigned long extra[5];\n};\n";
  Preamble += "__OBJC_RW_EXTERN void objc_enumerationMutation(struct objc_object *);\n";
  Preamble += "#define __FASTENUMERATIONSTATE\n";
  Preamble += "#endif\n";
  Preamble += "#ifndef __NSCONSTANTSTRINGIMPL\n";
  Preamble += "struct __NSConstantStringImpl {\n";
  Preamble += "  int *isa;\n";
  Preamble += "  int flags;\n";
  Preamble += "  char *str;\n";
  Preamble += "  long length;\n";
  Preamble += "};\n";
  Preamble += "#ifdef CF_EXPORT_CONSTANT_STRING\n";
  Preamble += "extern \"C\" __declspec(dllexport) int __CFConstantStringClassReference[];\n";
  Preamble += "#else\n";
  Preamble += "__OBJC_RW_EXTERN int __CFConstantStringClassReference[];\n";
  Preamble += "#endif\n";
  Preamble += "#define __NSCONSTANTSTRINGIMPL\n";
  Preamble += "#endif\n";
  // Blocks preamble.
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
  Preamble += "// Runtime copy/destroy helper functions\n";
  Preamble += "__OBJC_RW_EXTERN void _Block_copy_assign(void *, void *);\n";
  Preamble += "__OBJC_RW_EXTERN void _Block_byref_assign_copy(void *, void *);\n";
  Preamble += "__OBJC_RW_EXTERN void _Block_destroy(void *);\n";
  Preamble += "__OBJC_RW_EXTERN void _Block_byref_release(void *);\n";
  Preamble += "__OBJC_RW_EXTERN void *_NSConcreteGlobalBlock;\n";
  Preamble += "__OBJC_RW_EXTERN void *_NSConcreteStackBlock;\n";
  Preamble += "#endif\n";
  if (LangOpts.Microsoft) {
    Preamble += "#undef __OBJC_RW_EXTERN\n";
    Preamble += "#define __attribute__(X)\n";
  }
}


//===----------------------------------------------------------------------===//
// Top Level Driver Code
//===----------------------------------------------------------------------===//

void RewriteObjC::HandleTopLevelDecl(Decl *D) {
  // Two cases: either the decl could be in the main file, or it could be in a
  // #included file.  If the former, rewrite it now.  If the later, check to see
  // if we rewrote the #include/#import.
  SourceLocation Loc = D->getLocation();
  Loc = SM->getLogicalLoc(Loc);
  
  // If this is for a builtin, ignore it.
  if (Loc.isInvalid()) return;

  // Look for built-in declarations that we need to refer during the rewrite.
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    RewriteFunctionDecl(FD);
  } else if (VarDecl *FVD = dyn_cast<VarDecl>(D)) {
    // declared in <Foundation/NSString.h>
    if (strcmp(FVD->getName(), "_NSConstantStringClassReference") == 0) {
      ConstantStringClassReference = FVD;
      return;
    }
  } else if (ObjCInterfaceDecl *MD = dyn_cast<ObjCInterfaceDecl>(D)) {
    RewriteInterfaceDecl(MD);
  } else if (ObjCCategoryDecl *CD = dyn_cast<ObjCCategoryDecl>(D)) {
    RewriteCategoryDecl(CD);
  } else if (ObjCProtocolDecl *PD = dyn_cast<ObjCProtocolDecl>(D)) {
    RewriteProtocolDecl(PD);
  } else if (ObjCForwardProtocolDecl *FP = 
             dyn_cast<ObjCForwardProtocolDecl>(D)){
    RewriteForwardProtocolDecl(FP);
  }
  // If we have a decl in the main file, see if we should rewrite it.
  if (SM->isFromMainFile(Loc))
    return HandleDeclInMainFile(D);
}

//===----------------------------------------------------------------------===//
// Syntactic (non-AST) Rewriting Code
//===----------------------------------------------------------------------===//

void RewriteObjC::RewriteInclude() {
  SourceLocation LocStart = SourceLocation::getFileLoc(MainFileID, 0);
  std::pair<const char*, const char*> MainBuf = SM->getBufferData(MainFileID);
  const char *MainBufStart = MainBuf.first;
  const char *MainBufEnd = MainBuf.second;
  size_t ImportLen = strlen("import");
  size_t IncludeLen = strlen("include");
                             
  // Loop over the whole file, looking for includes.
  for (const char *BufPtr = MainBufStart; BufPtr < MainBufEnd; ++BufPtr) {
    if (*BufPtr == '#') {
      if (++BufPtr == MainBufEnd)
        return;
      while (*BufPtr == ' ' || *BufPtr == '\t')
        if (++BufPtr == MainBufEnd)
          return;
      if (!strncmp(BufPtr, "import", ImportLen)) {
        // replace import with include
        SourceLocation ImportLoc = 
          LocStart.getFileLocWithOffset(BufPtr-MainBufStart);
        ReplaceText(ImportLoc, ImportLen, "include", IncludeLen);
        BufPtr += ImportLen;
      }
    }
  }
}

void RewriteObjC::RewriteTabs() {
  std::pair<const char*, const char*> MainBuf = SM->getBufferData(MainFileID);
  const char *MainBufStart = MainBuf.first;
  const char *MainBufEnd = MainBuf.second;
  
  // Loop over the whole file, looking for tabs.
  for (const char *BufPtr = MainBufStart; BufPtr != MainBufEnd; ++BufPtr) {
    if (*BufPtr != '\t')
      continue;
    
    // Okay, we found a tab.  This tab will turn into at least one character,
    // but it depends on which 'virtual column' it is in.  Compute that now.
    unsigned VCol = 0;
    while (BufPtr-VCol != MainBufStart && BufPtr[-VCol-1] != '\t' &&
           BufPtr[-VCol-1] != '\n' && BufPtr[-VCol-1] != '\r')
      ++VCol;
    
    // Okay, now that we know the virtual column, we know how many spaces to
    // insert.  We assume 8-character tab-stops.
    unsigned Spaces = 8-(VCol & 7);
    
    // Get the location of the tab.
    SourceLocation TabLoc =
      SourceLocation::getFileLoc(MainFileID, BufPtr-MainBufStart);
    
    // Rewrite the single tab character into a sequence of spaces.
    ReplaceText(TabLoc, 1, "        ", Spaces);
  }
}


void RewriteObjC::RewriteForwardClassDecl(ObjCClassDecl *ClassDecl) {
  int numDecls = ClassDecl->getNumForwardDecls();
  ObjCInterfaceDecl **ForwardDecls = ClassDecl->getForwardDecls();
  
  // Get the start location and compute the semi location.
  SourceLocation startLoc = ClassDecl->getLocation();
  const char *startBuf = SM->getCharacterData(startLoc);
  const char *semiPtr = strchr(startBuf, ';');
  
  // Translate to typedef's that forward reference structs with the same name
  // as the class. As a convenience, we include the original declaration
  // as a comment.
  std::string typedefString;
  typedefString += "// ";
  typedefString.append(startBuf, semiPtr-startBuf+1);
  typedefString += "\n";
  for (int i = 0; i < numDecls; i++) {
    ObjCInterfaceDecl *ForwardDecl = ForwardDecls[i];
    typedefString += "#ifndef _REWRITER_typedef_";
    typedefString += ForwardDecl->getName();
    typedefString += "\n";
    typedefString += "#define _REWRITER_typedef_";
    typedefString += ForwardDecl->getName();
    typedefString += "\n";
    typedefString += "typedef struct objc_object ";
    typedefString += ForwardDecl->getName();
    typedefString += ";\n#endif\n";
  }
  
  // Replace the @class with typedefs corresponding to the classes.
  ReplaceText(startLoc, semiPtr-startBuf+1, 
              typedefString.c_str(), typedefString.size());
}

void RewriteObjC::RewriteMethodDeclaration(ObjCMethodDecl *Method) {
  SourceLocation LocStart = Method->getLocStart();
  SourceLocation LocEnd = Method->getLocEnd();
    
  if (SM->getLineNumber(LocEnd) > SM->getLineNumber(LocStart)) {
    InsertText(LocStart, "#if 0\n", 6);
    ReplaceText(LocEnd, 1, ";\n#endif\n", 9);
  } else {
    InsertText(LocStart, "// ", 3);
  }
}

void RewriteObjC::RewriteProperties(unsigned nProperties, ObjCPropertyDecl **Properties) 
{
  for (unsigned i = 0; i < nProperties; i++) {
    ObjCPropertyDecl *Property = Properties[i];
    SourceLocation Loc = Property->getLocation();
    
    ReplaceText(Loc, 0, "// ", 3);
    
    // FIXME: handle properties that are declared across multiple lines.
  }
}

void RewriteObjC::RewriteCategoryDecl(ObjCCategoryDecl *CatDecl) {
  SourceLocation LocStart = CatDecl->getLocStart();
  
  // FIXME: handle category headers that are declared across multiple lines.
  ReplaceText(LocStart, 0, "// ", 3);
  
  for (ObjCCategoryDecl::instmeth_iterator I = CatDecl->instmeth_begin(), 
       E = CatDecl->instmeth_end(); I != E; ++I)
    RewriteMethodDeclaration(*I);
  for (ObjCCategoryDecl::classmeth_iterator I = CatDecl->classmeth_begin(), 
       E = CatDecl->classmeth_end(); I != E; ++I)
    RewriteMethodDeclaration(*I);

  // Lastly, comment out the @end.
  ReplaceText(CatDecl->getAtEndLoc(), 0, "// ", 3);
}

void RewriteObjC::RewriteProtocolDecl(ObjCProtocolDecl *PDecl) {
  std::pair<const char*, const char*> MainBuf = SM->getBufferData(MainFileID);
  
  SourceLocation LocStart = PDecl->getLocStart();
  
  // FIXME: handle protocol headers that are declared across multiple lines.
  ReplaceText(LocStart, 0, "// ", 3);
  
  for (ObjCProtocolDecl::instmeth_iterator I = PDecl->instmeth_begin(), 
       E = PDecl->instmeth_end(); I != E; ++I)
    RewriteMethodDeclaration(*I);
  for (ObjCProtocolDecl::classmeth_iterator I = PDecl->classmeth_begin(), 
       E = PDecl->classmeth_end(); I != E; ++I)
    RewriteMethodDeclaration(*I);

  // Lastly, comment out the @end.
  SourceLocation LocEnd = PDecl->getAtEndLoc();
  ReplaceText(LocEnd, 0, "// ", 3);

  // Must comment out @optional/@required
  const char *startBuf = SM->getCharacterData(LocStart);
  const char *endBuf = SM->getCharacterData(LocEnd);
  for (const char *p = startBuf; p < endBuf; p++) {
    if (*p == '@' && !strncmp(p+1, "optional", strlen("optional"))) {
      std::string CommentedOptional = "/* @optional */";
      SourceLocation OptionalLoc = LocStart.getFileLocWithOffset(p-startBuf);
      ReplaceText(OptionalLoc, strlen("@optional"),
                  CommentedOptional.c_str(), CommentedOptional.size());
      
    }
    else if (*p == '@' && !strncmp(p+1, "required", strlen("required"))) {
      std::string CommentedRequired = "/* @required */";
      SourceLocation OptionalLoc = LocStart.getFileLocWithOffset(p-startBuf);
      ReplaceText(OptionalLoc, strlen("@required"),
                  CommentedRequired.c_str(), CommentedRequired.size());
      
    }
  }
}

void RewriteObjC::RewriteForwardProtocolDecl(ObjCForwardProtocolDecl *PDecl) {
  SourceLocation LocStart = PDecl->getLocation();
  if (LocStart.isInvalid())
    assert(false && "Invalid SourceLocation");
  // FIXME: handle forward protocol that are declared across multiple lines.
  ReplaceText(LocStart, 0, "// ", 3);
}

void RewriteObjC::RewriteObjCMethodDecl(ObjCMethodDecl *OMD, 
                                        std::string &ResultStr) {
  const FunctionType *FPRetType = 0;
  ResultStr += "\nstatic ";
  if (OMD->getResultType()->isObjCQualifiedIdType())
    ResultStr += "id";
  else if (OMD->getResultType()->isFunctionPointerType()) {
    // needs special handling, since pointer-to-functions have special
    // syntax (where a decaration models use).
    QualType retType = OMD->getResultType();
    if (const PointerType* PT = retType->getAsPointerType()) {
      QualType PointeeTy = PT->getPointeeType();
      if ((FPRetType = PointeeTy->getAsFunctionType())) {
        ResultStr += FPRetType->getResultType().getAsString();
        ResultStr += "(*";
      }
    }
  } else
    ResultStr += OMD->getResultType().getAsString();
  ResultStr += " ";
  
  // Unique method name
  std::string NameStr;
  
  if (OMD->isInstance())
    NameStr += "_I_";
  else
    NameStr += "_C_";
  
  NameStr += OMD->getClassInterface()->getName();
  NameStr += "_";
  
  NamedDecl *MethodContext = OMD->getMethodContext();
  if (ObjCCategoryImplDecl *CID = 
      dyn_cast<ObjCCategoryImplDecl>(MethodContext)) {
    NameStr += CID->getName();
    NameStr += "_";
  }
  // Append selector names, replacing ':' with '_' 
  if (OMD->getSelector().getName().find(':') == std::string::npos)
    NameStr +=  OMD->getSelector().getName();
  else {
    std::string selString = OMD->getSelector().getName();
    int len = selString.size();
    for (int i = 0; i < len; i++)
      if (selString[i] == ':')
        selString[i] = '_';
    NameStr += selString;
  }
  // Remember this name for metadata emission
  MethodInternalNames[OMD] = NameStr;
  ResultStr += NameStr;
  
  // Rewrite arguments
  ResultStr += "(";
  
  // invisible arguments
  if (OMD->isInstance()) {
    QualType selfTy = Context->getObjCInterfaceType(OMD->getClassInterface());
    selfTy = Context->getPointerType(selfTy);
    if (!LangOpts.Microsoft) {
      if (ObjCSynthesizedStructs.count(OMD->getClassInterface()))
        ResultStr += "struct ";
    }
    // When rewriting for Microsoft, explicitly omit the structure name.
    ResultStr += OMD->getClassInterface()->getName();
    ResultStr += " *";
  }
  else
    ResultStr += Context->getObjCIdType().getAsString();
  
  ResultStr += " self, ";
  ResultStr += Context->getObjCSelType().getAsString();
  ResultStr += " _cmd";
  
  // Method arguments.
  for (unsigned i = 0; i < OMD->getNumParams(); i++) {
    ParmVarDecl *PDecl = OMD->getParamDecl(i);
    ResultStr += ", ";
    if (PDecl->getType()->isObjCQualifiedIdType()) {
      ResultStr += "id ";
      ResultStr += PDecl->getName();
    } else {
      std::string Name = PDecl->getName();
      PDecl->getType().getAsStringInternal(Name);
      ResultStr += Name;
    }
  }
  if (OMD->isVariadic())
    ResultStr += ", ...";
  ResultStr += ") ";
  
  if (FPRetType) {
    ResultStr += ")"; // close the precedence "scope" for "*".
    
    // Now, emit the argument types (if any).
    if (const FunctionTypeProto *FT = dyn_cast<FunctionTypeProto>(FPRetType)) {
      ResultStr += "(";
      for (unsigned i = 0, e = FT->getNumArgs(); i != e; ++i) {
        if (i) ResultStr += ", ";
        std::string ParamStr = FT->getArgType(i).getAsString();
        ResultStr += ParamStr;
      }
      if (FT->isVariadic()) {
        if (FT->getNumArgs()) ResultStr += ", ";
        ResultStr += "...";
      }
      ResultStr += ")";
    } else {
      ResultStr += "()";
    }
  }
}
void RewriteObjC::RewriteImplementationDecl(NamedDecl *OID) {
  ObjCImplementationDecl *IMD = dyn_cast<ObjCImplementationDecl>(OID);
  ObjCCategoryImplDecl *CID = dyn_cast<ObjCCategoryImplDecl>(OID);
  
  if (IMD)
    InsertText(IMD->getLocStart(), "// ", 3);
  else
    InsertText(CID->getLocStart(), "// ", 3);
  
  for (ObjCCategoryImplDecl::instmeth_iterator
       I = IMD ? IMD->instmeth_begin() : CID->instmeth_begin(),
       E = IMD ? IMD->instmeth_end() : CID->instmeth_end(); I != E; ++I) {
    std::string ResultStr;
    ObjCMethodDecl *OMD = *I;
    RewriteObjCMethodDecl(OMD, ResultStr);
    SourceLocation LocStart = OMD->getLocStart();
    SourceLocation LocEnd = OMD->getBody()->getLocStart();
    
    const char *startBuf = SM->getCharacterData(LocStart);
    const char *endBuf = SM->getCharacterData(LocEnd);
    ReplaceText(LocStart, endBuf-startBuf,
                ResultStr.c_str(), ResultStr.size());
  }
  
  for (ObjCCategoryImplDecl::classmeth_iterator
       I = IMD ? IMD->classmeth_begin() : CID->classmeth_begin(),
       E = IMD ? IMD->classmeth_end() : CID->classmeth_end(); I != E; ++I) {
    std::string ResultStr;
    ObjCMethodDecl *OMD = *I;
    RewriteObjCMethodDecl(OMD, ResultStr);
    SourceLocation LocStart = OMD->getLocStart();
    SourceLocation LocEnd = OMD->getBody()->getLocStart();
    
    const char *startBuf = SM->getCharacterData(LocStart);
    const char *endBuf = SM->getCharacterData(LocEnd);
    ReplaceText(LocStart, endBuf-startBuf,
                ResultStr.c_str(), ResultStr.size());    
  }
  if (IMD)
    InsertText(IMD->getLocEnd(), "// ", 3);
  else
   InsertText(CID->getLocEnd(), "// ", 3); 
}

void RewriteObjC::RewriteInterfaceDecl(ObjCInterfaceDecl *ClassDecl) {
  std::string ResultStr;
  if (!ObjCForwardDecls.count(ClassDecl)) {
    // we haven't seen a forward decl - generate a typedef.
    ResultStr = "#ifndef _REWRITER_typedef_";
    ResultStr += ClassDecl->getName();
    ResultStr += "\n";
    ResultStr += "#define _REWRITER_typedef_";
    ResultStr += ClassDecl->getName();
    ResultStr += "\n";
    ResultStr += "typedef struct objc_object ";
    ResultStr += ClassDecl->getName();
    ResultStr += ";\n#endif\n";
    // Mark this typedef as having been generated.
    ObjCForwardDecls.insert(ClassDecl);
  }
  SynthesizeObjCInternalStruct(ClassDecl, ResultStr);
    
  RewriteProperties(ClassDecl->getNumPropertyDecl(),
                    ClassDecl->getPropertyDecl());
  for (ObjCInterfaceDecl::instmeth_iterator I = ClassDecl->instmeth_begin(), 
       E = ClassDecl->instmeth_end(); I != E; ++I)
    RewriteMethodDeclaration(*I);
  for (ObjCInterfaceDecl::classmeth_iterator I = ClassDecl->classmeth_begin(), 
       E = ClassDecl->classmeth_end(); I != E; ++I)
    RewriteMethodDeclaration(*I);

  // Lastly, comment out the @end.
  ReplaceText(ClassDecl->getAtEndLoc(), 0, "// ", 3);
}

Stmt *RewriteObjC::RewriteObjCIvarRefExpr(ObjCIvarRefExpr *IV, 
                                          SourceLocation OrigStart) {
  ObjCIvarDecl *D = IV->getDecl();
  if (CurMethodDef) {
    if (const PointerType *pType = IV->getBase()->getType()->getAsPointerType()) {
      ObjCInterfaceType *iFaceDecl = dyn_cast<ObjCInterfaceType>(pType->getPointeeType());
      // lookup which class implements the instance variable.
      ObjCInterfaceDecl *clsDeclared = 0;
      iFaceDecl->getDecl()->lookupInstanceVariable(D->getIdentifier(), clsDeclared);
      assert(clsDeclared && "RewriteObjCIvarRefExpr(): Can't find class");
      
      // Synthesize an explicit cast to gain access to the ivar.
      std::string RecName = clsDeclared->getIdentifier()->getName();
      RecName += "_IMPL";
      IdentifierInfo *II = &Context->Idents.get(RecName.c_str());
      RecordDecl *RD = RecordDecl::Create(*Context, TagDecl::TK_struct, TUDecl,
                                          SourceLocation(), II);
      assert(RD && "RewriteObjCIvarRefExpr(): Can't find RecordDecl");
      QualType castT = Context->getPointerType(Context->getTagDeclType(RD));
      CastExpr *castExpr = new CStyleCastExpr(castT, IV->getBase(), castT,
                                                 SourceLocation());
      // Don't forget the parens to enforce the proper binding.
      ParenExpr *PE = new ParenExpr(IV->getBase()->getLocStart(),
                                    IV->getBase()->getLocEnd(),
                                    castExpr);
      if (IV->isFreeIvar() && 
          CurMethodDef->getClassInterface() == iFaceDecl->getDecl()) {
        MemberExpr *ME = new MemberExpr(PE, true, D, IV->getLocation(),
                                        D->getType());
        ReplaceStmt(IV, ME);
        delete IV;
        return ME;
      }
       
      ReplaceStmt(IV->getBase(), PE);
      // Cannot delete IV->getBase(), since PE points to it.
      // Replace the old base with the cast. This is important when doing
      // embedded rewrites. For example, [newInv->_container addObject:0].
      IV->setBase(PE); 
      return IV;
    }
  } else { // we are outside a method.
    assert(!IV->isFreeIvar() && "Cannot have a free standing ivar outside a method");
      
    // Explicit ivar refs need to have a cast inserted.
    // FIXME: consider sharing some of this code with the code above.
    if (const PointerType *pType = IV->getBase()->getType()->getAsPointerType()) {
      ObjCInterfaceType *iFaceDecl = dyn_cast<ObjCInterfaceType>(pType->getPointeeType());
      // lookup which class implements the instance variable.
      ObjCInterfaceDecl *clsDeclared = 0;
      iFaceDecl->getDecl()->lookupInstanceVariable(D->getIdentifier(), clsDeclared);
      assert(clsDeclared && "RewriteObjCIvarRefExpr(): Can't find class");
      
      // Synthesize an explicit cast to gain access to the ivar.
      std::string RecName = clsDeclared->getIdentifier()->getName();
      RecName += "_IMPL";
      IdentifierInfo *II = &Context->Idents.get(RecName.c_str());
      RecordDecl *RD = RecordDecl::Create(*Context, TagDecl::TK_struct, TUDecl,
                                          SourceLocation(), II);
      assert(RD && "RewriteObjCIvarRefExpr(): Can't find RecordDecl");
      QualType castT = Context->getPointerType(Context->getTagDeclType(RD));
      CastExpr *castExpr = new CStyleCastExpr(castT, IV->getBase(), castT,
                                                 SourceLocation());
      // Don't forget the parens to enforce the proper binding.
      ParenExpr *PE = new ParenExpr(IV->getBase()->getLocStart(),
                                    IV->getBase()->getLocEnd(), castExpr);
      ReplaceStmt(IV->getBase(), PE);
      // Cannot delete IV->getBase(), since PE points to it.
      // Replace the old base with the cast. This is important when doing
      // embedded rewrites. For example, [newInv->_container addObject:0].
      IV->setBase(PE); 
      return IV;
    }
  }
  return IV;
}

/// SynthCountByEnumWithState - To print:
/// ((unsigned int (*)
///  (id, SEL, struct __objcFastEnumerationState *, id *, unsigned int))
///  (void *)objc_msgSend)((id)l_collection, 
///                        sel_registerName(
///                          "countByEnumeratingWithState:objects:count:"), 
///                        &enumState, 
///                        (id *)items, (unsigned int)16)
///
void RewriteObjC::SynthCountByEnumWithState(std::string &buf) {
  buf += "((unsigned int (*) (id, SEL, struct __objcFastEnumerationState *, "
  "id *, unsigned int))(void *)objc_msgSend)";
  buf += "\n\t\t";
  buf += "((id)l_collection,\n\t\t";
  buf += "sel_registerName(\"countByEnumeratingWithState:objects:count:\"),";
  buf += "\n\t\t";
  buf += "&enumState, "
         "(id *)items, (unsigned int)16)";
}

/// RewriteBreakStmt - Rewrite for a break-stmt inside an ObjC2's foreach
/// statement to exit to its outer synthesized loop.
///
Stmt *RewriteObjC::RewriteBreakStmt(BreakStmt *S) {
  if (Stmts.empty() || !isa<ObjCForCollectionStmt>(Stmts.back()))
    return S;
  // replace break with goto __break_label
  std::string buf;
  
  SourceLocation startLoc = S->getLocStart();
  buf = "goto __break_label_";
  buf += utostr(ObjCBcLabelNo.back());
  ReplaceText(startLoc, strlen("break"), buf.c_str(), buf.size());

  return 0;
}

/// RewriteContinueStmt - Rewrite for a continue-stmt inside an ObjC2's foreach
/// statement to continue with its inner synthesized loop.
///
Stmt *RewriteObjC::RewriteContinueStmt(ContinueStmt *S) {
  if (Stmts.empty() || !isa<ObjCForCollectionStmt>(Stmts.back()))
    return S;
  // replace continue with goto __continue_label
  std::string buf;
  
  SourceLocation startLoc = S->getLocStart();
  buf = "goto __continue_label_";
  buf += utostr(ObjCBcLabelNo.back());
  ReplaceText(startLoc, strlen("continue"), buf.c_str(), buf.size());
  
  return 0;
}

/// RewriteObjCForCollectionStmt - Rewriter for ObjC2's foreach statement.
///  It rewrites:
/// for ( type elem in collection) { stmts; }
 
/// Into:
/// {
///   type elem; 
///   struct __objcFastEnumerationState enumState = { 0 };
///   id items[16];
///   id l_collection = (id)collection;
///   unsigned long limit = [l_collection countByEnumeratingWithState:&enumState 
///                                       objects:items count:16];
/// if (limit) {
///   unsigned long startMutations = *enumState.mutationsPtr;
///   do {
///        unsigned long counter = 0;
///        do {
///             if (startMutations != *enumState.mutationsPtr) 
///               objc_enumerationMutation(l_collection);
///             elem = (type)enumState.itemsPtr[counter++];
///             stmts;
///             __continue_label: ;
///        } while (counter < limit);
///   } while (limit = [l_collection countByEnumeratingWithState:&enumState 
///                                  objects:items count:16]);
///   elem = nil;
///   __break_label: ;
///  }
///  else
///       elem = nil;
///  }
///
Stmt *RewriteObjC::RewriteObjCForCollectionStmt(ObjCForCollectionStmt *S,
                                                SourceLocation OrigEnd) {
  assert(!Stmts.empty() && "ObjCForCollectionStmt - Statement stack empty");
  assert(isa<ObjCForCollectionStmt>(Stmts.back()) && 
         "ObjCForCollectionStmt Statement stack mismatch");
  assert(!ObjCBcLabelNo.empty() && 
         "ObjCForCollectionStmt - Label No stack empty");
  
  SourceLocation startLoc = S->getLocStart();
  const char *startBuf = SM->getCharacterData(startLoc);
  const char *elementName;
  std::string elementTypeAsString;
  std::string buf;
  buf = "\n{\n\t";
  if (DeclStmt *DS = dyn_cast<DeclStmt>(S->getElement())) {
    // type elem;
    ScopedDecl* D = DS->getSolitaryDecl();
    QualType ElementType = cast<ValueDecl>(D)->getType();
    elementTypeAsString = ElementType.getAsString();
    buf += elementTypeAsString;
    buf += " ";
    elementName = D->getName();
    buf += elementName;
    buf += ";\n\t";
  }
  else {
    DeclRefExpr *DR = cast<DeclRefExpr>(S->getElement());
    elementName = DR->getDecl()->getName();
    elementTypeAsString 
      = cast<ValueDecl>(DR->getDecl())->getType().getAsString();
  }
  
  // struct __objcFastEnumerationState enumState = { 0 };
  buf += "struct __objcFastEnumerationState enumState = { 0 };\n\t";
  // id items[16];
  buf += "id items[16];\n\t";
  // id l_collection = (id)
  buf += "id l_collection = (id)";
  // Find start location of 'collection' the hard way!
  const char *startCollectionBuf = startBuf;
  startCollectionBuf += 3;  // skip 'for'
  startCollectionBuf = strchr(startCollectionBuf, '(');
  startCollectionBuf++; // skip '('
  // find 'in' and skip it.
  while (*startCollectionBuf != ' ' ||
         *(startCollectionBuf+1) != 'i' || *(startCollectionBuf+2) != 'n' ||
         (*(startCollectionBuf+3) != ' ' &&
          *(startCollectionBuf+3) != '[' && *(startCollectionBuf+3) != '('))
    startCollectionBuf++;
  startCollectionBuf += 3;
  
  // Replace: "for (type element in" with string constructed thus far. 
  ReplaceText(startLoc, startCollectionBuf - startBuf,
              buf.c_str(), buf.size());
  // Replace ')' in for '(' type elem in collection ')' with ';'
  SourceLocation rightParenLoc = S->getRParenLoc();
  const char *rparenBuf = SM->getCharacterData(rightParenLoc);
  SourceLocation lparenLoc = startLoc.getFileLocWithOffset(rparenBuf-startBuf);
  buf = ";\n\t";
  
  // unsigned long limit = [l_collection countByEnumeratingWithState:&enumState
  //                                   objects:items count:16];
  // which is synthesized into:
  // unsigned int limit = 
  // ((unsigned int (*)
  //  (id, SEL, struct __objcFastEnumerationState *, id *, unsigned int))
  //  (void *)objc_msgSend)((id)l_collection, 
  //                        sel_registerName(
  //                          "countByEnumeratingWithState:objects:count:"), 
  //                        (struct __objcFastEnumerationState *)&state, 
  //                        (id *)items, (unsigned int)16);
  buf += "unsigned long limit =\n\t\t";
  SynthCountByEnumWithState(buf);
  buf += ";\n\t";
  /// if (limit) {
  ///   unsigned long startMutations = *enumState.mutationsPtr;
  ///   do {
  ///        unsigned long counter = 0;
  ///        do {
  ///             if (startMutations != *enumState.mutationsPtr) 
  ///               objc_enumerationMutation(l_collection);
  ///             elem = (type)enumState.itemsPtr[counter++];
  buf += "if (limit) {\n\t";
  buf += "unsigned long startMutations = *enumState.mutationsPtr;\n\t";
  buf += "do {\n\t\t";
  buf += "unsigned long counter = 0;\n\t\t";
  buf += "do {\n\t\t\t";
  buf += "if (startMutations != *enumState.mutationsPtr)\n\t\t\t\t";
  buf += "objc_enumerationMutation(l_collection);\n\t\t\t";
  buf += elementName;
  buf += " = (";
  buf += elementTypeAsString;
  buf += ")enumState.itemsPtr[counter++];";
  // Replace ')' in for '(' type elem in collection ')' with all of these.
  ReplaceText(lparenLoc, 1, buf.c_str(), buf.size());
  
  ///            __continue_label: ;
  ///        } while (counter < limit);
  ///   } while (limit = [l_collection countByEnumeratingWithState:&enumState 
  ///                                  objects:items count:16]);
  ///   elem = nil;
  ///   __break_label: ;
  ///  }
  ///  else
  ///       elem = nil;
  ///  }
  /// 
  buf = ";\n\t";
  buf += "__continue_label_";
  buf += utostr(ObjCBcLabelNo.back());
  buf += ": ;";
  buf += "\n\t\t";
  buf += "} while (counter < limit);\n\t";
  buf += "} while (limit = ";
  SynthCountByEnumWithState(buf);
  buf += ");\n\t";
  buf += elementName;
  buf += " = nil;\n\t";
  buf += "__break_label_";
  buf += utostr(ObjCBcLabelNo.back());
  buf += ": ;\n\t";
  buf += "}\n\t";
  buf += "else\n\t\t";
  buf += elementName;
  buf += " = nil;\n";
  buf += "}\n";
  
  // Insert all these *after* the statement body.
  if (isa<CompoundStmt>(S->getBody())) {
    SourceLocation endBodyLoc = OrigEnd.getFileLocWithOffset(1);
    InsertText(endBodyLoc, buf.c_str(), buf.size());
  } else {
    /* Need to treat single statements specially. For example:
     *
     *     for (A *a in b) if (stuff()) break;
     *     for (A *a in b) xxxyy;
     *
     * The following code simply scans ahead to the semi to find the actual end.
     */
    const char *stmtBuf = SM->getCharacterData(OrigEnd);
    const char *semiBuf = strchr(stmtBuf, ';');
    assert(semiBuf && "Can't find ';'");
    SourceLocation endBodyLoc = OrigEnd.getFileLocWithOffset(semiBuf-stmtBuf+1);
    InsertText(endBodyLoc, buf.c_str(), buf.size());
  }
  Stmts.pop_back();
  ObjCBcLabelNo.pop_back();
  return 0;
}

/// RewriteObjCSynchronizedStmt - 
/// This routine rewrites @synchronized(expr) stmt;
/// into:
/// objc_sync_enter(expr);
/// @try stmt @finally { objc_sync_exit(expr); }
///
Stmt *RewriteObjC::RewriteObjCSynchronizedStmt(ObjCAtSynchronizedStmt *S) {
  // Get the start location and compute the semi location.
  SourceLocation startLoc = S->getLocStart();
  const char *startBuf = SM->getCharacterData(startLoc);
  
  assert((*startBuf == '@') && "bogus @synchronized location");
  
  std::string buf; 
  buf = "objc_sync_enter((id)";
  const char *lparenBuf = startBuf;
  while (*lparenBuf != '(') lparenBuf++;
  ReplaceText(startLoc, lparenBuf-startBuf+1, buf.c_str(), buf.size());
  // We can't use S->getSynchExpr()->getLocEnd() to find the end location, since 
  // the sync expression is typically a message expression that's already 
  // been rewritten! (which implies the SourceLocation's are invalid).
  SourceLocation endLoc = S->getSynchBody()->getLocStart();
  const char *endBuf = SM->getCharacterData(endLoc);
  while (*endBuf != ')') endBuf--;
  SourceLocation rparenLoc = startLoc.getFileLocWithOffset(endBuf-startBuf);
  buf = ");\n";
  // declare a new scope with two variables, _stack and _rethrow.
  buf += "/* @try scope begin */ \n{ struct _objc_exception_data {\n";
  buf += "int buf[18/*32-bit i386*/];\n";
  buf += "char *pointers[4];} _stack;\n";
  buf += "id volatile _rethrow = 0;\n";
  buf += "objc_exception_try_enter(&_stack);\n";
  buf += "if (!_setjmp(_stack.buf)) /* @try block continue */\n";
  ReplaceText(rparenLoc, 1, buf.c_str(), buf.size());
  startLoc = S->getSynchBody()->getLocEnd();
  startBuf = SM->getCharacterData(startLoc);
  
  assert((*startBuf == '}') && "bogus @synchronized block");
  SourceLocation lastCurlyLoc = startLoc;
  buf = "}\nelse {\n";
  buf += "  _rethrow = objc_exception_extract(&_stack);\n";
  buf += "  if (!_rethrow) objc_exception_try_exit(&_stack);\n";
  buf += "  objc_sync_exit(";
  Expr *syncExpr = new CStyleCastExpr(Context->getObjCIdType(), 
                                         S->getSynchExpr(), 
                                         Context->getObjCIdType(),
                                         SourceLocation());
  std::string syncExprBufS;
  llvm::raw_string_ostream syncExprBuf(syncExprBufS);
  syncExpr->printPretty(syncExprBuf);
  buf += syncExprBuf.str();
  buf += ");\n";
  buf += "  if (_rethrow) objc_exception_throw(_rethrow);\n";
  buf += "}\n";
  buf += "}";
  
  ReplaceText(lastCurlyLoc, 1, buf.c_str(), buf.size());
  return 0;
}

Stmt *RewriteObjC::RewriteObjCTryStmt(ObjCAtTryStmt *S) {
  // Get the start location and compute the semi location.
  SourceLocation startLoc = S->getLocStart();
  const char *startBuf = SM->getCharacterData(startLoc);
  
  assert((*startBuf == '@') && "bogus @try location");

  std::string buf;
  // declare a new scope with two variables, _stack and _rethrow.
  buf = "/* @try scope begin */ { struct _objc_exception_data {\n";
  buf += "int buf[18/*32-bit i386*/];\n";
  buf += "char *pointers[4];} _stack;\n";
  buf += "id volatile _rethrow = 0;\n";
  buf += "objc_exception_try_enter(&_stack);\n";
  buf += "if (!_setjmp(_stack.buf)) /* @try block continue */\n";

  ReplaceText(startLoc, 4, buf.c_str(), buf.size());
  
  startLoc = S->getTryBody()->getLocEnd();
  startBuf = SM->getCharacterData(startLoc);

  assert((*startBuf == '}') && "bogus @try block");
  
  SourceLocation lastCurlyLoc = startLoc;
  ObjCAtCatchStmt *catchList = S->getCatchStmts();
  if (catchList) {
    startLoc = startLoc.getFileLocWithOffset(1);
    buf = " /* @catch begin */ else {\n";
    buf += " id _caught = objc_exception_extract(&_stack);\n";
    buf += " objc_exception_try_enter (&_stack);\n";
    buf += " if (_setjmp(_stack.buf))\n";
    buf += "   _rethrow = objc_exception_extract(&_stack);\n";
    buf += " else { /* @catch continue */";
    
    InsertText(startLoc, buf.c_str(), buf.size());
  } else { /* no catch list */
    buf = "}\nelse {\n";
    buf += "  _rethrow = objc_exception_extract(&_stack);\n";
    buf += "}";
    ReplaceText(lastCurlyLoc, 1, buf.c_str(), buf.size());
  }
  bool sawIdTypedCatch = false;
  Stmt *lastCatchBody = 0;
  while (catchList) {
    Stmt *catchStmt = catchList->getCatchParamStmt();

    if (catchList == S->getCatchStmts()) 
      buf = "if ("; // we are generating code for the first catch clause
    else
      buf = "else if (";
    startLoc = catchList->getLocStart();
    startBuf = SM->getCharacterData(startLoc);
    
    assert((*startBuf == '@') && "bogus @catch location");
    
    const char *lParenLoc = strchr(startBuf, '(');

    if (catchList->hasEllipsis()) {
      // Now rewrite the body...
      lastCatchBody = catchList->getCatchBody();
      SourceLocation bodyLoc = lastCatchBody->getLocStart();
      const char *bodyBuf = SM->getCharacterData(bodyLoc);
      assert(*SM->getCharacterData(catchList->getRParenLoc()) == ')' &&
             "bogus @catch paren location");
      assert((*bodyBuf == '{') && "bogus @catch body location");
      
      buf += "1) { id _tmp = _caught;";
      Rewrite.ReplaceText(startLoc, bodyBuf-startBuf+1, 
                          buf.c_str(), buf.size());      
    } else if (DeclStmt *declStmt = dyn_cast<DeclStmt>(catchStmt)) {
      QualType t = dyn_cast<ValueDecl>(declStmt->getSolitaryDecl())->getType();
      if (t == Context->getObjCIdType()) {
        buf += "1) { ";
        ReplaceText(startLoc, lParenLoc-startBuf+1, buf.c_str(), buf.size());
        sawIdTypedCatch = true;
      } else if (const PointerType *pType = t->getAsPointerType()) { 
        ObjCInterfaceType *cls; // Should be a pointer to a class.
        
        cls = dyn_cast<ObjCInterfaceType>(pType->getPointeeType().getTypePtr());
        if (cls) {
          buf += "objc_exception_match((struct objc_class *)objc_getClass(\"";
          buf += cls->getDecl()->getName();
          buf += "\"), (struct objc_object *)_caught)) { ";
          ReplaceText(startLoc, lParenLoc-startBuf+1, buf.c_str(), buf.size());
        }
      }
      // Now rewrite the body...
      lastCatchBody = catchList->getCatchBody();
      SourceLocation rParenLoc = catchList->getRParenLoc();
      SourceLocation bodyLoc = lastCatchBody->getLocStart();
      const char *bodyBuf = SM->getCharacterData(bodyLoc);
      const char *rParenBuf = SM->getCharacterData(rParenLoc);
      assert((*rParenBuf == ')') && "bogus @catch paren location");
      assert((*bodyBuf == '{') && "bogus @catch body location");
        
      buf = " = _caught;";
      // Here we replace ") {" with "= _caught;" (which initializes and 
      // declares the @catch parameter).
      ReplaceText(rParenLoc, bodyBuf-rParenBuf+1, buf.c_str(), buf.size());
    } else if (!isa<NullStmt>(catchStmt)) {
      assert(false && "@catch rewrite bug");
    }
    // make sure all the catch bodies get rewritten!
    catchList = catchList->getNextCatchStmt();
  }
  // Complete the catch list...
  if (lastCatchBody) {
    SourceLocation bodyLoc = lastCatchBody->getLocEnd();
    assert(*SM->getCharacterData(bodyLoc) == '}' &&
           "bogus @catch body location");
  
    // Insert the last (implicit) else clause *before* the right curly brace.
    bodyLoc = bodyLoc.getFileLocWithOffset(-1);
    buf = "} /* last catch end */\n";
    buf += "else {\n";
    buf += " _rethrow = _caught;\n";
    buf += " objc_exception_try_exit(&_stack);\n";
    buf += "} } /* @catch end */\n";
    if (!S->getFinallyStmt())
      buf += "}\n";
    InsertText(bodyLoc, buf.c_str(), buf.size());
    
    // Set lastCurlyLoc
    lastCurlyLoc = lastCatchBody->getLocEnd();
  }
  if (ObjCAtFinallyStmt *finalStmt = S->getFinallyStmt()) {
    startLoc = finalStmt->getLocStart();
    startBuf = SM->getCharacterData(startLoc);
    assert((*startBuf == '@') && "bogus @finally start");
    
    buf = "/* @finally */";
    ReplaceText(startLoc, 8, buf.c_str(), buf.size());
    
    Stmt *body = finalStmt->getFinallyBody();
    SourceLocation startLoc = body->getLocStart();
    SourceLocation endLoc = body->getLocEnd();
    assert(*SM->getCharacterData(startLoc) == '{' &&
           "bogus @finally body location");
    assert(*SM->getCharacterData(endLoc) == '}' && 
           "bogus @finally body location");
  
    startLoc = startLoc.getFileLocWithOffset(1);
    buf = " if (!_rethrow) objc_exception_try_exit(&_stack);\n";
    InsertText(startLoc, buf.c_str(), buf.size());
    endLoc = endLoc.getFileLocWithOffset(-1);
    buf = " if (_rethrow) objc_exception_throw(_rethrow);\n";
    InsertText(endLoc, buf.c_str(), buf.size());
    
    // Set lastCurlyLoc
    lastCurlyLoc = body->getLocEnd();
  } else { /* no finally clause - make sure we synthesize an implicit one */
    buf = "{ /* implicit finally clause */\n";
    buf += " if (!_rethrow) objc_exception_try_exit(&_stack);\n";
    buf += " if (_rethrow) objc_exception_throw(_rethrow);\n";
    buf += "}";
    ReplaceText(lastCurlyLoc, 1, buf.c_str(), buf.size());
  }
  // Now emit the final closing curly brace...
  lastCurlyLoc = lastCurlyLoc.getFileLocWithOffset(1);
  buf = " } /* @try scope end */\n";
  InsertText(lastCurlyLoc, buf.c_str(), buf.size());
  return 0;
}

Stmt *RewriteObjC::RewriteObjCCatchStmt(ObjCAtCatchStmt *S) {
  return 0;
}

Stmt *RewriteObjC::RewriteObjCFinallyStmt(ObjCAtFinallyStmt *S) {
  return 0;
}

// This can't be done with ReplaceStmt(S, ThrowExpr), since 
// the throw expression is typically a message expression that's already 
// been rewritten! (which implies the SourceLocation's are invalid).
Stmt *RewriteObjC::RewriteObjCThrowStmt(ObjCAtThrowStmt *S) {
  // Get the start location and compute the semi location.
  SourceLocation startLoc = S->getLocStart();
  const char *startBuf = SM->getCharacterData(startLoc);
  
  assert((*startBuf == '@') && "bogus @throw location");

  std::string buf;
  /* void objc_exception_throw(id) __attribute__((noreturn)); */
  if (S->getThrowExpr())
    buf = "objc_exception_throw(";
  else // add an implicit argument
    buf = "objc_exception_throw(_caught";
  
  // handle "@  throw" correctly.
  const char *wBuf = strchr(startBuf, 'w');
  assert((*wBuf == 'w') && "@throw: can't find 'w'");
  ReplaceText(startLoc, wBuf-startBuf+1, buf.c_str(), buf.size());
  
  const char *semiBuf = strchr(startBuf, ';');
  assert((*semiBuf == ';') && "@throw: can't find ';'");
  SourceLocation semiLoc = startLoc.getFileLocWithOffset(semiBuf-startBuf);
  buf = ");";
  ReplaceText(semiLoc, 1, buf.c_str(), buf.size());
  return 0;
}

Stmt *RewriteObjC::RewriteAtEncode(ObjCEncodeExpr *Exp) {
  // Create a new string expression.
  QualType StrType = Context->getPointerType(Context->CharTy);
  std::string StrEncoding;
  Context->getObjCEncodingForType(Exp->getEncodedType(), StrEncoding);
  Expr *Replacement = new StringLiteral(StrEncoding.c_str(),
                                        StrEncoding.length(), false, StrType, 
                                        SourceLocation(), SourceLocation());
  ReplaceStmt(Exp, Replacement);
  
  // Replace this subexpr in the parent.
  delete Exp;
  return Replacement;
}

Stmt *RewriteObjC::RewriteAtSelector(ObjCSelectorExpr *Exp) {
  assert(SelGetUidFunctionDecl && "Can't find sel_registerName() decl");
  // Create a call to sel_registerName("selName").
  llvm::SmallVector<Expr*, 8> SelExprs;
  QualType argType = Context->getPointerType(Context->CharTy);
  SelExprs.push_back(new StringLiteral(Exp->getSelector().getName().c_str(),
                                       Exp->getSelector().getName().size(),
                                       false, argType, SourceLocation(),
                                       SourceLocation()));
  CallExpr *SelExp = SynthesizeCallToFunctionDecl(SelGetUidFunctionDecl,
                                                 &SelExprs[0], SelExprs.size());
  ReplaceStmt(Exp, SelExp);
  delete Exp;
  return SelExp;
}

CallExpr *RewriteObjC::SynthesizeCallToFunctionDecl(
  FunctionDecl *FD, Expr **args, unsigned nargs) {
  // Get the type, we will need to reference it in a couple spots.
  QualType msgSendType = FD->getType();
  
  // Create a reference to the objc_msgSend() declaration.
  DeclRefExpr *DRE = new DeclRefExpr(FD, msgSendType, SourceLocation());
                                     
  // Now, we cast the reference to a pointer to the objc_msgSend type.
  QualType pToFunc = Context->getPointerType(msgSendType);
  ImplicitCastExpr *ICE = new ImplicitCastExpr(pToFunc, DRE);
  
  const FunctionType *FT = msgSendType->getAsFunctionType();
  
  return new CallExpr(ICE, args, nargs, FT->getResultType(), SourceLocation());
}

static bool scanForProtocolRefs(const char *startBuf, const char *endBuf,
                                const char *&startRef, const char *&endRef) {
  while (startBuf < endBuf) {
    if (*startBuf == '<')
      startRef = startBuf; // mark the start.
    if (*startBuf == '>') {
      if (startRef && *startRef == '<') {
        endRef = startBuf; // mark the end.
        return true;
      }
      return false;
    }
    startBuf++;
  }
  return false;
}

static void scanToNextArgument(const char *&argRef) {
  int angle = 0;
  while (*argRef != ')' && (*argRef != ',' || angle > 0)) {
    if (*argRef == '<')
      angle++;
    else if (*argRef == '>')
      angle--;
    argRef++;
  }
  assert(angle == 0 && "scanToNextArgument - bad protocol type syntax");
}

bool RewriteObjC::needToScanForQualifiers(QualType T) {
  
  if (T->isObjCQualifiedIdType())
    return true;
  
  if (const PointerType *pType = T->getAsPointerType()) {
    Type *pointeeType = pType->getPointeeType().getTypePtr();
    if (isa<ObjCQualifiedInterfaceType>(pointeeType))
      return true; // we have "Class <Protocol> *".
  }
  return false;
}

void RewriteObjC::RewriteObjCQualifiedInterfaceTypes(Expr *E) {
  QualType Type = E->getType();
  if (needToScanForQualifiers(Type)) {
    SourceLocation Loc = E->getLocStart();
    const char *startBuf = SM->getCharacterData(Loc);
    const char *endBuf = SM->getCharacterData(E->getLocEnd());
    const char *startRef = 0, *endRef = 0;
    if (scanForProtocolRefs(startBuf, endBuf, startRef, endRef)) {
      // Get the locations of the startRef, endRef.
      SourceLocation LessLoc = Loc.getFileLocWithOffset(startRef-startBuf);
      SourceLocation GreaterLoc = Loc.getFileLocWithOffset(endRef-startBuf+1);
      // Comment out the protocol references.
      InsertText(LessLoc, "/*", 2);
      InsertText(GreaterLoc, "*/", 2);
    }
  }
}

void RewriteObjC::RewriteObjCQualifiedInterfaceTypes(Decl *Dcl) {
  SourceLocation Loc;
  QualType Type;
  const FunctionTypeProto *proto = 0;
  if (VarDecl *VD = dyn_cast<VarDecl>(Dcl)) {
    Loc = VD->getLocation();
    Type = VD->getType();
  }
  else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(Dcl)) {
    Loc = FD->getLocation();
    // Check for ObjC 'id' and class types that have been adorned with protocol
    // information (id<p>, C<p>*). The protocol references need to be rewritten!
    const FunctionType *funcType = FD->getType()->getAsFunctionType();
    assert(funcType && "missing function type");
    proto = dyn_cast<FunctionTypeProto>(funcType);
    if (!proto)
      return;
    Type = proto->getResultType();
  }
  else
    return;
  
  if (needToScanForQualifiers(Type)) {
    // Since types are unique, we need to scan the buffer.
    
    const char *endBuf = SM->getCharacterData(Loc);
    const char *startBuf = endBuf;
    while (*startBuf != ';' && *startBuf != '<' && startBuf != MainFileStart)
      startBuf--; // scan backward (from the decl location) for return type.
    const char *startRef = 0, *endRef = 0;
    if (scanForProtocolRefs(startBuf, endBuf, startRef, endRef)) {
      // Get the locations of the startRef, endRef.
      SourceLocation LessLoc = Loc.getFileLocWithOffset(startRef-endBuf);
      SourceLocation GreaterLoc = Loc.getFileLocWithOffset(endRef-endBuf+1);
      // Comment out the protocol references.
      InsertText(LessLoc, "/*", 2);
      InsertText(GreaterLoc, "*/", 2);
    }
  }
  if (!proto)
      return; // most likely, was a variable
  // Now check arguments.
  const char *startBuf = SM->getCharacterData(Loc);
  const char *startFuncBuf = startBuf;
  for (unsigned i = 0; i < proto->getNumArgs(); i++) {
    if (needToScanForQualifiers(proto->getArgType(i))) {
      // Since types are unique, we need to scan the buffer.
      
      const char *endBuf = startBuf;
      // scan forward (from the decl location) for argument types.
      scanToNextArgument(endBuf);
      const char *startRef = 0, *endRef = 0;
      if (scanForProtocolRefs(startBuf, endBuf, startRef, endRef)) {
        // Get the locations of the startRef, endRef.
        SourceLocation LessLoc = 
          Loc.getFileLocWithOffset(startRef-startFuncBuf);
        SourceLocation GreaterLoc = 
          Loc.getFileLocWithOffset(endRef-startFuncBuf+1);
        // Comment out the protocol references.
        InsertText(LessLoc, "/*", 2);
        InsertText(GreaterLoc, "*/", 2);
      }
      startBuf = ++endBuf;
    }
    else {
      // If the function name is derived from a macro expansion, then the
      // argument buffer will not follow the name. Need to speak with Chris.
      while (*startBuf && *startBuf != ')' && *startBuf != ',')
        startBuf++; // scan forward (from the decl location) for argument types.
      startBuf++;
    }
  }
}

// SynthSelGetUidFunctionDecl - SEL sel_registerName(const char *str);
void RewriteObjC::SynthSelGetUidFunctionDecl() {
  IdentifierInfo *SelGetUidIdent = &Context->Idents.get("sel_registerName");
  llvm::SmallVector<QualType, 16> ArgTys;
  ArgTys.push_back(Context->getPointerType(
    Context->CharTy.getQualifiedType(QualType::Const)));
  QualType getFuncType = Context->getFunctionType(Context->getObjCSelType(),
                                                   &ArgTys[0], ArgTys.size(),
                                                   false /*isVariadic*/, 0);
  SelGetUidFunctionDecl = FunctionDecl::Create(*Context, TUDecl,
                                           SourceLocation(), 
                                           SelGetUidIdent, getFuncType,
                                           FunctionDecl::Extern, false, 0);
}

// SynthGetProtocolFunctionDecl - Protocol objc_getProtocol(const char *proto);
void RewriteObjC::SynthGetProtocolFunctionDecl() {
  IdentifierInfo *SelGetProtoIdent = &Context->Idents.get("objc_getProtocol");
  llvm::SmallVector<QualType, 16> ArgTys;
  ArgTys.push_back(Context->getPointerType(
    Context->CharTy.getQualifiedType(QualType::Const)));
  QualType getFuncType = Context->getFunctionType(Context->getObjCProtoType(),
                                                  &ArgTys[0], ArgTys.size(),
                                                  false /*isVariadic*/, 0);
  GetProtocolFunctionDecl = FunctionDecl::Create(*Context, TUDecl,
                                             SourceLocation(), 
                                             SelGetProtoIdent, getFuncType,
                                             FunctionDecl::Extern, false, 0);
}

void RewriteObjC::RewriteFunctionDecl(FunctionDecl *FD) {
  // declared in <objc/objc.h>
  if (strcmp(FD->getName(), "sel_registerName") == 0) {
    SelGetUidFunctionDecl = FD;
    return;
  }
  RewriteObjCQualifiedInterfaceTypes(FD);
}

// SynthSuperContructorFunctionDecl - id objc_super(id obj, id super);
void RewriteObjC::SynthSuperContructorFunctionDecl() {
  if (SuperContructorFunctionDecl)
    return;
  IdentifierInfo *msgSendIdent = &Context->Idents.get("objc_super");
  llvm::SmallVector<QualType, 16> ArgTys;
  QualType argT = Context->getObjCIdType();
  assert(!argT.isNull() && "Can't find 'id' type");
  ArgTys.push_back(argT);
  ArgTys.push_back(argT);
  QualType msgSendType = Context->getFunctionType(Context->getObjCIdType(),
                                                  &ArgTys[0], ArgTys.size(),
                                                  false, 0);
  SuperContructorFunctionDecl = FunctionDecl::Create(*Context, TUDecl,
                                         SourceLocation(), 
                                         msgSendIdent, msgSendType,
                                         FunctionDecl::Extern, false, 0);
}

// SynthMsgSendFunctionDecl - id objc_msgSend(id self, SEL op, ...);
void RewriteObjC::SynthMsgSendFunctionDecl() {
  IdentifierInfo *msgSendIdent = &Context->Idents.get("objc_msgSend");
  llvm::SmallVector<QualType, 16> ArgTys;
  QualType argT = Context->getObjCIdType();
  assert(!argT.isNull() && "Can't find 'id' type");
  ArgTys.push_back(argT);
  argT = Context->getObjCSelType();
  assert(!argT.isNull() && "Can't find 'SEL' type");
  ArgTys.push_back(argT);
  QualType msgSendType = Context->getFunctionType(Context->getObjCIdType(),
                                                  &ArgTys[0], ArgTys.size(),
                                                  true /*isVariadic*/, 0);
  MsgSendFunctionDecl = FunctionDecl::Create(*Context, TUDecl,
                                         SourceLocation(),
                                         msgSendIdent, msgSendType,
                                         FunctionDecl::Extern, false, 0);
}

// SynthMsgSendSuperFunctionDecl - id objc_msgSendSuper(struct objc_super *, SEL op, ...);
void RewriteObjC::SynthMsgSendSuperFunctionDecl() {
  IdentifierInfo *msgSendIdent = &Context->Idents.get("objc_msgSendSuper");
  llvm::SmallVector<QualType, 16> ArgTys;
  RecordDecl *RD = RecordDecl::Create(*Context, TagDecl::TK_struct, TUDecl,
                                      SourceLocation(),
                                      &Context->Idents.get("objc_super"));
  QualType argT = Context->getPointerType(Context->getTagDeclType(RD));
  assert(!argT.isNull() && "Can't build 'struct objc_super *' type");
  ArgTys.push_back(argT);
  argT = Context->getObjCSelType();
  assert(!argT.isNull() && "Can't find 'SEL' type");
  ArgTys.push_back(argT);
  QualType msgSendType = Context->getFunctionType(Context->getObjCIdType(),
                                                  &ArgTys[0], ArgTys.size(),
                                                  true /*isVariadic*/, 0);
  MsgSendSuperFunctionDecl = FunctionDecl::Create(*Context, TUDecl,
                                              SourceLocation(), 
                                              msgSendIdent, msgSendType,
                                              FunctionDecl::Extern, false, 0);
}

// SynthMsgSendStretFunctionDecl - id objc_msgSend_stret(id self, SEL op, ...);
void RewriteObjC::SynthMsgSendStretFunctionDecl() {
  IdentifierInfo *msgSendIdent = &Context->Idents.get("objc_msgSend_stret");
  llvm::SmallVector<QualType, 16> ArgTys;
  QualType argT = Context->getObjCIdType();
  assert(!argT.isNull() && "Can't find 'id' type");
  ArgTys.push_back(argT);
  argT = Context->getObjCSelType();
  assert(!argT.isNull() && "Can't find 'SEL' type");
  ArgTys.push_back(argT);
  QualType msgSendType = Context->getFunctionType(Context->getObjCIdType(),
                                                  &ArgTys[0], ArgTys.size(),
                                                  true /*isVariadic*/, 0);
  MsgSendStretFunctionDecl = FunctionDecl::Create(*Context, TUDecl,
                                         SourceLocation(), 
                                         msgSendIdent, msgSendType,
                                         FunctionDecl::Extern, false, 0);
}

// SynthMsgSendSuperStretFunctionDecl - 
// id objc_msgSendSuper_stret(struct objc_super *, SEL op, ...);
void RewriteObjC::SynthMsgSendSuperStretFunctionDecl() {
  IdentifierInfo *msgSendIdent = 
    &Context->Idents.get("objc_msgSendSuper_stret");
  llvm::SmallVector<QualType, 16> ArgTys;
  RecordDecl *RD = RecordDecl::Create(*Context, TagDecl::TK_struct, TUDecl,
                                      SourceLocation(),
                                      &Context->Idents.get("objc_super"));
  QualType argT = Context->getPointerType(Context->getTagDeclType(RD));
  assert(!argT.isNull() && "Can't build 'struct objc_super *' type");
  ArgTys.push_back(argT);
  argT = Context->getObjCSelType();
  assert(!argT.isNull() && "Can't find 'SEL' type");
  ArgTys.push_back(argT);
  QualType msgSendType = Context->getFunctionType(Context->getObjCIdType(),
                                                  &ArgTys[0], ArgTys.size(),
                                                  true /*isVariadic*/, 0);
  MsgSendSuperStretFunctionDecl = FunctionDecl::Create(*Context, TUDecl,
                                                       SourceLocation(), 
                                              msgSendIdent, msgSendType,
                                              FunctionDecl::Extern, false, 0);
}

// SynthMsgSendFpretFunctionDecl - double objc_msgSend_fpret(id self, SEL op, ...);
void RewriteObjC::SynthMsgSendFpretFunctionDecl() {
  IdentifierInfo *msgSendIdent = &Context->Idents.get("objc_msgSend_fpret");
  llvm::SmallVector<QualType, 16> ArgTys;
  QualType argT = Context->getObjCIdType();
  assert(!argT.isNull() && "Can't find 'id' type");
  ArgTys.push_back(argT);
  argT = Context->getObjCSelType();
  assert(!argT.isNull() && "Can't find 'SEL' type");
  ArgTys.push_back(argT);
  QualType msgSendType = Context->getFunctionType(Context->DoubleTy,
                                                  &ArgTys[0], ArgTys.size(),
                                                  true /*isVariadic*/, 0);
  MsgSendFpretFunctionDecl = FunctionDecl::Create(*Context, TUDecl,
                                              SourceLocation(), 
                                              msgSendIdent, msgSendType,
                                              FunctionDecl::Extern, false, 0);
}

// SynthGetClassFunctionDecl - id objc_getClass(const char *name);
void RewriteObjC::SynthGetClassFunctionDecl() {
  IdentifierInfo *getClassIdent = &Context->Idents.get("objc_getClass");
  llvm::SmallVector<QualType, 16> ArgTys;
  ArgTys.push_back(Context->getPointerType(
                     Context->CharTy.getQualifiedType(QualType::Const)));
  QualType getClassType = Context->getFunctionType(Context->getObjCIdType(),
                                                   &ArgTys[0], ArgTys.size(),
                                                   false /*isVariadic*/, 0);
  GetClassFunctionDecl = FunctionDecl::Create(*Context, TUDecl,
                                          SourceLocation(), 
                                          getClassIdent, getClassType,
                                          FunctionDecl::Extern, false, 0);
}

// SynthGetMetaClassFunctionDecl - id objc_getClass(const char *name);
void RewriteObjC::SynthGetMetaClassFunctionDecl() {
  IdentifierInfo *getClassIdent = &Context->Idents.get("objc_getMetaClass");
  llvm::SmallVector<QualType, 16> ArgTys;
  ArgTys.push_back(Context->getPointerType(
                     Context->CharTy.getQualifiedType(QualType::Const)));
  QualType getClassType = Context->getFunctionType(Context->getObjCIdType(),
                                                   &ArgTys[0], ArgTys.size(),
                                                   false /*isVariadic*/, 0);
  GetMetaClassFunctionDecl = FunctionDecl::Create(*Context, TUDecl,
                                              SourceLocation(), 
                                              getClassIdent, getClassType,
                                              FunctionDecl::Extern, false, 0);
}

Stmt *RewriteObjC::RewriteObjCStringLiteral(ObjCStringLiteral *Exp) {
  QualType strType = getConstantStringStructType();

  std::string S = "__NSConstantStringImpl_";

  std::string tmpName = InFileName;
  unsigned i;
  for (i=0; i < tmpName.length(); i++) {
    char c = tmpName.at(i);
    // replace any non alphanumeric characters with '_'.
    if (!isalpha(c) && (c < '0' || c > '9'))
      tmpName[i] = '_';
  }
  S += tmpName;
  S += "_";
  S += utostr(NumObjCStringLiterals++);

  Preamble += "static __NSConstantStringImpl " + S;
  Preamble += " __attribute__ ((section (\"__DATA, __cfstring\"))) = {__CFConstantStringClassReference,";
  Preamble += "0x000007c8,"; // utf8_str
  // The pretty printer for StringLiteral handles escape characters properly.
  std::string prettyBufS;
  llvm::raw_string_ostream prettyBuf(prettyBufS);
  Exp->getString()->printPretty(prettyBuf);
  Preamble += prettyBuf.str();
  Preamble += ",";
  // The minus 2 removes the begin/end double quotes.
  Preamble += utostr(prettyBuf.str().size()-2) + "};\n";
  
  VarDecl *NewVD = VarDecl::Create(*Context, TUDecl, SourceLocation(), 
                                    &Context->Idents.get(S.c_str()), strType, 
                                    VarDecl::Static, NULL);
  DeclRefExpr *DRE = new DeclRefExpr(NewVD, strType, SourceLocation());
  Expr *Unop = new UnaryOperator(DRE, UnaryOperator::AddrOf,
                                 Context->getPointerType(DRE->getType()), 
                                 SourceLocation());
  // cast to NSConstantString *
  CastExpr *cast = new CStyleCastExpr(Exp->getType(), Unop, 
                                         Exp->getType(), SourceLocation());
  ReplaceStmt(Exp, cast);
  delete Exp;
  return cast;
}

ObjCInterfaceDecl *RewriteObjC::isSuperReceiver(Expr *recExpr) {
  // check if we are sending a message to 'super'
  if (!CurMethodDef || !CurMethodDef->isInstance()) return 0;
  
  if (PredefinedExpr *PDE = dyn_cast<PredefinedExpr>(recExpr))
    if (PDE->getIdentType() == PredefinedExpr::ObjCSuper) {
      const PointerType *PT = PDE->getType()->getAsPointerType();
      assert(PT);
      ObjCInterfaceType *IT = cast<ObjCInterfaceType>(PT->getPointeeType());
      return IT->getDecl();
    }
  return 0;
}

// struct objc_super { struct objc_object *receiver; struct objc_class *super; };
QualType RewriteObjC::getSuperStructType() {
  if (!SuperStructDecl) {
    SuperStructDecl = RecordDecl::Create(*Context, TagDecl::TK_struct, TUDecl,
                                         SourceLocation(), 
                                         &Context->Idents.get("objc_super"));
    QualType FieldTypes[2];
  
    // struct objc_object *receiver;
    FieldTypes[0] = Context->getObjCIdType();  
    // struct objc_class *super;
    FieldTypes[1] = Context->getObjCClassType();  
    // Create fields
    FieldDecl *FieldDecls[2];
  
    for (unsigned i = 0; i < 2; ++i)
      FieldDecls[i] = FieldDecl::Create(*Context, SourceLocation(), 0, 
                                        FieldTypes[i]);
  
    SuperStructDecl->defineBody(*Context, FieldDecls, 4);
  }
  return Context->getTagDeclType(SuperStructDecl);
}

QualType RewriteObjC::getConstantStringStructType() {
  if (!ConstantStringDecl) {
    ConstantStringDecl = RecordDecl::Create(*Context, TagDecl::TK_struct, TUDecl,
                                            SourceLocation(), 
                         &Context->Idents.get("__NSConstantStringImpl"));
    QualType FieldTypes[4];
  
    // struct objc_object *receiver;
    FieldTypes[0] = Context->getObjCIdType();  
    // int flags;
    FieldTypes[1] = Context->IntTy;  
    // char *str;
    FieldTypes[2] = Context->getPointerType(Context->CharTy);  
    // long length;
    FieldTypes[3] = Context->LongTy;  
    // Create fields
    FieldDecl *FieldDecls[4];
  
    for (unsigned i = 0; i < 4; ++i)
      FieldDecls[i] = FieldDecl::Create(*Context, SourceLocation(), 0,
                                        FieldTypes[i]);
  
    ConstantStringDecl->defineBody(*Context, FieldDecls, 4);
  }
  return Context->getTagDeclType(ConstantStringDecl);
}

Stmt *RewriteObjC::SynthMessageExpr(ObjCMessageExpr *Exp) {
  if (!SelGetUidFunctionDecl)
    SynthSelGetUidFunctionDecl();
  if (!MsgSendFunctionDecl)
    SynthMsgSendFunctionDecl();
  if (!MsgSendSuperFunctionDecl)
    SynthMsgSendSuperFunctionDecl();
  if (!MsgSendStretFunctionDecl)
    SynthMsgSendStretFunctionDecl();
  if (!MsgSendSuperStretFunctionDecl)
    SynthMsgSendSuperStretFunctionDecl();
  if (!MsgSendFpretFunctionDecl)
    SynthMsgSendFpretFunctionDecl();
  if (!GetClassFunctionDecl)
    SynthGetClassFunctionDecl();
  if (!GetMetaClassFunctionDecl)
    SynthGetMetaClassFunctionDecl();
  
  // default to objc_msgSend().
  FunctionDecl *MsgSendFlavor = MsgSendFunctionDecl;
  // May need to use objc_msgSend_stret() as well.
  FunctionDecl *MsgSendStretFlavor = 0;
  if (ObjCMethodDecl *mDecl = Exp->getMethodDecl()) {
    QualType resultType = mDecl->getResultType();
    if (resultType->isStructureType() || resultType->isUnionType())
      MsgSendStretFlavor = MsgSendStretFunctionDecl;
    else if (resultType->isRealFloatingType())
      MsgSendFlavor = MsgSendFpretFunctionDecl;
  }
  
  // Synthesize a call to objc_msgSend().
  llvm::SmallVector<Expr*, 8> MsgExprs;
  IdentifierInfo *clsName = Exp->getClassName();
  
  // Derive/push the receiver/selector, 2 implicit arguments to objc_msgSend().
  if (clsName) { // class message.
    // FIXME: We need to fix Sema (and the AST for ObjCMessageExpr) to handle
    // the 'super' idiom within a class method.
    if (!strcmp(clsName->getName(), "super")) {
      MsgSendFlavor = MsgSendSuperFunctionDecl;
      if (MsgSendStretFlavor)
        MsgSendStretFlavor = MsgSendSuperStretFunctionDecl;
      assert(MsgSendFlavor && "MsgSendFlavor is NULL!");
      
      ObjCInterfaceDecl *SuperDecl = 
        CurMethodDef->getClassInterface()->getSuperClass();

      llvm::SmallVector<Expr*, 4> InitExprs;
      
      // set the receiver to self, the first argument to all methods.
      InitExprs.push_back(new DeclRefExpr(
            CurMethodDef->getSelfDecl(), 
            Context->getObjCIdType(),
            SourceLocation())); 
      llvm::SmallVector<Expr*, 8> ClsExprs;
      QualType argType = Context->getPointerType(Context->CharTy);
      ClsExprs.push_back(new StringLiteral(SuperDecl->getIdentifier()->getName(), 
                                           SuperDecl->getIdentifier()->getLength(),
                                           false, argType, SourceLocation(),
                                           SourceLocation()));
      CallExpr *Cls = SynthesizeCallToFunctionDecl(GetMetaClassFunctionDecl,
                                                   &ClsExprs[0], 
                                                   ClsExprs.size());
      // To turn off a warning, type-cast to 'id'
      InitExprs.push_back( // set 'super class', using objc_getClass().
        new CStyleCastExpr(Context->getObjCIdType(), 
                              Cls, Context->getObjCIdType(),
                              SourceLocation())); 
      // struct objc_super
      QualType superType = getSuperStructType();
      Expr *SuperRep;
      
      if (LangOpts.Microsoft) {
        SynthSuperContructorFunctionDecl();
        // Simulate a contructor call...
        DeclRefExpr *DRE = new DeclRefExpr(SuperContructorFunctionDecl, 
                                           superType, SourceLocation());
        SuperRep = new CallExpr(DRE, &InitExprs[0], InitExprs.size(), 
                                superType, SourceLocation());
      } else {      
        // (struct objc_super) { <exprs from above> }
        InitListExpr *ILE = new InitListExpr(SourceLocation(), 
                                             &InitExprs[0], InitExprs.size(), 
                                             SourceLocation(), false);
        SuperRep = new CompoundLiteralExpr(SourceLocation(), superType, ILE,
                                           false);
      }
      // struct objc_super *
      Expr *Unop = new UnaryOperator(SuperRep, UnaryOperator::AddrOf,
                               Context->getPointerType(SuperRep->getType()), 
                               SourceLocation());
      MsgExprs.push_back(Unop);
    } else {
      llvm::SmallVector<Expr*, 8> ClsExprs;
      QualType argType = Context->getPointerType(Context->CharTy);
      ClsExprs.push_back(new StringLiteral(clsName->getName(), 
                                           clsName->getLength(),
                                           false, argType, SourceLocation(),
                                           SourceLocation()));
      CallExpr *Cls = SynthesizeCallToFunctionDecl(GetClassFunctionDecl,
                                                   &ClsExprs[0], 
                                                   ClsExprs.size());
      MsgExprs.push_back(Cls);
    }
  } else { // instance message.
    Expr *recExpr = Exp->getReceiver();

    if (ObjCInterfaceDecl *SuperDecl = isSuperReceiver(recExpr)) {
      MsgSendFlavor = MsgSendSuperFunctionDecl;
      if (MsgSendStretFlavor)
        MsgSendStretFlavor = MsgSendSuperStretFunctionDecl;
      assert(MsgSendFlavor && "MsgSendFlavor is NULL!");
      
      llvm::SmallVector<Expr*, 4> InitExprs;
      
      InitExprs.push_back(
        new CStyleCastExpr(Context->getObjCIdType(), 
                     new DeclRefExpr(CurMethodDef->getSelfDecl(), 
                                     Context->getObjCIdType(),
                                     SourceLocation()),
                     Context->getObjCIdType(),
                     SourceLocation())); // set the 'receiver'.
      
      llvm::SmallVector<Expr*, 8> ClsExprs;
      QualType argType = Context->getPointerType(Context->CharTy);
      ClsExprs.push_back(new StringLiteral(SuperDecl->getIdentifier()->getName(), 
                                           SuperDecl->getIdentifier()->getLength(),
                                           false, argType, SourceLocation(),
                                           SourceLocation()));
      CallExpr *Cls = SynthesizeCallToFunctionDecl(GetClassFunctionDecl,
                                                   &ClsExprs[0], 
                                                   ClsExprs.size());
      // To turn off a warning, type-cast to 'id'
      InitExprs.push_back(
        // set 'super class', using objc_getClass().
        new CStyleCastExpr(Context->getObjCIdType(), 
        Cls, Context->getObjCIdType(), SourceLocation())); 
      // struct objc_super
      QualType superType = getSuperStructType();
      Expr *SuperRep;
      
      if (LangOpts.Microsoft) {
        SynthSuperContructorFunctionDecl();
        // Simulate a contructor call...
        DeclRefExpr *DRE = new DeclRefExpr(SuperContructorFunctionDecl, 
                                           superType, SourceLocation());
        SuperRep = new CallExpr(DRE, &InitExprs[0], InitExprs.size(), 
                                superType, SourceLocation());
      } else {
        // (struct objc_super) { <exprs from above> }
        InitListExpr *ILE = new InitListExpr(SourceLocation(), 
                                             &InitExprs[0], InitExprs.size(), 
                                             SourceLocation(), false);
        SuperRep = new CompoundLiteralExpr(SourceLocation(), superType, ILE, false);
      }
      // struct objc_super *
      Expr *Unop = new UnaryOperator(SuperRep, UnaryOperator::AddrOf,
                               Context->getPointerType(SuperRep->getType()), 
                               SourceLocation());
      MsgExprs.push_back(Unop);
    } else {
      // Remove all type-casts because it may contain objc-style types; e.g.
      // Foo<Proto> *.
      while (CStyleCastExpr *CE = dyn_cast<CStyleCastExpr>(recExpr))
        recExpr = CE->getSubExpr();
      recExpr = new CStyleCastExpr(Context->getObjCIdType(), recExpr,
                                      Context->getObjCIdType(), 
                                      SourceLocation());
      MsgExprs.push_back(recExpr);
    }
  }
  // Create a call to sel_registerName("selName"), it will be the 2nd argument.
  llvm::SmallVector<Expr*, 8> SelExprs;
  QualType argType = Context->getPointerType(Context->CharTy);
  SelExprs.push_back(new StringLiteral(Exp->getSelector().getName().c_str(),
                                       Exp->getSelector().getName().size(),
                                       false, argType, SourceLocation(),
                                       SourceLocation()));
  CallExpr *SelExp = SynthesizeCallToFunctionDecl(SelGetUidFunctionDecl,
                                                 &SelExprs[0], SelExprs.size());
  MsgExprs.push_back(SelExp);
  
  // Now push any user supplied arguments.
  for (unsigned i = 0; i < Exp->getNumArgs(); i++) {
    Expr *userExpr = Exp->getArg(i);
    // Make all implicit casts explicit...ICE comes in handy:-)
    if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(userExpr)) {
      // Reuse the ICE type, it is exactly what the doctor ordered.
      QualType type = ICE->getType()->isObjCQualifiedIdType()
                                ? Context->getObjCIdType()
                                : ICE->getType();
      userExpr = new CStyleCastExpr(type, userExpr, type, SourceLocation());
    }
    // Make id<P...> cast into an 'id' cast.
    else if (CStyleCastExpr *CE = dyn_cast<CStyleCastExpr>(userExpr)) {
      if (CE->getType()->isObjCQualifiedIdType()) {
        while ((CE = dyn_cast<CStyleCastExpr>(userExpr)))
          userExpr = CE->getSubExpr();
        userExpr = new CStyleCastExpr(Context->getObjCIdType(), 
                                userExpr, Context->getObjCIdType(), 
                                SourceLocation());
      }
    } 
    MsgExprs.push_back(userExpr);
    // We've transferred the ownership to MsgExprs. Null out the argument in
    // the original expression, since we will delete it below.
    Exp->setArg(i, 0);
  }
  // Generate the funky cast.
  CastExpr *cast;
  llvm::SmallVector<QualType, 8> ArgTypes;
  QualType returnType;
  
  // Push 'id' and 'SEL', the 2 implicit arguments.
  if (MsgSendFlavor == MsgSendSuperFunctionDecl)
    ArgTypes.push_back(Context->getPointerType(getSuperStructType()));
  else
    ArgTypes.push_back(Context->getObjCIdType());
  ArgTypes.push_back(Context->getObjCSelType());
  if (ObjCMethodDecl *mDecl = Exp->getMethodDecl()) {
    // Push any user argument types.
    for (unsigned i = 0; i < mDecl->getNumParams(); i++) {
      QualType t = mDecl->getParamDecl(i)->getType()->isObjCQualifiedIdType()
                     ? Context->getObjCIdType() 
                     : mDecl->getParamDecl(i)->getType();
      // Make sure we convert "t (^)(...)" to "t (*)(...)".
      if (isBlockPointerType(t)) {
        const BlockPointerType *BPT = t->getAsBlockPointerType();
        t = Context->getPointerType(BPT->getPointeeType());
      }
      ArgTypes.push_back(t);
    }
    returnType = mDecl->getResultType()->isObjCQualifiedIdType()
                   ? Context->getObjCIdType() : mDecl->getResultType();
  } else {
    returnType = Context->getObjCIdType();
  }
  // Get the type, we will need to reference it in a couple spots.
  QualType msgSendType = MsgSendFlavor->getType();
  
  // Create a reference to the objc_msgSend() declaration.
  DeclRefExpr *DRE = new DeclRefExpr(MsgSendFlavor, msgSendType, 
                                     SourceLocation());

  // Need to cast objc_msgSend to "void *" (to workaround a GCC bandaid). 
  // If we don't do this cast, we get the following bizarre warning/note:
  // xx.m:13: warning: function called through a non-compatible type
  // xx.m:13: note: if this code is reached, the program will abort
  cast = new CStyleCastExpr(Context->getPointerType(Context->VoidTy), DRE, 
                               Context->getPointerType(Context->VoidTy),
                               SourceLocation());
    
  // Now do the "normal" pointer to function cast.
  QualType castType = Context->getFunctionType(returnType, 
    &ArgTypes[0], ArgTypes.size(),
    // If we don't have a method decl, force a variadic cast.
    Exp->getMethodDecl() ? Exp->getMethodDecl()->isVariadic() : true, 0);
  castType = Context->getPointerType(castType);
  cast = new CStyleCastExpr(castType, cast, castType, SourceLocation());

  // Don't forget the parens to enforce the proper binding.
  ParenExpr *PE = new ParenExpr(SourceLocation(), SourceLocation(), cast);
  
  const FunctionType *FT = msgSendType->getAsFunctionType();
  CallExpr *CE = new CallExpr(PE, &MsgExprs[0], MsgExprs.size(), 
                              FT->getResultType(), SourceLocation());
  Stmt *ReplacingStmt = CE;
  if (MsgSendStretFlavor) {
    // We have the method which returns a struct/union. Must also generate
    // call to objc_msgSend_stret and hang both varieties on a conditional
    // expression which dictate which one to envoke depending on size of
    // method's return type.
    
    // Create a reference to the objc_msgSend_stret() declaration.
    DeclRefExpr *STDRE = new DeclRefExpr(MsgSendStretFlavor, msgSendType, 
                                         SourceLocation());
    // Need to cast objc_msgSend_stret to "void *" (see above comment).
    cast = new CStyleCastExpr(Context->getPointerType(Context->VoidTy), STDRE, 
                                 Context->getPointerType(Context->VoidTy),
                                 SourceLocation());
    // Now do the "normal" pointer to function cast.
    castType = Context->getFunctionType(returnType, 
      &ArgTypes[0], ArgTypes.size(),
      Exp->getMethodDecl() ? Exp->getMethodDecl()->isVariadic() : false, 0);
    castType = Context->getPointerType(castType);
    cast = new CStyleCastExpr(castType, cast, castType, SourceLocation());
    
    // Don't forget the parens to enforce the proper binding.
    PE = new ParenExpr(SourceLocation(), SourceLocation(), cast);
    
    FT = msgSendType->getAsFunctionType();
    CallExpr *STCE = new CallExpr(PE, &MsgExprs[0], MsgExprs.size(), 
                                  FT->getResultType(), SourceLocation());
    
    // Build sizeof(returnType)
    SizeOfAlignOfTypeExpr *sizeofExpr = new SizeOfAlignOfTypeExpr(true, 
                                          returnType, Context->getSizeType(), 
                                          SourceLocation(), SourceLocation());
    // (sizeof(returnType) <= 8 ? objc_msgSend(...) : objc_msgSend_stret(...))
    // FIXME: Value of 8 is base on ppc32/x86 ABI for the most common cases.
    // For X86 it is more complicated and some kind of target specific routine
    // is needed to decide what to do.
    unsigned IntSize = 
      static_cast<unsigned>(Context->getTypeSize(Context->IntTy));
    IntegerLiteral *limit = new IntegerLiteral(llvm::APInt(IntSize, 8), 
                                               Context->IntTy,
                                               SourceLocation());
    BinaryOperator *lessThanExpr = new BinaryOperator(sizeofExpr, limit, 
                                                      BinaryOperator::LE, 
                                                      Context->IntTy, 
                                                      SourceLocation());
    // (sizeof(returnType) <= 8 ? objc_msgSend(...) : objc_msgSend_stret(...))
    ConditionalOperator *CondExpr = 
      new ConditionalOperator(lessThanExpr, CE, STCE, returnType);
    ReplacingStmt = new ParenExpr(SourceLocation(), SourceLocation(), CondExpr);
  }
  return ReplacingStmt;
}

Stmt *RewriteObjC::RewriteMessageExpr(ObjCMessageExpr *Exp) {
  Stmt *ReplacingStmt = SynthMessageExpr(Exp);
  
  // Now do the actual rewrite.
  ReplaceStmt(Exp, ReplacingStmt);
  
  delete Exp;
  return ReplacingStmt;
}

/// RewriteObjCProtocolExpr - Rewrite a protocol expression into
/// call to objc_getProtocol("proto-name").
Stmt *RewriteObjC::RewriteObjCProtocolExpr(ObjCProtocolExpr *Exp) {
  if (!GetProtocolFunctionDecl)
    SynthGetProtocolFunctionDecl();
  // Create a call to objc_getProtocol("ProtocolName").
  llvm::SmallVector<Expr*, 8> ProtoExprs;
  QualType argType = Context->getPointerType(Context->CharTy);
  ProtoExprs.push_back(new StringLiteral(Exp->getProtocol()->getName(),
                                       strlen(Exp->getProtocol()->getName()),
                                       false, argType, SourceLocation(),
                                       SourceLocation()));
  CallExpr *ProtoExp = SynthesizeCallToFunctionDecl(GetProtocolFunctionDecl,
                                                    &ProtoExprs[0], 
                                                    ProtoExprs.size());
  ReplaceStmt(Exp, ProtoExp);
  delete Exp;
  return ProtoExp;
  
}

bool RewriteObjC::BufferContainsPPDirectives(const char *startBuf, 
                                             const char *endBuf) {
  while (startBuf < endBuf) {
    if (*startBuf == '#') {
      // Skip whitespace.
      for (++startBuf; startBuf[0] == ' ' || startBuf[0] == '\t'; ++startBuf)
        ;
      if (!strncmp(startBuf, "if", strlen("if")) ||
          !strncmp(startBuf, "ifdef", strlen("ifdef")) ||
          !strncmp(startBuf, "ifndef", strlen("ifndef")) ||
          !strncmp(startBuf, "define", strlen("define")) ||
          !strncmp(startBuf, "undef", strlen("undef")) ||
          !strncmp(startBuf, "else", strlen("else")) ||
          !strncmp(startBuf, "elif", strlen("elif")) ||
          !strncmp(startBuf, "endif", strlen("endif")) ||
          !strncmp(startBuf, "pragma", strlen("pragma")) ||
          !strncmp(startBuf, "include", strlen("include")) ||
          !strncmp(startBuf, "import", strlen("import")) ||
          !strncmp(startBuf, "include_next", strlen("include_next")))
        return true;
    }
    startBuf++;
  }
  return false;
}

/// SynthesizeObjCInternalStruct - Rewrite one internal struct corresponding to
/// an objective-c class with ivars.
void RewriteObjC::SynthesizeObjCInternalStruct(ObjCInterfaceDecl *CDecl,
                                               std::string &Result) {
  assert(CDecl && "Class missing in SynthesizeObjCInternalStruct");
  assert(CDecl->getName() && "Name missing in SynthesizeObjCInternalStruct");
  // Do not synthesize more than once.
  if (ObjCSynthesizedStructs.count(CDecl))
    return;
  ObjCInterfaceDecl *RCDecl = CDecl->getSuperClass();
  int NumIvars = CDecl->ivar_size();
  SourceLocation LocStart = CDecl->getLocStart();
  SourceLocation LocEnd = CDecl->getLocEnd();
  
  const char *startBuf = SM->getCharacterData(LocStart);
  const char *endBuf = SM->getCharacterData(LocEnd);
  
  // If no ivars and no root or if its root, directly or indirectly,
  // have no ivars (thus not synthesized) then no need to synthesize this class.
  if ((CDecl->isForwardDecl() || NumIvars == 0) &&
      (!RCDecl || !ObjCSynthesizedStructs.count(RCDecl))) {
    endBuf += Lexer::MeasureTokenLength(LocEnd, *SM);
    ReplaceText(LocStart, endBuf-startBuf, Result.c_str(), Result.size());
    return;
  }
  
  // FIXME: This has potential of causing problem. If 
  // SynthesizeObjCInternalStruct is ever called recursively.
  Result += "\nstruct ";
  Result += CDecl->getName();
  if (LangOpts.Microsoft)
    Result += "_IMPL";

  if (NumIvars > 0) {
    const char *cursor = strchr(startBuf, '{');
    assert((cursor && endBuf) 
           && "SynthesizeObjCInternalStruct - malformed @interface");
    // If the buffer contains preprocessor directives, we do more fine-grained
    // rewrites. This is intended to fix code that looks like (which occurs in
    // NSURL.h, for example):
    //
    // #ifdef XYZ
    // @interface Foo : NSObject
    // #else
    // @interface FooBar : NSObject
    // #endif
    // {
    //    int i;
    // }
    // @end
    //
    // This clause is segregated to avoid breaking the common case.
    if (BufferContainsPPDirectives(startBuf, cursor)) {
      SourceLocation L = RCDecl ? CDecl->getSuperClassLoc() : 
                                  CDecl->getClassLoc();
      const char *endHeader = SM->getCharacterData(L);
      endHeader += Lexer::MeasureTokenLength(L, *SM);

      if (!CDecl->getReferencedProtocols().empty()) {
        // advance to the end of the referenced protocols.
        while (endHeader < cursor && *endHeader != '>') endHeader++;
        endHeader++;
      }
      // rewrite the original header
      ReplaceText(LocStart, endHeader-startBuf, Result.c_str(), Result.size());
    } else {
      // rewrite the original header *without* disturbing the '{'
      ReplaceText(LocStart, cursor-startBuf-1, Result.c_str(), Result.size());
    }
    if (RCDecl && ObjCSynthesizedStructs.count(RCDecl)) {
      Result = "\n    struct ";
      Result += RCDecl->getName();
      Result += "_IMPL ";
      Result += RCDecl->getName();
      Result += "_IVARS;\n";
      
      // insert the super class structure definition.
      SourceLocation OnePastCurly =
        LocStart.getFileLocWithOffset(cursor-startBuf+1);
      InsertText(OnePastCurly, Result.c_str(), Result.size());
    }
    cursor++; // past '{'
    
    // Now comment out any visibility specifiers.
    while (cursor < endBuf) {
      if (*cursor == '@') {
        SourceLocation atLoc = LocStart.getFileLocWithOffset(cursor-startBuf);
        // Skip whitespace.
        for (++cursor; cursor[0] == ' ' || cursor[0] == '\t'; ++cursor)
          /*scan*/;

        // FIXME: presence of @public, etc. inside comment results in
        // this transformation as well, which is still correct c-code.
        if (!strncmp(cursor, "public", strlen("public")) ||
            !strncmp(cursor, "private", strlen("private")) ||
            !strncmp(cursor, "package", strlen("package")) ||
            !strncmp(cursor, "protected", strlen("protected")))
          InsertText(atLoc, "// ", 3);
      }
      // FIXME: If there are cases where '<' is used in ivar declaration part
      // of user code, then scan the ivar list and use needToScanForQualifiers
      // for type checking.
      else if (*cursor == '<') {
        SourceLocation atLoc = LocStart.getFileLocWithOffset(cursor-startBuf);
        InsertText(atLoc, "/* ", 3);
        cursor = strchr(cursor, '>');
        cursor++;
        atLoc = LocStart.getFileLocWithOffset(cursor-startBuf);
        InsertText(atLoc, " */", 3);
      }
      cursor++;
    }
    // Don't forget to add a ';'!!
    InsertText(LocEnd.getFileLocWithOffset(1), ";", 1);
  } else { // we don't have any instance variables - insert super struct.
    endBuf += Lexer::MeasureTokenLength(LocEnd, *SM);
    Result += " {\n    struct ";
    Result += RCDecl->getName();
    Result += "_IMPL ";
    Result += RCDecl->getName();
    Result += "_IVARS;\n};\n";
    ReplaceText(LocStart, endBuf-startBuf, Result.c_str(), Result.size());
  }
  // Mark this struct as having been generated.
  if (!ObjCSynthesizedStructs.insert(CDecl))
    assert(false && "struct already synthesize- SynthesizeObjCInternalStruct");
}

// RewriteObjCMethodsMetaData - Rewrite methods metadata for instance or
/// class methods.
void RewriteObjC::RewriteObjCMethodsMetaData(instmeth_iterator MethodBegin,
                                             instmeth_iterator MethodEnd,
                                             bool IsInstanceMethod,
                                             const char *prefix,
                                             const char *ClassName,
                                             std::string &Result) {
  if (MethodBegin == MethodEnd) return;
  
  static bool objc_impl_method = false;
  if (!objc_impl_method) {
    /* struct _objc_method {
       SEL _cmd;
       char *method_types;
       void *_imp;
       }
     */
    Result += "\nstruct _objc_method {\n";
    Result += "\tSEL _cmd;\n";
    Result += "\tchar *method_types;\n";
    Result += "\tvoid *_imp;\n";
    Result += "};\n";
    
    objc_impl_method = true;
  }
  
  // Build _objc_method_list for class's methods if needed
  
  /* struct  {
   struct _objc_method_list *next_method;
   int method_count;
   struct _objc_method method_list[];
   }
   */
  Result += "\nstatic struct {\n";
  Result += "\tstruct _objc_method_list *next_method;\n";
  Result += "\tint method_count;\n";
  Result += "\tstruct _objc_method method_list[";
  Result += utostr(MethodEnd-MethodBegin);
  Result += "];\n} _OBJC_";
  Result += prefix;
  Result += IsInstanceMethod ? "INSTANCE" : "CLASS";
  Result += "_METHODS_";
  Result += ClassName;
  Result += " __attribute__ ((used, section (\"__OBJC, __";
  Result += IsInstanceMethod ? "inst" : "cls";
  Result += "_meth\")))= ";
  Result += "{\n\t0, " + utostr(MethodEnd-MethodBegin) + "\n";

  Result += "\t,{{(SEL)\"";
  Result += (*MethodBegin)->getSelector().getName().c_str();
  std::string MethodTypeString;
  Context->getObjCEncodingForMethodDecl(*MethodBegin, MethodTypeString);
  Result += "\", \"";
  Result += MethodTypeString;
  Result += "\", (void *)";
  Result += MethodInternalNames[*MethodBegin];
  Result += "}\n";
  for (++MethodBegin; MethodBegin != MethodEnd; ++MethodBegin) {
    Result += "\t  ,{(SEL)\"";
    Result += (*MethodBegin)->getSelector().getName().c_str();
    std::string MethodTypeString;
    Context->getObjCEncodingForMethodDecl(*MethodBegin, MethodTypeString);
    Result += "\", \"";
    Result += MethodTypeString;
    Result += "\", (void *)";
    Result += MethodInternalNames[*MethodBegin];
    Result += "}\n";
  }
  Result += "\t }\n};\n";
}

/// RewriteObjCProtocolsMetaData - Rewrite protocols meta-data.
void RewriteObjC::
RewriteObjCProtocolsMetaData(const ObjCList<ObjCProtocolDecl> &Protocols,
                             const char *prefix,
                             const char *ClassName,
                             std::string &Result) {
  static bool objc_protocol_methods = false;
  if (Protocols.empty()) return;
  
  for (unsigned i = 0; i != Protocols.size(); i++) {
    ObjCProtocolDecl *PDecl = Protocols[i];
    // Output struct protocol_methods holder of method selector and type.
    if (!objc_protocol_methods && !PDecl->isForwardDecl()) {
      /* struct protocol_methods {
       SEL _cmd;
       char *method_types;
       }
       */
      Result += "\nstruct protocol_methods {\n";
      Result += "\tSEL _cmd;\n";
      Result += "\tchar *method_types;\n";
      Result += "};\n";
      
      objc_protocol_methods = true;
    }
    // Do not synthesize the protocol more than once.
    if (ObjCSynthesizedProtocols.count(PDecl))
      continue;
           
    if (PDecl->instmeth_begin() != PDecl->instmeth_end()) {
      unsigned NumMethods = PDecl->getNumInstanceMethods();
      /* struct _objc_protocol_method_list {
       int protocol_method_count;
       struct protocol_methods protocols[];
       }
       */
      Result += "\nstatic struct {\n";
      Result += "\tint protocol_method_count;\n";
      Result += "\tstruct protocol_methods protocols[";
      Result += utostr(NumMethods);
      Result += "];\n} _OBJC_PROTOCOL_INSTANCE_METHODS_";
      Result += PDecl->getName();
      Result += " __attribute__ ((used, section (\"__OBJC, __cat_inst_meth\")))= "
        "{\n\t" + utostr(NumMethods) + "\n";
      
      // Output instance methods declared in this protocol.
      for (ObjCProtocolDecl::instmeth_iterator I = PDecl->instmeth_begin(), 
           E = PDecl->instmeth_end(); I != E; ++I) {
        if (I == PDecl->instmeth_begin())
          Result += "\t  ,{{(SEL)\"";
        else
          Result += "\t  ,{(SEL)\"";
        Result += (*I)->getSelector().getName().c_str();
        std::string MethodTypeString;
        Context->getObjCEncodingForMethodDecl((*I), MethodTypeString);
        Result += "\", \"";
        Result += MethodTypeString;
        Result += "\"}\n";
      }
      Result += "\t }\n};\n";
    }
    
    // Output class methods declared in this protocol.
    int NumMethods = PDecl->getNumClassMethods();
    if (NumMethods > 0) {
      /* struct _objc_protocol_method_list {
       int protocol_method_count;
       struct protocol_methods protocols[];
       }
       */
      Result += "\nstatic struct {\n";
      Result += "\tint protocol_method_count;\n";
      Result += "\tstruct protocol_methods protocols[";
      Result += utostr(NumMethods);
      Result += "];\n} _OBJC_PROTOCOL_CLASS_METHODS_";
      Result += PDecl->getName();
      Result += " __attribute__ ((used, section (\"__OBJC, __cat_cls_meth\")))= "
             "{\n\t";
      Result += utostr(NumMethods);
      Result += "\n";
      
      // Output instance methods declared in this protocol.
      for (ObjCProtocolDecl::classmeth_iterator I = PDecl->classmeth_begin(), 
           E = PDecl->classmeth_end(); I != E; ++I) {
        if (I == PDecl->classmeth_begin())
          Result += "\t  ,{{(SEL)\"";
        else
          Result += "\t  ,{(SEL)\"";
        Result += (*I)->getSelector().getName().c_str();
        std::string MethodTypeString;
        Context->getObjCEncodingForMethodDecl((*I), MethodTypeString);
        Result += "\", \"";
        Result += MethodTypeString;
        Result += "\"}\n";
      }
      Result += "\t }\n};\n";
    }

    // Output:
    /* struct _objc_protocol {
     // Objective-C 1.0 extensions
     struct _objc_protocol_extension *isa;
     char *protocol_name;
     struct _objc_protocol **protocol_list;
     struct _objc_protocol_method_list *instance_methods;
     struct _objc_protocol_method_list *class_methods;
     };  
     */
    static bool objc_protocol = false;
    if (!objc_protocol) {
      Result += "\nstruct _objc_protocol {\n";
      Result += "\tstruct _objc_protocol_extension *isa;\n";
      Result += "\tchar *protocol_name;\n";
      Result += "\tstruct _objc_protocol **protocol_list;\n";
      Result += "\tstruct _objc_protocol_method_list *instance_methods;\n";
      Result += "\tstruct _objc_protocol_method_list *class_methods;\n";
      Result += "};\n";
      
      objc_protocol = true;
    }
    
    Result += "\nstatic struct _objc_protocol _OBJC_PROTOCOL_";
    Result += PDecl->getName();
    Result += " __attribute__ ((used, section (\"__OBJC, __protocol\")))= "
      "{\n\t0, \"";
    Result += PDecl->getName();
    Result += "\", 0, ";
    if (PDecl->instmeth_begin() != PDecl->instmeth_end()) {
      Result += "(struct _objc_protocol_method_list *)&_OBJC_PROTOCOL_INSTANCE_METHODS_";
      Result += PDecl->getName();
      Result += ", ";
    }
    else
      Result += "0, ";
    if (PDecl->getNumClassMethods() > 0) {
      Result += "(struct _objc_protocol_method_list *)&_OBJC_PROTOCOL_CLASS_METHODS_";
      Result += PDecl->getName();
      Result += "\n";
    }
    else
      Result += "0\n";
    Result += "};\n";
    
    // Mark this protocol as having been generated.
    if (!ObjCSynthesizedProtocols.insert(PDecl))
      assert(false && "protocol already synthesized");
  }
  // Output the top lovel protocol meta-data for the class.
  /* struct _objc_protocol_list {
   struct _objc_protocol_list *next;
   int    protocol_count;
   struct _objc_protocol *class_protocols[];
   }
   */
  Result += "\nstatic struct {\n";
  Result += "\tstruct _objc_protocol_list *next;\n";
  Result += "\tint    protocol_count;\n";
  Result += "\tstruct _objc_protocol *class_protocols[";
  Result += utostr(Protocols.size());
  Result += "];\n} _OBJC_";
  Result += prefix;
  Result += "_PROTOCOLS_";
  Result += ClassName;
  Result += " __attribute__ ((used, section (\"__OBJC, __cat_cls_meth\")))= "
    "{\n\t0, ";
  Result += utostr(Protocols.size());
  Result += "\n";
  
  Result += "\t,{&_OBJC_PROTOCOL_";
  Result += Protocols[0]->getName();
  Result += " \n";
  
  for (unsigned i = 1; i != Protocols.size(); i++) {
    Result += "\t ,&_OBJC_PROTOCOL_";
    Result += Protocols[i]->getName();
    Result += "\n";
  }
  Result += "\t }\n};\n";
}

/// RewriteObjCCategoryImplDecl - Rewrite metadata for each category 
/// implementation.
void RewriteObjC::RewriteObjCCategoryImplDecl(ObjCCategoryImplDecl *IDecl,
                                              std::string &Result) {
  ObjCInterfaceDecl *ClassDecl = IDecl->getClassInterface();
  // Find category declaration for this implementation.
  ObjCCategoryDecl *CDecl;
  for (CDecl = ClassDecl->getCategoryList(); CDecl; 
       CDecl = CDecl->getNextClassCategory())
    if (CDecl->getIdentifier() == IDecl->getIdentifier())
      break;
  
  std::string FullCategoryName = ClassDecl->getName();
  FullCategoryName += '_';
  FullCategoryName += IDecl->getName();
    
  // Build _objc_method_list for class's instance methods if needed
  RewriteObjCMethodsMetaData(IDecl->instmeth_begin(), IDecl->instmeth_end(),
                             true, "CATEGORY_", FullCategoryName.c_str(),
                             Result);
  
  // Build _objc_method_list for class's class methods if needed
  RewriteObjCMethodsMetaData(IDecl->classmeth_begin(), IDecl->classmeth_end(),
                             false, "CATEGORY_", FullCategoryName.c_str(),
                             Result);
  
  // Protocols referenced in class declaration?
  // Null CDecl is case of a category implementation with no category interface
  if (CDecl)
    RewriteObjCProtocolsMetaData(CDecl->getReferencedProtocols(), "CATEGORY",
                                 FullCategoryName.c_str(), Result);
  
  /* struct _objc_category {
   char *category_name;
   char *class_name;
   struct _objc_method_list *instance_methods;
   struct _objc_method_list *class_methods;
   struct _objc_protocol_list *protocols;
   // Objective-C 1.0 extensions
   uint32_t size;     // sizeof (struct _objc_category)
   struct _objc_property_list *instance_properties;  // category's own 
                                                     // @property decl.
   };   
   */
  
  static bool objc_category = false;
  if (!objc_category) {
    Result += "\nstruct _objc_category {\n";
    Result += "\tchar *category_name;\n";
    Result += "\tchar *class_name;\n";
    Result += "\tstruct _objc_method_list *instance_methods;\n";
    Result += "\tstruct _objc_method_list *class_methods;\n";
    Result += "\tstruct _objc_protocol_list *protocols;\n";
    Result += "\tunsigned int size;\n";   
    Result += "\tstruct _objc_property_list *instance_properties;\n";
    Result += "};\n";
    objc_category = true;
  }
  Result += "\nstatic struct _objc_category _OBJC_CATEGORY_";
  Result += FullCategoryName;
  Result += " __attribute__ ((used, section (\"__OBJC, __category\")))= {\n\t\"";
  Result += IDecl->getName();
  Result += "\"\n\t, \"";
  Result += ClassDecl->getName();
  Result += "\"\n";
  
  if (IDecl->getNumInstanceMethods() > 0) {
    Result += "\t, (struct _objc_method_list *)"
           "&_OBJC_CATEGORY_INSTANCE_METHODS_";
    Result += FullCategoryName;
    Result += "\n";
  }
  else
    Result += "\t, 0\n";
  if (IDecl->getNumClassMethods() > 0) {
    Result += "\t, (struct _objc_method_list *)"
           "&_OBJC_CATEGORY_CLASS_METHODS_";
    Result += FullCategoryName;
    Result += "\n";
  }
  else
    Result += "\t, 0\n";
  
  if (CDecl && !CDecl->getReferencedProtocols().empty()) {
    Result += "\t, (struct _objc_protocol_list *)&_OBJC_CATEGORY_PROTOCOLS_"; 
    Result += FullCategoryName;
    Result += "\n";
  }
  else
    Result += "\t, 0\n";
  Result += "\t, sizeof(struct _objc_category), 0\n};\n";
}

/// SynthesizeIvarOffsetComputation - This rutine synthesizes computation of
/// ivar offset.
void RewriteObjC::SynthesizeIvarOffsetComputation(ObjCImplementationDecl *IDecl, 
                                                  ObjCIvarDecl *ivar, 
                                                  std::string &Result) {
  if (ivar->isBitField()) {
    // FIXME: The hack below doesn't work for bitfields. For now, we simply
    // place all bitfields at offset 0.
    Result += "0";
  } else {
    Result += "__OFFSETOFIVAR__(struct ";
    Result += IDecl->getName();
    if (LangOpts.Microsoft)
      Result += "_IMPL";
    Result += ", ";
    Result += ivar->getName();
    Result += ")";
  }
}

//===----------------------------------------------------------------------===//
// Meta Data Emission
//===----------------------------------------------------------------------===//

void RewriteObjC::RewriteObjCClassMetaData(ObjCImplementationDecl *IDecl,
                                           std::string &Result) {
  ObjCInterfaceDecl *CDecl = IDecl->getClassInterface();
  
  // Explictly declared @interface's are already synthesized.
  if (CDecl->ImplicitInterfaceDecl()) {
    // FIXME: Implementation of a class with no @interface (legacy) doese not 
    // produce correct synthesis as yet.
    SynthesizeObjCInternalStruct(CDecl, Result);
  }
  
  // Build _objc_ivar_list metadata for classes ivars if needed
  unsigned NumIvars = !IDecl->ivar_empty()
                      ? IDecl->ivar_size() 
                      : (CDecl ? CDecl->ivar_size() : 0);
  if (NumIvars > 0) {
    static bool objc_ivar = false;
    if (!objc_ivar) {
      /* struct _objc_ivar {
          char *ivar_name;
          char *ivar_type;
          int ivar_offset;
        };  
       */
      Result += "\nstruct _objc_ivar {\n";
      Result += "\tchar *ivar_name;\n";
      Result += "\tchar *ivar_type;\n";
      Result += "\tint ivar_offset;\n";
      Result += "};\n";
      
      objc_ivar = true;
    }

    /* struct {
       int ivar_count;
       struct _objc_ivar ivar_list[nIvars];
       };  
     */
    Result += "\nstatic struct {\n"; 
    Result += "\tint ivar_count;\n";
    Result += "\tstruct _objc_ivar ivar_list[";
    Result += utostr(NumIvars);
    Result += "];\n} _OBJC_INSTANCE_VARIABLES_";
    Result += IDecl->getName();
    Result += " __attribute__ ((used, section (\"__OBJC, __instance_vars\")))= "
      "{\n\t";
    Result += utostr(NumIvars);
    Result += "\n";
    
    ObjCInterfaceDecl::ivar_iterator IVI, IVE;
    if (!IDecl->ivar_empty()) {
      IVI = IDecl->ivar_begin();
      IVE = IDecl->ivar_end();
    } else {
      IVI = CDecl->ivar_begin();
      IVE = CDecl->ivar_end();
    }
    Result += "\t,{{\"";
    Result += (*IVI)->getName();
    Result += "\", \"";
    std::string StrEncoding;
    Context->getObjCEncodingForType((*IVI)->getType(), StrEncoding);
    Result += StrEncoding;
    Result += "\", ";
    SynthesizeIvarOffsetComputation(IDecl, *IVI, Result);
    Result += "}\n";
    for (++IVI; IVI != IVE; ++IVI) {
      Result += "\t  ,{\"";
      Result += (*IVI)->getName();
      Result += "\", \"";
      std::string StrEncoding;
      Context->getObjCEncodingForType((*IVI)->getType(), StrEncoding);
      Result += StrEncoding;
      Result += "\", ";
      SynthesizeIvarOffsetComputation(IDecl, (*IVI), Result);
      Result += "}\n";
    }
    
    Result += "\t }\n};\n";
  }
  
  // Build _objc_method_list for class's instance methods if needed
  RewriteObjCMethodsMetaData(IDecl->instmeth_begin(), IDecl->instmeth_end(), 
                             true, "", IDecl->getName(), Result);
  
  // Build _objc_method_list for class's class methods if needed
  RewriteObjCMethodsMetaData(IDecl->classmeth_begin(), IDecl->classmeth_end(),
                             false, "", IDecl->getName(), Result);
    
  // Protocols referenced in class declaration?
  RewriteObjCProtocolsMetaData(CDecl->getReferencedProtocols(),
                               "CLASS", CDecl->getName(), Result);
    
  
  // Declaration of class/meta-class metadata
  /* struct _objc_class {
   struct _objc_class *isa; // or const char *root_class_name when metadata
   const char *super_class_name;
   char *name;
   long version;
   long info;
   long instance_size;
   struct _objc_ivar_list *ivars;
   struct _objc_method_list *methods;
   struct objc_cache *cache;
   struct objc_protocol_list *protocols;
   const char *ivar_layout;
   struct _objc_class_ext  *ext;
   };  
  */
  static bool objc_class = false;
  if (!objc_class) {
    Result += "\nstruct _objc_class {\n";
    Result += "\tstruct _objc_class *isa;\n";
    Result += "\tconst char *super_class_name;\n";
    Result += "\tchar *name;\n";
    Result += "\tlong version;\n";
    Result += "\tlong info;\n";
    Result += "\tlong instance_size;\n";
    Result += "\tstruct _objc_ivar_list *ivars;\n";
    Result += "\tstruct _objc_method_list *methods;\n";
    Result += "\tstruct objc_cache *cache;\n";
    Result += "\tstruct _objc_protocol_list *protocols;\n";
    Result += "\tconst char *ivar_layout;\n";
    Result += "\tstruct _objc_class_ext  *ext;\n";
    Result += "};\n";
    objc_class = true;
  }
  
  // Meta-class metadata generation.
  ObjCInterfaceDecl *RootClass = 0;
  ObjCInterfaceDecl *SuperClass = CDecl->getSuperClass();
  while (SuperClass) {
    RootClass = SuperClass;
    SuperClass = SuperClass->getSuperClass();
  }
  SuperClass = CDecl->getSuperClass();
  
  Result += "\nstatic struct _objc_class _OBJC_METACLASS_";
  Result += CDecl->getName();
  Result += " __attribute__ ((used, section (\"__OBJC, __meta_class\")))= "
  "{\n\t(struct _objc_class *)\"";
  Result += (RootClass ? RootClass->getName() : CDecl->getName());
  Result += "\"";

  if (SuperClass) {
    Result += ", \"";
    Result += SuperClass->getName();
    Result += "\", \"";
    Result += CDecl->getName();
    Result += "\"";
  }
  else {
    Result += ", 0, \"";
    Result += CDecl->getName();
    Result += "\"";
  }
  // Set 'ivars' field for root class to 0. ObjC1 runtime does not use it.
  // 'info' field is initialized to CLS_META(2) for metaclass
  Result += ", 0,2, sizeof(struct _objc_class), 0";
  if (IDecl->getNumClassMethods() > 0) {
    Result += "\n\t, (struct _objc_method_list *)&_OBJC_CLASS_METHODS_";
    Result += IDecl->getName();
    Result += "\n"; 
  }
  else
    Result += ", 0\n";
  if (!CDecl->getReferencedProtocols().empty()) {
    Result += "\t,0, (struct _objc_protocol_list *)&_OBJC_CLASS_PROTOCOLS_";
    Result += CDecl->getName();
    Result += ",0,0\n";
  }
  else
    Result += "\t,0,0,0,0\n";
  Result += "};\n";
  
  // class metadata generation.
  Result += "\nstatic struct _objc_class _OBJC_CLASS_";
  Result += CDecl->getName();
  Result += " __attribute__ ((used, section (\"__OBJC, __class\")))= "
            "{\n\t&_OBJC_METACLASS_";
  Result += CDecl->getName();
  if (SuperClass) {
    Result += ", \"";
    Result += SuperClass->getName();
    Result += "\", \"";
    Result += CDecl->getName();
    Result += "\"";
  }
  else {
    Result += ", 0, \"";
    Result += CDecl->getName();
    Result += "\"";
  }
  // 'info' field is initialized to CLS_CLASS(1) for class
  Result += ", 0,1";
  if (!ObjCSynthesizedStructs.count(CDecl))
    Result += ",0";
  else {
    // class has size. Must synthesize its size.
    Result += ",sizeof(struct ";
    Result += CDecl->getName();
    if (LangOpts.Microsoft)
      Result += "_IMPL";
    Result += ")";
  }
  if (NumIvars > 0) {
    Result += ", (struct _objc_ivar_list *)&_OBJC_INSTANCE_VARIABLES_";
    Result += CDecl->getName();
    Result += "\n\t";
  }
  else
    Result += ",0";
  if (IDecl->getNumInstanceMethods() > 0) {
    Result += ", (struct _objc_method_list *)&_OBJC_INSTANCE_METHODS_";
    Result += CDecl->getName();
    Result += ", 0\n\t"; 
  }
  else
    Result += ",0,0";
  if (!CDecl->getReferencedProtocols().empty()) {
    Result += ", (struct _objc_protocol_list*)&_OBJC_CLASS_PROTOCOLS_";
    Result += CDecl->getName();
    Result += ", 0,0\n";
  }
  else
    Result += ",0,0,0\n";
  Result += "};\n";
}

/// RewriteImplementations - This routine rewrites all method implementations
/// and emits meta-data.

void RewriteObjC::RewriteImplementations(std::string &Result) {
  int ClsDefCount = ClassImplementation.size();
  int CatDefCount = CategoryImplementation.size();
  
  if (ClsDefCount == 0 && CatDefCount == 0)
    return;
  // Rewrite implemented methods
  for (int i = 0; i < ClsDefCount; i++)
    RewriteImplementationDecl(ClassImplementation[i]);
  
  for (int i = 0; i < CatDefCount; i++)
    RewriteImplementationDecl(CategoryImplementation[i]);
  
  // This is needed for determining instance variable offsets.
  Result += "\n#define __OFFSETOFIVAR__(TYPE, MEMBER) ((int) &((TYPE *)0)->MEMBER)\n";   
  // For each implemented class, write out all its meta data.
  for (int i = 0; i < ClsDefCount; i++)
    RewriteObjCClassMetaData(ClassImplementation[i], Result);
  
  // For each implemented category, write out all its meta data.
  for (int i = 0; i < CatDefCount; i++)
    RewriteObjCCategoryImplDecl(CategoryImplementation[i], Result);
  
  // Write objc_symtab metadata
  /*
   struct _objc_symtab
   {
   long sel_ref_cnt;
   SEL *refs;
   short cls_def_cnt;
   short cat_def_cnt;
   void *defs[cls_def_cnt + cat_def_cnt];
   }; 
   */
  
  Result += "\nstruct _objc_symtab {\n";
  Result += "\tlong sel_ref_cnt;\n";
  Result += "\tSEL *refs;\n";
  Result += "\tshort cls_def_cnt;\n";
  Result += "\tshort cat_def_cnt;\n";
  Result += "\tvoid *defs[" + utostr(ClsDefCount + CatDefCount)+ "];\n";
  Result += "};\n\n";
  
  Result += "static struct _objc_symtab "
         "_OBJC_SYMBOLS __attribute__((used, section (\"__OBJC, __symbols\")))= {\n";
  Result += "\t0, 0, " + utostr(ClsDefCount) 
            + ", " + utostr(CatDefCount) + "\n";
  for (int i = 0; i < ClsDefCount; i++) {
    Result += "\t,&_OBJC_CLASS_";
    Result += ClassImplementation[i]->getName();
    Result += "\n";
  }
  
  for (int i = 0; i < CatDefCount; i++) {
    Result += "\t,&_OBJC_CATEGORY_";
    Result += CategoryImplementation[i]->getClassInterface()->getName();
    Result += "_";
    Result += CategoryImplementation[i]->getName();
    Result += "\n";
  }
  
  Result += "};\n\n";
  
  // Write objc_module metadata
  
  /*
   struct _objc_module {
    long version;
    long size;
    const char *name;
    struct _objc_symtab *symtab;
   }
  */
  
  Result += "\nstruct _objc_module {\n";
  Result += "\tlong version;\n";
  Result += "\tlong size;\n";
  Result += "\tconst char *name;\n";
  Result += "\tstruct _objc_symtab *symtab;\n";
  Result += "};\n\n";
  Result += "static struct _objc_module "
    "_OBJC_MODULES __attribute__ ((used, section (\"__OBJC, __module_info\")))= {\n";
  Result += "\t" + utostr(OBJC_ABI_VERSION) + 
  ", sizeof(struct _objc_module), \"\", &_OBJC_SYMBOLS\n";
  Result += "};\n\n";

  if (LangOpts.Microsoft) {
    Result += "#pragma section(\".objc_module_info$B\",long,read,write)\n";
    Result += "#pragma data_seg(push, \".objc_module_info$B\")\n";
    Result += "static struct _objc_module *_POINTER_OBJC_MODULES = ";
    Result += "&_OBJC_MODULES;\n";
    Result += "#pragma data_seg(pop)\n\n";
  }
}

std::string RewriteObjC::SynthesizeBlockFunc(BlockExpr *CE, int i,
                                                   const char *funcName,
                                                   std::string Tag) {
  const FunctionType *AFT = CE->getFunctionType();
  QualType RT = AFT->getResultType();
  std::string StructRef = "struct " + Tag;
  std::string S = "static " + RT.getAsString() + " __" +
                  funcName + "_" + "block_func_" + utostr(i);

  BlockDecl *BD = CE->getBlockDecl();
  
  if (isa<FunctionTypeNoProto>(AFT)) {
    S += "()";
  } else if (BD->param_empty()) {
    S += "(" + StructRef + " *__cself)";
  } else {
    const FunctionTypeProto *FT = cast<FunctionTypeProto>(AFT);
    assert(FT && "SynthesizeBlockFunc: No function proto");
    S += '(';
    // first add the implicit argument.
    S += StructRef + " *__cself, ";
    std::string ParamStr;
    for (BlockDecl::param_iterator AI = BD->param_begin(),
         E = BD->param_end(); AI != E; ++AI) {
      if (AI != BD->param_begin()) S += ", ";
      ParamStr = (*AI)->getName();
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
  std::string RewrittenStr = RewrittenBlockExprs[CE];
  const char *cstr = RewrittenStr.c_str();
  while (*cstr++ != '{') ;
  S += cstr;
  S += "\n";
  return S;
}

std::string RewriteObjC::SynthesizeBlockHelperFuncs(BlockExpr *CE, int i,
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

std::string RewriteObjC::SynthesizeBlockImpl(BlockExpr *CE, std::string Tag,
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
    
    if (hasCopyDisposeHelpers)
      Constructor += "    copy = copyHelp;\n    dispose = disposeHelp;\n";
      
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
    if (hasCopyDisposeHelpers)
      Constructor += "    copy = copyHelp;\n    dispose = disposeHelp;\n";
  }
  Constructor += "  ";
  Constructor += "}\n";
  S += Constructor;
  S += "};\n";
  return S;
}

void RewriteObjC::SynthesizeBlockLiterals(SourceLocation FunLocStart,
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

void RewriteObjC::InsertBlockLiteralsWithinFunction(FunctionDecl *FD) {
  SourceLocation FunLocStart = FD->getTypeSpecStartLoc();
  const char *FuncName = FD->getName();
  
  SynthesizeBlockLiterals(FunLocStart, FuncName);
}

void RewriteObjC::InsertBlockLiteralsWithinMethod(ObjCMethodDecl *MD) {
  SourceLocation FunLocStart = MD->getLocStart();
  std::string FuncName = std::string(MD->getSelector().getName());
  // Convert colons to underscores.
  std::string::size_type loc = 0;
  while ((loc = FuncName.find(":", loc)) != std::string::npos)
    FuncName.replace(loc, 1, "_");
  
  SynthesizeBlockLiterals(FunLocStart, FuncName.c_str());
}

void RewriteObjC::GetBlockDeclRefExprs(Stmt *S) {
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

void RewriteObjC::GetBlockCallExprs(Stmt *S) {
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

Stmt *RewriteObjC::SynthesizeBlockCall(CallExpr *Exp) {
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
  
  RecordDecl *RD = RecordDecl::Create(*Context, TagDecl::TK_struct, TUDecl,
                                      SourceLocation(),
                                      &Context->Idents.get("__block_impl"));
  QualType PtrBlock = Context->getPointerType(Context->getTagDeclType(RD));

  // Generate a funky cast.
  llvm::SmallVector<QualType, 8> ArgTypes;
    
  // Push the block argument type.
  ArgTypes.push_back(PtrBlock);
  if (FTP) {
    for (FunctionTypeProto::arg_type_iterator I = FTP->arg_type_begin(), 
         E = FTP->arg_type_end(); I && (I != E); ++I) {
      QualType t = *I;
      // Make sure we convert "t (^)(...)" to "t (*)(...)".
      if (isBlockPointerType(t)) {
        const BlockPointerType *BPT = t->getAsBlockPointerType();
        t = Context->getPointerType(BPT->getPointeeType());
      }
      ArgTypes.push_back(t);
    }
  }
  // Now do the pointer to function cast.
  QualType PtrToFuncCastType = Context->getFunctionType(Exp->getType(), 
    &ArgTypes[0], ArgTypes.size(), false/*no variadic*/, 0);
    
  PtrToFuncCastType = Context->getPointerType(PtrToFuncCastType);
  
  CastExpr *BlkCast = new CStyleCastExpr(PtrBlock, Exp->getCallee(), PtrBlock, SourceLocation());
  // Don't forget the parens to enforce the proper binding.
  ParenExpr *PE = new ParenExpr(SourceLocation(), SourceLocation(), BlkCast);
  //PE->dump();
  
  FieldDecl *FD = FieldDecl::Create(*Context, SourceLocation(),
                     &Context->Idents.get("FuncPtr"), Context->VoidPtrTy);
  MemberExpr *ME = new MemberExpr(PE, true, FD, SourceLocation(), FD->getType());
  
  CastExpr *FunkCast = new CStyleCastExpr(PtrToFuncCastType, ME, PtrToFuncCastType, SourceLocation());
  PE = new ParenExpr(SourceLocation(), SourceLocation(), FunkCast);
  
  llvm::SmallVector<Expr*, 8> BlkExprs;
  // Add the implicit argument.
  BlkExprs.push_back(BlkCast);
  // Add the user arguments.
  for (CallExpr::arg_iterator I = Exp->arg_begin(), 
       E = Exp->arg_end(); I != E; ++I) {
    BlkExprs.push_back(*I);
  }
  CallExpr *CE = new CallExpr(PE, &BlkExprs[0], BlkExprs.size(), Exp->getType(), SourceLocation());
  return CE;
}

void RewriteObjC::RewriteBlockCall(CallExpr *Exp) {
  Stmt *BlockCall = SynthesizeBlockCall(Exp);
  ReplaceStmt(Exp, BlockCall);
}

void RewriteObjC::RewriteBlockDeclRefExpr(BlockDeclRefExpr *BDRE) {
  // FIXME: Add more elaborate code generation required by the ABI.
  InsertText(BDRE->getLocStart(), "*", 1);
}

void RewriteObjC::RewriteCastExpr(CastExpr *CE) {
  SourceLocation LocStart = CE->getLocStart();
  SourceLocation LocEnd = CE->getLocEnd();

  // Need to avoid trying to rewrite synthesized casts.
  if (LocStart.isInvalid())
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

void RewriteObjC::RewriteBlockPointerFunctionArgs(FunctionDecl *FD) {
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

bool RewriteObjC::PointerTypeTakesAnyBlockArguments(QualType QT) {
  const FunctionTypeProto *FTP;
  const PointerType *PT = QT->getAsPointerType();
  if (PT) {
    FTP = PT->getPointeeType()->getAsFunctionTypeProto();
  } else {
    const BlockPointerType *BPT = QT->getAsBlockPointerType();
    assert(BPT && "BlockPointerTypeTakeAnyBlockArguments(): not a block pointer type");
    FTP = BPT->getPointeeType()->getAsFunctionTypeProto();
  }
  if (FTP) {
    for (FunctionTypeProto::arg_type_iterator I = FTP->arg_type_begin(), 
         E = FTP->arg_type_end(); I != E; ++I)
      if (isBlockPointerType(*I))
        return true;
  }
  return false;
}

void RewriteObjC::GetExtentOfArgList(const char *Name, 
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

void RewriteObjC::RewriteBlockPointerDecl(NamedDecl *ND) {
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

void RewriteObjC::CollectBlockDeclRefInfo(BlockExpr *Exp) {  
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

FunctionDecl *RewriteObjC::SynthBlockInitFunctionDecl(const char *name) {
  IdentifierInfo *ID = &Context->Idents.get(name);
  QualType FType = Context->getFunctionTypeNoProto(Context->VoidPtrTy);
  return FunctionDecl::Create(*Context, TUDecl,SourceLocation(), 
                              ID, FType, FunctionDecl::Extern, false, 0);
}

Stmt *RewriteObjC::SynthBlockInitExpr(BlockExpr *Exp) {
  Blocks.push_back(Exp);

  CollectBlockDeclRefInfo(Exp);
  std::string FuncName;
  
  if (CurFunctionDef)
    FuncName = std::string(CurFunctionDef->getName());
  else if (CurMethodDef) {
    FuncName = std::string(CurMethodDef->getSelector().getName());
    // Convert colons to underscores.
    std::string::size_type loc = 0;
    while ((loc = FuncName.find(":", loc)) != std::string::npos)
      FuncName.replace(loc, 1, "_");
  } else if (GlobalVarDecl)
    FuncName = std::string(GlobalVarDecl->getName());
    
  std::string BlockNumber = utostr(Blocks.size()-1);
  
  std::string Tag = "__" + FuncName + "_block_impl_" + BlockNumber;
  std::string Func = "__" + FuncName + "_block_func_" + BlockNumber;
  
  // Get a pointer to the function type so we can cast appropriately.
  QualType FType = Context->getPointerType(QualType(Exp->getFunctionType(),0));

  FunctionDecl *FD;
  Expr *NewRep;
  
  // Simulate a contructor call...
  FD = SynthBlockInitFunctionDecl(Tag.c_str());
  DeclRefExpr *DRE = new DeclRefExpr(FD, FType, SourceLocation());
  
  llvm::SmallVector<Expr*, 4> InitExprs;
  
  // Initialize the block function.
  FD = SynthBlockInitFunctionDecl(Func.c_str());
  DeclRefExpr *Arg = new DeclRefExpr(FD, FD->getType(), SourceLocation());
  CastExpr *castExpr = new CStyleCastExpr(Context->VoidPtrTy, Arg, 
                                          Context->VoidPtrTy, SourceLocation());
  InitExprs.push_back(castExpr); 
  
  if (ImportedBlockDecls.size()) {
    std::string Buf = "__" + FuncName + "_block_copy_" + BlockNumber;
    FD = SynthBlockInitFunctionDecl(Buf.c_str());
    Arg = new DeclRefExpr(FD, FD->getType(), SourceLocation());
    castExpr = new CStyleCastExpr(Context->VoidPtrTy, Arg, 
                                  Context->VoidPtrTy, SourceLocation());
    InitExprs.push_back(castExpr); 
    
    Buf = "__" + FuncName + "_block_dispose_" + BlockNumber;
    FD = SynthBlockInitFunctionDecl(Buf.c_str());
    Arg = new DeclRefExpr(FD, FD->getType(), SourceLocation());
    castExpr = new CStyleCastExpr(Context->VoidPtrTy, Arg, 
                                  Context->VoidPtrTy, SourceLocation());
    InitExprs.push_back(castExpr); 
  }
  // Add initializers for any closure decl refs.
  if (BlockDeclRefs.size()) {
    Expr *Exp;
    // Output all "by copy" declarations.
    for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByCopyDecls.begin(), 
         E = BlockByCopyDecls.end(); I != E; ++I) {
      if (isObjCType((*I)->getType())) {
        // FIXME: Conform to ABI ([[obj retain] autorelease]).
        FD = SynthBlockInitFunctionDecl((*I)->getName());
        Exp = new DeclRefExpr(FD, FD->getType(), SourceLocation());
      } else if (isBlockPointerType((*I)->getType())) {
        FD = SynthBlockInitFunctionDecl((*I)->getName());
        Arg = new DeclRefExpr(FD, FD->getType(), SourceLocation());
        Exp = new CStyleCastExpr(Context->VoidPtrTy, Arg, 
                                 Context->VoidPtrTy, SourceLocation());
      } else {
        FD = SynthBlockInitFunctionDecl((*I)->getName());
        Exp = new DeclRefExpr(FD, FD->getType(), SourceLocation());
      }
      InitExprs.push_back(Exp); 
    }
    // Output all "by ref" declarations.
    for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByRefDecls.begin(), 
         E = BlockByRefDecls.end(); I != E; ++I) {
      FD = SynthBlockInitFunctionDecl((*I)->getName());
      Exp = new DeclRefExpr(FD, FD->getType(), SourceLocation());
      Exp = new UnaryOperator(Exp, UnaryOperator::AddrOf,
                              Context->getPointerType(Exp->getType()), 
                              SourceLocation());
    }
  }
  NewRep = new CallExpr(DRE, &InitExprs[0], InitExprs.size(),
                        FType, SourceLocation());
  NewRep = new UnaryOperator(NewRep, UnaryOperator::AddrOf,
                             Context->getPointerType(NewRep->getType()), 
                             SourceLocation());
  NewRep = new CStyleCastExpr(FType, NewRep, FType, SourceLocation());
  BlockDeclRefs.clear();
  BlockByRefDecls.clear();
  BlockByCopyDecls.clear();
  ImportedBlockDecls.clear();
  return NewRep;
}

//===----------------------------------------------------------------------===//
// Function Body / Expression rewriting
//===----------------------------------------------------------------------===//

Stmt *RewriteObjC::RewriteFunctionBodyOrGlobalInitializer(Stmt *S) {
  if (isa<SwitchStmt>(S) || isa<WhileStmt>(S) || 
      isa<DoStmt>(S) || isa<ForStmt>(S))
    Stmts.push_back(S);
  else if (isa<ObjCForCollectionStmt>(S)) {
    Stmts.push_back(S);
    ObjCBcLabelNo.push_back(++BcLabelCount);
  }
  
  SourceRange OrigStmtRange = S->getSourceRange();
  
  // Perform a bottom up rewrite of all children.
  for (Stmt::child_iterator CI = S->child_begin(), E = S->child_end();
       CI != E; ++CI)
    if (*CI) {
      Stmt *newStmt = RewriteFunctionBodyOrGlobalInitializer(*CI);
      if (newStmt) 
        *CI = newStmt;
    }
  
  if (BlockExpr *BE = dyn_cast<BlockExpr>(S)) {
    // Rewrite the block body in place.
    RewriteFunctionBodyOrGlobalInitializer(BE->getBody());
      
    // Now we snarf the rewritten text and stash it away for later use.
    std::string Str = Rewrite.getRewritenText(BE->getSourceRange());
    RewrittenBlockExprs[BE] = Str;
    
    Stmt *blockTranscribed = SynthBlockInitExpr(BE);
    //blockTranscribed->dump();
    ReplaceStmt(S, blockTranscribed);
    return blockTranscribed;
  }
  // Handle specific things.
  if (ObjCEncodeExpr *AtEncode = dyn_cast<ObjCEncodeExpr>(S))
    return RewriteAtEncode(AtEncode);
  
  if (ObjCIvarRefExpr *IvarRefExpr = dyn_cast<ObjCIvarRefExpr>(S))
    return RewriteObjCIvarRefExpr(IvarRefExpr, OrigStmtRange.getBegin());

  if (ObjCSelectorExpr *AtSelector = dyn_cast<ObjCSelectorExpr>(S))
    return RewriteAtSelector(AtSelector);
    
  if (ObjCStringLiteral *AtString = dyn_cast<ObjCStringLiteral>(S))
    return RewriteObjCStringLiteral(AtString);
    
  if (ObjCMessageExpr *MessExpr = dyn_cast<ObjCMessageExpr>(S)) {
    // Before we rewrite it, put the original message expression in a comment.
    SourceLocation startLoc = MessExpr->getLocStart();
    SourceLocation endLoc = MessExpr->getLocEnd();
    
    const char *startBuf = SM->getCharacterData(startLoc);
    const char *endBuf = SM->getCharacterData(endLoc);
    
    std::string messString;
    messString += "// ";
    messString.append(startBuf, endBuf-startBuf+1);
    messString += "\n";
        
    // FIXME: Missing definition of 
    // InsertText(clang::SourceLocation, char const*, unsigned int).
    // InsertText(startLoc, messString.c_str(), messString.size());
    // Tried this, but it didn't work either...
    // ReplaceText(startLoc, 0, messString.c_str(), messString.size());
    return RewriteMessageExpr(MessExpr);
  }
  
  if (ObjCAtTryStmt *StmtTry = dyn_cast<ObjCAtTryStmt>(S))
    return RewriteObjCTryStmt(StmtTry);

  if (ObjCAtSynchronizedStmt *StmtTry = dyn_cast<ObjCAtSynchronizedStmt>(S))
    return RewriteObjCSynchronizedStmt(StmtTry);

  if (ObjCAtThrowStmt *StmtThrow = dyn_cast<ObjCAtThrowStmt>(S))
    return RewriteObjCThrowStmt(StmtThrow);
  
  if (ObjCProtocolExpr *ProtocolExp = dyn_cast<ObjCProtocolExpr>(S))
    return RewriteObjCProtocolExpr(ProtocolExp);
  
  if (ObjCForCollectionStmt *StmtForCollection = 
        dyn_cast<ObjCForCollectionStmt>(S))
    return RewriteObjCForCollectionStmt(StmtForCollection, 
                                        OrigStmtRange.getEnd());
  if (BreakStmt *StmtBreakStmt =
      dyn_cast<BreakStmt>(S))
    return RewriteBreakStmt(StmtBreakStmt);
  if (ContinueStmt *StmtContinueStmt =
      dyn_cast<ContinueStmt>(S))
    return RewriteContinueStmt(StmtContinueStmt);
	
  // Need to check for protocol refs (id <P>, Foo <P> *) in variable decls 
  // and cast exprs.
  if (DeclStmt *DS = dyn_cast<DeclStmt>(S)) {
    // FIXME: What we're doing here is modifying the type-specifier that
    // precedes the first Decl.  In the future the DeclGroup should have
    // a separate type-specifier that we can rewrite.    
    RewriteObjCQualifiedInterfaceTypes(*DS->decl_begin());
    
    // Blocks rewrite rules.
    for (DeclStmt::decl_iterator DI = DS->decl_begin(), DE = DS->decl_end();
         DI != DE; ++DI) {
      ScopedDecl *SD = *DI;
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
  
  if (CStyleCastExpr *CE = dyn_cast<CStyleCastExpr>(S))
    RewriteObjCQualifiedInterfaceTypes(CE);
  
  if (isa<SwitchStmt>(S) || isa<WhileStmt>(S) || 
      isa<DoStmt>(S) || isa<ForStmt>(S)) {
    assert(!Stmts.empty() && "Statement stack is empty");
    assert ((isa<SwitchStmt>(Stmts.back()) || isa<WhileStmt>(Stmts.back()) || 
             isa<DoStmt>(Stmts.back()) || isa<ForStmt>(Stmts.back())) 
            && "Statement stack mismatch");
    Stmts.pop_back();
  }
  // Handle blocks rewriting.
  if (BlockDeclRefExpr *BDRE = dyn_cast<BlockDeclRefExpr>(S)) {
    if (BDRE->isByRef())
      RewriteBlockDeclRefExpr(BDRE);
  }
  if (CallExpr *CE = dyn_cast<CallExpr>(S)) {
    if (CE->getCallee()->getType()->isBlockPointerType()) {
      Stmt *BlockCall = SynthesizeBlockCall(CE);
      ReplaceStmt(S, BlockCall);
      return BlockCall;
    }
  }
  if (CastExpr *CE = dyn_cast<CastExpr>(S)) {
    RewriteCastExpr(CE);
  }
#if 0
  if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(S)) {
    CastExpr *Replacement = new CastExpr(ICE->getType(), ICE->getSubExpr(), SourceLocation());
    // Get the new text.
    std::string SStr;
    llvm::raw_string_ostream Buf(SStr);
    Replacement->printPretty(Buf);
    const std::string &Str = Buf.str();

    printf("CAST = %s\n", &Str[0]);
    InsertText(ICE->getSubExpr()->getLocStart(), &Str[0], Str.size());
    delete S;
    return Replacement;
  }
#endif
  // Return this stmt unmodified.
  return S;
}

/// HandleDeclInMainFile - This is called for each top-level decl defined in the
/// main file of the input.
void RewriteObjC::HandleDeclInMainFile(Decl *D) {
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // Since function prototypes don't have ParmDecl's, we check the function
    // prototype. This enables us to rewrite function declarations and
    // definitions using the same code.
    RewriteBlocksInFunctionTypeProto(FD->getType(), FD);

    if (Stmt *Body = FD->getBody()) {
      CurFunctionDef = FD;
      FD->setBody(RewriteFunctionBodyOrGlobalInitializer(Body));
      // This synthesizes and inserts the block "impl" struct, invoke function,
      // and any copy/dispose helper functions.
      InsertBlockLiteralsWithinFunction(FD);
      CurFunctionDef = 0;
    } 
    return;
  }
  if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
    if (Stmt *Body = MD->getBody()) {
      //Body->dump();
      CurMethodDef = MD;
      MD->setBody(RewriteFunctionBodyOrGlobalInitializer(Body));
      InsertBlockLiteralsWithinMethod(MD);
      CurMethodDef = 0;
    }
  }
  if (ObjCImplementationDecl *CI = dyn_cast<ObjCImplementationDecl>(D))
    ClassImplementation.push_back(CI);
  else if (ObjCCategoryImplDecl *CI = dyn_cast<ObjCCategoryImplDecl>(D))
    CategoryImplementation.push_back(CI);
  else if (ObjCClassDecl *CD = dyn_cast<ObjCClassDecl>(D))
    RewriteForwardClassDecl(CD);
  else if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
    RewriteObjCQualifiedInterfaceTypes(VD);
    if (isBlockPointerType(VD->getType()))
      RewriteBlockPointerDecl(VD);
    else if (VD->getType()->isFunctionPointerType()) {
      CheckFunctionPointerDecl(VD->getType(), VD);
      if (VD->getInit()) {
        if (CastExpr *CE = dyn_cast<CastExpr>(VD->getInit())) {
          RewriteCastExpr(CE);
        }
      }
    }
    if (VD->getInit()) {
      GlobalVarDecl = VD;
      RewriteFunctionBodyOrGlobalInitializer(VD->getInit());
      SynthesizeBlockLiterals(VD->getTypeSpecStartLoc(), VD->getName());
      GlobalVarDecl = 0;

      // This is needed for blocks.
      if (CastExpr *CE = dyn_cast<CastExpr>(VD->getInit())) {
        RewriteCastExpr(CE);
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
      for (RecordDecl::field_const_iterator i = RD->field_begin(), 
             e = RD->field_end(); i != e; ++i) {
        FieldDecl *FD = *i;
        if (isBlockPointerType(FD->getType()))
          RewriteBlockPointerDecl(FD);
      }
    }
    return;
  }
  // Nothing yet.
}

void RewriteObjC::HandleTranslationUnit(TranslationUnit& TU) {
  // Get the top-level buffer that this corresponds to.
  
  // Rewrite tabs if we care.
  //RewriteTabs();
  
  if (Diags.hasErrorOccurred())
    return;

  // Create the output file.
  
  llvm::OwningPtr<llvm::raw_ostream> OwnedStream;
  llvm::raw_ostream *OutFile;
  if (OutFileName == "-") {
    OutFile = &llvm::outs();
  } else if (!OutFileName.empty()) {
    std::string Err;
    OutFile = new llvm::raw_fd_ostream(OutFileName.c_str(), Err);
    OwnedStream.reset(OutFile);
  } else if (InFileName == "-") {
    OutFile = &llvm::outs();
  } else {
    llvm::sys::Path Path(InFileName);
    Path.eraseSuffix();
    Path.appendSuffix("cpp");
    std::string Err;
    OutFile = new llvm::raw_fd_ostream(Path.toString().c_str(), Err);
    OwnedStream.reset(OutFile);
  }
  
  RewriteInclude();
  
  InsertText(SourceLocation::getFileLoc(MainFileID, 0), 
             Preamble.c_str(), Preamble.size(), false);
  
  // Rewrite Objective-c meta data*
  std::string ResultStr;
  RewriteImplementations(ResultStr);
  
  // Get the buffer corresponding to MainFileID.  If we haven't changed it, then
  // we are done.
  if (const RewriteBuffer *RewriteBuf = 
      Rewrite.getRewriteBufferFor(MainFileID)) {
    //printf("Changed:\n");
    *OutFile << std::string(RewriteBuf->begin(), RewriteBuf->end());
  } else {
    fprintf(stderr, "No changes\n");
  }
  // Emit metadata.
  *OutFile << ResultStr;
  OutFile->flush();
}

