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

#include "clang/Frontend/ASTConsumers.h"
#include "clang/Rewrite/Rewriter.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ParentMap.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/DenseSet.h"
using namespace clang;
using llvm::utostr;

namespace {
  class RewriteObjC : public ASTConsumer {
    enum {
      BLOCK_FIELD_IS_OBJECT   =  3,  /* id, NSObject, __attribute__((NSObject)), 
                                        block, ... */
      BLOCK_FIELD_IS_BLOCK    =  7,  /* a block variable */
      BLOCK_FIELD_IS_BYREF    =  8,  /* the on stack structure holding the 
                                        __block variable */
      BLOCK_FIELD_IS_WEAK     = 16,  /* declared __weak, only used in byref copy 
                                        helpers */
      BLOCK_BYREF_CALLER      = 128, /* called from __block (byref) copy/dispose 
                                        support routines */
      BLOCK_BYREF_CURRENT_MAX = 256
    };
    
    enum {
      BLOCK_NEEDS_FREE =        (1 << 24),
      BLOCK_HAS_COPY_DISPOSE =  (1 << 25),
      BLOCK_HAS_CXX_OBJ =       (1 << 26),
      BLOCK_IS_GC =             (1 << 27),
      BLOCK_IS_GLOBAL =         (1 << 28),
      BLOCK_HAS_DESCRIPTOR =    (1 << 29)
    };
    
    Rewriter Rewrite;
    Diagnostic &Diags;
    const LangOptions &LangOpts;
    unsigned RewriteFailedDiag;
    unsigned TryFinallyContainsReturnDiag;

    ASTContext *Context;
    SourceManager *SM;
    TranslationUnitDecl *TUDecl;
    FileID MainFileID;
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
    // Remember all the @protocol(<expr>) expressions.
    llvm::SmallPtrSet<ObjCProtocolDecl *, 32> ProtocolExprDecls;
    
    llvm::DenseSet<uint64_t> CopyDestroyCache;
    
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

    TypeDecl *ProtocolTypeDecl;
    QualType getProtocolType();

    // Needed for header files being rewritten
    bool IsHeader;

    std::string InFileName;
    llvm::raw_ostream* OutFile;

    bool SilenceRewriteMacroWarning;
    bool objc_impl_method;

    std::string Preamble;

    // Block expressions.
    llvm::SmallVector<BlockExpr *, 32> Blocks;
    llvm::SmallVector<BlockDeclRefExpr *, 32> BlockDeclRefs;
    llvm::DenseMap<BlockDeclRefExpr *, CallExpr *> BlockCallExprs;

    // Block related declarations.
    llvm::SmallPtrSet<ValueDecl *, 8> BlockByCopyDecls;
    llvm::SmallPtrSet<ValueDecl *, 8> BlockByRefDecls;
    llvm::DenseMap<ValueDecl *, unsigned> BlockByRefDeclNo;
    llvm::SmallPtrSet<ValueDecl *, 8> ImportedBlockDecls;

    llvm::DenseMap<BlockExpr *, std::string> RewrittenBlockExprs;

    // This maps a property to it's assignment statement.
    llvm::DenseMap<ObjCPropertyRefExpr *, BinaryOperator *> PropSetters;
    // This maps a property to it's synthesied message expression.
    // This allows us to rewrite chained getters (e.g. o.a.b.c).
    llvm::DenseMap<ObjCPropertyRefExpr *, Stmt *> PropGetters;

    // This maps an original source AST to it's rewritten form. This allows
    // us to avoid rewriting the same node twice (which is very uncommon).
    // This is needed to support some of the exotic property rewriting.
    llvm::DenseMap<Stmt *, Stmt *> ReplacedNodes;

    FunctionDecl *CurFunctionDef;
    FunctionDecl *CurFunctionDeclToDeclareForBlock;
    VarDecl *GlobalVarDecl;

    bool DisableReplaceStmt;

    static const int OBJC_ABI_VERSION =7 ;
  public:
    virtual void Initialize(ASTContext &context);

    // Top Level Driver code.
    virtual void HandleTopLevelDecl(DeclGroupRef D) {
      for (DeclGroupRef::iterator I = D.begin(), E = D.end(); I != E; ++I)
        HandleTopLevelSingleDecl(*I);
    }
    void HandleTopLevelSingleDecl(Decl *D);
    void HandleDeclInMainFile(Decl *D);
    RewriteObjC(std::string inFile, llvm::raw_ostream *OS,
                Diagnostic &D, const LangOptions &LOpts,
                bool silenceMacroWarn);

    ~RewriteObjC() {}

    virtual void HandleTranslationUnit(ASTContext &C);

    void ReplaceStmt(Stmt *Old, Stmt *New) {
      Stmt *ReplacingStmt = ReplacedNodes[Old];

      if (ReplacingStmt)
        return; // We can't rewrite the same node twice.

      if (DisableReplaceStmt)
        return; // Used when rewriting the assignment of a property setter.

      // If replacement succeeded or warning disabled return with no warning.
      if (!Rewrite.ReplaceStmt(Old, New)) {
        ReplacedNodes[Old] = New;
        return;
      }
      if (SilenceRewriteMacroWarning)
        return;
      Diags.Report(Context->getFullLoc(Old->getLocStart()), RewriteFailedDiag)
                   << Old->getSourceRange();
    }

    void ReplaceStmtWithRange(Stmt *Old, Stmt *New, SourceRange SrcRange) {
      // Measaure the old text.
      int Size = Rewrite.getRangeSize(SrcRange);
      if (Size == -1) {
        Diags.Report(Context->getFullLoc(Old->getLocStart()), RewriteFailedDiag)
                     << Old->getSourceRange();
        return;
      }
      // Get the new text.
      std::string SStr;
      llvm::raw_string_ostream S(SStr);
      New->printPretty(S, *Context, 0, PrintingPolicy(LangOpts));
      const std::string &Str = S.str();

      // If replacement succeeded or warning disabled return with no warning.
      if (!Rewrite.ReplaceText(SrcRange.getBegin(), Size, Str)) {
        ReplacedNodes[Old] = New;
        return;
      }
      if (SilenceRewriteMacroWarning)
        return;
      Diags.Report(Context->getFullLoc(Old->getLocStart()), RewriteFailedDiag)
                   << Old->getSourceRange();
    }

    void InsertText(SourceLocation Loc, const char *StrData, unsigned StrLen,
                    bool InsertAfter = true) {
      // If insertion succeeded or warning disabled return with no warning.
      if (!Rewrite.InsertText(Loc, llvm::StringRef(StrData, StrLen),
                              InsertAfter) ||
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
      if (!Rewrite.ReplaceText(Start, OrigLength,
                               llvm::StringRef(NewStr, NewLength)) ||
          SilenceRewriteMacroWarning)
        return;

      Diags.Report(Context->getFullLoc(Start), RewriteFailedDiag);
    }

    // Syntactic Rewriting.
    void RewritePrologue(SourceLocation Loc);
    void RewriteInclude();
    void RewriteTabs();
    void RewriteForwardClassDecl(ObjCClassDecl *Dcl);
    void RewritePropertyImplDecl(ObjCPropertyImplDecl *PID,
                                 ObjCImplementationDecl *IMD,
                                 ObjCCategoryImplDecl *CID);
    void RewriteInterfaceDecl(ObjCInterfaceDecl *Dcl);
    void RewriteImplementationDecl(Decl *Dcl);
    void RewriteObjCMethodDecl(ObjCMethodDecl *MDecl, std::string &ResultStr);
    void RewriteByRefString(std::string &ResultStr, const std::string &Name,
                            ValueDecl *VD);
    void RewriteCategoryDecl(ObjCCategoryDecl *Dcl);
    void RewriteProtocolDecl(ObjCProtocolDecl *Dcl);
    void RewriteForwardProtocolDecl(ObjCForwardProtocolDecl *Dcl);
    void RewriteMethodDeclaration(ObjCMethodDecl *Method);
    void RewriteProperty(ObjCPropertyDecl *prop);
    void RewriteFunctionDecl(FunctionDecl *FD);
    void RewriteBlockLiteralFunctionDecl(FunctionDecl *FD);
    void RewriteObjCQualifiedInterfaceTypes(Decl *Dcl);
    void RewriteObjCQualifiedInterfaceTypes(Expr *E);
    bool needToScanForQualifiers(QualType T);
    ObjCInterfaceDecl *isSuperReceiver(Expr *recExpr);
    QualType getSuperStructType();
    QualType getConstantStringStructType();
    bool BufferContainsPPDirectives(const char *startBuf, const char *endBuf);

    // Expression Rewriting.
    Stmt *RewriteFunctionBodyOrGlobalInitializer(Stmt *S);
    void CollectPropertySetters(Stmt *S);

    Stmt *CurrentBody;
    ParentMap *PropParentMap; // created lazily.

    Stmt *RewriteAtEncode(ObjCEncodeExpr *Exp);
    Stmt *RewriteObjCIvarRefExpr(ObjCIvarRefExpr *IV, SourceLocation OrigStart);
    Stmt *RewritePropertyGetter(ObjCPropertyRefExpr *PropRefExpr);
    Stmt *RewritePropertySetter(BinaryOperator *BinOp, Expr *newStmt,
                                SourceRange SrcRange);
    Stmt *RewriteAtSelector(ObjCSelectorExpr *Exp);
    Stmt *RewriteMessageExpr(ObjCMessageExpr *Exp);
    Stmt *RewriteObjCStringLiteral(ObjCStringLiteral *Exp);
    Stmt *RewriteObjCProtocolExpr(ObjCProtocolExpr *Exp);
    void WarnAboutReturnGotoStmts(Stmt *S);
    void HasReturnStmts(Stmt *S, bool &hasReturns);
    void RewriteTryReturnStmts(Stmt *S);
    void RewriteSyncReturnStmts(Stmt *S, std::string buf);
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
    void SynthSuperContructorFunctionDecl();

    // Metadata emission.
    void RewriteObjCClassMetaData(ObjCImplementationDecl *IDecl,
                                  std::string &Result);

    void RewriteObjCCategoryImplDecl(ObjCCategoryImplDecl *CDecl,
                                     std::string &Result);

    template<typename MethodIterator>
    void RewriteObjCMethodsMetaData(MethodIterator MethodBegin,
                                    MethodIterator MethodEnd,
                                    bool IsInstanceMethod,
                                    const char *prefix,
                                    const char *ClassName,
                                    std::string &Result);

    void RewriteObjCProtocolMetaData(ObjCProtocolDecl *Protocol,
                                     const char *prefix,
                                     const char *ClassName,
                                     std::string &Result);
    void RewriteObjCProtocolListMetaData(const ObjCList<ObjCProtocolDecl> &Prots,
                                         const char *prefix,
                                         const char *ClassName,
                                         std::string &Result);
    void SynthesizeObjCInternalStruct(ObjCInterfaceDecl *CDecl,
                                      std::string &Result);
    void SynthesizeIvarOffsetComputation(ObjCImplementationDecl *IDecl,
                                         ObjCIvarDecl *ivar,
                                         std::string &Result);
    void RewriteImplementations();
    void SynthesizeMetaDataIntoBuffer(std::string &Result);

    // Block rewriting.
    void RewriteBlocksInFunctionProtoType(QualType funcType, NamedDecl *D);
    void CheckFunctionPointerDecl(QualType dType, NamedDecl *ND);

    void InsertBlockLiteralsWithinFunction(FunctionDecl *FD);
    void InsertBlockLiteralsWithinMethod(ObjCMethodDecl *MD);

    // Block specific rewrite rules.
    void RewriteBlockCall(CallExpr *Exp);
    void RewriteBlockPointerDecl(NamedDecl *VD);
    void RewriteByRefVar(VarDecl *VD);
    std::string SynthesizeByrefCopyDestroyHelper(VarDecl *VD, int flag);
    Stmt *RewriteBlockDeclRefExpr(Expr *VD);
    void RewriteBlockPointerFunctionArgs(FunctionDecl *FD);

    std::string SynthesizeBlockHelperFuncs(BlockExpr *CE, int i,
                                      const char *funcName, std::string Tag);
    std::string SynthesizeBlockFunc(BlockExpr *CE, int i,
                                      const char *funcName, std::string Tag);
    std::string SynthesizeBlockImpl(BlockExpr *CE, 
                                    std::string Tag, std::string Desc);
    std::string SynthesizeBlockDescriptor(std::string DescTag, 
                                          std::string ImplTag,
                                          int i, const char *funcName,
                                          unsigned hasCopy);
    Stmt *SynthesizeBlockCall(CallExpr *Exp, const Expr* BlockExp);
    void SynthesizeBlockLiterals(SourceLocation FunLocStart,
                                 const char *FunName);
    void RewriteRecordBody(RecordDecl *RD);

    void CollectBlockDeclRefInfo(BlockExpr *Exp);
    void GetBlockCallExprs(Stmt *S);
    void GetBlockDeclRefExprs(Stmt *S);

    // We avoid calling Type::isBlockPointerType(), since it operates on the
    // canonical type. We only care if the top-level type is a closure pointer.
    bool isTopLevelBlockPointerType(QualType T) {
      return isa<BlockPointerType>(T);
    }

    // FIXME: This predicate seems like it would be useful to add to ASTContext.
    bool isObjCType(QualType T) {
      if (!LangOpts.ObjC1 && !LangOpts.ObjC2)
        return false;

      QualType OCT = Context->getCanonicalType(T).getUnqualifiedType();

      if (OCT == Context->getCanonicalType(Context->getObjCIdType()) ||
          OCT == Context->getCanonicalType(Context->getObjCClassType()))
        return true;

      if (const PointerType *PT = OCT->getAs<PointerType>()) {
        if (isa<ObjCInterfaceType>(PT->getPointeeType()) ||
            PT->getPointeeType()->isObjCQualifiedIdType())
          return true;
      }
      return false;
    }
    bool PointerTypeTakesAnyBlockArguments(QualType QT);
    void GetExtentOfArgList(const char *Name, const char *&LParen,
                            const char *&RParen);
    void RewriteCastExpr(CStyleCastExpr *CE);

    FunctionDecl *SynthBlockInitFunctionDecl(const char *name);
    Stmt *SynthBlockInitExpr(BlockExpr *Exp);

    void QuoteDoublequotes(std::string &From, std::string &To) {
      for (unsigned i = 0; i < From.length(); i++) {
        if (From[i] == '"')
          To += "\\\"";
        else
          To += From[i];
      }
    }
  };

  // Helper function: create a CStyleCastExpr with trivial type source info.
  CStyleCastExpr* NoTypeInfoCStyleCastExpr(ASTContext *Ctx, QualType Ty,
                                           CastExpr::CastKind Kind, Expr *E) {
    TypeSourceInfo *TInfo = Ctx->getTrivialTypeSourceInfo(Ty, SourceLocation());
    return new (Ctx) CStyleCastExpr(Ty, Kind, E, TInfo,
                                    SourceLocation(), SourceLocation());
  }
}

void RewriteObjC::RewriteBlocksInFunctionProtoType(QualType funcType,
                                                   NamedDecl *D) {
  if (FunctionProtoType *fproto = dyn_cast<FunctionProtoType>(funcType)) {
    for (FunctionProtoType::arg_type_iterator I = fproto->arg_type_begin(),
         E = fproto->arg_type_end(); I && (I != E); ++I)
      if (isTopLevelBlockPointerType(*I)) {
        // All the args are checked/rewritten. Don't call twice!
        RewriteBlockPointerDecl(D);
        break;
      }
  }
}

void RewriteObjC::CheckFunctionPointerDecl(QualType funcType, NamedDecl *ND) {
  const PointerType *PT = funcType->getAs<PointerType>();
  if (PT && PointerTypeTakesAnyBlockArguments(funcType))
    RewriteBlocksInFunctionProtoType(PT->getPointeeType(), ND);
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

RewriteObjC::RewriteObjC(std::string inFile, llvm::raw_ostream* OS,
                         Diagnostic &D, const LangOptions &LOpts,
                         bool silenceMacroWarn)
      : Diags(D), LangOpts(LOpts), InFileName(inFile), OutFile(OS),
        SilenceRewriteMacroWarning(silenceMacroWarn) {
  IsHeader = IsHeaderFile(inFile);
  RewriteFailedDiag = Diags.getCustomDiagID(Diagnostic::Warning,
               "rewriting sub-expression within a macro (may not be correct)");
  TryFinallyContainsReturnDiag = Diags.getCustomDiagID(Diagnostic::Warning,
               "rewriter doesn't support user-specified control flow semantics "
               "for @try/@finally (code may not execute properly)");
}

ASTConsumer *clang::CreateObjCRewriter(const std::string& InFile,
                                       llvm::raw_ostream* OS,
                                       Diagnostic &Diags,
                                       const LangOptions &LOpts,
                                       bool SilenceRewriteMacroWarning) {
  return new RewriteObjC(InFile, OS, Diags, LOpts, SilenceRewriteMacroWarning);
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
  ConstantStringClassReference = 0;
  NSStringRecord = 0;
  CurMethodDef = 0;
  CurFunctionDef = 0;
  CurFunctionDeclToDeclareForBlock = 0;
  GlobalVarDecl = 0;
  SuperStructDecl = 0;
  ProtocolTypeDecl = 0;
  ConstantStringDecl = 0;
  BcLabelCount = 0;
  SuperContructorFunctionDecl = 0;
  NumObjCStringLiterals = 0;
  PropParentMap = 0;
  CurrentBody = 0;
  DisableReplaceStmt = false;
  objc_impl_method = false;

  // Get the ID and start/end of the main file.
  MainFileID = SM->getMainFileID();
  const llvm::MemoryBuffer *MainBuf = SM->getBuffer(MainFileID);
  MainFileStart = MainBuf->getBufferStart();
  MainFileEnd = MainBuf->getBufferEnd();

  Rewrite.setSourceMgr(Context->getSourceManager(), Context->getLangOptions());

  // declaring objc_selector outside the parameter list removes a silly
  // scope related warning...
  if (IsHeader)
    Preamble = "#pragma once\n";
  Preamble += "struct objc_selector; struct objc_class;\n";
  Preamble += "struct __rw_objc_super { struct objc_object *object; ";
  Preamble += "struct objc_object *superClass; ";
  if (LangOpts.Microsoft) {
    // Add a constructor for creating temporary objects.
    Preamble += "__rw_objc_super(struct objc_object *o, struct objc_object *s) "
                ": ";
    Preamble += "object(o), superClass(s) {} ";
  }
  Preamble += "};\n";
  Preamble += "#ifndef _REWRITER_typedef_Protocol\n";
  Preamble += "typedef struct objc_object Protocol;\n";
  Preamble += "#define _REWRITER_typedef_Protocol\n";
  Preamble += "#endif\n";
  if (LangOpts.Microsoft) {
    Preamble += "#define __OBJC_RW_DLLIMPORT extern \"C\" __declspec(dllimport)\n";
    Preamble += "#define __OBJC_RW_STATICIMPORT extern \"C\"\n";
  } else
    Preamble += "#define __OBJC_RW_DLLIMPORT extern\n";
  Preamble += "__OBJC_RW_DLLIMPORT struct objc_object *objc_msgSend";
  Preamble += "(struct objc_object *, struct objc_selector *, ...);\n";
  Preamble += "__OBJC_RW_DLLIMPORT struct objc_object *objc_msgSendSuper";
  Preamble += "(struct objc_super *, struct objc_selector *, ...);\n";
  Preamble += "__OBJC_RW_DLLIMPORT struct objc_object *objc_msgSend_stret";
  Preamble += "(struct objc_object *, struct objc_selector *, ...);\n";
  Preamble += "__OBJC_RW_DLLIMPORT struct objc_object *objc_msgSendSuper_stret";
  Preamble += "(struct objc_super *, struct objc_selector *, ...);\n";
  Preamble += "__OBJC_RW_DLLIMPORT double objc_msgSend_fpret";
  Preamble += "(struct objc_object *, struct objc_selector *, ...);\n";
  Preamble += "__OBJC_RW_DLLIMPORT struct objc_object *objc_getClass";
  Preamble += "(const char *);\n";
  Preamble += "__OBJC_RW_DLLIMPORT struct objc_object *objc_getMetaClass";
  Preamble += "(const char *);\n";
  Preamble += "__OBJC_RW_DLLIMPORT void objc_exception_throw(struct objc_object *);\n";
  Preamble += "__OBJC_RW_DLLIMPORT void objc_exception_try_enter(void *);\n";
  Preamble += "__OBJC_RW_DLLIMPORT void objc_exception_try_exit(void *);\n";
  Preamble += "__OBJC_RW_DLLIMPORT struct objc_object *objc_exception_extract(void *);\n";
  Preamble += "__OBJC_RW_DLLIMPORT int objc_exception_match";
  Preamble += "(struct objc_class *, struct objc_object *);\n";
  // @synchronized hooks.
  Preamble += "__OBJC_RW_DLLIMPORT void objc_sync_enter(struct objc_object *);\n";
  Preamble += "__OBJC_RW_DLLIMPORT void objc_sync_exit(struct objc_object *);\n";
  Preamble += "__OBJC_RW_DLLIMPORT Protocol *objc_getProtocol(const char *);\n";
  Preamble += "#ifndef __FASTENUMERATIONSTATE\n";
  Preamble += "struct __objcFastEnumerationState {\n\t";
  Preamble += "unsigned long state;\n\t";
  Preamble += "void **itemsPtr;\n\t";
  Preamble += "unsigned long *mutationsPtr;\n\t";
  Preamble += "unsigned long extra[5];\n};\n";
  Preamble += "__OBJC_RW_DLLIMPORT void objc_enumerationMutation(struct objc_object *);\n";
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
  Preamble += "__OBJC_RW_DLLIMPORT int __CFConstantStringClassReference[];\n";
  Preamble += "#endif\n";
  Preamble += "#define __NSCONSTANTSTRINGIMPL\n";
  Preamble += "#endif\n";
  // Blocks preamble.
  Preamble += "#ifndef BLOCK_IMPL\n";
  Preamble += "#define BLOCK_IMPL\n";
  Preamble += "struct __block_impl {\n";
  Preamble += "  void *isa;\n";
  Preamble += "  int Flags;\n";
  Preamble += "  int Reserved;\n";
  Preamble += "  void *FuncPtr;\n";
  Preamble += "};\n";
  Preamble += "// Runtime copy/destroy helper functions (from Block_private.h)\n";
  Preamble += "#ifdef __OBJC_EXPORT_BLOCKS\n";
  Preamble += "extern \"C\" __declspec(dllexport) void _Block_object_assign(void *, const void *, const int);\n";
  Preamble += "extern \"C\" __declspec(dllexport) void _Block_object_dispose(const void *, const int);\n";
  Preamble += "extern \"C\" __declspec(dllexport) void *_NSConcreteGlobalBlock[32];\n";
  Preamble += "extern \"C\" __declspec(dllexport) void *_NSConcreteStackBlock[32];\n";
  Preamble += "#else\n";
  Preamble += "__OBJC_RW_DLLIMPORT void _Block_object_assign(void *, const void *, const int);\n";
  Preamble += "__OBJC_RW_DLLIMPORT void _Block_object_dispose(const void *, const int);\n";
  Preamble += "__OBJC_RW_DLLIMPORT void *_NSConcreteGlobalBlock[32];\n";
  Preamble += "__OBJC_RW_DLLIMPORT void *_NSConcreteStackBlock[32];\n";
  Preamble += "#endif\n";
  Preamble += "#endif\n";
  if (LangOpts.Microsoft) {
    Preamble += "#undef __OBJC_RW_DLLIMPORT\n";
    Preamble += "#undef __OBJC_RW_STATICIMPORT\n";
    Preamble += "#define __attribute__(X)\n";
    Preamble += "#define __weak\n";
  }
  else {
    Preamble += "#define __block\n";
    Preamble += "#define __weak\n";
  }
}


//===----------------------------------------------------------------------===//
// Top Level Driver Code
//===----------------------------------------------------------------------===//

void RewriteObjC::HandleTopLevelSingleDecl(Decl *D) {
  // Two cases: either the decl could be in the main file, or it could be in a
  // #included file.  If the former, rewrite it now.  If the later, check to see
  // if we rewrote the #include/#import.
  SourceLocation Loc = D->getLocation();
  Loc = SM->getInstantiationLoc(Loc);

  // If this is for a builtin, ignore it.
  if (Loc.isInvalid()) return;

  // Look for built-in declarations that we need to refer during the rewrite.
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    RewriteFunctionDecl(FD);
  } else if (VarDecl *FVD = dyn_cast<VarDecl>(D)) {
    // declared in <Foundation/NSString.h>
    if (strcmp(FVD->getNameAsCString(), "_NSConstantStringClassReference") == 0) {
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
  } else if (LinkageSpecDecl *LSD = dyn_cast<LinkageSpecDecl>(D)) {
    // Recurse into linkage specifications
    for (DeclContext::decl_iterator DI = LSD->decls_begin(),
                                 DIEnd = LSD->decls_end();
         DI != DIEnd; ++DI)
      HandleTopLevelSingleDecl(*DI);
  }
  // If we have a decl in the main file, see if we should rewrite it.
  if (SM->isFromMainFile(Loc))
    return HandleDeclInMainFile(D);
}

//===----------------------------------------------------------------------===//
// Syntactic (non-AST) Rewriting Code
//===----------------------------------------------------------------------===//

void RewriteObjC::RewriteInclude() {
  SourceLocation LocStart = SM->getLocForStartOfFile(MainFileID);
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
    SourceLocation TabLoc = SM->getLocForStartOfFile(MainFileID);
    TabLoc = TabLoc.getFileLocWithOffset(BufPtr-MainBufStart);

    // Rewrite the single tab character into a sequence of spaces.
    ReplaceText(TabLoc, 1, "        ", Spaces);
  }
}

static std::string getIvarAccessString(ObjCInterfaceDecl *ClassDecl,
                                       ObjCIvarDecl *OID) {
  std::string S;
  S = "((struct ";
  S += ClassDecl->getIdentifier()->getName();
  S += "_IMPL *)self)->";
  S += OID->getName();
  return S;
}

void RewriteObjC::RewritePropertyImplDecl(ObjCPropertyImplDecl *PID,
                                          ObjCImplementationDecl *IMD,
                                          ObjCCategoryImplDecl *CID) {
  SourceLocation startLoc = PID->getLocStart();
  InsertText(startLoc, "// ", 3);
  const char *startBuf = SM->getCharacterData(startLoc);
  assert((*startBuf == '@') && "bogus @synthesize location");
  const char *semiBuf = strchr(startBuf, ';');
  assert((*semiBuf == ';') && "@synthesize: can't find ';'");
  SourceLocation onePastSemiLoc =
    startLoc.getFileLocWithOffset(semiBuf-startBuf+1);

  if (PID->getPropertyImplementation() == ObjCPropertyImplDecl::Dynamic)
    return; // FIXME: is this correct?

  // Generate the 'getter' function.
  ObjCPropertyDecl *PD = PID->getPropertyDecl();
  ObjCInterfaceDecl *ClassDecl = PD->getGetterMethodDecl()->getClassInterface();
  ObjCIvarDecl *OID = PID->getPropertyIvarDecl();

  if (!OID)
    return;

  std::string Getr;
  RewriteObjCMethodDecl(PD->getGetterMethodDecl(), Getr);
  Getr += "{ ";
  // Synthesize an explicit cast to gain access to the ivar.
  // FIXME: deal with code generation implications for various property
  // attributes (copy, retain, nonatomic).
  // See objc-act.c:objc_synthesize_new_getter() for details.
  Getr += "return " + getIvarAccessString(ClassDecl, OID);
  Getr += "; }";
  InsertText(onePastSemiLoc, Getr.c_str(), Getr.size());
  if (PD->isReadOnly())
    return;

  // Generate the 'setter' function.
  std::string Setr;
  RewriteObjCMethodDecl(PD->getSetterMethodDecl(), Setr);
  Setr += "{ ";
  // Synthesize an explicit cast to initialize the ivar.
  // FIXME: deal with code generation implications for various property
  // attributes (copy, retain, nonatomic).
  // See objc-act.c:objc_synthesize_new_setter() for details.
  Setr += getIvarAccessString(ClassDecl, OID) + " = ";
  Setr += PD->getNameAsCString();
  Setr += "; }";
  InsertText(onePastSemiLoc, Setr.c_str(), Setr.size());
}

void RewriteObjC::RewriteForwardClassDecl(ObjCClassDecl *ClassDecl) {
  // Get the start location and compute the semi location.
  SourceLocation startLoc = ClassDecl->getLocation();
  const char *startBuf = SM->getCharacterData(startLoc);
  const char *semiPtr = strchr(startBuf, ';');

  // Translate to typedef's that forward reference structs with the same name
  // as the class. As a convenience, we include the original declaration
  // as a comment.
  std::string typedefString;
  typedefString += "// @class ";
  for (ObjCClassDecl::iterator I = ClassDecl->begin(), E = ClassDecl->end();
       I != E; ++I) {
    ObjCInterfaceDecl *ForwardDecl = I->getInterface();
    typedefString += ForwardDecl->getNameAsString();
    if (I+1 != E)
      typedefString += ", ";
    else
      typedefString += ";\n";
  }
  
  for (ObjCClassDecl::iterator I = ClassDecl->begin(), E = ClassDecl->end();
       I != E; ++I) {
    ObjCInterfaceDecl *ForwardDecl = I->getInterface();
    typedefString += "#ifndef _REWRITER_typedef_";
    typedefString += ForwardDecl->getNameAsString();
    typedefString += "\n";
    typedefString += "#define _REWRITER_typedef_";
    typedefString += ForwardDecl->getNameAsString();
    typedefString += "\n";
    typedefString += "typedef struct objc_object ";
    typedefString += ForwardDecl->getNameAsString();
    typedefString += ";\n#endif\n";
  }

  // Replace the @class with typedefs corresponding to the classes.
  ReplaceText(startLoc, semiPtr-startBuf+1,
              typedefString.c_str(), typedefString.size());
}

void RewriteObjC::RewriteMethodDeclaration(ObjCMethodDecl *Method) {
  SourceLocation LocStart = Method->getLocStart();
  SourceLocation LocEnd = Method->getLocEnd();

  if (SM->getInstantiationLineNumber(LocEnd) >
      SM->getInstantiationLineNumber(LocStart)) {
    InsertText(LocStart, "#if 0\n", 6);
    ReplaceText(LocEnd, 1, ";\n#endif\n", 9);
  } else {
    InsertText(LocStart, "// ", 3);
  }
}

void RewriteObjC::RewriteProperty(ObjCPropertyDecl *prop) {
  SourceLocation Loc = prop->getLocation();

  ReplaceText(Loc, 0, "// ", 3);

  // FIXME: handle properties that are declared across multiple lines.
}

void RewriteObjC::RewriteCategoryDecl(ObjCCategoryDecl *CatDecl) {
  SourceLocation LocStart = CatDecl->getLocStart();

  // FIXME: handle category headers that are declared across multiple lines.
  ReplaceText(LocStart, 0, "// ", 3);

  for (ObjCCategoryDecl::instmeth_iterator
         I = CatDecl->instmeth_begin(), E = CatDecl->instmeth_end();
       I != E; ++I)
    RewriteMethodDeclaration(*I);
  for (ObjCCategoryDecl::classmeth_iterator
         I = CatDecl->classmeth_begin(), E = CatDecl->classmeth_end();
       I != E; ++I)
    RewriteMethodDeclaration(*I);

  // Lastly, comment out the @end.
  ReplaceText(CatDecl->getAtEndRange().getBegin(), 0, "// ", 3);
}

void RewriteObjC::RewriteProtocolDecl(ObjCProtocolDecl *PDecl) {
  std::pair<const char*, const char*> MainBuf = SM->getBufferData(MainFileID);

  SourceLocation LocStart = PDecl->getLocStart();

  // FIXME: handle protocol headers that are declared across multiple lines.
  ReplaceText(LocStart, 0, "// ", 3);

  for (ObjCProtocolDecl::instmeth_iterator
         I = PDecl->instmeth_begin(), E = PDecl->instmeth_end();
       I != E; ++I)
    RewriteMethodDeclaration(*I);
  for (ObjCProtocolDecl::classmeth_iterator
         I = PDecl->classmeth_begin(), E = PDecl->classmeth_end();
       I != E; ++I)
    RewriteMethodDeclaration(*I);

  // Lastly, comment out the @end.
  SourceLocation LocEnd = PDecl->getAtEndRange().getBegin();
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
  //fprintf(stderr,"In RewriteObjCMethodDecl\n");
  const FunctionType *FPRetType = 0;
  ResultStr += "\nstatic ";
  if (OMD->getResultType()->isObjCQualifiedIdType())
    ResultStr += "id";
  else if (OMD->getResultType()->isFunctionPointerType() ||
           OMD->getResultType()->isBlockPointerType()) {
    // needs special handling, since pointer-to-functions have special
    // syntax (where a decaration models use).
    QualType retType = OMD->getResultType();
    QualType PointeeTy;
    if (const PointerType* PT = retType->getAs<PointerType>())
      PointeeTy = PT->getPointeeType();
    else if (const BlockPointerType *BPT = retType->getAs<BlockPointerType>())
      PointeeTy = BPT->getPointeeType();
    if ((FPRetType = PointeeTy->getAs<FunctionType>())) {
      ResultStr += FPRetType->getResultType().getAsString();
      ResultStr += "(*";
    }
  } else
    ResultStr += OMD->getResultType().getAsString();
  ResultStr += " ";

  // Unique method name
  std::string NameStr;

  if (OMD->isInstanceMethod())
    NameStr += "_I_";
  else
    NameStr += "_C_";

  NameStr += OMD->getClassInterface()->getNameAsString();
  NameStr += "_";

  if (ObjCCategoryImplDecl *CID =
      dyn_cast<ObjCCategoryImplDecl>(OMD->getDeclContext())) {
    NameStr += CID->getNameAsString();
    NameStr += "_";
  }
  // Append selector names, replacing ':' with '_'
  {
    std::string selString = OMD->getSelector().getAsString();
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
  if (OMD->isInstanceMethod()) {
    QualType selfTy = Context->getObjCInterfaceType(OMD->getClassInterface());
    selfTy = Context->getPointerType(selfTy);
    if (!LangOpts.Microsoft) {
      if (ObjCSynthesizedStructs.count(OMD->getClassInterface()))
        ResultStr += "struct ";
    }
    // When rewriting for Microsoft, explicitly omit the structure name.
    ResultStr += OMD->getClassInterface()->getNameAsString();
    ResultStr += " *";
  }
  else
    ResultStr += Context->getObjCClassType().getAsString();

  ResultStr += " self, ";
  ResultStr += Context->getObjCSelType().getAsString();
  ResultStr += " _cmd";

  // Method arguments.
  for (ObjCMethodDecl::param_iterator PI = OMD->param_begin(),
       E = OMD->param_end(); PI != E; ++PI) {
    ParmVarDecl *PDecl = *PI;
    ResultStr += ", ";
    if (PDecl->getType()->isObjCQualifiedIdType()) {
      ResultStr += "id ";
      ResultStr += PDecl->getNameAsString();
    } else {
      std::string Name = PDecl->getNameAsString();
      if (isTopLevelBlockPointerType(PDecl->getType())) {
        // Make sure we convert "t (^)(...)" to "t (*)(...)".
        const BlockPointerType *BPT = PDecl->getType()->getAs<BlockPointerType>();
        Context->getPointerType(BPT->getPointeeType()).getAsStringInternal(Name,
                                                        Context->PrintingPolicy);
      } else
        PDecl->getType().getAsStringInternal(Name, Context->PrintingPolicy);
      ResultStr += Name;
    }
  }
  if (OMD->isVariadic())
    ResultStr += ", ...";
  ResultStr += ") ";

  if (FPRetType) {
    ResultStr += ")"; // close the precedence "scope" for "*".

    // Now, emit the argument types (if any).
    if (const FunctionProtoType *FT = dyn_cast<FunctionProtoType>(FPRetType)) {
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
void RewriteObjC::RewriteImplementationDecl(Decl *OID) {
  ObjCImplementationDecl *IMD = dyn_cast<ObjCImplementationDecl>(OID);
  ObjCCategoryImplDecl *CID = dyn_cast<ObjCCategoryImplDecl>(OID);

  if (IMD)
    InsertText(IMD->getLocStart(), "// ", 3);
  else
    InsertText(CID->getLocStart(), "// ", 3);

  for (ObjCCategoryImplDecl::instmeth_iterator
       I = IMD ? IMD->instmeth_begin() : CID->instmeth_begin(),
       E = IMD ? IMD->instmeth_end() : CID->instmeth_end();
       I != E; ++I) {
    std::string ResultStr;
    ObjCMethodDecl *OMD = *I;
    RewriteObjCMethodDecl(OMD, ResultStr);
    SourceLocation LocStart = OMD->getLocStart();
    SourceLocation LocEnd = OMD->getCompoundBody()->getLocStart();

    const char *startBuf = SM->getCharacterData(LocStart);
    const char *endBuf = SM->getCharacterData(LocEnd);
    ReplaceText(LocStart, endBuf-startBuf,
                ResultStr.c_str(), ResultStr.size());
  }

  for (ObjCCategoryImplDecl::classmeth_iterator
       I = IMD ? IMD->classmeth_begin() : CID->classmeth_begin(),
       E = IMD ? IMD->classmeth_end() : CID->classmeth_end();
       I != E; ++I) {
    std::string ResultStr;
    ObjCMethodDecl *OMD = *I;
    RewriteObjCMethodDecl(OMD, ResultStr);
    SourceLocation LocStart = OMD->getLocStart();
    SourceLocation LocEnd = OMD->getCompoundBody()->getLocStart();

    const char *startBuf = SM->getCharacterData(LocStart);
    const char *endBuf = SM->getCharacterData(LocEnd);
    ReplaceText(LocStart, endBuf-startBuf,
                ResultStr.c_str(), ResultStr.size());
  }
  for (ObjCCategoryImplDecl::propimpl_iterator
       I = IMD ? IMD->propimpl_begin() : CID->propimpl_begin(),
       E = IMD ? IMD->propimpl_end() : CID->propimpl_end();
       I != E; ++I) {
    RewritePropertyImplDecl(*I, IMD, CID);
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
    ResultStr += ClassDecl->getNameAsString();
    ResultStr += "\n";
    ResultStr += "#define _REWRITER_typedef_";
    ResultStr += ClassDecl->getNameAsString();
    ResultStr += "\n";
    ResultStr += "typedef struct objc_object ";
    ResultStr += ClassDecl->getNameAsString();
    ResultStr += ";\n#endif\n";
    // Mark this typedef as having been generated.
    ObjCForwardDecls.insert(ClassDecl);
  }
  SynthesizeObjCInternalStruct(ClassDecl, ResultStr);

  for (ObjCInterfaceDecl::prop_iterator I = ClassDecl->prop_begin(),
         E = ClassDecl->prop_end(); I != E; ++I)
    RewriteProperty(*I);
  for (ObjCInterfaceDecl::instmeth_iterator
         I = ClassDecl->instmeth_begin(), E = ClassDecl->instmeth_end();
       I != E; ++I)
    RewriteMethodDeclaration(*I);
  for (ObjCInterfaceDecl::classmeth_iterator
         I = ClassDecl->classmeth_begin(), E = ClassDecl->classmeth_end();
       I != E; ++I)
    RewriteMethodDeclaration(*I);

  // Lastly, comment out the @end.
  ReplaceText(ClassDecl->getAtEndRange().getBegin(), 0, "// ", 3);
}

Stmt *RewriteObjC::RewritePropertySetter(BinaryOperator *BinOp, Expr *newStmt,
                                         SourceRange SrcRange) {
  // Synthesize a ObjCMessageExpr from a ObjCPropertyRefExpr.
  // This allows us to reuse all the fun and games in SynthMessageExpr().
  ObjCPropertyRefExpr *PropRefExpr = dyn_cast<ObjCPropertyRefExpr>(BinOp->getLHS());
  ObjCMessageExpr *MsgExpr;
  ObjCPropertyDecl *PDecl = PropRefExpr->getProperty();
  llvm::SmallVector<Expr *, 1> ExprVec;
  ExprVec.push_back(newStmt);

  Stmt *Receiver = PropRefExpr->getBase();
  ObjCPropertyRefExpr *PRE = dyn_cast<ObjCPropertyRefExpr>(Receiver);
  if (PRE && PropGetters[PRE]) {
    // This allows us to handle chain/nested property getters.
    Receiver = PropGetters[PRE];
  }
  MsgExpr = new (Context) ObjCMessageExpr(dyn_cast<Expr>(Receiver),
                                PDecl->getSetterName(), PDecl->getType(),
                                PDecl->getSetterMethodDecl(),
                                SourceLocation(), SourceLocation(),
                                &ExprVec[0], 1);
  Stmt *ReplacingStmt = SynthMessageExpr(MsgExpr);

  // Now do the actual rewrite.
  ReplaceStmtWithRange(BinOp, ReplacingStmt, SrcRange);
  //delete BinOp;
  // NOTE: We don't want to call MsgExpr->Destroy(), as it holds references
  // to things that stay around.
  Context->Deallocate(MsgExpr);
  return ReplacingStmt;
}

Stmt *RewriteObjC::RewritePropertyGetter(ObjCPropertyRefExpr *PropRefExpr) {
  // Synthesize a ObjCMessageExpr from a ObjCPropertyRefExpr.
  // This allows us to reuse all the fun and games in SynthMessageExpr().
  ObjCMessageExpr *MsgExpr;
  ObjCPropertyDecl *PDecl = PropRefExpr->getProperty();

  Stmt *Receiver = PropRefExpr->getBase();

  ObjCPropertyRefExpr *PRE = dyn_cast<ObjCPropertyRefExpr>(Receiver);
  if (PRE && PropGetters[PRE]) {
    // This allows us to handle chain/nested property getters.
    Receiver = PropGetters[PRE];
  }
  MsgExpr = new (Context) ObjCMessageExpr(dyn_cast<Expr>(Receiver),
                                PDecl->getGetterName(), PDecl->getType(),
                                PDecl->getGetterMethodDecl(),
                                SourceLocation(), SourceLocation(),
                                0, 0);

  Stmt *ReplacingStmt = SynthMessageExpr(MsgExpr);

  if (!PropParentMap)
    PropParentMap = new ParentMap(CurrentBody);

  Stmt *Parent = PropParentMap->getParent(PropRefExpr);
  if (Parent && isa<ObjCPropertyRefExpr>(Parent)) {
    // We stash away the ReplacingStmt since actually doing the
    // replacement/rewrite won't work for nested getters (e.g. obj.p.i)
    PropGetters[PropRefExpr] = ReplacingStmt;
    // NOTE: We don't want to call MsgExpr->Destroy(), as it holds references
    // to things that stay around.
    Context->Deallocate(MsgExpr);
    return PropRefExpr; // return the original...
  } else {
    ReplaceStmt(PropRefExpr, ReplacingStmt);
    // delete PropRefExpr; elsewhere...
    // NOTE: We don't want to call MsgExpr->Destroy(), as it holds references
    // to things that stay around.
    Context->Deallocate(MsgExpr);
    return ReplacingStmt;
  }
}

Stmt *RewriteObjC::RewriteObjCIvarRefExpr(ObjCIvarRefExpr *IV,
                                          SourceLocation OrigStart) {
  ObjCIvarDecl *D = IV->getDecl();
  const Expr *BaseExpr = IV->getBase();
  if (CurMethodDef) {
    if (BaseExpr->getType()->isObjCObjectPointerType() && 
        isa<DeclRefExpr>(BaseExpr)) {
      ObjCInterfaceType *iFaceDecl =
        dyn_cast<ObjCInterfaceType>(BaseExpr->getType()->getPointeeType());
      // lookup which class implements the instance variable.
      ObjCInterfaceDecl *clsDeclared = 0;
      iFaceDecl->getDecl()->lookupInstanceVariable(D->getIdentifier(),
                                                   clsDeclared);
      assert(clsDeclared && "RewriteObjCIvarRefExpr(): Can't find class");

      // Synthesize an explicit cast to gain access to the ivar.
      std::string RecName = clsDeclared->getIdentifier()->getName();
      RecName += "_IMPL";
      IdentifierInfo *II = &Context->Idents.get(RecName.c_str());
      RecordDecl *RD = RecordDecl::Create(*Context, TagDecl::TK_struct, TUDecl,
                                          SourceLocation(), II);
      assert(RD && "RewriteObjCIvarRefExpr(): Can't find RecordDecl");
      QualType castT = Context->getPointerType(Context->getTagDeclType(RD));
      CastExpr *castExpr = NoTypeInfoCStyleCastExpr(Context, castT,
                                                    CastExpr::CK_Unknown,
                                                    IV->getBase());
      // Don't forget the parens to enforce the proper binding.
      ParenExpr *PE = new (Context) ParenExpr(IV->getBase()->getLocStart(),
                                               IV->getBase()->getLocEnd(),
                                               castExpr);
      if (IV->isFreeIvar() &&
          CurMethodDef->getClassInterface() == iFaceDecl->getDecl()) {
        MemberExpr *ME = new (Context) MemberExpr(PE, true, D,
                                                   IV->getLocation(),
                                                   D->getType());
        ReplaceStmt(IV, ME);
        // delete IV; leak for now, see RewritePropertySetter() usage for more info.
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
    if (BaseExpr->getType()->isObjCObjectPointerType()) {
      ObjCInterfaceType *iFaceDecl =
        dyn_cast<ObjCInterfaceType>(BaseExpr->getType()->getPointeeType());
      // lookup which class implements the instance variable.
      ObjCInterfaceDecl *clsDeclared = 0;
      iFaceDecl->getDecl()->lookupInstanceVariable(D->getIdentifier(),
                                                   clsDeclared);
      assert(clsDeclared && "RewriteObjCIvarRefExpr(): Can't find class");

      // Synthesize an explicit cast to gain access to the ivar.
      std::string RecName = clsDeclared->getIdentifier()->getName();
      RecName += "_IMPL";
      IdentifierInfo *II = &Context->Idents.get(RecName.c_str());
      RecordDecl *RD = RecordDecl::Create(*Context, TagDecl::TK_struct, TUDecl,
                                          SourceLocation(), II);
      assert(RD && "RewriteObjCIvarRefExpr(): Can't find RecordDecl");
      QualType castT = Context->getPointerType(Context->getTagDeclType(RD));
      CastExpr *castExpr = NoTypeInfoCStyleCastExpr(Context, castT,
                                                    CastExpr::CK_Unknown,
                                                    IV->getBase());
      // Don't forget the parens to enforce the proper binding.
      ParenExpr *PE = new (Context) ParenExpr(IV->getBase()->getLocStart(),
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
    NamedDecl* D = cast<NamedDecl>(DS->getSingleDecl());
    QualType ElementType = cast<ValueDecl>(D)->getType();
    if (ElementType->isObjCQualifiedIdType() ||
        ElementType->isObjCQualifiedInterfaceType())
      // Simply use 'id' for all qualified types.
      elementTypeAsString = "id";
    else
      elementTypeAsString = ElementType.getAsString();
    buf += elementTypeAsString;
    buf += " ";
    elementName = D->getNameAsCString();
    buf += elementName;
    buf += ";\n\t";
  }
  else {
    DeclRefExpr *DR = cast<DeclRefExpr>(S->getElement());
    elementName = DR->getDecl()->getNameAsCString();
    ValueDecl *VD = cast<ValueDecl>(DR->getDecl());
    if (VD->getType()->isObjCQualifiedIdType() ||
        VD->getType()->isObjCQualifiedInterfaceType())
      // Simply use 'id' for all qualified types.
      elementTypeAsString = "id";
    else
      elementTypeAsString = VD->getType().getAsString();
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
  buf += " = ((";
  buf += elementTypeAsString;
  buf += ")0);\n\t";
  buf += "__break_label_";
  buf += utostr(ObjCBcLabelNo.back());
  buf += ": ;\n\t";
  buf += "}\n\t";
  buf += "else\n\t\t";
  buf += elementName;
  buf += " = ((";
  buf += elementTypeAsString;
  buf += ")0);\n\t";
  buf += "}\n";

  // Insert all these *after* the statement body.
  // FIXME: If this should support Obj-C++, support CXXTryStmt
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
  buf += "}\n";
  buf += "{ /* implicit finally clause */\n";
  buf += "  if (!_rethrow) objc_exception_try_exit(&_stack);\n";
  
  std::string syncBuf;
  syncBuf += " objc_sync_exit(";
  Expr *syncExpr = NoTypeInfoCStyleCastExpr(Context, Context->getObjCIdType(),
                                            CastExpr::CK_Unknown,
                                            S->getSynchExpr());
  std::string syncExprBufS;
  llvm::raw_string_ostream syncExprBuf(syncExprBufS);
  syncExpr->printPretty(syncExprBuf, *Context, 0,
                        PrintingPolicy(LangOpts));
  syncBuf += syncExprBuf.str();
  syncBuf += ");";
  
  buf += syncBuf;
  buf += "\n  if (_rethrow) objc_exception_throw(_rethrow);\n";
  buf += "}\n";
  buf += "}";

  ReplaceText(lastCurlyLoc, 1, buf.c_str(), buf.size());

  bool hasReturns = false;
  HasReturnStmts(S->getSynchBody(), hasReturns);
  if (hasReturns)
    RewriteSyncReturnStmts(S->getSynchBody(), syncBuf);

  return 0;
}

void RewriteObjC::WarnAboutReturnGotoStmts(Stmt *S)
{
  // Perform a bottom up traversal of all children.
  for (Stmt::child_iterator CI = S->child_begin(), E = S->child_end();
       CI != E; ++CI)
    if (*CI)
      WarnAboutReturnGotoStmts(*CI);

  if (isa<ReturnStmt>(S) || isa<GotoStmt>(S)) {
    Diags.Report(Context->getFullLoc(S->getLocStart()),
                 TryFinallyContainsReturnDiag);
  }
  return;
}

void RewriteObjC::HasReturnStmts(Stmt *S, bool &hasReturns) 
{  
  // Perform a bottom up traversal of all children.
  for (Stmt::child_iterator CI = S->child_begin(), E = S->child_end();
        CI != E; ++CI)
   if (*CI)
     HasReturnStmts(*CI, hasReturns);

 if (isa<ReturnStmt>(S))
   hasReturns = true;
 return;
}

void RewriteObjC::RewriteTryReturnStmts(Stmt *S) {
 // Perform a bottom up traversal of all children.
 for (Stmt::child_iterator CI = S->child_begin(), E = S->child_end();
      CI != E; ++CI)
   if (*CI) {
     RewriteTryReturnStmts(*CI);
   }
 if (isa<ReturnStmt>(S)) {
   SourceLocation startLoc = S->getLocStart();
   const char *startBuf = SM->getCharacterData(startLoc);

   const char *semiBuf = strchr(startBuf, ';');
   assert((*semiBuf == ';') && "RewriteTryReturnStmts: can't find ';'");
   SourceLocation onePastSemiLoc = startLoc.getFileLocWithOffset(semiBuf-startBuf+1);

   std::string buf;
   buf = "{ objc_exception_try_exit(&_stack); return";
   
   ReplaceText(startLoc, 6, buf.c_str(), buf.size());
   InsertText(onePastSemiLoc, "}", 1);
 }
 return;
}

void RewriteObjC::RewriteSyncReturnStmts(Stmt *S, std::string syncExitBuf) {
  // Perform a bottom up traversal of all children.
  for (Stmt::child_iterator CI = S->child_begin(), E = S->child_end();
       CI != E; ++CI)
    if (*CI) {
      RewriteSyncReturnStmts(*CI, syncExitBuf);
    }
  if (isa<ReturnStmt>(S)) {
    SourceLocation startLoc = S->getLocStart();
    const char *startBuf = SM->getCharacterData(startLoc);

    const char *semiBuf = strchr(startBuf, ';');
    assert((*semiBuf == ';') && "RewriteSyncReturnStmts: can't find ';'");
    SourceLocation onePastSemiLoc = startLoc.getFileLocWithOffset(semiBuf-startBuf+1);

    std::string buf;
    buf = "{ objc_exception_try_exit(&_stack);";
    buf += syncExitBuf;
    buf += " return";
    
    ReplaceText(startLoc, 6, buf.c_str(), buf.size());
    InsertText(onePastSemiLoc, "}", 1);
  }
  return;
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
    ParmVarDecl *catchDecl = catchList->getCatchParamDecl();

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
      Rewrite.ReplaceText(startLoc, bodyBuf-startBuf+1, buf);
    } else if (catchDecl) {
      QualType t = catchDecl->getType();
      if (t == Context->getObjCIdType()) {
        buf += "1) { ";
        ReplaceText(startLoc, lParenLoc-startBuf+1, buf.c_str(), buf.size());
        sawIdTypedCatch = true;
      } else if (t->isObjCObjectPointerType()) {
        QualType InterfaceTy = t->getPointeeType();
        const ObjCInterfaceType *cls = // Should be a pointer to a class.
          InterfaceTy->getAs<ObjCInterfaceType>();
        if (cls) {
          buf += "objc_exception_match((struct objc_class *)objc_getClass(\"";
          buf += cls->getDecl()->getNameAsString();
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
    } else {
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

    // Now check for any return/continue/go statements within the @try.
    WarnAboutReturnGotoStmts(S->getTryBody());
  } else { /* no finally clause - make sure we synthesize an implicit one */
    buf = "{ /* implicit finally clause */\n";
    buf += " if (!_rethrow) objc_exception_try_exit(&_stack);\n";
    buf += " if (_rethrow) objc_exception_throw(_rethrow);\n";
    buf += "}";
    ReplaceText(lastCurlyLoc, 1, buf.c_str(), buf.size());
    
    // Now check for any return/continue/go statements within the @try.
    // The implicit finally clause won't called if the @try contains any
    // jump statements.
    bool hasReturns = false;
    HasReturnStmts(S->getTryBody(), hasReturns);
    if (hasReturns)
      RewriteTryReturnStmts(S->getTryBody());
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
  Expr *Replacement = StringLiteral::Create(*Context,StrEncoding.c_str(),
                                            StrEncoding.length(), false,StrType,
                                            SourceLocation());
  ReplaceStmt(Exp, Replacement);

  // Replace this subexpr in the parent.
  // delete Exp; leak for now, see RewritePropertySetter() usage for more info.
  return Replacement;
}

Stmt *RewriteObjC::RewriteAtSelector(ObjCSelectorExpr *Exp) {
  if (!SelGetUidFunctionDecl)
    SynthSelGetUidFunctionDecl();
  assert(SelGetUidFunctionDecl && "Can't find sel_registerName() decl");
  // Create a call to sel_registerName("selName").
  llvm::SmallVector<Expr*, 8> SelExprs;
  QualType argType = Context->getPointerType(Context->CharTy);
  SelExprs.push_back(StringLiteral::Create(*Context,
                                       Exp->getSelector().getAsString().c_str(),
                                       Exp->getSelector().getAsString().size(),
                                       false, argType, SourceLocation()));
  CallExpr *SelExp = SynthesizeCallToFunctionDecl(SelGetUidFunctionDecl,
                                                 &SelExprs[0], SelExprs.size());
  ReplaceStmt(Exp, SelExp);
  // delete Exp; leak for now, see RewritePropertySetter() usage for more info.
  return SelExp;
}

CallExpr *RewriteObjC::SynthesizeCallToFunctionDecl(
  FunctionDecl *FD, Expr **args, unsigned nargs) {
  // Get the type, we will need to reference it in a couple spots.
  QualType msgSendType = FD->getType();

  // Create a reference to the objc_msgSend() declaration.
  DeclRefExpr *DRE = new (Context) DeclRefExpr(FD, msgSendType, SourceLocation());

  // Now, we cast the reference to a pointer to the objc_msgSend type.
  QualType pToFunc = Context->getPointerType(msgSendType);
  ImplicitCastExpr *ICE = new (Context) ImplicitCastExpr(pToFunc,
                                                         CastExpr::CK_Unknown,
                                                         DRE,
                                               /*isLvalue=*/false);

  const FunctionType *FT = msgSendType->getAs<FunctionType>();

  return new (Context) CallExpr(*Context, ICE, args, nargs, FT->getResultType(),
                                SourceLocation());
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
  return T->isObjCQualifiedIdType() || T->isObjCQualifiedInterfaceType();
}

void RewriteObjC::RewriteObjCQualifiedInterfaceTypes(Expr *E) {
  QualType Type = E->getType();
  if (needToScanForQualifiers(Type)) {
    SourceLocation Loc, EndLoc;

    if (const CStyleCastExpr *ECE = dyn_cast<CStyleCastExpr>(E)) {
      Loc = ECE->getLParenLoc();
      EndLoc = ECE->getRParenLoc();
    } else {
      Loc = E->getLocStart();
      EndLoc = E->getLocEnd();
    }
    // This will defend against trying to rewrite synthesized expressions.
    if (Loc.isInvalid() || EndLoc.isInvalid())
      return;

    const char *startBuf = SM->getCharacterData(Loc);
    const char *endBuf = SM->getCharacterData(EndLoc);
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
  const FunctionProtoType *proto = 0;
  if (VarDecl *VD = dyn_cast<VarDecl>(Dcl)) {
    Loc = VD->getLocation();
    Type = VD->getType();
  }
  else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(Dcl)) {
    Loc = FD->getLocation();
    // Check for ObjC 'id' and class types that have been adorned with protocol
    // information (id<p>, C<p>*). The protocol references need to be rewritten!
    const FunctionType *funcType = FD->getType()->getAs<FunctionType>();
    assert(funcType && "missing function type");
    proto = dyn_cast<FunctionProtoType>(funcType);
    if (!proto)
      return;
    Type = proto->getResultType();
  }
  else if (FieldDecl *FD = dyn_cast<FieldDecl>(Dcl)) {
    Loc = FD->getLocation();
    Type = FD->getType();
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
  ArgTys.push_back(Context->getPointerType(Context->CharTy.withConst()));
  QualType getFuncType = Context->getFunctionType(Context->getObjCSelType(),
                                                   &ArgTys[0], ArgTys.size(),
                                                   false /*isVariadic*/, 0);
  SelGetUidFunctionDecl = FunctionDecl::Create(*Context, TUDecl,
                                           SourceLocation(),
                                           SelGetUidIdent, getFuncType, 0,
                                           FunctionDecl::Extern, false);
}

void RewriteObjC::RewriteFunctionDecl(FunctionDecl *FD) {
  // declared in <objc/objc.h>
  if (FD->getIdentifier() &&
      strcmp(FD->getNameAsCString(), "sel_registerName") == 0) {
    SelGetUidFunctionDecl = FD;
    return;
  }
  RewriteObjCQualifiedInterfaceTypes(FD);
}

void RewriteObjC::RewriteBlockLiteralFunctionDecl(FunctionDecl *FD) {
  SourceLocation FunLocStart = FD->getTypeSpecStartLoc();
  const FunctionType *funcType = FD->getType()->getAs<FunctionType>();
  const FunctionProtoType *proto = dyn_cast<FunctionProtoType>(funcType);
  if (!proto)
    return;
  QualType Type = proto->getResultType();
  std::string FdStr = Type.getAsString();
  FdStr += " ";
  FdStr += FD->getNameAsCString();
  FdStr +=  "(";
  unsigned numArgs = proto->getNumArgs();
  for (unsigned i = 0; i < numArgs; i++) {
    QualType ArgType = proto->getArgType(i);
    FdStr += ArgType.getAsString();
    
    if (i+1 < numArgs)
      FdStr += ", ";
  }
  FdStr +=  ");\n";
  InsertText(FunLocStart, FdStr.c_str(), FdStr.size());
  CurFunctionDeclToDeclareForBlock = 0;
}

// SynthSuperContructorFunctionDecl - id objc_super(id obj, id super);
void RewriteObjC::SynthSuperContructorFunctionDecl() {
  if (SuperContructorFunctionDecl)
    return;
  IdentifierInfo *msgSendIdent = &Context->Idents.get("__rw_objc_super");
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
                                         msgSendIdent, msgSendType, 0,
                                         FunctionDecl::Extern, false);
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
                                         msgSendIdent, msgSendType, 0,
                                         FunctionDecl::Extern, false);
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
                                              msgSendIdent, msgSendType, 0,
                                              FunctionDecl::Extern, false);
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
                                         msgSendIdent, msgSendType, 0,
                                         FunctionDecl::Extern, false);
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
                                              msgSendIdent, msgSendType, 0,
                                              FunctionDecl::Extern, false);
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
                                              msgSendIdent, msgSendType, 0,
                                              FunctionDecl::Extern, false);
}

// SynthGetClassFunctionDecl - id objc_getClass(const char *name);
void RewriteObjC::SynthGetClassFunctionDecl() {
  IdentifierInfo *getClassIdent = &Context->Idents.get("objc_getClass");
  llvm::SmallVector<QualType, 16> ArgTys;
  ArgTys.push_back(Context->getPointerType(Context->CharTy.withConst()));
  QualType getClassType = Context->getFunctionType(Context->getObjCIdType(),
                                                   &ArgTys[0], ArgTys.size(),
                                                   false /*isVariadic*/, 0);
  GetClassFunctionDecl = FunctionDecl::Create(*Context, TUDecl,
                                          SourceLocation(),
                                          getClassIdent, getClassType, 0,
                                          FunctionDecl::Extern, false);
}

// SynthGetMetaClassFunctionDecl - id objc_getClass(const char *name);
void RewriteObjC::SynthGetMetaClassFunctionDecl() {
  IdentifierInfo *getClassIdent = &Context->Idents.get("objc_getMetaClass");
  llvm::SmallVector<QualType, 16> ArgTys;
  ArgTys.push_back(Context->getPointerType(Context->CharTy.withConst()));
  QualType getClassType = Context->getFunctionType(Context->getObjCIdType(),
                                                   &ArgTys[0], ArgTys.size(),
                                                   false /*isVariadic*/, 0);
  GetMetaClassFunctionDecl = FunctionDecl::Create(*Context, TUDecl,
                                              SourceLocation(),
                                              getClassIdent, getClassType, 0,
                                              FunctionDecl::Extern, false);
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
  Exp->getString()->printPretty(prettyBuf, *Context, 0,
                                PrintingPolicy(LangOpts));
  Preamble += prettyBuf.str();
  Preamble += ",";
  Preamble += utostr(Exp->getString()->getByteLength()) + "};\n";

  VarDecl *NewVD = VarDecl::Create(*Context, TUDecl, SourceLocation(),
                                    &Context->Idents.get(S.c_str()), strType, 0,
                                    VarDecl::Static);
  DeclRefExpr *DRE = new (Context) DeclRefExpr(NewVD, strType, SourceLocation());
  Expr *Unop = new (Context) UnaryOperator(DRE, UnaryOperator::AddrOf,
                                 Context->getPointerType(DRE->getType()),
                                 SourceLocation());
  // cast to NSConstantString *
  CastExpr *cast = NoTypeInfoCStyleCastExpr(Context, Exp->getType(),
                                            CastExpr::CK_Unknown, Unop);
  ReplaceStmt(Exp, cast);
  // delete Exp; leak for now, see RewritePropertySetter() usage for more info.
  return cast;
}

ObjCInterfaceDecl *RewriteObjC::isSuperReceiver(Expr *recExpr) {
  // check if we are sending a message to 'super'
  if (!CurMethodDef || !CurMethodDef->isInstanceMethod()) return 0;

  if (ObjCSuperExpr *Super = dyn_cast<ObjCSuperExpr>(recExpr)) {
      const ObjCObjectPointerType *OPT =
        Super->getType()->getAs<ObjCObjectPointerType>();
      assert(OPT);
      const ObjCInterfaceType *IT = OPT->getInterfaceType();
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
    for (unsigned i = 0; i < 2; ++i) {
      SuperStructDecl->addDecl(FieldDecl::Create(*Context, SuperStructDecl,
                                                 SourceLocation(), 0,
                                                 FieldTypes[i], 0,
                                                 /*BitWidth=*/0,
                                                 /*Mutable=*/false));
    }

    SuperStructDecl->completeDefinition(*Context);
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
    for (unsigned i = 0; i < 4; ++i) {
      ConstantStringDecl->addDecl(FieldDecl::Create(*Context,
                                                    ConstantStringDecl,
                                                    SourceLocation(), 0,
                                                    FieldTypes[i], 0,
                                                    /*BitWidth=*/0,
                                                    /*Mutable=*/true));
    }

    ConstantStringDecl->completeDefinition(*Context);
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
    if (clsName->getName() == "super") {
      MsgSendFlavor = MsgSendSuperFunctionDecl;
      if (MsgSendStretFlavor)
        MsgSendStretFlavor = MsgSendSuperStretFunctionDecl;
      assert(MsgSendFlavor && "MsgSendFlavor is NULL!");

      ObjCInterfaceDecl *SuperDecl =
        CurMethodDef->getClassInterface()->getSuperClass();

      llvm::SmallVector<Expr*, 4> InitExprs;

      // set the receiver to self, the first argument to all methods.
      InitExprs.push_back(
        NoTypeInfoCStyleCastExpr(Context, Context->getObjCIdType(),
                                 CastExpr::CK_Unknown,
                     new (Context) DeclRefExpr(CurMethodDef->getSelfDecl(),
                                     Context->getObjCIdType(),
                                     SourceLocation()))
                          ); // set the 'receiver'.

      llvm::SmallVector<Expr*, 8> ClsExprs;
      QualType argType = Context->getPointerType(Context->CharTy);
      ClsExprs.push_back(StringLiteral::Create(*Context,
                                     SuperDecl->getIdentifier()->getNameStart(),
                                     SuperDecl->getIdentifier()->getLength(),
                                     false, argType, SourceLocation()));
      CallExpr *Cls = SynthesizeCallToFunctionDecl(GetMetaClassFunctionDecl,
                                                   &ClsExprs[0],
                                                   ClsExprs.size());
      // To turn off a warning, type-cast to 'id'
      InitExprs.push_back( // set 'super class', using objc_getClass().
                          NoTypeInfoCStyleCastExpr(Context,
                                                   Context->getObjCIdType(),
                                                   CastExpr::CK_Unknown, Cls));
      // struct objc_super
      QualType superType = getSuperStructType();
      Expr *SuperRep;

      if (LangOpts.Microsoft) {
        SynthSuperContructorFunctionDecl();
        // Simulate a contructor call...
        DeclRefExpr *DRE = new (Context) DeclRefExpr(SuperContructorFunctionDecl,
                                           superType, SourceLocation());
        SuperRep = new (Context) CallExpr(*Context, DRE, &InitExprs[0],
                                          InitExprs.size(),
                                          superType, SourceLocation());
        // The code for super is a little tricky to prevent collision with
        // the structure definition in the header. The rewriter has it's own
        // internal definition (__rw_objc_super) that is uses. This is why
        // we need the cast below. For example:
        // (struct objc_super *)&__rw_objc_super((id)self, (id)objc_getClass("SUPER"))
        //
        SuperRep = new (Context) UnaryOperator(SuperRep, UnaryOperator::AddrOf,
                                 Context->getPointerType(SuperRep->getType()),
                                 SourceLocation());
        SuperRep = NoTypeInfoCStyleCastExpr(Context,
                                            Context->getPointerType(superType),
                                            CastExpr::CK_Unknown, SuperRep);
      } else {
        // (struct objc_super) { <exprs from above> }
        InitListExpr *ILE = new (Context) InitListExpr(SourceLocation(),
                                             &InitExprs[0], InitExprs.size(),
                                             SourceLocation());
        TypeSourceInfo *superTInfo
          = Context->getTrivialTypeSourceInfo(superType);
        SuperRep = new (Context) CompoundLiteralExpr(SourceLocation(),
                                                     superTInfo, ILE, false);
        // struct objc_super *
        SuperRep = new (Context) UnaryOperator(SuperRep, UnaryOperator::AddrOf,
                                 Context->getPointerType(SuperRep->getType()),
                                 SourceLocation());
      }
      MsgExprs.push_back(SuperRep);
    } else {
      llvm::SmallVector<Expr*, 8> ClsExprs;
      QualType argType = Context->getPointerType(Context->CharTy);
      ClsExprs.push_back(StringLiteral::Create(*Context,
                                               clsName->getNameStart(),
                                               clsName->getLength(),
                                               false, argType,
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
        NoTypeInfoCStyleCastExpr(Context, Context->getObjCIdType(),
                                 CastExpr::CK_Unknown,
                     new (Context) DeclRefExpr(CurMethodDef->getSelfDecl(),
                                     Context->getObjCIdType(),
                                     SourceLocation()))
                          ); // set the 'receiver'.

      llvm::SmallVector<Expr*, 8> ClsExprs;
      QualType argType = Context->getPointerType(Context->CharTy);
      ClsExprs.push_back(StringLiteral::Create(*Context,
                                     SuperDecl->getIdentifier()->getNameStart(),
                                     SuperDecl->getIdentifier()->getLength(),
                                     false, argType, SourceLocation()));
      CallExpr *Cls = SynthesizeCallToFunctionDecl(GetClassFunctionDecl,
                                                   &ClsExprs[0],
                                                   ClsExprs.size());
      // To turn off a warning, type-cast to 'id'
      InitExprs.push_back(
        // set 'super class', using objc_getClass().
        NoTypeInfoCStyleCastExpr(Context, Context->getObjCIdType(),
                                 CastExpr::CK_Unknown, Cls));
      // struct objc_super
      QualType superType = getSuperStructType();
      Expr *SuperRep;

      if (LangOpts.Microsoft) {
        SynthSuperContructorFunctionDecl();
        // Simulate a contructor call...
        DeclRefExpr *DRE = new (Context) DeclRefExpr(SuperContructorFunctionDecl,
                                           superType, SourceLocation());
        SuperRep = new (Context) CallExpr(*Context, DRE, &InitExprs[0],
                                          InitExprs.size(),
                                          superType, SourceLocation());
        // The code for super is a little tricky to prevent collision with
        // the structure definition in the header. The rewriter has it's own
        // internal definition (__rw_objc_super) that is uses. This is why
        // we need the cast below. For example:
        // (struct objc_super *)&__rw_objc_super((id)self, (id)objc_getClass("SUPER"))
        //
        SuperRep = new (Context) UnaryOperator(SuperRep, UnaryOperator::AddrOf,
                                 Context->getPointerType(SuperRep->getType()),
                                 SourceLocation());
        SuperRep = NoTypeInfoCStyleCastExpr(Context,
                                 Context->getPointerType(superType),
                                 CastExpr::CK_Unknown, SuperRep);
      } else {
        // (struct objc_super) { <exprs from above> }
        InitListExpr *ILE = new (Context) InitListExpr(SourceLocation(),
                                             &InitExprs[0], InitExprs.size(),
                                             SourceLocation());
        TypeSourceInfo *superTInfo
          = Context->getTrivialTypeSourceInfo(superType);
        SuperRep = new (Context) CompoundLiteralExpr(SourceLocation(),
                                                     superTInfo, ILE, false);
      }
      MsgExprs.push_back(SuperRep);
    } else {
      // Remove all type-casts because it may contain objc-style types; e.g.
      // Foo<Proto> *.
      while (CStyleCastExpr *CE = dyn_cast<CStyleCastExpr>(recExpr))
        recExpr = CE->getSubExpr();
      recExpr = NoTypeInfoCStyleCastExpr(Context, Context->getObjCIdType(),
                                         CastExpr::CK_Unknown, recExpr);
      MsgExprs.push_back(recExpr);
    }
  }
  // Create a call to sel_registerName("selName"), it will be the 2nd argument.
  llvm::SmallVector<Expr*, 8> SelExprs;
  QualType argType = Context->getPointerType(Context->CharTy);
  SelExprs.push_back(StringLiteral::Create(*Context,
                                       Exp->getSelector().getAsString().c_str(),
                                       Exp->getSelector().getAsString().size(),
                                       false, argType, SourceLocation()));
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
      userExpr = NoTypeInfoCStyleCastExpr(Context, type, CastExpr::CK_Unknown,
                                          userExpr);
    }
    // Make id<P...> cast into an 'id' cast.
    else if (CStyleCastExpr *CE = dyn_cast<CStyleCastExpr>(userExpr)) {
      if (CE->getType()->isObjCQualifiedIdType()) {
        while ((CE = dyn_cast<CStyleCastExpr>(userExpr)))
          userExpr = CE->getSubExpr();
        userExpr = NoTypeInfoCStyleCastExpr(Context, Context->getObjCIdType(),
                                            CastExpr::CK_Unknown, userExpr);
      }
    }
    MsgExprs.push_back(userExpr);
    // We've transferred the ownership to MsgExprs. For now, we *don't* null
    // out the argument in the original expression (since we aren't deleting
    // the ObjCMessageExpr). See RewritePropertySetter() usage for more info.
    //Exp->setArg(i, 0);
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
  if (ObjCMethodDecl *OMD = Exp->getMethodDecl()) {
    // Push any user argument types.
    for (ObjCMethodDecl::param_iterator PI = OMD->param_begin(),
         E = OMD->param_end(); PI != E; ++PI) {
      QualType t = (*PI)->getType()->isObjCQualifiedIdType()
                     ? Context->getObjCIdType()
                     : (*PI)->getType();
      // Make sure we convert "t (^)(...)" to "t (*)(...)".
      if (isTopLevelBlockPointerType(t)) {
        const BlockPointerType *BPT = t->getAs<BlockPointerType>();
        t = Context->getPointerType(BPT->getPointeeType());
      }
      ArgTypes.push_back(t);
    }
    returnType = OMD->getResultType()->isObjCQualifiedIdType()
                   ? Context->getObjCIdType() : OMD->getResultType();
  } else {
    returnType = Context->getObjCIdType();
  }
  // Get the type, we will need to reference it in a couple spots.
  QualType msgSendType = MsgSendFlavor->getType();

  // Create a reference to the objc_msgSend() declaration.
  DeclRefExpr *DRE = new (Context) DeclRefExpr(MsgSendFlavor, msgSendType,
                                     SourceLocation());

  // Need to cast objc_msgSend to "void *" (to workaround a GCC bandaid).
  // If we don't do this cast, we get the following bizarre warning/note:
  // xx.m:13: warning: function called through a non-compatible type
  // xx.m:13: note: if this code is reached, the program will abort
  cast = NoTypeInfoCStyleCastExpr(Context,
                                  Context->getPointerType(Context->VoidTy),
                                  CastExpr::CK_Unknown, DRE);

  // Now do the "normal" pointer to function cast.
  QualType castType = Context->getFunctionType(returnType,
    &ArgTypes[0], ArgTypes.size(),
    // If we don't have a method decl, force a variadic cast.
    Exp->getMethodDecl() ? Exp->getMethodDecl()->isVariadic() : true, 0);
  castType = Context->getPointerType(castType);
  cast = NoTypeInfoCStyleCastExpr(Context, castType, CastExpr::CK_Unknown,
                                  cast);

  // Don't forget the parens to enforce the proper binding.
  ParenExpr *PE = new (Context) ParenExpr(SourceLocation(), SourceLocation(), cast);

  const FunctionType *FT = msgSendType->getAs<FunctionType>();
  CallExpr *CE = new (Context) CallExpr(*Context, PE, &MsgExprs[0],
                                        MsgExprs.size(),
                                        FT->getResultType(), SourceLocation());
  Stmt *ReplacingStmt = CE;
  if (MsgSendStretFlavor) {
    // We have the method which returns a struct/union. Must also generate
    // call to objc_msgSend_stret and hang both varieties on a conditional
    // expression which dictate which one to envoke depending on size of
    // method's return type.

    // Create a reference to the objc_msgSend_stret() declaration.
    DeclRefExpr *STDRE = new (Context) DeclRefExpr(MsgSendStretFlavor, msgSendType,
                                         SourceLocation());
    // Need to cast objc_msgSend_stret to "void *" (see above comment).
    cast = NoTypeInfoCStyleCastExpr(Context,
                                    Context->getPointerType(Context->VoidTy),
                                    CastExpr::CK_Unknown, STDRE);
    // Now do the "normal" pointer to function cast.
    castType = Context->getFunctionType(returnType,
      &ArgTypes[0], ArgTypes.size(),
      Exp->getMethodDecl() ? Exp->getMethodDecl()->isVariadic() : false, 0);
    castType = Context->getPointerType(castType);
    cast = NoTypeInfoCStyleCastExpr(Context, castType, CastExpr::CK_Unknown,
                                    cast);

    // Don't forget the parens to enforce the proper binding.
    PE = new (Context) ParenExpr(SourceLocation(), SourceLocation(), cast);

    FT = msgSendType->getAs<FunctionType>();
    CallExpr *STCE = new (Context) CallExpr(*Context, PE, &MsgExprs[0],
                                            MsgExprs.size(),
                                            FT->getResultType(), SourceLocation());

    // Build sizeof(returnType)
    SizeOfAlignOfExpr *sizeofExpr = new (Context) SizeOfAlignOfExpr(true,
                            Context->getTrivialTypeSourceInfo(returnType),
                                      Context->getSizeType(),
                                      SourceLocation(), SourceLocation());
    // (sizeof(returnType) <= 8 ? objc_msgSend(...) : objc_msgSend_stret(...))
    // FIXME: Value of 8 is base on ppc32/x86 ABI for the most common cases.
    // For X86 it is more complicated and some kind of target specific routine
    // is needed to decide what to do.
    unsigned IntSize =
      static_cast<unsigned>(Context->getTypeSize(Context->IntTy));
    IntegerLiteral *limit = new (Context) IntegerLiteral(llvm::APInt(IntSize, 8),
                                               Context->IntTy,
                                               SourceLocation());
    BinaryOperator *lessThanExpr = new (Context) BinaryOperator(sizeofExpr, limit,
                                                      BinaryOperator::LE,
                                                      Context->IntTy,
                                                      SourceLocation());
    // (sizeof(returnType) <= 8 ? objc_msgSend(...) : objc_msgSend_stret(...))
    ConditionalOperator *CondExpr =
      new (Context) ConditionalOperator(lessThanExpr,
                                        SourceLocation(), CE,
                                        SourceLocation(), STCE, returnType);
    ReplacingStmt = new (Context) ParenExpr(SourceLocation(), SourceLocation(), CondExpr);
  }
  // delete Exp; leak for now, see RewritePropertySetter() usage for more info.
  return ReplacingStmt;
}

Stmt *RewriteObjC::RewriteMessageExpr(ObjCMessageExpr *Exp) {
  Stmt *ReplacingStmt = SynthMessageExpr(Exp);

  // Now do the actual rewrite.
  ReplaceStmt(Exp, ReplacingStmt);

  // delete Exp; leak for now, see RewritePropertySetter() usage for more info.
  return ReplacingStmt;
}

// typedef struct objc_object Protocol;
QualType RewriteObjC::getProtocolType() {
  if (!ProtocolTypeDecl) {
    TypeSourceInfo *TInfo
      = Context->getTrivialTypeSourceInfo(Context->getObjCIdType());
    ProtocolTypeDecl = TypedefDecl::Create(*Context, TUDecl,
                                           SourceLocation(),
                                           &Context->Idents.get("Protocol"),
                                           TInfo);
  }
  return Context->getTypeDeclType(ProtocolTypeDecl);
}

/// RewriteObjCProtocolExpr - Rewrite a protocol expression into
/// a synthesized/forward data reference (to the protocol's metadata).
/// The forward references (and metadata) are generated in
/// RewriteObjC::HandleTranslationUnit().
Stmt *RewriteObjC::RewriteObjCProtocolExpr(ObjCProtocolExpr *Exp) {
  std::string Name = "_OBJC_PROTOCOL_" + Exp->getProtocol()->getNameAsString();
  IdentifierInfo *ID = &Context->Idents.get(Name);
  VarDecl *VD = VarDecl::Create(*Context, TUDecl, SourceLocation(),
                                ID, getProtocolType(), 0, VarDecl::Extern);
  DeclRefExpr *DRE = new (Context) DeclRefExpr(VD, getProtocolType(), SourceLocation());
  Expr *DerefExpr = new (Context) UnaryOperator(DRE, UnaryOperator::AddrOf,
                             Context->getPointerType(DRE->getType()),
                             SourceLocation());
  CastExpr *castExpr = NoTypeInfoCStyleCastExpr(Context, DerefExpr->getType(),
                                                CastExpr::CK_Unknown,
                                                DerefExpr);
  ReplaceStmt(Exp, castExpr);
  ProtocolExprDecls.insert(Exp->getProtocol());
  // delete Exp; leak for now, see RewritePropertySetter() usage for more info.
  return castExpr;

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
  assert(CDecl->getNameAsCString() &&
         "Name missing in SynthesizeObjCInternalStruct");
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
    endBuf += Lexer::MeasureTokenLength(LocEnd, *SM, LangOpts);
    ReplaceText(LocStart, endBuf-startBuf, Result.c_str(), Result.size());
    return;
  }

  // FIXME: This has potential of causing problem. If
  // SynthesizeObjCInternalStruct is ever called recursively.
  Result += "\nstruct ";
  Result += CDecl->getNameAsString();
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
      endHeader += Lexer::MeasureTokenLength(L, *SM, LangOpts);

      if (CDecl->protocol_begin() != CDecl->protocol_end()) {
        // advance to the end of the referenced protocols.
        while (endHeader < cursor && *endHeader != '>') endHeader++;
        endHeader++;
      }
      // rewrite the original header
      ReplaceText(LocStart, endHeader-startBuf, Result.c_str(), Result.size());
    } else {
      // rewrite the original header *without* disturbing the '{'
      ReplaceText(LocStart, cursor-startBuf, Result.c_str(), Result.size());
    }
    if (RCDecl && ObjCSynthesizedStructs.count(RCDecl)) {
      Result = "\n    struct ";
      Result += RCDecl->getNameAsString();
      Result += "_IMPL ";
      Result += RCDecl->getNameAsString();
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
      } else if (*cursor == '^') { // rewrite block specifier.
        SourceLocation caretLoc = LocStart.getFileLocWithOffset(cursor-startBuf);
        ReplaceText(caretLoc, 1, "*", 1);
      }
      cursor++;
    }
    // Don't forget to add a ';'!!
    InsertText(LocEnd.getFileLocWithOffset(1), ";", 1);
  } else { // we don't have any instance variables - insert super struct.
    endBuf += Lexer::MeasureTokenLength(LocEnd, *SM, LangOpts);
    Result += " {\n    struct ";
    Result += RCDecl->getNameAsString();
    Result += "_IMPL ";
    Result += RCDecl->getNameAsString();
    Result += "_IVARS;\n};\n";
    ReplaceText(LocStart, endBuf-startBuf, Result.c_str(), Result.size());
  }
  // Mark this struct as having been generated.
  if (!ObjCSynthesizedStructs.insert(CDecl))
    assert(false && "struct already synthesize- SynthesizeObjCInternalStruct");
}

// RewriteObjCMethodsMetaData - Rewrite methods metadata for instance or
/// class methods.
template<typename MethodIterator>
void RewriteObjC::RewriteObjCMethodsMetaData(MethodIterator MethodBegin,
                                             MethodIterator MethodEnd,
                                             bool IsInstanceMethod,
                                             const char *prefix,
                                             const char *ClassName,
                                             std::string &Result) {
  if (MethodBegin == MethodEnd) return;

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
  unsigned NumMethods = std::distance(MethodBegin, MethodEnd);
  Result += "\nstatic struct {\n";
  Result += "\tstruct _objc_method_list *next_method;\n";
  Result += "\tint method_count;\n";
  Result += "\tstruct _objc_method method_list[";
  Result += utostr(NumMethods);
  Result += "];\n} _OBJC_";
  Result += prefix;
  Result += IsInstanceMethod ? "INSTANCE" : "CLASS";
  Result += "_METHODS_";
  Result += ClassName;
  Result += " __attribute__ ((used, section (\"__OBJC, __";
  Result += IsInstanceMethod ? "inst" : "cls";
  Result += "_meth\")))= ";
  Result += "{\n\t0, " + utostr(NumMethods) + "\n";

  Result += "\t,{{(SEL)\"";
  Result += (*MethodBegin)->getSelector().getAsString().c_str();
  std::string MethodTypeString;
  Context->getObjCEncodingForMethodDecl(*MethodBegin, MethodTypeString);
  Result += "\", \"";
  Result += MethodTypeString;
  Result += "\", (void *)";
  Result += MethodInternalNames[*MethodBegin];
  Result += "}\n";
  for (++MethodBegin; MethodBegin != MethodEnd; ++MethodBegin) {
    Result += "\t  ,{(SEL)\"";
    Result += (*MethodBegin)->getSelector().getAsString().c_str();
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

/// RewriteObjCProtocolMetaData - Rewrite protocols meta-data.
void RewriteObjC::
RewriteObjCProtocolMetaData(ObjCProtocolDecl *PDecl, const char *prefix,
                            const char *ClassName, std::string &Result) {
  static bool objc_protocol_methods = false;

  // Output struct protocol_methods holder of method selector and type.
  if (!objc_protocol_methods && !PDecl->isForwardDecl()) {
    /* struct protocol_methods {
     SEL _cmd;
     char *method_types;
     }
     */
    Result += "\nstruct _protocol_methods {\n";
    Result += "\tstruct objc_selector *_cmd;\n";
    Result += "\tchar *method_types;\n";
    Result += "};\n";

    objc_protocol_methods = true;
  }
  // Do not synthesize the protocol more than once.
  if (ObjCSynthesizedProtocols.count(PDecl))
    return;

    if (PDecl->instmeth_begin() != PDecl->instmeth_end()) {
      unsigned NumMethods = std::distance(PDecl->instmeth_begin(),
                                          PDecl->instmeth_end());
    /* struct _objc_protocol_method_list {
     int protocol_method_count;
     struct protocol_methods protocols[];
     }
     */
    Result += "\nstatic struct {\n";
    Result += "\tint protocol_method_count;\n";
    Result += "\tstruct _protocol_methods protocol_methods[";
    Result += utostr(NumMethods);
    Result += "];\n} _OBJC_PROTOCOL_INSTANCE_METHODS_";
    Result += PDecl->getNameAsString();
    Result += " __attribute__ ((used, section (\"__OBJC, __cat_inst_meth\")))= "
      "{\n\t" + utostr(NumMethods) + "\n";

    // Output instance methods declared in this protocol.
    for (ObjCProtocolDecl::instmeth_iterator
           I = PDecl->instmeth_begin(), E = PDecl->instmeth_end();
         I != E; ++I) {
      if (I == PDecl->instmeth_begin())
        Result += "\t  ,{{(struct objc_selector *)\"";
      else
        Result += "\t  ,{(struct objc_selector *)\"";
      Result += (*I)->getSelector().getAsString().c_str();
      std::string MethodTypeString;
      Context->getObjCEncodingForMethodDecl((*I), MethodTypeString);
      Result += "\", \"";
      Result += MethodTypeString;
      Result += "\"}\n";
    }
    Result += "\t }\n};\n";
  }

  // Output class methods declared in this protocol.
  unsigned NumMethods = std::distance(PDecl->classmeth_begin(),
                                      PDecl->classmeth_end());
  if (NumMethods > 0) {
    /* struct _objc_protocol_method_list {
     int protocol_method_count;
     struct protocol_methods protocols[];
     }
     */
    Result += "\nstatic struct {\n";
    Result += "\tint protocol_method_count;\n";
    Result += "\tstruct _protocol_methods protocol_methods[";
    Result += utostr(NumMethods);
    Result += "];\n} _OBJC_PROTOCOL_CLASS_METHODS_";
    Result += PDecl->getNameAsString();
    Result += " __attribute__ ((used, section (\"__OBJC, __cat_cls_meth\")))= "
           "{\n\t";
    Result += utostr(NumMethods);
    Result += "\n";

    // Output instance methods declared in this protocol.
    for (ObjCProtocolDecl::classmeth_iterator
           I = PDecl->classmeth_begin(), E = PDecl->classmeth_end();
         I != E; ++I) {
      if (I == PDecl->classmeth_begin())
        Result += "\t  ,{{(struct objc_selector *)\"";
      else
        Result += "\t  ,{(struct objc_selector *)\"";
      Result += (*I)->getSelector().getAsString().c_str();
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
  Result += PDecl->getNameAsString();
  Result += " __attribute__ ((used, section (\"__OBJC, __protocol\")))= "
    "{\n\t0, \"";
  Result += PDecl->getNameAsString();
  Result += "\", 0, ";
  if (PDecl->instmeth_begin() != PDecl->instmeth_end()) {
    Result += "(struct _objc_protocol_method_list *)&_OBJC_PROTOCOL_INSTANCE_METHODS_";
    Result += PDecl->getNameAsString();
    Result += ", ";
  }
  else
    Result += "0, ";
  if (PDecl->classmeth_begin() != PDecl->classmeth_end()) {
    Result += "(struct _objc_protocol_method_list *)&_OBJC_PROTOCOL_CLASS_METHODS_";
    Result += PDecl->getNameAsString();
    Result += "\n";
  }
  else
    Result += "0\n";
  Result += "};\n";

  // Mark this protocol as having been generated.
  if (!ObjCSynthesizedProtocols.insert(PDecl))
    assert(false && "protocol already synthesized");

}

void RewriteObjC::
RewriteObjCProtocolListMetaData(const ObjCList<ObjCProtocolDecl> &Protocols,
                                const char *prefix, const char *ClassName,
                                std::string &Result) {
  if (Protocols.empty()) return;

  for (unsigned i = 0; i != Protocols.size(); i++)
    RewriteObjCProtocolMetaData(Protocols[i], prefix, ClassName, Result);

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
  Result += Protocols[0]->getNameAsString();
  Result += " \n";

  for (unsigned i = 1; i != Protocols.size(); i++) {
    Result += "\t ,&_OBJC_PROTOCOL_";
    Result += Protocols[i]->getNameAsString();
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

  std::string FullCategoryName = ClassDecl->getNameAsString();
  FullCategoryName += '_';
  FullCategoryName += IDecl->getNameAsString();

  // Build _objc_method_list for class's instance methods if needed
  llvm::SmallVector<ObjCMethodDecl *, 32>
    InstanceMethods(IDecl->instmeth_begin(), IDecl->instmeth_end());

  // If any of our property implementations have associated getters or
  // setters, produce metadata for them as well.
  for (ObjCImplDecl::propimpl_iterator Prop = IDecl->propimpl_begin(),
         PropEnd = IDecl->propimpl_end();
       Prop != PropEnd; ++Prop) {
    if ((*Prop)->getPropertyImplementation() == ObjCPropertyImplDecl::Dynamic)
      continue;
    if (!(*Prop)->getPropertyIvarDecl())
      continue;
    ObjCPropertyDecl *PD = (*Prop)->getPropertyDecl();
    if (!PD)
      continue;
    if (ObjCMethodDecl *Getter = PD->getGetterMethodDecl())
      InstanceMethods.push_back(Getter);
    if (PD->isReadOnly())
      continue;
    if (ObjCMethodDecl *Setter = PD->getSetterMethodDecl())
      InstanceMethods.push_back(Setter);
  }
  RewriteObjCMethodsMetaData(InstanceMethods.begin(), InstanceMethods.end(),
                             true, "CATEGORY_", FullCategoryName.c_str(),
                             Result);

  // Build _objc_method_list for class's class methods if needed
  RewriteObjCMethodsMetaData(IDecl->classmeth_begin(), IDecl->classmeth_end(),
                             false, "CATEGORY_", FullCategoryName.c_str(),
                             Result);

  // Protocols referenced in class declaration?
  // Null CDecl is case of a category implementation with no category interface
  if (CDecl)
    RewriteObjCProtocolListMetaData(CDecl->getReferencedProtocols(), "CATEGORY",
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
  Result += IDecl->getNameAsString();
  Result += "\"\n\t, \"";
  Result += ClassDecl->getNameAsString();
  Result += "\"\n";

  if (IDecl->instmeth_begin() != IDecl->instmeth_end()) {
    Result += "\t, (struct _objc_method_list *)"
           "&_OBJC_CATEGORY_INSTANCE_METHODS_";
    Result += FullCategoryName;
    Result += "\n";
  }
  else
    Result += "\t, 0\n";
  if (IDecl->classmeth_begin() != IDecl->classmeth_end()) {
    Result += "\t, (struct _objc_method_list *)"
           "&_OBJC_CATEGORY_CLASS_METHODS_";
    Result += FullCategoryName;
    Result += "\n";
  }
  else
    Result += "\t, 0\n";

  if (CDecl && CDecl->protocol_begin() != CDecl->protocol_end()) {
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
    Result += IDecl->getNameAsString();
    if (LangOpts.Microsoft)
      Result += "_IMPL";
    Result += ", ";
    Result += ivar->getNameAsString();
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
  if (CDecl->isImplicitInterfaceDecl()) {
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
    Result += IDecl->getNameAsString();
    Result += " __attribute__ ((used, section (\"__OBJC, __instance_vars\")))= "
      "{\n\t";
    Result += utostr(NumIvars);
    Result += "\n";

    ObjCInterfaceDecl::ivar_iterator IVI, IVE;
    llvm::SmallVector<ObjCIvarDecl *, 8> IVars;
    if (!IDecl->ivar_empty()) {
      for (ObjCImplementationDecl::ivar_iterator
             IV = IDecl->ivar_begin(), IVEnd = IDecl->ivar_end();
           IV != IVEnd; ++IV)
        IVars.push_back(*IV);
      IVI = IVars.begin();
      IVE = IVars.end();
    } else {
      IVI = CDecl->ivar_begin();
      IVE = CDecl->ivar_end();
    }
    Result += "\t,{{\"";
    Result += (*IVI)->getNameAsString();
    Result += "\", \"";
    std::string TmpString, StrEncoding;
    Context->getObjCEncodingForType((*IVI)->getType(), TmpString, *IVI);
    QuoteDoublequotes(TmpString, StrEncoding);
    Result += StrEncoding;
    Result += "\", ";
    SynthesizeIvarOffsetComputation(IDecl, *IVI, Result);
    Result += "}\n";
    for (++IVI; IVI != IVE; ++IVI) {
      Result += "\t  ,{\"";
      Result += (*IVI)->getNameAsString();
      Result += "\", \"";
      std::string TmpString, StrEncoding;
      Context->getObjCEncodingForType((*IVI)->getType(), TmpString, *IVI);
      QuoteDoublequotes(TmpString, StrEncoding);
      Result += StrEncoding;
      Result += "\", ";
      SynthesizeIvarOffsetComputation(IDecl, (*IVI), Result);
      Result += "}\n";
    }

    Result += "\t }\n};\n";
  }

  // Build _objc_method_list for class's instance methods if needed
  llvm::SmallVector<ObjCMethodDecl *, 32>
    InstanceMethods(IDecl->instmeth_begin(), IDecl->instmeth_end());

  // If any of our property implementations have associated getters or
  // setters, produce metadata for them as well.
  for (ObjCImplDecl::propimpl_iterator Prop = IDecl->propimpl_begin(),
         PropEnd = IDecl->propimpl_end();
       Prop != PropEnd; ++Prop) {
    if ((*Prop)->getPropertyImplementation() == ObjCPropertyImplDecl::Dynamic)
      continue;
    if (!(*Prop)->getPropertyIvarDecl())
      continue;
    ObjCPropertyDecl *PD = (*Prop)->getPropertyDecl();
    if (!PD)
      continue;
    if (ObjCMethodDecl *Getter = PD->getGetterMethodDecl())
      InstanceMethods.push_back(Getter);
    if (PD->isReadOnly())
      continue;
    if (ObjCMethodDecl *Setter = PD->getSetterMethodDecl())
      InstanceMethods.push_back(Setter);
  }
  RewriteObjCMethodsMetaData(InstanceMethods.begin(), InstanceMethods.end(),
                             true, "", IDecl->getNameAsCString(), Result);

  // Build _objc_method_list for class's class methods if needed
  RewriteObjCMethodsMetaData(IDecl->classmeth_begin(), IDecl->classmeth_end(),
                             false, "", IDecl->getNameAsCString(), Result);

  // Protocols referenced in class declaration?
  RewriteObjCProtocolListMetaData(CDecl->getReferencedProtocols(),
                                  "CLASS", CDecl->getNameAsCString(), Result);

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
  Result += CDecl->getNameAsString();
  Result += " __attribute__ ((used, section (\"__OBJC, __meta_class\")))= "
  "{\n\t(struct _objc_class *)\"";
  Result += (RootClass ? RootClass->getNameAsString() : CDecl->getNameAsString());
  Result += "\"";

  if (SuperClass) {
    Result += ", \"";
    Result += SuperClass->getNameAsString();
    Result += "\", \"";
    Result += CDecl->getNameAsString();
    Result += "\"";
  }
  else {
    Result += ", 0, \"";
    Result += CDecl->getNameAsString();
    Result += "\"";
  }
  // Set 'ivars' field for root class to 0. ObjC1 runtime does not use it.
  // 'info' field is initialized to CLS_META(2) for metaclass
  Result += ", 0,2, sizeof(struct _objc_class), 0";
  if (IDecl->classmeth_begin() != IDecl->classmeth_end()) {
    Result += "\n\t, (struct _objc_method_list *)&_OBJC_CLASS_METHODS_";
    Result += IDecl->getNameAsString();
    Result += "\n";
  }
  else
    Result += ", 0\n";
  if (CDecl->protocol_begin() != CDecl->protocol_end()) {
    Result += "\t,0, (struct _objc_protocol_list *)&_OBJC_CLASS_PROTOCOLS_";
    Result += CDecl->getNameAsString();
    Result += ",0,0\n";
  }
  else
    Result += "\t,0,0,0,0\n";
  Result += "};\n";

  // class metadata generation.
  Result += "\nstatic struct _objc_class _OBJC_CLASS_";
  Result += CDecl->getNameAsString();
  Result += " __attribute__ ((used, section (\"__OBJC, __class\")))= "
            "{\n\t&_OBJC_METACLASS_";
  Result += CDecl->getNameAsString();
  if (SuperClass) {
    Result += ", \"";
    Result += SuperClass->getNameAsString();
    Result += "\", \"";
    Result += CDecl->getNameAsString();
    Result += "\"";
  }
  else {
    Result += ", 0, \"";
    Result += CDecl->getNameAsString();
    Result += "\"";
  }
  // 'info' field is initialized to CLS_CLASS(1) for class
  Result += ", 0,1";
  if (!ObjCSynthesizedStructs.count(CDecl))
    Result += ",0";
  else {
    // class has size. Must synthesize its size.
    Result += ",sizeof(struct ";
    Result += CDecl->getNameAsString();
    if (LangOpts.Microsoft)
      Result += "_IMPL";
    Result += ")";
  }
  if (NumIvars > 0) {
    Result += ", (struct _objc_ivar_list *)&_OBJC_INSTANCE_VARIABLES_";
    Result += CDecl->getNameAsString();
    Result += "\n\t";
  }
  else
    Result += ",0";
  if (IDecl->instmeth_begin() != IDecl->instmeth_end()) {
    Result += ", (struct _objc_method_list *)&_OBJC_INSTANCE_METHODS_";
    Result += CDecl->getNameAsString();
    Result += ", 0\n\t";
  }
  else
    Result += ",0,0";
  if (CDecl->protocol_begin() != CDecl->protocol_end()) {
    Result += ", (struct _objc_protocol_list*)&_OBJC_CLASS_PROTOCOLS_";
    Result += CDecl->getNameAsString();
    Result += ", 0,0\n";
  }
  else
    Result += ",0,0,0\n";
  Result += "};\n";
}

/// RewriteImplementations - This routine rewrites all method implementations
/// and emits meta-data.

void RewriteObjC::RewriteImplementations() {
  int ClsDefCount = ClassImplementation.size();
  int CatDefCount = CategoryImplementation.size();

  // Rewrite implemented methods
  for (int i = 0; i < ClsDefCount; i++)
    RewriteImplementationDecl(ClassImplementation[i]);

  for (int i = 0; i < CatDefCount; i++)
    RewriteImplementationDecl(CategoryImplementation[i]);
}

void RewriteObjC::SynthesizeMetaDataIntoBuffer(std::string &Result) {
  int ClsDefCount = ClassImplementation.size();
  int CatDefCount = CategoryImplementation.size();

  // This is needed for determining instance variable offsets.
  Result += "\n#define __OFFSETOFIVAR__(TYPE, MEMBER) ((long) &((TYPE *)0)->MEMBER)\n";
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
    Result += ClassImplementation[i]->getNameAsString();
    Result += "\n";
  }

  for (int i = 0; i < CatDefCount; i++) {
    Result += "\t,&_OBJC_CATEGORY_";
    Result += CategoryImplementation[i]->getClassInterface()->getNameAsString();
    Result += "_";
    Result += CategoryImplementation[i]->getNameAsString();
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
    if (ProtocolExprDecls.size()) {
      Result += "#pragma section(\".objc_protocol$B\",long,read,write)\n";
      Result += "#pragma data_seg(push, \".objc_protocol$B\")\n";
      for (llvm::SmallPtrSet<ObjCProtocolDecl *,8>::iterator I = ProtocolExprDecls.begin(),
           E = ProtocolExprDecls.end(); I != E; ++I) {
        Result += "static struct _objc_protocol *_POINTER_OBJC_PROTOCOL_";
        Result += (*I)->getNameAsString();
        Result += " = &_OBJC_PROTOCOL_";
        Result += (*I)->getNameAsString();
        Result += ";\n";
      }
      Result += "#pragma data_seg(pop)\n\n";
    }
    Result += "#pragma section(\".objc_module_info$B\",long,read,write)\n";
    Result += "#pragma data_seg(push, \".objc_module_info$B\")\n";
    Result += "static struct _objc_module *_POINTER_OBJC_MODULES = ";
    Result += "&_OBJC_MODULES;\n";
    Result += "#pragma data_seg(pop)\n\n";
  }
}

void RewriteObjC::RewriteByRefString(std::string &ResultStr, 
                                     const std::string &Name,
                                     ValueDecl *VD) {
  assert(BlockByRefDeclNo.count(VD) && 
         "RewriteByRefString: ByRef decl missing");
  ResultStr += "struct __Block_byref_" + Name + 
    "_" + utostr(BlockByRefDeclNo[VD]) ;
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

  if (isa<FunctionNoProtoType>(AFT)) {
    // No user-supplied arguments. Still need to pass in a pointer to the
    // block (to reference imported block decl refs).
    S += "(" + StructRef + " *__cself)";
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
      (*AI)->getType().getAsStringInternal(ParamStr, Context->PrintingPolicy);
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
    std::string TypeString;
    RewriteByRefString(TypeString, Name, (*I));
    TypeString += " *";
    Name = TypeString + Name;
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
    if (isTopLevelBlockPointerType((*I)->getType()))
      S += "struct __block_impl *";
    else
      (*I)->getType().getAsStringInternal(Name, Context->PrintingPolicy);
    S += Name + " = __cself->" + (*I)->getNameAsString() + "; // bound by copy\n";
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
    S += "_Block_object_assign((void*)&dst->";
    S += (*I)->getNameAsString();
    S += ", (void*)src->";
    S += (*I)->getNameAsString();
    if (BlockByRefDecls.count((*I)))
      S += ", " + utostr(BLOCK_FIELD_IS_BYREF) + "/*BLOCK_FIELD_IS_BYREF*/);";
    else
      S += ", " + utostr(BLOCK_FIELD_IS_OBJECT) + "/*BLOCK_FIELD_IS_OBJECT*/);";
  }
  S += "}\n";
  
  S += "\nstatic void __";
  S += funcName;
  S += "_block_dispose_" + utostr(i);
  S += "(" + StructRef;
  S += "*src) {";
  for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = ImportedBlockDecls.begin(),
      E = ImportedBlockDecls.end(); I != E; ++I) {
    S += "_Block_object_dispose((void*)src->";
    S += (*I)->getNameAsString();
    if (BlockByRefDecls.count((*I)))
      S += ", " + utostr(BLOCK_FIELD_IS_BYREF) + "/*BLOCK_FIELD_IS_BYREF*/);";
    else
      S += ", " + utostr(BLOCK_FIELD_IS_OBJECT) + "/*BLOCK_FIELD_IS_OBJECT*/);";
  }
  S += "}\n";
  return S;
}

std::string RewriteObjC::SynthesizeBlockImpl(BlockExpr *CE, std::string Tag, 
                                             std::string Desc) {
  std::string S = "\nstruct " + Tag;
  std::string Constructor = "  " + Tag;

  S += " {\n  struct __block_impl impl;\n";
  S += "  struct " + Desc;
  S += "* Desc;\n";

  Constructor += "(void *fp, "; // Invoke function pointer.
  Constructor += "struct " + Desc; // Descriptor pointer.
  Constructor += " *desc";

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
      if (isTopLevelBlockPointerType((*I)->getType())) {
        S += "struct __block_impl *";
        Constructor += ", void *" + ArgName;
      } else {
        (*I)->getType().getAsStringInternal(FieldName, Context->PrintingPolicy);
        (*I)->getType().getAsStringInternal(ArgName, Context->PrintingPolicy);
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
      if (isTopLevelBlockPointerType((*I)->getType())) {
        S += "struct __block_impl *";
        Constructor += ", void *" + ArgName;
      } else {
        std::string TypeString;
        RewriteByRefString(TypeString, FieldName, (*I));
        TypeString += " *";
        FieldName = TypeString + FieldName;
        ArgName = TypeString + ArgName;
        Constructor += ", " + ArgName;
      }
      S += FieldName + "; // by ref\n";
    }
    // Finish writing the constructor.
    Constructor += ", int flags=0) {\n";
    if (GlobalVarDecl)
      Constructor += "    impl.isa = &_NSConcreteGlobalBlock;\n";
    else
      Constructor += "    impl.isa = &_NSConcreteStackBlock;\n";
    Constructor += "    impl.Flags = flags;\n    impl.FuncPtr = fp;\n";

    Constructor += "    Desc = desc;\n";

    // Initialize all "by copy" arguments.
    for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByCopyDecls.begin(),
         E = BlockByCopyDecls.end(); I != E; ++I) {
      std::string Name = (*I)->getNameAsString();
      Constructor += "    ";
      if (isTopLevelBlockPointerType((*I)->getType()))
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
      if (isTopLevelBlockPointerType((*I)->getType()))
        Constructor += Name + " = (struct __block_impl *)_";
      else
        Constructor += Name + " = _";
      Constructor += Name + "->__forwarding;\n";
    }
  } else {
    // Finish writing the constructor.
    Constructor += ", int flags=0) {\n";
    if (GlobalVarDecl)
      Constructor += "    impl.isa = &_NSConcreteGlobalBlock;\n";
    else
      Constructor += "    impl.isa = &_NSConcreteStackBlock;\n";
    Constructor += "    impl.Flags = flags;\n    impl.FuncPtr = fp;\n";
    Constructor += "    Desc = desc;\n";
  }
  Constructor += "  ";
  Constructor += "}\n";
  S += Constructor;
  S += "};\n";
  return S;
}

std::string RewriteObjC::SynthesizeBlockDescriptor(std::string DescTag, 
                                                   std::string ImplTag, int i,
                                                   const char *FunName,
                                                   unsigned hasCopy) {
  std::string S = "\nstatic struct " + DescTag;
  
  S += " {\n  unsigned long reserved;\n";
  S += "  unsigned long Block_size;\n";
  if (hasCopy) {
    S += "  void (*copy)(struct ";
    S += ImplTag; S += "*, struct ";
    S += ImplTag; S += "*);\n";
    
    S += "  void (*dispose)(struct ";
    S += ImplTag; S += "*);\n";
  }
  S += "} ";

  S += DescTag + "_DATA = { 0, sizeof(struct ";
  S += ImplTag + ")";
  if (hasCopy) {
    S += ", __" + std::string(FunName) + "_block_copy_" + utostr(i);
    S += ", __" + std::string(FunName) + "_block_dispose_" + utostr(i);
  }
  S += "};\n";
  return S;
}

void RewriteObjC::SynthesizeBlockLiterals(SourceLocation FunLocStart,
                                          const char *FunName) {
  // Insert declaration for the function in which block literal is used.
  if (CurFunctionDeclToDeclareForBlock && !Blocks.empty())
    RewriteBlockLiteralFunctionDecl(CurFunctionDeclToDeclareForBlock);
  // Insert closures that were part of the function.
  for (unsigned i = 0; i < Blocks.size(); i++) {

    CollectBlockDeclRefInfo(Blocks[i]);

    std::string ImplTag = "__" + std::string(FunName) + "_block_impl_" + utostr(i);
    std::string DescTag = "__" + std::string(FunName) + "_block_desc_" + utostr(i);

    std::string CI = SynthesizeBlockImpl(Blocks[i], ImplTag, DescTag);

    InsertText(FunLocStart, CI.c_str(), CI.size());

    std::string CF = SynthesizeBlockFunc(Blocks[i], i, FunName, ImplTag);

    InsertText(FunLocStart, CF.c_str(), CF.size());

    if (ImportedBlockDecls.size()) {
      std::string HF = SynthesizeBlockHelperFuncs(Blocks[i], i, FunName, ImplTag);
      InsertText(FunLocStart, HF.c_str(), HF.size());
    }
    std::string BD = SynthesizeBlockDescriptor(DescTag, ImplTag, i, FunName,
                                               ImportedBlockDecls.size() > 0);
    InsertText(FunLocStart, BD.c_str(), BD.size());    

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
  const char *FuncName = FD->getNameAsCString();

  SynthesizeBlockLiterals(FunLocStart, FuncName);
}

void RewriteObjC::InsertBlockLiteralsWithinMethod(ObjCMethodDecl *MD) {
  //fprintf(stderr,"In InsertBlockLiteralsWitinMethod\n");
  //SourceLocation FunLocStart = MD->getLocStart();
  // FIXME: This hack works around a bug in Rewrite.InsertText().
  SourceLocation FunLocStart = MD->getLocStart().getFileLocWithOffset(-1);
  std::string FuncName = MD->getSelector().getAsString();
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

Stmt *RewriteObjC::SynthesizeBlockCall(CallExpr *Exp, const Expr *BlockExp) {
  // Navigate to relevant type information.
  const BlockPointerType *CPT = 0;

  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(BlockExp)) {
    CPT = DRE->getType()->getAs<BlockPointerType>();
  } else if (const BlockDeclRefExpr *CDRE = 
              dyn_cast<BlockDeclRefExpr>(BlockExp)) {
    CPT = CDRE->getType()->getAs<BlockPointerType>();
  } else if (const MemberExpr *MExpr = dyn_cast<MemberExpr>(BlockExp)) {
    CPT = MExpr->getType()->getAs<BlockPointerType>();
  } 
  else if (const ParenExpr *PRE = dyn_cast<ParenExpr>(BlockExp)) {
    return SynthesizeBlockCall(Exp, PRE->getSubExpr());
  }
  else if (const ImplicitCastExpr *IEXPR = dyn_cast<ImplicitCastExpr>(BlockExp)) 
    CPT = IEXPR->getType()->getAs<BlockPointerType>();
  else if (const ConditionalOperator *CEXPR = 
            dyn_cast<ConditionalOperator>(BlockExp)) {
    Expr *LHSExp = CEXPR->getLHS();
    Stmt *LHSStmt = SynthesizeBlockCall(Exp, LHSExp);
    Expr *RHSExp = CEXPR->getRHS();
    Stmt *RHSStmt = SynthesizeBlockCall(Exp, RHSExp);
    Expr *CONDExp = CEXPR->getCond();
    ConditionalOperator *CondExpr =
      new (Context) ConditionalOperator(CONDExp,
                                      SourceLocation(), cast<Expr>(LHSStmt),
                                      SourceLocation(), cast<Expr>(RHSStmt), 
                                      Exp->getType());
    return CondExpr;
  } else if (const ObjCIvarRefExpr *IRE = dyn_cast<ObjCIvarRefExpr>(BlockExp)) {
    CPT = IRE->getType()->getAs<BlockPointerType>();
  } else {
    assert(1 && "RewriteBlockClass: Bad type");
  }
  assert(CPT && "RewriteBlockClass: Bad type");
  const FunctionType *FT = CPT->getPointeeType()->getAs<FunctionType>();
  assert(FT && "RewriteBlockClass: Bad type");
  const FunctionProtoType *FTP = dyn_cast<FunctionProtoType>(FT);
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
    for (FunctionProtoType::arg_type_iterator I = FTP->arg_type_begin(),
         E = FTP->arg_type_end(); I && (I != E); ++I) {
      QualType t = *I;
      // Make sure we convert "t (^)(...)" to "t (*)(...)".
      if (isTopLevelBlockPointerType(t)) {
        const BlockPointerType *BPT = t->getAs<BlockPointerType>();
        t = Context->getPointerType(BPT->getPointeeType());
      }
      ArgTypes.push_back(t);
    }
  }
  // Now do the pointer to function cast.
  QualType PtrToFuncCastType = Context->getFunctionType(Exp->getType(),
    &ArgTypes[0], ArgTypes.size(), false/*no variadic*/, 0);

  PtrToFuncCastType = Context->getPointerType(PtrToFuncCastType);

  CastExpr *BlkCast = NoTypeInfoCStyleCastExpr(Context, PtrBlock,
                                               CastExpr::CK_Unknown,
                                               const_cast<Expr*>(BlockExp));
  // Don't forget the parens to enforce the proper binding.
  ParenExpr *PE = new (Context) ParenExpr(SourceLocation(), SourceLocation(),
                                          BlkCast);
  //PE->dump();

  FieldDecl *FD = FieldDecl::Create(*Context, 0, SourceLocation(),
                     &Context->Idents.get("FuncPtr"), Context->VoidPtrTy, 0,
                                    /*BitWidth=*/0, /*Mutable=*/true);
  MemberExpr *ME = new (Context) MemberExpr(PE, true, FD, SourceLocation(),
                                            FD->getType());

  CastExpr *FunkCast = NoTypeInfoCStyleCastExpr(Context, PtrToFuncCastType,
                                                CastExpr::CK_Unknown, ME);
  PE = new (Context) ParenExpr(SourceLocation(), SourceLocation(), FunkCast);

  llvm::SmallVector<Expr*, 8> BlkExprs;
  // Add the implicit argument.
  BlkExprs.push_back(BlkCast);
  // Add the user arguments.
  for (CallExpr::arg_iterator I = Exp->arg_begin(),
       E = Exp->arg_end(); I != E; ++I) {
    BlkExprs.push_back(*I);
  }
  CallExpr *CE = new (Context) CallExpr(*Context, PE, &BlkExprs[0],
                                        BlkExprs.size(),
                                        Exp->getType(), SourceLocation());
  return CE;
}

void RewriteObjC::RewriteBlockCall(CallExpr *Exp) {
  Stmt *BlockCall = SynthesizeBlockCall(Exp, Exp->getCallee());
  ReplaceStmt(Exp, BlockCall);
}

// We need to return the rewritten expression to handle cases where the
// BlockDeclRefExpr is embedded in another expression being rewritten.
// For example:
//
// int main() {
//    __block Foo *f;
//    __block int i;
//
//    void (^myblock)() = ^() {
//        [f test]; // f is a BlockDeclRefExpr embedded in a message (which is being rewritten).
//        i = 77;
//    };
//}
Stmt *RewriteObjC::RewriteBlockDeclRefExpr(Expr *DeclRefExp) {
  // Rewrite the byref variable into BYREFVAR->__forwarding->BYREFVAR 
  // for each DeclRefExp where BYREFVAR is name of the variable.
  ValueDecl *VD;
  bool isArrow = true;
  if (BlockDeclRefExpr *BDRE = dyn_cast<BlockDeclRefExpr>(DeclRefExp))
    VD = BDRE->getDecl();
  else {
    VD = cast<DeclRefExpr>(DeclRefExp)->getDecl();
    isArrow = false;
  }
  
  FieldDecl *FD = FieldDecl::Create(*Context, 0, SourceLocation(),
                                    &Context->Idents.get("__forwarding"), 
                                    Context->VoidPtrTy, 0,
                                    /*BitWidth=*/0, /*Mutable=*/true);
  MemberExpr *ME = new (Context) MemberExpr(DeclRefExp, isArrow,
                                            FD, SourceLocation(),
                                            FD->getType());

  const char *Name = VD->getNameAsCString();
  FD = FieldDecl::Create(*Context, 0, SourceLocation(),
                         &Context->Idents.get(Name), 
                         Context->VoidPtrTy, 0,
                         /*BitWidth=*/0, /*Mutable=*/true);
  ME = new (Context) MemberExpr(ME, true, FD, SourceLocation(),
                                DeclRefExp->getType());
  
  
  
  // Need parens to enforce precedence.
  ParenExpr *PE = new (Context) ParenExpr(SourceLocation(), SourceLocation(), 
                                          ME);
  ReplaceStmt(DeclRefExp, PE);
  return PE;
}

void RewriteObjC::RewriteCastExpr(CStyleCastExpr *CE) {
  SourceLocation LocStart = CE->getLParenLoc();
  SourceLocation LocEnd = CE->getRParenLoc();

  // Need to avoid trying to rewrite synthesized casts.
  if (LocStart.isInvalid())
    return;
  // Need to avoid trying to rewrite casts contained in macros.
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
  const FunctionProtoType *FTP;
  const PointerType *PT = QT->getAs<PointerType>();
  if (PT) {
    FTP = PT->getPointeeType()->getAs<FunctionProtoType>();
  } else {
    const BlockPointerType *BPT = QT->getAs<BlockPointerType>();
    assert(BPT && "BlockPointerTypeTakeAnyBlockArguments(): not a block pointer type");
    FTP = BPT->getPointeeType()->getAs<FunctionProtoType>();
  }
  if (FTP) {
    for (FunctionProtoType::arg_type_iterator I = FTP->arg_type_begin(),
         E = FTP->arg_type_end(); I != E; ++I)
      if (isTopLevelBlockPointerType(*I))
        return true;
  }
  return false;
}

void RewriteObjC::GetExtentOfArgList(const char *Name, const char *&LParen,
                                     const char *&RParen) {
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


/// SynthesizeByrefCopyDestroyHelper - This routine synthesizes:
/// void __Block_byref_id_object_copy(struct Block_byref_id_object *dst,
///                    struct Block_byref_id_object *src) {
///  _Block_object_assign (&_dest->object, _src->object, 
///                        BLOCK_BYREF_CALLER | BLOCK_FIELD_IS_OBJECT
///                        [|BLOCK_FIELD_IS_WEAK]) // object
///  _Block_object_assign(&_dest->object, _src->object, 
///                       BLOCK_BYREF_CALLER | BLOCK_FIELD_IS_BLOCK
///                       [|BLOCK_FIELD_IS_WEAK]) // block
/// }
/// And:
/// void __Block_byref_id_object_dispose(struct Block_byref_id_object *_src) {
///  _Block_object_dispose(_src->object, 
///                        BLOCK_BYREF_CALLER | BLOCK_FIELD_IS_OBJECT
///                        [|BLOCK_FIELD_IS_WEAK]) // object
///  _Block_object_dispose(_src->object, 
///                         BLOCK_BYREF_CALLER | BLOCK_FIELD_IS_BLOCK
///                         [|BLOCK_FIELD_IS_WEAK]) // block
/// }

std::string RewriteObjC::SynthesizeByrefCopyDestroyHelper(VarDecl *VD,
                                                          int flag) {
  std::string S;
  if (CopyDestroyCache.count(flag))
    return S;
  CopyDestroyCache.insert(flag);
  S = "static void __Block_byref_id_object_copy_";
  S += utostr(flag);
  S += "(void *dst, void *src) {\n";
  
  // offset into the object pointer is computed as:
  // void * + void* + int + int + void* + void *
  unsigned IntSize = 
  static_cast<unsigned>(Context->getTypeSize(Context->IntTy));
  unsigned VoidPtrSize = 
  static_cast<unsigned>(Context->getTypeSize(Context->VoidPtrTy));
  
  unsigned offset = (VoidPtrSize*4 + IntSize + IntSize)/8;
  S += " _Block_object_assign((char*)dst + ";
  S += utostr(offset);
  S += ", *(void * *) ((char*)src + ";
  S += utostr(offset);
  S += "), ";
  S += utostr(flag);
  S += ");\n}\n";
  
  S += "static void __Block_byref_id_object_dispose_";
  S += utostr(flag);
  S += "(void *src) {\n";
  S += " _Block_object_dispose(*(void * *) ((char*)src + ";
  S += utostr(offset);
  S += "), ";
  S += utostr(flag);
  S += ");\n}\n";
  return S;
}

/// RewriteByRefVar - For each __block typex ND variable this routine transforms
/// the declaration into:
/// struct __Block_byref_ND {
/// void *__isa;                  // NULL for everything except __weak pointers
/// struct __Block_byref_ND *__forwarding;
/// int32_t __flags;
/// int32_t __size;
/// void *__Block_byref_id_object_copy; // If variable is __block ObjC object
/// void *__Block_byref_id_object_dispose; // If variable is __block ObjC object
/// typex ND;
/// };
///
/// It then replaces declaration of ND variable with:
/// struct __Block_byref_ND ND = {__isa=0B, __forwarding=&ND, __flags=some_flag, 
///                               __size=sizeof(struct __Block_byref_ND), 
///                               ND=initializer-if-any};
///
///
void RewriteObjC::RewriteByRefVar(VarDecl *ND) {
  // Insert declaration for the function in which block literal is
  // used.
  if (CurFunctionDeclToDeclareForBlock)
    RewriteBlockLiteralFunctionDecl(CurFunctionDeclToDeclareForBlock);
  int flag = 0;
  int isa = 0;
  SourceLocation DeclLoc = ND->getTypeSpecStartLoc();
  const char *startBuf = SM->getCharacterData(DeclLoc);
  SourceLocation X = ND->getLocEnd();
  X = SM->getInstantiationLoc(X);
  const char *endBuf = SM->getCharacterData(X);
  std::string Name(ND->getNameAsString());
  std::string ByrefType;
  RewriteByRefString(ByrefType, Name, ND);
  ByrefType += " {\n";
  ByrefType += "  void *__isa;\n";
  RewriteByRefString(ByrefType, Name, ND);
  ByrefType += " *__forwarding;\n";
  ByrefType += " int __flags;\n";
  ByrefType += " int __size;\n";
  // Add void *__Block_byref_id_object_copy; 
  // void *__Block_byref_id_object_dispose; if needed.
  QualType Ty = ND->getType();
  bool HasCopyAndDispose = Context->BlockRequiresCopying(Ty);
  if (HasCopyAndDispose) {
    ByrefType += " void (*__Block_byref_id_object_copy)(void*, void*);\n";
    ByrefType += " void (*__Block_byref_id_object_dispose)(void*);\n";
  }
  
  Ty.getAsStringInternal(Name, Context->PrintingPolicy);
  ByrefType += " " + Name + ";\n";
  ByrefType += "};\n";
  // Insert this type in global scope. It is needed by helper function.
  SourceLocation FunLocStart;
  if (CurFunctionDef)
     FunLocStart = CurFunctionDef->getTypeSpecStartLoc();
  else {
    assert(CurMethodDef && "RewriteByRefVar - CurMethodDef is null");
    FunLocStart = CurMethodDef->getLocStart();
  }
  InsertText(FunLocStart, ByrefType.c_str(), ByrefType.size());
  if (Ty.isObjCGCWeak()) {
    flag |= BLOCK_FIELD_IS_WEAK;
    isa = 1;
  }
  
  if (HasCopyAndDispose) {
    flag = BLOCK_BYREF_CALLER;
    QualType Ty = ND->getType();
    // FIXME. Handle __weak variable (BLOCK_FIELD_IS_WEAK) as well.
    if (Ty->isBlockPointerType())
      flag |= BLOCK_FIELD_IS_BLOCK;
    else
      flag |= BLOCK_FIELD_IS_OBJECT;
    std::string HF = SynthesizeByrefCopyDestroyHelper(ND, flag);
    if (!HF.empty())
      InsertText(FunLocStart, HF.c_str(), HF.size());
  }
  
  // struct __Block_byref_ND ND = 
  // {0, &ND, some_flag, __size=sizeof(struct __Block_byref_ND), 
  //  initializer-if-any};
  bool hasInit = (ND->getInit() != 0);
  unsigned flags = 0;
  if (HasCopyAndDispose)
    flags |= BLOCK_HAS_COPY_DISPOSE;
  Name = ND->getNameAsString();
  ByrefType.clear();
  RewriteByRefString(ByrefType, Name, ND);
  if (!hasInit) {
    ByrefType += " " + Name + " = {(void*)";
    ByrefType += utostr(isa);
    ByrefType += ", &" + Name + ", ";
    ByrefType += utostr(flags);
    ByrefType += ", ";
    ByrefType += "sizeof(";
    RewriteByRefString(ByrefType, Name, ND);
    ByrefType += ")";
    if (HasCopyAndDispose) {
      ByrefType += ", __Block_byref_id_object_copy_";
      ByrefType += utostr(flag);
      ByrefType += ", __Block_byref_id_object_dispose_";
      ByrefType += utostr(flag);
    }
    ByrefType += "};\n";
    ReplaceText(DeclLoc, endBuf-startBuf+Name.size(), 
                ByrefType.c_str(), ByrefType.size());
  }
  else {
    SourceLocation startLoc;
    Expr *E = ND->getInit();
    if (const CStyleCastExpr *ECE = dyn_cast<CStyleCastExpr>(E))
      startLoc = ECE->getLParenLoc();
    else
      startLoc = E->getLocStart();
    startLoc = SM->getInstantiationLoc(startLoc);
    endBuf = SM->getCharacterData(startLoc);
   
    ByrefType += " " + Name;
    ByrefType += " = {(void*)";
    ByrefType += utostr(isa);
    ByrefType += ", &" + Name + ", ";
    ByrefType += utostr(flags);
    ByrefType += ", ";
    ByrefType += "sizeof(";
    RewriteByRefString(ByrefType, Name, ND);
    ByrefType += "), ";
    if (HasCopyAndDispose) {
      ByrefType += "__Block_byref_id_object_copy_";
      ByrefType += utostr(flag);
      ByrefType += ", __Block_byref_id_object_dispose_";
      ByrefType += utostr(flag);
      ByrefType += ", ";
    }
    ReplaceText(DeclLoc, endBuf-startBuf, 
                ByrefType.c_str(), ByrefType.size());
    
    // Complete the newly synthesized compound expression by inserting a right
    // curly brace before the end of the declaration.
    // FIXME: This approach avoids rewriting the initializer expression. It
    // also assumes there is only one declarator. For example, the following
    // isn't currently supported by this routine (in general):
    // 
    // double __block BYREFVAR = 1.34, BYREFVAR2 = 1.37;
    //
    const char *startBuf = SM->getCharacterData(startLoc);
    const char *semiBuf = strchr(startBuf, ';');
    assert((*semiBuf == ';') && "RewriteByRefVar: can't find ';'");
    SourceLocation semiLoc =
      startLoc.getFileLocWithOffset(semiBuf-startBuf);

    InsertText(semiLoc, "}", 1);
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
      if (BlockDeclRefs[i]->isByRef() ||
          BlockDeclRefs[i]->getType()->isObjCObjectPointerType() || 
          BlockDeclRefs[i]->getType()->isBlockPointerType()) {
        GetBlockCallExprs(BlockDeclRefs[i]);
        ImportedBlockDecls.insert(BlockDeclRefs[i]->getDecl());
      }
  }
}

FunctionDecl *RewriteObjC::SynthBlockInitFunctionDecl(const char *name) {
  IdentifierInfo *ID = &Context->Idents.get(name);
  QualType FType = Context->getFunctionNoProtoType(Context->VoidPtrTy);
  return FunctionDecl::Create(*Context, TUDecl,SourceLocation(),
                              ID, FType, 0, FunctionDecl::Extern, false,
                              false);
}

Stmt *RewriteObjC::SynthBlockInitExpr(BlockExpr *Exp) {
  Blocks.push_back(Exp);

  CollectBlockDeclRefInfo(Exp);
  std::string FuncName;

  if (CurFunctionDef)
    FuncName = CurFunctionDef->getNameAsString();
  else if (CurMethodDef) {
    FuncName = CurMethodDef->getSelector().getAsString();
    // Convert colons to underscores.
    std::string::size_type loc = 0;
    while ((loc = FuncName.find(":", loc)) != std::string::npos)
      FuncName.replace(loc, 1, "_");
  } else if (GlobalVarDecl)
    FuncName = std::string(GlobalVarDecl->getNameAsString());

  std::string BlockNumber = utostr(Blocks.size()-1);

  std::string Tag = "__" + FuncName + "_block_impl_" + BlockNumber;
  std::string Func = "__" + FuncName + "_block_func_" + BlockNumber;

  // Get a pointer to the function type so we can cast appropriately.
  QualType FType = Context->getPointerType(QualType(Exp->getFunctionType(),0));

  FunctionDecl *FD;
  Expr *NewRep;

  // Simulate a contructor call...
  FD = SynthBlockInitFunctionDecl(Tag.c_str());
  DeclRefExpr *DRE = new (Context) DeclRefExpr(FD, FType, SourceLocation());

  llvm::SmallVector<Expr*, 4> InitExprs;

  // Initialize the block function.
  FD = SynthBlockInitFunctionDecl(Func.c_str());
  DeclRefExpr *Arg = new (Context) DeclRefExpr(FD, FD->getType(),
                                               SourceLocation());
  CastExpr *castExpr = NoTypeInfoCStyleCastExpr(Context, Context->VoidPtrTy,
                                                CastExpr::CK_Unknown, Arg);
  InitExprs.push_back(castExpr);

  // Initialize the block descriptor.
  std::string DescData = "__" + FuncName + "_block_desc_" + BlockNumber + "_DATA";

  VarDecl *NewVD = VarDecl::Create(*Context, TUDecl, SourceLocation(), 
                                    &Context->Idents.get(DescData.c_str()), 
                                    Context->VoidPtrTy, 0,
                                    VarDecl::Static);
  UnaryOperator *DescRefExpr = new (Context) UnaryOperator(
                                  new (Context) DeclRefExpr(NewVD, 
                                    Context->VoidPtrTy, SourceLocation()), 
                                  UnaryOperator::AddrOf,
                                  Context->getPointerType(Context->VoidPtrTy), 
                                  SourceLocation());
  InitExprs.push_back(DescRefExpr); 
  
  // Add initializers for any closure decl refs.
  if (BlockDeclRefs.size()) {
    Expr *Exp;
    // Output all "by copy" declarations.
    for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByCopyDecls.begin(),
         E = BlockByCopyDecls.end(); I != E; ++I) {
      if (isObjCType((*I)->getType())) {
        // FIXME: Conform to ABI ([[obj retain] autorelease]).
        FD = SynthBlockInitFunctionDecl((*I)->getNameAsCString());
        Exp = new (Context) DeclRefExpr(FD, FD->getType(), SourceLocation());
      } else if (isTopLevelBlockPointerType((*I)->getType())) {
        FD = SynthBlockInitFunctionDecl((*I)->getNameAsCString());
        Arg = new (Context) DeclRefExpr(FD, FD->getType(), SourceLocation());
        Exp = NoTypeInfoCStyleCastExpr(Context, Context->VoidPtrTy,
                                       CastExpr::CK_Unknown, Arg);
      } else {
        FD = SynthBlockInitFunctionDecl((*I)->getNameAsCString());
        Exp = new (Context) DeclRefExpr(FD, FD->getType(), SourceLocation());
      }
      InitExprs.push_back(Exp);
    }
    // Output all "by ref" declarations.
    for (llvm::SmallPtrSet<ValueDecl*,8>::iterator I = BlockByRefDecls.begin(),
         E = BlockByRefDecls.end(); I != E; ++I) {
      FD = SynthBlockInitFunctionDecl((*I)->getNameAsCString());
      Exp = new (Context) DeclRefExpr(FD, FD->getType(), SourceLocation());
      Exp = new (Context) UnaryOperator(Exp, UnaryOperator::AddrOf,
                              Context->getPointerType(Exp->getType()),
                              SourceLocation());
      InitExprs.push_back(Exp);
    }
  }
  if (ImportedBlockDecls.size()) {
    // generate BLOCK_HAS_COPY_DISPOSE(have helper funcs) | BLOCK_HAS_DESCRIPTOR
    int flag = (BLOCK_HAS_COPY_DISPOSE | BLOCK_HAS_DESCRIPTOR);
    unsigned IntSize = 
      static_cast<unsigned>(Context->getTypeSize(Context->IntTy));
    Expr *FlagExp = new (Context) IntegerLiteral(llvm::APInt(IntSize, flag), 
                                             Context->IntTy, SourceLocation());
    InitExprs.push_back(FlagExp);
  }
  NewRep = new (Context) CallExpr(*Context, DRE, &InitExprs[0], InitExprs.size(),
                                  FType, SourceLocation());
  NewRep = new (Context) UnaryOperator(NewRep, UnaryOperator::AddrOf,
                             Context->getPointerType(NewRep->getType()),
                             SourceLocation());
  NewRep = NoTypeInfoCStyleCastExpr(Context, FType, CastExpr::CK_Unknown,
                                    NewRep);
  BlockDeclRefs.clear();
  BlockByRefDecls.clear();
  BlockByCopyDecls.clear();
  ImportedBlockDecls.clear();
  return NewRep;
}

//===----------------------------------------------------------------------===//
// Function Body / Expression rewriting
//===----------------------------------------------------------------------===//

// This is run as a first "pass" prior to RewriteFunctionBodyOrGlobalInitializer().
// The allows the main rewrite loop to associate all ObjCPropertyRefExprs with
// their respective BinaryOperator. Without this knowledge, we'd need to rewrite
// the ObjCPropertyRefExpr twice (once as a getter, and later as a setter).
// Since the rewriter isn't capable of rewriting rewritten code, it's important
// we get this right.
void RewriteObjC::CollectPropertySetters(Stmt *S) {
  // Perform a bottom up traversal of all children.
  for (Stmt::child_iterator CI = S->child_begin(), E = S->child_end();
       CI != E; ++CI)
    if (*CI)
      CollectPropertySetters(*CI);

  if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(S)) {
    if (BinOp->isAssignmentOp()) {
      if (ObjCPropertyRefExpr *PRE = dyn_cast<ObjCPropertyRefExpr>(BinOp->getLHS()))
        PropSetters[PRE] = BinOp;
    }
  }
}

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
    std::string Str = Rewrite.getRewrittenText(BE->getSourceRange());
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

  if (ObjCPropertyRefExpr *PropRefExpr = dyn_cast<ObjCPropertyRefExpr>(S)) {
    BinaryOperator *BinOp = PropSetters[PropRefExpr];
    if (BinOp) {
      // Because the rewriter doesn't allow us to rewrite rewritten code,
      // we need to rewrite the right hand side prior to rewriting the setter.
      DisableReplaceStmt = true;
      // Save the source range. Even if we disable the replacement, the
      // rewritten node will have been inserted into the tree. If the synthesized
      // node is at the 'end', the rewriter will fail. Consider this:
      //    self.errorHandler = handler ? handler :
      //              ^(NSURL *errorURL, NSError *error) { return (BOOL)1; };
      SourceRange SrcRange = BinOp->getSourceRange();
      Stmt *newStmt = RewriteFunctionBodyOrGlobalInitializer(BinOp->getRHS());
      DisableReplaceStmt = false;
      //
      // Unlike the main iterator, we explicily avoid changing 'BinOp'. If
      // we changed the RHS of BinOp, the rewriter would fail (since it needs
      // to see the original expression). Consider this example:
      //
      // Foo *obj1, *obj2;
      //
      // obj1.i = [obj2 rrrr];
      //
      // 'BinOp' for the previous expression looks like:
      //
      // (BinaryOperator 0x231ccf0 'int' '='
      //   (ObjCPropertyRefExpr 0x231cc70 'int' Kind=PropertyRef Property="i"
      //     (DeclRefExpr 0x231cc50 'Foo *' Var='obj1' 0x231cbb0))
      //   (ObjCMessageExpr 0x231ccb0 'int' selector=rrrr
      //     (DeclRefExpr 0x231cc90 'Foo *' Var='obj2' 0x231cbe0)))
      //
      // 'newStmt' represents the rewritten message expression. For example:
      //
      // (CallExpr 0x231d300 'id':'struct objc_object *'
      //   (ParenExpr 0x231d2e0 'int (*)(id, SEL)'
      //     (CStyleCastExpr 0x231d2c0 'int (*)(id, SEL)'
      //       (CStyleCastExpr 0x231d220 'void *'
      //         (DeclRefExpr 0x231d200 'id (id, SEL, ...)' FunctionDecl='objc_msgSend' 0x231cdc0))))
      //
      // Note that 'newStmt' is passed to RewritePropertySetter so that it
      // can be used as the setter argument. ReplaceStmt() will still 'see'
      // the original RHS (since we haven't altered BinOp).
      //
      // This implies the Rewrite* routines can no longer delete the original
      // node. As a result, we now leak the original AST nodes.
      //
      return RewritePropertySetter(BinOp, dyn_cast<Expr>(newStmt), SrcRange);
    } else {
      return RewritePropertyGetter(PropRefExpr);
    }
  }
  if (ObjCSelectorExpr *AtSelector = dyn_cast<ObjCSelectorExpr>(S))
    return RewriteAtSelector(AtSelector);

  if (ObjCStringLiteral *AtString = dyn_cast<ObjCStringLiteral>(S))
    return RewriteObjCStringLiteral(AtString);

  if (ObjCMessageExpr *MessExpr = dyn_cast<ObjCMessageExpr>(S)) {
#if 0
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
#endif
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
    // NOTE: We need to avoid rewriting the DeclStmt if it is within
    // the context of an ObjCForCollectionStmt. For example:
    //   NSArray *someArray;
    //   for (id <FooProtocol> index in someArray) ;
    // This is because RewriteObjCForCollectionStmt() does textual rewriting 
    // and it depends on the original text locations/positions.
    if (Stmts.empty() || !isa<ObjCForCollectionStmt>(Stmts.back()))
      RewriteObjCQualifiedInterfaceTypes(*DS->decl_begin());

    // Blocks rewrite rules.
    for (DeclStmt::decl_iterator DI = DS->decl_begin(), DE = DS->decl_end();
         DI != DE; ++DI) {
      Decl *SD = *DI;
      if (ValueDecl *ND = dyn_cast<ValueDecl>(SD)) {
        if (isTopLevelBlockPointerType(ND->getType()))
          RewriteBlockPointerDecl(ND);
        else if (ND->getType()->isFunctionPointerType())
          CheckFunctionPointerDecl(ND->getType(), ND);
        if (VarDecl *VD = dyn_cast<VarDecl>(SD)) 
          if (VD->hasAttr<BlocksAttr>()) {
            static unsigned uniqueByrefDeclCount = 0;
            assert(!BlockByRefDeclNo.count(ND) &&
              "RewriteFunctionBodyOrGlobalInitializer: Duplicate byref decl");
            BlockByRefDeclNo[ND] = uniqueByrefDeclCount++;
            RewriteByRefVar(VD);
          }
      }
      if (TypedefDecl *TD = dyn_cast<TypedefDecl>(SD)) {
        if (isTopLevelBlockPointerType(TD->getUnderlyingType()))
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
      return RewriteBlockDeclRefExpr(BDRE);
  }
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(S)) {
    ValueDecl *VD = DRE->getDecl(); 
    if (VD->hasAttr<BlocksAttr>())
      return RewriteBlockDeclRefExpr(DRE);
  }
  
  if (CallExpr *CE = dyn_cast<CallExpr>(S)) {
    if (CE->getCallee()->getType()->isBlockPointerType()) {
      Stmt *BlockCall = SynthesizeBlockCall(CE, CE->getCallee());
      ReplaceStmt(S, BlockCall);
      return BlockCall;
    }
  }
  if (CStyleCastExpr *CE = dyn_cast<CStyleCastExpr>(S)) {
    RewriteCastExpr(CE);
  }
#if 0
  if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(S)) {
    CastExpr *Replacement = new (Context) CastExpr(ICE->getType(), ICE->getSubExpr(), SourceLocation());
    // Get the new text.
    std::string SStr;
    llvm::raw_string_ostream Buf(SStr);
    Replacement->printPretty(Buf, *Context);
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

void RewriteObjC::RewriteRecordBody(RecordDecl *RD) {
  for (RecordDecl::field_iterator i = RD->field_begin(), 
                                  e = RD->field_end(); i != e; ++i) {
    FieldDecl *FD = *i;
    if (isTopLevelBlockPointerType(FD->getType()))
      RewriteBlockPointerDecl(FD);
    if (FD->getType()->isObjCQualifiedIdType() ||
        FD->getType()->isObjCQualifiedInterfaceType())
      RewriteObjCQualifiedInterfaceTypes(FD);
  }
}

/// HandleDeclInMainFile - This is called for each top-level decl defined in the
/// main file of the input.
void RewriteObjC::HandleDeclInMainFile(Decl *D) {
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->isOverloadedOperator())
      return;

    // Since function prototypes don't have ParmDecl's, we check the function
    // prototype. This enables us to rewrite function declarations and
    // definitions using the same code.
    RewriteBlocksInFunctionProtoType(FD->getType(), FD);

    // FIXME: If this should support Obj-C++, support CXXTryStmt
    if (CompoundStmt *Body = FD->getCompoundBody()) {
      CurFunctionDef = FD;
      CurFunctionDeclToDeclareForBlock = FD;
      CollectPropertySetters(Body);
      CurrentBody = Body;
      Body =
       cast_or_null<CompoundStmt>(RewriteFunctionBodyOrGlobalInitializer(Body));
      FD->setBody(Body);
      CurrentBody = 0;
      if (PropParentMap) {
        delete PropParentMap;
        PropParentMap = 0;
      }
      // This synthesizes and inserts the block "impl" struct, invoke function,
      // and any copy/dispose helper functions.
      InsertBlockLiteralsWithinFunction(FD);
      CurFunctionDef = 0;
      CurFunctionDeclToDeclareForBlock = 0;
    }
    return;
  }
  if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
    if (CompoundStmt *Body = MD->getCompoundBody()) {
      CurMethodDef = MD;
      CollectPropertySetters(Body);
      CurrentBody = Body;
      Body =
       cast_or_null<CompoundStmt>(RewriteFunctionBodyOrGlobalInitializer(Body));
      MD->setBody(Body);
      CurrentBody = 0;
      if (PropParentMap) {
        delete PropParentMap;
        PropParentMap = 0;
      }
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
    if (isTopLevelBlockPointerType(VD->getType()))
      RewriteBlockPointerDecl(VD);
    else if (VD->getType()->isFunctionPointerType()) {
      CheckFunctionPointerDecl(VD->getType(), VD);
      if (VD->getInit()) {
        if (CStyleCastExpr *CE = dyn_cast<CStyleCastExpr>(VD->getInit())) {
          RewriteCastExpr(CE);
        }
      }
    } else if (VD->getType()->isRecordType()) {
      RecordDecl *RD = VD->getType()->getAs<RecordType>()->getDecl();
      if (RD->isDefinition())
        RewriteRecordBody(RD);
    }
    if (VD->getInit()) {
      GlobalVarDecl = VD;
      CollectPropertySetters(VD->getInit());
      CurrentBody = VD->getInit();
      RewriteFunctionBodyOrGlobalInitializer(VD->getInit());
      CurrentBody = 0;
      if (PropParentMap) {
        delete PropParentMap;
        PropParentMap = 0;
      }
      SynthesizeBlockLiterals(VD->getTypeSpecStartLoc(),
                              VD->getNameAsCString());
      GlobalVarDecl = 0;

      // This is needed for blocks.
      if (CStyleCastExpr *CE = dyn_cast<CStyleCastExpr>(VD->getInit())) {
        RewriteCastExpr(CE);
      }
    }
    return;
  }
  if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
    if (isTopLevelBlockPointerType(TD->getUnderlyingType()))
      RewriteBlockPointerDecl(TD);
    else if (TD->getUnderlyingType()->isFunctionPointerType())
      CheckFunctionPointerDecl(TD->getUnderlyingType(), TD);
    else if (TD->getUnderlyingType()->isRecordType()) {
      RecordDecl *RD = TD->getUnderlyingType()->getAs<RecordType>()->getDecl();
      if (RD->isDefinition())
        RewriteRecordBody(RD);
    }
    return;
  }
  if (RecordDecl *RD = dyn_cast<RecordDecl>(D)) {
    if (RD->isDefinition()) 
      RewriteRecordBody(RD);
    return;
  }
  // Nothing yet.
}

void RewriteObjC::HandleTranslationUnit(ASTContext &C) {
  // Get the top-level buffer that this corresponds to.

  // Rewrite tabs if we care.
  //RewriteTabs();

  if (Diags.hasErrorOccurred())
    return;

  RewriteInclude();

  // Here's a great place to add any extra declarations that may be needed.
  // Write out meta data for each @protocol(<expr>).
  for (llvm::SmallPtrSet<ObjCProtocolDecl *,8>::iterator I = ProtocolExprDecls.begin(),
       E = ProtocolExprDecls.end(); I != E; ++I)
    RewriteObjCProtocolMetaData(*I, "", "", Preamble);

  InsertText(SM->getLocForStartOfFile(MainFileID),
             Preamble.c_str(), Preamble.size(), false);
  if (ClassImplementation.size() || CategoryImplementation.size())
    RewriteImplementations();

  // Get the buffer corresponding to MainFileID.  If we haven't changed it, then
  // we are done.
  if (const RewriteBuffer *RewriteBuf =
      Rewrite.getRewriteBufferFor(MainFileID)) {
    //printf("Changed:\n");
    *OutFile << std::string(RewriteBuf->begin(), RewriteBuf->end());
  } else {
    fprintf(stderr, "No changes\n");
  }

  if (ClassImplementation.size() || CategoryImplementation.size() ||
      ProtocolExprDecls.size()) {
    // Rewrite Objective-c meta data*
    std::string ResultStr;
    SynthesizeMetaDataIntoBuffer(ResultStr);
    // Emit metadata.
    *OutFile << ResultStr;
  }
  OutFile->flush();
}

