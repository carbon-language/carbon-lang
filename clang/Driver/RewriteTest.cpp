//===--- RewriteTest.cpp - Playground for the code rewriter ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
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
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "clang/Lex/Lexer.h"
using namespace clang;
using llvm::utostr;

namespace {
  class RewriteTest : public ASTConsumer {
    Rewriter Rewrite;
    ASTContext *Context;
    SourceManager *SM;
    unsigned MainFileID;
    SourceLocation LastIncLoc;
    llvm::SmallVector<ObjcImplementationDecl *, 8> ClassImplementation;
    llvm::SmallVector<ObjcCategoryImplDecl *, 8> CategoryImplementation;
    llvm::SmallPtrSet<ObjcInterfaceDecl*, 8> ObjcSynthesizedStructs;
    llvm::SmallPtrSet<ObjcInterfaceDecl*, 8> ObjcForwardDecls;
    llvm::DenseMap<ObjcMethodDecl*, std::string> MethodInternalNames;
    
    FunctionDecl *MsgSendFunctionDecl;
    FunctionDecl *GetClassFunctionDecl;
    FunctionDecl *SelGetUidFunctionDecl;
    FunctionDecl *CFStringFunctionDecl;
    
    // ObjC string constant support.
    FileVarDecl *ConstantStringClassReference;
    RecordDecl *NSStringRecord;
    
    static const int OBJC_ABI_VERSION =7 ;
  public:
    void Initialize(ASTContext &context, unsigned mainFileID) {
      Context = &context;
      SM = &Context->SourceMgr;
      MainFileID = mainFileID;
      MsgSendFunctionDecl = 0;
      GetClassFunctionDecl = 0;
      SelGetUidFunctionDecl = 0;
      CFStringFunctionDecl = 0;
      ConstantStringClassReference = 0;
      NSStringRecord = 0;
      Rewrite.setSourceMgr(Context->SourceMgr);
      // declaring objc_selector outside the parameter list removes a silly
      // scope related warning...
      const char *s = "struct objc_selector; struct objc_class;\n"
                      "extern struct objc_object *objc_msgSend"
                      "(struct objc_object *, struct objc_selector *, ...);\n"
                      "extern struct objc_object *objc_getClass"
                      "(const char *);\n"
                      "extern void objc_exception_throw(struct objc_object *);\n"
                      "extern void objc_exception_try_enter(void *);\n"
                      "extern void objc_exception_try_exit(void *);\n"
                      "extern struct objc_object *objc_exception_extract(void *);\n"
                      "extern int objc_exception_match"
                      "(struct objc_class *, struct objc_object *, ...);\n";

      Rewrite.InsertText(SourceLocation::getFileLoc(mainFileID, 0), 
                         s, strlen(s));
    }

    // Top Level Driver code.
    virtual void HandleTopLevelDecl(Decl *D);
    void HandleDeclInMainFile(Decl *D);
    ~RewriteTest();

    // Syntactic Rewriting.
    void RewritePrologue(SourceLocation Loc);
    void RewriteInclude(SourceLocation Loc);
    void RewriteTabs();
    void RewriteForwardClassDecl(ObjcClassDecl *Dcl);
    void RewriteInterfaceDecl(ObjcInterfaceDecl *Dcl);
    void RewriteImplementationDecl(NamedDecl *Dcl);
    void RewriteObjcMethodDecl(ObjcMethodDecl *MDecl, std::string &ResultStr);
    void RewriteCategoryDecl(ObjcCategoryDecl *Dcl);
    void RewriteProtocolDecl(ObjcProtocolDecl *Dcl);
    void RewriteForwardProtocolDecl(ObjcForwardProtocolDecl *Dcl);
    void RewriteMethodDeclarations(int nMethods, ObjcMethodDecl **Methods);
    void RewriteProperties(int nProperties, ObjcPropertyDecl **Properties);
    void RewriteFunctionDecl(FunctionDecl *FD);
    void RewriteObjcQualifiedInterfaceTypes(
        const FunctionTypeProto *proto, FunctionDecl *FD);
    bool needToScanForQualifiers(QualType T);
    
    // Expression Rewriting.
    Stmt *RewriteFunctionBodyOrGlobalInitializer(Stmt *S);
    Stmt *RewriteAtEncode(ObjCEncodeExpr *Exp);
    Stmt *RewriteAtSelector(ObjCSelectorExpr *Exp);
    Stmt *RewriteMessageExpr(ObjCMessageExpr *Exp);
    Stmt *RewriteObjCStringLiteral(ObjCStringLiteral *Exp);
    Stmt *RewriteObjcTryStmt(ObjcAtTryStmt *S);
    Stmt *RewriteObjcCatchStmt(ObjcAtCatchStmt *S);
    Stmt *RewriteObjcFinallyStmt(ObjcAtFinallyStmt *S);
    Stmt *RewriteObjcThrowStmt(ObjcAtThrowStmt *S);
    CallExpr *SynthesizeCallToFunctionDecl(FunctionDecl *FD, 
                                           Expr **args, unsigned nargs);
    void SynthMsgSendFunctionDecl();
    void SynthGetClassFunctionDecl();
    void SynthCFStringFunctionDecl();
      
    // Metadata emission.
    void RewriteObjcClassMetaData(ObjcImplementationDecl *IDecl,
                                  std::string &Result);
    
    void RewriteObjcCategoryImplDecl(ObjcCategoryImplDecl *CDecl,
                                     std::string &Result);
    
    void RewriteObjcMethodsMetaData(ObjcMethodDecl *const*Methods,
                                    int NumMethods,
                                    bool IsInstanceMethod,
                                    const char *prefix,
                                    const char *ClassName,
                                    std::string &Result);
    
    void RewriteObjcProtocolsMetaData(ObjcProtocolDecl **Protocols,
                                      int NumProtocols,
                                      const char *prefix,
                                      const char *ClassName,
                                      std::string &Result);
    void SynthesizeObjcInternalStruct(ObjcInterfaceDecl *CDecl,
                                      std::string &Result);
    void SynthesizeIvarOffsetComputation(ObjcImplementationDecl *IDecl, 
                                         ObjcIvarDecl *ivar, 
                                         std::string &Result);
    void RewriteImplementations(std::string &Result);
  };
}

ASTConsumer *clang::CreateCodeRewriterTest() { return new RewriteTest(); }

//===----------------------------------------------------------------------===//
// Top Level Driver Code
//===----------------------------------------------------------------------===//

void RewriteTest::HandleTopLevelDecl(Decl *D) {
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
  } else if (FileVarDecl *FVD = dyn_cast<FileVarDecl>(D)) {
    // declared in <Foundation/NSString.h>
    if (strcmp(FVD->getName(), "_NSConstantStringClassReference") == 0) {
      ConstantStringClassReference = FVD;
      return;
    }
  } else if (ObjcInterfaceDecl *MD = dyn_cast<ObjcInterfaceDecl>(D)) {
    RewriteInterfaceDecl(MD);
  } else if (ObjcCategoryDecl *CD = dyn_cast<ObjcCategoryDecl>(D)) {
    RewriteCategoryDecl(CD);
  } else if (ObjcProtocolDecl *PD = dyn_cast<ObjcProtocolDecl>(D)) {
    RewriteProtocolDecl(PD);
  } else if (ObjcForwardProtocolDecl *FP = 
             dyn_cast<ObjcForwardProtocolDecl>(D)){
    RewriteForwardProtocolDecl(FP);
  }
  // If we have a decl in the main file, see if we should rewrite it.
  if (SM->getDecomposedFileLoc(Loc).first == MainFileID)
    return HandleDeclInMainFile(D);

  // Otherwise, see if there is a #import in the main file that should be
  // rewritten.
  //RewriteInclude(Loc);
}

/// HandleDeclInMainFile - This is called for each top-level decl defined in the
/// main file of the input.
void RewriteTest::HandleDeclInMainFile(Decl *D) {
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    if (Stmt *Body = FD->getBody())
      FD->setBody(RewriteFunctionBodyOrGlobalInitializer(Body));
	  
  if (ObjcMethodDecl *MD = dyn_cast<ObjcMethodDecl>(D)) {
    if (Stmt *Body = MD->getBody())
      MD->setBody(RewriteFunctionBodyOrGlobalInitializer(Body));
  }
  if (ObjcImplementationDecl *CI = dyn_cast<ObjcImplementationDecl>(D))
    ClassImplementation.push_back(CI);
  else if (ObjcCategoryImplDecl *CI = dyn_cast<ObjcCategoryImplDecl>(D))
    CategoryImplementation.push_back(CI);
  else if (ObjcClassDecl *CD = dyn_cast<ObjcClassDecl>(D))
    RewriteForwardClassDecl(CD);
  else if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
    if (VD->getInit())
      RewriteFunctionBodyOrGlobalInitializer(VD->getInit());
  }
  // Nothing yet.
}

RewriteTest::~RewriteTest() {
  // Get the top-level buffer that this corresponds to.
  
  // Rewrite tabs if we care.
  //RewriteTabs();
  
  // Rewrite Objective-c meta data*
  std::string ResultStr;
  RewriteImplementations(ResultStr);
  
  // Get the buffer corresponding to MainFileID.  If we haven't changed it, then
  // we are done.
  if (const RewriteBuffer *RewriteBuf = 
      Rewrite.getRewriteBufferFor(MainFileID)) {
    //printf("Changed:\n");
    std::string S(RewriteBuf->begin(), RewriteBuf->end());
    printf("%s\n", S.c_str());
  } else {
    printf("No changes\n");
  }
  // Emit metadata.
  printf("%s", ResultStr.c_str());
}

//===----------------------------------------------------------------------===//
// Syntactic (non-AST) Rewriting Code
//===----------------------------------------------------------------------===//

void RewriteTest::RewriteInclude(SourceLocation Loc) {
  // Rip up the #include stack to the main file.
  SourceLocation IncLoc = Loc, NextLoc = Loc;
  do {
    IncLoc = Loc;
    Loc = SM->getLogicalLoc(NextLoc);
    NextLoc = SM->getIncludeLoc(Loc);
  } while (!NextLoc.isInvalid());

  // Loc is now the location of the #include filename "foo" or <foo/bar.h>.
  // IncLoc indicates the header that was included if it is useful.
  IncLoc = SM->getLogicalLoc(IncLoc);
  if (SM->getDecomposedFileLoc(Loc).first != MainFileID ||
      Loc == LastIncLoc)
    return;
  LastIncLoc = Loc;
  
  unsigned IncCol = SM->getColumnNumber(Loc);
  SourceLocation LineStartLoc = Loc.getFileLocWithOffset(-IncCol+1);

  // Replace the #import with #include.
  Rewrite.ReplaceText(LineStartLoc, IncCol-1, "#include ", strlen("#include "));
}

void RewriteTest::RewriteTabs() {
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
    Rewrite.ReplaceText(TabLoc, 1, "        ", Spaces);
  }
}


void RewriteTest::RewriteForwardClassDecl(ObjcClassDecl *ClassDecl) {
  int numDecls = ClassDecl->getNumForwardDecls();
  ObjcInterfaceDecl **ForwardDecls = ClassDecl->getForwardDecls();
  
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
    ObjcInterfaceDecl *ForwardDecl = ForwardDecls[i];
    if (ObjcForwardDecls.count(ForwardDecl))
      continue;
    typedefString += "#ifndef _REWRITER_typedef_";
    typedefString += ForwardDecl->getName();
    typedefString += "\n";
    typedefString += "#define _REWRITER_typedef_";
    typedefString += ForwardDecl->getName();
    typedefString += "\n";
    typedefString += "typedef struct objc_object ";
    typedefString += ForwardDecl->getName();
    typedefString += ";\n#endif\n";
    // Mark this typedef as having been generated.
    if (!ObjcForwardDecls.insert(ForwardDecl))
      assert(false && "typedef already output");
  }
  
  // Replace the @class with typedefs corresponding to the classes.
  Rewrite.ReplaceText(startLoc, semiPtr-startBuf+1, 
                      typedefString.c_str(), typedefString.size());
}

void RewriteTest::RewriteMethodDeclarations(int nMethods, ObjcMethodDecl **Methods) {
  for (int i = 0; i < nMethods; i++) {
    ObjcMethodDecl *Method = Methods[i];
    SourceLocation LocStart = Method->getLocStart();
    SourceLocation LocEnd = Method->getLocEnd();
    
    if (SM->getLineNumber(LocEnd) > SM->getLineNumber(LocStart)) {
      Rewrite.InsertText(LocStart, "/* ", 3);
      Rewrite.ReplaceText(LocEnd, 1, ";*/ ", 4);
    } else {
      Rewrite.InsertText(LocStart, "// ", 3);
    }
  }
}

void RewriteTest::RewriteProperties(int nProperties, ObjcPropertyDecl **Properties) 
{
  for (int i = 0; i < nProperties; i++) {
    ObjcPropertyDecl *Property = Properties[i];
    SourceLocation Loc = Property->getLocation();
    
    Rewrite.ReplaceText(Loc, 0, "// ", 3);
    
    // FIXME: handle properties that are declared across multiple lines.
  }
}

void RewriteTest::RewriteCategoryDecl(ObjcCategoryDecl *CatDecl) {
  SourceLocation LocStart = CatDecl->getLocStart();
  
  // FIXME: handle category headers that are declared across multiple lines.
  Rewrite.ReplaceText(LocStart, 0, "// ", 3);
  
  RewriteMethodDeclarations(CatDecl->getNumInstanceMethods(),
                            CatDecl->getInstanceMethods());
  RewriteMethodDeclarations(CatDecl->getNumClassMethods(),
                            CatDecl->getClassMethods());
  // Lastly, comment out the @end.
  Rewrite.ReplaceText(CatDecl->getAtEndLoc(), 0, "// ", 3);
}

void RewriteTest::RewriteProtocolDecl(ObjcProtocolDecl *PDecl) {
  std::pair<const char*, const char*> MainBuf = SM->getBufferData(MainFileID);
  
  SourceLocation LocStart = PDecl->getLocStart();
  
  // FIXME: handle protocol headers that are declared across multiple lines.
  Rewrite.ReplaceText(LocStart, 0, "// ", 3);
  
  RewriteMethodDeclarations(PDecl->getNumInstanceMethods(),
                            PDecl->getInstanceMethods());
  RewriteMethodDeclarations(PDecl->getNumClassMethods(),
                            PDecl->getClassMethods());
  // Lastly, comment out the @end.
  SourceLocation LocEnd = PDecl->getAtEndLoc();
  Rewrite.ReplaceText(LocEnd, 0, "// ", 3);

  // Must comment out @optional/@required
  const char *startBuf = SM->getCharacterData(LocStart);
  const char *endBuf = SM->getCharacterData(LocEnd);
  for (const char *p = startBuf; p < endBuf; p++) {
    if (*p == '@' && !strncmp(p+1, "optional", strlen("optional"))) {
      std::string CommentedOptional = "/* @optional */";
      SourceLocation OptionalLoc = LocStart.getFileLocWithOffset(p-startBuf);
      Rewrite.ReplaceText(OptionalLoc, strlen("@optional"),
                          CommentedOptional.c_str(), CommentedOptional.size());
      
    }
    else if (*p == '@' && !strncmp(p+1, "required", strlen("required"))) {
      std::string CommentedRequired = "/* @required */";
      SourceLocation OptionalLoc = LocStart.getFileLocWithOffset(p-startBuf);
      Rewrite.ReplaceText(OptionalLoc, strlen("@required"),
                          CommentedRequired.c_str(), CommentedRequired.size());
      
    }
  }
}

void RewriteTest::RewriteForwardProtocolDecl(ObjcForwardProtocolDecl *PDecl) {
  SourceLocation LocStart = PDecl->getLocation();
  if (LocStart.isInvalid())
    assert(false && "Invalid SourceLocation");
  // FIXME: handle forward protocol that are declared across multiple lines.
  Rewrite.ReplaceText(LocStart, 0, "// ", 3);
}

void RewriteTest::RewriteObjcMethodDecl(ObjcMethodDecl *OMD, 
                                        std::string &ResultStr) {
  static bool includeObjc = false;
  if (!includeObjc) {
    ResultStr += "#include <Objc/objc.h>\n";
    includeObjc = true;
  }
  ResultStr += "\nstatic ";
  ResultStr += OMD->getResultType().getAsString();
  ResultStr += "\n";
  
  // Unique method name
  std::string NameStr;
  
  if (OMD->isInstance())
    NameStr += "_I_";
  else
    NameStr += "_C_";
  
  NameStr += OMD->getClassInterface()->getName();
  NameStr += "_";
  
  NamedDecl *MethodContext = OMD->getMethodContext();
  if (ObjcCategoryImplDecl *CID = 
      dyn_cast<ObjcCategoryImplDecl>(MethodContext)) {
    NameStr += CID->getName();
    NameStr += "_";
  }
  // Append selector names, replacing ':' with '_'
  const char *selName = OMD->getSelector().getName().c_str();
  if (!strchr(selName, ':'))
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
    QualType selfTy = Context->getObjcInterfaceType(OMD->getClassInterface());
    selfTy = Context->getPointerType(selfTy);
    if (ObjcSynthesizedStructs.count(OMD->getClassInterface()))
      ResultStr += "struct ";
    ResultStr += selfTy.getAsString();
  }
  else
    ResultStr += Context->getObjcIdType().getAsString();
  
  ResultStr += " self, ";
  ResultStr += Context->getObjcSelType().getAsString();
  ResultStr += " _cmd";
  
  // Method arguments.
  for (int i = 0; i < OMD->getNumParams(); i++) {
    ParmVarDecl *PDecl = OMD->getParamDecl(i);
    ResultStr += ", ";
    ResultStr += PDecl->getType().getAsString();
    ResultStr += " ";
    ResultStr += PDecl->getName();
  }
  ResultStr += ")";
  
}
void RewriteTest::RewriteImplementationDecl(NamedDecl *OID) {
  ObjcImplementationDecl *IMD = dyn_cast<ObjcImplementationDecl>(OID);
  ObjcCategoryImplDecl *CID = dyn_cast<ObjcCategoryImplDecl>(OID);
  
  if (IMD)
    Rewrite.InsertText(IMD->getLocStart(), "// ", 3);
  else
    Rewrite.InsertText(CID->getLocStart(), "// ", 3);
  
  int numMethods = IMD ? IMD->getNumInstanceMethods() 
                       : CID->getNumInstanceMethods();
  
  for (int i = 0; i < numMethods; i++) {
    std::string ResultStr;
    ObjcMethodDecl *OMD;
    if (IMD)
      OMD = IMD->getInstanceMethods()[i];
    else
      OMD = CID->getInstanceMethods()[i];
    RewriteObjcMethodDecl(OMD, ResultStr);
    SourceLocation LocStart = OMD->getLocStart();
    SourceLocation LocEnd = OMD->getBody()->getLocStart();
    
    const char *startBuf = SM->getCharacterData(LocStart);
    const char *endBuf = SM->getCharacterData(LocEnd);
    Rewrite.ReplaceText(LocStart, endBuf-startBuf,
                        ResultStr.c_str(), ResultStr.size());
  }
  
  numMethods = IMD ? IMD->getNumClassMethods() : CID->getNumClassMethods();
  for (int i = 0; i < numMethods; i++) {
    std::string ResultStr;
    ObjcMethodDecl *OMD;
    if (IMD)
      OMD = IMD->getClassMethods()[i];
    else
      OMD = CID->getClassMethods()[i];
    RewriteObjcMethodDecl(OMD, ResultStr);
    SourceLocation LocStart = OMD->getLocStart();
    SourceLocation LocEnd = OMD->getBody()->getLocStart();
    
    const char *startBuf = SM->getCharacterData(LocStart);
    const char *endBuf = SM->getCharacterData(LocEnd);
    Rewrite.ReplaceText(LocStart, endBuf-startBuf,
                        ResultStr.c_str(), ResultStr.size());    
  }
  if (IMD)
    Rewrite.InsertText(IMD->getLocEnd(), "// ", 3);
  else
   Rewrite.InsertText(CID->getLocEnd(), "// ", 3); 
}

void RewriteTest::RewriteInterfaceDecl(ObjcInterfaceDecl *ClassDecl) {
  std::string ResultStr;
  if (!ObjcForwardDecls.count(ClassDecl)) {
    // we haven't seen a forward decl - generate a typedef.
    ResultStr += "#ifndef _REWRITER_typedef_";
    ResultStr += ClassDecl->getName();
    ResultStr += "\n";
    ResultStr += "#define _REWRITER_typedef_";
    ResultStr += ClassDecl->getName();
    ResultStr += "\n";
    ResultStr += "typedef struct objc_object ";
    ResultStr += ClassDecl->getName();
    ResultStr += ";\n#endif\n";
    
    // Mark this typedef as having been generated.
    ObjcForwardDecls.insert(ClassDecl);
  }
  SynthesizeObjcInternalStruct(ClassDecl, ResultStr);
    
  RewriteProperties(ClassDecl->getNumPropertyDecl(),
                    ClassDecl->getPropertyDecl());
  RewriteMethodDeclarations(ClassDecl->getNumInstanceMethods(),
                            ClassDecl->getInstanceMethods());
  RewriteMethodDeclarations(ClassDecl->getNumClassMethods(),
                            ClassDecl->getClassMethods());
  
  // Lastly, comment out the @end.
  Rewrite.ReplaceText(ClassDecl->getAtEndLoc(), 0, "// ", 3);
}

//===----------------------------------------------------------------------===//
// Function Body / Expression rewriting
//===----------------------------------------------------------------------===//

Stmt *RewriteTest::RewriteFunctionBodyOrGlobalInitializer(Stmt *S) {
  // Otherwise, just rewrite all children.
  for (Stmt::child_iterator CI = S->child_begin(), E = S->child_end();
       CI != E; ++CI)
    if (*CI) {
      Stmt *newStmt = RewriteFunctionBodyOrGlobalInitializer(*CI);
      if (newStmt) 
        *CI = newStmt;
    }
      
  // Handle specific things.
  if (ObjCEncodeExpr *AtEncode = dyn_cast<ObjCEncodeExpr>(S))
    return RewriteAtEncode(AtEncode);

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
        
    // FIXME: Missing definition of Rewrite.InsertText(clang::SourceLocation, char const*, unsigned int).
    // Rewrite.InsertText(startLoc, messString.c_str(), messString.size());
    // Tried this, but it didn't work either...
    // Rewrite.ReplaceText(startLoc, 0, messString.c_str(), messString.size());
    return RewriteMessageExpr(MessExpr);
  }
  
  if (ObjcAtTryStmt *StmtTry = dyn_cast<ObjcAtTryStmt>(S))
    return RewriteObjcTryStmt(StmtTry);

  if (ObjcAtThrowStmt *StmtThrow = dyn_cast<ObjcAtThrowStmt>(S))
    return RewriteObjcThrowStmt(StmtThrow);
    
  // Return this stmt unmodified.
  return S;
}
 
Stmt *RewriteTest::RewriteObjcTryStmt(ObjcAtTryStmt *S) {
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

  Rewrite.ReplaceText(startLoc, 4, buf.c_str(), buf.size());
  
  startLoc = S->getTryBody()->getLocEnd();
  startBuf = SM->getCharacterData(startLoc);

  assert((*startBuf == '}') && "bogus @try block");

  SourceLocation lastCurlyLoc = startLoc;
  
  startLoc = startLoc.getFileLocWithOffset(1);
  buf = " /* @catch begin */ else {\n";
  buf += " id _caught = objc_exception_extract(&_stack);\n";
  buf += " objc_exception_try_enter (&_stack);\n";
  buf += " if (_setjmp(_stack.buf))\n";
  buf += "   _rethrow = objc_exception_extract(&_stack);\n";
  buf += " else { /* @catch continue */";
  
  Rewrite.InsertText(startLoc, buf.c_str(), buf.size());
  
  bool sawIdTypedCatch = false;
  Stmt *lastCatchBody = 0;
  ObjcAtCatchStmt *catchList = S->getCatchStmts();
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

    if (DeclStmt *declStmt = dyn_cast<DeclStmt>(catchStmt)) {
      QualType t = dyn_cast<ValueDecl>(declStmt->getDecl())->getType();
      if (t == Context->getObjcIdType()) {
        buf += "1) { ";
        Rewrite.ReplaceText(startLoc, lParenLoc-startBuf+1, 
                            buf.c_str(), buf.size());
        sawIdTypedCatch = true;
      } else if (const PointerType *pType = t->getAsPointerType()) { 
        ObjcInterfaceType *cls; // Should be a pointer to a class.
        
        cls = dyn_cast<ObjcInterfaceType>(pType->getPointeeType().getTypePtr());
        if (cls) {
          buf += "objc_exception_match((struct objc_class *)objc_getClass(\"";
          buf += cls->getDecl()->getName();
          buf += "\"), (struct objc_object *)_caught)) { ";
          Rewrite.ReplaceText(startLoc, lParenLoc-startBuf+1, 
                              buf.c_str(), buf.size());
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
      Rewrite.ReplaceText(rParenLoc, bodyBuf-rParenBuf+1, 
                          buf.c_str(), buf.size());
    } else if (!isa<NullStmt>(catchStmt)) {
      assert(false && "@catch rewrite bug");
    }
    catchList = catchList->getNextCatchStmt();
  }
  // Complete the catch list...
  if (lastCatchBody) {
    SourceLocation bodyLoc = lastCatchBody->getLocEnd();
    const char *bodyBuf = SM->getCharacterData(bodyLoc);
    assert((*bodyBuf == '}') && "bogus @catch body location");
    bodyLoc = bodyLoc.getFileLocWithOffset(1);
    buf = " } } /* @catch end */\n";
  
    Rewrite.InsertText(bodyLoc, buf.c_str(), buf.size());
    
    // Set lastCurlyLoc
    lastCurlyLoc = lastCatchBody->getLocEnd();
  }
  if (ObjcAtFinallyStmt *finalStmt = S->getFinallyStmt()) {
    startLoc = finalStmt->getLocStart();
    startBuf = SM->getCharacterData(startLoc);
    assert((*startBuf == '@') && "bogus @finally start");
    
    buf = "/* @finally */";
    Rewrite.ReplaceText(startLoc, 8, buf.c_str(), buf.size());
    
    Stmt *body = finalStmt->getFinallyBody();
    SourceLocation startLoc = body->getLocStart();
    SourceLocation endLoc = body->getLocEnd();
    const char *startBuf = SM->getCharacterData(startLoc);
    const char *endBuf = SM->getCharacterData(endLoc);
    assert((*startBuf == '{') && "bogus @finally body location");
    assert((*endBuf == '}') && "bogus @finally body location");
  
    startLoc = startLoc.getFileLocWithOffset(1);
    buf = " if (!_rethrow) objc_exception_try_exit(&_stack);\n";
    Rewrite.InsertText(startLoc, buf.c_str(), buf.size());
    endLoc = endLoc.getFileLocWithOffset(-1);
    buf = " if (_rethrow) objc_exception_throw(_rethrow);\n";
    Rewrite.InsertText(endLoc, buf.c_str(), buf.size());
    
    // Set lastCurlyLoc
    lastCurlyLoc = body->getLocEnd();
  }
  // Now emit the final closing curly brace...
  lastCurlyLoc = lastCurlyLoc.getFileLocWithOffset(1);
  buf = " } /* @try scope end */\n";
  Rewrite.InsertText(lastCurlyLoc, buf.c_str(), buf.size());
  return 0;
}

Stmt *RewriteTest::RewriteObjcCatchStmt(ObjcAtCatchStmt *S) {
  return 0;
}

Stmt *RewriteTest::RewriteObjcFinallyStmt(ObjcAtFinallyStmt *S) {
  return 0;
}

// This can't be done with Rewrite.ReplaceStmt(S, ThrowExpr), since 
// the throw expression is typically a message expression that's already 
// been rewritten! (which implies the SourceLocation's are invalid).
Stmt *RewriteTest::RewriteObjcThrowStmt(ObjcAtThrowStmt *S) {
  // Get the start location and compute the semi location.
  SourceLocation startLoc = S->getLocStart();
  const char *startBuf = SM->getCharacterData(startLoc);
  
  assert((*startBuf == '@') && "bogus @throw location");

  std::string buf;
  /* void objc_exception_throw(id) __attribute__((noreturn)); */
  buf = "objc_exception_throw(";
  Rewrite.ReplaceText(startLoc, 6, buf.c_str(), buf.size());
  const char *semiBuf = strchr(startBuf, ';');
  assert((*semiBuf == ';') && "@throw: can't find ';'");
  SourceLocation semiLoc = startLoc.getFileLocWithOffset(semiBuf-startBuf);
  buf = ");";
  Rewrite.ReplaceText(semiLoc, 1, buf.c_str(), buf.size());
  return 0;
}

Stmt *RewriteTest::RewriteAtEncode(ObjCEncodeExpr *Exp) {
  // Create a new string expression.
  QualType StrType = Context->getPointerType(Context->CharTy);
  std::string StrEncoding;
  Context->getObjcEncodingForType(Exp->getEncodedType(), StrEncoding);
  Expr *Replacement = new StringLiteral(StrEncoding.c_str(),
                                        StrEncoding.length(), false, StrType, 
                                        SourceLocation(), SourceLocation());
  Rewrite.ReplaceStmt(Exp, Replacement);
  delete Exp;
  return Replacement;
}

Stmt *RewriteTest::RewriteAtSelector(ObjCSelectorExpr *Exp) {
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
  Rewrite.ReplaceStmt(Exp, SelExp);
  delete Exp;
  return SelExp;
}

CallExpr *RewriteTest::SynthesizeCallToFunctionDecl(
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

bool RewriteTest::needToScanForQualifiers(QualType T) {
  // FIXME: we don't currently represent "id <Protocol>" in the type system.
  if (T == Context->getObjcIdType())
    return true;
    
  if (const PointerType *pType = T->getAsPointerType()) {
    Type *pointeeType = pType->getPointeeType().getTypePtr();
    if (isa<ObjcQualifiedInterfaceType>(pointeeType))
      return true; // we have "Class <Protocol> *".
  }
  return false;
}

void RewriteTest::RewriteObjcQualifiedInterfaceTypes(
  const FunctionTypeProto *proto, FunctionDecl *FD) {
  
  if (needToScanForQualifiers(proto->getResultType())) {
    // Since types are unique, we need to scan the buffer.
    SourceLocation Loc = FD->getLocation();
    
    const char *endBuf = SM->getCharacterData(Loc);
    const char *startBuf = endBuf;
    while (*startBuf != ';')
      startBuf--; // scan backward (from the decl location) for return type.
    const char *startRef = 0, *endRef = 0;
    if (scanForProtocolRefs(startBuf, endBuf, startRef, endRef)) {
      // Get the locations of the startRef, endRef.
      SourceLocation LessLoc = Loc.getFileLocWithOffset(startRef-endBuf);
      SourceLocation GreaterLoc = Loc.getFileLocWithOffset(endRef-endBuf+1);
      // Comment out the protocol references.
      Rewrite.InsertText(LessLoc, "/*", 2);
      Rewrite.InsertText(GreaterLoc, "*/", 2);
    }
  }
  // Now check arguments.
  for (unsigned i = 0; i < proto->getNumArgs(); i++) {
    if (needToScanForQualifiers(proto->getArgType(i))) {
      // Since types are unique, we need to scan the buffer.
      SourceLocation Loc = FD->getLocation();
      
      const char *startBuf = SM->getCharacterData(Loc);
      const char *endBuf = startBuf;
      while (*endBuf != ';')
        endBuf++; // scan forward (from the decl location) for argument types.
      const char *startRef = 0, *endRef = 0;
      if (scanForProtocolRefs(startBuf, endBuf, startRef, endRef)) {
        // Get the locations of the startRef, endRef.
        SourceLocation LessLoc = Loc.getFileLocWithOffset(startRef-startBuf);
        SourceLocation GreaterLoc = Loc.getFileLocWithOffset(endRef-startBuf+1);
        // Comment out the protocol references.
        Rewrite.InsertText(LessLoc, "/*", 2);
        Rewrite.InsertText(GreaterLoc, "*/", 2);
      }
    } 
  }
}

void RewriteTest::RewriteFunctionDecl(FunctionDecl *FD) {
  // declared in <objc/objc.h>
  if (strcmp(FD->getName(), "sel_registerName") == 0) {
    SelGetUidFunctionDecl = FD;
    return;
  }
  // Check for ObjC 'id' and class types that have been adorned with protocol
  // information (id<p>, C<p>*). The protocol references need to be rewritten!
  const FunctionType *funcType = FD->getType()->getAsFunctionType();
  assert(funcType && "missing function type");
  if (const FunctionTypeProto *proto = dyn_cast<FunctionTypeProto>(funcType))
    RewriteObjcQualifiedInterfaceTypes(proto, FD);
}

// SynthMsgSendFunctionDecl - id objc_msgSend(id self, SEL op, ...);
void RewriteTest::SynthMsgSendFunctionDecl() {
  IdentifierInfo *msgSendIdent = &Context->Idents.get("objc_msgSend");
  llvm::SmallVector<QualType, 16> ArgTys;
  QualType argT = Context->getObjcIdType();
  assert(!argT.isNull() && "Can't find 'id' type");
  ArgTys.push_back(argT);
  argT = Context->getObjcSelType();
  assert(!argT.isNull() && "Can't find 'SEL' type");
  ArgTys.push_back(argT);
  QualType msgSendType = Context->getFunctionType(Context->getObjcIdType(),
                                                  &ArgTys[0], ArgTys.size(),
                                                  true /*isVariadic*/);
  MsgSendFunctionDecl = new FunctionDecl(SourceLocation(), 
                                         msgSendIdent, msgSendType,
                                         FunctionDecl::Extern, false, 0);
}

// SynthGetClassFunctionDecl - id objc_getClass(const char *name);
void RewriteTest::SynthGetClassFunctionDecl() {
  IdentifierInfo *getClassIdent = &Context->Idents.get("objc_getClass");
  llvm::SmallVector<QualType, 16> ArgTys;
  ArgTys.push_back(Context->getPointerType(
                     Context->CharTy.getQualifiedType(QualType::Const)));
  QualType getClassType = Context->getFunctionType(Context->getObjcIdType(),
                                                   &ArgTys[0], ArgTys.size(),
                                                   false /*isVariadic*/);
  GetClassFunctionDecl = new FunctionDecl(SourceLocation(), 
                                          getClassIdent, getClassType,
                                          FunctionDecl::Extern, false, 0);
}

// SynthCFStringFunctionDecl - id __builtin___CFStringMakeConstantString(const char *name);
void RewriteTest::SynthCFStringFunctionDecl() {
  IdentifierInfo *getClassIdent = &Context->Idents.get("__builtin___CFStringMakeConstantString");
  llvm::SmallVector<QualType, 16> ArgTys;
  ArgTys.push_back(Context->getPointerType(
                     Context->CharTy.getQualifiedType(QualType::Const)));
  QualType getClassType = Context->getFunctionType(Context->getObjcIdType(),
                                                   &ArgTys[0], ArgTys.size(),
                                                   false /*isVariadic*/);
  CFStringFunctionDecl = new FunctionDecl(SourceLocation(), 
                                          getClassIdent, getClassType,
                                          FunctionDecl::Extern, false, 0);
}

Stmt *RewriteTest::RewriteObjCStringLiteral(ObjCStringLiteral *Exp) {
#if 1
  // This rewrite is specific to GCC, which has builtin support for CFString.
  if (!CFStringFunctionDecl)
    SynthCFStringFunctionDecl();
  // Create a call to __builtin___CFStringMakeConstantString("cstr").
  llvm::SmallVector<Expr*, 8> StrExpr;
  StrExpr.push_back(Exp->getString());
  CallExpr *call = SynthesizeCallToFunctionDecl(CFStringFunctionDecl,
                                                &StrExpr[0], StrExpr.size());
  // cast to NSConstantString *
  CastExpr *cast = new CastExpr(Exp->getType(), call, SourceLocation());
  Rewrite.ReplaceStmt(Exp, cast);
  delete Exp;
  return cast;
#else
  assert(ConstantStringClassReference && "Can't find constant string reference");
  llvm::SmallVector<Expr*, 4> InitExprs;
  
  // Synthesize "(Class)&_NSConstantStringClassReference"
  DeclRefExpr *ClsRef = new DeclRefExpr(ConstantStringClassReference,
                                        ConstantStringClassReference->getType(),
                                        SourceLocation());
  QualType expType = Context->getPointerType(ClsRef->getType());
  UnaryOperator *Unop = new UnaryOperator(ClsRef, UnaryOperator::AddrOf,
                                          expType, SourceLocation());
  CastExpr *cast = new CastExpr(Context->getObjcClassType(), Unop, 
                                SourceLocation());
  InitExprs.push_back(cast); // set the 'isa'.
  InitExprs.push_back(Exp->getString()); // set "char *bytes".
  unsigned IntSize = static_cast<unsigned>(
      Context->getTypeSize(Context->IntTy, Exp->getLocStart()));
  llvm::APInt IntVal(IntSize, Exp->getString()->getByteLength());
  IntegerLiteral *len = new IntegerLiteral(IntVal, Context->IntTy, 
                                           Exp->getLocStart());
  InitExprs.push_back(len); // set "int numBytes".
  
  // struct NSConstantString
  QualType CFConstantStrType = Context->getCFConstantStringType();
  // (struct NSConstantString) { <exprs from above> }
  InitListExpr *ILE = new InitListExpr(SourceLocation(), 
                                       &InitExprs[0], InitExprs.size(), 
                                       SourceLocation());
  CompoundLiteralExpr *StrRep = new CompoundLiteralExpr(CFConstantStrType, ILE);
  // struct NSConstantString *
  expType = Context->getPointerType(StrRep->getType());
  Unop = new UnaryOperator(StrRep, UnaryOperator::AddrOf, expType, 
                           SourceLocation());
  // cast to NSConstantString *
  cast = new CastExpr(Exp->getType(), Unop, SourceLocation());
  Rewrite.ReplaceStmt(Exp, cast);
  delete Exp;
  return cast;
#endif
}

Stmt *RewriteTest::RewriteMessageExpr(ObjCMessageExpr *Exp) {
  assert(SelGetUidFunctionDecl && "Can't find sel_registerName() decl");
  if (!MsgSendFunctionDecl)
    SynthMsgSendFunctionDecl();
  if (!GetClassFunctionDecl)
    SynthGetClassFunctionDecl();

  // Synthesize a call to objc_msgSend().
  llvm::SmallVector<Expr*, 8> MsgExprs;
  IdentifierInfo *clsName = Exp->getClassName();
  
  // Derive/push the receiver/selector, 2 implicit arguments to objc_msgSend().
  if (clsName) { // class message.
    llvm::SmallVector<Expr*, 8> ClsExprs;
    QualType argType = Context->getPointerType(Context->CharTy);
    ClsExprs.push_back(new StringLiteral(clsName->getName(), 
                                         clsName->getLength(),
                                         false, argType, SourceLocation(),
                                         SourceLocation()));
    CallExpr *Cls = SynthesizeCallToFunctionDecl(GetClassFunctionDecl,
                                                 &ClsExprs[0], ClsExprs.size());
    MsgExprs.push_back(Cls);
  } else // instance message.
    MsgExprs.push_back(Exp->getReceiver());
    
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
    MsgExprs.push_back(Exp->getArg(i));
    // We've transferred the ownership to MsgExprs. Null out the argument in
    // the original expression, since we will delete it below.
    Exp->setArg(i, 0);
  }
  // Generate the funky cast.
  CastExpr *cast;
  llvm::SmallVector<QualType, 8> ArgTypes;
  QualType returnType;
  
  // Push 'id' and 'SEL', the 2 implicit arguments.
  ArgTypes.push_back(Context->getObjcIdType());
  ArgTypes.push_back(Context->getObjcSelType());
  if (ObjcMethodDecl *mDecl = Exp->getMethodDecl()) {
    // Push any user argument types.
    for (int i = 0; i < mDecl->getNumParams(); i++) {
      QualType t = mDecl->getParamDecl(i)->getType();
      if (t == Context->getObjcClassType())
        t = Context->getObjcIdType(); // Convert "Class"->"id"
      ArgTypes.push_back(t);
    }
    returnType = mDecl->getResultType();
  } else {
    returnType = Context->getObjcIdType();
  }
  // Get the type, we will need to reference it in a couple spots.
  QualType msgSendType = MsgSendFunctionDecl->getType();
  
  // Create a reference to the objc_msgSend() declaration.
  DeclRefExpr *DRE = new DeclRefExpr(MsgSendFunctionDecl, msgSendType, SourceLocation());

  // Need to cast objc_msgSend to "void *" (to workaround a GCC bandaid). 
  // If we don't do this cast, we get the following bizarre warning/note:
  // xx.m:13: warning: function called through a non-compatible type
  // xx.m:13: note: if this code is reached, the program will abort
  cast = new CastExpr(Context->getPointerType(Context->VoidTy), DRE, 
                      SourceLocation());
                                                   
  // Now do the "normal" pointer to function cast.
  QualType castType = Context->getFunctionType(returnType, 
                                               &ArgTypes[0], ArgTypes.size(),
                                               false/*FIXME:variadic*/);
  castType = Context->getPointerType(castType);
  cast = new CastExpr(castType, cast, SourceLocation());

  // Don't forget the parens to enforce the proper binding.
  ParenExpr *PE = new ParenExpr(SourceLocation(), SourceLocation(), cast);
  
  const FunctionType *FT = msgSendType->getAsFunctionType();
  CallExpr *CE = new CallExpr(PE, &MsgExprs[0], MsgExprs.size(), 
                              FT->getResultType(), SourceLocation());
  // Now do the actual rewrite.
  Rewrite.ReplaceStmt(Exp, CE);
  
  delete Exp;
  return CE;
}

/// SynthesizeObjcInternalStruct - Rewrite one internal struct corresponding to
/// an objective-c class with ivars.
void RewriteTest::SynthesizeObjcInternalStruct(ObjcInterfaceDecl *CDecl,
                                               std::string &Result) {
  assert(CDecl && "Class missing in SynthesizeObjcInternalStruct");
  assert(CDecl->getName() && "Name missing in SynthesizeObjcInternalStruct");
  // Do not synthesize more than once.
  if (ObjcSynthesizedStructs.count(CDecl))
    return;
  ObjcInterfaceDecl *RCDecl = CDecl->getSuperClass();
  if (RCDecl && !ObjcSynthesizedStructs.count(RCDecl)) {
    // Do it for the root
    SynthesizeObjcInternalStruct(RCDecl, Result);
  }
  
  int NumIvars = CDecl->getNumInstanceVariables();
  // If no ivars and no root or if its root, directly or indirectly,
  // have no ivars (thus not synthesized) then no need to synthesize this class.
  if (NumIvars <= 0 && (!RCDecl || !ObjcSynthesizedStructs.count(RCDecl)))
    return;
  
  Result += "\nstruct ";
  Result += CDecl->getName();

  SourceLocation LocStart = CDecl->getLocStart();
  SourceLocation LocEnd = CDecl->getLocEnd();
  
  const char *startBuf = SM->getCharacterData(LocStart);
  const char *endBuf = SM->getCharacterData(LocEnd);
  
  if (NumIvars > 0) {
    const char *cursor = strchr(startBuf, '{');
    assert((cursor && endBuf) 
           && "SynthesizeObjcInternalStruct - malformed @interface");
    
    // rewrite the original header *without* disturbing the '{'
    Rewrite.ReplaceText(LocStart, cursor-startBuf-1, 
                        Result.c_str(), Result.size());
    if (RCDecl && ObjcSynthesizedStructs.count(RCDecl)) {
      Result = "\n    struct ";
      Result += RCDecl->getName();
      Result += " _";
      Result += RCDecl->getName();
      Result += ";\n";
      
      // insert the super class structure definition.
      SourceLocation OnePastCurly = LocStart.getFileLocWithOffset(cursor-startBuf+1);
      Rewrite.InsertText(OnePastCurly, Result.c_str(), Result.size());
    }
    cursor++; // past '{'
    
    // Now comment out any visibility specifiers.
    while (cursor < endBuf) {
      if (*cursor == '@') {
        SourceLocation atLoc = LocStart.getFileLocWithOffset(cursor-startBuf);
        cursor = strchr(cursor, 'p');
        // FIXME: presence of @public, etc. inside comment results in
        // this transformation as well, which is still correct c-code.
        if (!strncmp(cursor, "public", strlen("public")) ||
            !strncmp(cursor, "private", strlen("private")) ||
            !strncmp(cursor, "protected", strlen("private")))
          Rewrite.InsertText(atLoc, "// ", 3);
      }
      cursor++;
    }
    // Don't forget to add a ';'!!
    Rewrite.InsertText(LocEnd.getFileLocWithOffset(1), ";", 1);
  } else { // we don't have any instance variables - insert super struct.
    endBuf += Lexer::MeasureTokenLength(LocEnd, *SM);
    Result += " {\n    struct ";
    Result += RCDecl->getName();
    Result += " _";
    Result += RCDecl->getName();
    Result += ";\n};\n";
    Rewrite.ReplaceText(LocStart, endBuf-startBuf, 
                        Result.c_str(), Result.size());
  }
  // Mark this struct as having been generated.
  if (!ObjcSynthesizedStructs.insert(CDecl))
  assert(false && "struct already synthesize- SynthesizeObjcInternalStruct");
}

// RewriteObjcMethodsMetaData - Rewrite methods metadata for instance or
/// class methods.
void RewriteTest::RewriteObjcMethodsMetaData(ObjcMethodDecl *const*Methods,
                                             int NumMethods,
                                             bool IsInstanceMethod,
                                             const char *prefix,
                                             const char *ClassName,
                                             std::string &Result) {
  static bool objc_impl_method = false;
  if (NumMethods > 0 && !objc_impl_method) {
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
    
    /* struct _objc_method_list {
     struct _objc_method_list *next_method;
     int method_count;
     struct _objc_method method_list[];
     }
     */
    Result += "\nstruct _objc_method_list {\n";
    Result += "\tstruct _objc_method_list *next_method;\n";
    Result += "\tint method_count;\n";
    Result += "\tstruct _objc_method method_list[];\n};\n";
    objc_impl_method = true;
  }
  // Build _objc_method_list for class's methods if needed
  if (NumMethods > 0) {
    Result += "\nstatic struct _objc_method_list _OBJC_";
    Result += prefix;
    Result += IsInstanceMethod ? "INSTANCE" : "CLASS";
    Result += "_METHODS_";
    Result += ClassName;
    Result += " __attribute__ ((section (\"__OBJC, __";
    Result += IsInstanceMethod ? "inst" : "cls";
    Result += "_meth\")))= ";
    Result += "{\n\t0, " + utostr(NumMethods) + "\n";

    Result += "\t,{{(SEL)\"";
    Result += Methods[0]->getSelector().getName().c_str();
    std::string MethodTypeString;
    Context->getObjcEncodingForMethodDecl(Methods[0], MethodTypeString);
    Result += "\", \"";
    Result += MethodTypeString;
    Result += "\", ";
    Result += MethodInternalNames[Methods[0]];
    Result += "}\n";
    for (int i = 1; i < NumMethods; i++) {
      Result += "\t  ,{(SEL)\"";
      Result += Methods[i]->getSelector().getName().c_str();
      std::string MethodTypeString;
      Context->getObjcEncodingForMethodDecl(Methods[i], MethodTypeString);
      Result += "\", \"";
      Result += MethodTypeString;
      Result += "\", ";
      Result += MethodInternalNames[Methods[i]];
      Result += "}\n";
    }
    Result += "\t }\n};\n";
  }
}

/// RewriteObjcProtocolsMetaData - Rewrite protocols meta-data.
void RewriteTest::RewriteObjcProtocolsMetaData(ObjcProtocolDecl **Protocols,
                                               int NumProtocols,
                                               const char *prefix,
                                               const char *ClassName,
                                               std::string &Result) {
  static bool objc_protocol_methods = false;
  if (NumProtocols > 0) {
    for (int i = 0; i < NumProtocols; i++) {
      ObjcProtocolDecl *PDecl = Protocols[i];
      // Output struct protocol_methods holder of method selector and type.
      if (!objc_protocol_methods &&
          (PDecl->getNumInstanceMethods() > 0 
           || PDecl->getNumClassMethods() > 0)) {
        /* struct protocol_methods {
         SEL _cmd;
         char *method_types;
         }
         */
        Result += "\nstruct protocol_methods {\n";
        Result += "\tSEL _cmd;\n";
        Result += "\tchar *method_types;\n";
        Result += "};\n";
        
        /* struct _objc_protocol_method_list {
         int protocol_method_count;
         struct protocol_methods protocols[];
         }
         */
        Result += "\nstruct _objc_protocol_method_list {\n";
        Result += "\tint protocol_method_count;\n";
        Result += "\tstruct protocol_methods protocols[];\n};\n";
        objc_protocol_methods = true;
      }
      
      // Output instance methods declared in this protocol.
      int NumMethods = PDecl->getNumInstanceMethods();
      if (NumMethods > 0) {
        Result += "\nstatic struct _objc_protocol_method_list "
               "_OBJC_PROTOCOL_INSTANCE_METHODS_";
        Result += PDecl->getName();
        Result += " __attribute__ ((section (\"__OBJC, __cat_inst_meth\")))= "
          "{\n\t" + utostr(NumMethods) + "\n";
        
        ObjcMethodDecl **Methods = PDecl->getInstanceMethods();
        Result += "\t,{{(SEL)\"";
        Result += Methods[0]->getSelector().getName().c_str();
        Result += "\", \"\"}\n";
                       
        for (int i = 1; i < NumMethods; i++) {
          Result += "\t  ,{(SEL)\"";
          Result += Methods[i]->getSelector().getName().c_str();
          std::string MethodTypeString;
          Context->getObjcEncodingForMethodDecl(Methods[i], MethodTypeString);
          Result += "\", \"";
          Result += MethodTypeString;
          Result += "\"}\n";
        }
        Result += "\t }\n};\n";
      }
      
      // Output class methods declared in this protocol.
      NumMethods = PDecl->getNumClassMethods();
      if (NumMethods > 0) {
        Result += "\nstatic struct _objc_protocol_method_list "
               "_OBJC_PROTOCOL_CLASS_METHODS_";
        Result += PDecl->getName();
        Result += " __attribute__ ((section (\"__OBJC, __cat_cls_meth\")))= "
               "{\n\t";
        Result += utostr(NumMethods);
        Result += "\n";
        
        ObjcMethodDecl **Methods = PDecl->getClassMethods();
        Result += "\t,{{(SEL)\"";
        Result += Methods[0]->getSelector().getName().c_str();
        Result += "\", \"\"}\n";
            
        for (int i = 1; i < NumMethods; i++) {
          Result += "\t  ,{(SEL)\"";
          Result += Methods[i]->getSelector().getName().c_str();
          std::string MethodTypeString;
          Context->getObjcEncodingForMethodDecl(Methods[i], MethodTypeString);
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
        
        /* struct _objc_protocol_list {
         struct _objc_protocol_list *next;
         int    protocol_count;
         struct _objc_protocol *class_protocols[];
         }
         */
        Result += "\nstruct _objc_protocol_list {\n";
        Result += "\tstruct _objc_protocol_list *next;\n";
        Result += "\tint    protocol_count;\n";
        Result += "\tstruct _objc_protocol *class_protocols[];\n";
        Result += "};\n";
        objc_protocol = true;
      }
      
      Result += "\nstatic struct _objc_protocol _OBJC_PROTOCOL_";
      Result += PDecl->getName();
      Result += " __attribute__ ((section (\"__OBJC, __protocol\")))= "
        "{\n\t0, \"";
      Result += PDecl->getName();
      Result += "\", 0, ";
      if (PDecl->getInstanceMethods() > 0) {
        Result += "&_OBJC_PROTOCOL_INSTANCE_METHODS_";
        Result += PDecl->getName();
        Result += ", ";
      }
      else
        Result += "0, ";
      if (PDecl->getClassMethods() > 0) {
        Result += "&_OBJC_PROTOCOL_CLASS_METHODS_";
        Result += PDecl->getName();
        Result += "\n";
      }
      else
        Result += "0\n";
      Result += "};\n";
    }
    // Output the top lovel protocol meta-data for the class.
    Result += "\nstatic struct _objc_protocol_list _OBJC_";
    Result += prefix;
    Result += "_PROTOCOLS_";
    Result += ClassName;
    Result += " __attribute__ ((section (\"__OBJC, __cat_cls_meth\")))= "
      "{\n\t0, ";
    Result += utostr(NumProtocols);
    Result += "\n";
    
    Result += "\t,{&_OBJC_PROTOCOL_";
    Result += Protocols[0]->getName();
    Result += " \n";
    
    for (int i = 1; i < NumProtocols; i++) {
      ObjcProtocolDecl *PDecl = Protocols[i];
      Result += "\t ,&_OBJC_PROTOCOL_";
      Result += PDecl->getName();
      Result += "\n";
    }
    Result += "\t }\n};\n";
  }  
}

/// RewriteObjcCategoryImplDecl - Rewrite metadata for each category 
/// implementation.
void RewriteTest::RewriteObjcCategoryImplDecl(ObjcCategoryImplDecl *IDecl,
                                              std::string &Result) {
  ObjcInterfaceDecl *ClassDecl = IDecl->getClassInterface();
  // Find category declaration for this implementation.
  ObjcCategoryDecl *CDecl;
  for (CDecl = ClassDecl->getCategoryList(); CDecl; 
       CDecl = CDecl->getNextClassCategory())
    if (CDecl->getIdentifier() == IDecl->getIdentifier())
      break;
    
  char *FullCategoryName = (char*)alloca(
    strlen(ClassDecl->getName()) + strlen(IDecl->getName()) + 2);
  sprintf(FullCategoryName, "%s_%s", ClassDecl->getName(), IDecl->getName());
  
  // Build _objc_method_list for class's instance methods if needed
  RewriteObjcMethodsMetaData(IDecl->getInstanceMethods(),
                             IDecl->getNumInstanceMethods(),
                             true,
                             "CATEGORY_", FullCategoryName, Result);
  
  // Build _objc_method_list for class's class methods if needed
  RewriteObjcMethodsMetaData(IDecl->getClassMethods(),
                             IDecl->getNumClassMethods(),
                             false,
                             "CATEGORY_", FullCategoryName, Result);
  
  // Protocols referenced in class declaration?
  // Null CDecl is case of a category implementation with no category interface
  if (CDecl)
    RewriteObjcProtocolsMetaData(CDecl->getReferencedProtocols(),
                                 CDecl->getNumReferencedProtocols(),
                                 "CATEGORY",
                                 FullCategoryName, Result);
  
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
  Result += " __attribute__ ((section (\"__OBJC, __category\")))= {\n\t\"";
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
  
  if (CDecl && CDecl->getNumReferencedProtocols() > 0) {
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
void RewriteTest::SynthesizeIvarOffsetComputation(ObjcImplementationDecl *IDecl, 
                                                  ObjcIvarDecl *ivar, 
                                                  std::string &Result) {
  Result += "offsetof(struct ";
  Result += IDecl->getName();
  Result += ", ";
  Result += ivar->getName();
  Result += ")";
}

//===----------------------------------------------------------------------===//
// Meta Data Emission
//===----------------------------------------------------------------------===//

void RewriteTest::RewriteObjcClassMetaData(ObjcImplementationDecl *IDecl,
                                           std::string &Result) {
  ObjcInterfaceDecl *CDecl = IDecl->getClassInterface();
  
  // Build _objc_ivar_list metadata for classes ivars if needed
  int NumIvars = IDecl->getImplDeclNumIvars() > 0 
                   ? IDecl->getImplDeclNumIvars() 
                   : (CDecl ? CDecl->getNumInstanceVariables() : 0);
  
  SynthesizeObjcInternalStruct(CDecl, Result);
  
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
      
      /* struct _objc_ivar_list {
       int ivar_count;
       struct _objc_ivar ivar_list[];
       };  
       */
      Result += "\nstruct _objc_ivar_list {\n";
      Result += "\tint ivar_count;\n";
      Result += "\tstruct _objc_ivar ivar_list[];\n};\n";
      objc_ivar = true;
    }

    Result += "\nstatic struct _objc_ivar_list _OBJC_INSTANCE_VARIABLES_";
    Result += IDecl->getName();
    Result += " __attribute__ ((section (\"__OBJC, __instance_vars\")))= "
      "{\n\t";
    Result += utostr(NumIvars);
    Result += "\n";
          
    ObjcIvarDecl **Ivars = IDecl->getImplDeclIVars() 
                             ? IDecl->getImplDeclIVars() 
                             : CDecl->getInstanceVariables();
    Result += "\t,{{\"";
    Result += Ivars[0]->getName();
    Result += "\", \"";
    std::string StrEncoding;
    Context->getObjcEncodingForType(Ivars[0]->getType(), StrEncoding);
    Result += StrEncoding;
    Result += "\", ";
    SynthesizeIvarOffsetComputation(IDecl, Ivars[0], Result);
    Result += "}\n";
    for (int i = 1; i < NumIvars; i++) {
      Result += "\t  ,{\"";
      Result += Ivars[i]->getName();
      Result += "\", \"";
      std::string StrEncoding;
      Context->getObjcEncodingForType(Ivars[i]->getType(), StrEncoding);
      Result += StrEncoding;
      Result += "\", ";
      SynthesizeIvarOffsetComputation(IDecl, Ivars[i], Result);
      Result += "}\n";
    }
    
    Result += "\t }\n};\n";
  }
  
  // Build _objc_method_list for class's instance methods if needed
  RewriteObjcMethodsMetaData(IDecl->getInstanceMethods(), 
                             IDecl->getNumInstanceMethods(), 
                             true,
                             "", IDecl->getName(), Result);
  
  // Build _objc_method_list for class's class methods if needed
  RewriteObjcMethodsMetaData(IDecl->getClassMethods(), 
                             IDecl->getNumClassMethods(),
                             false,
                             "", IDecl->getName(), Result);
    
  // Protocols referenced in class declaration?
  RewriteObjcProtocolsMetaData(CDecl->getReferencedProtocols(), 
                               CDecl->getNumIntfRefProtocols(),
                               "CLASS",
                               CDecl->getName(), Result);
    
  
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
  ObjcInterfaceDecl *RootClass = 0;
  ObjcInterfaceDecl *SuperClass = CDecl->getSuperClass();
  while (SuperClass) {
    RootClass = SuperClass;
    SuperClass = SuperClass->getSuperClass();
  }
  SuperClass = CDecl->getSuperClass();
  
  Result += "\nstatic struct _objc_class _OBJC_METACLASS_";
  Result += CDecl->getName();
  Result += " __attribute__ ((section (\"__OBJC, __meta_class\")))= "
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
  // TODO: 'ivars' field for root class is currently set to 0.
  // 'info' field is initialized to CLS_META(2) for metaclass
  Result += ", 0,2, sizeof(struct _objc_class), 0";
  if (CDecl->getNumClassMethods() > 0) {
    Result += "\n\t, &_OBJC_CLASS_METHODS_";
    Result += CDecl->getName();
    Result += "\n"; 
  }
  else
    Result += ", 0\n";
  if (CDecl->getNumIntfRefProtocols() > 0) {
    Result += "\t,0, &_OBJC_CLASS_PROTOCOLS_";
    Result += CDecl->getName();
    Result += ",0,0\n";
  }
  else
    Result += "\t,0,0,0,0\n";
  Result += "};\n";
  
  // class metadata generation.
  Result += "\nstatic struct _objc_class _OBJC_CLASS_";
  Result += CDecl->getName();
  Result += " __attribute__ ((section (\"__OBJC, __class\")))= "
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
  if (!ObjcSynthesizedStructs.count(CDecl))
    Result += ",0";
  else {
    // class has size. Must synthesize its size.
    Result += ",sizeof(struct ";
    Result += CDecl->getName();
    Result += ")";
  }
  if (NumIvars > 0) {
    Result += ", &_OBJC_INSTANCE_VARIABLES_";
    Result += CDecl->getName();
    Result += "\n\t";
  }
  else
    Result += ",0";
  if (IDecl->getNumInstanceMethods() > 0) {
    Result += ", &_OBJC_INSTANCE_METHODS_";
    Result += CDecl->getName();
    Result += ", 0\n\t"; 
  }
  else
    Result += ",0,0";
  if (CDecl->getNumIntfRefProtocols() > 0) {
    Result += ", &_OBJC_CLASS_PROTOCOLS_";
    Result += CDecl->getName();
    Result += ", 0,0\n";
  }
  else
    Result += ",0,0,0\n";
  Result += "};\n";
}

/// RewriteImplementations - This routine rewrites all method implementations
/// and emits meta-data.

void RewriteTest::RewriteImplementations(std::string &Result) {
  int ClsDefCount = ClassImplementation.size();
  int CatDefCount = CategoryImplementation.size();
  
  if (ClsDefCount == 0 && CatDefCount == 0)
    return;
  // Rewrite implemented methods
  for (int i = 0; i < ClsDefCount; i++)
    RewriteImplementationDecl(ClassImplementation[i]);
  
  for (int i = 0; i < CatDefCount; i++)
    RewriteImplementationDecl(CategoryImplementation[i]);
  
  // TODO: This is temporary until we decide how to access objc types in a
  // c program
  Result += "#include <Objc/objc.h>\n";
  // This is needed for use of offsetof
  Result += "#include <stddef.h>\n";
    
  // For each implemented class, write out all its meta data.
  for (int i = 0; i < ClsDefCount; i++)
    RewriteObjcClassMetaData(ClassImplementation[i], Result);
  
  // For each implemented category, write out all its meta data.
  for (int i = 0; i < CatDefCount; i++)
    RewriteObjcCategoryImplDecl(CategoryImplementation[i], Result);
  
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
         "_OBJC_SYMBOLS __attribute__((section (\"__OBJC, __symbols\")))= {\n";
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
    "_OBJC_MODULES __attribute__ ((section (\"__OBJC, __module_info\")))= {\n";
  Result += "\t" + utostr(OBJC_ABI_VERSION) + 
  ", sizeof(struct _objc_module), \"\", &_OBJC_SYMBOLS\n";
  Result += "};\n\n";
  
}

