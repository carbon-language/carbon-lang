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
    
    FunctionDecl *MsgSendFunctionDecl;
    FunctionDecl *GetClassFunctionDecl;
    FunctionDecl *SelGetUidFunctionDecl;
    
    static const int OBJC_ABI_VERSION =7 ;
  public:
    void Initialize(ASTContext &context, unsigned mainFileID) {
      Context = &context;
      SM = &Context->SourceMgr;
      MainFileID = mainFileID;
      MsgSendFunctionDecl = 0;
      GetClassFunctionDecl = 0;
      SelGetUidFunctionDecl = 0;
      Rewrite.setSourceMgr(Context->SourceMgr);
    }

    // Top Level Driver code.
    virtual void HandleTopLevelDecl(Decl *D);
    void HandleDeclInMainFile(Decl *D);
    ~RewriteTest();

    // Syntactic Rewriting.
    void RewriteInclude(SourceLocation Loc);
    void RewriteTabs();
    void RewriteForwardClassDecl(ObjcClassDecl *Dcl);
    void RewriteInterfaceDecl(ObjcInterfaceDecl *Dcl);
    
    // Expression Rewriting.
    Stmt *RewriteFunctionBody(Stmt *S);
    Stmt *RewriteAtEncode(ObjCEncodeExpr *Exp);
    Stmt *RewriteMessageExpr(ObjCMessageExpr *Exp);
    CallExpr *SynthesizeCallToFunctionDecl(FunctionDecl *FD, 
                                           Expr **args, unsigned nargs);
    // Metadata emission.
    void HandleObjcMetaDataEmission();
    void RewriteObjcClassMetaData(ObjcImplementationDecl *IDecl,
                                  std::string &Result);
    
    void RewriteObjcCategoryImplDecl(ObjcCategoryImplDecl *CDecl,
                                     std::string &Result);
    
    void RewriteObjcMethodsMetaData(ObjcMethodDecl **Methods,
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
    void WriteObjcMetaData(std::string &Result);
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
    if (strcmp(FD->getName(), "objc_msgSend") == 0)
      MsgSendFunctionDecl = FD;
    else if (strcmp(FD->getName(), "objc_getClass") == 0)
      GetClassFunctionDecl = FD;
    else if (strcmp(FD->getName(), "sel_getUid") == 0)
      SelGetUidFunctionDecl = FD;
  } else if (ObjcInterfaceDecl *MD = dyn_cast<ObjcInterfaceDecl>(D)) {
    RewriteInterfaceDecl(MD);
  }
  // If we have a decl in the main file, see if we should rewrite it.
  if (SM->getDecomposedFileLoc(Loc).first == MainFileID)
    return HandleDeclInMainFile(D);

  // Otherwise, see if there is a #import in the main file that should be
  // rewritten.
  RewriteInclude(Loc);
}

/// HandleDeclInMainFile - This is called for each top-level decl defined in the
/// main file of the input.
void RewriteTest::HandleDeclInMainFile(Decl *D) {
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    if (Stmt *Body = FD->getBody())
      FD->setBody(RewriteFunctionBody(Body));
  
  if (ObjcImplementationDecl *CI = dyn_cast<ObjcImplementationDecl>(D))
    ClassImplementation.push_back(CI);
  else if (ObjcCategoryImplDecl *CI = dyn_cast<ObjcCategoryImplDecl>(D))
    CategoryImplementation.push_back(CI);
  else if (ObjcClassDecl *CD = dyn_cast<ObjcClassDecl>(D))
    RewriteForwardClassDecl(CD);
  // Nothing yet.
}

RewriteTest::~RewriteTest() {
  // Get the top-level buffer that this corresponds to.
  RewriteTabs();
  
  // Get the buffer corresponding to MainFileID.  If we haven't changed it, then
  // we are done.
  if (const RewriteBuffer *RewriteBuf = 
      Rewrite.getRewriteBufferFor(MainFileID)) {
    printf("Changed:\n");
    std::string S(RewriteBuf->begin(), RewriteBuf->end());
    printf("%s\n", S.c_str());
  } else {
    printf("No changes\n");
  }

}

/// HandleObjcMetaDataEmission - main routine to generate objective-c's 
/// metadata.
void RewriteTest::HandleObjcMetaDataEmission() {
  // Rewrite Objective-c meta data*
  std::string ResultStr;
  WriteObjcMetaData(ResultStr);
  // For now just print the string out.
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
    typedefString += "typedef struct ";
    typedefString += ForwardDecl->getName();
    typedefString += " ";
    typedefString += ForwardDecl->getName();
    typedefString += ";\n";
  }
  
  // Replace the @class with typedefs corresponding to the classes.
  Rewrite.ReplaceText(startLoc, semiPtr-startBuf+1, 
                      typedefString.c_str(), typedefString.size());
}

void RewriteTest::RewriteInterfaceDecl(ObjcInterfaceDecl *ClassDecl) {
  int nInstanceMethods = ClassDecl->getNumInstanceMethods();
  ObjcMethodDecl **instanceMethods = ClassDecl->getInstanceMethods();
  
  for (int i = 0; i < nInstanceMethods; i++) {
    ObjcMethodDecl *instanceMethod = instanceMethods[i];
    SourceLocation Loc = instanceMethod->getLocStart();

    Rewrite.ReplaceText(Loc, 0, "// ", 3);
    
    // FIXME: handle methods that are declared across multiple lines.
  }
  int nClassMethods = ClassDecl->getNumClassMethods();
  ObjcMethodDecl **classMethods = ClassDecl->getClassMethods();
  
  for (int i = 0; i < nClassMethods; i++) {
    ObjcMethodDecl *classMethod = classMethods[i];
    SourceLocation Loc = classMethod->getLocStart();

    Rewrite.ReplaceText(Loc, 0, "// ", 3);
    
    // FIXME: handle methods that are declared across multiple lines.
  }
}

//===----------------------------------------------------------------------===//
// Function Body / Expression rewriting
//===----------------------------------------------------------------------===//

Stmt *RewriteTest::RewriteFunctionBody(Stmt *S) {
  // Otherwise, just rewrite all children.
  for (Stmt::child_iterator CI = S->child_begin(), E = S->child_end();
       CI != E; ++CI)
    if (*CI)
      *CI = RewriteFunctionBody(*CI);
      
  // Handle specific things.
  if (ObjCEncodeExpr *AtEncode = dyn_cast<ObjCEncodeExpr>(S))
    return RewriteAtEncode(AtEncode);
    
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
    Rewrite.ReplaceText(startLoc, 0, messString.c_str(), messString.size());
    return RewriteMessageExpr(MessExpr);
  }
  // Return this stmt unmodified.
  return S;
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

Stmt *RewriteTest::RewriteMessageExpr(ObjCMessageExpr *Exp) {
  assert(MsgSendFunctionDecl && "Can't find objc_msgSend() decl");
  assert(SelGetUidFunctionDecl && "Can't find sel_getUid() decl");
  assert(GetClassFunctionDecl && "Can't find objc_getClass() decl");

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
    
  // Create a call to sel_getUid("selName"), it will be the 2nd argument.
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
  CallExpr *MessExp = SynthesizeCallToFunctionDecl(MsgSendFunctionDecl,
                                                 &MsgExprs[0], MsgExprs.size());
  // Now do the actual rewrite.
  Rewrite.ReplaceStmt(Exp, MessExp);
  
  delete Exp;
  return MessExp;
}

/// SynthesizeObjcInternalStruct - Rewrite one internal struct corresponding to
/// an objective-c class with ivars.
void RewriteTest::SynthesizeObjcInternalStruct(ObjcInterfaceDecl *CDecl,
                                               std::string &Result) {
  assert(CDecl && "Class missing in SynthesizeObjcInternalStruct");
  assert(CDecl->getName() && "Name missing in SynthesizeObjcInternalStruct");
  ObjcInterfaceDecl *RCDecl = CDecl->getSuperClass();
  if (RCDecl && !ObjcSynthesizedStructs.count(RCDecl)) {
    // Do it for the root
    SynthesizeObjcInternalStruct(RCDecl, Result);
  }
  
  int NumIvars = CDecl->getIntfDeclNumIvars();
  if (NumIvars <= 0 && (!RCDecl || !ObjcSynthesizedStructs.count(RCDecl)))
    return;
  
  Result += "\nstruct _interface_";
  Result += CDecl->getName();
  Result += " {\n";
  if (RCDecl && ObjcSynthesizedStructs.count(RCDecl)) {
    Result += "\tstruct _interface_";
    Result += RCDecl->getName();
    Result += " _";
    Result += RCDecl->getName();
    Result += ";\n";
  }
  
  ObjcIvarDecl **Ivars = CDecl->getIntfDeclIvars();
  for (int i = 0; i < NumIvars; i++) {
    Result += "\t";
    std::string Name = Ivars[i]->getName();
    Ivars[i]->getType().getAsStringInternal(Name);
    Result += Name;
    Result += ";\n";
  }
  Result += "};\n";
  // Mark this struct as having been generated.
  if (!ObjcSynthesizedStructs.insert(CDecl))
  assert(true && "struct already synthesize- SynthesizeObjcInternalStruct");
}

// RewriteObjcMethodsMetaData - Rewrite methods metadata for instance or
/// class methods.
void RewriteTest::RewriteObjcMethodsMetaData(ObjcMethodDecl **Methods,
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
    Result += "\", \"\", 0}\n";
  
    for (int i = 1; i < NumMethods; i++) {
      // TODO: 1) method selector name may hav to go into their own section
      // 2) encode method types for use here (which may have to go into 
      // __meth_var_types section, 3) Need method address as 3rd initializer.
      Result += "\t  ,{(SEL)\"";
      Result += Methods[i]->getSelector().getName().c_str();
      Result += "\", \"\", 0}\n";
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
          // TODO: 1) method selector name may hav to go into their own section
          // 2) encode method types for use here (which may have to go into 
          // __meth_var_types section.
          Result += "\t  ,{(SEL)\"";
          Result += Methods[i]->getSelector().getName().c_str();
          Result += "\", \"\"}\n";
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
          // TODO: 1) method selector name may hav to go into their own section
          // 2) encode method types for use here (which may have to go into 
          // __meth_var_types section.
          Result += "\t  ,{(SEL)\"";
          Result += Methods[i]->getSelector().getName().c_str();
          Result += "\", \"\"}\n";
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
  assert(CDecl && "RewriteObjcCategoryImplDecl - bad category");
  
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
  
  if (CDecl->getNumReferencedProtocols() > 0) {
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
  Result += "offsetof(struct _interface_";
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
                   : (CDecl ? CDecl->getIntfDeclNumIvars() : 0);
  
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
                             : CDecl->getIntfDeclIvars();
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
    Result += ",sizeof(struct _interface_";
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

void RewriteTest::WriteObjcMetaData(std::string &Result) {
  int ClsDefCount = ClassImplementation.size();
  int CatDefCount = CategoryImplementation.size();
  if (ClsDefCount == 0 && CatDefCount == 0)
    return;
  
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

