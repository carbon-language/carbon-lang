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
using namespace clang;

namespace {
  class RewriteTest : public ASTConsumer {
    Rewriter Rewrite;
    ASTContext *Context;
    SourceManager *SM;
    unsigned MainFileID;
    SourceLocation LastIncLoc;
    llvm::SmallVector<ObjcImplementationDecl *, 8> ClassImplementation;
    llvm::SmallVector<ObjcCategoryImplDecl *, 8> CategoryImplementation;
    
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
    
    // Expression Rewriting.
    Stmt *RewriteFunctionBody(Stmt *S);
    Stmt *RewriteAtEncode(ObjCEncodeExpr *Exp);
    Stmt *RewriteMessageExpr(ObjCMessageExpr *Exp);
    CallExpr *SynthesizeCallToFunctionDecl(FunctionDecl *FD, 
                                           Expr **args, unsigned nargs);
    // Metadata emission.
    void RewriteObjcClassMetaData(ObjcImplementationDecl *IDecl);
    
    void RewriteObjcCategoryImplDecl(ObjcCategoryImplDecl *CDecl);
    
    void RewriteObjcMethodsMetaData(ObjcMethodDecl **Methods,
                                    int NumMethods,
                                    const char *prefix,
                                    const char *MethodKind,
                                    const char *ClassName);
    
    void RewriteObjcProtocolsMetaData(ObjcProtocolDecl **Protocols,
                                      int NumProtocols,
                                      const char *prefix,
                                      const char *ClassName);
    void WriteObjcMetaData();
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
  
  // Rewrite Objective-c meta data*
  WriteObjcMetaData();
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
    // Rewrite.ReplaceText(startLoc, 0, messString.c_str(), messString.size());
    return RewriteMessageExpr(MessExpr);
  }
  // Return this stmt unmodified.
  return S;
}
 
Stmt *RewriteTest::RewriteAtEncode(ObjCEncodeExpr *Exp) {
  // Create a new string expression.
  QualType StrType = Context->getPointerType(Context->CharTy);
  Expr *Replacement = new StringLiteral("foo", 3, false, StrType, 
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

// RewriteObjcMethodsMetaData - Rewrite methods metadata for instance or
/// class methods.
void RewriteTest::RewriteObjcMethodsMetaData(ObjcMethodDecl **Methods,
                                             int NumMethods,
                                             const char *prefix,
                                             const char *MethodKind,
                                             const char *ClassName) {
  static bool objc_impl_method = false;
  if (NumMethods > 0 && !objc_impl_method) {
    /* struct _objc_method {
       SEL _cmd;
       char *method_types;
       void *_imp;
       }
     */
    printf("\nstruct _objc_method {\n");
    printf("\tSEL _cmd;\n");
    printf("\tchar *method_types;\n");
    printf("\tvoid *_imp;\n");
    printf("};\n");
    objc_impl_method = true;
  }
  // Build _objc_method_list for class's methods if needed
  if (NumMethods > 0) {
    /* struct _objc_method_list {
     struct _objc_method_list *next_method;
     int method_count;
     struct _objc_method method_list[method_count];
     }
     */
    printf("\nstatic struct {\n");
    printf("\tstruct _objc_method_list *next_method;\n");
    printf("\tint method_count;\n");
    printf("\tstruct _objc_method method_list[%d];\n", NumMethods);
    printf("} _OBJC_%s%s_METHODS_%s "
           "__attribute__ ((section (\"__OBJC, __inst_meth\")))= "
           "{\n\t0, %d\n", prefix, MethodKind, ClassName, NumMethods);
    for (int i = 0; i < NumMethods; i++)
      // TODO: 1) method selector name may hav to go into their own section
      // 2) encode method types for use here (which may have to go into 
      // __meth_var_types section, 3) Need method address as 3rd initializer.
      printf("\t,(SEL)\"%s\", \"\", 0\n", 
             Methods[i]->getSelector().getName().c_str());
    printf("};\n");
  }
}

/// RewriteObjcProtocolsMetaData - Rewrite protocols meta-data.
void RewriteTest::RewriteObjcProtocolsMetaData(ObjcProtocolDecl **Protocols,
                                               int NumProtocols,
                                               const char *prefix,
                                               const char *ClassName) {
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
        printf("\nstruct protocol_methods {\n");
        printf("\tSEL _cmd;\n");
        printf("\tchar *method_types;\n");
        printf("};\n");
        objc_protocol_methods = true;
      }
      // Output instance methods declared in this protocol.
      /* struct _objc_protocol_method_list {
       int protocol_method_count;
       struct protocol_methods protocols[protocol_method_count];
       }
       */      
      int NumMethods = PDecl->getNumInstanceMethods();
      if (NumMethods > 0) {
        printf("\nstatic struct {\n");
        printf("\tint protocol_method_count;\n");
        printf("\tstruct protocol_methods protocols[%d];\n", NumMethods);
        printf("} _OBJC_PROTOCOL_INSTANCE_METHODS_%s "
               "__attribute__ ((section (\"__OBJC, __cat_inst_meth\")))= "
               "{\n\t%d\n",PDecl->getName(), NumMethods);
        ObjcMethodDecl **Methods = PDecl->getInstanceMethods();
        for (int i = 0; i < NumMethods; i++)
          // TODO: 1) method selector name may hav to go into their own section
          // 2) encode method types for use here (which may have to go into 
          // __meth_var_types section.
          printf("\t,(SEL)\"%s\", \"\"\n", 
                 Methods[i]->getSelector().getName().c_str());
        printf("};\n");
      }
      
      // Output class methods declared in this protocol.
      NumMethods = PDecl->getNumClassMethods();
      if (NumMethods > 0) {
        printf("\nstatic struct {\n");
        printf("\tint protocol_method_count;\n");
        printf("\tstruct protocol_methods protocols[%d];\n", NumMethods);
        printf("} _OBJC_PROTOCOL_CLASS_METHODS_%s "
               "__attribute__ ((section (\"__OBJC, __cat_cls_meth\")))= "
               "{\n\t%d\n",PDecl->getName(), NumMethods);
        ObjcMethodDecl **Methods = PDecl->getClassMethods();
        for (int i = 0; i < NumMethods; i++)
          // TODO: 1) method selector name may hav to go into their own section
          // 2) encode method types for use here (which may have to go into 
          // __meth_var_types section.
          printf("\t,(SEL)\"%s\", \"\"\n", 
                 Methods[i]->getSelector().getName().c_str());
        printf("};\n");
      }
      // Output:
      /* struct _objc_protocol {
       // Objective-C 1.0 extensions
       struct _objc_protocol_extension *isa;
       char *protocol_name;
       struct _objc_protocol **protocol_list;
       struct _objc__method_prototype_list *instance_methods;
       struct _objc__method_prototype_list *class_methods;
       };  
       */
      static bool objc_protocol = false;
      if (!objc_protocol) {
        printf("\nstruct _objc_protocol {\n");
        printf("\tstruct _objc_protocol_extension *isa;\n");
        printf("\tchar *protocol_name;\n");
        printf("\tstruct _objc_protocol **protocol_list;\n");
        printf("\tstruct _objc__method_prototype_list *instance_methods;\n");
        printf("\tstruct _objc__method_prototype_list *class_methods;\n");
        printf("};\n");
        objc_protocol = true;
      }
      
      printf("\nstatic struct _objc_protocol _OBJC_PROTOCOL_%s "
             "__attribute__ ((section (\"__OBJC, __protocol\")))= "
             "{\n\t0, \"%s\", 0, ", PDecl->getName(), PDecl->getName());
      if (PDecl->getInstanceMethods() > 0)
        printf("(struct _objc__method_prototype_list *)"
               "&_OBJC_PROTOCOL_INSTANCE_METHODS_%s, ", PDecl->getName());
      else
        printf("0, ");
      if (PDecl->getClassMethods() > 0)
        printf("(struct _objc__method_prototype_list *)"
               "&_OBJC_PROTOCOL_CLASS_METHODS_%s\n", PDecl->getName());
      else
        printf("0\n");
      printf("};\n");
    }
    // Output the top lovel protocol meta-data for the class.
    /* struct _objc_protocol_list {
     struct _objc_protocol_list *next;
     int    protocol_count;
     struct _objc_protocol *class_protocols[protocol_count];
     }
     */
    printf("\nstatic struct {\n");
    printf("\tstruct _objc_protocol_list *next;\n");
    printf("\tint    protocol_count;\n");
    printf("\tstruct _objc_protocol *class_protocols[%d];\n"
           "} _OBJC_%s_PROTOCOLS_%s "
           "__attribute__ ((section (\"__OBJC, __cat_cls_meth\")))= "
           "{\n\t0, %d\n",NumProtocols, prefix,
           ClassName, NumProtocols);
    for (int i = 0; i < NumProtocols; i++) {
      ObjcProtocolDecl *PDecl = Protocols[i];
      printf("\t,&_OBJC_PROTOCOL_%s \n", 
             PDecl->getName());
    }
    printf("};\n");
  }  
}

/// RewriteObjcCategoryImplDecl - Rewrite metadata for each category 
/// implementation.
void RewriteTest::RewriteObjcCategoryImplDecl(ObjcCategoryImplDecl *IDecl) {
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
                             "CATEGORY_", "INSTANCE", FullCategoryName);
  
  // Build _objc_method_list for class's class methods if needed
  RewriteObjcMethodsMetaData(IDecl->getClassMethods(),
                             IDecl->getNumClassMethods(),
                             "CATEGORY_", "CLASS", FullCategoryName);
  
  // Protocols referenced in class declaration?
  RewriteObjcProtocolsMetaData(CDecl->getReferencedProtocols(),
                               CDecl->getNumReferencedProtocols(),
                               "CATEGORY",
                               FullCategoryName);
  
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
    printf("\nstruct _objc_category {\n");
    printf("\tchar *category_name;\n");
    printf("\tchar *class_name;\n");
    printf("\tstruct _objc_method_list *instance_methods;\n");
    printf("\tstruct _objc_method_list *class_methods;\n");
    printf("\tstruct _objc_protocol_list *protocols;\n");
    printf("\tunsigned int size;\n");   
    printf("\tstruct _objc_property_list *instance_properties;\n");
    printf("};\n");
    objc_category = true;
  }
  printf("\nstatic struct _objc_category _OBJC_CATEGORY_%s "
         "__attribute__ ((section (\"__OBJC, __category\")))= {\n"
         "\t\"%s\"\n\t, \"%s\"\n",FullCategoryName, 
         IDecl->getName(), 
         ClassDecl->getName());
  if (IDecl->getNumInstanceMethods() > 0)
    printf("\t, (struct _objc_method_list *)"
           "&_OBJC_CATEGORY_INSTANCE_METHODS_%s\n", 
           FullCategoryName);
  else
    printf("\t, 0\n");
  if (IDecl->getNumClassMethods() > 0)
    printf("\t, (struct _objc_method_list *)"
           "&_OBJC_CATEGORY_CLASS_METHODS_%s\n", 
           FullCategoryName);
  else
    printf("\t, 0\n");
  
  if (CDecl->getNumReferencedProtocols() > 0)
    printf("\t, (struct _objc_protocol_list *)&_OBJC_CATEGORY_PROTOCOLS_%s\n", 
           FullCategoryName);
  else
    printf("\t, 0\n");
  printf("\t, sizeof(struct _objc_category), 0\n};\n");
}

//===----------------------------------------------------------------------===//
// Meta Data Emission
//===----------------------------------------------------------------------===//

void RewriteTest::RewriteObjcClassMetaData(ObjcImplementationDecl *IDecl) {
  ObjcInterfaceDecl *CDecl = IDecl->getClassInterface();
  
  // Build _objc_ivar_list metadata for classes ivars if needed
  int NumIvars = IDecl->getImplDeclNumIvars() > 0 
                   ? IDecl->getImplDeclNumIvars() 
                   : (CDecl ? CDecl->getIntfDeclNumIvars() : 0);
  
  if (NumIvars > 0) {
    static bool objc_ivar = false;
    if (!objc_ivar) {
      /* struct _objc_ivar {
          char *ivar_name;
          char *ivar_type;
          int ivar_offset;
        };  
       */
      printf("\nstruct _objc_ivar {\n");
      printf("\tchar *ivar_name;\n");
      printf("\tchar *ivar_type;\n");
      printf("\tint ivar_offset;\n");
      printf("};\n");
      objc_ivar = true;
    }

    /* struct _objc_ivar_list {
        int ivar_count;
        struct _objc_ivar ivar_list[ivar_count];
     };  
    */
    printf("\nstatic struct {\n");
    printf("\tint ivar_count;\n");
    printf("\tstruct _objc_ivar ivar_list[%d];\n", NumIvars);
    printf("} _OBJC_INSTANCE_VARIABLES_%s "
      "__attribute__ ((section (\"__OBJC, __instance_vars\")))= "
      "{\n\t%d\n",IDecl->getName(), 
           NumIvars);
    ObjcIvarDecl **Ivars = IDecl->getImplDeclIVars() 
                             ? IDecl->getImplDeclIVars() 
                             : CDecl->getIntfDeclIvars();
    for (int i = 0; i < NumIvars; i++)
      // TODO: 1) ivar names may have to go to another section. 2) encode
      // ivar_type type of each ivar . 3) compute and add ivar offset.
      printf("\t,\"%s\", \"\", 0\n", Ivars[i]->getName());
    printf("};\n");
  }
  
  // Build _objc_method_list for class's instance methods if needed
  RewriteObjcMethodsMetaData(IDecl->getInstanceMethods(), 
                             IDecl->getNumInstanceMethods(), 
                             "", "INSTANCE", IDecl->getName());
  
  // Build _objc_method_list for class's class methods if needed
  RewriteObjcMethodsMetaData(IDecl->getClassMethods(), 
                             IDecl->getNumClassMethods(), 
                             "", "CLASS", IDecl->getName());
    
  // Protocols referenced in class declaration?
  RewriteObjcProtocolsMetaData(CDecl->getReferencedProtocols(), 
                               CDecl->getNumIntfRefProtocols(),
                               "CLASS",
                               CDecl->getName());
    
  
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
    printf("\nstruct _objc_class {\n");
    printf("\tstruct _objc_class *isa;\n");
    printf("\tconst char *super_class_name;\n");
    printf("\tchar *name;\n");
    printf("\tlong version;\n");
    printf("\tlong info;\n");
    printf("\tlong instance_size;\n");
    printf("\tstruct _objc_ivar_list *ivars;\n");
    printf("\tstruct _objc_method_list *methods;\n");
    printf("\tstruct objc_cache *cache;\n");
    printf("\tstruct _objc_protocol_list *protocols;\n");
    printf("\tconst char *ivar_layout;\n");
    printf("\tstruct _objc_class_ext  *ext;\n");
    printf("};\n");
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
  
  printf("\nstatic struct _objc_class _OBJC_METACLASS_%s "
         "__attribute__ ((section (\"__OBJC, __meta_class\")))= "
         "{\n\t(struct _objc_class *)\"%s\"", 
         CDecl->getName(), RootClass ? RootClass->getName() 
                                     :  CDecl->getName());
  if (SuperClass)
    printf(", \"%s\", \"%s\"", SuperClass->getName(), CDecl->getName());
  else
    printf(", 0, \"%s\"", CDecl->getName());
  // TODO: 'ivars' field for root class is currently set to 0.
  // 'info' field is initialized to CLS_META(2) for metaclass
  printf(", 0,2, sizeof(struct _objc_class), 0");
  if (CDecl->getNumClassMethods() > 0)
    printf("\n\t, (struct _objc_method_list *)&_OBJC_CLASS_METHODS_%s\n", 
           CDecl->getName());
  else
    printf(", 0\n");
  if (CDecl->getNumIntfRefProtocols() > 0)
    printf("\t,0,(struct _objc_protocol_list*)&_OBJC_CLASS_PROTOCOLS_%s,0,0\n", 
           CDecl->getName());
  else
    printf("\t,0,0,0,0\n");
  printf("};\n");
  
  // class metadata generation.
  printf("\nstatic struct _objc_class _OBJC_CLASS_%s "
         "__attribute__ ((section (\"__OBJC, __class\")))= "
         "{\n\t&_OBJC_METACLASS_%s", CDecl->getName(), CDecl->getName());
  if (SuperClass)
    printf(", \"%s\", \"%s\"", SuperClass->getName(), CDecl->getName());
  else
    printf(", 0, \"%s\"", CDecl->getName());
  // 'info' field is initialized to CLS_CLASS(1) for class
  // TODO: instance_size is curently set to 0.
  printf(", 0,1,0");
  if (NumIvars > 0)
    printf(", (struct _objc_ivar_list *)&_OBJC_INSTANCE_VARIABLES_%s\n\t", 
           CDecl->getName());
  else
    printf(",0");
  if (IDecl->getNumInstanceMethods() > 0)
    printf(", (struct _objc_method_list*)&_OBJC_INSTANCE_METHODS_%s, 0\n\t", 
           CDecl->getName());
  else
    printf(",0,0");
  if (CDecl->getNumIntfRefProtocols() > 0)
    printf(", (struct _objc_protocol_list*)&_OBJC_CLASS_PROTOCOLS_%s, 0,0\n", 
           CDecl->getName());
  else
    printf(",0,0,0\n");
  printf("};\n");
}

void RewriteTest::WriteObjcMetaData() {
  int ClsDefCount = ClassImplementation.size();
  int CatDefCount = CategoryImplementation.size();
  if (ClsDefCount == 0 && CatDefCount == 0)
    return;
  
  // TODO: This is temporary until we decide how to access objc types in a
  // c program
  printf("\n#include <Objc/objc.h>\n");
  
  // For each implemented class, write out all its meta data.
  for (int i = 0; i < ClsDefCount; i++)
    RewriteObjcClassMetaData(ClassImplementation[i]);
  
  // For each implemented category, write out all its meta data.
  for (int i = 0; i < CatDefCount; i++)
    RewriteObjcCategoryImplDecl(CategoryImplementation[i]);
  
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
  
  printf("\nstruct _objc_symtab {\n");
  printf("\tlong sel_ref_cnt;\n");
  printf("\tSEL *refs;\n");
  printf("\tshort cls_def_cnt;\n");
  printf("\tshort cat_def_cnt;\n");
  printf("\tvoid *defs[%d];\n", ClsDefCount + CatDefCount);
  printf("};\n\n");
  
  printf("static struct _objc_symtab "
         "_OBJC_SYMBOLS __attribute__((section (\"__OBJC, __symbols\")))= {\n");
  printf("\t0, 0, %d, %d\n", ClsDefCount, CatDefCount);
  for (int i = 0; i < ClsDefCount; i++)
    printf("\t,&_OBJC_CLASS_%s\n", ClassImplementation[i]->getName());
  
  for (int i = 0; i < CatDefCount; i++)
    printf("\t,&_OBJC_CATEGORY_%s_%s\n", 
           CategoryImplementation[i]->getClassInterface()->getName(), 
           CategoryImplementation[i]->getName());
  
  printf("};\n\n");
  
  // Write objc_module metadata
  
  /*
   struct _objc_module {
    long version;
    long size;
    const char *name;
    struct _objc_symtab *symtab;
   }
  */
  
  printf("\nstruct _objc_module {\n");
  printf("\tlong version;\n");
  printf("\tlong size;\n");
  printf("\tconst char *name;\n");
  printf("\tstruct _objc_symtab *symtab;\n");
  printf("};\n\n");
  printf("static struct _objc_module "
    "_OBJC_MODULES __attribute__ ((section (\"__OBJC, __module_info\")))= {\n");
  printf("\t%d, sizeof(struct _objc_module), \"\", &_OBJC_SYMBOLS\n", OBJC_ABI_VERSION);
  printf("};\n\n");
}

