//===------- CGObjCMac.cpp - Interface to Apple Objective-C Runtime -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides Objective-C code generation targetting the Apple runtime.
//
//===----------------------------------------------------------------------===//

#include "CGObjCRuntime.h"

#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/LangOptions.h"

#include "llvm/Module.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Target/TargetData.h"

using namespace clang;

namespace {

/// ObjCTypesHelper - Helper class that encapsulates lazy
/// construction of varies types used during ObjC generation.
class ObjCTypesHelper {
private:
  CodeGen::CodeGenModule &CGM;  
  
  const llvm::StructType *CFStringType;
  llvm::Constant *CFConstantStringClassReference;
  llvm::Function *MessageSendFn;

public:
  const llvm::Type *LongTy;

  /// ObjectPtrTy - LLVM type for object handles (typeof(id))
  const llvm::Type *ObjectPtrTy;
  /// SelectorPtrTy - LLVM type for selector handles (typeof(SEL))
  const llvm::Type *SelectorPtrTy;
  /// ProtocolPtrTy - LLVM type for protocol handles (typeof(Protocol))
  const llvm::Type *ProtocolPtrTy;

  /// SymtabTy - LLVM type for struct objc_symtab.
  const llvm::StructType *SymtabTy;
  /// ModuleTy - LLVM type for struct objc_module.
  const llvm::StructType *ModuleTy;

public:
  ObjCTypesHelper(CodeGen::CodeGenModule &cgm);
  ~ObjCTypesHelper();
  
  llvm::Constant *getCFConstantStringClassReference();
  const llvm::StructType *getCFStringType();
  llvm::Function *getMessageSendFn();
};

class CGObjCMac : public CodeGen::CGObjCRuntime {
private:
  CodeGen::CodeGenModule &CGM;  
  ObjCTypesHelper ObjCTypes;
  /// ObjCABI - FIXME: Not sure yet.
  unsigned ObjCABI;

  /// ClassNames - uniqued class names.
  llvm::DenseMap<Selector, llvm::GlobalVariable*> ClassNames;

  /// MethodVarNames - uniqued method variable names.
  llvm::DenseMap<Selector, llvm::GlobalVariable*> MethodVarNames;

  /// SelectorReferences - uniqued selector references.
  llvm::DenseMap<Selector, llvm::GlobalVariable*> SelectorReferences;

  /// UsedGlobals - list of globals to pack into the llvm.used metadata
  /// to prevent them from being clobbered.
  std::vector<llvm::GlobalVariable*> UsedGlobals;

  /// EmitImageInfo - Emit the image info marker used to encode some module
  /// level information.
  void EmitImageInfo();

  /// EmitModuleInfo - Another marker encoding module level
  /// information. 

  // FIXME: Not sure yet of the difference between this and
  // IMAGE_INFO. otool looks at this before printing out Obj-C info
  // though...
  void EmitModuleInfo();

  /// EmitModuleSymols - Emit module symbols, the result is a constant
  /// of type pointer-to SymtabTy. // FIXME: Describe more completely
  /// once known.
  llvm::Constant *EmitModuleSymbols();

  /// FinishModule - Write out global data structures at the end of
  /// processing a translation unit.
  void FinishModule();
  
  /// EmitSelector - Return a Value*, of type ObjCTypes.SelectorPtrTy,
  /// for the given selector.
  llvm::Value *EmitSelector(llvm::IRBuilder<> &Builder, Selector Sel);

  /// GetClassName - Return a unique constant for the given selector's
  /// name.
  llvm::Constant *GetClassName(Selector Sel);

  /// GetMethodVarName - Return a unique constant for the given
  /// selector's name.
  llvm::Constant *GetMethodVarName(Selector Sel);

public:
  CGObjCMac(CodeGen::CodeGenModule &cgm);
  virtual llvm::Constant *GenerateConstantString(const std::string &String);

  virtual llvm::Value *GenerateMessageSend(llvm::IRBuilder<> &Builder,
                                           const llvm::Type *ReturnTy,
                                           llvm::Value *Receiver,
                                           Selector Sel,
                                           llvm::Value** ArgV,
                                           unsigned ArgC);

  virtual llvm::Value *GenerateMessageSendSuper(llvm::IRBuilder<> &Builder,
                                                const llvm::Type *ReturnTy,
                                                const char *SuperClassName,
                                                llvm::Value *Receiver,
                                                Selector Sel,
                                                llvm::Value** ArgV,
                                                unsigned ArgC);

  virtual llvm::Value *LookupClass(llvm::IRBuilder<> &Builder,
                                   llvm::Value *ClassName);

  virtual llvm::Value *GetSelector(llvm::IRBuilder<> &Builder, Selector Sel);
  
  virtual llvm::Function *MethodPreamble(const std::string &ClassName,
                                         const std::string &CategoryName,
                                         const std::string &MethodName,
                                         const llvm::Type *ReturnTy,
                                         const llvm::Type *SelfTy,
                                         const llvm::Type **ArgTy,
                                         unsigned ArgC,
                                         bool isClassMethod,
                                         bool isVarArg);

  virtual void GenerateCategory(const char *ClassName, const char *CategoryName,
           const llvm::SmallVectorImpl<Selector>  &InstanceMethodSels,
           const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodTypes,
           const llvm::SmallVectorImpl<Selector>  &ClassMethodSels,
           const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodTypes,
           const llvm::SmallVectorImpl<std::string> &Protocols);

  virtual void GenerateClass(
           const char *ClassName,
           const char *SuperClassName,
           const int instanceSize,
           const llvm::SmallVectorImpl<llvm::Constant *>  &IvarNames,
           const llvm::SmallVectorImpl<llvm::Constant *>  &IvarTypes,
           const llvm::SmallVectorImpl<llvm::Constant *>  &IvarOffsets,
           const llvm::SmallVectorImpl<Selector>  &InstanceMethodSels,
           const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodTypes,
           const llvm::SmallVectorImpl<Selector>  &ClassMethodSels,
           const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodTypes,
           const llvm::SmallVectorImpl<std::string> &Protocols);

  virtual llvm::Value *GenerateProtocolRef(llvm::IRBuilder<> &Builder,
                                           const ObjCProtocolDecl *PD);

  virtual void GenerateProtocol(const ObjCProtocolDecl *PD);

  virtual llvm::Function *ModuleInitFunction();
};
} // end anonymous namespace

/* *** Helper Functions *** */

/// getConstantGEP() - Help routine to construct simple GEPs.
static llvm::Constant *getConstantGEP(llvm::Constant *C, 
                                      unsigned idx0,
                                      unsigned idx1) {
  llvm::Value *Idxs[] = {
    llvm::ConstantInt::get(llvm::Type::Int32Ty, idx0),
    llvm::ConstantInt::get(llvm::Type::Int32Ty, idx1)
  };
  return llvm::ConstantExpr::getGetElementPtr(C, Idxs, 2);
}

/* *** CGObjCMac Public Interface *** */
 
CGObjCMac::CGObjCMac(CodeGen::CodeGenModule &cgm) 
  : CGM(cgm),
    ObjCTypes(cgm),
    ObjCABI(1)
{
  // FIXME: How does this get set in GCC? And what does it even mean?
  if (ObjCTypes.LongTy != CGM.getTypes().ConvertType(CGM.getContext().IntTy))
      ObjCABI = 2;

  EmitImageInfo();  
}

// This has to perform the lookup every time, since posing and related
// techniques can modify the name -> class mapping.
llvm::Value *CGObjCMac::LookupClass(llvm::IRBuilder<> &Builder,                                    
                                    llvm::Value *ClassName) {
  assert(0 && "Cannot lookup classes on Mac runtime.");
  return 0;
}

/// GetSelector - Return the pointer to the unique'd string for this selector.
llvm::Value *CGObjCMac::GetSelector(llvm::IRBuilder<> &Builder, Selector Sel) {
  return EmitSelector(Builder, Sel);
}

/// Generate a constant CFString object.
/* 
   struct __builtin_CFString {
     const int *isa; // point to __CFConstantStringClassReference
     int flags;
     const char *str;
     long length;
   };
*/

llvm::Constant *CGObjCMac::GenerateConstantString(const std::string &String) {
  // FIXME: I have no idea what this constant is (it is a magic
  // constant in GCC as well). Most likely the encoding of the string
  // and at least one part of it relates to UTF-16. Is this just the
  // code for UTF-8? Where is this handled for us?
  //  See: <rdr://2996215>
  unsigned flags = 0x07c8;

  // FIXME: Use some machinery to unique this. We can't reuse the CGM
  // one since we put them in a different section.
  llvm::Constant *StringC = llvm::ConstantArray::get(String);
  llvm::Constant *StringGV = 
    new llvm::GlobalVariable(StringC->getType(), true, 
                             llvm::GlobalValue::InternalLinkage,
                             StringC, ".str", &CGM.getModule());
  llvm::Constant *Values[4] = {
    ObjCTypes.getCFConstantStringClassReference(),
    llvm::ConstantInt::get(llvm::Type::Int32Ty, flags),
    getConstantGEP(StringGV, 0, 0), // Decay array -> ptr
    llvm::ConstantInt::get(ObjCTypes.LongTy, String.size())
  };

  llvm::Constant *CFStringC = 
    llvm::ConstantStruct::get(ObjCTypes.getCFStringType(), 
                              std::vector<llvm::Constant*>(Values, Values+4));

  llvm::GlobalVariable *CFStringGV = 
    new llvm::GlobalVariable(CFStringC->getType(), true,
                             llvm::GlobalValue::InternalLinkage,
                             CFStringC, "",
                             &CGM.getModule());

  CFStringGV->setSection("__DATA, __cfstring");

  return CFStringGV;
}

/// Generates a message send where the super is the receiver.  This is
/// a message send to self with special delivery semantics indicating
/// which class's method should be called.
llvm::Value *CGObjCMac::GenerateMessageSendSuper(llvm::IRBuilder<> &Builder,
                                                 const llvm::Type *ReturnTy,
                                                 const char *SuperClassName,
                                                 llvm::Value *Receiver,
                                                 Selector Sel,
                                                 llvm::Value** ArgV,
                                                 unsigned ArgC) {
  assert(0 && "Cannot generate message send to super for Mac runtime.");
  return 0;
}

/// Generate code for a message send expression.  
llvm::Value *CGObjCMac::GenerateMessageSend(llvm::IRBuilder<> &Builder,
                                            const llvm::Type *ReturnTy,
                                            llvm::Value *Receiver,
                                            Selector Sel,
                                            llvm::Value** ArgV,
                                            unsigned ArgC) {
  llvm::Function *F = ObjCTypes.getMessageSendFn();
  llvm::Value **Args = new llvm::Value*[ArgC+2];
  Args[0] = Builder.CreateBitCast(Receiver, ObjCTypes.ObjectPtrTy, "tmp");
  Args[1] = EmitSelector(Builder, Sel);
  std::copy(ArgV, ArgV+ArgC, Args+2);
  llvm::CallInst *CI = Builder.CreateCall(F, Args, Args+ArgC+2, "tmp");
  delete[] Args;
  return Builder.CreateBitCast(CI, ReturnTy, "tmp");
}

llvm::Value *CGObjCMac::GenerateProtocolRef(llvm::IRBuilder<> &Builder, 
                                            const ObjCProtocolDecl *PD) {
  //  assert(0 && "Cannot get protocol reference on Mac runtime.");
  return llvm::Constant::getNullValue(ObjCTypes.ProtocolPtrTy);
  return 0;
}

void CGObjCMac::GenerateProtocol(const ObjCProtocolDecl *PD) {
  //  assert(0 && "Cannot generate protocol for Mac runtime.");
}

void CGObjCMac::GenerateCategory(
           const char *ClassName,
           const char *CategoryName,
           const llvm::SmallVectorImpl<Selector>  &InstanceMethodSels,
           const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodTypes,
           const llvm::SmallVectorImpl<Selector>  &ClassMethodSels,
           const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodTypes,
           const llvm::SmallVectorImpl<std::string> &Protocols) {
  assert(0 && "Cannot generate category for Mac runtime.");
}

void CGObjCMac::GenerateClass(
           const char *ClassName,
           const char *SuperClassName,
           const int instanceSize,
           const llvm::SmallVectorImpl<llvm::Constant *>  &IvarNames,
           const llvm::SmallVectorImpl<llvm::Constant *>  &IvarTypes,
           const llvm::SmallVectorImpl<llvm::Constant *>  &IvarOffsets,
           const llvm::SmallVectorImpl<Selector>  &InstanceMethodSels,
           const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodTypes,
           const llvm::SmallVectorImpl<Selector>  &ClassMethodSels,
           const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodTypes,
           const llvm::SmallVectorImpl<std::string> &Protocols) {
  assert(0 && "Cannot generate class for Mac runtime.");
}

llvm::Function *CGObjCMac::ModuleInitFunction() { 
  // Abuse this interface function as a place to finalize.
  FinishModule();

  return NULL;
}

llvm::Function *CGObjCMac::MethodPreamble(const std::string &ClassName,
                                          const std::string &CategoryName,
                                          const std::string &MethodName,
                                          const llvm::Type *ReturnTy,
                                          const llvm::Type *SelfTy,
                                          const llvm::Type **ArgTy,
                                          unsigned ArgC,
                                          bool isClassMethod,
                                          bool isVarArg) {
  assert(0 && "Cannot generate method preamble for Mac runtime.");
  return 0;
}

/* *** Private Interface *** */

/// EmitImageInfo - Emit the image info marker used to encode some module
/// level information.
///
/// See: <rdr://4810609&4810587&4810587>
/// struct IMAGE_INFO {
///   unsigned version;
///   unsigned flags;
/// };
enum ImageInfoFlags {
  eImageInfo_FixAndContinue   = (1 << 0), // FIXME: Not sure what this implies
  eImageInfo_GarbageCollected = (1 << 1), 
  eImageInfo_GCOnly           = (1 << 2)  
};

void CGObjCMac::EmitImageInfo() {
  unsigned version = 0; // Version is unused?
  unsigned flags = 0;

  // FIXME: Fix and continue?
  if (CGM.getLangOptions().getGCMode() != LangOptions::NonGC)
    flags |= eImageInfo_GarbageCollected;
  if (CGM.getLangOptions().getGCMode() == LangOptions::GCOnly)
    flags |= eImageInfo_GCOnly;

  // Emitted as int[2];
  llvm::Constant *values[2] = {
    llvm::ConstantInt::get(llvm::Type::Int32Ty, version),
    llvm::ConstantInt::get(llvm::Type::Int32Ty, flags)
  };
  llvm::ArrayType *AT = llvm::ArrayType::get(llvm::Type::Int32Ty, 2);  
  llvm::GlobalVariable *GV = 
    new llvm::GlobalVariable(AT, true,
                             llvm::GlobalValue::InternalLinkage,
                             llvm::ConstantArray::get(AT, values, 2),
                             "\01L_OBJC_IMAGE_INFO", 
                             &CGM.getModule());

  if (ObjCABI == 1) {
    GV->setSection("__OBJC, __image_info,regular");
  } else {
    GV->setSection("__DATA, __objc_imageinfo, regular, no_dead_strip");
  }

  UsedGlobals.push_back(GV);
}


// struct objc_module {
//   unsigned long version;
//   unsigned long size;
//   const char *name;
//   Symtab symtab;
// };

// FIXME: Get from somewhere
static const int ModuleVersion = 7;

void CGObjCMac::EmitModuleInfo() {
  IdentifierInfo *EmptyIdent = &CGM.getContext().Idents.get("");
  Selector EmptySel = CGM.getContext().Selectors.getNullarySelector(EmptyIdent);
  uint64_t Size = CGM.getTargetData().getABITypeSize(ObjCTypes.ModuleTy);
  
  std::vector<llvm::Constant*> Values(4);
  Values[0] = llvm::ConstantInt::get(ObjCTypes.LongTy, ModuleVersion);
  Values[1] = llvm::ConstantInt::get(ObjCTypes.LongTy, Size);
    // FIXME: GCC just appears to make up an empty name for this? Why?
  Values[2] = getConstantGEP(GetClassName(EmptySel), 0, 0);
  Values[3] = EmitModuleSymbols();

  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(ObjCTypes.ModuleTy, false,
                             llvm::GlobalValue::InternalLinkage,
                             llvm::ConstantStruct::get(ObjCTypes.ModuleTy, 
                                                       Values),
                             "\01L_OBJC_MODULE_INFO", 
                             &CGM.getModule());
  GV->setSection("__OBJC,__module_info,regular,no_dead_strip");
  UsedGlobals.push_back(GV);
}

llvm::Constant *CGObjCMac::EmitModuleSymbols() {
  // FIXME: Is this ever used?
  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(ObjCTypes.SymtabTy, false,
                             llvm::GlobalValue::InternalLinkage,
                             llvm::Constant::getNullValue(ObjCTypes.SymtabTy),
                             "\01L_OBJC_SYMBOLS", 
                             &CGM.getModule());
  GV->setSection("__OBJC,__symbols,regular,no_dead_strip");
  UsedGlobals.push_back(GV);
  return GV;
}

llvm::Value *CGObjCMac::EmitSelector(llvm::IRBuilder<> &Builder, Selector Sel) {
  llvm::GlobalVariable *&Entry = SelectorReferences[Sel];
  
  if (!Entry) {
    llvm::Constant *Casted = 
      llvm::ConstantExpr::getBitCast(GetMethodVarName(Sel),
                                     ObjCTypes.SelectorPtrTy);
    Entry = 
      new llvm::GlobalVariable(ObjCTypes.SelectorPtrTy, false,
                               llvm::GlobalValue::InternalLinkage,
                               Casted, "\01L_OBJC_SELECTOR_REFERENCES_",
                               &CGM.getModule());
    Entry->setSection("__OBJC,__message_refs,literal_pointers,no_dead_strip");
    UsedGlobals.push_back(Entry);
  }

  return Builder.CreateLoad(Entry, false, "tmp");
}

llvm::Constant *CGObjCMac::GetClassName(Selector Sel) {
  llvm::GlobalVariable *&Entry = ClassNames[Sel];

  if (!Entry) {
    llvm::Constant *C = llvm::ConstantArray::get(Sel.getName());
    Entry = 
      new llvm::GlobalVariable(C->getType(), true, 
                               llvm::GlobalValue::InternalLinkage,
                               C, "\01L_OBJC_CLASS_NAME_", 
                               &CGM.getModule());
    Entry->setSection("__TEXT,__cstring,cstring_literals");
    UsedGlobals.push_back(Entry);
  }

  return Entry;
}

llvm::Constant *CGObjCMac::GetMethodVarName(Selector Sel) {
  llvm::GlobalVariable *&Entry = MethodVarNames[Sel];

  if (!Entry) {
    llvm::Constant *C = llvm::ConstantArray::get(Sel.getName());
    Entry = 
      new llvm::GlobalVariable(C->getType(), true, 
                               llvm::GlobalValue::InternalLinkage,
                               C, "\01L_OBJC_METH_VAR_NAME_", 
                               &CGM.getModule());
    Entry->setSection("__TEXT,__cstring,cstring_literals");
    UsedGlobals.push_back(Entry);
  }

  return Entry;
}

void CGObjCMac::FinishModule() {
  EmitModuleInfo();

  std::vector<llvm::Constant*> Used;

  llvm::Type *I8Ptr = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
  for (std::vector<llvm::GlobalVariable*>::iterator i = UsedGlobals.begin(), 
         e = UsedGlobals.end(); i != e; ++i) {
    Used.push_back(llvm::ConstantExpr::getBitCast(*i, I8Ptr));
  }
  
  llvm::ArrayType *AT = llvm::ArrayType::get(I8Ptr, Used.size());
  llvm::GlobalValue *GV = 
    new llvm::GlobalVariable(AT, false,
                             llvm::GlobalValue::AppendingLinkage,
                             llvm::ConstantArray::get(AT, Used),
                             "llvm.used", 
                             &CGM.getModule());

  GV->setSection("llvm.metadata");
}

/* *** */

ObjCTypesHelper::ObjCTypesHelper(CodeGen::CodeGenModule &cgm) 
  : CGM(cgm),
    CFStringType(0),
    CFConstantStringClassReference(0),
    MessageSendFn(0)
{
  CodeGen::CodeGenTypes &Types = CGM.getTypes();
  ASTContext &Ctx = CGM.getContext();
  
  LongTy = Types.ConvertType(Ctx.LongTy);
  ObjectPtrTy = Types.ConvertType(Ctx.getObjCIdType());
  SelectorPtrTy = Types.ConvertType(Ctx.getObjCSelType());
  ProtocolPtrTy = Types.ConvertType(Ctx.getObjCProtoType());

  SymtabTy = llvm::StructType::get(LongTy,
                                   SelectorPtrTy,
                                   Types.ConvertType(Ctx.ShortTy),
                                   Types.ConvertType(Ctx.ShortTy),
                                   NULL);
  CGM.getModule().addTypeName("struct._objc_symtab", SymtabTy);

  ModuleTy = 
    llvm::StructType::get(LongTy,
                          LongTy,
                          llvm::PointerType::getUnqual(llvm::Type::Int8Ty),
                          llvm::PointerType::getUnqual(SymtabTy),
                          NULL);
  CGM.getModule().addTypeName("struct._objc_module", ModuleTy);
}

ObjCTypesHelper::~ObjCTypesHelper() {
}

const llvm::StructType *ObjCTypesHelper::getCFStringType() {
  if (!CFStringType) {
    CFStringType = 
      llvm::StructType::get(llvm::PointerType::getUnqual(llvm::Type::Int32Ty), 
                            llvm::Type::Int32Ty,
                            llvm::PointerType::getUnqual(llvm::Type::Int8Ty),
                            LongTy,
                            NULL);

    CGM.getModule().addTypeName("struct.__builtin_CFString", CFStringType);
  }

  return CFStringType;
}

llvm::Constant *ObjCTypesHelper::getCFConstantStringClassReference() {
  if (!CFConstantStringClassReference) {
    llvm::GlobalValue *GV = 
      new llvm::GlobalVariable(llvm::ArrayType::get(llvm::Type::Int32Ty, 0), 
                               false,
                               llvm::GlobalValue::ExternalLinkage,
                               0, "__CFConstantStringClassReference", 
                                                     &CGM.getModule());

    // Decay to pointer.
    CFConstantStringClassReference = getConstantGEP(GV, 0, 0);
  }

  return CFConstantStringClassReference;
}

llvm::Function *ObjCTypesHelper::getMessageSendFn() {
  if (!MessageSendFn) {
    std::vector<const llvm::Type*> Params;
    Params.push_back(ObjectPtrTy);
    Params.push_back(SelectorPtrTy);
    MessageSendFn = llvm::Function::Create(llvm::FunctionType::get(ObjectPtrTy,
                                                                   Params,
                                                                   true),
                                           llvm::Function::ExternalLinkage,
                                           "objc_msgSend",
                                           &CGM.getModule());
  }

  return MessageSendFn;
}

/* *** */

CodeGen::CGObjCRuntime *CodeGen::CreateMacObjCRuntime(CodeGen::CodeGenModule &CGM){
  return new CGObjCMac(CGM);
}
