//===------- CGObjCGNU.cpp - Emit LLVM Code from ASTs for a Module --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides Objective-C code generation targetting the GNU runtime.  The
// class in this file generates structures used by the GNU Objective-C runtime
// library.  These structures are defined in objc/objc.h and objc/objc-api.h in
// the GNU runtime distribution.
//
//===----------------------------------------------------------------------===//

#include "CGObjCRuntime.h"
#include "llvm/Module.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <map>

using llvm::dyn_cast;

// The version of the runtime that this class targets.  Must match the version
// in the runtime.
static const int RuntimeVersion = 8;
static const int ProtocolVersion = 2;

namespace {
class CGObjCGNU : public clang::CodeGen::CGObjCRuntime {
private:
  llvm::Module &TheModule;
  const llvm::StructType *SelStructTy;
  const llvm::Type *SelectorTy;
  const llvm::Type *PtrToInt8Ty;
  const llvm::Type *IMPTy;
  const llvm::Type *IdTy;
  const llvm::Type *IntTy;
  const llvm::Type *PtrTy;
  const llvm::Type *LongTy;
  const llvm::Type *PtrToIntTy;
  std::vector<llvm::Constant*> Classes;
  std::vector<llvm::Constant*> Categories;
  std::vector<llvm::Constant*> ConstantStrings;
  llvm::Function *LoadFunction;
  llvm::StringMap<llvm::Constant*> ExistingProtocols;
  typedef std::pair<std::string, std::string> TypedSelector;
  std::map<TypedSelector, llvm::GlobalAlias*> TypedSelectors;
  llvm::StringMap<llvm::GlobalAlias*> UntypedSelectors;
  // Some zeros used for GEPs in lots of places.
  llvm::Constant *Zeros[2];
  llvm::Constant *NULLPtr;
private:
  llvm::Constant *GenerateIvarList(
      const llvm::SmallVectorImpl<llvm::Constant *>  &IvarNames,
      const llvm::SmallVectorImpl<llvm::Constant *>  &IvarTypes,
      const llvm::SmallVectorImpl<llvm::Constant *>  &IvarOffsets);
  llvm::Constant *GenerateMethodList(const std::string &ClassName,
      const std::string &CategoryName,
      const llvm::SmallVectorImpl<llvm::Constant *>  &MethodNames, 
      const llvm::SmallVectorImpl<llvm::Constant *>  &MethodTypes, 
      bool isClassMethodList);
  llvm::Constant *GenerateProtocolList(
      const llvm::SmallVectorImpl<std::string> &Protocols);
  llvm::Constant *GenerateClassStructure(
      llvm::Constant *MetaClass,
      llvm::Constant *SuperClass,
      unsigned info,
      llvm::Constant *Name,
      llvm::Constant *Version,
      llvm::Constant *InstanceSize,
      llvm::Constant *IVars,
      llvm::Constant *Methods,
      llvm::Constant *Protocols);
  llvm::Constant *GenerateProtocolMethodList(
      const llvm::SmallVectorImpl<llvm::Constant *>  &MethodNames,
      const llvm::SmallVectorImpl<llvm::Constant *>  &MethodTypes);
  llvm::Constant *MakeConstantString(const std::string &Str, const std::string
      &Name="");
  llvm::Constant *MakeGlobal(const llvm::StructType *Ty,
      std::vector<llvm::Constant*> &V, const std::string &Name="");
  llvm::Constant *MakeGlobal(const llvm::ArrayType *Ty,
      std::vector<llvm::Constant*> &V, const std::string &Name="");
public:
  CGObjCGNU(llvm::Module &Mp,
    const llvm::Type *LLVMIntType,
    const llvm::Type *LLVMLongType);
  virtual llvm::Constant *GenerateConstantString(const char *String, 
      const size_t length);
  virtual llvm::Value *GenerateMessageSend(llvm::IRBuilder &Builder,
                                           const llvm::Type *ReturnTy,
                                           llvm::Value *Sender,
                                           llvm::Value *Receiver,
                                           llvm::Value *Selector,
                                           llvm::Value** ArgV,
                                           unsigned ArgC);
  virtual llvm::Value *GenerateMessageSendSuper(llvm::IRBuilder &Builder,
                                            const llvm::Type *ReturnTy,
                                            llvm::Value *Sender,
                                            const char *SuperClassName,
                                            llvm::Value *Receiver,
                                            llvm::Value *Selector,
                                            llvm::Value** ArgV,
                                            unsigned ArgC);
  virtual llvm::Value *LookupClass(llvm::IRBuilder &Builder, llvm::Value
      *ClassName);
  virtual llvm::Value *GetSelector(llvm::IRBuilder &Builder,
      llvm::Value *SelName,
      llvm::Value *SelTypes);
  virtual llvm::Function *MethodPreamble(
                                         const std::string &ClassName,
                                         const std::string &CategoryName,
                                         const std::string &MethodName,
                                         const llvm::Type *ReturnTy,
                                         const llvm::Type *SelfTy,
                                         const llvm::Type **ArgTy,
                                         unsigned ArgC,
                                         bool isClassMethod,
                                         bool isVarArg);
  virtual void GenerateCategory(const char *ClassName, const char *CategoryName,
           const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodNames,
           const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodTypes,
           const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodNames,
           const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodTypes,
           const llvm::SmallVectorImpl<std::string> &Protocols);
  virtual void GenerateClass(
           const char *ClassName,
           const char *SuperClassName,
           const int instanceSize,
           const llvm::SmallVectorImpl<llvm::Constant *>  &IvarNames,
           const llvm::SmallVectorImpl<llvm::Constant *>  &IvarTypes,
           const llvm::SmallVectorImpl<llvm::Constant *>  &IvarOffsets,
           const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodNames,
           const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodTypes,
           const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodNames,
           const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodTypes,
           const llvm::SmallVectorImpl<std::string> &Protocols);
  virtual llvm::Value *GenerateProtocolRef(llvm::IRBuilder &Builder, const char
      *ProtocolName);
  virtual void GenerateProtocol(const char *ProtocolName,
      const llvm::SmallVectorImpl<std::string> &Protocols,
      const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodNames,
      const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodTypes,
      const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodNames,
      const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodTypes);
  virtual llvm::Function *ModuleInitFunction();
};
} // end anonymous namespace



static std::string SymbolNameForClass(const std::string &ClassName) {
  return ".objc_class_" + ClassName;
}

static std::string SymbolNameForMethod(const std::string &ClassName, const
  std::string &CategoryName, const std::string &MethodName, bool isClassMethod)
{
  return "._objc_method_" + ClassName +"("+CategoryName+")"+
            (isClassMethod ? "+" : "-") + MethodName;
}

CGObjCGNU::CGObjCGNU(llvm::Module &M,
    const llvm::Type *LLVMIntType,
    const llvm::Type *LLVMLongType) : 
  TheModule(M),
  IntTy(LLVMIntType),
  LongTy(LLVMLongType)
{
  Zeros[0] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
  Zeros[1] = Zeros[0];
  NULLPtr = llvm::ConstantPointerNull::get(
    llvm::PointerType::getUnqual(llvm::Type::Int8Ty));
  // C string type.  Used in lots of places.
  PtrToInt8Ty = 
    llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
  // Get the selector Type.
  SelStructTy = llvm::StructType::get(
      PtrToInt8Ty,
      PtrToInt8Ty,
      NULL);
  SelectorTy = llvm::PointerType::getUnqual(SelStructTy);
  PtrToIntTy = llvm::PointerType::getUnqual(IntTy);
  PtrTy = PtrToInt8Ty;
 
  // Object type
  llvm::PATypeHolder OpaqueObjTy = llvm::OpaqueType::get();
  llvm::Type *OpaqueIdTy = llvm::PointerType::getUnqual(OpaqueObjTy);
  IdTy = llvm::StructType::get(OpaqueIdTy, NULL);
  llvm::cast<llvm::OpaqueType>(OpaqueObjTy.get())->refineAbstractTypeTo(IdTy);
  IdTy = llvm::cast<llvm::StructType>(OpaqueObjTy.get());
  IdTy = llvm::PointerType::getUnqual(IdTy);
 
  // IMP type
  std::vector<const llvm::Type*> IMPArgs;
  IMPArgs.push_back(IdTy);
  IMPArgs.push_back(SelectorTy);
  IMPTy = llvm::FunctionType::get(IdTy, IMPArgs, true);
}
// This has to perform the lookup every time, since posing and related
// techniques can modify the name -> class mapping.
llvm::Value *CGObjCGNU::LookupClass(llvm::IRBuilder &Builder,
    llvm::Value *ClassName) {
  llvm::Constant *ClassLookupFn =
    TheModule.getOrInsertFunction("objc_lookup_class", IdTy, PtrToInt8Ty,
        NULL);
  return Builder.CreateCall(ClassLookupFn, ClassName);
}

/// Looks up the selector for the specified name / type pair.
// FIXME: Selectors should be statically cached, not looked up on every call.
llvm::Value *CGObjCGNU::GetSelector(llvm::IRBuilder &Builder,
    llvm::Value *SelName,
    llvm::Value *SelTypes) {
  // For static selectors, we return an alias for now then store them all in a
  // list that the runtime will initialise later.
  if (llvm::Constant *CName = dyn_cast<llvm::Constant>(SelName)) {
    // Untyped selector
    if (SelTypes == 0) {
      // If it's already cached, return it.
      if (UntypedSelectors[CName->getStringValue()]) {
        // FIXME: Volatility
        return Builder.CreateLoad(UntypedSelectors[CName->getStringValue()]);
      }
      // If it isn't, cache it.
      llvm::GlobalAlias *Sel = new llvm::GlobalAlias(
          llvm::PointerType::getUnqual(SelectorTy),
          llvm::GlobalValue::InternalLinkage, ".objc_untyped_selector_alias",
          NULL, &TheModule);
      UntypedSelectors[CName->getStringValue()] = Sel;
      // FIXME: Volatility
      return Builder.CreateLoad(Sel);
    }
    // Typed selectors
    if (llvm::Constant *CTypes = dyn_cast<llvm::Constant>(SelTypes)) {
      TypedSelector Selector = TypedSelector(CName->getStringValue(),
          CTypes->getStringValue());
      // If it's already cached, return it.
      if (TypedSelectors[Selector]) {
        // FIXME: Volatility
        return Builder.CreateLoad(TypedSelectors[Selector]);
      }
      // If it isn't, cache it.
      llvm::GlobalAlias *Sel = new llvm::GlobalAlias(
          llvm::PointerType::getUnqual(SelectorTy),
          llvm::GlobalValue::InternalLinkage, ".objc_typed_selector_alias",
          NULL, &TheModule);
      TypedSelectors[Selector] = Sel;
      // FIXME: Volatility
      return Builder.CreateLoad(Sel);
    }
  }
  // Dynamically look up selectors from non-constant sources
  llvm::Value *cmd;
  if (SelTypes == 0) {
    llvm::Constant *SelFunction = TheModule.getOrInsertFunction("sel_get_uid", 
        SelectorTy, 
        PtrToInt8Ty, 
        NULL);
    cmd = Builder.CreateCall(SelFunction, SelName);
  }
  else {
    llvm::Constant *SelFunction = 
      TheModule.getOrInsertFunction("sel_get_typed_uid",
          SelectorTy,
          PtrToInt8Ty,
          PtrToInt8Ty,
          NULL);
    cmd = Builder.CreateCall2(SelFunction, SelName, SelTypes);
  }
  return cmd;
}


llvm::Constant *CGObjCGNU::MakeConstantString(const std::string &Str, const
    std::string &Name) {
  llvm::Constant * ConstStr = llvm::ConstantArray::get(Str);
  ConstStr = new llvm::GlobalVariable(ConstStr->getType(), true, 
                               llvm::GlobalValue::InternalLinkage,
                               ConstStr, Name, &TheModule);
  return llvm::ConstantExpr::getGetElementPtr(ConstStr, Zeros, 2);
}
llvm::Constant *CGObjCGNU::MakeGlobal(const llvm::StructType *Ty,
    std::vector<llvm::Constant*> &V, const std::string &Name) {
  llvm::Constant *C = llvm::ConstantStruct::get(Ty, V);
  return new llvm::GlobalVariable(Ty, false,
      llvm::GlobalValue::InternalLinkage, C, Name, &TheModule);
}
llvm::Constant *CGObjCGNU::MakeGlobal(const llvm::ArrayType *Ty,
    std::vector<llvm::Constant*> &V, const std::string &Name) {
  llvm::Constant *C = llvm::ConstantArray::get(Ty, V);
  return new llvm::GlobalVariable(Ty, false,
      llvm::GlobalValue::InternalLinkage, C, Name, &TheModule);
}

/// Generate an NSConstantString object.
//TODO: In case there are any crazy people still using the GNU runtime without
//an OpenStep implementation, this should let them select their own class for
//constant strings.
llvm::Constant *CGObjCGNU::GenerateConstantString(const char *String, const
    size_t length) {
  std::string Str(String, String +length);
  std::vector<llvm::Constant*> Ivars;
  Ivars.push_back(NULLPtr);
  Ivars.push_back(MakeConstantString(Str));
  Ivars.push_back(llvm::ConstantInt::get(IntTy, length));
  llvm::Constant *ObjCStr = MakeGlobal(
    llvm::StructType::get(PtrToInt8Ty, PtrToInt8Ty, IntTy, NULL),
    Ivars, ".objc_str");
  ConstantStrings.push_back(
      llvm::ConstantExpr::getBitCast(ObjCStr, PtrToInt8Ty));
  return ObjCStr;
}

///Generates a message send where the super is the receiver.  This is a message
///send to self with special delivery semantics indicating which class's method
///should be called.
llvm::Value *CGObjCGNU::GenerateMessageSendSuper(llvm::IRBuilder &Builder,
                                            const llvm::Type *ReturnTy,
                                            llvm::Value *Sender,
                                            const char *SuperClassName,
                                            llvm::Value *Receiver,
                                            llvm::Value *Selector,
                                            llvm::Value** ArgV,
                                            unsigned ArgC) {
  // TODO: This should be cached, not looked up every time.
  llvm::Value *ReceiverClass = LookupClass(Builder,
      MakeConstantString(SuperClassName));
  llvm::Value *cmd = GetSelector(Builder, Selector, 0);
  std::vector<const llvm::Type*> impArgTypes;
  impArgTypes.push_back(Receiver->getType());
  impArgTypes.push_back(SelectorTy);
  
  // Avoid an explicit cast on the IMP by getting a version that has the right
  // return type.
  llvm::FunctionType *impType = llvm::FunctionType::get(ReturnTy, impArgTypes,
                                                        true);
  // Construct the structure used to look up the IMP
  llvm::StructType *ObjCSuperTy = llvm::StructType::get(Receiver->getType(),
      IdTy, NULL);
  llvm::Value *ObjCSuper = Builder.CreateAlloca(ObjCSuperTy);
  // FIXME: volatility
  Builder.CreateStore(Receiver, Builder.CreateStructGEP(ObjCSuper, 0));
  Builder.CreateStore(ReceiverClass, Builder.CreateStructGEP(ObjCSuper, 1));

  // Get the IMP
  llvm::Constant *lookupFunction = 
     TheModule.getOrInsertFunction("objc_msg_lookup_super",
                                   llvm::PointerType::getUnqual(impType),
                                   llvm::PointerType::getUnqual(ObjCSuperTy),
                                   SelectorTy, NULL);
  llvm::Value *lookupArgs[] = {ObjCSuper, cmd};
  llvm::Value *imp = Builder.CreateCall(lookupFunction, lookupArgs,
      lookupArgs+2);

  // Call the method
  llvm::SmallVector<llvm::Value*, 8> callArgs;
  callArgs.push_back(Receiver);
  callArgs.push_back(cmd);
  callArgs.insert(callArgs.end(), ArgV, ArgV+ArgC);
  return Builder.CreateCall(imp, callArgs.begin(), callArgs.end());
}

/// Generate code for a message send expression.  
llvm::Value *CGObjCGNU::GenerateMessageSend(llvm::IRBuilder &Builder,
                                            const llvm::Type *ReturnTy,
                                            llvm::Value *Sender,
                                            llvm::Value *Receiver,
                                            llvm::Value *Selector,
                                            llvm::Value** ArgV,
                                            unsigned ArgC) {
  llvm::Value *cmd = GetSelector(Builder, Selector, 0);

  // Look up the method implementation.
  std::vector<const llvm::Type*> impArgTypes;
  const llvm::Type *RetTy;
  //TODO: Revisit this when LLVM supports aggregate return types.
  if (ReturnTy->isSingleValueType() && ReturnTy != llvm::Type::VoidTy) {
    RetTy = ReturnTy;
  } else {
    // For struct returns allocate the space in the caller and pass it up to
    // the sender.
    RetTy = llvm::Type::VoidTy;
    impArgTypes.push_back(llvm::PointerType::getUnqual(ReturnTy));
  }
  impArgTypes.push_back(Receiver->getType());
  impArgTypes.push_back(SelectorTy);
  
  // Avoid an explicit cast on the IMP by getting a version that has the right
  // return type.
  llvm::FunctionType *impType = llvm::FunctionType::get(RetTy, impArgTypes,
                                                        true);
  
  llvm::Constant *lookupFunction = 
     TheModule.getOrInsertFunction("objc_msg_lookup",
                                   llvm::PointerType::getUnqual(impType),
                                   Receiver->getType(), SelectorTy, NULL);
  llvm::Value *imp = Builder.CreateCall2(lookupFunction, Receiver, cmd);

  // Call the method.
  llvm::SmallVector<llvm::Value*, 16> Args;
  if (!ReturnTy->isSingleValueType()) {
    llvm::Value *Return = Builder.CreateAlloca(ReturnTy);
    Args.push_back(Return);
  }
  Args.push_back(Receiver);
  Args.push_back(cmd);
  Args.insert(Args.end(), ArgV, ArgV+ArgC);
  if (!ReturnTy->isSingleValueType()) {
    Builder.CreateCall(imp, Args.begin(), Args.end());
    return Args[0];
  }
  return Builder.CreateCall(imp, Args.begin(), Args.end());
}

/// Generates a MethodList.  Used in construction of a objc_class and 
/// objc_category structures.
llvm::Constant *CGObjCGNU::GenerateMethodList(const std::string &ClassName,
    const std::string &CategoryName, 
    const llvm::SmallVectorImpl<llvm::Constant *> &MethodNames, 
    const llvm::SmallVectorImpl<llvm::Constant *> &MethodTypes, 
    bool isClassMethodList) {
  // Get the method structure type.  
  llvm::StructType *ObjCMethodTy = llvm::StructType::get(
    PtrToInt8Ty, // Really a selector, but the runtime creates it us.
    PtrToInt8Ty, // Method types
    llvm::PointerType::getUnqual(IMPTy), //Method pointer
    NULL);
  std::vector<llvm::Constant*> Methods;
  std::vector<llvm::Constant*> Elements;
  for (unsigned int i = 0, e = MethodTypes.size(); i < e; ++i) {
    Elements.clear();
    Elements.push_back( llvm::ConstantExpr::getGetElementPtr(MethodNames[i],
          Zeros, 2));
    Elements.push_back(
          llvm::ConstantExpr::getGetElementPtr(MethodTypes[i], Zeros, 2));
    llvm::Constant *Method =
      TheModule.getFunction(SymbolNameForMethod(ClassName, CategoryName,
            MethodNames[i]->getStringValue(), isClassMethodList));
    Method = llvm::ConstantExpr::getBitCast(Method,
        llvm::PointerType::getUnqual(IMPTy));
    Elements.push_back(Method);
    Methods.push_back(llvm::ConstantStruct::get(ObjCMethodTy, Elements));
  }

  // Array of method structures
  llvm::ArrayType *ObjCMethodArrayTy = llvm::ArrayType::get(ObjCMethodTy,
      MethodNames.size());
  llvm::Constant *MethodArray = llvm::ConstantArray::get(ObjCMethodArrayTy,
      Methods);

  // Structure containing list pointer, array and array count
  llvm::SmallVector<const llvm::Type*, 16> ObjCMethodListFields;
  llvm::PATypeHolder OpaqueNextTy = llvm::OpaqueType::get();
  llvm::Type *NextPtrTy = llvm::PointerType::getUnqual(OpaqueNextTy);
  llvm::StructType *ObjCMethodListTy = llvm::StructType::get(NextPtrTy, 
      IntTy, 
      ObjCMethodArrayTy,
      NULL);
  // Refine next pointer type to concrete type
  llvm::cast<llvm::OpaqueType>(
      OpaqueNextTy.get())->refineAbstractTypeTo(ObjCMethodListTy);
  ObjCMethodListTy = llvm::cast<llvm::StructType>(OpaqueNextTy.get());

  Methods.clear();
  Methods.push_back(llvm::ConstantPointerNull::get(
        llvm::PointerType::getUnqual(ObjCMethodListTy)));
  Methods.push_back(llvm::ConstantInt::get(llvm::Type::Int32Ty,
        MethodTypes.size()));
  Methods.push_back(MethodArray);
  
  // Create an instance of the structure
  return MakeGlobal(ObjCMethodListTy, Methods, ".objc_method_list");
}

/// Generates an IvarList.  Used in construction of a objc_class.
llvm::Constant *CGObjCGNU::GenerateIvarList(
    const llvm::SmallVectorImpl<llvm::Constant *>  &IvarNames,
    const llvm::SmallVectorImpl<llvm::Constant *>  &IvarTypes,
    const llvm::SmallVectorImpl<llvm::Constant *>  &IvarOffsets) {
  // Get the method structure type.  
  llvm::StructType *ObjCIvarTy = llvm::StructType::get(
    PtrToInt8Ty,
    PtrToInt8Ty,
    IntTy,
    NULL);
  std::vector<llvm::Constant*> Ivars;
  std::vector<llvm::Constant*> Elements;
  for (unsigned int i = 0, e = IvarNames.size() ; i < e ; i++) {
    Elements.clear();
    Elements.push_back( llvm::ConstantExpr::getGetElementPtr(IvarNames[i],
          Zeros, 2));
    Elements.push_back( llvm::ConstantExpr::getGetElementPtr(IvarTypes[i],
          Zeros, 2));
    Elements.push_back(IvarOffsets[i]);
    Ivars.push_back(llvm::ConstantStruct::get(ObjCIvarTy, Elements));
  }

  // Array of method structures
  llvm::ArrayType *ObjCIvarArrayTy = llvm::ArrayType::get(ObjCIvarTy,
      IvarNames.size());

  
  Elements.clear();
  Elements.push_back(llvm::ConstantInt::get(
        llvm::cast<llvm::IntegerType>(IntTy), (int)IvarNames.size()));
  Elements.push_back(llvm::ConstantArray::get(ObjCIvarArrayTy, Ivars));
  // Structure containing array and array count
  llvm::StructType *ObjCIvarListTy = llvm::StructType::get(IntTy,
    ObjCIvarArrayTy,
    NULL);

  // Create an instance of the structure
  return MakeGlobal(ObjCIvarListTy, Elements, ".objc_ivar_list");
}

/// Generate a class structure
llvm::Constant *CGObjCGNU::GenerateClassStructure(
    llvm::Constant *MetaClass,
    llvm::Constant *SuperClass,
    unsigned info,
    llvm::Constant *Name,
    llvm::Constant *Version,
    llvm::Constant *InstanceSize,
    llvm::Constant *IVars,
    llvm::Constant *Methods,
    llvm::Constant *Protocols) {
  // Set up the class structure
  // Note:  Several of these are char*s when they should be ids.  This is
  // because the runtime performs this translation on load.
  llvm::StructType *ClassTy = llvm::StructType::get(
      PtrToInt8Ty,        // class_pointer
      PtrToInt8Ty,        // super_class
      PtrToInt8Ty,        // name
      LongTy,             // version
      LongTy,             // info
      LongTy,             // instance_size
      IVars->getType(),   // ivars
      Methods->getType(), // methods
      // These are all filled in by the runtime, so we pretend 
      PtrTy,              // dtable
      PtrTy,              // subclass_list
      PtrTy,              // sibling_class
      PtrTy,              // protocols
      PtrTy,              // gc_object_type
      NULL);
  llvm::Constant *Zero = llvm::ConstantInt::get(LongTy, 0);
  llvm::Constant *NullP =
    llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(PtrTy));
  // Fill in the structure
  std::vector<llvm::Constant*> Elements;
  Elements.push_back(llvm::ConstantExpr::getBitCast(MetaClass, PtrToInt8Ty));
  Elements.push_back(SuperClass);
  Elements.push_back(Name);
  Elements.push_back(Zero);
  Elements.push_back(llvm::ConstantInt::get(LongTy, info));
  Elements.push_back(InstanceSize);
  Elements.push_back(IVars);
  Elements.push_back(Methods);
  Elements.push_back(NullP);
  Elements.push_back(NullP);
  Elements.push_back(NullP);
  Elements.push_back(llvm::ConstantExpr::getBitCast(Protocols, PtrTy));
  Elements.push_back(NullP);
  // Create an instance of the structure
  return MakeGlobal(ClassTy, Elements,
      SymbolNameForClass(Name->getStringValue()));
}

llvm::Constant *CGObjCGNU::GenerateProtocolMethodList(
    const llvm::SmallVectorImpl<llvm::Constant *>  &MethodNames,
    const llvm::SmallVectorImpl<llvm::Constant *>  &MethodTypes) {
  // Get the method structure type.  
  llvm::StructType *ObjCMethodDescTy = llvm::StructType::get(
    PtrToInt8Ty, // Really a selector, but the runtime does the casting for us.
    PtrToInt8Ty,
    NULL);
  std::vector<llvm::Constant*> Methods;
  std::vector<llvm::Constant*> Elements;
  for (unsigned int i = 0, e = MethodTypes.size() ; i < e ; i++) {
    Elements.clear();
    Elements.push_back( llvm::ConstantExpr::getGetElementPtr(MethodNames[i],
          Zeros, 2)); 
    Elements.push_back(
          llvm::ConstantExpr::getGetElementPtr(MethodTypes[i], Zeros, 2));
    Methods.push_back(llvm::ConstantStruct::get(ObjCMethodDescTy, Elements));
  }
  llvm::ArrayType *ObjCMethodArrayTy = llvm::ArrayType::get(ObjCMethodDescTy,
      MethodNames.size());
  llvm::Constant *Array = llvm::ConstantArray::get(ObjCMethodArrayTy, Methods);
  llvm::StructType *ObjCMethodDescListTy = llvm::StructType::get(
      IntTy, ObjCMethodArrayTy, NULL);
  Methods.clear();
  Methods.push_back(llvm::ConstantInt::get(IntTy, MethodNames.size()));
  Methods.push_back(Array);
  return MakeGlobal(ObjCMethodDescListTy, Methods, ".objc_method_list");
}
// Create the protocol list structure used in classes, categories and so on
llvm::Constant *CGObjCGNU::GenerateProtocolList(
    const llvm::SmallVectorImpl<std::string> &Protocols) {
  llvm::ArrayType *ProtocolArrayTy = llvm::ArrayType::get(PtrToInt8Ty,
      Protocols.size());
  llvm::StructType *ProtocolListTy = llvm::StructType::get(
      PtrTy, //Should be a recurisve pointer, but it's always NULL here.
      LongTy,//FIXME: Should be size_t
      ProtocolArrayTy,
      NULL);
  std::vector<llvm::Constant*> Elements; 
  for (const std::string *iter = Protocols.begin(), *endIter = Protocols.end();
      iter != endIter ; iter++) {
    llvm::Constant *Ptr =
      llvm::ConstantExpr::getBitCast(ExistingProtocols[*iter], PtrToInt8Ty);
    Elements.push_back(Ptr);
  }
  llvm::Constant * ProtocolArray = llvm::ConstantArray::get(ProtocolArrayTy,
      Elements);
  Elements.clear();
  Elements.push_back(NULLPtr);
  Elements.push_back(llvm::ConstantInt::get(
        llvm::cast<llvm::IntegerType>(LongTy), Protocols.size()));
  Elements.push_back(ProtocolArray);
  return MakeGlobal(ProtocolListTy, Elements, ".objc_protocol_list");
}

llvm::Value *CGObjCGNU::GenerateProtocolRef(llvm::IRBuilder &Builder, const
    char *ProtocolName) {
  return ExistingProtocols[ProtocolName];
}

void CGObjCGNU::GenerateProtocol(const char *ProtocolName,
    const llvm::SmallVectorImpl<std::string> &Protocols,
    const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodNames,
    const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodTypes,
    const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodNames,
    const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodTypes) {

  llvm::Constant *ProtocolList = GenerateProtocolList(Protocols);
  llvm::Constant *InstanceMethodList =
    GenerateProtocolMethodList(InstanceMethodNames, InstanceMethodTypes);
  llvm::Constant *ClassMethodList =
    GenerateProtocolMethodList(ClassMethodNames, ClassMethodTypes);
  // Protocols are objects containing lists of the methods implemented and
  // protocols adopted.
  llvm::StructType *ProtocolTy = llvm::StructType::get(IdTy,
      PtrToInt8Ty,
      ProtocolList->getType(),
      InstanceMethodList->getType(),
      ClassMethodList->getType(),
      NULL);
  std::vector<llvm::Constant*> Elements; 
  // The isa pointer must be set to a magic number so the runtime knows it's
  // the correct layout.
  Elements.push_back(llvm::ConstantExpr::getIntToPtr(
        llvm::ConstantInt::get(llvm::Type::Int32Ty, ProtocolVersion), IdTy));
  Elements.push_back(MakeConstantString(ProtocolName, ".objc_protocol_name"));
  Elements.push_back(ProtocolList);
  Elements.push_back(InstanceMethodList);
  Elements.push_back(ClassMethodList);
  ExistingProtocols[ProtocolName] = 
    llvm::ConstantExpr::getBitCast(MakeGlobal(ProtocolTy, Elements,
          ".objc_protocol"), IdTy);
}

void CGObjCGNU::GenerateCategory(
           const char *ClassName,
           const char *CategoryName,
           const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodNames,
           const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodTypes,
           const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodNames,
           const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodTypes,
           const llvm::SmallVectorImpl<std::string> &Protocols) {
  std::vector<llvm::Constant*> Elements;
  Elements.push_back(MakeConstantString(CategoryName));
  Elements.push_back(MakeConstantString(ClassName));
  // Instance method list 
  Elements.push_back(llvm::ConstantExpr::getBitCast(GenerateMethodList(
          ClassName, CategoryName, InstanceMethodNames, InstanceMethodTypes,
          false), PtrTy));
  // Class method list
  Elements.push_back(llvm::ConstantExpr::getBitCast(GenerateMethodList(
          ClassName, CategoryName, ClassMethodNames, ClassMethodTypes, true),
        PtrTy));
  // Protocol list
  Elements.push_back(llvm::ConstantExpr::getBitCast(
        GenerateProtocolList(Protocols), PtrTy));
  Categories.push_back(llvm::ConstantExpr::getBitCast(
        MakeGlobal(llvm::StructType::get(PtrToInt8Ty, PtrToInt8Ty, PtrTy,
            PtrTy, PtrTy, NULL), Elements), PtrTy));
}
void CGObjCGNU::GenerateClass(
           const char *ClassName,
           const char *SuperClassName,
           const int instanceSize,
           const llvm::SmallVectorImpl<llvm::Constant *>  &IvarNames,
           const llvm::SmallVectorImpl<llvm::Constant *>  &IvarTypes,
           const llvm::SmallVectorImpl<llvm::Constant *>  &IvarOffsets,
           const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodNames,
           const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodTypes,
           const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodNames,
           const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodTypes,
           const llvm::SmallVectorImpl<std::string> &Protocols) {
  // Get the superclass pointer.
  llvm::Constant *SuperClass;
  if (SuperClassName) {
    SuperClass = MakeConstantString(SuperClassName, ".super_class_name");
  } else {
    SuperClass = llvm::ConstantPointerNull::get(
        llvm::cast<llvm::PointerType>(PtrToInt8Ty));
  }
  llvm::Constant * Name = MakeConstantString(ClassName, ".class_name");
  // Empty vector used to construct empty method lists
  llvm::SmallVector<llvm::Constant*, 1>  empty;
  // Generate the method and instance variable lists
  llvm::Constant *MethodList = GenerateMethodList(ClassName, "",
      InstanceMethodNames, InstanceMethodTypes, false);
  llvm::Constant *ClassMethodList = GenerateMethodList(ClassName, "",
      ClassMethodNames, ClassMethodTypes, true);
  llvm::Constant *IvarList = GenerateIvarList(IvarNames, IvarTypes,
      IvarOffsets);
  //Generate metaclass for class methods
  llvm::Constant *MetaClassStruct = GenerateClassStructure(NULLPtr,
      NULLPtr, 0x2L, NULLPtr, 0, Zeros[0], GenerateIvarList(
        empty, empty, empty), ClassMethodList, NULLPtr);
  // Generate the class structure
  llvm::Constant *ClassStruct = GenerateClassStructure(MetaClassStruct,
      SuperClass, 0x1L, Name, 0,
      llvm::ConstantInt::get(llvm::Type::Int32Ty, instanceSize), IvarList,
      MethodList, GenerateProtocolList(Protocols));
  // Add class structure to list to be added to the symtab later
  ClassStruct = llvm::ConstantExpr::getBitCast(ClassStruct, PtrToInt8Ty);
  Classes.push_back(ClassStruct);
}

llvm::Function *CGObjCGNU::ModuleInitFunction() { 
  // Only emit an ObjC load function if no Objective-C stuff has been called
  if (Classes.empty() && Categories.empty() && ConstantStrings.empty() &&
      ExistingProtocols.empty() && TypedSelectors.empty() &&
      UntypedSelectors.empty())
    return NULL;

  // Name the ObjC types to make the IR a bit easier to read
  TheModule.addTypeName(".objc_selector", SelectorTy);
  TheModule.addTypeName(".objc_id", IdTy);
  TheModule.addTypeName(".objc_imp", IMPTy);

  std::vector<llvm::Constant*> Elements;
  // Generate statics list:
  llvm::ArrayType *StaticsArrayTy = llvm::ArrayType::get(PtrToInt8Ty,
      ConstantStrings.size() + 1);
  ConstantStrings.push_back(NULLPtr);
  Elements.push_back(MakeConstantString("NSConstantString",
        ".objc_static_class_name"));
  Elements.push_back(llvm::ConstantArray::get(StaticsArrayTy, ConstantStrings));
  llvm::StructType *StaticsListTy = 
    llvm::StructType::get(PtrToInt8Ty, StaticsArrayTy, NULL);
  llvm::Constant *Statics = 
    MakeGlobal(StaticsListTy, Elements, ".objc_statics");
  llvm::ArrayType *StaticsListArrayTy =
    llvm::ArrayType::get(llvm::PointerType::getUnqual(StaticsListTy), 2);
  Elements.clear();
  Elements.push_back(Statics);
  Elements.push_back(llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(StaticsListTy)));
  Statics = MakeGlobal(StaticsListArrayTy, Elements, ".objc_statics_ptr");
  Statics = llvm::ConstantExpr::getBitCast(Statics, PtrTy);
  // Array of classes, categories, and constant objects
  llvm::ArrayType *ClassListTy = llvm::ArrayType::get(PtrToInt8Ty,
      Classes.size() + Categories.size()  + 2);
  llvm::StructType *SymTabTy = llvm::StructType::get(
      LongTy,
      SelectorTy,
      llvm::Type::Int16Ty,
      llvm::Type::Int16Ty,
      ClassListTy,
      NULL);

  Elements.clear();
  // Pointer to an array of selectors used in this module.
  std::vector<llvm::Constant*> Selectors;
  for (std::map<TypedSelector, llvm::GlobalAlias*>::iterator
     iter = TypedSelectors.begin(), iterEnd = TypedSelectors.end();
     iter != iterEnd ; ++iter) {
    Elements.push_back(MakeConstantString((*iter).first.first,
          ".objc_sel_name"));
    Elements.push_back(MakeConstantString((*iter).first.second,
          ".objc_sel_types"));
    Selectors.push_back(llvm::ConstantStruct::get(SelStructTy, Elements));
    Elements.clear();
  }
  for (llvm::StringMap<llvm::GlobalAlias*>::iterator
      iter = UntypedSelectors.begin(), iterEnd = UntypedSelectors.end();
      iter != iterEnd; iter++) {
    Elements.push_back(
        MakeConstantString((*iter).getKeyData(), ".objc_sel_name"));
    Elements.push_back(NULLPtr);
    Selectors.push_back(llvm::ConstantStruct::get(SelStructTy, Elements));
    Elements.clear();
  }
  Elements.push_back(NULLPtr);
  Elements.push_back(NULLPtr);
  Selectors.push_back(llvm::ConstantStruct::get(SelStructTy, Elements));
  Elements.clear();
  // Number of static selectors
  Elements.push_back(llvm::ConstantInt::get(LongTy, Selectors.size() ));
  llvm::Constant *SelectorList = MakeGlobal(
          llvm::ArrayType::get(SelStructTy, Selectors.size()), Selectors,
          ".objc_selector_list");
  Elements.push_back(llvm::ConstantExpr::getBitCast(SelectorList, SelectorTy));

  // Now that all of the static selectors exist, create pointers to them.
  int index = 0;
  for (std::map<TypedSelector, llvm::GlobalAlias*>::iterator
     iter=TypedSelectors.begin(), iterEnd =TypedSelectors.end();
     iter != iterEnd; ++iter) {
    llvm::Constant *Idxs[] = {Zeros[0],
      llvm::ConstantInt::get(llvm::Type::Int32Ty, index++), Zeros[0]};
    llvm::GlobalVariable *SelPtr = new llvm::GlobalVariable(SelectorTy, true,
        llvm::GlobalValue::InternalLinkage,
        llvm::ConstantExpr::getGetElementPtr(SelectorList, Idxs, 2),
        ".objc_sel_ptr", &TheModule);
    (*iter).second->setAliasee(SelPtr);
  }
  for (llvm::StringMap<llvm::GlobalAlias*>::iterator
      iter=UntypedSelectors.begin(), iterEnd = UntypedSelectors.end();
      iter != iterEnd; iter++) {
    llvm::Constant *Idxs[] = {Zeros[0],
      llvm::ConstantInt::get(llvm::Type::Int32Ty, index++), Zeros[0]};
    llvm::GlobalVariable *SelPtr = new llvm::GlobalVariable(SelectorTy, true,
        llvm::GlobalValue::InternalLinkage,
        llvm::ConstantExpr::getGetElementPtr(SelectorList, Idxs, 2),
        ".objc_sel_ptr", &TheModule);
    (*iter).second->setAliasee(SelPtr);
  }
  // Number of classes defined.
  Elements.push_back(llvm::ConstantInt::get(llvm::Type::Int16Ty, 
        Classes.size()));
  // Number of categories defined
  Elements.push_back(llvm::ConstantInt::get(llvm::Type::Int16Ty, 
        Categories.size()));
  // Create an array of classes, then categories, then static object instances
  Classes.insert(Classes.end(), Categories.begin(), Categories.end());
  //  NULL-terminated list of static object instances (mainly constant strings)
  Classes.push_back(Statics);
  Classes.push_back(NULLPtr);
  llvm::Constant *ClassList = llvm::ConstantArray::get(ClassListTy, Classes);
  Elements.push_back(ClassList);
  // Construct the symbol table 
  llvm::Constant *SymTab= MakeGlobal(SymTabTy, Elements);

  // The symbol table is contained in a module which has some version-checking
  // constants
  llvm::StructType * ModuleTy = llvm::StructType::get(LongTy, LongTy,
      PtrToInt8Ty, llvm::PointerType::getUnqual(SymTabTy), NULL);
  Elements.clear();
  // Runtime version used for compatibility checking.
  Elements.push_back(llvm::ConstantInt::get(LongTy, RuntimeVersion));
  //FIXME: Should be sizeof(ModuleTy)
  Elements.push_back(llvm::ConstantInt::get(LongTy, 16));
  //FIXME: Should be the path to the file where this module was declared
  Elements.push_back(NULLPtr);
  Elements.push_back(SymTab);
  llvm::Value *Module = MakeGlobal(ModuleTy, Elements);

  // Create the load function calling the runtime entry point with the module
  // structure
  std::vector<const llvm::Type*> VoidArgs;
  llvm::Function * LoadFunction = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::VoidTy, VoidArgs, false),
      llvm::GlobalValue::InternalLinkage, ".objc_load_function",
      &TheModule);
  llvm::BasicBlock *EntryBB = llvm::BasicBlock::Create("entry", LoadFunction);
  llvm::IRBuilder Builder;
  Builder.SetInsertPoint(EntryBB);
  llvm::Value *Register = TheModule.getOrInsertFunction("__objc_exec_class",
      llvm::Type::VoidTy, llvm::PointerType::getUnqual(ModuleTy), NULL);
  Builder.CreateCall(Register, Module);
  Builder.CreateRetVoid();
  return LoadFunction;
}
llvm::Function *CGObjCGNU::MethodPreamble(
                                         const std::string &ClassName,
                                         const std::string &CategoryName,
                                         const std::string &MethodName,
                                         const llvm::Type *ReturnTy,
                                         const llvm::Type *SelfTy,
                                         const llvm::Type **ArgTy,
                                         unsigned ArgC,
                                         bool isClassMethod,
                                         bool isVarArg) {
  std::vector<const llvm::Type*> Args;
  if (!ReturnTy->isSingleValueType() && ReturnTy != llvm::Type::VoidTy) {
    Args.push_back(llvm::PointerType::getUnqual(ReturnTy));
    ReturnTy = llvm::Type::VoidTy;
  }
  Args.push_back(SelfTy);
  Args.push_back(SelectorTy);
  Args.insert(Args.end(), ArgTy, ArgTy+ArgC);

  llvm::FunctionType *MethodTy = llvm::FunctionType::get(ReturnTy,
      Args,
      isVarArg);
  std::string FunctionName = SymbolNameForMethod(ClassName, CategoryName,
      MethodName, isClassMethod);

  llvm::Function *Method = llvm::Function::Create(MethodTy,
      llvm::GlobalValue::InternalLinkage,
      FunctionName,
      &TheModule);
  llvm::Function::arg_iterator AI = Method->arg_begin();
  // Name the struct return argument.
  // FIXME: This is probably the wrong test.
  if (!ReturnTy->isFirstClassType() && ReturnTy != llvm::Type::VoidTy) {
    AI->setName("agg.result");
    ++AI;
  }
  AI->setName("self");
  ++AI;
  AI->setName("_cmd");
  return Method;
}

clang::CodeGen::CGObjCRuntime *clang::CodeGen::CreateObjCRuntime(
    llvm::Module &M,
    const llvm::Type *LLVMIntType,
    const llvm::Type *LLVMLongType) {
  return new CGObjCGNU(M, LLVMIntType, LLVMLongType);
}
