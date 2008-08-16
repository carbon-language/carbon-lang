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
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/LangOptions.h"

#include "llvm/Module.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Target/TargetData.h"

using namespace clang;

namespace {

  // FIXME: We should find a nicer way to make the labels for
  // metadata, string concatenation is lame.

/// ObjCTypesHelper - Helper class that encapsulates lazy
/// construction of varies types used during ObjC generation.
class ObjCTypesHelper {
private:
  CodeGen::CodeGenModule &CGM;  
  
  const llvm::StructType *CFStringType;
  llvm::Constant *CFConstantStringClassReference;
  llvm::Function *MessageSendFn;

public:
  const llvm::Type *IntTy, *LongTy;

  /// ObjectPtrTy - LLVM type for object handles (typeof(id))
  const llvm::Type *ObjectPtrTy;
  /// SelectorPtrTy - LLVM type for selector handles (typeof(SEL))
  const llvm::Type *SelectorPtrTy;
  /// ProtocolPtrTy - LLVM type for external protocol handles
  /// (typeof(Protocol))
  const llvm::Type *ExternalProtocolPtrTy;

  /// SymtabTy - LLVM type for struct objc_symtab.
  const llvm::StructType *SymtabTy;
  /// ModuleTy - LLVM type for struct objc_module.
  const llvm::StructType *ModuleTy;

  /// ProtocolTy - LLVM type for struct objc_protocol.
  const llvm::StructType *ProtocolTy;
  /// ProtocolPtrTy - LLVM type for struct objc_protocol *.
  const llvm::Type *ProtocolPtrTy;
  /// ProtocolExtensionTy - LLVM type for struct
  /// objc_protocol_extension.
  const llvm::StructType *ProtocolExtensionTy;
  /// ProtocolExtensionTy - LLVM type for struct
  /// objc_protocol_extension *.
  const llvm::Type *ProtocolExtensionPtrTy;
  /// MethodDescriptionTy - LLVM type for struct
  /// objc_method_description.
  const llvm::StructType *MethodDescriptionTy;
  /// MethodDescriptionListTy - LLVM type for struct
  /// objc_method_description_list.
  const llvm::StructType *MethodDescriptionListTy;
  /// MethodDescriptionListPtrTy - LLVM type for struct
  /// objc_method_description_list *.
  const llvm::Type *MethodDescriptionListPtrTy;
  /// PropertyListTy - LLVM type for struct objc_property_list.
  const llvm::Type *PropertyListTy;
  /// PropertyListPtrTy - LLVM type for struct objc_property_list*.
  const llvm::Type *PropertyListPtrTy;
  /// ProtocolListTy - LLVM type for struct objc_property_list.
  const llvm::Type *ProtocolListTy;
  /// ProtocolListPtrTy - LLVM type for struct objc_property_list*.
  const llvm::Type *ProtocolListPtrTy;

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
  llvm::DenseMap<IdentifierInfo*, llvm::GlobalVariable*> ClassNames;

  /// MethodVarNames - uniqued method variable names.
  llvm::DenseMap<Selector, llvm::GlobalVariable*> MethodVarNames;

  /// MethodVarTypes - uniqued method type signatures. We have to use
  /// a StringMap here because have no other unique reference.
  llvm::StringMap<llvm::GlobalVariable*> MethodVarTypes;

  /// SelectorReferences - uniqued selector references.
  llvm::DenseMap<Selector, llvm::GlobalVariable*> SelectorReferences;

  /// Protocols - Protocols for which an objc_protocol structure has
  /// been emitted. Forward declarations are handled by creating an
  /// empty structure whose initializer is filled in when/if defined.
  llvm::DenseMap<IdentifierInfo*, llvm::GlobalVariable*> Protocols;

  /// UsedGlobals - List of globals to pack into the llvm.used metadata
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

  /// EmitMethodList - Emit a method description list for a list of
  /// method declarations. 
  ///  - TypeName: The name for the type containing the methods.
  ///  - IsProtocol: True iff these methods are for a protocol.
  ///  - ClassMethds: True iff these are class methods.
  ///  - Required: When true, only "required" methods are
  ///    listed. Similarly, when false only "optional" methods are
  ///    listed. For classes this should always be true.
  ///  - begin, end: The method list to output.
  ///
  /// The return value has type MethodDescriptionListPtrTy.
  llvm::Constant *EmitMethodList(const std::string &TypeName,
                                 bool IsProtocol,
                                 bool ClassMethods,
                                 bool Required,
                                 ObjCMethodDecl * const *begin,
                                 ObjCMethodDecl * const *end);

  /// EmitProtocolExtension - Generate the protocol extension
  /// structure used to store optional instance and class methods, and
  /// protocol properties. The return value has type
  /// ProtocolExtensionPtrTy.
  llvm::Constant *EmitProtocolExtension(const ObjCProtocolDecl *PD);

  /// EmitProtocolList - Generate the list of referenced
  /// protocols. The return value has type ProtocolListPtrTy.
  llvm::Constant *EmitProtocolList(const ObjCProtocolDecl *PD);

  /// GetProtocolRef - Return a reference to the internal protocol
  /// description, creating an empty one if it has not been
  /// defined. The return value has type pointer-to ProtocolTy.
  llvm::GlobalVariable *GetProtocolRef(const ObjCProtocolDecl *PD);

  /// EmitSelector - Return a Value*, of type ObjCTypes.SelectorPtrTy,
  /// for the given selector.
  llvm::Value *EmitSelector(llvm::IRBuilder<> &Builder, Selector Sel);

  /// GetClassName - Return a unique constant for the given selector's
  /// name.
  llvm::Constant *GetClassName(IdentifierInfo *Ident);

  /// GetMethodVarName - Return a unique constant for the given
  /// selector's name. This returns a constant i8* to the start of
  /// the name.
  llvm::Constant *GetMethodVarName(Selector Sel);

  /// GetMethodVarType - Return a unique constant for the given
  /// selector's name. This returns a constant i8* to the start of
  /// the name.
  llvm::Constant *GetMethodVarType(ObjCMethodDecl *D);

public:
  CGObjCMac(CodeGen::CodeGenModule &cgm);
  virtual llvm::Constant *GenerateConstantString(const std::string &String);

  virtual llvm::Value *GenerateMessageSend(llvm::IRBuilder<> &Builder,
                                           const llvm::Type *ReturnTy,
                                           llvm::Value *Receiver,
                                           Selector Sel,
                                           llvm::Value** ArgV,
                                           unsigned ArgC);

  virtual llvm::Value *
  GenerateMessageSendSuper(llvm::IRBuilder<> &Builder,
                           const llvm::Type *ReturnTy,
                           const ObjCInterfaceDecl *SuperClass,
                           llvm::Value *Receiver,
                           Selector Sel,
                           llvm::Value** ArgV,
                           unsigned ArgC);
  
  virtual llvm::Value *GetClass(llvm::IRBuilder<> &Builder,
                                const ObjCInterfaceDecl *SuperClass);

  virtual llvm::Value *GetSelector(llvm::IRBuilder<> &Builder, Selector Sel);
  
  virtual llvm::Function *GenerateMethod(const ObjCMethodDecl *OMD);

  virtual void GenerateCategory(const ObjCCategoryImplDecl *CMD);

  virtual void GenerateClass(const ObjCImplementationDecl *ClassDecl);

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

/// GetClass - Return a reference to the class for the given interface
/// decl.
llvm::Value *CGObjCMac::GetClass(llvm::IRBuilder<> &Builder,
                                    const ObjCInterfaceDecl *OID) {
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
llvm::Value *
CGObjCMac::GenerateMessageSendSuper(llvm::IRBuilder<> &Builder,
                                    const llvm::Type *ReturnTy,
                                    const ObjCInterfaceDecl *SuperClass,
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

  std::vector<const llvm::Type*> Params;
  Params.push_back(ObjCTypes.ObjectPtrTy);
  Params.push_back(ObjCTypes.SelectorPtrTy);
  llvm::FunctionType *CallFTy = llvm::FunctionType::get(ReturnTy,
                                                        Params,
                                                        true);
  llvm::Type *PCallFTy = llvm::PointerType::getUnqual(CallFTy);
  llvm::Constant *C = llvm::ConstantExpr::getBitCast(F, PCallFTy);
  llvm::CallInst *CI = Builder.CreateCall(C, 
                                          Args, Args+ArgC+2, "tmp");
  delete[] Args;
  return Builder.CreateBitCast(CI, ReturnTy, "tmp");
}

llvm::Value *CGObjCMac::GenerateProtocolRef(llvm::IRBuilder<> &Builder, 
                                            const ObjCProtocolDecl *PD) {
  return llvm::ConstantExpr::getBitCast(GetProtocolRef(PD),
                                        ObjCTypes.ExternalProtocolPtrTy);
}

/*
     // APPLE LOCAL radar 4585769 - Objective-C 1.0 extensions
  struct _objc_protocol {
    struct _objc_protocol_extension *isa;
    char *protocol_name;
    struct _objc_protocol_list *protocol_list;
    struct _objc__method_prototype_list *instance_methods;
    struct _objc__method_prototype_list *class_methods
  };

  See EmitProtocolExtension().
*/
void CGObjCMac::GenerateProtocol(const ObjCProtocolDecl *PD) { 
  const char *ProtocolName = PD->getName();
  
  std::vector<llvm::Constant*> Values(5);
  Values[0] = EmitProtocolExtension(PD);
  Values[1] = GetClassName(PD->getIdentifier());
  Values[2] = EmitProtocolList(PD);
  Values[3] = EmitMethodList(ProtocolName,
                             true, // IsProtocol
                             false, // ClassMethods
                             true, // Required
                             PD->instmeth_begin(),
                             PD->instmeth_end());
  Values[4] = EmitMethodList(ProtocolName,
                             true, // IsProtocol
                             true, // ClassMethods
                             true, // Required
                             PD->classmeth_begin(),
                             PD->classmeth_end());
  llvm::Constant *Init = llvm::ConstantStruct::get(ObjCTypes.ProtocolTy,
                                                   Values);
  
  llvm::GlobalVariable *&Entry = Protocols[PD->getIdentifier()];  
  if (Entry) {
    // Already created, just update the initializer
    Entry->setInitializer(Init);
  } else {
    Entry = 
      new llvm::GlobalVariable(ObjCTypes.ProtocolTy, false,
                               llvm::GlobalValue::InternalLinkage,
                               Init, 
                               std::string("\01L_OBJC_PROTOCOL_")+ProtocolName,
                               &CGM.getModule());
    Entry->setSection("__OBJC,__protocol,regular,no_dead_strip");
    UsedGlobals.push_back(Entry);
    // FIXME: Is this necessary? Why only for protocol?
    Entry->setAlignment(4);
  }
}

llvm::GlobalVariable *CGObjCMac::GetProtocolRef(const ObjCProtocolDecl *PD) {
  llvm::GlobalVariable *&Entry = Protocols[PD->getIdentifier()];

  if (!Entry) {
    std::vector<llvm::Constant*> Values(5);
    Values[0] = llvm::Constant::getNullValue(ObjCTypes.ProtocolExtensionPtrTy);
    Values[1] = GetClassName(PD->getIdentifier());
    Values[2] = llvm::Constant::getNullValue(ObjCTypes.ProtocolListPtrTy);
    Values[3] = Values[4] =
      llvm::Constant::getNullValue(ObjCTypes.MethodDescriptionListPtrTy);
    llvm::Constant *Init = llvm::ConstantStruct::get(ObjCTypes.ProtocolTy,
                                                     Values);

    Entry = 
      new llvm::GlobalVariable(ObjCTypes.ProtocolTy, false,
                               llvm::GlobalValue::InternalLinkage,
                               Init,
                               std::string("\01L_OBJC_PROTOCOL_")+PD->getName(),
                               &CGM.getModule());
    Entry->setSection("__OBJC,__protocol,regular,no_dead_strip");
    UsedGlobals.push_back(Entry);
    // FIXME: Is this necessary? Why only for protocol?
    Entry->setAlignment(4);
  }
  
  return Entry;
}

/*
  struct _objc_protocol_extension {
    uint32_t size;
    struct objc_method_description_list *optional_instance_methods;
    struct objc_method_description_list *optional_class_methods;
    struct objc_property_list *instance_properties;
  };
*/
llvm::Constant *CGObjCMac::EmitProtocolExtension(const ObjCProtocolDecl *PD) {
  uint64_t Size = 
    CGM.getTargetData().getABITypeSize(ObjCTypes.ProtocolExtensionTy);
  std::vector<llvm::Constant*> Values(4);
  Values[0] = llvm::ConstantInt::get(ObjCTypes.IntTy, Size);
  Values[1] = EmitMethodList(PD->getName(),
                             true, // IsProtocol
                             false, // ClassMethods
                             false, // Required
                             PD->instmeth_begin(),
                             PD->instmeth_end());
  Values[2] = EmitMethodList(PD->getName(),
                             true, // IsProtocol
                             true, // ClassMethods
                             false, // Required
                             PD->classmeth_begin(),
                             PD->classmeth_end());
  Values[3] = llvm::Constant::getNullValue(ObjCTypes.PropertyListPtrTy);
  assert(!PD->getNumPropertyDecl() && 
         "Cannot emit Obj-C protocol properties for NeXT runtime.");

  // Return null if no extension bits are used
  if (Values[1]->isNullValue() && Values[2]->isNullValue() && 
      Values[3]->isNullValue())
    return llvm::Constant::getNullValue(ObjCTypes.ProtocolExtensionPtrTy);

  llvm::Constant *Init = 
    llvm::ConstantStruct::get(ObjCTypes.ProtocolExtensionTy, Values);
  llvm::GlobalVariable *GV = 
      new llvm::GlobalVariable(ObjCTypes.ProtocolExtensionTy, false,
                               llvm::GlobalValue::InternalLinkage,
                               Init,
                               (std::string("\01L_OBJC_PROTOCOLEXT_") + 
                                PD->getName()),
                               &CGM.getModule());
  // No special section, but goes in llvm.used
  UsedGlobals.push_back(GV);

  return GV;
}

/*
  struct objc_protocol_list {
    struct objc_protocol_list *next;
    long count;
    Protocol *list[];
  };
*/
llvm::Constant *CGObjCMac::EmitProtocolList(const ObjCProtocolDecl *PD) {
  std::vector<llvm::Constant*> ProtocolRefs;

  for (ObjCProtocolDecl::protocol_iterator i = PD->protocol_begin(), 
         e = PD->protocol_end(); i != e; ++i)
    ProtocolRefs.push_back(GetProtocolRef(*i));

  // Just return null for empty protocol lists
  if (ProtocolRefs.empty()) 
    return llvm::Constant::getNullValue(ObjCTypes.ProtocolListPtrTy);

  // This list is null terminated?
  ProtocolRefs.push_back(llvm::Constant::getNullValue(ObjCTypes.ProtocolPtrTy));

  std::vector<llvm::Constant*> Values(3);
  // XXX: What is this for?
  Values[0] = llvm::Constant::getNullValue(ObjCTypes.ProtocolListPtrTy);
  Values[1] = llvm::ConstantInt::get(ObjCTypes.LongTy, ProtocolRefs.size() - 1);
  Values[2] = 
    llvm::ConstantArray::get(llvm::ArrayType::get(ObjCTypes.ProtocolPtrTy, 
                                                  ProtocolRefs.size()), 
                             ProtocolRefs);
  
  llvm::Constant *Init = llvm::ConstantStruct::get(Values);
  llvm::GlobalVariable *GV = 
    new llvm::GlobalVariable(Init->getType(), false,
                             llvm::GlobalValue::InternalLinkage,
                             Init,
                             (std::string("\01L_OBJC_PROTOCOL_REFS_") + 
                              PD->getName()), 
                             &CGM.getModule());
  GV->setSection("__OBJC,__cat_cls_meth,regular,no_dead_strip");
  return llvm::ConstantExpr::getBitCast(GV, ObjCTypes.ProtocolListPtrTy);
}

/*
  struct objc_method_description_list {
    int count;
    struct objc_method_description list[];
  };
*/
llvm::Constant *CGObjCMac::EmitMethodList(const std::string &TypeName,
                                          bool IsProtocol,
                                          bool ClassMethods,
                                          bool Required,
                                          ObjCMethodDecl * const *begin,
                                          ObjCMethodDecl * const *end) {
  std::vector<llvm::Constant*> Methods, Desc(2);
  for (; begin != end; ++begin) {
    ObjCMethodDecl *D = *begin;
    bool IsRequired = D->getImplementationControl() != ObjCMethodDecl::Optional;

    // Skip if this method is required and we are outputting optional
    // methods, or vice versa.
    if (Required != IsRequired)
      continue;

    Desc[0] = llvm::ConstantExpr::getBitCast(GetMethodVarName(D->getSelector()),
                                             ObjCTypes.SelectorPtrTy);
    Desc[1] = GetMethodVarType(D);
    Methods.push_back(llvm::ConstantStruct::get(ObjCTypes.MethodDescriptionTy,
                                                Desc));
  }

  // Return null for empty list.
  if (Methods.empty())
    return llvm::Constant::getNullValue(ObjCTypes.MethodDescriptionListPtrTy);

  std::vector<llvm::Constant*> Values(2);
  Values[0] = llvm::ConstantInt::get(ObjCTypes.IntTy, Methods.size());
  llvm::ArrayType *AT = llvm::ArrayType::get(ObjCTypes.MethodDescriptionTy, 
                                             Methods.size());
  Values[1] = llvm::ConstantArray::get(AT, Methods);
  llvm::Constant *Init = llvm::ConstantStruct::get(Values);

  char Prefix[256];
  sprintf(Prefix, "\01L_OBJC_%s%sMETHODS_%s",
          IsProtocol ? "PROTOCOL_" : "",
          ClassMethods ? "CLASS_" : "INSTANCE_",
          !Required ? "OPT_" : "");
  llvm::GlobalVariable *GV = 
    new llvm::GlobalVariable(Init->getType(), false,
                             llvm::GlobalValue::InternalLinkage,
                             Init,
                             std::string(Prefix) + TypeName,
                             &CGM.getModule());
  if (ClassMethods) {
    GV->setSection("__OBJC,__cat_cls_meth,regular,no_dead_strip");
  } else {
    GV->setSection("__OBJC,__cat_inst_meth,regular,no_dead_strip");
  }
  UsedGlobals.push_back(GV);
  return llvm::ConstantExpr::getBitCast(GV, 
                                        ObjCTypes.MethodDescriptionListPtrTy);
}

void CGObjCMac::GenerateCategory(const ObjCCategoryImplDecl *OCD) {
  assert(0 && "Cannot generate category for Mac runtime.");
}

void CGObjCMac::GenerateClass(const ObjCImplementationDecl *ClassDecl) {
  assert(0 && "Cannot generate class for Mac runtime.");
}

llvm::Function *CGObjCMac::ModuleInitFunction() { 
  // Abuse this interface function as a place to finalize.
  FinishModule();

  return NULL;
}

llvm::Function *CGObjCMac::GenerateMethod(const ObjCMethodDecl *OMD) {
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
  uint64_t Size = CGM.getTargetData().getABITypeSize(ObjCTypes.ModuleTy);
  
  std::vector<llvm::Constant*> Values(4);
  Values[0] = llvm::ConstantInt::get(ObjCTypes.LongTy, ModuleVersion);
  Values[1] = llvm::ConstantInt::get(ObjCTypes.LongTy, Size);
  // This used to be the filename, now it is unused. <rdr://4327263>
  Values[2] = GetClassName(&CGM.getContext().Idents.get(""));
  Values[3] = EmitModuleSymbols();

  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(ObjCTypes.ModuleTy, false,
                             llvm::GlobalValue::InternalLinkage,
                             llvm::ConstantStruct::get(ObjCTypes.ModuleTy, 
                                                       Values),
                             "\01L_OBJC_MODULES", 
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

llvm::Constant *CGObjCMac::GetClassName(IdentifierInfo *Ident) {
  llvm::GlobalVariable *&Entry = ClassNames[Ident];

  if (!Entry) {
    llvm::Constant *C = llvm::ConstantArray::get(Ident->getName());
    Entry = 
      new llvm::GlobalVariable(C->getType(), false, 
                               llvm::GlobalValue::InternalLinkage,
                               C, "\01L_OBJC_CLASS_NAME_", 
                               &CGM.getModule());
    Entry->setSection("__TEXT,__cstring,cstring_literals");
    UsedGlobals.push_back(Entry);
  }

  return getConstantGEP(Entry, 0, 0);
}

llvm::Constant *CGObjCMac::GetMethodVarName(Selector Sel) {
  llvm::GlobalVariable *&Entry = MethodVarNames[Sel];

  if (!Entry) {
    llvm::Constant *C = llvm::ConstantArray::get(Sel.getName());
    Entry = 
      new llvm::GlobalVariable(C->getType(), false, 
                               llvm::GlobalValue::InternalLinkage,
                               C, "\01L_OBJC_METH_VAR_NAME_", 
                               &CGM.getModule());
    Entry->setSection("__TEXT,__cstring,cstring_literals");
    UsedGlobals.push_back(Entry);
  }

  return getConstantGEP(Entry, 0, 0);
}

llvm::Constant *CGObjCMac::GetMethodVarType(ObjCMethodDecl *D) {
  std::string TypeStr;
  CGM.getContext().getObjCEncodingForMethodDecl(D, TypeStr);
  llvm::GlobalVariable *&Entry = MethodVarTypes[TypeStr];

  if (!Entry) {
    llvm::Constant *C = llvm::ConstantArray::get(TypeStr);
    Entry = 
      new llvm::GlobalVariable(C->getType(), false, 
                               llvm::GlobalValue::InternalLinkage,
                               C, "\01L_OBJC_METH_VAR_TYPE_", 
                               &CGM.getModule());
    Entry->setSection("__TEXT,__cstring,cstring_literals");
    UsedGlobals.push_back(Entry);
  }

  return getConstantGEP(Entry, 0, 0);
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

  IntTy = Types.ConvertType(Ctx.IntTy);
  LongTy = Types.ConvertType(Ctx.LongTy);
  ObjectPtrTy = Types.ConvertType(Ctx.getObjCIdType());
  SelectorPtrTy = Types.ConvertType(Ctx.getObjCSelType());
  
  // FIXME: It would be nice to unify this with the opaque type, so
  // that the IR comes out a bit cleaner.
  const llvm::Type *T = Types.ConvertType(Ctx.getObjCProtoType());
  ExternalProtocolPtrTy = llvm::PointerType::getUnqual(T);

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

  MethodDescriptionTy = 
    llvm::StructType::get(SelectorPtrTy,
                          llvm::PointerType::getUnqual(llvm::Type::Int8Ty),
                          NULL);
  CGM.getModule().addTypeName("struct._objc_method_description", 
                              MethodDescriptionTy);

  MethodDescriptionListTy = 
    llvm::StructType::get(IntTy,
                          llvm::ArrayType::get(MethodDescriptionTy, 0),
                          NULL);
  CGM.getModule().addTypeName("struct._objc_method_description_list", 
                              MethodDescriptionListTy);
  MethodDescriptionListPtrTy = 
    llvm::PointerType::getUnqual(MethodDescriptionListTy);

  PropertyListTy = llvm::OpaqueType::get();
  CGM.getModule().addTypeName("struct._objc_property_list", 
                              PropertyListTy);
  PropertyListPtrTy = llvm::PointerType::getUnqual(PropertyListTy);

  // Protocol description structures

  ProtocolExtensionTy = 
    llvm::StructType::get(Types.ConvertType(Ctx.IntTy),
                          llvm::PointerType::getUnqual(MethodDescriptionListTy),
                          llvm::PointerType::getUnqual(MethodDescriptionListTy),
                          PropertyListPtrTy,
                          NULL);
  CGM.getModule().addTypeName("struct._objc_protocol_extension", 
                              ProtocolExtensionTy);
  ProtocolExtensionPtrTy = llvm::PointerType::getUnqual(ProtocolExtensionTy);

  // Handle recursive construction of Protocl and ProtocolList types

  llvm::PATypeHolder ProtocolTyHolder = llvm::OpaqueType::get();
  llvm::PATypeHolder ProtocolListTyHolder = llvm::OpaqueType::get();

  T = llvm::StructType::get(llvm::PointerType::getUnqual(ProtocolListTyHolder),
                            LongTy,
                            llvm::ArrayType::get(ProtocolTyHolder, 0),
                            NULL);
  cast<llvm::OpaqueType>(ProtocolListTyHolder.get())->refineAbstractTypeTo(T);

  T = llvm::StructType::get(llvm::PointerType::getUnqual(ProtocolExtensionTy),
                            llvm::PointerType::getUnqual(llvm::Type::Int8Ty),
                            llvm::PointerType::getUnqual(ProtocolListTyHolder),
                            MethodDescriptionListPtrTy,
                            MethodDescriptionListPtrTy,
                            NULL);
  cast<llvm::OpaqueType>(ProtocolTyHolder.get())->refineAbstractTypeTo(T);

  ProtocolListTy = cast<llvm::StructType>(ProtocolListTyHolder.get());
  CGM.getModule().addTypeName("struct._objc_protocol_list", 
                              ProtocolListTy);
  ProtocolListPtrTy = llvm::PointerType::getUnqual(ProtocolListTy);

  ProtocolTy = cast<llvm::StructType>(ProtocolTyHolder.get());
  CGM.getModule().addTypeName("struct.__objc_protocol", ProtocolTy);
  ProtocolPtrTy = llvm::PointerType::getUnqual(ProtocolTy);
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

CodeGen::CGObjCRuntime *
CodeGen::CreateMacObjCRuntime(CodeGen::CodeGenModule &CGM) {
  return new CGObjCMac(CGM);
}
