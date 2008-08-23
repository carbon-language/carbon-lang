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
#include "CodeGenFunction.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/LangOptions.h"

#include "llvm/Module.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Target/TargetData.h"
#include <sstream>

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
  llvm::Function *MessageSendFn, *MessageSendStretFn;
  llvm::Function *MessageSendSuperFn, *MessageSendSuperStretFn;

public:
  const llvm::Type *ShortTy, *IntTy, *LongTy;
  const llvm::Type *Int8PtrTy;

  /// ObjectPtrTy - LLVM type for object handles (typeof(id))
  const llvm::Type *ObjectPtrTy;
  /// SelectorPtrTy - LLVM type for selector handles (typeof(SEL))
  const llvm::Type *SelectorPtrTy;
  /// ProtocolPtrTy - LLVM type for external protocol handles
  /// (typeof(Protocol))
  const llvm::Type *ExternalProtocolPtrTy;

  /// SuperTy - LLVM type for struct objc_super.
  const llvm::StructType *SuperTy;
  /// SuperPtrTy - LLVM type for struct objc_super *.
  const llvm::Type *SuperPtrTy;

  /// SymtabTy - LLVM type for struct objc_symtab.
  const llvm::StructType *SymtabTy;
  /// SymtabPtrTy - LLVM type for struct objc_symtab *.
  const llvm::Type *SymtabPtrTy;
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
  /// PropertyTy - LLVM type for struct objc_property (struct _prop_t
  /// in GCC parlance).
  const llvm::StructType *PropertyTy;
  /// PropertyListTy - LLVM type for struct objc_property_list
  /// (_prop_list_t in GCC parlance).
  const llvm::StructType *PropertyListTy;
  /// PropertyListPtrTy - LLVM type for struct objc_property_list*.
  const llvm::Type *PropertyListPtrTy;
  /// ProtocolListTy - LLVM type for struct objc_property_list.
  const llvm::Type *ProtocolListTy;
  /// ProtocolListPtrTy - LLVM type for struct objc_property_list*.
  const llvm::Type *ProtocolListPtrTy;
  /// CategoryTy - LLVM type for struct objc_category.
  const llvm::StructType *CategoryTy;
  /// ClassTy - LLVM type for struct objc_class.
  const llvm::StructType *ClassTy;
  /// ClassPtrTy - LLVM type for struct objc_class *.
  const llvm::Type *ClassPtrTy;
  /// ClassExtensionTy - LLVM type for struct objc_class_ext.
  const llvm::StructType *ClassExtensionTy;
  /// ClassExtensionPtrTy - LLVM type for struct objc_class_ext *.
  const llvm::Type *ClassExtensionPtrTy;
  /// CacheTy - LLVM type for struct objc_cache.
  const llvm::Type *CacheTy;
  /// CachePtrTy - LLVM type for struct objc_cache *.
  const llvm::Type *CachePtrTy;
  // IvarTy - LLVM type for struct objc_ivar.
  const llvm::StructType *IvarTy;
  /// IvarListTy - LLVM type for struct objc_ivar_list.
  const llvm::Type *IvarListTy;
  /// IvarListPtrTy - LLVM type for struct objc_ivar_list *.
  const llvm::Type *IvarListPtrTy;
  // MethodTy - LLVM type for struct objc_method.
  const llvm::StructType *MethodTy;
  /// MethodListTy - LLVM type for struct objc_method_list.
  const llvm::Type *MethodListTy;
  /// MethodListPtrTy - LLVM type for struct objc_method_list *.
  const llvm::Type *MethodListPtrTy;

public:
  ObjCTypesHelper(CodeGen::CodeGenModule &cgm);
  ~ObjCTypesHelper();
  
  llvm::Constant *getCFConstantStringClassReference();
  const llvm::StructType *getCFStringType();
  llvm::Value *getMessageSendFn(bool IsSuper, const llvm::Type *ReturnTy);
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

  /// PropertyNames - uniqued method variable names.
  llvm::DenseMap<IdentifierInfo*, llvm::GlobalVariable*> PropertyNames;

  /// ClassReferences - uniqued class references.
  llvm::DenseMap<IdentifierInfo*, llvm::GlobalVariable*> ClassReferences;

  /// SelectorReferences - uniqued selector references.
  llvm::DenseMap<Selector, llvm::GlobalVariable*> SelectorReferences;

  /// Protocols - Protocols for which an objc_protocol structure has
  /// been emitted. Forward declarations are handled by creating an
  /// empty structure whose initializer is filled in when/if defined.
  llvm::DenseMap<IdentifierInfo*, llvm::GlobalVariable*> Protocols;

  /// DefinedClasses - List of defined classes.
  std::vector<llvm::GlobalValue*> DefinedClasses;

  /// DefinedCategories - List of defined categories.
  std::vector<llvm::GlobalValue*> DefinedCategories;

  /// UsedGlobals - List of globals to pack into the llvm.used metadata
  /// to prevent them from being clobbered.
  std::vector<llvm::GlobalVariable*> UsedGlobals;

  /// EmitImageInfo - Emit the image info marker used to encode some module
  /// level information.
  void EmitImageInfo();

  /// EmitModuleInfo - Another marker encoding module level
  /// information. 
  void EmitModuleInfo();

  /// EmitModuleSymols - Emit module symbols, the list of defined
  /// classes and categories. The result has type SymtabPtrTy.
  llvm::Constant *EmitModuleSymbols();

  /// FinishModule - Write out global data structures at the end of
  /// processing a translation unit.
  void FinishModule();

  /// EmitClassExtension - Generate the class extension structure used
  /// to store the weak ivar layout and properties. The return value
  /// has type ClassExtensionPtrTy.
  llvm::Constant *EmitClassExtension(const ObjCImplementationDecl *ID);

  /// EmitClassRef - Return a Value*, of type ObjCTypes.ClassPtrTy,
  /// for the given class.
  llvm::Value *EmitClassRef(llvm::IRBuilder<> &Builder, 
                            const ObjCInterfaceDecl *ID);

  CodeGen::RValue EmitMessageSend(CodeGen::CodeGenFunction &CGF,
                                  const ObjCMessageExpr *E,
                                  llvm::Value *Arg0,
                                  bool IsSuper);

  /// EmitIvarList - Emit the ivar list for the given
  /// implementation. If ForClass is true the list of class ivars
  /// (i.e. metaclass ivars) is emitted, otherwise the list of
  /// interface ivars will be emitted. The return value has type
  /// IvarListPtrTy.
  llvm::Constant *EmitIvarList(const ObjCImplementationDecl *ID,
                               bool ForClass,
                               const llvm::Type *InterfaceTy);
  
  /// EmitMetaClass - Emit a class structure for the metaclass of the
  /// given implementation. return value has type ClassPtrTy.
  llvm::Constant *EmitMetaClass(const ObjCImplementationDecl *ID,
                                llvm::Constant *Protocols,
                                const llvm::Type *InterfaceTy);

  /// EmitMethodList - Emit the method list for the given
  /// implementation. If ForClass is true the list of class methods
  /// will be emitted, otherwise the list of instance methods will be
  /// generated. The return value has type MethodListPtrTy.
  llvm::Constant *EmitMethodList(const std::string &Name,
                                 const char *Section,
                   llvm::SmallVector<ObjCMethodDecl*, 32>::const_iterator begin,
                   llvm::SmallVector<ObjCMethodDecl*, 32>::const_iterator end);

  /// EmitMethodDescList - Emit a method description list for a list of
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
  llvm::Constant *EmitMethodDescList(const std::string &TypeName,
                                     bool IsProtocol,
                                     bool ClassMethods,
                                     bool Required,
                                     ObjCMethodDecl * const *begin,
                                     ObjCMethodDecl * const *end);

  /// EmitPropertyList - Emit the given property list. The return
  /// value has type PropertyListPtrTy.
  llvm::Constant *EmitPropertyList(const std::string &Name,
                                   ObjCPropertyDecl * const *begin,
                                   ObjCPropertyDecl * const *end);

  /// EmitProtocolExtension - Generate the protocol extension
  /// structure used to store optional instance and class methods, and
  /// protocol properties. The return value has type
  /// ProtocolExtensionPtrTy.
  llvm::Constant *EmitProtocolExtension(const ObjCProtocolDecl *PD);

  /// EmitProtocolList - Generate the list of referenced
  /// protocols. The return value has type ProtocolListPtrTy.
  llvm::Constant *EmitProtocolList(const std::string &Name,
                                   ObjCProtocolDecl::protocol_iterator begin,
                                   ObjCProtocolDecl::protocol_iterator end);

  /// EmitSelector - Return a Value*, of type ObjCTypes.SelectorPtrTy,
  /// for the given selector.
  llvm::Value *EmitSelector(llvm::IRBuilder<> &Builder, Selector Sel);

  /// GetProtocolRef - Return a reference to the internal protocol
  /// description, creating an empty one if it has not been
  /// defined. The return value has type pointer-to ProtocolTy.
  llvm::GlobalVariable *GetProtocolRef(const ObjCProtocolDecl *PD);

  /// GetClassName - Return a unique constant for the given selector's
  /// name. The return value has type char *.
  llvm::Constant *GetClassName(IdentifierInfo *Ident);

  /// GetMethodVarName - Return a unique constant for the given
  /// selector's name. The return value has type char *.
  llvm::Constant *GetMethodVarName(Selector Sel);
  llvm::Constant *GetMethodVarName(IdentifierInfo *Ident);
  llvm::Constant *GetMethodVarName(const std::string &Name);

  /// GetMethodVarType - Return a unique constant for the given
  /// selector's name. The return value has type char *.

  // FIXME: This is a horrible name.
  llvm::Constant *GetMethodVarType(ObjCMethodDecl *D);
  llvm::Constant *GetMethodVarType(const std::string &Name);

  /// GetPropertyName - Return a unique constant for the given
  /// name. The return value has type char *.
  llvm::Constant *GetPropertyName(IdentifierInfo *Ident);

  // FIXME: This is a horrible name too.
  llvm::Constant *GetPropertyType(const ObjCPropertyDecl *PD);

  /// GetNameForMethod - Return a name for the given method.
  /// \param[out] NameOut - The return value.
  void GetNameForMethod(const ObjCMethodDecl *OMD,
                        std::string &NameOut);

public:
  CGObjCMac(CodeGen::CodeGenModule &cgm);
  virtual llvm::Constant *GenerateConstantString(const std::string &String);

  virtual CodeGen::RValue GenerateMessageSend(CodeGen::CodeGenFunction &CGF,
                                              const ObjCMessageExpr *E,
                                              llvm::Value *Receiver);

  virtual CodeGen::RValue 
  GenerateMessageSendSuper(CodeGen::CodeGenFunction &CGF,
                           const ObjCMessageExpr *E,
                           const ObjCInterfaceDecl *SuperClass,
                           llvm::Value *Receiver);
  
  virtual llvm::Value *GetClass(llvm::IRBuilder<> &Builder,
                                const ObjCInterfaceDecl *ID);

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
                                 const ObjCInterfaceDecl *ID) {
  return EmitClassRef(Builder, ID);
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
CodeGen::RValue
CGObjCMac::GenerateMessageSendSuper(CodeGen::CodeGenFunction &CGF,
                                    const ObjCMessageExpr *E,
                                    const ObjCInterfaceDecl *SuperClass,
                                    llvm::Value *Receiver) {
  // FIXME: This should be cached, not looked up every time. Meh. We
  // should just make sure the optimizer hits it.
  llvm::Value *ReceiverClass = EmitClassRef(CGF.Builder, SuperClass);
  
  // Create and init a super structure; this is a (receiver, class)
  // pair we will pass to objc_msgSendSuper.
  llvm::Value *ObjCSuper = 
    CGF.Builder.CreateAlloca(ObjCTypes.SuperTy, 0, "objc_super");
  llvm::Value *ReceiverAsObject = 
    CGF.Builder.CreateBitCast(Receiver, ObjCTypes.ObjectPtrTy);
  CGF.Builder.CreateStore(ReceiverAsObject, 
                          CGF.Builder.CreateStructGEP(ObjCSuper, 0));
  CGF.Builder.CreateStore(ReceiverClass, 
                          CGF.Builder.CreateStructGEP(ObjCSuper, 1));

  return EmitMessageSend(CGF, E, ObjCSuper, true);
}
                                           
/// Generate code for a message send expression.  
CodeGen::RValue CGObjCMac::GenerateMessageSend(CodeGen::CodeGenFunction &CGF,
                                               const ObjCMessageExpr *E,
                                               llvm::Value *Receiver) {
  llvm::Value *Arg0 = 
    CGF.Builder.CreateBitCast(Receiver, ObjCTypes.ObjectPtrTy, "tmp");
  return EmitMessageSend(CGF, E, Arg0, false);
}

CodeGen::RValue CGObjCMac::EmitMessageSend(CodeGen::CodeGenFunction &CGF,
                                           const ObjCMessageExpr *E,
                                           llvm::Value *Arg0,
                                           bool IsSuper) {
  llvm::Value *Args[2];
  Args[0] = Arg0;
  Args[1] = EmitSelector(CGF.Builder, E->getSelector());

  // FIXME: This is a hack, we are implicitly coordinating with
  // EmitCallExprExt, which will move the return type to the first
  // parameter and set the structure return flag. See
  // getMessageSendFn().

                                                   
  const llvm::Type *ReturnTy = CGM.getTypes().ConvertType(E->getType());
  return CGF.EmitCallExprExt(ObjCTypes.getMessageSendFn(IsSuper, ReturnTy),
                             E->getType(),
                             E->arg_begin(),
                             E->arg_end(),
                             Args, 2);
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
  Values[2] = 
    EmitProtocolList(std::string("\01L_OBJC_PROTOCOL_REFS_")+PD->getName(),
                     PD->protocol_begin(),
                     PD->protocol_end());
  Values[3] = EmitMethodDescList(ProtocolName,
                                 true, // IsProtocol
                                 false, // ClassMethods
                                 true, // Required
                                 PD->instmeth_begin(),
                                 PD->instmeth_end());
  Values[4] = EmitMethodDescList(ProtocolName,
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
  Values[1] = EmitMethodDescList(PD->getName(),
                                 true, // IsProtocol
                                 false, // ClassMethods
                                 false, // Required
                                 PD->instmeth_begin(),
                                 PD->instmeth_end());
  Values[2] = EmitMethodDescList(PD->getName(),
                                 true, // IsProtocol
                                 true, // ClassMethods
                                 false, // Required
                                 PD->classmeth_begin(),
                                 PD->classmeth_end());
  Values[3] = EmitPropertyList(std::string("\01L_OBJC_$_PROP_PROTO_LIST_") + 
                               PD->getName(),
                               PD->classprop_begin(),
                               PD->classprop_end());

  // Return null if no extension bits are used.
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
llvm::Constant *
CGObjCMac::EmitProtocolList(const std::string &Name,
                            ObjCProtocolDecl::protocol_iterator begin,
                            ObjCProtocolDecl::protocol_iterator end) {
  std::vector<llvm::Constant*> ProtocolRefs;

  for (; begin != end; ++begin)
    ProtocolRefs.push_back(GetProtocolRef(*begin));

  // Just return null for empty protocol lists
  if (ProtocolRefs.empty()) 
    return llvm::Constant::getNullValue(ObjCTypes.ProtocolListPtrTy);

  // This list is null terminated.
  ProtocolRefs.push_back(llvm::Constant::getNullValue(ObjCTypes.ProtocolPtrTy));

  std::vector<llvm::Constant*> Values(3);
  // This field is only used by the runtime.
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
                             Name,
                             &CGM.getModule());
  GV->setSection("__OBJC,__cat_cls_meth,regular,no_dead_strip");
  return llvm::ConstantExpr::getBitCast(GV, ObjCTypes.ProtocolListPtrTy);
}

/*
  struct _objc_property {
    const char * const name;
    const char * const attributes;
  };

  struct _objc_property_list {
    uint32_t entsize; // sizeof (struct _objc_property)
    uint32_t prop_count;
    struct _objc_property[prop_count];
  };
*/
llvm::Constant *CGObjCMac::EmitPropertyList(const std::string &Name,
                                            ObjCPropertyDecl * const *begin,
                                            ObjCPropertyDecl * const *end) {
  std::vector<llvm::Constant*> Properties, Prop(2);
  for (; begin != end; ++begin) {
    const ObjCPropertyDecl *PD = *begin;
    Prop[0] = GetPropertyName(PD->getIdentifier());
    Prop[1] = GetPropertyType(PD);
    Properties.push_back(llvm::ConstantStruct::get(ObjCTypes.PropertyTy,
                                                   Prop));
  }

  // Return null for empty list.
  if (Properties.empty())
    return llvm::Constant::getNullValue(ObjCTypes.PropertyListPtrTy);

  unsigned PropertySize = 
    CGM.getTargetData().getABITypeSize(ObjCTypes.PropertyTy);
  std::vector<llvm::Constant*> Values(3);
  Values[0] = llvm::ConstantInt::get(ObjCTypes.IntTy, PropertySize);
  Values[1] = llvm::ConstantInt::get(ObjCTypes.IntTy, Properties.size());
  llvm::ArrayType *AT = llvm::ArrayType::get(ObjCTypes.PropertyTy, 
                                             Properties.size());
  Values[2] = llvm::ConstantArray::get(AT, Properties);
  llvm::Constant *Init = llvm::ConstantStruct::get(Values);

  llvm::GlobalVariable *GV = 
    new llvm::GlobalVariable(Init->getType(), false,
                             llvm::GlobalValue::InternalLinkage,
                             Init,
                             Name,
                             &CGM.getModule());
  // No special section on property lists?
  UsedGlobals.push_back(GV);
  return llvm::ConstantExpr::getBitCast(GV, 
                                        ObjCTypes.PropertyListPtrTy);
  
}

/*
  struct objc_method_description_list {
    int count;
    struct objc_method_description list[];
  };
*/
llvm::Constant *CGObjCMac::EmitMethodDescList(const std::string &TypeName,
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

/*
  struct _objc_category {
    char *category_name;
    char *class_name;
    struct _objc_method_list *instance_methods;
    struct _objc_method_list *class_methods;
    struct _objc_protocol_list *protocols;
    uint32_t size; // <rdar://4585769>
    struct _objc_property_list *instance_properties;
  };
 */
void CGObjCMac::GenerateCategory(const ObjCCategoryImplDecl *OCD) {
  unsigned Size = CGM.getTargetData().getABITypeSize(ObjCTypes.CategoryTy);

  const ObjCInterfaceDecl *Interface = OCD->getClassInterface();
  std::string ExtName(std::string(Interface->getName()) +
                      "_" +
                      OCD->getName());

  std::vector<llvm::Constant*> Values(7);
  Values[0] = GetClassName(OCD->getIdentifier());
  Values[1] = GetClassName(Interface->getIdentifier());
  Values[2] = 
    EmitMethodList(std::string("\01L_OBJC_CATEGORY_INSTANCE_METHODS_") + 
                   ExtName,
                   "__OBJC,__cat_inst_meth,regular,no_dead_strip",
                   OCD->instmeth_begin(),
                   OCD->instmeth_end());
  Values[3] = 
    EmitMethodList(std::string("\01L_OBJC_CATEGORY_CLASS_METHODS_") + ExtName,
                   "__OBJC,__cat_class_meth,regular,no_dead_strip",
                   OCD->classmeth_begin(),
                   OCD->classmeth_end());
  Values[4] = 
    EmitProtocolList(std::string("\01L_OBJC_CATEGORY_PROTOCOLS_") + ExtName,
                     Interface->protocol_begin(),
                     Interface->protocol_end());
  Values[5] = llvm::ConstantInt::get(ObjCTypes.IntTy, Size);
  Values[6] = EmitPropertyList(std::string("\01L_OBJC_$_PROP_LIST_") + ExtName,
                               Interface->classprop_begin(),
                               Interface->classprop_end());
  
  llvm::Constant *Init = llvm::ConstantStruct::get(ObjCTypes.CategoryTy,
                                                   Values);

  llvm::GlobalVariable *GV = 
    new llvm::GlobalVariable(ObjCTypes.CategoryTy, false,
                             llvm::GlobalValue::InternalLinkage,
                             Init,
                             std::string("\01L_OBJC_CATEGORY_")+ExtName,
                             &CGM.getModule());
  GV->setSection("__OBJC,__category,regular,no_dead_strip");
  UsedGlobals.push_back(GV);
  DefinedCategories.push_back(GV);
}

// FIXME: Get from somewhere?
enum ClassFlags {
  eClassFlags_Factory              = 0x00001,
  eClassFlags_Meta                 = 0x00002,
  // <rdr://5142207>
  eClassFlags_HasCXXStructors      = 0x02000,
  eClassFlags_Hidden               = 0x20000,
  eClassFlags_ABI2_Hidden          = 0x00010,
  eClassFlags_ABI2_HasCXXStructors = 0x00004   // <rdr://4923634>
};

// <rdr://5142207&4705298&4843145>
static bool IsClassHidden(const ObjCInterfaceDecl *ID) {
  if (const VisibilityAttr *attr = ID->getAttr<VisibilityAttr>()) {
    // FIXME: Support -fvisibility
    switch (attr->getVisibility()) {
    default: 
      assert(0 && "Unknown visibility");
      return false;
    case VisibilityAttr::DefaultVisibility:
    case VisibilityAttr::ProtectedVisibility:  // FIXME: What do we do here?
      return false;
    case VisibilityAttr::HiddenVisibility:
      return true;
    }
  } else {
    return false; // FIXME: Support -fvisibility
  }
}

/*
  struct _objc_class {
    Class isa;
    Class super_class;
    const char *name;
    long version;
    long info;
    long instance_size;
    struct _objc_ivar_list *ivars;
    struct _objc_method_list *methods;
    struct _objc_cache *cache;
    struct _objc_protocol_list *protocols;
    // Objective-C 1.0 extensions (<rdr://4585769>)
    const char *ivar_layout;
    struct _objc_class_ext *ext;
  };

  See EmitClassExtension();
 */
void CGObjCMac::GenerateClass(const ObjCImplementationDecl *ID) {
  const char *ClassName = ID->getName();
  // FIXME: Gross
  ObjCInterfaceDecl *Interface = 
    const_cast<ObjCInterfaceDecl*>(ID->getClassInterface());
  llvm::Constant *Protocols = 
    EmitProtocolList(std::string("\01L_OBJC_CLASS_PROTOCOLS_") + ID->getName(),
                     Interface->protocol_begin(),
                     Interface->protocol_end());
  const llvm::Type *InterfaceTy = 
   CGM.getTypes().ConvertType(CGM.getContext().getObjCInterfaceType(Interface));
  unsigned Flags = eClassFlags_Factory;
  unsigned Size = CGM.getTargetData().getABITypeSize(InterfaceTy);

  // FIXME: Set CXX-structors flag.
  if (IsClassHidden(ID->getClassInterface()))
    Flags |= eClassFlags_Hidden;

  std::vector<llvm::Constant*> Values(12);
  Values[ 0] = EmitMetaClass(ID, Protocols, InterfaceTy);
  if (ObjCInterfaceDecl *Super = Interface->getSuperClass()) {
    Values[ 1] = 
      llvm::ConstantExpr::getBitCast(GetClassName(Super->getIdentifier()),
                                     ObjCTypes.ClassPtrTy);
  } else {
    Values[ 1] = llvm::Constant::getNullValue(ObjCTypes.ClassPtrTy);
  }
  Values[ 2] = GetClassName(ID->getIdentifier());
  // Version is always 0.
  Values[ 3] = llvm::ConstantInt::get(ObjCTypes.LongTy, 0);
  Values[ 4] = llvm::ConstantInt::get(ObjCTypes.LongTy, Flags);
  Values[ 5] = llvm::ConstantInt::get(ObjCTypes.LongTy, Size);
  Values[ 6] = EmitIvarList(ID, false, InterfaceTy);
  Values[ 7] = 
    EmitMethodList(std::string("\01L_OBJC_INSTANCE_METHODS_") + ID->getName(),
                   "__OBJC,__inst_meth,regular,no_dead_strip",
                   ID->instmeth_begin(),
                   ID->instmeth_end());
  // cache is always NULL.
  Values[ 8] = llvm::Constant::getNullValue(ObjCTypes.CachePtrTy);
  Values[ 9] = Protocols;
  // FIXME: Set ivar_layout
  Values[10] = llvm::Constant::getNullValue(ObjCTypes.Int8PtrTy); 
  Values[11] = EmitClassExtension(ID);
  llvm::Constant *Init = llvm::ConstantStruct::get(ObjCTypes.ClassTy,
                                                   Values);

  llvm::GlobalVariable *GV = 
    new llvm::GlobalVariable(ObjCTypes.ClassTy, false,
                             llvm::GlobalValue::InternalLinkage,
                             Init,
                             std::string("\01L_OBJC_CLASS_")+ClassName,
                             &CGM.getModule());
  GV->setSection("__OBJC,__class,regular,no_dead_strip");
  UsedGlobals.push_back(GV);
  // FIXME: Why?
  GV->setAlignment(32);
  DefinedClasses.push_back(GV);
}

llvm::Constant *CGObjCMac::EmitMetaClass(const ObjCImplementationDecl *ID,
                                         llvm::Constant *Protocols,
                                         const llvm::Type *InterfaceTy) {
  const char *ClassName = ID->getName();
  unsigned Flags = eClassFlags_Meta;
  unsigned Size = CGM.getTargetData().getABITypeSize(ObjCTypes.ClassTy);

  if (IsClassHidden(ID->getClassInterface()))
    Flags |= eClassFlags_Hidden;
 
  std::vector<llvm::Constant*> Values(12);
  // The isa for the metaclass is the root of the hierarchy.
  const ObjCInterfaceDecl *Root = ID->getClassInterface();
  while (const ObjCInterfaceDecl *Super = Root->getSuperClass())
    Root = Super;
  Values[ 0] = 
    llvm::ConstantExpr::getBitCast(GetClassName(Root->getIdentifier()),
                                   ObjCTypes.ClassPtrTy);
  // The super class for the metaclass is emitted as the name of the
  // super class. The runtime fixes this up to point to the
  // *metaclass* for the super class.
  if (ObjCInterfaceDecl *Super = ID->getClassInterface()->getSuperClass()) {
    Values[ 1] = 
      llvm::ConstantExpr::getBitCast(GetClassName(Super->getIdentifier()),
                                     ObjCTypes.ClassPtrTy);
  } else {
    Values[ 1] = llvm::Constant::getNullValue(ObjCTypes.ClassPtrTy);
  }
  Values[ 2] = GetClassName(ID->getIdentifier());
  // Version is always 0.
  Values[ 3] = llvm::ConstantInt::get(ObjCTypes.LongTy, 0);
  Values[ 4] = llvm::ConstantInt::get(ObjCTypes.LongTy, Flags);
  Values[ 5] = llvm::ConstantInt::get(ObjCTypes.LongTy, Size);
  Values[ 6] = EmitIvarList(ID, true, InterfaceTy);
  Values[ 7] = 
    EmitMethodList(std::string("\01L_OBJC_CLASS_METHODS_") + ID->getName(),
                   "__OBJC,__inst_meth,regular,no_dead_strip",
                   ID->classmeth_begin(),
                   ID->classmeth_end());
  // cache is always NULL.
  Values[ 8] = llvm::Constant::getNullValue(ObjCTypes.CachePtrTy);
  Values[ 9] = Protocols;
  // ivar_layout for metaclass is always NULL.
  Values[10] = llvm::Constant::getNullValue(ObjCTypes.Int8PtrTy);
  // The class extension is always unused for metaclasses.
  Values[11] = llvm::Constant::getNullValue(ObjCTypes.ClassExtensionPtrTy);
  llvm::Constant *Init = llvm::ConstantStruct::get(ObjCTypes.ClassTy,
                                                   Values);

  llvm::GlobalVariable *GV = 
    new llvm::GlobalVariable(ObjCTypes.ClassTy, false,
                             llvm::GlobalValue::InternalLinkage,
                             Init,
                             std::string("\01L_OBJC_METACLASS_")+ClassName,
                             &CGM.getModule());
  GV->setSection("__OBJC,__meta_class,regular,no_dead_strip");
  UsedGlobals.push_back(GV);
  // FIXME: Why?
  GV->setAlignment(32);

  return GV;
}

/*
  struct objc_class_ext {
    uint32_t size;
    const char *weak_ivar_layout;
    struct _objc_property_list *properties;
  };
*/
llvm::Constant *
CGObjCMac::EmitClassExtension(const ObjCImplementationDecl *ID) {
  uint64_t Size = 
    CGM.getTargetData().getABITypeSize(ObjCTypes.ClassExtensionTy);

  std::vector<llvm::Constant*> Values(3);
  Values[0] = llvm::ConstantInt::get(ObjCTypes.IntTy, Size);
  // FIXME: Output weak_ivar_layout string.
  Values[1] = llvm::Constant::getNullValue(ObjCTypes.Int8PtrTy);
  Values[2] = EmitPropertyList(std::string("\01L_OBJC_$_PROP_LIST_") + 
                               ID->getName(),
                               ID->getClassInterface()->classprop_begin(),
                               ID->getClassInterface()->classprop_end());

  // Return null if no extension bits are used.
  if (Values[1]->isNullValue() && Values[2]->isNullValue())
    return llvm::Constant::getNullValue(ObjCTypes.ClassExtensionPtrTy);

  llvm::Constant *Init = 
    llvm::ConstantStruct::get(ObjCTypes.ClassExtensionTy, Values);
  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(ObjCTypes.ClassExtensionTy, false,
                             llvm::GlobalValue::InternalLinkage,
                             Init,
                             (std::string("\01L_OBJC_CLASSEXT_") +
                              ID->getName()),
                             &CGM.getModule());
  // No special section, but goes in llvm.used
  UsedGlobals.push_back(GV);
  
  return GV;
}

/*
  struct objc_ivar {
    char *ivar_name;
    char *ivar_type;
    int ivar_offset;
  };

  struct objc_ivar_list {
    int ivar_count;
    struct objc_ivar list[count];
  };
 */
llvm::Constant *CGObjCMac::EmitIvarList(const ObjCImplementationDecl *ID,
                                        bool ForClass,
                                        const llvm::Type *InterfaceTy) {
  std::vector<llvm::Constant*> Ivars, Ivar(3);

  // When emitting the root class GCC emits ivar entries for the
  // actual class structure. It is not clear if we need to follow this
  // behavior; for now lets try and get away with not doing it. If so,
  // the cleanest solution would be to make up an ObjCInterfaceDecl
  // for the class.
  if (ForClass)
    return llvm::Constant::getNullValue(ObjCTypes.IvarListPtrTy);

  const llvm::StructLayout *Layout =
    CGM.getTargetData().getStructLayout(cast<llvm::StructType>(InterfaceTy));
  for (ObjCInterfaceDecl::ivar_iterator 
         i = ID->getClassInterface()->ivar_begin(),
         e = ID->getClassInterface()->ivar_end(); i != e; ++i) {
    ObjCIvarDecl *V = *i;
    unsigned Offset = 
      Layout->getElementOffset(CGM.getTypes().getLLVMFieldNo(V));
    std::string TypeStr;
    llvm::SmallVector<const RecordType *, 8> EncodingRecordTypes;
    Ivar[0] = GetMethodVarName(V->getIdentifier());
    CGM.getContext().getObjCEncodingForType(V->getType(), TypeStr,
                                            EncodingRecordTypes);
    Ivar[1] = GetMethodVarType(TypeStr);
    Ivar[2] = llvm::ConstantInt::get(ObjCTypes.IntTy, Offset);
    Ivars.push_back(llvm::ConstantStruct::get(ObjCTypes.IvarTy,
                                              Ivar));
  }

  // Return null for empty list.
  if (Ivars.empty())
    return llvm::Constant::getNullValue(ObjCTypes.IvarListPtrTy);

  std::vector<llvm::Constant*> Values(2);
  Values[0] = llvm::ConstantInt::get(ObjCTypes.IntTy, Ivars.size());
  llvm::ArrayType *AT = llvm::ArrayType::get(ObjCTypes.IvarTy,
                                             Ivars.size());
  Values[1] = llvm::ConstantArray::get(AT, Ivars);
  llvm::Constant *Init = llvm::ConstantStruct::get(Values);

  const char *Prefix = (ForClass ? "\01L_OBJC_CLASS_VARIABLES_" :
                        "\01L_OBJC_INSTANCE_VARIABLES_");
  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(Init->getType(), false,
                             llvm::GlobalValue::InternalLinkage,
                             Init,
                             std::string(Prefix) + ID->getName(),
                             &CGM.getModule());
  if (ForClass) {
    GV->setSection("__OBJC,__cls_vars,regular,no_dead_strip");
    // FIXME: Why is this only here?
    GV->setAlignment(32);
  } else {
    GV->setSection("__OBJC,__instance_vars,regular,no_dead_strip");
  }
  UsedGlobals.push_back(GV);
  return llvm::ConstantExpr::getBitCast(GV,
                                        ObjCTypes.IvarListPtrTy);
}

/*
  struct objc_method {
    SEL method_name;
    char *method_types;
    void *method;
  };
  
  struct objc_method_list {
    struct objc_method_list *obsolete;
    int count;
    struct objc_method methods_list[count];
  };
*/
llvm::Constant *CGObjCMac::EmitMethodList(const std::string &Name,
                                          const char *Section,
                   llvm::SmallVector<ObjCMethodDecl*, 32>::const_iterator begin,
                   llvm::SmallVector<ObjCMethodDecl*, 32>::const_iterator end) {
  std::vector<llvm::Constant*> Methods, Method(3);

  for (; begin != end; ++begin) {
    ObjCMethodDecl *MD = *begin;

    Method[0] = 
      llvm::ConstantExpr::getBitCast(GetMethodVarName(MD->getSelector()),
                                     ObjCTypes.SelectorPtrTy);
    Method[1] = GetMethodVarType(MD);

    // FIXME: This is gross, we shouldn't be looking up by name.
    std::string Name;
    GetNameForMethod(MD, Name);
    Method[2] = 
      llvm::ConstantExpr::getBitCast(CGM.getModule().getFunction(Name),
                                     ObjCTypes.Int8PtrTy);
    Methods.push_back(llvm::ConstantStruct::get(ObjCTypes.MethodTy,
                                                Method));
  }

  // Return null for empty list.
  if (Methods.empty())
    return llvm::Constant::getNullValue(ObjCTypes.MethodListPtrTy);

  std::vector<llvm::Constant*> Values(3);
  Values[0] = llvm::Constant::getNullValue(ObjCTypes.Int8PtrTy);
  Values[1] = llvm::ConstantInt::get(ObjCTypes.IntTy, Methods.size());
  llvm::ArrayType *AT = llvm::ArrayType::get(ObjCTypes.MethodTy,
                                             Methods.size());
  Values[2] = llvm::ConstantArray::get(AT, Methods);
  llvm::Constant *Init = llvm::ConstantStruct::get(Values);

  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(Init->getType(), false,
                             llvm::GlobalValue::InternalLinkage,
                             Init,
                             Name,
                             &CGM.getModule());
  GV->setSection(Section);
  UsedGlobals.push_back(GV);
  return llvm::ConstantExpr::getBitCast(GV,
                                        ObjCTypes.MethodListPtrTy);
}

llvm::Function *CGObjCMac::GenerateMethod(const ObjCMethodDecl *OMD) { 
  const llvm::Type *ReturnTy = 
    CGM.getTypes().ConvertReturnType(OMD->getResultType());
  const llvm::Type *SelfTy = 
    CGM.getTypes().ConvertType(OMD->getSelfDecl()->getType());

  std::vector<const llvm::Type*> ArgTys;
  ArgTys.reserve(1 + 2 + OMD->param_size());

  // FIXME: This is not something we should have to be dealing with
  // here.
  bool useStructRet = 
    CodeGen::CodeGenFunction::hasAggregateLLVMType(OMD->getResultType());
  if (useStructRet) {
    ArgTys.push_back(llvm::PointerType::getUnqual(ReturnTy));
    ReturnTy = llvm::Type::VoidTy;
  }

  // Implicit arguments
  ArgTys.push_back(SelfTy);
  ArgTys.push_back(ObjCTypes.SelectorPtrTy);

  for (ObjCMethodDecl::param_const_iterator 
         i = OMD->param_begin(), e = OMD->param_end();
       i != e; ++i) {
    const llvm::Type *Ty = CGM.getTypes().ConvertType((*i)->getType());
    if (Ty->isSingleValueType()) {
      ArgTys.push_back(Ty);
    } else {
      ArgTys.push_back(llvm::PointerType::getUnqual(Ty));
    }
  }

  std::string Name;
  GetNameForMethod(OMD, Name);

  llvm::Function *Method = 
    llvm::Function::Create(llvm::FunctionType::get(ReturnTy,
                                                   ArgTys,
                                                   OMD->isVariadic()),
                           llvm::GlobalValue::InternalLinkage,
                           Name,
                           &CGM.getModule());

  unsigned Offset = 3; // Return plus self and selector implicit args.
  if (useStructRet) {
    Method->addParamAttr(1, llvm::ParamAttr::StructRet);
    ++Offset;
  }

  // FIXME: This is horrible, we need to be reusing the machinery in
  // CodeGenModule.cpp (SetFunctionAttributes).
  for (ObjCMethodDecl::param_const_iterator 
         i = OMD->param_begin(), e = OMD->param_end();
       i != e; ++i, ++Offset) {
    const llvm::Type *Ty = CGM.getTypes().ConvertType((*i)->getType());
    if (!Ty->isSingleValueType())
      Method->addParamAttr(Offset, llvm::ParamAttr::ByVal);
  }

  return Method;
}

llvm::Function *CGObjCMac::ModuleInitFunction() { 
  // Abuse this interface function as a place to finalize.
  FinishModule();

  return NULL;
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
  std::vector<llvm::Constant*> Values(5);
  unsigned NumClasses = DefinedClasses.size();
  unsigned NumCategories = DefinedCategories.size();

  Values[0] = llvm::ConstantInt::get(ObjCTypes.LongTy, 0);
  Values[1] = llvm::Constant::getNullValue(ObjCTypes.SelectorPtrTy);
  Values[2] = llvm::ConstantInt::get(ObjCTypes.ShortTy, NumClasses);
  Values[3] = llvm::ConstantInt::get(ObjCTypes.ShortTy, NumCategories);

  // The runtime expects exactly the list of defined classes followed
  // by the list of defined categories, in a single array.
  std::vector<llvm::Constant*> Symbols(NumClasses + NumCategories);
  for (unsigned i=0; i<NumClasses; i++)
    Symbols[i] = llvm::ConstantExpr::getBitCast(DefinedClasses[i],
                                                ObjCTypes.Int8PtrTy);
  for (unsigned i=0; i<NumCategories; i++)
    Symbols[NumClasses + i] = 
      llvm::ConstantExpr::getBitCast(DefinedCategories[i],
                                     ObjCTypes.Int8PtrTy);

  Values[4] = 
    llvm::ConstantArray::get(llvm::ArrayType::get(ObjCTypes.Int8PtrTy,
                                                  NumClasses + NumCategories),
                             Symbols);

  llvm::Constant *Init = llvm::ConstantStruct::get(Values);  

  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(Init->getType(), false,
                             llvm::GlobalValue::InternalLinkage,
                             Init,
                             "\01L_OBJC_SYMBOLS", 
                             &CGM.getModule());
  GV->setSection("__OBJC,__symbols,regular,no_dead_strip");
  UsedGlobals.push_back(GV);
  return llvm::ConstantExpr::getBitCast(GV, ObjCTypes.SymtabPtrTy);
}

llvm::Value *CGObjCMac::EmitClassRef(llvm::IRBuilder<> &Builder, 
                                     const ObjCInterfaceDecl *ID) {
  llvm::GlobalVariable *&Entry = ClassReferences[ID->getIdentifier()];
  
  if (!Entry) {
    llvm::Constant *Casted = 
      llvm::ConstantExpr::getBitCast(GetClassName(ID->getIdentifier()),
                                     ObjCTypes.ClassPtrTy);
    Entry = 
      new llvm::GlobalVariable(ObjCTypes.ClassPtrTy, false,
                               llvm::GlobalValue::InternalLinkage,
                               Casted, "\01L_OBJC_CLASS_REFERENCES_",
                               &CGM.getModule());
    Entry->setSection("__OBJC,__cls_refs,literal_pointers,no_dead_strip");
    UsedGlobals.push_back(Entry);
  }

  return Builder.CreateLoad(Entry, false, "tmp");
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

// FIXME: Merge into a single cstring creation function.
llvm::Constant *CGObjCMac::GetMethodVarName(IdentifierInfo *ID) {
  return GetMethodVarName(CGM.getContext().Selectors.getNullarySelector(ID));
}

// FIXME: Merge into a single cstring creation function.
llvm::Constant *CGObjCMac::GetMethodVarName(const std::string &Name) {
  return GetMethodVarName(&CGM.getContext().Idents.get(Name));
}

llvm::Constant *CGObjCMac::GetMethodVarType(const std::string &Name) {
  llvm::GlobalVariable *&Entry = MethodVarTypes[Name];

  if (!Entry) {
    llvm::Constant *C = llvm::ConstantArray::get(Name);
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

// FIXME: Merge into a single cstring creation function.
llvm::Constant *CGObjCMac::GetMethodVarType(ObjCMethodDecl *D) {
  std::string TypeStr;
  CGM.getContext().getObjCEncodingForMethodDecl(D, TypeStr);
  return GetMethodVarType(TypeStr);
}

// FIXME: Merge into a single cstring creation function.
llvm::Constant *CGObjCMac::GetPropertyName(IdentifierInfo *Ident) {
  llvm::GlobalVariable *&Entry = PropertyNames[Ident];
  
  if (!Entry) {
    llvm::Constant *C = llvm::ConstantArray::get(Ident->getName());
    Entry = 
      new llvm::GlobalVariable(C->getType(), false, 
                               llvm::GlobalValue::InternalLinkage,
                               C, "\01L_OBJC_PROP_NAME_ATTR_", 
                               &CGM.getModule());
    Entry->setSection("__TEXT,__cstring,cstring_literals");
    UsedGlobals.push_back(Entry);
  }

  return getConstantGEP(Entry, 0, 0);
}

// FIXME: Merge into a single cstring creation function.
llvm::Constant *CGObjCMac::GetPropertyType(const ObjCPropertyDecl *PD) {
  std::string TypeStr("MOOO!");
  //CGM.getContext().getObjCEncodingForMethodDecl(D, TypeStr);
  return GetPropertyName(&CGM.getContext().Idents.get(TypeStr));
}

void CGObjCMac::GetNameForMethod(const ObjCMethodDecl *D, 
                                 std::string &NameOut) {
  // FIXME: Find the mangling GCC uses.
  std::stringstream s;
  s << (D->isInstance() ? "-" : "+");
  s << "[";
  s << D->getClassInterface()->getName();
  s << " ";
  s << D->getSelector().getName();
  s << "]";
  NameOut = s.str();
}

void CGObjCMac::FinishModule() {
  EmitModuleInfo();

  std::vector<llvm::Constant*> Used;

  for (std::vector<llvm::GlobalVariable*>::iterator i = UsedGlobals.begin(), 
         e = UsedGlobals.end(); i != e; ++i) {
    Used.push_back(llvm::ConstantExpr::getBitCast(*i, ObjCTypes.Int8PtrTy));
  }
  
  llvm::ArrayType *AT = llvm::ArrayType::get(ObjCTypes.Int8PtrTy, Used.size());
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
    CFConstantStringClassReference(0)
{
  CodeGen::CodeGenTypes &Types = CGM.getTypes();
  ASTContext &Ctx = CGM.getContext();

  ShortTy = Types.ConvertType(Ctx.ShortTy);
  IntTy = Types.ConvertType(Ctx.IntTy);
  LongTy = Types.ConvertType(Ctx.LongTy);
  Int8PtrTy = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
  
  ObjectPtrTy = Types.ConvertType(Ctx.getObjCIdType());
  SelectorPtrTy = Types.ConvertType(Ctx.getObjCSelType());
  
  // FIXME: It would be nice to unify this with the opaque type, so
  // that the IR comes out a bit cleaner.
  const llvm::Type *T = Types.ConvertType(Ctx.getObjCProtoType());
  ExternalProtocolPtrTy = llvm::PointerType::getUnqual(T);

  MethodDescriptionTy = 
    llvm::StructType::get(SelectorPtrTy,
                          Int8PtrTy,
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

  PropertyTy = llvm::StructType::get(Int8PtrTy,
                                     Int8PtrTy,
                                     NULL);
  CGM.getModule().addTypeName("struct._objc_property", 
                              PropertyTy);

  PropertyListTy = llvm::StructType::get(IntTy,
                                         IntTy,
                                         llvm::ArrayType::get(PropertyTy, 0),
                                         NULL);
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
                            Int8PtrTy,
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

  // Class description structures

  IvarTy = llvm::StructType::get(Int8PtrTy, 
                                 Int8PtrTy, 
                                 IntTy, 
                                 NULL);
  CGM.getModule().addTypeName("struct._objc_ivar", IvarTy);

  IvarListTy = llvm::OpaqueType::get();
  CGM.getModule().addTypeName("struct._objc_ivar_list", IvarListTy);
  IvarListPtrTy = llvm::PointerType::getUnqual(IvarListTy);

  MethodTy = llvm::StructType::get(SelectorPtrTy,
                                   Int8PtrTy,
                                   Int8PtrTy,
                                   NULL);
  CGM.getModule().addTypeName("struct._objc_method", MethodTy);
  
  MethodListTy = llvm::OpaqueType::get();
  CGM.getModule().addTypeName("struct._objc_method_list", MethodListTy);
  MethodListPtrTy = llvm::PointerType::getUnqual(MethodListTy);

  CacheTy = llvm::OpaqueType::get();
  CGM.getModule().addTypeName("struct._objc_cache", CacheTy);
  CachePtrTy = llvm::PointerType::getUnqual(CacheTy);

  ClassExtensionTy = 
    llvm::StructType::get(IntTy,
                          Int8PtrTy,
                          PropertyListPtrTy,
                          NULL);
  CGM.getModule().addTypeName("struct._objc_class_extension", ClassExtensionTy);
  ClassExtensionPtrTy = llvm::PointerType::getUnqual(ClassExtensionTy);

  llvm::PATypeHolder ClassTyHolder = llvm::OpaqueType::get();

  T = llvm::StructType::get(llvm::PointerType::getUnqual(ClassTyHolder),
                            llvm::PointerType::getUnqual(ClassTyHolder),
                            Int8PtrTy,
                            LongTy,
                            LongTy,
                            LongTy,
                            IvarListPtrTy,
                            MethodListPtrTy,
                            CachePtrTy,
                            ProtocolListPtrTy,
                            Int8PtrTy,
                            ClassExtensionPtrTy,
                            NULL);
  cast<llvm::OpaqueType>(ClassTyHolder.get())->refineAbstractTypeTo(T);
  
  ClassTy = cast<llvm::StructType>(ClassTyHolder.get());
  CGM.getModule().addTypeName("struct._objc_class", ClassTy);
  ClassPtrTy = llvm::PointerType::getUnqual(ClassTy);

  CategoryTy = llvm::StructType::get(Int8PtrTy,
                                     Int8PtrTy,
                                     MethodListPtrTy,
                                     MethodListPtrTy,
                                     ProtocolListPtrTy,
                                     IntTy,
                                     PropertyListPtrTy,
                                     NULL);
  CGM.getModule().addTypeName("struct._objc_category", CategoryTy);

  SuperTy = 
    llvm::StructType::get(ObjectPtrTy,
                          ClassPtrTy,
                          NULL);
  CGM.getModule().addTypeName("struct._objc_super", 
                              SuperTy);
  SuperPtrTy = llvm::PointerType::getUnqual(SuperTy);

  // Global metadata structures

  SymtabTy = llvm::StructType::get(LongTy,
                                   SelectorPtrTy,
                                   ShortTy,
                                   ShortTy,
                                   llvm::ArrayType::get(Int8PtrTy, 0),
                                   NULL);
  CGM.getModule().addTypeName("struct._objc_symtab", SymtabTy);
  SymtabPtrTy = llvm::PointerType::getUnqual(SymtabTy);

  ModuleTy = 
    llvm::StructType::get(LongTy,
                          LongTy,
                          Int8PtrTy,
                          SymtabPtrTy,
                          NULL);
  CGM.getModule().addTypeName("struct._objc_module", ModuleTy);

  // Message send functions

  std::vector<const llvm::Type*> Params;
  Params.push_back(ObjectPtrTy);
  Params.push_back(SelectorPtrTy);
  MessageSendFn = llvm::Function::Create(llvm::FunctionType::get(ObjectPtrTy,
                                                                 Params,
                                                                 true),
                                         llvm::Function::ExternalLinkage,
                                         "objc_msgSend",
                                         &CGM.getModule());
  
  Params.clear();
  Params.push_back(Int8PtrTy);
  Params.push_back(ObjectPtrTy);
  Params.push_back(SelectorPtrTy);
  MessageSendStretFn = 
    llvm::Function::Create(llvm::FunctionType::get(llvm::Type::VoidTy,
                                                   Params,
                                                   true),
                             llvm::Function::ExternalLinkage,
                             "objc_msgSend_stret",
                             &CGM.getModule());
  
  Params.clear();
  Params.push_back(SuperPtrTy);
  Params.push_back(SelectorPtrTy);
  MessageSendSuperFn = 
    llvm::Function::Create(llvm::FunctionType::get(ObjectPtrTy,
                                                   Params,
                                                   true),
                           llvm::Function::ExternalLinkage,
                           "objc_msgSendSuper",
                           &CGM.getModule());

  Params.clear();
  Params.push_back(Int8PtrTy);
  Params.push_back(SuperPtrTy);
  Params.push_back(SelectorPtrTy);
  MessageSendSuperStretFn = 
    llvm::Function::Create(llvm::FunctionType::get(llvm::Type::VoidTy,
                                                   Params,
                                                   true),
                           llvm::Function::ExternalLinkage,
                           "objc_msgSendSuper_stret",
                           &CGM.getModule());
}

ObjCTypesHelper::~ObjCTypesHelper() {
}

const llvm::StructType *ObjCTypesHelper::getCFStringType() {
  if (!CFStringType) {
    CFStringType = 
      llvm::StructType::get(llvm::PointerType::getUnqual(llvm::Type::Int32Ty), 
                            llvm::Type::Int32Ty,
                            Int8PtrTy,
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

llvm::Value *ObjCTypesHelper::getMessageSendFn(bool IsSuper, 
                                               const llvm::Type *ReturnTy) {
  llvm::Function *F;
  llvm::FunctionType *CallFTy;
  
  // FIXME: Should we be caching any of this?
  if (!ReturnTy->isSingleValueType()) {
    F = IsSuper ? MessageSendSuperStretFn : MessageSendStretFn;
    std::vector<const llvm::Type*> Params(3);
    Params[0] = llvm::PointerType::getUnqual(ReturnTy);
    Params[1] = IsSuper ? SuperPtrTy : ObjectPtrTy;
    Params[2] = SelectorPtrTy;
    CallFTy = llvm::FunctionType::get(llvm::Type::VoidTy, Params, true);
  } else { // XXX floating point?
    F = IsSuper ? MessageSendSuperFn : MessageSendFn;
    std::vector<const llvm::Type*> Params(2);
    Params[0] = IsSuper ? SuperPtrTy : ObjectPtrTy;
    Params[1] = SelectorPtrTy;
    CallFTy = llvm::FunctionType::get(ReturnTy, Params, true);
  }

  return llvm::ConstantExpr::getBitCast(F, 
                                        llvm::PointerType::getUnqual(CallFTy));
}

/* *** */

CodeGen::CGObjCRuntime *
CodeGen::CreateMacObjCRuntime(CodeGen::CodeGenModule &CGM) {
  return new CGObjCMac(CGM);
}
