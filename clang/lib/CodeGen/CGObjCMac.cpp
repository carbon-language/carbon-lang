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
#include "llvm/ADT/DenseSet.h"
#include "llvm/Target/TargetData.h"
#include <sstream>

using namespace clang;
using namespace CodeGen;

namespace {

  typedef std::vector<llvm::Constant*> ConstantVector;

  // FIXME: We should find a nicer way to make the labels for
  // metadata, string concatenation is lame.

class ObjCCommonTypesHelper {
protected:
  CodeGen::CodeGenModule &CGM;
  
public:
  const llvm::Type *ShortTy, *IntTy, *LongTy;
  const llvm::Type *Int8PtrTy;
  
  /// ObjectPtrTy - LLVM type for object handles (typeof(id))
  const llvm::Type *ObjectPtrTy;
  
  /// PtrObjectPtrTy - LLVM type for id *
  const llvm::Type *PtrObjectPtrTy;
  
  /// SelectorPtrTy - LLVM type for selector handles (typeof(SEL))
  const llvm::Type *SelectorPtrTy;
  /// ProtocolPtrTy - LLVM type for external protocol handles
  /// (typeof(Protocol))
  const llvm::Type *ExternalProtocolPtrTy;
  
  // SuperCTy - clang type for struct objc_super.
  QualType SuperCTy;
  // SuperPtrCTy - clang type for struct objc_super *.
  QualType SuperPtrCTy;
  
  /// SuperTy - LLVM type for struct objc_super.
  const llvm::StructType *SuperTy;
  /// SuperPtrTy - LLVM type for struct objc_super *.
  const llvm::Type *SuperPtrTy;
  
  /// PropertyTy - LLVM type for struct objc_property (struct _prop_t
  /// in GCC parlance).
  const llvm::StructType *PropertyTy;
  
  /// PropertyListTy - LLVM type for struct objc_property_list
  /// (_prop_list_t in GCC parlance).
  const llvm::StructType *PropertyListTy;
  /// PropertyListPtrTy - LLVM type for struct objc_property_list*.
  const llvm::Type *PropertyListPtrTy;
  
  // MethodTy - LLVM type for struct objc_method.
  const llvm::StructType *MethodTy;
  
  /// CacheTy - LLVM type for struct objc_cache.
  const llvm::Type *CacheTy;
  /// CachePtrTy - LLVM type for struct objc_cache *.
  const llvm::Type *CachePtrTy;
  
  llvm::Function *GetPropertyFn, *SetPropertyFn;
  
  llvm::Function *EnumerationMutationFn;
  
  /// GcReadWeakFn -- LLVM objc_read_weak (id *src) function.
  llvm::Function *GcReadWeakFn;
  
  /// GcAssignWeakFn -- LLVM objc_assign_weak function.
  llvm::Function *GcAssignWeakFn;
  
  /// GcAssignGlobalFn -- LLVM objc_assign_global function.
  llvm::Function *GcAssignGlobalFn;
  
  /// GcAssignIvarFn -- LLVM objc_assign_ivar function.
  llvm::Function *GcAssignIvarFn;
  
  /// GcAssignStrongCastFn -- LLVM objc_assign_strongCast function.
  llvm::Function *GcAssignStrongCastFn;
    
  ObjCCommonTypesHelper(CodeGen::CodeGenModule &cgm);
  ~ObjCCommonTypesHelper(){}
};

/// ObjCTypesHelper - Helper class that encapsulates lazy
/// construction of varies types used during ObjC generation.
class ObjCTypesHelper : public ObjCCommonTypesHelper {
private:
  
  llvm::Function *MessageSendFn, *MessageSendStretFn, *MessageSendFpretFn;
  llvm::Function *MessageSendSuperFn, *MessageSendSuperStretFn, 
    *MessageSendSuperFpretFn;

public:
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
  // IvarTy - LLVM type for struct objc_ivar.
  const llvm::StructType *IvarTy;
  /// IvarListTy - LLVM type for struct objc_ivar_list.
  const llvm::Type *IvarListTy;
  /// IvarListPtrTy - LLVM type for struct objc_ivar_list *.
  const llvm::Type *IvarListPtrTy;
  /// MethodListTy - LLVM type for struct objc_method_list.
  const llvm::Type *MethodListTy;
  /// MethodListPtrTy - LLVM type for struct objc_method_list *.
  const llvm::Type *MethodListPtrTy;
  
  /// ExceptionDataTy - LLVM type for struct _objc_exception_data.
  const llvm::Type *ExceptionDataTy;
  
  /// ExceptionThrowFn - LLVM objc_exception_throw function.
  llvm::Function *ExceptionThrowFn;
  
  /// ExceptionTryEnterFn - LLVM objc_exception_try_enter function.
  llvm::Function *ExceptionTryEnterFn;

  /// ExceptionTryExitFn - LLVM objc_exception_try_exit function.
  llvm::Function *ExceptionTryExitFn;

  /// ExceptionExtractFn - LLVM objc_exception_extract function.
  llvm::Function *ExceptionExtractFn;
  
  /// ExceptionMatchFn - LLVM objc_exception_match function.
  llvm::Function *ExceptionMatchFn;
  
  /// SetJmpFn - LLVM _setjmp function.
  llvm::Function *SetJmpFn;
  
  /// SyncEnterFn - LLVM object_sync_enter function.
  llvm::Function *SyncEnterFn;
  
  /// SyncExitFn - LLVM object_sync_exit function.
  llvm::Function *SyncExitFn;
  
public:
  ObjCTypesHelper(CodeGen::CodeGenModule &cgm);
  ~ObjCTypesHelper() {}


  llvm::Function *getSendFn(bool IsSuper) {
    return IsSuper ? MessageSendSuperFn : MessageSendFn;
  }

  llvm::Function *getSendStretFn(bool IsSuper) {
    return IsSuper ? MessageSendSuperStretFn : MessageSendStretFn;
  }

  llvm::Function *getSendFpretFn(bool IsSuper) {
    return IsSuper ? MessageSendSuperFpretFn : MessageSendFpretFn;
  }
};

/// ObjCNonFragileABITypesHelper - will have all types needed by objective-c's
/// modern abi
class ObjCNonFragileABITypesHelper : public ObjCCommonTypesHelper {
public:
  llvm::Function *MessageSendFixupFn, *MessageSendFpretFixupFn,
                 *MessageSendStretFixupFn, *MessageSendIdFixupFn,
                 *MessageSendIdStretFixupFn, *MessageSendSuper2FixupFn,
                 *MessageSendSuper2StretFixupFn;
          
  // MethodListnfABITy - LLVM for struct _method_list_t
  const llvm::StructType *MethodListnfABITy;
  
  // MethodListnfABIPtrTy - LLVM for struct _method_list_t*
  const llvm::Type *MethodListnfABIPtrTy;
  
  // ProtocolnfABITy = LLVM for struct _protocol_t
  const llvm::StructType *ProtocolnfABITy;
  
  // ProtocolListnfABITy - LLVM for struct _objc_protocol_list
  const llvm::StructType *ProtocolListnfABITy;
  
  // ProtocolListnfABIPtrTy - LLVM for struct _objc_protocol_list*
  const llvm::Type *ProtocolListnfABIPtrTy;
  
  // ClassnfABITy - LLVM for struct _class_t
  const llvm::StructType *ClassnfABITy;
  
  // ClassnfABIPtrTy - LLVM for struct _class_t*
  const llvm::Type *ClassnfABIPtrTy;
  
  // IvarnfABITy - LLVM for struct _ivar_t
  const llvm::StructType *IvarnfABITy;
  
  // IvarListnfABITy - LLVM for struct _ivar_list_t
  const llvm::StructType *IvarListnfABITy;
  
  // IvarListnfABIPtrTy = LLVM for struct _ivar_list_t*
  const llvm::Type *IvarListnfABIPtrTy;
  
  // ClassRonfABITy - LLVM for struct _class_ro_t
  const llvm::StructType *ClassRonfABITy;
  
  // ImpnfABITy - LLVM for id (*)(id, SEL, ...)
  const llvm::Type *ImpnfABITy;
  
  // CategorynfABITy - LLVM for struct _category_t
  const llvm::StructType *CategorynfABITy;
  
  // New types for nonfragile abi messaging.
  
  // MessageRefTy - LLVM for:
  // struct _message_ref_t {
  //   IMP messenger;
  //   SEL name;
  // };
  const llvm::StructType *MessageRefTy;
  // MessageRefCTy - clang type for struct _message_ref_t
  QualType MessageRefCTy;
  
  // MessageRefPtrTy - LLVM for struct _message_ref_t*
  const llvm::Type *MessageRefPtrTy;
  // MessageRefCPtrTy - clang type for struct _message_ref_t*
  QualType MessageRefCPtrTy;
  
  // MessengerTy - Type of the messenger (shown as IMP above)
  const llvm::FunctionType *MessengerTy;
  
  // SuperMessageRefTy - LLVM for:
  // struct _super_message_ref_t {
  //   SUPER_IMP messenger;
  //   SEL name;
  // };
  const llvm::StructType *SuperMessageRefTy;
  
  // SuperMessageRefPtrTy - LLVM for struct _super_message_ref_t*
  const llvm::Type *SuperMessageRefPtrTy;
  
  ObjCNonFragileABITypesHelper(CodeGen::CodeGenModule &cgm);
  ~ObjCNonFragileABITypesHelper(){}
};
  
class CGObjCCommonMac : public CodeGen::CGObjCRuntime {
protected:
  CodeGen::CodeGenModule &CGM;
  // FIXME! May not be needing this after all.
  unsigned ObjCABI;
  
  /// LazySymbols - Symbols to generate a lazy reference for. See
  /// DefinedSymbols and FinishModule().
  std::set<IdentifierInfo*> LazySymbols;
  
  /// DefinedSymbols - External symbols which are defined by this
  /// module. The symbols in this list and LazySymbols are used to add
  /// special linker symbols which ensure that Objective-C modules are
  /// linked properly.
  std::set<IdentifierInfo*> DefinedSymbols;
  
  /// ClassNames - uniqued class names.
  llvm::DenseMap<IdentifierInfo*, llvm::GlobalVariable*> ClassNames;
  
  /// MethodVarNames - uniqued method variable names.
  llvm::DenseMap<Selector, llvm::GlobalVariable*> MethodVarNames;
  
  /// MethodVarTypes - uniqued method type signatures. We have to use
  /// a StringMap here because have no other unique reference.
  llvm::StringMap<llvm::GlobalVariable*> MethodVarTypes;
  
  /// MethodDefinitions - map of methods which have been defined in
  /// this translation unit.
  llvm::DenseMap<const ObjCMethodDecl*, llvm::Function*> MethodDefinitions;
  
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
  
  /// DefinedProtocols - Protocols which have actually been
  /// defined. We should not need this, see FIXME in GenerateProtocol.
  llvm::DenseSet<IdentifierInfo*> DefinedProtocols;
  
  /// DefinedClasses - List of defined classes.
  std::vector<llvm::GlobalValue*> DefinedClasses;
  
  /// DefinedCategories - List of defined categories.
  std::vector<llvm::GlobalValue*> DefinedCategories;
  
  /// UsedGlobals - List of globals to pack into the llvm.used metadata
  /// to prevent them from being clobbered.
  std::vector<llvm::GlobalVariable*> UsedGlobals;

  /// GetNameForMethod - Return a name for the given method.
  /// \param[out] NameOut - The return value.
  void GetNameForMethod(const ObjCMethodDecl *OMD,
                        const ObjCContainerDecl *CD,
                        std::string &NameOut);
  
  /// GetMethodVarName - Return a unique constant for the given
  /// selector's name. The return value has type char *.
  llvm::Constant *GetMethodVarName(Selector Sel);
  llvm::Constant *GetMethodVarName(IdentifierInfo *Ident);
  llvm::Constant *GetMethodVarName(const std::string &Name);
  
  /// GetMethodVarType - Return a unique constant for the given
  /// selector's name. The return value has type char *.
  
  // FIXME: This is a horrible name.
  llvm::Constant *GetMethodVarType(const ObjCMethodDecl *D);
  llvm::Constant *GetMethodVarType(const std::string &Name);
  
  /// GetPropertyName - Return a unique constant for the given
  /// name. The return value has type char *.
  llvm::Constant *GetPropertyName(IdentifierInfo *Ident);
  
  // FIXME: This can be dropped once string functions are unified.
  llvm::Constant *GetPropertyTypeString(const ObjCPropertyDecl *PD,
                                        const Decl *Container);
  
  /// GetClassName - Return a unique constant for the given selector's
  /// name. The return value has type char *.
  llvm::Constant *GetClassName(IdentifierInfo *Ident);
  
  const RecordDecl *GetFirstIvarInRecord(const ObjCInterfaceDecl *OID,
                                         RecordDecl::field_iterator &FIV,
                                         RecordDecl::field_iterator &PIV);  
  /// EmitPropertyList - Emit the given property list. The return
  /// value has type PropertyListPtrTy.
  llvm::Constant *EmitPropertyList(const std::string &Name,
                                   const Decl *Container, 
                                   const ObjCContainerDecl *OCD,
                                   const ObjCCommonTypesHelper &ObjCTypes);
  
  /// GetProtocolRef - Return a reference to the internal protocol
  /// description, creating an empty one if it has not been
  /// defined. The return value has type ProtocolPtrTy.
  llvm::Constant *GetProtocolRef(const ObjCProtocolDecl *PD);
  
public:
  CGObjCCommonMac(CodeGen::CodeGenModule &cgm) : CGM(cgm)
  { }
  
  virtual llvm::Constant *GenerateConstantString(const std::string &String);
  
  virtual llvm::Function *GenerateMethod(const ObjCMethodDecl *OMD,
                                         const ObjCContainerDecl *CD=0);
  
  virtual void GenerateProtocol(const ObjCProtocolDecl *PD);
  
  /// GetOrEmitProtocol - Get the protocol object for the given
  /// declaration, emitting it if necessary. The return value has type
  /// ProtocolPtrTy.
  virtual llvm::Constant *GetOrEmitProtocol(const ObjCProtocolDecl *PD)=0;
  
  /// GetOrEmitProtocolRef - Get a forward reference to the protocol
  /// object for the given declaration, emitting it if needed. These
  /// forward references will be filled in with empty bodies if no
  /// definition is seen. The return value has type ProtocolPtrTy.
  virtual llvm::Constant *GetOrEmitProtocolRef(const ObjCProtocolDecl *PD)=0;
};
  
class CGObjCMac : public CGObjCCommonMac {
private:
  ObjCTypesHelper ObjCTypes;
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
  llvm::Value *EmitClassRef(CGBuilderTy &Builder, 
                            const ObjCInterfaceDecl *ID);

  CodeGen::RValue EmitMessageSend(CodeGen::CodeGenFunction &CGF,
                                  QualType ResultType,
                                  Selector Sel,
                                  llvm::Value *Arg0,
                                  QualType Arg0Ty,
                                  bool IsSuper,
                                  const CallArgList &CallArgs);

  /// EmitIvarList - Emit the ivar list for the given
  /// implementation. If ForClass is true the list of class ivars
  /// (i.e. metaclass ivars) is emitted, otherwise the list of
  /// interface ivars will be emitted. The return value has type
  /// IvarListPtrTy.
  llvm::Constant *EmitIvarList(const ObjCImplementationDecl *ID,
                               bool ForClass);
                               
  /// EmitMetaClass - Emit a forward reference to the class structure
  /// for the metaclass of the given interface. The return value has
  /// type ClassPtrTy.
  llvm::Constant *EmitMetaClassRef(const ObjCInterfaceDecl *ID);

  /// EmitMetaClass - Emit a class structure for the metaclass of the
  /// given implementation. The return value has type ClassPtrTy.
  llvm::Constant *EmitMetaClass(const ObjCImplementationDecl *ID,
                                llvm::Constant *Protocols,
                                const llvm::Type *InterfaceTy,
                                const ConstantVector &Methods);
  
  llvm::Constant *GetMethodConstant(const ObjCMethodDecl *MD);
  
  llvm::Constant *GetMethodDescriptionConstant(const ObjCMethodDecl *MD);

  /// EmitMethodList - Emit the method list for the given
  /// implementation. The return value has type MethodListPtrTy.
  llvm::Constant *EmitMethodList(const std::string &Name,
                                 const char *Section,
                                 const ConstantVector &Methods);

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
  llvm::Constant *EmitMethodDescList(const std::string &Name,
                                     const char *Section,
                                     const ConstantVector &Methods);

  /// GetOrEmitProtocol - Get the protocol object for the given
  /// declaration, emitting it if necessary. The return value has type
  /// ProtocolPtrTy.
  virtual llvm::Constant *GetOrEmitProtocol(const ObjCProtocolDecl *PD);

  /// GetOrEmitProtocolRef - Get a forward reference to the protocol
  /// object for the given declaration, emitting it if needed. These
  /// forward references will be filled in with empty bodies if no
  /// definition is seen. The return value has type ProtocolPtrTy.
  virtual llvm::Constant *GetOrEmitProtocolRef(const ObjCProtocolDecl *PD);

  /// EmitProtocolExtension - Generate the protocol extension
  /// structure used to store optional instance and class methods, and
  /// protocol properties. The return value has type
  /// ProtocolExtensionPtrTy.
  llvm::Constant *
  EmitProtocolExtension(const ObjCProtocolDecl *PD,
                        const ConstantVector &OptInstanceMethods,
                        const ConstantVector &OptClassMethods);

  /// EmitProtocolList - Generate the list of referenced
  /// protocols. The return value has type ProtocolListPtrTy.
  llvm::Constant *EmitProtocolList(const std::string &Name,
                                   ObjCProtocolDecl::protocol_iterator begin,
                                   ObjCProtocolDecl::protocol_iterator end);

  /// EmitSelector - Return a Value*, of type ObjCTypes.SelectorPtrTy,
  /// for the given selector.
  llvm::Value *EmitSelector(CGBuilderTy &Builder, Selector Sel);
  
  public:
  CGObjCMac(CodeGen::CodeGenModule &cgm);

  virtual llvm::Function *ModuleInitFunction();
  
  virtual CodeGen::RValue GenerateMessageSend(CodeGen::CodeGenFunction &CGF,
                                              QualType ResultType,
                                              Selector Sel,
                                              llvm::Value *Receiver,
                                              bool IsClassMessage,
                                              const CallArgList &CallArgs);

  virtual CodeGen::RValue 
  GenerateMessageSendSuper(CodeGen::CodeGenFunction &CGF,
                           QualType ResultType,
                           Selector Sel,
                           const ObjCInterfaceDecl *Class,
                           llvm::Value *Receiver,
                           bool IsClassMessage,
                           const CallArgList &CallArgs);
  
  virtual llvm::Value *GetClass(CGBuilderTy &Builder,
                                const ObjCInterfaceDecl *ID);

  virtual llvm::Value *GetSelector(CGBuilderTy &Builder, Selector Sel);
  
  virtual void GenerateCategory(const ObjCCategoryImplDecl *CMD);

  virtual void GenerateClass(const ObjCImplementationDecl *ClassDecl);

  virtual llvm::Value *GenerateProtocolRef(CGBuilderTy &Builder,
                                           const ObjCProtocolDecl *PD);
    
  virtual llvm::Function *GetPropertyGetFunction();
  virtual llvm::Function *GetPropertySetFunction();
  virtual llvm::Function *EnumerationMutationFunction();
  
  virtual void EmitTryOrSynchronizedStmt(CodeGen::CodeGenFunction &CGF,
                                         const Stmt &S);
  virtual void EmitThrowStmt(CodeGen::CodeGenFunction &CGF,
                             const ObjCAtThrowStmt &S);
  virtual llvm::Value * EmitObjCWeakRead(CodeGen::CodeGenFunction &CGF,
                                         llvm::Value *AddrWeakObj); 
  virtual void EmitObjCWeakAssign(CodeGen::CodeGenFunction &CGF,
                                  llvm::Value *src, llvm::Value *dst); 
  virtual void EmitObjCGlobalAssign(CodeGen::CodeGenFunction &CGF,
                                    llvm::Value *src, llvm::Value *dest);
  virtual void EmitObjCIvarAssign(CodeGen::CodeGenFunction &CGF,
                                  llvm::Value *src, llvm::Value *dest);
  virtual void EmitObjCStrongCastAssign(CodeGen::CodeGenFunction &CGF,
                                        llvm::Value *src, llvm::Value *dest);
  
  virtual LValue EmitObjCValueForIvar(CodeGen::CodeGenFunction &CGF,
                                      QualType ObjectTy,
                                      llvm::Value *BaseValue,
                                      const ObjCIvarDecl *Ivar,
                                      const FieldDecl *Field,
                                      unsigned CVRQualifiers);
  virtual llvm::Value *EmitIvarOffset(CodeGen::CodeGenFunction &CGF,
                                      ObjCInterfaceDecl *Interface,
                                      const ObjCIvarDecl *Ivar);
};
  
class CGObjCNonFragileABIMac : public CGObjCCommonMac {
private:
  ObjCNonFragileABITypesHelper ObjCTypes;
  llvm::GlobalVariable* ObjCEmptyCacheVar;
  llvm::GlobalVariable* ObjCEmptyVtableVar;
  /// MetaClassReferences - uniqued meta class references.
  llvm::DenseMap<IdentifierInfo*, llvm::GlobalVariable*> MetaClassReferences;
  
  /// FinishNonFragileABIModule - Write out global data structures at the end of
  /// processing a translation unit.
  void FinishNonFragileABIModule();
  
  llvm::GlobalVariable * BuildClassRoTInitializer(unsigned flags, 
                                unsigned InstanceStart,
                                unsigned InstanceSize,
                                const ObjCImplementationDecl *ID);
  llvm::GlobalVariable * BuildClassMetaData(std::string &ClassName,
                                            llvm::Constant *IsAGV, 
                                            llvm::Constant *SuperClassGV,
                                            llvm::Constant *ClassRoGV,
                                            bool HiddenVisibility);
  
  llvm::Constant *GetMethodConstant(const ObjCMethodDecl *MD);
  
  llvm::Constant *GetMethodDescriptionConstant(const ObjCMethodDecl *MD);
  
  /// EmitMethodList - Emit the method list for the given
  /// implementation. The return value has type MethodListnfABITy.
  llvm::Constant *EmitMethodList(const std::string &Name,
                                 const char *Section,
                                 const ConstantVector &Methods);
  /// EmitIvarList - Emit the ivar list for the given
  /// implementation. If ForClass is true the list of class ivars
  /// (i.e. metaclass ivars) is emitted, otherwise the list of
  /// interface ivars will be emitted. The return value has type
  /// IvarListnfABIPtrTy.
  llvm::Constant *EmitIvarList(const ObjCImplementationDecl *ID);
  
  llvm::Constant *EmitIvarOffsetVar(const ObjCInterfaceDecl *ID,
                                    const ObjCIvarDecl *Ivar,
                                    unsigned long int offset);
  
  /// GetOrEmitProtocol - Get the protocol object for the given
  /// declaration, emitting it if necessary. The return value has type
  /// ProtocolPtrTy.
  virtual llvm::Constant *GetOrEmitProtocol(const ObjCProtocolDecl *PD);
    
  /// GetOrEmitProtocolRef - Get a forward reference to the protocol
  /// object for the given declaration, emitting it if needed. These
  /// forward references will be filled in with empty bodies if no
  /// definition is seen. The return value has type ProtocolPtrTy.
  virtual llvm::Constant *GetOrEmitProtocolRef(const ObjCProtocolDecl *PD);
    
  /// EmitProtocolList - Generate the list of referenced
  /// protocols. The return value has type ProtocolListPtrTy.
  llvm::Constant *EmitProtocolList(const std::string &Name,
                                   ObjCProtocolDecl::protocol_iterator begin,
                                   ObjCProtocolDecl::protocol_iterator end);
  
  CodeGen::RValue EmitMessageSend(CodeGen::CodeGenFunction &CGF,
                                  QualType ResultType,
                                  Selector Sel,
                                  llvm::Value *Receiver,
                                  QualType Arg0Ty,
                                  bool IsSuper,
                                  const CallArgList &CallArgs);
  
  /// EmitClassRef - Return a Value*, of type ObjCTypes.ClassPtrTy,
  /// for the given class.
  llvm::Value *EmitClassRef(CGBuilderTy &Builder, 
                            const ObjCInterfaceDecl *ID,
                            bool IsSuper = false);
  
  /// EmitMetaClassRef - Return a Value * of the address of _class_t
  /// meta-data
  llvm::Value *EmitMetaClassRef(CGBuilderTy &Builder, 
                                const ObjCInterfaceDecl *ID);

  /// ObjCIvarOffsetVariable - Returns the ivar offset variable for
  /// the given ivar.
  ///
  llvm::GlobalVariable * ObjCIvarOffsetVariable(std::string &Name, 
                              const ObjCInterfaceDecl *ID,
                              const ObjCIvarDecl *Ivar);
  
  /// EmitSelector - Return a Value*, of type ObjCTypes.SelectorPtrTy,
  /// for the given selector.
  llvm::Value *EmitSelector(CGBuilderTy &Builder, Selector Sel);
  
public:
  CGObjCNonFragileABIMac(CodeGen::CodeGenModule &cgm);
  // FIXME. All stubs for now!
  virtual llvm::Function *ModuleInitFunction();
  
  virtual CodeGen::RValue GenerateMessageSend(CodeGen::CodeGenFunction &CGF,
                                              QualType ResultType,
                                              Selector Sel,
                                              llvm::Value *Receiver,
                                              bool IsClassMessage,
                                              const CallArgList &CallArgs);
  
  virtual CodeGen::RValue 
  GenerateMessageSendSuper(CodeGen::CodeGenFunction &CGF,
                           QualType ResultType,
                           Selector Sel,
                           const ObjCInterfaceDecl *Class,
                           llvm::Value *Receiver,
                           bool IsClassMessage,
                           const CallArgList &CallArgs);
  
  virtual llvm::Value *GetClass(CGBuilderTy &Builder,
                                const ObjCInterfaceDecl *ID);
  
  virtual llvm::Value *GetSelector(CGBuilderTy &Builder, Selector Sel)
    { return EmitSelector(Builder, Sel); }
  
  virtual void GenerateCategory(const ObjCCategoryImplDecl *CMD);
  
  virtual void GenerateClass(const ObjCImplementationDecl *ClassDecl);
  virtual llvm::Value *GenerateProtocolRef(CGBuilderTy &Builder,
                                           const ObjCProtocolDecl *PD);
  
  virtual llvm::Function *GetPropertyGetFunction(){ 
    return ObjCTypes.GetPropertyFn;
  }
  virtual llvm::Function *GetPropertySetFunction(){ 
    return ObjCTypes.SetPropertyFn; 
  }
  virtual llvm::Function *EnumerationMutationFunction()
    { return ObjCTypes.EnumerationMutationFn; }
  
  virtual void EmitTryOrSynchronizedStmt(CodeGen::CodeGenFunction &CGF,
                                         const Stmt &S)
    { return; }
  virtual void EmitThrowStmt(CodeGen::CodeGenFunction &CGF,
                             const ObjCAtThrowStmt &S)
    { return; }
  virtual llvm::Value * EmitObjCWeakRead(CodeGen::CodeGenFunction &CGF,
                                         llvm::Value *AddrWeakObj)
    { return 0; }
  virtual void EmitObjCWeakAssign(CodeGen::CodeGenFunction &CGF,
                                  llvm::Value *src, llvm::Value *dst)
    { return; } 
  virtual void EmitObjCGlobalAssign(CodeGen::CodeGenFunction &CGF,
                                    llvm::Value *src, llvm::Value *dest)
    { return; }
  virtual void EmitObjCIvarAssign(CodeGen::CodeGenFunction &CGF,
                                  llvm::Value *src, llvm::Value *dest)
  { return; }
  virtual void EmitObjCStrongCastAssign(CodeGen::CodeGenFunction &CGF,
                                        llvm::Value *src, llvm::Value *dest)
    { return; }
  virtual LValue EmitObjCValueForIvar(CodeGen::CodeGenFunction &CGF,
                                      QualType ObjectTy,
                                      llvm::Value *BaseValue,
                                      const ObjCIvarDecl *Ivar,
                                      const FieldDecl *Field,
                                      unsigned CVRQualifiers);
  virtual llvm::Value *EmitIvarOffset(CodeGen::CodeGenFunction &CGF,
                                      ObjCInterfaceDecl *Interface,
                                      const ObjCIvarDecl *Ivar);
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
 
CGObjCMac::CGObjCMac(CodeGen::CodeGenModule &cgm) : CGObjCCommonMac(cgm),
                                                    ObjCTypes(cgm)
{
  ObjCABI = 1;
  EmitImageInfo();  
}

/// GetClass - Return a reference to the class for the given interface
/// decl.
llvm::Value *CGObjCMac::GetClass(CGBuilderTy &Builder,
                                 const ObjCInterfaceDecl *ID) {
  return EmitClassRef(Builder, ID);
}

/// GetSelector - Return the pointer to the unique'd string for this selector.
llvm::Value *CGObjCMac::GetSelector(CGBuilderTy &Builder, Selector Sel) {
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

llvm::Constant *CGObjCCommonMac::GenerateConstantString(
                                                  const std::string &String) {
  return CGM.GetAddrOfConstantCFString(String);
}

/// Generates a message send where the super is the receiver.  This is
/// a message send to self with special delivery semantics indicating
/// which class's method should be called.
CodeGen::RValue
CGObjCMac::GenerateMessageSendSuper(CodeGen::CodeGenFunction &CGF,
                                    QualType ResultType,
                                    Selector Sel,
                                    const ObjCInterfaceDecl *Class,
                                    llvm::Value *Receiver,
                                    bool IsClassMessage,
                                    const CodeGen::CallArgList &CallArgs) {
  // Create and init a super structure; this is a (receiver, class)
  // pair we will pass to objc_msgSendSuper.
  llvm::Value *ObjCSuper = 
    CGF.Builder.CreateAlloca(ObjCTypes.SuperTy, 0, "objc_super");
  llvm::Value *ReceiverAsObject = 
    CGF.Builder.CreateBitCast(Receiver, ObjCTypes.ObjectPtrTy);
  CGF.Builder.CreateStore(ReceiverAsObject, 
                          CGF.Builder.CreateStructGEP(ObjCSuper, 0));

  // If this is a class message the metaclass is passed as the target.
  llvm::Value *Target;
  if (IsClassMessage) {
    llvm::Value *MetaClassPtr = EmitMetaClassRef(Class);
    llvm::Value *SuperPtr = CGF.Builder.CreateStructGEP(MetaClassPtr, 1);
    llvm::Value *Super = CGF.Builder.CreateLoad(SuperPtr);
    Target = Super;
  } else {
    Target = EmitClassRef(CGF.Builder, Class->getSuperClass());
  }
  // FIXME: We shouldn't need to do this cast, rectify the ASTContext
  // and ObjCTypes types.
  const llvm::Type *ClassTy = 
    CGM.getTypes().ConvertType(CGF.getContext().getObjCClassType());
  Target = CGF.Builder.CreateBitCast(Target, ClassTy);
  CGF.Builder.CreateStore(Target, 
                          CGF.Builder.CreateStructGEP(ObjCSuper, 1));
    
  return EmitMessageSend(CGF, ResultType, Sel, 
                         ObjCSuper, ObjCTypes.SuperPtrCTy,
                         true, CallArgs);
}
                                           
/// Generate code for a message send expression.  
CodeGen::RValue CGObjCMac::GenerateMessageSend(CodeGen::CodeGenFunction &CGF,
                                               QualType ResultType,
                                               Selector Sel,
                                               llvm::Value *Receiver,
                                               bool IsClassMessage,
                                               const CallArgList &CallArgs) {
  llvm::Value *Arg0 = 
    CGF.Builder.CreateBitCast(Receiver, ObjCTypes.ObjectPtrTy, "tmp");
  return EmitMessageSend(CGF, ResultType, Sel,
                         Arg0, CGF.getContext().getObjCIdType(),
                         false, CallArgs);
}

CodeGen::RValue CGObjCMac::EmitMessageSend(CodeGen::CodeGenFunction &CGF,
                                           QualType ResultType,
                                           Selector Sel,
                                           llvm::Value *Arg0,
                                           QualType Arg0Ty,
                                           bool IsSuper,
                                           const CallArgList &CallArgs) {
  CallArgList ActualArgs;
  ActualArgs.push_back(std::make_pair(RValue::get(Arg0), Arg0Ty));
  ActualArgs.push_back(std::make_pair(RValue::get(EmitSelector(CGF.Builder, 
                                                               Sel)),
                                      CGF.getContext().getObjCSelType()));
  ActualArgs.insert(ActualArgs.end(), CallArgs.begin(), CallArgs.end());

  CodeGenTypes &Types = CGM.getTypes();
  const CGFunctionInfo &FnInfo = Types.getFunctionInfo(ResultType, ActualArgs);
  const llvm::FunctionType *FTy = Types.GetFunctionType(FnInfo, false);

  llvm::Constant *Fn;
  if (CGM.ReturnTypeUsesSret(FnInfo)) {
    Fn = ObjCTypes.getSendStretFn(IsSuper);
  } else if (ResultType->isFloatingType()) {
    // FIXME: Sadly, this is wrong. This actually depends on the
    // architecture. This happens to be right for x86-32 though.
    Fn = ObjCTypes.getSendFpretFn(IsSuper);
  } else {
    Fn = ObjCTypes.getSendFn(IsSuper);
  }
  Fn = llvm::ConstantExpr::getBitCast(Fn, llvm::PointerType::getUnqual(FTy));
  return CGF.EmitCall(FnInfo, Fn, ActualArgs);
}

llvm::Value *CGObjCMac::GenerateProtocolRef(CGBuilderTy &Builder, 
                                            const ObjCProtocolDecl *PD) {
  // FIXME: I don't understand why gcc generates this, or where it is
  // resolved. Investigate. Its also wasteful to look this up over and
  // over.
  LazySymbols.insert(&CGM.getContext().Idents.get("Protocol"));

  return llvm::ConstantExpr::getBitCast(GetProtocolRef(PD),
                                        ObjCTypes.ExternalProtocolPtrTy);
}

void CGObjCCommonMac::GenerateProtocol(const ObjCProtocolDecl *PD) {
  // FIXME: We shouldn't need this, the protocol decl should contain
  // enough information to tell us whether this was a declaration or a
  // definition.
  DefinedProtocols.insert(PD->getIdentifier());

  // If we have generated a forward reference to this protocol, emit
  // it now. Otherwise do nothing, the protocol objects are lazily
  // emitted.
  if (Protocols.count(PD->getIdentifier())) 
    GetOrEmitProtocol(PD);
}

llvm::Constant *CGObjCCommonMac::GetProtocolRef(const ObjCProtocolDecl *PD) {
  if (DefinedProtocols.count(PD->getIdentifier()))
    return GetOrEmitProtocol(PD);
  return GetOrEmitProtocolRef(PD);
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
llvm::Constant *CGObjCMac::GetOrEmitProtocol(const ObjCProtocolDecl *PD) {
  llvm::GlobalVariable *&Entry = Protocols[PD->getIdentifier()];

  // Early exit if a defining object has already been generated.
  if (Entry && Entry->hasInitializer())
    return Entry;

  // FIXME: I don't understand why gcc generates this, or where it is
  // resolved. Investigate. Its also wasteful to look this up over and
  // over.
  LazySymbols.insert(&CGM.getContext().Idents.get("Protocol"));

  const char *ProtocolName = PD->getNameAsCString();

  // Construct method lists.
  std::vector<llvm::Constant*> InstanceMethods, ClassMethods;
  std::vector<llvm::Constant*> OptInstanceMethods, OptClassMethods;
  for (ObjCProtocolDecl::instmeth_iterator i = PD->instmeth_begin(),
         e = PD->instmeth_end(); i != e; ++i) {
    ObjCMethodDecl *MD = *i;
    llvm::Constant *C = GetMethodDescriptionConstant(MD);
    if (MD->getImplementationControl() == ObjCMethodDecl::Optional) {
      OptInstanceMethods.push_back(C);
    } else {
      InstanceMethods.push_back(C);
    }      
  }

  for (ObjCProtocolDecl::classmeth_iterator i = PD->classmeth_begin(),
         e = PD->classmeth_end(); i != e; ++i) {
    ObjCMethodDecl *MD = *i;
    llvm::Constant *C = GetMethodDescriptionConstant(MD);
    if (MD->getImplementationControl() == ObjCMethodDecl::Optional) {
      OptClassMethods.push_back(C);
    } else {
      ClassMethods.push_back(C);
    }      
  }

  std::vector<llvm::Constant*> Values(5);
  Values[0] = EmitProtocolExtension(PD, OptInstanceMethods, OptClassMethods);
  Values[1] = GetClassName(PD->getIdentifier());
  Values[2] = 
    EmitProtocolList("\01L_OBJC_PROTOCOL_REFS_" + PD->getNameAsString(),
                     PD->protocol_begin(),
                     PD->protocol_end());
  Values[3] = 
    EmitMethodDescList("\01L_OBJC_PROTOCOL_INSTANCE_METHODS_"
                          + PD->getNameAsString(),
                       "__OBJC,__cat_inst_meth,regular,no_dead_strip",
                       InstanceMethods);
  Values[4] = 
    EmitMethodDescList("\01L_OBJC_PROTOCOL_CLASS_METHODS_" 
                            + PD->getNameAsString(),
                       "__OBJC,__cat_cls_meth,regular,no_dead_strip",
                       ClassMethods);
  llvm::Constant *Init = llvm::ConstantStruct::get(ObjCTypes.ProtocolTy,
                                                   Values);
  
  if (Entry) {
    // Already created, fix the linkage and update the initializer.
    Entry->setLinkage(llvm::GlobalValue::InternalLinkage);
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

  return Entry;
}

llvm::Constant *CGObjCMac::GetOrEmitProtocolRef(const ObjCProtocolDecl *PD) {
  llvm::GlobalVariable *&Entry = Protocols[PD->getIdentifier()];

  if (!Entry) {
    // We use the initializer as a marker of whether this is a forward
    // reference or not. At module finalization we add the empty
    // contents for protocols which were referenced but never defined.
    Entry = 
      new llvm::GlobalVariable(ObjCTypes.ProtocolTy, false,
                               llvm::GlobalValue::ExternalLinkage,
                               0,
                               "\01L_OBJC_PROTOCOL_" + PD->getNameAsString(),
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
llvm::Constant *
CGObjCMac::EmitProtocolExtension(const ObjCProtocolDecl *PD,
                                 const ConstantVector &OptInstanceMethods,
                                 const ConstantVector &OptClassMethods) {
  uint64_t Size = 
    CGM.getTargetData().getTypePaddedSize(ObjCTypes.ProtocolExtensionTy);
  std::vector<llvm::Constant*> Values(4);
  Values[0] = llvm::ConstantInt::get(ObjCTypes.IntTy, Size);
  Values[1] = 
    EmitMethodDescList("\01L_OBJC_PROTOCOL_INSTANCE_METHODS_OPT_" 
                           + PD->getNameAsString(),
                       "__OBJC,__cat_inst_meth,regular,no_dead_strip",
                       OptInstanceMethods);
  Values[2] = 
    EmitMethodDescList("\01L_OBJC_PROTOCOL_CLASS_METHODS_OPT_"
                          + PD->getNameAsString(),
                       "__OBJC,__cat_cls_meth,regular,no_dead_strip",
                       OptClassMethods);
  Values[3] = EmitPropertyList("\01L_OBJC_$_PROP_PROTO_LIST_" + 
                                   PD->getNameAsString(),
                               0, PD, ObjCTypes);

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
                               "\01L_OBJC_PROTOCOLEXT_" + PD->getNameAsString(),
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
llvm::Constant *CGObjCCommonMac::EmitPropertyList(const std::string &Name,
                                      const Decl *Container,
                                      const ObjCContainerDecl *OCD,
                                      const ObjCCommonTypesHelper &ObjCTypes) {
  std::vector<llvm::Constant*> Properties, Prop(2);
  for (ObjCContainerDecl::prop_iterator I = OCD->prop_begin(), 
       E = OCD->prop_end(); I != E; ++I) {
    const ObjCPropertyDecl *PD = *I;
    Prop[0] = GetPropertyName(PD->getIdentifier());
    Prop[1] = GetPropertyTypeString(PD, Container);
    Properties.push_back(llvm::ConstantStruct::get(ObjCTypes.PropertyTy,
                                                   Prop));
  }

  // Return null for empty list.
  if (Properties.empty())
    return llvm::Constant::getNullValue(ObjCTypes.PropertyListPtrTy);

  unsigned PropertySize = 
    CGM.getTargetData().getTypePaddedSize(ObjCTypes.PropertyTy);
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
  if (ObjCABI == 2)
    GV->setSection("__DATA, __objc_const");
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
llvm::Constant *
CGObjCMac::GetMethodDescriptionConstant(const ObjCMethodDecl *MD) {
  std::vector<llvm::Constant*> Desc(2);
  Desc[0] = llvm::ConstantExpr::getBitCast(GetMethodVarName(MD->getSelector()),
                                           ObjCTypes.SelectorPtrTy);
  Desc[1] = GetMethodVarType(MD);
  return llvm::ConstantStruct::get(ObjCTypes.MethodDescriptionTy,
                                   Desc);
}

llvm::Constant *CGObjCMac::EmitMethodDescList(const std::string &Name,
                                              const char *Section,
                                              const ConstantVector &Methods) {
  // Return null for empty list.
  if (Methods.empty())
    return llvm::Constant::getNullValue(ObjCTypes.MethodDescriptionListPtrTy);

  std::vector<llvm::Constant*> Values(2);
  Values[0] = llvm::ConstantInt::get(ObjCTypes.IntTy, Methods.size());
  llvm::ArrayType *AT = llvm::ArrayType::get(ObjCTypes.MethodDescriptionTy, 
                                             Methods.size());
  Values[1] = llvm::ConstantArray::get(AT, Methods);
  llvm::Constant *Init = llvm::ConstantStruct::get(Values);

  llvm::GlobalVariable *GV = 
    new llvm::GlobalVariable(Init->getType(), false,
                             llvm::GlobalValue::InternalLinkage,
                             Init, Name, &CGM.getModule());
  GV->setSection(Section);
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
  unsigned Size = CGM.getTargetData().getTypePaddedSize(ObjCTypes.CategoryTy);

  // FIXME: This is poor design, the OCD should have a pointer to the
  // category decl. Additionally, note that Category can be null for
  // the @implementation w/o an @interface case. Sema should just
  // create one for us as it does for @implementation so everyone else
  // can live life under a clear blue sky.
  const ObjCInterfaceDecl *Interface = OCD->getClassInterface();
  const ObjCCategoryDecl *Category = 
    Interface->FindCategoryDeclaration(OCD->getIdentifier());
  std::string ExtName(Interface->getNameAsString() + "_" +
                      OCD->getNameAsString());

  std::vector<llvm::Constant*> InstanceMethods, ClassMethods;
  for (ObjCCategoryImplDecl::instmeth_iterator i = OCD->instmeth_begin(),
         e = OCD->instmeth_end(); i != e; ++i) {
    // Instance methods should always be defined.
    InstanceMethods.push_back(GetMethodConstant(*i));
  }
  for (ObjCCategoryImplDecl::classmeth_iterator i = OCD->classmeth_begin(),
         e = OCD->classmeth_end(); i != e; ++i) {
    // Class methods should always be defined.
    ClassMethods.push_back(GetMethodConstant(*i));
  }

  std::vector<llvm::Constant*> Values(7);
  Values[0] = GetClassName(OCD->getIdentifier());
  Values[1] = GetClassName(Interface->getIdentifier());
  Values[2] = 
    EmitMethodList(std::string("\01L_OBJC_CATEGORY_INSTANCE_METHODS_") + 
                   ExtName,
                   "__OBJC,__cat_inst_meth,regular,no_dead_strip",
                   InstanceMethods);
  Values[3] = 
    EmitMethodList(std::string("\01L_OBJC_CATEGORY_CLASS_METHODS_") + ExtName,
                   "__OBJC,__cat_class_meth,regular,no_dead_strip",
                   ClassMethods);
  if (Category) {
    Values[4] = 
      EmitProtocolList(std::string("\01L_OBJC_CATEGORY_PROTOCOLS_") + ExtName,
                       Category->protocol_begin(),
                       Category->protocol_end());
  } else {
    Values[4] = llvm::Constant::getNullValue(ObjCTypes.ProtocolListPtrTy);
  }
  Values[5] = llvm::ConstantInt::get(ObjCTypes.IntTy, Size);

  // If there is no category @interface then there can be no properties.
  if (Category) {
    Values[6] = EmitPropertyList(std::string("\01L_OBJC_$_PROP_LIST_") + ExtName,
                                 OCD, Category, ObjCTypes);
  } else {
    Values[6] = llvm::Constant::getNullValue(ObjCTypes.PropertyListPtrTy);
  }
  
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
  DefinedSymbols.insert(ID->getIdentifier());

  std::string ClassName = ID->getNameAsString();
  // FIXME: Gross
  ObjCInterfaceDecl *Interface = 
    const_cast<ObjCInterfaceDecl*>(ID->getClassInterface());
  llvm::Constant *Protocols = 
    EmitProtocolList("\01L_OBJC_CLASS_PROTOCOLS_" + ID->getNameAsString(),
                     Interface->protocol_begin(),
                     Interface->protocol_end());
  const llvm::Type *InterfaceTy = 
   CGM.getTypes().ConvertType(CGM.getContext().buildObjCInterfaceType(Interface));
  unsigned Flags = eClassFlags_Factory;
  unsigned Size = CGM.getTargetData().getTypePaddedSize(InterfaceTy);

  // FIXME: Set CXX-structors flag.
  if (IsClassHidden(ID->getClassInterface()))
    Flags |= eClassFlags_Hidden;

  std::vector<llvm::Constant*> InstanceMethods, ClassMethods;
  for (ObjCImplementationDecl::instmeth_iterator i = ID->instmeth_begin(),
         e = ID->instmeth_end(); i != e; ++i) {
    // Instance methods should always be defined.
    InstanceMethods.push_back(GetMethodConstant(*i));
  }
  for (ObjCImplementationDecl::classmeth_iterator i = ID->classmeth_begin(),
         e = ID->classmeth_end(); i != e; ++i) {
    // Class methods should always be defined.
    ClassMethods.push_back(GetMethodConstant(*i));
  }

  for (ObjCImplementationDecl::propimpl_iterator i = ID->propimpl_begin(),
         e = ID->propimpl_end(); i != e; ++i) {
    ObjCPropertyImplDecl *PID = *i;

    if (PID->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize) {
      ObjCPropertyDecl *PD = PID->getPropertyDecl();

      if (ObjCMethodDecl *MD = PD->getGetterMethodDecl())
        if (llvm::Constant *C = GetMethodConstant(MD))
          InstanceMethods.push_back(C);
      if (ObjCMethodDecl *MD = PD->getSetterMethodDecl())
        if (llvm::Constant *C = GetMethodConstant(MD))
          InstanceMethods.push_back(C);
    }
  }

  std::vector<llvm::Constant*> Values(12);
  Values[ 0] = EmitMetaClass(ID, Protocols, InterfaceTy, ClassMethods);
  if (ObjCInterfaceDecl *Super = Interface->getSuperClass()) {
    // Record a reference to the super class.
    LazySymbols.insert(Super->getIdentifier());

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
  Values[ 6] = EmitIvarList(ID, false);
  Values[ 7] = 
    EmitMethodList("\01L_OBJC_INSTANCE_METHODS_" + ID->getNameAsString(),
                   "__OBJC,__inst_meth,regular,no_dead_strip",
                   InstanceMethods);
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
                                         const llvm::Type *InterfaceTy,
                                         const ConstantVector &Methods) {
  unsigned Flags = eClassFlags_Meta;
  unsigned Size = CGM.getTargetData().getTypePaddedSize(ObjCTypes.ClassTy);

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
  Values[ 6] = EmitIvarList(ID, true);
  Values[ 7] = 
    EmitMethodList("\01L_OBJC_CLASS_METHODS_" + ID->getNameAsString(),
                   "__OBJC,__inst_meth,regular,no_dead_strip",
                   Methods);
  // cache is always NULL.
  Values[ 8] = llvm::Constant::getNullValue(ObjCTypes.CachePtrTy);
  Values[ 9] = Protocols;
  // ivar_layout for metaclass is always NULL.
  Values[10] = llvm::Constant::getNullValue(ObjCTypes.Int8PtrTy);
  // The class extension is always unused for metaclasses.
  Values[11] = llvm::Constant::getNullValue(ObjCTypes.ClassExtensionPtrTy);
  llvm::Constant *Init = llvm::ConstantStruct::get(ObjCTypes.ClassTy,
                                                   Values);

  std::string Name("\01L_OBJC_METACLASS_");
  Name += ID->getNameAsCString();

  // Check for a forward reference.
  llvm::GlobalVariable *GV = CGM.getModule().getGlobalVariable(Name);
  if (GV) {
    assert(GV->getType()->getElementType() == ObjCTypes.ClassTy &&
           "Forward metaclass reference has incorrect type.");
    GV->setLinkage(llvm::GlobalValue::InternalLinkage);
    GV->setInitializer(Init);
  } else {
    GV = new llvm::GlobalVariable(ObjCTypes.ClassTy, false,
                                  llvm::GlobalValue::InternalLinkage,
                                  Init, Name,
                                  &CGM.getModule());
  }
  GV->setSection("__OBJC,__meta_class,regular,no_dead_strip");
  UsedGlobals.push_back(GV);
  // FIXME: Why?
  GV->setAlignment(32);

  return GV;
}

llvm::Constant *CGObjCMac::EmitMetaClassRef(const ObjCInterfaceDecl *ID) {  
  std::string Name = "\01L_OBJC_METACLASS_" + ID->getNameAsString();

  // FIXME: Should we look these up somewhere other than the
  // module. Its a bit silly since we only generate these while
  // processing an implementation, so exactly one pointer would work
  // if know when we entered/exitted an implementation block.

  // Check for an existing forward reference.
  // Previously, metaclass with internal linkage may have been defined.
  // pass 'true' as 2nd argument so it is returned.
  if (llvm::GlobalVariable *GV = CGM.getModule().getGlobalVariable(Name, true)) {
    assert(GV->getType()->getElementType() == ObjCTypes.ClassTy &&
           "Forward metaclass reference has incorrect type.");
    return GV;
  } else {
    // Generate as an external reference to keep a consistent
    // module. This will be patched up when we emit the metaclass.
    return new llvm::GlobalVariable(ObjCTypes.ClassTy, false,
                                    llvm::GlobalValue::ExternalLinkage,
                                    0,
                                    Name,
                                    &CGM.getModule());
  }
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
    CGM.getTargetData().getTypePaddedSize(ObjCTypes.ClassExtensionTy);

  std::vector<llvm::Constant*> Values(3);
  Values[0] = llvm::ConstantInt::get(ObjCTypes.IntTy, Size);
  // FIXME: Output weak_ivar_layout string.
  Values[1] = llvm::Constant::getNullValue(ObjCTypes.Int8PtrTy);
  Values[2] = EmitPropertyList("\01L_OBJC_$_PROP_LIST_" + ID->getNameAsString(),
                               ID, ID->getClassInterface(), ObjCTypes);

  // Return null if no extension bits are used.
  if (Values[1]->isNullValue() && Values[2]->isNullValue())
    return llvm::Constant::getNullValue(ObjCTypes.ClassExtensionPtrTy);

  llvm::Constant *Init = 
    llvm::ConstantStruct::get(ObjCTypes.ClassExtensionTy, Values);
  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(ObjCTypes.ClassExtensionTy, false,
                             llvm::GlobalValue::InternalLinkage,
                             Init,
                             "\01L_OBJC_CLASSEXT_" + ID->getNameAsString(),
                             &CGM.getModule());
  // No special section, but goes in llvm.used
  UsedGlobals.push_back(GV);
  
  return GV;
}

/// countInheritedIvars - count number of ivars in class and its super class(s)
///
static int countInheritedIvars(const ObjCInterfaceDecl *OI) {
  int count = 0;
  if (!OI)
    return 0;
  const ObjCInterfaceDecl *SuperClass = OI->getSuperClass();
  if (SuperClass)
    count += countInheritedIvars(SuperClass);
  for (ObjCInterfaceDecl::ivar_iterator I = OI->ivar_begin(),
       E = OI->ivar_end(); I != E; ++I)
    ++count;
  return count;
}

/// getInterfaceDeclForIvar - Get the interface declaration node where
/// this ivar is declared in.
/// FIXME. Ideally, this info should be in the ivar node. But currently 
/// it is not and prevailing wisdom is that ASTs should not have more
/// info than is absolutely needed, even though this info reflects the
/// source language. 
///
static const ObjCInterfaceDecl *getInterfaceDeclForIvar(
                                  const ObjCInterfaceDecl *OI,
                                  const ObjCIvarDecl *IVD) {
  if (!OI)
    return 0;
  assert(isa<ObjCInterfaceDecl>(OI) && "OI is not an interface");
  for (ObjCInterfaceDecl::ivar_iterator I = OI->ivar_begin(),
       E = OI->ivar_end(); I != E; ++I)
    if ((*I)->getIdentifier() == IVD->getIdentifier())
      return OI;
  return getInterfaceDeclForIvar(OI->getSuperClass(), IVD);
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
                                        bool ForClass) {
  std::vector<llvm::Constant*> Ivars, Ivar(3);

  // When emitting the root class GCC emits ivar entries for the
  // actual class structure. It is not clear if we need to follow this
  // behavior; for now lets try and get away with not doing it. If so,
  // the cleanest solution would be to make up an ObjCInterfaceDecl
  // for the class.
  if (ForClass)
    return llvm::Constant::getNullValue(ObjCTypes.IvarListPtrTy);
  
  ObjCInterfaceDecl *OID = 
    const_cast<ObjCInterfaceDecl*>(ID->getClassInterface());
  const llvm::Type *InterfaceTy = 
    CGM.getTypes().ConvertType(CGM.getContext().getObjCInterfaceType(OID));
  const llvm::StructLayout *Layout =
    CGM.getTargetData().getStructLayout(cast<llvm::StructType>(InterfaceTy));
  
  RecordDecl::field_iterator ifield, pfield;
  const RecordDecl *RD = GetFirstIvarInRecord(OID, ifield, pfield);
  for (RecordDecl::field_iterator e = RD->field_end(); ifield != e; ++ifield) {
    FieldDecl *Field = *ifield;
    unsigned Offset = Layout->getElementOffset(CGM.getTypes().
                                               getLLVMFieldNo(Field));
    if (Field->getIdentifier())
      Ivar[0] = GetMethodVarName(Field->getIdentifier());
    else
      Ivar[0] = llvm::Constant::getNullValue(ObjCTypes.Int8PtrTy);
    std::string TypeStr;
    CGM.getContext().getObjCEncodingForType(Field->getType(), TypeStr, Field);
    Ivar[1] = GetMethodVarType(TypeStr);
    Ivar[2] = llvm::ConstantInt::get(ObjCTypes.IntTy, Offset);
    Ivars.push_back(llvm::ConstantStruct::get(ObjCTypes.IvarTy, Ivar));
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
                             Prefix + ID->getNameAsString(),
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

/// GetMethodConstant - Return a struct objc_method constant for the
/// given method if it has been defined. The result is null if the
/// method has not been defined. The return value has type MethodPtrTy.
llvm::Constant *CGObjCMac::GetMethodConstant(const ObjCMethodDecl *MD) {
  // FIXME: Use DenseMap::lookup
  llvm::Function *Fn = MethodDefinitions[MD];
  if (!Fn)
    return 0;
  
  std::vector<llvm::Constant*> Method(3);
  Method[0] = 
    llvm::ConstantExpr::getBitCast(GetMethodVarName(MD->getSelector()),
                                   ObjCTypes.SelectorPtrTy);
  Method[1] = GetMethodVarType(MD);
  Method[2] = llvm::ConstantExpr::getBitCast(Fn, ObjCTypes.Int8PtrTy);
  return llvm::ConstantStruct::get(ObjCTypes.MethodTy, Method);
}

llvm::Constant *CGObjCMac::EmitMethodList(const std::string &Name,
                                          const char *Section,
                                          const ConstantVector &Methods) {
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

llvm::Function *CGObjCCommonMac::GenerateMethod(const ObjCMethodDecl *OMD,
                                                const ObjCContainerDecl *CD) { 
  std::string Name;
  GetNameForMethod(OMD, CD, Name);

  CodeGenTypes &Types = CGM.getTypes();
  const llvm::FunctionType *MethodTy =
    Types.GetFunctionType(Types.getFunctionInfo(OMD), OMD->isVariadic());
  llvm::Function *Method = 
    llvm::Function::Create(MethodTy,
                           llvm::GlobalValue::InternalLinkage,
                           Name,
                           &CGM.getModule());
  MethodDefinitions.insert(std::make_pair(OMD, Method));

  return Method;
}

llvm::Function *CGObjCMac::ModuleInitFunction() { 
  // Abuse this interface function as a place to finalize.
  FinishModule();

  return NULL;
}

llvm::Function *CGObjCMac::GetPropertyGetFunction() {
  return ObjCTypes.GetPropertyFn;
}

llvm::Function *CGObjCMac::GetPropertySetFunction() {
  return ObjCTypes.SetPropertyFn;
}

llvm::Function *CGObjCMac::EnumerationMutationFunction()
{
  return ObjCTypes.EnumerationMutationFn;
}

/* 

Objective-C setjmp-longjmp (sjlj) Exception Handling
--

The basic framework for a @try-catch-finally is as follows:
{
  objc_exception_data d;
  id _rethrow = null;
  bool _call_try_exit = true;
 
  objc_exception_try_enter(&d);
  if (!setjmp(d.jmp_buf)) {
    ... try body ... 
  } else {
    // exception path
    id _caught = objc_exception_extract(&d);
    
    // enter new try scope for handlers
    if (!setjmp(d.jmp_buf)) {
      ... match exception and execute catch blocks ...
      
      // fell off end, rethrow.
      _rethrow = _caught;
      ... jump-through-finally to finally_rethrow ...
    } else {
      // exception in catch block
      _rethrow = objc_exception_extract(&d);
      _call_try_exit = false;
      ... jump-through-finally to finally_rethrow ...
    }
  }
  ... jump-through-finally to finally_end ...

finally:
  if (_call_try_exit)
    objc_exception_try_exit(&d);

  ... finally block ....
  ... dispatch to finally destination ...

finally_rethrow:
  objc_exception_throw(_rethrow);

finally_end:
}

This framework differs slightly from the one gcc uses, in that gcc
uses _rethrow to determine if objc_exception_try_exit should be called
and if the object should be rethrown. This breaks in the face of
throwing nil and introduces unnecessary branches.

We specialize this framework for a few particular circumstances:

 - If there are no catch blocks, then we avoid emitting the second
   exception handling context.

 - If there is a catch-all catch block (i.e. @catch(...) or @catch(id
   e)) we avoid emitting the code to rethrow an uncaught exception.

 - FIXME: If there is no @finally block we can do a few more
   simplifications.

Rethrows and Jumps-Through-Finally
--

Support for implicit rethrows and jumping through the finally block is
handled by storing the current exception-handling context in
ObjCEHStack.

In order to implement proper @finally semantics, we support one basic
mechanism for jumping through the finally block to an arbitrary
destination. Constructs which generate exits from a @try or @catch
block use this mechanism to implement the proper semantics by chaining
jumps, as necessary.

This mechanism works like the one used for indirect goto: we
arbitrarily assign an ID to each destination and store the ID for the
destination in a variable prior to entering the finally block. At the
end of the finally block we simply create a switch to the proper
destination.
 
Code gen for @synchronized(expr) stmt;
Effectively generating code for:
objc_sync_enter(expr);
@try stmt @finally { objc_sync_exit(expr); }
*/

void CGObjCMac::EmitTryOrSynchronizedStmt(CodeGen::CodeGenFunction &CGF,
                                          const Stmt &S) {
  bool isTry = isa<ObjCAtTryStmt>(S);
  // Create various blocks we refer to for handling @finally.
  llvm::BasicBlock *FinallyBlock = CGF.createBasicBlock("finally");
  llvm::BasicBlock *FinallyExit = CGF.createBasicBlock("finally.exit");
  llvm::BasicBlock *FinallyNoExit = CGF.createBasicBlock("finally.noexit");
  llvm::BasicBlock *FinallyRethrow = CGF.createBasicBlock("finally.throw");
  llvm::BasicBlock *FinallyEnd = CGF.createBasicBlock("finally.end");

  // Push an EH context entry, used for handling rethrows and jumps
  // through finally.
  CGF.PushCleanupBlock(FinallyBlock);

  CGF.ObjCEHValueStack.push_back(0);
  
  // Allocate memory for the exception data and rethrow pointer.
  llvm::Value *ExceptionData = CGF.CreateTempAlloca(ObjCTypes.ExceptionDataTy,
                                                    "exceptiondata.ptr");
  llvm::Value *RethrowPtr = CGF.CreateTempAlloca(ObjCTypes.ObjectPtrTy, 
                                                 "_rethrow");
  llvm::Value *CallTryExitPtr = CGF.CreateTempAlloca(llvm::Type::Int1Ty,
                                                     "_call_try_exit");
  CGF.Builder.CreateStore(llvm::ConstantInt::getTrue(), CallTryExitPtr);
  
  if (!isTry) {
    // For @synchronized, call objc_sync_enter(sync.expr)
    llvm::Value *Arg = CGF.EmitScalarExpr(
                         cast<ObjCAtSynchronizedStmt>(S).getSynchExpr());
    Arg = CGF.Builder.CreateBitCast(Arg, ObjCTypes.ObjectPtrTy);
    CGF.Builder.CreateCall(ObjCTypes.SyncEnterFn, Arg);
  }
  
  // Enter a new try block and call setjmp.
  CGF.Builder.CreateCall(ObjCTypes.ExceptionTryEnterFn, ExceptionData);
  llvm::Value *JmpBufPtr = CGF.Builder.CreateStructGEP(ExceptionData, 0, 
                                                       "jmpbufarray");
  JmpBufPtr = CGF.Builder.CreateStructGEP(JmpBufPtr, 0, "tmp");
  llvm::Value *SetJmpResult = CGF.Builder.CreateCall(ObjCTypes.SetJmpFn,
                                                     JmpBufPtr, "result");

  llvm::BasicBlock *TryBlock = CGF.createBasicBlock("try");
  llvm::BasicBlock *TryHandler = CGF.createBasicBlock("try.handler");
  CGF.Builder.CreateCondBr(CGF.Builder.CreateIsNotNull(SetJmpResult, "threw"), 
                           TryHandler, TryBlock);

  // Emit the @try block.
  CGF.EmitBlock(TryBlock);
  CGF.EmitStmt(isTry ? cast<ObjCAtTryStmt>(S).getTryBody() 
                     : cast<ObjCAtSynchronizedStmt>(S).getSynchBody());
  CGF.EmitBranchThroughCleanup(FinallyEnd);
  
  // Emit the "exception in @try" block.
  CGF.EmitBlock(TryHandler);

  // Retrieve the exception object.  We may emit multiple blocks but
  // nothing can cross this so the value is already in SSA form.
  llvm::Value *Caught = CGF.Builder.CreateCall(ObjCTypes.ExceptionExtractFn,
                                               ExceptionData,
                                               "caught");
  CGF.ObjCEHValueStack.back() = Caught;
  if (!isTry)
  {
    CGF.Builder.CreateStore(Caught, RethrowPtr);
    CGF.Builder.CreateStore(llvm::ConstantInt::getFalse(), CallTryExitPtr);
    CGF.EmitBranchThroughCleanup(FinallyRethrow);
  }
  else if (const ObjCAtCatchStmt* CatchStmt = 
           cast<ObjCAtTryStmt>(S).getCatchStmts()) 
  {    
    // Enter a new exception try block (in case a @catch block throws
    // an exception).
    CGF.Builder.CreateCall(ObjCTypes.ExceptionTryEnterFn, ExceptionData);
        
    llvm::Value *SetJmpResult = CGF.Builder.CreateCall(ObjCTypes.SetJmpFn,
                                                       JmpBufPtr, "result");
    llvm::Value *Threw = CGF.Builder.CreateIsNotNull(SetJmpResult, "threw");

    llvm::BasicBlock *CatchBlock = CGF.createBasicBlock("catch");
    llvm::BasicBlock *CatchHandler = CGF.createBasicBlock("catch.handler");
    CGF.Builder.CreateCondBr(Threw, CatchHandler, CatchBlock);
    
    CGF.EmitBlock(CatchBlock);
        
    // Handle catch list. As a special case we check if everything is
    // matched and avoid generating code for falling off the end if
    // so.
    bool AllMatched = false;
    for (; CatchStmt; CatchStmt = CatchStmt->getNextCatchStmt()) {
      llvm::BasicBlock *NextCatchBlock = CGF.createBasicBlock("catch");

      const DeclStmt *CatchParam = 
        cast_or_null<DeclStmt>(CatchStmt->getCatchParamStmt());
      const VarDecl *VD = 0;
      const PointerType *PT = 0;

      // catch(...) always matches.
      if (!CatchParam) {
        AllMatched = true;
      } else {
        VD = cast<VarDecl>(CatchParam->getSolitaryDecl());
        PT = VD->getType()->getAsPointerType();
        
        // catch(id e) always matches. 
        // FIXME: For the time being we also match id<X>; this should
        // be rejected by Sema instead.
        if ((PT && CGF.getContext().isObjCIdStructType(PT->getPointeeType())) ||
            VD->getType()->isObjCQualifiedIdType())
          AllMatched = true;
      }
      
      if (AllMatched) {   
        if (CatchParam) {
          CGF.EmitStmt(CatchParam);
          assert(CGF.HaveInsertPoint() && "DeclStmt destroyed insert point?");
          CGF.Builder.CreateStore(Caught, CGF.GetAddrOfLocalVar(VD));
        }
        
        CGF.EmitStmt(CatchStmt->getCatchBody());
        CGF.EmitBranchThroughCleanup(FinallyEnd);
        break;
      }
      
      assert(PT && "Unexpected non-pointer type in @catch");
      QualType T = PT->getPointeeType();
      const ObjCInterfaceType *ObjCType = T->getAsObjCInterfaceType();
      assert(ObjCType && "Catch parameter must have Objective-C type!");

      // Check if the @catch block matches the exception object.
      llvm::Value *Class = EmitClassRef(CGF.Builder, ObjCType->getDecl());
      
      llvm::Value *Match = CGF.Builder.CreateCall2(ObjCTypes.ExceptionMatchFn,
                                                   Class, Caught, "match");
      
      llvm::BasicBlock *MatchedBlock = CGF.createBasicBlock("matched");
      
      CGF.Builder.CreateCondBr(CGF.Builder.CreateIsNotNull(Match, "matched"), 
                               MatchedBlock, NextCatchBlock);
      
      // Emit the @catch block.
      CGF.EmitBlock(MatchedBlock);
      CGF.EmitStmt(CatchParam);
      assert(CGF.HaveInsertPoint() && "DeclStmt destroyed insert point?");

      llvm::Value *Tmp = 
        CGF.Builder.CreateBitCast(Caught, CGF.ConvertType(VD->getType()), 
                                  "tmp");
      CGF.Builder.CreateStore(Tmp, CGF.GetAddrOfLocalVar(VD));
      
      CGF.EmitStmt(CatchStmt->getCatchBody());
      CGF.EmitBranchThroughCleanup(FinallyEnd);
      
      CGF.EmitBlock(NextCatchBlock);
    }

    if (!AllMatched) {
      // None of the handlers caught the exception, so store it to be
      // rethrown at the end of the @finally block.
      CGF.Builder.CreateStore(Caught, RethrowPtr);
      CGF.EmitBranchThroughCleanup(FinallyRethrow);
    }
    
    // Emit the exception handler for the @catch blocks.
    CGF.EmitBlock(CatchHandler);    
    CGF.Builder.CreateStore(CGF.Builder.CreateCall(ObjCTypes.ExceptionExtractFn,
                                                   ExceptionData), 
                            RethrowPtr);
    CGF.Builder.CreateStore(llvm::ConstantInt::getFalse(), CallTryExitPtr);
    CGF.EmitBranchThroughCleanup(FinallyRethrow);
  } else {
    CGF.Builder.CreateStore(Caught, RethrowPtr);
    CGF.Builder.CreateStore(llvm::ConstantInt::getFalse(), CallTryExitPtr);
    CGF.EmitBranchThroughCleanup(FinallyRethrow);
  }
  
  // Pop the exception-handling stack entry. It is important to do
  // this now, because the code in the @finally block is not in this
  // context.
  CodeGenFunction::CleanupBlockInfo Info = CGF.PopCleanupBlock();

  CGF.ObjCEHValueStack.pop_back();
  
  // Emit the @finally block.
  CGF.EmitBlock(FinallyBlock);
  llvm::Value* CallTryExit = CGF.Builder.CreateLoad(CallTryExitPtr, "tmp");
  
  CGF.Builder.CreateCondBr(CallTryExit, FinallyExit, FinallyNoExit);
  
  CGF.EmitBlock(FinallyExit);
  CGF.Builder.CreateCall(ObjCTypes.ExceptionTryExitFn, ExceptionData);

  CGF.EmitBlock(FinallyNoExit);
  if (isTry) {
    if (const ObjCAtFinallyStmt* FinallyStmt = 
          cast<ObjCAtTryStmt>(S).getFinallyStmt())
      CGF.EmitStmt(FinallyStmt->getFinallyBody());
  }
  else {
    // For @synchronized objc_sync_exit(expr); As finally's sole statement.
    // For @synchronized, call objc_sync_enter(sync.expr)
    llvm::Value *Arg = CGF.EmitScalarExpr(
                         cast<ObjCAtSynchronizedStmt>(S).getSynchExpr());
    Arg = CGF.Builder.CreateBitCast(Arg, ObjCTypes.ObjectPtrTy);
    CGF.Builder.CreateCall(ObjCTypes.SyncExitFn, Arg);
  }

  // Emit the switch block
  if (Info.SwitchBlock)
    CGF.EmitBlock(Info.SwitchBlock);
  if (Info.EndBlock)
    CGF.EmitBlock(Info.EndBlock);
  
  CGF.EmitBlock(FinallyRethrow);
  CGF.Builder.CreateCall(ObjCTypes.ExceptionThrowFn, 
                         CGF.Builder.CreateLoad(RethrowPtr));
  CGF.Builder.CreateUnreachable();
  
  CGF.EmitBlock(FinallyEnd);
}

void CGObjCMac::EmitThrowStmt(CodeGen::CodeGenFunction &CGF,
                              const ObjCAtThrowStmt &S) {
  llvm::Value *ExceptionAsObject;
  
  if (const Expr *ThrowExpr = S.getThrowExpr()) {
    llvm::Value *Exception = CGF.EmitScalarExpr(ThrowExpr);
    ExceptionAsObject = 
      CGF.Builder.CreateBitCast(Exception, ObjCTypes.ObjectPtrTy, "tmp");
  } else {
    assert((!CGF.ObjCEHValueStack.empty() && CGF.ObjCEHValueStack.back()) && 
           "Unexpected rethrow outside @catch block.");
    ExceptionAsObject = CGF.ObjCEHValueStack.back();
  }
  
  CGF.Builder.CreateCall(ObjCTypes.ExceptionThrowFn, ExceptionAsObject);
  CGF.Builder.CreateUnreachable();

  // Clear the insertion point to indicate we are in unreachable code.
  CGF.Builder.ClearInsertionPoint();
}

/// EmitObjCWeakRead - Code gen for loading value of a __weak
/// object: objc_read_weak (id *src)
///
llvm::Value * CGObjCMac::EmitObjCWeakRead(CodeGen::CodeGenFunction &CGF,
                                          llvm::Value *AddrWeakObj)
{
  AddrWeakObj = CGF.Builder.CreateBitCast(AddrWeakObj, ObjCTypes.PtrObjectPtrTy); 
  llvm::Value *read_weak = CGF.Builder.CreateCall(ObjCTypes.GcReadWeakFn,
                                                  AddrWeakObj, "weakread");
  return read_weak;
}

/// EmitObjCWeakAssign - Code gen for assigning to a __weak object.
/// objc_assign_weak (id src, id *dst)
///
void CGObjCMac::EmitObjCWeakAssign(CodeGen::CodeGenFunction &CGF,
                                   llvm::Value *src, llvm::Value *dst)
{
  src = CGF.Builder.CreateBitCast(src, ObjCTypes.ObjectPtrTy);
  dst = CGF.Builder.CreateBitCast(dst, ObjCTypes.PtrObjectPtrTy);
  CGF.Builder.CreateCall2(ObjCTypes.GcAssignWeakFn,
                          src, dst, "weakassign");
  return;
}

/// EmitObjCGlobalAssign - Code gen for assigning to a __strong object.
/// objc_assign_global (id src, id *dst)
///
void CGObjCMac::EmitObjCGlobalAssign(CodeGen::CodeGenFunction &CGF,
                                     llvm::Value *src, llvm::Value *dst)
{
  src = CGF.Builder.CreateBitCast(src, ObjCTypes.ObjectPtrTy);
  dst = CGF.Builder.CreateBitCast(dst, ObjCTypes.PtrObjectPtrTy);
  CGF.Builder.CreateCall2(ObjCTypes.GcAssignGlobalFn,
                          src, dst, "globalassign");
  return;
}

/// EmitObjCIvarAssign - Code gen for assigning to a __strong object.
/// objc_assign_ivar (id src, id *dst)
///
void CGObjCMac::EmitObjCIvarAssign(CodeGen::CodeGenFunction &CGF,
                                   llvm::Value *src, llvm::Value *dst)
{
  src = CGF.Builder.CreateBitCast(src, ObjCTypes.ObjectPtrTy);
  dst = CGF.Builder.CreateBitCast(dst, ObjCTypes.PtrObjectPtrTy);
  CGF.Builder.CreateCall2(ObjCTypes.GcAssignIvarFn,
                          src, dst, "assignivar");
  return;
}

/// EmitObjCStrongCastAssign - Code gen for assigning to a __strong cast object.
/// objc_assign_strongCast (id src, id *dst)
///
void CGObjCMac::EmitObjCStrongCastAssign(CodeGen::CodeGenFunction &CGF,
                                         llvm::Value *src, llvm::Value *dst)
{
  src = CGF.Builder.CreateBitCast(src, ObjCTypes.ObjectPtrTy);
  dst = CGF.Builder.CreateBitCast(dst, ObjCTypes.PtrObjectPtrTy);
  CGF.Builder.CreateCall2(ObjCTypes.GcAssignStrongCastFn,
                          src, dst, "weakassign");
  return;
}

/// EmitObjCValueForIvar - Code Gen for ivar reference.
///
LValue CGObjCMac::EmitObjCValueForIvar(CodeGen::CodeGenFunction &CGF,
                                       QualType ObjectTy,
                                       llvm::Value *BaseValue,
                                       const ObjCIvarDecl *Ivar,
                                       const FieldDecl *Field,
                                       unsigned CVRQualifiers) {
  if (Ivar->isBitField())
    return CGF.EmitLValueForBitfield(BaseValue, const_cast<FieldDecl *>(Field),
                                     CVRQualifiers);
  // TODO:  Add a special case for isa (index 0)
  unsigned Index = CGM.getTypes().getLLVMFieldNo(Field);
  llvm::Value *V = CGF.Builder.CreateStructGEP(BaseValue, Index, "tmp");
  LValue LV = LValue::MakeAddr(V, 
               Ivar->getType().getCVRQualifiers()|CVRQualifiers);
  LValue::SetObjCIvar(LV, true);
  return LV;
}

llvm::Value *CGObjCMac::EmitIvarOffset(CodeGen::CodeGenFunction &CGF,
                                       ObjCInterfaceDecl *Interface,
                                       const ObjCIvarDecl *Ivar) {
  const llvm::Type *InterfaceLTy =
  CGM.getTypes().ConvertType(CGM.getContext().getObjCInterfaceType(Interface));
  const llvm::StructLayout *Layout =
  CGM.getTargetData().getStructLayout(cast<llvm::StructType>(InterfaceLTy));
  FieldDecl *Field = Interface->lookupFieldDeclForIvar(CGM.getContext(), Ivar);
  uint64_t Offset =
  Layout->getElementOffset(CGM.getTypes().getLLVMFieldNo(Field));
  
  return llvm::ConstantInt::get(
                            CGM.getTypes().ConvertType(CGM.getContext().LongTy),
                            Offset);
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
  uint64_t Size = CGM.getTargetData().getTypePaddedSize(ObjCTypes.ModuleTy);
  
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
  unsigned NumClasses = DefinedClasses.size();
  unsigned NumCategories = DefinedCategories.size();

  // Return null if no symbols were defined.
  if (!NumClasses && !NumCategories)
    return llvm::Constant::getNullValue(ObjCTypes.SymtabPtrTy);

  std::vector<llvm::Constant*> Values(5);
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

llvm::Value *CGObjCMac::EmitClassRef(CGBuilderTy &Builder, 
                                     const ObjCInterfaceDecl *ID) {
  LazySymbols.insert(ID->getIdentifier());

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

llvm::Value *CGObjCMac::EmitSelector(CGBuilderTy &Builder, Selector Sel) {
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

llvm::Constant *CGObjCCommonMac::GetClassName(IdentifierInfo *Ident) {
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

llvm::Constant *CGObjCCommonMac::GetMethodVarName(Selector Sel) {
  llvm::GlobalVariable *&Entry = MethodVarNames[Sel];

  if (!Entry) {
    // FIXME: Avoid std::string copying.
    llvm::Constant *C = llvm::ConstantArray::get(Sel.getAsString());
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
llvm::Constant *CGObjCCommonMac::GetMethodVarName(IdentifierInfo *ID) {
  return GetMethodVarName(CGM.getContext().Selectors.getNullarySelector(ID));
}

// FIXME: Merge into a single cstring creation function.
llvm::Constant *CGObjCCommonMac::GetMethodVarName(const std::string &Name) {
  return GetMethodVarName(&CGM.getContext().Idents.get(Name));
}

llvm::Constant *CGObjCCommonMac::GetMethodVarType(const std::string &Name) {
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
llvm::Constant *CGObjCCommonMac::GetMethodVarType(const ObjCMethodDecl *D) {
  std::string TypeStr;
  CGM.getContext().getObjCEncodingForMethodDecl(const_cast<ObjCMethodDecl*>(D),
                                                TypeStr);
  return GetMethodVarType(TypeStr);
}

// FIXME: Merge into a single cstring creation function.
llvm::Constant *CGObjCCommonMac::GetPropertyName(IdentifierInfo *Ident) {
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
// FIXME: This Decl should be more precise.
llvm::Constant *CGObjCCommonMac::GetPropertyTypeString(const ObjCPropertyDecl *PD,
                                                 const Decl *Container) {
  std::string TypeStr;
  CGM.getContext().getObjCEncodingForPropertyDecl(PD, Container, TypeStr);
  return GetPropertyName(&CGM.getContext().Idents.get(TypeStr));
}

void CGObjCCommonMac::GetNameForMethod(const ObjCMethodDecl *D, 
                                       const ObjCContainerDecl *CD,
                                       std::string &NameOut) {
  // FIXME: Find the mangling GCC uses.
  NameOut = (D->isInstanceMethod() ? "-" : "+");
  NameOut += '[';
  assert (CD && "Missing container decl in GetNameForMethod");
  NameOut += CD->getNameAsString();
  // FIXME. For a method in a category, (CAT_NAME) is inserted here.
  // Right now! there is not enough info. to do this.
  NameOut += ' ';
  NameOut += D->getSelector().getAsString();
  NameOut += ']';
}

/// GetFirstIvarInRecord - This routine returns the record for the 
/// implementation of the fiven class OID. It also returns field
/// corresponding to the first ivar in the class in FIV. It also
/// returns the one before the first ivar. 
///
const RecordDecl *CGObjCCommonMac::GetFirstIvarInRecord(
                                          const ObjCInterfaceDecl *OID,
                                          RecordDecl::field_iterator &FIV,
                                          RecordDecl::field_iterator &PIV) {
  int countSuperClassIvars = countInheritedIvars(OID->getSuperClass());
  const RecordDecl *RD = CGM.getContext().addRecordToClass(OID);
  RecordDecl::field_iterator ifield = RD->field_begin();
  RecordDecl::field_iterator pfield = RD->field_end();
  while (countSuperClassIvars-- > 0) {
    pfield = ifield;
    ++ifield;
  }
  FIV = ifield;
  PIV = pfield;
  return RD;
}

void CGObjCMac::FinishModule() {
  EmitModuleInfo();

  // Emit the dummy bodies for any protocols which were referenced but
  // never defined.
  for (llvm::DenseMap<IdentifierInfo*, llvm::GlobalVariable*>::iterator 
         i = Protocols.begin(), e = Protocols.end(); i != e; ++i) {
    if (i->second->hasInitializer())
      continue;

    std::vector<llvm::Constant*> Values(5);
    Values[0] = llvm::Constant::getNullValue(ObjCTypes.ProtocolExtensionPtrTy);
    Values[1] = GetClassName(i->first);
    Values[2] = llvm::Constant::getNullValue(ObjCTypes.ProtocolListPtrTy);
    Values[3] = Values[4] =
      llvm::Constant::getNullValue(ObjCTypes.MethodDescriptionListPtrTy);
    i->second->setLinkage(llvm::GlobalValue::InternalLinkage);
    i->second->setInitializer(llvm::ConstantStruct::get(ObjCTypes.ProtocolTy,
                                                        Values));
  }

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

  // Add assembler directives to add lazy undefined symbol references
  // for classes which are referenced but not defined. This is
  // important for correct linker interaction.

  // FIXME: Uh, this isn't particularly portable.
  std::stringstream s;
  
  if (!CGM.getModule().getModuleInlineAsm().empty())
    s << "\n";
  
  for (std::set<IdentifierInfo*>::iterator i = LazySymbols.begin(),
         e = LazySymbols.end(); i != e; ++i) {
    s << "\t.lazy_reference .objc_class_name_" << (*i)->getName() << "\n";
  }
  for (std::set<IdentifierInfo*>::iterator i = DefinedSymbols.begin(),
         e = DefinedSymbols.end(); i != e; ++i) {
    s << "\t.objc_class_name_" << (*i)->getName() << "=0\n"
      << "\t.globl .objc_class_name_" << (*i)->getName() << "\n";
  }  
  
  CGM.getModule().appendModuleInlineAsm(s.str());
}

CGObjCNonFragileABIMac::CGObjCNonFragileABIMac(CodeGen::CodeGenModule &cgm) 
  : CGObjCCommonMac(cgm),
  ObjCTypes(cgm)
{
  ObjCEmptyCacheVar = ObjCEmptyVtableVar = NULL;
  ObjCABI = 2;
}

/* *** */

ObjCCommonTypesHelper::ObjCCommonTypesHelper(CodeGen::CodeGenModule &cgm)
: CGM(cgm)
{
  CodeGen::CodeGenTypes &Types = CGM.getTypes();
  ASTContext &Ctx = CGM.getContext();
  
  ShortTy = Types.ConvertType(Ctx.ShortTy);
  IntTy = Types.ConvertType(Ctx.IntTy);
  LongTy = Types.ConvertType(Ctx.LongTy);
  Int8PtrTy = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
  
  ObjectPtrTy = Types.ConvertType(Ctx.getObjCIdType());
  PtrObjectPtrTy = llvm::PointerType::getUnqual(ObjectPtrTy);
  SelectorPtrTy = Types.ConvertType(Ctx.getObjCSelType());
  
  // FIXME: It would be nice to unify this with the opaque type, so
  // that the IR comes out a bit cleaner.
  const llvm::Type *T = Types.ConvertType(Ctx.getObjCProtoType());
  ExternalProtocolPtrTy = llvm::PointerType::getUnqual(T);
  
  // I'm not sure I like this. The implicit coordination is a bit
  // gross. We should solve this in a reasonable fashion because this
  // is a pretty common task (match some runtime data structure with
  // an LLVM data structure).
  
  // FIXME: This is leaked.
  // FIXME: Merge with rewriter code?
  
  // struct _objc_super {
  //   id self;
  //   Class cls;
  // }
  RecordDecl *RD = RecordDecl::Create(Ctx, TagDecl::TK_struct, 0,
                                      SourceLocation(),
                                      &Ctx.Idents.get("_objc_super"));  
  RD->addDecl(FieldDecl::Create(Ctx, RD, SourceLocation(), 0, 
                                Ctx.getObjCIdType(), 0, false));
  RD->addDecl(FieldDecl::Create(Ctx, RD, SourceLocation(), 0,
                                Ctx.getObjCClassType(), 0, false));
  RD->completeDefinition(Ctx);
  
  SuperCTy = Ctx.getTagDeclType(RD);
  SuperPtrCTy = Ctx.getPointerType(SuperCTy);
  
  SuperTy = cast<llvm::StructType>(Types.ConvertType(SuperCTy));
  SuperPtrTy = llvm::PointerType::getUnqual(SuperTy); 
  
  // struct _prop_t {
  //   char *name;
  //   char *attributes; 
  // }
  PropertyTy = llvm::StructType::get(Int8PtrTy,
                                     Int8PtrTy,
                                     NULL);
  CGM.getModule().addTypeName("struct._prop_t", 
                              PropertyTy);
  
  // struct _prop_list_t {
  //   uint32_t entsize;      // sizeof(struct _prop_t)
  //   uint32_t count_of_properties;
  //   struct _prop_t prop_list[count_of_properties];
  // }
  PropertyListTy = llvm::StructType::get(IntTy,
                                         IntTy,
                                         llvm::ArrayType::get(PropertyTy, 0),
                                         NULL);
  CGM.getModule().addTypeName("struct._prop_list_t", 
                              PropertyListTy);
  // struct _prop_list_t *
  PropertyListPtrTy = llvm::PointerType::getUnqual(PropertyListTy);
  
  // struct _objc_method {
  //   SEL _cmd;
  //   char *method_type;
  //   char *_imp;
  // }
  MethodTy = llvm::StructType::get(SelectorPtrTy,
                                   Int8PtrTy,
                                   Int8PtrTy,
                                   NULL);
  CGM.getModule().addTypeName("struct._objc_method", MethodTy);
  
  // struct _objc_cache *
  CacheTy = llvm::OpaqueType::get();
  CGM.getModule().addTypeName("struct._objc_cache", CacheTy);
  CachePtrTy = llvm::PointerType::getUnqual(CacheTy);
    
  // Property manipulation functions.

  QualType IdType = Ctx.getObjCIdType();
  QualType SelType = Ctx.getObjCSelType();
  llvm::SmallVector<QualType,16> Params;
  const llvm::FunctionType *FTy;
  
  // id objc_getProperty (id, SEL, ptrdiff_t, bool)
  Params.push_back(IdType);
  Params.push_back(SelType);
  Params.push_back(Ctx.LongTy);
  Params.push_back(Ctx.BoolTy);
  FTy = Types.GetFunctionType(Types.getFunctionInfo(IdType, Params), 
                              false);
  GetPropertyFn = CGM.CreateRuntimeFunction(FTy, "objc_getProperty");
  
  // void objc_setProperty (id, SEL, ptrdiff_t, id, bool, bool)
  Params.clear();
  Params.push_back(IdType);
  Params.push_back(SelType);
  Params.push_back(Ctx.LongTy);
  Params.push_back(IdType);
  Params.push_back(Ctx.BoolTy);
  Params.push_back(Ctx.BoolTy);
  FTy = Types.GetFunctionType(Types.getFunctionInfo(Ctx.VoidTy, Params), false);
  SetPropertyFn = CGM.CreateRuntimeFunction(FTy, "objc_setProperty");

  // Enumeration mutation.

  // void objc_enumerationMutation (id)
  Params.clear();
  Params.push_back(IdType);
  FTy = Types.GetFunctionType(Types.getFunctionInfo(Ctx.VoidTy, Params), false);
  EnumerationMutationFn = CGM.CreateRuntimeFunction(FTy,
                                                    "objc_enumerationMutation");
  
  // gc's API
  // id objc_read_weak (id *)
  Params.clear();
  Params.push_back(Ctx.getPointerType(IdType));
  FTy = Types.GetFunctionType(Types.getFunctionInfo(IdType, Params), false);
  GcReadWeakFn = CGM.CreateRuntimeFunction(FTy, "objc_read_weak");

  // id objc_assign_weak (id, id *)
  Params.clear();
  Params.push_back(IdType);
  Params.push_back(Ctx.getPointerType(IdType));
  
  FTy = Types.GetFunctionType(Types.getFunctionInfo(IdType, Params), false);
  GcAssignWeakFn = CGM.CreateRuntimeFunction(FTy, "objc_assign_weak");
  GcAssignGlobalFn = CGM.CreateRuntimeFunction(FTy, "objc_assign_global");
  GcAssignIvarFn = CGM.CreateRuntimeFunction(FTy, "objc_assign_ivar");
  GcAssignStrongCastFn = 
    CGM.CreateRuntimeFunction(FTy, "objc_assign_strongCast");
}

ObjCTypesHelper::ObjCTypesHelper(CodeGen::CodeGenModule &cgm) 
  : ObjCCommonTypesHelper(cgm)
{
  // struct _objc_method_description {
  //   SEL name;
  //   char *types;
  // }
  MethodDescriptionTy = 
    llvm::StructType::get(SelectorPtrTy,
                          Int8PtrTy,
                          NULL);
  CGM.getModule().addTypeName("struct._objc_method_description", 
                              MethodDescriptionTy);

  // struct _objc_method_description_list {
  //   int count;
  //   struct _objc_method_description[1];
  // }
  MethodDescriptionListTy = 
    llvm::StructType::get(IntTy,
                          llvm::ArrayType::get(MethodDescriptionTy, 0),
                          NULL);
  CGM.getModule().addTypeName("struct._objc_method_description_list", 
                              MethodDescriptionListTy);
  
  // struct _objc_method_description_list *
  MethodDescriptionListPtrTy = 
    llvm::PointerType::getUnqual(MethodDescriptionListTy);

  // Protocol description structures

  // struct _objc_protocol_extension {
  //   uint32_t size;  // sizeof(struct _objc_protocol_extension)
  //   struct _objc_method_description_list *optional_instance_methods;
  //   struct _objc_method_description_list *optional_class_methods;
  //   struct _objc_property_list *instance_properties;
  // }
  ProtocolExtensionTy = 
    llvm::StructType::get(IntTy,
                          MethodDescriptionListPtrTy,
                          MethodDescriptionListPtrTy,
                          PropertyListPtrTy,
                          NULL);
  CGM.getModule().addTypeName("struct._objc_protocol_extension", 
                              ProtocolExtensionTy);
  
  // struct _objc_protocol_extension *
  ProtocolExtensionPtrTy = llvm::PointerType::getUnqual(ProtocolExtensionTy);

  // Handle recursive construction of Protocol and ProtocolList types

  llvm::PATypeHolder ProtocolTyHolder = llvm::OpaqueType::get();
  llvm::PATypeHolder ProtocolListTyHolder = llvm::OpaqueType::get();

  const llvm::Type *T = 
    llvm::StructType::get(llvm::PointerType::getUnqual(ProtocolListTyHolder),
                          LongTy,
                          llvm::ArrayType::get(ProtocolTyHolder, 0),
                          NULL);
  cast<llvm::OpaqueType>(ProtocolListTyHolder.get())->refineAbstractTypeTo(T);

  // struct _objc_protocol {
  //   struct _objc_protocol_extension *isa;
  //   char *protocol_name;
  //   struct _objc_protocol **_objc_protocol_list;
  //   struct _objc_method_description_list *instance_methods;
  //   struct _objc_method_description_list *class_methods;
  // }
  T = llvm::StructType::get(ProtocolExtensionPtrTy,
                            Int8PtrTy,
                            llvm::PointerType::getUnqual(ProtocolListTyHolder),
                            MethodDescriptionListPtrTy,
                            MethodDescriptionListPtrTy,
                            NULL);
  cast<llvm::OpaqueType>(ProtocolTyHolder.get())->refineAbstractTypeTo(T);

  ProtocolListTy = cast<llvm::StructType>(ProtocolListTyHolder.get());
  CGM.getModule().addTypeName("struct._objc_protocol_list", 
                              ProtocolListTy);
  // struct _objc_protocol_list *
  ProtocolListPtrTy = llvm::PointerType::getUnqual(ProtocolListTy);

  ProtocolTy = cast<llvm::StructType>(ProtocolTyHolder.get());
  CGM.getModule().addTypeName("struct._objc_protocol", ProtocolTy);
  ProtocolPtrTy = llvm::PointerType::getUnqual(ProtocolTy);

  // Class description structures

  // struct _objc_ivar {
  //   char *ivar_name;
  //   char *ivar_type;
  //   int  ivar_offset;
  // }
  IvarTy = llvm::StructType::get(Int8PtrTy, 
                                 Int8PtrTy, 
                                 IntTy, 
                                 NULL);
  CGM.getModule().addTypeName("struct._objc_ivar", IvarTy);

  // struct _objc_ivar_list *
  IvarListTy = llvm::OpaqueType::get();
  CGM.getModule().addTypeName("struct._objc_ivar_list", IvarListTy);
  IvarListPtrTy = llvm::PointerType::getUnqual(IvarListTy);

  // struct _objc_method_list *
  MethodListTy = llvm::OpaqueType::get();
  CGM.getModule().addTypeName("struct._objc_method_list", MethodListTy);
  MethodListPtrTy = llvm::PointerType::getUnqual(MethodListTy);

  // struct _objc_class_extension *
  ClassExtensionTy = 
    llvm::StructType::get(IntTy,
                          Int8PtrTy,
                          PropertyListPtrTy,
                          NULL);
  CGM.getModule().addTypeName("struct._objc_class_extension", ClassExtensionTy);
  ClassExtensionPtrTy = llvm::PointerType::getUnqual(ClassExtensionTy);

  llvm::PATypeHolder ClassTyHolder = llvm::OpaqueType::get();

  // struct _objc_class {
  //   Class isa;
  //   Class super_class;
  //   char *name;
  //   long version;
  //   long info;
  //   long instance_size;
  //   struct _objc_ivar_list *ivars;
  //   struct _objc_method_list *methods;
  //   struct _objc_cache *cache;
  //   struct _objc_protocol_list *protocols;
  //   char *ivar_layout;
  //   struct _objc_class_ext *ext;
  // };
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

  // struct _objc_category {
  //   char *category_name;
  //   char *class_name;
  //   struct _objc_method_list *instance_method;
  //   struct _objc_method_list *class_method;
  //   uint32_t size;  // sizeof(struct _objc_category)
  //   struct _objc_property_list *instance_properties;// category's @property
  // }
  CategoryTy = llvm::StructType::get(Int8PtrTy,
                                     Int8PtrTy,
                                     MethodListPtrTy,
                                     MethodListPtrTy,
                                     ProtocolListPtrTy,
                                     IntTy,
                                     PropertyListPtrTy,
                                     NULL);
  CGM.getModule().addTypeName("struct._objc_category", CategoryTy);

  // Global metadata structures

  // struct _objc_symtab {
  //   long sel_ref_cnt;
  //   SEL *refs;
  //   short cls_def_cnt;
  //   short cat_def_cnt;
  //   char *defs[cls_def_cnt + cat_def_cnt];
  // }
  SymtabTy = llvm::StructType::get(LongTy,
                                   SelectorPtrTy,
                                   ShortTy,
                                   ShortTy,
                                   llvm::ArrayType::get(Int8PtrTy, 0),
                                   NULL);
  CGM.getModule().addTypeName("struct._objc_symtab", SymtabTy);
  SymtabPtrTy = llvm::PointerType::getUnqual(SymtabTy);

  // struct _objc_module {
  //   long version;
  //   long size;   // sizeof(struct _objc_module)
  //   char *name;
  //   struct _objc_symtab* symtab;
  //  }
  ModuleTy = 
    llvm::StructType::get(LongTy,
                          LongTy,
                          Int8PtrTy,
                          SymtabPtrTy,
                          NULL);
  CGM.getModule().addTypeName("struct._objc_module", ModuleTy);

  // Message send functions.

  // id objc_msgSend (id, SEL, ...)
  std::vector<const llvm::Type*> Params;
  Params.push_back(ObjectPtrTy);
  Params.push_back(SelectorPtrTy);
  MessageSendFn = 
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(ObjectPtrTy,
                                                      Params,
                                                      true),
                              "objc_msgSend");
  
  // id objc_msgSend_stret (id, SEL, ...)
  Params.clear();
  Params.push_back(ObjectPtrTy);
  Params.push_back(SelectorPtrTy);
  MessageSendStretFn = 
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(llvm::Type::VoidTy,
                                                      Params,
                                                      true),
                              "objc_msgSend_stret");

  // 
  Params.clear();
  Params.push_back(ObjectPtrTy);
  Params.push_back(SelectorPtrTy);
  // FIXME: This should be long double on x86_64?
  // [double | long double] objc_msgSend_fpret(id self, SEL op, ...)
  MessageSendFpretFn = 
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(llvm::Type::DoubleTy,
                                                      Params,
                                                      true),
                              "objc_msgSend_fpret");
  
  // id objc_msgSendSuper(struct objc_super *super, SEL op, ...)
  Params.clear();
  Params.push_back(SuperPtrTy);
  Params.push_back(SelectorPtrTy);
  MessageSendSuperFn = 
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(ObjectPtrTy,
                                                      Params,
                                                      true),
                              "objc_msgSendSuper");

  // void objc_msgSendSuper_stret(void * stretAddr, struct objc_super *super, 
  //                              SEL op, ...)
  Params.clear();
  Params.push_back(Int8PtrTy);
  Params.push_back(SuperPtrTy);
  Params.push_back(SelectorPtrTy);
  MessageSendSuperStretFn = 
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(llvm::Type::VoidTy,
                                                      Params,
                                                      true),
                              "objc_msgSendSuper_stret");

  // There is no objc_msgSendSuper_fpret? How can that work?
  MessageSendSuperFpretFn = MessageSendSuperFn;
  
  // FIXME: This is the size of the setjmp buffer and should be 
  // target specific. 18 is what's used on 32-bit X86.
  uint64_t SetJmpBufferSize = 18;
 
  // Exceptions
  const llvm::Type *StackPtrTy = 
    llvm::ArrayType::get(llvm::PointerType::getUnqual(llvm::Type::Int8Ty), 4);
                           
  ExceptionDataTy = 
    llvm::StructType::get(llvm::ArrayType::get(llvm::Type::Int32Ty, 
                                               SetJmpBufferSize),
                          StackPtrTy, NULL);
  CGM.getModule().addTypeName("struct._objc_exception_data", 
                              ExceptionDataTy);

  Params.clear();
  Params.push_back(ObjectPtrTy);
  ExceptionThrowFn =
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(llvm::Type::VoidTy,
                                                      Params,
                                                      false),
                              "objc_exception_throw");
  
  Params.clear();
  Params.push_back(llvm::PointerType::getUnqual(ExceptionDataTy));
  ExceptionTryEnterFn = 
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(llvm::Type::VoidTy,
                                                      Params,
                                                      false),
                              "objc_exception_try_enter");
  ExceptionTryExitFn = 
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(llvm::Type::VoidTy,
                                                      Params,
                                                      false),
                              "objc_exception_try_exit");
  ExceptionExtractFn =
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(ObjectPtrTy,
                                                      Params,
                                                      false),
                              "objc_exception_extract");
  
  Params.clear();
  Params.push_back(ClassPtrTy);
  Params.push_back(ObjectPtrTy);
  ExceptionMatchFn = 
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(llvm::Type::Int32Ty,
                                                      Params,
                                                      false),
                              "objc_exception_match");
  
  // synchronized APIs
  // void objc_sync_enter (id)
  Params.clear();
  Params.push_back(ObjectPtrTy);
  SyncEnterFn =
  CGM.CreateRuntimeFunction(llvm::FunctionType::get(llvm::Type::VoidTy,
                                                    Params,
                                                    false),
                            "objc_sync_enter");
  // void objc_sync_exit (id)
  SyncExitFn =
  CGM.CreateRuntimeFunction(llvm::FunctionType::get(llvm::Type::VoidTy,
                                                    Params,
                                                    false),
                            "objc_sync_exit");
  

  Params.clear();
  Params.push_back(llvm::PointerType::getUnqual(llvm::Type::Int32Ty));
  SetJmpFn =
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(llvm::Type::Int32Ty,
                                                      Params,
                                                      false),
                              "_setjmp");
  
}

ObjCNonFragileABITypesHelper::ObjCNonFragileABITypesHelper(CodeGen::CodeGenModule &cgm) 
: ObjCCommonTypesHelper(cgm)
{
  // struct _method_list_t {
  //   uint32_t entsize;  // sizeof(struct _objc_method)
  //   uint32_t method_count;
  //   struct _objc_method method_list[method_count];
  // }
  MethodListnfABITy = llvm::StructType::get(IntTy,
                                            IntTy,
                                            llvm::ArrayType::get(MethodTy, 0),
                                            NULL);
  CGM.getModule().addTypeName("struct.__method_list_t",
                              MethodListnfABITy);
  // struct method_list_t *
  MethodListnfABIPtrTy = llvm::PointerType::getUnqual(MethodListnfABITy);
  
  // struct _protocol_t {
  //   id isa;  // NULL
  //   const char * const protocol_name;
  //   const struct _protocol_list_t * protocol_list; // super protocols
  //   const struct method_list_t * const instance_methods;
  //   const struct method_list_t * const class_methods;
  //   const struct method_list_t *optionalInstanceMethods;
  //   const struct method_list_t *optionalClassMethods;
  //   const struct _prop_list_t * properties;
  //   const uint32_t size;  // sizeof(struct _protocol_t)
  //   const uint32_t flags;  // = 0
  // }
  
  // Holder for struct _protocol_list_t *
  llvm::PATypeHolder ProtocolListTyHolder = llvm::OpaqueType::get();
  
  ProtocolnfABITy = llvm::StructType::get(ObjectPtrTy,
                                          Int8PtrTy,
                                          llvm::PointerType::getUnqual(
                                            ProtocolListTyHolder),
                                          MethodListnfABIPtrTy,
                                          MethodListnfABIPtrTy,
                                          MethodListnfABIPtrTy,
                                          MethodListnfABIPtrTy,
                                          PropertyListPtrTy,
                                          IntTy,
                                          IntTy,
                                          NULL);
  CGM.getModule().addTypeName("struct._protocol_t",
                              ProtocolnfABITy);
  
  // struct _protocol_list_t {
  //   long protocol_count;   // Note, this is 32/64 bit
  //   struct _protocol_t[protocol_count];
  // }
  ProtocolListnfABITy = llvm::StructType::get(LongTy,
                                              llvm::ArrayType::get(
                                                ProtocolnfABITy, 0),
                                              NULL);
  CGM.getModule().addTypeName("struct._objc_protocol_list",
                              ProtocolListnfABITy);
  
  // struct _objc_protocol_list*
  ProtocolListnfABIPtrTy = llvm::PointerType::getUnqual(ProtocolListnfABITy);
  
  // FIXME! Is this doing the right thing?
  cast<llvm::OpaqueType>(ProtocolListTyHolder.get())->refineAbstractTypeTo(
                                                      ProtocolListnfABIPtrTy);
  
  // struct _ivar_t {
  //   unsigned long int *offset;  // pointer to ivar offset location
  //   char *name;
  //   char *type;
  //   uint32_t alignment;
  //   uint32_t size;
  // }
  IvarnfABITy = llvm::StructType::get(llvm::PointerType::getUnqual(LongTy),
                                      Int8PtrTy,
                                      Int8PtrTy,
                                      IntTy,
                                      IntTy,
                                      NULL);
  CGM.getModule().addTypeName("struct._ivar_t", IvarnfABITy);
                                      
  // struct _ivar_list_t {
  //   uint32 entsize;  // sizeof(struct _ivar_t)
  //   uint32 count;
  //   struct _iver_t list[count];
  // }
  IvarListnfABITy = llvm::StructType::get(IntTy,
                                          IntTy,
                                          llvm::ArrayType::get(
                                                               IvarnfABITy, 0),
                                          NULL);
  CGM.getModule().addTypeName("struct._ivar_list_t", IvarListnfABITy);
  
  IvarListnfABIPtrTy = llvm::PointerType::getUnqual(IvarListnfABITy);
  
  // struct _class_ro_t {
  //   uint32_t const flags;
  //   uint32_t const instanceStart;
  //   uint32_t const instanceSize;
  //   uint32_t const reserved;  // only when building for 64bit targets
  //   const uint8_t * const ivarLayout;
  //   const char *const name;
  //   const struct _method_list_t * const baseMethods;
  //   const struct _objc_protocol_list *const baseProtocols;
  //   const struct _ivar_list_t *const ivars;
  //   const uint8_t * const weakIvarLayout;
  //   const struct _prop_list_t * const properties;
  // }
  
  // FIXME. Add 'reserved' field in 64bit abi mode!
  ClassRonfABITy = llvm::StructType::get(IntTy,
                                         IntTy,
                                         IntTy,
                                         Int8PtrTy,
                                         Int8PtrTy,
                                         MethodListnfABIPtrTy,
                                         ProtocolListnfABIPtrTy,
                                         IvarListnfABIPtrTy,
                                         Int8PtrTy,
                                         PropertyListPtrTy,
                                         NULL);
  CGM.getModule().addTypeName("struct._class_ro_t",
                              ClassRonfABITy);
  
  // ImpnfABITy - LLVM for id (*)(id, SEL, ...)
  std::vector<const llvm::Type*> Params;
  Params.push_back(ObjectPtrTy);
  Params.push_back(SelectorPtrTy);
  ImpnfABITy = llvm::PointerType::getUnqual(
                          llvm::FunctionType::get(ObjectPtrTy, Params, false));
  
  // struct _class_t {
  //   struct _class_t *isa;
  //   struct _class_t * const superclass;
  //   void *cache;
  //   IMP *vtable;
  //   struct class_ro_t *ro;
  // }
  
  llvm::PATypeHolder ClassTyHolder = llvm::OpaqueType::get();
  ClassnfABITy = llvm::StructType::get(llvm::PointerType::getUnqual(ClassTyHolder),
                                       llvm::PointerType::getUnqual(ClassTyHolder),
                                       CachePtrTy,
                                       llvm::PointerType::getUnqual(ImpnfABITy),
                                       llvm::PointerType::getUnqual(
                                                                ClassRonfABITy),
                                       NULL);
  CGM.getModule().addTypeName("struct._class_t", ClassnfABITy);

  cast<llvm::OpaqueType>(ClassTyHolder.get())->refineAbstractTypeTo(
                                                                ClassnfABITy);
  
  // LLVM for struct _class_t *
  ClassnfABIPtrTy = llvm::PointerType::getUnqual(ClassnfABITy);
  
  // struct _category_t {
  //   const char * const name;
  //   struct _class_t *const cls;
  //   const struct _method_list_t * const instance_methods;
  //   const struct _method_list_t * const class_methods;
  //   const struct _protocol_list_t * const protocols;
  //   const struct _prop_list_t * const properties;
  // }
  CategorynfABITy = llvm::StructType::get(Int8PtrTy,
                                          ClassnfABIPtrTy,
                                          MethodListnfABIPtrTy,
                                          MethodListnfABIPtrTy,
                                          ProtocolListnfABIPtrTy,
                                          PropertyListPtrTy,
                                          NULL);
  CGM.getModule().addTypeName("struct._category_t", CategorynfABITy);
  
  // New types for nonfragile abi messaging.
  CodeGen::CodeGenTypes &Types = CGM.getTypes();
  ASTContext &Ctx = CGM.getContext();
  
  // MessageRefTy - LLVM for:
  // struct _message_ref_t {
  //   IMP messenger;
  //   SEL name;
  // };
  
  // First the clang type for struct _message_ref_t
  RecordDecl *RD = RecordDecl::Create(Ctx, TagDecl::TK_struct, 0,
                                      SourceLocation(),
                                      &Ctx.Idents.get("_message_ref_t"));
  RD->addDecl(FieldDecl::Create(Ctx, RD, SourceLocation(), 0,
                                Ctx.VoidPtrTy, 0, false));
  RD->addDecl(FieldDecl::Create(Ctx, RD, SourceLocation(), 0,
                                Ctx.getObjCSelType(), 0, false));
  RD->completeDefinition(Ctx);
  
  MessageRefCTy = Ctx.getTagDeclType(RD);
  MessageRefCPtrTy = Ctx.getPointerType(MessageRefCTy);
  MessageRefTy = cast<llvm::StructType>(Types.ConvertType(MessageRefCTy));
  
  // MessageRefPtrTy - LLVM for struct _message_ref_t*
  MessageRefPtrTy = llvm::PointerType::getUnqual(MessageRefTy);
  
  // SuperMessageRefTy - LLVM for:
  // struct _super_message_ref_t {
  //   SUPER_IMP messenger;
  //   SEL name;
  // };
  SuperMessageRefTy = llvm::StructType::get(ImpnfABITy,
                                            SelectorPtrTy,
                                            NULL);
  CGM.getModule().addTypeName("struct._super_message_ref_t", SuperMessageRefTy);
  
  // SuperMessageRefPtrTy - LLVM for struct _super_message_ref_t*
  SuperMessageRefPtrTy = llvm::PointerType::getUnqual(SuperMessageRefTy);  
  
  // id objc_msgSend_fixup (id, struct message_ref_t*, ...)
  Params.clear();
  Params.push_back(ObjectPtrTy);
  Params.push_back(MessageRefPtrTy);
  MessengerTy = llvm::FunctionType::get(ObjectPtrTy,
                                        Params,
                                        true);
  MessageSendFixupFn = 
    CGM.CreateRuntimeFunction(MessengerTy,
                              "objc_msgSend_fixup");
  
  // id objc_msgSend_fpret_fixup (id, struct message_ref_t*, ...)
  MessageSendFpretFixupFn = 
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(ObjectPtrTy,
                                                      Params,
                                                      true),
                                  "objc_msgSend_fpret_fixup");
  
  // id objc_msgSend_stret_fixup (id, struct message_ref_t*, ...)
  MessageSendStretFixupFn =
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(ObjectPtrTy,
                                                      Params,
                                                      true),
                              "objc_msgSend_stret_fixup");
  
  // id objc_msgSendId_fixup (id, struct message_ref_t*, ...)
  MessageSendIdFixupFn =
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(ObjectPtrTy,
                                                      Params,
                                                      true),
                              "objc_msgSendId_fixup");
  
  
  // id objc_msgSendId_stret_fixup (id, struct message_ref_t*, ...)
  MessageSendIdStretFixupFn =
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(ObjectPtrTy,
                                                      Params,
                                                      true),
                              "objc_msgSendId_stret_fixup");
  
  // id objc_msgSendSuper2_fixup (struct objc_super *, 
  //                              struct _super_message_ref_t*, ...)
  Params.clear();
  Params.push_back(SuperPtrTy);
  Params.push_back(SuperMessageRefPtrTy);
  MessageSendSuper2FixupFn =
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(ObjectPtrTy,
                                                      Params,
                                                      true),
                              "objc_msgSendSuper2_fixup");
  
  
  // id objc_msgSendSuper2_stret_fixup (struct objc_super *, 
  //                                    struct _super_message_ref_t*, ...)
  MessageSendSuper2StretFixupFn =
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(ObjectPtrTy,
                                                      Params,
                                                      true),
                              "objc_msgSendSuper2_stret_fixup");
                                          
}

llvm::Function *CGObjCNonFragileABIMac::ModuleInitFunction() { 
  FinishNonFragileABIModule();
  
  return NULL;
}

void CGObjCNonFragileABIMac::FinishNonFragileABIModule() {
  // nonfragile abi has no module definition.
  
  // Build list of all implemented classe addresses in array
  // L_OBJC_LABEL_CLASS_$.
  // FIXME. Also generate in L_OBJC_LABEL_NONLAZY_CLASS_$
  // list of 'nonlazy' implementations (defined as those with a +load{}
  // method!!).
  unsigned NumClasses = DefinedClasses.size();
  if (NumClasses) {
    std::vector<llvm::Constant*> Symbols(NumClasses);
    for (unsigned i=0; i<NumClasses; i++)
      Symbols[i] = llvm::ConstantExpr::getBitCast(DefinedClasses[i],
                                                  ObjCTypes.Int8PtrTy);
    llvm::Constant* Init = 
      llvm::ConstantArray::get(llvm::ArrayType::get(ObjCTypes.Int8PtrTy,
                                                    NumClasses),
                               Symbols);
  
    llvm::GlobalVariable *GV =
      new llvm::GlobalVariable(Init->getType(), false,
                               llvm::GlobalValue::InternalLinkage,
                               Init,
                               "\01L_OBJC_LABEL_CLASS_$",
                               &CGM.getModule());
    GV->setSection("__DATA, __objc_classlist, regular, no_dead_strip");
    UsedGlobals.push_back(GV);
  }
  
  // Build list of all implemented category addresses in array
  // L_OBJC_LABEL_CATEGORY_$.
  // FIXME. Also generate in L_OBJC_LABEL_NONLAZY_CATEGORY_$
  // list of 'nonlazy' category implementations (defined as those with a +load{}
  // method!!).
  unsigned NumCategory = DefinedCategories.size();
  if (NumCategory) {
    std::vector<llvm::Constant*> Symbols(NumCategory);
    for (unsigned i=0; i<NumCategory; i++)
      Symbols[i] = llvm::ConstantExpr::getBitCast(DefinedCategories[i],
                                                  ObjCTypes.Int8PtrTy);
    llvm::Constant* Init = 
      llvm::ConstantArray::get(llvm::ArrayType::get(ObjCTypes.Int8PtrTy,
                                                    NumCategory),
                               Symbols);
    
    llvm::GlobalVariable *GV =
      new llvm::GlobalVariable(Init->getType(), false,
                               llvm::GlobalValue::InternalLinkage,
                               Init,
                               "\01L_OBJC_LABEL_CATEGORY_$",
                               &CGM.getModule());
    GV->setSection("__DATA, __objc_catlist, regular, no_dead_strip");
    UsedGlobals.push_back(GV);
  }
  
  //  static int L_OBJC_IMAGE_INFO[2] = { 0, flags };
  // FIXME. flags can be 0 | 1 | 2 | 6. For now just use 0
  std::vector<llvm::Constant*> Values(2);
  Values[0] = llvm::ConstantInt::get(ObjCTypes.IntTy, 0);
  Values[1] = llvm::ConstantInt::get(ObjCTypes.IntTy, 0);
  llvm::Constant* Init = llvm::ConstantArray::get(
                                      llvm::ArrayType::get(ObjCTypes.IntTy, 2),
                                      Values);   
  llvm::GlobalVariable *IMGV =
    new llvm::GlobalVariable(Init->getType(), false,
                             llvm::GlobalValue::InternalLinkage,
                             Init,
                             "\01L_OBJC_IMAGE_INFO",
                             &CGM.getModule());
  IMGV->setSection("__DATA, __objc_imageinfo, regular, no_dead_strip");
  UsedGlobals.push_back(IMGV);
  
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

// Metadata flags
enum MetaDataDlags {
  CLS = 0x0,
  CLS_META = 0x1,
  CLS_ROOT = 0x2,
  OBJC2_CLS_HIDDEN = 0x10,
  CLS_EXCEPTION = 0x20
};
/// BuildClassRoTInitializer - generate meta-data for:
/// struct _class_ro_t {
///   uint32_t const flags;
///   uint32_t const instanceStart;
///   uint32_t const instanceSize;
///   uint32_t const reserved;  // only when building for 64bit targets
///   const uint8_t * const ivarLayout;
///   const char *const name;
///   const struct _method_list_t * const baseMethods;
///   const struct _protocol_list_t *const baseProtocols;
///   const struct _ivar_list_t *const ivars;
///   const uint8_t * const weakIvarLayout;
///   const struct _prop_list_t * const properties;
/// }
///
llvm::GlobalVariable * CGObjCNonFragileABIMac::BuildClassRoTInitializer(
                                                unsigned flags, 
                                                unsigned InstanceStart,
                                                unsigned InstanceSize,
                                                const ObjCImplementationDecl *ID) {
  std::string ClassName = ID->getNameAsString();
  std::vector<llvm::Constant*> Values(10); // 11 for 64bit targets!
  Values[ 0] = llvm::ConstantInt::get(ObjCTypes.IntTy, flags);
  Values[ 1] = llvm::ConstantInt::get(ObjCTypes.IntTy, InstanceStart);
  Values[ 2] = llvm::ConstantInt::get(ObjCTypes.IntTy, InstanceSize);
  // FIXME. For 64bit targets add 0 here.
  // FIXME. ivarLayout is currently null!
  Values[ 3] = llvm::Constant::getNullValue(ObjCTypes.Int8PtrTy);
  Values[ 4] = GetClassName(ID->getIdentifier());
  // const struct _method_list_t * const baseMethods;
  std::vector<llvm::Constant*> Methods;
  std::string MethodListName("\01l_OBJC_$_");
  if (flags & CLS_META) {
    MethodListName += "CLASS_METHODS_" + ID->getNameAsString();
    for (ObjCImplementationDecl::classmeth_iterator i = ID->classmeth_begin(),
         e = ID->classmeth_end(); i != e; ++i) {
      // Class methods should always be defined.
      Methods.push_back(GetMethodConstant(*i));
    }
  } else {
    MethodListName += "INSTANCE_METHODS_" + ID->getNameAsString();
    for (ObjCImplementationDecl::instmeth_iterator i = ID->instmeth_begin(),
         e = ID->instmeth_end(); i != e; ++i) {
      // Instance methods should always be defined.
      Methods.push_back(GetMethodConstant(*i));
    }
    for (ObjCImplementationDecl::propimpl_iterator i = ID->propimpl_begin(),
         e = ID->propimpl_end(); i != e; ++i) {
      ObjCPropertyImplDecl *PID = *i;
      
      if (PID->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize){
        ObjCPropertyDecl *PD = PID->getPropertyDecl();
        
        if (ObjCMethodDecl *MD = PD->getGetterMethodDecl())
          if (llvm::Constant *C = GetMethodConstant(MD))
            Methods.push_back(C);
        if (ObjCMethodDecl *MD = PD->getSetterMethodDecl())
          if (llvm::Constant *C = GetMethodConstant(MD))
            Methods.push_back(C);
      }
    }
  }
  Values[ 5] = EmitMethodList(MethodListName, 
               "__DATA, __objc_const", Methods);
  
  const ObjCInterfaceDecl *OID = ID->getClassInterface();
  assert(OID && "CGObjCNonFragileABIMac::BuildClassRoTInitializer");
  Values[ 6] = EmitProtocolList("\01l_OBJC_CLASS_PROTOCOLS_$_" 
                                + OID->getNameAsString(),
                                OID->protocol_begin(),
                                OID->protocol_end());
  
  if (flags & CLS_META)
    Values[ 7] = llvm::Constant::getNullValue(ObjCTypes.IvarListnfABIPtrTy);
  else
    Values[ 7] = EmitIvarList(ID);
  // FIXME. weakIvarLayout is currently null.
  Values[ 8] = llvm::Constant::getNullValue(ObjCTypes.Int8PtrTy);
  if (flags & CLS_META)
    Values[ 9] = llvm::Constant::getNullValue(ObjCTypes.PropertyListPtrTy);
  else
    Values[ 9] = 
      EmitPropertyList(
                       "\01l_OBJC_$_PROP_LIST_" + ID->getNameAsString(),
                       ID, ID->getClassInterface(), ObjCTypes);
  llvm::Constant *Init = llvm::ConstantStruct::get(ObjCTypes.ClassRonfABITy,
                                                   Values);
  llvm::GlobalVariable *CLASS_RO_GV =
  new llvm::GlobalVariable(ObjCTypes.ClassRonfABITy, false,
                           llvm::GlobalValue::InternalLinkage,
                           Init,
                           (flags & CLS_META) ?
                           std::string("\01l_OBJC_METACLASS_RO_$_")+ClassName :
                           std::string("\01l_OBJC_CLASS_RO_$_")+ClassName,
                           &CGM.getModule());
  CLASS_RO_GV->setAlignment(
    CGM.getTargetData().getPrefTypeAlignment(ObjCTypes.ClassRonfABITy));
  CLASS_RO_GV->setSection("__DATA, __objc_const");
  UsedGlobals.push_back(CLASS_RO_GV);
  return CLASS_RO_GV;

}

/// BuildClassMetaData - This routine defines that to-level meta-data
/// for the given ClassName for:
/// struct _class_t {
///   struct _class_t *isa;
///   struct _class_t * const superclass;
///   void *cache;
///   IMP *vtable;
///   struct class_ro_t *ro;
/// }
///
llvm::GlobalVariable * CGObjCNonFragileABIMac::BuildClassMetaData(
                                                std::string &ClassName,
                                                llvm::Constant *IsAGV, 
                                                llvm::Constant *SuperClassGV,
                                                llvm::Constant *ClassRoGV,
                                                bool HiddenVisibility) {
  std::vector<llvm::Constant*> Values(5);
  Values[0] = IsAGV;
  Values[1] = SuperClassGV 
                ? SuperClassGV
                : llvm::Constant::getNullValue(ObjCTypes.ClassnfABIPtrTy);
  Values[2] = ObjCEmptyCacheVar;  // &ObjCEmptyCacheVar
  Values[3] = ObjCEmptyVtableVar; // &ObjCEmptyVtableVar
  Values[4] = ClassRoGV;                 // &CLASS_RO_GV
  llvm::Constant *Init = llvm::ConstantStruct::get(ObjCTypes.ClassnfABITy, 
                                                   Values);
  llvm::GlobalVariable *GV = CGM.getModule().getGlobalVariable(ClassName);
  if (GV)
    GV->setInitializer(Init);
  else
    GV =
    new llvm::GlobalVariable(ObjCTypes.ClassnfABITy, false,
                             llvm::GlobalValue::ExternalLinkage,
                             Init,
                             ClassName,
                             &CGM.getModule());
  GV->setSection("__DATA, __objc_data");
  GV->setAlignment(
    CGM.getTargetData().getPrefTypeAlignment(ObjCTypes.ClassnfABITy));
  if (HiddenVisibility)
    GV->setVisibility(llvm::GlobalValue::HiddenVisibility);
  UsedGlobals.push_back(GV);
  return GV;
}

void CGObjCNonFragileABIMac::GenerateClass(const ObjCImplementationDecl *ID) {
  std::string ClassName = ID->getNameAsString();
  if (!ObjCEmptyCacheVar) {
    ObjCEmptyCacheVar = new llvm::GlobalVariable(
                                            ObjCTypes.CachePtrTy,
                                            false,
                                            llvm::GlobalValue::ExternalLinkage,
                                            0,
                                            "\01__objc_empty_cache",
                                            &CGM.getModule());
    UsedGlobals.push_back(ObjCEmptyCacheVar);
    
    ObjCEmptyVtableVar = new llvm::GlobalVariable(
                            llvm::PointerType::getUnqual(
                                                  ObjCTypes.ImpnfABITy),
                            false,
                            llvm::GlobalValue::ExternalLinkage,
                            0,
                            "\01__objc_empty_vtable",
                            &CGM.getModule());
    UsedGlobals.push_back(ObjCEmptyVtableVar);
  }
  assert(ID->getClassInterface() && 
         "CGObjCNonFragileABIMac::GenerateClass - class is 0");
  uint32_t InstanceStart = 
    CGM.getTargetData().getTypePaddedSize(ObjCTypes.ClassnfABITy);
  uint32_t InstanceSize = InstanceStart;
  uint32_t flags = CLS_META;
  std::string ObjCMetaClassName("\01_OBJC_METACLASS_$_");
  std::string ObjCClassName("\01_OBJC_CLASS_$_");
  
  llvm::GlobalVariable *SuperClassGV, *IsAGV;
  
  bool classIsHidden = IsClassHidden(ID->getClassInterface());
  if (classIsHidden)
    flags |= OBJC2_CLS_HIDDEN;
  if (!ID->getClassInterface()->getSuperClass()) {
    // class is root
    flags |= CLS_ROOT;
    std::string SuperClassName = ObjCClassName + ClassName;
    SuperClassGV = CGM.getModule().getGlobalVariable(SuperClassName);
    if (!SuperClassGV) {
      SuperClassGV = 
        new llvm::GlobalVariable(ObjCTypes.ClassnfABITy, false,
                                llvm::GlobalValue::ExternalLinkage,
                                 0,
                                 SuperClassName,
                                 &CGM.getModule());
      UsedGlobals.push_back(SuperClassGV);
    }
    std::string IsAClassName = ObjCMetaClassName + ClassName;
    IsAGV = CGM.getModule().getGlobalVariable(IsAClassName);
    if (!IsAGV) {
      IsAGV = 
        new llvm::GlobalVariable(ObjCTypes.ClassnfABITy, false,
                                 llvm::GlobalValue::ExternalLinkage,
                                 0,
                                 IsAClassName,
                                 &CGM.getModule());
      UsedGlobals.push_back(IsAGV);
    }
  } else {
    // Has a root. Current class is not a root.
    std::string RootClassName = 
      ID->getClassInterface()->getSuperClass()->getNameAsString();
    std::string SuperClassName = ObjCMetaClassName + RootClassName;
    SuperClassGV = CGM.getModule().getGlobalVariable(SuperClassName);
    if (!SuperClassGV) {
      SuperClassGV = 
        new llvm::GlobalVariable(ObjCTypes.ClassnfABITy, false,
                                 llvm::GlobalValue::ExternalLinkage,
                                 0,
                                 SuperClassName,
                                 &CGM.getModule());
      UsedGlobals.push_back(SuperClassGV);
    }
    IsAGV = SuperClassGV;
  }
  llvm::GlobalVariable *CLASS_RO_GV = BuildClassRoTInitializer(flags,
                                                               InstanceStart,
                                                               InstanceSize,ID);
  std::string TClassName = ObjCMetaClassName + ClassName;
  llvm::GlobalVariable *MetaTClass = 
    BuildClassMetaData(TClassName, IsAGV, SuperClassGV, CLASS_RO_GV,
                       classIsHidden);
  
  // Metadata for the class
  flags = CLS;
  if (classIsHidden)
    flags |= OBJC2_CLS_HIDDEN;
  if (!ID->getClassInterface()->getSuperClass()) {
    flags |= CLS_ROOT;
    SuperClassGV = 0;
  }
  else {
    // Has a root. Current class is not a root.
    std::string RootClassName = 
      ID->getClassInterface()->getSuperClass()->getNameAsString();
    std::string SuperClassName = ObjCClassName + RootClassName;
    SuperClassGV = CGM.getModule().getGlobalVariable(SuperClassName);
    if (!SuperClassGV) {
      SuperClassGV = 
      new llvm::GlobalVariable(ObjCTypes.ClassnfABITy, false,
                               llvm::GlobalValue::ExternalLinkage,
                               0,
                               SuperClassName,
                               &CGM.getModule());
      UsedGlobals.push_back(SuperClassGV);
    }
  }
  
  InstanceStart = InstanceSize = 0;
  if (ObjCInterfaceDecl *OID = 
      const_cast<ObjCInterfaceDecl*>(ID->getClassInterface())) {
    // FIXME. Share this with the one in EmitIvarList.
    const llvm::Type *InterfaceTy = 
    CGM.getTypes().ConvertType(CGM.getContext().buildObjCInterfaceType(OID));
    const llvm::StructLayout *Layout =
    CGM.getTargetData().getStructLayout(cast<llvm::StructType>(InterfaceTy));
    
    RecordDecl::field_iterator firstField, lastField;
    const RecordDecl *RD = GetFirstIvarInRecord(OID, firstField, lastField);
    
    for (RecordDecl::field_iterator e = RD->field_end(),
         ifield = firstField; ifield != e; ++ifield)
      lastField = ifield;
    
    if (lastField != RD->field_end()) {
      FieldDecl *Field = *lastField;
      const llvm::Type *FieldTy =
        CGM.getTypes().ConvertTypeForMem(Field->getType());
      unsigned Size = CGM.getTargetData().getTypePaddedSize(FieldTy);
      InstanceSize = Layout->getElementOffset(
                                  CGM.getTypes().getLLVMFieldNo(Field)) + 
                                  Size;
      if (firstField == RD->field_end())
        InstanceStart = InstanceSize;
      else
        InstanceStart =  Layout->getElementOffset(CGM.getTypes().
                                                  getLLVMFieldNo(*firstField));
    }
  }
  CLASS_RO_GV = BuildClassRoTInitializer(flags,
                                         InstanceStart,
                                         InstanceSize, 
                                         ID);
  
  TClassName = ObjCClassName + ClassName;
  llvm::GlobalVariable *ClassMD = 
    BuildClassMetaData(TClassName, MetaTClass, SuperClassGV, CLASS_RO_GV,
                       classIsHidden);
  DefinedClasses.push_back(ClassMD);
}

/// GenerateProtocolRef - This routine is called to generate code for
/// a protocol reference expression; as in:
/// @code
///   @protocol(Proto1);
/// @endcode
/// It generates a weak reference to l_OBJC_PROTOCOL_REFERENCE_$_Proto1
/// which will hold address of the protocol meta-data.
///
llvm::Value *CGObjCNonFragileABIMac::GenerateProtocolRef(CGBuilderTy &Builder,
                                            const ObjCProtocolDecl *PD) {
  
  llvm::Constant *Init =  llvm::ConstantExpr::getBitCast(GetProtocolRef(PD),
                                        ObjCTypes.ExternalProtocolPtrTy);
  
  std::string ProtocolName("\01l_OBJC_PROTOCOL_REFERENCE_$_");
  ProtocolName += PD->getNameAsCString();
                            
  llvm::GlobalVariable *PTGV = CGM.getModule().getGlobalVariable(ProtocolName);
  if (PTGV)
    return Builder.CreateLoad(PTGV, false, "tmp");
  PTGV = new llvm::GlobalVariable(
                                Init->getType(), false,
                                llvm::GlobalValue::WeakLinkage,
                                Init,
                                ProtocolName,
                                &CGM.getModule());
  PTGV->setSection("__DATA, __objc_protorefs, coalesced, no_dead_strip");
  PTGV->setVisibility(llvm::GlobalValue::HiddenVisibility);
  UsedGlobals.push_back(PTGV);
  return Builder.CreateLoad(PTGV, false, "tmp");
}

/// GenerateCategory - Build metadata for a category implementation.
/// struct _category_t {
///   const char * const name;
///   struct _class_t *const cls;
///   const struct _method_list_t * const instance_methods;
///   const struct _method_list_t * const class_methods;
///   const struct _protocol_list_t * const protocols;
///   const struct _prop_list_t * const properties;
/// }
///
void CGObjCNonFragileABIMac::GenerateCategory(const ObjCCategoryImplDecl *OCD) 
{
  const ObjCInterfaceDecl *Interface = OCD->getClassInterface();
  const char *Prefix = "\01l_OBJC_$_CATEGORY_";
  std::string ExtCatName(Prefix + Interface->getNameAsString()+ 
                      "_$_" + OCD->getNameAsString());
  std::string ExtClassName("\01_OBJC_CLASS_$_" + Interface->getNameAsString());
  
  std::vector<llvm::Constant*> Values(6);
  Values[0] = GetClassName(OCD->getIdentifier());
  // meta-class entry symbol
  llvm::GlobalVariable *ClassGV = 
    CGM.getModule().getGlobalVariable(ExtClassName);
  if (!ClassGV)
    ClassGV =
    new llvm::GlobalVariable(ObjCTypes.ClassnfABITy, false,
                             llvm::GlobalValue::ExternalLinkage,
                             0,
                             ExtClassName,
                             &CGM.getModule());
  UsedGlobals.push_back(ClassGV);
  Values[1] = ClassGV;
  std::vector<llvm::Constant*> Methods;
  std::string MethodListName(Prefix);
  MethodListName += "INSTANCE_METHODS_" + Interface->getNameAsString() + 
    "_$_" + OCD->getNameAsString();
   
  for (ObjCCategoryImplDecl::instmeth_iterator i = OCD->instmeth_begin(),
       e = OCD->instmeth_end(); i != e; ++i) {
    // Instance methods should always be defined.
    Methods.push_back(GetMethodConstant(*i));
  }
  
  Values[2] = EmitMethodList(MethodListName, 
                             "__DATA, __objc_const", 
                             Methods);

  MethodListName = Prefix;
  MethodListName += "CLASS_METHODS_" + Interface->getNameAsString() + "_$_" +
    OCD->getNameAsString();
  Methods.clear();
  for (ObjCCategoryImplDecl::classmeth_iterator i = OCD->classmeth_begin(),
       e = OCD->classmeth_end(); i != e; ++i) {
    // Class methods should always be defined.
    Methods.push_back(GetMethodConstant(*i));
  }
  
  Values[3] = EmitMethodList(MethodListName, 
                             "__DATA, __objc_const", 
                             Methods);
  const ObjCCategoryDecl *Category = 
    Interface->FindCategoryDeclaration(OCD->getIdentifier());
  if (Category) {
    std::string ExtName(Interface->getNameAsString() + "_$_" +
                        OCD->getNameAsString());
    Values[4] = EmitProtocolList("\01l_OBJC_CATEGORY_PROTOCOLS_$_"
                                 + Interface->getNameAsString() + "_$_"
                                 + Category->getNameAsString(),
                                 Category->protocol_begin(),
                                 Category->protocol_end());
    Values[5] =
      EmitPropertyList(std::string("\01l_OBJC_$_PROP_LIST_") + ExtName,
                       OCD, Category, ObjCTypes);
  }
  else {
    Values[4] = llvm::Constant::getNullValue(ObjCTypes.ProtocolListnfABIPtrTy);
    Values[5] = llvm::Constant::getNullValue(ObjCTypes.PropertyListPtrTy);
  }
    
  llvm::Constant *Init = 
    llvm::ConstantStruct::get(ObjCTypes.CategorynfABITy, 
                              Values);
  llvm::GlobalVariable *GCATV
    = new llvm::GlobalVariable(ObjCTypes.CategorynfABITy, 
                               false,
                               llvm::GlobalValue::InternalLinkage,
                               Init,
                               ExtCatName,
                               &CGM.getModule());
  GCATV->setAlignment(
    CGM.getTargetData().getPrefTypeAlignment(ObjCTypes.CategorynfABITy));
  GCATV->setSection("__DATA, __objc_const");
  UsedGlobals.push_back(GCATV);
  DefinedCategories.push_back(GCATV);
}

/// GetMethodConstant - Return a struct objc_method constant for the
/// given method if it has been defined. The result is null if the
/// method has not been defined. The return value has type MethodPtrTy.
llvm::Constant *CGObjCNonFragileABIMac::GetMethodConstant(
                                                    const ObjCMethodDecl *MD) {
  // FIXME: Use DenseMap::lookup
  llvm::Function *Fn = MethodDefinitions[MD];
  if (!Fn)
    return 0;
  
  std::vector<llvm::Constant*> Method(3);
  Method[0] = 
  llvm::ConstantExpr::getBitCast(GetMethodVarName(MD->getSelector()),
                                 ObjCTypes.SelectorPtrTy);
  Method[1] = GetMethodVarType(MD);
  Method[2] = llvm::ConstantExpr::getBitCast(Fn, ObjCTypes.Int8PtrTy);
  return llvm::ConstantStruct::get(ObjCTypes.MethodTy, Method);
}

/// EmitMethodList - Build meta-data for method declarations
/// struct _method_list_t {
///   uint32_t entsize;  // sizeof(struct _objc_method)
///   uint32_t method_count;
///   struct _objc_method method_list[method_count];
/// }
///
llvm::Constant *CGObjCNonFragileABIMac::EmitMethodList(
                                              const std::string &Name,
                                              const char *Section,
                                              const ConstantVector &Methods) {
  // Return null for empty list.
  if (Methods.empty())
    return llvm::Constant::getNullValue(ObjCTypes.MethodListnfABIPtrTy);
  
  std::vector<llvm::Constant*> Values(3);
  // sizeof(struct _objc_method)
  unsigned Size = CGM.getTargetData().getTypePaddedSize(ObjCTypes.MethodTy);
  Values[0] = llvm::ConstantInt::get(ObjCTypes.IntTy, Size);
  // method_count
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
  GV->setAlignment(
    CGM.getTargetData().getPrefTypeAlignment(Init->getType()));
  GV->setSection(Section);
  UsedGlobals.push_back(GV);
  return llvm::ConstantExpr::getBitCast(GV,
                                        ObjCTypes.MethodListnfABIPtrTy);
}

/// ObjCIvarOffsetVariable - Returns the ivar offset variable for
/// the given ivar.
///
llvm::GlobalVariable * CGObjCNonFragileABIMac::ObjCIvarOffsetVariable(
                              std::string &Name, 
                              const ObjCInterfaceDecl *ID,
                              const ObjCIvarDecl *Ivar) {
  Name += "\01_OBJC_IVAR_$_" + 
          getInterfaceDeclForIvar(ID, Ivar)->getNameAsString() + '.'
          + Ivar->getNameAsString();
  llvm::GlobalVariable *IvarOffsetGV = 
    CGM.getModule().getGlobalVariable(Name);
  if (!IvarOffsetGV)
    IvarOffsetGV = 
      new llvm::GlobalVariable(ObjCTypes.LongTy,
                               false,
                               llvm::GlobalValue::ExternalLinkage,
                               0,
                               Name,
                               &CGM.getModule());
  return IvarOffsetGV;
}

llvm::Constant * CGObjCNonFragileABIMac::EmitIvarOffsetVar(
                                              const ObjCInterfaceDecl *ID,
                                              const ObjCIvarDecl *Ivar,
                                              unsigned long int Offset) {
  
  assert(ID && "EmitIvarOffsetVar - null interface decl.");
  std::string ExternalName("\01_OBJC_IVAR_$_" + ID->getNameAsString() + '.' 
                           + Ivar->getNameAsString());
  llvm::Constant *Init = llvm::ConstantInt::get(ObjCTypes.LongTy, Offset);
  
  llvm::GlobalVariable *IvarOffsetGV = 
    CGM.getModule().getGlobalVariable(ExternalName);
  if (IvarOffsetGV) {
    // ivar offset symbol already built due to user code referencing it.
    IvarOffsetGV->setAlignment(
      CGM.getTargetData().getPrefTypeAlignment(ObjCTypes.LongTy));
    IvarOffsetGV->setInitializer(Init);
    IvarOffsetGV->setSection("__DATA, __objc_const");
    UsedGlobals.push_back(IvarOffsetGV);
    return IvarOffsetGV;
  }
  
  IvarOffsetGV = 
    new llvm::GlobalVariable(Init->getType(),
                             false,
                             llvm::GlobalValue::ExternalLinkage,
                             Init,
                             ExternalName,
                             &CGM.getModule());
  IvarOffsetGV->setAlignment(
    CGM.getTargetData().getPrefTypeAlignment(ObjCTypes.LongTy));
  // @private and @package have hidden visibility.
  bool globalVisibility = (Ivar->getAccessControl() == ObjCIvarDecl::Public ||
                           Ivar->getAccessControl() == ObjCIvarDecl::Protected);
  if (!globalVisibility)
    IvarOffsetGV->setVisibility(llvm::GlobalValue::HiddenVisibility);
  else 
    if (IsClassHidden(ID))
      IvarOffsetGV->setVisibility(llvm::GlobalValue::HiddenVisibility);

  IvarOffsetGV->setSection("__DATA, __objc_const");
  UsedGlobals.push_back(IvarOffsetGV);
  return IvarOffsetGV;
}

/// EmitIvarList - Emit the ivar list for the given
/// implementation. If ForClass is true the list of class ivars
/// (i.e. metaclass ivars) is emitted, otherwise the list of
/// interface ivars will be emitted. The return value has type
/// IvarListnfABIPtrTy.
///  struct _ivar_t {
///   unsigned long int *offset;  // pointer to ivar offset location
///   char *name;
///   char *type;
///   uint32_t alignment;
///   uint32_t size;
/// }
/// struct _ivar_list_t {
///   uint32 entsize;  // sizeof(struct _ivar_t)
///   uint32 count;
///   struct _iver_t list[count];
/// }
///
llvm::Constant *CGObjCNonFragileABIMac::EmitIvarList(
                                            const ObjCImplementationDecl *ID) {
  
  std::vector<llvm::Constant*> Ivars, Ivar(5);
  
  const ObjCInterfaceDecl *OID = ID->getClassInterface();
  assert(OID && "CGObjCNonFragileABIMac::EmitIvarList - null interface");
  
  // FIXME. Consolidate this with similar code in GenerateClass.
  const llvm::Type *InterfaceTy =
    CGM.getTypes().ConvertType(CGM.getContext().getObjCInterfaceType(
                                        const_cast<ObjCInterfaceDecl*>(OID)));
  const llvm::StructLayout *Layout =
    CGM.getTargetData().getStructLayout(cast<llvm::StructType>(InterfaceTy));
  
  RecordDecl::field_iterator i,p;
  const RecordDecl *RD = GetFirstIvarInRecord(OID, i,p);
  ObjCInterfaceDecl::ivar_iterator I = OID->ivar_begin();
  
  for (RecordDecl::field_iterator e = RD->field_end(); i != e; ++i) {
    FieldDecl *Field = *i;
    unsigned long offset = Layout->getElementOffset(CGM.getTypes().
                                                    getLLVMFieldNo(Field));
    const ObjCIvarDecl *ivarDecl = *I++;
    Ivar[0] = EmitIvarOffsetVar(ID->getClassInterface(), ivarDecl, offset);
    if (Field->getIdentifier())
      Ivar[1] = GetMethodVarName(Field->getIdentifier());
    else
      Ivar[1] = llvm::Constant::getNullValue(ObjCTypes.Int8PtrTy);
    std::string TypeStr;
    CGM.getContext().getObjCEncodingForType(Field->getType(), TypeStr, Field);
    Ivar[2] = GetMethodVarType(TypeStr);
    const llvm::Type *FieldTy =
      CGM.getTypes().ConvertTypeForMem(Field->getType());
    unsigned Size = CGM.getTargetData().getTypePaddedSize(FieldTy);
    unsigned Align = CGM.getContext().getPreferredTypeAlign(
                       Field->getType().getTypePtr()) >> 3;
    Align = llvm::Log2_32(Align);
    Ivar[3] = llvm::ConstantInt::get(ObjCTypes.IntTy, Align);
    // NOTE. Size of a bitfield does not match gcc's, because of the way
    // bitfields are treated special in each. But I am told that 'size'
    // for bitfield ivars is ignored by the runtime so it does not matter.
    // (even if it matters, some day, there is enough info. to get the bitfield
    // right!
    Ivar[4] = llvm::ConstantInt::get(ObjCTypes.IntTy, Size);
    Ivars.push_back(llvm::ConstantStruct::get(ObjCTypes.IvarnfABITy, Ivar));
  }
  // Return null for empty list.
  if (Ivars.empty())
    return llvm::Constant::getNullValue(ObjCTypes.IvarListnfABIPtrTy);
  std::vector<llvm::Constant*> Values(3);
  unsigned Size = CGM.getTargetData().getTypePaddedSize(ObjCTypes.IvarnfABITy);
  Values[0] = llvm::ConstantInt::get(ObjCTypes.IntTy, Size);
  Values[1] = llvm::ConstantInt::get(ObjCTypes.IntTy, Ivars.size());
  llvm::ArrayType *AT = llvm::ArrayType::get(ObjCTypes.IvarnfABITy,
                                             Ivars.size());
  Values[2] = llvm::ConstantArray::get(AT, Ivars);
  llvm::Constant *Init = llvm::ConstantStruct::get(Values);
  const char *Prefix = "\01l_OBJC_$_INSTANCE_VARIABLES_";
  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(Init->getType(), false,
                             llvm::GlobalValue::InternalLinkage,
                             Init,
                             Prefix + OID->getNameAsString(),
                             &CGM.getModule());
  GV->setAlignment(
    CGM.getTargetData().getPrefTypeAlignment(Init->getType()));
  GV->setSection("__DATA, __objc_const");
                 
  UsedGlobals.push_back(GV);
  return llvm::ConstantExpr::getBitCast(GV,
                                        ObjCTypes.IvarListnfABIPtrTy);
}

llvm::Constant *CGObjCNonFragileABIMac::GetOrEmitProtocolRef(
                                                  const ObjCProtocolDecl *PD) {
  llvm::GlobalVariable *&Entry = Protocols[PD->getIdentifier()];
  
  if (!Entry) {
    // We use the initializer as a marker of whether this is a forward
    // reference or not. At module finalization we add the empty
    // contents for protocols which were referenced but never defined.
    Entry = 
    new llvm::GlobalVariable(ObjCTypes.ProtocolnfABITy, false,
                             llvm::GlobalValue::ExternalLinkage,
                             0,
                             "\01l_OBJC_PROTOCOL_$_" + PD->getNameAsString(),
                             &CGM.getModule());
    Entry->setSection("__DATA,__datacoal_nt,coalesced");
    UsedGlobals.push_back(Entry);
  }
  
  return Entry;
}

/// GetOrEmitProtocol - Generate the protocol meta-data:
/// @code
/// struct _protocol_t {
///   id isa;  // NULL
///   const char * const protocol_name;
///   const struct _protocol_list_t * protocol_list; // super protocols
///   const struct method_list_t * const instance_methods;
///   const struct method_list_t * const class_methods;
///   const struct method_list_t *optionalInstanceMethods;
///   const struct method_list_t *optionalClassMethods;
///   const struct _prop_list_t * properties;
///   const uint32_t size;  // sizeof(struct _protocol_t)
///   const uint32_t flags;  // = 0
/// }
/// @endcode
///

llvm::Constant *CGObjCNonFragileABIMac::GetOrEmitProtocol(
                                                  const ObjCProtocolDecl *PD) {
  llvm::GlobalVariable *&Entry = Protocols[PD->getIdentifier()];
  
  // Early exit if a defining object has already been generated.
  if (Entry && Entry->hasInitializer())
    return Entry;

  const char *ProtocolName = PD->getNameAsCString();
  
  // Construct method lists.
  std::vector<llvm::Constant*> InstanceMethods, ClassMethods;
  std::vector<llvm::Constant*> OptInstanceMethods, OptClassMethods;
  for (ObjCProtocolDecl::instmeth_iterator i = PD->instmeth_begin(),
       e = PD->instmeth_end(); i != e; ++i) {
    ObjCMethodDecl *MD = *i;
    llvm::Constant *C = GetMethodDescriptionConstant(MD);
    if (MD->getImplementationControl() == ObjCMethodDecl::Optional) {
      OptInstanceMethods.push_back(C);
    } else {
      InstanceMethods.push_back(C);
    }      
  }
  
  for (ObjCProtocolDecl::classmeth_iterator i = PD->classmeth_begin(),
       e = PD->classmeth_end(); i != e; ++i) {
    ObjCMethodDecl *MD = *i;
    llvm::Constant *C = GetMethodDescriptionConstant(MD);
    if (MD->getImplementationControl() == ObjCMethodDecl::Optional) {
      OptClassMethods.push_back(C);
    } else {
      ClassMethods.push_back(C);
    }      
  }
  
  std::vector<llvm::Constant*> Values(10);
  // isa is NULL
  Values[0] = llvm::Constant::getNullValue(ObjCTypes.ObjectPtrTy);
  Values[1] = GetClassName(PD->getIdentifier());
  Values[2] = EmitProtocolList(
                          "\01l_OBJC_$_PROTOCOL_REFS_" + PD->getNameAsString(),
                          PD->protocol_begin(),
                          PD->protocol_end());
  
  Values[3] = EmitMethodList("\01l_OBJC_$_PROTOCOL_INSTANCE_METHODS_"
                             + PD->getNameAsString(),
                             "__DATA, __objc_const",
                             InstanceMethods);
  Values[4] = EmitMethodList("\01l_OBJC_$_PROTOCOL_CLASS_METHODS_" 
                             + PD->getNameAsString(),
                             "__DATA, __objc_const",
                             ClassMethods);
  Values[5] = EmitMethodList("\01l_OBJC_$_PROTOCOL_INSTANCE_METHODS_OPT_"
                             + PD->getNameAsString(),
                             "__DATA, __objc_const",
                             OptInstanceMethods);
  Values[6] = EmitMethodList("\01l_OBJC_$_PROTOCOL_CLASS_METHODS_OPT_" 
                             + PD->getNameAsString(),
                             "__DATA, __objc_const",
                             OptClassMethods);
  Values[7] = EmitPropertyList("\01l_OBJC_$_PROP_LIST_" + PD->getNameAsString(),
                               0, PD, ObjCTypes);
  uint32_t Size = 
    CGM.getTargetData().getTypePaddedSize(ObjCTypes.ProtocolnfABITy);
  Values[8] = llvm::ConstantInt::get(ObjCTypes.IntTy, Size);
  Values[9] = llvm::Constant::getNullValue(ObjCTypes.IntTy);
  llvm::Constant *Init = llvm::ConstantStruct::get(ObjCTypes.ProtocolnfABITy,
                                                   Values);
  
  if (Entry) {
    // Already created, fix the linkage and update the initializer.
    Entry->setLinkage(llvm::GlobalValue::WeakLinkage);
    Entry->setInitializer(Init);
  } else {
    Entry = 
    new llvm::GlobalVariable(ObjCTypes.ProtocolnfABITy, false,
                             llvm::GlobalValue::WeakLinkage,
                             Init, 
                             std::string("\01l_OBJC_PROTOCOL_$_")+ProtocolName,
                             &CGM.getModule());
    Entry->setAlignment(
      CGM.getTargetData().getPrefTypeAlignment(ObjCTypes.ProtocolnfABITy));
    Entry->setSection("__DATA,__datacoal_nt,coalesced");
  }
  Entry->setVisibility(llvm::GlobalValue::HiddenVisibility);
  
  // Use this protocol meta-data to build protocol list table in section
  // __DATA, __objc_protolist
  llvm::Type *ptype = llvm::PointerType::getUnqual(ObjCTypes.ProtocolnfABITy);
  llvm::GlobalVariable *PTGV = new llvm::GlobalVariable(
                                      ptype, false,
                                      llvm::GlobalValue::WeakLinkage,
                                      Entry, 
                                      std::string("\01l_OBJC_LABEL_PROTOCOL_$_")
                                                  +ProtocolName,
                                      &CGM.getModule());
  PTGV->setAlignment(
    CGM.getTargetData().getPrefTypeAlignment(ptype));
  PTGV->setSection("__DATA, __objc_protolist");
  PTGV->setVisibility(llvm::GlobalValue::HiddenVisibility);
  UsedGlobals.push_back(PTGV);
  return Entry;
}

/// EmitProtocolList - Generate protocol list meta-data:
/// @code
/// struct _protocol_list_t {
///   long protocol_count;   // Note, this is 32/64 bit
///   struct _protocol_t[protocol_count];
/// }
/// @endcode
///
llvm::Constant *
CGObjCNonFragileABIMac::EmitProtocolList(const std::string &Name,
                            ObjCProtocolDecl::protocol_iterator begin,
                            ObjCProtocolDecl::protocol_iterator end) {
  std::vector<llvm::Constant*> ProtocolRefs;
  
  for (; begin != end; ++begin)
    ProtocolRefs.push_back(GetProtocolRef(*begin));  // Implemented???
  
  // Just return null for empty protocol lists
  if (ProtocolRefs.empty()) 
    return llvm::Constant::getNullValue(ObjCTypes.ProtocolListnfABIPtrTy);
  
  llvm::GlobalVariable *GV = CGM.getModule().getGlobalVariable(Name, true);
  if (GV)
    return GV;
  // This list is null terminated.
  ProtocolRefs.push_back(llvm::Constant::getNullValue(
                                            ObjCTypes.ProtocolListnfABIPtrTy));
  
  std::vector<llvm::Constant*> Values(2);
  Values[0] = llvm::ConstantInt::get(ObjCTypes.LongTy, ProtocolRefs.size() - 1);
  Values[1] = 
    llvm::ConstantArray::get(llvm::ArrayType::get(ObjCTypes.ProtocolListnfABIPtrTy, 
                                                  ProtocolRefs.size()), 
                                                  ProtocolRefs);
  
  llvm::Constant *Init = llvm::ConstantStruct::get(Values);
  GV = new llvm::GlobalVariable(Init->getType(), false,
                                llvm::GlobalValue::InternalLinkage,
                                Init,
                                Name,
                                &CGM.getModule());
  GV->setSection("__DATA, __objc_const");
  GV->setAlignment(
    CGM.getTargetData().getPrefTypeAlignment(Init->getType()));
  UsedGlobals.push_back(GV);
  return llvm::ConstantExpr::getBitCast(GV, ObjCTypes.ProtocolListnfABIPtrTy);
}

/// GetMethodDescriptionConstant - This routine build following meta-data:
/// struct _objc_method {
///   SEL _cmd;
///   char *method_type;
///   char *_imp;
/// }

llvm::Constant *
CGObjCNonFragileABIMac::GetMethodDescriptionConstant(const ObjCMethodDecl *MD) {
  std::vector<llvm::Constant*> Desc(3);
  Desc[0] = llvm::ConstantExpr::getBitCast(GetMethodVarName(MD->getSelector()),
                                           ObjCTypes.SelectorPtrTy);
  Desc[1] = GetMethodVarType(MD);
  // Protocol methods have no implementation. So, this entry is always NULL.
  Desc[2] = llvm::Constant::getNullValue(ObjCTypes.Int8PtrTy);
  return llvm::ConstantStruct::get(ObjCTypes.MethodTy, Desc);
}

/// EmitObjCValueForIvar - Code Gen for nonfragile ivar reference.
/// This code gen. amounts to generating code for:
/// @code
/// (type *)((char *)base + _OBJC_IVAR_$_.ivar;
/// @encode
/// 
LValue CGObjCNonFragileABIMac::EmitObjCValueForIvar(
                                             CodeGen::CodeGenFunction &CGF,
                                             QualType ObjectTy,
                                             llvm::Value *BaseValue,
                                             const ObjCIvarDecl *Ivar,
                                             const FieldDecl *Field,
                                             unsigned CVRQualifiers) {
  assert(ObjectTy->isObjCInterfaceType() && 
         "CGObjCNonFragileABIMac::EmitObjCValueForIvar");
  ObjCInterfaceDecl *ID = ObjectTy->getAsObjCInterfaceType()->getDecl();
  std::string ExternalName;
  llvm::GlobalVariable *IvarOffsetGV =
    ObjCIvarOffsetVariable(ExternalName, ID, Ivar);
  
  // (char *) BaseValue
  llvm::Value *V =  CGF.Builder.CreateBitCast(BaseValue,
                                              ObjCTypes.Int8PtrTy);
  llvm::Value *Offset = CGF.Builder.CreateLoad(IvarOffsetGV);
  // (char*)BaseValue + Offset_symbol
  V = CGF.Builder.CreateGEP(V, Offset, "add.ptr");
  // (type *)((char*)BaseValue + Offset_symbol)
  const llvm::Type *IvarTy = 
    CGM.getTypes().ConvertType(Ivar->getType());
  llvm::Type *ptrIvarTy = llvm::PointerType::getUnqual(IvarTy);
  V = CGF.Builder.CreateBitCast(V, ptrIvarTy);
  
  if (Ivar->isBitField())
    return CGF.EmitLValueForBitfield(V, const_cast<FieldDecl *>(Field), 
                                     CVRQualifiers);
  
  LValue LV = LValue::MakeAddr(V, 
              Ivar->getType().getCVRQualifiers()|CVRQualifiers);
  LValue::SetObjCIvar(LV, true);
  return LV;
}

llvm::Value *CGObjCNonFragileABIMac::EmitIvarOffset(
                                       CodeGen::CodeGenFunction &CGF,
                                       ObjCInterfaceDecl *Interface,
                                       const ObjCIvarDecl *Ivar) {
  std::string ExternalName;
  llvm::GlobalVariable *IvarOffsetGV =  
    ObjCIvarOffsetVariable(ExternalName, Interface, Ivar);
  
  return CGF.Builder.CreateLoad(IvarOffsetGV, false, "ivar");
}

CodeGen::RValue CGObjCNonFragileABIMac::EmitMessageSend(
                                           CodeGen::CodeGenFunction &CGF,
                                           QualType ResultType,
                                           Selector Sel,
                                           llvm::Value *Receiver,
                                           QualType Arg0Ty,
                                           bool IsSuper,
                                           const CallArgList &CallArgs) {
  // FIXME. Even though IsSuper is passes. This function doese not
  // handle calls to 'super' receivers.
  CodeGenTypes &Types = CGM.getTypes();
  llvm::Value *Arg0 = Receiver;
  if (!IsSuper)
    Arg0 = CGF.Builder.CreateBitCast(Arg0, ObjCTypes.ObjectPtrTy, "tmp");
  
  // Find the message function name.
  // FIXME. This is too much work to get the ABI-specific result type
  // needed to find the message name.
  const CGFunctionInfo &FnInfo = Types.getFunctionInfo(ResultType, 
                                        llvm::SmallVector<QualType, 16>());
  llvm::Constant *Fn;
  std::string Name("\01l_");
  if (CGM.ReturnTypeUsesSret(FnInfo)) {
#if 0
    // unlike what is documented. gcc never generates this API!!
    if (Receiver->getType() == ObjCTypes.ObjectPtrTy) {
      Fn = ObjCTypes.MessageSendIdStretFixupFn;
      // FIXME. Is there a better way of getting these names.
      // They are available in RuntimeFunctions vector pair.
      Name += "objc_msgSendId_stret_fixup";
    }
    else
#endif
    if (IsSuper) {
        Fn = ObjCTypes.MessageSendSuper2StretFixupFn;
        Name += "objc_msgSendSuper2_stret_fixup";
    } 
    else
    {
      Fn = ObjCTypes.MessageSendStretFixupFn;
      Name += "objc_msgSend_stret_fixup";
    }
  }
  else if (ResultType->isFloatingType() &&
           // Selection of frret API only happens in 32bit nonfragile ABI.
           CGM.getTargetData().getTypePaddedSize(ObjCTypes.LongTy) == 4) {
    Fn = ObjCTypes.MessageSendFpretFixupFn;
    Name += "objc_msgSend_fpret_fixup";
  }
  else {
#if 0
// unlike what is documented. gcc never generates this API!!
    if (Receiver->getType() == ObjCTypes.ObjectPtrTy) {
      Fn = ObjCTypes.MessageSendIdFixupFn;
      Name += "objc_msgSendId_fixup";
    }
    else 
#endif
    if (IsSuper) {
        Fn = ObjCTypes.MessageSendSuper2FixupFn;
        Name += "objc_msgSendSuper2_fixup";
    }
    else
    {
      Fn = ObjCTypes.MessageSendFixupFn;
      Name += "objc_msgSend_fixup";
    }
  }
  Name += '_';
  std::string SelName(Sel.getAsString());
  // Replace all ':' in selector name with '_'  ouch!
  for(unsigned i = 0; i < SelName.size(); i++)
    if (SelName[i] == ':')
      SelName[i] = '_';
  Name += SelName;
  llvm::GlobalVariable *GV = CGM.getModule().getGlobalVariable(Name);
  if (!GV) {
    // Build messafe ref table entry.
    std::vector<llvm::Constant*> Values(2);
    Values[0] = Fn;
    Values[1] = GetMethodVarName(Sel);
    llvm::Constant *Init = llvm::ConstantStruct::get(Values);
    GV =  new llvm::GlobalVariable(Init->getType(), false,
                                   llvm::GlobalValue::WeakLinkage,
                                   Init,
                                   Name,
                                   &CGM.getModule());
    GV->setVisibility(llvm::GlobalValue::HiddenVisibility);
    GV->setAlignment(
            CGM.getTargetData().getPrefTypeAlignment(ObjCTypes.MessageRefTy));
    GV->setSection("__DATA, __objc_msgrefs, coalesced");
    UsedGlobals.push_back(GV);
  }
  llvm::Value *Arg1 = CGF.Builder.CreateBitCast(GV, ObjCTypes.MessageRefPtrTy);
  
  CallArgList ActualArgs;
  ActualArgs.push_back(std::make_pair(RValue::get(Arg0), Arg0Ty));
  ActualArgs.push_back(std::make_pair(RValue::get(Arg1), 
                                      ObjCTypes.MessageRefCPtrTy));
  ActualArgs.insert(ActualArgs.end(), CallArgs.begin(), CallArgs.end());
  const CGFunctionInfo &FnInfo1 = Types.getFunctionInfo(ResultType, ActualArgs);
  llvm::Value *Callee = CGF.Builder.CreateStructGEP(Arg1, 0);
  Callee = CGF.Builder.CreateLoad(Callee);
  const llvm::FunctionType *FTy = Types.GetFunctionType(FnInfo1, true);
  Callee = CGF.Builder.CreateBitCast(Callee,
                                     llvm::PointerType::getUnqual(FTy));
  return CGF.EmitCall(FnInfo1, Callee, ActualArgs);
}

/// Generate code for a message send expression in the nonfragile abi.
CodeGen::RValue CGObjCNonFragileABIMac::GenerateMessageSend(
                                               CodeGen::CodeGenFunction &CGF,
                                               QualType ResultType,
                                               Selector Sel,
                                               llvm::Value *Receiver,
                                               bool IsClassMessage,
                                               const CallArgList &CallArgs) {
  return EmitMessageSend(CGF, ResultType, Sel,
                         Receiver, CGF.getContext().getObjCIdType(),
                         false, CallArgs);
}

llvm::Value *CGObjCNonFragileABIMac::EmitClassRef(CGBuilderTy &Builder, 
                                     const ObjCInterfaceDecl *ID,
                                     bool IsSuper) {
  
  llvm::GlobalVariable *&Entry = ClassReferences[ID->getIdentifier()];
  
  if (!Entry) {
    std::string ClassName("\01_OBJC_CLASS_$_" + ID->getNameAsString());
    llvm::GlobalVariable *ClassGV = 
      CGM.getModule().getGlobalVariable(ClassName);
    if (!ClassGV) {
      ClassGV =
        new llvm::GlobalVariable(ObjCTypes.ClassnfABITy, false,
                                 llvm::GlobalValue::ExternalLinkage,
                                 0,
                                 ClassName,
                                 &CGM.getModule());
      UsedGlobals.push_back(ClassGV);
    }
    Entry = 
      new llvm::GlobalVariable(ObjCTypes.ClassnfABIPtrTy, false,
                               llvm::GlobalValue::InternalLinkage,
                               ClassGV, 
                               IsSuper ? "\01L_OBJC_CLASSLIST_SUP_REFS_$_" 
                                       : "\01L_OBJC_CLASSLIST_REFERENCES_$_",
                               &CGM.getModule());
    Entry->setAlignment(
                     CGM.getTargetData().getPrefTypeAlignment(
                                                  ObjCTypes.ClassnfABIPtrTy));

    if (IsSuper)
      Entry->setSection("__OBJC,__objc_superrefs,regular,no_dead_strip");
    else
      Entry->setSection("__OBJC,__objc_classrefs,regular,no_dead_strip");
    UsedGlobals.push_back(Entry);
  }
  
  return Builder.CreateLoad(Entry, false, "tmp");
}

/// EmitMetaClassRef - Return a Value * of the address of _class_t
/// meta-data
///
llvm::Value *CGObjCNonFragileABIMac::EmitMetaClassRef(CGBuilderTy &Builder, 
                                                  const ObjCInterfaceDecl *ID) {
  llvm::GlobalVariable * &Entry = MetaClassReferences[ID->getIdentifier()];
  if (Entry)
    return Builder.CreateLoad(Entry, false, "tmp");
  
  std::string MetaClassName("\01_OBJC_METACLASS_$_" + ID->getNameAsString());
  llvm::GlobalVariable *MetaClassGV = 
    CGM.getModule().getGlobalVariable(MetaClassName);
  if (!MetaClassGV) {
    MetaClassGV =
      new llvm::GlobalVariable(ObjCTypes.ClassnfABITy, false,
                               llvm::GlobalValue::ExternalLinkage,
                               0,
                               MetaClassName,
                               &CGM.getModule());
      UsedGlobals.push_back(MetaClassGV);
  }

  Entry = 
    new llvm::GlobalVariable(ObjCTypes.ClassnfABIPtrTy, false,
                             llvm::GlobalValue::InternalLinkage,
                             MetaClassGV, 
                             "\01L_OBJC_CLASSLIST_SUP_REFS_$_",
                             &CGM.getModule());
  Entry->setAlignment(
                      CGM.getTargetData().getPrefTypeAlignment(
                                                  ObjCTypes.ClassnfABIPtrTy));
    
  Entry->setSection("__OBJC,__objc_superrefs,regular,no_dead_strip");
  UsedGlobals.push_back(Entry);
  
  return Builder.CreateLoad(Entry, false, "tmp");
}

/// GetClass - Return a reference to the class for the given interface
/// decl.
llvm::Value *CGObjCNonFragileABIMac::GetClass(CGBuilderTy &Builder,
                                              const ObjCInterfaceDecl *ID) {
  return EmitClassRef(Builder, ID);
}

/// Generates a message send where the super is the receiver.  This is
/// a message send to self with special delivery semantics indicating
/// which class's method should be called.
CodeGen::RValue
CGObjCNonFragileABIMac::GenerateMessageSendSuper(CodeGen::CodeGenFunction &CGF,
                                    QualType ResultType,
                                    Selector Sel,
                                    const ObjCInterfaceDecl *Class,
                                    llvm::Value *Receiver,
                                    bool IsClassMessage,
                                    const CodeGen::CallArgList &CallArgs) {
  // ...
  // Create and init a super structure; this is a (receiver, class)
  // pair we will pass to objc_msgSendSuper.
  llvm::Value *ObjCSuper =
    CGF.Builder.CreateAlloca(ObjCTypes.SuperTy, 0, "objc_super");
  
  llvm::Value *ReceiverAsObject =
    CGF.Builder.CreateBitCast(Receiver, ObjCTypes.ObjectPtrTy);
  CGF.Builder.CreateStore(ReceiverAsObject,
                          CGF.Builder.CreateStructGEP(ObjCSuper, 0));
  
  // If this is a class message the metaclass is passed as the target.
  llvm::Value *Target = 
    IsClassMessage ? EmitMetaClassRef(CGF.Builder, Class) 
                   : EmitClassRef(CGF.Builder, Class, true);
    
  // FIXME: We shouldn't need to do this cast, rectify the ASTContext
  // and ObjCTypes types.
  const llvm::Type *ClassTy =
    CGM.getTypes().ConvertType(CGF.getContext().getObjCClassType());
  Target = CGF.Builder.CreateBitCast(Target, ClassTy);
  CGF.Builder.CreateStore(Target,
                          CGF.Builder.CreateStructGEP(ObjCSuper, 1));
  
  return EmitMessageSend(CGF, ResultType, Sel,
                         ObjCSuper, ObjCTypes.SuperPtrCTy,
                         true, CallArgs);
}

llvm::Value *CGObjCNonFragileABIMac::EmitSelector(CGBuilderTy &Builder, 
                                                  Selector Sel) {
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
    Entry->setSection("__DATA,__objc_selrefs,literal_pointers,no_dead_strip");
    UsedGlobals.push_back(Entry);
  }
  
  return Builder.CreateLoad(Entry, false, "tmp");
}

/* *** */

CodeGen::CGObjCRuntime *
CodeGen::CreateMacObjCRuntime(CodeGen::CodeGenModule &CGM) {
  return new CGObjCMac(CGM);
}

CodeGen::CGObjCRuntime *
CodeGen::CreateMacNonFragileABIObjCRuntime(CodeGen::CodeGenModule &CGM) {
  return new CGObjCNonFragileABIMac(CGM);
}
