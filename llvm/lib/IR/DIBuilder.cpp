//===--- DIBuilder.cpp - Debug Information Builder ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the DIBuilder.
//
//===----------------------------------------------------------------------===//

#include "llvm/DIBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"

using namespace llvm;
using namespace llvm::dwarf;

static Constant *GetTagConstant(LLVMContext &VMContext, unsigned Tag) {
  assert((Tag & LLVMDebugVersionMask) == 0 &&
         "Tag too large for debug encoding!");
  return ConstantInt::get(Type::getInt32Ty(VMContext), Tag | LLVMDebugVersion);
}

DIBuilder::DIBuilder(Module &m)
  : M(m), VMContext(M.getContext()), TheCU(0), TempEnumTypes(0),
    TempRetainTypes(0), TempSubprograms(0), TempGVs(0), DeclareFn(0),
    ValueFn(0)
{}

/// finalize - Construct any deferred debug info descriptors.
void DIBuilder::finalize() {
  DIArray Enums = getOrCreateArray(AllEnumTypes);
  DIType(TempEnumTypes).replaceAllUsesWith(Enums);

  DIArray RetainTypes = getOrCreateArray(AllRetainTypes);
  DIType(TempRetainTypes).replaceAllUsesWith(RetainTypes);

  DIArray SPs = getOrCreateArray(AllSubprograms);
  DIType(TempSubprograms).replaceAllUsesWith(SPs);
  for (unsigned i = 0, e = SPs.getNumElements(); i != e; ++i) {
    DISubprogram SP(SPs.getElement(i));
    SmallVector<Value *, 4> Variables;
    if (NamedMDNode *NMD = getFnSpecificMDNode(M, SP)) {
      for (unsigned ii = 0, ee = NMD->getNumOperands(); ii != ee; ++ii)
        Variables.push_back(NMD->getOperand(ii));
      NMD->eraseFromParent();
    }
    if (MDNode *Temp = SP.getVariablesNodes()) {
      DIArray AV = getOrCreateArray(Variables);
      DIType(Temp).replaceAllUsesWith(AV);
    }
  }

  DIArray GVs = getOrCreateArray(AllGVs);
  DIType(TempGVs).replaceAllUsesWith(GVs);
}

/// getNonCompileUnitScope - If N is compile unit return NULL otherwise return
/// N.
static MDNode *getNonCompileUnitScope(MDNode *N) {
  if (DIDescriptor(N).isCompileUnit())
    return NULL;
  return N;
}

/// createCompileUnit - A CompileUnit provides an anchor for all debugging
/// information generated during this instance of compilation.
void DIBuilder::createCompileUnit(unsigned Lang, StringRef Filename,
                                  StringRef Directory, StringRef Producer,
                                  bool isOptimized, StringRef Flags,
                                  unsigned RunTimeVer, StringRef SplitName) {
  assert(((Lang <= dwarf::DW_LANG_Python && Lang >= dwarf::DW_LANG_C89) ||
          (Lang <= dwarf::DW_LANG_hi_user && Lang >= dwarf::DW_LANG_lo_user)) &&
         "Invalid Language tag");
  assert(!Filename.empty() &&
         "Unable to create compile unit without filename");
  Value *TElts[] = { GetTagConstant(VMContext, DW_TAG_base_type) };
  TempEnumTypes = MDNode::getTemporary(VMContext, TElts);

  TempRetainTypes = MDNode::getTemporary(VMContext, TElts);

  TempSubprograms = MDNode::getTemporary(VMContext, TElts);

  TempGVs = MDNode::getTemporary(VMContext, TElts);

  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_compile_unit),
    Constant::getNullValue(Type::getInt32Ty(VMContext)),
    ConstantInt::get(Type::getInt32Ty(VMContext), Lang),
    createFile(Filename, Directory),
    MDString::get(VMContext, Producer),
    ConstantInt::get(Type::getInt1Ty(VMContext), isOptimized),
    MDString::get(VMContext, Flags),
    ConstantInt::get(Type::getInt32Ty(VMContext), RunTimeVer),
    TempEnumTypes,
    TempRetainTypes,
    TempSubprograms,
    TempGVs,
    MDString::get(VMContext, SplitName)
  };
  TheCU = DICompileUnit(MDNode::get(VMContext, Elts));

  // Create a named metadata so that it is easier to find cu in a module.
  NamedMDNode *NMD = M.getOrInsertNamedMetadata("llvm.dbg.cu");
  NMD->addOperand(TheCU);
}

/// createFile - Create a file descriptor to hold debugging information
/// for a file.
DIFile DIBuilder::createFile(StringRef Filename, StringRef Directory) {
  assert(!Filename.empty() && "Unable to create file without name");
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_file_type),
    MDString::get(VMContext, Filename),
    MDString::get(VMContext, Directory),
    NULL // TheCU
  };
  return DIFile(MDNode::get(VMContext, Elts));
}

/// createEnumerator - Create a single enumerator value.
DIEnumerator DIBuilder::createEnumerator(StringRef Name, uint64_t Val) {
  assert(!Name.empty() && "Unable to create enumerator without name");
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_enumerator),
    MDString::get(VMContext, Name),
    ConstantInt::get(Type::getInt64Ty(VMContext), Val)
  };
  return DIEnumerator(MDNode::get(VMContext, Elts));
}

/// createNullPtrType - Create C++0x nullptr type.
DIType DIBuilder::createNullPtrType(StringRef Name) {
  assert(!Name.empty() && "Unable to create type without name");
  // nullptr is encoded in DIBasicType format. Line number, filename,
  // ,size, alignment, offset and flags are always empty here.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_unspecified_type),
    NULL, //TheCU,
    MDString::get(VMContext, Name),
    NULL, // Filename
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Line
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Size
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Align
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Offset
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Flags;
    ConstantInt::get(Type::getInt32Ty(VMContext), 0)  // Encoding
  };
  return DIType(MDNode::get(VMContext, Elts));
}

/// createBasicType - Create debugging information entry for a basic
/// type, e.g 'char'.
DIBasicType
DIBuilder::createBasicType(StringRef Name, uint64_t SizeInBits,
                           uint64_t AlignInBits, unsigned Encoding) {
  assert(!Name.empty() && "Unable to create type without name");
  // Basic types are encoded in DIBasicType format. Line number, filename,
  // offset and flags are always empty here.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_base_type),
    NULL, //TheCU,
    MDString::get(VMContext, Name),
    NULL, // Filename
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Line
    ConstantInt::get(Type::getInt64Ty(VMContext), SizeInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Offset
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Flags;
    ConstantInt::get(Type::getInt32Ty(VMContext), Encoding)
  };
  return DIBasicType(MDNode::get(VMContext, Elts));
}

/// createQualifiedType - Create debugging information entry for a qualified
/// type, e.g. 'const int'.
DIDerivedType DIBuilder::createQualifiedType(unsigned Tag, DIType FromTy) {
  // Qualified types are encoded in DIDerivedType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, Tag),
    NULL, //TheCU,
    MDString::get(VMContext, StringRef()), // Empty name.
    NULL, // Filename
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Line
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Size
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Align
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Offset
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Flags
    FromTy
  };
  return DIDerivedType(MDNode::get(VMContext, Elts));
}

/// createPointerType - Create debugging information entry for a pointer.
DIDerivedType
DIBuilder::createPointerType(DIType PointeeTy, uint64_t SizeInBits,
                             uint64_t AlignInBits, StringRef Name) {
  // Pointer types are encoded in DIDerivedType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_pointer_type),
    NULL, //TheCU,
    MDString::get(VMContext, Name),
    NULL, // Filename
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Line
    ConstantInt::get(Type::getInt64Ty(VMContext), SizeInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Offset
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Flags
    PointeeTy
  };
  return DIDerivedType(MDNode::get(VMContext, Elts));
}

DIDerivedType DIBuilder::createMemberPointerType(DIType PointeeTy, DIType Base) {
  // Pointer types are encoded in DIDerivedType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_ptr_to_member_type),
    NULL, //TheCU,
    NULL,
    NULL, // Filename
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Line
    ConstantInt::get(Type::getInt64Ty(VMContext), 0),
    ConstantInt::get(Type::getInt64Ty(VMContext), 0),
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Offset
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Flags
    PointeeTy,
    Base
  };
  return DIDerivedType(MDNode::get(VMContext, Elts));
}

/// createReferenceType - Create debugging information entry for a reference
/// type.
DIDerivedType DIBuilder::createReferenceType(unsigned Tag, DIType RTy) {
  assert(RTy.Verify() && "Unable to create reference type");
  // References are encoded in DIDerivedType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, Tag),
    NULL, // TheCU,
    NULL, // Name
    NULL, // Filename
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Line
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Size
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Align
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Offset
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Flags
    RTy
  };
  return DIDerivedType(MDNode::get(VMContext, Elts));
}

/// createTypedef - Create debugging information entry for a typedef.
DIDerivedType DIBuilder::createTypedef(DIType Ty, StringRef Name, DIFile File,
                                       unsigned LineNo, DIDescriptor Context) {
  // typedefs are encoded in DIDerivedType format.
  assert(Ty.Verify() && "Invalid typedef type!");
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_typedef),
    getNonCompileUnitScope(Context),
    MDString::get(VMContext, Name),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNo),
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Size
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Align
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Offset
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Flags
    Ty
  };
  return DIDerivedType(MDNode::get(VMContext, Elts));
}

/// createFriend - Create debugging information entry for a 'friend'.
DIType DIBuilder::createFriend(DIType Ty, DIType FriendTy) {
  // typedefs are encoded in DIDerivedType format.
  assert(Ty.Verify() && "Invalid type!");
  assert(FriendTy.Verify() && "Invalid friend type!");
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_friend),
    Ty,
    NULL, // Name
    Ty.getFile(),
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Line
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Size
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Align
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Offset
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Flags
    FriendTy
  };
  return DIType(MDNode::get(VMContext, Elts));
}

/// createInheritance - Create debugging information entry to establish
/// inheritance relationship between two types.
DIDerivedType DIBuilder::createInheritance(
    DIType Ty, DIType BaseTy, uint64_t BaseOffset, unsigned Flags) {
  assert(Ty.Verify() && "Unable to create inheritance");
  // TAG_inheritance is encoded in DIDerivedType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_inheritance),
    Ty,
    NULL, // Name
    Ty.getFile(),
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Line
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Size
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Align
    ConstantInt::get(Type::getInt64Ty(VMContext), BaseOffset),
    ConstantInt::get(Type::getInt32Ty(VMContext), Flags),
    BaseTy
  };
  return DIDerivedType(MDNode::get(VMContext, Elts));
}

/// createMemberType - Create debugging information entry for a member.
DIDerivedType DIBuilder::createMemberType(
    DIDescriptor Scope, StringRef Name, DIFile File, unsigned LineNumber,
    uint64_t SizeInBits, uint64_t AlignInBits, uint64_t OffsetInBits,
    unsigned Flags, DIType Ty) {
  // TAG_member is encoded in DIDerivedType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_member),
    getNonCompileUnitScope(Scope),
    MDString::get(VMContext, Name),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNumber),
    ConstantInt::get(Type::getInt64Ty(VMContext), SizeInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), OffsetInBits),
    ConstantInt::get(Type::getInt32Ty(VMContext), Flags),
    Ty
  };
  return DIDerivedType(MDNode::get(VMContext, Elts));
}

/// createStaticMemberType - Create debugging information entry for a
/// C++ static data member.
DIType DIBuilder::createStaticMemberType(DIDescriptor Scope, StringRef Name,
                                         DIFile File, unsigned LineNumber,
                                         DIType Ty, unsigned Flags,
                                         llvm::Value *Val) {
  // TAG_member is encoded in DIDerivedType format.
  Flags |= DIDescriptor::FlagStaticMember;
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_member),
    getNonCompileUnitScope(Scope),
    MDString::get(VMContext, Name),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNumber),
    ConstantInt::get(Type::getInt64Ty(VMContext), 0/*SizeInBits*/),
    ConstantInt::get(Type::getInt64Ty(VMContext), 0/*AlignInBits*/),
    ConstantInt::get(Type::getInt64Ty(VMContext), 0/*OffsetInBits*/),
    ConstantInt::get(Type::getInt32Ty(VMContext), Flags),
    Ty,
    Val
  };
  return DIType(MDNode::get(VMContext, Elts));
}

/// createObjCIVar - Create debugging information entry for Objective-C
/// instance variable.
DIType DIBuilder::createObjCIVar(StringRef Name,
                                 DIFile File, unsigned LineNumber,
                                 uint64_t SizeInBits, uint64_t AlignInBits,
                                 uint64_t OffsetInBits, unsigned Flags,
                                 DIType Ty, StringRef PropertyName,
                                 StringRef GetterName, StringRef SetterName,
                                 unsigned PropertyAttributes) {
  // TAG_member is encoded in DIDerivedType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_member),
    getNonCompileUnitScope(File),
    MDString::get(VMContext, Name),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNumber),
    ConstantInt::get(Type::getInt64Ty(VMContext), SizeInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), OffsetInBits),
    ConstantInt::get(Type::getInt32Ty(VMContext), Flags),
    Ty,
    MDString::get(VMContext, PropertyName),
    MDString::get(VMContext, GetterName),
    MDString::get(VMContext, SetterName),
    ConstantInt::get(Type::getInt32Ty(VMContext), PropertyAttributes)
  };
  return DIType(MDNode::get(VMContext, Elts));
}

/// createObjCIVar - Create debugging information entry for Objective-C
/// instance variable.
DIType DIBuilder::createObjCIVar(StringRef Name,
                                 DIFile File, unsigned LineNumber,
                                 uint64_t SizeInBits, uint64_t AlignInBits,
                                 uint64_t OffsetInBits, unsigned Flags,
                                 DIType Ty, MDNode *PropertyNode) {
  // TAG_member is encoded in DIDerivedType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_member),
    getNonCompileUnitScope(File),
    MDString::get(VMContext, Name),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNumber),
    ConstantInt::get(Type::getInt64Ty(VMContext), SizeInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), OffsetInBits),
    ConstantInt::get(Type::getInt32Ty(VMContext), Flags),
    Ty,
    PropertyNode
  };
  return DIType(MDNode::get(VMContext, Elts));
}

/// createObjCProperty - Create debugging information entry for Objective-C
/// property.
DIObjCProperty DIBuilder::createObjCProperty(StringRef Name,
                                             DIFile File, unsigned LineNumber,
                                             StringRef GetterName,
                                             StringRef SetterName, 
                                             unsigned PropertyAttributes,
                                             DIType Ty) {
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_APPLE_property),
    MDString::get(VMContext, Name),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNumber),
    MDString::get(VMContext, GetterName),
    MDString::get(VMContext, SetterName),
    ConstantInt::get(Type::getInt32Ty(VMContext), PropertyAttributes),
    Ty
  };
  return DIObjCProperty(MDNode::get(VMContext, Elts));
}

/// createTemplateTypeParameter - Create debugging information for template
/// type parameter.
DITemplateTypeParameter
DIBuilder::createTemplateTypeParameter(DIDescriptor Context, StringRef Name,
                                       DIType Ty, MDNode *File, unsigned LineNo,
                                       unsigned ColumnNo) {
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_template_type_parameter),
    getNonCompileUnitScope(Context),
    MDString::get(VMContext, Name),
    Ty,
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNo),
    ConstantInt::get(Type::getInt32Ty(VMContext), ColumnNo)
  };
  return DITemplateTypeParameter(MDNode::get(VMContext, Elts));
}

/// createTemplateValueParameter - Create debugging information for template
/// value parameter.
DITemplateValueParameter
DIBuilder::createTemplateValueParameter(DIDescriptor Context, StringRef Name,
                                        DIType Ty, uint64_t Val,
                                        MDNode *File, unsigned LineNo,
                                        unsigned ColumnNo) {
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_template_value_parameter),
    getNonCompileUnitScope(Context),
    MDString::get(VMContext, Name),
    Ty,
    ConstantInt::get(Type::getInt64Ty(VMContext), Val),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNo),
    ConstantInt::get(Type::getInt32Ty(VMContext), ColumnNo)
  };
  return DITemplateValueParameter(MDNode::get(VMContext, Elts));
}

/// createClassType - Create debugging information entry for a class.
DIType DIBuilder::createClassType(DIDescriptor Context, StringRef Name,
                                  DIFile File, unsigned LineNumber,
                                  uint64_t SizeInBits, uint64_t AlignInBits,
                                  uint64_t OffsetInBits, unsigned Flags,
                                  DIType DerivedFrom, DIArray Elements,
                                  MDNode *VTableHolder,
                                  MDNode *TemplateParams) {
  assert((!Context || Context.Verify()) &&
         "createClassType should be called with a valid Context");
  // TAG_class_type is encoded in DICompositeType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_class_type),
    getNonCompileUnitScope(Context),
    MDString::get(VMContext, Name),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNumber),
    ConstantInt::get(Type::getInt64Ty(VMContext), SizeInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits),
    ConstantInt::get(Type::getInt32Ty(VMContext), OffsetInBits),
    ConstantInt::get(Type::getInt32Ty(VMContext), Flags),
    DerivedFrom,
    Elements,
    ConstantInt::get(Type::getInt32Ty(VMContext), 0),
    VTableHolder,
    TemplateParams
  };
  DIType R(MDNode::get(VMContext, Elts));
  assert(R.Verify() && "createClassType should return a verifiable DIType");
  return R;
}

/// createStructType - Create debugging information entry for a struct.
DICompositeType DIBuilder::createStructType(DIDescriptor Context,
                                            StringRef Name, DIFile File,
                                            unsigned LineNumber,
                                            uint64_t SizeInBits,
                                            uint64_t AlignInBits,
                                            unsigned Flags, DIType DerivedFrom,
                                            DIArray Elements,
                                            unsigned RunTimeLang,
                                            MDNode *VTableHolder) {
 // TAG_structure_type is encoded in DICompositeType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_structure_type),
    getNonCompileUnitScope(Context),
    MDString::get(VMContext, Name),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNumber),
    ConstantInt::get(Type::getInt64Ty(VMContext), SizeInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits),
    ConstantInt::get(Type::getInt32Ty(VMContext), 0),
    ConstantInt::get(Type::getInt32Ty(VMContext), Flags),
    DerivedFrom,
    Elements,
    ConstantInt::get(Type::getInt32Ty(VMContext), RunTimeLang),
    VTableHolder,
    NULL,
  };
  DICompositeType R(MDNode::get(VMContext, Elts));
  assert(R.Verify() && "createStructType should return a verifiable DIType");
  return R;
}

/// createUnionType - Create debugging information entry for an union.
DICompositeType DIBuilder::createUnionType(
    DIDescriptor Scope, StringRef Name, DIFile File, unsigned LineNumber,
    uint64_t SizeInBits, uint64_t AlignInBits, unsigned Flags, DIArray Elements,
    unsigned RunTimeLang) {
  // TAG_union_type is encoded in DICompositeType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_union_type),
    getNonCompileUnitScope(Scope),
    MDString::get(VMContext, Name),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNumber),
    ConstantInt::get(Type::getInt64Ty(VMContext), SizeInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), 0),
    ConstantInt::get(Type::getInt32Ty(VMContext), Flags),
    NULL,
    Elements,
    ConstantInt::get(Type::getInt32Ty(VMContext), RunTimeLang),
    Constant::getNullValue(Type::getInt32Ty(VMContext))
  };
  return DICompositeType(MDNode::get(VMContext, Elts));
}

/// createSubroutineType - Create subroutine type.
DICompositeType
DIBuilder::createSubroutineType(DIFile File, DIArray ParameterTypes) {
  // TAG_subroutine_type is encoded in DICompositeType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_subroutine_type),
    Constant::getNullValue(Type::getInt32Ty(VMContext)),
    MDString::get(VMContext, ""),
    Constant::getNullValue(Type::getInt32Ty(VMContext)),
    ConstantInt::get(Type::getInt32Ty(VMContext), 0),
    ConstantInt::get(Type::getInt64Ty(VMContext), 0),
    ConstantInt::get(Type::getInt64Ty(VMContext), 0),
    ConstantInt::get(Type::getInt64Ty(VMContext), 0),
    ConstantInt::get(Type::getInt32Ty(VMContext), 0),
    NULL,
    ParameterTypes,
    ConstantInt::get(Type::getInt32Ty(VMContext), 0),
    Constant::getNullValue(Type::getInt32Ty(VMContext))
  };
  return DICompositeType(MDNode::get(VMContext, Elts));
}

/// createEnumerationType - Create debugging information entry for an
/// enumeration.
DICompositeType DIBuilder::createEnumerationType(
    DIDescriptor Scope, StringRef Name, DIFile File, unsigned LineNumber,
    uint64_t SizeInBits, uint64_t AlignInBits, DIArray Elements,
    DIType ClassType) {
  // TAG_enumeration_type is encoded in DICompositeType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_enumeration_type),
    getNonCompileUnitScope(Scope),
    MDString::get(VMContext, Name),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNumber),
    ConstantInt::get(Type::getInt64Ty(VMContext), SizeInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits),
    ConstantInt::get(Type::getInt32Ty(VMContext), 0),
    ConstantInt::get(Type::getInt32Ty(VMContext), 0),
    ClassType,
    Elements,
    ConstantInt::get(Type::getInt32Ty(VMContext), 0),
    Constant::getNullValue(Type::getInt32Ty(VMContext))
  };
  MDNode *Node = MDNode::get(VMContext, Elts);
  AllEnumTypes.push_back(Node);
  return DICompositeType(Node);
}

/// createArrayType - Create debugging information entry for an array.
DICompositeType DIBuilder::createArrayType(uint64_t Size, uint64_t AlignInBits,
                                           DIType Ty, DIArray Subscripts) {
  // TAG_array_type is encoded in DICompositeType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_array_type),
    NULL, //TheCU,
    MDString::get(VMContext, ""),
    NULL, //TheCU,
    ConstantInt::get(Type::getInt32Ty(VMContext), 0),
    ConstantInt::get(Type::getInt64Ty(VMContext), Size),
    ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits),
    ConstantInt::get(Type::getInt32Ty(VMContext), 0),
    ConstantInt::get(Type::getInt32Ty(VMContext), 0),
    Ty,
    Subscripts,
    ConstantInt::get(Type::getInt32Ty(VMContext), 0),
    Constant::getNullValue(Type::getInt32Ty(VMContext))
  };
  return DICompositeType(MDNode::get(VMContext, Elts));
}

/// createVectorType - Create debugging information entry for a vector.
DIType DIBuilder::createVectorType(uint64_t Size, uint64_t AlignInBits,
                                   DIType Ty, DIArray Subscripts) {

  // A vector is an array type with the FlagVector flag applied.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_array_type),
    NULL, //TheCU,
    MDString::get(VMContext, ""),
    NULL, //TheCU,
    ConstantInt::get(Type::getInt32Ty(VMContext), 0),
    ConstantInt::get(Type::getInt64Ty(VMContext), Size),
    ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits),
    ConstantInt::get(Type::getInt32Ty(VMContext), 0),
    ConstantInt::get(Type::getInt32Ty(VMContext), DIType::FlagVector),
    Ty,
    Subscripts,
    ConstantInt::get(Type::getInt32Ty(VMContext), 0),
    Constant::getNullValue(Type::getInt32Ty(VMContext))
  };
  return DIType(MDNode::get(VMContext, Elts));
}

/// createArtificialType - Create a new DIType with "artificial" flag set.
DIType DIBuilder::createArtificialType(DIType Ty) {
  if (Ty.isArtificial())
    return Ty;

  SmallVector<Value *, 9> Elts;
  MDNode *N = Ty;
  assert (N && "Unexpected input DIType!");
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    if (Value *V = N->getOperand(i))
      Elts.push_back(V);
    else
      Elts.push_back(Constant::getNullValue(Type::getInt32Ty(VMContext)));
  }

  unsigned CurFlags = Ty.getFlags();
  CurFlags = CurFlags | DIType::FlagArtificial;

  // Flags are stored at this slot.
  Elts[8] =  ConstantInt::get(Type::getInt32Ty(VMContext), CurFlags);

  return DIType(MDNode::get(VMContext, Elts));
}

/// createObjectPointerType - Create a new type with both the object pointer
/// and artificial flags set.
DIType DIBuilder::createObjectPointerType(DIType Ty) {
  if (Ty.isObjectPointer())
    return Ty;

  SmallVector<Value *, 9> Elts;
  MDNode *N = Ty;
  assert (N && "Unexpected input DIType!");
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    if (Value *V = N->getOperand(i))
      Elts.push_back(V);
    else
      Elts.push_back(Constant::getNullValue(Type::getInt32Ty(VMContext)));
  }

  unsigned CurFlags = Ty.getFlags();
  CurFlags = CurFlags | (DIType::FlagObjectPointer | DIType::FlagArtificial);

  // Flags are stored at this slot.
  Elts[8] = ConstantInt::get(Type::getInt32Ty(VMContext), CurFlags);

  return DIType(MDNode::get(VMContext, Elts));
}

/// retainType - Retain DIType in a module even if it is not referenced
/// through debug info anchors.
void DIBuilder::retainType(DIType T) {
  AllRetainTypes.push_back(T);
}

/// createUnspecifiedParameter - Create unspeicified type descriptor
/// for the subroutine type.
DIDescriptor DIBuilder::createUnspecifiedParameter() {
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_unspecified_parameters)
  };
  return DIDescriptor(MDNode::get(VMContext, Elts));
}

/// createTemporaryType - Create a temporary forward-declared type.
DIType DIBuilder::createTemporaryType() {
  // Give the temporary MDNode a tag. It doesn't matter what tag we
  // use here as long as DIType accepts it.
  Value *Elts[] = { GetTagConstant(VMContext, DW_TAG_base_type) };
  MDNode *Node = MDNode::getTemporary(VMContext, Elts);
  return DIType(Node);
}

/// createTemporaryType - Create a temporary forward-declared type.
DIType DIBuilder::createTemporaryType(DIFile F) {
  // Give the temporary MDNode a tag. It doesn't matter what tag we
  // use here as long as DIType accepts it.
  Value *Elts[] = {
    GetTagConstant(VMContext, DW_TAG_base_type),
    TheCU,
    NULL,
    F
  };
  MDNode *Node = MDNode::getTemporary(VMContext, Elts);
  return DIType(Node);
}

/// createForwardDecl - Create a temporary forward-declared type that
/// can be RAUW'd if the full type is seen.
DIType DIBuilder::createForwardDecl(unsigned Tag, StringRef Name,
                                    DIDescriptor Scope, DIFile F,
                                    unsigned Line, unsigned RuntimeLang,
                                    uint64_t SizeInBits,
                                    uint64_t AlignInBits) {
  // Create a temporary MDNode.
  Value *Elts[] = {
    GetTagConstant(VMContext, Tag),
    getNonCompileUnitScope(Scope),
    MDString::get(VMContext, Name),
    F,
    ConstantInt::get(Type::getInt32Ty(VMContext), Line),
    ConstantInt::get(Type::getInt64Ty(VMContext), SizeInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits),
    ConstantInt::get(Type::getInt32Ty(VMContext), 0),
    ConstantInt::get(Type::getInt32Ty(VMContext),
                     DIDescriptor::FlagFwdDecl),
    NULL,
    DIArray(),
    ConstantInt::get(Type::getInt32Ty(VMContext), RuntimeLang)
  };
  MDNode *Node = MDNode::getTemporary(VMContext, Elts);
  assert(DIType(Node).Verify() &&
         "createForwardDecl result should be verifiable");
  return DIType(Node);
}

/// getOrCreateArray - Get a DIArray, create one if required.
DIArray DIBuilder::getOrCreateArray(ArrayRef<Value *> Elements) {
  if (Elements.empty()) {
    Value *Null = Constant::getNullValue(Type::getInt32Ty(VMContext));
    return DIArray(MDNode::get(VMContext, Null));
  }
  return DIArray(MDNode::get(VMContext, Elements));
}

/// getOrCreateSubrange - Create a descriptor for a value range.  This
/// implicitly uniques the values returned.
DISubrange DIBuilder::getOrCreateSubrange(int64_t Lo, int64_t Count) {
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_subrange_type),
    ConstantInt::get(Type::getInt64Ty(VMContext), Lo),
    ConstantInt::get(Type::getInt64Ty(VMContext), Count)
  };

  return DISubrange(MDNode::get(VMContext, Elts));
}

/// createGlobalVariable - Create a new descriptor for the specified global.
DIGlobalVariable DIBuilder::
createGlobalVariable(StringRef Name, DIFile F, unsigned LineNumber,
                     DIType Ty, bool isLocalToUnit, Value *Val) {
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_variable),
    Constant::getNullValue(Type::getInt32Ty(VMContext)),
    NULL, // TheCU,
    MDString::get(VMContext, Name),
    MDString::get(VMContext, Name),
    MDString::get(VMContext, Name),
    F,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNumber),
    Ty,
    ConstantInt::get(Type::getInt32Ty(VMContext), isLocalToUnit),
    ConstantInt::get(Type::getInt32Ty(VMContext), 1), /* isDefinition*/
    Val,
    DIDescriptor()
  };
  MDNode *Node = MDNode::get(VMContext, Elts);
  AllGVs.push_back(Node);
  return DIGlobalVariable(Node);
}

/// createStaticVariable - Create a new descriptor for the specified static
/// variable.
DIGlobalVariable DIBuilder::
createStaticVariable(DIDescriptor Context, StringRef Name,
                     StringRef LinkageName, DIFile F, unsigned LineNumber,
                     DIType Ty, bool isLocalToUnit, Value *Val, MDNode *Decl) {
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_variable),
    Constant::getNullValue(Type::getInt32Ty(VMContext)),
    getNonCompileUnitScope(Context),
    MDString::get(VMContext, Name),
    MDString::get(VMContext, Name),
    MDString::get(VMContext, LinkageName),
    F,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNumber),
    Ty,
    ConstantInt::get(Type::getInt32Ty(VMContext), isLocalToUnit),
    ConstantInt::get(Type::getInt32Ty(VMContext), 1), /* isDefinition*/
    Val,
    DIDescriptor(Decl)
  };
  MDNode *Node = MDNode::get(VMContext, Elts);
  AllGVs.push_back(Node);
  return DIGlobalVariable(Node);
}

/// createVariable - Create a new descriptor for the specified variable.
DIVariable DIBuilder::createLocalVariable(unsigned Tag, DIDescriptor Scope,
                                          StringRef Name, DIFile File,
                                          unsigned LineNo, DIType Ty,
                                          bool AlwaysPreserve, unsigned Flags,
                                          unsigned ArgNo) {
  DIDescriptor Context(getNonCompileUnitScope(Scope));
  assert((!Context || Context.Verify()) &&
         "createLocalVariable should be called with a valid Context");
  assert(Ty.Verify() &&
         "createLocalVariable should be called with a valid type");
  Value *Elts[] = {
    GetTagConstant(VMContext, Tag),
    getNonCompileUnitScope(Scope),
    MDString::get(VMContext, Name),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), (LineNo | (ArgNo << 24))),
    Ty,
    ConstantInt::get(Type::getInt32Ty(VMContext), Flags),
    Constant::getNullValue(Type::getInt32Ty(VMContext))
  };
  MDNode *Node = MDNode::get(VMContext, Elts);
  if (AlwaysPreserve) {
    // The optimizer may remove local variable. If there is an interest
    // to preserve variable info in such situation then stash it in a
    // named mdnode.
    DISubprogram Fn(getDISubprogram(Scope));
    NamedMDNode *FnLocals = getOrInsertFnSpecificMDNode(M, Fn);
    FnLocals->addOperand(Node);
  }
  assert(DIVariable(Node).Verify() &&
         "createLocalVariable should return a verifiable DIVariable");
  return DIVariable(Node);
}

/// createComplexVariable - Create a new descriptor for the specified variable
/// which has a complex address expression for its address.
DIVariable DIBuilder::createComplexVariable(unsigned Tag, DIDescriptor Scope,
                                            StringRef Name, DIFile F,
                                            unsigned LineNo,
                                            DIType Ty, ArrayRef<Value *> Addr,
                                            unsigned ArgNo) {
  SmallVector<Value *, 15> Elts;
  Elts.push_back(GetTagConstant(VMContext, Tag));
  Elts.push_back(getNonCompileUnitScope(Scope)),
  Elts.push_back(MDString::get(VMContext, Name));
  Elts.push_back(F);
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext),
                                  (LineNo | (ArgNo << 24))));
  Elts.push_back(Ty);
  Elts.push_back(Constant::getNullValue(Type::getInt32Ty(VMContext)));
  Elts.push_back(Constant::getNullValue(Type::getInt32Ty(VMContext)));
  Elts.append(Addr.begin(), Addr.end());

  return DIVariable(MDNode::get(VMContext, Elts));
}

/// createFunction - Create a new descriptor for the specified function.
DISubprogram DIBuilder::createFunction(DIDescriptor Context,
                                       StringRef Name,
                                       StringRef LinkageName,
                                       DIFile File, unsigned LineNo,
                                       DIType Ty,
                                       bool isLocalToUnit, bool isDefinition,
                                       unsigned ScopeLine,
                                       unsigned Flags, bool isOptimized,
                                       Function *Fn,
                                       MDNode *TParams,
                                       MDNode *Decl) {
  Value *TElts[] = { GetTagConstant(VMContext, DW_TAG_base_type) };
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_subprogram),
    Constant::getNullValue(Type::getInt32Ty(VMContext)),
    getNonCompileUnitScope(Context),
    MDString::get(VMContext, Name),
    MDString::get(VMContext, Name),
    MDString::get(VMContext, LinkageName),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNo),
    Ty,
    ConstantInt::get(Type::getInt1Ty(VMContext), isLocalToUnit),
    ConstantInt::get(Type::getInt1Ty(VMContext), isDefinition),
    ConstantInt::get(Type::getInt32Ty(VMContext), 0),
    ConstantInt::get(Type::getInt32Ty(VMContext), 0),
    NULL,
    ConstantInt::get(Type::getInt32Ty(VMContext), Flags),
    ConstantInt::get(Type::getInt1Ty(VMContext), isOptimized),
    Fn,
    TParams,
    Decl,
    MDNode::getTemporary(VMContext, TElts),
    ConstantInt::get(Type::getInt32Ty(VMContext), ScopeLine)
  };
  MDNode *Node = MDNode::get(VMContext, Elts);

  // Create a named metadata so that we do not lose this mdnode.
  if (isDefinition)
    AllSubprograms.push_back(Node);
  return DISubprogram(Node);
}

/// createMethod - Create a new descriptor for the specified C++ method.
DISubprogram DIBuilder::createMethod(DIDescriptor Context,
                                     StringRef Name,
                                     StringRef LinkageName,
                                     DIFile F,
                                     unsigned LineNo, DIType Ty,
                                     bool isLocalToUnit,
                                     bool isDefinition,
                                     unsigned VK, unsigned VIndex,
                                     MDNode *VTableHolder,
                                     unsigned Flags,
                                     bool isOptimized,
                                     Function *Fn,
                                     MDNode *TParam) {
  Value *TElts[] = { GetTagConstant(VMContext, DW_TAG_base_type) };
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_subprogram),
    Constant::getNullValue(Type::getInt32Ty(VMContext)),
    getNonCompileUnitScope(Context),
    MDString::get(VMContext, Name),
    MDString::get(VMContext, Name),
    MDString::get(VMContext, LinkageName),
    F,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNo),
    Ty,
    ConstantInt::get(Type::getInt1Ty(VMContext), isLocalToUnit),
    ConstantInt::get(Type::getInt1Ty(VMContext), isDefinition),
    ConstantInt::get(Type::getInt32Ty(VMContext), (unsigned)VK),
    ConstantInt::get(Type::getInt32Ty(VMContext), VIndex),
    VTableHolder,
    ConstantInt::get(Type::getInt32Ty(VMContext), Flags),
    ConstantInt::get(Type::getInt1Ty(VMContext), isOptimized),
    Fn,
    TParam,
    Constant::getNullValue(Type::getInt32Ty(VMContext)),
    MDNode::getTemporary(VMContext, TElts),
    // FIXME: Do we want to use different scope/lines?
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNo)
  };
  MDNode *Node = MDNode::get(VMContext, Elts);
  if (isDefinition)
    AllSubprograms.push_back(Node);
  return DISubprogram(Node);
}

/// createNameSpace - This creates new descriptor for a namespace
/// with the specified parent scope.
DINameSpace DIBuilder::createNameSpace(DIDescriptor Scope, StringRef Name,
                                       DIFile File, unsigned LineNo) {
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_namespace),
    getNonCompileUnitScope(Scope),
    MDString::get(VMContext, Name),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNo)
  };
  DINameSpace R(MDNode::get(VMContext, Elts));
  assert(R.Verify() &&
         "createNameSpace should return a verifiable DINameSpace");
  return R;
}

/// createLexicalBlockFile - This creates a new MDNode that encapsulates
/// an existing scope with a new filename.
DILexicalBlockFile DIBuilder::createLexicalBlockFile(DIDescriptor Scope,
                                                     DIFile File) {
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_lexical_block),
    Scope,
    File
  };
  DILexicalBlockFile R(MDNode::get(VMContext, Elts));
  assert(
      R.Verify() &&
      "createLexicalBlockFile should return a verifiable DILexicalBlockFile");
  return R;
}

DILexicalBlock DIBuilder::createLexicalBlock(DIDescriptor Scope, DIFile File,
                                             unsigned Line, unsigned Col) {
  // Defeat MDNode uniqing for lexical blocks by using unique id.
  static unsigned int unique_id = 0;
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_lexical_block),
    getNonCompileUnitScope(Scope),
    ConstantInt::get(Type::getInt32Ty(VMContext), Line),
    ConstantInt::get(Type::getInt32Ty(VMContext), Col),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), unique_id++)
  };
  DILexicalBlock R(MDNode::get(VMContext, Elts));
  assert(R.Verify() &&
         "createLexicalBlock should return a verifiable DILexicalBlock");
  return R;
}

/// insertDeclare - Insert a new llvm.dbg.declare intrinsic call.
Instruction *DIBuilder::insertDeclare(Value *Storage, DIVariable VarInfo,
                                      Instruction *InsertBefore) {
  assert(Storage && "no storage passed to dbg.declare");
  assert(VarInfo.Verify() && "empty DIVariable passed to dbg.declare");
  if (!DeclareFn)
    DeclareFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_declare);

  Value *Args[] = { MDNode::get(Storage->getContext(), Storage), VarInfo };
  return CallInst::Create(DeclareFn, Args, "", InsertBefore);
}

/// insertDeclare - Insert a new llvm.dbg.declare intrinsic call.
Instruction *DIBuilder::insertDeclare(Value *Storage, DIVariable VarInfo,
                                      BasicBlock *InsertAtEnd) {
  assert(Storage && "no storage passed to dbg.declare");
  assert(VarInfo.Verify() && "invalid DIVariable passed to dbg.declare");
  if (!DeclareFn)
    DeclareFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_declare);

  Value *Args[] = { MDNode::get(Storage->getContext(), Storage), VarInfo };

  // If this block already has a terminator then insert this intrinsic
  // before the terminator.
  if (TerminatorInst *T = InsertAtEnd->getTerminator())
    return CallInst::Create(DeclareFn, Args, "", T);
  else
    return CallInst::Create(DeclareFn, Args, "", InsertAtEnd);
}

/// insertDbgValueIntrinsic - Insert a new llvm.dbg.value intrinsic call.
Instruction *DIBuilder::insertDbgValueIntrinsic(Value *V, uint64_t Offset,
                                                DIVariable VarInfo,
                                                Instruction *InsertBefore) {
  assert(V && "no value passed to dbg.value");
  assert(VarInfo.Verify() && "invalid DIVariable passed to dbg.value");
  if (!ValueFn)
    ValueFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_value);

  Value *Args[] = { MDNode::get(V->getContext(), V),
                    ConstantInt::get(Type::getInt64Ty(V->getContext()), Offset),
                    VarInfo };
  return CallInst::Create(ValueFn, Args, "", InsertBefore);
}

/// insertDbgValueIntrinsic - Insert a new llvm.dbg.value intrinsic call.
Instruction *DIBuilder::insertDbgValueIntrinsic(Value *V, uint64_t Offset,
                                                DIVariable VarInfo,
                                                BasicBlock *InsertAtEnd) {
  assert(V && "no value passed to dbg.value");
  assert(VarInfo.Verify() && "invalid DIVariable passed to dbg.value");
  if (!ValueFn)
    ValueFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_value);

  Value *Args[] = { MDNode::get(V->getContext(), V),
                    ConstantInt::get(Type::getInt64Ty(V->getContext()), Offset),
                    VarInfo };
  return CallInst::Create(ValueFn, Args, "", InsertAtEnd);
}
