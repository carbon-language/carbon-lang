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

#include "llvm/Analysis/DIBuilder.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Constants.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Dwarf.h"

using namespace llvm;
using namespace llvm::dwarf;

static Constant *GetTagConstant(LLVMContext &VMContext, unsigned Tag) {
  assert((Tag & LLVMDebugVersionMask) == 0 &&
         "Tag too large for debug encoding!");
  return ConstantInt::get(Type::getInt32Ty(VMContext), Tag | LLVMDebugVersion);
}
DIBuilder::DIBuilder(Module &m)
  : M(m), VMContext(M.getContext()), TheCU(0) {}

/// CreateCompileUnit - A CompileUnit provides an anchor for all debugging
/// information generated during this instance of compilation.
void DIBuilder::CreateCompileUnit(unsigned Lang, StringRef Filename, 
                                  StringRef Directory, StringRef Producer, 
                                  bool isOptimized, StringRef Flags, 
                                  unsigned RunTimeVer) {
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_compile_unit),
    llvm::Constant::getNullValue(Type::getInt32Ty(VMContext)),
    ConstantInt::get(Type::getInt32Ty(VMContext), Lang),
    MDString::get(VMContext, Filename),
    MDString::get(VMContext, Directory),
    MDString::get(VMContext, Producer),
    // Deprecate isMain field.
    ConstantInt::get(Type::getInt1Ty(VMContext), true), // isMain
    ConstantInt::get(Type::getInt1Ty(VMContext), isOptimized),
    MDString::get(VMContext, Flags),
    ConstantInt::get(Type::getInt32Ty(VMContext), RunTimeVer)
  };
  TheCU = DICompileUnit(MDNode::get(VMContext, &Elts[0], array_lengthof(Elts)));
}

/// CreateFile - Create a file descriptor to hold debugging information
/// for a file.
DIFile DIBuilder::CreateFile(StringRef Filename, StringRef Directory) {
  assert(TheCU && "Unable to create DW_TAG_file_type without CompileUnit");
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_file_type),
    MDString::get(VMContext, Filename),
    MDString::get(VMContext, Directory),
    TheCU
  };
  return DIFile(MDNode::get(VMContext, &Elts[0], array_lengthof(Elts)));
}

/// CreateEnumerator - Create a single enumerator value.
DIEnumerator DIBuilder::CreateEnumerator(StringRef Name, uint64_t Val) {
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_enumerator),
    MDString::get(VMContext, Name),
    ConstantInt::get(Type::getInt64Ty(VMContext), Val)
  };
  return DIEnumerator(MDNode::get(VMContext, &Elts[0], array_lengthof(Elts)));
}

/// CreateBasicType - Create debugging information entry for a basic 
/// type, e.g 'char'.
DIType DIBuilder::CreateBasicType(StringRef Name, uint64_t SizeInBits, 
                                  uint64_t AlignInBits,
                                  unsigned Encoding) {
  // Basic types are encoded in DIBasicType format. Line number, filename,
  // offset and flags are always empty here.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_base_type),
    TheCU,
    MDString::get(VMContext, Name),
    NULL, // Filename
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Line
    ConstantInt::get(Type::getInt64Ty(VMContext), SizeInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Offset
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Flags;
    ConstantInt::get(Type::getInt32Ty(VMContext), Encoding)
  };
  return DIType(MDNode::get(VMContext, &Elts[0], array_lengthof(Elts)));
}

/// CreateQaulifiedType - Create debugging information entry for a qualified
/// type, e.g. 'const int'.
DIType DIBuilder::CreateQualifiedType(unsigned Tag, DIType FromTy) {
  // Qualified types are encoded in DIDerivedType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, Tag),
    TheCU,
    MDString::get(VMContext, StringRef()), // Empty name.
    NULL, // Filename
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Line
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Size
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Align
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Offset
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Flags
    FromTy
  };
  return DIType(MDNode::get(VMContext, &Elts[0], array_lengthof(Elts)));
}

/// CreatePointerType - Create debugging information entry for a pointer.
DIType DIBuilder::CreatePointerType(DIType PointeeTy, uint64_t SizeInBits,
                                    uint64_t AlignInBits, StringRef Name) {
  // Pointer types are encoded in DIDerivedType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_pointer_type),
    TheCU,
    MDString::get(VMContext, Name),
    NULL, // Filename
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Line
    ConstantInt::get(Type::getInt64Ty(VMContext), SizeInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Offset
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Flags
    PointeeTy
  };
  return DIType(MDNode::get(VMContext, &Elts[0], array_lengthof(Elts)));
}

/// CreateReferenceType - Create debugging information entry for a reference.
DIType DIBuilder::CreateReferenceType(DIType RTy) {
  // References are encoded in DIDerivedType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_reference_type),
    TheCU,
    NULL, // Name
    NULL, // Filename
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Line
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Size
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Align
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Offset
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Flags
    RTy
  };
  return DIType(MDNode::get(VMContext, &Elts[0], array_lengthof(Elts)));
}

/// CreateTypedef - Create debugging information entry for a typedef.
DIType DIBuilder::CreateTypedef(DIType Ty, StringRef Name, DIFile File,
                                unsigned LineNo) {
  // typedefs are encoded in DIDerivedType format.
  assert(Ty.Verify() && "Invalid typedef type!");
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_typedef),
    Ty.getContext(),
    MDString::get(VMContext, Name),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNo),
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Size
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Align
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Offset
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Flags
    Ty
  };
  return DIType(MDNode::get(VMContext, &Elts[0], array_lengthof(Elts)));
}

/// CreateFriend - Create debugging information entry for a 'friend'.
DIType DIBuilder::CreateFriend(DIType Ty, DIType FriendTy) {
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
  return DIType(MDNode::get(VMContext, &Elts[0], array_lengthof(Elts)));
}

/// CreateInheritance - Create debugging information entry to establish
/// inheritnace relationship between two types.
DIType DIBuilder::CreateInheritance(DIType Ty, DIType BaseTy, 
                                    uint64_t BaseOffset, unsigned Flags) {
  // TAG_inheritance is encoded in DIDerivedType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_inheritance),
    Ty,
    NULL, // Name
    NULL, // File
    ConstantInt::get(Type::getInt32Ty(VMContext), 0), // Line
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Size
    ConstantInt::get(Type::getInt64Ty(VMContext), 0), // Align
    ConstantInt::get(Type::getInt64Ty(VMContext), BaseOffset),
    ConstantInt::get(Type::getInt32Ty(VMContext), Flags),
    BaseTy
  };
  return DIType(MDNode::get(VMContext, &Elts[0], array_lengthof(Elts)));
}

/// CreateMemberType - Create debugging information entry for a member.
DIType DIBuilder::CreateMemberType(DIDescriptor Context, StringRef Name, 
                                   DIFile F, unsigned LineNumber, 
                                   uint64_t SizeInBits, uint64_t AlignInBits,
                                   uint64_t OffsetInBits, unsigned Flags, 
                                   DIType Ty) {
  // TAG_member is encoded in DIDerivedType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_member),
    Context,
    MDString::get(VMContext, Name),
    F,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNumber),
    ConstantInt::get(Type::getInt64Ty(VMContext), SizeInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), OffsetInBits),
    ConstantInt::get(Type::getInt32Ty(VMContext), Flags),
    Ty
  };
  return DIType(MDNode::get(VMContext, &Elts[0], array_lengthof(Elts)));
}

/// CreateArtificialType - Create a new DIType with "artificial" flag set.
DIType DIBuilder::CreateArtificialType(DIType Ty) {
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

  return DIType(MDNode::get(VMContext, Elts.data(), Elts.size()));
}
