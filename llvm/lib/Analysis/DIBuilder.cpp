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
  SmallVector<Value *, 16> Elts;
  Elts.push_back(GetTagConstant(VMContext, dwarf::DW_TAG_compile_unit));
  Elts.push_back(llvm::Constant::getNullValue(Type::getInt32Ty(VMContext)));
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), Lang));
  Elts.push_back(MDString::get(VMContext, Filename));
  Elts.push_back(MDString::get(VMContext, Directory));
  Elts.push_back(MDString::get(VMContext, Producer));
  // Deprecate isMain field.
  Elts.push_back(ConstantInt::get(Type::getInt1Ty(VMContext), true)); // isMain
  Elts.push_back(ConstantInt::get(Type::getInt1Ty(VMContext), isOptimized));
  Elts.push_back(MDString::get(VMContext, Flags));
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), RunTimeVer));

  TheCU = DICompileUnit(MDNode::get(VMContext, Elts.data(), Elts.size()));
}

/// CreateFile - Create a file descriptor to hold debugging information
/// for a file.
DIFile DIBuilder::CreateFile(StringRef Filename, StringRef Directory) {
  assert (TheCU && "Unable to create DW_TAG_file_type without CompileUnit");
  SmallVector<Value *, 4> Elts;
  Elts.push_back(GetTagConstant(VMContext, dwarf::DW_TAG_file_type));
  Elts.push_back(MDString::get(VMContext, Filename));
  Elts.push_back(MDString::get(VMContext, Directory));
  Elts.push_back(TheCU);
  return DIFile(MDNode::get(VMContext, Elts.data(), Elts.size()));
}

/// CreateEnumerator - Create a single enumerator value.
DIEnumerator DIBuilder::CreateEnumerator(StringRef Name, uint64_t Val) {
  SmallVector<Value *, 4> Elts;
  Elts.push_back(GetTagConstant(VMContext, dwarf::DW_TAG_enumerator));
  Elts.push_back(MDString::get(VMContext, Name));
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), Val));
  return DIEnumerator(MDNode::get(VMContext, Elts.data(), Elts.size()));
}

/// CreateBasicType - Create debugging information entry for a basic 
/// type, e.g 'char'.
DIType DIBuilder::CreateBasicType(StringRef Name, uint64_t SizeInBits, 
                                  uint64_t AlignInBits,
                                  unsigned Encoding) {
  // Basic types are encoded in DIBasicType format. Line number, filename,
  // offset and flags are always empty here.
  SmallVector<Value *, 12> Elts;
  Elts.push_back(GetTagConstant(VMContext, dwarf::DW_TAG_base_type));
  Elts.push_back(TheCU);
  Elts.push_back(MDString::get(VMContext, Name));
  Elts.push_back(NULL); // Filename 
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), 0)); // Line
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), SizeInBits));
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits));
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), 0)); // Offset
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), 0)); // Flags;
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), Encoding));
  return DIType(MDNode::get(VMContext, Elts.data(), Elts.size()));
}

/// CreateQaulifiedType - Create debugging information entry for a qualified
/// type, e.g. 'const int'.
DIType DIBuilder::CreateQualifiedType(unsigned Tag, DIType FromTy) {
  /// Qualified types are encoded in DIDerivedType format.
  SmallVector<Value *, 12> Elts;
  Elts.push_back(GetTagConstant(VMContext, Tag));
  Elts.push_back(TheCU);
  Elts.push_back(MDString::get(VMContext, StringRef())); // Empty name.
  Elts.push_back(NULL); // Filename
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), 0)); // Line
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), 0)); // Size
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), 0)); // Align
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), 0)); // Offset
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), 0)); // Flags
  Elts.push_back(FromTy);
  return DIType(MDNode::get(VMContext, Elts.data(), Elts.size()));  
}

/// CreatePointerType - Create debugging information entry for a pointer.
DIType DIBuilder::CreatePointerType(DIType PointeeTy, uint64_t SizeInBits,
                                    uint64_t AlignInBits, StringRef Name) {
  /// pointer types are encoded in DIDerivedType format.
  SmallVector<Value *, 12> Elts;
  Elts.push_back(GetTagConstant(VMContext, dwarf::DW_TAG_pointer_type));
  Elts.push_back(TheCU);
  Elts.push_back(MDString::get(VMContext, Name));
  Elts.push_back(NULL); // Filename
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), 0)); // Line
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), SizeInBits));
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits));
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), 0)); // Offset
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), 0)); // Flags
  Elts.push_back(PointeeTy);
  return DIType(MDNode::get(VMContext, Elts.data(), Elts.size()));  
}

/// CreateReferenceType - Create debugging information entry for a reference.
DIType DIBuilder::CreateReferenceType(DIType RTy) {
  /// references are encoded in DIDerivedType format.
  SmallVector<Value *, 12> Elts;
  Elts.push_back(GetTagConstant(VMContext, dwarf::DW_TAG_reference_type));
  Elts.push_back(TheCU);
  Elts.push_back(NULL); // Name
  Elts.push_back(NULL); // Filename
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), 0)); // Line
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), 0)); // Size
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), 0)); // Align
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), 0)); // Offset
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), 0)); // Flags
  Elts.push_back(RTy);
  return DIType(MDNode::get(VMContext, Elts.data(), Elts.size()));  
}

/// CreateTypedef - Create debugging information entry for a typedef.
DIType DIBuilder::CreateTypedef(DIType Ty, StringRef Name, DIFile File,
                                unsigned LineNo) {
  /// typedefs are encoded in DIDerivedType format.
  assert (Ty.Verify() && "Invalid typedef type!");
  SmallVector<Value *, 12> Elts;
  Elts.push_back(GetTagConstant(VMContext, dwarf::DW_TAG_typedef));
  Elts.push_back(Ty.getContext());
  Elts.push_back(MDString::get(VMContext, Name));
  Elts.push_back(File);
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), LineNo));
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), 0)); // Size
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), 0)); // Align
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), 0)); // Offset
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), 0)); // Flags
  Elts.push_back(Ty);
  return DIType(MDNode::get(VMContext, Elts.data(), Elts.size()));  
}

/// CreateFriend - Create debugging information entry for a 'friend'.
DIType DIBuilder::CreateFriend(DIType Ty, DIType FriendTy) {
  /// typedefs are encoded in DIDerivedType format.
  assert (Ty.Verify() && "Invalid type!");
  assert (FriendTy.Verify() && "Invalid friend type!");
  SmallVector<Value *, 12> Elts;
  Elts.push_back(GetTagConstant(VMContext, dwarf::DW_TAG_friend));
  Elts.push_back(Ty);
  Elts.push_back(NULL); // Name
  Elts.push_back(Ty.getFile());
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), 0)); // Line
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), 0)); // Size
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), 0)); // Align
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), 0)); // Offset
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), 0)); // Flags
  Elts.push_back(FriendTy);
  return DIType(MDNode::get(VMContext, Elts.data(), Elts.size()));  
}

/// CreateInheritance - Create debugging information entry to establish
/// inheritnace relationship between two types.
DIType DIBuilder::CreateInheritance(DIType Ty, DIType BaseTy, 
                                    uint64_t BaseOffset, unsigned Flags) {
  /// TAG_inheritance is encoded in DIDerivedType format.
  SmallVector<Value *, 12> Elts;
  Elts.push_back(GetTagConstant(VMContext, dwarf::DW_TAG_inheritance));
  Elts.push_back(Ty);
  Elts.push_back(NULL); // Name
  Elts.push_back(NULL); // File
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), 0)); // Line
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), 0)); // Size
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), 0)); // Align
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), BaseOffset));
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), Flags));
  Elts.push_back(BaseTy);
  return DIType(MDNode::get(VMContext, Elts.data(), Elts.size()));  
}

/// CreateMemberType - Create debugging information entry for a member.
DIType DIBuilder::CreateMemberType(DIDescriptor Context, StringRef Name, 
                                   DIFile F, unsigned LineNumber, 
                                   uint64_t SizeInBits, uint64_t AlignInBits,
                                   uint64_t OffsetInBits, unsigned Flags, 
                                   DIType Ty) {
 /// TAG_member is encoded in DIDerivedType format.
  SmallVector<Value *, 12> Elts;
  Elts.push_back(GetTagConstant(VMContext, dwarf::DW_TAG_member));
  Elts.push_back(Context);
  Elts.push_back(MDString::get(VMContext, Name));
  Elts.push_back(F);
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), LineNumber));
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), SizeInBits));
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits));
  Elts.push_back(ConstantInt::get(Type::getInt64Ty(VMContext), OffsetInBits));
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), Flags));
  Elts.push_back(Ty);
  return DIType(MDNode::get(VMContext, Elts.data(), Elts.size()));  
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
