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
  : M(m), VMContext(M.getContext()), TheCU(0), DeclareFn(0), ValueFn(0) {}

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
DIType DIBuilder::CreateMemberType(StringRef Name, 
                                   DIFile File, unsigned LineNumber, 
                                   uint64_t SizeInBits, uint64_t AlignInBits,
                                   uint64_t OffsetInBits, unsigned Flags, 
                                   DIType Ty) {
  // TAG_member is encoded in DIDerivedType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_member),
    File, // Or TheCU ? Ty ?
    MDString::get(VMContext, Name),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNumber),
    ConstantInt::get(Type::getInt64Ty(VMContext), SizeInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), OffsetInBits),
    ConstantInt::get(Type::getInt32Ty(VMContext), Flags),
    Ty
  };
  return DIType(MDNode::get(VMContext, &Elts[0], array_lengthof(Elts)));
}

/// CreateStructType - Create debugging information entry for a struct.
DIType DIBuilder::CreateStructType(DIDescriptor Context, StringRef Name, DIFile F,
                                   unsigned LineNumber, uint64_t SizeInBits,
                                   uint64_t AlignInBits, unsigned Flags,
                                   DIArray Elements, unsigned RunTimeLang) {
 // TAG_structure_type is encoded in DICompositeType format.
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_structure_type),
    Context,
    MDString::get(VMContext, Name),
    F,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNumber),
    ConstantInt::get(Type::getInt64Ty(VMContext), SizeInBits),
    ConstantInt::get(Type::getInt64Ty(VMContext), AlignInBits),
    ConstantInt::get(Type::getInt32Ty(VMContext), Flags),
    llvm::Constant::getNullValue(Type::getInt32Ty(VMContext)),
    Elements,
    ConstantInt::get(Type::getInt32Ty(VMContext), RunTimeLang),
    llvm::Constant::getNullValue(Type::getInt32Ty(VMContext)),
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

/// CreateTemporaryType - Create a temporary forward-declared type.
DIType DIBuilder::CreateTemporaryType() {
  // Give the temporary MDNode a tag. It doesn't matter what tag we
  // use here as long as DIType accepts it.
  Value *Elts[] = { GetTagConstant(VMContext, DW_TAG_base_type) };
  MDNode *Node = MDNode::getTemporary(VMContext, Elts, array_lengthof(Elts));
  return DIType(Node);
}

/// CreateTemporaryType - Create a temporary forward-declared type.
DIType DIBuilder::CreateTemporaryType(DIFile F) {
  // Give the temporary MDNode a tag. It doesn't matter what tag we
  // use here as long as DIType accepts it.
  Value *Elts[] = {
    GetTagConstant(VMContext, DW_TAG_base_type),
    F.getCompileUnit(),
    NULL,
    F
  };
  MDNode *Node = MDNode::getTemporary(VMContext, Elts, array_lengthof(Elts));
  return DIType(Node);
}

/// GetOrCreateArray - Get a DIArray, create one if required.
DIArray DIBuilder::GetOrCreateArray(Value *const *Elements, unsigned NumElements) {
  if (NumElements == 0) {
    Value *Null = llvm::Constant::getNullValue(Type::getInt32Ty(VMContext));
    return DIArray(MDNode::get(VMContext, &Null, 1));
  }
  return DIArray(MDNode::get(VMContext, Elements, NumElements));
}

/// CreateGlobalVariable - Create a new descriptor for the specified global.
DIGlobalVariable DIBuilder::
CreateGlobalVariable(StringRef Name, DIFile F, unsigned LineNumber, 
                     DIType Ty, bool isLocalToUnit, llvm::Value *Val) {
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_variable),
    llvm::Constant::getNullValue(Type::getInt32Ty(VMContext)),
    TheCU,
    MDString::get(VMContext, Name),
    MDString::get(VMContext, Name),
    MDString::get(VMContext, Name),
    F,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNumber),
    Ty,
    ConstantInt::get(Type::getInt32Ty(VMContext), isLocalToUnit),
    ConstantInt::get(Type::getInt32Ty(VMContext), 1), /* isDefinition*/
    Val
  };
  MDNode *Node = MDNode::get(VMContext, &Elts[0], array_lengthof(Elts));
  // Create a named metadata so that we do not lose this mdnode.
  NamedMDNode *NMD = M.getOrInsertNamedMetadata("llvm.dbg.gv");
  NMD->addOperand(Node);
  return DIGlobalVariable(Node);
}

/// CreateStaticVariable - Create a new descriptor for the specified static
/// variable.
DIGlobalVariable DIBuilder::
CreateStaticVariable(DIDescriptor Context, StringRef Name, 
                     StringRef LinkageName, DIFile F, unsigned LineNumber, 
                     DIType Ty, bool isLocalToUnit, llvm::Value *Val) {
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_variable),
    llvm::Constant::getNullValue(Type::getInt32Ty(VMContext)),
    Context,
    MDString::get(VMContext, Name),
    MDString::get(VMContext, Name),
    MDString::get(VMContext, LinkageName),
    F,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNumber),
    Ty,
    ConstantInt::get(Type::getInt32Ty(VMContext), isLocalToUnit),
    ConstantInt::get(Type::getInt32Ty(VMContext), 1), /* isDefinition*/
    Val
  };
  MDNode *Node = MDNode::get(VMContext, &Elts[0], array_lengthof(Elts));
  // Create a named metadata so that we do not lose this mdnode.
  NamedMDNode *NMD = M.getOrInsertNamedMetadata("llvm.dbg.gv");
  NMD->addOperand(Node);
  return DIGlobalVariable(Node);
}

/// CreateVariable - Create a new descriptor for the specified variable.
DIVariable DIBuilder::CreateLocalVariable(unsigned Tag, DIDescriptor Scope,
                                          StringRef Name, DIFile File,
                                          unsigned LineNo, DIType Ty, 
                                          bool AlwaysPreserve, unsigned Flags) {
  Value *Elts[] = {
    GetTagConstant(VMContext, Tag),
    Scope,
    MDString::get(VMContext, Name),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNo),
    Ty,
    ConstantInt::get(Type::getInt32Ty(VMContext), Flags)
  };
  MDNode *Node = MDNode::get(VMContext, &Elts[0], array_lengthof(Elts));
  if (AlwaysPreserve) {
    // The optimizer may remove local variable. If there is an interest
    // to preserve variable info in such situation then stash it in a
    // named mdnode.
    DISubprogram Fn(getDISubprogram(Scope));
    StringRef FName = "fn";
    if (Fn.getFunction())
      FName = Fn.getFunction()->getName();
    char One = '\1';
    if (FName.startswith(StringRef(&One, 1)))
      FName = FName.substr(1);
    NamedMDNode *FnLocals = getOrInsertFnSpecificMDNode(M, FName);
    FnLocals->addOperand(Node);
  }
  return DIVariable(Node);
}

/// CreateComplexVariable - Create a new descriptor for the specified variable
/// which has a complex address expression for its address.
DIVariable DIBuilder::CreateComplexVariable(unsigned Tag, DIDescriptor Scope,
                                            StringRef Name, DIFile F,
                                            unsigned LineNo,
                                            DIType Ty, Value *const *Addr,
                                            unsigned NumAddr) {
  SmallVector<Value *, 15> Elts;
  Elts.push_back(GetTagConstant(VMContext, Tag));
  Elts.push_back(Scope);
  Elts.push_back(MDString::get(VMContext, Name));
  Elts.push_back(F);
  Elts.push_back(ConstantInt::get(Type::getInt32Ty(VMContext), LineNo));
  Elts.push_back(Ty);
  Elts.append(Addr, Addr+NumAddr);

  return DIVariable(MDNode::get(VMContext, Elts.data(), Elts.size()));
}


/// CreateNameSpace - This creates new descriptor for a namespace
/// with the specified parent scope.
DINameSpace DIBuilder::CreateNameSpace(DIDescriptor Scope, StringRef Name,
                                       DIFile File, unsigned LineNo) {
  Value *Elts[] = {
    GetTagConstant(VMContext, dwarf::DW_TAG_namespace),
    Scope,
    MDString::get(VMContext, Name),
    File,
    ConstantInt::get(Type::getInt32Ty(VMContext), LineNo)
  };
  return DINameSpace(MDNode::get(VMContext, &Elts[0], array_lengthof(Elts)));
}

/// InsertDeclare - Insert a new llvm.dbg.declare intrinsic call.
Instruction *DIBuilder::InsertDeclare(Value *Storage, DIVariable VarInfo,
                                      Instruction *InsertBefore) {
  assert(Storage && "no storage passed to dbg.declare");
  assert(VarInfo.Verify() && "empty DIVariable passed to dbg.declare");
  if (!DeclareFn)
    DeclareFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_declare);

  Value *Args[] = { MDNode::get(Storage->getContext(), &Storage, 1), VarInfo };
  return CallInst::Create(DeclareFn, Args, Args+2, "", InsertBefore);
}

/// InsertDeclare - Insert a new llvm.dbg.declare intrinsic call.
Instruction *DIBuilder::InsertDeclare(Value *Storage, DIVariable VarInfo,
                                      BasicBlock *InsertAtEnd) {
  assert(Storage && "no storage passed to dbg.declare");
  assert(VarInfo.Verify() && "invalid DIVariable passed to dbg.declare");
  if (!DeclareFn)
    DeclareFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_declare);

  Value *Args[] = { MDNode::get(Storage->getContext(), &Storage, 1), VarInfo };

  // If this block already has a terminator then insert this intrinsic
  // before the terminator.
  if (TerminatorInst *T = InsertAtEnd->getTerminator())
    return CallInst::Create(DeclareFn, Args, Args+2, "", T);
  else
    return CallInst::Create(DeclareFn, Args, Args+2, "", InsertAtEnd);
}

/// InsertDbgValueIntrinsic - Insert a new llvm.dbg.value intrinsic call.
Instruction *DIBuilder::InsertDbgValueIntrinsic(Value *V, uint64_t Offset,
                                                DIVariable VarInfo,
                                                Instruction *InsertBefore) {
  assert(V && "no value passed to dbg.value");
  assert(VarInfo.Verify() && "invalid DIVariable passed to dbg.value");
  if (!ValueFn)
    ValueFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_value);

  Value *Args[] = { MDNode::get(V->getContext(), &V, 1),
                    ConstantInt::get(Type::getInt64Ty(V->getContext()), Offset),
                    VarInfo };
  return CallInst::Create(ValueFn, Args, Args+3, "", InsertBefore);
}

/// InsertDbgValueIntrinsic - Insert a new llvm.dbg.value intrinsic call.
Instruction *DIBuilder::InsertDbgValueIntrinsic(Value *V, uint64_t Offset,
                                                DIVariable VarInfo,
                                                BasicBlock *InsertAtEnd) {
  assert(V && "no value passed to dbg.value");
  assert(VarInfo.Verify() && "invalid DIVariable passed to dbg.value");
  if (!ValueFn)
    ValueFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_value);

  Value *Args[] = { MDNode::get(V->getContext(), &V, 1),
                    ConstantInt::get(Type::getInt64Ty(V->getContext()), Offset),
                    VarInfo };
  return CallInst::Create(ValueFn, Args, Args+3, "", InsertAtEnd);
}

