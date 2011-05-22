//===---- TargetInfo.cpp - Encapsulate target details -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#include "TargetInfo.h"
#include "ABIInfo.h"
#include "CodeGenFunction.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/Type.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;
using namespace CodeGen;

static void AssignToArrayRange(CodeGen::CGBuilderTy &Builder,
                               llvm::Value *Array,
                               llvm::Value *Value,
                               unsigned FirstIndex,
                               unsigned LastIndex) {
  // Alternatively, we could emit this as a loop in the source.
  for (unsigned I = FirstIndex; I <= LastIndex; ++I) {
    llvm::Value *Cell = Builder.CreateConstInBoundsGEP1_32(Array, I);
    Builder.CreateStore(Value, Cell);
  }
}

static bool isAggregateTypeForABI(QualType T) {
  return CodeGenFunction::hasAggregateLLVMType(T) ||
         T->isMemberFunctionPointerType();
}

ABIInfo::~ABIInfo() {}

ASTContext &ABIInfo::getContext() const {
  return CGT.getContext();
}

llvm::LLVMContext &ABIInfo::getVMContext() const {
  return CGT.getLLVMContext();
}

const llvm::TargetData &ABIInfo::getTargetData() const {
  return CGT.getTargetData();
}


void ABIArgInfo::dump() const {
  llvm::raw_ostream &OS = llvm::errs();
  OS << "(ABIArgInfo Kind=";
  switch (TheKind) {
  case Direct:
    OS << "Direct Type=";
    if (const llvm::Type *Ty = getCoerceToType())
      Ty->print(OS);
    else
      OS << "null";
    break;
  case Extend:
    OS << "Extend";
    break;
  case Ignore:
    OS << "Ignore";
    break;
  case Indirect:
    OS << "Indirect Align=" << getIndirectAlign()
       << " Byal=" << getIndirectByVal()
       << " Realign=" << getIndirectRealign();
    break;
  case Expand:
    OS << "Expand";
    break;
  }
  OS << ")\n";
}

TargetCodeGenInfo::~TargetCodeGenInfo() { delete Info; }

static bool isEmptyRecord(ASTContext &Context, QualType T, bool AllowArrays);

/// isEmptyField - Return true iff a the field is "empty", that is it
/// is an unnamed bit-field or an (array of) empty record(s).
static bool isEmptyField(ASTContext &Context, const FieldDecl *FD,
                         bool AllowArrays) {
  if (FD->isUnnamedBitfield())
    return true;

  QualType FT = FD->getType();

    // Constant arrays of empty records count as empty, strip them off.
  if (AllowArrays)
    while (const ConstantArrayType *AT = Context.getAsConstantArrayType(FT))
      FT = AT->getElementType();

  const RecordType *RT = FT->getAs<RecordType>();
  if (!RT)
    return false;

  // C++ record fields are never empty, at least in the Itanium ABI.
  //
  // FIXME: We should use a predicate for whether this behavior is true in the
  // current ABI.
  if (isa<CXXRecordDecl>(RT->getDecl()))
    return false;

  return isEmptyRecord(Context, FT, AllowArrays);
}

/// isEmptyRecord - Return true iff a structure contains only empty
/// fields. Note that a structure with a flexible array member is not
/// considered empty.
static bool isEmptyRecord(ASTContext &Context, QualType T, bool AllowArrays) {
  const RecordType *RT = T->getAs<RecordType>();
  if (!RT)
    return 0;
  const RecordDecl *RD = RT->getDecl();
  if (RD->hasFlexibleArrayMember())
    return false;

  // If this is a C++ record, check the bases first.
  if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD))
    for (CXXRecordDecl::base_class_const_iterator i = CXXRD->bases_begin(),
           e = CXXRD->bases_end(); i != e; ++i)
      if (!isEmptyRecord(Context, i->getType(), true))
        return false;

  for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
         i != e; ++i)
    if (!isEmptyField(Context, *i, AllowArrays))
      return false;
  return true;
}

/// hasNonTrivialDestructorOrCopyConstructor - Determine if a type has either
/// a non-trivial destructor or a non-trivial copy constructor.
static bool hasNonTrivialDestructorOrCopyConstructor(const RecordType *RT) {
  const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl());
  if (!RD)
    return false;

  return !RD->hasTrivialDestructor() || !RD->hasTrivialCopyConstructor();
}

/// isRecordWithNonTrivialDestructorOrCopyConstructor - Determine if a type is
/// a record type with either a non-trivial destructor or a non-trivial copy
/// constructor.
static bool isRecordWithNonTrivialDestructorOrCopyConstructor(QualType T) {
  const RecordType *RT = T->getAs<RecordType>();
  if (!RT)
    return false;

  return hasNonTrivialDestructorOrCopyConstructor(RT);
}

/// isSingleElementStruct - Determine if a structure is a "single
/// element struct", i.e. it has exactly one non-empty field or
/// exactly one field which is itself a single element
/// struct. Structures with flexible array members are never
/// considered single element structs.
///
/// \return The field declaration for the single non-empty field, if
/// it exists.
static const Type *isSingleElementStruct(QualType T, ASTContext &Context) {
  const RecordType *RT = T->getAsStructureType();
  if (!RT)
    return 0;

  const RecordDecl *RD = RT->getDecl();
  if (RD->hasFlexibleArrayMember())
    return 0;

  const Type *Found = 0;

  // If this is a C++ record, check the bases first.
  if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
    for (CXXRecordDecl::base_class_const_iterator i = CXXRD->bases_begin(),
           e = CXXRD->bases_end(); i != e; ++i) {
      // Ignore empty records.
      if (isEmptyRecord(Context, i->getType(), true))
        continue;

      // If we already found an element then this isn't a single-element struct.
      if (Found)
        return 0;

      // If this is non-empty and not a single element struct, the composite
      // cannot be a single element struct.
      Found = isSingleElementStruct(i->getType(), Context);
      if (!Found)
        return 0;
    }
  }

  // Check for single element.
  for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
         i != e; ++i) {
    const FieldDecl *FD = *i;
    QualType FT = FD->getType();

    // Ignore empty fields.
    if (isEmptyField(Context, FD, true))
      continue;

    // If we already found an element then this isn't a single-element
    // struct.
    if (Found)
      return 0;

    // Treat single element arrays as the element.
    while (const ConstantArrayType *AT = Context.getAsConstantArrayType(FT)) {
      if (AT->getSize().getZExtValue() != 1)
        break;
      FT = AT->getElementType();
    }

    if (!isAggregateTypeForABI(FT)) {
      Found = FT.getTypePtr();
    } else {
      Found = isSingleElementStruct(FT, Context);
      if (!Found)
        return 0;
    }
  }

  return Found;
}

static bool is32Or64BitBasicType(QualType Ty, ASTContext &Context) {
  if (!Ty->getAs<BuiltinType>() && !Ty->hasPointerRepresentation() &&
      !Ty->isAnyComplexType() && !Ty->isEnumeralType() &&
      !Ty->isBlockPointerType())
    return false;

  uint64_t Size = Context.getTypeSize(Ty);
  return Size == 32 || Size == 64;
}

/// canExpandIndirectArgument - Test whether an argument type which is to be
/// passed indirectly (on the stack) would have the equivalent layout if it was
/// expanded into separate arguments. If so, we prefer to do the latter to avoid
/// inhibiting optimizations.
///
// FIXME: This predicate is missing many cases, currently it just follows
// llvm-gcc (checks that all fields are 32-bit or 64-bit primitive types). We
// should probably make this smarter, or better yet make the LLVM backend
// capable of handling it.
static bool canExpandIndirectArgument(QualType Ty, ASTContext &Context) {
  // We can only expand structure types.
  const RecordType *RT = Ty->getAs<RecordType>();
  if (!RT)
    return false;

  // We can only expand (C) structures.
  //
  // FIXME: This needs to be generalized to handle classes as well.
  const RecordDecl *RD = RT->getDecl();
  if (!RD->isStruct() || isa<CXXRecordDecl>(RD))
    return false;

  for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
         i != e; ++i) {
    const FieldDecl *FD = *i;

    if (!is32Or64BitBasicType(FD->getType(), Context))
      return false;

    // FIXME: Reject bit-fields wholesale; there are two problems, we don't know
    // how to expand them yet, and the predicate for telling if a bitfield still
    // counts as "basic" is more complicated than what we were doing previously.
    if (FD->isBitField())
      return false;
  }

  return true;
}

namespace {
/// DefaultABIInfo - The default implementation for ABI specific
/// details. This implementation provides information which results in
/// self-consistent and sensible LLVM IR generation, but does not
/// conform to any particular ABI.
class DefaultABIInfo : public ABIInfo {
public:
  DefaultABIInfo(CodeGen::CodeGenTypes &CGT) : ABIInfo(CGT) {}

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType RetTy) const;

  virtual void computeInfo(CGFunctionInfo &FI) const {
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
    for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
         it != ie; ++it)
      it->info = classifyArgumentType(it->type);
  }

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

class DefaultTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  DefaultTargetCodeGenInfo(CodeGen::CodeGenTypes &CGT)
    : TargetCodeGenInfo(new DefaultABIInfo(CGT)) {}
};

llvm::Value *DefaultABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                       CodeGenFunction &CGF) const {
  return 0;
}

ABIArgInfo DefaultABIInfo::classifyArgumentType(QualType Ty) const {
  if (isAggregateTypeForABI(Ty))
    return ABIArgInfo::getIndirect(0);

  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getDecl()->getIntegerType();

  return (Ty->isPromotableIntegerType() ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

ABIArgInfo DefaultABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  if (isAggregateTypeForABI(RetTy))
    return ABIArgInfo::getIndirect(0);

  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
    RetTy = EnumTy->getDecl()->getIntegerType();

  return (RetTy->isPromotableIntegerType() ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

/// UseX86_MMXType - Return true if this is an MMX type that should use the special
/// x86_mmx type.
bool UseX86_MMXType(const llvm::Type *IRType) {
  // If the type is an MMX type <2 x i32>, <4 x i16>, or <8 x i8>, use the
  // special x86_mmx type.
  return IRType->isVectorTy() && IRType->getPrimitiveSizeInBits() == 64 &&
    cast<llvm::VectorType>(IRType)->getElementType()->isIntegerTy() &&
    IRType->getScalarSizeInBits() != 64;
}

static const llvm::Type* X86AdjustInlineAsmType(CodeGen::CodeGenFunction &CGF,
                                                llvm::StringRef Constraint,
                                                const llvm::Type* Ty) {
  if ((Constraint == "y" || Constraint == "&y") && Ty->isVectorTy())
    return llvm::Type::getX86_MMXTy(CGF.getLLVMContext());
  return Ty;
}

//===----------------------------------------------------------------------===//
// X86-32 ABI Implementation
//===----------------------------------------------------------------------===//

/// X86_32ABIInfo - The X86-32 ABI information.
class X86_32ABIInfo : public ABIInfo {
  static const unsigned MinABIStackAlignInBytes = 4;

  bool IsDarwinVectorABI;
  bool IsSmallStructInRegABI;

  static bool isRegisterSize(unsigned Size) {
    return (Size == 8 || Size == 16 || Size == 32 || Size == 64);
  }

  static bool shouldReturnTypeInRegister(QualType Ty, ASTContext &Context);

  /// getIndirectResult - Give a source type \arg Ty, return a suitable result
  /// such that the argument will be passed in memory.
  ABIArgInfo getIndirectResult(QualType Ty, bool ByVal = true) const;

  /// \brief Return the alignment to use for the given type on the stack.
  unsigned getTypeStackAlignInBytes(QualType Ty, unsigned Align) const;

public:

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType RetTy) const;

  virtual void computeInfo(CGFunctionInfo &FI) const {
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
    for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
         it != ie; ++it)
      it->info = classifyArgumentType(it->type);
  }

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;

  X86_32ABIInfo(CodeGen::CodeGenTypes &CGT, bool d, bool p)
    : ABIInfo(CGT), IsDarwinVectorABI(d), IsSmallStructInRegABI(p) {}
};

class X86_32TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  X86_32TargetCodeGenInfo(CodeGen::CodeGenTypes &CGT, bool d, bool p)
    :TargetCodeGenInfo(new X86_32ABIInfo(CGT, d, p)) {}

  void SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &CGM) const;

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &CGM) const {
    // Darwin uses different dwarf register numbers for EH.
    if (CGM.isTargetDarwin()) return 5;

    return 4;
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const;

  const llvm::Type* adjustInlineAsmType(CodeGen::CodeGenFunction &CGF,
                                        llvm::StringRef Constraint,
                                        const llvm::Type* Ty) const {
    return X86AdjustInlineAsmType(CGF, Constraint, Ty);
  }

};

}

/// shouldReturnTypeInRegister - Determine if the given type should be
/// passed in a register (for the Darwin ABI).
bool X86_32ABIInfo::shouldReturnTypeInRegister(QualType Ty,
                                               ASTContext &Context) {
  uint64_t Size = Context.getTypeSize(Ty);

  // Type must be register sized.
  if (!isRegisterSize(Size))
    return false;

  if (Ty->isVectorType()) {
    // 64- and 128- bit vectors inside structures are not returned in
    // registers.
    if (Size == 64 || Size == 128)
      return false;

    return true;
  }

  // If this is a builtin, pointer, enum, complex type, member pointer, or
  // member function pointer it is ok.
  if (Ty->getAs<BuiltinType>() || Ty->hasPointerRepresentation() ||
      Ty->isAnyComplexType() || Ty->isEnumeralType() ||
      Ty->isBlockPointerType() || Ty->isMemberPointerType())
    return true;

  // Arrays are treated like records.
  if (const ConstantArrayType *AT = Context.getAsConstantArrayType(Ty))
    return shouldReturnTypeInRegister(AT->getElementType(), Context);

  // Otherwise, it must be a record type.
  const RecordType *RT = Ty->getAs<RecordType>();
  if (!RT) return false;

  // FIXME: Traverse bases here too.

  // Structure types are passed in register if all fields would be
  // passed in a register.
  for (RecordDecl::field_iterator i = RT->getDecl()->field_begin(),
         e = RT->getDecl()->field_end(); i != e; ++i) {
    const FieldDecl *FD = *i;

    // Empty fields are ignored.
    if (isEmptyField(Context, FD, true))
      continue;

    // Check fields recursively.
    if (!shouldReturnTypeInRegister(FD->getType(), Context))
      return false;
  }

  return true;
}

ABIArgInfo X86_32ABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  if (const VectorType *VT = RetTy->getAs<VectorType>()) {
    // On Darwin, some vectors are returned in registers.
    if (IsDarwinVectorABI) {
      uint64_t Size = getContext().getTypeSize(RetTy);

      // 128-bit vectors are a special case; they are returned in
      // registers and we need to make sure to pick a type the LLVM
      // backend will like.
      if (Size == 128)
        return ABIArgInfo::getDirect(llvm::VectorType::get(
                  llvm::Type::getInt64Ty(getVMContext()), 2));

      // Always return in register if it fits in a general purpose
      // register, or if it is 64 bits and has a single element.
      if ((Size == 8 || Size == 16 || Size == 32) ||
          (Size == 64 && VT->getNumElements() == 1))
        return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(),
                                                            Size));

      return ABIArgInfo::getIndirect(0);
    }

    return ABIArgInfo::getDirect();
  }

  if (isAggregateTypeForABI(RetTy)) {
    if (const RecordType *RT = RetTy->getAs<RecordType>()) {
      // Structures with either a non-trivial destructor or a non-trivial
      // copy constructor are always indirect.
      if (hasNonTrivialDestructorOrCopyConstructor(RT))
        return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

      // Structures with flexible arrays are always indirect.
      if (RT->getDecl()->hasFlexibleArrayMember())
        return ABIArgInfo::getIndirect(0);
    }

    // If specified, structs and unions are always indirect.
    if (!IsSmallStructInRegABI && !RetTy->isAnyComplexType())
      return ABIArgInfo::getIndirect(0);

    // Classify "single element" structs as their element type.
    if (const Type *SeltTy = isSingleElementStruct(RetTy, getContext())) {
      if (const BuiltinType *BT = SeltTy->getAs<BuiltinType>()) {
        if (BT->isIntegerType()) {
          // We need to use the size of the structure, padding
          // bit-fields can adjust that to be larger than the single
          // element type.
          uint64_t Size = getContext().getTypeSize(RetTy);
          return ABIArgInfo::getDirect(
            llvm::IntegerType::get(getVMContext(), (unsigned)Size));
        }

        if (BT->getKind() == BuiltinType::Float) {
          assert(getContext().getTypeSize(RetTy) ==
                 getContext().getTypeSize(SeltTy) &&
                 "Unexpect single element structure size!");
          return ABIArgInfo::getDirect(llvm::Type::getFloatTy(getVMContext()));
        }

        if (BT->getKind() == BuiltinType::Double) {
          assert(getContext().getTypeSize(RetTy) ==
                 getContext().getTypeSize(SeltTy) &&
                 "Unexpect single element structure size!");
          return ABIArgInfo::getDirect(llvm::Type::getDoubleTy(getVMContext()));
        }
      } else if (SeltTy->isPointerType()) {
        // FIXME: It would be really nice if this could come out as the proper
        // pointer type.
        const llvm::Type *PtrTy = llvm::Type::getInt8PtrTy(getVMContext());
        return ABIArgInfo::getDirect(PtrTy);
      } else if (SeltTy->isVectorType()) {
        // 64- and 128-bit vectors are never returned in a
        // register when inside a structure.
        uint64_t Size = getContext().getTypeSize(RetTy);
        if (Size == 64 || Size == 128)
          return ABIArgInfo::getIndirect(0);

        return classifyReturnType(QualType(SeltTy, 0));
      }
    }

    // Small structures which are register sized are generally returned
    // in a register.
    if (X86_32ABIInfo::shouldReturnTypeInRegister(RetTy, getContext())) {
      uint64_t Size = getContext().getTypeSize(RetTy);
      return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(),Size));
    }

    return ABIArgInfo::getIndirect(0);
  }

  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
    RetTy = EnumTy->getDecl()->getIntegerType();

  return (RetTy->isPromotableIntegerType() ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

static bool isRecordWithSSEVectorType(ASTContext &Context, QualType Ty) {
  const RecordType *RT = Ty->getAs<RecordType>();
  if (!RT)
    return 0;
  const RecordDecl *RD = RT->getDecl();

  // If this is a C++ record, check the bases first.
  if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD))
    for (CXXRecordDecl::base_class_const_iterator i = CXXRD->bases_begin(),
           e = CXXRD->bases_end(); i != e; ++i)
      if (!isRecordWithSSEVectorType(Context, i->getType()))
        return false;

  for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
       i != e; ++i) {
    QualType FT = i->getType();

    if (FT->getAs<VectorType>() && Context.getTypeSize(Ty) == 128)
      return true;

    if (isRecordWithSSEVectorType(Context, FT))
      return true;
  }

  return false;
}

unsigned X86_32ABIInfo::getTypeStackAlignInBytes(QualType Ty,
                                                 unsigned Align) const {
  // Otherwise, if the alignment is less than or equal to the minimum ABI
  // alignment, just use the default; the backend will handle this.
  if (Align <= MinABIStackAlignInBytes)
    return 0; // Use default alignment.

  // On non-Darwin, the stack type alignment is always 4.
  if (!IsDarwinVectorABI) {
    // Set explicit alignment, since we may need to realign the top.
    return MinABIStackAlignInBytes;
  }

  // Otherwise, if the type contains an SSE vector type, the alignment is 16.
  if (isRecordWithSSEVectorType(getContext(), Ty))
    return 16;

  return MinABIStackAlignInBytes;
}

ABIArgInfo X86_32ABIInfo::getIndirectResult(QualType Ty, bool ByVal) const {
  if (!ByVal)
    return ABIArgInfo::getIndirect(0, false);

  // Compute the byval alignment.
  unsigned TypeAlign = getContext().getTypeAlign(Ty) / 8;
  unsigned StackAlign = getTypeStackAlignInBytes(Ty, TypeAlign);
  if (StackAlign == 0)
    return ABIArgInfo::getIndirect(4);

  // If the stack alignment is less than the type alignment, realign the
  // argument.
  if (StackAlign < TypeAlign)
    return ABIArgInfo::getIndirect(StackAlign, /*ByVal=*/true,
                                   /*Realign=*/true);

  return ABIArgInfo::getIndirect(StackAlign);
}

ABIArgInfo X86_32ABIInfo::classifyArgumentType(QualType Ty) const {
  // FIXME: Set alignment on indirect arguments.
  if (isAggregateTypeForABI(Ty)) {
    // Structures with flexible arrays are always indirect.
    if (const RecordType *RT = Ty->getAs<RecordType>()) {
      // Structures with either a non-trivial destructor or a non-trivial
      // copy constructor are always indirect.
      if (hasNonTrivialDestructorOrCopyConstructor(RT))
        return getIndirectResult(Ty, /*ByVal=*/false);

      if (RT->getDecl()->hasFlexibleArrayMember())
        return getIndirectResult(Ty);
    }

    // Ignore empty structs.
    if (Ty->isStructureType() && getContext().getTypeSize(Ty) == 0)
      return ABIArgInfo::getIgnore();

    // Expand small (<= 128-bit) record types when we know that the stack layout
    // of those arguments will match the struct. This is important because the
    // LLVM backend isn't smart enough to remove byval, which inhibits many
    // optimizations.
    if (getContext().getTypeSize(Ty) <= 4*32 &&
        canExpandIndirectArgument(Ty, getContext()))
      return ABIArgInfo::getExpand();

    return getIndirectResult(Ty);
  }

  if (const VectorType *VT = Ty->getAs<VectorType>()) {
    // On Darwin, some vectors are passed in memory, we handle this by passing
    // it as an i8/i16/i32/i64.
    if (IsDarwinVectorABI) {
      uint64_t Size = getContext().getTypeSize(Ty);
      if ((Size == 8 || Size == 16 || Size == 32) ||
          (Size == 64 && VT->getNumElements() == 1))
        return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(),
                                                            Size));
    }

    const llvm::Type *IRType = CGT.ConvertTypeRecursive(Ty);
    if (UseX86_MMXType(IRType)) {
      ABIArgInfo AAI = ABIArgInfo::getDirect(IRType);
      AAI.setCoerceToType(llvm::Type::getX86_MMXTy(getVMContext()));
      return AAI;
    }

    return ABIArgInfo::getDirect();
  }


  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getDecl()->getIntegerType();

  return (Ty->isPromotableIntegerType() ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

llvm::Value *X86_32ABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                      CodeGenFunction &CGF) const {
  const llvm::Type *BP = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  const llvm::Type *BPP = llvm::PointerType::getUnqual(BP);

  CGBuilderTy &Builder = CGF.Builder;
  llvm::Value *VAListAddrAsBPP = Builder.CreateBitCast(VAListAddr, BPP,
                                                       "ap");
  llvm::Value *Addr = Builder.CreateLoad(VAListAddrAsBPP, "ap.cur");
  llvm::Type *PTy =
    llvm::PointerType::getUnqual(CGF.ConvertType(Ty));
  llvm::Value *AddrTyped = Builder.CreateBitCast(Addr, PTy);

  uint64_t Offset =
    llvm::RoundUpToAlignment(CGF.getContext().getTypeSize(Ty) / 8, 4);
  llvm::Value *NextAddr =
    Builder.CreateGEP(Addr, llvm::ConstantInt::get(CGF.Int32Ty, Offset),
                      "ap.next");
  Builder.CreateStore(NextAddr, VAListAddrAsBPP);

  return AddrTyped;
}

void X86_32TargetCodeGenInfo::SetTargetAttributes(const Decl *D,
                                                  llvm::GlobalValue *GV,
                                            CodeGen::CodeGenModule &CGM) const {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->hasAttr<X86ForceAlignArgPointerAttr>()) {
      // Get the LLVM function.
      llvm::Function *Fn = cast<llvm::Function>(GV);

      // Now add the 'alignstack' attribute with a value of 16.
      Fn->addFnAttr(llvm::Attribute::constructStackAlignmentFromInt(16));
    }
  }
}

bool X86_32TargetCodeGenInfo::initDwarfEHRegSizeTable(
                                               CodeGen::CodeGenFunction &CGF,
                                               llvm::Value *Address) const {
  CodeGen::CGBuilderTy &Builder = CGF.Builder;
  llvm::LLVMContext &Context = CGF.getLLVMContext();

  const llvm::IntegerType *i8 = llvm::Type::getInt8Ty(Context);
  llvm::Value *Four8 = llvm::ConstantInt::get(i8, 4);

  // 0-7 are the eight integer registers;  the order is different
  //   on Darwin (for EH), but the range is the same.
  // 8 is %eip.
  AssignToArrayRange(Builder, Address, Four8, 0, 8);

  if (CGF.CGM.isTargetDarwin()) {
    // 12-16 are st(0..4).  Not sure why we stop at 4.
    // These have size 16, which is sizeof(long double) on
    // platforms with 8-byte alignment for that type.
    llvm::Value *Sixteen8 = llvm::ConstantInt::get(i8, 16);
    AssignToArrayRange(Builder, Address, Sixteen8, 12, 16);

  } else {
    // 9 is %eflags, which doesn't get a size on Darwin for some
    // reason.
    Builder.CreateStore(Four8, Builder.CreateConstInBoundsGEP1_32(Address, 9));

    // 11-16 are st(0..5).  Not sure why we stop at 5.
    // These have size 12, which is sizeof(long double) on
    // platforms with 4-byte alignment for that type.
    llvm::Value *Twelve8 = llvm::ConstantInt::get(i8, 12);
    AssignToArrayRange(Builder, Address, Twelve8, 11, 16);
  }

  return false;
}

//===----------------------------------------------------------------------===//
// X86-64 ABI Implementation
//===----------------------------------------------------------------------===//


namespace {
/// X86_64ABIInfo - The X86_64 ABI information.
class X86_64ABIInfo : public ABIInfo {
  enum Class {
    Integer = 0,
    SSE,
    SSEUp,
    X87,
    X87Up,
    ComplexX87,
    NoClass,
    Memory
  };

  /// merge - Implement the X86_64 ABI merging algorithm.
  ///
  /// Merge an accumulating classification \arg Accum with a field
  /// classification \arg Field.
  ///
  /// \param Accum - The accumulating classification. This should
  /// always be either NoClass or the result of a previous merge
  /// call. In addition, this should never be Memory (the caller
  /// should just return Memory for the aggregate).
  static Class merge(Class Accum, Class Field);

  /// classify - Determine the x86_64 register classes in which the
  /// given type T should be passed.
  ///
  /// \param Lo - The classification for the parts of the type
  /// residing in the low word of the containing object.
  ///
  /// \param Hi - The classification for the parts of the type
  /// residing in the high word of the containing object.
  ///
  /// \param OffsetBase - The bit offset of this type in the
  /// containing object.  Some parameters are classified different
  /// depending on whether they straddle an eightbyte boundary.
  ///
  /// If a word is unused its result will be NoClass; if a type should
  /// be passed in Memory then at least the classification of \arg Lo
  /// will be Memory.
  ///
  /// The \arg Lo class will be NoClass iff the argument is ignored.
  ///
  /// If the \arg Lo class is ComplexX87, then the \arg Hi class will
  /// also be ComplexX87.
  void classify(QualType T, uint64_t OffsetBase, Class &Lo, Class &Hi) const;

  const llvm::Type *Get16ByteVectorType(QualType Ty) const;
  const llvm::Type *GetSSETypeAtOffset(const llvm::Type *IRType,
                                       unsigned IROffset, QualType SourceTy,
                                       unsigned SourceOffset) const;
  const llvm::Type *GetINTEGERTypeAtOffset(const llvm::Type *IRType,
                                           unsigned IROffset, QualType SourceTy,
                                           unsigned SourceOffset) const;

  /// getIndirectResult - Give a source type \arg Ty, return a suitable result
  /// such that the argument will be returned in memory.
  ABIArgInfo getIndirectReturnResult(QualType Ty) const;

  /// getIndirectResult - Give a source type \arg Ty, return a suitable result
  /// such that the argument will be passed in memory.
  ABIArgInfo getIndirectResult(QualType Ty) const;

  ABIArgInfo classifyReturnType(QualType RetTy) const;

  ABIArgInfo classifyArgumentType(QualType Ty,
                                  unsigned &neededInt,
                                  unsigned &neededSSE) const;

  /// The 0.98 ABI revision clarified a lot of ambiguities,
  /// unfortunately in ways that were not always consistent with
  /// certain previous compilers.  In particular, platforms which
  /// required strict binary compatibility with older versions of GCC
  /// may need to exempt themselves.
  bool honorsRevision0_98() const {
    return !getContext().Target.getTriple().isOSDarwin();
  }

public:
  X86_64ABIInfo(CodeGen::CodeGenTypes &CGT) : ABIInfo(CGT) {}

  virtual void computeInfo(CGFunctionInfo &FI) const;

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

/// WinX86_64ABIInfo - The Windows X86_64 ABI information.
class WinX86_64ABIInfo : public ABIInfo {

  ABIArgInfo classify(QualType Ty) const;

public:
  WinX86_64ABIInfo(CodeGen::CodeGenTypes &CGT) : ABIInfo(CGT) {}

  virtual void computeInfo(CGFunctionInfo &FI) const;

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

class X86_64TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  X86_64TargetCodeGenInfo(CodeGen::CodeGenTypes &CGT)
    : TargetCodeGenInfo(new X86_64ABIInfo(CGT)) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &CGM) const {
    return 7;
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const {
    CodeGen::CGBuilderTy &Builder = CGF.Builder;
    llvm::LLVMContext &Context = CGF.getLLVMContext();

    const llvm::IntegerType *i8 = llvm::Type::getInt8Ty(Context);
    llvm::Value *Eight8 = llvm::ConstantInt::get(i8, 8);

    // 0-15 are the 16 integer registers.
    // 16 is %rip.
    AssignToArrayRange(Builder, Address, Eight8, 0, 16);

    return false;
  }

  const llvm::Type* adjustInlineAsmType(CodeGen::CodeGenFunction &CGF,
                                        llvm::StringRef Constraint,
                                        const llvm::Type* Ty) const {
    return X86AdjustInlineAsmType(CGF, Constraint, Ty);
  }

};

class WinX86_64TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  WinX86_64TargetCodeGenInfo(CodeGen::CodeGenTypes &CGT)
    : TargetCodeGenInfo(new WinX86_64ABIInfo(CGT)) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &CGM) const {
    return 7;
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const {
    CodeGen::CGBuilderTy &Builder = CGF.Builder;
    llvm::LLVMContext &Context = CGF.getLLVMContext();

    const llvm::IntegerType *i8 = llvm::Type::getInt8Ty(Context);
    llvm::Value *Eight8 = llvm::ConstantInt::get(i8, 8);

    // 0-15 are the 16 integer registers.
    // 16 is %rip.
    AssignToArrayRange(Builder, Address, Eight8, 0, 16);

    return false;
  }
};

}

X86_64ABIInfo::Class X86_64ABIInfo::merge(Class Accum, Class Field) {
  // AMD64-ABI 3.2.3p2: Rule 4. Each field of an object is
  // classified recursively so that always two fields are
  // considered. The resulting class is calculated according to
  // the classes of the fields in the eightbyte:
  //
  // (a) If both classes are equal, this is the resulting class.
  //
  // (b) If one of the classes is NO_CLASS, the resulting class is
  // the other class.
  //
  // (c) If one of the classes is MEMORY, the result is the MEMORY
  // class.
  //
  // (d) If one of the classes is INTEGER, the result is the
  // INTEGER.
  //
  // (e) If one of the classes is X87, X87UP, COMPLEX_X87 class,
  // MEMORY is used as class.
  //
  // (f) Otherwise class SSE is used.

  // Accum should never be memory (we should have returned) or
  // ComplexX87 (because this cannot be passed in a structure).
  assert((Accum != Memory && Accum != ComplexX87) &&
         "Invalid accumulated classification during merge.");
  if (Accum == Field || Field == NoClass)
    return Accum;
  if (Field == Memory)
    return Memory;
  if (Accum == NoClass)
    return Field;
  if (Accum == Integer || Field == Integer)
    return Integer;
  if (Field == X87 || Field == X87Up || Field == ComplexX87 ||
      Accum == X87 || Accum == X87Up)
    return Memory;
  return SSE;
}

void X86_64ABIInfo::classify(QualType Ty, uint64_t OffsetBase,
                             Class &Lo, Class &Hi) const {
  // FIXME: This code can be simplified by introducing a simple value class for
  // Class pairs with appropriate constructor methods for the various
  // situations.

  // FIXME: Some of the split computations are wrong; unaligned vectors
  // shouldn't be passed in registers for example, so there is no chance they
  // can straddle an eightbyte. Verify & simplify.

  Lo = Hi = NoClass;

  Class &Current = OffsetBase < 64 ? Lo : Hi;
  Current = Memory;

  if (const BuiltinType *BT = Ty->getAs<BuiltinType>()) {
    BuiltinType::Kind k = BT->getKind();

    if (k == BuiltinType::Void) {
      Current = NoClass;
    } else if (k == BuiltinType::Int128 || k == BuiltinType::UInt128) {
      Lo = Integer;
      Hi = Integer;
    } else if (k >= BuiltinType::Bool && k <= BuiltinType::LongLong) {
      Current = Integer;
    } else if (k == BuiltinType::Float || k == BuiltinType::Double) {
      Current = SSE;
    } else if (k == BuiltinType::LongDouble) {
      Lo = X87;
      Hi = X87Up;
    }
    // FIXME: _Decimal32 and _Decimal64 are SSE.
    // FIXME: _float128 and _Decimal128 are (SSE, SSEUp).
    return;
  }

  if (const EnumType *ET = Ty->getAs<EnumType>()) {
    // Classify the underlying integer type.
    classify(ET->getDecl()->getIntegerType(), OffsetBase, Lo, Hi);
    return;
  }

  if (Ty->hasPointerRepresentation()) {
    Current = Integer;
    return;
  }

  if (Ty->isMemberPointerType()) {
    if (Ty->isMemberFunctionPointerType())
      Lo = Hi = Integer;
    else
      Current = Integer;
    return;
  }

  if (const VectorType *VT = Ty->getAs<VectorType>()) {
    uint64_t Size = getContext().getTypeSize(VT);
    if (Size == 32) {
      // gcc passes all <4 x char>, <2 x short>, <1 x int>, <1 x
      // float> as integer.
      Current = Integer;

      // If this type crosses an eightbyte boundary, it should be
      // split.
      uint64_t EB_Real = (OffsetBase) / 64;
      uint64_t EB_Imag = (OffsetBase + Size - 1) / 64;
      if (EB_Real != EB_Imag)
        Hi = Lo;
    } else if (Size == 64) {
      // gcc passes <1 x double> in memory. :(
      if (VT->getElementType()->isSpecificBuiltinType(BuiltinType::Double))
        return;

      // gcc passes <1 x long long> as INTEGER.
      if (VT->getElementType()->isSpecificBuiltinType(BuiltinType::LongLong) ||
          VT->getElementType()->isSpecificBuiltinType(BuiltinType::ULongLong) ||
          VT->getElementType()->isSpecificBuiltinType(BuiltinType::Long) ||
          VT->getElementType()->isSpecificBuiltinType(BuiltinType::ULong))
        Current = Integer;
      else
        Current = SSE;

      // If this type crosses an eightbyte boundary, it should be
      // split.
      if (OffsetBase && OffsetBase != 64)
        Hi = Lo;
    } else if (Size == 128) {
      Lo = SSE;
      Hi = SSEUp;
    }
    return;
  }

  if (const ComplexType *CT = Ty->getAs<ComplexType>()) {
    QualType ET = getContext().getCanonicalType(CT->getElementType());

    uint64_t Size = getContext().getTypeSize(Ty);
    if (ET->isIntegralOrEnumerationType()) {
      if (Size <= 64)
        Current = Integer;
      else if (Size <= 128)
        Lo = Hi = Integer;
    } else if (ET == getContext().FloatTy)
      Current = SSE;
    else if (ET == getContext().DoubleTy)
      Lo = Hi = SSE;
    else if (ET == getContext().LongDoubleTy)
      Current = ComplexX87;

    // If this complex type crosses an eightbyte boundary then it
    // should be split.
    uint64_t EB_Real = (OffsetBase) / 64;
    uint64_t EB_Imag = (OffsetBase + getContext().getTypeSize(ET)) / 64;
    if (Hi == NoClass && EB_Real != EB_Imag)
      Hi = Lo;

    return;
  }

  if (const ConstantArrayType *AT = getContext().getAsConstantArrayType(Ty)) {
    // Arrays are treated like structures.

    uint64_t Size = getContext().getTypeSize(Ty);

    // AMD64-ABI 3.2.3p2: Rule 1. If the size of an object is larger
    // than two eightbytes, ..., it has class MEMORY.
    if (Size > 128)
      return;

    // AMD64-ABI 3.2.3p2: Rule 1. If ..., or it contains unaligned
    // fields, it has class MEMORY.
    //
    // Only need to check alignment of array base.
    if (OffsetBase % getContext().getTypeAlign(AT->getElementType()))
      return;

    // Otherwise implement simplified merge. We could be smarter about
    // this, but it isn't worth it and would be harder to verify.
    Current = NoClass;
    uint64_t EltSize = getContext().getTypeSize(AT->getElementType());
    uint64_t ArraySize = AT->getSize().getZExtValue();
    for (uint64_t i=0, Offset=OffsetBase; i<ArraySize; ++i, Offset += EltSize) {
      Class FieldLo, FieldHi;
      classify(AT->getElementType(), Offset, FieldLo, FieldHi);
      Lo = merge(Lo, FieldLo);
      Hi = merge(Hi, FieldHi);
      if (Lo == Memory || Hi == Memory)
        break;
    }

    // Do post merger cleanup (see below). Only case we worry about is Memory.
    if (Hi == Memory)
      Lo = Memory;
    assert((Hi != SSEUp || Lo == SSE) && "Invalid SSEUp array classification.");
    return;
  }

  if (const RecordType *RT = Ty->getAs<RecordType>()) {
    uint64_t Size = getContext().getTypeSize(Ty);

    // AMD64-ABI 3.2.3p2: Rule 1. If the size of an object is larger
    // than two eightbytes, ..., it has class MEMORY.
    if (Size > 128)
      return;

    // AMD64-ABI 3.2.3p2: Rule 2. If a C++ object has either a non-trivial
    // copy constructor or a non-trivial destructor, it is passed by invisible
    // reference.
    if (hasNonTrivialDestructorOrCopyConstructor(RT))
      return;

    const RecordDecl *RD = RT->getDecl();

    // Assume variable sized types are passed in memory.
    if (RD->hasFlexibleArrayMember())
      return;

    const ASTRecordLayout &Layout = getContext().getASTRecordLayout(RD);

    // Reset Lo class, this will be recomputed.
    Current = NoClass;

    // If this is a C++ record, classify the bases first.
    if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
      for (CXXRecordDecl::base_class_const_iterator i = CXXRD->bases_begin(),
             e = CXXRD->bases_end(); i != e; ++i) {
        assert(!i->isVirtual() && !i->getType()->isDependentType() &&
               "Unexpected base class!");
        const CXXRecordDecl *Base =
          cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());

        // Classify this field.
        //
        // AMD64-ABI 3.2.3p2: Rule 3. If the size of the aggregate exceeds a
        // single eightbyte, each is classified separately. Each eightbyte gets
        // initialized to class NO_CLASS.
        Class FieldLo, FieldHi;
        uint64_t Offset = OffsetBase + Layout.getBaseClassOffsetInBits(Base);
        classify(i->getType(), Offset, FieldLo, FieldHi);
        Lo = merge(Lo, FieldLo);
        Hi = merge(Hi, FieldHi);
        if (Lo == Memory || Hi == Memory)
          break;
      }
    }

    // Classify the fields one at a time, merging the results.
    unsigned idx = 0;
    for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
           i != e; ++i, ++idx) {
      uint64_t Offset = OffsetBase + Layout.getFieldOffset(idx);
      bool BitField = i->isBitField();

      // AMD64-ABI 3.2.3p2: Rule 1. If ..., or it contains unaligned
      // fields, it has class MEMORY.
      //
      // Note, skip this test for bit-fields, see below.
      if (!BitField && Offset % getContext().getTypeAlign(i->getType())) {
        Lo = Memory;
        return;
      }

      // Classify this field.
      //
      // AMD64-ABI 3.2.3p2: Rule 3. If the size of the aggregate
      // exceeds a single eightbyte, each is classified
      // separately. Each eightbyte gets initialized to class
      // NO_CLASS.
      Class FieldLo, FieldHi;

      // Bit-fields require special handling, they do not force the
      // structure to be passed in memory even if unaligned, and
      // therefore they can straddle an eightbyte.
      if (BitField) {
        // Ignore padding bit-fields.
        if (i->isUnnamedBitfield())
          continue;

        uint64_t Offset = OffsetBase + Layout.getFieldOffset(idx);
        uint64_t Size =
          i->getBitWidth()->EvaluateAsInt(getContext()).getZExtValue();

        uint64_t EB_Lo = Offset / 64;
        uint64_t EB_Hi = (Offset + Size - 1) / 64;
        FieldLo = FieldHi = NoClass;
        if (EB_Lo) {
          assert(EB_Hi == EB_Lo && "Invalid classification, type > 16 bytes.");
          FieldLo = NoClass;
          FieldHi = Integer;
        } else {
          FieldLo = Integer;
          FieldHi = EB_Hi ? Integer : NoClass;
        }
      } else
        classify(i->getType(), Offset, FieldLo, FieldHi);
      Lo = merge(Lo, FieldLo);
      Hi = merge(Hi, FieldHi);
      if (Lo == Memory || Hi == Memory)
        break;
    }

    // AMD64-ABI 3.2.3p2: Rule 5. Then a post merger cleanup is done:
    //
    // (a) If one of the classes is MEMORY, the whole argument is
    // passed in memory.
    //
    // (b) If X87UP is not preceded by X87, the whole argument is 
    // passed in memory.
    // 
    // (c) If the size of the aggregate exceeds two eightbytes and the first
    // eight-byte isn’t SSE or any other eightbyte isn’t SSEUP, the whole 
    // argument is passed in memory.
    // 
    // (d) If SSEUP is not preceded by SSE or SSEUP, it is converted to SSE.
    //
    // Some of these are enforced by the merging logic.  Others can arise
    // only with unions; for example:
    //   union { _Complex double; unsigned; }
    //
    // Note that clauses (b) and (c) were added in 0.98.
    if (Hi == Memory)
      Lo = Memory;
    if (Hi == X87Up && Lo != X87 && honorsRevision0_98())
      Lo = Memory;
    if (Hi == SSEUp && Lo != SSE)
      Hi = SSE;
  }
}

ABIArgInfo X86_64ABIInfo::getIndirectReturnResult(QualType Ty) const {
  // If this is a scalar LLVM value then assume LLVM will pass it in the right
  // place naturally.
  if (!isAggregateTypeForABI(Ty)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = Ty->getAs<EnumType>())
      Ty = EnumTy->getDecl()->getIntegerType();

    return (Ty->isPromotableIntegerType() ?
            ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  }

  return ABIArgInfo::getIndirect(0);
}

ABIArgInfo X86_64ABIInfo::getIndirectResult(QualType Ty) const {
  // If this is a scalar LLVM value then assume LLVM will pass it in the right
  // place naturally.
  if (!isAggregateTypeForABI(Ty)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = Ty->getAs<EnumType>())
      Ty = EnumTy->getDecl()->getIntegerType();

    return (Ty->isPromotableIntegerType() ?
            ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  }

  if (isRecordWithNonTrivialDestructorOrCopyConstructor(Ty))
    return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

  // Compute the byval alignment. We specify the alignment of the byval in all
  // cases so that the mid-level optimizer knows the alignment of the byval.
  unsigned Align = std::max(getContext().getTypeAlign(Ty) / 8, 8U);
  return ABIArgInfo::getIndirect(Align);
}

/// Get16ByteVectorType - The ABI specifies that a value should be passed in an
/// full vector XMM register.  Pick an LLVM IR type that will be passed as a
/// vector register.
const llvm::Type *X86_64ABIInfo::Get16ByteVectorType(QualType Ty) const {
  const llvm::Type *IRType = CGT.ConvertTypeRecursive(Ty);

  // Wrapper structs that just contain vectors are passed just like vectors,
  // strip them off if present.
  const llvm::StructType *STy = dyn_cast<llvm::StructType>(IRType);
  while (STy && STy->getNumElements() == 1) {
    IRType = STy->getElementType(0);
    STy = dyn_cast<llvm::StructType>(IRType);
  }

  // If the preferred type is a 16-byte vector, prefer to pass it.
  if (const llvm::VectorType *VT = dyn_cast<llvm::VectorType>(IRType)){
    const llvm::Type *EltTy = VT->getElementType();
    if (VT->getBitWidth() == 128 &&
        (EltTy->isFloatTy() || EltTy->isDoubleTy() ||
         EltTy->isIntegerTy(8) || EltTy->isIntegerTy(16) ||
         EltTy->isIntegerTy(32) || EltTy->isIntegerTy(64) ||
         EltTy->isIntegerTy(128)))
      return VT;
  }

  return llvm::VectorType::get(llvm::Type::getDoubleTy(getVMContext()), 2);
}

/// BitsContainNoUserData - Return true if the specified [start,end) bit range
/// is known to either be off the end of the specified type or being in
/// alignment padding.  The user type specified is known to be at most 128 bits
/// in size, and have passed through X86_64ABIInfo::classify with a successful
/// classification that put one of the two halves in the INTEGER class.
///
/// It is conservatively correct to return false.
static bool BitsContainNoUserData(QualType Ty, unsigned StartBit,
                                  unsigned EndBit, ASTContext &Context) {
  // If the bytes being queried are off the end of the type, there is no user
  // data hiding here.  This handles analysis of builtins, vectors and other
  // types that don't contain interesting padding.
  unsigned TySize = (unsigned)Context.getTypeSize(Ty);
  if (TySize <= StartBit)
    return true;

  if (const ConstantArrayType *AT = Context.getAsConstantArrayType(Ty)) {
    unsigned EltSize = (unsigned)Context.getTypeSize(AT->getElementType());
    unsigned NumElts = (unsigned)AT->getSize().getZExtValue();

    // Check each element to see if the element overlaps with the queried range.
    for (unsigned i = 0; i != NumElts; ++i) {
      // If the element is after the span we care about, then we're done..
      unsigned EltOffset = i*EltSize;
      if (EltOffset >= EndBit) break;

      unsigned EltStart = EltOffset < StartBit ? StartBit-EltOffset :0;
      if (!BitsContainNoUserData(AT->getElementType(), EltStart,
                                 EndBit-EltOffset, Context))
        return false;
    }
    // If it overlaps no elements, then it is safe to process as padding.
    return true;
  }

  if (const RecordType *RT = Ty->getAs<RecordType>()) {
    const RecordDecl *RD = RT->getDecl();
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

    // If this is a C++ record, check the bases first.
    if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
      for (CXXRecordDecl::base_class_const_iterator i = CXXRD->bases_begin(),
           e = CXXRD->bases_end(); i != e; ++i) {
        assert(!i->isVirtual() && !i->getType()->isDependentType() &&
               "Unexpected base class!");
        const CXXRecordDecl *Base =
          cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());

        // If the base is after the span we care about, ignore it.
        unsigned BaseOffset = (unsigned)Layout.getBaseClassOffsetInBits(Base);
        if (BaseOffset >= EndBit) continue;

        unsigned BaseStart = BaseOffset < StartBit ? StartBit-BaseOffset :0;
        if (!BitsContainNoUserData(i->getType(), BaseStart,
                                   EndBit-BaseOffset, Context))
          return false;
      }
    }

    // Verify that no field has data that overlaps the region of interest.  Yes
    // this could be sped up a lot by being smarter about queried fields,
    // however we're only looking at structs up to 16 bytes, so we don't care
    // much.
    unsigned idx = 0;
    for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
         i != e; ++i, ++idx) {
      unsigned FieldOffset = (unsigned)Layout.getFieldOffset(idx);

      // If we found a field after the region we care about, then we're done.
      if (FieldOffset >= EndBit) break;

      unsigned FieldStart = FieldOffset < StartBit ? StartBit-FieldOffset :0;
      if (!BitsContainNoUserData(i->getType(), FieldStart, EndBit-FieldOffset,
                                 Context))
        return false;
    }

    // If nothing in this record overlapped the area of interest, then we're
    // clean.
    return true;
  }

  return false;
}

/// ContainsFloatAtOffset - Return true if the specified LLVM IR type has a
/// float member at the specified offset.  For example, {int,{float}} has a
/// float at offset 4.  It is conservatively correct for this routine to return
/// false.
static bool ContainsFloatAtOffset(const llvm::Type *IRType, unsigned IROffset,
                                  const llvm::TargetData &TD) {
  // Base case if we find a float.
  if (IROffset == 0 && IRType->isFloatTy())
    return true;

  // If this is a struct, recurse into the field at the specified offset.
  if (const llvm::StructType *STy = dyn_cast<llvm::StructType>(IRType)) {
    const llvm::StructLayout *SL = TD.getStructLayout(STy);
    unsigned Elt = SL->getElementContainingOffset(IROffset);
    IROffset -= SL->getElementOffset(Elt);
    return ContainsFloatAtOffset(STy->getElementType(Elt), IROffset, TD);
  }

  // If this is an array, recurse into the field at the specified offset.
  if (const llvm::ArrayType *ATy = dyn_cast<llvm::ArrayType>(IRType)) {
    const llvm::Type *EltTy = ATy->getElementType();
    unsigned EltSize = TD.getTypeAllocSize(EltTy);
    IROffset -= IROffset/EltSize*EltSize;
    return ContainsFloatAtOffset(EltTy, IROffset, TD);
  }

  return false;
}


/// GetSSETypeAtOffset - Return a type that will be passed by the backend in the
/// low 8 bytes of an XMM register, corresponding to the SSE class.
const llvm::Type *X86_64ABIInfo::
GetSSETypeAtOffset(const llvm::Type *IRType, unsigned IROffset,
                   QualType SourceTy, unsigned SourceOffset) const {
  // The only three choices we have are either double, <2 x float>, or float. We
  // pass as float if the last 4 bytes is just padding.  This happens for
  // structs that contain 3 floats.
  if (BitsContainNoUserData(SourceTy, SourceOffset*8+32,
                            SourceOffset*8+64, getContext()))
    return llvm::Type::getFloatTy(getVMContext());

  // We want to pass as <2 x float> if the LLVM IR type contains a float at
  // offset+0 and offset+4.  Walk the LLVM IR type to find out if this is the
  // case.
  if (ContainsFloatAtOffset(IRType, IROffset, getTargetData()) &&
      ContainsFloatAtOffset(IRType, IROffset+4, getTargetData()))
    return llvm::VectorType::get(llvm::Type::getFloatTy(getVMContext()), 2);

  return llvm::Type::getDoubleTy(getVMContext());
}


/// GetINTEGERTypeAtOffset - The ABI specifies that a value should be passed in
/// an 8-byte GPR.  This means that we either have a scalar or we are talking
/// about the high or low part of an up-to-16-byte struct.  This routine picks
/// the best LLVM IR type to represent this, which may be i64 or may be anything
/// else that the backend will pass in a GPR that works better (e.g. i8, %foo*,
/// etc).
///
/// PrefType is an LLVM IR type that corresponds to (part of) the IR type for
/// the source type.  IROffset is an offset in bytes into the LLVM IR type that
/// the 8-byte value references.  PrefType may be null.
///
/// SourceTy is the source level type for the entire argument.  SourceOffset is
/// an offset into this that we're processing (which is always either 0 or 8).
///
const llvm::Type *X86_64ABIInfo::
GetINTEGERTypeAtOffset(const llvm::Type *IRType, unsigned IROffset,
                       QualType SourceTy, unsigned SourceOffset) const {
  // If we're dealing with an un-offset LLVM IR type, then it means that we're
  // returning an 8-byte unit starting with it.  See if we can safely use it.
  if (IROffset == 0) {
    // Pointers and int64's always fill the 8-byte unit.
    if (isa<llvm::PointerType>(IRType) || IRType->isIntegerTy(64))
      return IRType;

    // If we have a 1/2/4-byte integer, we can use it only if the rest of the
    // goodness in the source type is just tail padding.  This is allowed to
    // kick in for struct {double,int} on the int, but not on
    // struct{double,int,int} because we wouldn't return the second int.  We
    // have to do this analysis on the source type because we can't depend on
    // unions being lowered a specific way etc.
    if (IRType->isIntegerTy(8) || IRType->isIntegerTy(16) ||
        IRType->isIntegerTy(32)) {
      unsigned BitWidth = cast<llvm::IntegerType>(IRType)->getBitWidth();

      if (BitsContainNoUserData(SourceTy, SourceOffset*8+BitWidth,
                                SourceOffset*8+64, getContext()))
        return IRType;
    }
  }

  if (const llvm::StructType *STy = dyn_cast<llvm::StructType>(IRType)) {
    // If this is a struct, recurse into the field at the specified offset.
    const llvm::StructLayout *SL = getTargetData().getStructLayout(STy);
    if (IROffset < SL->getSizeInBytes()) {
      unsigned FieldIdx = SL->getElementContainingOffset(IROffset);
      IROffset -= SL->getElementOffset(FieldIdx);

      return GetINTEGERTypeAtOffset(STy->getElementType(FieldIdx), IROffset,
                                    SourceTy, SourceOffset);
    }
  }

  if (const llvm::ArrayType *ATy = dyn_cast<llvm::ArrayType>(IRType)) {
    const llvm::Type *EltTy = ATy->getElementType();
    unsigned EltSize = getTargetData().getTypeAllocSize(EltTy);
    unsigned EltOffset = IROffset/EltSize*EltSize;
    return GetINTEGERTypeAtOffset(EltTy, IROffset-EltOffset, SourceTy,
                                  SourceOffset);
  }

  // Okay, we don't have any better idea of what to pass, so we pass this in an
  // integer register that isn't too big to fit the rest of the struct.
  unsigned TySizeInBytes =
    (unsigned)getContext().getTypeSizeInChars(SourceTy).getQuantity();

  assert(TySizeInBytes != SourceOffset && "Empty field?");

  // It is always safe to classify this as an integer type up to i64 that
  // isn't larger than the structure.
  return llvm::IntegerType::get(getVMContext(),
                                std::min(TySizeInBytes-SourceOffset, 8U)*8);
}


/// GetX86_64ByValArgumentPair - Given a high and low type that can ideally
/// be used as elements of a two register pair to pass or return, return a
/// first class aggregate to represent them.  For example, if the low part of
/// a by-value argument should be passed as i32* and the high part as float,
/// return {i32*, float}.
static const llvm::Type *
GetX86_64ByValArgumentPair(const llvm::Type *Lo, const llvm::Type *Hi,
                           const llvm::TargetData &TD) {
  // In order to correctly satisfy the ABI, we need to the high part to start
  // at offset 8.  If the high and low parts we inferred are both 4-byte types
  // (e.g. i32 and i32) then the resultant struct type ({i32,i32}) won't have
  // the second element at offset 8.  Check for this:
  unsigned LoSize = (unsigned)TD.getTypeAllocSize(Lo);
  unsigned HiAlign = TD.getABITypeAlignment(Hi);
  unsigned HiStart = llvm::TargetData::RoundUpAlignment(LoSize, HiAlign);
  assert(HiStart != 0 && HiStart <= 8 && "Invalid x86-64 argument pair!");

  // To handle this, we have to increase the size of the low part so that the
  // second element will start at an 8 byte offset.  We can't increase the size
  // of the second element because it might make us access off the end of the
  // struct.
  if (HiStart != 8) {
    // There are only two sorts of types the ABI generation code can produce for
    // the low part of a pair that aren't 8 bytes in size: float or i8/i16/i32.
    // Promote these to a larger type.
    if (Lo->isFloatTy())
      Lo = llvm::Type::getDoubleTy(Lo->getContext());
    else {
      assert(Lo->isIntegerTy() && "Invalid/unknown lo type");
      Lo = llvm::Type::getInt64Ty(Lo->getContext());
    }
  }

  const llvm::StructType *Result =
    llvm::StructType::get(Lo->getContext(), Lo, Hi, NULL);


  // Verify that the second element is at an 8-byte offset.
  assert(TD.getStructLayout(Result)->getElementOffset(1) == 8 &&
         "Invalid x86-64 argument pair!");
  return Result;
}

ABIArgInfo X86_64ABIInfo::
classifyReturnType(QualType RetTy) const {
  // AMD64-ABI 3.2.3p4: Rule 1. Classify the return type with the
  // classification algorithm.
  X86_64ABIInfo::Class Lo, Hi;
  classify(RetTy, 0, Lo, Hi);

  // Check some invariants.
  assert((Hi != Memory || Lo == Memory) && "Invalid memory classification.");
  assert((Hi != SSEUp || Lo == SSE) && "Invalid SSEUp classification.");

  const llvm::Type *ResType = 0;
  switch (Lo) {
  case NoClass:
    if (Hi == NoClass)
      return ABIArgInfo::getIgnore();
    // If the low part is just padding, it takes no register, leave ResType
    // null.
    assert((Hi == SSE || Hi == Integer || Hi == X87Up) &&
           "Unknown missing lo part");
    break;

  case SSEUp:
  case X87Up:
    assert(0 && "Invalid classification for lo word.");

    // AMD64-ABI 3.2.3p4: Rule 2. Types of class memory are returned via
    // hidden argument.
  case Memory:
    return getIndirectReturnResult(RetTy);

    // AMD64-ABI 3.2.3p4: Rule 3. If the class is INTEGER, the next
    // available register of the sequence %rax, %rdx is used.
  case Integer:
    ResType = GetINTEGERTypeAtOffset(CGT.ConvertTypeRecursive(RetTy), 0,
                                     RetTy, 0);

    // If we have a sign or zero extended integer, make sure to return Extend
    // so that the parameter gets the right LLVM IR attributes.
    if (Hi == NoClass && isa<llvm::IntegerType>(ResType)) {
      // Treat an enum type as its underlying type.
      if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
        RetTy = EnumTy->getDecl()->getIntegerType();

      if (RetTy->isIntegralOrEnumerationType() &&
          RetTy->isPromotableIntegerType())
        return ABIArgInfo::getExtend();
    }
    break;

    // AMD64-ABI 3.2.3p4: Rule 4. If the class is SSE, the next
    // available SSE register of the sequence %xmm0, %xmm1 is used.
  case SSE:
    ResType = GetSSETypeAtOffset(CGT.ConvertTypeRecursive(RetTy), 0, RetTy, 0);
    break;

    // AMD64-ABI 3.2.3p4: Rule 6. If the class is X87, the value is
    // returned on the X87 stack in %st0 as 80-bit x87 number.
  case X87:
    ResType = llvm::Type::getX86_FP80Ty(getVMContext());
    break;

    // AMD64-ABI 3.2.3p4: Rule 8. If the class is COMPLEX_X87, the real
    // part of the value is returned in %st0 and the imaginary part in
    // %st1.
  case ComplexX87:
    assert(Hi == ComplexX87 && "Unexpected ComplexX87 classification.");
    ResType = llvm::StructType::get(getVMContext(),
                                    llvm::Type::getX86_FP80Ty(getVMContext()),
                                    llvm::Type::getX86_FP80Ty(getVMContext()),
                                    NULL);
    break;
  }

  const llvm::Type *HighPart = 0;
  switch (Hi) {
    // Memory was handled previously and X87 should
    // never occur as a hi class.
  case Memory:
  case X87:
    assert(0 && "Invalid classification for hi word.");

  case ComplexX87: // Previously handled.
  case NoClass:
    break;

  case Integer:
    HighPart = GetINTEGERTypeAtOffset(CGT.ConvertTypeRecursive(RetTy),
                                      8, RetTy, 8);
    if (Lo == NoClass)  // Return HighPart at offset 8 in memory.
      return ABIArgInfo::getDirect(HighPart, 8);
    break;
  case SSE:
    HighPart = GetSSETypeAtOffset(CGT.ConvertTypeRecursive(RetTy), 8, RetTy, 8);
    if (Lo == NoClass)  // Return HighPart at offset 8 in memory.
      return ABIArgInfo::getDirect(HighPart, 8);
    break;

    // AMD64-ABI 3.2.3p4: Rule 5. If the class is SSEUP, the eightbyte
    // is passed in the upper half of the last used SSE register.
    //
    // SSEUP should always be preceded by SSE, just widen.
  case SSEUp:
    assert(Lo == SSE && "Unexpected SSEUp classification.");
    ResType = Get16ByteVectorType(RetTy);
    break;

    // AMD64-ABI 3.2.3p4: Rule 7. If the class is X87UP, the value is
    // returned together with the previous X87 value in %st0.
  case X87Up:
    // If X87Up is preceded by X87, we don't need to do
    // anything. However, in some cases with unions it may not be
    // preceded by X87. In such situations we follow gcc and pass the
    // extra bits in an SSE reg.
    if (Lo != X87) {
      HighPart = GetSSETypeAtOffset(CGT.ConvertTypeRecursive(RetTy),
                                    8, RetTy, 8);
      if (Lo == NoClass)  // Return HighPart at offset 8 in memory.
        return ABIArgInfo::getDirect(HighPart, 8);
    }
    break;
  }

  // If a high part was specified, merge it together with the low part.  It is
  // known to pass in the high eightbyte of the result.  We do this by forming a
  // first class struct aggregate with the high and low part: {low, high}
  if (HighPart)
    ResType = GetX86_64ByValArgumentPair(ResType, HighPart, getTargetData());

  return ABIArgInfo::getDirect(ResType);
}

ABIArgInfo X86_64ABIInfo::classifyArgumentType(QualType Ty, unsigned &neededInt,
                                               unsigned &neededSSE) const {
  X86_64ABIInfo::Class Lo, Hi;
  classify(Ty, 0, Lo, Hi);

  // Check some invariants.
  // FIXME: Enforce these by construction.
  assert((Hi != Memory || Lo == Memory) && "Invalid memory classification.");
  assert((Hi != SSEUp || Lo == SSE) && "Invalid SSEUp classification.");

  neededInt = 0;
  neededSSE = 0;
  const llvm::Type *ResType = 0;
  switch (Lo) {
  case NoClass:
    if (Hi == NoClass)
      return ABIArgInfo::getIgnore();
    // If the low part is just padding, it takes no register, leave ResType
    // null.
    assert((Hi == SSE || Hi == Integer || Hi == X87Up) &&
           "Unknown missing lo part");
    break;

    // AMD64-ABI 3.2.3p3: Rule 1. If the class is MEMORY, pass the argument
    // on the stack.
  case Memory:

    // AMD64-ABI 3.2.3p3: Rule 5. If the class is X87, X87UP or
    // COMPLEX_X87, it is passed in memory.
  case X87:
  case ComplexX87:
    return getIndirectResult(Ty);

  case SSEUp:
  case X87Up:
    assert(0 && "Invalid classification for lo word.");

    // AMD64-ABI 3.2.3p3: Rule 2. If the class is INTEGER, the next
    // available register of the sequence %rdi, %rsi, %rdx, %rcx, %r8
    // and %r9 is used.
  case Integer:
    ++neededInt;

    // Pick an 8-byte type based on the preferred type.
    ResType = GetINTEGERTypeAtOffset(CGT.ConvertTypeRecursive(Ty), 0, Ty, 0);

    // If we have a sign or zero extended integer, make sure to return Extend
    // so that the parameter gets the right LLVM IR attributes.
    if (Hi == NoClass && isa<llvm::IntegerType>(ResType)) {
      // Treat an enum type as its underlying type.
      if (const EnumType *EnumTy = Ty->getAs<EnumType>())
        Ty = EnumTy->getDecl()->getIntegerType();

      if (Ty->isIntegralOrEnumerationType() &&
          Ty->isPromotableIntegerType())
        return ABIArgInfo::getExtend();
    }

    break;

    // AMD64-ABI 3.2.3p3: Rule 3. If the class is SSE, the next
    // available SSE register is used, the registers are taken in the
    // order from %xmm0 to %xmm7.
  case SSE: {
    const llvm::Type *IRType = CGT.ConvertTypeRecursive(Ty);
    if (Hi != NoClass || !UseX86_MMXType(IRType))
      ResType = GetSSETypeAtOffset(IRType, 0, Ty, 0);
    else
      // This is an MMX type. Treat it as such.
      ResType = llvm::Type::getX86_MMXTy(getVMContext());

    ++neededSSE;
    break;
  }
  }

  const llvm::Type *HighPart = 0;
  switch (Hi) {
    // Memory was handled previously, ComplexX87 and X87 should
    // never occur as hi classes, and X87Up must be preceded by X87,
    // which is passed in memory.
  case Memory:
  case X87:
  case ComplexX87:
    assert(0 && "Invalid classification for hi word.");
    break;

  case NoClass: break;

  case Integer:
    ++neededInt;
    // Pick an 8-byte type based on the preferred type.
    HighPart = GetINTEGERTypeAtOffset(CGT.ConvertTypeRecursive(Ty), 8, Ty, 8);

    if (Lo == NoClass)  // Pass HighPart at offset 8 in memory.
      return ABIArgInfo::getDirect(HighPart, 8);
    break;

    // X87Up generally doesn't occur here (long double is passed in
    // memory), except in situations involving unions.
  case X87Up:
  case SSE:
    HighPart = GetSSETypeAtOffset(CGT.ConvertTypeRecursive(Ty), 8, Ty, 8);

    if (Lo == NoClass)  // Pass HighPart at offset 8 in memory.
      return ABIArgInfo::getDirect(HighPart, 8);

    ++neededSSE;
    break;

    // AMD64-ABI 3.2.3p3: Rule 4. If the class is SSEUP, the
    // eightbyte is passed in the upper half of the last used SSE
    // register.  This only happens when 128-bit vectors are passed.
  case SSEUp:
    assert(Lo == SSE && "Unexpected SSEUp classification");
    ResType = Get16ByteVectorType(Ty);
    break;
  }

  // If a high part was specified, merge it together with the low part.  It is
  // known to pass in the high eightbyte of the result.  We do this by forming a
  // first class struct aggregate with the high and low part: {low, high}
  if (HighPart)
    ResType = GetX86_64ByValArgumentPair(ResType, HighPart, getTargetData());

  return ABIArgInfo::getDirect(ResType);
}

void X86_64ABIInfo::computeInfo(CGFunctionInfo &FI) const {

  FI.getReturnInfo() = classifyReturnType(FI.getReturnType());

  // Keep track of the number of assigned registers.
  unsigned freeIntRegs = 6, freeSSERegs = 8;

  // If the return value is indirect, then the hidden argument is consuming one
  // integer register.
  if (FI.getReturnInfo().isIndirect())
    --freeIntRegs;

  // AMD64-ABI 3.2.3p3: Once arguments are classified, the registers
  // get assigned (in left-to-right order) for passing as follows...
  for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
       it != ie; ++it) {
    unsigned neededInt, neededSSE;
    it->info = classifyArgumentType(it->type, neededInt, neededSSE);

    // AMD64-ABI 3.2.3p3: If there are no registers available for any
    // eightbyte of an argument, the whole argument is passed on the
    // stack. If registers have already been assigned for some
    // eightbytes of such an argument, the assignments get reverted.
    if (freeIntRegs >= neededInt && freeSSERegs >= neededSSE) {
      freeIntRegs -= neededInt;
      freeSSERegs -= neededSSE;
    } else {
      it->info = getIndirectResult(it->type);
    }
  }
}

static llvm::Value *EmitVAArgFromMemory(llvm::Value *VAListAddr,
                                        QualType Ty,
                                        CodeGenFunction &CGF) {
  llvm::Value *overflow_arg_area_p =
    CGF.Builder.CreateStructGEP(VAListAddr, 2, "overflow_arg_area_p");
  llvm::Value *overflow_arg_area =
    CGF.Builder.CreateLoad(overflow_arg_area_p, "overflow_arg_area");

  // AMD64-ABI 3.5.7p5: Step 7. Align l->overflow_arg_area upwards to a 16
  // byte boundary if alignment needed by type exceeds 8 byte boundary.
  uint64_t Align = CGF.getContext().getTypeAlign(Ty) / 8;
  if (Align > 8) {
    // Note that we follow the ABI & gcc here, even though the type
    // could in theory have an alignment greater than 16. This case
    // shouldn't ever matter in practice.

    // overflow_arg_area = (overflow_arg_area + 15) & ~15;
    llvm::Value *Offset =
      llvm::ConstantInt::get(CGF.Int32Ty, 15);
    overflow_arg_area = CGF.Builder.CreateGEP(overflow_arg_area, Offset);
    llvm::Value *AsInt = CGF.Builder.CreatePtrToInt(overflow_arg_area,
                                                    CGF.Int64Ty);
    llvm::Value *Mask = llvm::ConstantInt::get(CGF.Int64Ty, ~15LL);
    overflow_arg_area =
      CGF.Builder.CreateIntToPtr(CGF.Builder.CreateAnd(AsInt, Mask),
                                 overflow_arg_area->getType(),
                                 "overflow_arg_area.align");
  }

  // AMD64-ABI 3.5.7p5: Step 8. Fetch type from l->overflow_arg_area.
  const llvm::Type *LTy = CGF.ConvertTypeForMem(Ty);
  llvm::Value *Res =
    CGF.Builder.CreateBitCast(overflow_arg_area,
                              llvm::PointerType::getUnqual(LTy));

  // AMD64-ABI 3.5.7p5: Step 9. Set l->overflow_arg_area to:
  // l->overflow_arg_area + sizeof(type).
  // AMD64-ABI 3.5.7p5: Step 10. Align l->overflow_arg_area upwards to
  // an 8 byte boundary.

  uint64_t SizeInBytes = (CGF.getContext().getTypeSize(Ty) + 7) / 8;
  llvm::Value *Offset =
      llvm::ConstantInt::get(CGF.Int32Ty, (SizeInBytes + 7)  & ~7);
  overflow_arg_area = CGF.Builder.CreateGEP(overflow_arg_area, Offset,
                                            "overflow_arg_area.next");
  CGF.Builder.CreateStore(overflow_arg_area, overflow_arg_area_p);

  // AMD64-ABI 3.5.7p5: Step 11. Return the fetched type.
  return Res;
}

llvm::Value *X86_64ABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                      CodeGenFunction &CGF) const {
  llvm::LLVMContext &VMContext = CGF.getLLVMContext();

  // Assume that va_list type is correct; should be pointer to LLVM type:
  // struct {
  //   i32 gp_offset;
  //   i32 fp_offset;
  //   i8* overflow_arg_area;
  //   i8* reg_save_area;
  // };
  unsigned neededInt, neededSSE;

  Ty = CGF.getContext().getCanonicalType(Ty);
  ABIArgInfo AI = classifyArgumentType(Ty, neededInt, neededSSE);

  // AMD64-ABI 3.5.7p5: Step 1. Determine whether type may be passed
  // in the registers. If not go to step 7.
  if (!neededInt && !neededSSE)
    return EmitVAArgFromMemory(VAListAddr, Ty, CGF);

  // AMD64-ABI 3.5.7p5: Step 2. Compute num_gp to hold the number of
  // general purpose registers needed to pass type and num_fp to hold
  // the number of floating point registers needed.

  // AMD64-ABI 3.5.7p5: Step 3. Verify whether arguments fit into
  // registers. In the case: l->gp_offset > 48 - num_gp * 8 or
  // l->fp_offset > 304 - num_fp * 16 go to step 7.
  //
  // NOTE: 304 is a typo, there are (6 * 8 + 8 * 16) = 176 bytes of
  // register save space).

  llvm::Value *InRegs = 0;
  llvm::Value *gp_offset_p = 0, *gp_offset = 0;
  llvm::Value *fp_offset_p = 0, *fp_offset = 0;
  if (neededInt) {
    gp_offset_p = CGF.Builder.CreateStructGEP(VAListAddr, 0, "gp_offset_p");
    gp_offset = CGF.Builder.CreateLoad(gp_offset_p, "gp_offset");
    InRegs = llvm::ConstantInt::get(CGF.Int32Ty, 48 - neededInt * 8);
    InRegs = CGF.Builder.CreateICmpULE(gp_offset, InRegs, "fits_in_gp");
  }

  if (neededSSE) {
    fp_offset_p = CGF.Builder.CreateStructGEP(VAListAddr, 1, "fp_offset_p");
    fp_offset = CGF.Builder.CreateLoad(fp_offset_p, "fp_offset");
    llvm::Value *FitsInFP =
      llvm::ConstantInt::get(CGF.Int32Ty, 176 - neededSSE * 16);
    FitsInFP = CGF.Builder.CreateICmpULE(fp_offset, FitsInFP, "fits_in_fp");
    InRegs = InRegs ? CGF.Builder.CreateAnd(InRegs, FitsInFP) : FitsInFP;
  }

  llvm::BasicBlock *InRegBlock = CGF.createBasicBlock("vaarg.in_reg");
  llvm::BasicBlock *InMemBlock = CGF.createBasicBlock("vaarg.in_mem");
  llvm::BasicBlock *ContBlock = CGF.createBasicBlock("vaarg.end");
  CGF.Builder.CreateCondBr(InRegs, InRegBlock, InMemBlock);

  // Emit code to load the value if it was passed in registers.

  CGF.EmitBlock(InRegBlock);

  // AMD64-ABI 3.5.7p5: Step 4. Fetch type from l->reg_save_area with
  // an offset of l->gp_offset and/or l->fp_offset. This may require
  // copying to a temporary location in case the parameter is passed
  // in different register classes or requires an alignment greater
  // than 8 for general purpose registers and 16 for XMM registers.
  //
  // FIXME: This really results in shameful code when we end up needing to
  // collect arguments from different places; often what should result in a
  // simple assembling of a structure from scattered addresses has many more
  // loads than necessary. Can we clean this up?
  const llvm::Type *LTy = CGF.ConvertTypeForMem(Ty);
  llvm::Value *RegAddr =
    CGF.Builder.CreateLoad(CGF.Builder.CreateStructGEP(VAListAddr, 3),
                           "reg_save_area");
  if (neededInt && neededSSE) {
    // FIXME: Cleanup.
    assert(AI.isDirect() && "Unexpected ABI info for mixed regs");
    const llvm::StructType *ST = cast<llvm::StructType>(AI.getCoerceToType());
    llvm::Value *Tmp = CGF.CreateTempAlloca(ST);
    assert(ST->getNumElements() == 2 && "Unexpected ABI info for mixed regs");
    const llvm::Type *TyLo = ST->getElementType(0);
    const llvm::Type *TyHi = ST->getElementType(1);
    assert((TyLo->isFPOrFPVectorTy() ^ TyHi->isFPOrFPVectorTy()) &&
           "Unexpected ABI info for mixed regs");
    const llvm::Type *PTyLo = llvm::PointerType::getUnqual(TyLo);
    const llvm::Type *PTyHi = llvm::PointerType::getUnqual(TyHi);
    llvm::Value *GPAddr = CGF.Builder.CreateGEP(RegAddr, gp_offset);
    llvm::Value *FPAddr = CGF.Builder.CreateGEP(RegAddr, fp_offset);
    llvm::Value *RegLoAddr = TyLo->isFloatingPointTy() ? FPAddr : GPAddr;
    llvm::Value *RegHiAddr = TyLo->isFloatingPointTy() ? GPAddr : FPAddr;
    llvm::Value *V =
      CGF.Builder.CreateLoad(CGF.Builder.CreateBitCast(RegLoAddr, PTyLo));
    CGF.Builder.CreateStore(V, CGF.Builder.CreateStructGEP(Tmp, 0));
    V = CGF.Builder.CreateLoad(CGF.Builder.CreateBitCast(RegHiAddr, PTyHi));
    CGF.Builder.CreateStore(V, CGF.Builder.CreateStructGEP(Tmp, 1));

    RegAddr = CGF.Builder.CreateBitCast(Tmp,
                                        llvm::PointerType::getUnqual(LTy));
  } else if (neededInt) {
    RegAddr = CGF.Builder.CreateGEP(RegAddr, gp_offset);
    RegAddr = CGF.Builder.CreateBitCast(RegAddr,
                                        llvm::PointerType::getUnqual(LTy));
  } else if (neededSSE == 1) {
    RegAddr = CGF.Builder.CreateGEP(RegAddr, fp_offset);
    RegAddr = CGF.Builder.CreateBitCast(RegAddr,
                                        llvm::PointerType::getUnqual(LTy));
  } else {
    assert(neededSSE == 2 && "Invalid number of needed registers!");
    // SSE registers are spaced 16 bytes apart in the register save
    // area, we need to collect the two eightbytes together.
    llvm::Value *RegAddrLo = CGF.Builder.CreateGEP(RegAddr, fp_offset);
    llvm::Value *RegAddrHi = CGF.Builder.CreateConstGEP1_32(RegAddrLo, 16);
    const llvm::Type *DoubleTy = llvm::Type::getDoubleTy(VMContext);
    const llvm::Type *DblPtrTy =
      llvm::PointerType::getUnqual(DoubleTy);
    const llvm::StructType *ST = llvm::StructType::get(VMContext, DoubleTy,
                                                       DoubleTy, NULL);
    llvm::Value *V, *Tmp = CGF.CreateTempAlloca(ST);
    V = CGF.Builder.CreateLoad(CGF.Builder.CreateBitCast(RegAddrLo,
                                                         DblPtrTy));
    CGF.Builder.CreateStore(V, CGF.Builder.CreateStructGEP(Tmp, 0));
    V = CGF.Builder.CreateLoad(CGF.Builder.CreateBitCast(RegAddrHi,
                                                         DblPtrTy));
    CGF.Builder.CreateStore(V, CGF.Builder.CreateStructGEP(Tmp, 1));
    RegAddr = CGF.Builder.CreateBitCast(Tmp,
                                        llvm::PointerType::getUnqual(LTy));
  }

  // AMD64-ABI 3.5.7p5: Step 5. Set:
  // l->gp_offset = l->gp_offset + num_gp * 8
  // l->fp_offset = l->fp_offset + num_fp * 16.
  if (neededInt) {
    llvm::Value *Offset = llvm::ConstantInt::get(CGF.Int32Ty, neededInt * 8);
    CGF.Builder.CreateStore(CGF.Builder.CreateAdd(gp_offset, Offset),
                            gp_offset_p);
  }
  if (neededSSE) {
    llvm::Value *Offset = llvm::ConstantInt::get(CGF.Int32Ty, neededSSE * 16);
    CGF.Builder.CreateStore(CGF.Builder.CreateAdd(fp_offset, Offset),
                            fp_offset_p);
  }
  CGF.EmitBranch(ContBlock);

  // Emit code to load the value if it was passed in memory.

  CGF.EmitBlock(InMemBlock);
  llvm::Value *MemAddr = EmitVAArgFromMemory(VAListAddr, Ty, CGF);

  // Return the appropriate result.

  CGF.EmitBlock(ContBlock);
  llvm::PHINode *ResAddr = CGF.Builder.CreatePHI(RegAddr->getType(), 2,
                                                 "vaarg.addr");
  ResAddr->addIncoming(RegAddr, InRegBlock);
  ResAddr->addIncoming(MemAddr, InMemBlock);
  return ResAddr;
}

ABIArgInfo WinX86_64ABIInfo::classify(QualType Ty) const {

  if (Ty->isVoidType())
    return ABIArgInfo::getIgnore();

  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getDecl()->getIntegerType();

  uint64_t Size = getContext().getTypeSize(Ty);

  if (const RecordType *RT = Ty->getAs<RecordType>()) {
    if (hasNonTrivialDestructorOrCopyConstructor(RT) ||
        RT->getDecl()->hasFlexibleArrayMember())
      return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

    // FIXME: mingw-w64-gcc emits 128-bit struct as i128
    if (Size == 128 &&
        getContext().Target.getTriple().getOS() == llvm::Triple::MinGW32)
      return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(),
                                                          Size));

    // MS x64 ABI requirement: "Any argument that doesn't fit in 8 bytes, or is
    // not 1, 2, 4, or 8 bytes, must be passed by reference."
    if (Size <= 64 &&
        (Size & (Size - 1)) == 0)
      return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(),
                                                          Size));

    return ABIArgInfo::getIndirect(0, /*ByVal=*/false);
  }

  if (Ty->isPromotableIntegerType())
    return ABIArgInfo::getExtend();

  return ABIArgInfo::getDirect();
}

void WinX86_64ABIInfo::computeInfo(CGFunctionInfo &FI) const {

  QualType RetTy = FI.getReturnType();
  FI.getReturnInfo() = classify(RetTy);

  for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
       it != ie; ++it)
    it->info = classify(it->type);
}

llvm::Value *WinX86_64ABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                      CodeGenFunction &CGF) const {
  const llvm::Type *BP = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  const llvm::Type *BPP = llvm::PointerType::getUnqual(BP);

  CGBuilderTy &Builder = CGF.Builder;
  llvm::Value *VAListAddrAsBPP = Builder.CreateBitCast(VAListAddr, BPP,
                                                       "ap");
  llvm::Value *Addr = Builder.CreateLoad(VAListAddrAsBPP, "ap.cur");
  llvm::Type *PTy =
    llvm::PointerType::getUnqual(CGF.ConvertType(Ty));
  llvm::Value *AddrTyped = Builder.CreateBitCast(Addr, PTy);

  uint64_t Offset =
    llvm::RoundUpToAlignment(CGF.getContext().getTypeSize(Ty) / 8, 8);
  llvm::Value *NextAddr =
    Builder.CreateGEP(Addr, llvm::ConstantInt::get(CGF.Int32Ty, Offset),
                      "ap.next");
  Builder.CreateStore(NextAddr, VAListAddrAsBPP);

  return AddrTyped;
}

// PowerPC-32

namespace {
class PPC32TargetCodeGenInfo : public DefaultTargetCodeGenInfo {
public:
  PPC32TargetCodeGenInfo(CodeGenTypes &CGT) : DefaultTargetCodeGenInfo(CGT) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &M) const {
    // This is recovered from gcc output.
    return 1; // r1 is the dedicated stack pointer
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const;
};

}

bool
PPC32TargetCodeGenInfo::initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                                                llvm::Value *Address) const {
  // This is calculated from the LLVM and GCC tables and verified
  // against gcc output.  AFAIK all ABIs use the same encoding.

  CodeGen::CGBuilderTy &Builder = CGF.Builder;
  llvm::LLVMContext &Context = CGF.getLLVMContext();

  const llvm::IntegerType *i8 = llvm::Type::getInt8Ty(Context);
  llvm::Value *Four8 = llvm::ConstantInt::get(i8, 4);
  llvm::Value *Eight8 = llvm::ConstantInt::get(i8, 8);
  llvm::Value *Sixteen8 = llvm::ConstantInt::get(i8, 16);

  // 0-31: r0-31, the 4-byte general-purpose registers
  AssignToArrayRange(Builder, Address, Four8, 0, 31);

  // 32-63: fp0-31, the 8-byte floating-point registers
  AssignToArrayRange(Builder, Address, Eight8, 32, 63);

  // 64-76 are various 4-byte special-purpose registers:
  // 64: mq
  // 65: lr
  // 66: ctr
  // 67: ap
  // 68-75 cr0-7
  // 76: xer
  AssignToArrayRange(Builder, Address, Four8, 64, 76);

  // 77-108: v0-31, the 16-byte vector registers
  AssignToArrayRange(Builder, Address, Sixteen8, 77, 108);

  // 109: vrsave
  // 110: vscr
  // 111: spe_acc
  // 112: spefscr
  // 113: sfp
  AssignToArrayRange(Builder, Address, Four8, 109, 113);

  return false;
}


//===----------------------------------------------------------------------===//
// ARM ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class ARMABIInfo : public ABIInfo {
public:
  enum ABIKind {
    APCS = 0,
    AAPCS = 1,
    AAPCS_VFP
  };

private:
  ABIKind Kind;

public:
  ARMABIInfo(CodeGenTypes &CGT, ABIKind _Kind) : ABIInfo(CGT), Kind(_Kind) {}

private:
  ABIKind getABIKind() const { return Kind; }

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType RetTy) const;

  virtual void computeInfo(CGFunctionInfo &FI) const;

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

class ARMTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  ARMTargetCodeGenInfo(CodeGenTypes &CGT, ARMABIInfo::ABIKind K)
    :TargetCodeGenInfo(new ARMABIInfo(CGT, K)) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &M) const {
    return 13;
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const {
    CodeGen::CGBuilderTy &Builder = CGF.Builder;
    llvm::LLVMContext &Context = CGF.getLLVMContext();

    const llvm::IntegerType *i8 = llvm::Type::getInt8Ty(Context);
    llvm::Value *Four8 = llvm::ConstantInt::get(i8, 4);

    // 0-15 are the 16 integer registers.
    AssignToArrayRange(Builder, Address, Four8, 0, 15);

    return false;
  }


};

}

void ARMABIInfo::computeInfo(CGFunctionInfo &FI) const {
  FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
  for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
       it != ie; ++it)
    it->info = classifyArgumentType(it->type);

  // Always honor user-specified calling convention.
  if (FI.getCallingConvention() != llvm::CallingConv::C)
    return;

  // Calling convention as default by an ABI.
  llvm::CallingConv::ID DefaultCC;
  llvm::StringRef Env = getContext().Target.getTriple().getEnvironmentName();
  if (Env == "gnueabi" || Env == "eabi")
    DefaultCC = llvm::CallingConv::ARM_AAPCS;
  else
    DefaultCC = llvm::CallingConv::ARM_APCS;

  // If user did not ask for specific calling convention explicitly (e.g. via
  // pcs attribute), set effective calling convention if it's different than ABI
  // default.
  switch (getABIKind()) {
  case APCS:
    if (DefaultCC != llvm::CallingConv::ARM_APCS)
      FI.setEffectiveCallingConvention(llvm::CallingConv::ARM_APCS);
    break;
  case AAPCS:
    if (DefaultCC != llvm::CallingConv::ARM_AAPCS)
      FI.setEffectiveCallingConvention(llvm::CallingConv::ARM_AAPCS);
    break;
  case AAPCS_VFP:
    if (DefaultCC != llvm::CallingConv::ARM_AAPCS_VFP)
      FI.setEffectiveCallingConvention(llvm::CallingConv::ARM_AAPCS_VFP);
    break;
  }
}

ABIArgInfo ARMABIInfo::classifyArgumentType(QualType Ty) const {
  if (!isAggregateTypeForABI(Ty)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = Ty->getAs<EnumType>())
      Ty = EnumTy->getDecl()->getIntegerType();

    return (Ty->isPromotableIntegerType() ?
            ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  }

  // Ignore empty records.
  if (isEmptyRecord(getContext(), Ty, true))
    return ABIArgInfo::getIgnore();

  // Structures with either a non-trivial destructor or a non-trivial
  // copy constructor are always indirect.
  if (isRecordWithNonTrivialDestructorOrCopyConstructor(Ty))
    return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

  // Otherwise, pass by coercing to a structure of the appropriate size.
  //
  // FIXME: This doesn't handle alignment > 64 bits.
  const llvm::Type* ElemTy;
  unsigned SizeRegs;
  if (getContext().getTypeSizeInChars(Ty) <= CharUnits::fromQuantity(64)) {
    ElemTy = llvm::Type::getInt32Ty(getVMContext());
    SizeRegs = (getContext().getTypeSize(Ty) + 31) / 32;
  } else if (getABIKind() == ARMABIInfo::APCS) {
    // Initial ARM ByVal support is APCS-only.
    return ABIArgInfo::getIndirect(0, /*ByVal=*/true);
  } else {
    // FIXME: This is kind of nasty... but there isn't much choice
    // because most of the ARM calling conventions don't yet support
    // byval.
    ElemTy = llvm::Type::getInt64Ty(getVMContext());
    SizeRegs = (getContext().getTypeSize(Ty) + 63) / 64;
  }

  const llvm::Type *STy =
    llvm::StructType::get(getVMContext(),
                          llvm::ArrayType::get(ElemTy, SizeRegs), NULL, NULL);
  return ABIArgInfo::getDirect(STy);
}

static bool isIntegerLikeType(QualType Ty, ASTContext &Context,
                              llvm::LLVMContext &VMContext) {
  // APCS, C Language Calling Conventions, Non-Simple Return Values: A structure
  // is called integer-like if its size is less than or equal to one word, and
  // the offset of each of its addressable sub-fields is zero.

  uint64_t Size = Context.getTypeSize(Ty);

  // Check that the type fits in a word.
  if (Size > 32)
    return false;

  // FIXME: Handle vector types!
  if (Ty->isVectorType())
    return false;

  // Float types are never treated as "integer like".
  if (Ty->isRealFloatingType())
    return false;

  // If this is a builtin or pointer type then it is ok.
  if (Ty->getAs<BuiltinType>() || Ty->isPointerType())
    return true;

  // Small complex integer types are "integer like".
  if (const ComplexType *CT = Ty->getAs<ComplexType>())
    return isIntegerLikeType(CT->getElementType(), Context, VMContext);

  // Single element and zero sized arrays should be allowed, by the definition
  // above, but they are not.

  // Otherwise, it must be a record type.
  const RecordType *RT = Ty->getAs<RecordType>();
  if (!RT) return false;

  // Ignore records with flexible arrays.
  const RecordDecl *RD = RT->getDecl();
  if (RD->hasFlexibleArrayMember())
    return false;

  // Check that all sub-fields are at offset 0, and are themselves "integer
  // like".
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  bool HadField = false;
  unsigned idx = 0;
  for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
       i != e; ++i, ++idx) {
    const FieldDecl *FD = *i;

    // Bit-fields are not addressable, we only need to verify they are "integer
    // like". We still have to disallow a subsequent non-bitfield, for example:
    //   struct { int : 0; int x }
    // is non-integer like according to gcc.
    if (FD->isBitField()) {
      if (!RD->isUnion())
        HadField = true;

      if (!isIntegerLikeType(FD->getType(), Context, VMContext))
        return false;

      continue;
    }

    // Check if this field is at offset 0.
    if (Layout.getFieldOffset(idx) != 0)
      return false;

    if (!isIntegerLikeType(FD->getType(), Context, VMContext))
      return false;

    // Only allow at most one field in a structure. This doesn't match the
    // wording above, but follows gcc in situations with a field following an
    // empty structure.
    if (!RD->isUnion()) {
      if (HadField)
        return false;

      HadField = true;
    }
  }

  return true;
}

ABIArgInfo ARMABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  // Large vector types should be returned via memory.
  if (RetTy->isVectorType() && getContext().getTypeSize(RetTy) > 128)
    return ABIArgInfo::getIndirect(0);

  if (!isAggregateTypeForABI(RetTy)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
      RetTy = EnumTy->getDecl()->getIntegerType();

    return (RetTy->isPromotableIntegerType() ?
            ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  }

  // Structures with either a non-trivial destructor or a non-trivial
  // copy constructor are always indirect.
  if (isRecordWithNonTrivialDestructorOrCopyConstructor(RetTy))
    return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

  // Are we following APCS?
  if (getABIKind() == APCS) {
    if (isEmptyRecord(getContext(), RetTy, false))
      return ABIArgInfo::getIgnore();

    // Complex types are all returned as packed integers.
    //
    // FIXME: Consider using 2 x vector types if the back end handles them
    // correctly.
    if (RetTy->isAnyComplexType())
      return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(),
                                              getContext().getTypeSize(RetTy)));

    // Integer like structures are returned in r0.
    if (isIntegerLikeType(RetTy, getContext(), getVMContext())) {
      // Return in the smallest viable integer type.
      uint64_t Size = getContext().getTypeSize(RetTy);
      if (Size <= 8)
        return ABIArgInfo::getDirect(llvm::Type::getInt8Ty(getVMContext()));
      if (Size <= 16)
        return ABIArgInfo::getDirect(llvm::Type::getInt16Ty(getVMContext()));
      return ABIArgInfo::getDirect(llvm::Type::getInt32Ty(getVMContext()));
    }

    // Otherwise return in memory.
    return ABIArgInfo::getIndirect(0);
  }

  // Otherwise this is an AAPCS variant.

  if (isEmptyRecord(getContext(), RetTy, true))
    return ABIArgInfo::getIgnore();

  // Aggregates <= 4 bytes are returned in r0; other aggregates
  // are returned indirectly.
  uint64_t Size = getContext().getTypeSize(RetTy);
  if (Size <= 32) {
    // Return in the smallest viable integer type.
    if (Size <= 8)
      return ABIArgInfo::getDirect(llvm::Type::getInt8Ty(getVMContext()));
    if (Size <= 16)
      return ABIArgInfo::getDirect(llvm::Type::getInt16Ty(getVMContext()));
    return ABIArgInfo::getDirect(llvm::Type::getInt32Ty(getVMContext()));
  }

  return ABIArgInfo::getIndirect(0);
}

llvm::Value *ARMABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                   CodeGenFunction &CGF) const {
  // FIXME: Need to handle alignment
  const llvm::Type *BP = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  const llvm::Type *BPP = llvm::PointerType::getUnqual(BP);

  CGBuilderTy &Builder = CGF.Builder;
  llvm::Value *VAListAddrAsBPP = Builder.CreateBitCast(VAListAddr, BPP,
                                                       "ap");
  llvm::Value *Addr = Builder.CreateLoad(VAListAddrAsBPP, "ap.cur");
  llvm::Type *PTy =
    llvm::PointerType::getUnqual(CGF.ConvertType(Ty));
  llvm::Value *AddrTyped = Builder.CreateBitCast(Addr, PTy);

  uint64_t Offset =
    llvm::RoundUpToAlignment(CGF.getContext().getTypeSize(Ty) / 8, 4);
  llvm::Value *NextAddr =
    Builder.CreateGEP(Addr, llvm::ConstantInt::get(CGF.Int32Ty, Offset),
                      "ap.next");
  Builder.CreateStore(NextAddr, VAListAddrAsBPP);

  return AddrTyped;
}

//===----------------------------------------------------------------------===//
// PTX ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class PTXABIInfo : public ABIInfo {
public:
  PTXABIInfo(CodeGenTypes &CGT) : ABIInfo(CGT) {}

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType Ty) const;

  virtual void computeInfo(CGFunctionInfo &FI) const;
  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CFG) const;
};

class PTXTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  PTXTargetCodeGenInfo(CodeGenTypes &CGT)
    : TargetCodeGenInfo(new PTXABIInfo(CGT)) {}
};

ABIArgInfo PTXABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();
  if (isAggregateTypeForABI(RetTy))
    return ABIArgInfo::getIndirect(0);
  return ABIArgInfo::getDirect();
}

ABIArgInfo PTXABIInfo::classifyArgumentType(QualType Ty) const {
  if (isAggregateTypeForABI(Ty))
    return ABIArgInfo::getIndirect(0);

  return ABIArgInfo::getDirect();
}

void PTXABIInfo::computeInfo(CGFunctionInfo &FI) const {
  FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
  for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
       it != ie; ++it)
    it->info = classifyArgumentType(it->type);

  // Always honor user-specified calling convention.
  if (FI.getCallingConvention() != llvm::CallingConv::C)
    return;

  // Calling convention as default by an ABI.
  llvm::CallingConv::ID DefaultCC;
  llvm::StringRef Env = getContext().Target.getTriple().getEnvironmentName();
  if (Env == "device")
    DefaultCC = llvm::CallingConv::PTX_Device;
  else
    DefaultCC = llvm::CallingConv::PTX_Kernel;

  FI.setEffectiveCallingConvention(DefaultCC);
}

llvm::Value *PTXABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                   CodeGenFunction &CFG) const {
  llvm_unreachable("PTX does not support varargs");
  return 0;
}

}

//===----------------------------------------------------------------------===//
// SystemZ ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class SystemZABIInfo : public ABIInfo {
public:
  SystemZABIInfo(CodeGenTypes &CGT) : ABIInfo(CGT) {}

  bool isPromotableIntegerType(QualType Ty) const;

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType RetTy) const;

  virtual void computeInfo(CGFunctionInfo &FI) const {
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
    for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
         it != ie; ++it)
      it->info = classifyArgumentType(it->type);
  }

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

class SystemZTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  SystemZTargetCodeGenInfo(CodeGenTypes &CGT)
    : TargetCodeGenInfo(new SystemZABIInfo(CGT)) {}
};

}

bool SystemZABIInfo::isPromotableIntegerType(QualType Ty) const {
  // SystemZ ABI requires all 8, 16 and 32 bit quantities to be extended.
  if (const BuiltinType *BT = Ty->getAs<BuiltinType>())
    switch (BT->getKind()) {
    case BuiltinType::Bool:
    case BuiltinType::Char_S:
    case BuiltinType::Char_U:
    case BuiltinType::SChar:
    case BuiltinType::UChar:
    case BuiltinType::Short:
    case BuiltinType::UShort:
    case BuiltinType::Int:
    case BuiltinType::UInt:
      return true;
    default:
      return false;
    }
  return false;
}

llvm::Value *SystemZABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                       CodeGenFunction &CGF) const {
  // FIXME: Implement
  return 0;
}


ABIArgInfo SystemZABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();
  if (isAggregateTypeForABI(RetTy))
    return ABIArgInfo::getIndirect(0);

  return (isPromotableIntegerType(RetTy) ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

ABIArgInfo SystemZABIInfo::classifyArgumentType(QualType Ty) const {
  if (isAggregateTypeForABI(Ty))
    return ABIArgInfo::getIndirect(0);

  return (isPromotableIntegerType(Ty) ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

//===----------------------------------------------------------------------===//
// MBlaze ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class MBlazeABIInfo : public ABIInfo {
public:
  MBlazeABIInfo(CodeGenTypes &CGT) : ABIInfo(CGT) {}

  bool isPromotableIntegerType(QualType Ty) const;

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType RetTy) const;

  virtual void computeInfo(CGFunctionInfo &FI) const {
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
    for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
         it != ie; ++it)
      it->info = classifyArgumentType(it->type);
  }

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

class MBlazeTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  MBlazeTargetCodeGenInfo(CodeGenTypes &CGT)
    : TargetCodeGenInfo(new MBlazeABIInfo(CGT)) {}
  void SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &M) const;
};

}

bool MBlazeABIInfo::isPromotableIntegerType(QualType Ty) const {
  // MBlaze ABI requires all 8 and 16 bit quantities to be extended.
  if (const BuiltinType *BT = Ty->getAs<BuiltinType>())
    switch (BT->getKind()) {
    case BuiltinType::Bool:
    case BuiltinType::Char_S:
    case BuiltinType::Char_U:
    case BuiltinType::SChar:
    case BuiltinType::UChar:
    case BuiltinType::Short:
    case BuiltinType::UShort:
      return true;
    default:
      return false;
    }
  return false;
}

llvm::Value *MBlazeABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                      CodeGenFunction &CGF) const {
  // FIXME: Implement
  return 0;
}


ABIArgInfo MBlazeABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();
  if (isAggregateTypeForABI(RetTy))
    return ABIArgInfo::getIndirect(0);

  return (isPromotableIntegerType(RetTy) ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

ABIArgInfo MBlazeABIInfo::classifyArgumentType(QualType Ty) const {
  if (isAggregateTypeForABI(Ty))
    return ABIArgInfo::getIndirect(0);

  return (isPromotableIntegerType(Ty) ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

void MBlazeTargetCodeGenInfo::SetTargetAttributes(const Decl *D,
                                                  llvm::GlobalValue *GV,
                                                  CodeGen::CodeGenModule &M)
                                                  const {
  const FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
  if (!FD) return;

  llvm::CallingConv::ID CC = llvm::CallingConv::C;
  if (FD->hasAttr<MBlazeInterruptHandlerAttr>())
    CC = llvm::CallingConv::MBLAZE_INTR;
  else if (FD->hasAttr<MBlazeSaveVolatilesAttr>())
    CC = llvm::CallingConv::MBLAZE_SVOL;

  if (CC != llvm::CallingConv::C) {
      // Handle 'interrupt_handler' attribute:
      llvm::Function *F = cast<llvm::Function>(GV);

      // Step 1: Set ISR calling convention.
      F->setCallingConv(CC);

      // Step 2: Add attributes goodness.
      F->addFnAttr(llvm::Attribute::NoInline);
  }

  // Step 3: Emit _interrupt_handler alias.
  if (CC == llvm::CallingConv::MBLAZE_INTR)
    new llvm::GlobalAlias(GV->getType(), llvm::Function::ExternalLinkage,
                          "_interrupt_handler", GV, &M.getModule());
}


//===----------------------------------------------------------------------===//
// MSP430 ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class MSP430TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  MSP430TargetCodeGenInfo(CodeGenTypes &CGT)
    : TargetCodeGenInfo(new DefaultABIInfo(CGT)) {}
  void SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &M) const;
};

}

void MSP430TargetCodeGenInfo::SetTargetAttributes(const Decl *D,
                                                  llvm::GlobalValue *GV,
                                             CodeGen::CodeGenModule &M) const {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (const MSP430InterruptAttr *attr = FD->getAttr<MSP430InterruptAttr>()) {
      // Handle 'interrupt' attribute:
      llvm::Function *F = cast<llvm::Function>(GV);

      // Step 1: Set ISR calling convention.
      F->setCallingConv(llvm::CallingConv::MSP430_INTR);

      // Step 2: Add attributes goodness.
      F->addFnAttr(llvm::Attribute::NoInline);

      // Step 3: Emit ISR vector alias.
      unsigned Num = attr->getNumber() + 0xffe0;
      new llvm::GlobalAlias(GV->getType(), llvm::Function::ExternalLinkage,
                            "vector_" + llvm::Twine::utohexstr(Num),
                            GV, &M.getModule());
    }
  }
}

//===----------------------------------------------------------------------===//
// MIPS ABI Implementation.  This works for both little-endian and
// big-endian variants.
//===----------------------------------------------------------------------===//

namespace {
class MIPSTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  MIPSTargetCodeGenInfo(CodeGenTypes &CGT)
    : TargetCodeGenInfo(new DefaultABIInfo(CGT)) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &CGM) const {
    return 29;
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const;
};
}

bool
MIPSTargetCodeGenInfo::initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                                               llvm::Value *Address) const {
  // This information comes from gcc's implementation, which seems to
  // as canonical as it gets.

  CodeGen::CGBuilderTy &Builder = CGF.Builder;
  llvm::LLVMContext &Context = CGF.getLLVMContext();

  // Everything on MIPS is 4 bytes.  Double-precision FP registers
  // are aliased to pairs of single-precision FP registers.
  const llvm::IntegerType *i8 = llvm::Type::getInt8Ty(Context);
  llvm::Value *Four8 = llvm::ConstantInt::get(i8, 4);

  // 0-31 are the general purpose registers, $0 - $31.
  // 32-63 are the floating-point registers, $f0 - $f31.
  // 64 and 65 are the multiply/divide registers, $hi and $lo.
  // 66 is the (notional, I think) register for signal-handler return.
  AssignToArrayRange(Builder, Address, Four8, 0, 65);

  // 67-74 are the floating-point status registers, $fcc0 - $fcc7.
  // They are one bit wide and ignored here.

  // 80-111 are the coprocessor 0 registers, $c0r0 - $c0r31.
  // (coprocessor 1 is the FP unit)
  // 112-143 are the coprocessor 2 registers, $c2r0 - $c2r31.
  // 144-175 are the coprocessor 3 registers, $c3r0 - $c3r31.
  // 176-181 are the DSP accumulator registers.
  AssignToArrayRange(Builder, Address, Four8, 80, 181);

  return false;
}


const TargetCodeGenInfo &CodeGenModule::getTargetCodeGenInfo() {
  if (TheTargetCodeGenInfo)
    return *TheTargetCodeGenInfo;

  // For now we just cache the TargetCodeGenInfo in CodeGenModule and don't
  // free it.

  const llvm::Triple &Triple = getContext().Target.getTriple();
  switch (Triple.getArch()) {
  default:
    return *(TheTargetCodeGenInfo = new DefaultTargetCodeGenInfo(Types));

  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
    return *(TheTargetCodeGenInfo = new MIPSTargetCodeGenInfo(Types));

  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    {
      ARMABIInfo::ABIKind Kind = ARMABIInfo::AAPCS;

      if (strcmp(getContext().Target.getABI(), "apcs-gnu") == 0)
        Kind = ARMABIInfo::APCS;
      else if (CodeGenOpts.FloatABI == "hard")
        Kind = ARMABIInfo::AAPCS_VFP;

      return *(TheTargetCodeGenInfo = new ARMTargetCodeGenInfo(Types, Kind));
    }

  case llvm::Triple::ppc:
    return *(TheTargetCodeGenInfo = new PPC32TargetCodeGenInfo(Types));

  case llvm::Triple::ptx32:
  case llvm::Triple::ptx64:
    return *(TheTargetCodeGenInfo = new PTXTargetCodeGenInfo(Types));

  case llvm::Triple::systemz:
    return *(TheTargetCodeGenInfo = new SystemZTargetCodeGenInfo(Types));

  case llvm::Triple::mblaze:
    return *(TheTargetCodeGenInfo = new MBlazeTargetCodeGenInfo(Types));

  case llvm::Triple::msp430:
    return *(TheTargetCodeGenInfo = new MSP430TargetCodeGenInfo(Types));

  case llvm::Triple::x86:
    if (Triple.isOSDarwin())
      return *(TheTargetCodeGenInfo =
               new X86_32TargetCodeGenInfo(Types, true, true));

    switch (Triple.getOS()) {
    case llvm::Triple::Cygwin:
    case llvm::Triple::MinGW32:
    case llvm::Triple::AuroraUX:
    case llvm::Triple::DragonFly:
    case llvm::Triple::FreeBSD:
    case llvm::Triple::OpenBSD:
    case llvm::Triple::NetBSD:
      return *(TheTargetCodeGenInfo =
               new X86_32TargetCodeGenInfo(Types, false, true));

    default:
      return *(TheTargetCodeGenInfo =
               new X86_32TargetCodeGenInfo(Types, false, false));
    }

  case llvm::Triple::x86_64:
    switch (Triple.getOS()) {
    case llvm::Triple::Win32:
    case llvm::Triple::MinGW32:
    case llvm::Triple::Cygwin:
      return *(TheTargetCodeGenInfo = new WinX86_64TargetCodeGenInfo(Types));
    default:
      return *(TheTargetCodeGenInfo = new X86_64TargetCodeGenInfo(Types));
    }
  }
}
