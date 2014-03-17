//==-- CGFunctionInfo.h - Representation of function argument/return types -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines CGFunctionInfo and associated types used in representing the
// LLVM source types and ABI-coerced types for function arguments and
// return values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGEN_FUNCTION_INFO_H
#define LLVM_CLANG_CODEGEN_FUNCTION_INFO_H

#include "clang/AST/CanonicalType.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/FoldingSet.h"
#include <cassert>

namespace llvm {
  class Type;
  class StructType;
}

namespace clang {
class Decl;

namespace CodeGen {

/// ABIArgInfo - Helper class to encapsulate information about how a
/// specific C type should be passed to or returned from a function.
class ABIArgInfo {
public:
  enum Kind {
    /// Direct - Pass the argument directly using the normal converted LLVM
    /// type, or by coercing to another specified type stored in
    /// 'CoerceToType').  If an offset is specified (in UIntData), then the
    /// argument passed is offset by some number of bytes in the memory
    /// representation. A dummy argument is emitted before the real argument
    /// if the specified type stored in "PaddingType" is not zero.
    Direct,

    /// Extend - Valid only for integer argument types. Same as 'direct'
    /// but also emit a zero/sign extension attribute.
    Extend,

    /// Indirect - Pass the argument indirectly via a hidden pointer
    /// with the specified alignment (0 indicates default alignment).
    Indirect,

    /// Ignore - Ignore the argument (treat as void). Useful for void and
    /// empty structs.
    Ignore,

    /// Expand - Only valid for aggregate argument types. The structure should
    /// be expanded into consecutive arguments for its constituent fields.
    /// Currently expand is only allowed on structures whose fields
    /// are all scalar types or are themselves expandable types.
    Expand,

    /// InAlloca - Pass the argument directly using the LLVM inalloca attribute.
    /// This is similar to 'direct', except it only applies to arguments stored
    /// in memory and forbids any implicit copies.  When applied to a return
    /// type, it means the value is returned indirectly via an implicit sret
    /// parameter stored in the argument struct.
    InAlloca,

    KindFirst = Direct,
    KindLast = InAlloca
  };

private:
  Kind TheKind;
  llvm::Type *TypeData;
  llvm::Type *PaddingType;
  unsigned UIntData;
  bool BoolData0;
  bool BoolData1;
  bool InReg;
  bool PaddingInReg;

  ABIArgInfo(Kind K, llvm::Type *TD, unsigned UI, bool B0, bool B1, bool IR,
             bool PIR, llvm::Type* P)
    : TheKind(K), TypeData(TD), PaddingType(P), UIntData(UI), BoolData0(B0),
      BoolData1(B1), InReg(IR), PaddingInReg(PIR) {}

public:
  ABIArgInfo() : TheKind(Direct), TypeData(0), UIntData(0) {}

  static ABIArgInfo getDirect(llvm::Type *T = 0, unsigned Offset = 0,
                              llvm::Type *Padding = 0) {
    return ABIArgInfo(Direct, T, Offset, false, false, false, false, Padding);
  }
  static ABIArgInfo getDirectInReg(llvm::Type *T = 0) {
    return ABIArgInfo(Direct, T, 0, false, false, true, false, 0);
  }
  static ABIArgInfo getExtend(llvm::Type *T = 0) {
    return ABIArgInfo(Extend, T, 0, false, false, false, false, 0);
  }
  static ABIArgInfo getExtendInReg(llvm::Type *T = 0) {
    return ABIArgInfo(Extend, T, 0, false, false, true, false, 0);
  }
  static ABIArgInfo getIgnore() {
    return ABIArgInfo(Ignore, 0, 0, false, false, false, false, 0);
  }
  static ABIArgInfo getIndirect(unsigned Alignment, bool ByVal = true
                                , bool Realign = false
                                , llvm::Type *Padding = 0) {
    return ABIArgInfo(Indirect, 0, Alignment, ByVal, Realign, false, false,
                      Padding);
  }
  static ABIArgInfo getInAlloca(unsigned FieldIndex) {
    return ABIArgInfo(InAlloca, 0, FieldIndex, false, false, false, false, 0);
  }
  static ABIArgInfo getIndirectInReg(unsigned Alignment, bool ByVal = true
                                , bool Realign = false) {
    return ABIArgInfo(Indirect, 0, Alignment, ByVal, Realign, true, false, 0);
  }
  static ABIArgInfo getExpand() {
    return ABIArgInfo(Expand, 0, 0, false, false, false, false, 0);
  }
  static ABIArgInfo getExpandWithPadding(bool PaddingInReg,
                                         llvm::Type *Padding) {
   return ABIArgInfo(Expand, 0, 0, false, false, false, PaddingInReg,
                     Padding);
  }

  Kind getKind() const { return TheKind; }
  bool isDirect() const { return TheKind == Direct; }
  bool isInAlloca() const { return TheKind == InAlloca; }
  bool isExtend() const { return TheKind == Extend; }
  bool isIgnore() const { return TheKind == Ignore; }
  bool isIndirect() const { return TheKind == Indirect; }
  bool isExpand() const { return TheKind == Expand; }

  bool canHaveCoerceToType() const {
    return TheKind == Direct || TheKind == Extend;
  }

  // Direct/Extend accessors
  unsigned getDirectOffset() const {
    assert((isDirect() || isExtend()) && "Not a direct or extend kind");
    return UIntData;
  }

  llvm::Type *getPaddingType() const {
    return PaddingType;
  }

  bool getPaddingInReg() const {
    return PaddingInReg;
  }

  llvm::Type *getCoerceToType() const {
    assert(canHaveCoerceToType() && "Invalid kind!");
    return TypeData;
  }

  void setCoerceToType(llvm::Type *T) {
    assert(canHaveCoerceToType() && "Invalid kind!");
    TypeData = T;
  }

  bool getInReg() const {
    assert((isDirect() || isExtend() || isIndirect()) && "Invalid kind!");
    return InReg;
  }

  // Indirect accessors
  unsigned getIndirectAlign() const {
    assert(TheKind == Indirect && "Invalid kind!");
    return UIntData;
  }

  bool getIndirectByVal() const {
    assert(TheKind == Indirect && "Invalid kind!");
    return BoolData0;
  }

  bool getIndirectRealign() const {
    assert(TheKind == Indirect && "Invalid kind!");
    return BoolData1;
  }

  unsigned getInAllocaFieldIndex() const {
    assert(TheKind == InAlloca && "Invalid kind!");
    return UIntData;
  }

  /// \brief Return true if this field of an inalloca struct should be returned
  /// to implement a struct return calling convention.
  bool getInAllocaSRet() const {
    assert(TheKind == InAlloca && "Invalid kind!");
    return BoolData0;
  }

  void setInAllocaSRet(bool SRet) {
    assert(TheKind == InAlloca && "Invalid kind!");
    BoolData0 = SRet;
  }

  void dump() const;
};

/// A class for recording the number of arguments that a function
/// signature requires.
class RequiredArgs {
  /// The number of required arguments, or ~0 if the signature does
  /// not permit optional arguments.
  unsigned NumRequired;
public:
  enum All_t { All };

  RequiredArgs(All_t _) : NumRequired(~0U) {}
  explicit RequiredArgs(unsigned n) : NumRequired(n) {
    assert(n != ~0U);
  }

  /// Compute the arguments required by the given formal prototype,
  /// given that there may be some additional, non-formal arguments
  /// in play.
  static RequiredArgs forPrototypePlus(const FunctionProtoType *prototype,
                                       unsigned additional) {
    if (!prototype->isVariadic()) return All;
    return RequiredArgs(prototype->getNumParams() + additional);
  }

  static RequiredArgs forPrototype(const FunctionProtoType *prototype) {
    return forPrototypePlus(prototype, 0);
  }

  static RequiredArgs forPrototype(CanQual<FunctionProtoType> prototype) {
    return forPrototype(prototype.getTypePtr());
  }

  static RequiredArgs forPrototypePlus(CanQual<FunctionProtoType> prototype,
                                       unsigned additional) {
    return forPrototypePlus(prototype.getTypePtr(), additional);
  }

  bool allowsOptionalArgs() const { return NumRequired != ~0U; }
  unsigned getNumRequiredArgs() const {
    assert(allowsOptionalArgs());
    return NumRequired;
  }

  unsigned getOpaqueData() const { return NumRequired; }
  static RequiredArgs getFromOpaqueData(unsigned value) {
    if (value == ~0U) return All;
    return RequiredArgs(value);
  }
};

/// CGFunctionInfo - Class to encapsulate the information about a
/// function definition.
class CGFunctionInfo : public llvm::FoldingSetNode {
  struct ArgInfo {
    CanQualType type;
    ABIArgInfo info;
  };

  /// The LLVM::CallingConv to use for this function (as specified by the
  /// user).
  unsigned CallingConvention : 8;

  /// The LLVM::CallingConv to actually use for this function, which may
  /// depend on the ABI.
  unsigned EffectiveCallingConvention : 8;

  /// The clang::CallingConv that this was originally created with.
  unsigned ASTCallingConvention : 8;

  /// Whether this is an instance method.
  unsigned InstanceMethod : 1;

  /// Whether this function is noreturn.
  unsigned NoReturn : 1;

  /// Whether this function is returns-retained.
  unsigned ReturnsRetained : 1;

  /// How many arguments to pass inreg.
  unsigned HasRegParm : 1;
  unsigned RegParm : 4;

  RequiredArgs Required;

  /// The struct representing all arguments passed in memory.  Only used when
  /// passing non-trivial types with inalloca.  Not part of the profile.
  llvm::StructType *ArgStruct;

  unsigned NumArgs;
  ArgInfo *getArgsBuffer() {
    return reinterpret_cast<ArgInfo*>(this+1);
  }
  const ArgInfo *getArgsBuffer() const {
    return reinterpret_cast<const ArgInfo*>(this + 1);
  }

  CGFunctionInfo() : Required(RequiredArgs::All) {}

public:
  static CGFunctionInfo *create(unsigned llvmCC,
                                bool InstanceMethod,
                                const FunctionType::ExtInfo &extInfo,
                                CanQualType resultType,
                                ArrayRef<CanQualType> argTypes,
                                RequiredArgs required);

  typedef const ArgInfo *const_arg_iterator;
  typedef ArgInfo *arg_iterator;

  typedef llvm::iterator_range<arg_iterator> arg_range;
  typedef llvm::iterator_range<const_arg_iterator> arg_const_range;

  arg_range arguments() { return arg_range(arg_begin(), arg_end()); }
  arg_const_range arguments() const {
    return arg_const_range(arg_begin(), arg_end());
  }

  const_arg_iterator arg_begin() const { return getArgsBuffer() + 1; }
  const_arg_iterator arg_end() const { return getArgsBuffer() + 1 + NumArgs; }
  arg_iterator arg_begin() { return getArgsBuffer() + 1; }
  arg_iterator arg_end() { return getArgsBuffer() + 1 + NumArgs; }

  unsigned  arg_size() const { return NumArgs; }

  bool isVariadic() const { return Required.allowsOptionalArgs(); }
  RequiredArgs getRequiredArgs() const { return Required; }

  bool isInstanceMethod() const { return InstanceMethod; }

  bool isNoReturn() const { return NoReturn; }

  /// In ARC, whether this function retains its return value.  This
  /// is not always reliable for call sites.
  bool isReturnsRetained() const { return ReturnsRetained; }

  /// getASTCallingConvention() - Return the AST-specified calling
  /// convention.
  CallingConv getASTCallingConvention() const {
    return CallingConv(ASTCallingConvention);
  }

  /// getCallingConvention - Return the user specified calling
  /// convention, which has been translated into an LLVM CC.
  unsigned getCallingConvention() const { return CallingConvention; }

  /// getEffectiveCallingConvention - Return the actual calling convention to
  /// use, which may depend on the ABI.
  unsigned getEffectiveCallingConvention() const {
    return EffectiveCallingConvention;
  }
  void setEffectiveCallingConvention(unsigned Value) {
    EffectiveCallingConvention = Value;
  }

  bool getHasRegParm() const { return HasRegParm; }
  unsigned getRegParm() const { return RegParm; }

  FunctionType::ExtInfo getExtInfo() const {
    return FunctionType::ExtInfo(isNoReturn(),
                                 getHasRegParm(), getRegParm(),
                                 getASTCallingConvention(),
                                 isReturnsRetained());
  }

  CanQualType getReturnType() const { return getArgsBuffer()[0].type; }

  ABIArgInfo &getReturnInfo() { return getArgsBuffer()[0].info; }
  const ABIArgInfo &getReturnInfo() const { return getArgsBuffer()[0].info; }

  /// \brief Return true if this function uses inalloca arguments.
  bool usesInAlloca() const { return ArgStruct; }

  /// \brief Get the struct type used to represent all the arguments in memory.
  llvm::StructType *getArgStruct() const { return ArgStruct; }
  void setArgStruct(llvm::StructType *Ty) { ArgStruct = Ty; }

  void Profile(llvm::FoldingSetNodeID &ID) {
    ID.AddInteger(getASTCallingConvention());
    ID.AddBoolean(InstanceMethod);
    ID.AddBoolean(NoReturn);
    ID.AddBoolean(ReturnsRetained);
    ID.AddBoolean(HasRegParm);
    ID.AddInteger(RegParm);
    ID.AddInteger(Required.getOpaqueData());
    getReturnType().Profile(ID);
    for (const auto &I : arguments())
      I.type.Profile(ID);
  }
  static void Profile(llvm::FoldingSetNodeID &ID,
                      bool InstanceMethod,
                      const FunctionType::ExtInfo &info,
                      RequiredArgs required,
                      CanQualType resultType,
                      ArrayRef<CanQualType> argTypes) {
    ID.AddInteger(info.getCC());
    ID.AddBoolean(InstanceMethod);
    ID.AddBoolean(info.getNoReturn());
    ID.AddBoolean(info.getProducesResult());
    ID.AddBoolean(info.getHasRegParm());
    ID.AddInteger(info.getRegParm());
    ID.AddInteger(required.getOpaqueData());
    resultType.Profile(ID);
    for (ArrayRef<CanQualType>::iterator
           i = argTypes.begin(), e = argTypes.end(); i != e; ++i) {
      i->Profile(ID);
    }
  }
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
