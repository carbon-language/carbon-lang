//===--- MCPlus.h - helpers for MCPlus-level instructions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_MCPLUS_H
#define LLVM_TOOLS_LLVM_BOLT_MCPLUS_H

#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Casting.h"

namespace llvm {
namespace bolt {

namespace MCPlus {

/// This type represents C++ EH info for a callsite.  The symbol is the landing
/// pad and the uint64_t represents the action.
using MCLandingPad = std::pair<const MCSymbol *, uint64_t>;

/// An extension to MCInst is provided via an extra operand of type Inst with
/// ANNOTATION_LABEL opcode (i.e. we are tying an annotation instruction to an
/// existing one). The annotation instruction contains a list of Immediate
/// operands. Each operand either contains a value, or is a pointer to
/// an instance of class MCAnnotation.
///
/// There are 2 distinct groups of annotations. The first group is a first-class
/// annotation that affects semantics of the instruction, such as an
/// exception-handling or jump table information. The second group contains
/// information that is supplement, and could be discarded without affecting
/// correctness of the program. Debugging information, and profile information
/// belong to the second group.
///
/// For the first group, we use a reserved operand number/index. Operands in
/// the first groups store a value of an annotation.
///
/// Annotations in the second group are addressed by name, and their respective
/// operands store a pointer to an instance of MCAnnotation class.
class MCAnnotation {
public:
  enum Kind {
    kEHLandingPad,        /// Exception handling landing pad.
    kEHAction,            /// Action for exception handler.
    kGnuArgsSize,         /// GNU args size.
    kJumpTable,           /// Jump Table.
    kConditionalTailCall, /// CTC.
    kGeneric              /// First generic annotation.
  };

  virtual void print(raw_ostream &OS) const = 0;
  virtual bool equals(const MCAnnotation &) const = 0;
  virtual ~MCAnnotation() {}
protected:
  MCAnnotation() {}
private:
  // noncopyable
  MCAnnotation(const MCAnnotation &Other) = delete;
  MCAnnotation &operator=(const MCAnnotation &Other) = delete;
};

/// Instances of this class represent a simple annotation with a
/// specific value type.
/// Note that if ValueType contains any heap allocated memory, it will
/// only be freed if the annotation is removed with the
/// MCPlusBuilder::removeAnnotation method.  This is because all
/// annotations are arena allocated.
template <typename ValueType>
class MCSimpleAnnotation : public MCAnnotation {
public:
  const ValueType &getValue() const { return Value; }
  bool equals(const MCAnnotation &Other) const override {
    return Value == static_cast<const MCSimpleAnnotation &>(Other).Value;
  }
  explicit MCSimpleAnnotation(const ValueType &Val)
    : Value(Val) {}

  void print(raw_ostream &OS) const override {
    OS << Value;
  }

private:
  ValueType Value;
};

/// Return a number of operands in \Inst excluding operands representing
/// annotations.
inline unsigned getNumPrimeOperands(const MCInst &Inst) {
  if (Inst.getNumOperands() > 0 && std::prev(Inst.end())->isInst()) {
    assert(std::prev(Inst.end())->
             getInst()->getOpcode() == TargetOpcode::ANNOTATION_LABEL);
    return Inst.getNumOperands() - 1;
  }
  return Inst.getNumOperands();
}

template <class Compare>
inline bool equals(const MCExpr &A, const MCExpr &B, Compare Comp) {
  if (A.getKind() != B.getKind())
    return false;

  switch (A.getKind()) {
  case MCExpr::Constant: {
    const auto &ConstA = cast<MCConstantExpr>(A);
    const auto &ConstB = cast<MCConstantExpr>(B);
    return ConstA.getValue() == ConstB.getValue();
  }

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr &SymbolA = cast<MCSymbolRefExpr>(A);
    const MCSymbolRefExpr &SymbolB = cast<MCSymbolRefExpr>(B);
    return SymbolA.getKind() == SymbolB.getKind() &&
           Comp(&SymbolA.getSymbol(), &SymbolB.getSymbol());
  }

  case MCExpr::Unary: {
    const auto &UnaryA = cast<MCUnaryExpr>(A);
    const auto &UnaryB = cast<MCUnaryExpr>(B);
    return UnaryA.getOpcode() == UnaryB.getOpcode() &&
           equals(*UnaryA.getSubExpr(), *UnaryB.getSubExpr(), Comp);
  }

  case MCExpr::Binary: {
    const auto &BinaryA = cast<MCBinaryExpr>(A);
    const auto &BinaryB = cast<MCBinaryExpr>(B);
    return BinaryA.getOpcode() == BinaryB.getOpcode() &&
           equals(*BinaryA.getLHS(), *BinaryB.getLHS(), Comp) &&
           equals(*BinaryA.getRHS(), *BinaryB.getRHS(), Comp);
  }

  case MCExpr::Target: {
    llvm_unreachable("target-specific expressions are unsupported");
  }
  }

  llvm_unreachable("Invalid expression kind!");
}

template <class Compare>
inline bool equals(const MCOperand &A, const MCOperand &B, Compare Comp) {
  if (A.isReg()) {
    if (!B.isReg())
      return false;
    return A.getReg() == B.getReg();
  } else if (A.isImm()) {
    if (!B.isImm())
      return false;
    return A.getImm() == B.getImm();
  } else if (A.isFPImm()) {
    if (!B.isFPImm())
      return false;
    return A.getFPImm() == B.getFPImm();
  } else if (A.isExpr()) {
    if (!B.isExpr())
      return false;
    return equals(*A.getExpr(), *B.getExpr(), Comp);
  } else {
    llvm_unreachable("unexpected operand kind");
    return false;
  }
}

template <class Compare>
inline bool equals(const MCInst &A, const MCInst &B, Compare Comp) {
  if (A.getOpcode() != B.getOpcode())
    return false;

  unsigned NumOperands = MCPlus::getNumPrimeOperands(A);
  if (NumOperands != MCPlus::getNumPrimeOperands(B))
    return false;

  for (unsigned Index = 0; Index < NumOperands; ++Index) {
    if (!equals(A.getOperand(Index), B.getOperand(Index), Comp))
      return false;
  }

  return true;
}

} // namespace MCPlus

} // namespace bolt
} // namespace llvm

#endif
