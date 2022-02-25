//===- VPlanValue.h - Represent Values in Vectorizer Plan -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of the entities induced by Vectorization
/// Plans, e.g. the instructions the VPlan intends to generate if executed.
/// VPlan models the following entities:
/// VPValue   VPUser   VPDef
///    |        |
///   VPInstruction
/// These are documented in docs/VectorizationPlan.rst.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLAN_VALUE_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLAN_VALUE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/ADT/iterator_range.h"

namespace llvm {

// Forward declarations.
class raw_ostream;
class Value;
class VPDef;
class VPSlotTracker;
class VPUser;
class VPRecipeBase;
class VPWidenMemoryInstructionRecipe;

// This is the base class of the VPlan Def/Use graph, used for modeling the data
// flow into, within and out of the VPlan. VPValues can stand for live-ins
// coming from the input IR, instructions which VPlan will generate if executed
// and live-outs which the VPlan will need to fix accordingly.
class VPValue {
  friend class VPBuilder;
  friend class VPDef;
  friend class VPInstruction;
  friend struct VPlanTransforms;
  friend class VPBasicBlock;
  friend class VPInterleavedAccessInfo;
  friend class VPSlotTracker;
  friend class VPRecipeBase;
  friend class VPWidenMemoryInstructionRecipe;

  const unsigned char SubclassID; ///< Subclass identifier (for isa/dyn_cast).

  SmallVector<VPUser *, 1> Users;

protected:
  // Hold the underlying Value, if any, attached to this VPValue.
  Value *UnderlyingVal;

  /// Pointer to the VPDef that defines this VPValue. If it is nullptr, the
  /// VPValue is not defined by any recipe modeled in VPlan.
  VPDef *Def;

  VPValue(const unsigned char SC, Value *UV = nullptr, VPDef *Def = nullptr);

  // DESIGN PRINCIPLE: Access to the underlying IR must be strictly limited to
  // the front-end and back-end of VPlan so that the middle-end is as
  // independent as possible of the underlying IR. We grant access to the
  // underlying IR using friendship. In that way, we should be able to use VPlan
  // for multiple underlying IRs (Polly?) by providing a new VPlan front-end,
  // back-end and analysis information for the new IR.

  // Set \p Val as the underlying Value of this VPValue.
  void setUnderlyingValue(Value *Val) {
    assert(!UnderlyingVal && "Underlying Value is already set.");
    UnderlyingVal = Val;
  }

public:
  /// Return the underlying Value attached to this VPValue.
  Value *getUnderlyingValue() { return UnderlyingVal; }
  const Value *getUnderlyingValue() const { return UnderlyingVal; }

  /// An enumeration for keeping track of the concrete subclass of VPValue that
  /// are actually instantiated. Values of this enumeration are kept in the
  /// SubclassID field of the VPValue objects. They are used for concrete
  /// type identification.
  enum {
    VPValueSC,
    VPVInstructionSC,
    VPVMemoryInstructionSC,
    VPVReductionSC,
    VPVReplicateSC,
    VPVWidenSC,
    VPVWidenCallSC,
    VPVWidenGEPSC,
    VPVWidenSelectSC,

    // Phi-like VPValues. Need to be kept together.
    VPVBlendSC,
    VPVFirstOrderRecurrencePHISC,
    VPVWidenPHISC,
    VPVWidenCanonicalIVSC,
    VPVWidenIntOrFpInductionSC,
    VPVPredInstPHI,
    VPVReductionPHISC,
  };

  VPValue(Value *UV = nullptr, VPDef *Def = nullptr)
      : VPValue(VPValueSC, UV, Def) {}
  VPValue(const VPValue &) = delete;
  VPValue &operator=(const VPValue &) = delete;

  virtual ~VPValue();

  /// \return an ID for the concrete type of this object.
  /// This is used to implement the classof checks. This should not be used
  /// for any other purpose, as the values may change as LLVM evolves.
  unsigned getVPValueID() const { return SubclassID; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void printAsOperand(raw_ostream &OS, VPSlotTracker &Tracker) const;
  void print(raw_ostream &OS, VPSlotTracker &Tracker) const;

  /// Dump the value to stderr (for debugging).
  void dump() const;
#endif

  unsigned getNumUsers() const { return Users.size(); }
  void addUser(VPUser &User) { Users.push_back(&User); }

  /// Remove a single \p User from the list of users.
  void removeUser(VPUser &User) {
    bool Found = false;
    // The same user can be added multiple times, e.g. because the same VPValue
    // is used twice by the same VPUser. Remove a single one.
    erase_if(Users, [&User, &Found](VPUser *Other) {
      if (Found)
        return false;
      if (Other == &User) {
        Found = true;
        return true;
      }
      return false;
    });
  }

  typedef SmallVectorImpl<VPUser *>::iterator user_iterator;
  typedef SmallVectorImpl<VPUser *>::const_iterator const_user_iterator;
  typedef iterator_range<user_iterator> user_range;
  typedef iterator_range<const_user_iterator> const_user_range;

  user_iterator user_begin() { return Users.begin(); }
  const_user_iterator user_begin() const { return Users.begin(); }
  user_iterator user_end() { return Users.end(); }
  const_user_iterator user_end() const { return Users.end(); }
  user_range users() { return user_range(user_begin(), user_end()); }
  const_user_range users() const {
    return const_user_range(user_begin(), user_end());
  }

  /// Returns true if the value has more than one unique user.
  bool hasMoreThanOneUniqueUser() {
    if (getNumUsers() == 0)
      return false;

    // Check if all users match the first user.
    auto Current = std::next(user_begin());
    while (Current != user_end() && *user_begin() == *Current)
      Current++;
    return Current != user_end();
  }

  void replaceAllUsesWith(VPValue *New);

  VPDef *getDef() { return Def; }

  /// Returns the underlying IR value, if this VPValue is defined outside the
  /// scope of VPlan. Returns nullptr if the VPValue is defined by a VPDef
  /// inside a VPlan.
  Value *getLiveInIRValue() {
    assert(!getDef() &&
           "VPValue is not a live-in; it is defined by a VPDef inside a VPlan");
    return getUnderlyingValue();
  }
};

typedef DenseMap<Value *, VPValue *> Value2VPValueTy;
typedef DenseMap<VPValue *, Value *> VPValue2ValueTy;

raw_ostream &operator<<(raw_ostream &OS, const VPValue &V);

/// This class augments VPValue with operands which provide the inverse def-use
/// edges from VPValue's users to their defs.
class VPUser {
public:
  /// Subclass identifier (for isa/dyn_cast).
  enum class VPUserID {
    Recipe,
    // TODO: Currently VPUsers are used in VPBlockBase, but in the future the
    // only VPUsers should either be recipes or live-outs.
    Block
  };

private:
  SmallVector<VPValue *, 2> Operands;

  VPUserID ID;

protected:
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the operands to \p O.
  void printOperands(raw_ostream &O, VPSlotTracker &SlotTracker) const;
#endif

  VPUser(ArrayRef<VPValue *> Operands, VPUserID ID) : ID(ID) {
    for (VPValue *Operand : Operands)
      addOperand(Operand);
  }

  VPUser(std::initializer_list<VPValue *> Operands, VPUserID ID)
      : VPUser(ArrayRef<VPValue *>(Operands), ID) {}

  template <typename IterT>
  VPUser(iterator_range<IterT> Operands, VPUserID ID) : ID(ID) {
    for (VPValue *Operand : Operands)
      addOperand(Operand);
  }

public:
  VPUser() = delete;
  VPUser(const VPUser &) = delete;
  VPUser &operator=(const VPUser &) = delete;
  virtual ~VPUser() {
    for (VPValue *Op : operands())
      Op->removeUser(*this);
  }

  VPUserID getVPUserID() const { return ID; }

  void addOperand(VPValue *Operand) {
    Operands.push_back(Operand);
    Operand->addUser(*this);
  }

  unsigned getNumOperands() const { return Operands.size(); }
  inline VPValue *getOperand(unsigned N) const {
    assert(N < Operands.size() && "Operand index out of bounds");
    return Operands[N];
  }

  void setOperand(unsigned I, VPValue *New) {
    Operands[I]->removeUser(*this);
    Operands[I] = New;
    New->addUser(*this);
  }

  void removeLastOperand() {
    VPValue *Op = Operands.pop_back_val();
    Op->removeUser(*this);
  }

  typedef SmallVectorImpl<VPValue *>::iterator operand_iterator;
  typedef SmallVectorImpl<VPValue *>::const_iterator const_operand_iterator;
  typedef iterator_range<operand_iterator> operand_range;
  typedef iterator_range<const_operand_iterator> const_operand_range;

  operand_iterator op_begin() { return Operands.begin(); }
  const_operand_iterator op_begin() const { return Operands.begin(); }
  operand_iterator op_end() { return Operands.end(); }
  const_operand_iterator op_end() const { return Operands.end(); }
  operand_range operands() { return operand_range(op_begin(), op_end()); }
  const_operand_range operands() const {
    return const_operand_range(op_begin(), op_end());
  }

  /// Method to support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const VPDef *Recipe);
};

/// This class augments a recipe with a set of VPValues defined by the recipe.
/// It allows recipes to define zero, one or multiple VPValues. A VPDef owns
/// the VPValues it defines and is responsible for deleting its defined values.
/// Single-value VPDefs that also inherit from VPValue must make sure to inherit
/// from VPDef before VPValue.
class VPDef {
  friend class VPValue;

  /// Subclass identifier (for isa/dyn_cast).
  const unsigned char SubclassID;

  /// The VPValues defined by this VPDef.
  TinyPtrVector<VPValue *> DefinedValues;

  /// Add \p V as a defined value by this VPDef.
  void addDefinedValue(VPValue *V) {
    assert(V->getDef() == this &&
           "can only add VPValue already linked with this VPDef");
    DefinedValues.push_back(V);
  }

  /// Remove \p V from the values defined by this VPDef. \p V must be a defined
  /// value of this VPDef.
  void removeDefinedValue(VPValue *V) {
    assert(V->getDef() == this &&
           "can only remove VPValue linked with this VPDef");
    assert(is_contained(DefinedValues, V) &&
           "VPValue to remove must be in DefinedValues");
    erase_value(DefinedValues, V);
    V->Def = nullptr;
  }

public:
  /// An enumeration for keeping track of the concrete subclass of VPRecipeBase
  /// that is actually instantiated. Values of this enumeration are kept in the
  /// SubclassID field of the VPRecipeBase objects. They are used for concrete
  /// type identification.
  using VPRecipeTy = enum {
    VPBranchOnMaskSC,
    VPInstructionSC,
    VPInterleaveSC,
    VPReductionSC,
    VPReplicateSC,
    VPWidenCallSC,
    VPWidenGEPSC,
    VPWidenMemoryInstructionSC,
    VPWidenSC,
    VPWidenSelectSC,

    // Phi-like recipes. Need to be kept together.
    VPBlendSC,
    VPFirstOrderRecurrencePHISC,
    VPWidenPHISC,
    VPWidenCanonicalIVSC,
    VPWidenIntOrFpInductionSC,
    VPPredInstPHISC,
    VPReductionPHISC,
    VPFirstPHISC = VPBlendSC,
    VPLastPHISC = VPReductionPHISC,
  };

  VPDef(const unsigned char SC) : SubclassID(SC) {}

  virtual ~VPDef() {
    for (VPValue *D : make_early_inc_range(DefinedValues)) {
      assert(D->Def == this &&
             "all defined VPValues should point to the containing VPDef");
      assert(D->getNumUsers() == 0 &&
             "all defined VPValues should have no more users");
      D->Def = nullptr;
      delete D;
    }
  }

  /// Returns the only VPValue defined by the VPDef. Can only be called for
  /// VPDefs with a single defined value.
  VPValue *getVPSingleValue() {
    assert(DefinedValues.size() == 1 && "must have exactly one defined value");
    assert(DefinedValues[0] && "defined value must be non-null");
    return DefinedValues[0];
  }
  const VPValue *getVPSingleValue() const {
    assert(DefinedValues.size() == 1 && "must have exactly one defined value");
    assert(DefinedValues[0] && "defined value must be non-null");
    return DefinedValues[0];
  }

  /// Returns the VPValue with index \p I defined by the VPDef.
  VPValue *getVPValue(unsigned I) {
    assert(DefinedValues[I] && "defined value must be non-null");
    return DefinedValues[I];
  }
  const VPValue *getVPValue(unsigned I) const {
    assert(DefinedValues[I] && "defined value must be non-null");
    return DefinedValues[I];
  }

  /// Returns an ArrayRef of the values defined by the VPDef.
  ArrayRef<VPValue *> definedValues() { return DefinedValues; }
  /// Returns an ArrayRef of the values defined by the VPDef.
  ArrayRef<VPValue *> definedValues() const { return DefinedValues; }

  /// Returns the number of values defined by the VPDef.
  unsigned getNumDefinedValues() const { return DefinedValues.size(); }

  /// \return an ID for the concrete type of this object.
  /// This is used to implement the classof checks. This should not be used
  /// for any other purpose, as the values may change as LLVM evolves.
  unsigned getVPDefID() const { return SubclassID; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Dump the VPDef to stderr (for debugging).
  void dump() const;

  /// Each concrete VPDef prints itself.
  virtual void print(raw_ostream &O, const Twine &Indent,
                     VPSlotTracker &SlotTracker) const = 0;
#endif
};

class VPlan;
class VPBasicBlock;
class VPRegionBlock;

/// This class can be used to assign consecutive numbers to all VPValues in a
/// VPlan and allows querying the numbering for printing, similar to the
/// ModuleSlotTracker for IR values.
class VPSlotTracker {
  DenseMap<const VPValue *, unsigned> Slots;
  unsigned NextSlot = 0;

  void assignSlot(const VPValue *V);
  void assignSlots(const VPlan &Plan);

public:
  VPSlotTracker(const VPlan *Plan = nullptr) {
    if (Plan)
      assignSlots(*Plan);
  }

  unsigned getSlot(const VPValue *V) const {
    auto I = Slots.find(V);
    if (I == Slots.end())
      return -1;
    return I->second;
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLAN_VALUE_H
