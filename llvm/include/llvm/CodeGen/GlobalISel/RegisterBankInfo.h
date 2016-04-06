//==-- llvm/CodeGen/GlobalISel/RegisterBankInfo.h ----------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file declares the API for the register bank info.
/// This API is responsible for handling the register banks.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_REGBANKINFO_H
#define LLVM_CODEGEN_GLOBALISEL_REGBANKINFO_H

#include "llvm/ADT/APInt.h"
#include "llvm/CodeGen/GlobalISel/RegisterBank.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <memory> // For unique_ptr.

namespace llvm {
class MachineInstr;
class TargetRegisterInfo;
class raw_ostream;

/// Holds all the information related to register banks.
class RegisterBankInfo {
public:
  /// Helper struct that represents how a value is partially mapped
  /// into a register.
  /// The Mask is used to represent this partial mapping. Ones represent
  /// where the value lives in RegBank and the width of the Mask represents
  /// the size of the whole value.
  struct PartialMapping {
    /// Mask where the partial value lives.
    APInt Mask;
    /// Register bank where the partial value lives.
    const RegisterBank *RegBank;

    PartialMapping() = default;

    /// Provide a shortcut for quickly building PartialMapping.
    PartialMapping(const APInt &Mask, const RegisterBank &RegBank)
        : Mask(Mask), RegBank(&RegBank) {}

    /// Print this partial mapping on dbgs() stream.
    void dump() const;

    /// Print this partial mapping on \p OS;
    void print(raw_ostream &OS) const;

    /// Check that the Mask is compatible with the RegBank.
    /// Indeed, if the RegBank cannot accomadate the "active bits" of the mask,
    /// there is no way this mapping is valid.
    void verify() const;
  };

  /// Helper struct that represents how a value is mapped through
  /// different register banks.
  struct ValueMapping {
    /// How the value is broken down between the different register banks.
    SmallVector<PartialMapping, 2> BreakDown;
    /// Verify that this mapping makes sense for a value of \p ExpectedBitWidth.
    void verify(unsigned ExpectedBitWidth) const;
  };

  /// Helper class that represents how the value of an instruction may be
  /// mapped and what is the related cost of such mapping.
  class InstructionMapping {
    /// Identifier of the mapping.
    /// This is used to communicate between the target and the optimizers
    /// which mapping should be realized.
    unsigned ID;
    /// Cost of this mapping.
    unsigned Cost;
    /// Mapping of all the operands.
    std::unique_ptr<ValueMapping[]> OperandsMapping;
    /// Number of operands.
    unsigned NumOperands;

    ValueMapping &getOperandMapping(unsigned i) {
      assert(i < getNumOperands() && "Out of bound operand");
      return OperandsMapping[i];
    }

  public:
    /// Constructor for the mapping of an instruction.
    /// \p NumOperands must be equal to number of all the operands of
    /// the related instruction.
    /// The rationale is that it is more efficient for the optimizers
    /// to be able to assume that the mapping of the ith operand is
    /// at the index i.
    InstructionMapping(unsigned ID, unsigned Cost, unsigned NumOperands)
        : ID(ID), Cost(Cost), NumOperands(NumOperands) {
      OperandsMapping.reset(new ValueMapping[getNumOperands()]);
    }

    /// Get the cost.
    unsigned getCost() const { return Cost; }

    /// Get the ID.
    unsigned getID() const { return ID; }

    /// Get the number of operands.
    unsigned getNumOperands() const { return NumOperands; }

    /// Get the value mapping of the ith operand.
    const ValueMapping &getOperandMapping(unsigned i) const {
      return const_cast<InstructionMapping *>(this)->getOperandMapping(i);
    }

    /// Get the value mapping of the ith operand.
    void setOperandMapping(unsigned i, const ValueMapping &ValMapping) {
      getOperandMapping(i) = ValMapping;
    }

    /// Verifiy that this mapping makes sense for \p MI.
    void verify(const MachineInstr &MI) const;
  };

protected:
  /// Hold the set of supported register banks.
  std::unique_ptr<RegisterBank[]> RegBanks;
  /// Total number of register banks.
  unsigned NumRegBanks;

  /// Create a RegisterBankInfo that can accomodate up to \p NumRegBanks
  /// RegisterBank instances.
  ///
  /// \note For the verify method to succeed all the \p NumRegBanks
  /// must be initialized by createRegisterBank and updated with
  /// addRegBankCoverage RegisterBank.
  RegisterBankInfo(unsigned NumRegBanks);

  /// This constructor is meaningless.
  /// It just provides a default constructor that can be used at link time
  /// when GlobalISel is not built.
  /// That way, targets can still inherit from this class without doing
  /// crazy gymnastic to avoid link time failures.
  /// \note That works because the constructor is inlined.
  RegisterBankInfo() {
    llvm_unreachable("This constructor should not be executed");
  }

  virtual ~RegisterBankInfo() {}

  /// Create a new register bank with the given parameter and add it
  /// to RegBanks.
  /// \pre \p ID must not already be used.
  /// \pre \p ID < NumRegBanks.
  void createRegisterBank(unsigned ID, const char *Name);

  /// Add \p RCId to the set of register class that the register bank
  /// identified \p ID covers.
  /// This method transitively adds all the sub classes and the subreg-classes
  /// of \p RCId to the set of covered register classes.
  /// It also adjusts the size of the register bank to reflect the maximal
  /// size of a value that can be hold into that register bank.
  ///
  /// \note This method does *not* add the super classes of \p RCId.
  /// The rationale is if \p ID covers the registers of \p RCId, that
  /// does not necessarily mean that \p ID covers the set of registers
  /// of RCId's superclasses.
  /// This method does *not* add the superreg classes as well for consistents.
  /// The expected use is to add the coverage top-down with respect to the
  /// register hierarchy.
  ///
  /// \todo TableGen should just generate the BitSet vector for us.
  void addRegBankCoverage(unsigned ID, unsigned RCId,
                          const TargetRegisterInfo &TRI);

  /// Get the register bank identified by \p ID.
  RegisterBank &getRegBank(unsigned ID) {
    assert(ID < getNumRegBanks() && "Accessing an unknown register bank");
    return RegBanks[ID];
  }

public:
  /// Get the register bank identified by \p ID.
  const RegisterBank &getRegBank(unsigned ID) const {
    return const_cast<RegisterBankInfo *>(this)->getRegBank(ID);
  }

  /// Get the total number of register banks.
  unsigned getNumRegBanks() const { return NumRegBanks; }

  /// Get the cost of a copy from \p B to \p A, or put differently,
  /// get the cost of A = COPY B.
  virtual unsigned copyCost(const RegisterBank &A,
                            const RegisterBank &B) const {
    return 0;
  }

  void verify(const TargetRegisterInfo &TRI) const;
};

inline raw_ostream &
operator<<(raw_ostream &OS,
           const RegisterBankInfo::PartialMapping &PartMapping) {
  PartMapping.print(OS);
  return OS;
}
} // End namespace llvm.

#endif
