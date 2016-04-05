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

#include <memory>

namespace llvm {
class RegisterBank;
class TargetRegisterInfo;

/// Holds all the information related to register banks.
class RegisterBankInfo {
protected:
  std::unique_ptr<RegisterBank[]> RegBanks;
  unsigned NbOfRegBanks;

  RegisterBankInfo(unsigned NbOfRegBanks);

  virtual ~RegisterBankInfo();

public:
  /// Get the register bank identified by \p ID.
  const RegisterBank &getRegBank(unsigned ID) const {
    assert(ID < NbOfRegBanks && "Accessing an unknown register bank");
    return RegBanks[ID];
  }

  /// Get the cost of a copy from \p B to \p A, or put differently,
  /// get the cost of A = COPY B.
  virtual unsigned copyCost(const RegisterBank &A,
                            const RegisterBank &B) const {
    return 0;
  }

  void verify(const TargetRegisterInfo &TRI) const;
};
} // End namespace llvm.

#endif
