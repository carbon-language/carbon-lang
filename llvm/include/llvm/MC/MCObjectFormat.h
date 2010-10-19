//===-- llvm/MC/MCObjectFormat.h - Object Format Info -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCOBJECTFORMAT_H
#define LLVM_MC_MCOBJECTFORMAT_H

namespace llvm {
class MCSymbol;

class MCObjectFormat {
public:
  virtual ~MCObjectFormat();

  /// isAbsolute - Check if A - B is an absolute value
  ///
  /// \param InSet - True if this expression is in a set. For example:
  ///   a:
  ///   ...
  ///   b:
  ///   tmp = a - b
  ///       .long tmp
  /// \param A - LHS
  /// \param B - RHS
  virtual bool isAbsolute(bool InSet, const MCSymbol &A,
                          const MCSymbol &B) const = 0;
};

class MCELFObjectFormat : public MCObjectFormat {
public:
  virtual bool isAbsolute(bool InSet, const MCSymbol &A,
                          const MCSymbol &B) const;
};

class MCMachOObjectFormat : public MCObjectFormat {
public:
  virtual bool isAbsolute(bool InSet, const MCSymbol &A,
                          const MCSymbol &B) const;
};

class MCCOFFObjectFormat : public MCObjectFormat {
public:
  virtual bool isAbsolute(bool InSet, const MCSymbol &A,
                          const MCSymbol &B) const;
};

}  // End llvm namespace

#endif
