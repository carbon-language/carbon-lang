//===- llvm/Support/Dump.h - Easy way to tailor dump output -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the PrefixPrinter interface to pass to MachineFunction
// and MachineBasicBlock print methods to output additional information before
// blocks and instructions are printed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DUMP_H
#define LLVM_CODEGEN_DUMP_H

#include <iosfwd>

namespace llvm {

class MachineBasicBlock;
class MachineInstr;
class raw_ostream;

/// PrefixPrinter - Print some additional information before printing
/// basic blocks and instructions.
class PrefixPrinter {
public:
  virtual ~PrefixPrinter();

  /// operator() - Print a prefix before each MachineBasicBlock
  virtual raw_ostream &operator()(raw_ostream &out,
                                  const MachineBasicBlock &) const {
    return out; 
  }

  /// operator() - Print a prefix before each MachineInstr
  virtual raw_ostream &operator()(raw_ostream &out,
                                  const MachineInstr &) const {
    return out; 
  }

  /// operator() - Print a prefix before each MachineBasicBlock
  virtual std::ostream &operator()(std::ostream &out,
                                   const MachineBasicBlock &) const {
    return out; 
  }

  /// operator() - Print a prefix before each MachineInstr
  virtual std::ostream &operator()(std::ostream &out,
                                   const MachineInstr &) const {
    return out; 
  }
};
 
} // End llvm namespace

#endif
