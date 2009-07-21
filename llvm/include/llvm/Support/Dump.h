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

#ifndef LLVM_SUPPORT_DUMP_H
#define LLVM_SUPPORT_DUMP_H

namespace llvm {

class MachineBasicBlock;
class MachineInstr;

// PrefixPrinter - Print some additional information before printing
// basic blocks and instructions.
class PrefixPrinter {
public:
  virtual ~PrefixPrinter() {}

  virtual std::string operator()(const MachineBasicBlock &) const {
    return("");
  };

  virtual std::string operator()(const MachineInstr &) const {
    return("");
  };  
};
 
} // End llvm namespace

#endif
