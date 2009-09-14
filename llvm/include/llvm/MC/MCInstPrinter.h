//===-- MCInstPrinter.h - Convert an MCInst to target assembly syntax -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCINSTPRINTER_H
#define LLVM_MC_MCINSTPRINTER_H

namespace llvm {
class MCInst;
class raw_ostream;
  
/// MCInstPrinter - This is an instance of a target assembly language printer
/// that converts an MCInst to valid target assembly syntax.
class MCInstPrinter {
  raw_ostream &O;
public:
  MCInstPrinter(raw_ostream &o) : O(o) {}
  
  virtual ~MCInstPrinter();
  
  /// printInst - Print the specified MCInst to the current raw_ostream.
  ///
  virtual void printInst(const MCInst *MI) = 0;
};
  
} // namespace llvm

#endif
