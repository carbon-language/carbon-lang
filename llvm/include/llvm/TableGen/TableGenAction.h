//===- llvm/TableGen/TableGenAction.h - defines TableGenAction --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TableGenAction base class to be derived from by
// tblgen tools.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_TABLEGENACTION_H
#define LLVM_TABLEGEN_TABLEGENACTION_H

namespace llvm {

class raw_ostream;
class RecordKeeper;

class TableGenAction {
public:
  virtual ~TableGenAction() {}

  /// Perform the action using Records, and write output to OS.
  /// @returns true on error, false otherwise
  virtual bool operator()(raw_ostream &OS, RecordKeeper &Records) = 0;
};

}

#endif
