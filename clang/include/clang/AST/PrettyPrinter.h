//===--- PrettyPrinter.h - Classes for aiding with AST printing -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PrinterHelper interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_PRETTY_PRINTER_H
#define LLVM_CLANG_AST_PRETTY_PRINTER_H

#include "llvm/Support/raw_ostream.h"

namespace clang {

class Stmt;
  
class PrinterHelper {
public:
  virtual ~PrinterHelper();
  virtual bool handledStmt(Stmt* E, llvm::raw_ostream& OS) = 0;
};

} // end namespace clang

#endif
