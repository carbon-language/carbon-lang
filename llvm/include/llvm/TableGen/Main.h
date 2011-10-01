//===- llvm/TableGen/Main.h - tblgen entry point ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the common entry point for tblgen tools.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_MAIN_H
#define LLVM_TABLEGEN_MAIN_H

namespace llvm {

class TableGenAction;

/// Run the table generator, performing the specified Action on parsed records.
int TableGenMain(char *argv0, TableGenAction &Action);

}

#endif
