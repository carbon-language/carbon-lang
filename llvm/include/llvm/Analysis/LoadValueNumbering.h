//===- llvm/Analysis/LoadValueNumbering.h - Value # Load Insts --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines a value numbering pass that value #'s load instructions.
// To do this, it finds lexically identical load instructions, and uses alias
// analysis to determine which loads are guaranteed to produce the same value.
//
// This pass builds off of another value numbering pass to implement value
// numbering for non-load instructions.  It uses Alias Analysis so that it can
// disambiguate the load instructions.  The more powerful these base analyses
// are, the more powerful the resultant analysis will be.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOAD_VALUE_NUMBERING_H
#define LLVM_ANALYSIS_LOAD_VALUE_NUMBERING_H

namespace llvm {

class Pass;

/// createLoadValueNumberingPass - Create and return a new pass that implements
/// the ValueNumbering interface.
///
Pass *createLoadValueNumberingPass();

} // End llvm namespace

#endif
