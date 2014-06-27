//===---- RuntimeDyldChecker.h - RuntimeDyld tester framework -----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_RUNTIMEDYLDCHECKER_H
#define LLVM_RUNTIMEDYLDCHECKER_H

#include "RuntimeDyld.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

namespace llvm {

class MCDisassembler;
class MCInstPrinter;

/// \brief RuntimeDyld invariant checker for verifying that RuntimeDyld has
///        correctly applied relocations.
///
/// The RuntimeDyldChecker class evaluates expressions against an attached
/// RuntimeDyld instance to verify that relocations have been applied
/// correctly.
///
/// The expression language supports basic pointer arithmetic and bit-masking,
/// and has limited disassembler integration for accessing instruction
/// operands and the next PC (program counter) address for each instruction.
///
/// The language syntax is:
///
/// check = expr '=' expr
///
/// expr = binary_expr
///      | sliceable_expr
///
/// sliceable_expr = '*{' number '}' load_addr_expr [slice]
///                | '(' expr ')' [slice]
///                | ident_expr [slice]
///                | number [slice]
///
/// slice = '[' high-bit-index ':' low-bit-index ']'
///
/// load_addr_expr = symbol
///                | '(' symbol '+' number ')'
///                | '(' symbol '-' number ')'
///
/// ident_expr = 'decode_operand' '(' symbol ',' operand-index ')'
///            | 'next_pc'        '(' symbol ')'
///            | symbol
///
/// binary_expr = expr '+' expr
///             | expr '-' expr
///             | expr '&' expr
///             | expr '|' expr
///             | expr '<<' expr
///             | expr '>>' expr
///
class RuntimeDyldChecker {
  friend class RuntimeDyldCheckerExprEval;
public:
  RuntimeDyldChecker(RuntimeDyld &RTDyld,
                     MCDisassembler *Disassembler,
                     MCInstPrinter *InstPrinter,
                     llvm::raw_ostream &ErrStream)
    : RTDyld(*RTDyld.Dyld), Disassembler(Disassembler),
      InstPrinter(InstPrinter), ErrStream(ErrStream) {}

  /// \brief Check a single expression against the attached RuntimeDyld
  ///        instance.
  bool check(StringRef CheckExpr) const;

  /// \brief Scan the given memory buffer for lines beginning with the string
  ///        in RulePrefix. The remainder of the line is passed to the check
  ///        method to be evaluated as an expression.
  bool checkAllRulesInBuffer(StringRef RulePrefix, MemoryBuffer *MemBuf) const;

private:

  bool checkSymbolIsValidForLoad(StringRef Symbol) const;
  uint64_t getSymbolAddress(StringRef Symbol) const;
  uint64_t readMemoryAtSymbol(StringRef Symbol, int64_t Offset,
                              unsigned Size) const;
  StringRef getSubsectionStartingAt(StringRef Name) const;

  RuntimeDyldImpl &RTDyld;
  MCDisassembler *Disassembler;
  MCInstPrinter *InstPrinter;
  llvm::raw_ostream &ErrStream;
};

} // end namespace llvm

#endif // LLVM_RUNTIMEDYLDCHECKER_H
