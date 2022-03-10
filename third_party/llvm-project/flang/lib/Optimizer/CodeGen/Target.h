//===- Target.h - target specific details -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTMIZER_CODEGEN_TARGET_H
#define FORTRAN_OPTMIZER_CODEGEN_TARGET_H

#include "flang/Optimizer/Support/KindMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/Triple.h"
#include <memory>
#include <tuple>
#include <vector>

namespace fir {

namespace details {
/// Extra information about how to marshal an argument or return value that
/// modifies a signature per a particular ABI's calling convention.
/// Note: llvm::Attribute is not used directly, because its use depends on an
/// LLVMContext.
class Attributes {
public:
  Attributes(unsigned short alignment = 0, bool byval = false,
             bool sret = false, bool append = false)
      : alignment{alignment}, byval{byval}, sret{sret}, append{append} {}

  unsigned getAlignment() const { return alignment; }
  bool hasAlignment() const { return alignment != 0; }
  bool isByVal() const { return byval; }
  bool isSRet() const { return sret; }
  bool isAppend() const { return append; }

private:
  unsigned short alignment{};
  bool byval : 1;
  bool sret : 1;
  bool append : 1;
};

} // namespace details

/// Some details of how to represent certain features depend on the target and
/// ABI that is being used.  These specifics are captured here and guide the
/// lowering of FIR to LLVM-IR dialect.
class CodeGenSpecifics {
public:
  using Attributes = details::Attributes;
  using Marshalling = std::vector<std::tuple<mlir::Type, Attributes>>;

  static std::unique_ptr<CodeGenSpecifics>
  get(mlir::MLIRContext *ctx, llvm::Triple &&trp, KindMapping &&kindMap);

  CodeGenSpecifics(mlir::MLIRContext *ctx, llvm::Triple &&trp,
                   KindMapping &&kindMap)
      : context{*ctx}, triple{std::move(trp)}, kindMap{std::move(kindMap)} {}
  CodeGenSpecifics() = delete;
  virtual ~CodeGenSpecifics() {}

  /// Type presentation of a `complex<ele>` type value in memory.
  virtual mlir::Type complexMemoryType(mlir::Type eleTy) const = 0;

  /// Type representation of a `complex<eleTy>` type argument when passed by
  /// value. An argument value may need to be passed as a (safe) reference
  /// argument.
  virtual Marshalling complexArgumentType(mlir::Type eleTy) const = 0;

  /// Type representation of a `complex<eleTy>` type return value. Such a return
  /// value may need to be converted to a hidden reference argument.
  virtual Marshalling complexReturnType(mlir::Type eleTy) const = 0;

  /// Type presentation of a `boxchar<n>` type value in memory.
  virtual mlir::Type boxcharMemoryType(mlir::Type eleTy) const = 0;

  /// Type representation of a `boxchar<n>` type argument when passed by value.
  /// An argument value may need to be passed as a (safe) reference argument.
  ///
  /// A function that returns a `boxchar<n>` type value must already have
  /// converted that return value to a parameter decorated with the 'sret'
  /// Attribute (https://llvm.org/docs/LangRef.html#parameter-attributes).
  /// This requirement is in keeping with Fortran semantics, which require the
  /// caller to allocate the space for the return CHARACTER value and pass
  /// a pointer and the length of that space (a boxchar) to the called function.
  virtual Marshalling boxcharArgumentType(mlir::Type eleTy,
                                          bool sret = false) const = 0;

protected:
  mlir::MLIRContext &context;
  llvm::Triple triple;
  KindMapping kindMap;
};

} // namespace fir

#endif // FORTRAN_OPTMIZER_CODEGEN_TARGET_H
