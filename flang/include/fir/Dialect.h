// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FIR_DIALECT_H
#define FIR_DIALECT_H

#include "mlir/IR/Dialect.h"

namespace llvm {
class raw_ostream;
class StringRef;
}

namespace mlir {
class Attribute;
class Location;
class MLIRContext;
class Type;
}

namespace fir {

/// FIR dialect
class FIROpsDialect final : public mlir::Dialect {
public:
  explicit FIROpsDialect(mlir::MLIRContext *ctx);
  virtual ~FIROpsDialect();

  static llvm::StringRef getDialectNamespace() { return "fir"; }

  mlir::Type parseType(
      llvm::StringRef rawData, mlir::Location loc) const override;
  void printType(mlir::Type ty, llvm::raw_ostream &os) const override;

  mlir::Attribute parseAttribute(llvm::StringRef attrData, mlir::Type type,
      mlir::Location loc) const override;
  void printAttribute(
      mlir::Attribute attr, llvm::raw_ostream &os) const override;
};

}  // fir

#endif  // FIR_DIALECT_H
