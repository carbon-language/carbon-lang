//===- Attributes.cpp - MLIR Affine Expr Classes --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// Attribute
//===----------------------------------------------------------------------===//

/// Return the context this attribute belongs to.
MLIRContext *Attribute::getContext() const { return getDialect().getContext(); }

//===----------------------------------------------------------------------===//
// NamedAttribute
//===----------------------------------------------------------------------===//

NamedAttribute::NamedAttribute(StringAttr name, Attribute value)
    : name(name), value(value) {
  assert(name && value && "expected valid attribute name and value");
  assert(name.size() != 0 && "expected valid attribute name");
}

StringAttr NamedAttribute::getName() const { return name.cast<StringAttr>(); }

Dialect *NamedAttribute::getNameDialect() const {
  return getName().getReferencedDialect();
}

void NamedAttribute::setName(StringAttr newName) {
  assert(name && "expected valid attribute name");
  name = newName;
}

bool NamedAttribute::operator<(const NamedAttribute &rhs) const {
  return getName().compare(rhs.getName()) < 0;
}

bool NamedAttribute::operator<(StringRef rhs) const {
  return getName().getValue().compare(rhs) < 0;
}
