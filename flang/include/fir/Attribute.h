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

#ifndef FIR_ATTRIBUTE_H
#define FIR_ATTRIBUTE_H

#include "mlir/IR/Attributes.h"

namespace fir {

class FIROpsDialect;

namespace detail {
struct TypeAttributeStorage;
}

enum AttributeKind {
  FIR_ATTR = mlir::Attribute::FIRST_FIR_ATTR,
  FIR_EXACTTYPE,  // instance_of, precise type relation
  FIR_SUBCLASS,  // subsumed_by, is-a (subclass) relation
  FIR_POINT,
  FIR_CLOSEDCLOSED_INTERVAL,
  FIR_OPENCLOSED_INTERVAL,
  FIR_CLOSEDOPEN_INTERVAL,
};

class ExactTypeAttr : public mlir::Attribute::AttrBase<ExactTypeAttr,
                          mlir::Attribute, detail::TypeAttributeStorage> {
public:
  using Base::Base;
  using ValueType = mlir::Type;

  static llvm::StringRef getAttrName() { return "instance"; }
  static ExactTypeAttr get(mlir::Type value);

  mlir::Type getType() const;

  constexpr static bool kindof(unsigned kind) { return kind == getId(); }
  constexpr static unsigned getId() { return AttributeKind::FIR_EXACTTYPE; }
};

class SubclassAttr : public mlir::Attribute::AttrBase<SubclassAttr,
                         mlir::Attribute, detail::TypeAttributeStorage> {
public:
  using Base::Base;
  using ValueType = mlir::Type;

  static llvm::StringRef getAttrName() { return "subsumed"; }
  static SubclassAttr get(mlir::Type value);

  mlir::Type getType() const;

  constexpr static bool kindof(unsigned kind) { return kind == getId(); }
  constexpr static unsigned getId() { return AttributeKind::FIR_SUBCLASS; }
};

class ClosedIntervalAttr
  : public mlir::Attribute::AttrBase<ClosedIntervalAttr> {
public:
  using Base::Base;

  static llvm::StringRef getAttrName() { return "interval"; }
  static ClosedIntervalAttr get(mlir::MLIRContext *ctxt);
  constexpr static bool kindof(unsigned kind) { return kind == getId(); }
  constexpr static unsigned getId() {
    return AttributeKind::FIR_CLOSEDCLOSED_INTERVAL;
  }
};

class UpperBoundAttr : public mlir::Attribute::AttrBase<UpperBoundAttr> {
public:
  using Base::Base;

  static llvm::StringRef getAttrName() { return "upper"; }
  static UpperBoundAttr get(mlir::MLIRContext *ctxt);
  constexpr static bool kindof(unsigned kind) { return kind == getId(); }
  constexpr static unsigned getId() {
    return AttributeKind::FIR_OPENCLOSED_INTERVAL;
  }
};

class LowerBoundAttr : public mlir::Attribute::AttrBase<LowerBoundAttr> {
public:
  using Base::Base;

  static llvm::StringRef getAttrName() { return "lower"; }
  static LowerBoundAttr get(mlir::MLIRContext *ctxt);
  constexpr static bool kindof(unsigned kind) { return kind == getId(); }
  constexpr static unsigned getId() {
    return AttributeKind::FIR_CLOSEDOPEN_INTERVAL;
  }
};

class PointIntervalAttr : public mlir::Attribute::AttrBase<PointIntervalAttr> {
public:
  using Base::Base;

  static llvm::StringRef getAttrName() { return "point"; }
  static PointIntervalAttr get(mlir::MLIRContext *ctxt);
  constexpr static bool kindof(unsigned kind) { return kind == getId(); }
  constexpr static unsigned getId() { return AttributeKind::FIR_POINT; }
};

mlir::Attribute parseFirAttribute(FIROpsDialect *dialect,
    llvm::StringRef rawText, mlir::Type type, mlir::Location loc);

}  // fir

#endif  // FIR_ATTRIBUTE_H
