//===- DialectLinalg.cpp - Pybind module for Linalg dialect API support --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects.h"
#include "IRModule.h"
#include "mlir-c/Dialect/Linalg.h"
#include "mlir-c/IR.h"

// TODO: Port this to operate only on the public PybindAdaptors.h
#include "PybindUtils.h"

namespace py = pybind11;
using namespace mlir;
using namespace mlir::python;

void mlir::python::populateDialectLinalgSubmodule(py::module m) {
  m.def(
      "fill_builtin_region",
      [](PyDialectDescriptor &dialect, PyOperation &op) {
        mlirLinalgFillBuiltinNamedOpRegion(dialect.get(), op.get());
      },
      py::arg("dialect"), py::arg("op"),
      "Fill the region for `op`, which is assumed to be a builtin named Linalg "
      "op.");
}
