//===- PythonTestModule.cpp - Python extension for the PythonTest dialect -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PythonTestCAPI.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;

PYBIND11_MODULE(_mlirPythonTest, m) {
  m.def(
      "register_python_test_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle pythonTestDialect =
            mlirGetDialectHandle__python_test__();
        mlirDialectHandleRegisterDialect(pythonTestDialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(pythonTestDialect, context);
        }
      },
      py::arg("context"), py::arg("load") = true);
}
