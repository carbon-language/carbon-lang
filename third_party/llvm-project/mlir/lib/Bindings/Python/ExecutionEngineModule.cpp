//===- ExecutionEngineModule.cpp - Python module for execution engine -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/ExecutionEngine.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir;
using namespace mlir::python;

namespace {

/// Owning Wrapper around an ExecutionEngine.
class PyExecutionEngine {
public:
  PyExecutionEngine(MlirExecutionEngine executionEngine)
      : executionEngine(executionEngine) {}
  PyExecutionEngine(PyExecutionEngine &&other)
      : executionEngine(other.executionEngine) {
    other.executionEngine.ptr = nullptr;
  }
  ~PyExecutionEngine() {
    if (!mlirExecutionEngineIsNull(executionEngine))
      mlirExecutionEngineDestroy(executionEngine);
  }
  MlirExecutionEngine get() { return executionEngine; }

  void release() {
    executionEngine.ptr = nullptr;
    referencedObjects.clear();
  }
  pybind11::object getCapsule() {
    return py::reinterpret_steal<py::object>(
        mlirPythonExecutionEngineToCapsule(get()));
  }

  // Add an object to the list of referenced objects whose lifetime must exceed
  // those of the ExecutionEngine.
  void addReferencedObject(const pybind11::object &obj) {
    referencedObjects.push_back(obj);
  }

  static pybind11::object createFromCapsule(pybind11::object capsule) {
    MlirExecutionEngine rawPm =
        mlirPythonCapsuleToExecutionEngine(capsule.ptr());
    if (mlirExecutionEngineIsNull(rawPm))
      throw py::error_already_set();
    return py::cast(PyExecutionEngine(rawPm), py::return_value_policy::move);
  }

private:
  MlirExecutionEngine executionEngine;
  // We support Python ctypes closures as callbacks. Keep a list of the objects
  // so that they don't get garbage collected. (The ExecutionEngine itself
  // just holds raw pointers with no lifetime semantics).
  std::vector<py::object> referencedObjects;
};

} // namespace

/// Create the `mlir.execution_engine` module here.
PYBIND11_MODULE(_mlirExecutionEngine, m) {
  m.doc() = "MLIR Execution Engine";

  //----------------------------------------------------------------------------
  // Mapping of the top-level PassManager
  //----------------------------------------------------------------------------
  py::class_<PyExecutionEngine>(m, "ExecutionEngine", py::module_local())
      .def(py::init<>([](MlirModule module, int optLevel,
                         const std::vector<std::string> &sharedLibPaths) {
             llvm::SmallVector<MlirStringRef, 4> libPaths;
             for (const std::string &path : sharedLibPaths)
               libPaths.push_back({path.c_str(), path.length()});
             MlirExecutionEngine executionEngine = mlirExecutionEngineCreate(
                 module, optLevel, libPaths.size(), libPaths.data());
             if (mlirExecutionEngineIsNull(executionEngine))
               throw std::runtime_error(
                   "Failure while creating the ExecutionEngine.");
             return new PyExecutionEngine(executionEngine);
           }),
           py::arg("module"), py::arg("opt_level") = 2,
           py::arg("shared_libs") = py::list(),
           "Create a new ExecutionEngine instance for the given Module. The "
           "module must contain only dialects that can be translated to LLVM. "
           "Perform transformations and code generation at the optimization "
           "level `opt_level` if specified, or otherwise at the default "
           "level of two (-O2). Load a list of libraries specified in "
           "`shared_libs`.")
      .def_property_readonly(MLIR_PYTHON_CAPI_PTR_ATTR,
                             &PyExecutionEngine::getCapsule)
      .def("_testing_release", &PyExecutionEngine::release,
           "Releases (leaks) the backing ExecutionEngine (for testing purpose)")
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyExecutionEngine::createFromCapsule)
      .def(
          "raw_lookup",
          [](PyExecutionEngine &executionEngine, const std::string &func) {
            auto *res = mlirExecutionEngineLookupPacked(
                executionEngine.get(),
                mlirStringRefCreate(func.c_str(), func.size()));
            return reinterpret_cast<uintptr_t>(res);
          },
          py::arg("func_name"),
          "Lookup function `func` in the ExecutionEngine.")
      .def(
          "raw_register_runtime",
          [](PyExecutionEngine &executionEngine, const std::string &name,
             py::object callbackObj) {
            executionEngine.addReferencedObject(callbackObj);
            uintptr_t rawSym =
                py::cast<uintptr_t>(py::getattr(callbackObj, "value"));
            mlirExecutionEngineRegisterSymbol(
                executionEngine.get(),
                mlirStringRefCreate(name.c_str(), name.size()),
                reinterpret_cast<void *>(rawSym));
          },
          py::arg("name"), py::arg("callback"),
          "Register `callback` as the runtime symbol `name`.")
      .def(
          "dump_to_object_file",
          [](PyExecutionEngine &executionEngine, const std::string &fileName) {
            mlirExecutionEngineDumpToObjectFile(
                executionEngine.get(),
                mlirStringRefCreate(fileName.c_str(), fileName.size()));
          },
          py::arg("file_name"), "Dump ExecutionEngine to an object file.");
}
