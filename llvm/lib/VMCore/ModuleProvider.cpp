//===-- ModuleProvider.cpp - Base implementation for module providers -----===//
//
// Minimal implementation of the abstract interface for providing a module.
//
//===----------------------------------------------------------------------===//

#include "llvm/ModuleProvider.h"
#include "llvm/Module.h"

/// ctor - always have a valid Module
///
AbstractModuleProvider::AbstractModuleProvider() {
  M = new Module("");
}

/// dtor - when we leave, we take our Module with us
///
AbstractModuleProvider::~AbstractModuleProvider() {
  delete M;
}

/// materializeFunction - make sure the given function is fully read.
///
void AbstractModuleProvider::materializeModule() {
  for (Module::iterator i = M->begin(), e = M->end(); i != e; ++i) {
    materializeFunction(i);
  }
}
