//===-- ModuleProvider.cpp - Base implementation for module providers -----===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Minimal implementation of the abstract interface for providing a module.
//
//===----------------------------------------------------------------------===//

#include "llvm/ModuleProvider.h"
#include "llvm/Module.h"

/// ctor - always have a valid Module
///
ModuleProvider::ModuleProvider() : TheModule(0) { }

/// dtor - when we leave, we take our Module with us
///
ModuleProvider::~ModuleProvider() {
  delete TheModule;
}

/// materializeFunction - make sure the given function is fully read.
///
Module* ModuleProvider::materializeModule() {
  assert(TheModule && "Attempting to materialize an invalid module!");

  for (Module::iterator i = TheModule->begin(), e = TheModule->end();
       i != e; ++i)
    materializeFunction(i);

  return TheModule;
}
