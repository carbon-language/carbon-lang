//===-- llvm/ModuleProvider.h - Interface for module providers --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides an abstract interface for loading a module from some
// place.  This interface allows incremental or random access loading of
// functions from the file.  This is useful for applications like JIT compilers
// or interprocedural optimizers that do not need the entire program in memory
// at the same time.
//
//===----------------------------------------------------------------------===//

#ifndef MODULEPROVIDER_H
#define MODULEPROVIDER_H

namespace llvm {

class Function;
class Module;

class ModuleProvider {
protected:
  Module *TheModule;
  ModuleProvider();

public:
  virtual ~ModuleProvider();

  /// getModule - returns the module this provider is encapsulating.
  ///
  Module* getModule() { return TheModule; }

  /// materializeFunction - make sure the given function is fully read.
  ///
  virtual void materializeFunction(Function *F) = 0;

  /// materializeModule - make sure the entire Module has been completely read.
  ///
  Module* materializeModule();

  /// releaseModule - no longer delete the Module* when provider is destroyed.
  ///
  virtual Module* releaseModule() { 
    // Since we're losing control of this Module, we must hand it back complete
    materializeModule();
    Module *tempM = TheModule; 
    TheModule = 0; 
    return tempM; 
  }
};


/// ExistingModuleProvider - Allow conversion from a fully materialized Module
/// into a ModuleProvider, allowing code that expects a ModuleProvider to work
/// if we just have a Module.  Note that the ModuleProvider takes ownership of
/// the Module specified.
struct ExistingModuleProvider : public ModuleProvider {
  ExistingModuleProvider(Module *M) {
    TheModule = M;
  }
  void materializeFunction(Function *F) {}
};

} // End llvm namespace

#endif
