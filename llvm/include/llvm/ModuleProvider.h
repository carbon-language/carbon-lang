//===-- llvm/ModuleProvider.h - Interface for module providers --*- C++ -*-===//
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
  void materializeModule();

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

#endif
