//===-- llvm/ModuleProvider.h - Interface for module providers --*- C++ -*-===//
//
// Abstract interface for providing a module.
//
//===----------------------------------------------------------------------===//

#ifndef MODULEPROVIDER_H
#define MODULEPROVIDER_H

class Function;
class Module;

class AbstractModuleProvider {
protected:
  Module *TheModule;
  AbstractModuleProvider();

public:
  virtual ~AbstractModuleProvider();

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
