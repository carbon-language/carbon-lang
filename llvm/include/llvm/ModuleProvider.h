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
  Module *M;

protected:
  AbstractModuleProvider();

public:
  virtual ~AbstractModuleProvider();

  /// getModule - returns the module this provider is encapsulating
  ///
  Module* getModule() { return M; }

  /// materializeFunction - make sure the given function is fully read.
  ///
  virtual void materializeFunction(Function *F) = 0;

  /// materializeModule - make sure the entire Module has been completely read.
  ///
  void materializeModule();

  /// releaseModule - no longer delete the Module* when provider is destroyed.
  ///
  Module* releaseModule() { Module *tempM = M; M = 0; return tempM; }

};

#endif
