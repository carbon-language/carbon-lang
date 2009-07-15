//===-- Target/TargetMachineRegistry.h - Target Registration ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes two classes: the TargetMachineRegistry class, which allows
// tools to inspect all of registered targets, and the RegisterTarget class,
// which TargetMachine implementations should use to register themselves with
// the system.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETMACHINEREGISTRY_H
#define LLVM_TARGET_TARGETMACHINEREGISTRY_H

#include "llvm/Module.h"
#include "llvm/Support/Registry.h"
#include "llvm/Target/TargetRegistry.h"

namespace llvm {
  class Module;
  class Target;
  class TargetMachine;

  struct TargetMachineRegistryEntry {
    const Target &TheTarget;
    const char *Name;
    const char *ShortDesc;

  public:
    TargetMachineRegistryEntry(const Target &T, const char *N, const char *SD)
      : TheTarget(T), Name(N), ShortDesc(SD) {}
  };

  template<>
  class RegistryTraits<TargetMachine> {
  public:
    typedef TargetMachineRegistryEntry entry;

    static const char *nameof(const entry &Entry) { return Entry.Name; }
    static const char *descof(const entry &Entry) { return Entry.ShortDesc; }
  };

  struct TargetMachineRegistry : public Registry<TargetMachine> {

  };

  //===--------------------------------------------------------------------===//
  /// RegisterTarget - This class is used to make targets automatically register
  /// themselves with the tools they are linked with.  Targets should define an
  /// single global Target instance and register it using the TargetRegistry
  /// interfaces. Targets must also include a static instance of this class.
  ///
  /// The type 'TargetMachineImpl' should provide a constructor with two
  /// parameters:
  /// - const Module& M: the module that is being compiled:
  /// - const std::string& FS: target-specific string describing target
  ///   flavour.

  template<class TargetMachineImpl>
  struct RegisterTarget {
    RegisterTarget(Target &T, const char *Name, const char *ShortDesc)
      : Entry(T, Name, ShortDesc),
        Node(Entry) {
      TargetRegistry::RegisterTargetMachine(T, &Allocator);
    }

  private:
    TargetMachineRegistry::entry Entry;
    TargetMachineRegistry::node Node;

    static TargetMachine *Allocator(const Module &M, const std::string &FS) {
      return new TargetMachineImpl(M, FS);
    }
  };

}

#endif
