//===-- Target/TargetMachineRegistry.h - Target Registration ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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

namespace llvm {
  class Module;
  class TargetMachine;
  class IntrinsicLowering;

  struct TargetMachineRegistry {
    /// Entry - One instance of this struct is created for each target that is
    /// registered.
    struct Entry {
      const char *Name;
      const char *ShortDesc;
      TargetMachine *(*CtorFn)(const Module &, IntrinsicLowering*);
      unsigned (*ModuleMatchQualityFn)(const Module &M);
      unsigned (*JITMatchQualityFn)();

      const Entry *getNext() const { return Next; }

    protected:
      Entry(const char *N, const char *SD,
            TargetMachine *(*CF)(const Module &, IntrinsicLowering*),
            unsigned (*MMF)(const Module &M), unsigned (*JMF)())
      : Name(N), ShortDesc(SD), CtorFn(CF), ModuleMatchQualityFn(MMF),
      JITMatchQualityFn(JMF), Next(List) {
        List = this;
      }
    private:
      const Entry *Next;  // Next entry in the linked list.
    };

    /// TargetMachineRegistry::getList - This static method returns the list of
    /// target machines that are registered with the system.
    static const Entry *getList() { return List; }

  private:
    static const Entry *List;
  };

  //===--------------------------------------------------------------------===//
  /// RegisterTarget - This class is used to make targets automatically register
  /// themselves with the tool they are linked.  Targets should define an
  /// instance of this and implement the static methods described in the
  /// TargetMachine comments..
  template<class TargetMachineImpl>
  struct RegisterTarget : public TargetMachineRegistry::Entry {
    RegisterTarget(const char *Name, const char *ShortDesc) : 
      TargetMachineRegistry::Entry(Name, ShortDesc, &Allocator,
                                   &TargetMachineImpl::getModuleMatchQuality,
                                   &TargetMachineImpl::getJITMatchQuality) {
    }
  private:
    static TargetMachine *Allocator(const Module &M, IntrinsicLowering *IL) {
      return new TargetMachineImpl(M, IL);
    }
  };
}

#endif
