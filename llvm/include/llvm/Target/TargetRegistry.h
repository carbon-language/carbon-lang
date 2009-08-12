//===-- Target/TargetRegistry.h - Target Registration -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes the TargetRegistry interface, which tools can use to access
// the appropriate target specific classes (TargetMachine, AsmPrinter, etc.)
// which have been registered.
//
// Target specific class implementations should register themselves using the
// appropriate TargetRegistry interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETREGISTRY_H
#define LLVM_TARGET_TARGETREGISTRY_H

#include "llvm/ADT/Triple.h"
// FIXME: We shouldn't need this header, but we need it until there is a
// different interface to get the TargetAsmInfo.
#include "llvm/Target/TargetMachine.h"
#include <string>
#include <cassert>

namespace llvm {
  class FunctionPass;
  class MCAsmParser;
  class Module;
  class TargetAsmParser;
  class TargetMachine;
  class formatted_raw_ostream;

  /// Target - Wrapper for Target specific information.
  ///
  /// For registration purposes, this is a POD type so that targets can be
  /// registered without the use of static constructors.
  ///
  /// Targets should implement a single global instance of this class (which
  /// will be zero initialized), and pass that instance to the TargetRegistry as
  /// part of their initialization.
  class Target {
  public:
    friend struct TargetRegistry;

    typedef unsigned (*TripleMatchQualityFnTy)(const std::string &TT);

    typedef const TargetAsmInfo *(*AsmInfoCtorFnTy)(const Target &T,
                                                    const StringRef &TT);
    typedef TargetMachine *(*TargetMachineCtorTy)(const Target &T,
                                                  const std::string &TT,
                                                  const std::string &Features);
    typedef FunctionPass *(*AsmPrinterCtorTy)(formatted_raw_ostream &,
                                              TargetMachine &,
                                              bool);
    typedef TargetAsmParser *(*AsmParserCtorTy)(const Target &,
                                                MCAsmParser &);
  private:
    /// Next - The next registered target in the linked list, maintained by the
    /// TargetRegistry.
    Target *Next;

    /// TripleMatchQualityFn - The target function for rating the match quality
    /// of a triple.
    TripleMatchQualityFnTy TripleMatchQualityFn;

    /// Name - The target name.
    const char *Name;

    /// ShortDesc - A short description of the target.
    const char *ShortDesc;

    /// HasJIT - Whether this target supports the JIT.
    bool HasJIT;

    AsmInfoCtorFnTy AsmInfoCtorFn;
    
    /// TargetMachineCtorFn - Construction function for this target's
    /// TargetMachine, if registered.
    TargetMachineCtorTy TargetMachineCtorFn;

    /// AsmPrinterCtorFn - Construction function for this target's AsmPrinter,
    /// if registered.
    AsmPrinterCtorTy AsmPrinterCtorFn;

    /// AsmParserCtorFn - Construction function for this target's AsmParser,
    /// if registered.
    AsmParserCtorTy AsmParserCtorFn;

  public:
    // getNext - Return the next registered target.
    const Target *getNext() const { return Next; }

    /// getName - Get the target name.
    const char *getName() const { return Name; }

    /// getShortDescription - Get a short description of the target.
    const char *getShortDescription() const { return ShortDesc; }

    bool hasJIT() const { return HasJIT; }

    /// hasTargetMachine - Check if this target supports code generation.
    bool hasTargetMachine() const { return TargetMachineCtorFn != 0; }

    /// hasAsmPrinter - Check if this target supports .s printing.
    bool hasAsmPrinter() const { return AsmPrinterCtorFn != 0; }

    /// hasAsmParser - Check if this target supports .s parsing.
    bool hasAsmParser() const { return AsmParserCtorFn != 0; }

    
    /// createAsmInfo - Create a TargetAsmInfo implementation for the specified
    /// target triple.
    ///
    /// \arg Triple - This argument is used to determine the target machine
    /// feature set; it should always be provided. Generally this should be
    /// either the target triple from the module, or the target triple of the
    /// host if that does not exist.
    const TargetAsmInfo *createAsmInfo(const StringRef &Triple) const {
      if (!AsmInfoCtorFn)
        return 0;
      return AsmInfoCtorFn(*this, Triple);
    }
    
    /// createTargetMachine - Create a target specific machine implementation
    /// for the specified \arg Triple.
    ///
    /// \arg Triple - This argument is used to determine the target machine
    /// feature set; it should always be provided. Generally this should be
    /// either the target triple from the module, or the target triple of the
    /// host if that does not exist.
    TargetMachine *createTargetMachine(const std::string &Triple,
                                       const std::string &Features) const {
      if (!TargetMachineCtorFn)
        return 0;
      return TargetMachineCtorFn(*this, Triple, Features);
    }

    /// createAsmPrinter - Create a target specific assembly printer pass.
    FunctionPass *createAsmPrinter(formatted_raw_ostream &OS,
                                   TargetMachine &M,
                                   bool Verbose) const {
      if (!AsmPrinterCtorFn)
        return 0;
      return AsmPrinterCtorFn(OS, M, Verbose);
    }

    /// createAsmParser - Create a target specific assembly parser.
    ///
    /// \arg Parser - The target independent parser implementation to use for
    /// parsing and lexing.
    TargetAsmParser *createAsmParser(MCAsmParser &Parser) const {
      if (!AsmParserCtorFn)
        return 0;
      return AsmParserCtorFn(*this, Parser);
    }
  };

  /// TargetRegistry - Generic interface to target specific features.
  struct TargetRegistry {
    class iterator {
      const Target *Current;
      explicit iterator(Target *T) : Current(T) {}
      friend struct TargetRegistry;
    public:
      iterator(const iterator &I) : Current(I.Current) {}
      iterator() : Current(0) {}

      bool operator==(const iterator &x) const {
        return Current == x.Current;
      }
      bool operator!=(const iterator &x) const {
        return !operator==(x);
      }

      // Iterator traversal: forward iteration only
      iterator &operator++() {          // Preincrement
        assert(Current && "Cannot increment end iterator!");
        Current = Current->getNext();
        return *this;
      }
      iterator operator++(int) {        // Postincrement
        iterator tmp = *this; 
        ++*this; 
        return tmp;
      }

      const Target &operator*() const {
        assert(Current && "Cannot dereference end iterator!");
        return *Current;
      }

      const Target *operator->() const {
        return &operator*();
      }
    };

    /// @name Registry Access
    /// @{

    static iterator begin();

    static iterator end() { return iterator(); }

    /// lookupTarget - Lookup a target based on a target triple.
    ///
    /// \param Triple - The triple to use for finding a target.
    /// \param Error - On failure, an error string describing why no target was
    /// found.
    static const Target *lookupTarget(const std::string &Triple,
                                      std::string &Error);

    /// getClosestTargetForJIT - Pick the best target that is compatible with
    /// the current host.  If no close target can be found, this returns null
    /// and sets the Error string to a reason.
    ///
    /// Mainted for compatibility through 2.6.
    static const Target *getClosestTargetForJIT(std::string &Error);

    /// @}
    /// @name Target Registration
    /// @{

    /// RegisterTarget - Register the given target. Attempts to register a
    /// target which has already been registered will be ignored.
    /// 
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Name - The target name. This should be a static string.
    /// @param ShortDesc - A short target description. This should be a static
    /// string. 
    /// @param TQualityFn - The triple match quality computation function for
    /// this target.
    /// @param HasJIT - Whether the target supports JIT code
    /// generation.
    static void RegisterTarget(Target &T,
                               const char *Name,
                               const char *ShortDesc,
                               Target::TripleMatchQualityFnTy TQualityFn,
                               bool HasJIT = false);

    /// RegisterAsmInfo - Register a TargetAsmInfo implementation for the
    /// given target.
    /// 
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    /// 
    /// @param T - The target being registered.
    /// @param Fn - A function to construct a TargetAsmInfo for the target.
    static void RegisterAsmInfo(Target &T, Target::AsmInfoCtorFnTy Fn) {
      // Ignore duplicate registration.
      if (!T.AsmInfoCtorFn)
        T.AsmInfoCtorFn = Fn;
    }
    
    /// RegisterTargetMachine - Register a TargetMachine implementation for the
    /// given target.
    /// 
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    /// 
    /// @param T - The target being registered.
    /// @param Fn - A function to construct a TargetMachine for the target.
    static void RegisterTargetMachine(Target &T, 
                                      Target::TargetMachineCtorTy Fn) {
      // Ignore duplicate registration.
      if (!T.TargetMachineCtorFn)
        T.TargetMachineCtorFn = Fn;
    }

    /// RegisterAsmPrinter - Register an AsmPrinter implementation for the given
    /// target.
    /// 
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Fn - A function to construct an AsmPrinter for the target.
    static void RegisterAsmPrinter(Target &T, Target::AsmPrinterCtorTy Fn) {
      // Ignore duplicate registration.
      if (!T.AsmPrinterCtorFn)
        T.AsmPrinterCtorFn = Fn;
    }

    /// RegisterAsmParser - Register a TargetAsmParser implementation for the
    /// given target.
    /// 
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Fn - A function to construct an AsmPrinter for the target.
    static void RegisterAsmParser(Target &T, Target::AsmParserCtorTy Fn) {
      if (!T.AsmParserCtorFn)
        T.AsmParserCtorFn = Fn;
    }

    /// @}
  };


  //===--------------------------------------------------------------------===//

  /// RegisterTarget - Helper template for registering a target, for use in the
  /// target's initialization function. Usage:
  ///
  ///
  /// Target TheFooTarget; // The global target instance.
  ///
  /// extern "C" void LLVMInitializeFooTargetInfo() {
  ///   RegisterTarget<Triple::foo> X(TheFooTarget, "foo", "Foo description");
  /// }
  template<Triple::ArchType TargetArchType = Triple::InvalidArch,
           bool HasJIT = false>
  struct RegisterTarget {
    RegisterTarget(Target &T, const char *Name, const char *Desc) {
      TargetRegistry::RegisterTarget(T, Name, Desc,
                                     &getTripleMatchQuality,
                                     HasJIT);
    }

    static unsigned getTripleMatchQuality(const std::string &TT) {
      if (Triple(TT.c_str()).getArch() == TargetArchType)
        return 20;
      return 0;
    }
  };

  /// RegisterAsmInfo - Helper template for registering a target assembly info
  /// implementation.  This invokes the static "Create" method on the class to
  /// actually do the construction.  Usage:
  ///
  /// extern "C" void LLVMInitializeFooTarget() {
  ///   extern Target TheFooTarget;
  ///   RegisterAsmInfo<FooTargetAsmInfo> X(TheFooTarget);
  /// }
  template<class TargetAsmInfoImpl>
  struct RegisterAsmInfo {
    RegisterAsmInfo(Target &T) {
      TargetRegistry::RegisterAsmInfo(T, &Allocator);
    }
  private:
    static const TargetAsmInfo *Allocator(const Target &T, const StringRef &TT){
      return new TargetAsmInfoImpl(T, TT);
    }
    
  };

  /// RegisterAsmInfoFn - Helper template for registering a target assembly info
  /// implementation.  This invokes the specified function to do the
  /// construction.  Usage:
  ///
  /// extern "C" void LLVMInitializeFooTarget() {
  ///   extern Target TheFooTarget;
  ///   RegisterAsmInfoFn X(TheFooTarget, TheFunction);
  /// }
  struct RegisterAsmInfoFn {
    RegisterAsmInfoFn(Target &T, Target::AsmInfoCtorFnTy Fn) {
      TargetRegistry::RegisterAsmInfo(T, Fn);
    }
  };


  /// RegisterTargetMachine - Helper template for registering a target machine
  /// implementation, for use in the target machine initialization
  /// function. Usage:
  ///
  /// extern "C" void LLVMInitializeFooTarget() {
  ///   extern Target TheFooTarget;
  ///   RegisterTargetMachine<FooTargetMachine> X(TheFooTarget);
  /// }
  template<class TargetMachineImpl>
  struct RegisterTargetMachine {
    RegisterTargetMachine(Target &T) {
      TargetRegistry::RegisterTargetMachine(T, &Allocator);
    }

  private:
    static TargetMachine *Allocator(const Target &T, const std::string &TT,
                                    const std::string &FS) {
      return new TargetMachineImpl(T, TT, FS);
    }
  };

  /// RegisterAsmPrinter - Helper template for registering a target specific
  /// assembly printer, for use in the target machine initialization
  /// function. Usage:
  ///
  /// extern "C" void LLVMInitializeFooAsmPrinter() {
  ///   extern Target TheFooTarget;
  ///   RegisterAsmPrinter<FooAsmPrinter> X(TheFooTarget);
  /// }
  template<class AsmPrinterImpl>
  struct RegisterAsmPrinter {
    RegisterAsmPrinter(Target &T) {
      TargetRegistry::RegisterAsmPrinter(T, &Allocator);
    }

  private:
    static FunctionPass *Allocator(formatted_raw_ostream &OS,
                                   TargetMachine &TM,
                                   bool Verbose) {
      return new AsmPrinterImpl(OS, TM, TM.getTargetAsmInfo(), Verbose);
    }
  };

  /// RegisterAsmParser - Helper template for registering a target specific
  /// assembly parser, for use in the target machine initialization
  /// function. Usage:
  ///
  /// extern "C" void LLVMInitializeFooAsmParser() {
  ///   extern Target TheFooTarget;
  ///   RegisterAsmParser<FooAsmParser> X(TheFooTarget);
  /// }
  template<class AsmParserImpl>
  struct RegisterAsmParser {
    RegisterAsmParser(Target &T) {
      TargetRegistry::RegisterAsmParser(T, &Allocator);
    }

  private:
    static TargetAsmParser *Allocator(const Target &T, MCAsmParser &P) {
      return new AsmParserImpl(T, P);
    }
  };

}

#endif
