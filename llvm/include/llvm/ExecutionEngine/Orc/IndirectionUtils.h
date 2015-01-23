//===-- IndirectionUtils.h - Utilities for adding indirections --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains utilities for adding indirections and breaking up modules.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_INDIRECTIONUTILS_H
#define LLVM_EXECUTIONENGINE_ORC_INDIRECTIONUTILS_H

#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include <sstream>

namespace llvm {

/// @brief Persistent name mangling.
///
///   This class provides name mangling that can outlive a Module (and its
/// DataLayout).
class PersistentMangler {
public:
  PersistentMangler(DataLayout DL) : DL(std::move(DL)), M(&this->DL) {}

  std::string getMangledName(StringRef Name) const {
    std::string MangledName;
    {
      raw_string_ostream MangledNameStream(MangledName);
      M.getNameWithPrefix(MangledNameStream, Name);
    }
    return MangledName;
  }

private:
  DataLayout DL;
  Mangler M;
};

/// @brief Handle callbacks from the JIT process requesting the definitions of
///        symbols.
///
///   This utility is intended to be used to support compile-on-demand for
/// functions.
class JITResolveCallbackHandler {
private:
  typedef std::vector<std::string> FuncNameList;

public:
  typedef FuncNameList::size_type StubIndex;

public:
  /// @brief Create a JITResolveCallbackHandler with the given functors for
  ///        looking up symbols and updating their use-sites.
  ///
  /// @return A JITResolveCallbackHandler instance that will invoke the
  ///         Lookup and Update functors as needed to resolve missing symbol
  ///         definitions.
  template <typename LookupFtor, typename UpdateFtor>
  static std::unique_ptr<JITResolveCallbackHandler> create(LookupFtor Lookup,
                                                           UpdateFtor Update);

  /// @brief Destroy instance. Does not modify existing emitted symbols.
  ///
  ///   Not-yet-emitted symbols will need to be resolved some other way after
  /// this class is destroyed.
  virtual ~JITResolveCallbackHandler() {}

  /// @brief Add a function to be resolved on demand.
  void addFuncName(std::string Name) { FuncNames.push_back(std::move(Name)); }

  /// @brief Get the name associated with the given index.
  const std::string &getFuncName(StubIndex Idx) const { return FuncNames[Idx]; }

  /// @brief Returns the number of symbols being managed by this instance.
  StubIndex getNumFuncs() const { return FuncNames.size(); }

  /// @brief Get the address for the symbol associated with the given index.
  ///
  ///   This is expected to be called by code in the JIT process itself, in
  /// order to resolve a function.
  virtual uint64_t resolve(StubIndex StubIdx) = 0;

private:
  FuncNameList FuncNames;
};

// Implementation class for JITResolveCallbackHandler.
template <typename LookupFtor, typename UpdateFtor>
class JITResolveCallbackHandlerImpl : public JITResolveCallbackHandler {
public:
  JITResolveCallbackHandlerImpl(LookupFtor Lookup, UpdateFtor Update)
      : Lookup(std::move(Lookup)), Update(std::move(Update)) {}

  uint64_t resolve(StubIndex StubIdx) override {
    const std::string &FuncName = getFuncName(StubIdx);
    uint64_t Addr = Lookup(FuncName);
    Update(FuncName, Addr);
    return Addr;
  }

private:
  LookupFtor Lookup;
  UpdateFtor Update;
};

template <typename LookupFtor, typename UpdateFtor>
std::unique_ptr<JITResolveCallbackHandler>
JITResolveCallbackHandler::create(LookupFtor Lookup, UpdateFtor Update) {
  typedef JITResolveCallbackHandlerImpl<LookupFtor, UpdateFtor> Impl;
  return make_unique<Impl>(std::move(Lookup), std::move(Update));
}

/// @brief Holds a list of the function names that were indirected, plus
///        mappings from each of these names to (a) the name of function
///        providing the implementation for that name (GetImplNames), and
///        (b) the name of the global variable holding the address of the
///        implementation.
///
///   This data structure can be used with a JITCallbackHandler to look up and
/// update function implementations when lazily compiling.
class JITIndirections {
public:
  JITIndirections(std::vector<std::string> IndirectedNames,
                  std::function<std::string(StringRef)> GetImplName,
                  std::function<std::string(StringRef)> GetAddrName)
      : IndirectedNames(std::move(IndirectedNames)),
        GetImplName(std::move(GetImplName)),
        GetAddrName(std::move(GetAddrName)) {}

  std::vector<std::string> IndirectedNames;
  std::function<std::string(StringRef Name)> GetImplName;
  std::function<std::string(StringRef Name)> GetAddrName;
};

/// @brief Indirect all calls to functions matching the predicate
///        ShouldIndirect through a global variable containing the address
///        of the implementation.
///
/// @return An indirection structure containing the functions that had their
///         call-sites re-written.
///
///   For each function 'F' that meets the ShouldIndirect predicate, and that
/// is called in this Module, add a common-linkage global variable to the
/// module that will hold the address of the implementation of that function.
/// Rewrite all call-sites of 'F' to be indirect calls (via the global).
/// This allows clients, either directly or via a JITCallbackHandler, to
/// change the address of the implementation of 'F' at runtime.
///
/// Important notes:
///
///   Single indirection does not preserve pointer equality for 'F'. If the
/// program was already calling 'F' indirectly through function pointers, or
/// if it was taking the address of 'F' for the purpose of pointer comparisons
/// or arithmetic double indirection should be used instead.
///
///   This method does *not* initialize the function implementation addresses.
/// The client must do this prior to running any call-sites that have been
/// indirected.
JITIndirections makeCallsSingleIndirect(
    llvm::Module &M,
    const std::function<bool(const Function &)> &ShouldIndirect,
    const char *JITImplSuffix, const char *JITAddrSuffix);

/// @brief Replace the body of functions matching the predicate ShouldIndirect
///        with indirect calls to the implementation.
///
/// @return An indirections structure containing the functions that had their
///         implementations re-written.
///
///   For each function 'F' that meets the ShouldIndirect predicate, add a
/// common-linkage global variable to the module that will hold the address of
/// the implementation of that function and rewrite the implementation of 'F'
/// to call through to the implementation indirectly (via the global).
/// This allows clients, either directly or via a JITCallbackHandler, to
/// change the address of the implementation of 'F' at runtime.
///
/// Important notes:
///
///   Double indirection is slower than single indirection, but preserves
/// function pointer relation tests and correct behavior for function pointers
/// (all calls to 'F', direct or indirect) go the address stored in the global
/// variable at the time of the call.
///
///   This method does *not* initialize the function implementation addresses.
/// The client must do this prior to running any call-sites that have been
/// indirected.
JITIndirections makeCallsDoubleIndirect(
    llvm::Module &M,
    const std::function<bool(const Function &)> &ShouldIndirect,
    const char *JITImplSuffix, const char *JITAddrSuffix);

/// @brief Given a set of indirections and a symbol lookup functor, create a
///        JITResolveCallbackHandler instance that will resolve the
///        implementations for the indirected symbols on demand.
template <typename SymbolLookupFtor>
std::unique_ptr<JITResolveCallbackHandler>
createCallbackHandlerFromJITIndirections(const JITIndirections &Indirs,
                                         const PersistentMangler &NM,
                                         SymbolLookupFtor Lookup) {
  auto GetImplName = Indirs.GetImplName;
  auto GetAddrName = Indirs.GetAddrName;

  std::unique_ptr<JITResolveCallbackHandler> J =
      JITResolveCallbackHandler::create(
          [=](const std::string &S) {
            return Lookup(NM.getMangledName(GetImplName(S)));
          },
          [=](const std::string &S, uint64_t Addr) {
            void *ImplPtr = reinterpret_cast<void *>(
                Lookup(NM.getMangledName(GetAddrName(S))));
            memcpy(ImplPtr, &Addr, sizeof(uint64_t));
          });

  for (const auto &FuncName : Indirs.IndirectedNames)
    J->addFuncName(FuncName);

  return J;
}

/// @brief Insert callback asm into module M for the symbols managed by
///        JITResolveCallbackHandler J.
void insertX86CallbackAsm(Module &M, JITResolveCallbackHandler &J);

/// @brief Initialize global indirects to point into the callback asm.
template <typename LookupFtor>
void initializeFuncAddrs(JITResolveCallbackHandler &J,
                         const JITIndirections &Indirs,
                         const PersistentMangler &NM, LookupFtor Lookup) {
  // Forward declare so that we can access this, even though it's an
  // implementation detail.
  std::string getJITResolveCallbackIndexLabel(unsigned I);

  if (J.getNumFuncs() == 0)
    return;

  //   Force a look up one of the global addresses for a function that has been
  // indirected. We need to do this to trigger the emission of the module
  // holding the callback asm. We can't rely on that emission happening
  // automatically when we look up the callback asm symbols, since lazy-emitting
  // layers can't see those.
  Lookup(NM.getMangledName(Indirs.GetAddrName(J.getFuncName(0))));

  // Now update indirects to point to the JIT resolve callback asm.
  for (JITResolveCallbackHandler::StubIndex I = 0; I < J.getNumFuncs(); ++I) {
    uint64_t ResolveCallbackIdxAddr =
        Lookup(getJITResolveCallbackIndexLabel(I));
    void *AddrPtr = reinterpret_cast<void *>(
        Lookup(NM.getMangledName(Indirs.GetAddrName(J.getFuncName(I)))));
    assert(AddrPtr && "Can't find stub addr global to initialize.");
    memcpy(AddrPtr, &ResolveCallbackIdxAddr, sizeof(uint64_t));
  }
}

/// @brief Extract all functions matching the predicate ShouldExtract in to
///        their own modules. (Does not modify the original module.)
///
/// @return A set of modules, the first containing all symbols (including
///         globals and aliases) that did not pass ShouldExtract, and each
///         subsequent module containing one of the functions that did meet
///         ShouldExtract.
///
///   By adding the resulting modules separately (not as a set) to a
/// LazyEmittingLayer instance, compilation can be deferred until symbols are
/// actually needed.
std::vector<std::unique_ptr<llvm::Module>>
explode(const llvm::Module &OrigMod,
        const std::function<bool(const Function &)> &ShouldExtract);

/// @brief Given a module that has been indirectified, break each function
///        that has been indirected out into its own module. (Does not modify
///        the original module).
///
/// @returns A set of modules covering the symbols provided by OrigMod.
std::vector<std::unique_ptr<llvm::Module>>
explode(const llvm::Module &OrigMod, const JITIndirections &Indirections);
}

#endif // LLVM_EXECUTIONENGINE_ORC_INDIRECTIONUTILS_H
