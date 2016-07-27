//=== Registry.h - Linker-supported plugin registries -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines a registry template for discovering pluggable modules.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_REGISTRY_H
#define LLVM_SUPPORT_REGISTRY_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DynamicLibrary.h"
#include <memory>

namespace llvm {
  /// A simple registry entry which provides only a name, description, and
  /// no-argument constructor.
  template <typename T>
  class SimpleRegistryEntry {
    const char *Name, *Desc;
    std::unique_ptr<T> (*Ctor)();

  public:
    SimpleRegistryEntry(const char *N, const char *D, std::unique_ptr<T> (*C)())
      : Name(N), Desc(D), Ctor(C)
    {}

    const char *getName() const { return Name; }
    const char *getDesc() const { return Desc; }
    std::unique_ptr<T> instantiate() const { return Ctor(); }
  };

  /// A global registry used in conjunction with static constructors to make
  /// pluggable components (like targets or garbage collectors) "just work" when
  /// linked with an executable.
  template <typename T>
  class Registry {
  public:
    typedef SimpleRegistryEntry<T> entry;

    class node;
    class iterator;

  private:
    Registry() = delete;

    friend class node;
    static node *Head, *Tail;

  public:
    /// Node in linked list of entries.
    ///
    class node {
      friend class iterator;
      friend Registry<T>;

      node *Next;
      const entry& Val;

    public:
      node(const entry &V) : Next(nullptr), Val(V) {}
    };

    /// Add a node to the Registry: this is the interface between the plugin and
    /// the executable.
    ///
    /// This function is exported by the executable and called by the plugin to
    /// add a node to the executable's registry. Therefore it's not defined here
    /// to avoid it being instantiated in the plugin and is instead defined in
    /// the executable (see LLVM_INSTANTIATE_REGISTRY below).
    static void add_node(node *N);

    /// Iterators for registry entries.
    ///
    class iterator {
      const node *Cur;

    public:
      explicit iterator(const node *N) : Cur(N) {}

      bool operator==(const iterator &That) const { return Cur == That.Cur; }
      bool operator!=(const iterator &That) const { return Cur != That.Cur; }
      iterator &operator++() { Cur = Cur->Next; return *this; }
      const entry &operator*() const { return Cur->Val; }
      const entry *operator->() const { return &Cur->Val; }
    };

    static iterator begin() { return iterator(Head); }
    static iterator end()   { return iterator(nullptr); }

    static iterator_range<iterator> entries() {
      return make_range(begin(), end());
    }

    /// A static registration template. Use like such:
    ///
    ///   Registry<Collector>::Add<FancyGC>
    ///   X("fancy-gc", "Newfangled garbage collector.");
    ///
    /// Use of this template requires that:
    ///
    ///  1. The registered subclass has a default constructor.
    template <typename V>
    class Add {
      entry Entry;
      node Node;

      static std::unique_ptr<T> CtorFn() { return make_unique<V>(); }

    public:
      Add(const char *Name, const char *Desc)
          : Entry(Name, Desc, CtorFn), Node(Entry) {
        add_node(&Node);
      }
    };
  };
} // end namespace llvm

/// Instantiate a registry class.
///
/// This instantiates add_node and the Head and Tail pointers.
#define LLVM_INSTANTIATE_REGISTRY(REGISTRY_CLASS) \
  namespace llvm { \
  template<> void REGISTRY_CLASS::add_node(REGISTRY_CLASS::node *N) { \
    if (Tail) \
     Tail->Next = N; \
    else \
      Head = N; \
    Tail = N; \
  } \
  template<> typename REGISTRY_CLASS::node *REGISTRY_CLASS::Head = nullptr; \
  template<> typename REGISTRY_CLASS::node *REGISTRY_CLASS::Tail = nullptr; \
  }

#endif // LLVM_SUPPORT_REGISTRY_H
