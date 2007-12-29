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

#include "llvm/Support/CommandLine.h"

namespace llvm {
  /// A simple registry entry which provides only a name, description, and
  /// no-argument constructor.
  template <typename T>
  class SimpleRegistryEntry {
    const char *Name, *Desc;
    T *(*Ctor)();
    
  public:
    SimpleRegistryEntry(const char *N, const char *D, T *(*C)())
      : Name(N), Desc(D), Ctor(C)
    {}
    
    const char *getName() const { return Name; }
    const char *getDesc() const { return Desc; }
    T *instantiate() const { return Ctor(); }
  };
  
  
  /// Traits for registry entries. If using other than SimpleRegistryEntry, it
  /// is necessary to define an alternate traits class.
  template <typename T>
  class RegistryTraits {
    RegistryTraits(); // Do not implement.
    
  public:
    typedef SimpleRegistryEntry<T> entry;
    
    /// Accessors for .
    /// 
    static const char *nameof(const entry &Entry) { return Entry.getName(); }
    static const char *descof(const entry &Entry) { return Entry.getDesc(); }
  };
  
  
  /// A global registry used in conjunction with static constructors to make
  /// pluggable components (like targets or garbage collectors) "just work" when
  /// linked with an executable.
  template <typename T, typename U = RegistryTraits<T> >
  class Registry {
  public:
    typedef U traits;
    typedef typename U::entry entry;
    
    class node;
    class listener;
    class iterator;
  
  private:
    Registry(); // Do not implement.
    
    static void Announce(const entry &E) {
      for (listener *Cur = ListenerHead; Cur; Cur = Cur->Next)
        Cur->registered(E);
    }
    
    friend class node;
    static node *Head, *Tail;
    
    friend class listener;
    static listener *ListenerHead, *ListenerTail;
    
  public:
    class iterator;
    
    
    /// Node in linked list of entries.
    /// 
    class node {
      friend class iterator;
      
      node *Next;
      const entry& Val;
      
    public:
      node(const entry& V) : Next(0), Val(V) {
        if (Tail)
          Tail->Next = this;
        else
          Head = this;
        Tail = this;
        
        Announce(V);
      }
    };
    
    
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
    static iterator end()   { return iterator(0); }
    
    
    /// Abstract base class for registry listeners, which are informed when new
    /// entries are added to the registry. Simply subclass and instantiate:
    /// 
    ///   class CollectorPrinter : public Registry<Collector>::listener {
    ///   protected:
    ///     void registered(const Registry<Collector>::entry &e) {
    ///       cerr << "collector now available: " << e->getName() << "\n";
    ///     }
    ///   
    ///   public:
    ///     CollectorPrinter() { init(); }  // Print those already registered.
    ///   };
    /// 
    ///   CollectorPrinter Printer;
    /// 
    class listener {
      listener *Prev, *Next;
      
      friend void Registry::Announce(const entry &E);
      
    protected:
      /// Called when an entry is added to the registry.
      /// 
      virtual void registered(const entry &) = 0;
      
      /// Calls 'registered' for each pre-existing entry.
      /// 
      void init() {
        for (iterator I = begin(), E = end(); I != E; ++I)
          registered(*I);
      }
      
    public:
      listener() : Prev(ListenerTail), Next(0) {
        if (Prev)
          Prev->Next = this;
        else
          ListenerHead = this;
        ListenerTail = this;
      }
      
      virtual ~listener() {
        if (Next)
          Next->Prev = Prev;
        else
          ListenerTail = Prev;
        if (Prev)
          Prev->Next = Next;
        else
          ListenerHead = Next;
      }
    };
    
    
    /// A static registration template. Use like such:
    /// 
    ///   Registry<Collector>::Add<FancyGC>
    ///   X("fancy-gc", "Newfangled garbage collector.");
    /// 
    /// Use of this template requires that:
    /// 
    ///  1. The registered subclass has a default constructor.
    // 
    ///  2. The registry entry type has a constructor compatible with this
    ///     signature:
    /// 
    ///       entry(const char *Name, const char *ShortDesc, T *(*Ctor)());
    /// 
    /// If you have more elaborate requirements, then copy and modify.
    /// 
    template <typename V>
    class Add {
      entry Entry;
      node Node;
      
      static T *CtorFn() { return new V(); }
      
    public:
      Add(const char *Name, const char *Desc)
        : Entry(Name, Desc, CtorFn), Node(Entry) {}
    };
    
    
    /// A command-line parser for a registry. Use like such:
    /// 
    ///   static cl::opt<Registry<Collector>::entry, false,
    ///                  Registry<Collector>::Parser>
    ///   GCOpt("gc", cl::desc("Garbage collector to use."),
    ///               cl::value_desc());
    ///   
    /// To make use of the value:
    /// 
    ///   Collector *TheCollector = GCOpt->instantiate();
    /// 
    class Parser : public cl::parser<const typename U::entry*>, public listener{
      typedef U traits;
      typedef typename U::entry entry;
      
    protected:
      void registered(const entry &E) {
        addLiteralOption(traits::nameof(E), &E, traits::descof(E));
      }
      
    public:
      void initialize(cl::Option &O) {
        listener::init();
        cl::parser<const typename U::entry*>::initialize(O);
      }
    };
    
  };
  
}

#endif
