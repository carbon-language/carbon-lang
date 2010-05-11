//===--- Ownership.h - Parser Ownership Helpers -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file contains classes for managing ownership of Stmt and Expr nodes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_OWNERSHIP_H
#define LLVM_CLANG_PARSE_OWNERSHIP_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/PointerIntPair.h"

//===----------------------------------------------------------------------===//
// OpaquePtr
//===----------------------------------------------------------------------===//

namespace clang {
  class ActionBase;

  /// OpaquePtr - This is a very simple POD type that wraps a pointer that the
  /// Parser doesn't know about but that Sema or another client does.  The UID
  /// template argument is used to make sure that "Decl" pointers are not
  /// compatible with "Type" pointers for example.
  template<int UID>
  class OpaquePtr {
    void *Ptr;
  public:
    OpaquePtr() : Ptr(0) {}

    template <typename T>
    T* getAs() const {
      return llvm::PointerLikeTypeTraits<T*>::getFromVoidPointer(Ptr);
    }

    template <typename T>
    T getAsVal() const {
      return llvm::PointerLikeTypeTraits<T>::getFromVoidPointer(Ptr);
    }

    void *get() const { return Ptr; }

    template<typename T>
    static OpaquePtr make(T P) {
      OpaquePtr R; R.set(P); return R;
    }

    template<typename T>
    void set(T P) {
      Ptr = llvm::PointerLikeTypeTraits<T>::getAsVoidPointer(P);
    }

    operator bool() const { return Ptr != 0; }
  };
}

namespace llvm {
  template <int UID>
  class PointerLikeTypeTraits<clang::OpaquePtr<UID> > {
  public:
    static inline void *getAsVoidPointer(clang::OpaquePtr<UID> P) {
      // FIXME: Doesn't work? return P.getAs< void >();
      return P.get();
    }
    static inline clang::OpaquePtr<UID> getFromVoidPointer(void *P) {
      return clang::OpaquePtr<UID>::make(P);
    }
    enum { NumLowBitsAvailable = 3 };
  };
}



// -------------------------- About Move Emulation -------------------------- //
// The smart pointer classes in this file attempt to emulate move semantics
// as they appear in C++0x with rvalue references. Since C++03 doesn't have
// rvalue references, some tricks are needed to get similar results.
// Move semantics in C++0x have the following properties:
// 1) "Moving" means transferring the value of an object to another object,
//    similar to copying, but without caring what happens to the old object.
//    In particular, this means that the new object can steal the old object's
//    resources instead of creating a copy.
// 2) Since moving can modify the source object, it must either be explicitly
//    requested by the user, or the modifications must be unnoticeable.
// 3) As such, C++0x moving is only allowed in three contexts:
//    * By explicitly using std::move() to request it.
//    * From a temporary object, since that object cannot be accessed
//      afterwards anyway, thus making the state unobservable.
//    * On function return, since the object is not observable afterwards.
//
// To sum up: moving from a named object should only be possible with an
// explicit std::move(), or on function return. Moving from a temporary should
// be implicitly done. Moving from a const object is forbidden.
//
// The emulation is not perfect, and has the following shortcomings:
// * move() is not in namespace std.
// * move() is required on function return.
// * There are difficulties with implicit conversions.
// * Microsoft's compiler must be given the /Za switch to successfully compile.
//
// -------------------------- Implementation -------------------------------- //
// The move emulation relies on the peculiar reference binding semantics of
// C++03: as a rule, a non-const reference may not bind to a temporary object,
// except for the implicit object parameter in a member function call, which
// can refer to a temporary even when not being const.
// The moveable object has five important functions to facilitate moving:
// * A private, unimplemented constructor taking a non-const reference to its
//   own class. This constructor serves a two-fold purpose.
//   - It prevents the creation of a copy constructor that takes a const
//     reference. Temporaries would be able to bind to the argument of such a
//     constructor, and that would be bad.
//   - Named objects will bind to the non-const reference, but since it's
//     private, this will fail to compile. This prevents implicit moving from
//     named objects.
//   There's also a copy assignment operator for the same purpose.
// * An implicit, non-const conversion operator to a special mover type. This
//   type represents the rvalue reference of C++0x. Being a non-const member,
//   its implicit this parameter can bind to temporaries.
// * A constructor that takes an object of this mover type. This constructor
//   performs the actual move operation. There is an equivalent assignment
//   operator.
// There is also a free move() function that takes a non-const reference to
// an object and returns a temporary. Internally, this function uses explicit
// constructor calls to move the value from the referenced object to the return
// value.
//
// There are now three possible scenarios of use.
// * Copying from a const object. Constructor overload resolution will find the
//   non-const copy constructor, and the move constructor. The first is not
//   viable because the const object cannot be bound to the non-const reference.
//   The second fails because the conversion to the mover object is non-const.
//   Moving from a const object fails as intended.
// * Copying from a named object. Constructor overload resolution will select
//   the non-const copy constructor, but fail as intended, because this
//   constructor is private.
// * Copying from a temporary. Constructor overload resolution cannot select
//   the non-const copy constructor, because the temporary cannot be bound to
//   the non-const reference. It thus selects the move constructor. The
//   temporary can be bound to the implicit this parameter of the conversion
//   operator, because of the special binding rule. Construction succeeds.
//   Note that the Microsoft compiler, as an extension, allows binding
//   temporaries against non-const references. The compiler thus selects the
//   non-const copy constructor and fails, because the constructor is private.
//   Passing /Za (disable extensions) disables this behaviour.
// The free move() function is used to move from a named object.
//
// Note that when passing an object of a different type (the classes below
// have OwningResult and OwningPtr, which should be mixable), you get a problem.
// Argument passing and function return use copy initialization rules. The
// effect of this is that, when the source object is not already of the target
// type, the compiler will first seek a way to convert the source object to the
// target type, and only then attempt to copy the resulting object. This means
// that when passing an OwningResult where an OwningPtr is expected, the
// compiler will first seek a conversion from OwningResult to OwningPtr, then
// copy the OwningPtr. The resulting conversion sequence is:
// OwningResult object -> ResultMover -> OwningResult argument to
// OwningPtr(OwningResult) -> OwningPtr -> PtrMover -> final OwningPtr
// This conversion sequence is too complex to be allowed. Thus the special
// move_* functions, which help the compiler out with some explicit
// conversions.

// Flip this switch to measure performance impact of the smart pointers.
// #define DISABLE_SMART_POINTERS

namespace llvm {
  template<>
  class PointerLikeTypeTraits<clang::ActionBase*> {
    typedef clang::ActionBase* PT;
  public:
    static inline void *getAsVoidPointer(PT P) { return P; }
    static inline PT getFromVoidPointer(void *P) {
      return static_cast<PT>(P);
    }
    enum { NumLowBitsAvailable = 2 };
  };
}

namespace clang {
  // Basic
  class DiagnosticBuilder;

  // Determines whether the low bit of the result pointer for the
  // given UID is always zero. If so, ActionResult will use that bit
  // for it's "invalid" flag.
  template<unsigned UID>
  struct IsResultPtrLowBitFree {
    static const bool value = false;
  };

  /// ActionBase - A small part split from Action because of the horrible
  /// definition order dependencies between Action and the smart pointers.
  class ActionBase {
  public:
    /// Out-of-line virtual destructor to provide home for this class.
    virtual ~ActionBase();

    // Types - Though these don't actually enforce strong typing, they document
    // what types are required to be identical for the actions.
    typedef OpaquePtr<0> DeclPtrTy;
    typedef OpaquePtr<1> DeclGroupPtrTy;
    typedef OpaquePtr<2> TemplateTy;
    typedef void AttrTy;
    typedef void BaseTy;
    typedef void MemInitTy;
    typedef void ExprTy;
    typedef void StmtTy;
    typedef void TemplateParamsTy;
    typedef void CXXScopeTy;
    typedef void TypeTy;  // FIXME: Change TypeTy to use OpaquePtr<N>.

    /// ActionResult - This structure is used while parsing/acting on
    /// expressions, stmts, etc.  It encapsulates both the object returned by
    /// the action, plus a sense of whether or not it is valid.
    /// When CompressInvalid is true, the "invalid" flag will be
    /// stored in the low bit of the Val pointer.
    template<unsigned UID,
             typename PtrTy = void*,
             bool CompressInvalid = IsResultPtrLowBitFree<UID>::value>
    class ActionResult {
      PtrTy Val;
      bool Invalid;

    public:
      ActionResult(bool Invalid = false) : Val(PtrTy()), Invalid(Invalid) {}
      template<typename ActualExprTy>
      ActionResult(ActualExprTy val) : Val(val), Invalid(false) {}
      ActionResult(const DiagnosticBuilder &) : Val(PtrTy()), Invalid(true) {}

      PtrTy get() const { return Val; }
      void set(PtrTy V) { Val = V; }
      bool isInvalid() const { return Invalid; }

      const ActionResult &operator=(PtrTy RHS) {
        Val = RHS;
        Invalid = false;
        return *this;
      }
    };

    // This ActionResult partial specialization places the "invalid"
    // flag into the low bit of the pointer.
    template<unsigned UID, typename PtrTy>
    class ActionResult<UID, PtrTy, true> {
      // A pointer whose low bit is 1 if this result is invalid, 0
      // otherwise.
      uintptr_t PtrWithInvalid;
      typedef llvm::PointerLikeTypeTraits<PtrTy> PtrTraits;
    public:
      ActionResult(bool Invalid = false)
        : PtrWithInvalid(static_cast<uintptr_t>(Invalid)) { }

      template<typename ActualExprTy>
      ActionResult(ActualExprTy *val) {
        PtrTy V(val);
        void *VP = PtrTraits::getAsVoidPointer(V);
        PtrWithInvalid = reinterpret_cast<uintptr_t>(VP);
        assert((PtrWithInvalid & 0x01) == 0 && "Badly aligned pointer");
      }

      ActionResult(PtrTy V) {
        void *VP = PtrTraits::getAsVoidPointer(V);
        PtrWithInvalid = reinterpret_cast<uintptr_t>(VP);
        assert((PtrWithInvalid & 0x01) == 0 && "Badly aligned pointer");
      }

      ActionResult(const DiagnosticBuilder &) : PtrWithInvalid(0x01) { }

      PtrTy get() const {
        void *VP = reinterpret_cast<void *>(PtrWithInvalid & ~0x01);
        return PtrTraits::getFromVoidPointer(VP);
      }

      void set(PtrTy V) {
        void *VP = PtrTraits::getAsVoidPointer(V);
        PtrWithInvalid = reinterpret_cast<uintptr_t>(VP);
        assert((PtrWithInvalid & 0x01) == 0 && "Badly aligned pointer");
      }

      bool isInvalid() const { return PtrWithInvalid & 0x01; }

      const ActionResult &operator=(PtrTy RHS) {
        void *VP = PtrTraits::getAsVoidPointer(RHS);
        PtrWithInvalid = reinterpret_cast<uintptr_t>(VP);
        assert((PtrWithInvalid & 0x01) == 0 && "Badly aligned pointer");
        return *this;
      }
    };

    /// Deletion callbacks - Since the parser doesn't know the concrete types of
    /// the AST nodes being generated, it must do callbacks to delete objects
    /// when recovering from errors. These are in ActionBase because the smart
    /// pointers need access to them.
    virtual void DeleteExpr(ExprTy *E) {}
    virtual void DeleteStmt(StmtTy *S) {}
    virtual void DeleteTemplateParams(TemplateParamsTy *P) {}
  };

  /// ASTDestroyer - The type of an AST node destruction function pointer.
  typedef void (ActionBase::*ASTDestroyer)(void *);

  /// For the transition phase: translate from an ASTDestroyer to its
  /// ActionResult UID.
  template <ASTDestroyer Destroyer> struct DestroyerToUID;
  template <> struct DestroyerToUID<&ActionBase::DeleteExpr> {
    static const unsigned UID = 0;
  };
  template <> struct DestroyerToUID<&ActionBase::DeleteStmt> {
    static const unsigned UID = 1;
  };
  /// ASTOwningResult - A moveable smart pointer for AST nodes that also
  /// has an extra flag to indicate an additional success status.
  template <ASTDestroyer Destroyer> class ASTOwningResult;

  /// ASTMultiPtr - A moveable smart pointer to multiple AST nodes. Only owns
  /// the individual pointers, not the array holding them.
  template <ASTDestroyer Destroyer> class ASTMultiPtr;

#if !defined(DISABLE_SMART_POINTERS)
  namespace moving {
    /// Move emulation helper for ASTOwningResult. NEVER EVER use this class
    /// directly if you don't know what you're doing.
    template <ASTDestroyer Destroyer>
    class ASTResultMover {
      ASTOwningResult<Destroyer> &Moved;

    public:
      ASTResultMover(ASTOwningResult<Destroyer> &moved) : Moved(moved) {}

      ASTOwningResult<Destroyer> * operator ->() { return &Moved; }
    };

    /// Move emulation helper for ASTMultiPtr. NEVER EVER use this class
    /// directly if you don't know what you're doing.
    template <ASTDestroyer Destroyer>
    class ASTMultiMover {
      ASTMultiPtr<Destroyer> &Moved;

    public:
      ASTMultiMover(ASTMultiPtr<Destroyer> &moved) : Moved(moved) {}

      ASTMultiPtr<Destroyer> * operator ->() { return &Moved; }

      /// Reset the moved object's internal structures.
      void release();
    };
  }
#else

  /// Kept only as a type-safe wrapper for a void pointer, when smart pointers
  /// are disabled. When they are enabled, ASTOwningResult takes over.
  template <ASTDestroyer Destroyer>
  class ASTOwningPtr {
    void *Node;

  public:
    explicit ASTOwningPtr(ActionBase &) : Node(0) {}
    ASTOwningPtr(ActionBase &, void *node) : Node(node) {}
    // Normal copying operators are defined implicitly.
    ASTOwningPtr(const ASTOwningResult<Destroyer> &o);

    ASTOwningPtr & operator =(void *raw) {
      Node = raw;
      return *this;
    }

    /// Access to the raw pointer.
    void * get() const { return Node; }

    /// Release the raw pointer.
    void * take() {
      return Node;
    }

    /// Take outside ownership of the raw pointer and cast it down.
    template<typename T>
    T *takeAs() {
      return static_cast<T*>(Node);
    }

    /// Alias for interface familiarity with unique_ptr.
    void * release() {
      return take();
    }
  };
#endif

  // Important: There are two different implementations of
  // ASTOwningResult below, depending on whether
  // DISABLE_SMART_POINTERS is defined. If you make changes that
  // affect the interface, be sure to compile and test both ways!

#if !defined(DISABLE_SMART_POINTERS)
  template <ASTDestroyer Destroyer>
  class ASTOwningResult {
    llvm::PointerIntPair<ActionBase*, 1, bool> ActionInv;
    void *Ptr;

    friend class moving::ASTResultMover<Destroyer>;

#if !(defined(_MSC_VER) && _MSC_VER >= 1600)
    ASTOwningResult(ASTOwningResult&); // DO NOT IMPLEMENT
    ASTOwningResult& operator =(ASTOwningResult&); // DO NOT IMPLEMENT
#endif

    void destroy() {
      if (Ptr) {
        assert(ActionInv.getPointer() &&
               "Smart pointer has node but no action.");
        (ActionInv.getPointer()->*Destroyer)(Ptr);
        Ptr = 0;
      }
    }

  public:
    typedef ActionBase::ActionResult<DestroyerToUID<Destroyer>::UID> DumbResult;

    explicit ASTOwningResult(ActionBase &actions, bool invalid = false)
      : ActionInv(&actions, invalid), Ptr(0) {}
    ASTOwningResult(ActionBase &actions, void *node)
      : ActionInv(&actions, false), Ptr(node) {}
    ASTOwningResult(ActionBase &actions, const DumbResult &res)
      : ActionInv(&actions, res.isInvalid()), Ptr(res.get()) {}
    /// Move from another owning result
    ASTOwningResult(moving::ASTResultMover<Destroyer> mover)
      : ActionInv(mover->ActionInv),
        Ptr(mover->Ptr) {
      mover->Ptr = 0;
    }

    ~ASTOwningResult() {
      destroy();
    }

    /// Move assignment from another owning result
    ASTOwningResult &operator=(moving::ASTResultMover<Destroyer> mover) {
      destroy();
      ActionInv = mover->ActionInv;
      Ptr = mover->Ptr;
      mover->Ptr = 0;
      return *this;
    }

#if defined(_MSC_VER) && _MSC_VER >= 1600
    // Emulated move semantics don't work with msvc.
    ASTOwningResult(ASTOwningResult &&mover)
      : ActionInv(mover.ActionInv),
        Ptr(mover.Ptr) {
      mover.Ptr = 0;
    }
    ASTOwningResult &operator=(ASTOwningResult &&mover) {
      *this = moving::ASTResultMover<Destroyer>(mover);
      return *this;
    }
#endif

    /// Assignment from a raw pointer. Takes ownership - beware!
    ASTOwningResult &operator=(void *raw) {
      destroy();
      Ptr = raw;
      ActionInv.setInt(false);
      return *this;
    }

    /// Assignment from an ActionResult. Takes ownership - beware!
    ASTOwningResult &operator=(const DumbResult &res) {
      destroy();
      Ptr = res.get();
      ActionInv.setInt(res.isInvalid());
      return *this;
    }

    /// Access to the raw pointer.
    void *get() const { return Ptr; }

    bool isInvalid() const { return ActionInv.getInt(); }

    /// Does this point to a usable AST node? To be usable, the node must be
    /// valid and non-null.
    bool isUsable() const { return !isInvalid() && get(); }

    /// Take outside ownership of the raw pointer.
    void *take() {
      if (isInvalid())
        return 0;
      void *tmp = Ptr;
      Ptr = 0;
      return tmp;
    }

    /// Take outside ownership of the raw pointer and cast it down.
    template<typename T>
    T *takeAs() {
      return static_cast<T*>(take());
    }

    /// Alias for interface familiarity with unique_ptr.
    void *release() { return take(); }

    /// Pass ownership to a classical ActionResult.
    DumbResult result() {
      if (isInvalid())
        return true;
      return take();
    }

    /// Move hook
    operator moving::ASTResultMover<Destroyer>() {
      return moving::ASTResultMover<Destroyer>(*this);
    }
  };
#else
  template <ASTDestroyer Destroyer>
  class ASTOwningResult {
  public:
    typedef ActionBase::ActionResult<DestroyerToUID<Destroyer>::UID> DumbResult;

  private:
    DumbResult Result;

  public:
    explicit ASTOwningResult(ActionBase &actions, bool invalid = false)
      : Result(invalid) { }
    ASTOwningResult(ActionBase &actions, void *node) : Result(node) { }
    ASTOwningResult(ActionBase &actions, const DumbResult &res) : Result(res) { }
    // Normal copying semantics are defined implicitly.
    ASTOwningResult(const ASTOwningPtr<Destroyer> &o) : Result(o.get()) { }

    /// Assignment from a raw pointer. Takes ownership - beware!
    ASTOwningResult & operator =(void *raw) {
      Result = raw;
      return *this;
    }

    /// Assignment from an ActionResult. Takes ownership - beware!
    ASTOwningResult & operator =(const DumbResult &res) {
      Result = res;
      return *this;
    }

    /// Access to the raw pointer.
    void * get() const { return Result.get(); }

    bool isInvalid() const { return Result.isInvalid(); }

    /// Does this point to a usable AST node? To be usable, the node must be
    /// valid and non-null.
    bool isUsable() const { return !Result.isInvalid() && get(); }

    /// Take outside ownership of the raw pointer.
    void * take() {
      return Result.get();
    }

    /// Take outside ownership of the raw pointer and cast it down.
    template<typename T>
    T *takeAs() {
      return static_cast<T*>(take());
    }

    /// Alias for interface familiarity with unique_ptr.
    void * release() { return take(); }

    /// Pass ownership to a classical ActionResult.
    DumbResult result() { return Result; }
  };
#endif

  template <ASTDestroyer Destroyer>
  class ASTMultiPtr {
#if !defined(DISABLE_SMART_POINTERS)
    ActionBase &Actions;
#endif
    void **Nodes;
    unsigned Count;

#if !defined(DISABLE_SMART_POINTERS)
    friend class moving::ASTMultiMover<Destroyer>;

#if defined(_MSC_VER)
    //  Last tested with Visual Studio 2008.
    //  Visual C++ appears to have a bug where it does not recognise
    //  the return value from ASTMultiMover<Destroyer>::opeator-> as
    //  being a pointer to ASTMultiPtr.  However, the diagnostics
    //  suggest it has the right name, simply that the pointer type
    //  is not convertible to itself.
    //  Either way, a classic C-style hard cast resolves any issue.
     static ASTMultiPtr* hack(moving::ASTMultiMover<Destroyer> & source) {
       return (ASTMultiPtr*)source.operator->();
     }
#endif

    ASTMultiPtr(ASTMultiPtr&); // DO NOT IMPLEMENT
    // Reference member prevents copy assignment.

    void destroy() {
      assert((Count == 0 || Nodes) && "No nodes when count is not zero.");
      for (unsigned i = 0; i < Count; ++i) {
        if (Nodes[i])
          (Actions.*Destroyer)(Nodes[i]);
      }
    }
#endif

  public:
#if !defined(DISABLE_SMART_POINTERS)
    explicit ASTMultiPtr(ActionBase &actions)
      : Actions(actions), Nodes(0), Count(0) {}
    ASTMultiPtr(ActionBase &actions, void **nodes, unsigned count)
      : Actions(actions), Nodes(nodes), Count(count) {}
    /// Move constructor
    ASTMultiPtr(moving::ASTMultiMover<Destroyer> mover)
#if defined(_MSC_VER)
    //  Apply the visual C++ hack supplied above.
    //  Last tested with Visual Studio 2008.
      : Actions(hack(mover)->Actions), Nodes(hack(mover)->Nodes), Count(hack(mover)->Count) {
#else
      : Actions(mover->Actions), Nodes(mover->Nodes), Count(mover->Count) {
#endif
      mover.release();
    }
#else
    // Normal copying implicitly defined
    explicit ASTMultiPtr(ActionBase &) : Nodes(0), Count(0) {}
    ASTMultiPtr(ActionBase &, void **nodes, unsigned count)
      : Nodes(nodes), Count(count) {}
    // Fake mover in Parse/AstGuard.h needs this:
    ASTMultiPtr(void **nodes, unsigned count) : Nodes(nodes), Count(count) {}
#endif

#if !defined(DISABLE_SMART_POINTERS)
    /// Move assignment
    ASTMultiPtr & operator =(moving::ASTMultiMover<Destroyer> mover) {
      destroy();
      Nodes = mover->Nodes;
      Count = mover->Count;
      mover.release();
      return *this;
    }
#endif

    /// Access to the raw pointers.
    void ** get() const { return Nodes; }

    /// Access to the count.
    unsigned size() const { return Count; }

    void ** release() {
#if !defined(DISABLE_SMART_POINTERS)
      void **tmp = Nodes;
      Nodes = 0;
      Count = 0;
      return tmp;
#else
      return Nodes;
#endif
    }

#if !defined(DISABLE_SMART_POINTERS)
    /// Move hook
    operator moving::ASTMultiMover<Destroyer>() {
      return moving::ASTMultiMover<Destroyer>(*this);
    }
#endif
  };

  class ParsedTemplateArgument;
    
  class ASTTemplateArgsPtr {
#if !defined(DISABLE_SMART_POINTERS)
    ActionBase &Actions;
#endif
    ParsedTemplateArgument *Args;
    mutable unsigned Count;

#if !defined(DISABLE_SMART_POINTERS)
    void destroy();
#endif
    
  public:
    ASTTemplateArgsPtr(ActionBase &actions, ParsedTemplateArgument *args,
                       unsigned count) :
#if !defined(DISABLE_SMART_POINTERS)
      Actions(actions),
#endif
      Args(args), Count(count) { }

    // FIXME: Lame, not-fully-type-safe emulation of 'move semantics'.
    ASTTemplateArgsPtr(ASTTemplateArgsPtr &Other) :
#if !defined(DISABLE_SMART_POINTERS)
      Actions(Other.Actions),
#endif
      Args(Other.Args), Count(Other.Count) {
#if !defined(DISABLE_SMART_POINTERS)
      Other.Count = 0;
#endif
    }

    // FIXME: Lame, not-fully-type-safe emulation of 'move semantics'.
    ASTTemplateArgsPtr& operator=(ASTTemplateArgsPtr &Other)  {
#if !defined(DISABLE_SMART_POINTERS)
      Actions = Other.Actions;
#endif
      Args = Other.Args;
      Count = Other.Count;
#if !defined(DISABLE_SMART_POINTERS)
      Other.Count = 0;
#endif
      return *this;
    }

#if !defined(DISABLE_SMART_POINTERS)
    ~ASTTemplateArgsPtr() { destroy(); }
#endif

    ParsedTemplateArgument *getArgs() const { return Args; }
    unsigned size() const { return Count; }

    void reset(ParsedTemplateArgument *args, unsigned count) {
#if !defined(DISABLE_SMART_POINTERS)
      destroy();
#endif
      Args = args;
      Count = count;
    }

    const ParsedTemplateArgument &operator[](unsigned Arg) const;

    ParsedTemplateArgument *release() const {
#if !defined(DISABLE_SMART_POINTERS)
      Count = 0;
#endif
      return Args;
    }
  };

  /// \brief A small vector that owns a set of AST nodes.
  template <ASTDestroyer Destroyer, unsigned N = 8>
  class ASTOwningVector : public llvm::SmallVector<void *, N> {
#if !defined(DISABLE_SMART_POINTERS)
    ActionBase &Actions;
    bool Owned;
#endif

    ASTOwningVector(ASTOwningVector &); // do not implement
    ASTOwningVector &operator=(ASTOwningVector &); // do not implement

  public:
    explicit ASTOwningVector(ActionBase &Actions)
#if !defined(DISABLE_SMART_POINTERS)
      : Actions(Actions), Owned(true)
#endif
    { }

#if !defined(DISABLE_SMART_POINTERS)
    ~ASTOwningVector() {
      if (!Owned)
        return;

      for (unsigned I = 0, Last = this->size(); I != Last; ++I)
        (Actions.*Destroyer)((*this)[I]);
    }
#endif

    void **take() {
#if !defined(DISABLE_SMART_POINTERS)
      Owned = false;
#endif
      return &this->front();
    }

    template<typename T> T **takeAs() { return (T**)take(); }

#if !defined(DISABLE_SMART_POINTERS)
    ActionBase &getActions() const { return Actions; }
#endif
  };

  /// A SmallVector of statements, with stack size 32 (as that is the only one
  /// used.)
  typedef ASTOwningVector<&ActionBase::DeleteStmt, 32> StmtVector;
  /// A SmallVector of expressions, with stack size 12 (the maximum used.)
  typedef ASTOwningVector<&ActionBase::DeleteExpr, 12> ExprVector;

  template <ASTDestroyer Destroyer, unsigned N> inline
  ASTMultiPtr<Destroyer> move_arg(ASTOwningVector<Destroyer, N> &vec) {
#if !defined(DISABLE_SMART_POINTERS)
    return ASTMultiPtr<Destroyer>(vec.getActions(), vec.take(), vec.size());
#else
    return ASTMultiPtr<Destroyer>(vec.take(), vec.size());
#endif
  }

#if !defined(DISABLE_SMART_POINTERS)

  // Out-of-line implementations due to definition dependencies

  template <ASTDestroyer Destroyer> inline
  void moving::ASTMultiMover<Destroyer>::release() {
    Moved.Nodes = 0;
    Moved.Count = 0;
  }

  // Move overloads.

  template <ASTDestroyer Destroyer> inline
  ASTOwningResult<Destroyer> move(ASTOwningResult<Destroyer> &ptr) {
    return ASTOwningResult<Destroyer>(moving::ASTResultMover<Destroyer>(ptr));
  }

  template <ASTDestroyer Destroyer> inline
  ASTMultiPtr<Destroyer> move(ASTMultiPtr<Destroyer> &ptr) {
    return ASTMultiPtr<Destroyer>(moving::ASTMultiMover<Destroyer>(ptr));
  }

#else

  template <ASTDestroyer Destroyer> inline
  ASTOwningPtr<Destroyer>::ASTOwningPtr(const ASTOwningResult<Destroyer> &o)
    : Node(o.get()) { }

  // These versions are hopefully no-ops.
  template <ASTDestroyer Destroyer> inline
  ASTOwningResult<Destroyer>& move(ASTOwningResult<Destroyer> &ptr) {
    return ptr;
  }

  template <ASTDestroyer Destroyer> inline
  ASTOwningPtr<Destroyer>& move(ASTOwningPtr<Destroyer> &ptr) {
    return ptr;
  }

  template <ASTDestroyer Destroyer> inline
  ASTMultiPtr<Destroyer>& move(ASTMultiPtr<Destroyer> &ptr) {
    return ptr;
  }
#endif
}

#endif
