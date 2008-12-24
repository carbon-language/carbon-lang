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
// move_convert functions, which help the compiler out with some explicit
// conversions.

namespace clang
{
  // Basic
  class DiagnosticBuilder;

  /// ActionBase - A small part split from Action because of the horrible
  /// definition order dependencies between Action and the smart pointers.
  class ActionBase {
  public:
    /// Out-of-line virtual destructor to provide home for this class.
    virtual ~ActionBase();

    // Types - Though these don't actually enforce strong typing, they document
    // what types are required to be identical for the actions.
    typedef void ExprTy;
    typedef void StmtTy;
    typedef void TemplateParamsTy;
    typedef void TemplateArgTy;

    /// ActionResult - This structure is used while parsing/acting on
    /// expressions, stmts, etc.  It encapsulates both the object returned by
    /// the action, plus a sense of whether or not it is valid.
    template<unsigned UID>
    struct ActionResult {
      void *Val;
      bool isInvalid;

      ActionResult(bool Invalid = false) : Val(0), isInvalid(Invalid) {}
      template<typename ActualExprTy>
      ActionResult(ActualExprTy *val) : Val(val), isInvalid(false) {}
      ActionResult(const DiagnosticBuilder &) : Val(0), isInvalid(true) {}

      const ActionResult &operator=(void *RHS) {
        Val = RHS;
        isInvalid = false;
        return *this;
      }
    };

    /// Deletion callbacks - Since the parser doesn't know the concrete types of
    /// the AST nodes being generated, it must do callbacks to delete objects
    /// when recovering from errors. These are in ActionBase because the smart
    /// pointers need access to them.
    virtual void DeleteExpr(ExprTy *E) {}
    virtual void DeleteStmt(StmtTy *E) {}
    virtual void DeleteTemplateParams(TemplateParamsTy *E) {}
    virtual void DeleteTemplateArg(TemplateArgTy *E) {}
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
  template <> struct DestroyerToUID<&ActionBase::DeleteTemplateArg> {
    static const unsigned UID = 1;
  };

  /// ASTOwningResult - A moveable smart pointer for AST nodes that also
  /// has an extra flag to indicate an additional success status.
  template <ASTDestroyer Destroyer> class ASTOwningResult;

  /// ASTOwningPtr - A moveable smart pointer for AST nodes.
  template <ASTDestroyer Destroyer> class ASTOwningPtr;

  /// ASTMultiPtr - A moveable smart pointer to multiple AST nodes. Only owns
  /// the individual pointers, not the array holding them.
  template <ASTDestroyer Destroyer> class ASTMultiPtr;

  namespace moving {
    /// Move emulation helper for ASTOwningResult. NEVER EVER use this class
    /// directly if you don't know what you're doing.
    template <ASTDestroyer Destroyer>
    class ASTResultMover
    {
      ASTOwningResult<Destroyer> &Moved;

    public:
      ASTResultMover(ASTOwningResult<Destroyer> &moved) : Moved(moved) {}

      ASTOwningResult<Destroyer> * operator ->() { return &Moved; }
    };

    /// Move emulation helper for ASTOwningPtr. NEVER EVER use this class
    /// directly if you don't know what you're doing.
    template <ASTDestroyer Destroyer>
    class ASTPtrMover
    {
      ASTOwningPtr<Destroyer> &Moved;

    public:
      ASTPtrMover(ASTOwningPtr<Destroyer> &moved) : Moved(moved) {}

      ASTOwningPtr<Destroyer> * operator ->() { return &Moved; }
    };

    /// Move emulation helper for ASTMultiPtr. NEVER EVER use this class
    /// directly if you don't know what you're doing.
    template <ASTDestroyer Destroyer>
    class ASTMultiMover
    {
      ASTMultiPtr<Destroyer> &Moved;

    public:
      ASTMultiMover(ASTMultiPtr<Destroyer> &moved) : Moved(moved) {}

      ASTMultiPtr<Destroyer> * operator ->() { return &Moved; }

      /// Reset the moved object's internal structures.
      void release();
    };
  }

  template <ASTDestroyer Destroyer>
  class ASTOwningPtr
  {
    ActionBase *Actions;
    void *Node;

    friend class moving::ASTPtrMover<Destroyer>;

    ASTOwningPtr(ASTOwningPtr&); // DO NOT IMPLEMENT
    ASTOwningPtr& operator =(ASTOwningPtr&); // DO NOT IMPLEMENT

    void destroy() {
      if (Node) {
        assert(Actions && "Owning pointer without Action owns node.");
        (Actions->*Destroyer)(Node);
      }
    }

  public:
    explicit ASTOwningPtr(ActionBase &actions)
      : Actions(&actions), Node(0) {}
    ASTOwningPtr(ActionBase &actions, void *node)
      : Actions(&actions), Node(node) {}
    /// Move from another owning pointer
    ASTOwningPtr(moving::ASTPtrMover<Destroyer> mover)
      : Actions(mover->Actions), Node(mover->take()) {}

    /// Move assignment from another owning pointer
    ASTOwningPtr & operator =(moving::ASTPtrMover<Destroyer> mover) {
      Actions = mover->Actions;
      Node = mover->take();
      return *this;
    }

    /// Assignment from a raw pointer. Takes ownership - beware!
    ASTOwningPtr & operator =(void *raw)
    {
      assert((Actions || !raw) && "Cannot assign non-null raw without Action");
      Node = raw;
      return *this;
    }

    /// Access to the raw pointer.
    void * get() const { return Node; }

    /// Release the raw pointer.
    void * take() {
      void *tmp = Node;
      Node = 0;
      return tmp;
    }

    /// Alias for interface familiarity with unique_ptr.
    void * release() {
      return take();
    }

    /// Get the Action associated with the node.
    ActionBase* getActions() const { return Actions; }

    /// Move hook
    operator moving::ASTPtrMover<Destroyer>() {
      return moving::ASTPtrMover<Destroyer>(*this);
    }
  };

  template <ASTDestroyer Destroyer>
  class ASTOwningResult
  {
    ASTOwningPtr<Destroyer> Ptr;
    bool Invalid;

    friend class moving::ASTResultMover<Destroyer>;

    ASTOwningResult(ASTOwningResult&); // DO NOT IMPLEMENT
    ASTOwningResult& operator =(ASTOwningResult&); // DO NOT IMPLEMENT

  public:
    typedef ActionBase::ActionResult<DestroyerToUID<Destroyer>::UID> DumbResult;

    explicit ASTOwningResult(ActionBase &actions, bool invalid = false)
      : Ptr(actions, 0), Invalid(invalid) {}
    ASTOwningResult(ActionBase &actions, void *node)
      : Ptr(actions, node), Invalid(false) {}
    ASTOwningResult(ActionBase &actions, const DumbResult &res)
      : Ptr(actions, res.Val), Invalid(res.isInvalid) {}
    /// Move from another owning result
    ASTOwningResult(moving::ASTResultMover<Destroyer> mover)
      : Ptr(moving::ASTPtrMover<Destroyer>(mover->Ptr)),
        Invalid(mover->Invalid) {}
    /// Move from an owning pointer
    ASTOwningResult(moving::ASTPtrMover<Destroyer> mover)
      : Ptr(mover), Invalid(false) {}

    /// Move assignment from another owning result
    ASTOwningResult & operator =(moving::ASTResultMover<Destroyer> mover) {
      Ptr = move(mover->Ptr);
      Invalid = mover->Invalid;
      return *this;
    }

    /// Move assignment from an owning ptr
    ASTOwningResult & operator =(moving::ASTPtrMover<Destroyer> mover) {
      Ptr = mover;
      Invalid = false;
      return *this;
    }

    /// Assignment from a raw pointer. Takes ownership - beware!
    ASTOwningResult & operator =(void *raw)
    {
      Ptr = raw;
      Invalid = false;
      return *this;
    }

    /// Assignment from an ActionResult. Takes ownership - beware!
    ASTOwningResult & operator =(const DumbResult &res) {
      Ptr = res.Val;
      Invalid = res.isInvalid;
      return *this;
    }

    /// Access to the raw pointer.
    void * get() const { return Ptr.get(); }

    bool isInvalid() const { return Invalid; }

    /// Does this point to a usable AST node? To be usable, the node must be
    /// valid and non-null.
    bool isUsable() const { return !Invalid && get(); }

    /// Take outside ownership of the raw pointer.
    void * take() {
      if (Invalid)
        return 0;
      return Ptr.take();
    }

    /// Alias for interface familiarity with unique_ptr.
    void * release() { return take(); }

    /// Pass ownership to a classical ActionResult.
    DumbResult result() {
      if (Invalid)
        return true;
      return Ptr.take();
    }

    /// Get the Action associated with the node.
    ActionBase* getActions() const { return Ptr.getActions(); }

    /// Move hook
    operator moving::ASTResultMover<Destroyer>() {
      return moving::ASTResultMover<Destroyer>(*this);
    }

    /// Special function for moving to an OwningPtr.
    moving::ASTPtrMover<Destroyer> ptr_move() {
      return moving::ASTPtrMover<Destroyer>(Ptr);
    }
  };

  template <ASTDestroyer Destroyer>
  class ASTMultiPtr
  {
    ActionBase &Actions;
    void **Nodes;
    unsigned Count;

    friend class moving::ASTMultiMover<Destroyer>;

    ASTMultiPtr(ASTMultiPtr&); // DO NOT IMPLEMENT
    // Reference member prevents copy assignment.

    void destroy() {
      assert((Count == 0 || Nodes) && "No nodes when count is not zero.");
      for (unsigned i = 0; i < Count; ++i) {
        if (Nodes[i])
          (Actions.*Destroyer)(Nodes[i]);
      }
    }

  public:
    explicit ASTMultiPtr(ActionBase &actions)
      : Actions(actions), Nodes(0), Count(0) {}
    ASTMultiPtr(ActionBase &actions, void **nodes, unsigned count)
      : Actions(actions), Nodes(nodes), Count(count) {}
    /// Move constructor
    ASTMultiPtr(moving::ASTMultiMover<Destroyer> mover)
      : Actions(mover->Actions), Nodes(mover->Nodes), Count(mover->Count) {
      mover.release();
    }

    /// Move assignment
    ASTMultiPtr & operator =(moving::ASTMultiMover<Destroyer> mover) {
      destroy();
      Nodes = mover->Nodes;
      Count = mover->Count;
      mover.release();
      return *this;
    }

    /// Access to the raw pointers.
    void ** get() const { return Nodes; }

    /// Access to the count.
    unsigned size() const { return Count; }

    void ** release() {
      void **tmp = Nodes;
      Nodes = 0;
      Count = 0;
      return tmp;
    }

    /// Move hook
    operator moving::ASTMultiMover<Destroyer>() {
      return moving::ASTMultiMover<Destroyer>(*this);
    }
  };

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
  ASTOwningPtr<Destroyer> move(ASTOwningPtr<Destroyer> &ptr) {
    return ASTOwningPtr<Destroyer>(moving::ASTPtrMover<Destroyer>(ptr));
  }

  template <ASTDestroyer Destroyer> inline
  ASTMultiPtr<Destroyer> move(ASTMultiPtr<Destroyer> &ptr) {
    return ASTMultiPtr<Destroyer>(moving::ASTMultiMover<Destroyer>(ptr));
  }

  // These are necessary because of ambiguity problems.

  template <ASTDestroyer Destroyer> inline
  ASTOwningPtr<Destroyer> move_convert(ASTOwningResult<Destroyer> &ptr) {
    return ASTOwningPtr<Destroyer>(ptr.ptr_move());
  }

  template <ASTDestroyer Destroyer> inline
  ASTOwningResult<Destroyer> move_convert(ASTOwningPtr<Destroyer> &ptr) {
    return ASTOwningResult<Destroyer>(moving::ASTPtrMover<Destroyer>(ptr));
  }

}

#endif
