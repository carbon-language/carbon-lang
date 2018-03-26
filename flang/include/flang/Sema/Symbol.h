#ifndef FLANG_SEMA_SYMBOL_H
#define FLANG_SEMA_SYMBOL_H

#include "flang/Sema/Attr.h"
#include "flang/Sema/Identifier.h"
#include "flang/Sema/Type.h"

#include <vector>

namespace Fortran::semantics {

class Scope;

// Forward declaration of all Symbol classes
#define SEMA_DEFINE_SYMBOL(Classname) class Classname;
#include "flang/Sema/Symbol.def"
#undef SEMA_DEFINE_SYMBOL

// Describe a symbol.
class Symbol {
public:
  enum ClassId {
#define SEMA_DEFINE_SYMBOL(Classname) Classname##Id,
#include "flang/Sema/Symbol.def"
#undef SEMA_DEFINE_SYMBOL
    // ... and probably more to come
    last_ClassId
  };
private:
  ClassId cid_;
  Scope *owner_;  // The scope that owns this symbol
  Identifier ident_;  // The local name of that symbol
private:
  TypeSpec *type_ = nullptr;  // The type associated to that symbol
public:
  Symbol(ClassId cid, Scope *owner, Identifier name);

public:
  // For now, provide some toXXXX() member to perform the dynamic cast.
  //
  //

#define SEMA_DEFINE_SYMBOL(Classname) \
  Classname *to##Classname(void) { \
    return (cid_ == Classname##Id) ? (Classname *)this : nullptr; \
  } \
  const Classname *to##Classname(void) const { \
    return (cid_ == Classname##Id) ? (const Classname *)this : nullptr; \
  }
#include "flang/Sema/Symbol.def"
#undef SEMA_DEFINE_SYMBOL

public:
  Scope *owner() { return owner_; }

  Identifier name() { return ident_; }

  bool Match(Identifier ident) { return ident == ident_; }

  std::string toString() const { return ident_.name(); }
};

//
// A local symbol is a temporary placeholder for symbols whose nature cannot be
// infered when the it is first encounterd. It shall eventually be replaced by
// an actual symbol.
//
// For instance:
//
//    INTEGER :: x,y,z              ! x, y, and z start as TemporarySymbol
//    VOLATILE :: x                 ! x is resolved as a VariableSymbol
//    PARAMETER(y=42)               ! y is resolved as a ParameterSymbol
//    print *, z(10)                ! z is resolved as an ExternalSymbol
//
class TemporarySymbol : public Symbol {
public:
  TemporarySymbol(Scope *owner, Identifier name)
    : Symbol(TemporarySymbolId, owner, name) {}

private:
  // If not NULL, then this is the actual symbol to be used instead of this
  // symbol.
  Symbol *actual_;

private:
};

//
// A symbol representing the name of a Module
//
class ModuleSymbol : public Symbol {
public:
  ModuleSymbol(Scope *owner, Identifier name)
    : Symbol(ProgramSymbolId, owner, name) {}

private:
};

//
// A symbol representing the name of a Program
//
class ProgramSymbol : public Symbol {
public:
  ProgramSymbol(Scope *owner, Identifier name)
    : Symbol(ProgramSymbolId, owner, name) {}

private:
};

//
// A symbol representing the name of a Program
//
class BlockDataSymbol : public Symbol {
public:
  BlockDataSymbol(Scope *owner, Identifier name)
    : Symbol(BlockDataSymbolId, owner, name) {}

private:
};

// A symbol representing a parameter whose value can be queried.
//
//
class ParameterSymbol : public Symbol {
public:
  ParameterSymbol(Scope *owner, Identifier name)
    : Symbol(ParameterSymbolId, owner, name) {}
};

// A symbol representing an EXTERNAL function.
class ExternalSymbol : public Symbol {
public:
  ExternalSymbol(Scope *owner, Identifier name)
    : Symbol(ExternalSymbolId, owner, name) {}
};

//
// A symbol representing a variable.
//
// The variable may be local or part of a common.
//
// Question: Do we want to represent pointers using VariableSymbol or a
// dedicated class?
//
class VariableSymbol : public Symbol {
public:
  VariableSymbol(Scope *owner, Identifier name)
    : Symbol(VariableSymbolId, owner, name) {}
};

// A symbol representing a dummy argument.
class DummyArgumentSymbol : public Symbol {
public:
  DummyArgumentSymbol(Scope *owner, Identifier name)
    : Symbol(DummyArgumentSymbolId, owner, name) {}
};

// A symbol representing a COMMON block.
//
// TODO: Each member of common shall appear as a VariableSymbol probably with a
//       boolean flag to indicate that it belongs to a COMMON. The actual
//       CommonSymbol is then easy to figure out by the local scope.
//
//
class CommonSymbol : public Symbol {
public:
  CommonSymbol(Scope *owner, Identifier name)
    : Symbol(CommonSymbolId, owner, name) {}

private:
  // The content of the common section.
  std::vector<Identifier > content_;
};

//
// A symbol describing a subprogram declared by a FUNCTION or SUBROUTINE
// construct (either as a main unit, a internal subprogram, or an interface).
//
// The SubprogramSymbol only occurs in the local scope providing the FUNCTION
// or SUBROUTINE construct.
//
class SubroutineSymbol : public Symbol {
public:
  SubroutineSymbol(Scope *owner, Identifier name)
    : Symbol(SubroutineSymbolId, owner, name) {}

private:
  Scope *inner_scope_;
  std::vector<DummyArgumentSymbol *> args;
};

class FunctionSymbol : public Symbol {
public:
  FunctionSymbol(Scope *owner, Identifier name)
    : Symbol(FunctionSymbolId, owner, name) {}

private:
  Scope *inner_scope_;
  VariableSymbol *result;  // Shortcut to the variable that holds the result
  std::vector<DummyArgumentSymbol *> args;
};

// Symbol describing an interface.
// A null name is allowed to represent an unnamed interface.
class InterfaceSymbol : public Symbol {
public:
  InterfaceSymbol(Scope *owner, Identifier name)
    : Symbol(InterfaceSymbolId, owner, name) {}

private:
  Scope *inner_scope_;
};

// A symbol imported from the parent or host scope of the local scope either
// automatically or via an 'import' statement.
class ImportedSymbol : public Symbol {
public:
  ImportedSymbol(Scope *owner, Identifier name)
    : Symbol(ImportedSymbolId, owner, name) {}

private:
  // Provide the target symbol in the parent or host scope.
  Symbol *target_;
};

// A symbol imported from a module within a submodule.
//
// FIXME:
// I am not sure how submodules scopes shall be implemented.
// On one side, a submodule inherit all the symbols from its parent
// module or submodule so that part looks a lot like regular host
// association.
//
// However, a submodule may also provide the implementation of
// a subroutine or function already declared in the parent module.
// A dedicated mecanism is probably needed here. A boolean flag
// added to SubprogramSymbol to tag all 'module function' and
// 'module subroutine' could be enough. TO BE INVESTIGATED
//
//
class SubSymbol : public Symbol {
public:
  SubSymbol(Scope *owner, Identifier name)
    : Symbol(SubSymbolId, owner, name) {}

private:
  // Provide the target symbol in the parent or host scope.
  Symbol *target_;
};

// A symbol provided by one or more USE statements.
//
// Reminder: a UsedSymbol may be ambiguous even when it has
// only one target. This is because ambiguities are always
// resolved when the symbol is actually used so a module
// may export ambiguous symbols.
//
//
class UsedSymbol : public Symbol {
public:
  UsedSymbol(Scope *owner, Identifier name)
    : Symbol(UsedSymbolId, owner, name) {}

private:
  // Provide the list of target symbols in the
  std::vector<Symbol *> targets_;
  bool ambiguous_;
};

//
// A symbol representing a derived type.
//
class DerivedTypeSymbol : public Symbol {
public:
  DerivedTypeSymbol(Scope *owner, Identifier name)
    : Symbol(DerivedTypeSymbolId, owner, name) {}

private:
  Scope *inner_scope_;
};

//
// A symbol representing a member in a Derived type
//
class MemberSymbol : public Symbol {
public:
  MemberSymbol(Scope *owner, Identifier name)
    : Symbol(MemberSymbolId, owner, name) {}

private:
};

//
// A symbol representing a namelist-group
//
class NamelistSymbol : public Symbol {
public:
  NamelistSymbol(Scope *owner, Identifier name)
    : Symbol(NamelistSymbolId, owner, name) {}

private:
  std::vector<Symbol *> content_;
};

//
// A symbol representing the name of a construct
//
class ConstructSymbol : public Symbol {
public:
  ConstructSymbol(Scope *owner, Identifier name)
    : Symbol(ConstructSymbolId, owner, name) {}

private:
};

}  // namespace Fortran::semantics

// NOTE:  Support for EQUIVALENCE ...

#endif // of FLANG_SEMA_SYMBOL_H
