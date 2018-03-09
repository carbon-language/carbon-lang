#ifndef FLANG_SEMA_SCOPE_H
#define FLANG_SEMA_SCOPE_H

#include "flang/Sema/Attr.h"
#include "flang/Sema/Identifier.h"
#include "flang/Sema/Symbol.h"
#include "flang/Sema/Type.h"

#include <cassert>
#include <vector>

namespace Fortran::semantics {

class Scope;

// =======================================================

//
// Describe a scope (or a symbol table)
//
// For now, there is only one class. Not sure if we need more.
// Of course, there are multiple kinds of scopes in Fortran and
// they all behave slightly differently. However, a single Scope class
// that adjust its behavior according to its Kind is probably
// easier to implement than a dozen of classes.
//
// Remark: ImportedSymbol and UsedSymbol may be created on the fly the first
// time they are referenced (in order to avoid an explosion of the number of
// symbols)
//
class Scope {
public:
  enum class Kind {
    SK_SYSTEM,  // the unique scope associated to the system (and providing
                // intrinsic procedures)
    SK_GLOBAL,  // the unique global scope containing all units
    SK_PROGRAM,  // a scope associated to a PROGRAM
    SK_MODULE,  // a scope associated to a MODULE
    SK_SUBMODULE,  // a scope associated to a SUBMODULE
    SK_FUNCTION,  // a scope associated to a FUNCTION
    SK_SUBROUTINE,  // a scope associated to a SUBROUTNE
    SK_BLOCKDATA,  // a scope associated to a BLOCKDATA
    SK_USE_MODULE,  // a scope describing all the public symbols of a module.
    SK_BLOCK,  // a scope associated to a BLOCK construct
    SK_DERIVED,  // a scope associated to a derived TYPE construct
    SK_INTERFACE  // a scope associated to an interface

    // ... and probably more to come
  };

public:
  Scope(Kind k, Scope *p, Symbol *s);
  ~Scope();

private:
  Kind kind_;

  //
  // For most ranks, id contains a district identifier that is increased
  // each time a new Scope is created.
  //
  // The System and Global scope respectively have 0 and 1
  //
  // Scopes that are created within the System scope (i.e. intrinsics)
  // are assigned a negative id.
  //
  // Scopes that are created within the Global scope (i.e. modules or
  // user code) are assigned an id larger than 1
  //
  // The unique System and Global scopes do not use that field to store
  // their respectived id (which are hardcoded). Instead they use it
  // to count the number of scopes already created.
  //
  int id_or_counter_;

  // For scopes associated to a name (so most of them except BLOCK
  // and unnamed INTERFACE) provide the associated symbol in the
  // parent scope.
  Symbol *self_;

  // The UsedModule describes how a module shall be used.
  //
  // For instance, consider the statement
  //
  //   use foobar , A => B, C => D
  //
  // The 'exclude' vector will contain the identifiers for "B" and "D"
  //
  // Remark: A and C are descibed as UsedSymbol entries in the local
  //         scope.
  //
  //
  // Reminder: Module that are imported with a 'only' qualifier are
  // entirely handled via UsedSymbol entries and so, do not have
  // a UsedModule descriptor
  //
  struct UsedModule {
    // A reference to the
    //
    Scope *module;
    // All identifiers to exclude when doing a lookup
    // in that module
    std::vector<Identifier> exclude;
  };

  //
  // The parent scope is the scope that lexically owns this local scope.
  // Symbols are normally not searched is the parent scope except in the case of
  // symbols declared by an 'import, only'.
  //

  Scope *parent_scope_;

  // The host scope is the scope that is searched last.   
  Scope *host_scope_;  

  std::list<Scope *> use_;  // Scopes for the modules that are entirely used
  std::list<Scope *>
      use_only_;  // Scopes for the modules that are partially used

  // All visible entries in the scope.
  //
  // The entries are sorted by lexical order of declaration (except for
  // subprogram that are added at the end of the declaration part) which
  // means that lookups must be performed in reverse order.
  //
  // The reason to do that instead of using a map or any other data structure
  // optimized for fast lookup is to allow inner scopes (e.g. BLOCK) to see only
  // a subset of their parent scope.
  //
  // Example:
  //
  //    PROGRAM test
  //      implicit integer (a-z)
  //      integer :: i
  //      do i = 1,2
  //         if (i==1) bar=0
  //         if (i==2) then
  //            BLOCK
  //              bar=42        ! this is the 'bar' declared above
  //              foo=42        ! this 'foo' is private to the BLOCK
  //            END BLOCK
  //         endif
  //         if (i==1)  foo=0
  //         print *,'foo=',foo, 'bar=',bar
  //      enddo
  //    END PROGRAM test
  //
  // The expected output is
  //     foo=0   bar=0
  //     foo=0   bar=42
  //
  // In that case, the idea is that the block shall not have access
  // to the (implicit) declarations of its parent scope after 'bar'
  // This is easy to do with a vector by keeping the number of entries
  // in the parent scope at the time the block construct was created.
  //
  // Remark: That mecanism may not be required if all symbols can
  //         be resolved in a single pass.
  //
  //  Another potential reason to keep the declaration order is when
  //  a symbol hides or extends another symbol. A typical example would be
  //  named interfaces that can be redeclared multiple times.
  //

  std::vector<Symbol *> entries_;

  //
  // A vector containing all Symbols owned by that scope
  //
  std::vector<Symbol *> owned_;

public:
  Kind kind() const { return kind_; }

  const Scope *getParentScope() const { return parent_scope_; }
  Scope *getParentScope() { return parent_scope_; }

  //
  // The system scope is the scope that provides all intrinsics definitions
  //
  const Scope *getSystemScope() const;

  //
  // The Global scope is the scope that is immediately below the System scope.
  //
  // It contains the symbols for the Program unit (PROGRAM, MODULE, SUBMODULE,
  // FUNCTION, SUBROUTINE and BLOCKDATA)
  //
  // However, that scope is not standard in the sense that it requires a
  // dedicated lookup to find its symbols.
  // This is because the program units do not see each others by default
  //
  // This is also the scope that owns all USE_MODULE.
  //
  //
  Scope *getGlobalScope();
  const Scope *getGlobalScope() const;

  // Look for a symbol locally and in the host scope (if any)
  Symbol *Lookup(Identifier name);
  const Symbol *Lookup(Identifier name) const;

  // Look for a symbol locally (do not explore the host scope).
  //
  Symbol *LookupLocal(Identifier name);
  const Symbol *LookupLocal(Identifier name) const;

  //
  // Lookup for a Program Unit by name
  //
  Symbol *LookupProgramUnit(Identifier name);
  const Symbol *LookupProgramUnit(Identifier name) const;

  //
  // Lookup for a known module.
  //
  // The result is either null or a scope of kind SK_USE_MODULE.
  //
  Symbol *LookupModule(Identifier name);
  const Symbol *LookupModule(Identifier name) const;

  //
  // Provide the distinct id associated to each scope 
  //
  int id() const;

  //
  // Add a symbol to the scope.
  //
  void add(Symbol *s);

public:
  std::string toString(void);

public:
  // a temporary method to fail
  void fail(const std::string &msg) const;
};

}  // namespace Fortran::semantics

#endif // FLANG_SEMA_SCOPE_H
